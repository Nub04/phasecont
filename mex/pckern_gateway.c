/* pckern: the MEX gateway between MATLAB and the pckern engine library.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Internal. The callers are phasecont.Model, phasecont.internal.Run and the
 * tests. The library is loaded at run time from the folder of this MEX file
 * (pckern.dll, libpckern.so or libpckern.dylib), and every command calls one
 * member of the table that pc_api_v1() returns (crates/pc-ffi/include/pckern.h).
 *
 * Build: mex -R2018a -I<repo>/crates/pc-ffi/include -output pckern pckern_gateway.c
 * (add -ldl on Linux). buildtool mex does this.
 *
 * Commands (indexes are 0-based, as in the C ABI; the MATLAB classes convert):
 *   [v, lib] = pckern("version")
 *   pckern("panic")                       always raises phasecont:engine:panic
 *   r = pckern("panic_in_worker")         run_poll on r raises the panic
 *   [m, diag, msg] = pckern("model_from_ode", text)   msg: last_error on a model error
 *   pckern("model_compile", m [, opts])
 *   info = pckern("model_info", m)
 *   f = pckern("model_rhs", m, x, p)       a model that uses t raises nonAutonomous
 *   f = pckern("model_rhs", m, x, p, t)    f(x, p, t); t is a real double scalar
 *   [u, ev] = pckern("model_equilibrium", m, u0, p, opts)
 *   r = pckern("run_start", m, problem, u0, p)
 *   status = pckern("run_poll", r)
 *   rows = pckern("run_copy", r, branch, what, start, count, stride)
 *   pckern("run_cancel", r)
 *   pckern("free", h)
 *
 * Only this thread (the MATLAB thread) calls mx* and mex* functions. The
 * library returns error codes, and the gateway raises phasecont:engine:<kind>.
 */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <stdlib.h>
#endif

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mex.h"
#include "pckern.h"

#ifdef _WIN32
#define PC_LIB_NAME "pckern.dll"
#elif defined(__APPLE__)
#define PC_LIB_NAME "libpckern.dylib"
#else
#define PC_LIB_NAME "libpckern.so"
#endif

#ifdef _WIN32
#define PC_PATH_MAX 32768
#elif defined(PATH_MAX)
#define PC_PATH_MAX (PATH_MAX + 64)
#else
#define PC_PATH_MAX 4160
#endif

static const pc_api_v1_t *g_api = NULL;
static char g_lib_path[3 * PC_PATH_MAX]; /* UTF-8 path of the loaded library */

/* Formats the message first and passes it as an argument, so a "%" in it
 * is never read as a format directive. Does not return. */
static void fail(const char *id, const char *fmt, ...)
{
    char buf[4096];
    int n;
    va_list ap;
    va_start(ap, fmt);
    n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof buf) {
        /* Truncated: drop a UTF-8 sequence that was cut in the middle. */
        size_t i = sizeof buf - 1;
        while (i > 0 && ((unsigned char)buf[i - 1] & 0xC0) == 0x80)
            i--;
        if (i > 0) {
            unsigned char lead = (unsigned char)buf[i - 1];
            size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
            if (need > 1 && sizeof buf - 1 - (i - 1) < need)
                buf[i - 1] = '\0';
        }
    }
    mexErrMsgIdAndTxt(id, "%s", buf);
}

static void bad_arg(const char *fmt, const char *what)
{
    fail("phasecont:engine:badArg", fmt, what);
}

static const char *kind_id(int32_t rc)
{
    switch (rc) {
    case PC_ERR_PANIC: return "phasecont:engine:panic";
    case PC_ERR_BAD_ARG: return "phasecont:engine:badArg";
    case PC_ERR_BAD_HANDLE: return "phasecont:engine:badHandle";
    case PC_ERR_INTERNAL: return "phasecont:engine:internal";
    case PC_ERR_MODEL: return "phasecont:engine:model";
    case PC_ERR_BAD_PROBLEM: return "phasecont:engine:badProblem";
    case PC_ERR_NON_AUTONOMOUS: return "phasecont:engine:nonAutonomous";
    case PC_ERR_NUMERIC: return "phasecont:engine:numeric";
    default: return NULL;
    }
}

/* Raises the error for a nonzero code. Free Rust strings before calling,
 * since mexErrMsgIdAndTxt does not return. */
static void check(int32_t rc)
{
    const char *msg, *id;
    if (rc == PC_OK)
        return;
    msg = g_api->last_error();
    if (!msg || !msg[0])
        msg = "(no message)";
    id = kind_id(rc);
    if (!id)
        fail("phasecont:engine:unknown", "unknown error code %d: %s", (int)rc, msg);
    fail(id, "%s", msg);
}

/* ---- Loader ------------------------------------------------------------ */

#ifdef _WIN32

static void to_utf8(const wchar_t *w, char *out, int out_size)
{
    if (WideCharToMultiByte(CP_UTF8, 0, w, -1, out, out_size, NULL, NULL) == 0)
        out[0] = '\0';
}

static void fail_win32(const char *what, DWORD err)
{
    wchar_t sysw[1024];
    char sys[3 * 1024];
    DWORD n = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                             err, 0, sysw, ARRAYSIZE(sysw), NULL);
    while (n > 0 && (sysw[n - 1] == L'\r' || sysw[n - 1] == L'\n' || sysw[n - 1] == L' '))
        sysw[--n] = L'\0';
    sys[0] = '\0';
    if (n > 0)
        to_utf8(sysw, sys, (int)sizeof sys);
    if (sys[0] == '\0')
        strcpy_s(sys, sizeof sys, "unknown error");
    fail("phasecont:engine:load", "%s: %s (Windows error %lu)", what, sys, (unsigned long)err);
}

/* Loads pckern.dll from the folder of this MEX file. */
static const pc_api_v1_t *open_library(void)
{
    static wchar_t path[PC_PATH_MAX];
    static char what[3 * PC_PATH_MAX + 64];
    static char p8[3 * PC_PATH_MAX];
    const wchar_t *dll = L"pckern.dll";
    HMODULE self = NULL, lib;
    DWORD len;
    wchar_t *slash;
    FARPROC sym;
    pc_api_v1_fn entry;

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)(void *)&open_library, &self))
        fail_win32("cannot find the MEX file's own module", GetLastError());
    len = GetModuleFileNameW(self, path, ARRAYSIZE(path));
    if (len == 0 || len >= ARRAYSIZE(path))
        fail_win32("cannot read the MEX file's path", GetLastError());
    slash = wcsrchr(path, L'\\');
    if (!slash || (size_t)(slash + 1 - path) + wcslen(dll) + 1 > ARRAYSIZE(path))
        fail("phasecont:engine:load", "unexpected MEX file path");
    wcscpy_s(slash + 1, ARRAYSIZE(path) - (size_t)(slash + 1 - path), dll);
    to_utf8(path, p8, (int)sizeof p8);

    lib = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!lib) {
        DWORD err = GetLastError();
        snprintf(what, sizeof what, "cannot load %s", p8);
        fail_win32(what, err);
    }
    sym = GetProcAddress(lib, PC_API_V1_SYMBOL);
    if (!sym)
        fail_win32("pckern.dll has no pc_api_v1 export", GetLastError());
    memcpy(&entry, &sym, sizeof entry);
    strcpy_s(g_lib_path, sizeof g_lib_path, p8);
    return entry();
}

#else /* POSIX */

static const pc_api_v1_t *open_library(void)
{
    static char path[PC_PATH_MAX];
    Dl_info info;
    void *self, *lib, *sym;
    void (*fn)(int, mxArray **, int, const mxArray **) = mexFunction;
    const char *name = PC_LIB_NAME;
    static char resolved[PC_PATH_MAX];
    const char *self_path, *slash, *err;
    size_t dir_len;
    pc_api_v1_fn entry;

    memcpy(&self, &fn, sizeof self);
    if (!dladdr(self, &info) || !info.dli_fname)
        fail("phasecont:engine:load", "cannot find the MEX file's own path");
    /* Resolve symlinks and relative names so the sibling library is found
     * next to the real file. */
    self_path = realpath(info.dli_fname, resolved) ? resolved : info.dli_fname;
    slash = strrchr(self_path, '/');
    if (!slash)
        fail("phasecont:engine:load", "the MEX file's path \"%s\" has no folder", self_path);
    dir_len = (size_t)(slash + 1 - self_path);
    if (dir_len + strlen(name) + 1 > sizeof path)
        fail("phasecont:engine:load", "the MEX file's path is too long");
    memcpy(path, self_path, dir_len);
    strcpy(path + dir_len, name);

    lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        err = dlerror();
        fail("phasecont:engine:load", "cannot load %s: %s", path, err ? err : "unknown error");
    }
    sym = dlsym(lib, PC_API_V1_SYMBOL);
    if (!sym) {
        err = dlerror();
        fail("phasecont:engine:load", "%s has no pc_api_v1 export: %s", path,
             err ? err : "unknown error");
    }
    memcpy(&entry, &sym, sizeof entry);
    memcpy(g_lib_path, path, strlen(path) + 1);
    return entry();
}

#endif

/* Loads the library on first use. The library is never unloaded (no
 * FreeLibrary, no dlclose): run worker threads and rayon threads may still
 * run its code, so it stays loaded until MATLAB exits. A failure leaves
 * g_api NULL, so the next call tries again. */
static void load_api(void)
{
    const pc_api_v1_t *api;
    if (g_api)
        return;
    api = open_library();
    /* pc_api_v1 returns NULL only when its panic barrier caught a panic.
     * There is no table yet, so last_error cannot be read here. */
    if (!api)
        fail("phasecont:engine:load",
             "%s: pc_api_v1 returned NULL (a panic inside it was caught)", PC_LIB_NAME);
    if (api->abi_version != (uint32_t)PC_ABI_VERSION ||
        api->struct_size < (uint32_t)sizeof(pc_api_v1_t))
        fail("phasecont:engine:load",
             "%s ABI mismatch: abi_version %u, struct_size %u, gateway expects %u and at least %u",
             PC_LIB_NAME, (unsigned)api->abi_version, (unsigned)api->struct_size,
             (unsigned)PC_ABI_VERSION, (unsigned)sizeof(pc_api_v1_t));
    g_api = api;
}

/* ---- Argument helpers --------------------------------------------------- */

static int is_char_row(const mxArray *a)
{
    return mxIsChar(a) && mxGetNumberOfDimensions(a) == 2 && mxGetM(a) <= 1;
}

/* Returns a UTF-8 copy of a char row vector or string scalar. Free with mxFree.
 * The C Matrix API shows every string array as 1x1, even string.empty, so a
 * string goes through convertStringsToChars, which returns a char row only
 * for a string scalar. U+0000 in the text is badArg, since it would cut the
 * C string short. */
static char *get_text(const mxArray *a, const char *what)
{
    mxArray *c = NULL;
    char *s;
    const mxChar *chars;
    size_t k, n;
    if (mxIsClass(a, "string")) {
        mxArray *in = (mxArray *)a;
        mxArray *err = mexCallMATLABWithTrap(1, &c, 1, &in, "convertStringsToChars");
        if (err || !c || !is_char_row(c))
            bad_arg("%s must be a string scalar", what);
        a = c;
    } else if (!is_char_row(a)) {
        bad_arg("%s must be a character vector or string scalar", what);
    }
    chars = mxGetChars(a);
    n = mxGetNumberOfElements(a);
    for (k = 0; k < n; k++) {
        if (chars[k] == 0)
            bad_arg("%s contains U+0000, which text arguments cannot hold", what);
    }
    s = mxArrayToUTF8String(a);
    if (c)
        mxDestroyArray(c);
    if (!s)
        bad_arg("%s could not be converted to UTF-8", what);
    return s;
}

/* A real, full double vector in any orientation (empty allowed). Never NULL. */
static const double *get_vec(const mxArray *a, const char *what, size_t *len)
{
    static const double none = 0.0;
    const double *d;
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfDimensions(a) != 2)
        bad_arg("%s must be a real, full double vector", what);
    if (mxGetNumberOfElements(a) > 0 && mxGetM(a) != 1 && mxGetN(a) != 1)
        bad_arg("%s must be a vector, not a matrix", what);
    *len = mxGetNumberOfElements(a);
    d = mxGetDoubles(a);
    return d ? d : &none;
}

static double get_real_scalar(const mxArray *a, const char *what)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfElements(a) != 1)
        bad_arg("%s must be a real double scalar", what);
    return *mxGetDoubles(a);
}

static size_t get_size(const mxArray *a, const char *what)
{
    double v = get_real_scalar(a, what);
    if (!(v >= 0.0 && v <= 9007199254740992.0) || v != (double)(size_t)v)
        bad_arg("%s must be a non-negative integer", what);
    return (size_t)v;
}

static uint32_t get_u32(const mxArray *a, const char *what)
{
    size_t v = get_size(a, what);
    if (v > UINT32_MAX)
        bad_arg("%s is too large", what);
    return (uint32_t)v;
}

static uint64_t get_handle(const mxArray *a)
{
    if (!mxIsUint64(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
        fail("phasecont:engine:badArg", "a handle must be a real uint64 scalar");
    return *mxGetUint64s(a);
}

static mxArray *uint64_scalar(uint64_t v)
{
    mxArray *a = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(a) = v;
    return a;
}

/* Decodes one code point from UTF-8 and advances *p. Invalid, overlong,
 * surrogate, out-of-range and truncated sequences give U+FFFD and consume
 * the bytes that cannot belong to a valid sequence. */
static uint32_t next_code_point(const unsigned char **p, const unsigned char *end)
{
    const unsigned char *q = *p;
    unsigned char c = *q++;
    uint32_t cp, min;
    int extra, k;
    if (c < 0x80) {
        *p = q;
        return c;
    }
    if (c >= 0xC2 && c <= 0xDF) { extra = 1; cp = c & 0x1F; min = 0x80; }
    else if (c >= 0xE0 && c <= 0xEF) { extra = 2; cp = c & 0x0F; min = 0x800; }
    else if (c >= 0xF0 && c <= 0xF4) { extra = 3; cp = c & 0x07; min = 0x10000; }
    else { *p = q; return 0xFFFD; }
    for (k = 0; k < extra; k++) {
        if (q >= end || (*q & 0xC0) != 0x80) {
            *p = q;
            return 0xFFFD;
        }
        cp = (cp << 6) | (*q++ & 0x3F);
    }
    *p = q;
    if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return 0xFFFD;
    return cp;
}

/* Builds a 1-by-N char row from UTF-8 text. Code points above U+FFFF become
 * surrogate pairs. mxCreateString would read each byte as one character. */
static mxArray *utf8_to_char(const char *s)
{
    const unsigned char *b = (const unsigned char *)s, *end = b + strlen(s), *q;
    size_t units = 0;
    mwSize dims[2];
    mxArray *a;
    mxChar *out;
    for (q = b; q < end;)
        units += next_code_point(&q, end) > 0xFFFF ? 2 : 1;
    dims[0] = 1;
    dims[1] = (mwSize)units;
    a = mxCreateCharArray(2, dims);
    out = mxGetChars(a);
    for (q = b; q < end;) {
        uint32_t cp = next_code_point(&q, end);
        if (cp > 0xFFFF) {
            cp -= 0x10000;
            *out++ = (mxChar)(0xD800 + (cp >> 10));
            *out++ = (mxChar)(0xDC00 + (cp & 0x3FF));
        } else {
            *out++ = (mxChar)cp;
        }
    }
    return a;
}

/* Copies a Rust string into a MATLAB char array and frees the Rust copy. */
static mxArray *take_rust_string(char *s)
{
    mxArray *a = utf8_to_char(s ? s : "");
    g_api->free_string(s);
    return a;
}

static void need_args(int nrhs, int lo, int hi, const char *cmd)
{
    if (nrhs < lo || nrhs > hi) {
        if (lo == hi)
            fail("phasecont:engine:badArg",
                 "pckern(\"%s\", ...) takes %d input(s) after the command", cmd, lo - 1);
        fail("phasecont:engine:badArg",
             "pckern(\"%s\", ...) takes %d to %d inputs after the command", cmd, lo - 1,
             hi - 1);
    }
}

static void need_outs(int nlhs, int max, const char *cmd)
{
    if (nlhs > max)
        fail("phasecont:engine:badArg", "pckern(\"%s\", ...) returns at most %d output(s)", cmd,
             max);
}

/* ---- Commands ----------------------------------------------------------- */

static void cmd_version(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const char *v;
    (void)prhs;
    need_args(nrhs, 1, 1, "version");
    need_outs(nlhs, 2, "version");
    v = g_api->version();
    if (!v)
        fail("phasecont:engine:internal", "version() returned NULL");
    plhs[0] = utf8_to_char(v);
    if (nlhs > 1)
        plhs[1] = utf8_to_char(g_lib_path);
}

static void cmd_panic(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t run = 0;
    (void)plhs; (void)prhs;
    need_args(nrhs, 1, 1, "panic");
    need_outs(nlhs, 0, "panic");
    check(g_api->debug_panic(0, &run));
    fail("phasecont:engine:internal", "debug_panic(0) did not panic");
}

static void cmd_panic_in_worker(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t run = 0;
    (void)prhs;
    need_args(nrhs, 1, 1, "panic_in_worker");
    need_outs(nlhs, 1, "panic_in_worker");
    check(g_api->debug_panic(1, &run));
    plhs[0] = uint64_scalar(run);
}

static void cmd_model_from_ode(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *text, *diag = NULL;
    uint64_t m = 0;
    char *msg_copy = NULL;
    int32_t rc;
    need_args(nrhs, 2, 2, "model_from_ode");
    need_outs(nlhs, 3, "model_from_ode");
    text = get_text(prhs[1], "the model text");
    rc = g_api->model_from_ode(text, strlen(text), &m, &diag);
    mxFree(text);
    if (rc != PC_OK && rc != PC_ERR_MODEL) {
        g_api->free_string(diag);
        check(rc);
    }
    /* Read last_error before any other library call. A lowering failure
     * has no error entry in the diagnostics, only this text. */
    if (rc == PC_ERR_MODEL && nlhs > 2) {
        const char *e = g_api->last_error();
        size_t n = e ? strlen(e) : 0;
        msg_copy = (char *)mxMalloc(n + 1);
        memcpy(msg_copy, e ? e : "", n + 1);
    }
    /* A model error is not raised: phasecont.Model formats the diagnostics. */
    plhs[0] = uint64_scalar(rc == PC_OK ? m : 0);
    if (nlhs > 1)
        plhs[1] = take_rust_string(diag);
    else
        g_api->free_string(diag);
    if (nlhs > 2) {
        plhs[2] = utf8_to_char(msg_copy ? msg_copy : "");
        if (msg_copy)
            mxFree(msg_copy);
    }
}

static void cmd_model_compile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t m;
    char *opts = NULL;
    int32_t rc;
    (void)plhs;
    need_args(nrhs, 2, 3, "model_compile");
    need_outs(nlhs, 0, "model_compile");
    m = get_handle(prhs[1]);
    if (nrhs == 3)
        opts = get_text(prhs[2], "opts");
    rc = g_api->model_compile(m, opts);
    if (opts)
        mxFree(opts);
    check(rc);
}

static void cmd_model_info(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *json = NULL;
    int32_t rc;
    need_args(nrhs, 2, 2, "model_info");
    need_outs(nlhs, 1, "model_info");
    rc = g_api->model_info(get_handle(prhs[1]), &json);
    if (rc != PC_OK) {
        g_api->free_string(json);
        check(rc);
    }
    plhs[0] = take_rust_string(json);
}

static void cmd_model_rhs(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t m;
    const double *x, *p;
    size_t nx, np;
    double t = 0.0;
    mxArray *f;
    need_args(nrhs, 4, 5, "model_rhs");
    need_outs(nlhs, 1, "model_rhs");
    m = get_handle(prhs[1]);
    x = get_vec(prhs[2], "x", &nx);
    p = get_vec(prhs[3], "p", &np);
    if (nrhs == 5)
        t = get_real_scalar(prhs[4], "t");
    f = mxCreateDoubleMatrix(nx, 1, mxREAL);
    if (nrhs == 5)
        check(g_api->model_rhs_t(m, t, x, nx, p, np, mxGetDoubles(f), nx));
    else
        check(g_api->model_rhs(m, x, nx, p, np, mxGetDoubles(f), nx));
    plhs[0] = f;
}

static void cmd_model_equilibrium(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t m;
    const double *u0, *p;
    size_t n, np, k;
    char *opts;
    mxArray *u, *ev;
    double *eig;
    mxComplexDouble *z;
    int32_t rc;
    need_args(nrhs, 5, 5, "model_equilibrium");
    need_outs(nlhs, 2, "model_equilibrium");
    m = get_handle(prhs[1]);
    u0 = get_vec(prhs[2], "u0", &n);
    p = get_vec(prhs[3], "p", &np);
    opts = get_text(prhs[4], "opts");
    u = mxCreateDoubleMatrix(n, 1, mxREAL);
    eig = (double *)mxCalloc(2 * n + 1, sizeof(double));
    rc = g_api->model_equilibrium(m, u0, n, p, np, opts[0] ? opts : NULL, mxGetDoubles(u), n,
                                  eig, 2 * n);
    mxFree(opts);
    check(rc);
    ev = mxCreateDoubleMatrix(n, 1, mxCOMPLEX);
    z = mxGetComplexDoubles(ev);
    for (k = 0; k < n; k++) {
        z[k].real = eig[k];
        z[k].imag = eig[n + k];
    }
    mxFree(eig);
    plhs[0] = u;
    if (nlhs > 1)
        plhs[1] = ev;
    else
        mxDestroyArray(ev);
}

static void cmd_run_start(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t m, run = 0;
    char *problem;
    const double *u0, *p;
    size_t nu, np;
    int32_t rc;
    need_args(nrhs, 5, 5, "run_start");
    need_outs(nlhs, 1, "run_start");
    m = get_handle(prhs[1]);
    problem = get_text(prhs[2], "the problem");
    u0 = get_vec(prhs[3], "u0", &nu);
    p = get_vec(prhs[4], "p", &np);
    rc = g_api->run_start(m, problem, u0, nu, p, np, &run);
    mxFree(problem);
    check(rc);
    plhs[0] = uint64_scalar(run);
}

static void cmd_run_poll(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *json = NULL;
    int32_t rc;
    need_args(nrhs, 2, 2, "run_poll");
    need_outs(nlhs, 1, "run_poll");
    rc = g_api->run_poll(get_handle(prhs[1]), &json);
    if (rc != PC_OK) {
        g_api->free_string(json);
        check(rc);
    }
    plhs[0] = take_rust_string(json);
}

static void cmd_run_copy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t r;
    uint32_t branch, what;
    size_t start, count, stride, rows = 0, total;
    mxArray *out;
    need_args(nrhs, 7, 7, "run_copy");
    need_outs(nlhs, 1, "run_copy");
    r = get_handle(prhs[1]);
    branch = get_u32(prhs[2], "branch");
    what = get_u32(prhs[3], "what");
    start = get_size(prhs[4], "start");
    count = get_size(prhs[5], "count");
    stride = get_size(prhs[6], "stride");
    if (stride == 0)
        bad_arg("%s must be positive", "stride");
    if (count > SIZE_MAX / stride)
        bad_arg("%s * stride overflows", "count");
    total = count * stride;
    /* Callers clamp count to the point count from the status JSON. A huge
     * count raises MATLAB's out-of-memory error, not a phasecont ID. */
    /* Each column is one C row: MATLAB is column-major and the rows are
     * contiguous. Shrink to the rows written after the call. */
    out = mxCreateDoubleMatrix(stride, count, mxREAL);
    if (count > 0) {
        check(g_api->run_copy(r, branch, what, start, count, mxGetDoubles(out), total, &rows));
        if (rows > count)
            fail("phasecont:engine:internal", "run_copy reported %zu rows for %zu requested",
                 rows, count);
        mxSetN(out, rows);
    }
    plhs[0] = out;
}

static void cmd_run_cancel(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs;
    need_args(nrhs, 2, 2, "run_cancel");
    need_outs(nlhs, 0, "run_cancel");
    check(g_api->run_cancel(get_handle(prhs[1])));
}

static void cmd_free(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)plhs;
    need_args(nrhs, 2, 2, "free");
    need_outs(nlhs, 0, "free");
    check(g_api->free_handle(get_handle(prhs[1])));
}

typedef void (*cmd_fn)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static const struct {
    const char *name;
    cmd_fn fn;
} k_cmds[] = {
    {"version", cmd_version},
    {"panic", cmd_panic},
    {"panic_in_worker", cmd_panic_in_worker},
    {"model_from_ode", cmd_model_from_ode},
    {"model_compile", cmd_model_compile},
    {"model_info", cmd_model_info},
    {"model_rhs", cmd_model_rhs},
    {"model_equilibrium", cmd_model_equilibrium},
    {"run_start", cmd_run_start},
    {"run_poll", cmd_run_poll},
    {"run_copy", cmd_run_copy},
    {"run_cancel", cmd_run_cancel},
    {"free", cmd_free},
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *cmd;
    size_t k;
    if (nrhs < 1)
        fail("phasecont:engine:badArg", "usage: pckern(command, ...)");
    cmd = get_text(prhs[0], "the command");
    for (k = 0; k < sizeof k_cmds / sizeof k_cmds[0]; k++) {
        if (strcmp(cmd, k_cmds[k].name) == 0) {
            mxFree(cmd);
            load_api();
            k_cmds[k].fn(nlhs, plhs, nrhs, prhs);
            return;
        }
    }
    fail("phasecont:engine:badArg", "unknown pckern command \"%s\"", cmd);
}
