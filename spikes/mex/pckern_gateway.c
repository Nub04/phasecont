/* Phase 0 spike gateway for pckern.dll. This is not the P7a gateway.
 *
 * Build from the staging folder on C: (spikes/stage.sh does this):
 *   mex -R2018a -output pckern pckern_gateway.c fhn_aot.c
 *
 * Commands:
 *   [info, dll] = pckern("info")
 *   y = pckern("scale", x, a)
 *   pckern("panic")                  pckern("panic_in_worker")
 *   h = pckern("test_handle")        pckern("free_handle", h)
 *   json = pckern("run_json", name, args)
 *   json = pckern("bench_c", args)   times fhn_aot.c inside the Rust loop
 *   [h, n, nnz] = pckern("s3_build", args)
 *   [I, J, V, b] = pckern("s3_copy", h, n, nnz)
 *   [x, json] = pckern("s3_solve", h, n, args)
 *   path = pckern("module_path", name)   full path of a loaded DLL, or ""
 *   path = pckern("preload", fullpath)   loads a DLL by absolute path
 *
 * Only this thread (the MATLAB thread) calls mx* and mex* functions. Rust
 * returns an error code, and the gateway raises phasecont:engine:<kind>.
 */
#ifndef _WIN32
#error "The spike gateway is Windows only."
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mex.h"

typedef void (*pc_rhs_fn)(const double *x, const double *p, double *out);

/* Field order must match PcApiV1 in spikes/pckern/src/lib.rs. */
typedef struct PcApiV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *(*info)(void);
    const char *(*last_error)(void);
    void (*free_string)(char *s);
    int32_t (*scale)(const double *x, size_t n, double a, double *out);
    int32_t (*panic)(void);
    int32_t (*panic_in_worker)(void);
    int32_t (*test_handle)(uint64_t *out);
    int32_t (*free_handle)(uint64_t id);
    int32_t (*run_json)(const char *name, const char *args, char **out);
    int32_t (*bench_extern)(pc_rhs_fn rhs, pc_rhs_fn jac, const char *args, char **out);
    int32_t (*s3_build)(const char *args, uint64_t *out_handle, size_t *out_n, size_t *out_nnz);
    int32_t (*s3_copy)(uint64_t h, size_t n, size_t nnz, double *i, double *j, double *v,
                       double *b);
    int32_t (*s3_solve)(uint64_t h, size_t n, const char *args, double *out_x, char **out_json);
} PcApiV1;

#define PC_ABI_VERSION 1u

/* fhn_aot.c */
void fhn_aot_rhs(const double *x, const double *p, double *out);
void fhn_aot_jac(const double *x, const double *p, double *out);

static const PcApiV1 *g_api = NULL;
static wchar_t g_dll_path[32768];
static char g_dll_path_utf8[3 * 32768];

/* Formats the message first and passes it as an argument, so a "%" in it
 * is never read as a format directive. Does not return. */
static void fail(const char *id, const char *fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    mexErrMsgIdAndTxt(id, "%s", buf);
}

static const char *kind_id(int32_t rc)
{
    switch (rc) {
    case 1: return "phasecont:engine:panic";
    case 2: return "phasecont:engine:badArg";
    case 3: return "phasecont:engine:badHandle";
    case 4: return "phasecont:engine:internal";
    default: return "phasecont:engine:unknown";
    }
}

/* Raises the error for a nonzero code. Free Rust strings before calling. */
static void check(int32_t rc)
{
    const char *msg;
    if (rc == 0)
        return;
    msg = g_api->last_error();
    fail(kind_id(rc), "%s", (msg && msg[0]) ? msg : "(no message)");
}

static void to_utf8(const wchar_t *w, char *out, int out_size)
{
    if (WideCharToMultiByte(CP_UTF8, 0, w, -1, out, out_size, NULL, NULL) == 0)
        out[0] = '\0';
}

static void fail_win32(const char *what, DWORD err)
{
    char sys[1024];
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                             err, 0, sys, sizeof sys, NULL);
    while (n > 0 && (sys[n - 1] == '\r' || sys[n - 1] == '\n' || sys[n - 1] == ' '))
        sys[--n] = '\0';
    if (n == 0)
        strcpy_s(sys, sizeof sys, "unknown error");
    fail("phasecont:engine:load", "%s: %s (Windows error %lu)", what, sys, (unsigned long)err);
}

/* Loads pckern.dll from the folder of this MEX file. The DLL is never
 * unloaded: rayon threads may still run its code, so it stays loaded (and
 * its file locked) until MATLAB exits. */
static void load_api(void)
{
    static wchar_t path[32768];
    static char what[3 * 32768 + 64];
    const wchar_t *dll = L"pckern.dll";
    HMODULE self = NULL, lib;
    DWORD len;
    wchar_t *slash;
    const PcApiV1 *(*entry)(void);
    const PcApiV1 *api;

    if (g_api)
        return;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)(void *)&load_api, &self))
        fail_win32("cannot find the MEX file's own module", GetLastError());
    len = GetModuleFileNameW(self, path, ARRAYSIZE(path));
    if (len == 0 || len >= ARRAYSIZE(path))
        fail_win32("cannot read the MEX file's path", GetLastError());
    slash = wcsrchr(path, L'\\');
    if (!slash || (size_t)(slash + 1 - path) + wcslen(dll) + 1 > ARRAYSIZE(path))
        fail("phasecont:engine:load", "unexpected MEX file path");
    wcscpy_s(slash + 1, ARRAYSIZE(path) - (size_t)(slash + 1 - path), dll);

    lib = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!lib) {
        DWORD err = GetLastError();
        static char p8[3 * 32768];
        to_utf8(path, p8, sizeof p8);
        snprintf(what, sizeof what, "cannot load %s", p8);
        fail_win32(what, err);
    }
    entry = (const PcApiV1 *(*)(void))(void *)GetProcAddress(lib, "pc_api_v1");
    if (!entry)
        fail_win32("pckern.dll has no pc_api_v1 export", GetLastError());
    api = entry();
    /* pc_api_v1 returns NULL only when its panic barrier caught a panic.
     * There is no table yet, so last_error cannot be read here. */
    if (!api)
        fail("phasecont:engine:load",
             "pckern.dll: pc_api_v1 returned NULL (a panic inside it was caught)");
    else if (api->abi_version != PC_ABI_VERSION || api->struct_size < sizeof(PcApiV1))
        fail("phasecont:engine:load",
             "pckern.dll ABI mismatch: abi_version %u, struct_size %u, gateway expects %u and %u",
             api->abi_version, api->struct_size, PC_ABI_VERSION, (unsigned)sizeof(PcApiV1));
    wcscpy_s(g_dll_path, ARRAYSIZE(g_dll_path), path);
    to_utf8(g_dll_path, g_dll_path_utf8, sizeof g_dll_path_utf8);
    g_api = api;
}

static int is_char_row(const mxArray *a)
{
    return mxIsChar(a) && mxGetNumberOfDimensions(a) == 2 && mxGetM(a) <= 1;
}

/* Returns a UTF-8 copy of a char row vector or string scalar. Free with mxFree.
 * The C Matrix API shows every string array as 1x1, even string.empty, so a
 * string goes through convertStringsToChars, which returns a char row only
 * for a string scalar. */
static char *get_text(const mxArray *a, const char *what)
{
    mxArray *c = NULL;
    char *s;
    if (mxIsClass(a, "string")) {
        mxArray *in = (mxArray *)a;
        mxArray *err = mexCallMATLABWithTrap(1, &c, 1, &in, "convertStringsToChars");
        if (err || !c || !is_char_row(c))
            fail("phasecont:engine:badArg", "%s must be a string scalar", what);
        a = c;
    } else if (!is_char_row(a)) {
        fail("phasecont:engine:badArg", "%s must be a character vector or string scalar", what);
    }
    s = mxArrayToUTF8String(a);
    if (c)
        mxDestroyArray(c);
    if (!s)
        fail("phasecont:engine:badArg", "%s could not be converted to UTF-8", what);
    return s;
}

static double get_real_scalar(const mxArray *a, const char *what)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfElements(a) != 1)
        fail("phasecont:engine:badArg", "%s must be a real double scalar", what);
    return *mxGetDoubles(a);
}

static size_t get_size(const mxArray *a, const char *what)
{
    double v = get_real_scalar(a, what);
    if (!(v >= 0.0 && v <= 9007199254740992.0) || v != (double)(size_t)v)
        fail("phasecont:engine:badArg", "%s must be a non-negative integer", what);
    return (size_t)v;
}

static uint64_t get_handle(const mxArray *a)
{
    if (!mxIsUint64(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
        fail("phasecont:engine:badArg", "the handle must be a uint64 scalar");
    return *mxGetUint64s(a);
}

static mxArray *uint64_scalar(uint64_t v)
{
    mxArray *a = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(a) = v;
    return a;
}

/* Copies a Rust string into a MATLAB char array and frees the Rust copy. */
static mxArray *take_rust_string(char *s)
{
    mxArray *a = mxCreateString(s ? s : "");
    g_api->free_string(s);
    return a;
}

static void need_args(int nrhs, int n, const char *cmd)
{
    if (nrhs != n)
        fail("phasecont:engine:badArg", "pckern(\"%s\", ...) takes %d input(s) after the command",
             cmd, n - 1);
}

static void cmd_info(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const char *stamp, *msg;
    (void)prhs;
    need_args(nrhs, 1, "info");
    stamp = g_api->info();
    if (!stamp) {
        /* info() returns NULL only when its panic barrier caught a panic,
         * and then last_error holds the "panic: ..." text. */
        msg = g_api->last_error();
        fail("phasecont:engine:panic", "%s", (msg && msg[0]) ? msg : "info() returned NULL");
    }
    plhs[0] = mxCreateString(stamp);
    if (nlhs > 1)
        plhs[1] = mxCreateString(g_dll_path_utf8);
}

static void cmd_scale(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray *x;
    mxArray *y;
    double a;
    (void)nlhs;
    need_args(nrhs, 3, "scale");
    x = prhs[1];
    if (!mxIsDouble(x) || mxIsComplex(x) || mxIsSparse(x))
        fail("phasecont:engine:badArg", "x must be a real, full double array");
    a = get_real_scalar(prhs[2], "a");
    y = mxCreateNumericArray(mxGetNumberOfDimensions(x), mxGetDimensions(x), mxDOUBLE_CLASS,
                             mxREAL);
    check(g_api->scale(mxGetDoubles(x), mxGetNumberOfElements(x), a, mxGetDoubles(y)));
    plhs[0] = y;
}

static void cmd_panic(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)plhs; (void)prhs;
    need_args(nrhs, 1, "panic");
    check(g_api->panic());
}

static void cmd_panic_in_worker(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)plhs; (void)prhs;
    need_args(nrhs, 1, "panic_in_worker");
    check(g_api->panic_in_worker());
}

static void cmd_test_handle(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t h = 0;
    (void)nlhs; (void)prhs;
    need_args(nrhs, 1, "test_handle");
    check(g_api->test_handle(&h));
    plhs[0] = uint64_scalar(h);
}

static void cmd_free_handle(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)plhs;
    need_args(nrhs, 2, "free_handle");
    check(g_api->free_handle(get_handle(prhs[1])));
}

static void cmd_run_json(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *name, *args, *out = NULL;
    int32_t rc;
    (void)nlhs;
    need_args(nrhs, 3, "run_json");
    name = get_text(prhs[1], "name");
    args = get_text(prhs[2], "args");
    rc = g_api->run_json(name, args, &out);
    mxFree(name);
    mxFree(args);
    if (rc != 0) {
        g_api->free_string(out);
        check(rc);
    }
    plhs[0] = take_rust_string(out);
}

static void cmd_bench_c(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *args, *out = NULL;
    int32_t rc;
    (void)nlhs;
    need_args(nrhs, 2, "bench_c");
    args = get_text(prhs[1], "args");
    rc = g_api->bench_extern(fhn_aot_rhs, fhn_aot_jac, args, &out);
    mxFree(args);
    if (rc != 0) {
        g_api->free_string(out);
        check(rc);
    }
    plhs[0] = take_rust_string(out);
}

static void cmd_s3_build(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *args;
    uint64_t h = 0;
    size_t n = 0, nnz = 0;
    int32_t rc;
    need_args(nrhs, 2, "s3_build");
    args = get_text(prhs[1], "args");
    rc = g_api->s3_build(args, &h, &n, &nnz);
    mxFree(args);
    check(rc);
    plhs[0] = uint64_scalar(h);
    if (nlhs > 1)
        plhs[1] = mxCreateDoubleScalar((double)n);
    if (nlhs > 2)
        plhs[2] = mxCreateDoubleScalar((double)nnz);
}

static void cmd_s3_copy(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t h;
    size_t n, nnz;
    mxArray *out[4];
    int k;
    need_args(nrhs, 4, "s3_copy");
    h = get_handle(prhs[1]);
    n = get_size(prhs[2], "n");
    nnz = get_size(prhs[3], "nnz");
    out[0] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    out[1] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    out[2] = mxCreateDoubleMatrix(nnz, 1, mxREAL);
    out[3] = mxCreateDoubleMatrix(n, 1, mxREAL);
    check(g_api->s3_copy(h, n, nnz, mxGetDoubles(out[0]), mxGetDoubles(out[1]),
                         mxGetDoubles(out[2]), mxGetDoubles(out[3])));
    for (k = 0; k < 4; k++) {
        if (k == 0 || k < nlhs)
            plhs[k] = out[k];
        else
            mxDestroyArray(out[k]);
    }
}

static void cmd_s3_solve(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint64_t h;
    size_t n;
    char *args, *json = NULL;
    mxArray *x;
    int32_t rc;
    need_args(nrhs, 4, "s3_solve");
    h = get_handle(prhs[1]);
    n = get_size(prhs[2], "n");
    args = get_text(prhs[3], "args");
    x = mxCreateDoubleMatrix(n, 1, mxREAL);
    rc = g_api->s3_solve(h, n, args, mxGetDoubles(x), &json);
    mxFree(args);
    if (rc != 0) {
        g_api->free_string(json);
        check(rc);
    }
    plhs[0] = x;
    if (nlhs > 1)
        plhs[1] = take_rust_string(json);
    else
        g_api->free_string(json);
}

static mxArray *module_file_name(HMODULE m)
{
    static wchar_t p[32768];
    static char p8[3 * 32768];
    DWORD len = GetModuleFileNameW(m, p, ARRAYSIZE(p));
    if (len == 0 || len >= ARRAYSIZE(p))
        fail_win32("cannot read a module path", GetLastError());
    to_utf8(p, p8, sizeof p8);
    return mxCreateString(p8);
}

static void utf8_to_wide(const char *s, wchar_t *w, int w_size, const char *what)
{
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, w_size) == 0)
        fail("phasecont:engine:badArg", "%s is not valid UTF-8 or is too long", what);
}

static void cmd_module_path(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    static wchar_t wname[32768];
    char *name;
    HMODULE m;
    (void)nlhs;
    need_args(nrhs, 2, "module_path");
    name = get_text(prhs[1], "name");
    utf8_to_wide(name, wname, ARRAYSIZE(wname), "name");
    mxFree(name);
    m = GetModuleHandleW(wname);
    plhs[0] = m ? module_file_name(m) : mxCreateString("");
}

/* Loads a DLL by absolute path and keeps it loaded, so a later load of the
 * same file name gets this module. */
static void cmd_preload(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    static wchar_t wpath[32768];
    static char what[3 * 32768 + 64];
    char *path;
    HMODULE m;
    (void)nlhs;
    need_args(nrhs, 2, "preload");
    path = get_text(prhs[1], "path");
    utf8_to_wide(path, wpath, ARRAYSIZE(wpath), "path");
    snprintf(what, sizeof what, "cannot load %s", path);
    mxFree(path);
    m = LoadLibraryExW(wpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!m)
        fail_win32(what, GetLastError());
    plhs[0] = module_file_name(m);
}

typedef void (*cmd_fn)(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

static const struct {
    const char *name;
    cmd_fn fn;
    int needs_api;
} k_cmds[] = {
    {"info", cmd_info, 1},
    {"scale", cmd_scale, 1},
    {"panic", cmd_panic, 1},
    {"panic_in_worker", cmd_panic_in_worker, 1},
    {"test_handle", cmd_test_handle, 1},
    {"free_handle", cmd_free_handle, 1},
    {"run_json", cmd_run_json, 1},
    {"bench_c", cmd_bench_c, 1},
    {"s3_build", cmd_s3_build, 1},
    {"s3_copy", cmd_s3_copy, 1},
    {"s3_solve", cmd_s3_solve, 1},
    {"module_path", cmd_module_path, 0},
    {"preload", cmd_preload, 0},
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char cmd[64];
    char *text;
    size_t k;
    if (nrhs < 1)
        fail("phasecont:engine:badArg", "usage: pckern(command, ...)");
    text = get_text(prhs[0], "the command");
    strncpy_s(cmd, sizeof cmd, text, _TRUNCATE);
    mxFree(text);
    for (k = 0; k < ARRAYSIZE(k_cmds); k++) {
        if (strcmp(cmd, k_cmds[k].name) == 0) {
            if (k_cmds[k].needs_api)
                load_api();
            k_cmds[k].fn(nlhs, plhs, nrhs, prhs);
            return;
        }
    }
    fail("phasecont:engine:badArg", "unknown command \"%s\"", cmd);
}
