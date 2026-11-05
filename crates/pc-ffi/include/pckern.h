/*
 * pckern.h: C interface of the PhaseCont engine library (pckern).
 * SPDX-License-Identifier: Apache-2.0
 *
 * The library exports one symbol, pc_api_v1(), which returns a table of
 * function pointers. This is ABI v1. The table is frozen: later versions may
 * only append members, so a caller must accept a struct_size at least as large
 * as sizeof(pc_api_v1_t). The caller loads the library at run time and looks
 * the symbol up by name (see PC_API_V1_SYMBOL); do not link against it.
 *
 * Rules for the caller:
 *  - Every function returns an error code (PC_OK or PC_ERR_*), except
 *    free_string, last_error and version. The message of the last failure is
 *    kept in a thread-local slot that last_error() reads. It stays valid until
 *    the next call on the same thread and is "" after a call that succeeded.
 *  - Strings returned through char** are owned by the caller and must be freed
 *    with free_string. The strings from last_error and version are not freed.
 *  - Out-pointers are set to 0 or NULL before any check that can fail, so the
 *    caller can always read them after the call.
 *  - Handles are opaque uint64_t ids that the library checks against a
 *    registry. 0 and freed ids give PC_ERR_BAD_HANDLE. Never store a handle
 *    in a double.
 *  - Buffers are copied during the call. The library never keeps a pointer
 *    into caller memory after the call returns.
 *  - free_handle on a running run cancels it and joins its worker, so it can
 *    block until the worker reaches its next check between Newton iterations.
 *
 * Layout constants below are for 64-bit targets and are checked at compile
 * time (define PCKERN_NO_LAYOUT_CHECK to skip) and by a Rust test.
 */
#ifndef PCKERN_H
#define PCKERN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PC_ABI_VERSION 1

/* Error codes. The gateway raises "load" and "unknown" itself. */
#define PC_OK 0
#define PC_ERR_PANIC 1          /* a panic was caught, including on a run worker */
#define PC_ERR_BAD_ARG 2
#define PC_ERR_BAD_HANDLE 3
#define PC_ERR_INTERNAL 4
#define PC_ERR_MODEL 5
#define PC_ERR_BAD_PROBLEM 6
#define PC_ERR_NON_AUTONOMOUS 7
#define PC_ERR_NUMERIC 8        /* singular matrix, Newton failure, step too small */
#define PC_ERR_GPU_UNAVAILABLE 9 /* no such backend or device, or its library is missing */
#define PC_ERR_GPU_COMPILE 10   /* the device compiler refused the kernel */
#define PC_ERR_GPU_LAUNCH 11    /* a launch or a copy to or from the device failed */
#define PC_ERR_GPU_TIMEOUT 12   /* one launch ran past the time limit */
#define PC_ERR_UNSUPPORTED_MODEL 13 /* the model uses something a kernel cannot lower */

/* Selectors for the "what" argument of run_copy. */
#define PC_COPY_POINTS 0
#define PC_COPY_SPECIALS 1

/* Selectors for the "what" argument of ens_copy. Lengths are exact: states is
 * members * samples * n values (member, then sample, then state), times is samples
 * values, and the other four are members values. status holds 0 ok, 1 bounds,
 * 2 step too small, 3 max steps, 4 not finite. */
#define PC_ENS_STATES 0
#define PC_ENS_TIMES 1
#define PC_ENS_STATUS 2
#define PC_ENS_T_REACHED 3
#define PC_ENS_ACCEPTED 4
#define PC_ENS_REJECTED 5

/*
 * problem_json (run_start): the problem object without "u0" and "params",
 * which come only as buffers (either key is PC_ERR_BAD_PROBLEM here):
 *   {"type":"EP","free":"r","start_is_special":false,"options":{...},
 *    "switch":"BP1"}            ("switch" is optional)
 * A malformed switch label fails in run_start. A well-formed label that does
 * not exist (for example "XY1") is reported by run_poll as PC_ERR_BAD_PROBLEM.
 *
 * Rows copied by run_copy have "stride" doubles, read from status_json; the
 * stride depends on the problem type. EP rows have stride 3n + 6, in this
 * order:
 *   [alpha, u(1..n), eig_re(1..n), eig_im(1..n),
 *    psi_LP, psi_BP, psi_H, h, newton_iters]
 * docs/SPEC.md section 10 lists the rows of the LP, H, LC, LPC, PD and NS
 * types. Missing values are NaN. Tangents stay inside the library.
 *
 * status_json (run_poll):
 *   {"state":"running"|"done"|"cancelled","n":3,"stride":15,
 *    "branches":[{"points":120,"stop":"MaxSteps"|null,"warnings":[...],
 *      "specials":[{"label":"H","index":57,"coefficient":0.1,"omega":9.6,
 *                   "normalization":"...","refined":true}]}]}
 * "specials" appear once the branch has stopped. "stop" is null for the last
 * branch while the run is running. Otherwise it is "MaxSteps", "Bounds",
 * "StepTooSmall", "Singular", "Cancelled" or "Error".
 */

typedef struct pc_api_v1_t {
  uint32_t abi_version;     /* PC_ABI_VERSION */
  uint32_t struct_size;     /* sizeof the table in this build of pckern */

  /* Parses model text (UTF-8, at most 16 MiB). *model is 0 and *diag_json is
   * NULL unless set. diag_json may be NULL; otherwise it gets a diagnostics
   * JSON string on success and on PC_ERR_MODEL (free it with free_string). */
  int32_t (*model_from_ode)(const char *text, size_t len, uint64_t *model,
                            char **diag_json);

  /* Builds the evaluator. opts_json is NULL or an object with the keys
   * "evaluator" ("jit", "fallback" or "aot"), "aot_lib" (needed by and only
   * valid with "aot") and "emit_c" (a folder for the model's C text); any
   * other key is PC_ERR_BAD_ARG. NULL means "jit". See docs/SPEC.md section 10. */
  int32_t (*model_compile)(uint64_t model, const char *opts_json);

  /* Evaluates f(x, p). Needs a compiled model. nx and nout must equal n, np
   * must equal the number of parameters. A model that uses t
   * gives PC_ERR_NON_AUTONOMOUS here: call model_rhs_t. */
  int32_t (*model_rhs)(uint64_t model, const double *x, size_t nx,
                       const double *p, size_t np, double *out, size_t nout);

  /* Starts a continuation run on its own worker thread. Needs a compiled,
   * autonomous model. np must equal the number of parameters. nu is n, except
   * for a cycle run that starts from an orbit (LC) or from a point of an LC
   * branch (an LC run from a PD point, or an LPC, PD or NS curve): there u0
   * holds the orbit and the period, n (N m + 1) + 1 values, with N and m of
   * the start orbit. The run keeps its own copy of the model. *run is 0 unless
   * set. */
  int32_t (*run_start)(uint64_t model, const char *problem_json,
                       const double *u0, size_t nu, const double *p, size_t np,
                       uint64_t *run);

  /* Writes the status_json string (free it with free_string). On a non-zero
   * code *status_json is NULL: PC_ERR_PANIC if the worker panicked, or the
   * kind code of the failure, with the message in last_error. */
  int32_t (*run_poll)(uint64_t run, char **status_json);

  /* Copies up to count rows from row start of the given branch into buf.
   * what is PC_COPY_POINTS or PC_COPY_SPECIALS. buf_len is in doubles and
   * must equal count * stride, with stride from status_json, so a caller with
   * the wrong stride gets PC_ERR_BAD_ARG. buf may be NULL when count is 0.
   * *rows gets the number of rows written (0 on failure). */
  int32_t (*run_copy)(uint64_t run, uint32_t branch, uint32_t what,
                      size_t start, size_t count, double *buf, size_t buf_len,
                      size_t *rows);

  /* Asks the run to stop. Returns at once; poll for state "cancelled". */
  int32_t (*run_cancel)(uint64_t run);

  /* Frees a model or run handle. A running run is cancelled and joined first. */
  int32_t (*free_handle)(uint64_t handle);

  /* Frees a string returned through char**. NULL is allowed. */
  void (*free_string)(char *s);

  /* The message of the last failure on this thread. Not freed by the caller. */
  const char *(*last_error)(void);

  /* Writes the model description JSON (free it with free_string):
   *   {"n":3,"n_p":3,"variables":[...],"parameters":[...],"values":[...],
   *    "init":[...],"aux":[...],"autonomous":true,"evaluator":null,
   *    "jit_max_frame":0}
   * n and n_p are the state and parameter counts. variables and parameters
   * are the names, values the parameter values from the file and init the
   * initial values from the file. aux lists the names of the auxiliary quantities and
   * autonomous is false when the right-hand side uses t. evaluator is "jit",
   * "fallback" or null before model_compile. jit_max_frame is the largest
   * JIT stack frame compiled so far (0 before model_compile). */
  int32_t (*model_info)(uint64_t model, char **info_json);

  /* Newton solve from u0 with the given parameters. opts_json is NULL or the
   * engine options object. u_out gets n values and eig_out gets 2n values as
   * [re(1..n), im(1..n)]. The two buffers must not overlap. */
  int32_t (*model_equilibrium)(uint64_t model, const double *u0, size_t nu,
                               const double *p, size_t np,
                               const char *opts_json, double *u_out,
                               size_t nu_out, double *eig_out,
                               size_t neig_out);

  /* Test hook. where 0 panics on the calling thread (returns PC_ERR_PANIC).
   * where 1 starts a run whose worker panics and writes its id to *run; the
   * panic then shows up in run_poll. Other values are PC_ERR_BAD_ARG. */
  int32_t (*debug_panic)(int32_t where, uint64_t *run);

  /* "pckern <crate version> abi 1". Not freed by the caller. */
  const char *(*version)(void);

  /* Evaluates f(x, p, t). Same rules as model_rhs; t must be finite
   * (PC_ERR_BAD_ARG otherwise). A model that does not use t ignores it. */
  int32_t (*model_rhs_t)(uint64_t model, double t, const double *x, size_t nx,
                         const double *p, size_t np, double *out, size_t nout);

  /* Translates a model written in another form into .ode text. format is
   * PC_IMPORT_SBML (bytes is SBML, UTF-8) or PC_IMPORT_MATLAB (bytes is the
   * text of func2str(h) or of a function file, UTF-8). At most 17 MiB. opts_json
   * is NULL or a JSON object. SBML takes no options (NULL or {}). MATLAB takes
   *   {"captures":[{"name":"k","value":[1,2]}],      values a handle captured
   *    "params":[{"name":"a","value":0.1}],          p(1), p(2), ... in order
   *    "state_names":["v","w"],                      one per state
   *    "init":[0,1],                                 one per state
   *    "title":"text"}
   * Every key is optional. A one-element list may be written as its item, as
   * MATLAB's jsonencode does. Any other key is PC_ERR_BAD_ARG. *result_json is NULL
   * unless set; free it with free_string. On PC_OK it is
   *   {"ok":true,"ode":"...","warnings":["..."]}
   * A model the importer refuses (syntax, unsupported, undeclared, missingValue,
   * mismatch, nameCollision, tooLarge, noDynamics, xml) returns PC_ERR_MODEL and
   *   {"ok":false,"error":{"id":"phasecont:import:<kind>","kind":"<kind>",
   *                        "message":"...","line":3,"col":9}}
   * with line and col only when the error has a position. Nothing defaults to 0:
   * a missing initial value or an undeclared name is an error. The importers
   * read text and never run it. */
  int32_t (*import)(uint32_t format, const char *bytes, size_t len,
                    const char *opts_json, char **result_json);

  /* Ensemble runs: many trajectories of one model, on the CPU or a GPU. These
   * five members were appended after import; they change nothing before it. */

  /* Lists every backend and its devices as JSON, for example
   *   {"backends":[{"backend":"cpu","state":"available"},
   *                {"backend":"cuda","state":"unavailable","why":"..."},
   *                {"backend":"metal","state":"not_built"}],
   *    "devices":[{"backend":"cpu","index":0,"name":"...","driver":"...",
   *                "fp32":true,"fp64":true,"memory_bytes":0}]}
   * *out_json is NULL unless set; free it with free_string. */
  int32_t (*ens_devices)(char **out_json);

  /* Checks an ensemble job and starts it on a worker thread. job_json is
   *   {"backend":"cpu"|"cuda"|"opencl"|"metal","device":0,
   *    "precision":"single"|"double","t0":0,"t1":10,
   *    "method":{"type":"rk4","dt":0.01}
   *            or {"type":"dp5","rtol":1e-6,"atol":1e-9,"hmax":null},
   *    "output":"final" or {"samples":[t, ...]},
   *    "bounds":null or a positive number,"max_steps":100000,
   *    "params":"shared"|"per_member"}
   * Every key but "device" is required; nothing is filled in. x0 holds nx / n
   * states per member, one member after the other, where n is the number of
   * states. params holds one set of the model's parameters, or one set per
   * member when "params" is "per_member". A job that is wrong in any way,
   * including a tolerance that single precision cannot resolve (the message
   * starts "tolerance too tight"), is PC_ERR_BAD_PROBLEM and nothing starts.
   * A missing backend or device is PC_ERR_GPU_UNAVAILABLE. *out_ens is 0 unless
   * set; free the handle with free_handle, which cancels a run that is still
   * going and waits for its worker. */
  int32_t (*ens_start)(uint64_t model, const char *job_json, const double *x0,
                       size_t nx, const double *params, size_t np,
                       uint64_t *out_ens);

  /* {"state":"running"} or {"state":"cancelled"}, or on a finished run
   *   {"state":"done","members":M,"n":N,"samples":S,
   *    "counts":{"ok":0,"bounds":0,"step_too_small":0,"max_steps":0,
   *              "nonfinite":0},
   *    "label":{"backend":"cpu","device":"...","driver":"...",
   *             "precision":"double","single_precision":false}}
   * A run that failed returns its error code here and *status_json stays NULL:
   * PC_ERR_PANIC if the worker panicked, or a GPU code. Free the string with
   * free_string. */
  int32_t (*ens_poll)(uint64_t ens, char **status_json);

  /* Copies one array of a finished run (PC_ENS_*) into buf. buf_len must be
   * the exact length. A run that has not finished, or was cancelled, is
   * PC_ERR_BAD_ARG. */
  int32_t (*ens_copy)(uint64_t ens, uint32_t what, double *buf, size_t buf_len);

  /* Asks the worker to stop at its next check. Safe to call twice and after the
   * run ended. The run then polls as cancelled. */
  int32_t (*ens_cancel)(uint64_t ens);

  /* Renders a figure or table document (UTF-8 JSON, NUL-terminated) to LaTeX.
   * kind is PC_LATEX_*. On success *result_json is
   * {"ok":true,"tex":"...","data_files":[{"name":"...","content":"..."}]}.
   * A document the renderer rejects returns PC_ERR_BAD_PROBLEM and sets
   * *result_json to {"ok":false,"error":{"id":"phasecont:latex:<kind>",
   * "kind":"<kind>","message":"..."}}. An unknown kind or a bad pointer is
   * PC_ERR_BAD_ARG and leaves *result_json NULL. Free the string with
   * free_string. The call reads and writes no files. */
  int32_t (*latex_render)(uint32_t kind, const char *doc_json,
                          char **result_json);
} pc_api_v1_t;

#define PC_IMPORT_SBML 1
#define PC_IMPORT_MATLAB 2

#define PC_LATEX_PGFPLOTS 1
#define PC_LATEX_TABLE 2

/* The entry point, for loaders that resolve it by name. It returns NULL only
 * if a panic was caught. It is deliberately not declared as an imported
 * function. */
#define PC_API_V1_SYMBOL "pc_api_v1"
typedef const pc_api_v1_t *(*pc_api_v1_fn)(void);

/* Layout of pc_api_v1_t on 64-bit targets, in bytes. */
#define PC_API_V1_SIZE 184
#define PC_API_V1_OFFSET_ABI_VERSION 0
#define PC_API_V1_OFFSET_STRUCT_SIZE 4
#define PC_API_V1_OFFSET_MODEL_FROM_ODE 8
#define PC_API_V1_OFFSET_MODEL_COMPILE 16
#define PC_API_V1_OFFSET_MODEL_RHS 24
#define PC_API_V1_OFFSET_RUN_START 32
#define PC_API_V1_OFFSET_RUN_POLL 40
#define PC_API_V1_OFFSET_RUN_COPY 48
#define PC_API_V1_OFFSET_RUN_CANCEL 56
#define PC_API_V1_OFFSET_FREE_HANDLE 64
#define PC_API_V1_OFFSET_FREE_STRING 72
#define PC_API_V1_OFFSET_LAST_ERROR 80
#define PC_API_V1_OFFSET_MODEL_INFO 88
#define PC_API_V1_OFFSET_MODEL_EQUILIBRIUM 96
#define PC_API_V1_OFFSET_DEBUG_PANIC 104
#define PC_API_V1_OFFSET_VERSION 112
#define PC_API_V1_OFFSET_MODEL_RHS_T 120
#define PC_API_V1_OFFSET_IMPORT 128
#define PC_API_V1_OFFSET_ENS_DEVICES 136
#define PC_API_V1_OFFSET_ENS_START 144
#define PC_API_V1_OFFSET_ENS_POLL 152
#define PC_API_V1_OFFSET_ENS_COPY 160
#define PC_API_V1_OFFSET_ENS_CANCEL 168
#define PC_API_V1_OFFSET_LATEX_RENDER 176

#ifndef PCKERN_NO_LAYOUT_CHECK
/* C89 negative-array checks: _Static_assert needs /std:c11 on MSVC. */
typedef char pc_check_ptr[(sizeof(void *) == 8) ? 1 : -1];
typedef char pc_check_size[(sizeof(pc_api_v1_t) == PC_API_V1_SIZE) ? 1 : -1];
#define PC_CHECK_OFFSET(field, off) \
  typedef char pc_check_off_##field[(offsetof(pc_api_v1_t, field) == (off)) ? 1 : -1]
PC_CHECK_OFFSET(abi_version, PC_API_V1_OFFSET_ABI_VERSION);
PC_CHECK_OFFSET(struct_size, PC_API_V1_OFFSET_STRUCT_SIZE);
PC_CHECK_OFFSET(model_from_ode, PC_API_V1_OFFSET_MODEL_FROM_ODE);
PC_CHECK_OFFSET(model_compile, PC_API_V1_OFFSET_MODEL_COMPILE);
PC_CHECK_OFFSET(model_rhs, PC_API_V1_OFFSET_MODEL_RHS);
PC_CHECK_OFFSET(run_start, PC_API_V1_OFFSET_RUN_START);
PC_CHECK_OFFSET(run_poll, PC_API_V1_OFFSET_RUN_POLL);
PC_CHECK_OFFSET(run_copy, PC_API_V1_OFFSET_RUN_COPY);
PC_CHECK_OFFSET(run_cancel, PC_API_V1_OFFSET_RUN_CANCEL);
PC_CHECK_OFFSET(free_handle, PC_API_V1_OFFSET_FREE_HANDLE);
PC_CHECK_OFFSET(free_string, PC_API_V1_OFFSET_FREE_STRING);
PC_CHECK_OFFSET(last_error, PC_API_V1_OFFSET_LAST_ERROR);
PC_CHECK_OFFSET(model_info, PC_API_V1_OFFSET_MODEL_INFO);
PC_CHECK_OFFSET(model_equilibrium, PC_API_V1_OFFSET_MODEL_EQUILIBRIUM);
PC_CHECK_OFFSET(debug_panic, PC_API_V1_OFFSET_DEBUG_PANIC);
PC_CHECK_OFFSET(version, PC_API_V1_OFFSET_VERSION);
PC_CHECK_OFFSET(model_rhs_t, PC_API_V1_OFFSET_MODEL_RHS_T);
PC_CHECK_OFFSET(import, PC_API_V1_OFFSET_IMPORT);
PC_CHECK_OFFSET(ens_devices, PC_API_V1_OFFSET_ENS_DEVICES);
PC_CHECK_OFFSET(ens_start, PC_API_V1_OFFSET_ENS_START);
PC_CHECK_OFFSET(ens_poll, PC_API_V1_OFFSET_ENS_POLL);
PC_CHECK_OFFSET(ens_copy, PC_API_V1_OFFSET_ENS_COPY);
PC_CHECK_OFFSET(ens_cancel, PC_API_V1_OFFSET_ENS_CANCEL);
PC_CHECK_OFFSET(latex_render, PC_API_V1_OFFSET_LATEX_RENDER);
#endif

#ifdef __cplusplus
}
#endif

#endif /* PCKERN_H */
