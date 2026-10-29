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

/* Selectors for the "what" argument of run_copy. */
#define PC_COPY_POINTS 0
#define PC_COPY_SPECIALS 1

/*
 * problem_json (run_start): the problem object without "u0" and "params",
 * which come only as buffers (either key is PC_ERR_BAD_PROBLEM here):
 *   {"type":"EP","free":"r","start_is_special":false,"options":{...},
 *    "switch":"BP1"}            ("switch" is optional)
 * A malformed switch label fails in run_start. A well-formed label that does
 * not exist (for example "XY1") is reported by run_poll as PC_ERR_BAD_PROBLEM.
 *
 * Rows copied by run_copy have stride 3n + 6 doubles, in this order:
 *   [alpha, u(1..n), eig_re(1..n), eig_im(1..n),
 *    psi_LP, psi_BP, psi_H, h, newton_iters]
 * Missing values are NaN. Tangents stay inside the library.
 *
 * status_json (run_poll):
 *   {"state":"running"|"done"|"cancelled","n":3,"stride":15,
 *    "branches":[{"points":120,"stop":"MaxSteps"|null,"warnings":[...],
 *      "specials":[{"label":"H","index":57,"coefficient":0.1,"omega":9.6,
 *                   "normalization":"...","refined":true}]}]}
 * "specials" appear once the branch has stopped. "stop" is null for the last
 * branch while the run is running.
 */

typedef struct pc_api_v1_t {
  uint32_t abi_version;     /* PC_ABI_VERSION */
  uint32_t struct_size;     /* sizeof the table in this build of pckern */

  /* Parses model text (UTF-8, at most 16 MiB). *model is 0 and *diag_json is
   * NULL unless set. diag_json may be NULL; otherwise it gets a diagnostics
   * JSON string on success and on PC_ERR_MODEL (free it with free_string). */
  int32_t (*model_from_ode)(const char *text, size_t len, uint64_t *model,
                            char **diag_json);

  /* Builds the evaluator. opts_json is NULL or {"evaluator":"jit"|"fallback"};
   * any other key is PC_ERR_BAD_ARG. NULL means "jit". */
  int32_t (*model_compile)(uint64_t model, const char *opts_json);

  /* Evaluates f(x, p). Needs a compiled model. nx and nout must equal n, np
   * must equal the number of parameters. */
  int32_t (*model_rhs)(uint64_t model, const double *x, size_t nx,
                       const double *p, size_t np, double *out, size_t nout);

  /* Starts a continuation run on its own worker thread. Needs a compiled,
   * autonomous model. nu must equal n and np the number of parameters. The
   * run keeps its own copy of the model. *run is 0 unless set. */
  int32_t (*run_start)(uint64_t model, const char *problem_json,
                       const double *u0, size_t nu, const double *p, size_t np,
                       uint64_t *run);

  /* Writes the status_json string (free it with free_string). On a non-zero
   * code *status_json is NULL: PC_ERR_PANIC if the worker panicked, or the
   * kind code of the failure, with the message in last_error. */
  int32_t (*run_poll)(uint64_t run, char **status_json);

  /* Copies up to count rows from row start of the given branch into buf.
   * what is PC_COPY_POINTS or PC_COPY_SPECIALS. buf_len is in doubles and
   * must be at least count * (3n + 6). *rows gets the number of rows written
   * (0 on failure). */
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

  /* Writes the model description JSON (names, values, init, aux, n, n_p,
   * autonomous, evaluator, jit_max_frame). Free it with free_string. */
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
} pc_api_v1_t;

/* The entry point, for loaders that resolve it by name. It returns NULL only
 * if a panic was caught. It is deliberately not declared as an imported
 * function. */
#define PC_API_V1_SYMBOL "pc_api_v1"
typedef const pc_api_v1_t *(*pc_api_v1_fn)(void);

/* Layout of pc_api_v1_t on 64-bit targets, in bytes. */
#define PC_API_V1_SIZE 120
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
#endif

#ifdef __cplusplus
}
#endif

#endif /* PCKERN_H */
