# PhaseCont technical spec

This file is the source of truth for what the engine computes and how each phase is accepted. `CLAUDE.md` holds the project rules. Where a formula here conflicts with a cited reference, the reference wins: stop, note the conflict in `docs/STATUS.md`, and ask.

## 1. Data flow

```
.ode text
  -> pc-ode      parse, validate, diagnostics, AST
  -> pc-ir       expression DAG, constant folding, CSE, symbolic derivatives
  -> pc-jit      Cranelift native functions (default)
     pc-codegen  C (optional AOT), CUDA C, OpenCL C, MSL, LaTeX, MATLAB reference
  -> pc-engine   continuation, test functions, normal forms, limit cycles
  -> pc-ffi      C ABI, handle registry, panic barrier
  -> mex gateway
  -> toolbox/+phasecont  MATLAB API, apps, plots, sessions
```

`pc-cli` drives the same Rust crates without MATLAB, so the engine can be tested and benchmarked on CI runners that have no MATLAB license.

## 2. `.ode` support for v1

The rules below were checked against the XPPAUT manuals (v5.0, v5.41, v8 notes) and measured with XPPAUT 8.0 on our own probe files (`references/xpp-probes/`). The full language reference, with the precedence table, the builtin semantics table and the diagnostic codes, is `crates/pc-ode/docs/language.md`.

Parsing rules:
- **Keywords:** only the first letter is significant for the nine declarations `markov`, `parameter`, `wiener`, `table`, `aux`, `init`, `bndry`, `global` and `done` (`p`, `par` and `pzz` all declare parameters). Other keywords (`number`, `set`, `options`, `only`, `special`, `solv`, `export`, `volt`) need the full word. A first word is a keyword only when a blank follows it and the next non-blank character is not `=`, so `p=3` and `p = 3` define a fixed quantity named `p`.
- **Case:** the language is case-insensitive.
- **Names:** XPPAUT 8.0 truncates names to 10 characters. A longer name gives a portability warning, not an error.
- **Line continuation:** a backslash at the end of a line continues it.
- **Order:** names may be used before their declaration, except fixed quantities: XPPAUT evaluates them in file order, so a fixed quantity that uses one defined on a later line is an error.
- **Spaces around `=`:** accepted in `par`, `number` and `init` lists and in formulas. In `@` lists XPPAUT 8.0 accepts the line but ignores the option, so PhaseCont gives a warning with a fix-it.
- **Declared values:** `par`, `number`, `init` and `x(0)=` values are signed numeric literals (XPPAUT reads only a leading number). A repeated `par` is an error.
- **Comments:** `#` at the start of a line. A trailing `#` comment is accepted after `par`, `number` and `@` lists and is an error elsewhere, as in XPPAUT 8.0.
- **Arrays:** range lines (`x[1..10]'=-x[j]`) and `%[1..n]` ... `%` blocks are expanded during loading.
- **Old-style files**, whose first line starts with a digit, are an error.

| Construct | Example | v1 status |
|---|---|---|
| ODE | `x'=...`, `dx/dt=...` | full |
| Map | `x(t+1)=...` | parse, simulate; continuation of maps later |
| Initial data | `init x=1,y=2`, `x(0)=1` | full |
| Parameters | `par a=1,b=2` | full |
| Numbers, derived parameters | `number c=3`, `!d=a*b` | full |
| Fixed quantities | `w=...` | full (inlined in the DAG) |
| Auxiliary output | `aux I=...` | full |
| User functions | `f(x,y)=...` | full (inlined) |
| Tables | `table h % 101 0 6.283 sin(t)` | inline formula tables; file tables later |
| Options | `@ xp=V, yp=w, total=100, dt=.05, meth=stiff` | full |
| Include, sets | `#include`, `set name {...}` | full |
| Arrays | `x[1..n]'=...`, `%[1..n]` ... `%` | full (expanded) |
| Global flags | `global -1 {u-.2} {m=.5*m}` | simulation only |
| Markov, Wiener | `markov z 2`, `wiener w` | simulation only |
| Delay | `delay(x,tau)` | warn; v2 |
| Volterra, BVP | `volt`, `bdry` | warn; unsupported |
| DAEs | `0=...`, `solv` | warn; unsupported |
| Other constructs | `special`, `export`, `only`, `options`, `shift`, `del_shft`, `sum(..)of(..)`, `hom_bcs` | warn; unsupported |
| End | `done` | full |

**Expressions.**
- Operators: `+ - * /`, `^` and `**` for powers, prefix `-` and `not`, comparisons `< > <= >= ==`, and `&` and `|`. XPPAUT's precedence differs from C: `&` binds like `*`, `|` like `+`, comparisons bind tighter than arithmetic, and `^` is left-associative. PhaseCont follows XPPAUT and warns where the reading is surprising. The measured table is in `language.md`.
- A number followed directly by a name or `(` (`2v`) multiplies, with a portability warning, since XPPAUT 8.0 rejects it.
- Built-in functions: `sin cos tan asin acos atan atan2 sinh cosh tanh exp ln log log10 sqrt abs heav sign mod min max flr ceil erf erfc lgamma besselj bessely`, and `if(c)then(a)else(b)`. `ceil` gets a portability warning.
- `ran`, `normal` and `poisson` make a model simulation-only, as do `markov`, `wiener` and `global`.
- The exact meaning of each function (for example that `log` is natural and `atan2(y,x)` uses C's order), with its source, is in the semantics table in `language.md`. Division by zero and whether `if` evaluates both branches are open questions for P2.

**Non-smooth models.**
- Any of `heav sign abs mod flr ceil min max if`, or a table, marks a model as non-smooth.
- Equilibrium continuation is allowed, with a warning, as long as no switching surface is crossed.
- Derivatives use one-sided rules and the result is flagged.

**Diagnostics.**
- Each diagnostic has a code (`ODE0001`...), a line and column, a message, and an optional fix-it.
- The same diagnostics feed the MATLAB import report and, later, the VS Code language server.

**Round trip.** For every model, `parse(write(parse(s)))` must equal `parse(s)` (checked with `proptest`).

## 3. IR and derivatives (pc-ir)

**DAG structure.**
- A hash-consed DAG of `f64` operations, with constant folding and common subexpression elimination.
- Fixed quantities and user functions are inlined.

**Derivatives produced by symbolic forward differentiation on the DAG.** Each one is a new DAG that goes through CSE again:
- $f(u,\alpha)$
- the Jacobian $A = f_u$ ($n \times n$)
- the parameter Jacobian $f_\alpha$ ($n \times n_p$)
- the multilinear forms $B(v_1,v_2) = D^2 f(u)[v_1,v_2]$ and $C(v_1,v_2,v_3) = D^3 f(u)[v_1,v_2,v_3]$
- directional derivatives up to order 5, for normal forms
- the mixed parameter derivatives needed by branch switching

**Complex directions.** Directional derivatives must accept complex direction vectors, because $q$ and $p$ in the normal forms are complex. Do this either by splitting real and imaginary parts and using multilinearity, or by emitting a complex-valued entry point.

**Validation uses two independent evaluators:**
1. A tree-walking dual-number interpreter on the AST. Its results must match the generated derivatives to within 4 ulp per entry, or $10^{-13}$ relative error.
2. Central finite differences on random inputs, with relative error at most $10^{-6}$ for $A$ and at most $10^{-4}$ for $B$ and $C$.

## 4. JIT and AOT (pc-jit, pc-codegen)

**Entry points.** Lower each needed DAG to Cranelift IR as an `extern "C"` function. A typical signature is `fn(x: *const f64, p: *const f64, out: *mut f64)`. Directional entries take extra vector pointers. Matrix outputs such as the Jacobian are written column major.

**Code generation flags.** Set `is_pic` only on x86_64 and set `use_colocated_libcalls` to false, as `JITBuilder::new` does. Set `enable_probestack` to true and `probestack_strategy` to `inline`. Cranelift leaves stack probes off by default, Windows needs them for frames over 4096 bytes, and the default `outline` strategy calls `__cranelift_probestack`, which `pckern` does not provide.

**Transcendentals.** Call Rust `extern "C"` shims around the `f64` methods (`sin`, `exp`, and so on). Register every shim with `JITBuilder::symbol`. Shim bodies must not panic: a panic that leaves an `extern "C"` function aborts the process, JIT frames have no unwind information, and the panic barrier sits outside the JIT call.

**Ownership.** Each compiled model owns its `JITModule`. Dropping a `JITModule` leaves its code pages mapped, so the model handle calls `JITModule::free_memory` when it is freed and on every error path after `JITModule::new`, and only when no thread can still call the compiled functions.

**Equivalence.** JIT output must match the dual-number interpreter to within 4 ulp. Exact equality is not expected, because of fused multiply-add.

**AOT path.** `pc-codegen` emits C with the same signatures. It is compiled through `mex` when a compiler exists and selected with `phasecont.config("Codegen","aot")`. AOT output must match JIT output to within 4 ulp. Compile AOT code with floating-point contraction turned off, as the spike's gcc build does with `-ffp-contract=off`, and store the compiler, its version and its flags with every benchmark or test result that uses AOT code.

**macOS arm64.** Confirm that JIT pages can be mapped executable inside the MATLAB process. Phase 0 ran only on Windows and WSL, so this check moves to Phase 2, which runs it on a Mac with MATLAB or on the macOS runner of the MATLAB job in section 12. `cranelift-jit` 0.137.0 accepts `is_pic=true` only on x86_64. Its default `SystemMemoryProvider` maps code pages read-write without `MAP_JIT` and makes them read-execute with `mprotect`, which the hardened runtime allows only with `com.apple.security.cs.allow-unsigned-executable-memory` or `com.apple.security.cs.disable-executable-page-protection`. `codesign -d --entitlements - <matlabroot>/bin/maca64/MATLAB`, where `<matlabroot>` is the folder that `matlabroot` returns, lists the entitlements MATLAB has. Without either entitlement, `pc-jit` needs its own `JITMemoryProvider`, set with `JITBuilder::memory_provider`, that maps pages with `MAP_JIT`, which needs `com.apple.security.cs.allow-jit`, and switches them with `pthread_jit_write_protect_np`. With none of the three, rule 5 of `CLAUDE.md` still holds: models run on an `f64` evaluator in Rust built from the same code as the dual-number interpreter of section 3, and PhaseCont warns that this is slower than compiled code. The AOT path stays the optional speed-up on Macs that have the Xcode Command Line Tools, selected with `phasecont.config("Codegen","aot")`. No Mac user is required to install a compiler.

## 5. Continuation core (pc-engine)

**Notation.** Let $x = (u,\alpha) \in \mathbb{R}^{n+1}$ and $F(x) = f(u,\alpha)$.

**Tangent.** Solve
$$\begin{pmatrix} F_x(x_k) \\ v_{k-1}^\top \end{pmatrix} v_k = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \qquad v_k \leftarrow \frac{v_k}{\|v_k\|}.$$

**Pseudo-arclength (PALC) step.** Predict $x^{(0)} = x_k + h v_k$, then correct with Newton's method on
$$G(x) = \begin{pmatrix} F(x) \\ v_k^\top (x - x_k) - h \end{pmatrix} = 0.$$

**Convergence and step control.**
- The corrector has converged when $\|\Delta x\|_\infty \le 10^{-10}$ and $\|F\|_\infty \le 10^{-10}$ (both configurable).
- If Newton converges in 3 or fewer iterations, set $h \leftarrow \min(1.3h, h_{\max})$.
- If it fails after 6 iterations (configurable), set $h \leftarrow h/2$ and retry.
- Below $h_{\min}$, stop the branch and report the reason.

**Test functions on equilibrium branches.** $\lambda_i$ are the eigenvalues of $A$.
- **LP:** $\psi_{LP} = v_{k,\alpha}$, the parameter component of the tangent.
- **BP:** $\psi_{BP} = \det \begin{pmatrix} F_x \\ v^\top \end{pmatrix}$.
- **H:** $\psi_H = \prod_{i<j} (\lambda_i + \lambda_j)$, which equals $\det(2A \odot I)$. It vanishes at Hopf points and at neutral saddles. Check the eigenvalues to tell them apart and label neutral saddles but do not report them as Hopf. For $n > 100$, switch to the minimally augmented test function from Govaerts.

**Locating special points.**
1. When a test function changes sign between $x_k$ and $x_{k+1}$, find the zero along the step with the secant method, using a bisection safeguard.
2. Refine the point by solving its defining system (Section 6) with Newton's method.
3. Store the refined point, its eigenvalues and its normal form coefficient on the branch.

**Branch storage.** Columns are $u$, $\alpha$, tangent, eigenvalues (or multipliers), test function values, step size and Newton iteration count. Special points carry a label, an index and a struct of extras.

**Background runs.** A continuation runs on a Rust background thread. MATLAB polls for new points with `pc_run_poll` from a `timer`, so plots update live, and `pc_run_cancel` stops the run. A panic on the background thread is caught on that thread, with its message and source location, and the next `pc_run_poll` reports it as an error.

## 6. Equilibrium bifurcations and normal forms

**Fold (LP).** With $Aq = 0$, $A^\top p = 0$, $p^\top q = 1$ and $\|q\| = 1$, the quadratic coefficient is
$$a = \tfrac{1}{2}\, p^\top B(q,q).$$

**Minimally augmented LP curve** (Govaerts 2000). Solve
$$\begin{pmatrix} A & w_{\text{bor}} \\ v_{\text{bor}}^\top & 0 \end{pmatrix}\begin{pmatrix} q \\ g \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.$$
The LP curve is $F = 0$, $g = 0$ with two free parameters. Update the borders $v_{\text{bor}}, w_{\text{bor}}$ from the current null vectors after each accepted step.

**Hopf.** The eigenvectors satisfy $Aq = i\omega q$ and $A^\top p = -i\omega p$. Use the normalization $\langle q,q\rangle = 1$ and $\langle p,q\rangle = 1$, where $\langle a,b\rangle = \bar a^\top b$. The first Lyapunov coefficient (Kuznetsov) is
$$l_1 = \frac{1}{2\omega}\,\mathrm{Re}\Big(\langle p, C(q,q,\bar q)\rangle - 2\langle p, B(q, A^{-1}B(q,\bar q))\rangle + \langle p, B(\bar q, (2i\omega I - A)^{-1}B(q,q))\rangle\Big).$$
The value of $l_1$ depends on the normalization of $q$; only its sign is invariant. Always report which normalization was used.

**Minimally augmented Hopf curve.** Use the bordered $A^2 + \kappa I$ formulation from Govaerts (2000), with $\kappa = \omega^2$ as an extra unknown. The extra test function $\kappa \to 0$ detects BT.

**Codim-2 points.**
- On LP curves: CP ($a = 0$), BT, ZH.
- On H curves: GH ($l_1 = 0$), BT, ZH, HH.
- Normal form coefficients come from Kuznetsov (1999), "Numerical normalization techniques for all codim 2 bifurcations of equilibria in ODEs."

**Branch switching at BP.** Compute the second tangent from the null space of the extended Jacobian and the algebraic branching equation (Allgower and Georg; Govaerts).

## 7. Limit cycles

**Formulation.** Rescale time to $\tau \in [0,1]$:
$$\dot u = T f(u,\alpha), \qquad u(0) = u(1), \qquad \int_0^1 u(\tau)^\top \dot u_{\text{prev}}(\tau)\, d\tau = 0.$$
The integral is the phase condition.

**Discretization.**
- Use $N$ mesh intervals with $m$ Gauss-Legendre collocation points each. On each interval, $u$ is a degree-$m$ polynomial stored by its values at $m+1$ points, and neighboring intervals share endpoints.
- Unknowns: $n(Nm+1) + 2$ (the mesh values, $T$ and $\alpha$).
- Equations: $nNm$ collocation equations, $n$ periodicity conditions, 1 phase condition and 1 PALC equation.
- Evaluate the phase integral with the same Gauss weights.

**Linear solves by condensation of parameters** (Doedel, Keller and Kernévez 1991; AUTO-07p):
1. On each interval, eliminate the interior unknowns with dense LU (faer).
2. This leaves a block system in the mesh-point values plus the bordering rows and columns.
3. Solve that reduced system with faer's sparse LU. `pc-linalg` chooses between faer's simplicial and supernodal variants itself, starting with the simplicial one, instead of using faer's automatic choice. Tests compare the result with a sparse LU of the full matrix.

Assemble and condense in Rust. Run sequentially by default, and parallelize across intervals in a rayon pool that `pc-linalg` owns, only above a value of $N$ and with a thread count, both of which Phase 5 measures. `pc-linalg` sets faer's global parallelism once and checks a memory budget before large solves, since a failed allocation in `Vec` or in dyn-stack's `MemBuffer::new` aborts the MATLAB process.

**Floquet multipliers.**
- The condensed blocks give the linear map from $u(0)$ to $u(1)$.
- v1: form the monodromy matrix as a product of the blocks, compute its eigenvalues, and check that one multiplier satisfies $|\mu_1 - 1| \le 10^{-6}$. Flag the result when the check fails.
- Later: replace the product with a periodic Schur method (Fairgrieve and Jepson 1991), so that multipliers spanning many orders of magnitude stay accurate.

**Test functions on cycle branches.** The products run over the nontrivial multipliers.
- **LPC:** $\psi_{LPC} = \prod_i (\mu_i - 1)$, together with the parameter component of the tangent.
- **PD:** $\psi_{PD} = \prod_i (\mu_i + 1)$.
- **NS:** $\psi_{NS} = \prod_{i<j} (\mu_i \mu_j - 1)$. This also vanishes for a real pair with product 1, which is a neutral saddle cycle; check the multipliers.

**Mesh adaptation.** Redistribute the mesh so that a monitor of the $m$-th derivative jumps is equidistributed (Russell and Christiansen 1978; AUTO-07p). Run it every $k$ steps (default 3) and whenever a step fails. Record the mesh on the branch.

**Starting a cycle branch from a Hopf point.** Use the initial guess
$$u(\tau) = u_0 + \varepsilon\, \mathrm{Re}\big(q\, e^{2\pi i \tau}\big), \qquad T = \frac{2\pi}{\omega},$$
with $\varepsilon$ chosen from the step size. If the corrector fails, retry with $\varepsilon/2$, up to 4 times.

**Period doubling switch.** Start the doubled branch at period $2T$ with an initial guess perturbed along the eigenfunction of multiplier $-1$.

## 8. Codim-1 cycle curves (phase E6)

Continue LPC, PD and NS curves in two parameters with minimally augmented defining systems on the collocation discretization (Doedel, Govaerts and Kuznetsov 2003, "Computation of periodic solution bifurcations in ODEs using bordered systems"). The periodic normal forms of phase E7 follow Kuznetsov, Govaerts, Doedel and Dhooge (2005) for codim 1 and De Witte, Della Rossa, Govaerts and Kuznetsov (2013) for codim 2.

## 9. Acceptance tests

Exact tests (no oracle needed):

1. **Fold normal form** $\dot x = \alpha + x^2$: LP at $(x,\alpha) = (0,0)$ within $10^{-10}$, and $a = 1$.
2. **Hopf normal form in real coordinates**, with $c = \pm 1$:
$$\dot x = \alpha x - y + c\,x(x^2+y^2), \qquad \dot y = x + \alpha y + c\,y(x^2+y^2).$$
   - The Hopf point is at $\alpha = 0$ with $\omega = 1$.
   - With $\langle q,q\rangle = 1$, the formula gives $l_1 = 2c$. With that normalization $|x|^2 = 2|z|^2$, so the complex coefficient doubles.
   - The sign of $l_1$ must match the sign of $c$.
3. **Hopf normal form limit cycle**, $c = -1$, $\alpha = 0.5$:
   - radius $\sqrt{\alpha}$ within $10^{-10}$ at $N = 40$, $m = 4$
   - period $2\pi$ within $10^{-10}$
   - nontrivial Floquet multiplier $e^{-4\pi\alpha}$ within $10^{-8}$
4. **Collocation convergence** on test 3: the mesh-point error must scale like $O(N^{-2m})$, with the observed order within 0.5 of $2m$ for $m = 2, 3, 4$.
5. **Lorenz**, $\sigma = 10$, $b = 8/3$:
   - BP of the origin at $r = 1$
   - Hopf on the nontrivial equilibria at $r = \sigma(\sigma+b+3)/(\sigma-b-1) = 470/19$, within $10^{-9}$
   - $l_1 > 0$ there (subcritical)
6. **Brusselator** $\dot x = a - (b+1)x + x^2 y$, $\dot y = bx - x^2 y$: the equilibrium $(a, b/a)$ has its Hopf point at $b = 1 + a^2$ with $\omega = a$, within $10^{-9}$.
7. **Cusp normal form** $\dot x = \beta_1 + \beta_2 x - x^3$: the LP curve in $(\beta_1,\beta_2)$ passes through a CP at the origin.
8. **Bogdanov-Takens normal form** $\dot x = y$, $\dot y = \beta_1 + \beta_2 y + x^2 + s\,xy$ with $s = \pm 1$: the LP and H curves meet at a BT point at the origin.
9. **Bautin normal form**, in real coordinates of $\dot z = (\beta_1 + i)z + \beta_2 z|z|^2 + s\,z|z|^4$: the H curve has a GH point at $\beta = (0,0)$, and the second Lyapunov coefficient has the sign of $s$.

Oracle tests compare against `references/*.json`:
- **Models:** Morris-Lecar (type I and type II parameter sets), FitzHugh-Nagumo, Hodgkin-Huxley and Rössler (PD cascade).
- **Generating references:** the developer produces them locally with AUTO-07p or MatCont. Each file records the tool, its version and the settings used.
- **Tolerance:** special-point parameter values must agree to $10^{-6}$ relative.
- **Missing references:** a test without its reference file is skipped and reported as "oracle missing", not failed.

FFI tests:
- Calling `pckern("panic")` raises a MATLAB error with ID `phasecont:engine:panic`, and MATLAB keeps running.
- Freed or unknown handles raise `phasecont:engine:badHandle`.
- `pc-cli` runs clean under AddressSanitizer on Linux CI.

Performance:
- `pc-cli bench` reports time per step and Newton iterations per step for each model, written to `benches/RESULTS.md`.
- If the developer has CL_MatCont installed locally, compare against it. Set targets after the first phase E5 measurement; do not promise numbers before then.

## 10. C ABI and MEX gateway

**Shape of the boundary.** The Rust side is a `cdylib` named `pckern` that exports a single symbol, `pc_api_v1()`. That function returns a struct of function pointers that starts with an ABI version number and the struct's size in bytes. Sketch:

```c
typedef struct {
  uint32_t abi_version;
  uint32_t struct_size;   /* sizeof the table in this build of pckern */
  int32_t (*model_from_ode)(const char* text, size_t len, uint64_t* model, char** diag_json);
  int32_t (*model_compile)(uint64_t model, const char* opts_json);
  int32_t (*model_rhs)(uint64_t model, const double* x, const double* p, double* out);
  int32_t (*run_start)(uint64_t model, const char* problem_json, uint64_t* run);
  int32_t (*run_poll)(uint64_t run, char** status_json, size_t* new_points);
  int32_t (*run_copy)(uint64_t run, size_t start, size_t count, double* buf, size_t buf_len);
  int32_t (*run_cancel)(uint64_t run);
  int32_t (*free_handle)(uint64_t handle);
  void    (*free_string)(char* s);
  const char* (*last_error)(void);  /* thread-local */
} pc_api_v1_t;
```

**Gateway.** `mex/pckern_gateway.c` is compiled with `mex -R2018a`. The first command that needs `pckern` loads it from the folder that holds the MEX file: on Windows the gateway finds its own file with `GetModuleHandleExW` and `GetModuleFileNameW` and loads `pckern.dll` by full path with `LoadLibraryExW` and `LOAD_WITH_ALTERED_SEARCH_PATH`, and elsewhere it uses `dlopen`. The gateway then calls `pc_api_v1`, requires `abi_version` to equal its own and `struct_size` to be at least the size of its own table, and dispatches on a command string. A failed load raises `phasecont:engine:load`, and the next call tries again. The gateway never unloads `pckern`, since worker threads may still run its code. Loading the library at runtime keeps the gateway independent of the Rust toolchain (MSVC or MinGW on Windows).

**Data rules.** Metadata and options cross as JSON. Numeric arrays are copied into buffers that MATLAB allocated. Rust never holds a pointer into MATLAB memory after a call returns.

## 11. MATLAB API sketch

```matlab
m  = phasecont.Model("ml.ode");                  % parse + JIT in Rust
eq = phasecont.equilibrium(m, x0, Params=p0);
br = phasecont.continuation(m, eq, Free="I", MaxSteps=500);
h  = br.points("H");
lc = phasecont.continuation(m, h(1), Type="LC", N=40, Ncol=4);
plot(br); hold on; plot(lc)
s = phasecont.Session(m, {br, lc}); save(s, "ml.pcsession.json");
phasecont.session.replay("ml.pcsession.json");   % headless rerun
```

- **Handle classes.** `Model`, `Branch` and `Run` are handle classes whose `delete` frees the Rust handle.
- **Apps.** They are programmatic `uifigure` code, not `.mlapp` files.
- **Session JSON.** Records the model source and its SHA-256, the engine name and version, the platform, the MATLAB release, every run's starting point and options, and the precision used.

## 12. Build, CI and packaging

**`buildfile.m` tasks.**
- `cargo`: release build for the current platform. For Windows the task lists the DLLs that `pckern.dll` imports and fails when one of them is not on an allow-list of DLLs that ship with Windows. Before a Windows build in WSL it checks that `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` report the same version, and stops with both versions in the message when they differ.
- `mex`: build the gateway.
- `test`: run `matlab.unittest` with JUnit output.
- `package`: build the `.mltbx` with `matlab.addons.toolbox.ToolboxOptions`, using a fixed UUID that never changes between versions. Before packaging, the task writes the license text of every crate linked into `pckern` on each platform into the toolbox, next to `LICENSE` and `THIRD_PARTY_NOTICES.md`. A Rust tool in this repo builds that file from `cargo metadata --filter-platform` and the license files in each crate's source.

**Base MATLAB check.** A test in `tests/` runs `matlab.codetools.requiredFilesAndProducts` on every file under `toolbox/` and fails when it lists a product other than MATLAB. Files of the optional `gpuArray` route may also list Parallel Computing Toolbox (rule 8 of `CLAUDE.md`). The test starts with the first MATLAB code under `toolbox/`.

**GitHub Actions.**
- **Rust job:** fmt, clippy, test and deny on Ubuntu, Windows and macOS.
- **MATLAB job:** `matlab-actions/setup-matlab` and `matlab-actions/run-build` on all three operating systems.
- **Release job:** collects the per-platform binaries into one `.mltbx` and creates the GitHub Release with the `.mltbx` attached in the same step, so File Exchange serves the toolbox file.
- **Linux binaries:** built against glibc 2.28.

**macOS quarantine.** Downloaded binaries may be quarantined. Test installing from the `.mltbx` on a clean Mac account, and document any `xattr` step if one turns out to be needed.

## 13. Later phases (pointers only)

- **GPU ensemble engine** (`pc-ensemble`). Backends:
  - `cuda`: `cudarc`, driver API plus NVRTC; primary for NVIDIA. Spike S4 compiled and ran an `sm_120` kernel with it on the developer's GPU, inside MATLAB R2026b and in WSL.
  - `opencl`: the `opencl3` crate; for AMD, Intel, and NVIDIA as a fallback.
  - `metal`: the `objc2-metal` crate; for every Mac, in `f32`.
  - `hip`: bindgen bindings loaded with `libloading`; for AMD.
  - `cpu`: rayon.
  - Kernels are emitted by `pc-codegen` and compiled at runtime. The precision policy is in `CLAUDE.md`, rule 6.
  - `cuda` host code calls `is_culib_present` for the driver and NVRTC before any other `cudarc` call, because `cudarc` panics when it cannot load a library. It compiles kernels through the raw NVRTC calls and destroys the program on every path: when a compile fails and reading the program log fails too, `cudarc`'s `compile_ptx_with_opts` panics, and the `Drop` of its program panics again if destroying the program also fails. That second panic aborts the process, which `catch_unwind` cannot stop.
  - NVRTC contracts `a*b + c` into an FMA by default (`--fmad`, "Default: true" in `/usr/local/cuda-13.4/include/nvrtc.h:960-965`). A test that compares a GPU result with a CPU reference at the level of single ulps rounds each contracted operation once in the reference, and does not get that rounding from `f32::mul_add` on `x86_64-pc-windows-gnu` at the baseline target features.
- **LaTeX companion toolbox:** pgfplots figures, tables of special points, a side-by-side editor with a pdf.js preview driving `latexmk` or Tectonic.
- **VS Code extension:** a TypeScript client plus `pc-lsp` (Rust, `tower-lsp`) that reuses `pc-ode` diagnostics.
- **Optional external backends:** AUTO-07p and XPPAUT, run as subprocesses the user installs separately.
- **Engine phases E7 to E9:** periodic normal forms, homoclinic orbits, delay equations.

## 14. Phases and acceptance

| Phase | Scope | Accepted when |
|---|---|---|
| P0 | Discovery, scaffold, spikes S1 to S4 | `docs/spikes.md` written, go/no-go decisions approved |
| P1 | `pc-ode` parser, diagnostics, writer, `pc-cli parse` | Section 2 table covered, round-trip property test passes, own test corpus parses |
| P2 | `pc-ir`, derivatives, `pc-jit`, AOT C emitter | Section 3 and 4 checks pass |
| P3 | Continuation core, equilibria, LP/H/BP, $a$, $l_1$, BP switching | Tests 1, 2, 5, 6 |
| P7a | Minimal MATLAB API and plotting, live polling | MATLAB tests for `Model`, `continuation`, `plot`; handle and panic tests |
| P8a | Package v0.3 | `.mltbx` installs on a clean MATLAB and runs the FitzHugh-Nagumo example |
| P4 | LP and H curves, codim-2 detection and normal forms | Tests 7, 8, 9 |
| P5 | Limit cycles: collocation, condensation, Floquet, mesh, Hopf start, PD switch | Tests 3, 4; Brusselator and Rössler oracles |
| P6 | LPC, PD, NS curves | Oracle tests for Morris-Lecar and Rössler |
| P7b | Phase-plane app, sessions, replay | Session round trip reproduces a full Morris-Lecar diagram |
| P8 | Package v1.0 | Release workflow publishes a `.mltbx` for all platforms |

## 15. References

Check every citation before quoting it in user-facing docs.

- Allgower, E. L., and Georg, K. *Introduction to Numerical Continuation Methods.* SIAM Classics, 2003.
- Govaerts, W. *Numerical Methods for Bifurcations of Dynamical Equilibria.* SIAM, 2000.
- Kuznetsov, Yu. A. *Elements of Applied Bifurcation Theory*, 4th ed. Springer, 2023.
- Kuznetsov, Yu. A. "Numerical normalization techniques for all codim 2 bifurcations of equilibria in ODEs." *SIAM J. Numer. Anal.*, 1999.
- Doedel, E., Keller, H. B., and Kernévez, J. P. "Numerical analysis and control of bifurcation problems," Parts I and II. *Int. J. Bifurcation and Chaos*, 1991.
- Doedel, E., Govaerts, W., and Kuznetsov, Yu. A. "Computation of periodic solution bifurcations in ODEs using bordered systems." *SIAM J. Numer. Anal.*, 2003.
- Kuznetsov, Yu. A., Govaerts, W., Doedel, E., and Dhooge, A. "Numerical periodic normalization for codim 1 bifurcations of limit cycles." *SIAM J. Numer. Anal.*, 2005.
- De Witte, V., Della Rossa, F., Govaerts, W., and Kuznetsov, Yu. A. "Numerical periodic normalization for codim 2 bifurcations of limit cycles." *SIAM J. Appl. Dyn. Syst.*, 2013.
- Fairgrieve, T. F., and Jepson, A. D. "O.K. Floquet multipliers." *SIAM J. Numer. Anal.*, 1991.
- Russell, R. D., and Christiansen, J. "Adaptive mesh selection strategies for solving boundary value problems." *SIAM J. Numer. Anal.*, 1978.
- Beyn, W.-J. "The numerical computation of connecting orbits in dynamical systems." *IMA J. Numer. Anal.*, 1990.
- Griewank, A., and Walther, A. *Evaluating Derivatives*, 2nd ed. SIAM, 2008.
- Ermentrout, B. *Simulating, Analyzing, and Animating Dynamical Systems: A Guide to XPPAUT for Researchers and Students.* SIAM, 2002.
- Dankowicz, H., and Schilder, F. *Recipes for Continuation.* SIAM, 2013. Read for ideas only; COCO's code is GPLv3.
