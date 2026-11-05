# PhaseCont technical spec

This file is the source of truth for what the engine computes and how each phase is accepted. Where a formula here conflicts with a cited reference, the reference wins: stop, note the conflict in `docs/STATUS.md`, and ask.

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
- **Keywords:** only the first letter is significant for the nine declarations `markov`, `parameter`, `wiener`, `table`, `aux`, `init`, `bndry`, `global` and `done` (`p`, `par` and `pzz` all declare parameters). Other keywords (`number`, `set`, `options`, `only`, `special`, `solv`, `export`, `volt`, `hom_bcs`) need the full word. A first word is a keyword only when a blank or the end of the line follows it and the next non-blank character is not `=`, so `p=3` and `p = 3` define a fixed quantity named `p`.
- **Case:** the language is case-insensitive.
- **Names:** XPPAUT 8.0 accepts a declared name longer than 10 characters but stops with an error where the name is used in a formula. A longer name gives a portability warning, not an error.
- **Line continuation:** a backslash at the end of a line continues it.
- **Order:** names may be used before their declaration, except fixed quantities: XPPAUT evaluates them in file order, so a fixed quantity that uses one defined on a later line is an error.
- **Spaces around `=`:** accepted in `par`, `number` and `init` lists and in formulas. In `@` lists XPPAUT 8.0 accepts the line but ignores the option, so PhaseCont gives a warning with a fix-it.
- **Declared values:** `par`, `number`, `init` and `x(0)=` values are signed numeric literals (XPPAUT reads only a leading number); inside an array template a value may also be a bare index form (`x[j](0)=[j]`). A repeated `par` is an error.
- **Comments:** `#` at the start of a line. A trailing `#` comment is accepted after `par`, `number` and `@` lists, where XPPAUT 8.0 ignores it, and is an error elsewhere: XPPAUT 8.0 rejects it after formulas and `init` lists, and PhaseCont also rejects it after the statements that were not measured.
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
| Tables | `table h % 101 0 6.283 sin(t)` | inline formula tables; file tables give an unsupported warning |
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
- The exact meaning of each function (for example that `log` is natural and `atan2(y,x)` uses C's order), with its source, is in the semantics table in `language.md`. Division follows IEEE (`1/0` is inf, `0/0` is NaN), unlike XPPAUT's guarded division. `mod(x,y)` is `r = fmod(x,y)`, plus `y` when `r < 0`. `if` evaluates both branches and selects one with the test `c != 0`. Constant folding in `pc-ir` drops NaN and inf from terms it removes (`x*0` is 0), and it can change the sign of a zero (`x*0` is `+0`, so `atan2(x*0,-1)` at `x = -1` is `+pi` in the DAG and `-pi` in the interpreter); `language.md` lists the rules.

**Non-smooth models.**
- Any of `heav sign abs mod flr ceil min max if`, a comparison (`< <= > >= ==`), `&`, `|`, `not`, or a table, marks a model as non-smooth.
- Equilibrium continuation is allowed, with a warning, as long as no switching surface is crossed.
- Derivatives use one-sided rules and the result is flagged.

**Diagnostics.**
- Each diagnostic has a code (`ODE0001`...), a line and column, a message, and an optional fix-it.
- The same diagnostics feed the MATLAB import report and, later, the VS Code language server.

**Round trip.** For every model, `parse(write(parse(s)))` must equal `parse(s)` (checked with `proptest`).

## 3. IR and derivatives (pc-ir)

**DAG structure.**
- A hash-consed DAG of `f64` operations, with constant folding and common subexpression elimination. Folding happens when a node is built, so hash-consing gives CSE for free. The algebraic rules and their effect on NaN are in `language.md`; there is no reassociation.
- Fixed quantities and user functions are inlined. Inline formula tables are tabulated when the model is lowered and read with linear interpolation.
- Time `t` is a hidden parameter slot `p[n_p]`, so every entry keeps the same signature. The IR records whether the model is autonomous.
- Every transcendental goes through one table of shims in `pc-ir`, backed by the `libm` crate. The interpreter, the DAG evaluator, the JIT and the AOT code all call the same functions.
- `pc-ir` refuses simulation-only models, `delay` and unsupported statements with an error that points at the source. A DAG over 1,000,000 nodes is an error.

**Derivatives produced by symbolic forward differentiation on the DAG.** Each one is a new DAG that goes through CSE again:
- $f(u,\alpha)$
- the Jacobian $A = f_u$ ($n \times n$)
- the parameter Jacobian $f_\alpha$ ($n \times n_p$)
- the multilinear forms $B(v_1,v_2) = D^2 f(u)[v_1,v_2]$ and $C(v_1,v_2,v_3) = D^3 f(u)[v_1,v_2,v_3]$
- directional derivatives up to order 5, for normal forms
- the mixed parameter derivatives needed by branch switching

$A$ and $f_\alpha$ come from first-order symbolic differentiation. Orders 2 to 5 use a multilinear (hyper-dual) expansion on the DAG: each node gets one component per subset of the $k$ direction slots, a product splits over subsets, and a call $g(w)$ uses Faà di Bruno's formula over set partitions, with $g^{(m)}$ from repeated symbolic differentiation. The result is exact, with no polarization. Directions live in the extended space $(u,\alpha)$ of length $n+n_p$, so one mechanism gives $B$, $C$, the order 4 and 5 forms, and mixed forms such as $A_\alpha v = D^2 f[(v,0),(0,e_j)]$. Each slot is marked as $u$ only, $\alpha$ only or full, so zero blocks fold away. Non-smooth builtins use the derivative of the active branch (`language.md` lists the rules).

**Complex directions.** Directional derivatives must accept complex direction vectors, because $q$ and $p$ in the normal forms are complex. PhaseCont splits each complex direction into its real and imaginary parts and uses multilinearity, so an order $k$ form takes at most $2^k$ real evaluations. There is no complex-valued entry point.

**Validation uses two independent evaluators:**
1. A tree-walking dual-number interpreter on the AST, using nested dual numbers up to depth 5. Its results must match the generated derivatives to within 4 ulp per entry, or $10^{-13}$ relative error. Let $a$ be the value from the DAG (or the JIT) and $b$ the value from the interpreter, and let $s$ be the largest magnitude in that output vector. An entry passes when any of these holds:
   - $a$ and $b$ are both NaN, or $a = b$;
   - $|a-b| \le 4\,\mathrm{ulp}(\max(|a|,|b|))$;
   - $|a-b| \le 10^{-13} s$;
   - $|a-b| \le 10^{-15}$, because values near zero carry rounding from larger intermediate terms;
   - $|a-b| \le 8\,\varepsilon\,E_i$, where $\varepsilon$ is the machine epsilon and $E_i$ is the running error bound of output $i$ of the DAG evaluation. A NaN or infinite $E_i$ adds no pass.

   $E$ is computed beside the value by Wilkinson's running error analysis (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, section 3.3), so that the rounding error of the computed value is at most $uE$ to first order, with $u = \varepsilon/2$. Leaves are exact ($E = 0$) and every rounded operation adds $|r|$ for its result $r$: `Add` and `Sub` give $E_a + E_b + |r|$, `Mul` gives $|a|E_b + |b|E_a + |r|$ and `Div` gives $(E_a + |r|E_b)/|b| + |r|$. Input errors pass through the exact first derivatives (the partials of the dual-number rules), so `Call1` gives $|f'(a)|E_a + 2|r|$ and `Call2` gives $|\partial_a f|E_a + |\partial_b f|E_b + 2|r|$. `PowI` with exponent $n$ gives $|n||r|E_a/|a| + (|n|-1)|r|$, because the roundings of binary powering compound by the `Mul` rule, plus $|r|$ for the reciprocal when $n < 0$. `Select` takes the bound of the branch taken, a table gives $|\mathrm{slope}|E_a + |r|$, `TableSlope` gives 0, and comparisons and logic give 0. An exact input ($E_a = 0$) adds nothing through a partial that is infinite or NaN, and any other non-finite partial makes $E$ non-finite.

   The factor 8 is 16 times the first-order bound $uE$ of one evaluator ($8\varepsilon = 16u$). That leaves room for the rounding of the interpreter, which $E$ does not model. The condition covers cancellation in high-order directional derivatives, where the terms are far larger than $s$ and $10^{-13}s$ sits below their rounding noise. $E$ is a first-order bound: `floor`, `ceil`, `heav`, `sign`, comparisons and `Select` contribute nothing for a jump, so a rounding difference that crosses a jump on a non-smooth model can still fail a row.
2. Central finite differences on random inputs, with relative error at most $10^{-6}$ for $A$ and at most $10^{-4}$ for $B$ and $C$, measured against the same scale $s$.

## 4. JIT and AOT (pc-jit, pc-codegen)

**Entry points.** Lower each needed DAG to Cranelift IR as an `extern "C"` function. A typical signature is `fn(x: *const f64, p: *const f64, out: *mut f64)`. Directional entries take extra vector pointers. Matrix outputs such as the Jacobian are written column major.

**Code generation flags.** Set `is_pic` only on x86_64 and set `use_colocated_libcalls` to false, as `JITBuilder::new` does. Set `enable_probestack` to true and `probestack_strategy` to `inline`. Cranelift leaves stack probes off by default, Windows needs them for frames over 4096 bytes, and the default `outline` strategy calls `__cranelift_probestack`, which `pckern` does not provide.

**Transcendentals.** Call the `extern "C"` shims of `pc-ir` (section 3), which wrap the `libm` crate, not the `f64` methods of std. Register every shim with `JITBuilder::symbol`. Shim bodies must not panic: a panic that leaves an `extern "C"` function aborts the process, JIT frames have no unwind information, and the panic barrier sits outside the JIT call.

**Ownership.** Each compiled model owns its `JITModule`. Dropping a `JITModule` leaves its code pages mapped, so the model handle calls `JITModule::free_memory` when it is freed and on every error path after `JITModule::new`, and only when no thread can still call the compiled functions.

**Equivalence.** JIT output for $f$ must match the dual-number interpreter to within 4 ulp. For derivatives the two use different algorithms, so they are compared with the tolerance of section 3, including the $8\,\varepsilon\,E_i$ condition with $E_i$ taken from the DAG evaluation of the same entry. JIT output must also match a Rust evaluator of the same DAG exactly (0 ulp), since Cranelift emits no fused multiply-add unless asked.

**AOT path.** `pc-codegen` emits C with the same signatures. The C does not call the C library: it declares `const pc_math_t *pc_math` and exports `pc_aot_init(const pc_math_t *)`, and the loader passes the `pc-ir` shim table, so AOT and JIT call the same math functions. It is compiled through `mex` when a compiler exists and selected with `phasecont.config("Codegen","aot")`. AOT output must match JIT output to within 4 ulp. Compile AOT code with floating-point contraction turned off, as the spike's gcc build does with `-ffp-contract=off`, and store the compiler, its version and its flags with every benchmark or test result that uses AOT code.

In P8a the loader is `Evaluator::Aot` in pc-engine. `libloading` opens the library that `mex` built, and the loader calls `pc_aot_init` with the `pc-ir` shim table before any entry. The library handle is kept for the life of the model. `model_compile` takes the keys `"emit_c":"<dir>"`, which writes the C text into that folder, and `"evaluator":"aot"` with `"aot_lib":"<path>"`, which loads the built library (section 10). On the MATLAB side, `internal.aotBuild` runs `mex` on the emitted C plus a stub `mexFunction`, with contraction off (`/fp:strict` on MSVC, `-ffp-contract=off` elsewhere). It caches the library under `tempdir`, and `Model.AotInfo` stores the compiler, its version and the flags. With no compiler, a failed compile or a blocked load, the model runs on the JIT and PhaseCont warns with a `phasecont:aot:*` identifier. Both paths agree within 4 ulp, so the warning is the only visible change. NaNs compare by class, not by sign.

**macOS arm64.** Confirm that JIT pages can be mapped executable inside the MATLAB process. Phase 0 ran only on Windows and WSL, so Phase 2 runs the Rust side of this check on the macOS runner of section 12, and the part inside MATLAB waits for a MATLAB licensing token for the macOS runner (deferred from Phase 7a, see `docs/STATUS.md`). `cranelift-jit` 0.137.0 accepts `is_pic=true` only on x86_64. Its default `SystemMemoryProvider` maps code pages read-write without `MAP_JIT` and makes them read-execute with `mprotect`, which the hardened runtime allows only with `com.apple.security.cs.allow-unsigned-executable-memory` or `com.apple.security.cs.disable-executable-page-protection`. `codesign -d --entitlements - <matlabroot>/bin/maca64/MATLAB`, where `<matlabroot>` is the folder that `matlabroot` returns, lists the entitlements MATLAB has. Without either entitlement, `pc-jit` needs its own `JITMemoryProvider`, set with `JITBuilder::memory_provider`, that maps pages with `MAP_JIT`, which needs `com.apple.security.cs.allow-jit`, and switches them with `pthread_jit_write_protect_np`. With none of the three, users still need no compiler: models run on an `f64` evaluator in Rust built from the same code as the dual-number interpreter of section 3, and PhaseCont warns that this is slower than compiled code. The AOT path stays the optional speed-up on Macs that have the Xcode Command Line Tools, selected with `phasecont.config("Codegen","aot")`. No Mac user is required to install a compiler.

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
- A branch also stops at the maximum step count, at a bound on $\alpha$ (the last step is cut back so that $\alpha$ lands on the bound), when the bordered matrix is singular, and on cancel. Each stop records its reason.

**Starting point.** The `init` values of a model are rarely an equilibrium, so a branch starts with Newton's method on $F(u) = 0$ at fixed parameters, with the same tolerances as the corrector. The first tangent solves the system above with $e_\alpha$ in place of $v_{k-1}$, and its sign is chosen by the requested direction. Models that depend on $t$ are refused, and so are models with a formula table that reads a parameter, since `lower` tabulates it once from the declared values.

**Test functions on equilibrium branches.** $\lambda_i$ are the eigenvalues of $A$.
- **LP:** $\psi_{LP} = v_{k,\alpha}$, the parameter component of the tangent.
- **BP:** $\psi_{BP} = \det \begin{pmatrix} F_x \\ v^\top \end{pmatrix}$, taken from the LU factors of the tangent solve.
- **H:** the zeros of $\prod_{i<j} (\lambda_i + \lambda_j)$, which equals $\det(2A \odot I)$. Let $s_{ij} = (\lambda_i + \lambda_j) / \max(1, |\lambda_i| + |\lambda_j|)$ and let $s_k$ be the factor of smallest modulus. The engine computes
  $$\psi_H = \operatorname{sign}\Big( \prod_{i<j} s_{ij} \Big)\, |s_k|.$$
  It is continuous in $A$, it has the sign of the plain product, and it is 0 exactly where some factor is 0 (two factors vanishing together also give 0). Its modulus is at most 1, so it neither overflows nor underflows to 0 away from a zero, even with many slow modes. For $n = 1$ there are no pairs and $\psi_H = 1$. Near a zero, $\psi_H$ can be nearly flat when another factor is tiny and hardly changes, so the zero search along a step uses the plain product divided by its modulus at point $k$, computed through logarithms.
- $\psi_H$ vanishes at Hopf points and at neutral saddles. At a zero, the pair with the smallest scaled $|\lambda_i + \lambda_j|$ decides: a complex conjugate pair is a Hopf point (`H`), a real pair $\pm\lambda$ is a neutral saddle, labeled `NSA` and not reported as Hopf. (`NS` is Neimark-Sacker.)
- Hopf detection has no limit on $n$. The product costs $O(n^2)$ on the dense eigenvalues, which the engine computes at every point anyway for the $O(n^3)$ eigenvalue step, so it adds a few percent (`docs/STATUS.md` lists timings up to $n = 1000$). Phase 4 kept the dense route. Govaerts' minimally augmented Hopf function needs $\kappa = \omega^2$ as an unknown, so it is not a scalar test function along an equilibrium branch, and no source that gives such a test was confirmed.

**Locating special points.**
1. When a test function changes sign strictly between $x_k$ and $x_{k+1}$, find $s \in (0,1)$ where it vanishes. Each trial point is corrected from $x_k + s h v_k$ on the hyperplane $v_k^\top(x - x_k) = s h$. The iteration is a secant method with a bisection safeguard (Illinois regula falsi), at most 30 iterations.
2. Refine the point by solving its defining system (Section 6) with Newton's method. If that Newton fails, the located point is kept and marked as not refined.
3. Store the point, its eigenvalues and its normal form coefficient on the branch.
4. A branch that starts at a special point (after branch switching) skips detection on its first step.

**Branch storage.** Columns are $u$, $\alpha$, tangent, eigenvalues (or multipliers), test function values, step size and Newton iteration count. Special points carry a label, an index and a struct of extras.

**Background runs.** A continuation runs on a Rust background thread. MATLAB polls for new points with `pc_run_poll` from a `timer`, so plots update live, and `pc_run_cancel` stops the run. Cancel is checked between Newton iterations and between steps. A panic on the background thread is caught on that thread, with its message and source location, and the next `pc_run_poll` reports it as an error.
- A compiled model is not `Send` (it holds code pointers), and the lowered `IrModel` is not `Clone`. A run therefore takes copies of the checked model and its `ModelInfo`, and the worker lowers and compiles them on its own thread.
- `pc-engine` owns the panic hook. It is installed once, stores the message and `file:line:col` of a panic on an engine worker in a thread local, and prints nothing, since stderr is lost inside MATLAB. `pc-ffi` reuses it in Phase 7a.

## 6. Equilibrium bifurcations and normal forms

**Fold (LP).** With $Aq = 0$, $A^\top p = 0$, $p^\top q = 1$ and $\|q\| = 1$, the quadratic coefficient is
$$a = \tfrac{1}{2}\, p^\top B(q,q).$$
The sign of $q$ is chosen so that $p^\top F_\alpha > 0$. The normal form is then $\dot w = \beta + a w^2$ with $\beta$ increasing in $\alpha$ (Kuznetsov, Theorem 3.1), so the sign of $a$ tells on which side of the fold the two equilibria lie. If $p^\top F_\alpha$ is exactly 0, the fold is not generic and $q$ keeps the sign of the null vector solve.

**Minimally augmented LP curve** (Govaerts 2000). Solve
$$\begin{pmatrix} A & w_{\text{bor}} \\ v_{\text{bor}}^\top & 0 \end{pmatrix}\begin{pmatrix} q \\ g \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}.$$
The LP curve is $F = 0$, $g = 0$ with two free parameters. Update the borders $v_{\text{bor}}, w_{\text{bor}}$ from the current null vectors after each accepted step.

**Hopf.** The eigenvectors satisfy $Aq = i\omega q$ and $A^\top p = -i\omega p$. Use the normalization $\langle q,q\rangle = 1$ and $\langle p,q\rangle = 1$, where $\langle a,b\rangle = \bar a^\top b$. The first Lyapunov coefficient (Kuznetsov) is
$$l_1 = \frac{1}{2\omega}\,\mathrm{Re}\Big(\langle p, C(q,q,\bar q)\rangle - 2\langle p, B(q, A^{-1}B(q,\bar q))\rangle + \langle p, B(\bar q, (2i\omega I - A)^{-1}B(q,q))\rangle\Big).$$
The value of $l_1$ depends on the normalization of $q$; only its sign is invariant. Always report which normalization was used. Every stored LP and H point records its normalization as a string (`|q|=1, p'q=1, p'F_alpha>0` and `<q,q>=1, <p,q>=1`). The complex solve with $2i\omega I - A$ runs in the real form $\begin{pmatrix} A - \sigma I & \omega I \\ -\omega I & A - \sigma I \end{pmatrix}$ of $A - (\sigma + i\omega) I$, so only real LU is needed.

**Refining points on an equilibrium branch.** Newton's method on these systems, with exact Jacobians from the `JacU`, `JacP` and `Dir` entries:
- LP: $F = 0$, $g = 0$ in $(u, \alpha)$, with $g$ from the bordered system above and borders from the null vectors of $A$ at the located point. The derivative of $g$ with respect to a component $z$ of $(u,\alpha)$ is $g_z = -w^\top (A_z q)$, where $w$ solves the adjoint system $\begin{pmatrix} A^\top & v_{\text{bor}} \\ w_{\text{bor}}^\top & 0 \end{pmatrix}\begin{pmatrix} w \\ h \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$.
- H: the standard extended system in $(u, \alpha, q_r, q_i, \omega)$ with $3n + 2$ equations: $F = 0$, $A q_r + \omega q_i = 0$, $A q_i - \omega q_r = 0$, $c^\top q_r = 1$, $c^\top q_i = 0$, where $c$ is the real part of the eigenvector at the located point. It needs only first and second derivatives.
- BP: $F(x) + b\varphi = 0$, $F_x^\top \varphi = 0$, $\varphi^\top \varphi = 1$ in $(x, b, \varphi)$, with $2n + 2$ equations; $b = 0$ at the solution.

**Two-parameter curves.** Phase 4 continues LP and H curves in `pc-engine` (`continue_lp_curve`, `continue_hopf_curve`) under the same `run_branch` as equilibrium branches. They are reachable from the C ABI (problem `type` `LP` or `H`) and from MATLAB (`Type="LP"|"H"`, Section 11). The unknown is $x = (u, a_1, a_2)$ for the LP curve and $x = (u, a_1, \kappa, a_2)$ for the H curve, and `CurveSetup` gives the two free parameters and a start state. On a curve, `Point.u` is $x$ without its last entry, `Point.alpha` is $a_2$, `Options::bounds` applies to $a_2$, `Point.tests` is NaN and the codim-2 test values are in `Point.codim2`. The start tangent is the unit null vector of the curve Jacobian. `Options::direction` $-1$ reverses it.

**LP curve.** The equations are $F = 0$ and $g = 0$ with $g$ from the bordered system above. The zero set of $g$ does not depend on the borders, so each evaluation of the test functions replaces them by the unit null vectors $q$ and $p$ at that point. The sign of $q$ and $p$ stays continuous because $v_{\text{bor}}^\top q = 1$ holds for the old borders.

**Minimally augmented Hopf curve.** Let $M = A^2 + \kappa I$. At a Hopf point $M$ has the eigenvalue 0 on the real eigenplane $E$ of the pair, so its nullity is 2. On $E$, $A = \mu I + \omega J$ with $J$ the rotation by 90 degrees, and $M|_E = aI + bJ$ with $a = \mu^2 - \omega^2 + \kappa$ and $b = 2\mu\omega$. With $n \times 2$ borders $V$ and $W$ solve
$$\begin{pmatrix} M & W \\ V^\top & 0 \end{pmatrix}\begin{pmatrix} X \\ G \end{pmatrix} = \begin{pmatrix} 0 \\ I_2 \end{pmatrix}.$$
Then $G e_1 = 0$ exactly when $M|_E = 0$, that is when $\mu = 0$ and $\kappa = \omega^2$. The curve is $F = 0$, $G_{11} = 0$, $G_{21} = 0$, which are $n + 2$ equations in $n + 3$ unknowns. This is Govaerts' minimally extended system for a nullity-2 matrix with the second column of $G$ left out (Govaerts 2000, Sec. 5.6). The gradient is $G_z[i,0] = -Y_i^\top M_z X_0$, with $Y$ from the adjoint system $\begin{pmatrix} M^\top & V \\ W^\top & 0 \end{pmatrix}\begin{pmatrix} Y \\ H \end{pmatrix} = \begin{pmatrix} 0 \\ I_2 \end{pmatrix}$. For $z$ in $(u, a_1, a_2)$, $M_z = A_z A + A A_z$, so $G_z[i,0] = -\big(Y_i \cdot A_z(AX_0) + (A^\top Y_i)\cdot(A_z X_0)\big)$. For $\kappa$, $M_\kappa = I$ and $G_\kappa[i,0] = -Y_i \cdot X_0$. The borders are replaced by an orthonormal basis of the columns of $X$ and $Y$ at each evaluation of the codim-2 tests. The start $\kappa_0 = \omega^2 - \mu^2$ comes from the critical pair of $A$ at the start state. A start whose residual exceeds `tol_f` is a `BadProblem` error. The Brusselator Hopf curve $b = 1 + a^2$ is an exact check ($\kappa = a^2$, equilibrium $(a, b/a)$), and the run matches it to $10^{-9}$.

**Codim-2 points.** `Point.codim2` holds the test values in the order below, and `Problem::codim2_label(i, eigs)` names a zero. `locate.rs` treats the three equilibrium tests and the codim-2 tests as one index space. The zero is located by the Illinois search of Section 5 and then polished by Newton's method along the curve: the slope of the test is a forward difference along the tangent, each step is a corrector solve of that length, and the run ends when a step is shorter than `tol_x`. A point that converges has `refined = true` and is accurate to $10^{-10}$. If the test is undefined next to the point, a step is longer than the search step, the corrector fails or the iteration does not converge in 8 steps, the Illinois point is kept with `refined = false`, accurate to $10^{-8}$. A codim-2 sign change counts as a zero only if the located value is at most $10^{-2}$ times the smaller end value of the step, so a pole is not reported. A zero that `codim2_label` rejects gives no special point and no warning. A curve run also looks for specials on its first step.
- On LP curves, with unit $q$ and $p$:
  - CP: $\langle p, B(q,q)\rangle$. It stays smooth through a BT point, where $\langle p, q\rangle$ vanishes.
  - BT: $\langle p, q\rangle$.
  - ZH: $\psi_H$ of the eigenvalues without the one of smallest modulus. A zero from two real eigenvalues of opposite sign is a neutral saddle and is not labeled (`classify_h_zero`).
- On H curves, with $\omega = \sqrt{\kappa}$:
  - GH: $l_1$ at $\omega$ for $\kappa > 0$, NaN otherwise.
  - BT: $\kappa$. It changes sign where $\omega$ passes through 0, so the curve continues through BT into the neutral saddle side.
  - ZH: the product of the real parts of the eigenvalues outside the critical pair. It is labeled only if the eigenvalue nearest the imaginary axis outside the pair is real.
  - HH: $\psi_H$ of the eigenvalues outside the critical pair. It needs at least two of them and is labeled only if the zero is a conjugate pair.
  - ZH and HH are NaN for $\kappa \le 0$, since the critical pair is not a Hopf pair there.

**Normal form coefficients.** The formulas follow Kuznetsov (1999) and his book (Section 15), with $B, C, D, E$ the second to fifth derivatives of $F$ in the state at the point.
- CP: $c = \tfrac{1}{6}\langle p, C(q,q,q) + 3B(q, h_2)\rangle$ with $Aq = 0$, $A^\top p = 0$, $\langle p,q\rangle = 1$, $Ah_2 = \langle p, B(q,q)\rangle q - B(q,q)$ and $\langle p, h_2\rangle = 0$. The value depends on the scale of $q$ and its sign does not. The normalization string is `|q| = 1 before scaling, <p,q> = 1, <p,h2> = 0`.
- BT: with the chains $Aq_0 = 0$, $Aq_1 = q_0$, $A^\top p_1 = 0$, $A^\top p_0 = p_1$ and $\langle q_0,p_0\rangle = \langle q_1,p_1\rangle = 1$, $\langle q_1,p_0\rangle = \langle q_0,p_1\rangle = 0$, the normal form $\dot w_0 = w_1$, $\dot w_1 = a w_0^2 + b w_0 w_1$ has $a = \tfrac12\langle p_1, B(q_0,q_0)\rangle$ and $b = \langle p_0, B(q_0,q_0)\rangle + \langle p_1, B(q_0,q_1)\rangle$. Here $|q_0| = 1$ and the sign of $q_0$ makes $a > 0$. This sign convention is a choice of this project, and the stored normalization string says so.
- GH: $l_2 = \mathrm{Re}\,G_{32} / (12\omega)$ from the recursion for $h_{20}, h_{11}, h_{30}, G_{21}, h_{21}, h_{31}, h_{22}$ in Kuznetsov's Bautin section, with $\langle q,q\rangle = \langle p,q\rangle = 1$ and $\langle p, h_{21}\rangle = 0$. The normalization string is `<q,q>=1, <p,q>=1, <p,h21>=0`. For the Bautin form of Test 9 with $|q| = 1$ the expected value is $l_2 = 4s$, and a near-identity coordinate change of the model gives the same value to $10^{-6}$.
- ZH: with $Aq_0 = 0$, $Aq_1 = q_0$, $A^\top p_1 = 0$, $A^\top p_0 = p_1$ for the double-zero chain, and $q_1, p_1$ the eigenvector pair for $\pm i\omega$, the coefficients follow Kuznetsov (1999, Section 8): $G_{200} = \langle p_0, B(q_0,q_0)\rangle$, $G_{110} = \langle p_1, B(q_0,q_1)\rangle$, $G_{011} = \langle p_0, B(q_1,\bar q_1)\rangle$, then $b = G_{200}/2$, $c = G_{011}$, $d = G_{110} - i\omega G_{300}/(3G_{200})$ and $e = \tfrac12\mathrm{Re}\,[G_{210} + G_{110}(\mathrm{Re}\,G_{021}/G_{011} - G_{300}/G_{200} + G_{111}/G_{011}) - G_{021}G_{200}/(2G_{011})]$, with $G_{300}, G_{111}, G_{210}, G_{021}$ from the $h_{200}, h_{020}, h_{110}, h_{011}$ solves of that section. The sign of $q_0$ makes $b > 0$, a choice of this project like $a > 0$ at BT. $G_{200} = 0$ or $G_{011} = 0$ is an error. `Extras.coefficient` is $b$, `coefficient2` is $c$, and `values` holds `re_d`, `im_d` and `e`. The normalization string is `b>0, |q0|=1, <p0,q0>=1, <q1,q1>=1, <p1,q1>=1`.
- HH: with the eigenpairs of $\pm i\omega_1$ and $\pm i\omega_2$ ordered so that $\omega_1 > \omega_2$, and $\langle q_j,q_j\rangle = \langle p_j,q_j\rangle = 1$, the cubic coefficients follow Kuznetsov (1999, Section 9). The $h_{1100}, h_{2000}, h_{0011}, h_{0020}, h_{1010}, h_{1001}$ solves give $G_{2100}, G_{1011}, G_{1110}, G_{0021}$, whose real parts are $P_{11}, P_{12}, P_{21}, P_{22}$. `coefficient` is $P_{11}$, `values` holds `re_p12`, `re_p21`, `re_p22` and `omega2`, and `omega` is $\omega_1$. $\omega_1 = \omega_2$ and $\omega_1 = 2\omega_2$ are errors. The normalization string is `<q1,q1>=1, <p1,q1>=1, <q2,q2>=1, <p2,q2>=1, w1>w2`. The fifth-order coefficients $S_1$ and $S_2$ are not computed. The open copy of the paper has misprints in those formulas and no second source confirms them, so rule 9 keeps them out. They stay open in `docs/STATUS.md`.

**Branch switching at BP.** Compute the second tangent from the null space of the extended Jacobian and the algebraic branching equation (Allgower and Georg; Govaerts). At a simple BP $x^*$, $\varphi_1, \varphi_2$ span the kernel of the $n \times (n+1)$ matrix $F_x$ and $\psi$ spans its left kernel. With $c_{ij} = \psi^\top F_{xx}[\varphi_i, \varphi_j]$, the tangents of the two branches are $\beta_1\varphi_1 + \beta_2\varphi_2$ for the roots of $c_{11}\beta_1^2 + 2c_{12}\beta_1\beta_2 + c_{22}\beta_2^2 = 0$. The new branch takes the root farther from the current tangent. A negative discriminant or a double root means the BP is not simple, and switching stops with an error.

## 7. Limit cycles

**Formulation.** Rescale time to $\tau \in [0,1]$:
$$\dot u = T f(u,\alpha), \qquad u(0) = u(1), \qquad \int_0^1 u(\tau)^\top \dot u_{\text{prev}}(\tau)\, d\tau = 0.$$
The integral is the phase condition.

**Discretization.**
- Use $N$ mesh intervals with $m$ Gauss-Legendre collocation points each. On each interval, $u$ is a degree-$m$ polynomial stored by its values at $m+1$ points, and neighboring intervals share endpoints.
- Unknowns: $n(Nm+1) + 2$ (the mesh values, $T$ and $\alpha$).
- Equations: $nNm$ collocation equations, $n$ periodicity conditions, 1 phase condition and 1 PALC equation.
- Evaluate the phase integral with the same Gauss weights.
- $N$ and $m$ stay fixed for the whole run. Mesh adaptation moves the $N+1$ mesh points and keeps both, so the row stride of Section 10 does not change during a run.
- The PALC equation scales the profile unknowns by $1/\sqrt{Nm}$, so the arclength approximates the $L^2$ norm of the orbit. $T$ and $\alpha$ are not scaled.

**Linear solves by condensation of parameters** (Doedel, Keller and Kernévez 1991; AUTO-07p):
1. On each interval, eliminate the interior unknowns with dense LU (faer).
2. This leaves a block system in the mesh-point values plus the bordering rows and columns.
3. Solve that reduced system with faer's sparse LU. `pc-linalg` chooses between faer's simplicial and supernodal variants itself, starting with the simplicial one, instead of using faer's automatic choice. Tests compare the result with a sparse LU of the full matrix.

Assemble and condense in Rust. Run sequentially by default, and condense the intervals in parallel on the current rayon pool when the interval elimination work, $N \cdot nm \cdot n_i (n_i + 2n + k + 1)$ multiply-adds with $n_i = n(m-1)$ interior columns and $k$ bordering columns, is at least `PARALLEL_WORK` $= 2 \cdot 10^7$ (`CondenseShape::parallel_pays`). Phase 5 measured this threshold on the default pool, with one thread per logical CPU (`docs/STATUS.md`, P5 package 7). The parallel and sequential solves give the same bits. Large buffers are allocated with `try_reserve` or `MemBuffer::try_new` and fail with `OutOfMemory`, since a failed allocation in `Vec` or in dyn-stack's `MemBuffer::new` aborts the MATLAB process.

**Floquet multipliers.**
- The condensed blocks give the linear map from $u(0)$ to $u(1)$.
- v1: form the monodromy matrix as a product of the blocks, compute its eigenvalues, and check that one multiplier satisfies $|\mu_1 - 1| \le 10^{-6}$. Flag the result when the check fails.
- Later: replace the product with a periodic Schur method (Fairgrieve and Jepson 1991), so that multipliers spanning many orders of magnitude stay accurate.

**Test functions on cycle branches.** The products run over the nontrivial multipliers.
- **LPC:** $\psi_{LPC}$ is the $\alpha$ component of the tangent, which changes sign at a fold of the cycle branch. The product $\prod_i (\mu_i - 1)$ over the nontrivial multipliers also vanishes there, but P5 does not use it.
- **PD:** $\psi_{PD} = \prod_i (\mu_i + 1)$.
- **NS:** $\psi_{NS} = \prod_{i<j} (\mu_i \mu_j - 1)$. This also vanishes for a real pair with product 1, which is a neutral saddle cycle; check the multipliers.

**Mesh adaptation.** Redistribute the mesh so that a monitor of the $m$-th derivative jumps is equidistributed (Russell and Christiansen 1978; AUTO-07p). `crates/pc-engine/src/cycle/mesh.rs` ports AUTO-07p's EQDF, NEWMSH and INTERP (see `THIRD_PARTY_NOTICES.md`). It runs every `Options::mesh_adapt` accepted points (default 3, 0 turns it off) and at most once per point after a failed corrector. When every $m$-th derivative is below $10^{-7}$ the mesh stays, and a new mesh that is not strictly increasing or not finite is refused. Every point records its mesh.

**Starting a cycle branch from a Hopf point.** Use the initial guess
$$u(\tau) = u_0 + \varepsilon\, \mathrm{Re}\big(q\, e^{2\pi i \tau}\big), \qquad T = \frac{2\pi}{\omega},$$
with $\varepsilon$ = `Options::h0`. If the corrector fails, retry with $\varepsilon/2$, up to 4 times. The run always moves away from the equilibrium, so `Options::direction` $< 0$ is ignored with a warning.

**Period doubling switch.** Start the doubled branch at period $2T$ with an initial guess perturbed along the eigenfunction $\psi$ of multiplier $-1$. $\psi$ is solved from the condensed system of the PD orbit. The orbit and $\psi$ (with $-\psi$ on the second half) are doubled onto $[0, 1]$ and interpolated onto the new $N$ and $m$, which may differ from those of the PD orbit. The new mesh takes its $N + 1$ points evenly in the index of the doubled mesh. The first orbit is corrected at $\varepsilon$ = `Options::h0` along that direction, with the same halving as a Hopf start. The start warns when the multiplier closest to $-1$ is more than $10^{-6}$ away from it and refuses a complex one. Both directions give the same orbits half a period apart, so `Options::direction` $< 0$ is ignored with a warning.

**Trivial multiplier.** A point with no multiplier within $10^{-6}$ of 1 gets a warning on its branch.

## 8. Codim-1 cycle curves (phase E6)

Continue LPC, PD and NS curves in two parameters $(a_1, a_2)$ with minimally augmented defining systems on the collocation discretization of Section 7 (Doedel, Govaerts and Kuznetsov 2003, "Computation of periodic solution bifurcations in ODEs using bordered systems", DGK below). The periodic normal forms of phase E7 follow Kuznetsov, Govaerts, Doedel and Dhooge (2005) for codim 1 and De Witte, Della Rossa, Govaerts and Kuznetsov (2013) for codim 2.

**Unknowns.** An LPC or PD curve has $x = (u, T, a_1, a_2)$, where $u$ is the orbit at the $Nm+1$ base points. An NS curve adds $\kappa = \cos\theta$ of the critical multipliers $e^{\pm i\theta}$, so $x = (u, T, a_1, \kappa, a_2)$. The row alpha is $a_2$, and `Point.u` holds everything before it.

**Outer system.** The rows of Section 7 (collocation, periodicity and the phase condition) come first, then the defining equation $G = 0$ (one row for LPC and PD, two for NS), then PALC. $T$, $a_1$, $\kappa$ and $a_2$ border the condensed system, which has $k = 3$ bordering columns for LPC and PD and $k = 4$ for NS. The phase reference, the PALC scaling and mesh adaptation (`options.mesh_adapt`) work as on a cycle branch.

**Defining functions.** Each $G$ is the bottom right entry (or $2 \times 2$ block) of the solution of a bordered inner system $N$ that is singular on the curve. With $N (v, G)^\top = (0, 1)^\top$ and $(w, G)\, N = (0, 1)$, the derivatives are $G_z = -w^\top N_z v$ (DGK §3), from one factored solve and one transposed solve of the condensed inner system. The second derivatives $f_{xx}$ and $f_{xp}$ in $N_z$ are exact directional derivatives from the JIT (`Model2::a_z_q`).
- **LPC.** $N$ is the Jacobian of the periodic orbit system in $(u, T)$, with the phase row of Section 7 and its fixed reference in place of DGK's $\mathrm{Int}_f$ row, bordered by one row and one column. $G = 0$ is then exactly the singularity of the discrete Jacobian in $(u, T)$. The zero set does not depend on the phase vector as long as $\int_0^1 \varphi^\top \dot u \, d\tau \ne 0$. The inner condensation has $k = 2$.
- **PD.** $N = [D - T f_x;\ v(0) + v(1)]$, bordered by one row and one column (DGK, PD system), with no $T$ unknown. The inner condensation has $k = 1$.
- **NS.** DGK write the NS system on $[0, 2]$ with $v(0) - 2\kappa v(1) + v(2) = 0$. The engine writes it exactly on $[0, 1]$ with a doubled state $(v_a, v_b)$ of size $2n$ and the boundary conditions $v_b(0) - v_a(1) = 0$ and $v_a(0) - 2\kappa v_a(1) + v_b(1) = 0$, bordered by two rows and two columns. The curve continues $G_{11}$ and $G_{21}$, the first column of $G$, from one factored solve and two transposed solves. $G_\kappa$ comes from the $2 v_a(1)$ term. The inner condensation has $n \to 2n$ and $k = 2$.

For $|\kappa| < 1$ the real kernel of $M^2 - 2\kappa M + I$ has even dimension, so a zero first column of $G$ forces $G = 0$, and the two equations have no spurious zeros. For $|\kappa| > 1$ (a neutral saddle cycle) they can: the branch warns when a point lacks one of the real multipliers $\mu, 1/\mu$ with $\mu + 1/\mu = 2\kappa$, within $10^{-6}$ relative.

**Borders.** The row border over the profile is the previous right kernel vector. After mesh adaptation it is interpolated onto the new mesh. The column border lies on the boundary rows (and the phase row for LPC), so it does not depend on the mesh. Both are refreshed at every accepted point. At the start they come from two steps of inverse iteration from a generic border, and the NS start uses two Weyl sequences for its two borders.

**Start.** A curve starts from an LPC, PD or NS special point of an LC branch, with its orbit, period and mesh.
- When the requested `ntst` and `ncol` differ from those of the point, the orbit is regridded onto $N + 1$ mesh points taken evenly in the index of the old mesh (Section 7, period doubling switch).
- An NS start takes $\kappa$ from the complex multiplier closest to the unit circle, and a start with no complex pair is `BadProblem`.
- Section 7 locates these points with other test functions, so $G$ is small at the start but not zero. Newton first corrects the start at fixed $a_2$, and a failure is `BadProblem` with the hint "start from a matching special point of a cycle branch".
- The first tangent comes from a bordered solve with the row $e_{a_2}$, or the next unknown when that solve fails. It points toward growing $a_2$ times the sign of `Options::direction`.

**Codim-2 tests.** They use the Floquet multipliers of the point. Each curve leaves out the trivial multiplier and its own critical multipliers, removing each time the multiplier nearest to the excluded value, so a zero comes from one more multiplier, or pair, on the unit circle. A test over an empty set of remaining multipliers is NaN.

| Curve | Labels and tests |
|---|---|
| LPC | LPPD $= \prod(\mu + 1)$, LPNS $= \prod_{i<j}(\mu_i \mu_j - 1)$, without the double multiplier 1 |
| PD | LPPD $= \prod(\mu - 1)$, PDNS $= \prod_{i<j}(\mu_i \mu_j - 1)$, without 1 and $-1$ |
| NS | R1 $= \kappa - 1$, R2 $= \kappa + 1$, R3 $= \kappa + \tfrac12$, R4 $= \kappa$, LPNS $= \prod(\mu - 1)$, PDNS $= \prod(\mu + 1)$, without 1 and the roots of $\mu^2 - 2\kappa\mu + 1$ |

A zero of LPNS on an LPC curve or of PDNS on a PD curve gets its label only when the pair whose product is closest to 1 is complex, since a real pair with product 1 is a neutral saddle. On an LPC curve the multiplier 1 is double with a Jordan block, so the trivial multiplier check of Section 7 uses the mean of the two multipliers nearest to 1.

**Deferred to E7.** CPC, GPD and CH need periodic normal forms. R1 on an LPC curve and R2 on a PD curve need a coalescence of multipliers that the exclusion rule cannot separate.

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
5. **Lorenz**, $\sigma = 10$, $b = 8/3$ (set to `8.0/3.0` exactly, not a rounded literal such as `2.6666667`, which moves the Hopf value by about $2 \cdot 10^{-7}$):
   - BP of the origin at $r = 1$ within $10^{-10}$
   - switching at that BP gives the nontrivial branch, whose points satisfy $x = y$ and $z = x^2/b$
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
- `pc-cli` runs clean under AddressSanitizer on Linux CI. The job uses a nightly Rust toolchain on GitHub's Ubuntu runner with `-Zsanitizer=address -Zbuild-std`, because Fedora's packaged rustc has no sanitizer runtime.

Performance:
- `pc-cli bench` reports time per step and Newton iterations per step for each model, written to `benches/RESULTS.md`.
- If the developer has CL_MatCont installed locally, compare against it. Set targets after the first phase E5 measurement; do not promise numbers before then.

## 10. C ABI and MEX gateway

**Shape of the boundary.** The Rust side is a `cdylib` named `pckern` that exports a single symbol, `pc_api_v1()`. That function returns a struct of function pointers that starts with an ABI version number and the struct's size in bytes. The table is frozen at ABI version 1 from P7a on. Later fields may only be appended, since the gateway already accepts a `struct_size` at least its own.

```c
#define PC_ABI_VERSION 1
typedef struct {
  uint32_t abi_version;
  uint32_t struct_size;   /* sizeof the table in this build of pckern */
  int32_t (*model_from_ode)(const char* text, size_t len, uint64_t* model, char** diag_json);
  int32_t (*model_compile)(uint64_t model, const char* opts_json);        /* {"evaluator":"jit"|"fallback"|"aot", "aot_lib", "emit_c"} */
  int32_t (*model_rhs)(uint64_t model, const double* x, size_t nx,
                       const double* p, size_t np, double* out, size_t nout);
  int32_t (*run_start)(uint64_t model, const char* problem_json,
                       const double* u0, size_t nu, const double* p, size_t np, uint64_t* run);
  int32_t (*run_poll)(uint64_t run, char** status_json);
  int32_t (*run_copy)(uint64_t run, uint32_t branch, uint32_t what,
                      size_t start, size_t count, double* buf, size_t buf_len, size_t* rows);
  int32_t (*run_cancel)(uint64_t run);
  int32_t (*free_handle)(uint64_t handle);
  void    (*free_string)(char* s);
  const char* (*last_error)(void);                                        /* thread-local */
  int32_t (*model_info)(uint64_t model, char** info_json);                /* names, values, init, n, n_p, evaluator, aot_lib, c_file, jit_max_frame, autonomous, source_sha256, options */
  int32_t (*model_equilibrium)(uint64_t model, const double* u0, size_t nu, const double* p, size_t np,
                               const char* opts_json, double* u_out, size_t nu_out,
                               double* eig_out, size_t neig_out);          /* eig as [re; im], 2n */
  int32_t (*debug_panic)(int32_t where, uint64_t* run);                   /* 0: this thread, 1: a run whose worker panics */
  const char* (*version)(void);
  int32_t (*model_rhs_t)(uint64_t model, double t, const double* x, size_t nx,
                         const double* p, size_t np, double* out, size_t nout);  /* appended after version, offset 120, table size 128 */
} pc_api_v1_t;
```

**Time.** `model_compile` accepts a model whose equations use `t`, and `model_info` reports `autonomous: false` for it. `model_rhs` on such a model returns nonAutonomous with the message `the model depends on t; call model_rhs_t`. `model_rhs_t` evaluates `f(x, p, t)` for any model, needs a finite `t` (badArg otherwise), and a model that does not use `t` ignores it. There is no default for `t`. `run_start` and `model_equilibrium` need an autonomous model and return nonAutonomous otherwise. Adding `model_rhs_t` grew the table from 120 to 128 bytes and left `abi_version` at 1.

**`model_info` keys (P7b).** `source_sha256` is the SHA-256 of the exact bytes passed to `model_from_ode`, in lower-case hex, with no newline or encoding change. It comes from an in-repo implementation of FIPS 180-4 (`pc-ffi/src/sha256.rs`) that is tested against the NIST vectors. `options` lists the `@` options of the model as `[{"name":"total","value":"100"},...]`, with lower-case names, values as written, and the last value of a repeated name winning. These are JSON keys, so the table keeps its 17 members and 128 bytes, and `abi_version` stays 1.

**Equilibrium without a free parameter (P7b).** `model_equilibrium` solves F(u, p) = 0 by plain Newton and does not need a free parameter, so a model with no parameters runs `phasecont.equilibrium`. Before P7b it raised "parameter index 0 is out of range". Continuation still needs a free parameter: `EpProblem::new` and the start of a continuation run refuse a setup whose `free` index is not below `n_p`.

**`model_compile` keys (P8a).** `opts_json` takes `evaluator` (`jit`, `fallback` or `aot`). With `"emit_c":"<dir>"`, the call writes the model's C text into that folder and returns its path in `model_info` as `c_file`. With `"evaluator":"aot"`, the call also takes `"aot_lib":"<path>"`, the library that `mex` built from that C text. The engine opens it with `libloading` and calls `pc_aot_init` (section 4). These are JSON keys, so the table keeps its 17 members and 128 bytes, and `abi_version` stays 1.

**Error codes.** 0 ok, 1 panic, 2 badArg, 3 badHandle, 4 internal, 5 model, 6 badProblem, 7 nonAutonomous, 8 numeric (`Singular`, `NewtonFailed`, `StepTooSmall`). The gateway alone raises `load` and `unknown`. MATLAB error IDs are `phasecont:engine:<kind>`.

**Runs.**
- `problem_json` is the P3 problem JSON without `u0` and `params`, which come only as buffers: `{"type":"EP","free":"r","start_is_special":false,"options":{...},"switch":"BP1"?}`. On the ABI path a `u0` or `params` key is badProblem. pc-cli keeps accepting both keys. `start_is_special` (new in the shared parser, default false) maps to `EpSetup.start_is_special`. MATLAB sets it when the start Point's label is not `EP`, so a run from an H or BP point does not find its own start again. A special start is not solved for an equilibrium first (the Jacobian is singular at an LP). The engine checks that `u0` solves F = 0 to `tol_f`, takes the tangent from the null vector of `[F_u F_alpha]`, and `Direction` picks its sign. An LP or H start runs. A BP start, where that matrix has rank n-1 (smallest singular value below 1e-8 of the largest), ends with `badProblem` and a message that points to the `Switch` option.
- Curve problems use `{"type":"LP"|"H","free":["a1","a2"],"options":{...}}`, with two distinct parameter names. The state buffer holds the point on the curve, and `u0` and `params` give the start as for an EP run. The row alpha is `a2` (`free[1]`), and `Bounds` bound `a2`. `start_is_special`, `switch` and `detect` are not meaningful for a curve, and the MATLAB layer does not send them. The Hopf-curve `kappa` stays in the engine.
- Cycle problems (P5) use `{"type":"LC","free":"a","ntst":N,"ncol":m,"start":"hopf"|"orbit"|"pd","mesh":[...]?,"options":{...}}`. `ntst` is at least 1, `ncol` is from 2 to 7, and `start` has no default. A `hopf` start takes the equilibrium as the state buffer. `orbit` and `pd` take the orbit at the base points, point-major, followed by the period. `mesh` (the $N_{old}+1$ mesh points of the PD orbit) is required for `pd` and refused otherwise; the old $N$ comes from `mesh` and the old $m$ from the length of the state buffer, so the new `ntst` and `ncol` may differ. `orbit` assumes a uniform mesh. `detect` is refused, since LPC, PD and NS are always tested, and `options.mesh_adapt` is accepted for LC only. A problem whose stride would overflow `usize` is refused when it is parsed. `start_is_special` and `switch` do not apply.
- Curves of cycles (P6) use `{"type":"LPC"|"PD"|"NS","free":["a1","a2"],"ntst":N,"ncol":m,"mesh":[...],"options":{...}}`, with two distinct parameter names. The state buffer holds the orbit of the start point at its base points, point-major, followed by the period, and `params` holds the parameters there. `mesh` is required: the $N_{old}+1$ mesh points of the start, at least two, rising strictly from exactly 0 to exactly 1. The old $N$ comes from `mesh` and the old $m$ from the length of the state buffer, and the start is regridded when they differ from `ntst` and `ncol`. `start` is refused, and so are `detect`, `start_is_special` and `switch`. `options.mesh_adapt` is accepted. The row alpha is `a2`, and `Bounds` bound `a2`.
- Initial value problems (P7b) use `{"type":"IVP","tspan":[t0,t1],"options":{"rtol":1e-8,"atol":1e-10,"max_steps":100000,"hmax":0.5}}`. The state buffer holds the start state `x0` and `params` holds the `n_p` parameters. Time reaches a model that uses `t` as the last entry of `p`, so an IVP run accepts a non-autonomous model, while EP and the curves still refuse one. `tspan` is two finite, different numbers, and `t1 < t0` integrates backward. Every key in `options` is optional. The defaults are `rtol` 1e-8, `atol` 1e-10, `max_steps` 100000 and no cap on the step (`hmax` omitted); `rtol`, `atol` and `hmax` must be positive, `max_steps` a positive integer, and an unknown key is badProblem. `tspan` is refused on every other type, and `free` is refused on IVP (an initial value run has no free parameter). The integrator is the Dormand-Prince 5(4) pair with the first-same-as-last property, the error norm and the starting step of Hairer, Nørsett and Wanner (section 15), and the step size controller with the stabilizing exponent beta = 0.04 as in their `dopri5`. A run has one branch. A row is `[t, x(1..n)]` with stride n + 1, one row for the start and one per accepted step, so the last row sits exactly at `t1` when the run ends with the stop reason `End`. Other stops are `MaxSteps`, `StepTooSmall`, `Cancelled` and `Error` (a non-finite right-hand side). `run_copy` with `what = 0` returns these rows, and an IVP branch has no special points.
- `status_json`: `{"state":"running|done|cancelled","kind":"EP"|"LP"|"H"|"LC"|"LPC"|"PD"|"NS"|"IVP","codim2_labels":[],"n":3,"stride":15,"branches":[{"points":120,"stop":"MaxSteps"|null,"warnings":[...],"specials":[{"label":"H","index":57,"coefficient":0.1,"coefficient2":null,"coefficients":{},"omega":9.6,"normalization":"...","refined":true}]}]}`. `specials` appear once the branch has stopped. `codim2_labels` is empty for EP, `["CP","BT","ZH"]` for LP, `["GH","BT","ZH","HH"]` for H, `["LPC","PD","NS"]` for LC, `["LPPD","LPNS"]` for LPC, `["LPPD","PDNS"]` for PD and `["R1","R2","R3","R4","LPNS","PDNS"]` for NS. The status of every run of cycles (LC, LPC, PD and NS) also has `ntst` and `ncol`. `coefficient2` is null when a point has none, and `coefficients` is an object with the named extras (`{}` when there are none).
- An EP run uses stride 3n + 6, with the row layout `[alpha, u(1..n), eig_re(1..n), eig_im(1..n), psi_LP, psi_BP, psi_H, h, newton_iters]`, with NaN for missing values. An LP curve run uses `[a2, x(1..n), a1, eig_re(1..n), eig_im(1..n), codim2(1..3), h, newton_iters]`, stride 3n + 7, and an H curve run uses `[a2, x(1..n), a1, kappa, eig_re(1..n), eig_im(1..n), codim2(1..4), h, newton_iters]`, stride 3n + 9. An LC run uses `[alpha, u(1..n(Nm+1)), T, mesh(1..N+1), mult_re(1..n), mult_im(1..n), psi_LPC, psi_PD, psi_NS, h, newton_iters]`, stride n(Nm + 1) + N + 2n + 8, where `u` is the orbit at the base points $\tau_j + (k/m) h_j$ and then $\tau_N = 1$, point-major. An LPC or PD curve run uses `[a2, u(1..n(Nm+1)), T, a1, mesh(1..N+1), mult_re(1..n), mult_im(1..n), codim2(1..2), h, newton_iters]`, stride n(Nm + 1) + N + 2n + 8, and an NS curve run uses `[a2, u, T, a1, kappa, mesh(1..N+1), mult_re(1..n), mult_im(1..n), codim2(1..6), h, newton_iters]`, stride n(Nm + 1) + N + 2n + 13. The caller reads the stride from the status JSON. `run_copy` needs `buf_len` equal to `count * stride`, so a caller with the wrong stride gets badArg, and `buf` may be NULL when `count` is 0. It writes at most `count` rows and returns the number in `rows`. Tangents stay in Rust.
- `run_poll` returns 1 when the worker panicked and the kind code when the run failed, with the message in `last_error`. Cancelled is a state in the JSON.
- `free_handle` on a running run cancels and joins inside the barrier. That blocks the MATLAB thread until the worker sees the flag, between Newton iterations.
- `run_start` gives the run its own clone of the model, so a run does not depend on the Model handle after it starts.

**Threads.** Each Model handle owns a thread with an 8 MiB stack. That thread builds the evaluator (the JIT is not `Send`) and serves `model_rhs` and `model_equilibrium` over a channel. Freeing the handle stops and joins the thread. `model_from_ode` parses on an 8 MiB thread. Input over 16 MiB is badArg. The diagnostics JSON keeps the first 1000 diagnostics plus a count of the rest. The registry is one global mutex over `Arc` entries. No call holds the lock across engine work, and a poisoned lock is recovered, so one panic does not break later calls.

**Gateway.** `mex/pckern_gateway.c` is compiled with `mex -R2018a`. The first command that needs `pckern` loads it from the folder that holds the MEX file: on Windows the gateway finds its own file with `GetModuleHandleExW` and `GetModuleFileNameW` and loads `pckern.dll` by full path with `LoadLibraryExW` and `LOAD_WITH_ALTERED_SEARCH_PATH`, and elsewhere it uses `dlopen`. On Linux and macOS the gateway finds its own path with `dladdr(&mexFunction)` and calls `dlopen` with the full path. The commands are `version`, `model_from_ode`, `model_compile`, `model_info`, `model_rhs`, `model_rhs_batch`, `model_equilibrium`, `run_start`, `run_poll`, `run_copy`, `run_cancel` and `free`. `model_rhs_batch(m, X, p [, t])` takes X as n-by-k, one point per column, and returns the n-by-k right-hand sides. The gateway loops over the columns with `model_rhs` or `model_rhs_t`, so one call serves a direction field or a nullcline grid and the C ABI has no batch entry. The binaries sit side by side: `pckern.<mexext>` plus `pckern.dll`, `libpckern.so` or `libpckern.dylib` in `toolbox/bin/<arch>/`. The gateway then calls `pc_api_v1`, requires `abi_version` to equal its own and `struct_size` to be at least the size of its own table, and dispatches on a command string. A failed load raises `phasecont:engine:load`, and the next call tries again. The gateway never unloads `pckern`, since worker threads may still run its code. Loading the library at runtime keeps the gateway independent of the Rust toolchain (MSVC or MinGW on Windows).

**Data rules.** Metadata and options cross as JSON. Numeric arrays are copied into buffers that MATLAB allocated. Rust never holds a pointer into MATLAB memory after a call returns. MATLAB writes numeric option fields with `sprintf('%.17g')`, not `jsonencode`, since `jsonencode` can drop digits. Rust writes JSON numbers in serde_json's shortest round-trip form, and MATLAB reads them with `jsondecode`.

## 11. MATLAB API sketch

```matlab
m  = phasecont.Model("ml.ode");                  % parse + JIT in Rust
eq = phasecont.equilibrium(m, x0, Params=p0);
br = phasecont.continuation(m, eq, Free="I", MaxSteps=500);
h  = br.points("H");
lc = phasecont.continuation(m, h(1), Type="LC", Free="I", NTST=40, NCOL=4);
plot(br); hold on; plot(lc)
s = phasecont.Session(m, {br, lc}); save(s, "ml.pcsession.json");
phasecont.session.replay("ml.pcsession.json");   % headless rerun
```

The P7a calls, the first working part of this sketch:

```matlab
m   = phasecont.Model("lorenz.ode", Evaluator="jit");     % or "fallback"
f   = m.rhs(x, p);                                        % p: vector or struct, default m.ParameterValues
f   = m.rhs(x, p, t);                                     % a model that uses t (m.Autonomous false) needs t
eq  = phasecont.equilibrium(m, x0, Params=struct(b=8/3)); % -> phasecont.Point, Label "EP"
br  = phasecont.continuation(m, eq, Free="r", MaxSteps=500, Bounds=[0 30], Plot=true);
brs = phasecont.continuation(m, eq, Free="r", Switch="BP1");  % 1x2 Branch: followed and switched
h   = br.points("H");                                     % Point array; br.points() gives all specials
plot(br, Var="x");                                        % stable solid, unstable dashed, labeled specials
```

- **Branch plots.** `plot(br)` draws stable runs as solid lines (Tag `pc-stable`), unstable runs as dashed lines (`pc-unstable`), and each special point as a marker (`pc-special`) with a text label (`pc-label`). It uses `newplot`, so `hold on` is respected, and `Parent=ax` draws into an axes without resetting it. `continuation(Plot=true)` ends with this plot.
- **Two-parameter curves.** `continuation(m, pt, Type="LP"|"H", Free=["a1" "a2"], ...)` continues an LP or Hopf curve from an LP or H point `pt` (or from any state Point). `Type` defaults to `"EP"`. `Free` takes two distinct parameter names for a curve, and `Bounds` bound `a2`. `Detect`, `Switch` and `Var` apply to EP only and raise `phasecont:model:badOption` on a curve. The Branch has `Type`, `Free`, `Alpha` (N-by-2, columns `a1` and `a2`), `Codim2` (the codim-2 test values) and `Codim2Labels` (`["CP" "BT" "ZH"]` for LP, `["GH" "BT" "ZH" "HH"]` for H). Special points carry the label of the codim-2 point. `Point.Data` has `coefficient` and `coefficient2` (BT: a and b; ZH: b and c; HH: Re P11 and NaN; GH: l2; CP: the cusp coefficient), plus `values`, a struct of the named extras (ZH: `re_d`, `im_d`, `e`; HH: `re_p12`, `re_p21`, `re_p22`, `omega2`). `Branch.Stable` is false on a curve, since the curve has no stability of its own. `plot(br)` draws a curve in the (a1, a2) plane as one solid line (Tag `pc-curve`) with the special markers and labels as for EP, and `continuation(Plot=true)` grows the same plot.
- **Limit cycles (P5).** `continuation(m, start, Type="LC", Free="a", NTST=N, NCOL=m, ...)` continues periodic orbits. `NTST` and `NCOL` are required for LC, have no default and raise `phasecont:model:badOption` on the other types; LC refuses `Detect` and `Switch`, and `Var` works as for EP. The start is a Hopf point (or any equilibrium Point or state vector, which the engine checks; a state with no complex pair fails the run with `phasecont:engine:badProblem`) or a PD point of an earlier LC branch, which starts the period-doubled branch. Any other cycle Point raises `phasecont:model:badStart`. The Branch has `Period` (N-by-1), `Mesh` (N-by-(NTST+1)), `NTST` and `NCOL`. `U` holds the orbit at the NTST·NCOL+1 base points, point-major. `Eigenvalues` holds the Floquet multipliers and `Tests` the LPC, PD and NS test values. `Stable` is true when every multiplier except the one closest to 1 has modulus below 1. `[t, x] = lc.cycle(i)` returns the base-point times from 0 to the period and the states there, with the last row equal to the first. A PD special point carries `Data.period`, `Data.mesh`, `Data.ntst` and `Data.ncol`. `plot(lc, Var="x")` draws the maximum and minimum of x over the orbit against the parameter, stable solid and unstable dashed, with the y label "max, min x", and `continuation(Plot=true)` grows the maximum while the run goes and ends with the same plot. The engine option `mesh_adapt` is not exposed.
- **Curves of cycles (P6).** `continuation(m, pt, Type="LPC"|"PD"|"NS", Free=["a1" "a2"], ...)` continues a curve of fold, period doubling or Neimark-Sacker points of cycles from a special point `pt` with the same label on an LC branch. `Free` takes two distinct parameter names, and `Bounds` bound `a2`. `NTST` and `NCOL` default to `pt.Data.ntst` and `pt.Data.ncol`, and other values regrid the start orbit. `Detect`, `Switch` and `Var` raise `phasecont:model:badOption`, an unknown name raises `phasecont:model:unknownParameter`, and a start that is not such a Point raises `phasecont:model:badStart`. The Branch has `Alpha` (N-by-2, columns `a1` and `a2`), `Period`, `Mesh`, `NTST` and `NCOL` as for LC, `Kappa` (N-by-1, NS only), `Codim2` and `Codim2Labels` (`["LPPD" "LPNS"]` for LPC, `["LPPD" "PDNS"]` for PD, `["R1" "R2" "R3" "R4" "LPNS" "PDNS"]` for NS). `Eigenvalues` holds the Floquet multipliers, and `Stable` is false. `br.cycle(i)` works as for LC. Special points carry `Data.period`, `Data.mesh`, `Data.ntst` and `Data.ncol`, plus `Data.kappa` on an NS curve. `plot(br)` draws the curve in the (a1, a2) plane as for an LP or H curve.
- **Handle classes.** Only `Model` and the internal `phasecont.internal.Run` own Rust handles, and their `delete` frees the handle. `Branch` is a handle class that holds copied data, and `Point` is a value class. `continuation` blocks and polls about every 50 ms with `drawnow`. It grows the plot when `Plot=true`, and Ctrl+C cancels the run and returns nothing.
- **Time in `rhs`.** `m.rhs(x, p, t)` takes a finite real scalar `t`, else `phasecont:model:badTime`. A model with `Autonomous == false` called without `t` raises `phasecont:model:missingTime`; the toolbox never assumes `t = 0`. A model that does not use `t` accepts `t` and ignores it. `equilibrium` and `continuation` still need an autonomous model and raise `phasecont:engine:nonAutonomous`.
- **P8a additions.** `phasecont.config("Codegen", "aot"|"jit")` selects the code path, and `Model(file, Evaluator="aot")` asks for AOT for one model. `Model.AotInfo` has the fields `Compiler`, `Version`, `Flags` and `Library`. They hold the compiler name, its version, the flags used to build the library and the absolute path of that library. They are empty when the model does not run on AOT.
- **Trajectories (P7b).** `tr = phasecont.simulate(m, x0, [t0 t1], Params=, RelTol=1e-8, AbsTol=1e-10, MaxSteps=100000, Hmax=Inf)` integrates an initial value problem with the IVP run of section 10 and returns a `phasecont.Trajectory` handle. `x0` is a state vector or a `Point`, and `t1 < t0` integrates backward. `Params` is a struct or a vector as for `equilibrium`, and a model that uses `t` is integrated with `t` running from `t0` to `t1`. The defaults are the ones in the call above. `Hmax=Inf` means no cap on the step, so a plot of the rows is as smooth as the steps are short, since the solver has no dense output. The Trajectory has `T` (N-by-1), `X` (N-by-n), `X0`, `TSpan`, `P`, `Stop` (`"End"` when the run reached `t1`), `Warnings` and `Model`, and `plot(tr, Var=[x y])` draws a phase plane (Tag `pc-trajectory`) or, with one name, that state against `t`. A run that stops for another reason (`MaxSteps`, `StepTooSmall` or `Error`) warns with `phasecont:simulate:stoppedEarly` and keeps the steps up to then. Ctrl+C cancels the run and returns nothing. `simulate` refuses `Free` and every other option it does not list.
- **Models from text (P7b).** `m = phasecont.Model.fromSource(text, Name="ml.ode", Evaluator="jit")` builds a model from source text, with `Name` used in diagnostics and in `m.File`, and no file read. `m.SourceSha256` is the hash from `model_info`, and `m.Options` is a struct of the `@` options with lower-case field names and string values. `m.rhs(X, p[, t])` accepts an n-by-k matrix X, one point per column, and returns n-by-k through `model_rhs_batch`; a single point keeps its old behavior.
- **Sessions (P7b).** `s = phasecont.Session(m, {br, lc, tr, eq})` holds a model and the Branch, Trajectory and Point objects that `continuation`, `simulate` and `equilibrium` made on it. `save(s, file)` writes a session file and `phasecont.Session.load(file)` reads one. A load builds the Model again from the stored source, checks the hash (`phasecont:session:hashMismatch`) and rebuilds the items from the stored rows without running anything. An item with no run of its own, such as a Point taken from a Branch, raises `phasecont:session:noRecord`, items from another Model raise `phasecont:session:modelMismatch`, and a file that is not a version 1 session raises `phasecont:session:badFile`, or `phasecont:session:badRecord` when a run in it is damaged. Items from one run, such as the two branches of a `Switch`, are stored once.
- **Replay (P7b).** `r = phasecont.session.replay(file, Tol=1e-8, Evaluator=<recorded>)` runs every stored run again through the same engine calls and returns one struct per run with `Kind`, `Type`, `Rows` (`Stored` and `New`), `Identical`, `MaxRelDiff`, `LabelsMatch`, `StopMatch` and `Pass`. A run passes when its shapes, the positions of NaN and Inf, the point counts, the labels with their indices and the stop reasons agree and `MaxRelDiff`, the largest `|a-b|/max(1,|b|)` over the finite numbers with `a` the new value, is at most `Tol`. `Identical` reports bit-for-bit equality and does not decide the pass. The Rust replay (`pc-cli session replay`) of the full Morris-Lecar diagram is `Identical` on all six runs on the build machine. Other platforms and the other evaluators can differ in the last digits, and an adaptive run can then take another step sequence, so a cross-platform replay can fail the shape check. That has not been tested and STATUS lists it as open.
- **Session JSON v1 (`*.pcsession.json`).** One JSON object with `format` (`"phasecont-session"`), `version` (1), `created` (UTC, ISO 8601), `engine` (`name`, `version`), `platform`, `matlab` (the release), `precision` (`"f64"`), `model` (`name`, the file name only, `source`, `sha256`, `evaluator`), `runs` and `items`. A run has `kind` (`"continuation"`, `"simulate"` or `"equilibrium"`), `problem`, `type`, `free`, `grid`, `n`, `u0` and `p`. A continuation or simulate run also has `status`, `rows` and `specials`, and an equilibrium run has `result` with `u`, `eig_re` and `eig_im`. `problem` and `status` are the exact strings that went to `run_start` and came back from `run_poll` (for an equilibrium, `problem` is the options string that went to `model_equilibrium`), since a JSON parse and print can change digits. Every array of numbers is a block `{"shape":[r,c],"f64le":"<base64>"}` holding the r-by-c matrix as row-major little-endian binary64 in base64 (RFC 4648), so NaN, Inf and -0 survive and MATLAB and Rust read the same bits. `rows` and `specials` hold one block per branch with the shape `[count, stride]`. `items` lists `{"run":i,"branch":j}` pairs, one per saved object, with 1-based indices. `model.source` is hashed as given, with no newline or encoding change, and `.gitattributes` keeps fixtures at LF so they hash the same on every runner.
- **Phase-plane app (P7b).** `app = phasecont.app.PhasePlane(m)` opens a programmatic `uifigure` for a Model with at least two variables, and `PhasePlane("diagram.pcsession.json")` opens a saved session instead. `Visible="off"` builds the same figure hidden. The figure has a `uiaxes`, dropdowns for the two plotted states, a table of the parameters, a table of the other states (the slice) when n > 2, a field for T, Forward, Backward or Both, an Integrate or Equilibrium click mode, check boxes for the direction field and the nullclines, and Clear, Save session and Open session. A click in Integrate mode runs `simulate` from that point, with the other states at their slice values. A click in Equilibrium mode runs `equilibrium` and marks the result filled when every eigenvalue has a negative real part and open otherwise (Tags `pc-equilibrium-stable` and `pc-equilibrium-unstable`). The direction field (Tag `pc-field`) is a normalized quiver on a 20-by-20 grid, and the nullclines (Tag `pc-nullcline`, one line per variable with the name in `UserData`) are the zero contours of the two plotted right-hand sides on a 101-by-101 grid, both through the batch `rhs`. Trajectories use Tag `pc-trajectory` and start markers use `pc-start`. Methods for scripts and tests: `clickAt`, `addTrajectory`, `findEquilibrium`, `setParameter`, `setSlice`, `setAxes`, `setWindow`, `setT`, `setDirection`, `setClickMode`, `showField`, `showNullclines`, `clear`, `saveSession`, `openSession` and `delete`. The choices that change what the user sees:
  - The `@` options `xp` and `yp` name the plotted variables (the first two by default). `xlo`, `xhi`, `ylo` and `yhi` give the view range, and only for the pair the model names; any other pair starts at the start value plus or minus max(1, |value|). `total` gives T. When the model sets no `total`, T stays empty and a click raises `phasecont:app:missingTime`; the app does not pick a T. The grid sizes and the view range affect only the display.
  - The app integrates with Dormand-Prince only, and shows a note when the model sets `meth`, `dt` or `bounds`, which it ignores.
  - It passes `Hmax = |T|/500` so a drawn trajectory is smooth. `simulate` itself keeps `Hmax = Inf`.
  - A model that uses `t` integrates with `t` running from 0 to T (Forward), from 0 to -T (Backward) or both.
  - A change of a parameter removes the trajectories and equilibria, which belong to the old values, and redraws the field and the nullclines. A change of a slice value redraws the field and the nullclines and keeps the runs. Clear removes every item.
  - The equilibrium search, the direction field and the nullclines need an autonomous model and raise `phasecont:app:nonAutonomous` for a model that uses `t`. A click can still integrate such a model.
  - A saved session holds the trajectories and equilibria. Branches from a session opened in the app are kept, saved again and not drawn.
  - Error IDs: `phasecont:app:badSource`, `needTwoStates`, `missingTime`, `nonAutonomous`, `unknownParameter`, `unknownVariable`, `sameAxes`, `badValue` and `badWindow`.
- **Apps.** They are programmatic `uifigure` code, not `.mlapp` files.

## 12. Build, CI and packaging

**`buildfile.m` tasks.**
- `cargo`: release build for the current platform. For Windows the allow-list check is `pc-cli dll-imports FILE`, a small PE reader in this repo. The allow-list holds the Windows system DLLs, `msvcrt.dll` and `api-ms-win-*`, and the task fails when `pckern.dll` imports anything else. Before a Windows build in WSL it checks that `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` report the same version, and stops with both versions in the message when they differ. On Windows MATLAB with the repo under `\\wsl.localhost`, the task runs `wsl.exe` to call `scripts/build-pckern win64 <stage>`. With `PHASECONT_PREBUILT` set, it copies prebuilt binaries instead.
- `mex`: build the gateway.
- `test`: run `matlab.unittest` with JUnit output.
- `package`: build the `.mltbx` with `matlab.addons.toolbox.ToolboxOptions`, using a fixed UUID that never changes between versions. Version 1.0.0, set in `buildfile.m`, has `MinimumMatlabRelease` R2026b and ships win64, glnxa64 and maca64. MATLAB R2026b no longer runs on Intel Macs, so there is no maci64 build. `scripts/packageToolbox.m` stages the package in `build/package/PhaseCont/` from `toolbox/`, the binaries in `bin/<arch>/`, the examples, `LICENSE` and `THIRD_PARTY_NOTICES.md`, and fails with `phasecont:package:missingBinary` when a MEX file or library of a listed platform is absent. With `PHASECONT_BINROOT` set to a folder holding `win64/`, `glnxa64/` and `maca64/`, the package covers all three platforms (the release workflow sets it); without it, the package covers the current platform only. Before packaging, the task writes the license text of every crate linked into `pckern` into the toolbox, next to `LICENSE` and `THIRD_PARTY_NOTICES.md`. `pc-cli licenses --target <triple> [--target <triple> ...] --out FILE` builds that file from `cargo metadata --filter-platform` and the license files in each crate's source, and fails when a crate has none. With the optional `--overrides DIR`, a crate that ships no license file takes `LICENSE` and `SOURCE` from `DIR/<crate>/`.

**Session replay (P7b).** `pc-cli session replay [--tol X] [--fallback] [--] FILE` reads a `*.pcsession.json` file, rebuilds the model from the stored source, and checks the stored SHA-256 against it. It then reruns every run through the same pckern functions that MATLAB calls: `model_from_ode`, `model_compile`, `run_start`, `run_poll` and `run_copy` for continuations and trajectories, and the Newton entry for equilibria. `--tol` sets the pass rule of §11 (default 1e-8) and `--fallback` evaluates the model with the f64 interpreter in place of the JIT. The command prints a JSON report to stdout with `pass`, `tol`, `engine`, `evaluator` and, for each run, `index`, `kind`, `type`, `pass`, `identical`, `max_rel_diff`, `rows_match`, `labels_match`, `stop_match`, `seconds` and `message`. The exit code is 0 when every run passes, 1 when a run does not match or cannot be rerun, and 2 for an unreadable or unsupported file, a hash mismatch or a usage error. A test in `crates/pc-cli/tests/` builds the full Morris-Lecar diagram, writes a session, and checks that the replay passes with `identical` true on every run; it also checks that a changed row exits 1, and that a changed source and an unknown version exit 2.

**C header.** The C header `crates/pc-ffi/include/pckern.h` is hand-written. A Rust test checks the table size and field offsets against it.

**Base MATLAB check.** A test in `tests/` runs `matlab.codetools.requiredFilesAndProducts` on every file under `toolbox/` and fails when it lists a product other than MATLAB. Files of the optional `gpuArray` route may also list Parallel Computing Toolbox, which is allowed only on that route. The test starts with the first MATLAB code under `toolbox/`.

**GitHub Actions.** Every job runs on a GitHub-hosted runner, so a pull request from a fork never runs on a machine of ours. Every third-party action is pinned by commit SHA, no checkout keeps the token (`persist-credentials: false`), and the workflows have `contents: read` except the `release` job. `setup-matlab` licenses MATLAB R2026b for a public repository without a secret.
- **`ci.yml`:**
  - `rust`: fmt, clippy and test on Ubuntu, Windows and macOS, with `cargo deny` on Ubuntu only, since licenses do not depend on the OS.
  - `pckern-win`: on Ubuntu, cross-builds `pckern.dll` with MinGW, checks its imports and host paths, and uploads the DLL and the license texts of the three targets as artifacts.
  - `matlab`: on Windows, Ubuntu and macOS, runs `buildtool cargo mex test package`, then `scripts/cleanInstallCheck.m` from `RUNNER_TEMP` in three steps (install, example, remove). Windows takes the DLL from `pckern-win` through `PHASECONT_PREBUILT`; Linux and macOS build the library from source. On Linux it also runs `scripts/check-glibc` on the MEX file.
  - `asan`: `pc-ffi` tests and `pc-cli cont` under AddressSanitizer, on pull requests, the nightly schedule and manual runs.
- **`release.yml`:** runs on a `v*` tag. A manual run (`workflow_dispatch`) is a dry run that builds, tests and installs the package and uploads it as an artifact, but creates no release.
  - `lib-win`, `lib-linux`, `lib-mac`: build the three libraries. Linux uses `cargo zigbuild --target x86_64-unknown-linux-gnu.2.28`, and macOS sets `MACOSX_DEPLOYMENT_TARGET` to 15.0 and links with the install name `@rpath/libpckern.dylib`, since the default install name is the absolute build path. Each build passes `--remap-path-prefix` for the checkout and the cargo home, and `scripts/check-host-paths` fails when the binary still holds a runner path. `scripts/check-glibc` fails when a Linux binary needs a symbol version above `GLIBC_2.28`.
  - `licenses`: `pc-cli licenses` for the three targets.
  - `mex`: on each platform, the gateway build and the test suites with the prebuilt library, then the `bin/<arch>` folder as an artifact. The gateway pins its `dl*` symbols to `GLIBC_2.2.5` on Linux x86_64, so a MEX file built on a current Ubuntu loads on glibc 2.28.
  - `package`: one `.mltbx` from the three `bin/<arch>` folders (`PHASECONT_BINROOT`) and the license texts.
  - `install`: on each platform, the three steps of `scripts/cleanInstallCheck.m` with the `.mltbx` from `package`.
  - `release`: only for a tag. It checks that the tag equals `v` plus the `buildfile.m` version, writes `SHA256SUMS` and runs `gh release create` with the `.mltbx` attached. Only this job has `contents: write`.
- **`cleanInstallCheck.m`:** installs the `.mltbx` into the MATLAB Add-Ons folder and asserts that `phasecont.Model` resolves into that add-on folder and not the repo, or fails with `phasecont:install:notAddon`. The example runs in a MATLAB process that did not install the toolbox, and the script then uninstalls it. `PHASECONT_INSTALL_STEP` set to `install`, `example` or `remove` runs one step, and the CI jobs run each step in its own MATLAB, because a MATLAB licensed on a GitHub-hosted runner cannot start a second licensed MATLAB. With the variable unset, the script runs all three and starts the example MATLAB itself. An install made under `-batch` is not registered for later sessions on R2026b, though its files stay on disk, so the remove step installs the same `.mltbx` again when the toolbox is not listed and its folder exists, then uninstalls it.
- **Hosted runs:** the `matlab` legs first ran on 2026-11-04, after the repo went public, and passed on all three runners at ee109cc. The `release.yml` dry run passed every job except `release` on 2026-11-04 at 21fc71e.
- **Upgrades.** The gateway never unloads `pckern` while MATLAB runs, because worker threads may still run its code. Installing a new version or uninstalling in a session that used PhaseCont fails until MATLAB exits; the README says to close MATLAB first.
- **File Exchange.** The R2026b help says a package with MEX files cannot be submitted to File Exchange, and it does not say whether a GitHub-linked entry is exempt. GitHub Releases is the download route for 1.0 (see `docs/STATUS.md`).

**macOS quarantine.** Downloaded binaries may be quarantined. Test installing from the `.mltbx` on a clean Mac account, and document any `xattr` step if one turns out to be needed.

## 13. Later phases (pointers only)

- **GPU ensemble engine** (`pc-ensemble`). One thread runs one trajectory. A kernel advances each trajectory by at most K steps and the host relaunches until all are done, with K sized so one launch takes about 100 ms (the Windows watchdog). A GPU speeds up ensembles only: one run and all continuation stay on the CPU in `f64`. Backends:
  - `cpu`: rayon; the reference for every other backend, in both precisions.
  - `cuda`: `cudarc`, driver API plus NVRTC; primary for NVIDIA. Spike S4 compiled and ran an `sm_120` kernel with it on the developer's GPU, inside MATLAB R2026b and in WSL. NVRTC is not bundled, since it is a proprietary NVIDIA binary. The engine looks for it at run time, first MATLAB's own copy and then a CUDA toolkit on the machine. With neither, an NVIDIA GPU runs through `opencl`.
  - `opencl`: the `opencl3` crate with its `dynamic` feature, so the library is loaded at run time and never linked; for AMD, Intel and NVIDIA. AMD and Intel have no other backend. The kernel uses `f` literals and `-cl-single-precision-constant`, and double precision needs `cl_khr_fp64`.
  - `metal`: the `objc2-metal` crate, macOS only, `f32` only (MSL has no `double`).
  - There is no `hip` backend. `hiprtc` is not shipped with consumer AMD drivers.
  - Kernels are emitted as text for each dialect (C, CUDA C, OpenCL C, MSL) from one shared body and compiled at runtime. The engine uses `f64` everywhere. `f32` exists only in GPU ensemble screening, and those results are labeled as single precision and recomputed in `f64` before they feed continuation, normal forms or Floquet multipliers.
  - A job has no default precision and no default time span, and a tolerance that single precision cannot resolve (`rtol < 13 * 2^-23`) is an error, not a clamp. Every result carries its backend, device, driver and precision.
  - `cuda` host code calls `is_culib_present` for the driver and NVRTC before any other `cudarc` call, because `cudarc` panics when it cannot load a library. It compiles kernels through the raw NVRTC calls and destroys the program on every path: when a compile fails and reading the program log fails too, `cudarc`'s `compile_ptx_with_opts` panics, and the `Drop` of its program panics again if destroying the program also fails. That second panic aborts the process, which `catch_unwind` cannot stop.
  - NVRTC contracts `a*b + c` into an FMA by default (`--fmad`, "Default: true" in `/usr/local/cuda-13.4/include/nvrtc.h:960-965`). Kernels are compiled with `--fmad=false`, and a test that compares a GPU result with a CPU reference at the level of single ulps rounds each operation once in the reference. It does not get that rounding from `f32::mul_add` on `x86_64-pc-windows-gnu` at the baseline target features.
  - `pc-cli gpu list` prints every backend and device. `pc-cli gpu selftest --backend B --precision P [--device N]` checks closed-form error, determinism and, for a device that is not the CPU, bit-exactness with the CPU backend. It exits 0 on a pass, 1 on a failure and 3 when the backend is not built or the device does not exist, which is never a pass. `pc-cli ensemble MODEL JOB` runs a job file, and `pc-cli ensemble --bench` times it.
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
| P8a | Package v0.3 (win64) | `.mltbx` installs on a clean MATLAB and runs the FitzHugh-Nagumo example |
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
- Hairer, E., Nørsett, S. P., and Wanner, G. *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993. Sections II.4 and II.5 for the step-size control and the starting step.
- Dormand, J. R., and Prince, P. J. "A family of embedded Runge-Kutta formulae." *J. Comput. Appl. Math.*, 1980.
- NIST. *Secure Hash Standard (SHS)*, FIPS 180-4, 2015.
- Josefsson, S. *The Base16, Base32, and Base64 Data Encodings*, RFC 4648, 2006.
