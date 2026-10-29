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

**macOS arm64.** Confirm that JIT pages can be mapped executable inside the MATLAB process. Phase 0 ran only on Windows and WSL, so Phase 2 runs the Rust side of this check on the macOS runner of section 12, and Phase 7a runs the part inside MATLAB once the MEX gateway exists. `cranelift-jit` 0.137.0 accepts `is_pic=true` only on x86_64. Its default `SystemMemoryProvider` maps code pages read-write without `MAP_JIT` and makes them read-execute with `mprotect`, which the hardened runtime allows only with `com.apple.security.cs.allow-unsigned-executable-memory` or `com.apple.security.cs.disable-executable-page-protection`. `codesign -d --entitlements - <matlabroot>/bin/maca64/MATLAB`, where `<matlabroot>` is the folder that `matlabroot` returns, lists the entitlements MATLAB has. Without either entitlement, `pc-jit` needs its own `JITMemoryProvider`, set with `JITBuilder::memory_provider`, that maps pages with `MAP_JIT`, which needs `com.apple.security.cs.allow-jit`, and switches them with `pthread_jit_write_protect_np`. With none of the three, rule 5 of `CLAUDE.md` still holds: models run on an `f64` evaluator in Rust built from the same code as the dual-number interpreter of section 3, and PhaseCont warns that this is slower than compiled code. The AOT path stays the optional speed-up on Macs that have the Xcode Command Line Tools, selected with `phasecont.config("Codegen","aot")`. No Mac user is required to install a compiler.

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
- For $n > 100$, Phase 3 runs with Hopf detection off and puts a warning in the run status. Phase 4 adds the minimally augmented test function from Govaerts.

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

**Minimally augmented Hopf curve.** Phase 4 continues Hopf curves with the bordered $A^2 + \kappa I$ formulation from Govaerts (2000), with $\kappa = \omega^2$ as an extra unknown. The extra test function $\kappa \to 0$ detects BT.

**Codim-2 points.**
- On LP curves: CP ($a = 0$), BT, ZH.
- On H curves: GH ($l_1 = 0$), BT, ZH, HH.
- Normal form coefficients come from Kuznetsov (1999), "Numerical normalization techniques for all codim 2 bifurcations of equilibria in ODEs."

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
  int32_t (*model_compile)(uint64_t model, const char* opts_json);        /* {"evaluator":"jit"|"fallback"} */
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
  int32_t (*model_info)(uint64_t model, char** info_json);                /* names, values, init, n, n_p, evaluator, jit_max_frame */
  int32_t (*model_equilibrium)(uint64_t model, const double* u0, size_t nu, const double* p, size_t np,
                               const char* opts_json, double* u_out, size_t nu_out,
                               double* eig_out, size_t neig_out);          /* eig as [re; im], 2n */
  int32_t (*debug_panic)(int32_t where, uint64_t* run);                   /* 0: this thread, 1: a run whose worker panics */
  const char* (*version)(void);
} pc_api_v1_t;
```

**Error codes.** 0 ok, 1 panic, 2 badArg, 3 badHandle, 4 internal, 5 model, 6 badProblem, 7 nonAutonomous, 8 numeric (`Singular`, `NewtonFailed`, `StepTooSmall`). The gateway alone raises `load` and `unknown`. MATLAB error IDs are `phasecont:engine:<kind>`.

**Runs.**
- `problem_json` is the P3 problem JSON without `u0` and `params`, which come only as buffers: `{"type":"EP","free":"r","start_is_special":false,"options":{...},"switch":"BP1"?}`. On the ABI path a `u0` or `params` key is badProblem. pc-cli keeps accepting both keys. `start_is_special` (new in the shared parser, default false) maps to `EpSetup.start_is_special`. MATLAB sets it when the start Point's label is not `EP`, so a run from an H or BP point does not find its own start again.
- `status_json`: `{"state":"running|done|cancelled","n":3,"stride":15,"branches":[{"points":120,"stop":"MaxSteps"|null,"warnings":[...],"specials":[{"label":"H","index":57,"coefficient":0.1,"omega":9.6,"normalization":"...","refined":true}]}]}`. `specials` appear once the branch has stopped.
- Points and specials use stride 3n + 6, with the row layout `[alpha, u(1..n), eig_re(1..n), eig_im(1..n), psi_LP, psi_BP, psi_H, h, newton_iters]`, with NaN for missing values. Tangents stay in Rust.
- `run_poll` returns 1 when the worker panicked and the kind code when the run failed, with the message in `last_error`. Cancelled is a state in the JSON.
- `free_handle` on a running run cancels and joins inside the barrier. That blocks the MATLAB thread until the worker sees the flag, between Newton iterations.

**Threads.** Each Model handle owns a thread with an 8 MiB stack. That thread builds the evaluator (the JIT is not `Send`) and serves `model_rhs` and `model_equilibrium` over a channel. Freeing the handle stops and joins the thread. `model_from_ode` parses on an 8 MiB thread. Input over 16 MiB is badArg. The diagnostics JSON keeps the first 1000 diagnostics plus a count of the rest. The registry is one global mutex over `Arc` entries. No call holds the lock across engine work, and a poisoned lock is recovered, so one panic does not break later calls.

**Gateway.** `mex/pckern_gateway.c` is compiled with `mex -R2018a`. The first command that needs `pckern` loads it from the folder that holds the MEX file: on Windows the gateway finds its own file with `GetModuleHandleExW` and `GetModuleFileNameW` and loads `pckern.dll` by full path with `LoadLibraryExW` and `LOAD_WITH_ALTERED_SEARCH_PATH`, and elsewhere it uses `dlopen`. On Linux and macOS the gateway finds its own path with `dladdr(&mexFunction)` and calls `dlopen` with the full path. The binaries sit side by side: `pckern.<mexext>` plus `pckern.dll`, `libpckern.so` or `libpckern.dylib` in `toolbox/bin/<arch>/`. The gateway then calls `pc_api_v1`, requires `abi_version` to equal its own and `struct_size` to be at least the size of its own table, and dispatches on a command string. A failed load raises `phasecont:engine:load`, and the next call tries again. The gateway never unloads `pckern`, since worker threads may still run its code. Loading the library at runtime keeps the gateway independent of the Rust toolchain (MSVC or MinGW on Windows).

**Data rules.** Metadata and options cross as JSON. Numeric arrays are copied into buffers that MATLAB allocated. Rust never holds a pointer into MATLAB memory after a call returns. MATLAB writes numeric option fields with `sprintf('%.17g')`, not `jsonencode`, since `jsonencode` can drop digits. Rust writes JSON numbers in serde_json's shortest round-trip form, and MATLAB reads them with `jsondecode`.

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

The P7a calls, the first working part of this sketch:

```matlab
m   = phasecont.Model("lorenz.ode", Evaluator="jit");     % or "fallback"
f   = m.rhs(x, p);                                        % p: vector or struct, default m.ParameterValues
eq  = phasecont.equilibrium(m, x0, Params=struct(b=8/3)); % -> phasecont.Point, Label "EP"
br  = phasecont.continuation(m, eq, Free="r", MaxSteps=500, Bounds=[0 30], Plot=true);
brs = phasecont.continuation(m, eq, Free="r", Switch="BP1");  % 1x2 Branch: followed and switched
h   = br.points("H");                                     % Point array; br.points() gives all specials
plot(br, Var="x");                                        % stable solid, unstable dashed, labeled specials
```

- **Handle classes.** Only `Model` and the internal `phasecont.internal.Run` own Rust handles, and their `delete` frees the handle. `Branch` is a handle class that holds copied data, and `Point` is a value class. `continuation` blocks and polls about every 50 ms with `drawnow`. It grows the plot when `Plot=true`, and Ctrl+C cancels the run and returns nothing.
- **Apps.** They are programmatic `uifigure` code, not `.mlapp` files.
- **Session JSON.** Records the model source and its SHA-256, the engine name and version, the platform, the MATLAB release, every run's starting point and options, and the precision used.

## 12. Build, CI and packaging

**`buildfile.m` tasks.**
- `cargo`: release build for the current platform. For Windows the allow-list check is `pc-cli dll-imports FILE`, a small PE reader in this repo. The allow-list holds the Windows system DLLs, `msvcrt.dll` and `api-ms-win-*`, and the task fails when `pckern.dll` imports anything else. Before a Windows build in WSL it checks that `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` report the same version, and stops with both versions in the message when they differ. On Windows MATLAB with the repo under `\\wsl.localhost`, the task runs `wsl.exe` to call `scripts/build-pckern win64 <stage>`. With `PHASECONT_PREBUILT` set, it copies prebuilt binaries instead.
- `mex`: build the gateway.
- `test`: run `matlab.unittest` with JUnit output.
- `package`: build the `.mltbx` with `matlab.addons.toolbox.ToolboxOptions`, using a fixed UUID that never changes between versions. Before packaging, the task writes the license text of every crate linked into `pckern` on each platform into the toolbox, next to `LICENSE` and `THIRD_PARTY_NOTICES.md`. A Rust tool in this repo builds that file from `cargo metadata --filter-platform` and the license files in each crate's source.

**C header.** The C header `crates/pc-ffi/include/pckern.h` is hand-written. A Rust test checks the table size and field offsets against it.

**Base MATLAB check.** A test in `tests/` runs `matlab.codetools.requiredFilesAndProducts` on every file under `toolbox/` and fails when it lists a product other than MATLAB. Files of the optional `gpuArray` route may also list Parallel Computing Toolbox (rule 8 of `CLAUDE.md`). The test starts with the first MATLAB code under `toolbox/`.

**GitHub Actions.**
- **Rust job:** fmt, clippy and test on Ubuntu, Windows and macOS, with `cargo deny` on Ubuntu only, since licenses do not depend on the OS.
- **`pckern-win` job:** on Ubuntu, cross-builds `pckern.dll` with MinGW, checks its imports and uploads the DLL as an artifact.
- **MATLAB job:** `matlab-actions/setup-matlab` and `matlab-actions/run-build` on all three operating systems. It uses the `MLM_LICENSE_TOKEN` secret for the batch license. The Windows job uses the `pckern-win` artifact, so the Windows runner needs no MinGW.
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
