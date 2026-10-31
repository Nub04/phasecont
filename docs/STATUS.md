# Status

## Phase

P4b closed the open items of P4 and merged into `main` locally on 2026-10-31 as a fast-forward to 78be5c3 (not pushed). P4 (LP and H curves, codim-2 detection and normal forms; SPEC section 14) merged into `main` locally on 2026-10-30 (14d6455, not pushed). P8a (package v0.3; SPEC section 14) was accepted and merged into `main` on 2026-10-30 as a fast-forward to 78c5df6. P7a (C ABI, MEX gateway and the first MATLAB API; SPEC section 14) merged into `main` on 2026-10-29 as PR #2 (3748707). P3 (continuation core, equilibria, LP/H/BP, a, l1, BP switching) merged into `main` on 2026-10-28 as PR #1 (dfcfdab). P2 was accepted on 2026-10-28 and merged into `main`. P1 was accepted on 2026-10-24.

## Done

- P3 merged into `main` on 2026-10-28 as PR #1 (dfcfdab).
- P7a merged into `main` on 2026-10-29 as PR #2 (3748707).

P4 (branch `phase4-codim2`, commits 8a8d207 to 51f2922, Rust only: no MATLAB API change, no C ABI change, no new dependency).

- 8a8d207 adds the `cusp.ode`, `bt.ode` and `bautin.ode` test models. bd44a92 adds `Point.codim2` and the `Problem` hooks `codim2_labels`, `codim2_tests` and `codim2_label`, and `locate.rs` searches one index space for the old and the codim-2 tests.
- 4cc03d6: `continue_lp_curve` with CP detection and the cusp coefficient c. 6995fd5: BT and ZH detection on the LP curve and the BT coefficients (a, b). 51f2922: `continue_hopf_curve` with GH, BT, ZH and HH detection and l2.
- Acceptance (SPEC section 9): Test 7 (cusp, CP at the origin), Test 8 (BT, LP and H curves meet at the origin, s = +1 and s = -1) and Test 9 (Bautin, GH at the origin, l2 = 4s) pass in `crates/pc-engine/tests/codim2.rs`, 10 tests. Two extra checks pass: a warped Bautin model (`bautin_warped.ode`) gives the same l2 to 1e-6, and the Brusselator Hopf curve b = 1 + a^2 matches to 1e-9.
- Checks at 51f2922: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (no failures), `cargo deny check` (advisories, bans, licenses, sources ok) and `cargo build --release --target x86_64-pc-windows-gnu -p pc-ffi` all pass. `buildtool` did not run, since no MATLAB-side file or `pc-ffi` code changed.
- SPEC section 6 now has the curve equations, the Hopf bordered function and its gradient, the codim-2 tests and the normal form formulas.
- Review fixes after the branch read-through: `locate_one` rejects a codim-2 sign change through a pole, a codim-2 zero that its problem drops gives no warning, the warning names the codim-2 test, ZH and HH are NaN for kappa <= 0, and LP and H curve runs detect specials on their first step. Each has a unit test.

P8a (package v0.3, win64 only, branch `phase8a-package`, commits 5efc474 to cbbc82c, plus the status commits; merged into `main` on 2026-10-30). `MinimumMatlabRelease` is R2026b, the version is 0.3.0, and the toolbox UUID is fixed in `buildfile.m`.

- `Evaluator::Aot` in `pc-engine` loads the library with `libloading`, which is now a normal dependency. `model_compile` takes the keys `emit_c`, `evaluator` and `aot_lib`. The C ABI is unchanged: 17 members, 128 bytes. `internal.aotBuild` runs `mex` with floating-point contraction off. On the MATLAB side, `phasecont.config("Codegen")`, `Model(file, Evaluator="aot")` and `Model.AotInfo` select and report AOT.
- `pc-cli licenses` writes `THIRD_PARTY_LICENSES.txt` with one section per crate in the closure. The real run gave 104 sections, 18 of them from `licenses/overrides`. The command fails when a crate has no license file and no override.
- `toolbox/examples` has a FitzHugh-Nagumo example and `ExampleTest`. It finds two Hopf points, at I = 1.4187187 and I = 0.33128134, with omega = 0.27550681. An independent `fzero` on the Jacobian trace confirms both.
- `scripts/packageToolbox.m`, `PackageTest` (31 tests) and the `package` buildtool task build `build/PhaseCont-0.3.0.mltbx`, about 4.87 MB. The script checks the stage path before it deletes anything, and it refuses a repo path that contains `&` or `;`.
- `scripts/cleanInstallCheck.m`, `scripts/buildfileValue.m`, `BuildfileValueTest` and the `package-win` CI job. The check ran in the real Windows MATLAB R2026b Add-Ons folder. It installed 0.3.0, found `phasecont.Model` in the add-on folder, ran the example and uninstalled the toolbox. The run used the same Windows MATLAB install that built the package, not a second machine, and it does not prove that a brand new MATLAB session finds the toolbox on its path. Smart App Control did not block the freshly built `pckern.mexw64`.
- Checks at the tip (cbbc82c): `cargo test` gave 894 passed and 0 failed. `scripts/matlab-win "buildtool cargo mex test package"` gave 150 tests, 149 passed, 0 failed and 1 incomplete (`BuildTest/wslRootElsewhereUsesToolboxBin`, skipped by an assumption, as before).
- Not run yet: the `package-win` job and the licenses step in `pckern-win` have never run on a GitHub runner.

P0 (accepted on 2026-10-22, when the owner approved the decisions in `docs/spikes.md`):

- Discovery results are in `docs/ENVIRONMENT.md`.
- The project license is Apache-2.0. Rule 2 of `CLAUDE.md` says so; `docs/KICKOFF.md` keeps the original wording.
- Scaffold: a Cargo workspace with the nine crates from the `CLAUDE.md` layout, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `deny.toml`, `rustfmt.toml`, `buildfile.m` (only the `test` task works so far) and `tests/ScaffoldTest.m`.
- Windows build route 1 is installed: the `x86_64-pc-windows-gnu` std and MinGW-w64, both from Fedora.
- Four spikes in `spikes/` (a separate Cargo workspace) pass their checks: S1 FFI round trip through the MEX gateway, S2 Cranelift JIT, S3 collocation solve with faer, S4 NVRTC kernel launch. `docs/spikes.md` has the commands, numbers and lessons.
- The four recommendations are approved: Windows build route 1, the Cranelift JIT as default with AOT C as an opt-in, condensation with faer sparse LU, and the 17 changes to `CLAUDE.md` and `docs/SPEC.md`, which are applied. "Decisions after review" in `docs/spikes.md` settles the open questions of P0.

P1 (the numbers below were measured on 2026-10-23 at the head of the branch):

- `crates/pc-ode` has these modules: `span`, `diag`, `lexer`, `ast`, `options`, `builtins`, `parser` (expressions and statements), `include`, `expand` (pseudo-arrays and array blocks), `check` (name resolution and model checks) and `writer`. `crates/pc-ode/docs/language.md` is the language reference.
- `pc-cli parse FILE` parses a model, prints each diagnostic with its source line, then a model summary, and has a `--json` flag.
- 73 diagnostic codes, ODE0001 to ODE0073. Counted with `grep -E "^\| ODE[0-9]{4}" crates/pc-ode/docs/language.md | wc -l`. Each code has an invalid file in `crates/pc-ode/tests/invalid/` (72 files without the `_b` suffix, plus ODE0018, which is built in `invalid.rs`).
- 409 tests pass in `cargo test`: 18 unit tests and 16 + 3 integration tests in `pc-cli`; 307 unit tests, 13 corpus, 5 invalid-file, 9 limit and 38 round-trip tests in `pc-ode`. `PROPTEST_CASES=10000 cargo test -p pc-ode --release round_trip` passes with 11 round-trip tests.
- 21 corpus models in `crates/pc-ode/tests/corpus/`, each with a golden file. All 21 parse with `target/debug/pc-cli parse` without a failure. `crates/pc-ode/tests/README.md` holds the coverage matrix; every test and file it names exists.
- 162 XPPAUT 8.0 probe files in `references/xpp-probes/` at P1 acceptance (164 after the P2 probes `fn-mod-more` and `table-range`, with 328 PASS lines). `references/xpp-probes/run.sh` runs each through XPPAUT and `pc-cli`, gives 324 PASS lines and no FAIL, and ends with "all checks passed". The probes back the grammar rulings in SPEC section 2 and `language.md`.
- The four checks pass: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` and `cargo deny check`. `cargo build --target x86_64-pc-windows-gnu -p pc-cli` and `RUSTDOCFLAGS="-D warnings" cargo doc -p pc-ode -p pc-cli --no-deps` also pass.

P2 (accepted on 2026-10-28; the numbers below were measured that day at `765fc3c`):

- `crates/pc-ir` has these modules: `math` (the shim table over `libm`, `polygamma`, tables), `dag` (hash-consed DAG with folding), `lower`, `entry`, `error`, `diff` (first-order symbolic derivatives), `multi` (multilinear forms up to order 5), `complex` (real and imaginary split of complex directions), `eval` (straight-line evaluator and running error bound), `interp` (the generic interpreter, nested dual numbers and `Fallback`) and `check` (tolerance and finite-difference checks). `crates/pc-jit` has `flags`, `lower`, `module` (the `JitEntries` guard), `compiled` (lazy, length-checked `CompiledModel::call`) and `check`. `crates/pc-codegen` has the C emitter `c`.
- New dependencies, as approved in the plan: `libm` 0.2.16 and `proptest` 1.12.0 (dev) in `pc-ir`, Cranelift 0.137.0 (five crates) in `pc-jit`, `libloading` 0.9.0 (dev) in `pc-codegen`.
- `pc-cli ir FILE`, `pc-cli check-derivs FILE` and `pc-cli emit-c FILE` work.
- 603 tests pass in `cargo test`: 372 in `pc-ode`, 135 in `pc-ir`, 64 in `pc-cli`, 25 in `pc-jit` and 7 in `pc-codegen`. `PROPTEST_CASES=10000 cargo test -p pc-ir --release` passes 135 tests in about 1 s, and `PROPTEST_CASES=1000000` runs `diff_prop.rs` in about 90 s. `diff_prop.rs` now reads `PROPTEST_CASES` (default 300); before that fix the count was fixed at 300, so earlier 10000-case runs on this branch ran only 300 cases.
- `for f in crates/pc-ode/tests/corpus/*.ode; do target/debug/pc-cli check-derivs "$f"; done`: 19 models pass every check and 2 are refused with their `IrError` (`stochastic.ode`: `ir:simulation-only`; `unsupported.ode`: `ir:delay`). None fails.
- `crates/pc-codegen/tests/corpus.rs` runs the SPEC section 3 and 4 checks on 28 models (the 19 accepted corpus models and the 9 models in `crates/pc-ir/tests/models`) at seed 1 and 6 points per entry kind, in about 16 s in a debug build. The JIT matches the DAG evaluator bit for bit everywhere (a NaN matches any NaN), and the AOT C code matches the JIT at 0 ulp on all 28 models. The AOT test compiles with `gcc (GCC) 16.2.1` and `-O2 -fPIC -shared -ffp-contract=off -std=c99 -Wall -Wextra -Werror` (recorded in `references/aot/compiler.txt`).
- Section 3 tolerance. Four DAG-vs-Fallback rows of the Hodgkin-Huxley model failed the first rule (worst ratio 40.7, in `Dir[P,P,Full,P]`). A double-double reference, exact rational models and a permutation check showed cancellation, not a bug: in v' the terms of about 1e3 cancel, so at the failing points the result sits up to 6.8e4 times below the term size and 1e-13 of the output scale falls under the rounding noise. The owner chose a fifth pass condition, |a-b| <= 8 eps E, where E is Wilkinson's running error bound of the DAG evaluation (Higham 2002, section 3.3). SPEC section 3 defines it. On 40000 points per entry kind it gives no failure, with a worst |a-b|/(eps E) of 0.66. `crates/pc-ir/tests/hh_cancellation.rs` keeps the four vectors and checks that a sign flip and relative changes of 1e-8 and 1e-2 still fail.
- `cargo build --target x86_64-pc-windows-gnu -p pc-jit -p pc-cli` and `RUSTDOCFLAGS="-D warnings" cargo doc -p pc-ir -p pc-jit -p pc-codegen -p pc-ode -p pc-cli --no-deps` pass. The four checks pass.
- macOS arm64 (SPEC section 4, measured on 2026-10-28 on GitHub's `macos-latest` runner, macOS 26.6.2, in the private repo through `.github/workflows/macos-jit.yml`):
  - The four checks pass there, and `cargo test` passes all 603 tests (run 40644502330 at `d1b60ec`), the `pc-jit` tests and the AOT test included. The AOT test compiles with Apple clang 21.0.0 (also called as `gcc`) and the same flags, and matches the JIT at 0 ulp.
  - `crates/pc-jit/examples/jit_probe.rs` compiles a Hopf model with the JIT and runs `F` and `JacU`. The results equal `Fallback` bit for bit, unsigned and under the hardened runtime with each JIT entitlement. The runner has SIP disabled and does not enforce the hardened runtime (`DYLD_PRINT_LIBRARIES` still prints with `--options runtime`), so that matrix only shows the JIT runs.
  - MATLAB R2026b on that runner is not hardened: `bin/maca64/MATLAB` and `matlab_helper` have `flags=0x0(none)` and no entitlements. `cranelift-jit`'s default memory provider (read-write, then executable, without `MAP_JIT`) therefore works inside MATLAB, and no `MAP_JIT` provider is needed. A guard step fails the job if a later release adopts the hardened runtime without `allow-unsigned-executable-memory` or `disable-executable-page-protection`.
  - Not tested: the JIT inside the MATLAB process on macOS. That needs a MATLAB batch licensing token in CI (deferred from P7a, see Open questions).
- The review of the whole branch (2026-10-28) found four minor defects and some stale text. One fix round (`8e15f85` to `765fc3c`) fixed them, each code fix with a test that fails without it:
  - The x-derivative of `besselj` and `bessely` now shifts the truncated order: for an order between -1 and 1 it used to give the wrong value.
  - An entry request that hits the node cap now restores the DAG, so later requests on the same model still work.
  - The corpus check holds `F` and `Aux` to 4 ulp between the JIT and `Fallback`, as SPEC section 4 says.
  - `diff_prop.rs` compares `F` between the DAG and `Fallback` and uses the shared tolerance, and runs of 100000 cases or more no longer stop on rejected inputs.
  - The docs now state how folding changes the sign of a zero, and the probe README states the rulings on division, `mod` and `if`.
- Known differences, all documented or by ruling:
  - A literal Bessel order that is NaN, infinite or so large that `trunc(a) - 1` and `trunc(a) + 1` round to the same value gives an x-derivative of 0 in the DAG and the JIT, and NaN in `Fallback`. The two shifted calls become one node and `x - x` folds to 0, as `language.md` documents for folding. The function value is NaN in every evaluator.
  - `polygamma(n, -inf)` is NaN. Every evaluator, the AOT code included, calls the same shim, so they agree.
  - A table formula that reads parameters, states or fixed quantities is tabulated from their default and initial values when the model is lowered (`language.md`, statement details). Changing them needs a new lowering; P3 sessions must do that.
  - At a singular point, where a partial is infinite or NaN and the tangent is 0 at run time, the DAG and the JIT can give NaN while `Fallback` gives a finite value. The interpreter skips a zero tangent at run time and the DAG folds only constant zeros. Both are right at regular points, and the checks use regular points.
  - In JIT, DAG and AOT comparisons a NaN matches any NaN, since IEEE 754 leaves the sign and payload of a NaN result unspecified. DAG constants and table data are still compared bit for bit.
- Parked minor findings, none affecting results: `pc-jit` and `pc-codegen` keep their own copies of `children` and `live_set`; `jit_rows` repeats `compare_kind` across the crate boundary; the running error bound of a table ignores the roundings inside `math::table` (all evaluators share it); user-function re-lowering can grow exponentially with nesting; sanitized C entry names can collide (the compile then fails); the module guard's error-path counter covers one injection point; `math::besselj` and `math::bessely` take time linear in the order (about 0.5 s at order 1e8 and 10 s near 2^31 per call), which P3 bounds by capping the order at 10000 (see P3 below).

P3 (the numbers below were measured on 2026-10-28 at `fb14bdb` plus the SPEC and status commit):

- `crates/pc-linalg` has `dense` (LU solve with a singularity check, transpose solve, determinant), `eig` (eigenvalues of a real matrix as sorted `(re, im)` pairs), `null` (null vectors of A and A^T, kernel of an n x (n+1) matrix) and `block` (the real 2n x 2n form of complex shifted solves). No faer type leaves the crate.
- `crates/pc-engine` has `model` (the `ModelEval` trait for `CompiledModel` and `Fallback`), `error`, `problem`, `ep` (the EP problem and the equilibrium solve), `palc` (tangent, predictor, corrector and step control), `tests_fn` (ψ_LP, ψ_BP, ψ_H and the neutral saddle check), `locate` (secant with a bisection safeguard along the step), `refine` (the LP, BP and Hopf defining systems), `normal` (a and l1), `switch` (the algebraic branching equation), `branch`, `run` (the worker thread with poll, cancel and stop) and `panic` (the hook that `pc-ffi` reuses in P7a).
- `pc-cli cont FILE PROBLEM.json [--fallback]` runs a continuation and prints the branches as JSON, then one line per special point. On `crates/pc-engine/tests/models/lorenz.ode` with `crates/pc-engine/tests/problems/lorenz.json` it prints BP at r = 1 and NSA at r = 4.644444 on the origin branch, and H at r = 24.73684 with l1 = 2.655190e-4 and ω = 9.624530 on the switched branch. The NSA is real: there the eigenvalue 8/3 pairs with -b, which gives r = 1 + 328/90.
- New dependency, as approved in the plan: `faer` 0.24.4 in `pc-linalg`, without default features and without rayon. `cargo deny check` passes with one ignore in `deny.toml`, RUSTSEC-2024-0436, an unmaintained build-time proc-macro under faer with no upgrade.
- `besselj` and `bessely` with an order of magnitude above 10000 give NaN in every evaluator, as `language.md` states. This bounds the time of one evaluation, which a cancel cannot interrupt.
- The fold coefficient a is computed with q oriented so that p^T F_α > 0, so its sign follows the normal form of Kuznetsov's Theorem 3.1; SPEC section 6 states it and the stored normalization string is `|q|=1, p'q=1, p'F_alpha>0`.
- ψ_H is the sign of the plain product times its smallest scaled factor (SPEC section 5), so it does not underflow on models with many slow modes (`a451941`). The zero search along a step uses the plain product divided by its modulus at the step start, so a tiny, nearly constant factor does not make the search stop on the wrong pair (`d7cc8fb`).
- The review of the whole branch (2026-10-28) found eight minor defects. One fix round (`12fd0e1` to `9894938`) fixed them together with the minor items parked during the task reviews. Among them: the bound landing now uses `tol_refine` in both of its checks, options with an infinite step size are refused, a refined point that moved more than twice the step from its secant estimate is not used, a BP whose state Jacobian does not have the expected rank is refused for n = 1 too, switch names ignore case, and `EngineError::Panicked`, which nothing built, is gone.
- 754 tests pass in `cargo test`: 372 in `pc-ode`, 139 in `pc-ir`, 26 in `pc-jit`, 8 in `pc-codegen`, 16 in `pc-linalg`, 113 in `pc-engine` and 80 in `pc-cli`.
- `cargo test -p pc-engine --release --test acceptance -- --nocapture` runs SPEC section 9 tests 1, 2, 5 and 6 with the JIT and with `Fallback`, and both give the same numbers:
  - Test 1: one LP, at x = 0 and α = 1.7e-41, with a = 1 exactly.
  - Test 2: H at α = 0 with ω = 1, and l1 = 2c for c = 1 and c = -1 with an error of 4.4e-16. No NSA.
  - Test 5: BP at r = 1 exactly. On the switched branch x = y exactly and z = x²/b within 7.1e-15, H at r = 470/19 within 3.6e-15, and l1 = 2.655189725836717e-4 > 0.
  - Test 6: H at b = 1 + a² exactly for a = 1 and a = 2, with ω = a within 6.7e-16.
- `cargo build --target x86_64-pc-windows-gnu -p pc-engine` passes. The four checks pass.
- Known limit: on a long step across a branch crossing, the trial corrector of the zero search can jump to the other branch, which puts the BP at the wrong place (α = 0.04 instead of 0 in one case). `hmax` bounds it; the test models use 0.1 or 0.5.

P4b (branch `phase4b-open-items`, cut from 14d6455, merged into `main` on 2026-10-31; closes the open items of P4):

- Hopf detection has no dimension limit. `MAX_HOPF_DIM` and its warning are gone. `hopf_is_detected_above_100_variables` runs a 101-variable model (`hopf101.ode`, a planar Hopf normal form plus 99 stable variables) on both evaluators and finds H at al = 0 to 1e-8 with no warning. Release-mode cost of one `psi_h` call against one dense eigenvalue call on the same size: n = 101, 78 us against 1.4 ms; n = 300, 0.83 ms against 15 ms; n = 1000, 8.3 ms against 0.42 s.
- Codim-2 points are polished by Newton's method along the curve (`refine_zero` in `locate.rs`) and carry `refined = true`. CP (Test 7), BT on both curves (Test 8) and GH (Test 9) now sit within 1e-10 of the exact positions, up from 1e-8. The slope of the test is a forward difference along the tangent, so each step costs one extra test evaluation. A point that does not converge keeps its Illinois position with `refined = false`. Unit tests in `locate.rs` cover a capped search that Newton finishes and a test that is undefined next to the point.
- ZH normal form coefficients (436f39e): b, c, d and e from Kuznetsov (1999) section 8, at ZH points on both the LP curve and the H curve. The sign of q0 makes b > 0, a convention of this project like a > 0 at BT. Tests use our own models with the G values read off by hand, plus a coordinate change by an orthogonal map (the |q0| = 1 normalization allows no other). The warp test asserts b, c, Re d and e.
- HH cubic coefficients (68927d2): P11, P12, P21, P22 and omega2 from section 9, with the pairs ordered w1 > w2. The warp test asserts all four real parts. The fifth-order coefficients S1 and S2 stay unimplemented: the open copy of the paper has misprints in those formulas and no second source confirms them (rule 9).
- LP and H curves in the C ABI and MATLAB (4adcfeb, 61badc2, d236c08). The engine has a `RunKind`, and `pc-ffi` and `pc-cli` take problem `type` `LP` or `H` with two free names. The ABI struct is unchanged (17 members, 128 bytes, `abi_version` 1); the JSON gained `kind`, `codim2_labels`, `coefficient2` and `coefficients`, and curve rows have their own layout (SPEC section 10). MATLAB has `continuation(..., Type="LP"|"H", Free=[a1 a2])`, `Branch.Type`, `Alpha` (N-by-2), `Codim2`, `Codim2Labels` and `Point.Data.values` (SPEC section 11). `kappa` of the Hopf curve is not exposed, and `Stable` is false on a curve.
- Checks at 61badc2: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (939 tests, no failures), `cargo deny check` and `cargo build --release --target x86_64-pc-windows-gnu -p pc-ffi` pass. Commit d236c08 changed no Rust.
- MATLAB suite at d236c08: 165 tests, 159 passed, 5 failed and 4 incomplete. All 5 failures are `AotTest`. Windows Smart App Control blocks freshly built unsigned MEX and DLL files here, so `buildtool cargo mex test` does not run green on this machine. The run used `buildtool cargo`, then the previously staged `pckern.mexw64` gateway (24064 bytes) with the fresh `pckern.dll`, then `buildtool test`. The 4 incomplete are the AOT tests and `BuildTest/wslRootElsewhereUsesToolboxBin` (an assumption). `CurveTest` (11 tests) and the 4 new `Rows2BranchTest` tests pass.

P7a (complete on branch `phase7a-matlab-api`, cut from dfcfdab; merged into `main` as PR #2, 3748707):

- `pc-ffi` C ABI v1: 17 members in 128 bytes, with `model_rhs_t` appended at offset 120. It has the handle registry, the panic barrier, model threads and runs, and a hand-written `pckern.h` with a layout test.
- `pc-cli dll-imports`, the build tasks `cargo` and `mex` with `scripts/build-pckern`, and the MEX gateway in `mex/pckern_gateway.c`.
- MATLAB API: `phasecont.Model`, `Point`, `equilibrium`, `continuation` (blocking, with a live plot), and `Branch` with `plot` and `points`.
- A model that uses `t` compiles and evaluates with `m.rhs(x, p, t)`. `t` has no default and raises `phasecont:model:missingTime`. Runs and equilibria still need an autonomous model.
- Continuation can start from an LP, H or BP point. A BP start raises `badProblem` and names `Switch`.
- Tests at Task 15: Rust 868 passed; MATLAB 105 tests, 104 passed, 0 failed, 1 skipped (a BuildTest assumption).
- CI in `.github/workflows/ci.yml` has four jobs: rust (ubuntu, windows, macos), pckern-win (cross build, dll-imports, artifact), matlab-windows (self-hosted Windows runner `nubom-windows`, prebuilt DLL, no license secret) and asan. None has run on GitHub, since the branch is not pushed.

## In progress

P5 (limit cycles, plan approved on 2026-10-31) on branch `phase5-limit-cycles`, not pushed. Packages 1 to 9 of 10 are done:

- 137aaa1: `cycle/basis.rs`, Gauss-Legendre nodes and weights by Newton on the Legendre polynomials, and the Lagrange basis on m + 1 base points.
- 513f8cc: `pc-linalg` `condense.rs`, the condensation solver for the bordered collocation block system, with the monodromy matrix from the condensed blocks. rayon is a `pc-linalg` dependency and faer's `sparse-linalg` feature is on, both approved in the plan.
- 4363ed1: `cycle/problem.rs`, `CycleProblem` and `periodic_orbit`. `Problem` gains `jacobian_len` so a problem can keep its Jacobian in its own storage. SPEC section 9 Test 3 passes on both evaluators (radius error 1.4e-14, period error 9.1e-14, |mu1 - 1| 3.3e-15, mu2 error 2.5e-15 against exp(-2 pi)), and Test 4 gives observed orders 3.96 to 3.99 (m = 2), 5.89 to 5.97 (m = 3) and 7.92 to 7.98 (m = 4). `cargo test`: 960 passed, 0 failed.

- 93f4280: `cycle/continuation.rs`, `continue_cycle` from a Hopf point or a guessed orbit, with Floquet multipliers per point and LPC, PD and NS detection. `Problem` gains an `accept` hook that moves the phase reference to the last accepted orbit. On the Hopf normal form (N = 40, m = 4, alpha from 1e-4 to 0.5) the radius error is at most 1.4e-13 and |mu1 - 1| 6.8e-15. The Brusselator cycle at b = 2.5 closes under an independent RK4 integration over one period with a gap of 2.4e-10. `cargo test`: 965 passed, 0 failed.

Rulings in package 4:

- PALC metric. The profile unknowns enter PALC multiplied by 1/sqrt(N m), so the arclength approximates the L2 norm of the orbit (Doedel, Keller and Kernévez 1991). T and alpha are not scaled. The factor stays fixed for the whole run, also after mesh adaptation. Points and tangents are returned unscaled. A side effect is that tol_x on profile entries is looser by sqrt(N m); tol_f still applies in full.
- A Hopf start always continues away from the equilibrium, so `direction` < 0 is ignored with a warning.
- The trivial multiplier check warns when a point has no multiplier within 1e-6 of 1. The Hopf normal form cannot trigger it, since its rotation symmetry keeps the multiplier at 1 on any mesh; the test uses the Brusselator on a 3 by 2 mesh, where the trivial multiplier is 1.39 at b = 2.5 and moves to 0.99999 at 20 by 4.

- 3bfd726: `cycle/mesh.rs`, mesh adaptation ported from AUTO-07p (EQDF, NEWMSH and INTERP, see `THIRD_PARTY_NOTICES.md`). The N + 1 mesh points move so that the jumps of the m-th derivative are equidistributed, and the orbit and the tangent are interpolated onto the new mesh. `Options::mesh_adapt` sets the number of accepted points between adaptations (default 3, 0 turns it off), and a failed corrector adapts at most once per point. Every `Point` carries its mesh. On the Brusselator at b = 3.5 with N = 20 and m = 4 the largest mesh-point error against RK4 drops from 3.8e-3 (uniform mesh) to 1.1e-5. `pc-cli licenses` now also copies the `NOTICE*` files of workspace crates, so `crates/pc-engine/NOTICE` reaches `THIRD_PARTY_LICENSES.txt`. `cargo test`: 970 passed, 0 failed; MATLAB 165 tests, 164 passed, 0 failed, 1 filtered by assumption.

Rulings in package 5:

- The plan said mesh adaptation would be written from the papers with no notice entry. SPEC section 7 names AUTO-07p for it, so `mesh.rs` is a port of AUTO's routines with a header comment, an entry in `THIRD_PARTY_NOTICES.md` and the BSD text in `crates/pc-engine/NOTICE`.
- When every m-th derivative is below 1e-7 the mesh stays, as in AUTO. A new mesh that is not strictly increasing or not finite is refused and the old one kept.
- The interpolated orbit is not an exact solution on the new mesh; the next corrector step fixes it, as in AUTO. The PALC scale factor stays as it was.

- e588cdd: `CycleStart::Pd(PdStart)`, the switch to the period-doubled branch. The eigenfunction psi of the multiplier -1 is solved from the condensed system of the PD orbit, the orbit and psi (with -psi on the second half) are doubled onto [0, 1] and interpolated onto the new N and m, and the first orbit is corrected at h0 along that direction, halving it up to 4 times. On Rössler (a = b = 0.2, N = 40, m = 4) the run from an RK4 orbit at c = 2.5 finds one PD at c = 2.8324450, period 5.7698. The switched run (N = 60, m = 4) goes to c = 3 with no warnings, and there its period 11.5310741 matches the sum of two RK4 crossing times (6.0085632 + 5.5225106) to 1e-4, it closes under RK4 over the period with a gap of 6.8e-9, and the gap at half the period is 1.38. `mesh::interpolate` now takes the number of points per interval on the new mesh. `cargo test`: 973 passed, 0 failed.

Rulings in package 6:

- This test stands in for the Rössler oracle of SPEC section 9, as the plan says, until `references/rossler.json` exists.
- The PD start warns when the multiplier closest to -1 is more than 1e-6 away from it and refuses a complex one. Like a Hopf start, it ignores `direction` < 0 with a warning, because both directions give the same orbits half a period apart.
- The new mesh takes its N + 1 points evenly in the index of the doubled mesh, interpolated linearly between its points, so the adapted spacing of the PD orbit carries over to both halves.

- f7528bd: the rayon threshold. `CondenseShape::parallel_pays` holds when the interval elimination work, N nm ni (ni + 2n + k + 1) multiply-adds, is at least `PARALLEL_WORK` = 2e7, and `CycleProblem` then condenses on the current rayon pool. The ignored test `timing_sequential_and_parallel` (`cargo test --release -p pc-linalg timing -- --ignored --nocapture`, 12 logical CPUs, m = 4) measured: on the default pool of 12 threads, parallel was 1.20 times faster at n = 10 and N = 320 and 1.47 times at N = 640, but 1.0 at n = 4 and N = 640, 0.55 at n = 2 and N = 640, and 0.02 at n = 2 and N = 20. Pools of 4 or 6 threads break even earlier (about N = 160 at n = 2), with at most 1.6 times. The parallel and sequential solves give the same bits, so the threshold changes only the time. `cargo test`: 974 passed, 0 failed.

Rulings in package 7:

- The threshold is set for the default pool, which has one thread per logical CPU, since that is what a MATLAB session gets. It is conservative for smaller pools.
- The plan's memory check is the fallible allocation already in place: the blocks, the condenser store and the sparse workspaces use `try_reserve` or `MemBuffer::try_new` and return `OutOfMemory` instead of aborting.

- f93a9bb: limit cycles in the run machinery, the C ABI and the CLI. `RunKind::Cycle` takes a `CycleStart`, and the problem JSON `{"type": "LC", "free", "ntst", "ncol", "start", "mesh"}` builds one. An LC row is `[alpha, u (n (N m + 1)), T, mesh (N + 1), mult_re (n), mult_im (n), LPC, PD, NS, h, newton_iters]`, so its stride is n (N m + 1) + N + 2n + 8, and the status JSON reports `kind` "LC", `ntst`, `ncol` and the labels LPC, PD and NS. The C ABI struct and `abi_version` are unchanged. CLI points and special points gain `eigenvalues` and `mesh`. On Rössler (a = b = 0.2, c = 2.5, free a) the CLI run from the Hopf point at a = 0.0278973 (N = 40, m = 4) finds a PD at a = 0.2261598 with period 5.6796 in 75 points, and a `pd` start built from that special point's CLI output begins the doubled branch at period 11.3592, twice the old one. `cargo test`: 984 passed, 0 failed; MATLAB 165 tests, 164 passed, 0 failed, 1 filtered by assumption.

Rulings in package 8:

- `start` is required for LC and has no default. `hopf` takes the equilibrium as `u0`; `orbit` and `pd` take the orbit at the base points followed by the period, and the CLI gives them no default `u0`. `mesh` is required for `pd` and refused for the other starts. `orbit` assumes a uniform mesh.
- For `pd` the old N comes from `mesh` and the old m from the length of `u0`, so the new `ntst` and `ncol` may differ from the PD orbit's. A wrong `u0` length keeps error code 2.
- `detect` is refused for LC, since LPC, PD and NS are always tested; `mesh_adapt` is refused for the other types.
- A problem whose stride would overflow `usize` is refused when it is parsed ("ntst: ntst times ncol is too large"). `stride` still maps an overflow to `usize::MAX` as a second guard, so the `count * stride` check in `pc_run_points` fails instead of wrapping.

- d0805d1: limit cycles in the MATLAB API. `phasecont.continuation(m, start, Type = "LC", NTST = N, NCOL = m)` starts from a Hopf point (or any equilibrium point or vector, which the engine checks) or from a PD point of an earlier cycle branch. A cycle `Branch` has `Period`, `Mesh`, `NTST` and `NCOL`, keeps the multipliers in `Eigenvalues` and the LPC, PD and NS test values in `Tests`, and `cycle(i)` returns the orbit of point i as times and states over one period. `plot` draws the maximum and minimum of the chosen state, and `Plot = true` draws the maximum while the run goes and ends with the same lines. `tests/CycleTest.m` runs the Brusselator from its Hopf point at b = 2 (first period 2 pi within 1e-3, trivial multiplier within 1e-6 at every point, all points stable) and Rössler (a = b = 0.2, c = 2.5, free a) from its Hopf point to a PD point with a in (0.2, 0.25) and a multiplier within 1e-4 of -1, then the doubled branch from that point with its first period twice the old one within 1e-3. `cargo test`: 984 passed, 0 failed; MATLAB 173 tests, 172 passed, 0 failed, 1 filtered by assumption.

Rulings in package 9:

- `NTST` and `NCOL` are required for LC and have no default. Other types refuse them, and LC refuses `Detect` and `Switch`. `Var` works as for EP.
- A cycle point is a start only when its label is PD. Any other start goes to the engine as a Hopf start, and a state with no complex pair fails the run with `phasecont:engine:badProblem`.
- A cycle point is stable when every multiplier except the one closest to 1 has modulus below 1. A point with a NaN multiplier is not stable.
- The Rössler checks are self-consistent, since no oracle file exists (SPEC section 14).

Next is package 10: docs.

## Next

- P5, P6, P7b and P8 follow P4b.
- P4 and later: a continuation that changes a parameter or state read by a table formula needs a new lowering of the model (P2 known difference).

## Blocked

- Nothing.

## Open questions

- Open from P5: the engine option `mesh_adapt` (default 3) is not exposed in MATLAB, so a MATLAB LC run always adapts the mesh every 3 accepted points. A `MeshAdapt` option would change the public API.
- Open from P4:
  - The BT coefficients use the sign convention a > 0, and the ZH coefficients use b > 0. The normalization strings record both. The owner may prefer another convention.
  - HH fifth-order coefficients S1 and S2 are not computed (see P4b). They need a second source for the formulas.
- Owner: File Exchange route for v0.3, checked on 2026-10-29. Direct upload accepts BSD only. Apache-2.0 is allowed only for a GitHub-linked entry. The R2026b help says a package with MEX or DLL files cannot be submitted to File Exchange, and it does not say whether a GitHub-linked entry is exempt. The repo is private, which blocks the GitHub link route. Ask MathWorks before relying on either route.
- P8 release build: `pckern.dll` embeds host paths, about 377 references to `/home/nub04/.cargo/registry` and 8 to `target/`. Path remapping belongs in the release build.
- P8 or later: the gateway never unloads `pckern.dll`. Uninstalling or upgrading the toolbox in a MATLAB session that has used PhaseCont fails with "Permission denied" and leaves a stale folder under `%APPDATA%\MathWorks\MATLAB Add-Ons` until MATLAB exits. Add an unload or shutdown function.
- P8: AOT on Linux and macOS. The mex link may not export `pc_aot_init` on glnxa64 or maca64. The win64 build exports it through `PC_EXPORT`. Check with a visibility attribute and a Linux test. The AOT test in `pc-codegen` still runs gcc only.
- Settled on 2026-10-28: the in-MATLAB part of the macOS arm64 check in SPEC section 4 moved to P7a. P7a did not run it; it is deferred (see below).
- The P1 questions are settled for P2: division follows IEEE; `mod` is `fmod(x,y)` plus `y` when the remainder is negative, which probe `fn-mod-more` confirmed on eight more cases; `if` evaluates both branches and selects one.
- Open from P7a: `model_from_ode` parses one in-memory string with no file loader, so `#include` lines fail with a diagnostic. The MATLAB layer has to inline included files, or a later task adds a loader callback.
- Carried into the open questions from the P1, P2 and P7a notes:
  - Closed in P7a by c2ad18f: `pc-ffi` parses on an 8 MiB worker thread and rejects input over 16 MiB. Each Model handle owns an 8 MiB thread that lowers the model and runs either evaluator, so `JitEntries` never moves between threads and needs no `unsafe impl`. The diagnostics JSON keeps the first 1000 diagnostics.
  - The JIT inside MATLAB on Windows:
    - `ModelTest` evaluates a model whose JIT frame passes 4 KiB inside MATLAB (it asserts `jit_max_frame > 4096`), so the inline stack probes run on Windows.
    - `cranelift-jit` registers no Windows unwind information. No panic or exception may unwind through JIT code, which rule 7's `catch_unwind` on the Rust side of every call already requires.
  - The JIT inside MATLAB on macOS arm64 (moved from P2 by the owner on 2026-10-28). The Rust side of the SPEC section 4 check passes on the macOS runner already (see "Done").
- Deferred from P7a, with reasons:
  - (a) The MATLAB jobs on Linux and macOS, and the in-MATLAB macOS JIT check. MathWorks' batch token pilot was closed to new requests (checked 2026-10-29), and the repo is private. These resume when a token or a public repo exists.
  - (b) AOT through `mex`, moved to P8a on 2026-10-28.
  - (c) ASan and the self-hosted MATLAB job have never run on GitHub. The P8a `package-win` job and the licenses step in `pckern-win` have not run on a real runner either.
  - The self-hosted runner is registered but not a Windows service. It runs only while `C:\actions-runner\run.cmd` is running.
- Known minors from reviews, parked:
  - `special_tangent` threshold flips LP orientation with refinement noise.
  - The BP rank threshold needs a comment.
  - The LP test does not assert alpha at or below the LP alpha.
  - SPEC does not say that LP Direction 1 leaves toward increasing first state component.
  - Variable lookup is duplicated in `Branch.m` and `continuation.m`.
  - The special-marker edge is black on dark themes.
  - The live line advances ColorOrderIndex, so the final plot uses the second color.
  - From the final P7a review (2026-10-29):
    - `Stable` is false for a point whose eigenvalues are NaN. The engine fills NaN only after an internal fault, and carrying the last value forward would be a silent default, so it stays.
    - `run_copy` with `count == 0` does not reach the library, so a freed run id returns an empty matrix, and the gateway trusts the caller's `stride` instead of reading the run's.
    - `dll-imports` accepts any `api-ms-win-` prefix and reads the whole file without a size limit.
    - `scripts/build-pckern` and `buildfile.m` ignore `CARGO_TARGET_DIR`, and the `wsl.exe` command does not escape `$` or quotes in paths.
    - A `Branch` keeps its `Model` handle, so variable lookup fails after the Model is deleted, and `continuation` rejects a start Point from another Model object with the same source.
    - `BuildTest/gatewayIsBuiltIntoTheBinFolder` cannot fail when the gateway is missing.
  - AOT cache: a cache hit trusts `isfile` alone, with no staged rename, no rebuild after a failed load and no flags in the key, so a partial library warns until the cache folder is deleted.
  - AOT hash: `textHash` is a per-byte MATLAB loop on every Model construction; read the hash from the `c_file` name when no hook is set.
  - AOT cache folder: the path is predictable in a shared `tempdir` with no ownership check (a Linux `/tmp` concern), and AotTest uses the real cache.
  - AOT Model: the class help does not mention `Evaluator="aot"`, `AotInfo` or the config key, `Evaluator=""` is accepted, and a failed AOT attempt compiles the JIT a second time.
  - Licenses: `--overrides` on a regular file says "not found", the help text calls SOURCE an "upstream URL and tag", and the exit-1 list leaves out SOURCE, UTF-8 and `--overrides`.
  - Licenses: the `licenses.rs` doc lists `LICENSE*` only, a read failure names the folder and not the file, and two "names the crate" tests also match the fixture path.
  - Licenses: overrides are keyed by crate name only, so a `Cargo.lock` bump can reuse an old text for a new version.
  - Example: the comment says every setting is given, but H0, Hmin, Hmax and the tolerances use engine defaults, and the output leaves out `Branch.Stop` and `Warnings`.
  - Example: the test checks only that the printed output is non-empty, does not assert distinct Hopf points or table columns, and `traceAt` never checks Newton convergence.
  - Package: `PackageTest` reads metadata with regexes that depend on attribute spelling, the `toolbox/bin` skip is untested, and the stage root's `matlabPath` entry is an empty string.
  - Package: `toolbox/examples` with no packageable file fails after the stage is deleted, a stage on a missing drive gives a raw MKDIR error, and files in dot-folders under `toolbox/` are shipped.
  - Clean install: the `notAddon` negative path is untested, `settings()` has a silent catch fallback, and `buildfileValue` has no negative tests for single-quoted or concatenated literals.
  - Clean install: the `.mltbx` is uploaded only after the check passes, and a failure in the licenses step stops the `matlab-windows` job.
  - AOT load: `AotEval::open` checks only that the symbols exist. The emitted C has no dimension or `pc_math_t` layout export, so a hand-picked `aot_lib` built for another model passes `open` and reads out of bounds. The MATLAB flow cannot reach this, since the cache folder is named from a hash of the C text. A fix needs an exported tag in `pc-codegen`.
