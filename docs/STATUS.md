# Status

## Phase

P3 (continuation core, equilibria, LP/H/BP, a, l1, BP switching; SPEC section 14) is implemented on branch `phase3-pc-engine` and waits for the review of the whole branch and for acceptance; its plan was approved on 2026-10-28. P2 was accepted on 2026-10-28 and merged into `main`. P1 was accepted on 2026-10-24.

## Done

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
  - Not tested: the JIT inside the MATLAB process on macOS. That needs the MEX gateway and `pc-ffi` (P7a) and a MATLAB batch licensing token in CI.
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
- 738 tests pass in `cargo test`: 372 in `pc-ode`, 139 in `pc-ir`, 26 in `pc-jit`, 8 in `pc-codegen`, 16 in `pc-linalg`, 98 in `pc-engine` and 79 in `pc-cli`.
- `cargo test -p pc-engine --release --test acceptance -- --nocapture` runs SPEC section 9 tests 1, 2, 5 and 6 with the JIT and with `Fallback`, and both give the same numbers:
  - Test 1: one LP, at x = 0 and α = 1.7e-41, with a = 1 exactly.
  - Test 2: H at α = 0 with ω = 1, and l1 = 2c for c = 1 and c = -1 with an error of 4.4e-16. No NSA.
  - Test 5: BP at r = 1 exactly. On the switched branch x = y exactly and z = x²/b within 7.1e-15, H at r = 470/19 within 3.6e-15, and l1 = 2.655189725836717e-4 > 0.
  - Test 6: H at b = 1 + a² exactly for a = 1 and a = 2, with ω = a within 6.7e-16.
- `cargo build --target x86_64-pc-windows-gnu -p pc-engine` passes. The four checks pass.
- Known limit: on a long step across a branch crossing, the trial corrector of the zero search can jump to the other branch, which puts the BP at the wrong place (α = 0.04 instead of 0 in one case). `hmax` bounds it; the test models use 0.1 or 0.5.

## In progress

- P3: all 14 plan tasks are done. The review of the whole branch, its fix round and the owner's acceptance remain.

## Next

- P4: Hopf detection for n > 100 with the minimally augmented test function (Govaerts), and the LP and Hopf curves. Each step now evaluates `JacU` and all eigenvalues even when Hopf detection is off; P4 should reuse F_u from the step's Jacobian.
- P4 and later: a continuation that changes a parameter or state read by a table formula needs a new lowering of the model (P2 known difference).
- P7a: the `pc-ffi` worker thread. Parsing at depth 200 needs about 800 KB of stack in a debug build, so `pc-ffi` must parse on a worker thread with an explicit stack size (for example 8 MB), not on MATLAB's thread (parked P1). It must also cap the input size, since parse diagnostics grow with the input (about 253 MB for 300000 bad lines in a debug build).
- P7a: the JIT and AOT inside MATLAB on Windows, which P2 could not test here.
  - Run a model whose frame passes 4 KiB inside MATLAB, so the inline stack probes run on Windows.
  - `cranelift-jit` registers no Windows unwind information. No panic or exception may unwind through JIT code, which rule 7's `catch_unwind` on the Rust side of every call already requires.
  - `JitEntries` is neither `Send` nor `Sync`. Moving a compiled model to the worker thread needs a justified `unsafe impl`.
  - Run `Fallback` on the 8 MB worker thread too: its recursion budget is sized for a 1 MiB stack at five nested dual levels.
  - Lower models on that thread as well. `lower`'s depth limit of 1000 is not a stack bound: about 400 nested user functions overflow a 1 MiB stack in a debug build (a release build at 1 MiB and a debug build at 2 MiB return `TooLarge`).
  - AOT through `mex` with MSVC: compile with floating-point contraction off (the counterpart of `-ffp-contract=off`), call `pc_aot_init` with pckern's shim table before any entry, and compare NaNs by class, not by sign. The project test runs gcc only.
- P7a: the JIT inside MATLAB on macOS arm64 (moved here from P2 by the owner on 2026-10-28), on the macOS runner once the owner adds a MATLAB batch licensing token as a repository secret. The Rust side of the SPEC section 4 check passes there already (see "Done").
- P8a: confirm on the File Exchange submission form that it accepts an Apache-2.0 toolbox linked to GitHub.

## Blocked

- Nothing.

## Open questions

- Settled on 2026-10-28: the in-MATLAB part of the macOS arm64 check in SPEC section 4 moves to P7a, where the MEX gateway first exists (see "Next").
- The P1 questions are settled for P2: division follows IEEE; `mod` is `fmod(x,y)` plus `y` when the remainder is negative, which probe `fn-mod-more` confirmed on eight more cases; `if` evaluates both branches and selects one.
