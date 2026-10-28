# Status

## Phase

P2 (`pc-ir`, derivatives, `pc-jit` and the AOT C emitter; SPEC section 14) is in progress. P1 was accepted on 2026-10-24 and merged into `main`.

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

P2 (all 15 plan tasks are in; the numbers below were measured on 2026-10-28 at `c34499a`; acceptance is pending, see "In progress"):

- `crates/pc-ir` has these modules: `math` (the shim table over `libm`, `polygamma`, tables), `dag` (hash-consed DAG with folding), `lower`, `entry`, `error`, `diff` (first-order symbolic derivatives), `multi` (multilinear forms up to order 5), `complex` (real and imaginary split of complex directions), `eval` (straight-line evaluator and running error bound), `interp` (the generic interpreter, nested dual numbers and `Fallback`) and `check` (tolerance and finite-difference checks). `crates/pc-jit` has `flags`, `lower`, `module` (the `JitEntries` guard), `compiled` (lazy, length-checked `CompiledModel::call`) and `check`. `crates/pc-codegen` has the C emitter `c`.
- New dependencies, as approved in the plan: `libm` 0.2.16 and `proptest` 1.12.0 (dev) in `pc-ir`, Cranelift 0.137.0 (five crates) in `pc-jit`, `libloading` 0.9.0 (dev) in `pc-codegen`.
- `pc-cli ir FILE`, `pc-cli check-derivs FILE` and `pc-cli emit-c FILE` work.
- 590 tests pass in `cargo test`: 372 in `pc-ode`, 124 in `pc-ir`, 64 in `pc-cli`, 23 in `pc-jit` and 7 in `pc-codegen`. `PROPTEST_CASES=10000 cargo test -p pc-ir --release` passes 124 tests in 4.8 s. `diff_prop.rs` now reads `PROPTEST_CASES` (default 300); before that fix the count was fixed at 300, so earlier 10000-case runs on this branch ran only 300 cases.
- `for f in crates/pc-ode/tests/corpus/*.ode; do target/debug/pc-cli check-derivs "$f"; done`: 19 models pass every check and 2 are refused with their `IrError` (`stochastic.ode`: `ir:simulation-only`; `unsupported.ode`: `ir:delay`). None fails.
- `crates/pc-codegen/tests/corpus.rs` runs the SPEC section 3 and 4 checks on 28 models (the 19 accepted corpus models and the 9 models in `crates/pc-ir/tests/models`) at seed 1 and 6 points per entry kind, in about 16 s in a debug build. The JIT matches the DAG evaluator bit for bit everywhere (a NaN matches any NaN), and the AOT C code matches the JIT at 0 ulp on all 28 models. The AOT test compiles with `gcc (GCC) 16.2.1` and `-O2 -fPIC -shared -ffp-contract=off -std=c99 -Wall -Wextra -Werror` (recorded in `references/aot/compiler.txt`).
- Section 3 tolerance. Four DAG-vs-Fallback rows of the Hodgkin-Huxley model failed the first rule (worst ratio 40.7, in `Dir[P,P,Full,P]`). A double-double reference, exact rational models and a permutation check showed cancellation, not a bug: in v' the terms of about 1e3 cancel, so at the failing points the result sits up to 6.8e4 times below the term size and 1e-13 of the output scale falls under the rounding noise. The owner chose a fifth pass condition, |a-b| <= 8 eps E, where E is Wilkinson's running error bound of the DAG evaluation (Higham 2002, section 3.3). SPEC section 3 defines it. On 40000 points per entry kind it gives no failure, with a worst |a-b|/(eps E) of 0.66. `crates/pc-ir/tests/hh_cancellation.rs` keeps the four vectors and checks that a sign flip and relative changes of 1e-8 and 1e-2 still fail.
- `cargo build --target x86_64-pc-windows-gnu -p pc-jit -p pc-cli` and `RUSTDOCFLAGS="-D warnings" cargo doc -p pc-ir -p pc-jit -p pc-codegen -p pc-ode -p pc-cli --no-deps` pass. The four checks pass.
- Known differences, all documented or by ruling:
  - `polygamma(n, -inf)` is NaN. Every evaluator, the AOT code included, calls the same shim, so they agree.
  - A table formula that reads parameters, states or fixed quantities is tabulated from their default and initial values when the model is lowered (`language.md`, statement details). Changing them needs a new lowering; P3 sessions must do that.
  - At a singular point, where a partial is infinite or NaN and the tangent is 0 at run time, the DAG and the JIT can give NaN while `Fallback` gives a finite value. The interpreter skips a zero tangent at run time and the DAG folds only constant zeros. Both are right at regular points, and the checks use regular points.
  - In JIT, DAG and AOT comparisons a NaN matches any NaN, since IEEE 754 leaves the sign and payload of a NaN result unspecified. DAG constants and table data are still compared bit for bit.
- Parked minor findings, none affecting results: `pc-jit` and `pc-codegen` keep their own copies of `children` and `live_set`; `jit_rows` repeats `compare_kind` across the crate boundary; the running error bound of a table ignores the roundings inside `math::table` (all evaluators share it); user-function re-lowering can grow exponentially with nesting; sanitized C entry names can collide (the compile then fails); the module guard's error-path counter covers one injection point.

## In progress

- P2 (`pc-ir`, derivatives, `pc-jit`, AOT C emitter; SPEC section 14) on branch `phase2-pc-ir`. The plan was approved on 2026-10-25. It settles the three open questions of P1 and the derivative method; SPEC sections 2 to 4 and `language.md` record the rulings. All plan tasks are done and the section 3 and 4 checks pass (see "Done"). Left: the review of the whole branch, and the owner's decision on carrying the macOS check past P2.

## Next

- P7a: the `pc-ffi` worker thread. Parsing at depth 200 needs about 800 KB of stack in a debug build, so `pc-ffi` must parse on a worker thread with an explicit stack size (for example 8 MB), not on MATLAB's thread (parked P1). It must also cap the input size, since parse diagnostics grow with the input (about 253 MB for 300000 bad lines in a debug build).
- P7a: the JIT and AOT inside MATLAB on Windows, which P2 could not test here.
  - Run a model whose frame passes 4 KiB inside MATLAB, so the inline stack probes run on Windows.
  - `cranelift-jit` registers no Windows unwind information. No panic or exception may unwind through JIT code, which rule 7's `catch_unwind` on the Rust side of every call already requires.
  - `JitEntries` is neither `Send` nor `Sync`. Moving a compiled model to the worker thread needs a justified `unsafe impl`.
  - Run `Fallback` on the 8 MB worker thread too: its recursion budget is sized for a 1 MiB stack at five nested dual levels.
  - AOT through `mex` with MSVC: compile with floating-point contraction off (the counterpart of `-ffp-contract=off`), call `pc_aot_init` with pckern's shim table before any entry, and compare NaNs by class, not by sign. The project test runs gcc only.
- The macOS arm64 JIT check in SPEC section 4, on a Mac with MATLAB or on the macOS runner of the MATLAB CI job. No Mac, GitHub repo or CI exists yet, so the P2 plan proposes carrying this check past P2 acceptance; P2 builds and tests the `f64` fallback evaluator on every platform.
- P8a: confirm on the File Exchange submission form that it accepts an Apache-2.0 toolbox linked to GitHub.

## Blocked

- Nothing.

## Open questions

- For the owner: may the macOS arm64 check in SPEC section 4 stay open past P2 acceptance (see "Next")?
- The P1 questions are settled for P2: division follows IEEE; `mod` is `fmod(x,y)` plus `y` when the remainder is negative, which probe `fn-mod-more` confirmed on eight more cases; `if` evaluates both branches and selects one.
