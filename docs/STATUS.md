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
- 162 XPPAUT 8.0 probe files in `references/xpp-probes/` at P1 acceptance (163 after the P2 probe `fn-mod-more`, with 326 PASS lines). `references/xpp-probes/run.sh` runs each through XPPAUT and `pc-cli`, gives 324 PASS lines and no FAIL, and ends with "all checks passed". The probes back the grammar rulings in SPEC section 2 and `language.md`.
- The four checks pass: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` and `cargo deny check`. `cargo build --target x86_64-pc-windows-gnu -p pc-cli` and `RUSTDOCFLAGS="-D warnings" cargo doc -p pc-ode -p pc-cli --no-deps` also pass.

## In progress

- P2 (`pc-ir`, derivatives, `pc-jit`, AOT C emitter; SPEC section 14) on branch `phase2-pc-ir`. The plan was approved on 2026-10-25. It settles the three open questions of P1 and the derivative method; SPEC sections 2 to 4 and `language.md` record the rulings.

## Next

- P7a: the `pc-ffi` worker thread. Parsing at depth 200 needs about 800 KB of stack in a debug build, so `pc-ffi` must parse on a worker thread with an explicit stack size (for example 8 MB), not on MATLAB's thread (parked P1). It must also cap the input size, since parse diagnostics grow with the input (about 253 MB for 300000 bad lines in a debug build).
- The macOS arm64 JIT check in SPEC section 4, on a Mac with MATLAB or on the macOS runner of the MATLAB CI job. No Mac, GitHub repo or CI exists yet, so the P2 plan proposes carrying this check past P2 acceptance; P2 builds and tests the `f64` fallback evaluator on every platform.
- P8a: confirm on the File Exchange submission form that it accepts an Apache-2.0 toolbox linked to GitHub.

## Blocked

- Nothing.

## Open questions

- None. The P1 questions are settled for P2: division follows IEEE; `mod` is `fmod(x,y)` plus `y` when the remainder is negative, which probe `fn-mod-more` confirmed on eight more cases; `if` evaluates both branches and selects one.
