# Status

## Phase

P0: discovery, scaffold and spikes S1 to S4 (`docs/KICKOFF.md`). The plan was approved on 2026-10-21. P0 was accepted on 2026-10-22, when the owner approved the decisions in `docs/spikes.md` (SPEC section 14).

## Done

- Discovery results are in `docs/ENVIRONMENT.md`, updated after the Phase 0 installs.
- The project license is Apache-2.0 instead of BSD-3-Clause. Rule 2 of `CLAUDE.md` now says so; `docs/KICKOFF.md` keeps the original wording.
- Scaffold: a Cargo workspace with the nine crates from the `CLAUDE.md` layout, all empty; `LICENSE`, `THIRD_PARTY_NOTICES.md`, `deny.toml`, `rustfmt.toml`; `buildfile.m` with `cargo`, `mex`, `test` and `package` tasks, of which only `test` works so far; `tests/ScaffoldTest.m`.
- Windows build route 1 is installed: the `x86_64-pc-windows-gnu` std and MinGW-w64, both from Fedora, so one `rustc` builds for WSL and for Windows.
- Spike code in `spikes/`, a separate Cargo workspace. In WSL, `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` are clean, `cargo test` passes 62 tests, one of which runs on the GPU, and `cargo deny check licenses` passes.
  - S1: the FFI round trip from MATLAB through the MEX gateway into `pckern.dll`, with the panic barrier, the handle registry and the `phasecont:engine:<kind>` error IDs.
  - S2: Cranelift JIT of the FitzHugh-Nagumo right-hand side and Jacobian, timed against plain Rust (called through a function pointer and inlined), C and MATLAB anonymous functions.
  - S3: the collocation matrix from SPEC section 7, solved with faer sparse LU on the full matrix and by condensation, and compared with MATLAB's sparse solvers.
  - S4: a saxpy kernel compiled by NVRTC to PTX and to a cubin and launched through the CUDA driver API, once with MATLAB's NVRTC and once with the toolkit's.
  - The last run of each spike passes its checks inside MATLAB, and S2 to S4 also pass in WSL. The first S4 run inside MATLAB failed on its CPU reference; `docs/spikes.md` gives the cause and the fix. Raw output is under `spikes/results/`.
  - The 29 files in `spikes/matlab` give 29 `checkcode` messages, all of them the CFERR message about a missing settings file, and no `codeIssues` finding.
  - Each spike, the MATLAB scripts, the whole tree and the S4 fix were checked for defects, and the fixes that held up are in the code. Minor items in the throwaway spike code stay open; the lessons later phases need are in `docs/spikes.md`.
- `docs/spikes.md`: for each spike, what was built, the commands, the numbers and what went wrong, then the four recommendations that `docs/KICKOFF.md` asks for.
- The four recommendations are approved: Windows build route 1, the Cranelift JIT as the default backend with AOT C as an opt-in speed-up, condensation with faer sparse LU as the first linear solver, and all 17 changes to `CLAUDE.md` and `docs/SPEC.md`, which are applied.
- The open questions of P0 are decided; "Decisions after review" in `docs/spikes.md` gives each decision and its reason.

## In progress

- Nothing.

## Next

- P1 (`pc-ode` parser, diagnostics, writer, `pc-cli parse`), started in plan mode.
- P2: the macOS arm64 JIT check in SPEC section 4, on a Mac with MATLAB or on the macOS runner of the MATLAB CI job.
- P8a: confirm on the File Exchange submission form that it accepts an Apache-2.0 toolbox linked to GitHub.

## Blocked

- Nothing.

## Open questions

- Nothing.
