# Phase 0 spike report

Phase 0 asked four feasibility questions and answered each with a throwaway spike. Each section below has the same four parts: what was built, the commands that ran, the numbers measured and what went wrong. The report ends with one recommendation for each of four decisions. Phase 0 is accepted when the project owner approves those decisions (`docs/SPEC.md` section 14).

| Spike | Question | Outcome |
|---|---|---|
| S1, FFI round trip | Can a Rust `cdylib` and a C MEX gateway talk to Windows MATLAB from this WSL setup? | Yes. All 39 checks of `s1_ffi.m` passed, including four panics that arrived in MATLAB as `phasecont:engine:panic` errors. |
| S2, Cranelift JIT | Can Cranelift compile model code inside the MATLAB process, and how fast is that code compared with plain Rust, C and MATLAB? | Yes on win64. For the FitzHugh-Nagumo model in MATLAB, the median ratios of JIT loop time to the loop time of the same function in plain Rust, called through a pointer, ran from 0.937 to 1.151, and the median ratios of JIT loop time to C loop time from 0.976 to 1.124. Each loop also builds its input and folds its output. The loop that calls the right-hand side written as a MATLAB anonymous function took 46 to 59 times as long per iteration as the loop that calls the JIT version. macOS arm64 was not tested. |
| S3, collocation linear algebra | Which solve is fastest for one collocation Newton step? | `condensed_sparse` (condense each interval, then a sparse LU of the reduced system) was fastest at 1 thread for N = 40, 100 and 400, in WSL and in MATLAB. Thread scaling is open. faer's automatic choice of LU variant picked the slower supernodal variant for 2 of 5 probed matrices, so recommendation 3 chooses the variant in the code. |
| S4, CUDA through NVRTC | Can `cudarc` compile a kernel with NVRTC and run it from inside MATLAB? | Yes, after a fix to the CPU reference. PTX for `compute_120` and a cubin for `sm_120` passed with NVRTC 13.1 (MATLAB's copy) and 13.4 (the toolkit's copy). |

The spike code is in `spikes/`, a Cargo workspace of its own that the root `Cargo.toml` excludes. `spikes/results/` holds the raw output of the `spikecli` runs and MATLAB scripts reported here, plus `s1_crash_dumps.txt`, a PowerShell listing of MATLAB crash dumps. Output of `cargo build`, `spikes/stage.sh` and `cargo test` was not saved there. Results files named without a folder are in `spikes/results/`, and `<registry>/` stands for `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`, where Cargo unpacked the crates that this report cites. `docs/ENVIRONMENT.md` describes the machine and its toolchains and is not copied here. A measured number in this report is copied from a file in `spikes/results/` as printed there. A ratio is computed from such numbers. A number about third-party code or about the machine cites its source.

The spike commits add S1, S2 and S3 one at a time, but the workspace compiles only from the S4 commit on. `spikes/pckern/src/lib.rs` declares all four spike modules (`s1`, `s2_jit`, `s3_linalg` and `s4_cuda`), its function table points to functions in `s1`, `s2_jit` and `s3_linalg`, and `run_json` dispatches to `s2_jit::run`, `s3_linalg::run` and `s4_cuda::run`. `spikes/pckern/Cargo.toml` already lists the dependencies of all four spikes (Cranelift, `cudarc`, faer and rayon).

Rows labeled MATLAB ran inside Windows MATLAB R2026b through `scripts/matlab-win`, each run in a new `-batch` process, with `pckern.dll` built for `x86_64-pc-windows-gnu`. Rows labeled WSL ran natively through the `spikecli` binary, built for `x86_64-unknown-linux-gnu`. Several results files hold more than one entry, each with its own `date:` line, and the sections say which entry a number comes from. Each WSL entry records its `rustc:` line. Each MATLAB capture that loaded the DLL records the build stamp that the `info` function returns (`built=`, a Unix time in seconds).

The S1 to S3 MATLAB captures come from build `1792599836` (2026-10-21T16:23:56Z). The S4 captures differ. `s4_matlab_mingw_fmaf.txt`, the first S4 capture in MATLAB, used `1792599836`, and `s4_matlab.txt` and `s4_matlab_preload.txt`, made after the `fma_ref` fix, used `1792610350` (2026-10-21T19:19:10Z). Two captures loaded no DLL and carry no stamp: `s1_gateway_nodll.txt` and `s2_matlab_baseline.txt`.

Each header records the Windows power overlay and the AC state, and every `power:` line reports AC power with the battery at 95 to 100 percent. Two runs used "Best power efficiency": `s1_ffi.txt` and `s2_matlab_power_efficiency.txt`. Every other file with a `power:` line used "Best performance", and `s1_crash_dumps.txt` has no such line. S1 times nothing, so its overlay does not matter. The S2 timing run under "Best power efficiency" is labeled where it appears and stays out of the S2 comparison.

## S1, FFI round trip

### What was built

`spikes/pckern/` is a Rust crate that builds as `pckern.dll` (crate type `cdylib`) for `x86_64-pc-windows-gnu`. It exports one symbol, `pc_api_v1`, which returns a table of function pointers that starts with `abi_version` and `struct_size` (`spikes/pckern/src/lib.rs`). The release profile sets `panic = "unwind"`, because the barrier depends on `catch_unwind`.

`spikes/pckern/src/ffi.rs` holds the panic barrier. Every table function except `last_error` and `free_string` runs its body under `catch_unwind` and keeps any error message in a thread-local that `last_error` returns; after a caught panic it returns code 1, or NULL if it returns a pointer. Those two call `catch_unwind` directly and record no panic message. While a barrier runs on the panicking thread, a panic hook records the message with its file and line.

`spikes/pckern/src/s1.rs` has the test functions. `scale` writes `a * x` into a buffer that MATLAB allocated, `panic` panics on the calling thread, `panic_in_worker` panics at item 37 of 64 in a local rayon pool with two threads, and `test_handle` and `free_handle` use the registry in `spikes/pckern/src/registry.rs`, which hands out ids `TAG | (counter << 1) | 1` with the tag `0x5043` in the high 16 bits. `spikes/pckern/build.rs` stamps the build time into the string that `info` returns, so each capture records which DLL it loaded. Native tests in `spikes/pckern/tests/s1_barrier.rs` and `spikes/pckern/tests/s1_registry.rs` cover the same paths on Linux, but no results file records a run of them.

`spikes/mex/pckern_gateway.c` is the MEX gateway, built by the MSVC compiler that Windows MATLAB found. The first command that needs the table finds the gateway's own file with `GetModuleHandleExW` and `GetModuleFileNameW`, loads `pckern.dll` from that folder with `LoadLibraryExW` and `LOAD_WITH_ALTERED_SEARCH_PATH`, calls `pc_api_v1`, and checks `abi_version` and `struct_size`. The gateway checks argument types before it calls Rust.

`spikes/stage.sh` runs `cargo build --release --target x86_64-pc-windows-gnu` in `spikes/`, stops if the DLL imports `libgcc`, `libwinpthread` or `libstdc`, copies the DLL and the gateway sources to `%USERPROFILE%\phasecont-stage\spikes` and runs `mex -R2018a` there. With `--mex-only` it copies the C sources and runs `mex` only. `spikes/matlab/s1_gateway_nodll.m` builds the gateway once from the repo's `\\wsl.localhost\` folder, then copies the staged gateway into a temporary folder without `pckern.dll` and checks argument errors, the missing-DLL error and the `module_path` and `preload` commands. `spikes/matlab/s1_ffi.m` checks `scale` on nine input shapes with a string and a char command, raises each panic twice with a `scale` call after each, and checks handles and argument errors. A `scale` result must equal `2*x` exactly, NaN and signed zeros included, with the size of `x` and class `double`.

### Commands

```
spikes/stage.sh --mex-only
scripts/matlab-win --run spikes/matlab/s1_gateway_nodll.m
spikes/stage.sh
powershell.exe -NoProfile -Command 'Get-ChildItem -Path $env:TEMP -Filter "matlab_crash_dump.*"'
scripts/matlab-win --run spikes/matlab/s1_ffi.m
powershell.exe -NoProfile -Command 'Get-ChildItem -Path $env:TEMP -Filter "matlab_crash_dump.*"'
```

`s1_gateway_nodll.m` wrote `spikes/results/s1_gateway_nodll.txt`, `s1_ffi.m` wrote `spikes/results/s1_ffi.txt`, the two listings went to `spikes/results/s1_crash_dumps.txt`, and `spikes/stage.sh` writes no results file.

### Numbers

Both captures ran inside Windows MATLAB.

| Run | UTC start | Power overlay | DLL build | PASS lines | Last lines |
|---|---|---|---|---|---|
| `s1_gateway_nodll.m` | 2026-10-21T14:26:04Z | Best performance | none loaded | 15 | `all passed`, `exit code: 0` |
| `s1_ffi.m` | 2026-10-21T16:28:34Z | Best power efficiency | `built=1792599836` (2026-10-21T16:23:56Z) | 39 | `all passed`, `exit code: 0` |

S1 times nothing, so the power overlay does not affect either result. Two of the 15 PASS lines of the no-DLL run check less than they seem: the UNC `mex` check cannot fail the script, and the missing-string check accepts any message.

| Code | Error ID | Detected by | Seen in the captures |
|---|---|---|---|
| 1 | `phasecont:engine:panic` | Rust barrier | `panic` and `panic_in_worker`, twice each |
| 2 | `phasecont:engine:badArg` | Rust function | not seen; native tests only |
| 3 | `phasecont:engine:badHandle` | Rust registry | second `free_handle` of a handle, `free_handle` of id 0 |
| 4 | `phasecont:engine:internal` | Rust function | not tested |
| other | `phasecont:engine:unknown` | gateway | not tested |
| none | `phasecont:engine:badArg` | gateway argument checks | 7 command checks, a double handle, 6 `scale` argument checks |
| none | `phasecont:engine:load` | gateway, missing DLL | three calls without `pckern.dll` and `preload` of a missing file, all `Windows error 126` |
| none | `phasecont:engine:load` | gateway, missing export, NULL table or ABI mismatch | not tested |

The messages after the ID were `panic: spike panic: 100% on purpose (at pckern/src/s1.rs:33)` for both `panic` calls and `panic: spike worker panic at item 37 (raised on another thread)` for both `panic_in_worker` calls. The "%" arrived intact, since the gateway passes every message through a `"%s"` format. `scale` returned the right result after each of the four panics. `test_handle` returned `5783466346473848835`, which is `0x5043000000000003`, the tag with counter 1.

| Crash dump listing | UTC time | `count=` |
|---|---|---|
| before `s1_ffi.m` | 2026-10-21T16:28:24Z | 3 |
| after `s1_ffi.m` | 2026-10-21T16:29:23Z | 3 |

Both listings name the same three files with the same sizes and write times, so the run left no crash dump. These four panics are the only evidence that unwinding in an `x86_64-pc-windows-gnu` DLL works inside MATLAB, since the native tests run on Linux.

### What went wrong

Nothing failed in the captures; the points below are for later phases.

- Worker panics lose their location. No barrier runs on a rayon worker, so the hook hands a worker panic to Rust's default hook, whose stderr output does not appear in the capture. The calling barrier gets only the payload, so the MATLAB message has no file and line. The engine should keep a worker panic's location for the calling barrier, and every thread it starts must catch its own panics and pass the message back.
- Handle ids and doubles. With consecutive tagged ids, an id converted to a double and back could equal another live id, because every double in [2^62, 2^63) is a multiple of 1024. The registry now hands out odd ids, which no value rounded through a double can equal while the counter stays below 2^47; `s1_registry.rs` checks 2048 ids. The gateway also rejects a double where a handle belongs. `pc-ffi` should keep both.
- Pointer results. After a caught panic `pc_api_v1` returns NULL, and the gateway raises `phasecont:engine:load` without the panic text, since `last_error` is reached through the table. A NULL from `info` raises `phasecont:engine:panic` with the `last_error` text. Neither path was tested.
- Runtime DLLs. `pckern.dll` imports `msvcrt.dll` and no MinGW runtime DLL, and the gateway imports `VCRUNTIME140.dll` and the UCRT (`api-ms-win-crt-*`), so the DLL and the gateway use separate C runtimes. These lists come from the current build `1792610350` and the gateway staged with it; the S1 build is known only to have passed the `spikes/stage.sh` check. That deny-list would pass a new runtime DLL, so Phase 7a should allow Windows system DLLs only.
- The DLL stays loaded. The gateway never calls `FreeLibrary`, because rayon threads may still run DLL code, so a desktop MATLAB would keep the file locked until it exits and `spikes/stage.sh` could not replace it. Every capture used `scripts/matlab-win`, which starts a new `-batch` process, so the lock was not tested. The Phase 7a build tasks must handle it.
- Smart App Control is on. It blocked a test executable that `cargo test` built for `x86_64-pc-windows-gnu` (CodeIntegrity events 3033 and 3077) and has not blocked `pckern.dll` or `pckern.mexw64`. Only `docs/ENVIRONMENT.md` records the block. Whether it accepts an unsigned release DLL on a user's machine was not tested.
- The Windows std and `rustc` must come from the same Fedora build, and no version lock is set, so an update of `rustc` alone would break Windows builds. `docs/ENVIRONMENT.md` gives the `dnf versionlock` command that would set one.
- `mex` works with the UNC repo folder as the current folder and its output on `C:`, as `buildfile.m` assumes.
- Reading the code turned up gaps that no capture checks. The Phase 7a gateway should check the number of outputs, reject missing strings, NUL characters inside text and relative `preload` paths with `phasecont:engine:badArg`, and use `FormatMessageW` for Windows errors. `pc-ffi` should save the panic record when a barrier starts and restore it at the end, since a barrier run by a `Drop` during an unwind now discards the outer panic's location.

## S2, Cranelift JIT

### What was built

`spikes/pckern/src/s2_jit.rs` compiles the FitzHugh-Nagumo right-hand side and Jacobian to native code with Cranelift 0.137.0 inside `pckern`. Both use the signature `extern "C" fn(x, p, out)` that `docs/SPEC.md` plans for `pc-jit`, with the Jacobian column major. The IR uses separate `fmul`, `fdiv`, `fsub` and `fadd` instructions in the order of the Rust reference functions and no fused multiply-add, so JIT and Rust results should agree bit for bit. A third function calls `pc_shim_exp`, a Rust `extern "C"` shim around `f64::exp`, to test the shim route planned for transcendentals.

The suite in the same file records the compile time of the module, times the JIT right-hand side and Jacobian against the same Rust code called through a function pointer and inlined, reports the largest ulp distance from the Rust results over random inputs, and checks that all variants give the same checksum.

A C reference, `spikes/mex/fhn_aot.c`, stands in for the optional AOT path. In the native WSL build, `spikes/pckern/build.rs` compiles it with gcc and `-O2 -fPIC -ffp-contract=off`, so it rounds like the JIT. In MATLAB, `spikes/stage.sh` compiles it with MSVC through `mex`. The gateway command `bench_c` in `spikes/mex/pckern_gateway.c` passes its two function pointers to `bench_extern` in the DLL, which adds C variants and jit/c ratios.

`spikes/matlab/s2_jit.m` first times the same functions written as MATLAB anonymous functions with `spikes/matlab/s2_baseline.m`. It then loads `pckern.dll` through the gateway, calls `run_json` for the first JIT compile in the process and `bench_c` for a second compile with the C variants, and asserts that all checksums match and every ulp distance is 0. `spikes/pckern/src/bin/spikecli.rs` runs the same suite natively in WSL.

Every timed iteration also builds its input from the loop index and folds the output into a running value: a bit-level checksum in Rust (`spikes/pckern/src/s2_jit.rs:270-323`) and the sum of the first output in MATLAB (`spikes/matlab/s2_baseline.m:34-43`). Each per-call time and ratio in this section therefore covers the whole loop and not the function alone. No run timed a loop around an empty function, so the share of each time that belongs to the function is not known. In the `bench_c` blocks of `spikes/results/s2_matlab.txt` the medians for the inlined Rust Jacobian were 2.10, 1.97 and 2.22 ns per iteration, against 3.39, 3.32 and 3.32 ns for the JIT Jacobian and 3.07, 3.11 and 3.16 ns for the C Jacobian. When the shared work adds the same time to both variants of a ratio, the ratio of the functions alone lies farther from 1 than the printed ratio.

### Commands

```
(cd spikes && cargo build --release)
spikes/target/release/spikecli s2
scripts/matlab-win --run spikes/matlab/s2_jit.m
spikes/stage.sh
scripts/matlab-win --run spikes/matlab/s2_jit.m
```

The `spikecli` runs wrote the 4 entries of `spikes/results/s2_wsl.txt`; entry 1 came from an earlier build. The first `matlab-win` run found no staged DLL and wrote `spikes/results/s2_matlab_baseline.txt`; after `spikes/stage.sh`, the run under "Best power efficiency" wrote `spikes/results/s2_matlab_power_efficiency.txt` and the next three runs wrote `spikes/results/s2_matlab.txt`.

### Numbers

Runs inside MATLAB load the Windows `pckern.dll` (target `x86_64-pc-windows-gnu`) under MATLAB 26.2.0.3386108 (R2026b) with `maxNumCompThreads` 16. Each Rust timing is the median of 5 repetitions of 10000000 calls, after one untimed warm-up run of the same variant, and each `max_ulp_vs_rust` value comes from 100000 random inputs (`calls`, `reps` and `samples` in the JSON). The MATLAB anonymous-function baseline times 100000 calls per repetition, after one warm-up run, over 5 repetitions. Anonymous-function cells give the median per call, the range and the median scaled to 1e7 calls, as printed. The last column is the printed anonymous/JIT factor for `run_json`, then `bench_c`.

| Run (`date:`) | File, entry | `built=` | `compile_ms`, first JIT compile in this process | `compile_ms`, second compile, warm | MATLAB anonymous rhs | MATLAB anonymous jac | anonymous rhs / jit rhs |
|---|---|---|---|---|---|---|---|
| 2026-10-21T14:26:29Z | `s2_matlab_baseline.txt`, only entry | no DLL loaded | not run | not run | 138.5 ns (118.8 to 156.7 ns), about 1.39 s | 162.3 ns (160.9 to 182.2 ns), about 1.62 s | not run |
| 2026-10-21T16:44:45Z | `s2_matlab.txt`, entry 1 of 3 | `1792599836` | 28.082 ms | 0.665 ms | 208.4 ns (193.5 to 221.7 ns), about 2.08 s | 318.4 ns (291.4 to 331.5 ns), about 3.18 s | 59 and 56 |
| 2026-10-21T16:45:08Z | `s2_matlab.txt`, entry 2 of 3 | `1792599836` | 1.386 ms | 0.555 ms | 177.0 ns (165.5 to 179.7 ns), about 1.77 s | 310.4 ns (297.3 to 322.8 ns), about 3.10 s | 46 and 48 |
| 2026-10-21T16:45:28Z | `s2_matlab.txt`, entry 3 of 3 | `1792599836` | 1.477 ms | 1.252 ms | 202.0 ns (192.9 to 215.7 ns), about 2.02 s | 353.7 ns (321.7 to 382.9 ns), about 3.54 s | 55 and 57 |

| Run under "Best power efficiency", not compared (`date:`) | File, entry | `built=` | `compile_ms`, first JIT compile in this process | `compile_ms`, second compile, warm | MATLAB anonymous rhs | MATLAB anonymous jac | anonymous rhs / jit rhs |
|---|---|---|---|---|---|---|---|
| 2026-10-21T16:29:49Z | `s2_matlab_power_efficiency.txt`, only entry | `1792599836` | 49.270 ms | 0.749 ms | 451.1 ns (268.4 to 486.2 ns), about 4.51 s | 617.4 ns (513.5 to 802.6 ns), about 6.17 s | 74 and 72 |

Each MATLAB run is a new process. The results do not show why the first compile in entry 1 took 28.082 ms. Each `compile_ms` value is one sample that covers flag setup, ISA detection, three function compiles and finalization (`spikes/pckern/src/s2_jit.rs:67-135`). The anonymous-function medians move between processes (right-hand side 138.5 to 208.4 ns under "Best performance"), so each anonymous/JIT factor holds only inside its own run.

| Entry of `s2_matlab.txt` | Block | Ratio | Median (min to max) | Per repetition |
|---|---|---|---|---|
| 1 (16:44:45Z) | `run_json` | rhs jit/pointer | 0.998 (0.925 to 1.106) | 1.106 0.925 1.008 0.998 0.930 |
| 1 (16:44:45Z) | `run_json` | jac jit/pointer | 1.070 (1.017 to 1.273) | 1.017 1.273 1.070 1.023 1.222 |
| 1 (16:44:45Z) | `bench_c` | rhs jit/pointer | 0.990 (0.880 to 1.167) | 0.995 0.880 1.167 0.990 0.965 |
| 1 (16:44:45Z) | `bench_c` | rhs jit/c | 1.044 (0.885 to 1.102) | 1.048 0.885 1.044 1.102 0.981 |
| 1 (16:44:45Z) | `bench_c` | jac jit/pointer | 1.151 (1.040 to 1.227) | 1.167 1.151 1.227 1.040 1.115 |
| 1 (16:44:45Z) | `bench_c` | jac jit/c | 1.124 (0.922 to 1.160) | 1.160 0.932 0.922 1.124 1.132 |
| 2 (16:45:08Z) | `run_json` | rhs jit/pointer | 0.978 (0.867 to 1.088) | 0.978 0.867 0.992 1.088 0.916 |
| 2 (16:45:08Z) | `run_json` | jac jit/pointer | 0.979 (0.953 to 1.290) | 0.979 0.968 1.290 1.255 0.953 |
| 2 (16:45:08Z) | `bench_c` | rhs jit/pointer | 1.011 (0.939 to 1.036) | 1.036 1.011 1.029 0.955 0.939 |
| 2 (16:45:08Z) | `bench_c` | rhs jit/c | 0.984 (0.893 to 1.078) | 0.984 1.020 0.976 1.078 0.893 |
| 2 (16:45:08Z) | `bench_c` | jac jit/pointer | 1.085 (0.954 to 1.189) | 1.025 1.088 1.085 1.189 0.954 |
| 2 (16:45:08Z) | `bench_c` | jac jit/c | 1.032 (0.907 to 1.118) | 1.118 1.032 1.068 1.007 0.907 |
| 3 (16:45:28Z) | `run_json` | rhs jit/pointer | 0.937 (0.894 to 1.077) | 0.937 0.894 1.077 0.975 0.932 |
| 3 (16:45:28Z) | `run_json` | jac jit/pointer | 1.119 (1.086 to 1.184) | 1.086 1.161 1.119 1.118 1.184 |
| 3 (16:45:28Z) | `bench_c` | rhs jit/pointer | 1.036 (0.859 to 1.080) | 1.033 1.036 1.080 1.062 0.859 |
| 3 (16:45:28Z) | `bench_c` | rhs jit/c | 0.976 (0.567 to 1.095) | 0.976 0.968 0.996 1.095 0.567 |
| 3 (16:45:28Z) | `bench_c` | jac jit/pointer | 1.018 (0.853 to 1.346) | 0.982 0.853 1.346 1.321 1.018 |
| 3 (16:45:28Z) | `bench_c` | jac jit/c | 1.078 (0.924 to 1.186) | 1.078 0.930 1.113 1.186 0.924 |

In every block of the four runs that loaded the DLL, both checksum pairs match and every `max_ulp_vs_rust` value, C included, is `0e0`. In the table above, medians lie between 0.937 and 1.151 and single repetitions between 0.567 and 1.346. The right-hand side ratios show no consistent winner. For the Jacobian, the jit/c median was above 1 in each of the three runs (1.124, 1.032 and 1.078) and the JIT was slower than the MSVC-compiled C code in 10 of the 15 repetitions. Five of the six Jacobian jit/pointer medians are above 1.

Medians in ns per call from the same runs, as printed in `s2_matlab.txt`, with (min to max) after each median. The ratio medians above are medians of per-repetition ratios, so they differ from the ratios of the medians below. The `run_json` blocks have no C variant.

| Entry of `s2_matlab.txt` | Block | Function | JIT | Rust via pointer | Rust inlined | C |
|---|---|---|---|---|---|---|
| 1 (16:44:45Z) | `run_json` | rhs | 3.53 (3.29 to 4.16) | 3.69 (3.54 to 3.78) | 3.47 (3.36 to 3.68) | no C variant |
| 1 (16:44:45Z) | `run_json` | jac | 3.15 (2.69 to 3.34) | 2.73 (2.48 to 3.02) | 1.93 (1.68 to 2.26) | no C variant |
| 1 (16:44:45Z) | `bench_c` | rhs | 3.72 (3.33 to 3.74) | 3.73 (3.20 to 3.85) | 3.56 (3.41 to 3.79) | 3.58 (3.20 to 3.79) |
| 1 (16:44:45Z) | `bench_c` | jac | 3.39 (3.03 to 3.55) | 3.03 (2.47 to 3.07) | 2.10 (1.62 to 2.16) | 3.07 (2.80 to 3.64) |
| 2 (16:45:08Z) | `run_json` | rhs | 3.83 (3.62 to 4.09) | 3.94 (3.76 to 4.17) | 3.78 (3.37 to 3.89) | no C variant |
| 2 (16:45:08Z) | `run_json` | jac | 3.31 (3.15 to 3.65) | 3.31 (2.46 to 3.42) | 2.18 (2.03 to 2.32) | no C variant |
| 2 (16:45:08Z) | `bench_c` | rhs | 3.70 (3.58 to 4.22) | 3.81 (3.60 to 4.08) | 3.77 (3.11 to 3.94) | 4.00 (3.43 to 4.29) |
| 2 (16:45:08Z) | `bench_c` | jac | 3.32 (3.06 to 3.57) | 3.06 (2.82 to 3.42) | 1.97 (1.77 to 2.59) | 3.11 (2.97 to 3.60) |
| 3 (16:45:28Z) | `run_json` | rhs | 3.65 (3.51 to 3.90) | 3.81 (3.37 to 4.17) | 3.45 (3.22 to 3.91) | no C variant |
| 3 (16:45:28Z) | `run_json` | jac | 3.28 (3.20 to 3.37) | 2.93 (2.82 to 2.96) | 2.16 (1.52 to 2.37) | no C variant |
| 3 (16:45:28Z) | `bench_c` | rhs | 3.57 (2.38 to 4.07) | 3.36 (2.77 to 3.93) | 3.49 (2.32 to 4.22) | 3.69 (3.27 to 4.19) |
| 3 (16:45:28Z) | `bench_c` | jac | 3.32 (2.94 to 3.94) | 3.08 (2.61 to 3.45) | 2.22 (1.91 to 2.40) | 3.16 (3.08 to 3.39) |

Native WSL runs use `spikecli` (target `x86_64-unknown-linux-gnu`) on the host `Linux 6.18.54.1-microsoft-standard-WSL2, 12 vCPUs, Intel(R) Core(TM) Ultra 7 255H`, under "Best performance". `s2_wsl.txt` has 4 entries. The table gives each ratio of entries 2 to 4 as median (min to max), as printed.

| Entry (`date:`) | `rustc:` line | `compile_ms` | `code_bytes` | rhs jit/pointer | rhs jit/c | jac jit/pointer | jac jit/c |
|---|---|---|---|---|---|---|---|
| 1 (2026-10-21T05:10:00Z) | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 4.28328e-1 | 262 | not printed | not printed | not printed | not printed |
| 2 (2026-10-21T14:21:05Z) | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 4.77756e-1 | 262 | 1.0923920196115438e0 (1.0775655287951647e0 to 1.1138838232716983e0) | 1.1103094175932655e0 (1.063549563550996e0 to 1.1604706197595607e0) | 1.0035124839699336e0 (8.265123376539016e-1 to 1.034278147895085e0) | 1.0310260114630359e0 (9.751417709997888e-1 to 1.1260335817973675e0) |
| 3 (2026-10-21T14:21:07Z) | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 7.919729999999999e-1 | 262 | 1.1088946682151266e0 (9.607086496757813e-1 to 1.2321362047388669e0) | 1.1210299380132351e0 (9.946989659798345e-1 to 1.2096312614409652e0) | 1.0086853029577922e0 (8.748266697718218e-1 to 1.0800402198230048e0) | 1.078071595514323e0 (8.740795621523578e-1 to 1.0893249745194273e0) |
| 4 (2026-10-21T14:21:08Z) | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 4.81738e-1 | 262 | 1.0271996226619e0 (8.331304554140888e-1 to 1.4229648493235862e0) | 1.0964660830862596e0 (9.633824059263075e-1 to 1.3552715332421894e0) | 1.036007998659603e0 (9.815203795331008e-1 to 1.1697217579957533e0) | 1.0670299358891553e0 (9.652900830741462e-1 to 1.163296114693503e0) |

Entry 1 (2026-10-21T05:10:00Z) is left out of the comparison because it predates the interleaved repetitions and per-repetition ratios. In entries 2 to 4 every ratio median is above 1, and the jit/c medians run from 1.0310260114630359e0 to 1.1210299380132351e0, so the gcc-compiled C code was faster than the JIT at the median of each entry and in 25 of the 30 repetitions. All checksums match and every `max_ulp_vs_rust` value is `0e0`. The results do not show why `code_bytes` is 262 here and 279 in the DLL. WSL and MATLAB timings differ in operating system, Rust target and C compiler, so they are not compared.

Medians of entries 2 to 4 of `s2_wsl.txt` in ns per call, as printed in `median_ns`, with `min_ns` to `max_ns` in parentheses.

| Entry (`date:`) | Function | JIT | Rust via pointer | Rust inlined | C |
|---|---|---|---|---|---|
| 2 (2026-10-21T14:21:05Z) | rhs | 2.1513121e0 (2.0789767e0 to 2.2329713e0) | 1.9714219e0 (1.9024774e0 to 2.0300477e0) | 1.6793111e0 (1.6364953000000002e0 to 1.7595459e0) | 1.9375789e0 (1.9161233e0 to 1.9921909e0) |
| 2 (2026-10-21T14:21:05Z) | jac | 1.8334144e0 (1.8002739e0 to 2.0261417e0) | 1.8458215e0 (1.7726512e0 to 2.1860396e0) | 9.196852e-1 (9.093657e-1 to 9.345417e-1) | 1.7993617e0 (1.7272448e0 to 1.8528472e0) |
| 3 (2026-10-21T14:21:07Z) | rhs | 2.213642e0 (2.0766389e0 to 2.415424e0) | 2.0621781e0 (1.9603547e0 to 2.2539261e0) | 1.6351742999999999e0 (1.5826179e0 to 1.9352808e0) | 1.9968267e0 (1.9601292e0 to 2.0877059e0) |
| 3 (2026-10-21T14:21:07Z) | jac | 1.8106082e0 (1.7730745e0 to 2.0571876e0) | 1.9047324e0 (1.7702664e0 to 2.0696765e0) | 9.284578e-1 (9.227209e-1 to 1.0635814e0) | 1.8056149e0 (1.6794879e0 to 2.0285047e0) |
| 4 (2026-10-21T14:21:08Z) | rhs | 2.1314418e0 (2.0701985e0 to 2.6124889e0) | 2.0798324e0 (1.8359476e0 to 2.558353e0) | 1.6581426e0 (1.6143962000000003e0 to 2.0094606e0) | 1.9439195e0 (1.8744345e0 to 2.298092e0) |
| 4 (2026-10-21T14:21:08Z) | jac | 1.9050717e0 (1.7809229e0 to 2.1082376e0) | 1.8023411e0 (1.7477015e0 to 1.8388581e0) | 9.012926e-1 (8.629753e-1 to 9.586378e-1) | 1.7226852e0 (1.6690468e0 to 1.9735743e0) |

In all 9 blocks (6 in MATLAB, 3 in WSL), the median of the inlined Rust Jacobian is below the median of the JIT Jacobian.

### What went wrong

The spike ran only on Windows and WSL, so the macOS arm64 check that `docs/SPEC.md` assigns to S2 is still open. Reading `cranelift-jit` 0.137.0 shows two problems for a Mac build. `JITModule::new` accepts `is_pic=true` only on x86_64 (`<registry>/cranelift-jit-0.137.0/src/backend.rs:370-374`), but `Jit::new` sets it on every target (`spikes/pckern/src/s2_jit.rs:71`), so the spike code would fail on aarch64 before compiling. Phase 2 should set the flag only on x86_64, as `JITBuilder::with_flags` does (`<registry>/cranelift-jit-0.137.0/src/backend.rs:75-80`). The default `SystemMemoryProvider` maps code as read-write pages without `MAP_JIT` and switches them to read-execute with `mprotect`. Apple's documentation allows this under the hardened runtime only with the `com.apple.security.cs.allow-unsigned-executable-memory` entitlement or one that includes it. With only `com.apple.security.cs.allow-jit`, `pc-jit` needs its own `JITMemoryProvider` that maps pages with `MAP_JIT`, and with neither entitlement only the AOT fallback is left. Item 4.4 of the recommendations gives the sources and the `codesign` check; none of this was tried on a Mac.

Dropping a `JITModule` does not free its code. `SystemMemoryProvider` forgets its allocations in the `Drop` of its `Memory` regions (`<registry>/cranelift-jit-0.137.0/src/memory/system.rs:146-152`), and only `free_memory` unmaps them (`<registry>/cranelift-jit-0.137.0/src/backend.rs:198-208`). The spike's `Jit` calls `free_memory` in its own `Drop` (`spikes/pckern/src/s2_jit.rs:45-52`). Still open: an error after `JITModule::new` and before the module moves into `Jit` drops it without `free_memory`, which leaks every page that is already full or finalized. The Phase 2 model handle must call `free_memory` on every path, and only once no thread can still call its functions.

Imported symbols resolve through the builder's own table, then through the process: `dlsym(RTLD_DEFAULT)` on unix, and on Windows the running executable and `ucrtbase.dll` (`<registry>/cranelift-jit-0.137.0/src/backend.rs:662-704`). Inside MATLAB a shim exported only from `pckern.dll` would not resolve, and finalization would panic. The spike registers `pc_shim_exp` with `JITBuilder::symbol` (`spikes/pckern/src/s2_jit.rs:83`), and Phase 2 must register every shim this way. Shims must also never panic. A panic that leaves an `extern "C"` function such as `pc_shim_exp` aborts the process, MATLAB included, and JIT frames have no unwind information either (`cranelift-jit` registers none with the system unwinder), while the panic barrier sits outside the JIT call.

Cranelift leaves stack probes off by default, and its default outline strategy calls `__cranelift_probestack`, which nothing provides. Windows grows the stack through a guard page, so frames over 4096 bytes need probes, and the spike sets `enable_probestack=true` with `probestack_strategy=inline` (`spikes/pckern/src/s2_jit.rs:73-76`). The test functions have small frames, so this path was not tested; code for large models must keep both flags.

The first WSL entry timed the variants in a fixed order without per-repetition ratios, so drift during a run could favor one variant. The harness now interleaves repetitions in a rotated order and records a ratio per repetition (`spikes/pckern/src/s2_jit.rs:337-392`). It also folds checksums differently (`spikes/pckern/src/s2_jit.rs:270-283`), which is why entry 1 has other checksums.

Larger models and MATLAB on glnxa64 were not tested. In MATLAB, `c_compiler` is "none" in `run_json`, which has no C variants, and only "passed in by the caller" in `bench_c`. The `mex` call in `spikes/stage.sh` adds only `/W4` to the defaults, so the AOT path should record the compiler and its flags with each result, as the WSL JSON does.

## S3, collocation linear algebra

### What was built

`spikes/pckern/src/s3_linalg/matrix.rs` builds the bordered system of one collocation Newton step for a periodic orbit, the solve the limit-cycle phase repeats, with n components, m Gauss-Legendre points and N intervals. Each Jacobian block is minus the identity plus a scaled uniform random matrix, drawn per interval and Gauss point where the Phase 0 plan had one per interval (`matrix.rs:159-163`, `spikes/NOTES.md:36`). The `s3` runs use n = m = 4.

`spikes/pckern/src/s3_linalg/run.rs` solves the system three ways:

- `sparse_lu`: faer's sparse LU of the full matrix, always sequential.
- `condensed_sparse` (`condense.rs`): a dense LU of each interval's interior columns eliminates the interior unknowns, faer's sparse LU solves the reduced system, and back-substitution recovers the interiors.
- `condensed_dense`: the same with a dense LU of the reduced system, run up to a fixed size limit (`run.rs:19`).

Each timing row holds 20 repetitions after a warm-up. For the two sparse methods, "first" includes the symbolic analysis and "repeat" reuses it. The condensed methods also time their condense, reduced and backsub steps. More than 1 thread runs in a local rayon pool built outside the timed region (`run.rs:207-228`).

`run` (`run.rs:286-312`) sizes the pool to the system's thread count unless `threads=` is given. A run passes when every backward error `norm(A*x-b,inf) / (norm(A,inf)*norm(x,inf) + norm(b,inf))` is at most 1e-13, every pairwise difference `norm(x_i-x_j) / norm(x_0)` (2-norm, `x_0` from `sparse_lu`) is at most 1e-10, and `condensed_sparse` gives the same bits at 1 thread and in the pool.

`spikes/matlab/s3_linalg.m` repeats this in MATLAB with the Windows `pckern.dll`. It copies the Rust matrix into MATLAB (`s3_build`, `s3_copy`), calls each method through `s3_solve`, recomputes the backward errors, times `decomposition(A)` and `A\b` with `timeit`, compares all 7 solutions and runs the Rust suite through `run_json("s3")`.

`spikes/pckern/src/s3_linalg/lu_probe.rs` (`spikecli s3lu`) times faer's sparse LU under AUTO, its automatic choice between the simplicial and supernodal variants, and under each variant forced, all with `Par::Seq`.

### Commands

```
(cd spikes && cargo build --release --bin spikecli)
spikes/target/release/spikecli s3 N=40
spikes/target/release/spikecli s3 N=100
spikes/target/release/spikecli s3 N=400
spikes/target/release/spikecli s3 N=400 threads=6
spikes/target/release/spikecli s3lu n=4 m=4 N=40
spikes/target/release/spikecli s3lu n=16 m=4 N=100 budget_ms=10000
spikes/target/release/spikecli s3lu n=32 m=2 N=157 on=reduced budget_ms=10000
spikes/stage.sh
scripts/matlab-win --run spikes/matlab/s3_linalg.m
```

The `s3` runs wrote entries 1 to 4 of `spikes/results/s3_wsl.txt`, the `s3lu` runs entries 1 to 3 of `spikes/results/s3lu_wsl.txt` and the MATLAB run the one capture in `spikes/results/s3_matlab.txt`; the two build commands wrote none.

### Numbers

Every run used the "Best performance" power overlay. The WSL runs use `spikecli` (target `x86_64-unknown-linux-gnu`).

WSL `s3` entries, with `cond1_est`, a 1-norm condition estimate:

| Entry, `date:` | `rustc:` | N | threads | nr | nnz | nred | `cond1_est` | largest backward error | pairwise difference |
|---|---|---|---|---|---|---|---|---|---|
| 1, 2026-10-21T14:21:09Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 40 | 12 | 646 | 15378 | 166 | 1.488473678137417e5 | 1.7447853724986725e-17 | 1.0158159759701378e-14 |
| 2, 2026-10-21T14:21:11Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 100 | 12 | 1606 | 38418 | 406 | 2.383512891660256e5 | 6.487327944933953e-18 | 1.4705540050589163e-14 |
| 3, 2026-10-21T14:21:17Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 400 | 12 | 6406 | 153618 | 1606 | 7.972520936075523e6 | 3.0430044600240353e-18 | 2.672448648623745e-14 |
| 4, 2026-10-21T14:21:21Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | 400 | 6 | 6406 | 153618 | 1606 | 7.972520936075523e6 | 3.2441948375462848e-18 | 2.672448648623745e-14 |

All 4 entries pass, bit check included. The largest backward error in each comes from `condensed_dense`.

WSL "repeat" medians in ns, with (min to max):

| Method, threads | Entry 1, N = 40 | Entry 2, N = 100 | Entry 3, N = 400 | Entry 4, N = 400 |
|---|---|---|---|---|
| `sparse_lu`, 1 | 3.5723e5 (3.51073e5 to 3.86304e5) | 6.36196e5 (5.91e5 to 9.78603e5) | 2.663346e6 (2.574631e6 to 3.431032e6) | 2.6446105e6 (2.504354e6 to 4.957648e6) |
| `condensed_sparse`, 1 | 8.9348e4 (8.7822e4 to 1.90487e5) | 2.34177e5 (2.28474e5 to 3.3018e5) | 9.425525e5 (9.15492e5 to 1.297153e6) | 9.528575e5 (9.2879e5 to 1.082397e6) |
| `condensed_sparse`, 12 (6 in entry 4) | 6.89374e5 (3.89375e5 to 1.254522e6) | 7.375285e5 (3.46477e5 to 1.372879e6) | 1.520801e6 (1.105977e6 to 1.998801e6) | 6.56455e5 (5.71073e5 to 8.20977e5) |
| `condensed_dense`, 1 | 1.63234e5 (1.60073e5 to 1.82125e5) | 1.120905e6 (1.101479e6 to 1.183757e6) | 5.59271955e7 (5.0736469e7 to 6.8335044e7) | 5.3445419e7 (5.1251306e7 to 6.1761409e7) |
| `condensed_dense`, 12 (6 in entry 4) | 2.288198e6 (7.1929e5 to 4.308303e6) | 5.0897555e6 (3.211038e6 to 2.0446755e7) | 4.43986635e7 (3.2347243e7 to 1.00669171e8) | 2.2579603e7 (1.8410172e7 to 3.6995508e7) |

`condensed_sparse` at 1 thread is fastest in entries 1 to 3, where `sparse_lu` takes 3.9982, 2.7167 and 2.8257 times as long and the 12-thread pool raises the `condensed_sparse` median. With 6 threads in entry 4 it is 1.4515 times faster than at 1 thread, and the spreads do not overlap.

`condensed_sparse` "repeat" steps at N = 400, in ns:

| Step | Entry 3, 1 thread | Entry 3, 12 threads | Entry 4, 1 thread | Entry 4, 6 threads |
|---|---|---|---|---|
| condense | 5.431775e5 (5.21654e5 to 7.19668e5) | 5.38402e5 (1.22789e5 to 9.81009e5) | 5.487655e5 (5.32011e5 to 5.83057e5) | 1.69338e5 (1.36089e5 to 3.09555e5) |
| reduced | 3.13007e5 (3.02813e5 to 4.00752e5) | 3.814815e5 (3.43611e5 to 5.01175e5) | 3.134045e5 (2.99902e5 to 4.48883e5) | 3.69854e5 (3.21175e5 to 4.45711e5) |
| backsub | 7.2375e4 (7.0275e4 to 2.80778e5) | 6.42656e5 (8.8803e4 to 8.05738e5) | 7.4113e4 (6.9956e4 to 8.6878e4) | 6.59225e4 (5.5828e4 to 9.9462e4) |

With 6 threads the condense step is 3.2407 times faster, and the sequential reduced step has a higher median. With 12 the condense median barely moves, and the reduced and backsub medians rise.

`sparse_lu` builds its CSC matrix once, outside the timer (`run.rs:86-89`), so its "first" time starts at the symbolic LU and its "repeat" time at the numeric LU. The condensed timings include their own value fill, and their "repeat" includes `condense_all` (`run.rs:125`). `condensed_sparse` refills the CSC values of its reduced matrix before its numeric LU in every repetition (`condense.rs:267-272`), so the comparison favors `sparse_lu`. For `condensed_dense` nothing is reused, so its "first" and "repeat" rows in the results files time the same work (`run.rs:118-120`, `:128-129`, `:148-150`), and any difference between them is noise and shows no symbolic-analysis cost.

The MATLAB capture (`date:` 2026-10-21T16:46:05Z) ran MATLAB 26.2.0.3386108 (R2026b) with `maxNumCompThreads` 16 and the DLL `built=1792599836`, target `x86_64-pc-windows-gnu` (`s3_matlab.txt:6-7`). Every check passed (`s3_matlab.txt:402`), including the `run_json("s3")` report at each N (`:138`, `:269`, `:400`). The 7 solutions compared are the 5 Rust solves (`sparse_lu` at 1 thread, and `condensed_sparse` and `condensed_dense` at 1 and 16 threads), the `decomposition(A)` solve and `A\b`. The largest pairwise difference is 1.016e-14 at N = 40, 1.471e-14 at N = 100 and 2.672e-14 at N = 400, against the limit of 1e-10, printed next to `condest(A)` values of 1.488e+05, 2.384e+05 and 7.973e+06 (`s3_matlab.txt:89`, `:220`, `:351`). Rounded to four digits, the three differences equal the pairwise differences of the WSL entries for the same N, and the `condest(A)` values equal their `cond1_est` values.

MATLAB `s3_solve` "repeat" medians in us, with (min to max):

| Method, threads | N = 40 | N = 100 | N = 400 |
|---|---|---|---|
| `sparse_lu`, 1 | 621.0 (531.1 to 754.4) | 1210.0 (1088.6 to 1474.8) | 7165.9 (5966.2 to 10016.0) |
| `condensed_sparse`, 1 | 198.9 (148.5 to 313.2) | 421.8 (378.9 to 515.1) | 1987.5 (1760.9 to 2549.4) |
| `condensed_sparse`, 16 | 188.0 (151.3 to 341.0) | 861.2 (629.7 to 1655.1) | 1877.3 (1480.1 to 2260.8) |
| `condensed_dense`, 1 | 376.4 (261.9 to 833.8) | 2857.1 (2563.9 to 3776.4) | 115994.8 (94719.5 to 133883.0) |
| `condensed_dense`, 16 | 1989.4 (1234.7 to 6093.6) | 15863.2 (8936.7 to 25574.4) | 151765.0 (131937.2 to 174370.0) |

The 1-thread methods rank as in WSL, and `sparse_lu` takes 3.6055 times as long as `condensed_sparse` at N = 400. At 16 threads `condensed_sparse` is slower at N = 100 and overlaps the 1-thread spread at N = 40 and 400. Parallel timings in WSL (12 processors) and MATLAB (16) do not compare directly (`docs/ENVIRONMENT.md:17`).

MATLAB's own solves, against the Rust "first" median of `condensed_sparse` at 1 thread:

| N | backward error of both MATLAB solves | `decomposition(A)` + solve | `A\b` | `condensed_sparse` "first", 1 thread |
|---|---|---|---|---|
| 40 | 7.044e-18 | 2583.6 us | 1798.4 us | 293.4 us (266.9 to 426.4) |
| 100 | 3.697e-18 | 5716.2 us | 5657.3 us | 648.2 us (563.8 to 824.0) |
| 400 | 1.885e-18 | 25819.0 us | 25184.5 us | 3861.3 us (2626.5 to 6411.2) |

`timeit` reports no spread. Both MATLAB solves include the factorization (`spikes/matlab/s3_time_matlab.m:3`), and at N = 400 `A\b` takes 6.5223 times as long as the Rust solve. The accuracy fields of the three `run_json` reports (`s3_matlab.txt:91`, `:222`, `:353`) match WSL entries 1 to 3 digit for digit, except the `condensed_dense` backward error at 16 threads at N = 400.

`spikecli s3lu`, numeric LU medians in ns, with (min to max):

| Entry, `date:` | `rustc:` | Shape | Matrix: size, nnz | AUTO picks | AUTO | simplicial | supernodal |
|---|---|---|---|---|---|---|---|
| 1, 2026-10-21T14:21:22Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | n = 4, m = 4, N = 40 | full: 646, 15378 | simplicial | 2.19661e5 (2.15913e5 to 2.28215e5) | 2.15917e5 (2.14608e5 to 2.24961e5) | 3.559501e6 (3.460692e6 to 5.39203e6) |
| 1 | same entry | same | reduced: 166, 1940 | simplicial | 2.3835e4 (2.2747e4 to 1.32437e5) | 6.22e4 (2.909e4 to 1.45528e5) | 1.19034e5 (1.08472e5 to 2.66551e5) |
| 2, 2026-10-21T14:21:56Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | n = 16, m = 4, N = 100 | full: 6418, 537666 | supernodal | 3.179234024e9 (3.172845732e9 to 3.219033118e9) | 2.259175e7 (2.1891406e7 to 2.401351e7) | 3.2182471e9 (3.172566507e9 to 3.243350146e9) |
| 2 | same entry | same | reduced: 1618, 57668 | simplicial | 1.439604e6 (1.368375e6 to 1.456009e6) | 1.461863e6 (1.415201e6 to 2.696621e6) | 5.1934941e7 (5.1442297e7 to 5.2053449e7) |
| 3, 2026-10-21T14:22:17Z | `rustc 1.99.0 (b940084d7 2026-09-28) (Fedora 1.99.0-1.fc45)` | n = 32, m = 2, N = 157 | reduced: 5058, 341764 | supernodal | 1.563831194e9 (1.54944642e9 to 1.597892135e9) | 1.6597388e7 (1.5943392e7 to 1.8196509e7) | 1.575876546e9 (1.568222279e9 to 1.634829953e9) |

The simplicial variant, forced or picked by AUTO, has the lowest median for every matrix. Where AUTO picked supernodal, the numeric LU took 140.7254 times (entry 2, full) and 94.2215 times (entry 3) as long as simplicial. Each setting ran 5 repetitions, except AUTO and forced supernodal on the entry 2 full matrix, which stopped at 4 under the 10000 ms budget.

### What went wrong

- faer's AUTO can pick the slow variant. It chose the supernodal LU for 2 of the 5 probed matrices, because its symbolic estimate of flops per factor entry exceeded 1.0 times 40.0 (`<registry>/faer-0.24.4/src/sparse/linalg/lu.rs:2301-2309`, `mod.rs:4`, `:46`). The sparse `s3` methods use AUTO, which picked simplicial for their matrix at N = 40; N = 100 and 400 were not probed at n = m = 4. `pc-linalg` must choose the variant itself and measure again at larger n.
- Thread scaling. Neither default pool (12 threads in WSL, 16 in MATLAB) was shown to speed up `condensed_sparse` up to N = 400, while 6 threads at N = 400 did. The cause was not investigated. This is open: later phases must choose the thread count, and the N above which to use a pool, from measurements.
- `condensed_dense` results depend on the thread count. At N = 400 its backward error is 2.2445301492326025e-18 at 1 thread, 3.0430044600240353e-18 at 12, 3.2441948375462848e-18 at 6 and 2.2665353467741005e-18 at 16 in MATLAB, since, with more than 1 thread, its dense LU and triangular solves get `Par::rayon(0)` (`run.rs:122`, `condense.rs:284-306`). The bit check covers `condensed_sparse` only (`run.rs:311`). This is open: a dense reduced solve must stay sequential or get a test across thread counts.
- Process-wide state. faer's global parallelism is one value per process, `Rayon(0)` by default (`<registry>/faer-0.24.4/src/lib.rs:1107-1115`), so faer calls from the MATLAB thread run on rayon's global pool until `solve` sets it to `Par::Seq` (`run.rs:212`). rayon never tears down its global pool (`<registry>/rayon-core-1.13.0/src/registry.rs:156-157`, `:187-207`), and a dropped local pool does not join its workers (`<registry>/rayon-core-1.13.0/src/thread_pool/mod.rs:399-403`). Each thread that runs a faer product larger than `16 * 16 * 16` on `x86_64` keeps a buffer of twice the L3 cache size until it exits (`<registry>/faer-0.24.4/src/linalg/matmul/mod.rs:16`; `<registry>/private-gemm-x86-0.1.22/src/lib.rs:1286-1294`). The engine must own its pool, set faer's global parallelism once and count these buffers in its memory budget.
- Allocation failures. A failed allocation in faer's `Mat` constructors panics (`<registry>/faer-0.24.4/src/mat/matown.rs:384-388`), which the barrier turns into `phasecont:engine:panic` (`spikes/mex/pckern_gateway.c:83`). dyn-stack's `MemBuffer::new` (`condense.rs:42`, `:290`; `<registry>/dyn-stack-0.13.2/src/mem.rs:45-47` and `:315-317`) and std `Vec` allocation, the gemm buffer included, abort, which ends the MATLAB session. `pc-linalg` must check a memory budget before large solves and allocate fallibly where it can, with `try_reserve_exact` as `matrix.rs:134-140` does, or `MemBuffer::try_new` (`mem.rs:63`).
- Seed output. `spikecli s3lu` prints a seed above `i64::MAX` as an `f64` (`lu_probe.rs:356-359`), because the spike's JSON writer has no `u64` field (`spikes/pckern/src/ffi.rs:303`, `:309`). The engine's JSON writer needs an exact `u64` path.

## S4, CUDA through NVRTC

### What was built

`spikes/pckern/src/s4_cuda.rs` loads the CUDA driver and NVRTC through `cudarc` 0.19.10 with dynamic loading. It compiles one single-precision saxpy kernel, `y = a*x + y`, to PTX for `compute_120` and to a cubin for `sm_120`, runs each on 16777216 elements and compares the result with a CPU reference. `compile_ms` covers only the NVRTC compile and `load_ms` only the module load. After one checked launch and one warm-up, the kernel row times 20 launches, each with a stream synchronize; `gbps` counts 12 bytes per element over the median.

NVRTC contracts `a*x + y` into a fused multiply-add by default, so the reference is `a*x + y` rounded once. A path passes when it reports no error and its largest distance from the reference is at most 1 ulp (`max_ulp_fma <= 1`). `fma_ref` forms the exact product in `f64`, recovers the exact error of the `f64` sum with 2Sum (Algorithm 2 in Kornerup, Lefèvre, Louvet and Muller, IEEE Trans. Computers 61(3), 289-298, 2012, doi 10.1109/TC.2011.27), rounds the sum to odd and converts it to `f32` (Boldo and Melquiond, IEEE Trans. Computers 57(4), 462-471, 2008, doi 10.1109/TC.2007.70819). The last step rounds as the exact value would, because every `f32`, every midpoint of two adjacent `f32` values and the overflow threshold is an `f64` whose last bit is 0.

`select` picks PTX when the driver version is at least that of the NVRTC that compiled it, else the cubin when the driver reports CUDA 13.0 or newer. Every run chose PTX.

`report` calls `is_culib_present` for the driver and NVRTC before any other `cudarc` call, because `cudarc` panics when a library is missing. The whole run and each path run inside `catch_panic` (`spikes/pckern/src/ffi.rs`) behind the panic barrier of `run_json`, so no panic reaches the C ABI and one failed path does not hide the other.

Unit tests in `s4_cuda.rs` check `fma_ref` against known results on double-rounding cases and special values, and against the x86 FMA instruction on random inputs and all 16777216 spike inputs. `spikes/pckern/tests/s4_cuda.rs` runs a small GPU job (`n=4097 reps=2`).

`spikes/matlab/s4_cuda.m` (variant a) runs the spike in a fresh MATLAB process with the default DLL search. `spikes/matlab/s4_cuda_preload.m` (variant b) first loads the toolkit's `nvrtc-builtins64_134.dll` and `nvrtc64_130_0.dll` by absolute path through `pckern("preload", path)`, which calls `LoadLibraryExW` in `spikes/mex/pckern_gateway.c`. Both print the loaded module paths and recheck the pass and selection rules in MATLAB.

### Commands

```
(cd spikes && cargo build --release --bin spikecli)
spikes/target/release/spikecli s4
CUDA_CACHE_DISABLE=1 spikes/target/release/spikecli s4
spikes/target/release/spikecli s4
spikes/stage.sh
scripts/matlab-win --run spikes/matlab/s4_cuda.m
spikes/stage.sh
scripts/matlab-win --run spikes/matlab/s4_cuda.m
scripts/matlab-win --run spikes/matlab/s4_cuda_preload.m
(cd spikes && cargo build --release --bin spikecli)
spikes/target/release/spikecli s4
CUDA_CACHE_DISABLE=1 spikes/target/release/spikecli s4
spikes/target/release/spikecli s4
```

The first three runs wrote entries 1 to 3 of `spikes/results/s4_wsl.txt`; the failed first S4 run inside MATLAB is kept as `spikes/results/s4_matlab_mingw_fmaf.txt`; the second wrote `spikes/results/s4_matlab.txt`, the preload run `spikes/results/s4_matlab_preload.txt`, and the last three runs entries 4 to 6 of `spikes/results/s4_wsl.txt`.

### Numbers

The first two tables cover the native WSL runs (`spikecli`, target `x86_64-unknown-linux-gnu`), all six entries of `spikes/results/s4_wsl.txt`. Times are the JSON values as printed, in ms in the first table and in ns in the second.

| Entry | `date:` (UTC) | `rustc:` | CUDA cache | PTX `compile_ms` | PTX `load_ms` | Cubin `compile_ms` | Cubin `load_ms` |
|---|---|---|---|---|---|---|---|
| 1 | 2026-10-21T14:22:19Z | 1.99.0 | default, first run | 1.8113901000000002e1 | 5.36971e-1 | 4.3903119999999994e1 | 2.86403e-1 |
| 2 | 2026-10-21T14:22:22Z | 1.99.0 | `CUDA_CACHE_DISABLE=1` (cold) | 2.05742916e2 | 1.8508925200000002e2 | 1.18786741e2 | 2.5991600000000004e-1 |
| 3 | 2026-10-21T14:22:24Z | 1.99.0 | default (warm) | 1.9288460999999998e1 | 4.92562e-1 | 8.329006999999999e0 | 3.18318e-1 |
| 4 | 2026-10-21T19:35:58Z | 1.99.1 | default, first run | 3.5012689e1 | 1.044332e0 | 9.149623e0 | 3.12101e-1 |
| 5 | 2026-10-21T19:36:01Z | 1.99.1 | `CUDA_CACHE_DISABLE=1` (cold) | 1.06510796e2 | 2.02810935e2 | 7.549624e1 | 2.20507e-1 |
| 6 | 2026-10-21T19:36:03Z | 1.99.1 | default (warm) | 1.652219e1 | 4.82147e-1 | 8.814082e0 | 2.82027e-1 |

Every entry reports `driver_version` 13040, NVRTC 13.4 from `/usr/local/cuda-13.4/targets/x86_64-linux/lib/libnvrtc.so.13.4.92`, the NVIDIA GeForce RTX 5060 Laptop GPU at compute capability 12.0, chosen `ptx`, `max_ulp_fma` 0e0 on both paths and `pass` true, under "Best performance". Entries 1 to 3 come from a binary built before `fma_ref` existed and were checked against `f32::mul_add`; entries 4 to 6 come from the rebuilt binary and use `fma_ref`. Without the cache the PTX load took 1.8508925200000002e2 ms and 2.02810935e2 ms, against 4.92562e-1 ms and 4.82147e-1 ms warm; every cubin load stayed below 1 ms. Entries 1 and 4 have PTX loads far below the cold loads, so the cache probably held the module already. Compile times are single samples, highest in the cold entries for a reason not measured.

| Entry | PTX `median_ns` | PTX `min_ns` to `max_ns` | Cubin `median_ns` | Cubin `min_ns` to `max_ns` |
|---|---|---|---|---|
| 1 | 7.264925e5 | 7.09732e5 to 7.62877e5 | 7.30354e5 | 7.03205e5 to 8.07994e5 |
| 2 | 6.89763e5 | 6.69137e5 to 7.52521e5 | 7.18434e5 | 6.92016e5 to 7.84248e5 |
| 3 | 7.309875e5 | 7.01864e5 to 8.6459e5 | 7.05098e5 | 6.86922e5 to 8.77495e5 |
| 4 | 6.739335e5 | 6.63594e5 to 7.2866e5 | 6.80776e5 | 6.62127e5 to 1.0040159999999999e6 |
| 5 | 6.941955e5 | 6.67691e5 to 7.33911e5 | 7.211515e5 | 6.64198e5 to 8.91127e5 |
| 6 | 6.962535e5 | 6.75114e5 to 7.6969e5 | 6.64181e5 | 6.29517e5 to 7.14573e5 |

All twelve kernel medians lie between 6.64181e5 and 7.309875e5 ns; PTX and cubin run at the same speed within that spread.

The third table covers the runs inside MATLAB R2026b (`pckern.dll`, target `x86_64-pc-windows-gnu`). Each file holds one entry, and each run used "Best performance".

| File | `date:` (UTC) | `built=` | NVRTC | Path | `compile_ms` | `load_ms` | `gbps` | Kernel median, min, max | `max_ulp_fma` |
|---|---|---|---|---|---|---|---|---|---|
| `s4_matlab_mingw_fmaf.txt` | 2026-10-21T16:50:23Z | 1792599836 | 13.1, MATLAB `bin\win64` | ptx | 132.359 | 147.398 | 254.4 | 791.3 us, 783.0 us, 865.7 us | 2 |
| | | | | cubin | 25.320 | 32.718 | 256.6 | 784.5 us, 774.7 us, 937.3 us | 2 |
| `s4_matlab.txt` | 2026-10-21T19:26:15Z | 1792610350 | 13.1, MATLAB `bin\win64` | ptx | 67.685 | 2.340 | 304.5 | 661.2 us, 652.9 us, 697.0 us | 0 |
| | | | | cubin | 13.464 | 1.591 | 309.0 | 651.5 us, 646.1 us, 698.8 us | 0 |
| `s4_matlab_preload.txt` | 2026-10-21T19:29:17Z | 1792610350 | 13.4, toolkit `bin\x64` | ptx | 193.760 | 90.068 | 305.5 | 658.9 us, 650.2 us, 692.4 us | 0 |
| | | | | cubin | 86.664 | 0.179 | 299.1 | 673.0 us, 650.8 us, 713.1 us | 0 |

All three runs report `driver_version` 13040, as in WSL, the NVIDIA GeForce RTX 5060 Laptop GPU at compute capability 12.0 and chosen `ptx`; the two from build 1792610350 pass. No path in MATLAB or WSL reported an error. The PTX load of variant b, 90.068 ms against 2.340 ms in variant a, probably includes a first driver JIT of PTX from NVRTC 13.4; neither run set `CUDA_CACHE_DISABLE`. The failed run, the first S4 run inside MATLAB, had a 147.398 ms PTX load that looks like the same cold case; the causes of its 32.718 ms cubin load and slower kernel were not measured.

### What went wrong

- The first S4 run inside MATLAB failed with `max_ulp_fma` 2 on both paths while every WSL run passed. The CPU reference was wrong; the GPU results were right. On `x86_64-pc-windows-gnu` at the baseline target features, `f32::mul_add` becomes a call to `fmaf` from MinGW-w64, a software routine that sums partial products in single precision and is not correctly rounded. The compiler output that shows the call, the disassembly of the routine and the archive listing that places `fmaf` in `libmsvcrt.a` (Fedora `mingw64-crt` 14.0.0-3) are not saved in `spikes/results/`. The placement in `libmsvcrt.a` was inferred from the GCC specs and the archive contents, because the build wrote no link map and the DLL from before the fix was overwritten. `fma_ref` replaced `mul_add`, and the rebuilt DLL passes. `x86_64-w64-mingw32-objdump -t` lists no symbol named `fmaf` or `fma` in it, and no results file records that list. Engine code for this target must not rely on `mul_add` for correct rounding; `f64::mul_add` there was not tested.
- Inside MATLAB the default DLL search returns MATLAB's own `nvrtc64_130_0.dll`, NVRTC 13.1, which has the same file name as the toolkit's 13.4. `cudarc` loads the bare name with `LoadLibraryExW` and no flags. The engine must load its NVRTC and the matching builtins DLL by absolute path before the first `cudarc` call, or accept MATLAB's copy, and check `nvrtcVersion` either way. The spike's `cmd_preload` does not check that the path is absolute, which `LOAD_WITH_ALTERED_SEARCH_PATH` requires; the engine version must.
- Two `cudarc` failures abort the process. When a compile fails and reading the program log fails too, `compile_ptx_with_opts` panics, and the `Drop` of its program panics again if destroying the program also fails. A `Drop` impl that first resolves a missing driver symbol panics during unwinding. Neither happened here. Engine host code should compile both routes through one raw-API helper, as `compile_cubin` does; the `Drop` case has no tested fix.
- The GPU integration test skips only when a library is missing, so a machine with the libraries and no usable device fails it. A CI runner without a GPU needs its own gate.
- `s4_cuda.rs` is still the source of the capture DLL, so its comment errors remain: the module doc says a passing path "matches" the reference, where the code allows 1 ulp, and the `fma_ref` comment leaves out "adjacent" and cites Boldo and Melquiond without pages or DOI. Its `fma_ref` tests lack an overflow-midpoint case, a double-rounding case with a subnormal result, a random double-rounding case with `|y| > |a*x|` and odd-arm cases one `f64` ulp from a midpoint. On the spike inputs plain double rounding gives the same result as `fma_ref`, so the test over all of them guards only against an unfused reference. An engine port should add the missing cases.
- The spike did not check the floating-point control state inside MATLAB, and `fma_ref` assumes round to nearest.
- When edited S4 sources were copied into the spike tree with `cp -p`, the copies kept modification times older than the last build, so `cargo` kept the old stub module and the first test run failed. A `touch` of the copied files fixed it; a plain `cp` avoids it. Build scripts in later phases must not preserve modification times on the sources they stage.

## Recommendations

### 1, Windows build route

Use route 1: build `pckern.dll` for `x86_64-pc-windows-gnu` in WSL with Fedora's `rust-std-static-x86_64-pc-windows-gnu` and `mingw64-gcc`, and build the MEX gateway with the Microsoft Visual C++ 2026 compiler that Windows MATLAB already had.

No Windows route was free of installs, although the Phase 0 brief preferred one that needs no new install. At discovery WSL had the host std only, neither `rustup` nor `cargo-xwin` was installed, and Windows had no `cargo`, `rustc` or `rustup` on its `PATH` (`docs/ENVIRONMENT.md:30`, `:34`, `:61`, `:74`). Route 3 would have needed a Rust install on Windows, where Visual Studio 2026, the MSVC toolsets and the Windows SDK are present (`docs/ENVIRONMENT.md:75-77`). Route 2 would have needed `rustup` and `cargo-xwin`. Route 1 took the two `dnf` commands at `docs/ENVIRONMENT.md:132-133`, which installed the Windows std, `mingw64-gcc` and `mingw64-winpthreads-static` and moved the Rust packages from 1.99.0 to 1.99.1.

Route 1 was picked anyway, for four reasons. It is the first choice in the project rules. Everything it installed is a Fedora package, and Fedora has no package for `cargo-xwin` (`docs/ENVIRONMENT.md:61`, `:149`). One `rustc` builds the Linux and the Windows targets. The Rust build runs in WSL on the repo, and only the finished DLL of that build is copied to `C:` (`spikes/stage.sh:30`, `:39`).

Evidence:

- S1: MATLAB R2026b loaded the DLL through the MSVC-built gateway, and all 39 checks passed, including four panics that arrived as `phasecont:engine:panic` and left no new crash dump (`spikes/results/s1_ffi.txt`, `spikes/results/s1_crash_dumps.txt`).
- S2: the DLL compiled Cranelift code inside MATLAB and ran it, and every `max_ulp_vs_rust` value is `0e0` in the four runs that loaded the DLL (`spikes/results/s2_matlab.txt`, `spikes/results/s2_matlab_power_efficiency.txt`).
- S4: the DLL loaded the CUDA driver and NVRTC through `cudarc` and passed in both runs of build `1792610350` (`spikes/results/s4_matlab.txt`, `spikes/results/s4_matlab_preload.txt`). The cross build needs no CUDA toolkit and no `nvcc`: `cudarc` is built with `dynamic-loading` and the fixed feature `cuda-13000`, so its build script neither runs `nvcc` nor links a CUDA library (`spikes/pckern/Cargo.toml:24`). `<registry>/cudarc-0.19.10/build.rs` runs `nvcc` only in `cuda_version_from_build_system` (`:158-159`). `main` calls that function only when neither `CUDARC_CUDA_VERSION` nor a version feature such as `cuda-13000` gives the version and the `cuda-version-from-build-system` feature is on (`:112-127`). Every link directive sits in `dynamic_linking` (`:187-249`) and `static_linking` (`:252-332`), which `main` calls only under the `dynamic-linking` or `static-linking` feature (`:150-154`). The DLL imports no CUDA DLL (next bullet).
- The DLL needs no MinGW runtime DLL. `x86_64-w64-mingw32-objdump -p` on the build with stamp `1792610350` lists `kernel32.dll`, `ntdll.dll`, `userenv.dll`, `ws2_32.dll`, `api-ms-win-core-synch-l1-2-0.dll`, `bcryptprimitives.dll` and `msvcrt.dll`, and no results file records that list. `spikes/stage.sh:32-39` stages a DLL only if it imports none of `libgcc`, `libwinpthread` and `libstdc`, and the DLL of the S1 and S2 captures (stamp `1792599836`) was staged that way.
- The gateway needed no install. `mex.getCompilerConfigurations('C','Installed')` lists only Microsoft Visual C++ 2026 (C) (`docs/ENVIRONMENT.md:88`, `:94`), and the no-DLL run of S1 built the gateway with it at 2026-10-21T14:26:04Z, before the route 1 packages existed (`spikes/results/s1_gateway_nodll.txt`).

Not tested: routes 2 and 3 were neither installed nor run, so nothing in this report compares them with route 1. A MinGW build of the gateway was not tried. The DLL and the gateway ran on this machine only. Smart App Control has not blocked the unsigned `pckern.dll` and `pckern.mexw64` here (`docs/ENVIRONMENT.md:15`); whether it accepts a release DLL on a user's machine was not tested. Loading the DLL on a machine without the NVIDIA driver was not tested.

Upkeep: the Windows std and `rustc` must come from one Fedora build. After any Rust update, `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` must report the same version, or Windows builds fail (`docs/ENVIRONMENT.md:34`, `:139`). No version lock is set, and after review the owner decided against one ("Decisions after review" below). The `dnf versionlock add` command at `docs/ENVIRONMENT.md:142` has not been run. The 1.99.1 packages came from `updates-testing`, which is disabled by default (`docs/ENVIRONMENT.md:32`). Smart App Control blocked one test executable that `cargo test` built for the Windows target (`docs/ENVIRONMENT.md:15`), and every results file that records the Windows target is a MATLAB capture. Whether it would block a test executable built by route 2 or 3 was not tested. The import check in `spikes/stage.sh` is a deny-list of three names, so the `cargo` task of Phase 7a needs an allow-list of Windows system DLLs (the `docs/SPEC.md:277` item of recommendation 4).

### 2, default model backend

Keep the Cranelift JIT as the default model backend, and keep AOT C as the optional path that is used only when `mex` finds a compiler. This follows rule 5 of the project rules ("Users never need a compiler."). On the two-variable model of S2 the median ratios of JIT loop time to C loop time ran from 0.976 to 1.124 in MATLAB (S2, Numbers), each loop also builds its inputs and folds its outputs, and the JIT and C code give the same results, so S2 shows no reason on win64 to make a compiler a requirement.

Evidence:

- Inside MATLAB the JIT compiled the three test functions (the FitzHugh-Nagumo right-hand side, its Jacobian and a function that calls an `exp` shim) into 279 code bytes. All results equal the Rust reference, with `max_ulp_vs_rust` at `0e0` in every block of `spikes/results/s2_matlab.txt`. The first compile in each of three MATLAB processes took 28.082 ms, 1.386 ms and 1.477 ms (S2, Numbers).
- The median ratios of JIT time to C time ran from 0.976 to 1.124 in MATLAB against the MSVC-compiled `spikes/mex/fhn_aot.c`, and from 1.0310260114630359e0 to 1.1210299380132351e0 in WSL against gcc with `-ffp-contract=off` (S2, Numbers). For the Jacobian the median is above 1 in all six entries, so the JIT was slower than the C code there, although the smallest repetition in each entry is below 1. The right-hand side medians in MATLAB are 1.044, 0.984 and 0.976 and show no consistent winner. MATLAB and WSL differ in operating system, Rust target and C compiler, so the two sets are not compared with each other.
- The Rust code inlined into the timing loop, the best case for ahead-of-time code, had a lower Jacobian median than the JIT in all 9 blocks of S2 (6 in MATLAB, 3 in WSL), while the JIT and the C code were called through a function pointer (S2, Numbers).
- The results are for one two-variable model. Nothing in S2 compares the two backends on a larger model.

macOS arm64 was not tested, because the spikes ran only on Windows and WSL. The check that `docs/SPEC.md:99` assigns to S2 moves to Phase 2. Reading `cranelift-jit` 0.137.0 gives two points for that check (S2, What went wrong). `JITModule::new` accepts `is_pic=true` only on x86_64, and the spike sets it on every target, so the spike code would fail on aarch64. The default memory provider maps code pages without `MAP_JIT` and switches them to read-execute with `mprotect`, which under the hardened runtime needs an entitlement that allows unsigned executable memory in the MATLAB process. That last point follows Apple's documentation (item 4.4) and was not checked on a Mac.

Owner decision: the fallback in `docs/SPEC.md:99` (AOT plus a documented requirement for the Xcode Command Line Tools on Macs) conflicts with rule 5 of the project rules. If the Phase 2 check fails, one of the two has to give way, and S2 does not settle which. Decided on 2026-10-22: rule 5 holds ("Decisions after review" below).

Not tested: macOS arm64, MATLAB on glnxa64, models larger than two variables, directional derivatives, calls from several threads, a shim that panics, and C emitted by `pc-codegen` (the C reference is the fixed file `spikes/mex/fhn_aot.c`). The results do not record the MSVC flags that `mex` used.

Upkeep: the JIT code pages are freed only by `JITModule::free_memory`, shims are registered with `JITBuilder::symbol` and must not panic, `is_pic` is set only on x86_64, and stack probes are switched on (the `docs/SPEC.md:89`, `:91` and `:93` items of recommendation 4). AOT code is compiled with floating-point contraction off, and the compiler, its version and its flags are stored with every result that uses it (the `docs/SPEC.md:97` item). The 4 ulp bounds in `docs/SPEC.md:95` and `:97` are untested, since S2 had no dual-number interpreter and no AOT code from `pc-codegen`.

### 3, first linear solver

Use condensation of parameters as `docs/SPEC.md` section 7 describes it (lines 165 to 170), and solve the reduced system with faer's sparse LU, with the simplicial variant chosen in the code. Run the solve sequentially by default. Keep the sparse LU of the full matrix as the reference that tests compare against, and leave the dense LU of the reduced system out of the engine. Run the condense step in a rayon pool only above a value of N and with a thread count, both of which Phase 5 measures.

Evidence:

- All three methods passed every check in the four WSL entries and in the MATLAB capture. The largest backward error is 1.7447853724986725e-17 against the limit of 1e-13, and the largest pairwise difference is 2.672448648623745e-14 against 1e-10, with `cond1_est` up to 7.972520936075523e6 (S3, Numbers).
- At 1 thread `condensed_sparse` has the lowest "repeat" median. In WSL entries 1 to 3 `sparse_lu` takes 3.9982, 2.7167 and 2.8257 times as long, and in the MATLAB `s3_solve` calls it takes 3.6055 times as long at N = 400. The setup favors `sparse_lu`, which builds its matrix once outside the timed region, while `condensed_sparse` refills its reduced matrix in every repetition.
- At N = 400 in MATLAB, `A\b` took 25184.5 us and `decomposition(A)` plus one solve took 25819.0 us, both including the factorization. The "first" median of `condensed_sparse` at 1 thread in the `s3_solve` call, which includes the symbolic analysis, is 3861.3 us.
- The dense reduced solve is slow at the largest size. At N = 400 in WSL entry 3, `condensed_dense` has a median of 5.59271955e7 ns at 1 thread against 9.425525e5 ns for `condensed_sparse`, and its backward error changes with the thread count (S3, What went wrong).
- faer's automatic choice between its two sparse LU variants picked the supernodal one for 2 of the 5 matrices that `spikecli s3lu` probed. There the numeric LU took 140.7254 times (n = 16, full matrix) and 94.2215 times (n = 32, m = 2, N = 157, reduced system) as long as with the simplicial variant, and the supernodal variant was the slower one on all five matrices.
- Neither default pool (12 threads in WSL, 16 in MATLAB) was shown to speed up `condensed_sparse` up to N = 400. A pool of 6 threads at N = 400 made it 1.4515 times faster, and its condense step 3.2407 times faster (WSL entry 4).

Not tested: the `s3` runs used n = 4 and m = 4 only, with N up to 400, and the `s3lu` probe went to n = 32 for the LU alone. The Jacobian blocks are minus the identity plus 0.3 times a random matrix with entries uniform between -1 and 1, and no model Jacobian was tried. Mesh adaptation and Floquet multipliers from the condensed blocks were not tried. The runs did not record which variant faer's automatic choice took in the `s3` runs at N = 100 and 400. The cause of the poor thread scaling was not investigated.

Upkeep: `pc-linalg` chooses the LU variant itself, sets faer's global parallelism once and owns its rayon pool, and it checks a memory budget before large solves because a failed allocation in `Vec` or in dyn-stack's `MemBuffer::new` aborts the MATLAB process (S3, What went wrong). These are the `docs/SPEC.md:168` and `:170` items of recommendation 4.

### 4, changes to the project rules and `docs/SPEC.md`

All 17 items were approved on 2026-10-22 and are applied, with the changes to 4.4 and 4.16 that "Decisions after review" lists. The text below is the proposal as it was reviewed. Proposals for the same lines are merged into one item. The first group holds text that a spike result contradicts or that Phase 0 has settled. The second group holds text that is correct but lacks something the spikes found. Each item gives the line, a quote of the current text, the proposed wording and the evidence. "Not tested" marks what a spike did not exercise; it does not mean the current text is wrong.

Line numbers and quotes are from the working tree. Quotes and proposed wording leave out the bold label that starts a line (for example `Rust.` or `Gateway.`), and the label stays as it is.

#### Group 1, text that a result contradicts or that Phase 0 settled

##### 4.1, project rules, the Rust toolchain

Current text:

```text
The WSL Rust toolchain is Fedora's packaged `rustc`/`cargo` 1.99, not rustup, so no Windows target is installed yet. Phase 0 picks the Windows build route. Candidates, in order of preference:
```

Proposed:

```text
The WSL Rust toolchain is Fedora's packaged `rustc`/`cargo` 1.99.1, not rustup, with the Windows std `rust-std-static-x86_64-pc-windows-gnu` and the MinGW linker `x86_64-w64-mingw32-gcc`. Phase 0 chose route 1 below. After any Rust update, `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` must report the same version, or Windows builds fail. The routes Phase 0 considered, in order of preference:
```

Evidence: `docs/ENVIRONMENT.md:25-30` (versions, packages and targets, no rustup), `:34` and `:139` (the std and `rustc` must come from the same build), `:48` (MinGW gcc) and `:129-137` (the installs); `spikes/.cargo/config.toml:1-2` (the linker); `s1_ffi.txt:7-8` (a DLL from this toolchain loaded in MATLAB). Recommendation 1 gives the reasons for route 1.

##### 4.2, project rules, the gateway compiler

Current text:

```text
Either the compiler that `mex -setup C` finds in Windows MATLAB, or a MinGW cross-compile from WSL linked against MATLAB's import libraries under `/mnt/c/Program Files/MATLAB/R2026b/extern/lib/win64/`. Check which exists before proposing an install.
```

Proposed:

```text
Windows MATLAB lists Microsoft Visual C++ 2026 (C) as its only installed C compiler, `mex -setup C` selects it, and the gateway builds with it (`mex -R2018a`). A MinGW cross-compile of the gateway is not needed.
```

Evidence: `docs/ENVIRONMENT.md:88`, `:94`; `s1_gateway_nodll.txt:9-10`; `spikes/stage.sh:45`; `s1_ffi.txt:7-8`.

Not tested: a MinGW build of the gateway.

##### 4.3, project rules, the GPU line

Two sentences of the line change. The first proposed block replaces the line's second sentence, and the second block replaces its fourth sentence.

Current text:

```text
That is outside the 5.0 to 9.x range MATLAB officially supports for `gpuArray`, so the native CUDA backend (driver API plus NVRTC through the `cudarc` crate) is the main NVIDIA path.

Code that runs inside Windows MATLAB needs the Windows NVRTC DLL (`nvrtc64_*.dll`), not the WSL CUDA install.
```

Proposed:

```text
MATLAB R2026b's `gpuDevice` reports this card as supported, but `gpuArray` needs Parallel Computing Toolbox (rule 8), so the native CUDA backend (driver API plus NVRTC through the `cudarc` crate) is the main NVIDIA path. Spike S4 ran it inside R2026b on this card.

Code that runs inside Windows MATLAB needs a Windows NVRTC DLL (`nvrtc64_*.dll`), not the WSL CUDA install. The R2026b installation here has its own `nvrtc64_130_0.dll` (NVRTC 13.1) in `bin\win64`. The CUDA 13.4 toolkit's DLL has the same name, and a bare-name load from inside MATLAB gets MATLAB's copy. To use another NVRTC, load it and its `nvrtc-builtins64_*.dll` by absolute path before the first `cudarc` call. Check `nvrtcVersion` either way.
```

Evidence for the first sentence:

- `docs/ENVIRONMENT.md:90`: `gpuDevice` reports `ComputeCapability` 12.0 and `DeviceSupported` 1. `docs/ENVIRONMENT.md:117` says the range in the project rules does not hold for R2026b and leaves the change to this report.
- Rule 8 of the project rules allows Parallel Computing Toolbox only inside the optional `gpuArray` route.
- The `cudarc` backend ran on this card inside R2026b: `s4_matlab.txt:13`, `:36`; `s4_matlab_preload.txt:17`, `:42`.

Evidence for the second sentence:

- With the default search, the run loaded NVRTC 13.1 from MATLAB's folder together with `nvrtc-builtins64_131.dll` (`s4_matlab.txt:12`, `:33`, `:35`).
- With the preload by absolute path, the run loaded NVRTC 13.4 from the toolkit folder (`s4_matlab_preload.txt:11`, `:12`, `:16`, `:37`, `:38`, `:40`, `:41`). Both variants passed (`s4_matlab.txt:36`, `s4_matlab_preload.txt:42`).
- `cudarc` tries bare file names in order, `nvrtc64_130_0.dll` among them (`<registry>/cudarc-0.19.10/src/lib.rs:218-223`, `<registry>/cudarc-0.19.10/src/nvrtc/sys/mod.rs:540-550`), and `libloading` hands each name to `LoadLibraryExW` with flags 0 (`<registry>/libloading-0.9.0/src/os/windows/mod.rs:69-71`, `:169`).
- The spike's preload calls `LoadLibraryExW` with `LOAD_WITH_ALTERED_SEARCH_PATH` (`spikes/mex/pckern_gateway.c:445-461`, call at `:457`) and does not check that the path is absolute.
- `docs/ENVIRONMENT.md:92`, `:113` and `:115` list both copies.

Not tested: `gpuArray` itself (S4 made no `gpuArray` call), other MATLAB releases, whether a MATLAB installation without Parallel Computing Toolbox has the `bin\win64` copy (the toolbox is installed here, `docs/ENVIRONMENT.md:100`), and a preload of `nvrtc64_130_0.dll` without its builtins DLL. NVIDIA's terms for shipping an NVRTC DLL with the toolbox were not checked.

##### 4.4, `docs/SPEC.md:99`, the macOS arm64 check

Current text:

```text
Confirm that JIT pages can be mapped executable inside the MATLAB process. This is spike S2. If it fails there, the fallback is AOT plus a documented requirement for the Xcode Command Line Tools on Macs only.
```

Proposed, appended to the current text:

```text
Phase 0 ran only on Windows and WSL, so this check moves to Phase 2. `cranelift-jit` 0.137.0 accepts `is_pic=true` only on x86_64. Its default `SystemMemoryProvider` maps code pages read-write without `MAP_JIT` and makes them read-execute with `mprotect`, which the hardened runtime allows only with `com.apple.security.cs.allow-unsigned-executable-memory` or `com.apple.security.cs.disable-executable-page-protection`. `codesign -d --entitlements - <matlabroot>/bin/maca64/MATLAB`, where `<matlabroot>` is the folder that `matlabroot` returns, lists the entitlements MATLAB has. Without either entitlement, `pc-jit` needs its own `JITMemoryProvider`, set with `JITBuilder::memory_provider`, that maps pages with `MAP_JIT`, which needs `com.apple.security.cs.allow-jit`, and switches them with `pthread_jit_write_protect_np`. With none of the three, the AOT fallback applies.
```

Evidence: the `is_pic` assert (`<registry>/cranelift-jit-0.137.0/src/backend.rs:370-374`), which `JITBuilder::new` satisfies through `with_flags`, which sets `is_pic` only on x86_64 (`:44-48`, `:75-80`), and the spike does not (`spikes/pckern/src/s2_jit.rs:71`). The default memory maps with `MmapMut::map_anon` (`<registry>/cranelift-jit-0.137.0/src/memory/system.rs:31`), which passes `PROT_READ | PROT_WRITE` and `MAP_PRIVATE | MAP_ANON` (`<registry>/memmap2-0.9.11/src/unix.rs:352-359`). It then calls `region::protect` with `READ_EXECUTE` (`<registry>/cranelift-jit-0.137.0/src/memory/mod.rs:60-66`), which calls `libc::mprotect` (`<registry>/region-3.0.2/src/os/unix.rs:50`). `grep -rn 'MAP_JIT\|pthread_jit'` over `<registry>/cranelift-jit-0.137.0/src` finds nothing. A custom provider implements `JITMemoryProvider` (`<registry>/cranelift-jit-0.137.0/src/memory/mod.rs:30`) and is set with `JITBuilder::memory_provider` (`<registry>/cranelift-jit-0.137.0/src/backend.rs:161-167`). Apple's entitlement pages say that `allow-unsigned-executable-memory` permits writable and executable memory without the `MAP_JIT` restrictions, that `disable-executable-page-protection` includes it, and that `allow-jit` permits `MAP_JIT` (https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.allow-unsigned-executable-memory, https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.disable-executable-page-protection, https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.security.cs.allow-jit). Apple's porting guide adds two limits for a `MAP_JIT` provider: a hardened app with `allow-jit` can create only one `MAP_JIT` region, and the `com.apple.security.cs.jit-write-allowlist` entitlement turns `pthread_jit_write_protect_np` off (https://developer.apple.com/documentation/apple-silicon/porting-just-in-time-compilers-to-apple-silicon). Whether MATLAB holds any of these entitlements or already uses its `MAP_JIT` region, and the `bin/maca64/MATLAB` path, were not checked.

Not tested: macOS arm64, since S2 ran only on Windows and WSL, so S2 does not show the current text to be wrong. If the Phase 2 check fails, the fallback in this paragraph conflicts with rule 5 of the project rules, the owner decision in recommendation 2. Item 4.4 now carries the fallback chosen after review.

##### 4.5, `docs/SPEC.md:293`, the `cuda` backend

Current text:

```text
  - `cuda`: `cudarc`, driver API plus NVRTC; primary for NVIDIA, and required for the developer's sm_120 GPU.
```

Proposed:

```text
  - `cuda`: `cudarc`, driver API plus NVRTC; primary for NVIDIA. Spike S4 compiled and ran an `sm_120` kernel with it on the developer's GPU, inside MATLAB R2026b and in WSL.
```

Evidence:

- The `cudarc` path ran on this GPU in both passing MATLAB runs (`s4_matlab.txt:13`, `:15`, `:19`, `:36`; `s4_matlab_preload.txt:17`, `:19`, `:23`, `:42`) and in all six WSL entries (`s4_wsl.txt:6`, `:13`, `:20`, `:27`, `:34`, `:41`).
- S4 shows that this path works, and nothing in it shows that the path is required. The reason the project rules give for the native path, the `gpuArray` range, does not hold for R2026b (`docs/ENVIRONMENT.md:90`, `:117`).

Not tested: `gpuArray` and the `opencl` backend on this GPU.

#### Group 2, text that is correct but lacks what the spikes found

##### 4.6, project rules, the panic rule

Current text:

```text
   - No panic may cross the C ABI. Wrap every exported function in `catch_unwind` and return an error code. The gateway turns errors into `mexErrMsgIdAndTxt` with IDs of the form `phasecont:engine:<kind>`.
```

Proposed:

```text
   - No panic may cross the C ABI. Wrap every exported function in `catch_unwind`. After a caught panic, a function that returns a status returns an error code, and a function that returns a pointer returns NULL. A panic on a Rust worker thread must also reach MATLAB as an error: through the barrier of the call that waits for the worker or, for a thread that outlives its call, through a later call such as `pc_run_poll`. The gateway turns errors into `mexErrMsgIdAndTxt` with IDs of the form `phasecont:engine:<kind>`.
```

Evidence: the exported symbol `pc_api_v1` and the table entries `info` and `last_error` return pointers, which are NULL after a caught panic, and `free_string` returns nothing, so none of them can return an error code (`spikes/pckern/src/lib.rs:28-47`, `:67-73`, `:75-88`; `spikes/pckern/src/ffi.rs:155-167`, `:169-175`, `:177-187`). The gateway checks for a NULL table and for a NULL from `info` (`spikes/mex/pckern_gateway.c:158-163`, `:252-258`). A panic inside a rayon pool reached the barrier of the calling function on both calls (`s1_ffi.txt:32`, `:34`), without its file and line. A thread that outlives its call has no barrier waiting for it (item 4.12).

Not tested: neither NULL path ran in a capture, and no capture shows a panic on a worker thread outside a rayon pool.

##### 4.7, project rules, the GNU target and `mul_add`

Current text:

```text
This targets `x86_64-pc-windows-gnu`, which is fine because the gateway loads `pckern` through a plain C ABI.
```

Proposed, appended to the current text:

```text
On this target at the baseline target features, `f32::mul_add` calls MinGW-w64's `fmaf`, which is not correctly rounded, so code that needs a single rounding does not use it there (spike S4 in `docs/spikes.md`). `f64::mul_add` on this target has not been tested.
```

Evidence:

- The current sentence holds: the GNU DLL ran behind the MSVC-built gateway (`s1_gateway_nodll.txt:9`, `spikes/stage.sh:45`) and passed (`s4_matlab.txt:7`, `:8`, `:36`).
- The first S4 capture in MATLAB failed on the CPU reference that `f32::mul_add` computed through `fmaf` (`s4_matlab_mingw_fmaf.txt:7`, `:16`, `:20`), while the WSL entries with the same code passed (`s4_wsl.txt:6`, `:13`, `:20`). S4, What went wrong, gives the cause and the fix in `spikes/pckern/src/s4_cuda.rs:147-150`. The rebuilt DLL passes (`s4_matlab.txt:7`, `:16`, `:20`), and `x86_64-w64-mingw32-objdump -t` lists no symbol named `fmaf` or `fma` in it. No results file records that symbol list.

Not tested: `f64::mul_add` on this target, and the same check on `x86_64-pc-windows-msvc`.

##### 4.8, `docs/SPEC.md:89`, entry points and code generation flags

Current text:

```text
Lower each needed DAG to Cranelift IR as an `extern "C"` function. A typical signature is `fn(x: *const f64, p: *const f64, out: *mut f64)`. Directional entries take extra vector pointers.
```

Proposed: append the first block to the current text, and add the second block as a new paragraph after it, with the bold label `Code generation flags.`

```text
Matrix outputs such as the Jacobian are written column major.
```

```text
Set `is_pic` only on x86_64 and set `use_colocated_libcalls` to false, as `JITBuilder::new` does. Set `enable_probestack` to true and `probestack_strategy` to `inline`. Cranelift leaves stack probes off by default, Windows needs them for frames over 4096 bytes, and the default `outline` strategy calls `__cranelift_probestack`, which `pckern` does not provide.
```

Evidence for the layout: the spike stores the Jacobian column major (`spikes/pckern/src/s2_jit.rs:3-5`), and `jit_jac` is `0e0` against the Rust reference in that layout in every block (`s2_matlab.txt:12`, `:52`, `:92`; `s2_wsl.txt:13`, `:20`, `:27`). `docs/SPEC.md:75` lists the Jacobian without an order. S2 did not compare layouts, so the sentence records the layout the spike used and nothing more.

Evidence for the flags: the spike's flags and their reasons (`spikes/pckern/src/s2_jit.rs:69-79`). `JITBuilder::with_flags`, which `JITBuilder::new` calls, sets `use_colocated_libcalls` to false because AArch64 colocated calls use short-range relocations, and sets `is_pic` only on x86_64 (`<registry>/cranelift-jit-0.137.0/src/backend.rs:44-48`, `:65-80`); the assert at `:370-374` rejects `is_pic=true` elsewhere. `enable_probestack` defaults to false (`<registry>/cranelift-codegen-meta-0.137.0/src/shared/settings.rs:255-260`), frames larger than the 4096-byte guard region get probes (`:262-272`), and `outline` is the first value of `probestack_strategy` (`:274-283`), which is the default because an enum setting's default byte is 0 (`<registry>/cranelift-codegen-meta-0.137.0/src/cdsl/settings.rs:43`). The outline probe calls `__cranelift_probestack` (`<registry>/cranelift-module-0.137.0/src/lib.rs:41`), and the only mentions of that name under `spikes/` are the comments at `spikes/pckern/src/s2_jit.rs:74` and `spikes/API_NOTES.md:31`.

Not tested: the S2 functions have small frames, and no aarch64 build ran.

##### 4.9, `docs/SPEC.md:91`, shims for transcendentals

Current text:

```text
Call Rust `extern "C"` shims around the `f64` methods (`sin`, `exp`, and so on).
```

Proposed, appended to the current text:

```text
Register every shim with `JITBuilder::symbol`. Shim bodies must not panic: a panic that leaves an `extern "C"` function aborts the process, JIT frames have no unwind information, and the panic barrier sits outside the JIT call.
```

Evidence: `JITBuilder::with_isa` installs only the platform lookup (`<registry>/cranelift-jit-0.137.0/src/backend.rs:98-103`), which runs after the builder's own table (`:210-222`). On Windows it searches the running executable and `ucrtbase.dll`, and on other systems it calls `dlsym(RTLD_DEFAULT)` (`:662-704`), so inside MATLAB on Windows it does not find a function that only `pckern.dll` exports. An import that does not resolve panics with "can't resolve symbol" (`:258-263`). `cranelift-jit` registers no unwind information with the system unwinder: `grep -rn 'register_frame\|RtlAddFunctionTable\|RtlInstallFunctionTableCallback'` over `<registry>/cranelift-jit-0.137.0/src` finds nothing. Its optional `wasmtime-unwinder` feature (`<registry>/cranelift-jit-0.137.0/Cargo.toml:31-33`) adds only exception tables for Wasmtime's own unwinder (`<registry>/cranelift-jit-0.137.0/src/backend.rs:394-401`). The abort comes from the `catch_unwind` documentation (`std/src/panic.rs:296-300`) and `panic_cannot_unwind` (`core/src/panicking.rs:346-351`) in the installed rust-src under `/usr/lib/rustlib/src/rust/library/`. The spike registers `pc_shim_exp` with `JITBuilder::symbol` (`spikes/pckern/src/s2_jit.rs:27-31`, `:83`), and `jit_shim_exp` is `0e0` in every block (`s2_matlab.txt:12`, `:24`, `:52`, `:64`, `:92`, `:104`; `s2_wsl.txt:13`, `:20`, `:27`).

Not tested: a panicking shim.

##### 4.10, `docs/SPEC.md:93`, freeing JIT code

Current text:

```text
Each compiled model owns its `JITModule` and frees it when its handle is freed.
```

Proposed:

```text
Each compiled model owns its `JITModule`. Dropping a `JITModule` leaves its code pages mapped, so the model handle calls `JITModule::free_memory` when it is freed and on every error path after `JITModule::new`, and only when no thread can still call the compiled functions.
```

Evidence: the `Drop` of the default memory forgets its allocations to keep function pointers valid (`<registry>/cranelift-jit-0.137.0/src/memory/system.rs:146-152`). Only `free_memory` releases them, and its safety note forbids running or calling the functions afterwards (`<registry>/cranelift-jit-0.137.0/src/backend.rs:198-208`, `:283-284`). The spike frees the module in the `Drop` of `Jit` (`spikes/pckern/src/s2_jit.rs:45-52`), but an error between `JITModule::new` (`spikes/pckern/src/s2_jit.rs:84`) and the construction of `Jit` drops the module without `free_memory`. That leak is still open in the spike.

Not tested: the leak was found by reading the code, and no run measured it.

##### 4.11, `docs/SPEC.md:97`, AOT compiler flags

Current text:

```text
`pc-codegen` emits C with the same signatures. It is compiled through `mex` when a compiler exists and selected with `phasecont.config("Codegen","aot")`. AOT output must match JIT output to within 4 ulp.
```

Proposed, appended to the current text:

```text
Compile AOT code with floating-point contraction turned off, as the spike's gcc build does with `-ffp-contract=off`, and store the compiler, its version and its flags with every benchmark or test result that uses AOT code.
```

Evidence: the WSL build passes `-O2 -fPIC -ffp-contract=off`, with a comment that without contraction the C code rounds like the JIT and Rust code (`spikes/pckern/build.rs:33-35`). `c_rhs` and `c_jac` are `0e0` in every block (`s2_matlab.txt:24`, `:64`, `:104`; `s2_wsl.txt:13`, `:20`, `:27`). The WSL JSON records the compiler with its flags (`"c_compiler":"cc (GCC) 16.2.1 20260819 (Red Hat 16.2.1-2), -O2 -fPIC -ffp-contract=off"`, `s2_wsl.txt:13`, `:20`, `:27`). The MATLAB `bench_c` JSON records only "passed in by the caller" (`s2_matlab.txt:24`, `:64`, `:104`), and `spikes/stage.sh:45` adds only `/W4` to the `mex` defaults.

Not tested: C code built with contraction allowed, clang on maca64, and which MSVC flags control contraction.

##### 4.12, `docs/SPEC.md:129`, panics on the background thread

Current text:

```text
A continuation runs on a Rust background thread. MATLAB polls for new points with `pc_run_poll` from a `timer`, so plots update live, and `pc_run_cancel` stops the run.
```

Proposed, appended to the current text:

```text
A panic on the background thread is caught on that thread, with its message and source location, and the next `pc_run_poll` reports it as an error.
```

Evidence: the panic hook records a location only while a barrier runs on the panicking thread and hands every other panic to the previous hook, which writes to stderr (`spikes/pckern/src/ffi.rs:69-74`). That output does not appear in the capture, and the worker panic arrived without its file and line (`s1_ffi.txt:32`, `:34`, against `:28`, `:30`). It reached MATLAB because rayon carries a panic back to the thread that called `install`, where the barrier catches it (`spikes/pckern/src/s1.rs:36-38`, `:46-52`). A background run outlives the call that starts it, so no barrier waits for it.

Not tested: a panic on a thread that outlives the call that started it.

##### 4.13, `docs/SPEC.md:168`, the reduced solve

Current text:

```text
3. Solve that reduced system.
```

Proposed:

```text
3. Solve that reduced system with faer's sparse LU. `pc-linalg` chooses between faer's simplicial and supernodal variants itself, starting with the simplicial one, instead of using faer's automatic choice. Tests compare the result with a sparse LU of the full matrix.
```

Evidence: recommendation 3. Every method passed every check, and `condensed_sparse` has the lowest 1-thread "repeat" median in every entry (S3, Numbers). faer's automatic choice picks the supernodal variant when its symbolic estimate of flops per factor entry exceeds 40 (`<registry>/faer-0.24.4/src/sparse/linalg/lu.rs:2301-2309`; the factor 40.0 is at `<registry>/faer-0.24.4/src/sparse/linalg/mod.rs:4` and the automatic threshold 1.0 at `:46`). It did so for 2 of the 5 probed matrices, where the numeric LU took 140.7254 and 94.2215 times as long as with the simplicial variant (`s3lu_wsl.txt:13`, `:20`).

Not tested: the `s3` runs did not record which variant faer chose at N = 100 and 400, and the probe stopped at n = 32.

##### 4.14, `docs/SPEC.md:170`, threads and memory in the condense step

Current text:

```text
Assemble and condense in Rust, parallelizing across intervals with rayon when $N$ is large.
```

Proposed:

```text
Assemble and condense in Rust. Run sequentially by default, and parallelize across intervals in a rayon pool that `pc-linalg` owns, only above a value of $N$ and with a thread count, both of which Phase 5 measures. `pc-linalg` sets faer's global parallelism once and checks a memory budget before large solves, since a failed allocation in `Vec` or in dyn-stack's `MemBuffer::new` aborts the MATLAB process.
```

Evidence: neither default pool (12 threads in WSL, 16 in MATLAB) was shown to speed up `condensed_sparse` up to N = 400, while a pool of 6 threads at N = 400 made it 1.4515 times faster (`s3_wsl.txt:27`). faer's global parallelism is one value per process (`<registry>/faer-0.24.4/src/lib.rs:1107-1115`), and rayon never tears down its global pool (`<registry>/rayon-core-1.13.0/src/registry.rs:156-157`, `:187-207`). dyn-stack's `MemBuffer::new` (`<registry>/dyn-stack-0.13.2/src/mem.rs:45-47`, which calls `new_in` at `:315-317`) and std `Vec` abort when an allocation fails (S3, What went wrong).

Not tested: the cause of the poor scaling with 12 and 16 threads, N above 400, and an allocation failure (the abort comes from reading the sources).

##### 4.15, `docs/SPEC.md:235`, `:239` and `:253`, the ABI table and the gateway

Current text, line 235:

```text
The Rust side is a `cdylib` named `pckern` that exports a single symbol, `pc_api_v1()`. That function returns a struct of function pointers plus an ABI version number. Sketch:
```

Line 239, inside the `pc_api_v1_t` sketch:

```text
  uint32_t abi_version;
```

Line 253:

```text
`mex/pckern_gateway.c` is compiled with `mex -R2018a`. On its first call it loads `pckern` from the folder that holds the MEX file (`LoadLibraryW` on Windows, `dlopen` elsewhere), checks `abi_version`, and dispatches on a command string. Loading the library at runtime keeps the gateway independent of the Rust toolchain (MSVC or MinGW on Windows).
```

Proposed: the first block replaces the second sentence of line 235, the second block is a new line after line 239, and the third block replaces line 253.

```text
That function returns a struct of function pointers that starts with an ABI version number and the struct's size in bytes.
```

```text
  uint32_t struct_size;   /* sizeof the table in this build of pckern */
```

```text
`mex/pckern_gateway.c` is compiled with `mex -R2018a`. The first command that needs `pckern` loads it from the folder that holds the MEX file: on Windows the gateway finds its own file with `GetModuleHandleExW` and `GetModuleFileNameW` and loads `pckern.dll` by full path with `LoadLibraryExW` and `LOAD_WITH_ALTERED_SEARCH_PATH`, and elsewhere it uses `dlopen`. The gateway then calls `pc_api_v1`, requires `abi_version` to equal its own and `struct_size` to be at least the size of its own table, and dispatches on a command string. A failed load raises `phasecont:engine:load`, and the next call tries again. The gateway never unloads `pckern`, since worker threads may still run its code. Loading the library at runtime keeps the gateway independent of the Rust toolchain (MSVC or MinGW on Windows).
```

Evidence: the spike's table starts with `abi_version` and `struct_size` on both sides (`spikes/pckern/src/lib.rs:28-30`, `spikes/mex/pckern_gateway.c:40-41`). `load_api` (`spikes/mex/pckern_gateway.c:122-171`) finds the MEX file's own module and path (`:135`, `:139`) and loads `pckern.dll` by full path (`:147`). It raises `phasecont:engine:load` when `pc_api_v1` returns NULL (`:158-163`) and when the version differs or `struct_size` is smaller than the gateway's table (`:164-167`), and it keeps the table only after every check passes (`:170`), so the next call tries the load again (`:133-134`). Only commands that need the table call it (`:481-482`, `:497-498`), and the comment at `:119-121` gives the reason it never unloads the DLL. Without `pckern.dll`, three calls in a row each tried the load again and failed with Windows error 126 (`s1_gateway_nodll.txt:20-22`), while `module_path` and `preload` still worked (`:23-26`). With it, the gateway loaded `pckern.dll` from the staged folder (`s1_ffi.txt:7-8`).

Not tested: `dlopen`, plain `LoadLibraryW`, the file lock, a current folder other than the MEX file's folder, and the load errors for a missing export, a NULL table and an ABI mismatch (S1, Numbers).

##### 4.16, `docs/SPEC.md:277`, the import check in the `cargo` task

Current text:

```text
- `cargo`: release build for the current platform.
```

Proposed, appended to the current text:

```text
For Windows the task lists the DLLs that `pckern.dll` imports and fails when one of them is not on an allow-list of DLLs that ship with Windows.
```

Evidence: `spikes/stage.sh:32-38` lists the imports with `x86_64-w64-mingw32-objdump -p` and stops only on names that match `libgcc`, `libwinpthread` or `libstdc`, so it would pass any other runtime DLL. Build `1792610350` imports only DLLs that ship with Windows (recommendation 1).

Not tested: an allow-list.

##### 4.17, `docs/SPEC.md:298`, `cuda` host code

Current text:

```text
  - Kernels are emitted by `pc-codegen` and compiled at runtime. The precision policy is rule 6 of the project rules.
```

Proposed, two new lines after the current text:

```text
  - `cuda` host code calls `is_culib_present` for the driver and NVRTC before any other `cudarc` call, because `cudarc` panics when it cannot load a library. It compiles kernels through the raw NVRTC calls and destroys the program on every path: when a compile fails and reading the program log fails too, `cudarc`'s `compile_ptx_with_opts` panics, and the `Drop` of its program panics again if destroying the program also fails. That second panic aborts the process, which `catch_unwind` cannot stop.
  - NVRTC contracts `a*b + c` into an FMA by default (`--fmad`, "Default: true" in `/usr/local/cuda-13.4/include/nvrtc.h:960-965`). A test that compares a GPU result with a CPU reference at the level of single ulps rounds each contracted operation once in the reference, and does not get that rounding from `f32::mul_add` on `x86_64-pc-windows-gnu` at the baseline target features.
```

Evidence for the first line: `cudarc` panics when it finds no library (`<registry>/cudarc-0.19.10/src/lib.rs:199-200`, `<registry>/cudarc-0.19.10/src/nvrtc/sys/mod.rs:550`) or no symbol (`<registry>/cudarc-0.19.10/src/driver/sys/mod.rs:11-12`), and `is_culib_present` only tries to open the library (`<registry>/cudarc-0.19.10/src/nvrtc/sys/mod.rs:529-538`, `<registry>/cudarc-0.19.10/src/driver/sys/mod.rs:16868`). `compile_ptx_with_opts` (`<registry>/cudarc-0.19.10/src/nvrtc/safe.rs:116-122`) unwraps the program log after a failed compile (`:151-152`), and the `Drop` of `Program` unwraps `destroy_program` during the unwind (`:168-173`). A panic in a `Drop` that runs during an unwind aborts the process (`panic_in_cleanup`, `core/src/panicking.rs:353-367` in the rust-src under `/usr/lib/rustlib/src/rust/library/`). The spike checks both libraries first (`spikes/pckern/src/s4_cuda.rs:58-65`), builds the cubin through the raw calls and destroys the program on every path (`:393-407`), but still compiles PTX with `compile_ptx_with_opts` (`:306-310`). No compile failed in the captures.

Evidence for the second line: `cudarc` adds `--fmad` to the NVRTC options only when the caller sets `fmad` or `use_fast_math` (`<registry>/cudarc-0.19.10/src/nvrtc/safe.rs:260-266`), and the spike sets only the architecture (`spikes/pckern/src/s4_cuda.rs:306-308`, `:412`). Against a reference rounded once, `max_ulp_fma` is 0 on all 16777216 elements (`s4_matlab.txt:16`, `:20`; `s4_matlab_preload.txt:20`, `:24`; all 12 values in `s4_wsl.txt`). The first run inside MATLAB used `f32::mul_add` for the reference and reported `max_ulp_fma` 2 (`s4_matlab_mingw_fmaf.txt:16`, `:20`), because on this target `mul_add` calls MinGW-w64's `fmaf` (`spikes/pckern/src/s4_cuda.rs:147-150`; S4, What went wrong).

Not tested: `--fmad=false`, a failed compile, and `f64::mul_add` on `x86_64-pc-windows-gnu`. The other abort in S4, a driver symbol that `cudarc` first resolves inside a `Drop` (`CudaEvent::drop`, `<registry>/cudarc-0.19.10/src/driver/safe/core.rs:540-546`, calls `<registry>/cudarc-0.19.10/src/driver/result.rs:1306-1308`, which calls `cuEventDestroy_v2` at `<registry>/cudarc-0.19.10/src/driver/sys/mod.rs:9943-9949`; S4, What went wrong), gets no text here, since the spike tested no fix for it.

## Decisions after review

The owner approved recommendations 1 to 4 on 2026-10-22, which accepts Phase 0. The items of recommendation 4 are applied to the project rules and `docs/SPEC.md`. The owner left the five open questions of `docs/STATUS.md` to the recommendations below, which are applied too.

- Default backend: the Cranelift JIT is the default on every platform. The AOT C path is used only when `mex` finds a compiler and the user selects it with `phasecont.config("Codegen","aot")`. It does not switch on by itself when a compiler is found.
- macOS arm64: rule 5 of the project rules holds, and the fallback in `docs/SPEC.md` section 4 changes. Phase 2 checks the entitlements of MATLAB on a Mac. Without the entitlements that the default memory provider or a `MAP_JIT` provider needs, models run on an `f64` evaluator in Rust built from the same code as the dual-number interpreter that section 3 already requires, and PhaseCont warns that it is slower. AOT through `mex` stays an optional speed-up on Macs with the Xcode Command Line Tools. This replaces item 4.4's last sentence, "With none of the three, the AOT fallback applies."
- File Exchange: the license stays Apache-2.0. The File Exchange licensing FAQ says that a submission linked to GitHub can use a license other than BSD (https://www.mathworks.com/matlabcentral/content/fx/fx-transition-faq.html). It does not list Apache-2.0 by name, so Phase 8a confirms it on the submission form.
- License notices in the `.mltbx`: the `package` task writes the license text of every crate linked into `pckern` into the toolbox. A Rust tool in this repo builds it from `cargo metadata --filter-platform` and each crate's license files (`docs/SPEC.md`, section 12). Installing `cargo-about` instead would need approval.
- Rule 8: a test runs `matlab.codetools.requiredFilesAndProducts` on every file under `toolbox/` and fails on any product other than MATLAB, with Parallel Computing Toolbox allowed only in the `gpuArray` route (`docs/SPEC.md`, section 12).
- Version lock: none. The Windows std requires `rust = 1.99.1-2.fc45`, so dnf cannot leave the two at different versions, and a lock would hold back Rust security updates. The `cargo` task compares `rustc -vV` with `rpm -q rust-std-static-x86_64-pc-windows-gnu` before a Windows build, which extends item 4.16. `docs/ENVIRONMENT.md` keeps the lock command for the case where the std lags behind `rust`.
