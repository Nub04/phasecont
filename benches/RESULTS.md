- A kept kernel saves 12 to 13 ms per run on CUDA at N = 1000 and N = 10000, and 5 to 7 ms on OpenCL on the RTX 5060. That matches the 11 ms NVRTC compile and the 6 ms estimate. At N = 100000 the saving is 19 ms on CUDA and 11 ms on OpenCL. I did not isolate why the CUDA saving is larger there than the compile alone.
- On the Arc 140T the saving is 0.4 ms at N = 1000 and 0.4 to 8.6 ms at the larger sizes. I did not measure the spread between runs, so the larger numbers may be noise. The Arc driver builds a program in 0.3 to 0.7 ms, so there was little to save.
# Speed against XPPAUT

These numbers compare `pc-cli cont` on an IVP problem with XPPAUT 8.0 run headless on the same model. They measure integration only. XPPAUT's AUTO cannot run headless, so continuation is not compared.

## Setup

- Machine: Intel Core Ultra 7 255H, 12 threads, WSL2 (Fedora).
- pc-cli: release build, commit 2e4e8c9, Dormand-Prince 5(4) with rtol 1e-8 and atol 1e-10.
- XPPAUT: `xppaut model.ode -silent -outfile out.dat`, with the line `@ meth=qualrk, toler=1e-8, atoler=1e-10, total=T, dt=0.5, nout=1, maxstor=4000000, bounds=1e8` before `done`. A second set of runs uses `meth=cvode`.
- `meth=dorprince` is not used, because XPPAUT 8.0 ignores `total` with it when run headless.
- Time is wall clock for the whole process, including start and output, median of 7 runs.
- pc-cli prints every accepted step as JSON. XPPAUT writes one row per `dt`. The output sizes differ.

## Models

Both programs read the same equations, written in each program's syntax. They are our own models, not XPPAUT examples.

- Morris-Lecar, 2 states, T = 30000.
- Chain of 40 Morris-Lecar cells with diffusive coupling, 80 states, T = 3000.
- Chain of 200 cells, 400 states, T = 300.

## Results (ms)

| Model | pc-cli | XPPAUT qualrk | XPPAUT cvode |
|---|---|---|---|
| Morris-Lecar, T = 30000 | 74 | 108 | 107 |
| 40-cell chain, T = 3000 | 71 | 485 | 700 |
| 200-cell chain, T = 300 | 74 | 262 | 3655 |
| Startup only, T = 1 | 4 | 2 | |

The Morris-Lecar end point agrees: pc-cli gives v = -35.950125, w = 0.13281562 and XPPAUT prints -35.950127, 0.13281561.

## Notes

- Before commit 2e4e8c9, pc-cli took 235 ms on Morris-Lecar. Most of that was building a `serde_json::Value` for each of about 51000 points. The result is now written straight into one buffer, and the output is byte-identical.
- Startup is about 2 ms slower than XPPAUT.
- These are single runs on one machine and one build. Treat them as a rough comparison.

# GPU ensemble

Three measurements on 2026-11-08, all through a Windows `pc-cli.exe` (cross-built for `x86_64-pc-windows-gnu`) started from WSL:

```sh
pc-cli ensemble --bench
```

- Before: commit 7d09cc2, where every run creates and drops its own GPU context.
- After: commit 5cf1e4c, where the CUDA and OpenCL backends keep one context per device between runs in a process. The bench rows also report `first_seconds`, the first repeat.
- After, kernel kept: commit 80d82b1, where each kept device entry also keeps up to 8 compiled kernels. The cache is keyed by the kernel text, the compute capability or build options, and the precision, so a repeated run of the same model and precision skips the compile.

Setup, the same for all three:

- Machine: Intel Core Ultra 7 255H laptop, CPU backend on 16 rayon threads.
- NVIDIA: RTX 5060 Laptop GPU, CUDA driver 13.5, NVRTC 13.5, OpenCL driver 621.14.
- Intel: Arc 140T, OpenCL 3.0 NEO, driver 32.0.101.9150.
- Model: van der Pol, mu = 1.5, RK4 with dt 0.01 from t = 0 to t = 10, so each member takes 1000 steps.
- Time: the fastest of 3 repeats, including the host work around the launches. Each cell is wall time in milliseconds for one run.
- The Metal backend is not built on this machine, and no AMD or Apple device has run.

## After: the context is kept

Single precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 6.2 | 55.0 | 559.7 |
| CUDA, RTX 5060 | 12.5 | 19.9 | 88.5 |
| OpenCL, RTX 5060 | 8.3 | 14.0 | 78.7 |
| OpenCL, Arc 140T | 2.4 | 18.8 | 191.2 |

Double precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 5.6 | 51.9 | 525.5 |
| CUDA, RTX 5060 | 13.9 | 20.5 | 99.3 |
| OpenCL, RTX 5060 | 7.5 | 14.3 | 86.7 |
| OpenCL, Arc 140T | 4.5 | 22.8 | 232.8 |

The first run of a device in the process, which is the first single-precision row at N = 1000 and also creates the context: 126 ms on CUDA for the RTX 5060, 160 ms on OpenCL for the RTX 5060 and 9 ms on the Arc 140T. Every later row on that device and backend reuses the context, and its first repeat is within 0 to 7 ms of its fastest one.

What the numbers say:

- The RTX 5060 now beats the CPU from N = 10000 on. It is 2.5 to 3.9 times faster at N = 10000 and 5.3 to 7.1 times faster at N = 100000, at roughly 1000 to 1270 million member-steps per second at N = 100000 against 180 to 190 million for the CPU. At N = 1000 it is still slower: OpenCL takes 1.3 times as long as the CPU and CUDA 2.0 to 2.5 times as long.
- The Arc 140T is faster than the CPU at all three sizes, by 1.2 to 2.6 times at N = 1000, 2.3 to 2.9 times at N = 10000 and 2.3 to 2.9 times at N = 100000. Keeping the context changed it little, since it creates one in under 1 ms.
- Double precision on the RTX 5060 is within 12 percent of single (OpenCL at N = 1000 was 9 percent faster in double, which is noise). On the Arc 140T it takes 21 to 87 percent longer, depending on N.
- What is left of the fixed cost on the RTX 5060 is the kernel compile that every run still does: 11 ms in NVRTC for CUDA, and for OpenCL on the RTX 5060 about 6 ms in the program build, which is an estimate (8.3 ms at N = 1000 less 2 to 3 ms to launch and copy) and was not timed separately. That is most of the 12.5 and 8.3 ms at N = 1000. Where the break-even sits between 1000 and 10000 members was not searched.
- These are single runs on one laptop. Treat them as a rough comparison.

## Before: every run creates its own context

Single precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 5.7 | 54.5 | 556 |
| CUDA, RTX 5060 | 104 | 112 | 162 |
| OpenCL, RTX 5060 | 91 | 101 | 154 |
| OpenCL, Arc 140T | 3.9 | 20.1 | 188 |

Double precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 6.1 | 52.4 | 538 |
| CUDA, RTX 5060 | 94 | 106 | 172 |
| OpenCL, RTX 5060 | 87 | 93 | 158 |
| OpenCL, Arc 140T | 5.5 | 25.5 | 227 |

The RTX 5060 was slower than the CPU at N = 1000 and N = 10000 and its time barely changed from 1000 to 10000 members, which pointed at a fixed cost per run. Timings printed from a throwaway build (not committed) for N = 1000 gave, per run: 65 to 76 ms to create the context (CUDA and OpenCL on the RTX 5060), 11 ms to compile the kernel (36 to 47 ms on some runs), under 1 ms to load it and 2 to 3 ms to launch and copy. On the Arc 140T, creating the context took 0.2 to 0.7 ms and building the program 0.3 to 0.7 ms.

## After: the kernel is kept too

Wall time in milliseconds, the fastest of 3 repeats. The kernel is compiled on the first repeat of a model and precision, so these numbers leave the compile out. The previous run (context kept, kernel compiled every run) is in the second number of each cell.

Single precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 5.9 | 55.5 | 567.7 |
| CUDA, RTX 5060 | 0.9 (12.5) | 6.7 (19.9) | 69.6 (88.5) |
| OpenCL, RTX 5060 | 1.7 (8.3) | 6.9 (14.0) | 67.6 (78.7) |
| OpenCL, Arc 140T | 2.0 (2.4) | 18.4 (18.8) | 187.7 (191.2) |

Double precision:

| Backend and device | N = 1000 | N = 10000 | N = 100000 |
|---|---|---|---|
| CPU (rayon) | 6.0 | 52.7 | 539.5 |
| CUDA, RTX 5060 | 1.8 (13.9) | 8.2 (20.5) | 80.3 (99.3) |
| OpenCL, RTX 5060 | 2.2 (7.5) | 9.2 (14.3) | 75.3 (86.7) |
| OpenCL, Arc 140T | 4.1 (4.5) | 22.3 (22.8) | 224.2 (232.8) |

- A kept kernel saves 11 to 13 ms per run on CUDA at N = 1000 and N = 10000, and 6 to 7 ms on OpenCL on the RTX 5060. That matches the 11 ms NVRTC compile and the 6 ms estimate. At N = 100000 the saving is 19 ms on CUDA and 11 to 12 ms on OpenCL. I did not isolate why the CUDA saving is larger there than the compile alone.
- On the Arc 140T the saving is 0.4 to 0.5 ms at N = 1000 and 0.4 to 8 ms at the larger sizes, which is within the repeat noise at the large sizes. The Arc driver builds a program in 0.3 to 0.7 ms, so there was little to save.
- The RTX 5060 now beats the 16-thread CPU at N = 1000 as well: 0.9 ms against 5.9 ms on CUDA in single precision, 1.8 ms against 6.0 ms in double.
- A new model or precision still pays for one compile on its first run. For the double-precision rows at N = 1000, the first repeat took 15.4 ms on CUDA and 7.4 ms on OpenCL for the RTX 5060, against 1.8 ms and 2.2 ms for the fastest. The very first single-precision run in the process also creates the context: 163 ms on CUDA and 112 ms on OpenCL for the RTX 5060.
