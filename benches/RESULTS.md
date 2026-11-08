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

Measured on 2026-11-08 with commit 7d09cc2, using a Windows `pc-cli.exe` (cross-built for `x86_64-pc-windows-gnu`) started from WSL:

```sh
pc-cli ensemble --bench
```

- Machine: Intel Core Ultra 7 255H laptop, CPU backend on 16 rayon threads.
- NVIDIA: RTX 5060 Laptop GPU, CUDA driver 13.5, NVRTC 13.5, OpenCL driver 621.14.
- Intel: Arc 140T, OpenCL 3.0 NEO, driver 32.0.101.9150.
- Model: van der Pol, mu = 1.5, RK4 with dt 0.01 from t = 0 to t = 10, so each member takes 1000 steps.
- Time: the fastest of 3 repeats, including the host work around the launches. Each cell is wall time in milliseconds for one run.
- The Metal backend is not built on this machine, and no AMD or Apple device has run.

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

What the numbers say:

- The RTX 5060 is slower than the CPU at N = 1000 and N = 10000, in both precisions and on both backends. Its time barely changes from 1000 to 10000 members (about 90 to 100 ms to about 100 to 110 ms), so a fixed cost of about 90 ms per run dominates. At N = 100000 it is 3.1 to 3.6 times faster than the CPU, at roughly 580 to 650 million member-steps per second against about 180 million for the CPU.
- The Arc 140T has almost no fixed cost. It is faster than the CPU at all three sizes, by 1.1 to 1.5 times at N = 1000, 2.1 to 2.7 times at N = 10000 and 2.4 to 3.0 times at N = 100000.
- Double precision on the RTX 5060 costs about the same as single. On the Arc 140T it is 20 to 40 percent slower, depending on N.
- Where the break-even sits for the RTX 5060 is between 10000 and 100000 members. This run did not search for it.
- The cause of the fixed cost on the RTX 5060 is not investigated. Kernel compilation on every run, context setup or something else would all fit these numbers.
- These are single runs on one laptop. Treat them as a rough comparison.
