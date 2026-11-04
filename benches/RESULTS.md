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
