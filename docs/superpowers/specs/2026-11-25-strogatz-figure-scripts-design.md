# Strogatz figure scripts (sub-project A, part 4b): design

Date: 2026-11-25. State: decisions D1 to D5 ruled on 2026-11-25 (see "Decisions"). Slice 0 plan next; no code yet.

## Goal

On 2026-11-17 the owner asked that PhaseCont can make all the graphs in the books before any release. This part covers Strogatz, *Nonlinear Dynamics and Chaos*, 3rd edition. For every figure that today's PhaseCont can compute, a script in this repo makes it from our own `.ode` file, and a test checks the result against an independent calculation. Figures that need a missing feature go on a gap list for the owner. This part builds no new API.

## What the triage found

The triage read every figure in the book (PDF on the owner's machine, never copied into the repo) and sorted the 229 figures that rest on a model:

| Class | Rows |
|---|---|
| Covered by today's API | 153 (127 computed, 25 sketch, 1 schematic) |
| Gap, needs a feature PhaseCont lacks | 64 |
| n/a, nothing to compute | 12 |

The covered rows per chapter: ch1 1, ch2 13, ch3 31, ch4 1, ch5 8, ch6 20, ch7 16, ch8 23, ch9 12, ch10 18, ch12 10.

The covered rows name 146 distinct model strings. Grouped by hand, they come to about 65 to 75 model families, for example the Lorenz equations, the logistic flow and map, the saddle-node, transcritical and pitchfork normal forms, the imperfect bifurcation, the budworm model, the bead on a rotating hoop, the harmonic, damped and van der Pol oscillators, the rabbits and sheep model, the pendulum, the glycolysis model, the genetic switch, the chemical oscillator, the driven pendulum, the Rossler equations, the Henon, baker, tent and sine maps and the forced Duffing equation.

The API calls the covered rows need, counted over the rows (a row can need several): `simulate` 67, `PhasePlane` 58, `continuation` 26, `equilibrium` 24, `cobweb` 13, `ensemble` 6, `tdplot` 1. 22 rows also need some MATLAB glue (see the glue rule).

The 153 rows are an upper bound. A row counts as done only when its script runs and its check passes.

## Rules

1. **Done.** A row is done when its script runs from a clean MATLAB session and its check passes in the test class. A row that turns out to need a missing feature moves to the gap list with the reason. No workarounds for blocked rows.
2. **Glue.** A script may plot and do array arithmetic on what PhaseCont returns (a mean, a difference, `interp1` on a returned trajectory, a `surf` over returned points). A script that iterates a map, integrates, or finds roots in MATLAB is doing PhaseCont's job, so that row goes on the gap list. Tests are different: they may use base MATLAB `fzero`, `eig` and `ode45` as independent oracles.
3. **Models.** One `.ode` file per model family, written by us from the equations. Each figure that uses the model is a case of it: a parameter set, a start, a window. No XPPAUT example files and no files from Ermentrout's book.
4. **Values.** Each row records its parameter values, starts and windows, and whether each is stated in the book or chosen by us. A chosen value is one the book does not print (for example a start point read off a sketch). The owner sees the list of chosen values (decision D2).
5. **No book text.** The manifest and the scripts hold figure numbers, our own model files, values and checks. They hold no captions, no book text and no printed data tables.
6. **Rule 9.** Every closed form a check uses is derived in the test comment from the model equations, or the check uses an independent numeric oracle. A value that cannot be confirmed either way stops the row and goes to the owner.

## Layout

Ruled (decision D1): a top-level folder that the `.mltbx` does not ship.

```
books/strogatz/
  models/          our .ode files, one per model family
  figures/         one script per figure, fig_<chapter>_<section>_<n>.m
  manifest.json    one entry per covered row
tests/StrogatzFigureTest.m
```

`buildfile.m` packages `toolbox/` only, and `toolbox/examples/` ships, so `books/` stays out of the toolbox. A later option is to ship a small subset (one script per API family) under `toolbox/examples/`.

## Figure scripts

Each script is a function:

```matlab
out = fig_9_3_2(Plot = true)
```

- With `Plot = true` it draws the figure with PhaseCont's plotting (`Trajectory.plot`, `Branch.plot`, `cobweb`, the PhasePlane app) plus plain MATLAB axes calls.
- With `Plot = false` it draws nothing and opens no app window.
- `out` is a struct of what the check needs: the `Trajectory`, `Branch` or `Point` objects, or for app rows the app handle built with `Visible = "off"`.
- A PhasePlane row drives the app through its public methods (`addTrajectory`, `findEquilibrium`, `setParameter`, `setWindow`, `showNullclines`, `showField`), so the test exercises the app itself.

## Manifest

`manifest.json`, read with base MATLAB `jsondecode`. One entry per covered row:

| Field | Meaning |
|---|---|
| `figure` | figure number, for example `"9.3.2"` |
| `edition` | `3` |
| `model` | file name under `models/` |
| `script` | function name under `figures/` |
| `api` | PhaseCont calls used, for example `["simulate", "Trajectory.plot"]` |
| `values` | parameters, starts and windows used |
| `source` | `"stated"` or `"chosen"` for each entry in `values` |
| `check` | the check kind and its oracle, for example `"ode45 at t = 1, RelTol 1e-10"` |
| `state` | `"done"` or `"blocked"` (with the item that blocks it) |

The gap and n/a rows go in a second file, `gaps.json`, with the same `figure` field and a reason. A covered row that turns out to need a missing feature leaves `manifest.json` and moves to `gaps.json`. The triage files (local only, in `~/phasecont-wip/strogatz-triage-2026-11-25/`) are the source for both and are not committed.

## Checks

- **Computed rows** are checked against an independent calculation: `ode45` with tight tolerances over a short horizon for trajectories (short for chaotic systems), `fzero` or a closed form for equilibria, folds, Hopf points and period-doubling points, `eig` of a Jacobian built by central differences on `Model.rhs` for stability types.
- **Sketch rows** check the qualitative content: the number and types of equilibria in the window and the sign of `f` between them for 1-D flows.
- **The schematic row** (5.1.4) checks the orbit only. The spring drawings are not producible and are noted as such.

## Test class

`tests/StrogatzFigureTest.m`, one parameterized test over the manifest entries with `state == "done"`.

- It calls each script with `Plot = false`, then runs the row's check.
- A teardown closes any figure and deletes any app the script made.
- It skips when `pckern` is not built, as `ExampleTest` does.
- Slice 0 measures the time per row on this machine and on CI. From those numbers the class gets a total time budget, which the owner signs. Times are logged per row, and only the total is checked, so one slow CI runner does not fail a single row.
- `AppTest` already runs on CI with `Visible = "off"`, so the PhasePlane rows run on the CI MATLAB job on all three operating systems.

## Slices

**Slice 0** takes one row per API family to measure how many covered rows really work and what each costs:

| Family | Row | Check |
|---|---|---|
| `simulate` | 9.3.2 (Lorenz, x against z) | `ode45` over a short horizon |
| `PhasePlane` | 6.4.7 (rabbits and sheep) | four equilibria, types from `eig` |
| `equilibrium` | 2.2.6 (x' = x - cos x) | `fzero` |
| EP + branch point + switch | 3.4.2 (pitchfork) | branch point at r = 0, branches x = ±sqrt(r) |
| LP curve with cusp | 3.6.2 (imperfect bifurcation) | fold curve closed form, cusp at the origin |
| Hopf curve | 8.3.4 (chemical oscillator) | Hopf locus closed form |
| LC from Hopf | 9.5.1 (Lorenz) | r_H closed form; cycle checked with `ode45` |
| HOM | 8.4.3 | mu_c from an `ode45` shooting oracle |
| `cobweb` | 10.1.3 (cosine map) | fixed point by `fzero` |
| map continuation | 10.3.4 (logistic map) | fold or branch point at r = 1, period doubling at r = 3 |
| map `ensemble` | 12.2.2 (Henon map) | image of the start set by the map formula as array arithmetic |
| `tdplot` | 5.2.8 | region boundaries tau^2 = 4 Delta, Delta = 0, and tau = 0 for Delta > 0 |

Slice 0 ends with a short report for the owner: rows done, rows moved to the gap list and why, time per row. Then the work goes chapter by chapter, smallest open chapter first, each chapter one local commit with its rows, models and manifest entries.

## Known blocks and deviations

- **2.5.1** (x' = x^(1/3)) is blocked by the fractional-power item in STATUS: at the singular point the DAG and the JIT can give NaN where the fallback gives a finite value. The row waits for that fix.
- **`x*y<=0`.** PhaseCont reads it as `x*(y<=0)`, as XPPAUT does, and warns. Our model files write the parentheses they mean, so this is not a block (decision D5).

## Out of scope

- The 64 gap rows. By reason: 1-D phase line plot 10, fractal set construction 9, angle-valued state on a cylinder or torus 7, Poincare map 5, plot on the circle, cylinder or torus 5, potential or energy surface 4, potential plot V(x) 3, phase plot on the circle 3, period or time-average sweep 3, orbit diagram 3, Lyapunov exponent 2, correlation dimension 2, and one each of a model-term plot, a Lorenz map of successive maxima, a global unstable manifold, an escape-time image, a closed-curve vector plot and a basin of attraction. The two Kuramoto rows (13.2.3, 13.2.4) are on the list too, because the `.ode` reader has no `sum(..)of(..)` (decision D4). Building any of these is a feature with its own spec.
- Kuznetsov and Ermentrout. They follow the same design once the owner supplies the PDFs.
- Changes to the public API, the C ABI or a dependency.

## Decisions

The owner delegated D1 to D5 to the advisor on 2026-11-25. The rulings below stand until the owner changes them.

- **D1. Location.** Ruled: `books/strogatz/` at the repo root, not shipped. A subset under `toolbox/examples/` stays a later option.
- **D2. Chosen values.** Ruled: accept chosen values, each marked `"chosen"` in the manifest. 45 rows use at least one value the book does not print: 1.1.1, 2.4.1, 2.5.1, 3.1.3, 3.4.8, 3.5.8, 4.3.6, 5.1.2, 5.3.1, 6.3.2, 6.7.7, 7.1.3, 7.3.4, 7.5.1, 7.6.1, 8.1.1, 8.1.6, 8.2.3, 8.2.5, 8.3.5, 8.4.2, 8.5.1, 8.7.3, 9.2.5, 9.4.1, 9.5.1, 9.5.2, 9.5.4, 10.1.2, 10.2.1, 10.2.2, 10.3.1, 10.3.2, 10.3.4, 10.4.3, 10.4.4, 10.5.1, 10.6.1, 12.1.4, 12.1.5, 12.2.2, 12.2.3, 12.5.3, 12.5.5, 12.5.8. Three of them (9.5.1, 10.3.4, 12.2.2) are slice 0 rows, so the slice 0 report lists every chosen value it used.
- **D3. The 12 n/a rows** (3.5.7, 4.2.1, 4.5.2, 6.7.6, 9.2.3, 9.6.2, 10.7.2, 10.7.3, 11.5.4, 12.2.1, 13.2.5, 13.5.3) are cartoons, measured data or diagrams without numbers. Ruled: list them in `gaps.json` as n/a and build nothing.
- **D4. Kuramoto.** Ruled: 13.2.3 and 13.2.4 go on the gap list until the `.ode` reader has `sum(..)of(..)`. A model text generated in MATLAB is not a model file (rule 3), and the cost of N = 1000 states was not measured.
- **D5. `x*y<=0`.** Ruled: the XPPAUT reading is correct, and our model files write the parentheses they mean. Making it a parse error is a behavior change and not part of this work.
