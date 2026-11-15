# Changelog

## Unreleased

The version number of this release is not chosen yet. The toolbox still reports 1.0.0.

### Added

- Homoclinic curves: `phasecont.continuation(..., Type = "HOM")` follows the curve of orbits homoclinic to a saddle in two parameters, starting from a point of long period on an LC branch, and labels its codim-2 points. The inclination flips IFS and IFU are checked only with `InclinationFlips = true`, because they cost an adjoint solve at every point.
- Codim-2 points on curves of cycles: LPPD, LPNS, CPC and R1 on LPC curves, LPPD, PDNS, GPD and R2 on PD curves, and R1 to R4, LPNS, PDNS and CH on NS curves.
- Delay models. A model that calls `delay(x, tau)` runs in `continuation`:
  - equilibria in one parameter, with the roots of the characteristic equation (`Roots`) and LP, BP and H labels;
  - Hopf curves in two parameters, with BT, ZH and HH labels, GH with `Lyapunov = true`, and normal form coefficients at ZH and HH points with `Coefficients = true`;
  - cycles from a Hopf point, with Floquet multipliers (`Floquet = true`), LPC, PD and NS labels, BPC with `Detect = "BPC"`, and a switch onto the other branch at a BPC point (`Switch = "BPC1"`).
- Mesh adaptation of cycles with the `MeshAdapt` option. It is 3 for ODE cycles and 0 for delay cycles unless you pass it.
- Ensembles: `phasecont.ensemble` integrates many parameter sets and initial states at once on the CPU or on a GPU through CUDA, OpenCL or Metal. `Precision`, the time span and the tolerances (or `Dt`) have no default, single-precision results are labeled as such, and `refine` recomputes chosen members in double precision. `phasecont.gpuDevices` lists the devices and `phasecont.gpuSelfTest` checks one of them.
- Models without a `.ode` file: `Model.fromEquations`, `Model.fromTable`, `Model.fromFunction` and `Model.fromFile`, which also reads SBML. The model editor app is `phasecont.app.ModelEditor`.
- LaTeX output: `phasecont.latex.pgfplots`, `phasecont.latex.figureFromBranch`, `phasecont.latex.specialTable` and `phasecont.latex.build`, which runs `latexmk`. `build` has no time limit unless you pass `Timeout`. The LaTeX editor app is `phasecont.app.LatexStudio`.
- Iteration of maps: `simulate(m, x0, [n0 n1])` on a map iterates it from n0 to n1, and `plot` draws the iterates as dots. `Model.IsMap` tells a map from an ODE. Sessions save and replay these runs.
- Fixed points of maps: `equilibrium` on a map solves F(x) = x and returns the multipliers in `Eigenvalues`, and `continuation` with Type `"EP"` follows the fixed points in one parameter with LP, BP, PD and NS labels and their normal form coefficients. On a map, `Tests` has the four columns `[LP BP PD NS]`, `Detect` takes a subset of `["LP" "BP" "PD" "NS"]`, `Stable` is true where every multiplier has modulus below 1, and an NS point has its angle in `Data.values.theta`. Sessions save and replay these runs.
- `Point.Stable`: an `"EP"` Point from `equilibrium` or a session says whether the equilibrium or the fixed point is stable, from its eigenvalues or multipliers. It is empty at a special point such as LP, PD or H, where the linear test decides nothing. It is not stored, so session files do not change.
- A language server (`pc-lsp`) and a VS Code extension for `.ode` files. The extension is built as a `.vsix` file and is not on the Marketplace.

### Changed

- `simulate` warns once per model with `phasecont:simulate:ignoredOption` and names the `@` options of the model, such as `meth` and `dt`, since it uses none of them. The results do not change. The warning comes after the run, so a run that fails gives no warning.
- `simulate` iterates a map, and `equilibrium` and `continuation` with Type `"EP"` compute its fixed points and their multipliers. In 1.0.0 they treated the right-hand side of a map as an ODE and gave wrong results. Other types of `continuation` on a map now raise a `phasecont:` error, `ensemble` raises `phasecont:engine:unsupportedModel`, and `phasecont.app.PhasePlane` raises `phasecont:app:mapModel`. `Model.rhs` still evaluates the right-hand side.
- A file of `x'=...` lines with `@ meth=discrete` (any `meth` value that starts with `d`) is now a map, as in XPPAUT, so it runs as a map in every call above. In 1.0.0 it was integrated as an ODE, and `ensemble` integrated it too.
- `continuation`, `simulate` and session replay check for the result more often in the first few milliseconds, so short runs return in about 5 ms instead of about 50 ms. The results do not change.
- Rows of LPC, PD and NS curves have more `Tests` columns, one per new codim-2 test. Code that reads these columns by position needs the new widths.
- `shift` and `del_shft` are now refused with a message that names the function, instead of an error about the number of arguments.

### Compatibility

- Session files saved by 1.0.0 still load. Their LPC, PD and NS rows get NaN in the new test columns. Replaying such a run reports that the row layout changed.
- A session saved by 1.0.0 with a run on a map loads with its stored rows, which are wrong. An EP branch of a map from such a file gets NaN in its PD and NS columns, since 1.0.0 stored an H test there. Replaying any of these runs gives `Pass = false`, since 1.0.0 treated the map as an ODE. A run that the engine refuses, such as an integration of a map or an EP branch that asks for the H test, also warns with `phasecont:session:runFailed`.
- The engine library and the MEX gateway must come from the same release. A new gateway refuses an old library, which can happen only after a partial upgrade.
- As in 1.0.0, close every MATLAB session that used PhaseCont before you install a new version, or the install fails with "Permission denied".

### Known limits

- Maps (`x(t+1)=...`, or `x'=...` with `@ meth=discrete`) run only in `simulate`, `equilibrium` and `continuation` with Type `"EP"`. Other types of `continuation`, `ensemble` and `PhasePlane` refuse them. Curves of fixed points in two parameters and period-k points are planned.
- Delay models run only in `continuation`. `simulate`, `equilibrium` and `ensemble` refuse them.
- The `.ode` reader refuses noise, `global`, differential-algebraic and Volterra equations, boundary value problems, `sum`, `shift`, `special`, `export`, tables read from a file and `#include`. The README lists them.
- `simulate` integrates only with the Dormand-Prince 5(4) pair.
- The classification of Hopf-Hopf points into unfolding cases is not yet checked against a published table, and help marks the case letters as unconfirmed.
- GPU ensembles have run on an NVIDIA RTX 5060 Laptop GPU (CUDA and OpenCL, single and double precision) and an Intel Arc 140T (OpenCL, single and double precision). Metal has run only on the virtual GPU of GitHub's hosted macOS runners, in single precision, never on Apple hardware. No AMD device has run.

## 1.0.0 (2026-11-04)

First public release.
