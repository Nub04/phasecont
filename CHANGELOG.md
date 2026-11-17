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
- Curves of fixed points of maps: `continuation` with Type `"LP"`, `"PD"` or `"NS"` and two free parameters follows the fixed points of a map with a multiplier 1, a multiplier -1 or a pair of multipliers on the unit circle. The Branch has `U` the fixed point, `Alpha` the two parameters and `Eigenvalues` the multipliers, an NS curve has `Kappa`, and `Bounds` bound the second free parameter. The curves label LPPD, LPNS, CP and R1 on LP curves; LPPD, PDNS, GPD and R2 on PD curves; and R1 to R4, LPNS, PDNS, NSNS and CH on NS curves, with NaN `Data.coefficient`. Sessions save and replay these runs.
- Period-k points of maps: `Iterate = k` on `equilibrium` and `continuation` works with the k-th iterate of the map, and `Branch.Iterate` holds k. `Switch = "PD1"` on an EP run of a map returns the branch at iterate k and the branch at iterate 2k that starts at the first PD point of the first branch, and a run with Type `"EP"` from a PD point at iterate k starts the branch at iterate 2k. LP, PD and NS curves run at the iterate of the special point they start from. Sessions save and replay runs at any iterate.
- Maps in the phase-plane app: on a map with two or more variables, `phasecont.app.PhasePlane` iterates from a click (the T field is labeled N), marks fixed points as stable or unstable from their multipliers and draws the curves where the map leaves a variable unchanged as its nullclines. On a map with one variable that does not use `t`, it opens a cobweb view. `phasecont.cobweb(m, x0, N)` draws the cobweb of a map with one variable in any axes. `phasecont:app:mapModel` now means that an action, such as a backward run or the direction field, does not apply to a map.
- Ensembles of maps: `phasecont.ensemble` iterates a map on the CPU and on CUDA, OpenCL and Metal, with `t` inside the model the iterate index. `TSPAN` holds whole numbers, `Precision` is the only required option, and `Dt`, `Rtol`, `Atol`, `Hmax` and `MaxSteps` raise `phasecont:ensemble:mapOption`. `Job.Method` is `"iter"`, and `refine` iterates again on the CPU in double precision. The self-test of `pc-cli gpu selftest` has an `iterate` tier, and its determinism and bit-exactness tiers also run the Henon map.
- Data fitting: `phasecont.fit` fits chosen parameters of an ODE model to trajectory data with the Levenberg-Marquardt method, with one `ensemble` call per iteration in double precision on the CPU, and returns a `phasecont.FitResult` with the estimate, the residual, the Jacobian and the history of the iterations. The difference step, the damping and the stopping rules have no default.
- `Point.Stable`: an `"EP"` Point from `equilibrium` or a session says whether the equilibrium or the fixed point is stable, from its eigenvalues or multipliers. It is empty at a special point such as LP, PD or H, where the linear test decides nothing. It is not stored, so session files do not change.
- A language server (`pc-lsp`) and a VS Code extension for `.ode` files. The extension is built as a `.vsix` file and is not on the Marketplace.

### Security

- Windows: the GPU backends never load a library from MATLAB's current folder. cudarc and the OpenCL loader open `cuda.dll`, `nvrtc.dll`, `OpenCL.dll` and similar names by bare name, and Windows searches the current folder for those, so before this change a file with one of those names in the current folder ran inside MATLAB at the first call of `phasecont.gpuDevices`, `phasecont.gpuSelfTest` or `phasecont.ensemble`. The first loads now skip the current folder, the CUDA driver and the OpenCL loader come from the Windows system folder, and the NVRTC builtins library is loaded by path from the folder of the NVRTC in use. The folders on `PATH` are still searched, as for any Windows program. Release 1.0.0 has no GPU backend and is not affected.
- Windows: `phasecont.latex.build` and the LaTeX editor app never run a `latexmk` from MATLAB's current folder. A bare program name went to cmd.exe, which looks in the current folder before `PATH`, so a `latexmk.bat` in a downloaded folder ran on the first build. The name is now looked up in the `PATH` folders only and runs by its full path. Release 1.0.0 has no LaTeX build and is not affected.
- `Session.load` and `phasecont.session.replay` no longer build an AOT library because a session file records `"aot"`. Before this change, loading such a file ran `mex` on C code generated from the model source in the file. The C text comes from our own code generator, but a session file from someone else should not start a compiler on its own. Such a file now loads with the evaluator that `phasecont.config("Codegen")` names (`"jit"` unless you changed it), and the `Evaluator` option still picks one. A file that records any evaluator other than `"jit"`, `"fallback"` or `"aot"` raises `phasecont:session:badFile`. Release 1.0.0 is affected.

### Changed

- Single-precision CPU ensembles run a compiled right-hand side, as double precision already did. `vdp-rk4` at 100000 members runs about 1.9 times faster, with the same bits as before. Models that use tables or erf, erfc, lgamma, the Bessel functions or polygamma still run on the interpreter in single precision.
- `simulate` warns once per model with `phasecont:simulate:ignoredOption` and names the `@` options of the model, such as `meth` and `dt`, since it uses none of them. The results do not change. The warning comes after the run, so a run that fails gives no warning.
- `simulate` iterates a map, and `equilibrium` and `continuation` with Types `"EP"`, `"LP"`, `"PD"` and `"NS"` compute its fixed points, their multipliers and the curves of its LP, PD and NS points. In 1.0.0 they treated the right-hand side of a map as an ODE and gave wrong results. Types `"H"`, `"LC"`, `"LPC"` and `"HOM"` on a map now raise a `phasecont:` error. `phasecont.app.PhasePlane` shows a map instead of a vector field (see Added). `Model.rhs` still evaluates the right-hand side.
- A file of `x'=...` lines with `@ meth=discrete` (any `meth` value that starts with `d`) is now a map, as in XPPAUT, so it runs as a map in every call above. In 1.0.0 it was integrated as an ODE, and `ensemble` integrated it too.
- `continuation`, `simulate` and session replay check for the result more often in the first few milliseconds, so short runs return in about 5 ms instead of about 50 ms. The results do not change.
- Rows of LPC, PD and NS curves have more `Tests` columns, one per new codim-2 test. Code that reads these columns by position needs the new widths.
- `shift` and `del_shft` are now refused with a message that names the function, instead of an error about the number of arguments.

### Fixed

- Models built with the AOT evaluator run LPC, PD and NS curves. The AOT library had no `F_up` entry, which those curves use, so they stopped with "AOT library has no entry d_up".
- Maps built with the AOT evaluator run LP, PD and NS curves at `Iterate` 2 or more. The k-th iterate asks for the directional derivative of the whole map, which the AOT library did not hold, so those curves stopped with "AOT library has no entry d_f". Adding the entry changes the tag of every AOT library, so a cached library is built again on first use.

### Compatibility

- Session files saved by 1.0.0 still load. Their LPC, PD and NS rows get NaN in the new test columns. Replaying such a run reports that the row layout changed.
- A session saved by 1.0.0 with a run on a map loads with its stored rows, which are wrong. An EP branch of a map from such a file gets NaN in its PD and NS columns, since 1.0.0 stored an H test there. Replaying any of these runs gives `Pass = false`, since 1.0.0 treated the map as an ODE. A run that the engine refuses, such as an integration of a map or an EP branch that asks for the H test, also warns with `phasecont:session:runFailed`.
- The engine library and the MEX gateway must come from the same release. A new gateway refuses an old library, which can happen only after a partial upgrade.
- As in 1.0.0, close every MATLAB session that used PhaseCont before you install a new version, or the install fails with "Permission denied".

### Known limits

- Maps (`x(t+1)=...`, or `x'=...` with `@ meth=discrete`) run only in `simulate`, `equilibrium`, `continuation` with Types `"EP"`, `"LP"`, `"PD"` and `"NS"`, `PhasePlane`, `cobweb` and `ensemble`. Types `"H"`, `"LC"`, `"LPC"` and `"HOM"` refuse them, so a cycle of a map is followed as a fixed point of its k-th iterate with `Iterate`. `PhasePlane` iterates a map forward only. The codim-2 points of map curves have no normal form coefficients, and the least period of a period-k point is not checked. `fit` refuses them with `phasecont:fit:mapModel`.
- Three bounds in the continuation of map curves have no published source: a start must lie within 10^-6 of its curve, a CH sign change counts only above 10^-8, and the CH test divides by c1 and checks only for an exact zero. The `Switch = "PD1"` option and a start at a PD point of a map, and `Switch = "BP1"` on a map or an ODE, refuse a point whose two smallest singular values lie within a factor of 10^3 of each other as not simple, and that factor has no published source either.
- Delay models run only in `continuation`. `simulate`, `equilibrium` and `ensemble` refuse them.
- The `.ode` reader refuses noise, `global`, differential-algebraic and Volterra equations, boundary value problems, `sum`, `shift`, `special`, `export`, tables read from a file and `#include`. The README lists them.
- `simulate` integrates only with the Dormand-Prince 5(4) pair.
- The classification of Hopf-Hopf points into unfolding cases is not yet checked against a published table, and help marks the case letters as unconfirmed.
- GPU ensembles have run on an NVIDIA RTX 5060 Laptop GPU (CUDA and OpenCL, single and double precision) and an Intel Arc 140T (OpenCL, single and double precision). Metal has run only on the virtual GPU of GitHub's hosted macOS runners, in single precision, never on Apple hardware. No AMD device has run.

## 1.0.0 (2026-11-04)

First public release.
