# PhaseCont

PhaseCont (working name) is a MATLAB toolbox for phase-plane and bifurcation analysis of ODE models. It reads XPP/XPPAUT `.ode` files and runs its own continuation engine, written in Rust. MATLAB is the front end for the API, the phase-plane app, plots and sessions. Rust does the parsing, code generation, JIT compilation and all of the numerics, in `f64`.

## What it does

- Parse `.ode` files, with diagnostics that give file, line and column (`phasecont.Model`).
- Integrate trajectories with the Dormand-Prince 5(4) pair (`phasecont.simulate`).
- Find equilibria with Newton's method (`phasecont.equilibrium`).
- Continue equilibria in one parameter and detect fold (LP), branch point (BP) and Hopf (H) points, with their normal form coefficients (`phasecont.continuation`).
- Continue limit cycles from a Hopf point by orthogonal collocation and detect folds (LPC), period doublings (PD) and Neimark-Sacker (NS) points from the Floquet multipliers.
- Continue LP, H, LPC, PD and NS points in two parameters, and detect their codim-2 points: CP, BT, ZH, GH and HH on LP and H curves, LPPD, LPNS, CPC and R1 on LPC curves, LPPD, PDNS, GPD and R2 on PD curves, and R1 to R4, LPNS, PDNS and CH on NS curves.
- Continue curves of orbits homoclinic to a saddle in two parameters, from a long-period orbit of a cycle branch or from a given orbit, and detect the codim-2 points NSA, BLS, BLU, NSF, NDS, NDU, TLS, TLU, ZES, ZEU, OFS and OFU (`Type = "HOM"`). The inclination flips IFS and IFU need `InclinationFlips = true`, because they cost an adjoint solve at every point.
- Move the collocation mesh of cycles and homoclinic orbits every k steps (`MeshAdapt = k`). The default is 3 for ODE models and 0 for delay cycles.
- Continue the equilibria of a delay model, one that calls `delay(x, tau)`, with the characteristic roots counted at every point (`Roots`), and detect LP, BP and H points. Follow the Hopf curves of a delay model in two parameters with their BT and ZH points, GH points with `Lyapunov = true` and HH points with `Roots`. `Coefficients = true` adds the zero-Hopf and Hopf-Hopf normal form coefficients.
- Continue the cycles that branch off a Hopf point of a delay model, with their Floquet multipliers (`Floquet`) and their LPC, PD and NS points. `Detect = "BPC"` labels the branch points of the discretized cycle problem, and `Switch = "BPC1"` follows the branch that leaves one.
- Integrate many trajectories of one model at once, for a grid of start states or parameter sets, on the CPU or on a GPU through CUDA, OpenCL or Metal (`phasecont.ensemble`). `Precision` has no default. Single-precision results are labeled as such, and `refine` runs a member again in double precision on the CPU. `phasecont.gpuDevices` lists the devices, and `phasecont.gpuSelfTest` checks one against exact solutions.
- Write a branch as a pgfplots figure and its special points as a booktabs table, compile the document with `latexmk`, or edit it in the LaTeX studio app (`phasecont.latex`, `phasecont.app.LatexStudio`).
- Save a model and its results to a session file, load it again, and replay every run to check the result (`phasecont.Session`, `phasecont.session.replay`).
- Explore a model in the phase-plane app, with direction field, nullclines, trajectories and equilibria (`phasecont.app.PhasePlane`).
- Build a model from equation text, a table, a MATLAB function handle, a MATLAB function file or an SBML file, with no `.ode` file (`phasecont.Model.fromEquations`, `fromTable`, `fromFunction`, `fromFile`), or type it into the model editor (`phasecont.app.ModelEditor`).

`Lyapunov`, `Coefficients` and `Floquet` (delay models), `InclinationFlips` (homoclinic curves) and `Detect = "BPC"` (delay cycles) are off unless you pass them, because each one adds work at every point. `help phasecont.continuation` describes them.

Users never need a compiler. The engine ships as a prebuilt library, and models are compiled to native code inside the MATLAB process. When `mex` finds a C compiler, `phasecont.config("Codegen", "aot")` builds new models as C libraries instead. If that build or its load fails, the model runs on the JIT and a `phasecont:aot:*` warning says so.

The `.ode` reader refuses some XPPAUT constructs, and delay models run only in `continuation`. [What the `.ode` reader refuses](#what-the-ode-reader-refuses) lists them.

## Requirements

- MATLAB R2026b or later, on Windows (win64), Linux (glnxa64, glibc 2.28 or later) or macOS on Apple silicon (maca64, macOS 15 or later). Intel Macs are not supported.
- Base MATLAB only. No add-on toolbox is needed.

## Install

1. Download `PhaseCont-<version>.mltbx` from the [Releases page](https://github.com/Nub04/phasecont/releases). `SHA256SUMS` next to it lists the checksum of the file.
2. Open the `.mltbx` file in MATLAB, or run `matlab.addons.install("PhaseCont-<version>.mltbx")`.
3. Check the install:

   ```matlab
   which phasecont.Model
   ```

The macOS binaries are not signed by Apple. If macOS refuses to load `pckern.mexmaca64` or `libpckern.dylib`, please open an issue with the message.

## Upgrading

Close every MATLAB session that has used PhaseCont before you install a new version or uninstall the old one. The MEX gateway never unloads the engine library while MATLAB runs, so the old files stay locked and the upgrade fails with "Permission denied". Start a new MATLAB, then install the new `.mltbx`.

## First run

The toolbox ships two examples. The FitzHugh-Nagumo example finds two Hopf points:

```matlab
branches = fitzhugh_nagumo_example();
```

The Morris-Lecar example builds a small bifurcation diagram (rest states, limit cycles, a fold of cycles, the curves of folds and Hopf points in two parameters, and one trajectory), saves it to a session file, replays every run and opens the result in the phase-plane app:

```matlab
r = morris_lecar_session_example(Open = true);
```

To work with your own model, save this as `fold.ode`:

```
par a=-1
x'=a+x^2
init x=1
done
```

```matlab
m  = phasecont.Model("fold.ode");
eq = phasecont.equilibrium(m, -1);                 % a Point with Label "EP"
br = phasecont.continuation(m, eq, Free="a", Direction=1, ...
         MaxSteps=100, Bounds=[-5 5], Plot=true);
lp = br.points("LP");                              % the fold at a = 0, x = 0
lp.Data.coefficient                                % the normal form coefficient, 1
```

`continuation` blocks while it runs and polls the engine, first after 1 ms and then at growing intervals up to 50 ms. Ctrl+C cancels the run and frees it. It returns a `phasecont.Branch` that holds copies of the data, so it stays valid after the model is deleted. `Type` selects the problem: `"EP"` (the default), `"LC"` from a Hopf point, and `"LP"`, `"H"`, `"LPC"`, `"PD"`, `"NS"` or `"HOM"` with two free parameters. `help phasecont.continuation` lists the options.

## `@` options

PhaseCont reads only a few of the `@` options in an `.ode` file, and only in the phase-plane app:

- `xp` and `yp` pick the variables on the two axes.
- `xlo`, `xhi`, `ylo` and `yhi` set the window.
- `total` fills in the integration length T.

The app names `meth`, `dt` and `bound` in its notes and does not use them, since it always integrates with the Dormand-Prince 5(4) pair. `simulate`, `equilibrium`, `continuation` and `ensemble` read no `@` options. The first `simulate` call on a model that sets any of them warns once with the ID `phasecont:simulate:ignoredOption` and names them, and the result is the same as without them. The one exception is a `meth` value that starts with `d`, such as `meth=discrete`, which makes the model a map (see below) and gives no warning. Set the solver with the `RelTol`, `AbsTol`, `MaxSteps` and `Hmax` options of `simulate` instead.

## What the `.ode` reader refuses

`phasecont.Model` refuses a model that uses any of the constructs below. The error has the ID `phasecont:engine:model`, and its message quotes the statement or names the function.

| Construct | Used in XPPAUT for |
|---|---|
| `wiener`, `markov`, `ran`, `normal`, `poisson` | noise and random numbers (the message says the model is for simulation only) |
| `global` | events that reset the state during a run (also "for simulation only") |
| `0=...`, `solv` | differential-algebraic equations |
| `volt`, `int{...}` | Volterra integral equations |
| `bndry`, `hom_bcs` | boundary value problems |
| `sum(...)of(...)` | sums over an index |
| `shift`, `del_shft` | variables picked by their position in the list |
| `special`, `export`, `only`, `options` | network functions, values passed to a DLL, output selection and option files |
| `table` read from a file | tabulated data; a table given by a formula (`table h % 51 0 1 t^2`) works |

The reader also stops with an error on these files:

- A file whose first line starts with a number is in the old XPPAUT format. Convert it to the current format.
- A file that has both ODE lines (`x'=`) and map lines (`x(t+1)=`).
- A file with `#include`. MATLAB passes the model to the engine as text, so the included file cannot be found. Paste its lines into the model instead.

Some lines load and do nothing. Active comments (lines that start with `"`) give the warning `phasecont:model:parseWarning`, and `set` blocks are ignored with no warning.

A delay model, one that calls `delay(x, tau)`, loads and runs in `continuation` (equilibria, Hopf curves and cycles from a Hopf point). `simulate`, `equilibrium` and `ensemble` refuse it with the error `phasecont:engine:badArg`.

A map is a file of `x(t+1)=...` lines, or of `x'=...` lines with `@ meth=discrete`, which XPPAUT also iterates as a map. `Model.IsMap` is true for a map, and `simulate` iterates it. Save this as `logistic.ode`:

```
par r=2.5
x(t+1)=r*x*(1-x)
init x=0.1
done
```

```matlab
m  = phasecont.Model("logistic.ode");
tr = phasecont.simulate(m, 0.1, [0 50]);   % rows for n = 0, 1, ..., 50
plot(tr)                                    % the iterates, drawn as dots
```

For a map, `TSPAN` is `[n0 n1]`, two integers with `n0 < n1`, and `t` inside the right-hand side is the iterate index n. One run gives at most 1000000 iterates. `RelTol`, `AbsTol`, `MaxSteps` and `Hmax` do not apply and raise `phasecont:simulate:mapOption`. A value that is not finite stops the run early with a warning and keeps the iterates up to that point.

`equilibrium` on a map solves F(x) = x, and `Eigenvalues` holds the multipliers, the eigenvalues of the Jacobian of F. The fixed point is stable when every multiplier has modulus below 1. `continuation` with Type `"EP"` follows the fixed points in one parameter:

```matlab
eq = phasecont.equilibrium(m, 0.6);                % x = 0.6, multiplier -0.5
br = phasecont.continuation(m, eq, Free = "r", Direction = 1, Bounds = [2 3.5]);
br.points("PD")                                    % r = 3, x = 2/3
```

On a map, `Tests` has the four columns `[LP BP PD NS]`, `Stable` is true where every multiplier has modulus below 1, and `Detect` takes a subset of `["LP" "BP" "PD" "NS"]`, all four by default. `Data.coefficient` holds the fold coefficient at an LP point, the period-doubling coefficient b at a PD point (b > 0 is supercritical) and the Neimark-Sacker coefficient d at an NS point (d < 0 is supercritical). An NS point also has the angle θ of its multipliers e^{±iθ} in `Data.values.theta`. Sessions save and replay iterations and branches of fixed points. Other types of `continuation` on a map raise a `phasecont:` error, `ensemble` refuses a map with `phasecont:engine:unsupportedModel`, and `phasecont.app.PhasePlane` refuses it with `phasecont:app:mapModel`. `Model.rhs` evaluates the right-hand side of a map. Curves of fixed points in two parameters and period-k points are planned.

## Models without an `.ode` file

Every route below writes `.ode` text and builds the model from it, so sessions, continuation and the phase-plane app work the same way. Nothing is defaulted: a missing initial value or parameter value raises an error that names it.

Equation text, with the parameters and initial values in structs:

```matlab
m = phasecont.Model.fromEquations(["x' = y" "dy/dt = -x - c*y"], ...
        Params=struct(c=0.1), Init=struct(x=1, y=0));
```

A table (or a struct) per part:

```matlab
states = table(["x"; "y"], ["y"; "-x - c*y"], [1; 0], VariableNames=["Name" "Rhs" "Init"]);
params = table("c", 0.1, VariableNames=["Name" "Value"]);
m = phasecont.Model.fromTable(States=states, Params=params);
```

A function handle or a function file. PhaseCont reads the text of the handle and never calls it, so only arithmetic, indexing and the usual math functions work. A loop, an `if` or an unknown call raises `phasecont:import:unsupported` with its position:

```matlab
m = phasecont.Model.fromFunction(@(t, x, p) [x(2); -x(1) - p(1)*x(2)], ...
        Init=[1; 0], Params=struct(c=0.1));
m = phasecont.Model.fromFile("pendulum.m", Init=[1; 0], Params=struct(c=0.1));
```

An SBML file (level 2 or 3: species, parameters, compartments, reactions, rate rules and assignment rules). Events, delays and algebraic rules are refused by name:

```matlab
m = phasecont.Model.fromFile("decay.xml");
```

The model editor has a tab for equations, a tab for a table and a tab for `.ode` text. It lists each error with its line and column, opens `.ode`, SBML and `.m` files, and opens the last good model in the phase-plane app:

```matlab
app = phasecont.app.ModelEditor;
```

## Test your GPU

An ensemble runs many trajectories of one model at once, for a grid of start states or parameter sets. It does not speed up a single run or a continuation. Those stay on the CPU in double precision.

The ensemble engine has backends for the CPU, for CUDA (NVIDIA, through NVRTC), for OpenCL (NVIDIA, AMD and Intel) and for Metal (Apple GPUs, single precision only). The CUDA and OpenCL backends have run on an NVIDIA RTX 5060 (2026-11-07), and the OpenCL backend on an Intel Arc 140T (2026-11-08, again on 2026-11-10 through MATLAB and the command line tool). The self-test below passed on both in single and double precision. No AMD device has run. The Metal backend has run only on the virtual GPU of GitHub's hosted macOS runners, never on Apple hardware, so a run on an AMD card or a Mac is the first real check. Run these in order:

1. List what the engine finds:

   ```matlab
   [devices, backends] = phasecont.gpuDevices()
   ```

   `devices` has one row per device, and `backends` says why a backend is unavailable (for example no driver, or no NVRTC for CUDA). A missing library only changes that backend's state. CUDA needs NVRTC, which MATLAB ships and a CUDA toolkit also provides. Without it, an NVIDIA GPU runs through OpenCL.

2. Run the self-test on one device. It compares small ensembles with closed-form solutions and checks that a repeated run and a smaller run give the same numbers. `Precision` has no default:

   ```matlab
   s = phasecont.gpuSelfTest(Backend="opencl", Device=0, Precision="single")
   s.Checks
   ```

   `s.Pass` is true when all four checks pass. The thresholds are the test's own and not a general accuracy claim.

3. Run an ensemble and check a member in double precision. A single-precision result is labeled and is for screening. `refine` runs members again on the CPU in double precision, and only that result may feed continuation:

   ```matlab
   r = phasecont.ensemble(m, X0, [], [0 50], Backend="auto", Precision="single", ...
           Rtol=1e-4, Atol=1e-6);
   r.Label
   f64 = r.refine(3);
   ```

4. From a shell, the command line tool gives the same answers and runs the full tiers. It exits with 0 on a pass, 1 on a failure and 3 when the backend or device does not exist (which is not a pass):

   ```sh
   target/debug/pc-cli gpu list
   target/debug/pc-cli gpu selftest --backend opencl --device 0 --precision single
   target/debug/pc-cli ensemble --bench
   ```

5. Run the test suites so that a missing GPU fails instead of skipping. Without the variable, a GPU test with no device prints `SKIPPED: no <backend> device` and passes:

   ```sh
   PC_REQUIRE_GPU=cuda,opencl cargo test -p pc-ensemble
   PC_REQUIRE_GPU=cuda,opencl scripts/matlab-win "buildtool test"
   ```

If you have an AMD or Apple GPU, the most useful report comes from one command. Build `pc-cli` (`cargo build --release -p pc-cli`), then run this from the repo root:

```sh
scripts/gpu-report
```

It lists the devices, runs the self-test on every GPU it finds in single precision (and in double precision where the device has it), and writes `gpu-report-<date>.txt`. The file holds device names, driver strings, the OS name and the self-test output, and no user name, host name or paths. Attach it to a new "GPU report" issue. The script exits with 0 when every self-test that ran passed, 1 when one failed and 2 when `pc-cli` is missing. To use another binary, pass `--cli PATH`, and `--out FILE` sets the file name. By hand, the same data is the output of `pc-cli gpu list` and of `pc-cli gpu selftest --backend opencl --precision single` (`--backend metal` on a Mac). AMD runs through OpenCL, and nobody has run it on AMD hardware yet. Nobody has run the Metal backend on a physical Mac yet either.

## LaTeX figures and tables

PhaseCont writes a branch as a pgfplots figure and its special points as a booktabs table. The text comes from the engine, so the same branch always gives the same LaTeX. To compile a document you need `latexmk` from TeX Live or MiKTeX. PhaseCont does not ship a TeX distribution and has not been tested with MiKTeX.

Your document needs these lines in its preamble:

```latex
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{booktabs}
```

Build a figure from a branch, then render it. `Digits` counts significant digits, from 1 to 17, and has no default, so the call raises `phasecont:latex:missingDigits` without it:

```matlab
fig = phasecont.latex.figureFromBranch(br, Var = "x", Title = "Equilibria");
r = phasecont.latex.pgfplots(fig, Digits = 6, Folder = "paper");
```

`r.Tex` holds the text and `paper/figure.tex` holds the same text, ready for `\input{figure}`. Stable runs are solid, unstable runs are dashed and each special point has a mark and a label. The points are written into the `.tex` file. Pass `Data = "external"` to write one `.dat` table per series next to it instead. Edit the fields of `fig` (`XLabel`, `YLabel`, `Title`) before you render to rename the axes.

The special points of a branch come out as a table the same way:

```matlab
t = phasecont.latex.specialTable(br, Digits = 4, Folder = "paper");
```

Both calls take `Standalone = true`, which adds a preamble so the file compiles on its own. To compile it, pass the text to `build`:

```matlab
r = phasecont.latex.pgfplots(fig, Digits = 6, Standalone = true);
b = phasecont.latex.build(r.Tex, Files = r.Files);
```

`b.Ok` is true when a PDF came out, `b.Pdf` is its path, and `b.Errors` lists each TeX error with its line. A TeX error does not raise, and a missing `latexmk` raises `phasecont:latex:engineNotFound`. `build` writes the document to a new folder under `tempdir`, runs `latexmk` there with shell escape off, and leaves the folder in place. By default it sets no time limit, so a document that never ends keeps MATLAB waiting until you press Ctrl-C, which stops `latexmk` and the TeX programs it started. Pass `Timeout = 60` to stop them after 60 seconds, and `b.TimedOut` is then true. `build` runs `latexmk` through the JVM of MATLAB, and on Windows it finds a `latexmk.cmd` wrapper through `PATHEXT`. When the JVM cannot start `latexmk` (MATLAB started with `-nojvm`, for example), a finite `Timeout` warns with `phasecont:latex:noTimeout` and `latexmk` runs with no limit. With `Inf` the same fallback is silent, and Ctrl-C does not stop it.

The editor opens with:

```matlab
app = phasecont.app.LatexStudio;
```

Type or open a `.tex` file, enter `Digits`, pick a branch variable from the workspace and press Insert figure or Insert table. Compile runs `latexmk` and lists the errors. Selecting one shows the lines around it with the error line marked. Open PDF hands the file to your system viewer. The window does not respond while a compile runs. There is no time limit by default, so a document that never ends blocks the window until you press Ctrl-C in the MATLAB window. Set one with `phasecont.app.LatexStudio(Timeout = 60)`.

The command line renders the same text from a JSON file. The format is in `docs/SPEC.md`, section 13:

```sh
cargo run -p pc-cli -- latex pgfplots figure.json --standalone --out figure.tex
```

## Build from source

You need this only to change PhaseCont itself.

1. Install MATLAB R2026b, Rust 1.99.1 and a C compiler that `mex` supports (`mex -setup C`).
2. Clone the repo:

   ```sh
   git clone https://github.com/Nub04/phasecont.git
   cd phasecont
   ```

3. From the repo root in MATLAB, build the engine and the gateway and run the tests:

   ```matlab
   buildtool cargo mex test
   ```

   - `cargo` builds the engine library (`pckern`) and copies it to the binary folder named in `tests/+fixtures/binFolder.m`.
   - `mex` builds the gateway next to it.
   - `test` runs the `matlab.unittest` suites in `tests/`.

4. Optionally, build a package for this platform in `build/`:

   ```matlab
   buildtool package
   ```

On Windows the repo is built from WSL, with the MinGW cross compiler and the Windows Rust std (see `docs/ENVIRONMENT.md`). From WSL, run the same commands through the wrapper, which converts paths and starts Windows MATLAB:

```sh
scripts/matlab-win "buildtool cargo mex test"
```

If a MATLAB session still has the engine library loaded, the copy fails. Close that session and run the task again.

## VS Code extension

`editors/vscode` holds a VS Code extension for `.ode` files: syntax highlighting, and diagnostics, hover, completion, go to definition and an outline from `pc-lsp`, a language server that runs the PhaseCont parser. There is no Marketplace listing yet. To try it from a checkout:

```sh
cargo build --release -p pc-lsp
mkdir -p editors/vscode/server/linux-x64
cp target/release/pc-lsp editors/vscode/server/linux-x64/
cd editors/vscode
npm ci --ignore-scripts
npm test
npm run bundle
```

Then open `editors/vscode` in VS Code and press F5. On Windows and macOS use the folder `server/win32-x64` (`pc-lsp.exe`), `server/darwin-arm64` or `server/darwin-x64`, or set `phasecont.server.path` to the binary. `editors/vscode/README.md` lists the features and settings.

To build an installable package for the platform of the server you copied, run this in `editors/vscode`, with `<platform>` set to `linux-x64`, `win32-x64`, `darwin-arm64` or `darwin-x64`:

```sh
npm run package -- --target <platform>
code --install-extension phasecont-ode-<platform>-0.1.0.vsix
```

`npm run package` calls `vsce`, which is a build tool only. It never goes into the `.vsix`, and the `.vsix` holds `dist/extension.js`, the server binary and the manifest files, with no `node_modules`.

## Checks

Before a commit:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo deny check
```

`pc-cli` is a headless tool for the same engine. For example, `target/debug/pc-cli parse FILE` parses a model and prints its diagnostics, and `pc-cli session replay FILE` replays a session file.

## Layout

| Path | Content |
|---|---|
| `crates/pc-ode` | lexer, parser, AST, diagnostics, `.ode` writer |
| `crates/pc-ir` | expression DAG and symbolic differentiation |
| `crates/pc-jit` | Cranelift lowering of the right-hand side and its derivatives |
| `crates/pc-codegen` | C emitter for the optional AOT path |
| `crates/pc-linalg` | bordered solvers, condensation and Floquet multipliers |
| `crates/pc-engine` | continuation core and problem types |
| `crates/pc-ffi` | C ABI, handle registry, panic barrier |
| `crates/pc-cli` | headless runner |
| `crates/pc-lsp` | language server for `.ode` files |
| `editors/vscode` | VS Code extension (TypeScript client) |
| `mex/` | the MEX gateway (C) |
| `toolbox/+phasecont` | MATLAB API and the phase-plane app |
| `toolbox/examples` | the examples above |
| `tests/` | `matlab.unittest` suites |
| `references/` | oracle values for regression tests |
| `docs/` | `SPEC.md`, `STATUS.md`, `ENVIRONMENT.md`, `spikes.md` |

The technical spec is `docs/SPEC.md`. Section 10 describes the C ABI, section 11 the MATLAB API and section 12 the build and release jobs. `docs/STATUS.md` lists what is done and the known limits.

## Licensing

PhaseCont is licensed under Apache-2.0 (see `LICENSE`). Third-party Rust dependencies must be MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib or Unicode-3.0, and `cargo deny check licenses` enforces that. The package ships the license text of every crate linked into the engine in `THIRD_PARTY_LICENSES.txt`. Ported code is listed in `THIRD_PARTY_NOTICES.md`. The project does not use or copy MatCont.
