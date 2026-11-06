# PhaseCont

PhaseCont (working name) is a MATLAB toolbox for phase-plane and bifurcation analysis of ODE models. It reads XPP/XPPAUT `.ode` files and runs its own continuation engine, written in Rust. MATLAB is the front end for the API, the phase-plane app, plots and sessions. Rust does the parsing, code generation, JIT compilation and all of the numerics, in `f64`.

## What it does

- Parse `.ode` files, with diagnostics that give file, line and column (`phasecont.Model`).
- Integrate trajectories with the Dormand-Prince 5(4) pair (`phasecont.simulate`).
- Find equilibria with Newton's method (`phasecont.equilibrium`).
- Continue equilibria in one parameter and detect fold (LP), branch point (BP) and Hopf (H) points, with their normal form coefficients (`phasecont.continuation`).
- Continue limit cycles from a Hopf point by orthogonal collocation and detect folds (LPC), period doublings (PD) and Neimark-Sacker (NS) points from the Floquet multipliers.
- Continue LP, H, LPC, PD and NS points in two parameters, and detect the codim-2 points on LP and H curves (CP, BT, ZH, GH, HH).
- Save a model and its results to a session file, load it again, and replay every run to check the result (`phasecont.Session`, `phasecont.session.replay`).
- Explore a model in the phase-plane app, with direction field, nullclines, trajectories and equilibria (`phasecont.app.PhasePlane`).
- Build a model from equation text, a table, a MATLAB function handle, a MATLAB function file or an SBML file, with no `.ode` file (`phasecont.Model.fromEquations`, `fromTable`, `fromFunction`, `fromFile`), or type it into the model editor (`phasecont.app.ModelEditor`).

Users never need a compiler. The engine ships as a prebuilt library, and models are compiled to native code inside the MATLAB process.

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

`continuation` blocks while it runs and polls the engine about every 50 ms. Ctrl+C cancels the run and frees it. It returns a `phasecont.Branch` that holds copies of the data, so it stays valid after the model is deleted. `Type` selects the problem: `"EP"` (the default), `"LC"` from a Hopf point, and `"LP"`, `"H"`, `"LPC"`, `"PD"` or `"NS"` with two free parameters. `help phasecont.continuation` lists the options.

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

The ensemble engine has backends for the CPU, for CUDA (NVIDIA, through NVRTC), for OpenCL (NVIDIA, AMD and Intel) and for Metal (Apple GPUs, single precision only). No GPU backend has run on a GPU so far. The code is tested on the CPU only, and the Metal host code has been type-checked for macOS but never run, so a GPU run is the first real check. Run these in order:

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

If you have an AMD or Apple GPU, the most useful report is the JSON that `pc-cli gpu selftest --backend opencl --precision single` prints (`--backend metal` on a Mac), together with the output of `pc-cli gpu list`. Please attach both to an issue. AMD runs through OpenCL, and nobody has run it on AMD hardware yet. Nobody has run the Metal backend on a Mac yet either.

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

`b.Ok` is true when a PDF came out, `b.Pdf` is its path, and `b.Errors` lists each TeX error with its line. A TeX error does not raise, and a missing `latexmk` raises `phasecont:latex:engineNotFound`. `build` writes the document to a new folder under `tempdir`, runs `latexmk` there with shell escape off, and leaves the folder in place. It stops `latexmk` and the TeX programs it started after 300 seconds, and `b.TimedOut` is then true. Pass `Timeout = 60` for another limit or `Timeout = Inf` for none. A finite limit runs `latexmk` through the JVM of MATLAB. When MATLAB cannot do that, `build` warns with `phasecont:latex:noTimeout` and runs `latexmk` with no limit.

The editor opens with:

```matlab
app = phasecont.app.LatexStudio;
```

Type or open a `.tex` file, enter `Digits`, pick a branch variable from the workspace and press Insert figure or Insert table. Compile runs `latexmk` and lists the errors. Selecting one shows the lines around it with the error line marked. Open PDF hands the file to your system viewer. The window does not respond while a compile runs, which is at most `Timeout` seconds (default 300, set with `phasecont.app.LatexStudio(Timeout = 60)`).

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
