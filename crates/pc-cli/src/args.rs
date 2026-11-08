//! Command line parsing. There is no argument crate: the grammar is small.
//!
//! The command and the flags are matched as UTF-8. The file operand stays an
//! `OsString`, so a path that is not valid UTF-8 still opens.

use std::ffi::OsString;
use std::path::PathBuf;

use pc_ensemble::Precision;
use pc_ensemble::device::Backend;
use pckern::json::ensemble::precision_from_name;

/// Random points per entry when `--points` is not given.
pub const DEFAULT_POINTS: usize = 20;
/// Seed when `--seed` is not given.
pub const DEFAULT_SEED: u64 = 1;
/// Ensemble sizes of `ensemble --bench` when `--members` is not given. The sizes are
/// printed with the results, so they are a measurement setting and not a hidden default.
pub const DEFAULT_BENCH_MEMBERS: [usize; 3] = [1_000, 10_000, 100_000];

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Print the top-level help to stdout.
    Help,
    /// Print the help text of a command to stdout.
    ToolHelp(&'static str),
    /// Print `pc-cli <version>`.
    Version,
    /// Parse one file.
    Parse { file: PathBuf, json: bool },
    /// Print the shape and the DAG sizes of one model.
    Ir { file: PathBuf },
    /// Check the derivative entries of one model.
    CheckDerivs {
        file: PathBuf,
        points: usize,
        seed: u64,
    },
    /// Write C code for F, JacU and JacP.
    EmitC { file: PathBuf, out: Option<PathBuf> },
    /// List the DLLs a 64-bit PE file imports and check them against the allow-list.
    DllImports { file: PathBuf },
    /// Write the license texts of the dependency closure of a package.
    Licenses {
        /// Target triples; each `--target` adds one.
        targets: Vec<String>,
        package: String,
        out: PathBuf,
        overrides: Option<PathBuf>,
    },
    /// Follow equilibrium branches described by a problem file.
    Cont {
        file: PathBuf,
        problem: PathBuf,
        fallback: bool,
    },
    /// Find the characteristic roots of a linear delay equation described by a problem file.
    DdeRoots { file: PathBuf },
    /// Find an equilibrium of a model with delay calls and the roots of the linear equation there.
    DdeEq { file: PathBuf },
    /// Continue an equilibrium of a model with delay calls and report its Hopf points.
    DdeBranch { file: PathBuf },
    /// Continue the Hopf point of a model with delay calls in two parameters.
    DdeHopf { file: PathBuf },
    /// Correct a periodic orbit of a model with delay calls, from an orbit or a Hopf point.
    DdeCycle { file: PathBuf },
    /// Compute the first Lyapunov coefficient at a Hopf point of a delay equilibrium.
    DdeLyap { file: PathBuf },
    /// Compute the zero-Hopf normal-form coefficients at a point of a delay equilibrium.
    DdeZh { file: PathBuf },
    /// Compute the Hopf-Hopf normal-form coefficients at a point of a delay equilibrium.
    DdeHh { file: PathBuf },
    /// Rerun the runs of a session file and compare the results.
    SessionReplay {
        file: PathBuf,
        tol: f64,
        fallback: bool,
    },
    /// List the ensemble backends and their devices.
    GpuList,
    /// Run the self-test on one device.
    GpuSelftest {
        backend: Backend,
        device: usize,
        precision: Precision,
    },
    /// Run the ensemble described by a job file.
    Ensemble { file: PathBuf, job: PathBuf },
    /// Time the ensemble engine on a built-in model.
    EnsembleBench { members: Vec<usize> },
    /// Render a LaTeX figure or table from a JSON document.
    Latex {
        kind: pc_latex::Kind,
        file: PathBuf,
        out: Option<PathBuf>,
        standalone: bool,
    },
}

/// Why the command line is not valid. The text is a short usage line for
/// stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

pub const USAGE: &str = "\
usage: pc-cli parse [--json] [--] <file.ode>
       pc-cli ir [--] <file.ode>
       pc-cli check-derivs [--points N] [--seed S] [--] <file.ode>
       pc-cli emit-c [-o out.c] [--] <file.ode>
       pc-cli cont [--fallback] [--] <file.ode> <problem.json>
       pc-cli dde-roots [--] <problem.json>
       pc-cli dde-eq [--] <problem.json>
       pc-cli dde-branch [--] <problem.json>
       pc-cli dde-hopf [--] <problem.json>
       pc-cli dde-cycle [--] <problem.json>
       pc-cli dde-lyap [--] <problem.json>
       pc-cli dde-zh [--] <problem.json>
       pc-cli dde-hh [--] <problem.json>
       pc-cli session replay [--tol X] [--fallback] [--] <file.pcsession.json>
       pc-cli gpu list
       pc-cli gpu selftest --backend B [--device N] --precision P
       pc-cli ensemble [--] <file.ode> <job.json>
       pc-cli ensemble --bench [--members N,N,...]
       pc-cli dll-imports [--] <file.dll>
       pc-cli licenses --target TRIPLE... [--package NAME] --out FILE [--overrides DIR]
       pc-cli latex pgfplots|table [--standalone] [--out FILE.tex] [--] <input.json>
       pc-cli --help | pc-cli --version";

pub const HELP: &str = "\
pc-cli: headless runner for PhaseCont

usage:
  pc-cli parse <file.ode> [--json]
  pc-cli ir <file.ode>
  pc-cli check-derivs <file.ode> [--points N] [--seed S]
  pc-cli emit-c <file.ode> [-o out.c]
  pc-cli cont <file.ode> <problem.json> [--fallback]
  pc-cli dde-roots <problem.json>
  pc-cli dde-eq <problem.json>
  pc-cli dde-branch <problem.json>
  pc-cli dde-hopf <problem.json>
  pc-cli dde-cycle <problem.json>
  pc-cli dde-lyap <problem.json>
  pc-cli dde-zh <problem.json>
  pc-cli dde-hh <problem.json>
  pc-cli session replay <file.pcsession.json> [--tol X] [--fallback]
  pc-cli gpu list
  pc-cli gpu selftest --backend B [--device N] --precision P
  pc-cli ensemble <file.ode> <job.json>
  pc-cli ensemble --bench [--members N,N,...]
  pc-cli dll-imports <file.dll>
  pc-cli licenses --target TRIPLE... [--package NAME] --out FILE [--overrides DIR]
  pc-cli latex pgfplots|table [--standalone] [--out FILE.tex] [--] <input.json>
  pc-cli --help
  pc-cli --version

commands:
  parse         read an XPPAUT .ode file and report its diagnostics and a summary
  ir            print the size of the model and of each derivative entry
  check-derivs  compare the derivative entries of the DAG, the JIT and the
                f64 fallback, and with finite differences
  emit-c        write C code for F, JacU and JacP
  cont          follow an equilibrium branch and print its points and special points
  dde-roots     find the characteristic roots of a linear delay equation
  dde-eq        find an equilibrium of a model with delay calls and its roots
  dde-branch    continue such an equilibrium in one parameter and find its Hopf points
  dde-hopf      continue a Hopf point of such an equilibrium in two parameters
  dde-cycle     correct, or continue, a periodic orbit of such a model
  dde-lyap      compute the first Lyapunov coefficient (and the second, if asked) at such a Hopf point
  dde-zh        compute the zero-Hopf normal-form coefficients at such a point
  dde-hh        compute the Hopf-Hopf normal-form coefficients at such a point
  session      rerun the runs of a session file and compare them with the stored ones
  gpu           list the ensemble devices and test one of them
  ensemble      run many trajectories of one model, or time the engine
  dll-imports   list the DLLs a 64-bit Windows PE file imports and check the allow-list
  licenses      write the license texts of the crates a package links
  latex         write a pgfplots figure or a booktabs table from a JSON document
";

/// Largest relative difference a replay accepts when `--tol` is not given.
pub const DEFAULT_TOL: f64 = 1e-8;

pub const SESSION_HELP: &str = "\
usage: pc-cli session replay [--tol X] [--fallback] [--] <file.pcsession.json>

Loads the model of a session file through the engine library that MATLAB loads,
checks that its source hashes to the stored SHA-256, then reruns every run of
the file (equilibrium solves, continuations and trajectories) and compares the
new rows with the stored ones. Prints one JSON report on stdout.

A run passes when the row counts, the special point labels and indexes and the
stop reasons are equal and no value differs by more than the tolerance,
|a - b| / max(1, |b|). NaNs must sit in the same places and infinities must be
equal. \"identical\" in the report says whether every value has the same bits.

options:
  --tol X     largest relative difference that passes, a finite number of at
              least 0 (default 1e-8)
  --fallback  evaluate the model with the f64 interpreter instead of the JIT
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 every run passes, 1 a run does not match or cannot be rerun, 2
usage error, a file that is not a version 1 session, or a model that does not
load or does not match its stored hash
";

pub const GPU_HELP: &str = "\
usage: pc-cli gpu list
       pc-cli gpu selftest --backend B [--device N] --precision P

`list` prints one JSON object: the state of every backend (cpu, cuda, opencl,
metal) and the devices that can run an ensemble now. A backend is `available`,
`not_built` (this binary does not contain it) or `unavailable` (it is built in
but cannot start; the reason follows).

`selftest` runs a few small ensembles on one device and prints one JSON object:
the device, the driver, the precision, the worst error of each tier against its
limit, whether the device matches the CPU backend bit for bit (`null` for the
CPU backend itself) and `pass`. The device is the first of the backend unless
--device says otherwise, and the report names the device that ran.

options:
  --backend B    cpu, cuda, opencl or metal (required)
  --device N     index among the devices of the backend (default 0)
  --precision P  single or double (required)
  -h, --help     print this help

exit codes: 0 every check passes (list: always), 1 a check fails or the device
refuses a job, 2 usage error, 3 the backend is not built in or has no such
device (no report is printed, and this is never a pass)
";

pub const ENSEMBLE_HELP: &str = "\
usage: pc-cli ensemble [--] <file.ode> <job.json>
       pc-cli ensemble --bench [--members N,N,...]

Runs many trajectories of one model and prints one JSON object. The job file
needs every key below except `device`; nothing is filled in for a missing key.

  backend     cpu, cuda, opencl or metal
  device      index among the devices of the backend (default 0)
  precision   single or double
  t0, t1      start and end time, t0 < t1
  method      {\"type\": \"rk4\", \"dt\": X}, a largest fixed step, or
              {\"type\": \"dp5\", \"rtol\": X, \"atol\": X, \"hmax\": X or null}
  x0          one array of initial states per member
  params      {\"shared\": [..]} for one set, or {\"per_member\": [[..], ..]}
  output      \"final\", or {\"samples\": [t, ..]} (increasing, in (t0, t1], last = t1)
  bounds      stop a member when a state leaves [-X, X]; a number or null
  max_steps   most steps per member, at least 1

The result has members, n, times, status (one of ok, bounds, step_too_small,
max_steps, nonfinite per member), states[member][sample][state] (null for a
value that is not finite), t_reached, accepted, rejected and a label with the
backend, the device, the driver, the precision and single_precision. A result
with single_precision true is a screening result: run the member again in double
precision before it feeds continuation, normal forms or Floquet multipliers.

--bench times a built-in van der Pol ensemble on every device, in both
precisions, at each size (default 1000,10000,100000) and prints one row each.

options:
  --bench        time the engine instead of reading a job file
  --members L    sizes for --bench, a comma-separated list of integers >= 1
  --             the next argument is the file, even if it starts with `-`
  -h, --help     print this help

exit codes: 0 the ensemble ran (members that did not finish are reported in
status), 1 the model does not load or the engine refuses the job, 2 usage error
or a bad job file (the message names the key), 3 the backend is not built in or
has no such device
";

pub const PARSE_HELP: &str = "\
usage: pc-cli parse [--json] [--] <file.ode>

Reads the file and the files it includes, prints each diagnostic with its
source line, then a summary of the model.

options:
  --json     print one JSON object instead of text
  --         the next argument is the file, even if it starts with `-`
  -h, --help print this help

Columns count characters, not display cells, so a wide character or a byte
order mark before the span can shift the caret on screen.

exit codes: 0 no errors, 1 errors (or the file cannot be read), 2 usage error
";

pub const IR_HELP: &str = "\
usage: pc-cli ir [--] <file.ode>

Loads and lowers the model, then prints n, n_p, n_aux, whether the model is
autonomous and smooth, and the number of live DAG nodes of each entry: F, Aux,
JacU, JacP and the directional entries of orders 2 to 5 with U slots. An entry
that cannot be built (too large, unsupported) gets its own line with the
reason; that does not change the exit code, since `ir` only reports. Diagnostics
and refusals go to stderr with their source position.

options:
  --         the next argument is the file, even if it starts with `-`
  -h, --help print this help

exit codes: 0 the model was lowered (even if an entry could not be built), 1 the
model has errors, is refused by the
IR or cannot be read, 2 usage error
";

pub const CHECK_HELP: &str = "\
usage: pc-cli check-derivs [--points N] [--seed S] [--] <file.ode>

Evaluates every entry at N random points near the initial state and checks
  - the DAG entries against the f64 fallback and against finite differences,
  - the JIT entries against the DAG (bit for bit; a NaN matches any NaN),
  - the JIT entries against the f64 fallback (4 ulp for F and Aux, SPEC section 3 tolerance
    for derivatives).
Prints one line per check. For a model that is not smooth, the finite
difference lines are printed as information and do not set the exit code,
since a kink between the two sides of a difference makes them unreliable.

options:
  --points N  random points per entry, at least 1 (default 20)
  --seed S    seed of the point generator, an unsigned integer (default 1)
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 every counted check passes, 1 a check fails, the model has
errors, is refused by the IR or cannot be read, 2 usage error
";

pub const CONT_HELP: &str = "\
usage: pc-cli cont [--fallback] [--] <file.ode> <problem.json>

Follows a branch of the model and prints the branches as JSON on stdout: for
each branch its points, its special points and why it stopped. Stderr gets one
line per special point.

The problem file is a JSON object:
  type      \"EP\" equilibria, \"LP\" or \"H\" a curve of fold or Hopf points in
            two parameters, \"LC\" periodic orbits, \"LPC\", \"PD\" or \"NS\" a
            curve of fold, period-doubling or Neimark-Sacker points of cycles in
            two parameters, \"HOM\" a curve of orbits homoclinic to a saddle in
            two parameters (required)
  free      name of the continuation parameter, or [a1, a2] for LP, H, LPC, PD,
            NS and HOM, with a1 the parameter of the branch the start point is on
            (required)
  u0        starting state (default: the model's init): one number per state,
            on an LP or H curve the point there, for LC \"hopf\" the equilibrium
            at the Hopf point, for LC \"orbit\" or \"pd\", for LPC, PD and NS and
            for HOM the orbit at the N m + 1 base points of its mesh, point by
            point, then the period (for HOM \"orbit\" the truncation time T)
            (required for those)
  params    object of parameter values by name (default: the model's values)
  options   h0, hmin, hmax, max_steps, tol_x, tol_f, max_newton, direction,
            bounds [lo, hi], detect {lp, bp, h} (EP, LP and H only),
            mesh_adapt (cycles and HOM only: adapt the mesh every that many
            steps, 0 never); engine defaults for the rest
  switch    EP only: name of a special point of the first branch, for example
            \"BP1\": the run switches there and follows the second branch

LC only:
  ntst      number of mesh intervals N, at least 1 (required)
  ncol      collocation points per interval m, 2 to 7 (required)
  start     \"hopf\" from a Hopf point, \"orbit\" from an orbit on the uniform
            mesh, \"pd\" the doubled orbit from a PD point (required)
  mesh      for \"pd\": the mesh of the orbit in u0, from 0 to 1, rising

LPC, PD and NS only:
  ntst      number of mesh intervals N of the curve, at least 1 (required)
  ncol      collocation points per interval m of the curve, 2 to 7 (required)
  mesh      the mesh of the orbit in u0, from 0 to 1, rising (required); the
            orbit is regridded when it differs from the N and m of the curve

HOM only:
  ntst      number of mesh intervals N of the curve, at least 1 (required)
  ncol      collocation points per interval m of the curve, 2 to 7 (required)
  start     \"orbit\" from an orbit on the uniform mesh over [0, 1], with u0
            ending in T; \"cycle\" from a long-period point of an LC run, with
            u0 ending in its period (required)
  mesh      for \"cycle\": the mesh of the orbit in u0, from 0 to 1, rising
  inclination_flips  true to add the inclination flip tests IFS and IFU and
            their labels; false when missing (a solve of the adjoint equation
            at every point, so a run takes longer)

options:
  --fallback  evaluate the model with the f64 interpreter instead of the JIT
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the run finished, 1 the run failed, panicked or was cancelled, or
the model cannot be read, has errors or is refused, 2 usage error or bad
problem file (the message names the key)
";

pub const DDE_ROOTS_HELP: &str = "\
usage: pc-cli dde-roots [--] <problem.json>

Finds the characteristic roots with Re >= re_min of the linear delay equation
  x'(t) = A0 x(t) + sum_j A_j x(t - tau_j)
and prints them as JSON on stdout. Each eigenvalue of a pseudospectral matrix of
order d (n + 1) starts a Newton iteration on det(lambda I - A0 - sum A_j
exp(-lambda tau_j)) = 0. A root is printed only when its scaled residual is at
most 1e-12.

The problem file is a JSON object, and every key is required:
  dim     number of states d, at least 1
  a0      the matrix A0 as d*d numbers, row by row
  delays  array of {\"tau\": delay > 0, \"a\": the matrix A_j as d*d numbers}
  n       number of Chebyshev intervals, at least 1, with d (n + 1) <= 4000
  re_min  smallest real part to report

The output is {\"n\", \"roots\": [{\"re\", \"im\", \"residual\"}], \"unrefined\":
[{\"re\", \"im\"}]}, with the roots by real part, largest first. An entry of
\"unrefined\" is a start value with Re >= re_min whose iteration did not give a
root. A root that no eigenvalue of the matrix approximates is missing without a
message. Raise n until the list stops changing; the strip holds roots of large
imaginary part when re_min is far below zero, and n has to grow with it.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the roots were computed, 1 the computation failed, 2 usage error,
an unreadable file, invalid JSON or a bad problem (the message names the key)
";

pub const DDE_EQ_HELP: &str = "\
usage: pc-cli dde-eq [--] <problem.json>

Reads a model whose equations call delay(x,tau), solves f(x, x, p) = 0 for an
equilibrium by Newton's method, and prints the equilibrium with the characteristic
roots of the linear equation there (see dde-roots). A delay call has a state name
as its first argument and a positive literal, a par name or a number name as its
second. Every other command refuses a model with a delay call.

The problem file is a JSON object, and every key is required:
  model   path of the .ode file, relative to the problem file
  params  object of numbers by parameter name; a parameter left out keeps the value
          of the model file
  x0      start values of Newton's method, one per state
  n       number of Chebyshev intervals for the roots, at least 1, with
          d (n + 1) <= 4000
  re_min  smallest real part to report

The output is {\"x\", \"residual\", \"delays\", \"roots\", \"unrefined\"}. \"x\" is the
equilibrium, \"residual\" the largest entry of f(x, x, p), and \"delays\" the distinct
delay times, ascending. \"roots\" and \"unrefined\" are as for dde-roots. The Newton
tolerances are 1e-10 on the step and on the residual, with at most 6 iterations. A
model that reads t is refused.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the equilibrium and roots were computed, 1 the model has errors or no
delay call, or Newton's method did not converge, 2 usage error, an unreadable
problem file, invalid JSON or a bad problem (the message names the key)
";

pub const DDE_BRANCH_HELP: &str = "\
usage: pc-cli dde-branch [--] <problem.json>

Continues an equilibrium of a model whose equations call delay(x,tau) in one declared
parameter, and reports where characteristic roots cross the imaginary axis. At each point
of the branch it counts the roots with a positive real part (see dde-roots). Where the
count changes between two points it bisects the step. A pair that crosses at +-i w is a
Hopf point, labelled H, and a root that crosses zero comes with an LP or a BP of the
branch. dde-hopf continues a Hopf point in two parameters, and dde-lyap computes the first
Lyapunov coefficient at one.

The problem file is a JSON object. Every key but options is required:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file
  free     name of the declared parameter to continue in; the run starts at its value
           in params or, if params leaves it out, in the model file. A delay time is
           allowed, and then options.bounds must give a lower bound above 0
  x0       start values of Newton's method at the start value of free, one per state
  n        number of Chebyshev intervals for the roots, at least 1, with
           d (n + 1) <= 4000
  re_min   smallest real part to look for, below 0
  options  optional; the keys of the cont problem files: h0, hmin, hmax, max_steps,
           tol_x, tol_f, max_newton, direction, bounds and detect.lp, detect.bp.
           detect.h and mesh_adapt are refused

The output is {\"stop\", \"points\", \"specials\", \"warnings\"}. A point is {\"alpha\",
\"x\", \"unstable\"}. A special is {\"label\", \"alpha\", \"x\", \"omega\", \"between\",
\"unstable_before\", \"unstable_after\"}, in order along the branch; \"omega\" is null for
LP and BP. \"stop\" is one of end, max_steps, bounds, step_too_small, singular, cancelled
and error, as for cont.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the run finished (a run that stops on an error prints what it found and
        the message on stderr), 1 the model has errors or no delay call, or the Newton
        start did not converge, 2 usage error, an unreadable problem file, invalid JSON or
        a bad problem (the message names the key)
";

pub const DDE_LYAP_HELP: &str = "\
usage: pc-cli dde-lyap [--] <problem.json>

Computes the first Lyapunov coefficient l1 at a Hopf point of a model whose equations call
delay(x,tau). The point is an equilibrium x0, the parameter values in params and the angular
frequency omega, for example an H point of dde-branch or a point of dde-hopf. The sign of l1
says whether the Hopf bifurcation is supercritical (l1 < 0) or subcritical (l1 > 0). The
command evaluates l1 at that one point. It does not follow a curve and it does not look for a
generalized Hopf point. The key l2 below adds the second coefficient, which is meant for a
generalized Hopf point, where l1 is zero. The x0 you give is used as it is, and its residual
comes back in the output.

The problem file is a JSON object. Every key but l2 is required:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file
  x0       the equilibrium, one value per state
  omega    the angular frequency of the Hopf point, above 1e-8; l1 changes by about
           86 per unit of omega for the FitzHugh-Nagumo example of the spec, so give
           it to all its digits
  l2       optional boolean, false if left out. true computes the second Lyapunov
           coefficient as well. It does not check that l1 is zero, so read l2 only
           at a point where it is

The output is {\"omega\", \"l1\", \"c1\", \"null_residual\", \"equilibrium_residual\"}.
c1 is [real, imaginary] and l1 is the real part of c1 over omega, with the critical
eigenvector q scaled to a 2-norm of 1. null_residual is the relative size of
Delta(i omega) q, and a value above 1e-4 means omega is not a Hopf frequency at this x0 and
params. With l2 true the output also has \"c2\" ([real, imaginary]) and \"l2\", the real
part of c2 over omega. The keys of the output are in alphabetical order. l2 means something
only at a point where l1 is zero. A point where Delta(0), Delta(2 i omega) or Delta(3 i omega) is
singular is refused.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 l1 was computed (and l2, if asked), 1 the model has errors or no delay call,
        or the engine failed for another reason (a null space of dimension 2 or more at
        Delta(i omega), a singular Delta(0) or Delta(2 i omega) included, and Delta(3 i omega)
        when l2 is true),
        2 usage error, an unreadable problem file, invalid JSON, a bad problem (the message
        names the key) or an omega that is not a Hopf frequency
";

pub const DDE_ZH_HELP: &str = "\
usage: pc-cli dde-zh [--] <problem.json>

Computes the critical normal-form coefficients at a zero-Hopf point of a model whose
equations call delay(x,tau). The point is an equilibrium x0, the parameter values in
params and the angular frequency omega, where the characteristic function has a simple zero
root and a simple pair +-i omega, for example a ZH point of dde-hopf. The command uses the
point as it is. It does not polish x0 or omega, it does not locate zero-Hopf points, and it
does not compute the predictors of the unfolding or say which unfolding scenario the point
has.

The problem file is a JSON object. Every key is required, and an unknown key is an error:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file
  x0       the equilibrium, one value per state
  omega    the angular frequency of the Hopf pair, above 1e-8

The output has the keys e, equilibrium_residual, g011, g021, g110, g111, g200, g210, g300,
null_residual_hopf, null_residual_zero, omega, s, s_product and theta, in alphabetical
order. Each g is [real, imaginary]. s_product is g200 g011, s is its sign (0 if it is
exactly 0), theta is Re g110 / Re g200 and e is the combination of the coefficients of the
paper, section 6.2.1 of arXiv 1903.08276. theta is null where Re g200 is 0, and e is null
where it is not a finite number. The critical eigenvectors have a 2-norm of 1 and the sign
of the real one is fixed by a rule of this program, so g200, g110 and g011 are fixed only up
to that sign. The coefficients g210, g111 and g021 depend on the coordinates; g200, g110,
g011, g300, s_product, theta and e do not. null_residual_zero and null_residual_hopf are the
relative sizes of Delta(0) q0 and Delta(i omega) q1, and a value above 1e-4 means the
point is not a zero-Hopf point.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the coefficients were computed, 1 the model has errors or no delay call, or the
        engine failed for another reason (a zero or imaginary root that is not simple, a null
        space of dimension 2 or more at Delta(0) or Delta(i omega), or a singular
        Delta(2 i omega), included),
        2 usage error, an unreadable problem file, invalid JSON, a bad problem (the message
        names the key) or a point that is not a zero-Hopf point
";

pub const DDE_HH_HELP: &str = "\
usage: pc-cli dde-hh [--] <problem.json>

Computes the critical normal-form coefficients at a Hopf-Hopf point of a model whose
equations call delay(x,tau). The point is an equilibrium x0, the parameter values in params
and two angular frequencies omega1 > omega2, where the characteristic function has a simple
pair +-i omega1 and a simple pair +-i omega2 and no root at 0, for example the crossing of
two Hopf curves of dde-hopf. The command uses the point as it is. It does not polish x0 or
the frequencies, it does not locate Hopf-Hopf points, it does not compute the predictors of
the unfolding, and it does not sort the frequencies for you.

The problem file is a JSON object. Every key is required, and an unknown key is an error:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file
  x0       the equilibrium, one value per state
  omega1   the larger angular frequency, above 1e-8
  omega2   the smaller angular frequency, above 1e-8 and below omega1

The output has the keys case, delta, equilibrium_residual, g0021, g1011, g1110, g2100,
null_residual_1, null_residual_2, omega1, omega2, product, sign and theta, in alphabetical
order. Each g is [real, imaginary]. They are the coefficients of z1 |z1|^2 and z1 |z2|^2 in
the equation of z1 (g2100, g1011) and of z2 |z1|^2 and z2 |z2|^2 in the equation of z2
(g1110, g0021), with the critical eigenvectors scaled to a 2-norm of 1. product is
Re g2100 Re g0021, sign is its sign (0 if it is exactly 0), theta is Re g1011 / Re g0021 and
delta is Re g1110 / Re g2100, the numbers that section 8.3 of arXiv 1903.08276 (with its
supplement S3.2) uses to sort a point into a case. theta and delta are null where the
denominator is 0. A negative product is the difficult case. theta,
delta and the sign of product do not depend on the scale of the eigenvectors, but the g
values and product do. null_residual_1 and null_residual_2 are the
relative sizes of Delta(i omega1) q1 and Delta(i omega2) q2, and a value above 1e-4 means
the frequency is not a root.

case names the unfolding case of the cubic amplitude system in the lettering of Guckenheimer
and Holmes (Ia, Ib, II, III, IVa, IVb, V, VIa, VIb, VIIa, VIIb or VIII), from the signs of
Re g2100, Re g1011, Re g1110 and Re g0021. It is null when one of them is 0 or when
Re g2100 Re g0021 = Re g1011 Re g1110. omega1 is the first oscillator, and swapping the
oscillators exchanges II with III and V with VIII. The label does not say how close the
point is to a boundary between cases. SPEC section 8j-2 lists what is not verified.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the coefficients were computed, 1 the model has errors or no delay call, or the
        engine failed for another reason (a root that is not simple, a null space of dimension
        2 or more at omega1 or omega2, or a singular Delta(0), Delta(2 i omega1),
        Delta(2 i omega2), Delta(i (omega1 + omega2)) or Delta(i (omega1 - omega2)),
        included),
        2 usage error, an unreadable problem file, invalid JSON, a bad problem (the message
        names the key), frequencies that are not in the order omega1 > omega2 or a frequency
        that is not a root of the characteristic function
";

pub const DDE_HOPF_HELP: &str = "\
usage: pc-cli dde-hopf [--] <problem.json>

Continues a Hopf point of a model whose equations call delay(x,tau) in two declared
parameters. The start is a Hopf point that you supply: a state x0, an angular frequency
omega and the values of both parameters in params. The command polishes it with Newton's
method and follows the curve of Hopf points in the plane of the two parameters. On the
way it labels BT, where omega^2 changes sign (Bogdanov-Takens), and ZH, where the
characteristic function has a zero root as well (zero-Hopf). It does not count the roots
to the right of the curve. dde-lyap computes the first Lyapunov coefficient at one point,
and the key lyapunov below does it at every point of the curve. At a GH point it also
computes the second coefficient l2. dde-zh and dde-hh compute the normal-form coefficients
at one ZH point and one HH point.

The problem file is a JSON object. Every key but options is required:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file. The two names in free give the start point
  free     array of two different declared parameter names, [first, second]
  x0       start values of Newton's method for the state, one per state
  omega    start value of the angular frequency, above 1e-8
  lyapunov optional boolean, false if left out. true adds \"l1\" to every point and
           labels GH (generalized Hopf), where l1 changes sign. l1 is null where kappa
           is not above 0 or where dde-lyap would refuse the point. Each point then
           costs one more solve. A GH special then also has \"l2\", the second
           coefficient at the located zero, or null where dde-lyap with l2 would
           refuse it
  hh       optional object {\"n\", \"re_min\"}, both required, off if left out. n is an
           integer of at least 1 and re_min a negative number, the options of
           dde-roots. With it, the characteristic roots are computed at every point
           with kappa above 0, and a sign change of the real part of the second pair
           of roots (the largest one off the axis) is labeled HH (Hopf-Hopf). Each
           point costs one root search. The key \"hh\" of a point is that real part,
           or null where the pair at i omega is not single or there is no second
           pair. An HH special also has \"omega2\", the frequency of the second pair.
           Warnings hh_unrefined and hh_lost count the points where the search
           lost roots, and then n should be raised
  options  optional; the keys of the cont problem files: h0, hmin, hmax, max_steps,
           tol_x, tol_f, max_newton, direction, bounds. bounds limits the second name
           in free, and a delay time there needs a lower bound above 0. detect and
           mesh_adapt are refused

The output is {\"stop\", \"points\", \"specials\", \"warnings\"}. A point is {\"alpha\",
\"x\", \"kappa\", \"omega\"} plus \"l1\" when lyapunov is true and \"hh\" when hh is
given, where alpha is [first, second], kappa is omega^2 and \"omega\" is null where kappa
is not above 0. A special is {\"label\", \"alpha\", \"x\", \"kappa\", \"index\"}, plus \"l2\"
on a GH special when lyapunov is true and \"omega2\" on an HH special when hh is given,
in order along the curve. \"stop\" is one of end, max_steps, bounds,
step_too_small, singular, cancelled and error, as for cont.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the run finished (a run that stops on an error prints what it found and
        the message on stderr), 1 the model has errors or no delay call, or the Newton
        start did not converge, 2 usage error, an unreadable problem file, invalid JSON or
        a bad problem (the message names the key)
";

pub const DDE_CYCLE_HELP: &str = "\
usage: pc-cli dde-cycle [--] <problem.json>

Corrects a periodic orbit of a model whose equations call delay(x,tau). The orbit is a
piecewise polynomial of degree ncol on ntst intervals, fixed by Gauss collocation, and
Newton's method solves for the orbit, its period and one declared parameter in free. The
phase condition is an integral condition against the start profile. The system has
(ntst*ncol+1)*n + 2 unknowns, at most 3000. With the floquet key it also prints the
leading Floquet multipliers of the corrected orbit. With the branch key it continues the
corrected orbit in the free parameter instead and labels LPC points, and with the floquet
key also PD and NS points. With the adapt key the mesh moves along the branch, and with
the bpc key it also labels branch points of cycles.

The problem file is a JSON object. Every key but options is required:
  model    path of the .ode file, relative to the problem file
  params   object of numbers by parameter name; a parameter left out keeps the value
           of the model file
  free     the name of one declared parameter, which Newton's method adjusts. With
           hopf, params must give its value at the Hopf point
  ntst     number of mesh intervals, an integer of at least 1
  ncol     collocation points per interval, an integer from 2 to 7
  orbit    the start, given as an object {\"profile\", \"period\"}, or
  hopf     a start from a Hopf point, given as an object {\"x0\", \"omega\", \"eps\"}.
           Give exactly one of orbit and hopf
  options  optional; without branch only max_newton, tol_x and tol_f are read and any
           other key is refused. With branch the keys of the cont command are read,
           except detect and mesh_adapt, which are refused (the adapt key moves the
           mesh)
  floquet  optional; an object {\"nhist\", \"count\"}, both integers of at least 1, that
           asks for the multipliers. There is no default and the key is off without it.
           With branch it adds the multipliers to every point and special point, and
           turns on the PD and NS labels
  branch   optional; true continues the orbit in the free parameter, false or no key
           prints the corrected orbit. There is no default and the key is off without it
  adapt    optional, and only with branch; an object {\"every\"} with an integer of at
           least 1 that moves the mesh after every that many accepted points. There is no
           default for every and the key is off without it. It needs ntst of at least 2
  bpc      optional, and only with branch; true labels branch points of cycles as BPC,
           false or no key does not. There is no default and the key is off without it.
           The points of the run are the same with and without it

orbit.profile holds the state at ntst*ncol+1 equally spaced points of one period, point
after point with n values each. orbit.period is the start period. hopf.x0 is the
equilibrium at the Hopf point, hopf.omega the angular frequency there and hopf.eps the
amplitude of the orbit, measured along the critical vector. It has no default and Newton's
method keeps it fixed. The first guess is x0 + eps*Re(q exp(2 pi i s)) with the critical
vector q and the period 2 pi/omega. The command does not check that x0 is an equilibrium or
that omega is a root.

The output is {\"alpha\", \"period\", \"iters\", \"residual\", \"ntst\", \"ncol\", \"profile\"},
where alpha is the value of the free parameter, iters is the number of Newton steps and
residual is the largest entry of the final residual vector.

With floquet, the output also has \"multipliers\", an array of [re, im] pairs, and
\"trivial_error\". The history of the delay equation is stored at nhist+1 Chebyshev points,
which gives n*(nhist+1) multipliers, and count of them are printed, largest modulus first
(1 <= count <= n*(nhist+1), and ncol*n*(nhist+1) at most 1000). A cut can fall inside a
conjugate pair. The multipliers of small modulus come from the discretization and mean
nothing. trivial_error is the distance of the multiplier nearest to 1 from 1, which is the
multiplier that every periodic orbit has, so it measures the discretization error.
The multipliers are taken at the corrected value of the free parameter.

With branch set to true, the start is first corrected by Newton's method with free held at
the value that params gives (for hopf, the Hopf corrector runs first and finds that value;
a bad floquet value is refused before either runs), and pseudo-arclength continuation
then follows the same equations on the same mesh. The output is
{\"ntst\", \"ncol\", \"stop\", \"points\", \"specials\", \"warnings\"}. Each point is {\"alpha\",
\"period\", \"profile\", \"h\", \"iters\"}: the first has h and iters 0 and is the corrected
start. stop is one of end, max_steps, bounds, step_too_small, singular, cancelled or
error. The run reads h0, hmin, hmax, max_steps, tol_x, tol_f, max_newton, direction and
bounds from options as cont does. The mesh stays uniform and fixed unless adapt is given.
When free is a delay time, bounds must have a lower bound above 0.

With adapt, the start is corrected on the uniform mesh, and after every `every` accepted
points (and once after a failed corrector) the mesh points move so that they equidistribute
the same monitor function as the ODE cycle continuation, which puts more of them where the
orbit is steep. The profile is interpolated onto the new mesh. Each point and each special point also has \"mesh\", the
ntst+1 mesh points of that point in [0, 1], and \"profile\" holds the values at that mesh.
The Floquet multipliers of an adapted point use its own mesh. The first point's mesh is
uniform. Without adapt there is no \"mesh\" key and the output is the one described above.

specials is an array in branch order, empty if there is none. Each element is {\"label\",
\"index\", \"alpha\", \"period\", \"profile\"}, where index is the index of the first point
at or after the special point. An LPC (limit point of cycles) is labeled on every run with
branch. With floquet the points and the special points also have \"multipliers\" and
\"trivial_error\" as above, and PD (period doubling) and NS (Neimark-Sacker) are labeled
too. The tests use all n*(nhist+1) multipliers, not only the count that is printed, and
the small ones that come from the discretization are assumed to stay small. Each point of
the run builds the monodromy matrix, so a run with floquet costs more.

With bpc, a BPC (branch point of cycles) is labeled where the sign of the determinant of
the bordered Jacobian changes. That happens at a branch point where an odd number of
branches cross, and not at a fold, a PD or an NS. A branch point where an even number
cross is missed.

With switch, an object with the keys special and direction (both required) that needs
branch and cannot be combined with adapt, the command goes on along the second branch at a
BPC. special is the index of the BPC in the specials of a first run, a non-negative integer,
and direction is 1 or -1. The first run is the run with bpc and the problem's options, with
no floquet and no adapt. The second run starts at that BPC, goes along the other branch in
the given direction (options.direction applies to the first run only) and uses floquet and
bpc as the problem gives them, off if absent. The output is the one of the second run, with
one more key, \"switch\": {\"special\", \"alpha\", \"period\"}, the index, the parameter value
and the period of the BPC. A special that is out of range or is not a BPC, and a BPC that is
not simple, are errors (exit 2). The start of the second run is not corrected, the BPC at
its start is not labeled again, and bpc still labels the later branch points on the new
branch.

options:
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the orbit converged (with branch, the run finished, and a run that stops on
        an error prints what it found and the message on stderr), 1 the model has errors or
        no delay call, or Newton's method did not converge at the start, 2 usage error, an
        unreadable problem file, invalid JSON or a bad problem (the message names the key)
";

pub const DLL_IMPORTS_HELP: &str = "\
usage: pc-cli dll-imports [--] <file.dll>

Reads a 64-bit Windows PE file (the cross-built pckern.dll) and prints the DLLs
it imports, one per line in table order, each once. Delay-loaded imports follow
with the suffix \" (delay)\". The names are compared without regard to case with
an allow-list: kernel32, ntdll, userenv, ws2_32, bcryptprimitives, bcrypt,
advapi32, msvcrt and api-ms-win-*. A name outside the list is reported on stderr.

options:
  --         the next argument is the file, even if it starts with `-`
  -h, --help print this help

exit codes: 0 every import is allowed, 1 an import is outside the allow-list or
the file cannot be read or is not a 64-bit PE file, 2 usage error
";

pub const LICENSES_HELP: &str = "\
usage: pc-cli licenses --target TRIPLE... [--package NAME] --out FILE [--overrides DIR]

Runs `cargo metadata --locked` with one `--filter-platform` per target, takes
the dependency closure of the package on all of them without dev-dependencies
and writes, for each crate that is not a workspace member, a header line
`== name version (SPDX) ==` and the text of every file named LICENSE*,
LICENCE*, COPYING* or NOTICE* in the crate folder. Crates are sorted by name,
then version. Workspace members are covered by the repo LICENSE and are left
out.

options:
  --target TRIPLE  Rust target triple (required; repeat it for more targets)
  --package NAME   package whose closure is listed (default pc-ffi)
  --out FILE       file to write (required); not written on failure
  --overrides DIR  folder with one subfolder per crate that ships no license file;
                   its LICENSE*, LICENCE*, COPYING* and NOTICE* files and its SOURCE file
                   (upstream URL and tag) stand in for the missing files (no default)
  -h, --help       print this help

exit codes: 0 the file was written, 1 cargo failed, the package is unknown, a
crate ships no license file or the file cannot be written, 2 usage error
";

pub const LATEX_HELP: &str = "\
usage: pc-cli latex pgfplots|table [--standalone] [--out FILE.tex] [--] <input.json>

Writes LaTeX for a figure (pgfplots) or a table of special points (booktabs)
from a JSON document. The document must name `digits`, the number of
significant digits to print (1 to 17); nothing is filled in for it. A value
that is null, NaN or infinite is an error that names the series and the point.

pgfplots keys: xlabel, ylabel, digits (required); title, data (\"inline\" or
\"external\"), basename (required for external), series[] with class (\"stable\" or
\"unstable\"), x[], y[] and optional name and color (#RRGGBB), marks[] with
label, x, y. table keys: columns[], rows[] with label, type, values[], digits.
Unknown keys are errors. See docs/SPEC.md section 13.

Without --out the text goes to stdout, which works for inline data only. With
external data, --out is required: the data files BASENAME-s1.dat, BASENAME-s2.dat,
... are written into the folder of the output file.

options:
  --standalone  write a whole `standalone` document instead of a fragment for \\input
  --out FILE    write the text to FILE instead of stdout
  --            the next argument is the input file, even if it starts with `-`
  -h, --help    print this help

exit codes: 0 the text was written, 1 the document is invalid, a file cannot be
read or written, or external data has no --out, 2 usage error
";

pub const EMIT_HELP: &str = "\
usage: pc-cli emit-c [-o out.c] [--] <file.ode>

Writes C code with the functions F, JacU and JacP (JacP has no outputs when
the model has no parameters) and pc_aot_init. Without -o the code goes to
stdout. Diagnostics and refusals go to stderr.

options:
  -o FILE    write the code to FILE instead of stdout
  --         the next argument is the file, even if it starts with `-`
  -h, --help print this help

exit codes: 0 the code was written, 1 the model has errors, is refused by the
IR, an entry cannot be built, or a file cannot be read or written, 2 usage error
";

fn utf8(arg: &OsString) -> Result<&str, UsageError> {
    arg.to_str().ok_or_else(|| {
        UsageError(format!(
            "argument {} is not valid UTF-8",
            arg.to_string_lossy()
        ))
    })
}

/// Parses the arguments after the program name.
pub fn parse_args<I>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = OsString>,
{
    let args: Vec<OsString> = args.into_iter().collect();
    let Some(first) = args.first() else {
        return Err(UsageError("no command given".to_string()));
    };
    let first = utf8(first)?;
    match first {
        "-h" | "--help" if args.len() == 1 => Ok(Command::Help),
        "-V" | "--version" if args.len() == 1 => Ok(Command::Version),
        "parse" => parse_command(&args[1..]),
        "ir" => ir_command(&args[1..]),
        "check-derivs" => check_command(&args[1..]),
        "emit-c" => emit_command(&args[1..]),
        "cont" => cont_command(&args[1..]),
        "dde-roots" => dde_roots_command(&args[1..]),
        "dde-eq" => dde_eq_command(&args[1..]),
        "dde-branch" => dde_branch_command(&args[1..]),
        "dde-hopf" => dde_hopf_command(&args[1..]),
        "dde-cycle" => dde_cycle_command(&args[1..]),
        "dde-lyap" => dde_lyap_command(&args[1..]),
        "dde-zh" => dde_zh_command(&args[1..]),
        "dde-hh" => dde_hh_command(&args[1..]),
        "session" => session_command(&args[1..]),
        "gpu" => gpu_command(&args[1..]),
        "ensemble" => ensemble_command(&args[1..]),
        "dll-imports" => dll_imports_command(&args[1..]),
        "licenses" => licenses_command(&args[1..]),
        "latex" => latex_command(&args[1..]),
        "-h" | "--help" | "-V" | "--version" => {
            Err(UsageError(format!("unexpected argument after {first}")))
        }
        other if other.starts_with('-') => Err(UsageError(format!("unknown option {other}"))),
        other => Err(UsageError(format!("unknown command {other}"))),
    }
}

/// What one command's arguments held, before the command checks them.
#[derive(Default)]
struct Parts {
    file: Option<PathBuf>,
    json: bool,
    help: bool,
    points: Option<usize>,
    seed: Option<u64>,
    out: Option<PathBuf>,
    problem: Option<PathBuf>,
    fallback: bool,
}

/// Which flags a command accepts.
#[derive(Clone, Copy, Default)]
struct Accepts {
    json: bool,
    points_seed: bool,
    out: bool,
    cont: bool,
}

/// The value after a flag that takes one.
fn value<'a>(
    flag: &str,
    it: &mut impl Iterator<Item = &'a OsString>,
) -> Result<&'a OsString, UsageError> {
    it.next()
        .ok_or_else(|| UsageError(format!("{flag} needs a value")))
}

fn number<T: std::str::FromStr>(flag: &str, arg: &OsString) -> Result<T, UsageError> {
    let text = utf8(arg)?;
    text.parse().map_err(|_| {
        UsageError(format!(
            "{flag} needs an unsigned integer, got {}",
            arg.to_string_lossy()
        ))
    })
}

/// Splits the arguments of one command into its flags and its file.
fn split(rest: &[OsString], ok: Accepts) -> Result<Parts, UsageError> {
    let mut parts = Parts::default();
    let mut only_files = false;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        let flag = if only_files { None } else { arg.to_str() };
        match flag {
            Some("--") => only_files = true,
            Some("--json") if ok.json => parts.json = true,
            Some("--fallback") if ok.cont => parts.fallback = true,
            Some("-h" | "--help") => parts.help = true,
            Some("--points") if ok.points_seed => {
                let n: usize = number("--points", value("--points", &mut it)?)?;
                if n == 0 {
                    return Err(UsageError("--points must be at least 1".to_string()));
                }
                parts.points = Some(n);
            }
            Some("--seed") if ok.points_seed => {
                parts.seed = Some(number("--seed", value("--seed", &mut it)?)?);
            }
            Some("-o") if ok.out => {
                let v = value("-o", &mut it)?;
                if v.as_encoded_bytes().first() == Some(&b'-') {
                    return Err(UsageError(format!(
                        "-o needs a file name, got {}",
                        v.to_string_lossy()
                    )));
                }
                parts.out = Some(PathBuf::from(v));
            }
            Some(other) if other.starts_with('-') && other.len() > 1 => {
                return Err(UsageError(format!("unknown option {other}")));
            }
            _ => {
                // Not a flag: a non-UTF-8 argument can only be the file.
                if !only_files
                    && arg.to_str().is_none()
                    && arg.as_encoded_bytes().first() == Some(&b'-')
                {
                    return Err(UsageError(format!(
                        "argument {} is not valid UTF-8",
                        arg.to_string_lossy()
                    )));
                }
                if parts.file.is_some() && ok.cont && parts.problem.is_none() {
                    parts.problem = Some(PathBuf::from(arg));
                    continue;
                }
                if parts.file.is_some() {
                    return Err(UsageError(format!(
                        "unexpected extra argument {}",
                        arg.to_string_lossy()
                    )));
                }
                parts.file = Some(PathBuf::from(arg));
            }
        }
    }
    Ok(parts)
}

fn parse_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(
        rest,
        Accepts {
            json: true,
            ..Accepts::default()
        },
    )?;
    if parts.help {
        return Ok(Command::ToolHelp(PARSE_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::Parse {
            file,
            json: parts.json,
        }),
        None => Err(UsageError("parse needs a file".to_string())),
    }
}

fn ir_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(IR_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::Ir { file }),
        None => Err(UsageError("ir needs a file".to_string())),
    }
}

fn check_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(
        rest,
        Accepts {
            points_seed: true,
            ..Accepts::default()
        },
    )?;
    if parts.help {
        return Ok(Command::ToolHelp(CHECK_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::CheckDerivs {
            file,
            points: parts.points.unwrap_or(DEFAULT_POINTS),
            seed: parts.seed.unwrap_or(DEFAULT_SEED),
        }),
        None => Err(UsageError("check-derivs needs a file".to_string())),
    }
}

fn emit_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(
        rest,
        Accepts {
            out: true,
            ..Accepts::default()
        },
    )?;
    if parts.help {
        return Ok(Command::ToolHelp(EMIT_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::EmitC {
            file,
            out: parts.out,
        }),
        None => Err(UsageError("emit-c needs a file".to_string())),
    }
}

fn dde_roots_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_ROOTS_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeRoots { file }),
        None => Err(UsageError("dde-roots needs a problem file".to_string())),
    }
}

fn dde_eq_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_EQ_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeEq { file }),
        None => Err(UsageError("dde-eq needs a problem file".to_string())),
    }
}

fn dde_branch_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_BRANCH_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeBranch { file }),
        None => Err(UsageError("dde-branch needs a problem file".to_string())),
    }
}

fn dde_hh_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_HH_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeHh { file }),
        None => Err(UsageError("dde-hh needs a problem file".to_string())),
    }
}

fn dde_zh_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_ZH_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeZh { file }),
        None => Err(UsageError("dde-zh needs a problem file".to_string())),
    }
}

fn dde_lyap_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_LYAP_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeLyap { file }),
        None => Err(UsageError("dde-lyap needs a problem file".to_string())),
    }
}

fn dde_hopf_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_HOPF_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeHopf { file }),
        None => Err(UsageError("dde-hopf needs a problem file".to_string())),
    }
}

fn dde_cycle_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DDE_CYCLE_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DdeCycle { file }),
        None => Err(UsageError("dde-cycle needs a problem file".to_string())),
    }
}

fn dll_imports_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(rest, Accepts::default())?;
    if parts.help {
        return Ok(Command::ToolHelp(DLL_IMPORTS_HELP));
    }
    match parts.file {
        Some(file) => Ok(Command::DllImports { file }),
        None => Err(UsageError("dll-imports needs a file".to_string())),
    }
}

/// Parses the arguments of `session`. `replay` is the only subcommand.
fn session_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut it = rest.iter();
    match it.next().map(utf8).transpose()? {
        Some("-h" | "--help") => return Ok(Command::ToolHelp(SESSION_HELP)),
        Some("replay") => {}
        Some(other) => {
            return Err(UsageError(format!(
                "unknown session command {other}, expected replay"
            )));
        }
        None => return Err(UsageError("session needs a subcommand: replay".to_string())),
    }
    let mut file: Option<PathBuf> = None;
    let mut tol: Option<f64> = None;
    let mut fallback = false;
    let mut literal = false;
    while let Some(arg) = it.next() {
        let text = utf8(arg)?;
        if literal || !text.starts_with('-') || text == "-" {
            if file.replace(PathBuf::from(arg)).is_some() {
                return Err(UsageError(format!("unexpected extra argument {text}")));
            }
            continue;
        }
        match text {
            "--" => literal = true,
            "-h" | "--help" => return Ok(Command::ToolHelp(SESSION_HELP)),
            "--fallback" => fallback = true,
            "--tol" => {
                let v = utf8(value("--tol", &mut it)?)?;
                let x: f64 = v.parse().map_err(|_| {
                    UsageError(format!("--tol needs a non-negative number, got {v}"))
                })?;
                if !(x.is_finite() && x >= 0.0) {
                    return Err(UsageError(format!(
                        "--tol needs a finite number of at least 0, got {v}"
                    )));
                }
                tol = Some(x);
            }
            other => return Err(UsageError(format!("unknown option {other}"))),
        }
    }
    match file {
        Some(file) => Ok(Command::SessionReplay {
            file,
            tol: tol.unwrap_or(DEFAULT_TOL),
            fallback,
        }),
        None => Err(UsageError(
            "session replay needs a session file".to_string(),
        )),
    }
}

/// Parses the arguments of `licenses`. `--out` and at least one `--target` are
/// required; `--target` may repeat.
fn licenses_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut targets: Vec<String> = Vec::new();
    let mut package: Option<String> = None;
    let mut overrides: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match utf8(arg)? {
            "-h" | "--help" => return Ok(Command::ToolHelp(LICENSES_HELP)),
            "--target" => targets.push(utf8(value("--target", &mut it)?)?.to_string()),
            "--package" => package = Some(utf8(value("--package", &mut it)?)?.to_string()),
            "--out" => out = Some(PathBuf::from(value("--out", &mut it)?)),
            "--overrides" => overrides = Some(PathBuf::from(value("--overrides", &mut it)?)),
            other if other.starts_with('-') => {
                return Err(UsageError(format!("unknown option {other}")));
            }
            other => {
                return Err(UsageError(format!("unexpected extra argument {other}")));
            }
        }
    }
    match (targets.is_empty(), out) {
        (false, Some(out)) => Ok(Command::Licenses {
            targets,
            package: package.unwrap_or_else(|| "pc-ffi".to_string()),
            out,
            overrides,
        }),
        _ => Err(UsageError("licenses needs --target and --out".to_string())),
    }
}

/// Parses the arguments of `latex`: a kind, an input file and the flags.
fn latex_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut it = rest.iter();
    let kind = match it.next().map(utf8).transpose()? {
        Some("-h" | "--help") => return Ok(Command::ToolHelp(LATEX_HELP)),
        Some(name) => pc_latex::Kind::from_name(name).ok_or_else(|| {
            UsageError(format!(
                "unknown latex kind {name}, expected pgfplots or table"
            ))
        })?,
        None => {
            return Err(UsageError(
                "latex needs a kind: pgfplots or table".to_string(),
            ));
        }
    };
    let mut file: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut standalone = false;
    let mut literal = false;
    while let Some(arg) = it.next() {
        let text = utf8(arg)?;
        if literal || !text.starts_with('-') || text == "-" {
            if file.replace(PathBuf::from(arg)).is_some() {
                return Err(UsageError(format!("unexpected extra argument {text}")));
            }
            continue;
        }
        match text {
            "--" => literal = true,
            "-h" | "--help" => return Ok(Command::ToolHelp(LATEX_HELP)),
            "--standalone" => standalone = true,
            "--out" | "-o" => out = Some(PathBuf::from(value(text, &mut it)?)),
            other => return Err(UsageError(format!("unknown option {other}"))),
        }
    }
    match file {
        Some(file) => Ok(Command::Latex {
            kind,
            file,
            out,
            standalone,
        }),
        None => Err(UsageError("latex needs an input file".to_string())),
    }
}

/// Parses the arguments of `gpu`: `list` and `selftest`.
fn gpu_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut it = rest.iter();
    let sub = match it.next().map(utf8).transpose()? {
        Some("-h" | "--help") => return Ok(Command::ToolHelp(GPU_HELP)),
        Some(s @ ("list" | "selftest")) => s,
        Some(other) => {
            return Err(UsageError(format!(
                "unknown gpu command {other}, expected list or selftest"
            )));
        }
        None => {
            return Err(UsageError(
                "gpu needs a subcommand: list or selftest".to_string(),
            ));
        }
    };
    let mut backend: Option<Backend> = None;
    let mut device: Option<usize> = None;
    let mut precision: Option<Precision> = None;
    while let Some(arg) = it.next() {
        let text = utf8(arg)?;
        match (sub, text) {
            (_, "-h" | "--help") => return Ok(Command::ToolHelp(GPU_HELP)),
            ("selftest", "--backend") => {
                let v = utf8(value("--backend", &mut it)?)?;
                backend = Some(Backend::from_name(v).ok_or_else(|| {
                    UsageError(format!(
                        "--backend needs cpu, cuda, opencl or metal, got {v}"
                    ))
                })?);
            }
            ("selftest", "--device") => {
                device = Some(number("--device", value("--device", &mut it)?)?);
            }
            ("selftest", "--precision") => {
                let v = utf8(value("--precision", &mut it)?)?;
                precision = Some(precision_from_name(v).ok_or_else(|| {
                    UsageError(format!("--precision needs single or double, got {v}"))
                })?);
            }
            (_, other) if other.starts_with('-') => {
                return Err(UsageError(format!("unknown option {other}")));
            }
            (_, other) => {
                return Err(UsageError(format!("unexpected extra argument {other}")));
            }
        }
    }
    if sub == "list" {
        return Ok(Command::GpuList);
    }
    match (backend, precision) {
        (Some(backend), Some(precision)) => Ok(Command::GpuSelftest {
            backend,
            device: device.unwrap_or(0),
            precision,
        }),
        (None, _) => Err(UsageError(
            "gpu selftest needs --backend (cpu, cuda, opencl or metal)".to_string(),
        )),
        (_, None) => Err(UsageError(
            "gpu selftest needs --precision (single or double)".to_string(),
        )),
    }
}

/// Parses the arguments of `ensemble`: a model and a job file, or `--bench`.
fn ensemble_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut bench = false;
    let mut members: Option<Vec<usize>> = None;
    let mut literal = false;
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        let flag = if literal { None } else { arg.to_str() };
        match flag {
            Some("--") => literal = true,
            Some("-h" | "--help") => return Ok(Command::ToolHelp(ENSEMBLE_HELP)),
            Some("--bench") => bench = true,
            Some("--members") => {
                let v = utf8(value("--members", &mut it)?)?;
                members = Some(member_list(v)?);
            }
            Some(other) if other.starts_with('-') && other.len() > 1 => {
                return Err(UsageError(format!("unknown option {other}")));
            }
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if bench {
        if !files.is_empty() {
            return Err(UsageError(
                "ensemble --bench takes no model or job file".to_string(),
            ));
        }
        return Ok(Command::EnsembleBench {
            members: members.unwrap_or_else(|| DEFAULT_BENCH_MEMBERS.to_vec()),
        });
    }
    if members.is_some() {
        return Err(UsageError("--members only goes with --bench".to_string()));
    }
    let mut files = files.into_iter();
    match (files.next(), files.next(), files.next()) {
        (Some(file), Some(job), None) => Ok(Command::Ensemble { file, job }),
        (_, _, Some(extra)) => Err(UsageError(format!(
            "unexpected extra argument {}",
            extra.display()
        ))),
        _ => Err(UsageError(
            "ensemble needs a model file and a job file, or --bench".to_string(),
        )),
    }
}

/// `10,2000,5`: positive integers separated by commas.
fn member_list(text: &str) -> Result<Vec<usize>, UsageError> {
    text.split(',')
        .map(|part| match part.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(n),
            _ => Err(UsageError(format!(
                "--members needs integers of at least 1 separated by commas, got {text}"
            ))),
        })
        .collect()
}

fn cont_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let parts = split(
        rest,
        Accepts {
            cont: true,
            ..Accepts::default()
        },
    )?;
    if parts.help {
        return Ok(Command::ToolHelp(CONT_HELP));
    }
    match (parts.file, parts.problem) {
        (Some(file), Some(problem)) => Ok(Command::Cont {
            file,
            problem,
            fallback: parts.fallback,
        }),
        _ => Err(UsageError(
            "cont needs a model file and a problem file".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(args: &[&str]) -> Result<Command, UsageError> {
        parse_args(args.iter().map(OsString::from))
    }

    fn parse(file: &str, json: bool) -> Command {
        Command::Parse {
            file: file.into(),
            json,
        }
    }

    #[test]
    fn parse_with_flag_before_or_after() {
        assert_eq!(p(&["parse", "a.ode", "--json"]), Ok(parse("a.ode", true)));
        assert_eq!(p(&["parse", "--json", "a.ode"]), Ok(parse("a.ode", true)));
        assert_eq!(p(&["parse", "a.ode"]), Ok(parse("a.ode", false)));
    }

    #[test]
    fn double_dash_makes_the_next_argument_the_file() {
        assert_eq!(p(&["parse", "--", "-x.ode"]), Ok(parse("-x.ode", false)));
        assert_eq!(
            p(&["parse", "--json", "--", "--json"]),
            Ok(parse("--json", true))
        );
        assert!(p(&["parse", "--", "a.ode", "b.ode"]).is_err());
        assert!(p(&["parse", "--"]).is_err());
    }

    #[test]
    fn help_and_version() {
        assert_eq!(p(&["--help"]), Ok(Command::Help));
        assert_eq!(p(&["-h"]), Ok(Command::Help));
        assert_eq!(p(&["--version"]), Ok(Command::Version));
        assert_eq!(p(&["-V"]), Ok(Command::Version));
        assert_eq!(p(&["parse", "--help"]), Ok(Command::ToolHelp(PARSE_HELP)));
        assert_eq!(
            p(&["parse", "x.ode", "-h"]),
            Ok(Command::ToolHelp(PARSE_HELP))
        );
    }

    #[test]
    fn ir_check_and_emit_commands() {
        assert_eq!(
            p(&["ir", "a.ode"]),
            Ok(Command::Ir {
                file: "a.ode".into()
            })
        );
        assert_eq!(
            p(&["check-derivs", "a.ode"]),
            Ok(Command::CheckDerivs {
                file: "a.ode".into(),
                points: DEFAULT_POINTS,
                seed: DEFAULT_SEED
            })
        );
        assert_eq!(
            p(&["check-derivs", "--seed", "7", "a.ode", "--points", "3"]),
            Ok(Command::CheckDerivs {
                file: "a.ode".into(),
                points: 3,
                seed: 7
            })
        );
        assert_eq!(
            p(&["emit-c", "a.ode"]),
            Ok(Command::EmitC {
                file: "a.ode".into(),
                out: None
            })
        );
        assert_eq!(
            p(&["emit-c", "-o", "o.c", "a.ode"]),
            Ok(Command::EmitC {
                file: "a.ode".into(),
                out: Some("o.c".into())
            })
        );
        assert_eq!(p(&["ir", "--help"]), Ok(Command::ToolHelp(IR_HELP)));
        assert_eq!(
            p(&["check-derivs", "x.ode", "-h"]),
            Ok(Command::ToolHelp(CHECK_HELP))
        );
        assert_eq!(p(&["emit-c", "-h"]), Ok(Command::ToolHelp(EMIT_HELP)));
    }

    #[test]
    fn dll_imports_command_takes_one_file() {
        assert_eq!(
            p(&["dll-imports", "a.dll"]),
            Ok(Command::DllImports {
                file: "a.dll".into()
            })
        );
        assert_eq!(
            p(&["dll-imports", "--", "-a.dll"]),
            Ok(Command::DllImports {
                file: "-a.dll".into()
            })
        );
        assert_eq!(
            p(&["dll-imports", "-h"]),
            Ok(Command::ToolHelp(DLL_IMPORTS_HELP))
        );
        for args in [
            &["dll-imports"][..],
            &["dll-imports", "a.dll", "b.dll"],
            &["dll-imports", "--json", "a.dll"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_roots_command_takes_one_file() {
        assert_eq!(
            p(&["dde-roots", "p.json"]),
            Ok(Command::DdeRoots {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-roots", "--", "-p.json"]),
            Ok(Command::DdeRoots {
                file: "-p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-roots", "--help"]),
            Ok(Command::ToolHelp(DDE_ROOTS_HELP))
        );
        for args in [
            &["dde-roots"][..],
            &["dde-roots", "a.json", "b.json"],
            &["dde-roots", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_eq_command_takes_one_file() {
        assert_eq!(
            p(&["dde-eq", "p.json"]),
            Ok(Command::DdeEq {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-eq", "--", "-p.json"]),
            Ok(Command::DdeEq {
                file: "-p.json".into()
            })
        );
        assert_eq!(p(&["dde-eq", "--help"]), Ok(Command::ToolHelp(DDE_EQ_HELP)));
        for args in [
            &["dde-eq"][..],
            &["dde-eq", "a.json", "b.json"],
            &["dde-eq", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_branch_command_takes_one_file() {
        assert_eq!(
            p(&["dde-branch", "p.json"]),
            Ok(Command::DdeBranch {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-branch", "--", "-p.json"]),
            Ok(Command::DdeBranch {
                file: "-p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-branch", "--help"]),
            Ok(Command::ToolHelp(DDE_BRANCH_HELP))
        );
        for args in [
            &["dde-branch"][..],
            &["dde-branch", "a.json", "b.json"],
            &["dde-branch", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_lyap_command_takes_one_file() {
        assert_eq!(
            p(&["dde-lyap", "p.json"]),
            Ok(Command::DdeLyap {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-lyap", "--", "-p.json"]),
            Ok(Command::DdeLyap {
                file: "-p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-lyap", "--help"]),
            Ok(Command::ToolHelp(DDE_LYAP_HELP))
        );
        for args in [
            &["dde-lyap"][..],
            &["dde-lyap", "a.json", "b.json"],
            &["dde-lyap", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_hh_command_takes_one_file() {
        assert_eq!(
            p(&["dde-hh", "p.json"]),
            Ok(Command::DdeHh {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-hh", "--", "-p.json"]),
            Ok(Command::DdeHh {
                file: "-p.json".into()
            })
        );
        assert_eq!(p(&["dde-hh", "--help"]), Ok(Command::ToolHelp(DDE_HH_HELP)));
        for args in [
            &["dde-hh"][..],
            &["dde-hh", "a.json", "b.json"],
            &["dde-hh", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_zh_command_takes_one_file() {
        assert_eq!(
            p(&["dde-zh", "p.json"]),
            Ok(Command::DdeZh {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-zh", "--", "-p.json"]),
            Ok(Command::DdeZh {
                file: "-p.json".into()
            })
        );
        assert_eq!(p(&["dde-zh", "--help"]), Ok(Command::ToolHelp(DDE_ZH_HELP)));
        for args in [
            &["dde-zh"][..],
            &["dde-zh", "a.json", "b.json"],
            &["dde-zh", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_cycle_command_takes_one_file() {
        assert_eq!(
            p(&["dde-cycle", "p.json"]),
            Ok(Command::DdeCycle {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-cycle", "--", "-p.json"]),
            Ok(Command::DdeCycle {
                file: "-p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-cycle", "--help"]),
            Ok(Command::ToolHelp(DDE_CYCLE_HELP))
        );
        for args in [
            &["dde-cycle"][..],
            &["dde-cycle", "a.json", "b.json"],
            &["dde-cycle", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn dde_hopf_command_takes_one_file() {
        assert_eq!(
            p(&["dde-hopf", "p.json"]),
            Ok(Command::DdeHopf {
                file: "p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-hopf", "--", "-p.json"]),
            Ok(Command::DdeHopf {
                file: "-p.json".into()
            })
        );
        assert_eq!(
            p(&["dde-hopf", "--help"]),
            Ok(Command::ToolHelp(DDE_HOPF_HELP))
        );
        for args in [
            &["dde-hopf"][..],
            &["dde-hopf", "a.json", "b.json"],
            &["dde-hopf", "--fallback", "a.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn licenses_command_needs_target_and_out() {
        assert_eq!(
            p(&["licenses", "--out", "o.txt", "--target", "t"]),
            Ok(Command::Licenses {
                targets: vec!["t".into()],
                package: "pc-ffi".into(),
                out: "o.txt".into(),
                overrides: None
            })
        );
        assert_eq!(
            p(&["licenses", "--target", "t", "--out", "o", "--package", "x"]),
            Ok(Command::Licenses {
                targets: vec!["t".into()],
                package: "x".into(),
                out: "o".into(),
                overrides: None
            })
        );
        assert_eq!(
            p(&[
                "licenses",
                "--target",
                "t",
                "--out",
                "o",
                "--overrides",
                "d"
            ]),
            Ok(Command::Licenses {
                targets: vec!["t".into()],
                package: "pc-ffi".into(),
                out: "o".into(),
                overrides: Some("d".into())
            })
        );
        assert_eq!(
            p(&["licenses", "--target", "a", "--out", "o", "--target", "b"]),
            Ok(Command::Licenses {
                targets: vec!["a".into(), "b".into()],
                package: "pc-ffi".into(),
                out: "o".into(),
                overrides: None
            })
        );
        assert_eq!(p(&["licenses", "-h"]), Ok(Command::ToolHelp(LICENSES_HELP)));
        for args in [
            &["licenses"][..],
            &["licenses", "--target", "t"],
            &["licenses", "--out", "o"],
            &["licenses", "--target"],
            &["licenses", "--target", "t", "--out", "o", "--overrides"],
            &["licenses", "--target", "t", "--out", "o", "x"],
            &["licenses", "--target", "t", "--out", "o", "--frob"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn cont_command_takes_two_files_and_a_flag() {
        let want = Command::Cont {
            file: "a.ode".into(),
            problem: "p.json".into(),
            fallback: false,
        };
        assert_eq!(p(&["cont", "a.ode", "p.json"]), Ok(want));
        assert_eq!(
            p(&["cont", "--fallback", "a.ode", "p.json"]),
            Ok(Command::Cont {
                file: "a.ode".into(),
                problem: "p.json".into(),
                fallback: true
            })
        );
        assert_eq!(p(&["cont", "-h"]), Ok(Command::ToolHelp(CONT_HELP)));
        for args in [
            &["cont"][..],
            &["cont", "a.ode"],
            &["cont", "a.ode", "p.json", "x"],
            &["cont", "a.ode", "p.json", "--json"],
            &["parse", "a.ode", "--fallback"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    fn replay(file: &str, tol: f64, fallback: bool) -> Command {
        Command::SessionReplay {
            file: file.into(),
            tol,
            fallback,
        }
    }

    #[test]
    fn session_replay_takes_a_file_a_tolerance_and_a_flag() {
        assert_eq!(
            p(&["session", "replay", "a.json"]),
            Ok(replay("a.json", DEFAULT_TOL, false))
        );
        assert_eq!(
            p(&["session", "replay", "--tol", "1e-6", "a.json", "--fallback"]),
            Ok(replay("a.json", 1e-6, true))
        );
        assert_eq!(
            p(&["session", "replay", "--tol", "0", "--", "-a.json"]),
            Ok(replay("-a.json", 0.0, false))
        );
        assert_eq!(p(&["session", "-h"]), Ok(Command::ToolHelp(SESSION_HELP)));
        assert_eq!(
            p(&["session", "replay", "-h"]),
            Ok(Command::ToolHelp(SESSION_HELP))
        );
        for args in [
            &["session"][..],
            &["session", "play", "a.json"],
            &["session", "replay"],
            &["session", "replay", "a.json", "b.json"],
            &["session", "replay", "a.json", "--tol"],
            &["session", "replay", "a.json", "--tol", "x"],
            &["session", "replay", "a.json", "--tol", "-1"],
            &["session", "replay", "a.json", "--tol", "inf"],
            &["session", "replay", "a.json", "--tol", "NaN"],
            &["session", "replay", "a.json", "--json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn bad_flags_and_values_are_usage_errors() {
        for args in [
            &["ir"][..],
            &["ir", "a.ode", "--json"],
            &["ir", "a.ode", "b.ode"],
            &["check-derivs"],
            &["check-derivs", "a.ode", "--points"],
            &["check-derivs", "a.ode", "--points", "0"],
            &["check-derivs", "a.ode", "--points", "-3"],
            &["check-derivs", "a.ode", "--points", "x"],
            &["check-derivs", "a.ode", "--points", "1.5"],
            &["check-derivs", "a.ode", "--seed", "-1"],
            &["check-derivs", "a.ode", "--seed"],
            &["check-derivs", "a.ode", "-o", "x.c"],
            &["emit-c"],
            &["emit-c", "a.ode", "-o"],
            &["emit-c", "a.ode", "-o", "--"],
            &["emit-c", "a.ode", "-o", "-x.c"],
            &["emit-c", "a.ode", "--points", "3"],
            &["parse", "a.ode", "--points", "3"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn everything_else_is_a_usage_error() {
        for args in [
            &[][..],
            &["frob"],
            &["--frob"],
            &["parse"],
            &["parse", "--json"],
            &["parse", "a.ode", "b.ode"],
            &["parse", "a.ode", "--frob"],
            &["--help", "x"],
            &["--version", "x"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    fn selftest(backend: Backend, device: usize, precision: Precision) -> Command {
        Command::GpuSelftest {
            backend,
            device,
            precision,
        }
    }

    #[test]
    fn gpu_list_takes_no_arguments() {
        assert_eq!(p(&["gpu", "list"]), Ok(Command::GpuList));
        assert!(p(&["gpu", "list", "x"]).is_err());
        assert!(p(&["gpu", "list", "--json"]).is_err());
    }

    #[test]
    fn gpu_selftest_needs_a_backend_and_a_precision() {
        assert_eq!(
            p(&[
                "gpu",
                "selftest",
                "--backend",
                "cuda",
                "--precision",
                "single"
            ]),
            Ok(selftest(Backend::Cuda, 0, Precision::Single))
        );
        assert_eq!(
            p(&[
                "gpu",
                "selftest",
                "--precision",
                "double",
                "--device",
                "2",
                "--backend",
                "opencl"
            ]),
            Ok(selftest(Backend::OpenCl, 2, Precision::Double))
        );
        let missing_backend = p(&["gpu", "selftest", "--precision", "single"]).unwrap_err();
        assert!(
            missing_backend.0.contains("--backend"),
            "{missing_backend:?}"
        );
        let missing_precision = p(&["gpu", "selftest", "--backend", "cpu"]).unwrap_err();
        assert!(
            missing_precision.0.contains("--precision"),
            "{missing_precision:?}"
        );
    }

    #[test]
    fn gpu_selftest_names_a_bad_value() {
        let e = p(&[
            "gpu",
            "selftest",
            "--backend",
            "hip",
            "--precision",
            "single",
        ])
        .unwrap_err();
        assert!(e.0.contains("hip") && e.0.contains("cuda"), "{e:?}");
        let e = p(&["gpu", "selftest", "--backend", "cpu", "--precision", "half"]).unwrap_err();
        assert!(e.0.contains("half") && e.0.contains("single"), "{e:?}");
        for args in [
            &["gpu", "selftest", "--backend"][..],
            &["gpu", "selftest", "--backend", "cpu", "--precision"],
            &[
                "gpu",
                "selftest",
                "--backend",
                "cpu",
                "--precision",
                "single",
                "--device",
                "x",
            ],
            &[
                "gpu",
                "selftest",
                "--backend",
                "cpu",
                "--precision",
                "single",
                "--device",
                "-1",
            ],
            &[
                "gpu",
                "selftest",
                "--backend",
                "cpu",
                "--precision",
                "single",
                "extra",
            ],
            &[
                "gpu",
                "selftest",
                "--backend",
                "cpu",
                "--precision",
                "single",
                "--frob",
            ],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn gpu_needs_a_subcommand_and_has_help() {
        assert!(p(&["gpu"]).is_err());
        assert!(p(&["gpu", "frob"]).unwrap_err().0.contains("frob"));
        assert_eq!(p(&["gpu", "--help"]), Ok(Command::ToolHelp(GPU_HELP)));
        assert_eq!(p(&["gpu", "-h"]), Ok(Command::ToolHelp(GPU_HELP)));
        assert_eq!(
            p(&["gpu", "selftest", "-h"]),
            Ok(Command::ToolHelp(GPU_HELP))
        );
        assert_eq!(p(&["gpu", "list", "-h"]), Ok(Command::ToolHelp(GPU_HELP)));
    }

    #[test]
    fn ensemble_takes_a_model_and_a_job() {
        assert_eq!(
            p(&["ensemble", "a.ode", "job.json"]),
            Ok(Command::Ensemble {
                file: "a.ode".into(),
                job: "job.json".into()
            })
        );
        assert_eq!(
            p(&["ensemble", "--", "-a.ode", "job.json"]),
            Ok(Command::Ensemble {
                file: "-a.ode".into(),
                job: "job.json".into()
            })
        );
        for args in [
            &["ensemble"][..],
            &["ensemble", "a.ode"],
            &["ensemble", "a.ode", "b.json", "c"],
            &["ensemble", "--members", "10"],
            &["ensemble", "a.ode", "b.json", "--frob"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
        assert_eq!(
            p(&["ensemble", "--help"]),
            Ok(Command::ToolHelp(ENSEMBLE_HELP))
        );
    }

    #[test]
    fn ensemble_bench_takes_sizes() {
        assert_eq!(
            p(&["ensemble", "--bench"]),
            Ok(Command::EnsembleBench {
                members: DEFAULT_BENCH_MEMBERS.to_vec()
            })
        );
        assert_eq!(
            p(&["ensemble", "--bench", "--members", "10,2000,5"]),
            Ok(Command::EnsembleBench {
                members: vec![10, 2000, 5]
            })
        );
        assert_eq!(
            p(&["ensemble", "--members", "7", "--bench"]),
            Ok(Command::EnsembleBench { members: vec![7] })
        );
        for args in [
            &["ensemble", "--bench", "--members", "0"][..],
            &["ensemble", "--bench", "--members", "10,x"],
            &["ensemble", "--bench", "--members", "10,,20"],
            &["ensemble", "--bench", "--members", ""],
            &["ensemble", "--bench", "--members"],
            &["ensemble", "--bench", "a.ode"],
            &["ensemble", "--bench", "a.ode", "b.json"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_arguments() {
        use std::os::unix::ffi::OsStringExt;
        let bad = OsString::from_vec(vec![b'b', 0xff, b'.']);
        // As the command or a flag: a usage error, no panic.
        assert!(parse_args([bad.clone()]).is_err());
        let flag = OsString::from_vec(vec![b'-', 0xff]);
        assert!(parse_args([OsString::from("parse"), flag]).is_err());
        // As the file: kept as it is.
        let got = parse_args([OsString::from("parse"), bad.clone()]);
        assert_eq!(
            got,
            Ok(Command::Parse {
                file: PathBuf::from(bad),
                json: false
            })
        );
    }

    #[test]
    fn latex_command_parses_kind_file_and_flags() {
        use pc_latex::Kind;
        let latex = |kind, file: &str, out: Option<&str>, standalone| {
            Ok(Command::Latex {
                kind,
                file: file.into(),
                out: out.map(Into::into),
                standalone,
            })
        };
        assert_eq!(
            p(&["latex", "pgfplots", "f.json"]),
            latex(Kind::Pgfplots, "f.json", None, false)
        );
        assert_eq!(
            p(&["latex", "table", "--standalone", "--out", "t.tex", "t.json"]),
            latex(Kind::Table, "t.json", Some("t.tex"), true)
        );
        assert_eq!(
            p(&["latex", "pgfplots", "f.json", "-o", "x.tex"]),
            latex(Kind::Pgfplots, "f.json", Some("x.tex"), false)
        );
        assert_eq!(
            p(&["latex", "table", "--", "--standalone"]),
            latex(Kind::Table, "--standalone", None, false)
        );
        assert_eq!(p(&["latex", "-h"]), Ok(Command::ToolHelp(LATEX_HELP)));
        assert_eq!(
            p(&["latex", "table", "--help"]),
            Ok(Command::ToolHelp(LATEX_HELP))
        );
        for args in [
            &["latex"][..],
            &["latex", "figure", "f.json"],
            &["latex", "pgfplots"],
            &["latex", "pgfplots", "a.json", "b.json"],
            &["latex", "pgfplots", "f.json", "--out"],
            &["latex", "pgfplots", "f.json", "--digits", "4"],
            &["latex", "pgfplots", "--"],
        ] {
            assert!(p(args).is_err(), "{args:?}");
        }
    }
}
