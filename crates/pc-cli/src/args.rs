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
       pc-cli session replay [--tol X] [--fallback] [--] <file.pcsession.json>
       pc-cli gpu list
       pc-cli gpu selftest --backend B [--device N] --precision P
       pc-cli ensemble [--] <file.ode> <job.json>
       pc-cli ensemble --bench [--members N,N,...]
       pc-cli dll-imports [--] <file.dll>
       pc-cli licenses --target TRIPLE... [--package NAME] --out FILE [--overrides DIR]
       pc-cli --help | pc-cli --version";

pub const HELP: &str = "\
pc-cli: headless runner for PhaseCont

usage:
  pc-cli parse <file.ode> [--json]
  pc-cli ir <file.ode>
  pc-cli check-derivs <file.ode> [--points N] [--seed S]
  pc-cli emit-c <file.ode> [-o out.c]
  pc-cli cont <file.ode> <problem.json> [--fallback]
  pc-cli session replay <file.pcsession.json> [--tol X] [--fallback]
  pc-cli gpu list
  pc-cli gpu selftest --backend B [--device N] --precision P
  pc-cli ensemble <file.ode> <job.json>
  pc-cli ensemble --bench [--members N,N,...]
  pc-cli dll-imports <file.dll>
  pc-cli licenses --target TRIPLE... [--package NAME] --out FILE [--overrides DIR]
  pc-cli --help
  pc-cli --version

commands:
  parse         read an XPPAUT .ode file and report its diagnostics and a summary
  ir            print the size of the model and of each derivative entry
  check-derivs  compare the derivative entries of the DAG, the JIT and the
                f64 fallback, and with finite differences
  emit-c        write C code for F, JacU and JacP
  cont          follow an equilibrium branch and print its points and special points
  session       rerun the runs of a session file and compare them with the stored ones
  gpu           list the ensemble devices and test one of them
  ensemble      run many trajectories of one model, or time the engine
  dll-imports   list the DLLs a 64-bit Windows PE file imports and check the allow-list
  licenses      write the license texts of the crates a package links
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
            two parameters (required)
  free      name of the continuation parameter, or [a1, a2] for LP, H, LPC, PD
            and NS, with a1 the parameter of the branch the start point is on
            (required)
  u0        starting state (default: the model's init): one number per state,
            on an LP or H curve the point there, for LC \"hopf\" the equilibrium
            at the Hopf point, for LC \"orbit\" or \"pd\" and for LPC, PD and NS
            the orbit at the N m + 1 base points of its mesh, point by point,
            then the period (required for those)
  params    object of parameter values by name (default: the model's values)
  options   h0, hmin, hmax, max_steps, tol_x, tol_f, max_newton, direction,
            bounds [lo, hi], detect {lp, bp, h} (EP, LP and H only),
            mesh_adapt (cycles only: adapt the mesh every that many steps, 0
            never); engine defaults for the rest
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

options:
  --fallback  evaluate the model with the f64 interpreter instead of the JIT
  --          the next argument is the file, even if it starts with `-`
  -h, --help  print this help

exit codes: 0 the run finished, 1 the run failed, panicked or was cancelled, or
the model cannot be read, has errors or is refused, 2 usage error or bad
problem file (the message names the key)
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
        "session" => session_command(&args[1..]),
        "gpu" => gpu_command(&args[1..]),
        "ensemble" => ensemble_command(&args[1..]),
        "dll-imports" => dll_imports_command(&args[1..]),
        "licenses" => licenses_command(&args[1..]),
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
}
