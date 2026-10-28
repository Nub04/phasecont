//! Command line parsing. There is no argument crate: the grammar is small.
//!
//! The command and the flags are matched as UTF-8. The file operand stays an
//! `OsString`, so a path that is not valid UTF-8 still opens.

use std::ffi::OsString;
use std::path::PathBuf;

/// Random points per entry when `--points` is not given.
pub const DEFAULT_POINTS: usize = 20;
/// Seed when `--seed` is not given.
pub const DEFAULT_SEED: u64 = 1;

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
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
       pc-cli --help | pc-cli --version";

pub const HELP: &str = "\
pc-cli: headless runner for PhaseCont

usage:
  pc-cli parse <file.ode> [--json]
  pc-cli ir <file.ode>
  pc-cli check-derivs <file.ode> [--points N] [--seed S]
  pc-cli emit-c <file.ode> [-o out.c]
  pc-cli --help
  pc-cli --version

commands:
  parse         read an XPPAUT .ode file and report its diagnostics and a summary
  ir            print the size of the model and of each derivative entry
  check-derivs  compare the derivative entries of the DAG, the JIT and the
                f64 fallback, and with finite differences
  emit-c        write C code for F, JacU and JacP
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
}

/// Which flags a command accepts.
#[derive(Clone, Copy, Default)]
struct Accepts {
    json: bool,
    points_seed: bool,
    out: bool,
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
