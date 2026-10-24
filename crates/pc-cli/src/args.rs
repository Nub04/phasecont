//! Command line parsing. There is no argument crate: the grammar is small.
//!
//! The command and the flags are matched as UTF-8. The file operand stays an
//! `OsString`, so a path that is not valid UTF-8 still opens.

use std::ffi::OsString;
use std::path::PathBuf;

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Print the top-level help to stdout.
    Help,
    /// Print the help for `parse` to stdout.
    ParseHelp,
    /// Print `pc-cli <version>`.
    Version,
    /// Parse one file.
    Parse { file: PathBuf, json: bool },
}

/// Why the command line is not valid. The text is a short usage line for
/// stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

pub const USAGE: &str =
    "usage: pc-cli parse [--json] [--] <file.ode> | pc-cli --help | pc-cli --version";

pub const HELP: &str = "\
pc-cli: headless runner for PhaseCont

usage:
  pc-cli parse <file.ode> [--json]
  pc-cli --help
  pc-cli --version

commands:
  parse    read an XPPAUT .ode file and report its diagnostics and a summary
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
        "-h" | "--help" | "-V" | "--version" => {
            Err(UsageError(format!("unexpected argument after {first}")))
        }
        other if other.starts_with('-') => Err(UsageError(format!("unknown option {other}"))),
        other => Err(UsageError(format!("unknown command {other}"))),
    }
}

fn parse_command(rest: &[OsString]) -> Result<Command, UsageError> {
    let mut file: Option<PathBuf> = None;
    let mut json = false;
    let mut help = false;
    let mut only_files = false;
    for arg in rest {
        let flag = if only_files { None } else { arg.to_str() };
        match flag {
            Some("--") => only_files = true,
            Some("--json") => json = true,
            Some("-h" | "--help") => help = true,
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
                if file.is_some() {
                    return Err(UsageError(format!(
                        "unexpected extra argument {}",
                        arg.to_string_lossy()
                    )));
                }
                file = Some(PathBuf::from(arg));
            }
        }
    }
    if help {
        return Ok(Command::ParseHelp);
    }
    match file {
        Some(file) => Ok(Command::Parse { file, json }),
        None => Err(UsageError("parse needs a file".to_string())),
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
        assert_eq!(p(&["parse", "--help"]), Ok(Command::ParseHelp));
        assert_eq!(p(&["parse", "x.ode", "-h"]), Ok(Command::ParseHelp));
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
