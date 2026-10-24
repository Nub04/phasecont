//! Command line parsing. There is no argument crate: the grammar is small.

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
    Parse { file: String, json: bool },
}

/// Why the command line is not valid. The text is a short usage line for
/// stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);

pub const USAGE: &str =
    "usage: pc-cli parse <file.ode> [--json] | pc-cli --help | pc-cli --version";

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
usage: pc-cli parse <file.ode> [--json]

Reads the file and the files it includes, prints each diagnostic with its
source line, then a summary of the model.

options:
  --json     print one JSON object instead of text
  -h, --help print this help

exit codes: 0 no errors, 1 errors (or the file cannot be read), 2 usage error
";

/// Parses the arguments after the program name.
pub fn parse_args<I, S>(args: I) -> Result<Command, UsageError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args.into_iter().map(|a| a.as_ref().to_string()).collect();
    let Some(first) = args.first() else {
        return Err(UsageError("no command given".to_string()));
    };
    match first.as_str() {
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

fn parse_command(rest: &[String]) -> Result<Command, UsageError> {
    let mut file: Option<String> = None;
    let mut json = false;
    let mut help = false;
    for arg in rest {
        match arg.as_str() {
            "--json" => json = true,
            "-h" | "--help" => help = true,
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(UsageError(format!("unknown option {other}")));
            }
            other => {
                if file.is_some() {
                    return Err(UsageError(format!("unexpected extra argument {other}")));
                }
                file = Some(other.to_string());
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
        parse_args(args.iter().copied())
    }

    #[test]
    fn parse_with_flag_before_or_after() {
        let want = Command::Parse {
            file: "a.ode".into(),
            json: true,
        };
        assert_eq!(p(&["parse", "a.ode", "--json"]), Ok(want.clone()));
        assert_eq!(p(&["parse", "--json", "a.ode"]), Ok(want));
        assert_eq!(
            p(&["parse", "a.ode"]),
            Ok(Command::Parse {
                file: "a.ode".into(),
                json: false
            })
        );
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
}
