//! Headless runner: parse models, run session JSON files, run benchmarks.

mod args;
mod json;
mod summary;
mod text;

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use args::{Command, HELP, PARSE_HELP, USAGE, parse_args};
use pc_ode::{FsLoader, Severity, SourceMap, load};

fn main() -> ExitCode {
    match parse_args(std::env::args_os().skip(1)) {
        Ok(Command::Help) => out(HELP, 0),
        Ok(Command::ParseHelp) => out(PARSE_HELP, 0),
        Ok(Command::Version) => out(concat!("pc-cli ", env!("CARGO_PKG_VERSION"), "\n"), 0),
        Ok(Command::Parse { file, json }) => parse(&file, json),
        Err(e) => {
            eprintln!("pc-cli: {}", e.0);
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Writes `text` to stdout and returns `code`. A failed write exits 1 with a
/// message on stderr.
fn out(text: &str, code: u8) -> ExitCode {
    let mut stdout = std::io::stdout().lock();
    match stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => ExitCode::from(code),
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "pc-cli: cannot write output: {e}");
            ExitCode::from(1)
        }
    }
}

fn parse(file: &Path, json: bool) -> ExitCode {
    let shown = file.to_string_lossy();
    // The top file is read as strict UTF-8, like included files in
    // `FsLoader`: invalid bytes are a read error, never replaced.
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("pc-cli: cannot read {shown}: {e}");
            return ExitCode::from(1);
        }
    };
    let mut map = SourceMap::new();
    let top = map.add(shown.as_ref(), text);
    let loaded = load(top, &mut map, &mut FsLoader::new());
    let report = if json {
        json::render(&shown, &map, &loaded)
    } else {
        text::render(&map, &loaded)
    };
    let failed = loaded
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error);
    out(&report, u8::from(failed))
}
