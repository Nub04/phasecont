//! Headless runner: parse models, run session JSON files, run benchmarks.

mod args;
mod cont;
mod ddebranch;
mod ddecmd;
mod ddecycle;
mod ddeeq;
mod ddehh;
mod ddehopf;
mod ddelyap;
mod ddezh;
mod dllimports;
mod ensemble;
mod gpu;
mod ircmd;
mod latexcmd;
mod licenses;
mod pe;
mod text;

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use args::{Command, HELP, USAGE, parse_args};
use pc_cli::session::{ReplayOptions, Session, replay};
use pc_ode::{FsLoader, Loaded, Severity, SourceMap, load};

fn main() -> ExitCode {
    match parse_args(std::env::args_os().skip(1)) {
        Ok(Command::Help) => out(HELP, 0),

        Ok(Command::ToolHelp(text)) => out(text, 0),
        Ok(Command::Version) => out(concat!("pc-cli ", env!("CARGO_PKG_VERSION"), "\n"), 0),
        Ok(Command::Parse { file, json }) => parse(&file, json),
        Ok(Command::Ir { file }) => ircmd::ir(&file),
        Ok(Command::CheckDerivs { file, points, seed }) => ircmd::check_derivs(&file, points, seed),
        Ok(Command::EmitC { file, out }) => ircmd::emit_c(&file, out.as_deref()),
        Ok(Command::DllImports { file }) => dllimports::dll_imports(&file),
        Ok(Command::Licenses {
            targets,
            package,
            out,
            overrides,
        }) => licenses::licenses(&targets, &package, &out, overrides.as_deref()),
        Ok(Command::Latex {
            kind,
            file,
            out,
            standalone,
        }) => latexcmd::latex(kind, &file, out.as_deref(), standalone),
        Ok(Command::Cont {
            file,
            problem,
            fallback,
        }) => cont::cont(&file, &problem, fallback),
        Ok(Command::DdeRoots { file }) => ddecmd::dde_roots(&file),
        Ok(Command::DdeEq { file }) => ddeeq::dde_eq(&file),
        Ok(Command::DdeBranch { file }) => ddebranch::dde_branch(&file),
        Ok(Command::DdeHopf { file }) => ddehopf::dde_hopf(&file),
        Ok(Command::DdeCycle { file }) => ddecycle::dde_cycle(&file),
        Ok(Command::DdeLyap { file }) => ddelyap::dde_lyap(&file),
        Ok(Command::DdeZh { file }) => ddezh::dde_zh(&file),
        Ok(Command::DdeHh { file }) => ddehh::dde_hh(&file),
        Ok(Command::GpuList) => gpu::list(),
        Ok(Command::GpuSelftest {
            backend,
            device,
            precision,
        }) => gpu::selftest(backend, device, precision),
        Ok(Command::Ensemble { file, job }) => ensemble::ensemble(&file, &job),
        Ok(Command::EnsembleBench { members }) => ensemble::bench(&members),
        Ok(Command::SessionReplay {
            file,
            tol,
            fallback,
        }) => session_replay(&file, tol, fallback),
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

/// Replays a session file and prints the JSON report. Exit code 0 when every run passes, 1
/// when one does not, 2 when the file is not a version 1 session or its model does not load
/// or does not match the stored hash.
fn session_replay(file: &Path, tol: f64, fallback: bool) -> ExitCode {
    let session = match Session::read(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("pc-cli: {e}");
            return ExitCode::from(2);
        }
    };
    match replay(&session, ReplayOptions { tol, fallback }) {
        Ok(report) => {
            let mut text = serde_json::to_string_pretty(&report.to_json())
                .unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
            text.push('\n');
            out(&text, u8::from(!report.pass()))
        }
        Err(e) => {
            eprintln!("pc-cli: {e}");
            ExitCode::from(2)
        }
    }
}

/// Reads and loads `file` and its includes. An unreadable file prints a
/// message on stderr and gives exit code 1 as the error.
fn load_file(file: &Path) -> Result<(SourceMap, Loaded), ExitCode> {
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    // One read path for the top file and its includes: strict UTF-8, so
    // invalid bytes are a read error, never replaced.
    let top = match loader.load_top(&mut map, file) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("pc-cli: cannot read {e}");
            return Err(ExitCode::from(1));
        }
    };
    let loaded = load(top, &mut map, &mut loader);
    Ok((map, loaded))
}

fn parse(file: &Path, json: bool) -> ExitCode {
    let shown = file.to_string_lossy();
    let (map, loaded) = match load_file(file) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let report = if json {
        pckern::json::diag::render(&shown, &map, &loaded)
    } else {
        text::render(&map, &loaded)
    };
    let failed = loaded
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error);
    out(&report, u8::from(failed))
}
