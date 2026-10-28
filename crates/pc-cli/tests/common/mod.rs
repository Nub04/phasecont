//! Helpers shared by the integration tests that run the `pc-cli` binary.

use std::path::Path;
use std::process::{Command, Output};

/// Runs the binary in `dir` with `args`.
pub fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pc-cli"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("cannot run pc-cli")
}

pub fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("stdout is UTF-8")
}

pub fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("stderr is UTF-8")
}

pub fn code(o: &Output) -> i32 {
    o.status.code().expect("pc-cli was killed by a signal")
}
