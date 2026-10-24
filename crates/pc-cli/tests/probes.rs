//! Checks the PhaseCont column of the XPPAUT probe files.
//!
//! Every `references/xpp-probes/*.ode` has a `# pc:` header line that lists
//! the diagnostic codes `pc-cli parse` gives for the file (in order of first
//! appearance), or `ok` when it gives none. This test runs the built binary
//! on each probe, so the column stays true without XPPAUT. The XPPAUT column
//! is checked by `references/xpp-probes/run.sh`.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

fn probe_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../references/xpp-probes")
}

fn probes() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(probe_dir())
        .expect("references/xpp-probes exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .collect();
    v.sort();
    v
}

/// The text after `# pc:` on the first such line.
fn pc_header(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("# pc:"))
        .map(|s| s.trim().to_string())
}

#[test]
fn probes_exist() {
    assert!(
        probes().len() >= 100,
        "expected at least 100 probes, found {}",
        probes().len()
    );
}

#[test]
fn every_probe_matches_its_pc_header() {
    let mut problems = Vec::new();
    for path in probes() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).unwrap();
        let Some(want) = pc_header(&text) else {
            problems.push(format!("{name}: no `# pc:` line"));
            continue;
        };
        let out = Command::new(env!("CARGO_BIN_EXE_pc-cli"))
            .args(["parse", "--json"])
            .arg(&path)
            .output()
            .expect("cannot run pc-cli");
        let exit = out.status.code().expect("pc-cli was killed by a signal");
        let json: Value = serde_json::from_slice(&out.stdout).expect("stdout is JSON");
        let diags = json["diagnostics"].as_array().expect("diagnostics array");

        let mut codes: Vec<&str> = Vec::new();
        let mut errors = 0;
        for d in diags {
            let code = d["code"].as_str().expect("diagnostic code");
            if !codes.contains(&code) {
                codes.push(code);
            }
            if d["severity"] == "error" {
                errors += 1;
            }
        }
        let got = if codes.is_empty() {
            "ok".to_string()
        } else {
            codes.join(",")
        };
        if got != want {
            problems.push(format!("{name}: `# pc:` says {want}, pc-cli gives {got}"));
        }
        let want_exit = i32::from(errors > 0);
        if exit != want_exit {
            problems.push(format!(
                "{name}: exit code {exit}, expected {want_exit} for {errors} errors"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn header_check_rejects_a_missing_line() {
    assert_eq!(pc_header("# a: b\nx'=0\n"), None);
    assert_eq!(
        pc_header("# pc: ODE0010, ODE0014\n").as_deref(),
        Some("ODE0010, ODE0014")
    );
}
