//! Runs the built `pc-cli` binary on the `pc-ode` corpus and invalid files.
//!
//! The golden file of `foo.ode` is `tests/golden/foo.json`. Run
//! `PC_BLESS=1 cargo test -p pc-cli --test cli` to rewrite them after a
//! change that is meant to alter the JSON, and read the diff.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn ode_tests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Runs the binary in `dir` with `args`.
fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pc-cli"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("cannot run pc-cli")
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(o: &Output) -> String {
    String::from_utf8(o.stderr.clone()).expect("stderr is UTF-8")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("pc-cli was killed by a signal")
}

/// Sorted names of the `*.ode` files directly in `dir`.
fn ode_files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
        .collect();
    v.sort();
    v
}

#[test]
fn corpus_files_exit_zero_in_both_modes() {
    let dir = ode_tests().join("corpus");
    let files = ode_files(&dir);
    assert!(files.len() >= 21);
    for f in &files {
        let text = run(&dir, &["parse", f]);
        assert_eq!(code(&text), 0, "{f}: {}", stderr(&text));
        assert!(stdout(&text).contains("kind: "), "{f}: no summary");
        let json = run(&dir, &["parse", f, "--json"]);
        assert_eq!(code(&json), 0, "{f}: {}", stderr(&json));
        // The flag may come first.
        let json2 = run(&dir, &["parse", "--json", f]);
        assert_eq!(stdout(&json), stdout(&json2), "{f}");
    }
}

#[test]
fn corpus_json_has_the_documented_keys() {
    let dir = ode_tests().join("corpus");
    for f in ode_files(&dir) {
        let out = run(&dir, &["parse", &f, "--json"]);
        let v: Value = serde_json::from_str(&stdout(&out)).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(v["file"], f.as_str());
        assert!(v["diagnostics"].is_array(), "{f}");
        for key in [
            "kind",
            "states",
            "parameters",
            "fixed",
            "functions",
            "aux",
            "smooth",
            "simulation_only",
            "errors",
            "warnings",
        ] {
            assert!(!v["summary"][key].is_null(), "{f}: summary.{key}");
        }
        for d in v["diagnostics"].as_array().unwrap() {
            for key in [
                "code", "severity", "file", "line", "col", "end_line", "end_col", "message",
                "fix_it", "notes",
            ] {
                assert!(d.get(key).is_some(), "{f}: diagnostic.{key}");
            }
        }
    }
}

#[test]
fn corpus_json_matches_the_golden_files() {
    let bless = std::env::var_os("PC_BLESS").is_some_and(|v| v == "1");
    let dir = ode_tests().join("corpus");
    let mut failures = Vec::new();
    for f in ode_files(&dir) {
        let actual = stdout(&run(&dir, &["parse", &f, "--json"]));
        // A backslash would mean a Windows path separator leaked into a file
        // name, which would make the goldens differ between platforms. No
        // corpus diagnostic text has a backslash or an escaped quote.
        assert!(!actual.contains('\\'), "{f}: the JSON has a backslash");
        let golden = golden_dir().join(f.replace(".ode", ".json"));
        if bless {
            std::fs::create_dir_all(golden_dir()).unwrap();
            std::fs::write(&golden, &actual).unwrap();
            continue;
        }
        match std::fs::read_to_string(&golden) {
            Err(_) => failures.push(format!(
                "{}: golden file is missing (run with PC_BLESS=1 to create it)",
                golden.display()
            )),
            Ok(want) if want != actual => failures.push(format!(
                "{}: golden file differs (run with PC_BLESS=1 to rewrite it, then read the diff)",
                golden.display()
            )),
            Ok(_) => {}
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn every_golden_file_has_a_corpus_file() {
    for entry in std::fs::read_dir(golden_dir()).unwrap() {
        let path = entry.unwrap().path();
        let ode = ode_tests()
            .join("corpus")
            .join(path.file_name().unwrap())
            .with_extension("ode");
        assert!(ode.is_file(), "{} has no corpus file", path.display());
    }
}

/// `ode0012.ode` and `ode0012_b.ode` both give code 12.
fn code_of_name(name: &str) -> u16 {
    name[3..7].parse().unwrap()
}

#[test]
fn invalid_files_exit_by_severity_and_name_their_code() {
    let dir = ode_tests().join("invalid");
    let files = ode_files(&dir);
    assert!(files.len() >= 70);
    for f in &files {
        let n = code_of_name(f);
        let label = format!("ODE{n:04}");
        let is_error = pc_ode::Code(n).severity() == pc_ode::Severity::Error;
        let want = i32::from(is_error);
        let text = run(&dir, &["parse", f]);
        assert_eq!(code(&text), want, "{f}: {}", stdout(&text));
        let t = stdout(&text);
        assert!(t.contains(&format!("[{label}]")), "{f}: {t}");
        assert!(t.contains(&format!("{f}:")), "{f}: {t}");
        let json = run(&dir, &["parse", f, "--json"]);
        assert_eq!(code(&json), want, "{f}");
        let v: Value = serde_json::from_str(&stdout(&json)).unwrap();
        let found = v["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["code"] == label.as_str());
        assert!(found, "{f}: no {label} in JSON");
    }
}

#[test]
fn text_output_shows_source_line_caret_and_summary() {
    let dir = ode_tests().join("invalid");
    let out = stdout(&run(&dir, &["parse", "ode0012.ode"]));
    let mut lines = out.lines();
    let first = lines.next().unwrap();
    assert!(first.starts_with("ode0012.ode:"), "{out}");
    assert!(first.contains(": error[ODE0012]: "), "{out}");
    let source = lines.next().unwrap();
    let caret = lines.next().unwrap();
    assert!(caret.trim_start().starts_with('^'), "{out}");
    assert!(caret.chars().count() <= source.chars().count() + 1, "{out}");
    assert!(out.contains("\n\nkind: "), "{out}");
    assert!(out.contains("errors, "), "{out}");
}

#[test]
fn json_for_an_invalid_file_has_positions() {
    let dir = ode_tests().join("invalid");
    let out = run(&dir, &["parse", "ode0012.ode", "--json"]);
    let v: Value = serde_json::from_str(&stdout(&out)).unwrap();
    let d = &v["diagnostics"][0];
    assert_eq!(d["code"], "ODE0012");
    assert_eq!(d["severity"], "error");
    assert!(d["line"].as_u64().unwrap() >= 1);
    assert!(d["col"].as_u64().unwrap() >= 1);
    assert!(v["summary"]["errors"].as_u64().unwrap() >= 1);
}

#[test]
fn usage_errors_exit_two_and_write_to_stderr() {
    let dir = ode_tests().join("corpus");
    for args in [
        &[][..],
        &["frob"],
        &["--frob"],
        &["parse"],
        &["parse", "--json"],
        &["parse", "lorenz.ode", "rossler.ode"],
        &["parse", "lorenz.ode", "--frob"],
    ] {
        let o = run(&dir, args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(stdout(&o).is_empty(), "{args:?}");
        assert!(stderr(&o).contains("usage:"), "{args:?}");
    }
}

#[test]
fn help_and_version_exit_zero_on_stdout() {
    let dir = ode_tests().join("corpus");
    for args in [&["--help"][..], &["-h"], &["parse", "--help"]] {
        let o = run(&dir, args);
        assert_eq!(code(&o), 0, "{args:?}");
        assert!(stdout(&o).contains("usage:"), "{args:?}");
        assert!(stderr(&o).is_empty(), "{args:?}");
    }
    for flag in ["--version", "-V"] {
        let o = run(&dir, &[flag]);
        assert_eq!(code(&o), 0);
        assert_eq!(
            stdout(&o),
            format!("pc-cli {}\n", env!("CARGO_PKG_VERSION"))
        );
    }
}

#[test]
fn unreadable_path_exits_one() {
    let dir = ode_tests().join("corpus");
    let o = run(&dir, &["parse", "no_such_file.ode"]);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).is_empty());
    assert!(stderr(&o).starts_with("pc-cli: cannot read no_such_file.ode: "));
}

#[test]
fn invalid_utf8_top_file_is_a_read_error() {
    let dir = std::env::temp_dir().join(format!("pc-cli-utf8-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("bad.ode"), b"x'=-x\n# \xff\xfe\ndone\n").unwrap();
    let o = run(&dir, &["parse", "bad.ode"]);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(code(&o), 1);
    assert!(
        stderr(&o).starts_with("pc-cli: cannot read bad.ode: "),
        "{}",
        stderr(&o)
    );
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn error_in_an_included_file_names_that_file() {
    let dir = fixtures();
    let o = run(&dir, &["parse", "top_inc.ode"]);
    assert_eq!(code(&o), 1);
    let t = stdout(&o);
    assert!(
        t.starts_with(
            "inc/bad.ode:2:5: error[ODE0012]: XPPAUT deletes this blank and joins `1` and `2` into one word; delete the blank or add an operator\ny'=1 2\n    ^\n"
        ),
        "{t}"
    );
    let o = run(&dir, &["parse", "top_inc.ode", "--json"]);
    assert_eq!(code(&o), 1);
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["file"], "top_inc.ode");
    let d = &v["diagnostics"][0];
    assert_eq!(d["file"], "inc/bad.ode");
    assert_eq!(d["code"], "ODE0012");
    assert_eq!(d["line"], 2);
    assert_eq!(d["col"], 5);
    assert_eq!(d["end_col"], 6);
}

#[test]
fn double_dash_takes_a_file_that_starts_with_a_dash() {
    let dir = ode_tests().join("corpus");
    // The file does not exist: a read error (1), not a usage error (2).
    let o = run(&dir, &["parse", "--", "-x.ode"]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).starts_with("pc-cli: cannot read -x.ode: "));
    let dir = std::env::temp_dir().join(format!("pc-cli-dash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("-x.ode"), "x'=-x\ndone\n").unwrap();
    let o = run(&dir, &["parse", "--", "-x.ode"]);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[cfg(unix)]
#[test]
fn non_utf8_arguments_do_not_panic() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let bin = env!("CARGO_BIN_EXE_pc-cli");
    let bad = OsStr::from_bytes(b"bad\xff.ode");
    // As the command: usage error.
    let o = Command::new(bin).arg(bad).output().unwrap();
    assert_eq!(code(&o), 2);
    // As a flag: usage error.
    let o = Command::new(bin)
        .arg("parse")
        .arg(OsStr::from_bytes(b"-\xff"))
        .output()
        .unwrap();
    assert_eq!(code(&o), 2);
    // As the file: it opens when it exists, and is a read error when not.
    let o = Command::new(bin).arg("parse").arg(bad).output().unwrap();
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).starts_with("pc-cli: cannot read "));
    let dir = std::env::temp_dir().join(format!("pc-cli-nonutf8-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(bad), "x'=-x\ndone\n").unwrap();
    let o = Command::new(bin)
        .arg("parse")
        .arg(bad)
        .current_dir(&dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[cfg(unix)]
#[test]
fn a_failed_write_exits_one_with_a_message() {
    let Ok(full) = std::fs::OpenOptions::new().write(true).open("/dev/full") else {
        return; // No /dev/full here.
    };
    let dir = ode_tests().join("corpus");
    let o = Command::new(env!("CARGO_BIN_EXE_pc-cli"))
        .args(["parse", "lorenz.ode"])
        .current_dir(&dir)
        .stdout(full)
        .output()
        .unwrap();
    assert_eq!(code(&o), 1);
    assert!(
        stderr(&o).starts_with("pc-cli: cannot write output: "),
        "{}",
        stderr(&o)
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_top_path_still_resolves_relative_includes() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let bin = env!("CARGO_BIN_EXE_pc-cli");
    let dir = std::env::temp_dir().join(format!("pc-cli-nonutf8-inc-{}", std::process::id()));
    std::fs::create_dir_all(dir.join(OsStr::from_bytes(b"d\xff"))).unwrap();
    let sub = dir.join(OsStr::from_bytes(b"d\xff"));
    std::fs::write(sub.join("top.ode"), "#include inc.ode\ndone\n").unwrap();
    std::fs::write(sub.join("inc.ode"), "x'=-x\n").unwrap();
    // The working directory is not the folder of the model, so a lossy
    // path would not find `inc.ode`.
    let o = Command::new(bin)
        .arg("parse")
        .arg(sub.join("top.ode"))
        .current_dir(&dir)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(code(&o), 0, "{}{}", stdout(&o), stderr(&o));
    assert!(stdout(&o).contains("states: 1\n"), "{}", stdout(&o));
}
