//! Runs `pc-cli ir`, `check-derivs` and `emit-c` on the models in
//! `tests/fixtures`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Runs the binary in the fixtures directory with `args`.
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pc-cli"))
        .args(args)
        .current_dir(fixtures())
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

/// The number after `name` on the line of `text` that starts with `name`.
fn count_after(text: &str, name: &str) -> usize {
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with(name))
        .unwrap_or_else(|| panic!("no line for {name}:\n{text}"));
    line[line.find(name).unwrap() + name.len()..]
        .split_whitespace()
        .next()
        .and_then(|w| w.parse().ok())
        .unwrap_or_else(|| panic!("no number in {line:?}"))
}

#[test]
fn ir_prints_the_shape_and_the_node_counts() {
    let o = run(&["ir", "vdp.ode"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let t = stdout(&o);
    assert!(t.starts_with("n: 2\nn_p: 1\nn_aux: 1\n"), "{t}");
    assert!(t.contains("autonomous: true\n"), "{t}");
    assert!(t.contains("smooth: true\n"), "{t}");
    for name in [
        "F ",
        "Aux ",
        "JacU ",
        "JacP ",
        "Dir[U,U] ",
        "Dir[U,U,U] ",
        "Dir[U,U,U,U] ",
        "Dir[U,U,U,U,U] ",
    ] {
        assert!(count_after(&t, name) >= 1, "{name}: {t}");
    }
    // Outputs: F has n, JacU n*n, JacP n*n_p, Aux n_aux.
    let outputs = |name: &str| {
        let line = t
            .lines()
            .find(|l| l.trim_start().starts_with(name))
            .unwrap();
        let w: Vec<&str> = line.split_whitespace().collect();
        w[w.len() - 2].parse::<usize>().unwrap()
    };
    assert_eq!(outputs("F "), 2);
    assert_eq!(outputs("Aux "), 1);
    assert_eq!(outputs("JacU "), 4);
    assert_eq!(outputs("JacP "), 2);
    assert!(t.contains("dag nodes after building the entries: "), "{t}");
}

#[test]
fn ir_reports_a_model_that_is_not_smooth() {
    let o = run(&["ir", "kinked.ode"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stdout(&o).contains("smooth: false\n"), "{}", stdout(&o));
}

#[test]
fn check_derivs_passes_on_a_smooth_model() {
    let o = run(&["check-derivs", "vdp.ode", "--points", "6", "--seed", "3"]);
    assert_eq!(code(&o), 0, "{}\n{}", stdout(&o), stderr(&o));
    let t = stdout(&o);
    assert!(t.starts_with("6 points per entry, seed 3\n"), "{t}");
    assert!(t.trim_end().ends_with("checks pass"), "{t}");
    assert!(!t.contains("FAIL"), "{t}");
    for label in [
        "F ",
        "Aux ",
        "JacU ",
        "JacP ",
        "A (FD)",
        "C (FD)",
        "F (JIT vs DAG, 0 ulp)",
        "F (JIT vs Fallback)",
        "Aux (JIT vs DAG, 0 ulp)",
        "JacU (JIT vs Fallback)",
        "JacP (JIT vs DAG, 0 ulp)",
        "Dir[U,U,U,U,U] (JIT vs DAG, 0 ulp)",
        "Dir[U,U,P] (JIT vs Fallback)",
    ] {
        assert!(
            t.lines().any(|l| l.starts_with(label)),
            "no line for {label}:\n{t}"
        );
    }
    // The same seed gives the same text.
    let again = run(&["check-derivs", "vdp.ode", "--points", "6", "--seed", "3"]);
    assert_eq!(stdout(&again), t);
}

#[test]
fn check_derivs_defaults_are_20_points_and_seed_1() {
    let o = run(&["check-derivs", "vdp.ode"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(
        stdout(&o).starts_with("20 points per entry, seed 1\n"),
        "{}",
        stdout(&o)
    );
}

#[test]
fn check_derivs_does_not_count_finite_differences_of_a_kinked_model() {
    let o = run(&["check-derivs", "kinked.ode", "--points", "8"]);
    assert_eq!(code(&o), 0, "{}\n{}", stdout(&o), stderr(&o));
    let t = stdout(&o);
    for label in ["A (FD)", "f_alpha (FD)", "B (FD)", "C (FD)"] {
        let line = t
            .lines()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("no line for {label}:\n{t}"));
        assert!(line.contains("info (non-smooth"), "{line}");
    }
    // The exact-versus-exact checks still run and count.
    assert!(
        t.lines()
            .any(|l| l.starts_with("F (JIT vs DAG, 0 ulp)") && l.contains(" pass ")),
        "{t}"
    );
    assert!(t.contains("finite-difference lines not counted"), "{t}");
    assert!(t.contains("all "), "{t}");
}

#[test]
fn a_smooth_model_prints_no_info_lines() {
    let o = run(&["check-derivs", "vdp.ode", "--points", "2"]);
    assert!(!stdout(&o).contains("info (non-smooth"), "{}", stdout(&o));
    assert!(!stdout(&o).contains("not counted"), "{}", stdout(&o));
}

#[test]
fn refused_models_exit_1_with_the_span_on_stderr() {
    for (file, tag, line) in [
        ("delayed.ode", "error[ir:delay]", "x'=-delay(x,tau)"),
        ("noisy.ode", "error[ir:simulation-only]", "ran(1)"),
    ] {
        for cmd in ["ir", "check-derivs", "emit-c"] {
            let o = run(&[cmd, file]);
            assert_eq!(code(&o), 1, "{cmd} {file}");
            assert!(stdout(&o).is_empty(), "{cmd} {file}: {}", stdout(&o));
            let e = stderr(&o);
            assert!(e.contains(&format!("{file}:")), "{cmd} {file}: {e}");
            assert!(e.contains(tag), "{cmd} {file}: {e}");
            assert!(e.contains(line), "{cmd} {file}: {e}");
            assert!(e.contains('^'), "{cmd} {file}: {e}");
        }
    }
}

#[test]
fn a_model_with_load_errors_prints_them_and_exits_1() {
    for cmd in ["ir", "check-derivs", "emit-c"] {
        let o = run(&[cmd, "broken.ode"]);
        assert_eq!(code(&o), 1, "{cmd}");
        assert!(stdout(&o).is_empty(), "{cmd}");
        assert!(
            stderr(&o).contains("broken.ode:1:7: error[ODE0004]"),
            "{cmd}"
        );
    }
}

#[test]
fn a_missing_file_exits_1() {
    for cmd in ["ir", "check-derivs", "emit-c"] {
        let o = run(&[cmd, "no_such_file.ode"]);
        assert_eq!(code(&o), 1, "{cmd}");
        assert!(stderr(&o).contains("cannot read"), "{cmd}: {}", stderr(&o));
    }
}

fn assert_c_text(c: &str) {
    assert!(c.contains("pc_aot_init"), "{c}");
    for f in ["F", "JacU", "JacP"] {
        assert!(
            c.contains(&format!(
                "PC_EXPORT void {f}(const double *x, const double *p, double *out)"
            )),
            "no function {f}"
        );
    }
}

#[test]
fn emit_c_writes_to_stdout_without_o() {
    let o = run(&["emit-c", "vdp.ode"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_c_text(&stdout(&o));
}

#[test]
fn emit_c_writes_the_file_given_with_o() {
    let dir = std::env::temp_dir().join(format!("pc-cli-emit-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.join("vdp.c");
    let o = run(&["emit-c", "vdp.ode", "-o", target.to_str().unwrap()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{}", stdout(&o));
    let c = std::fs::read_to_string(&target).unwrap();
    assert_c_text(&c);
    // The flag may come first, and the text is the same as on stdout.
    let again = run(&["emit-c", "-o", target.to_str().unwrap(), "vdp.ode"]);
    assert_eq!(code(&again), 0);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), c);
    assert_eq!(stdout(&run(&["emit-c", "vdp.ode"])), c);
    // An unwritable target is an error, not a panic.
    let bad = dir.join("no_such_dir").join("x.c");
    let o = run(&["emit-c", "vdp.ode", "-o", bad.to_str().unwrap()]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("cannot write"), "{}", stderr(&o));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn emit_c_of_a_model_without_parameters_still_has_three_functions() {
    // An ode with no `par` line: JacP has no outputs.
    let o = run(&["emit-c", "plain.ode"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_c_text(&stdout(&o));
}

#[test]
fn bad_flags_are_usage_errors() {
    for args in [
        &["ir"][..],
        &["ir", "vdp.ode", "--points", "3"],
        &["ir", "vdp.ode", "--json"],
        &["check-derivs", "vdp.ode", "--points", "0"],
        &["check-derivs", "vdp.ode", "--points", "many"],
        &["check-derivs", "vdp.ode", "--points"],
        &["check-derivs", "vdp.ode", "--seed", "-4"],
        &["check-derivs", "vdp.ode", "--seed", "x"],
        &["check-derivs", "vdp.ode", "-o", "x.c"],
        &["emit-c"],
        &["emit-c", "vdp.ode", "-o"],
        &["emit-c", "vdp.ode", "--bogus"],
        &["emit-c", "vdp.ode", "extra.ode"],
    ] {
        let o = run(args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(stdout(&o).is_empty(), "{args:?}");
        assert!(stderr(&o).starts_with("pc-cli: "), "{args:?}");
        assert!(stderr(&o).contains("usage: pc-cli"), "{args:?}");
    }
}

#[test]
fn help_for_each_command_exits_0() {
    for (cmd, word) in [
        ("ir", "pc-cli ir"),
        ("check-derivs", "--points N"),
        ("emit-c", "-o FILE"),
    ] {
        let o = run(&[cmd, "--help"]);
        assert_eq!(code(&o), 0, "{cmd}");
        assert!(stdout(&o).contains(word), "{cmd}: {}", stdout(&o));
    }
    assert!(stdout(&run(&["--help"])).contains("check-derivs"));
}
