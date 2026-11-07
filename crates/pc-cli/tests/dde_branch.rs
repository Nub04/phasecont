//! Runs `pc-cli dde-branch` on Hutchinson's equation x' = r x (1 - x(t - tau)) with tau = 1.
//! The equilibrium x = 1 has the characteristic equation lambda = -r exp(-lambda), and a
//! pair of roots crosses the imaginary axis at r = pi/2 with omega = pi/2. Also runs the
//! bad problem files and the model errors.

mod common;

use std::f64::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const HUTCHINSON: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_branch");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `text` to a file called `name` in the scratch directory and returns its path.
/// The text goes to a file of its own first and is renamed over `name`, so that a reader
/// never sees a half-written file when tests run in parallel.
fn write(name: &str, text: &str) -> String {
    let path = scratch().join(name);
    let tmp = scratch().join(format!("{name}.{:?}.tmp", std::thread::current().id()));
    std::fs::write(&tmp, text).unwrap();
    std::fs::rename(&tmp, &path).unwrap();
    path.to_string_lossy().into_owned()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// The problem of SPEC 8e test 1, with the model named by a relative path.
fn hutchinson() -> String {
    write("hutchinson.ode", HUTCHINSON);
    r#"{"model":"hutchinson.ode","params":{"tau":1.0,"r":1.3},"free":"r","x0":[1.0],
        "n":30,"re_min":-0.5,
        "options":{"direction":1,"hmax":0.05,"max_steps":400,"bounds":[1.2,2.0]}}"#
        .to_string()
}

#[test]
fn the_hopf_point_of_hutchinson_is_found() {
    let file = write("hutchinson.json", &hutchinson());
    let o = run(&scratch(), &["dde-branch", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds", "{}", stderr(&o));
    assert!(v["warnings"].as_array().unwrap().is_empty(), "{v}");
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    let s = &specials[0];
    assert_eq!(s["label"], "H");
    assert!((num(&s["alpha"]) - FRAC_PI_2).abs() <= 1e-9, "{s}");
    assert!((num(&s["omega"]) - FRAC_PI_2).abs() <= 1e-9, "{s}");
    assert!((num(&s["x"][0]) - 1.0).abs() <= 1e-10, "{s}");
    assert_eq!(s["unstable_before"], 0);
    assert_eq!(s["unstable_after"], 2);
    let between = s["between"].as_array().unwrap();
    assert_eq!(between.len(), 2);
    assert_eq!(
        between[1].as_u64().unwrap(),
        between[0].as_u64().unwrap() + 1
    );
    let points = v["points"].as_array().unwrap();
    assert!(points.len() > 10, "{} points", points.len());
    for p in points {
        let unstable = p["unstable"].as_u64().unwrap();
        assert_eq!(
            unstable,
            if num(&p["alpha"]) < FRAC_PI_2 { 0 } else { 2 },
            "{p}"
        );
        assert!((num(&p["x"][0]) - 1.0).abs() <= 1e-10, "{p}");
    }
    let last = points.last().unwrap();
    assert!((num(&last["alpha"]) - 2.0).abs() <= 1e-9, "{last}");
}

#[test]
fn a_fold_has_a_null_omega() {
    // x' = mu - x^2 - 0.5 x(t - 1): a fold at mu = -1/16, x = -1/4 (SPEC 8e test 4).
    write("fold.ode", "par mu=-0.04\nx'=mu-x*x-0.5*delay(x,1)\ndone\n");
    let file = write(
        "fold.json",
        r#"{"model":"fold.ode","params":{"mu":-0.04},"free":"mu","x0":[-0.1],
            "n":30,"re_min":-0.5,
            "options":{"direction":-1,"hmax":0.01,"max_steps":400,"bounds":[-0.07,-0.03]}}"#,
    );
    let o = run(&scratch(), &["dde-branch", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    let s = &specials[0];
    assert_eq!(s["label"], "LP");
    assert!(s["omega"].is_null(), "{s}");
    assert!((num(&s["alpha"]) + 0.0625).abs() <= 1e-8, "{s}");
    assert!((num(&s["x"][0]) + 0.25).abs() <= 1e-8, "{s}");
    assert_eq!(s["unstable_before"], 0);
    assert_eq!(s["unstable_after"], 1);
}

#[test]
fn the_model_path_is_relative_to_the_problem_file() {
    write("hutchinson.ode", HUTCHINSON);
    let sub = scratch().join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    let file = sub.join("p.json");
    let text = hutchinson().replace("\"hutchinson.ode\"", "\"../hutchinson.ode\"");
    std::fs::write(&file, text).unwrap();
    // Run from another directory than the one that holds the model.
    let o = run(&sub, &["dde-branch", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["specials"].as_array().unwrap().len(), 1, "{v}");
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("hutchinson.ode", HUTCHINSON);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-branch", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = hutchinson();
    for (key, text) in [
        ("model", r#""model":"hutchinson.ode","#),
        ("params", r#""params":{"tau":1.0,"r":1.3},"#),
        ("free", r#""free":"r","#),
        ("x0", r#""x0":[1.0],"#),
        ("n", r#""n":30,"#),
    ] {
        let missing = good.replace(text, "");
        let msg = bad(&format!("no_{key}.json"), &missing);
        assert!(
            msg.contains(&format!("missing key \"{key}\"")),
            "{key}: {msg}"
        );
    }
    let no_re = good.replace(r#""re_min":-0.5,"#, "");
    assert!(bad("no_re.json", &no_re).contains("missing key \"re_min\""));
    let extra = good.replace(r#""n":30,"#, r#""n":30,"extra":0,"#);
    assert!(bad("extra.json", &extra).contains("unknown key \"extra\""));
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    let unknown_free = good.replace(r#""free":"r""#, r#""free":"q""#);
    assert!(bad("free.json", &unknown_free).contains("no parameter \"q\""));
    let unknown_par = good.replace(r#""r":1.3"#, r#""rr":1.3"#);
    assert!(bad("par.json", &unknown_par).contains("no parameter \"rr\""));
    let wrong_len = good.replace("[1.0]", "[1.0,2.0]");
    assert!(bad("len.json", &wrong_len).contains("1 state"));
    let bad_type = good.replace("[1.0]", r#""1.0""#);
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
    let free_type = good.replace(r#""free":"r""#, r#""free":0"#);
    assert!(bad("free_type.json", &free_type).contains("\"free\""));
    let detect_h = good.replace(r#""hmax":0.05"#, r#""detect":{"h":true}"#);
    assert!(bad("detect_h.json", &detect_h).contains("options.detect.h"));
    let mesh = good.replace(r#""hmax":0.05"#, r#""mesh_adapt":2"#);
    assert!(bad("mesh.json", &mesh).contains("options.mesh_adapt"));
    let bad_option = good.replace(r#""hmax":0.05"#, r#""hmax":-1"#);
    assert!(bad("option.json", &bad_option).contains("options.hmax"));
    let zero = good.replace(r#""re_min":-0.5"#, r#""re_min":0.0"#);
    assert!(bad("re_zero.json", &zero).contains("re_min"));
    let big = good.replace(r#""n":30"#, r#""n":4000"#);
    assert!(bad("big.json", &big).contains("must be at most 4000"));
}

#[test]
fn a_delay_time_as_free_parameter_needs_a_positive_lower_bound() {
    let good = hutchinson();
    let tau = good.replace(r#""free":"r""#, r#""free":"tau""#);
    // No bounds at all, and a lower bound of 0.
    let no_bounds = tau.replace(r#","bounds":[1.2,2.0]"#, "");
    assert!(bad("tau_nobounds.json", &no_bounds).contains("delay time"));
    let zero = tau.replace("[1.2,2.0]", "[0.0,2.0]");
    assert!(bad("tau_zero.json", &zero).contains("delay time"));
    // With a positive lower bound the run is accepted.
    let ok = tau.replace("[1.2,2.0]", "[0.5,2.0]");
    let file = write("tau_ok.json", &ok);
    let o = run(&scratch(), &["dde-branch", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-branch", &missing.to_string_lossy()]);
    assert_eq!(code(&o), 2);
    assert!(
        stderr(&o).contains("cannot read problem file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn a_model_that_cannot_be_used_exits_one() {
    let good = hutchinson();
    // An unreadable model file.
    let gone = good.replace("hutchinson.ode", "no_such_model.ode");
    let o = run(&scratch(), &["dde-branch", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par r=1.3, tau=1\nx'=-r*x\ndone\n");
    let plain = good.replace("hutchinson.ode", "plain.ode");
    let o = run(&scratch(), &["dde-branch", &write("plain.json", &plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
    // A failed Newton start: the field is the constant c, so f = 0 has no solution.
    write("noeq.ode", "par c=1\nx'=c+0*delay(x,1)\ndone\n");
    let noeq = r#"{"model":"noeq.ode","params":{},"free":"c","x0":[0.0],"n":5,"re_min":-1.0,
        "options":{"bounds":[0.5,2.0]}}"#;
    let o = run(&scratch(), &["dde-branch", &write("noeq.json", noeq)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stdout(&o).is_empty());
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-branch"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-branch needs a problem file"));
    let o = run(&scratch(), &["dde-branch", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-branch"));
}
