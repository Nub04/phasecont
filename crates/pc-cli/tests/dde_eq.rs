//! Runs `pc-cli dde-eq` on the delayed logistic equation x' = r x (1 - x(t - tau)), whose
//! equilibrium x = 1 has the characteristic equation lambda = -r exp(-lambda tau). At
//! r tau = pi/2 the roots with real part zero are exactly +-i pi/(2 tau). Also runs the
//! bad problem files, and checks that `cont` and `ir` still refuse a delay model.

mod common;

use std::f64::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const LOGISTIC: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_eq");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `text` to a file called `name` in the scratch directory and returns its path.
fn write(name: &str, text: &str) -> String {
    let path = scratch().join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

/// The problem for the logistic model at r = pi/2, with the model named by a relative path.
fn neutral() -> String {
    write("logistic.ode", LOGISTIC);
    format!(
        r#"{{"model":"logistic.ode","params":{{"r":{:?}}},"x0":[0.8],"n":20,"re_min":-3.0}}"#,
        FRAC_PI_2
    )
}

#[test]
fn the_equilibrium_and_the_neutral_pair_are_found() {
    let file = write("neutral.json", &neutral());
    let o = run(&scratch(), &["dde-eq", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let x = v["x"].as_array().unwrap();
    assert_eq!(x.len(), 1);
    assert!((num(&x[0]) - 1.0).abs() <= 1e-10, "x = {}", x[0]);
    assert!(num(&v["residual"]) <= 1e-10, "{v}");
    let delays = v["delays"].as_array().unwrap();
    assert_eq!(delays.len(), 1);
    assert_eq!(num(&delays[0]), 1.0);
    assert!(v["unrefined"].as_array().unwrap().is_empty(), "{v}");
    let roots = v["roots"].as_array().unwrap();
    for r in roots {
        assert!(num(&r["residual"]) <= 1e-12, "{r}");
    }
    assert!(num(&roots[0]["re"]).abs() <= 1e-10, "{v}");
    for sign in [1.0, -1.0] {
        let hit = roots.iter().any(|r| {
            num(&r["re"]).abs() <= 1e-10 && (num(&r["im"]) - sign * FRAC_PI_2).abs() <= 1e-10
        });
        assert!(hit, "no root at {}i in {v}", sign * FRAC_PI_2);
    }
}

#[test]
fn a_parameter_not_given_keeps_the_value_of_the_model_file() {
    // The file has r = 2 and tau = 1, so r tau > pi/2 and the leading pair has Re > 0.
    write(
        "defaults.ode",
        "par r=2, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n",
    );
    let file = write(
        "defaults.json",
        r#"{"model":"defaults.ode","params":{},"x0":[0.8],"n":20,"re_min":-1.0}"#,
    );
    let o = run(&scratch(), &["dde-eq", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let top = num(&v["roots"][0]["re"]);
    assert!(top > 1e-3, "a delay of 1 at r = 2 is unstable, got {top}");
}

#[test]
fn the_model_path_is_relative_to_the_problem_file() {
    write("logistic.ode", LOGISTIC);
    let sub = scratch().join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    let file = sub.join("p.json");
    std::fs::write(
        &file,
        r#"{"model":"../logistic.ode","params":{},"x0":[0.8],"n":10,"re_min":-1.0}"#,
    )
    .unwrap();
    // Run from another directory than the one that holds the model.
    let o = run(&sub, &["dde-eq", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("logistic.ode", LOGISTIC);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-eq", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = neutral();
    let no_model = good.replace(r#""model":"logistic.ode","#, "");
    assert!(bad("no_model.json", &no_model).contains("missing key \"model\""));
    let no_params = good.replace(r#""params":{"r":1.5707963267948966},"#, "");
    assert!(bad("no_params.json", &no_params).contains("missing key \"params\""));
    let no_x0 = good.replace(r#""x0":[0.8],"#, "");
    assert!(bad("no_x0.json", &no_x0).contains("missing key \"x0\""));
    let no_n = good.replace(r#""n":20,"#, "");
    assert!(bad("no_n.json", &no_n).contains("missing key \"n\""));
    let no_r = good.replace(r#","re_min":-3.0"#, "");
    assert!(bad("no_r.json", &no_r).contains("missing key \"re_min\""));
    let extra = good.replace(r#""n":20,"#, r#""n":20,"extra":0,"#);
    assert!(bad("extra.json", &extra).contains("unknown key \"extra\""));
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    let unknown = good.replace(r#""r":"#, r#""rr":"#);
    assert!(bad("unknown.json", &unknown).contains("no parameter \"rr\""));
    let wrong_len = good.replace("[0.8]", "[0.8,0.9]");
    assert!(bad("len.json", &wrong_len).contains("1 state"));
    let zero_tau = good.replace(r#""r":"#, r#""tau":0.0,"r":"#);
    assert!(bad("tau.json", &zero_tau).contains("delay time"));
    let big = good.replace(r#""n":20"#, r#""n":4000"#);
    assert!(bad("big.json", &big).contains("must be at most 4000"));
    let bad_type = good.replace("[0.8]", r#""0.8""#);
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-eq", &missing.to_string_lossy()]);
    assert_eq!(code(&o), 2);
    assert!(
        stderr(&o).contains("cannot read problem file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn a_model_that_cannot_be_used_exits_one() {
    let good = neutral();
    // An unreadable model file.
    let gone = good.replace("logistic.ode", "no_such_model.ode");
    let o = run(&scratch(), &["dde-eq", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par r=1\nx'=-r*x\ndone\n");
    let plain = good.replace("logistic.ode", "plain.ode");
    let o = run(&scratch(), &["dde-eq", &write("plain.json", &plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
    // A delay of the wrong form.
    write("wrong.ode", "par tau=1\nx'=-delay(x,2*tau)\ndone\n");
    let wrong = good.replace("logistic.ode", "wrong.ode");
    let o = run(&scratch(), &["dde-eq", &write("wrong.json", &wrong)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(
        stderr(&o).contains("delay(x,2*tau)"),
        "the call is named: {}",
        stderr(&o)
    );
    // A model that reads t.
    write("forced.ode", "par tau=1\nx'=-delay(x,tau)+t\ndone\n");
    let forced = r#"{"model":"forced.ode","params":{},"x0":[0.8],"n":20,"re_min":-3.0}"#;
    let o = run(&scratch(), &["dde-eq", &write("forced.json", forced)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
}

#[test]
fn a_failed_newton_run_exits_one() {
    // The field is the constant c, so f(x, x, p) = 0 has no solution and Newton's method
    // cannot converge. The delay call keeps the model valid for `lower_dde`.
    write("noeq.ode", "par c=1\nx'=c+0*delay(x,1)\ndone\n");
    let file = write(
        "noeq.json",
        r#"{"model":"noeq.ode","params":{},"x0":[0.0],"n":5,"re_min":-1.0}"#,
    );
    let o = run(&scratch(), &["dde-eq", &file]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stdout(&o).is_empty());
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-eq"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-eq needs a problem file"));
    let o = run(&scratch(), &["dde-eq", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-eq"));
}

#[test]
fn the_other_commands_still_refuse_a_delay() {
    let model = write("logistic.ode", LOGISTIC);
    let o = run(&scratch(), &["ir", &model]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
    let cont = write(
        "cont.json",
        r#"{"type":"EP","free":0,"u0":[1.0],"steps":3}"#,
    );
    let o = run(&scratch(), &["cont", &model, &cont]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("delay"), "{}", stderr(&o));
}
