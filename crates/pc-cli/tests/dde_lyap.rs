//! Runs `pc-cli dde-lyap` on Wright's equation u' = -a u(t-1) (1 + u). At a = omega = pi/2
//! the first Lyapunov coefficient is 2(2 - 3 pi)/(5(4 + pi^2)) (SPEC section 8g, test 7).
//! Also runs the bad problem files and the model errors.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const WRIGHT: &str = "par a=1.0\nu'=-a*delay(u,1)*(1+u)\ndone\n";

/// SPEC 8g, test 1: the closed form and the numpy value of Im c1.
const L1: f64 = -0.21413092244177392;
const C1: [f64; 2] = [-0.33635606642474136, -0.41413092244177396];

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_lyap");
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

/// The problem of SPEC 8g test 1, with the model named by a relative path.
fn wright() -> String {
    write("wright.ode", WRIGHT);
    r#"{"model":"wright.ode","params":{"a":1.5707963267948966},"x0":[0.0],
        "omega":1.5707963267948966}"#
        .to_string()
}

#[test]
fn the_coefficient_of_wright_matches_the_closed_form() {
    let file = write("wright.json", &wright());
    let o = run(&scratch(), &["dde-lyap", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!((num(&v["l1"]) - L1).abs() <= 1e-9, "{v}");
    let c1 = v["c1"].as_array().unwrap();
    assert_eq!(c1.len(), 2, "{v}");
    assert!((num(&c1[0]) - C1[0]).abs() <= 1e-9, "{v}");
    assert!((num(&c1[1]) - C1[1]).abs() <= 1e-9, "{v}");
    assert!(
        (num(&v["l1"]) - num(&c1[0]) / num(&v["omega"])).abs() <= 1e-15,
        "{v}"
    );
    assert!(
        (num(&v["omega"]) - std::f64::consts::FRAC_PI_2).abs() <= 1e-15,
        "{v}"
    );
    assert!(num(&v["null_residual"]) <= 1e-12, "{v}");
    assert_eq!(num(&v["equilibrium_residual"]), 0.0, "{v}");
}

#[test]
fn a_parameter_left_out_of_params_keeps_the_model_value() {
    // The model file has a = 1, which is not a Hopf point at omega = pi/2, so the run is
    // refused. Putting a back in makes it pass.
    let text = wright().replace(r#""params":{"a":1.5707963267948966}"#, r#""params":{}"#);
    let o = run(&scratch(), &["dde-lyap", &write("no_a.json", &text)]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("Hopf frequency"), "{}", stderr(&o));
}

#[test]
fn the_model_path_is_relative_to_the_problem_file() {
    write("wright.ode", WRIGHT);
    let sub = scratch().join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    let file = sub.join("p.json");
    let text = wright().replace("\"wright.ode\"", "\"../wright.ode\"");
    std::fs::write(&file, text).unwrap();
    // Run from another directory than the one that holds the model.
    let o = run(&sub, &["dde-lyap", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!((num(&v["l1"]) - L1).abs() <= 1e-9, "{v}");
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("wright.ode", WRIGHT);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-lyap", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = wright();
    for (key, text) in [
        ("model", r#""model":"wright.ode","#),
        ("params", r#""params":{"a":1.5707963267948966},"#),
        ("x0", r#""x0":[0.0],"#),
    ] {
        let missing = good.replace(text, "");
        let msg = bad(&format!("no_{key}.json"), &missing);
        assert!(
            msg.contains(&format!("missing key \"{key}\"")),
            "{key}: {msg}"
        );
    }
    let mut no_omega: Value = serde_json::from_str(&good).unwrap();
    no_omega.as_object_mut().unwrap().remove("omega");
    let msg = bad("no_omega.json", &no_omega.to_string());
    assert!(msg.contains("missing key \"omega\""), "{msg}");
    // The keys of dde-hopf that mean nothing here are unknown.
    for key in ["free", "options"] {
        let extra = good.replace(r#""x0":[0.0],"#, &format!(r#""x0":[0.0],"{key}":0,"#));
        let msg = bad(&format!("extra_{key}.json"), &extra);
        assert!(msg.contains(&format!("unknown key \"{key}\"")), "{msg}");
    }
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    assert!(bad("array.json", "[1]").contains("JSON object"));
    let unknown_par = good.replace(r#""a":1.57"#, r#""b":1.57"#);
    assert!(bad("par.json", &unknown_par).contains("no parameter \"b\""));
    let wrong_len = good.replace("[0.0]", "[0.0,1.0]");
    assert!(bad("len.json", &wrong_len).contains("1 state"));
    let bad_type = good.replace("[0.0]", r#""0.0""#);
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
    let bad_par = good.replace(r#""a":1.5707963267948966"#, r#""a":"x""#);
    assert!(bad("par_type.json", &bad_par).contains("\"a\""));
    let bad_omega = good.replace(r#""omega":1.5707963267948966"#, r#""omega":"fast""#);
    assert!(bad("omega_type.json", &bad_omega).contains("\"omega\""));
}

#[test]
fn an_omega_that_is_not_a_hopf_frequency_exits_two() {
    let good = wright();
    let off = good.replace(r#""omega":1.5707963267948966"#, r#""omega":1.2"#);
    assert!(bad("omega_off.json", &off).contains("Hopf frequency"));
    let zero = good.replace(r#""omega":1.5707963267948966"#, r#""omega":0.0"#);
    assert!(bad("omega_zero.json", &zero).contains("omega"));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-lyap", &missing.to_string_lossy()]);
    assert_eq!(code(&o), 2);
    assert!(
        stderr(&o).contains("cannot read problem file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn a_model_that_cannot_be_used_exits_one() {
    let good = wright();
    // An unreadable model file.
    let gone = good.replace("wright.ode", "no_such_model.ode");
    let o = run(&scratch(), &["dde-lyap", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par a=1.0\nu'=-a*u\ndone\n");
    let plain = good.replace("wright.ode", "plain.ode");
    let o = run(&scratch(), &["dde-lyap", &write("plain.json", &plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-lyap"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-lyap needs a problem file"));
    let o = run(&scratch(), &["dde-lyap", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-lyap"));
}
