//! Runs `pc-cli dde-hopf` on Hutchinson's equation x' = r x (1 - x(t - tau)). The Hopf curve
//! is r tau = pi/2 with x = 1 and omega = r (SPEC section 8f, test 7). Also runs the bad
//! problem files and the model errors.

mod common;

use std::f64::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const HUTCHINSON: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_hopf");
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

/// The problem of SPEC 8f test 2 for `direction` 1, with the model named by a relative path.
fn hutchinson() -> String {
    write("hutchinson.ode", HUTCHINSON);
    r#"{"model":"hutchinson.ode","params":{"r":1.57,"tau":1.0},"free":["r","tau"],
        "x0":[1.001],"omega":1.57,
        "options":{"direction":1,"hmax":0.05,"max_steps":400,"bounds":[0.5,3.0],
                   "tol_x":1e-12,"tol_f":1e-12,"max_newton":20}}"#
        .to_string()
}

#[test]
fn the_hopf_curve_of_hutchinson_is_followed() {
    let file = write("hutchinson.json", &hutchinson());
    let o = run(&scratch(), &["dde-hopf", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds", "{}", stderr(&o));
    assert!(v["warnings"].as_array().unwrap().is_empty(), "{v}");
    assert!(v["specials"].as_array().unwrap().is_empty(), "{v}");
    let points = v["points"].as_array().unwrap();
    assert!(points.len() > 10, "{} points", points.len());
    for p in points {
        let alpha = p["alpha"].as_array().unwrap();
        assert_eq!(alpha.len(), 2, "{p}");
        let (r, tau) = (num(&alpha[0]), num(&alpha[1]));
        assert!((r * tau - FRAC_PI_2).abs() <= 1e-8, "{p}");
        assert!((num(&p["kappa"]) - r * r).abs() <= 1e-8, "{p}");
        assert!((num(&p["omega"]) - r).abs() <= 1e-8, "{p}");
        assert!((num(&p["x"][0]) - 1.0).abs() <= 1e-8, "{p}");
    }
    // The polished start has r = pi/2, and tau grows to the upper bound.
    let first = &points[0];
    assert!(
        (num(&first["alpha"][0]) - FRAC_PI_2).abs() <= 1e-10,
        "{first}"
    );
    let last = points.last().unwrap();
    assert!((num(&last["alpha"][1]) - 3.0).abs() <= 1e-8, "{last}");
}

#[test]
fn a_parameter_left_out_of_params_keeps_the_model_value() {
    // The model file has tau = 1, which is also the value the test above gives.
    let text = hutchinson().replace(r#","tau":1.0}"#, "}");
    let file = write("no_tau_value.json", &text);
    let o = run(&scratch(), &["dde-hopf", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds");
    let first = &v["points"][0];
    assert!(
        (num(&first["alpha"][0]) - FRAC_PI_2).abs() <= 1e-10,
        "{first}"
    );
}

#[test]
fn a_bt_point_is_printed_as_a_special() {
    // x' = c - a x - b x(t - tau) - x^2 with c = -1/4 and b = 1: BT at x = 1/2,
    // (a, tau) = (-2, 1) (SPEC 8f test 4).
    write(
        "bt.ode",
        "par a=-2, tau=1, b=1, c=-0.25\nx'=c-a*x-b*delay(x,tau)-x*x\ndone\n",
    );
    let file = write(
        "bt.json",
        r#"{"model":"bt.ode","params":{"a":-2.24,"tau":1.3613444250345883},
            "free":["a","tau"],"x0":[1.001],"omega":0.97,
            "options":{"direction":-1,"hmax":0.05,"max_steps":400,"bounds":[0.8,1.5],
                       "tol_x":1e-12,"tol_f":1e-12,"max_newton":20}}"#,
    );
    let o = run(&scratch(), &["dde-hopf", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds", "{}", stderr(&o));
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    let s = &specials[0];
    assert_eq!(s["label"], "BT");
    assert!((num(&s["alpha"][0]) + 2.0).abs() <= 1e-8, "{s}");
    assert!((num(&s["alpha"][1]) - 1.0).abs() <= 1e-8, "{s}");
    assert!((num(&s["x"][0]) - 0.5).abs() <= 1e-8, "{s}");
    assert!(num(&s["kappa"]).abs() <= 1e-8, "{s}");
    let index = s["index"].as_u64().unwrap() as usize;
    let points = v["points"].as_array().unwrap();
    assert!(index > 0 && index < points.len(), "index {index}");
    // The frequency is null where kappa is not positive.
    for p in points {
        assert_eq!(p["omega"].is_null(), num(&p["kappa"]) <= 0.0, "{p}");
    }
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
    let o = run(&sub, &["dde-hopf", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds", "{}", stderr(&o));
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("hutchinson.ode", HUTCHINSON);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-hopf", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = hutchinson();
    for (key, text) in [
        ("model", r#""model":"hutchinson.ode","#),
        ("params", r#""params":{"r":1.57,"tau":1.0},"#),
        ("free", r#""free":["r","tau"],"#),
        ("x0", r#""x0":[1.001],"#),
        ("omega", r#""omega":1.57,"#),
    ] {
        let missing = good.replace(text, "");
        let msg = bad(&format!("no_{key}.json"), &missing);
        assert!(
            msg.contains(&format!("missing key \"{key}\"")),
            "{key}: {msg}"
        );
    }
    let extra = good.replace(r#""omega":1.57,"#, r#""omega":1.57,"extra":0,"#);
    assert!(bad("extra.json", &extra).contains("unknown key \"extra\""));
    // The keys of dde-branch and dde-eq that mean nothing here are unknown.
    let n = good.replace(r#""omega":1.57,"#, r#""omega":1.57,"n":30,"#);
    assert!(bad("n.json", &n).contains("unknown key \"n\""));
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    assert!(bad("array.json", "[1]").contains("JSON object"));
    let one = good.replace(r#"["r","tau"]"#, r#"["r"]"#);
    assert!(bad("free_one.json", &one).contains("\"free\""));
    let three = good.replace(r#"["r","tau"]"#, r#"["r","tau","r"]"#);
    assert!(bad("free_three.json", &three).contains("\"free\""));
    let twice = good.replace(r#"["r","tau"]"#, r#"["r","r"]"#);
    assert!(bad("free_twice.json", &twice).contains("twice"));
    let scalar = good.replace(r#"["r","tau"]"#, r#""r""#);
    assert!(bad("free_scalar.json", &scalar).contains("\"free\""));
    let number = good.replace(r#"["r","tau"]"#, r#"["r",1]"#);
    assert!(bad("free_number.json", &number).contains("\"free\""));
    let unknown_free = good.replace(r#"["r","tau"]"#, r#"["r","q"]"#);
    assert!(bad("free_unknown.json", &unknown_free).contains("no parameter \"q\""));
    let unknown_par = good.replace(r#""r":1.57"#, r#""rr":1.57"#);
    assert!(bad("par.json", &unknown_par).contains("no parameter \"rr\""));
    let wrong_len = good.replace("[1.001]", "[1.0,2.0]");
    assert!(bad("len.json", &wrong_len).contains("1 state"));
    let bad_type = good.replace("[1.001]", r#""1.0""#);
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
    let bad_omega = good.replace(r#""omega":1.57"#, r#""omega":"fast""#);
    assert!(bad("omega_type.json", &bad_omega).contains("\"omega\""));
    let zero = good.replace(r#""omega":1.57"#, r#""omega":0.0"#);
    assert!(bad("omega_zero.json", &zero).contains("omega"));
    let detect = good.replace(r#""hmax":0.05"#, r#""detect":{"lp":false}"#);
    assert!(bad("detect.json", &detect).contains("options.detect"));
    let mesh = good.replace(r#""hmax":0.05"#, r#""mesh_adapt":2"#);
    assert!(bad("mesh.json", &mesh).contains("options.mesh_adapt"));
    let bad_option = good.replace(r#""hmax":0.05"#, r#""hmax":-1"#);
    assert!(bad("option.json", &bad_option).contains("options.hmax"));
    let lyap = good.replace(r#""omega":1.57,"#, r#""omega":1.57,"lyapunov":1,"#);
    assert!(bad("lyapunov_type.json", &lyap).contains("\"lyapunov\""));
}

// ---- The opt-in first Lyapunov coefficient (SPEC 8g-2) ----
//
// u' = -a u(t - 1) (1 + u) + b u^3 has the Hopf line a = pi/2, omega = pi/2 for every b,
// and l1(b) = l1(0) + 6 b / (pi (1 + pi^2/4)) with l1(0) = -0.21413092244177392. The
// generalized Hopf point is at b = 0.38875...

const WRIGHT3: &str = "par a=1.5, b=0\nu'=-a*delay(u,1)*(1+u)+b*u^3\ndone\n";
const L1_AT_ZERO: f64 = -0.214_130_922_441_773_92;

fn wright3(extra: &str) -> String {
    write("wright3.ode", WRIGHT3);
    format!(
        r#"{{"model":"wright3.ode","params":{{"a":1.57,"b":0.0}},"free":["a","b"],
            "x0":[0.001],"omega":1.57,{extra}
            "options":{{"direction":1,"hmax":0.05,"max_steps":400,"bounds":[-1.0,1.0],
                       "tol_x":1e-12,"tol_f":1e-12,"max_newton":20}}}}"#
    )
}

#[test]
fn lyapunov_adds_l1_to_every_point_and_labels_gh() {
    let slope = 6.0 / (std::f64::consts::PI * (1.0 + std::f64::consts::PI.powi(2) / 4.0));
    let file = write("wright3_on.json", &wright3(r#""lyapunov":true,"#));
    let o = run(&scratch(), &["dde-hopf", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["stop"], "bounds", "{}", stderr(&o));
    let points = v["points"].as_array().unwrap();
    assert!(points.len() > 20, "{} points", points.len());
    for p in points {
        let b = num(&p["alpha"][1]);
        let want = L1_AT_ZERO + slope * b;
        assert!((num(&p["l1"]) - want).abs() <= 1e-7, "{p}");
    }
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    let gh = &specials[0];
    assert_eq!(gh["label"], "GH");
    assert!(
        (num(&gh["alpha"][1]) + L1_AT_ZERO / slope).abs() <= 1e-6,
        "{gh}"
    );
    assert!(
        (num(&gh["alpha"][0]) - std::f64::consts::FRAC_PI_2).abs() <= 1e-8,
        "{gh}"
    );
}

#[test]
fn without_lyapunov_the_output_has_no_l1_and_false_equals_absent() {
    let absent = write("wright3_absent.json", &wright3(""));
    let off = write("wright3_off.json", &wright3(r#""lyapunov":false,"#));
    let a = run(&scratch(), &["dde-hopf", &absent]);
    let b = run(&scratch(), &["dde-hopf", &off]);
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    assert_eq!(code(&b), 0, "{}", stderr(&b));
    assert_eq!(
        stdout(&a),
        stdout(&b),
        "false must print what an absent key prints"
    );
    let v: Value = serde_json::from_str(&stdout(&a)).unwrap();
    assert!(v["specials"].as_array().unwrap().is_empty(), "{v}");
    for p in v["points"].as_array().unwrap() {
        assert!(p.get("l1").is_none(), "{p}");
    }
}

#[test]
fn a_delay_time_as_second_parameter_needs_a_positive_lower_bound() {
    let good = hutchinson();
    // No bounds at all, and a lower bound of 0. tau is the second name in free.
    let no_bounds = good.replace(r#""bounds":[0.5,3.0],"#, "");
    assert!(bad("tau_nobounds.json", &no_bounds).contains("delay time"));
    let zero = good.replace("[0.5,3.0]", "[0.0,3.0]");
    assert!(bad("tau_zero.json", &zero).contains("delay time"));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-hopf", &missing.to_string_lossy()]);
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
    let o = run(&scratch(), &["dde-hopf", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par r=1.5, tau=1\nx'=-r*x\ndone\n");
    let plain = good.replace("hutchinson.ode", "plain.ode");
    let o = run(&scratch(), &["dde-hopf", &write("plain.json", &plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
    // A start that Newton's method cannot polish: one iteration is not enough.
    let slow = good.replace(r#""max_newton":20"#, r#""max_newton":1"#);
    let o = run(&scratch(), &["dde-hopf", &write("slow.json", &slow)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{}", stdout(&o));
    assert!(!stderr(&o).is_empty());
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-hopf"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-hopf needs a problem file"));
    let o = run(&scratch(), &["dde-hopf", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-hopf"));
}

// Test 12 of SPEC section 8h: `l2` on a GH special. The Wright equation with a quintic term,
// u' = -a u(t-1) (1+u) + b u^3 + c u^5, has the curve a = pi/2 and the same l1 as without
// the quintic term, so the GH point stays at b = 0.38875... and l2 = 1.7362145915396248 for
// c = 1 (numpy oracle of section 8h, test 11).

const QUINTIC_MODEL: &str = "par a=1.5, b=0, c=1\nu'=-a*delay(u,1)*(1+u)+b*u^3+c*u^5\ndone\n";
// The same, with a decoupled block z' = 4.71238898038469 z(t-1). At omega = pi/2 its
// Delta_z(3 i omega) is zero up to rounding, so lyapunov2 refuses every point of the curve.
const QUINTIC_Z_MODEL: &str = "par a=1.5, b=0, c=1\nu'=-a*delay(u,1)*(1+u)+b*u^3+c*u^5\n\
                               z'=4.71238898038469*delay(z,1)\ndone\n";
const L2_AT_GH: f64 = 1.736_214_591_539_624_8;

fn quintic(model: &str, text: &str, x0: &str, extra: &str) -> String {
    write(model, text);
    format!(
        r#"{{"model":"{model}","params":{{"a":1.57,"b":0.0}},"free":["a","b"],
            "x0":{x0},"omega":1.57,{extra}
            "options":{{"direction":1,"hmax":0.05,"max_steps":400,"bounds":[-1.0,1.0],
                       "tol_x":1e-12,"tol_f":1e-12,"max_newton":20}}}}"#
    )
}

#[test]
fn a_gh_special_has_l2_and_no_other_key_changes() {
    let on = write(
        "quintic_on.json",
        &quintic(
            "quintic.ode",
            QUINTIC_MODEL,
            "[0.001]",
            r#""lyapunov":true,"#,
        ),
    );
    let o = run(&scratch(), &["dde-hopf", &on]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    let gh = &specials[0];
    assert_eq!(gh["label"], "GH");
    let l2 = num(&gh["l2"]);
    assert!((l2 - L2_AT_GH).abs() <= 1e-4, "l2 = {l2}, want {L2_AT_GH}");
    for s in specials.iter().filter(|s| s["label"] != "GH") {
        assert!(s.get("l2").is_none(), "{s}");
    }
    for p in v["points"].as_array().unwrap() {
        assert!(p.get("l2").is_none(), "{p}");
    }
}

#[test]
fn l2_is_null_where_lyapunov2_refuses_the_gh_point() {
    let on = write(
        "quintic_z_on.json",
        &quintic(
            "quintic_z.ode",
            QUINTIC_Z_MODEL,
            "[0.001, 0.0]",
            r#""lyapunov":true,"#,
        ),
    );
    let o = run(&scratch(), &["dde-hopf", &on]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let specials = v["specials"].as_array().unwrap();
    assert_eq!(specials.len(), 1, "{v}");
    assert_eq!(specials[0]["label"], "GH");
    assert!(
        specials[0].get("l2").is_some_and(Value::is_null),
        "the key must be present and null: {}",
        specials[0]
    );
}

#[test]
fn without_lyapunov_no_special_has_l2_and_false_equals_absent() {
    let absent = write(
        "quintic_absent.json",
        &quintic("quintic.ode", QUINTIC_MODEL, "[0.001]", ""),
    );
    let off = write(
        "quintic_off.json",
        &quintic(
            "quintic.ode",
            QUINTIC_MODEL,
            "[0.001]",
            r#""lyapunov":false,"#,
        ),
    );
    let a = run(&scratch(), &["dde-hopf", &absent]);
    let b = run(&scratch(), &["dde-hopf", &off]);
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    assert_eq!(stdout(&a), stdout(&b));
    let v: Value = serde_json::from_str(&stdout(&a)).unwrap();
    for s in v["specials"].as_array().unwrap() {
        assert!(s.get("l2").is_none(), "{s}");
    }
}
