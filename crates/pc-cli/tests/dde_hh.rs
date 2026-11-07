//! Runs `pc-cli dde-hh` on the active control system of Bosschaert, Janssens and Kuznetsov
//! (arXiv 1903.08276 v1, Sec. 8.3), at the Hopf-Hopf point of SPEC section 8j, test 1. Also
//! runs the bad problem files and the model errors (test 7).

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const ACS: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427\n\
                   s'=tau*y\n\
                   y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1)+0.1*delay(s,1)^3)\n\
                   done\n";

/// SPEC 8j, test 1: the numpy values, each as [re, im].
const G2100: [f64; 2] = [-0.091_452_514_113_883_53, 0.121_439_974_396_385_96];
const G1011: [f64; 2] = [-0.308_448_621_979_563_9, 0.409_589_535_276_780_66];
const G1110: [f64; 2] = [0.215_070_731_513_221_98, 0.387_604_330_232_565_2];
const G0021: [f64; 2] = [0.181_346_219_418_256_54, 0.326_825_409_590_890_07];
const THETA: f64 = -1.700_882_560_270_852_2;
const DELTA: f64 = -2.351_720_273_599_037;
const PRODUCT: f64 = -0.016_584_567_690_847_526;

/// The keys of the output, in the order the spec fixes.
const KEYS: [&str; 13] = [
    "delta",
    "equilibrium_residual",
    "g0021",
    "g1011",
    "g1110",
    "g2100",
    "null_residual_1",
    "null_residual_2",
    "omega1",
    "omega2",
    "product",
    "sign",
    "theta",
];

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_hh");
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

/// `|got - want| <= rel |want|`, with `what` and the output in the message.
fn close(got: f64, want: f64, rel: f64, what: &str, v: &Value) {
    assert!(
        (got - want).abs() <= rel * want.abs(),
        "{what}: got {got}, want {want}\n{v}"
    );
}

/// The problem of SPEC 8j test 1, with the model named by a relative path and the two
/// frequencies given by the caller.
fn acs(omega1: f64, omega2: f64) -> String {
    write("acs.ode", ACS);
    format!(
        r#"{{"model":"acs.ode","params":{{"zeta":-0.01622437043055295,"tau":5.898023025699427}},
            "x0":[0.0,0.0],"omega1":{omega1:?},"omega2":{omega2:?}}}"#
    )
}

fn point1() -> String {
    acs(7.644_926_613_449_257_5, 4.527_486_915_496_89)
}

#[test]
fn the_published_point_matches_the_numpy_values() {
    let file = write("acs_1.json", &point1());
    let o = run(&scratch(), &["dde-hh", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = stdout(&o);
    let v: Value = serde_json::from_str(&text).unwrap();
    for (name, want) in [
        ("g2100", G2100),
        ("g1011", G1011),
        ("g1110", G1110),
        ("g0021", G0021),
    ] {
        let g = v[name].as_array().unwrap_or_else(|| panic!("{name}: {v}"));
        assert_eq!(g.len(), 2, "{name}: {v}");
        close(num(&g[0]), want[0], 1e-8, &format!("{name} re"), &v);
        close(num(&g[1]), want[1], 1e-8, &format!("{name} im"), &v);
    }
    close(num(&v["theta"]), THETA, 1e-8, "theta", &v);
    close(num(&v["delta"]), DELTA, 1e-8, "delta", &v);
    close(num(&v["product"]), PRODUCT, 1e-8, "product", &v);
    assert_eq!(num(&v["sign"]), -1.0, "{v}");
    close(
        num(&v["omega1"]),
        7.644_926_613_449_257_5,
        1e-15,
        "omega1",
        &v,
    );
    close(num(&v["omega2"]), 4.527_486_915_496_89, 1e-15, "omega2", &v);
    assert!(num(&v["null_residual_1"]) <= 1e-10, "{v}");
    assert!(num(&v["null_residual_2"]) <= 1e-10, "{v}");
    assert_eq!(num(&v["equilibrium_residual"]), 0.0, "{v}");
    // The thirteen keys, in alphabetical order, in the text itself and not only in the
    // parsed object.
    let mut at = 0;
    for key in KEYS {
        let found = text[at..]
            .find(&format!("\"{key}\":"))
            .unwrap_or_else(|| panic!("key {key} is missing or out of order in {text}"));
        at += found + key.len();
    }
    assert_eq!(v.as_object().unwrap().len(), KEYS.len(), "{v}");
}

#[test]
fn a_zero_denominator_prints_null() {
    // Two oscillators with no radial coupling: every coefficient is 0, so theta and delta
    // have a zero denominator and print as null, and the sign is 0.
    write(
        "plain_osc.ode",
        "x0'=-2.3*x1\nx1'=2.3*x0\nx2'=-1.0*x3\nx3'=1.0*x2\nw'=-2*w-delay(w,1)\ndone\n",
    );
    let text = r#"{"model":"plain_osc.ode","params":{},"x0":[0,0,0,0,0],
                    "omega1":2.3,"omega2":1.0}"#;
    let o = run(&scratch(), &["dde-hh", &write("plain_osc.json", text)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v["theta"].is_null(), "{v}");
    assert!(v["delta"].is_null(), "{v}");
    assert_eq!(num(&v["sign"]), 0.0, "{v}");
    assert_eq!(num(&v["product"]), 0.0, "{v}");
}

#[test]
fn a_parameter_left_out_of_params_keeps_the_model_value() {
    // The model file has the values of the point, so leaving zeta out gives the same numbers
    // as giving it.
    let without = point1().replace(r#""zeta":-0.01622437043055295,"#, "");
    assert_ne!(without, point1());
    let a = run(
        &scratch(),
        &["dde-hh", &write("acs_no_zeta.json", &without)],
    );
    let b = run(
        &scratch(),
        &["dde-hh", &write("acs_with_zeta.json", &point1())],
    );
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    assert_eq!(stdout(&a), stdout(&b));
}

#[test]
fn the_model_path_is_relative_to_the_problem_file() {
    // The model sits next to the problem file and nowhere else, and pc-cli runs from the
    // parent directory, so a path read against the working directory would not find it.
    let sub = scratch().join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("own_model.ode"), ACS).unwrap();
    let file = sub.join("p.json");
    let text = point1().replace("\"acs.ode\"", "\"own_model.ode\"");
    std::fs::write(&file, text).unwrap();
    assert!(!scratch().join("own_model.ode").exists());
    let o = run(&scratch(), &["dde-hh", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    close(num(&v["theta"]), THETA, 1e-8, "theta", &v);
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("acs.ode", ACS);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-hh", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = point1();
    for key in ["model", "params", "x0", "omega1", "omega2"] {
        let mut v: Value = serde_json::from_str(&good).unwrap();
        v.as_object_mut().unwrap().remove(key);
        let msg = bad(&format!("no_{key}.json"), &v.to_string());
        assert!(
            msg.contains(&format!("missing key \"{key}\"")),
            "{key}: {msg}"
        );
    }
    // The key of dde-zh and the keys of dde-lyap and dde-hopf that mean nothing here are
    // unknown.
    for key in ["omega", "l2", "free", "options"] {
        let mut v: Value = serde_json::from_str(&good).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert(key.to_string(), Value::from(0));
        let msg = bad(&format!("extra_{key}.json"), &v.to_string());
        assert!(msg.contains(&format!("unknown key \"{key}\"")), "{msg}");
    }
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    assert!(bad("array.json", "[1]").contains("JSON object"));
    let unknown_par = good.replace(r#""zeta""#, r#""zeta2""#);
    assert!(bad("par.json", &unknown_par).contains("no parameter \"zeta2\""));
    let wrong_len = good.replace("[0.0,0.0]", "[0.0]");
    assert_ne!(wrong_len, good);
    assert!(bad("len.json", &wrong_len).contains("2 states"));
    let bad_type = good.replace("[0.0,0.0]", "\"[0.0,0.0]\"");
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
    let bad_par = good.replace(r#""tau":5.898023025699427"#, r#""tau":"x""#);
    assert!(bad("par_type.json", &bad_par).contains("\"tau\""));
    let bad_omega1 = good.replace("7.6449266134492575", "\"fast\"");
    assert!(bad("omega1_type.json", &bad_omega1).contains("\"omega1\""));
    let bad_omega2 = good.replace("4.52748691549689", "\"slow\"");
    assert!(bad("omega2_type.json", &bad_omega2).contains("\"omega2\""));
}

#[test]
fn a_pair_that_is_not_a_hopf_hopf_point_exits_two_and_names_the_frequency() {
    assert!(bad("off_1.json", &acs(7.5, 4.527_486_915_496_89)).contains("Delta(i omega1)"));
    assert!(bad("off_2.json", &acs(7.644_926_613_449_257_5, 4.4)).contains("Delta(i omega2)"));
    let swapped = acs(4.527_486_915_496_89, 7.644_926_613_449_257_5);
    assert!(bad("swapped.json", &swapped).contains("greater than"));
    assert!(bad("zero.json", &acs(7.644_926_613_449_257_5, 0.0)).contains("omega2"));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-hh", &missing.to_string_lossy()]);
    assert_eq!(code(&o), 2);
    assert!(
        stderr(&o).contains("cannot read problem file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn a_model_that_cannot_be_used_exits_one() {
    let good = point1();
    // An unreadable model file.
    let gone = good.replace("acs.ode", "no_such_model.ode");
    let o = run(&scratch(), &["dde-hh", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par a=1.0\nu'=-a*u\ndone\n");
    let plain = r#"{"model":"plain.ode","params":{"a":1.0},"x0":[0.0],"omega1":2.0,"omega2":1.0}"#;
    let o = run(&scratch(), &["dde-hh", &write("plain.json", plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
}

#[test]
fn a_resonant_pair_exits_one() {
    // Oscillators at 2 and 1 and the decoupled w: a 1:2 resonance, so Delta(2 i omega2) is
    // singular and the engine refuses the point (SPEC 8j, test 6).
    write(
        "resonant.ode",
        "u0'=-2*v0\nv0'=2*u0\nu1'=-v1\nv1'=u1\nw'=-2*w-delay(w,1)\ndone\n",
    );
    let text = r#"{"model":"resonant.ode","params":{},"x0":[0,0,0,0,0],
                    "omega1":2.0,"omega2":1.0}"#;
    let o = run(&scratch(), &["dde-hh", &write("resonant.json", text)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stdout(&o).is_empty(), "stdout {}", stdout(&o));
    assert!(stderr(&o).contains("Delta(2 i omega2)"), "{}", stderr(&o));
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-hh"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-hh needs a problem file"));
    let o = run(&scratch(), &["dde-hh", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-hh"));
}
