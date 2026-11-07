//! Runs `pc-cli dde-zh` on the Hindmarsh-Rose neuron with a delay in the quadratic term, at
//! the first zero-Hopf point of Bosschaert, Janssens and Kuznetsov (arXiv 1903.08276 v1,
//! Sec. 6.2.1), and on a model with a closed form (SPEC section 8i, test 9). Also runs the
//! bad problem files and the model errors.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

/// The Hindmarsh-Rose neuron. `sl` is the paper's `S`.
const HR: &str = "par a=1, b=3, c=1, d=5, chi=-1.6, r=0.001, sl=-0.57452592, iapp=0, tau=1\n\
                  x'=y-a*x^3+b*delay(x,tau)^2-c*z+iapp\n\
                  y'=c-d*x^2-y\n\
                  z'=r*(sl*(x-chi)-z)\ndone\n";

/// `x' = -(a+y) x - b x(t - tau)`, `y' = a - y^2`: a zero-Hopf point at (0, pi/2, 1).
const CLOSED: &str = "par a=0, tau=1, b=1\n\
                      x'=-(a+y)*x-b*delay(x,tau)\n\
                      y'=a-y*y\ndone\n";

/// SPEC 8i, test 1 (point 1): the numpy values, each as [re, im].
const G200: [f64; 2] = [-0.002_370_511_705_141_935_4, 0.0];
const G110: [f64; 2] = [0.329_575_777_314_695_7, 0.700_582_959_211_202_2];
const G011: [f64; 2] = [-0.007_798_607_170_293_79, 0.0];
const G300: [f64; 2] = [0.010_646_551_403_411_44, 0.0];
const G111: [f64; 2] = [-0.014_622_249_811_153_01, 0.0];
const G210: [f64; 2] = [-2.776_405_821_689_472_5, -3.180_644_918_483_814_4];
const G021: [f64; 2] = [-0.374_515_574_007_687_87, -2.475_382_718_221_687];
const S_PRODUCT: f64 = 1.848_668_958_098_525_5e-5;
const THETA: f64 = -139.031_491_217_615_47;
const E: f64 = 15.694_064_382_557_59;

/// The keys of the output, in the order the spec fixes.
const KEYS: [&str; 15] = [
    "e",
    "equilibrium_residual",
    "g011",
    "g021",
    "g110",
    "g111",
    "g200",
    "g210",
    "g300",
    "null_residual_hopf",
    "null_residual_zero",
    "omega",
    "s",
    "s_product",
    "theta",
];

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_zh");
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

/// The problem of SPEC 8i test 1, with the model named by a relative path and `r`, `sl` and
/// `omega` given by the caller.
fn hr(r: f64, sl: f64, omega: f64) -> String {
    write("hr.ode", HR);
    format!(
        r#"{{"model":"hr.ode","params":{{"a":1.0,"b":3.0,"c":1.0,"d":5.0,"chi":-1.6,"r":{r:?},
            "sl":{sl:?},"iapp":-1.9579343722240001,"tau":5.768830919756731}},
            "x0":[0.13079999999999994,0.9144568000000001,-0.9943894623360001],
            "omega":{omega:?}}}"#
    )
}

fn point1() -> String {
    hr(0.001, -0.57452592, 1.0079161435515798)
}

/// The problem of SPEC 8i test 7: `a = 0`, `tau = pi/2`, `b = 1`, `omega = 1`.
fn closed() -> String {
    write("closed.ode", CLOSED);
    r#"{"model":"closed.ode","params":{"a":0.0,"tau":1.5707963267948966,"b":1.0},
        "x0":[0.0,0.0],"omega":1.0}"#
        .to_string()
}

#[test]
fn hindmarsh_rose_point_one_matches_the_numpy_values() {
    let file = write("hr_1.json", &point1());
    let o = run(&scratch(), &["dde-zh", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = stdout(&o);
    let v: Value = serde_json::from_str(&text).unwrap();
    for (name, want) in [
        ("g200", G200),
        ("g110", G110),
        ("g011", G011),
        ("g300", G300),
        ("g111", G111),
        ("g210", G210),
        ("g021", G021),
    ] {
        let g = v[name].as_array().unwrap_or_else(|| panic!("{name}: {v}"));
        assert_eq!(g.len(), 2, "{name}: {v}");
        close(num(&g[0]), want[0], 1e-8, &format!("{name} re"), &v);
        if want[1] == 0.0 {
            // The four coefficients that are real by theory.
            assert!(num(&g[1]).abs() <= 1e-12 * want[0].abs(), "{name} im: {v}");
        } else {
            close(num(&g[1]), want[1], 1e-8, &format!("{name} im"), &v);
        }
    }
    close(num(&v["s_product"]), S_PRODUCT, 1e-8, "s_product", &v);
    close(num(&v["theta"]), THETA, 1e-8, "theta", &v);
    close(num(&v["e"]), E, 1e-8, "e", &v);
    assert_eq!(num(&v["s"]), 1.0, "{v}");
    close(num(&v["omega"]), 1.0079161435515798, 1e-15, "omega", &v);
    assert!(num(&v["null_residual_zero"]) <= 1e-10, "{v}");
    assert!(num(&v["null_residual_hopf"]) <= 1e-10, "{v}");
    assert!(num(&v["equilibrium_residual"]) <= 1e-12, "{v}");
    // The fifteen keys, in alphabetical order, in the text itself and not only in the parsed
    // object.
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
fn a_closed_form_point_prints_null_for_e_and_s_zero() {
    let file = write("closed.json", &closed());
    let o = run(&scratch(), &["dde-zh", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = stdout(&o);
    assert!(text.contains("\"e\":null"), "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert!(v["e"].is_null(), "{v}");
    assert_eq!(num(&v["s"]), 0.0, "{v}");
    assert!(num(&v["s_product"]).abs() <= 1e-12, "{v}");
    close(num(&v["g200"][0]), -1.0, 1e-12, "g200", &v);
    close(
        num(&v["g110"][0]),
        -0.288_400_439_142_001,
        1e-12,
        "g110 re",
        &v,
    );
    close(
        num(&v["g110"][1]),
        0.453_018_350_450_290_2,
        1e-12,
        "g110 im",
        &v,
    );
    close(num(&v["theta"]), 0.288_400_439_142_001, 1e-12, "theta", &v);
    assert!(num(&v["g011"][0]).abs() <= 1e-12, "{v}");
}

#[test]
fn a_parameter_left_out_of_params_keeps_the_model_value() {
    // The model file has r = 0.001, the value of point 1, so leaving r out gives the same
    // numbers as giving it.
    let without = point1().replace(r#""r":0.001,"#, "");
    assert_ne!(without, point1());
    let a = run(&scratch(), &["dde-zh", &write("hr_no_r.json", &without)]);
    let b = run(&scratch(), &["dde-zh", &write("hr_with_r.json", &point1())]);
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    assert_eq!(stdout(&a), stdout(&b));
}

#[test]
fn the_model_path_is_relative_to_the_problem_file() {
    // The model sits next to the problem file and nowhere else, and pc-cli runs from the
    // parent directory, so a path read against the working directory would not find it.
    let sub = scratch().join("elsewhere");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("own_model.ode"), HR).unwrap();
    let file = sub.join("p.json");
    let text = point1().replace("\"hr.ode\"", "\"own_model.ode\"");
    std::fs::write(&file, text).unwrap();
    assert!(!scratch().join("own_model.ode").exists());
    let o = run(&scratch(), &["dde-zh", &file.to_string_lossy()]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    close(num(&v["theta"]), THETA, 1e-8, "theta", &v);
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    write("hr.ode", HR);
    let file = write(name, text);
    let o = run(&scratch(), &["dde-zh", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = point1();
    let mut no_model: Value = serde_json::from_str(&good).unwrap();
    no_model.as_object_mut().unwrap().remove("model");
    let mut no_params = no_model.clone();
    no_params.as_object_mut().unwrap().remove("params");
    for key in ["model", "params", "x0", "omega"] {
        let mut v: Value = serde_json::from_str(&good).unwrap();
        v.as_object_mut().unwrap().remove(key);
        let msg = bad(&format!("no_{key}.json"), &v.to_string());
        assert!(
            msg.contains(&format!("missing key \"{key}\"")),
            "{key}: {msg}"
        );
    }
    // The keys of dde-lyap and dde-hopf that mean nothing here are unknown.
    for key in ["l2", "free", "options"] {
        let mut v: Value = serde_json::from_str(&good).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert(key.to_string(), Value::from(0));
        let msg = bad(&format!("extra_{key}.json"), &v.to_string());
        assert!(msg.contains(&format!("unknown key \"{key}\"")), "{msg}");
    }
    assert!(bad("json.json", "{\"model\":").contains("invalid JSON"));
    assert!(bad("array.json", "[1]").contains("JSON object"));
    let unknown_par = good.replace(r#""chi":-1.6"#, r#""chi2":-1.6"#);
    assert!(bad("par.json", &unknown_par).contains("no parameter \"chi2\""));
    let wrong_len = good.replace(
        "[0.13079999999999994,0.9144568000000001,-0.9943894623360001]",
        "[0.0,1.0]",
    );
    assert_ne!(wrong_len, good);
    assert!(bad("len.json", &wrong_len).contains("3 states"));
    let bad_type = good.replace("\"x0\":[", "\"x0\":\"[").replace("]", "]\"");
    assert!(bad("type.json", &bad_type).contains("\"x0\""));
    let bad_par = good.replace(r#""a":1.0"#, r#""a":"x""#);
    assert!(bad("par_type.json", &bad_par).contains("\"a\""));
    let bad_omega = point1().replace("1.0079161435515798", "\"fast\"");
    assert!(bad("omega_type.json", &bad_omega).contains("\"omega\""));
}

#[test]
fn a_point_that_is_not_zero_hopf_exits_two_and_names_the_root() {
    // Row 3 of Delta(0) is r (sl, 0, -1), so det Delta(0) = r (A + B - sl): sl moves the
    // determinant off zero and r alone would not.
    let off_zero = hr(0.5, 0.5, 1.0079161435515798);
    assert!(bad("off_zero.json", &off_zero).contains("Delta(0)"));
    let off_hopf = hr(0.001, -0.57452592, 0.9);
    assert!(bad("off_hopf.json", &off_hopf).contains("Delta(i omega)"));
    let zero = hr(0.001, -0.57452592, 0.0);
    assert!(bad("omega_zero.json", &zero).contains("omega"));
}

#[test]
fn an_unreadable_problem_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-zh", &missing.to_string_lossy()]);
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
    let gone = good.replace("hr.ode", "no_such_model.ode");
    let o = run(&scratch(), &["dde-zh", &write("gone.json", &gone)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    // A model without a delay call.
    write("plain.ode", "par a=1.0\nu'=-a*u\ndone\n");
    let plain = r#"{"model":"plain.ode","params":{"a":1.0},"x0":[0.0],"omega":1.0}"#;
    let o = run(&scratch(), &["dde-zh", &write("plain.json", plain)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("error[ir:delay]"), "{}", stderr(&o));
}

#[test]
fn a_jordan_block_at_zero_exits_one() {
    // x' = y, y' = 0 has a double zero root, an oscillator carries the pair +-i, and the
    // decoupled w gives the model a delay (SPEC 8i, test 8). The engine refuses the root.
    write(
        "jordan.ode",
        "x'=y\ny'=0*x\nu'=-v\nv'=u\nw'=-2*w-delay(w,1)\ndone\n",
    );
    let text = r#"{"model":"jordan.ode","params":{},"x0":[0,0,0,0,0],"omega":1.0}"#;
    let o = run(&scratch(), &["dde-zh", &write("jordan.json", text)]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stdout(&o).is_empty(), "stdout {}", stdout(&o));
    assert!(stderr(&o).contains("root 0 "), "{}", stderr(&o));
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-zh"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-zh needs a problem file"));
    let o = run(&scratch(), &["dde-zh", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-zh"));
}
