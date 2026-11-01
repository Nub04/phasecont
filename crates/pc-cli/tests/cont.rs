//! Runs `pc-cli cont` on the four problem files of `pc-engine` and on bad problem files.
//!
//! The expected special points are hand anchors: the fold of `x' = a + x^2`, the Hopf
//! point of the planar normal form, the pitchfork and Hopf points of the Lorenz system
//! (r = 1 and r = 470/19 with s = 10, b = 8/3) and the Hopf point of the Brusselator.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

const TOL: f64 = 1e-9;

fn engine_tests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-engine/tests")
}

fn model(name: &str) -> String {
    engine_tests()
        .join(format!("models/{name}.ode"))
        .to_string_lossy()
        .into_owned()
}

fn problem(name: &str) -> String {
    engine_tests()
        .join(format!("problems/{name}.json"))
        .to_string_lossy()
        .into_owned()
}

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cont");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `cont` on a model and a problem file, expecting exit 0, and returns stdout as JSON
/// together with stderr.
fn cont_ok(name: &str) -> (Value, String) {
    let o = run(&scratch(), &["cont", &model(name), &problem(name)]);
    assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    (v, stderr(&o))
}

fn specials<'a>(v: &'a Value, branch: usize, label: &str) -> Vec<&'a Value> {
    v["branches"][branch]["specials"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["label"] == label)
        .collect()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

fn near(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() < TOL,
        "{what}: got {got:e}, want {want:e}"
    );
}

#[test]
fn fold_prints_one_lp_at_zero() {
    let (v, err) = cont_ok("fold");
    assert_eq!(v["branches"].as_array().unwrap().len(), 1);
    let lp = specials(&v, 0, "LP");
    assert_eq!(lp.len(), 1, "{v}");
    near(num(&lp[0]["alpha"]), 0.0, "LP alpha");
    near(num(&lp[0]["u"][0]), 0.0, "LP u");
    near(num(&lp[0]["coefficient"]).abs(), 1.0, "LP |a|");
    assert_eq!(lp[0]["refined"], true);
    assert!(lp[0]["omega"].is_null());
    assert_eq!(v["branches"][0]["stop"], "max_steps");
    assert!(err.starts_with("branch 0\n"), "{err}");
    let line = err.lines().find(|l| l.starts_with("LP ")).unwrap();
    assert!(line.contains("index "), "{line}");
    assert!(line.contains("a="), "{line}");
    let free = line
        .split("free=")
        .nth(1)
        .unwrap_or_else(|| panic!("no free= in {line:?}"));
    let free: f64 = free.trim().parse().unwrap();
    assert!(free.abs() < TOL, "{line}");
}

#[test]
fn point_records_have_the_documented_keys() {
    let (v, _) = cont_ok("fold");
    let pts = v["branches"][0]["points"].as_array().unwrap();
    assert!(pts.len() > 10);
    for p in pts {
        for key in [
            "u",
            "alpha",
            "tests",
            "codim2",
            "eigenvalues",
            "mesh",
            "h",
            "newton_iters",
        ] {
            assert!(p.get(key).is_some(), "point.{key}");
        }
        assert_eq!(p["tests"].as_array().unwrap().len(), 3);
        assert_eq!(p["u"].as_array().unwrap().len(), 1);
    }
    for s in v["branches"][0]["specials"].as_array().unwrap() {
        for key in [
            "label",
            "index",
            "u",
            "alpha",
            "eigenvalues",
            "mesh",
            "coefficient",
            "omega",
            "normalization",
            "refined",
        ] {
            assert!(s.get(key).is_some(), "special.{key}");
        }
    }
}

#[test]
fn hopf_normal_form_prints_h_at_zero() {
    let (v, err) = cont_ok("hopf_c");
    let h = specials(&v, 0, "H");
    assert_eq!(h.len(), 1, "{v}");
    near(num(&h[0]["alpha"]), 0.0, "H alpha");
    near(num(&h[0]["omega"]), 1.0, "H omega");
    near(num(&h[0]["coefficient"]), -2.0, "H l1");
    let line = err.lines().find(|l| l.starts_with("H ")).unwrap();
    assert!(line.contains("l1="), "{line}");
    assert!(line.contains("omega="), "{line}");
}

#[test]
fn lorenz_switches_at_bp1_and_finds_h_on_branch_one() {
    let (v, err) = cont_ok("lorenz");
    assert_eq!(v["branches"].as_array().unwrap().len(), 2);
    let bp = specials(&v, 0, "BP");
    assert_eq!(bp.len(), 1, "{v}");
    near(num(&bp[0]["alpha"]), 1.0, "BP r");
    assert!(bp[0]["coefficient"].is_null());
    assert!(
        err.lines()
            .any(|l| l.starts_with("BP ") && !l.contains("a=") && !l.contains("l1=")),
        "{err}"
    );
    let h = specials(&v, 1, "H");
    assert_eq!(h.len(), 1, "{v}");
    near(num(&h[0]["alpha"]), 470.0 / 19.0, "H r");
    assert!(num(&h[0]["coefficient"]) > 0.0);
    assert_eq!(v["branches"][1]["stop"], "bounds");
    assert!(
        err.contains("branch 0\n") && err.contains("branch 1\n"),
        "{err}"
    );
}

#[test]
fn brusselator_prints_h_at_two() {
    let (v, _) = cont_ok("brusselator");
    let h = specials(&v, 0, "H");
    assert_eq!(h.len(), 1, "{v}");
    near(num(&h[0]["alpha"]), 2.0, "H b");
    near(num(&h[0]["omega"]), 1.0, "H omega");
}

#[test]
fn fallback_evaluator_gives_the_same_fold() {
    let o = run(
        &scratch(),
        &["cont", &model("fold"), &problem("fold"), "--fallback"],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    near(num(&specials(&v, 0, "LP")[0]["alpha"]), 0.0, "LP alpha");
}

#[test]
fn u0_defaults_to_the_model_init() {
    // fold.ode starts at x = 1; the branch still reaches a fold on the other side.
    let path = scratch().join("no_u0.json");
    std::fs::write(
        &path,
        r#"{"type":"EP","free":"a","options":{"direction":-1,"max_steps":50}}"#,
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("fold"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(num(&v["branches"][0]["points"][0]["u"][0]) > 0.0);
}

/// Runs `cont` on fold.ode with the problem text and returns (exit code, stderr).
fn bad(text: &str) -> (i32, String) {
    let path = scratch().join("bad.json");
    std::fs::write(&path, text).unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("fold"), path.to_str().unwrap()],
    );
    assert!(stdout(&o).is_empty(), "stdout: {}", stdout(&o));
    (code(&o), stderr(&o))
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key_path() {
    for (text, path) in [
        (r#"{"type":"EP","free":"a","frob":1}"#, "frob"),
        (r#"{"free":"a"}"#, "type"),
        (r#"{"type":"BP","free":"a"}"#, "type"),
        (r#"{"type":"LP","free":"a"}"#, "free"),
        (r#"{"type":"H","free":["a","a"]}"#, "free"),
        (r#"{"type":"LP","free":["a","b"]}"#, "free[1]"),
        (r#"{"type":"EP"}"#, "free"),
        (r#"{"type":"EP","free":3}"#, "free"),
        (r#"{"type":"EP","free":"zz"}"#, "free"),
        (r#"{"type":"EP","free":"a","u0":[1,2]}"#, "u0"),
        (r#"{"type":"EP","free":"a","u0":"x"}"#, "u0"),
        (r#"{"type":"EP","free":"a","u0":[true]}"#, "u0[0]"),
        (r#"{"type":"EP","free":"a","params":{"zz":1}}"#, "params.zz"),
        (r#"{"type":"EP","free":"a","params":{"a":"x"}}"#, "params.a"),
        (r#"{"type":"EP","free":"a","params":[1]}"#, "params"),
        (
            r#"{"type":"EP","free":"a","options":{"h0":"x"}}"#,
            "options.h0",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"h0":-1}}"#,
            "options.h0",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"max_steps":1.5}}"#,
            "options.max_steps",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"max_steps":-3}}"#,
            "options.max_steps",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"bounds":[1]}}"#,
            "options.bounds",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"bounds":[1,"x"]}}"#,
            "options.bounds[1]",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"bounds":[2,1]}}"#,
            "options.bounds",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"direction":0}}"#,
            "options.direction",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"detect":{"lp":1}}}"#,
            "options.detect.lp",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"detect":{"zz":true}}}"#,
            "options.detect.zz",
        ),
        (
            r#"{"type":"EP","free":"a","options":{"zz":1}}"#,
            "options.zz",
        ),
        (r#"{"type":"EP","free":"a","options":5}"#, "options"),
        (r#"{"type":"EP","free":"a","switch":"BP"}"#, "switch"),
        (r#"{"type":"EP","free":"a","switch":"1"}"#, "switch"),
        (r#"{"type":"EP","free":"a","switch":"BP0"}"#, "switch"),
        (r#"{"type":"EP","free":"a","switch":4}"#, "switch"),
        (r#"[1]"#, "$"),
        (r#"{"type":"EP","free":"a""#, "JSON"),
    ] {
        let (c, err) = bad(text);
        assert_eq!(c, 2, "{text}: {err}");
        assert!(err.contains(path), "{text}: want `{path}` in: {err}");
    }
}

#[test]
fn a_missing_problem_file_exits_two() {
    let o = run(
        &scratch(),
        &["cont", &model("fold"), "no_such_problem.json"],
    );
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("no_such_problem.json"));
}

#[test]
fn engine_and_model_failures_exit_one() {
    // A switch name that the first branch does not have.
    let path = scratch().join("noswitch.json");
    std::fs::write(
        &path,
        r#"{"type":"EP","free":"a","u0":[-1],"switch":"BP3","options":{"max_steps":20}}"#,
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("fold"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("BP3"), "{}", stderr(&o));
    // The branches computed so far are still on stdout.
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["branches"].as_array().unwrap().len(), 1);
    // A model that depends on t.
    let forced = scratch().join("forced.ode");
    std::fs::write(&forced, "par a=1\nx'=a-x+sin(t)\ninit x=0\ndone\n").unwrap();
    let o = run(
        &scratch(),
        &["cont", forced.to_str().unwrap(), &problem("fold")],
    );
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    // A model file that does not exist.
    let o = run(&scratch(), &["cont", "no_such_model.ode", &problem("fold")]);
    assert_eq!(code(&o), 1, "{}", stderr(&o));
}

#[test]
fn a_branch_that_fails_at_the_start_has_stop_error() {
    // log of a negative number gives NaN at the start point.
    let model = scratch().join("nan_start.ode");
    std::fs::write(&model, "par a=1\nx'=a*log(x)\ninit x=-1\ndone\n").unwrap();
    let path = scratch().join("nan_start.json");
    std::fs::write(&path, r#"{"type":"EP","free":"a","u0":[-1]}"#).unwrap();
    let o = run(
        &scratch(),
        &["cont", model.to_str().unwrap(), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["branches"][0]["stop"], "error", "{v}");
    assert!(stderr(&o).contains("run failed"), "{}", stderr(&o));
}

#[test]
fn help_and_usage() {
    let o = run(&scratch(), &["cont", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(
        stdout(&o).starts_with("usage: pc-cli cont"),
        "{}",
        stdout(&o)
    );
    let o = run(&scratch(), &["cont", "only_one.ode"]);
    assert_eq!(code(&o), 2);
}

#[test]
fn a_cycle_run_prints_the_orbit_the_mesh_and_the_multipliers() {
    let path = scratch().join("lc.json");
    std::fs::write(
        &path,
        r#"{"type":"LC","free":"al","params":{"al":0,"c":-1},"u0":[0,0],"ntst":8,"ncol":3,
            "start":"hopf","options":{"hmax":0.05,"max_steps":10,"bounds":[-1,0.3]}}"#,
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("hopf_c"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let pts = v["branches"][0]["points"].as_array().unwrap();
    assert!(pts.len() > 3, "{v}");
    for p in pts {
        // The orbit at 8 * 3 + 1 points of 2 states, then the period.
        assert_eq!(p["u"].as_array().unwrap().len(), 2 * 25 + 1);
        assert_eq!(p["mesh"].as_array().unwrap().len(), 9);
        assert_eq!(p["eigenvalues"].as_array().unwrap().len(), 2);
        assert_eq!(p["codim2"].as_array().unwrap().len(), 3);
    }
}

#[test]
fn an_lpc_curve_run_follows_the_bautin_fold() {
    let (ntst, ncol) = (6, 3);
    let pts = ntst * ncol + 1;
    let r = 0.5_f64.sqrt();
    let mut u0: Vec<f64> = (0..pts)
        .flat_map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / (pts - 1) as f64;
            [r * t.cos(), r * t.sin()]
        })
        .collect();
    u0.push(2.0 * std::f64::consts::PI / 1.25);
    let mesh: Vec<f64> = (0..=ntst).map(|j| j as f64 / ntst as f64).collect();
    let problem = serde_json::json!({
        "type": "LPC", "free": ["b1", "b2"], "params": {"b1": -0.25, "b2": 1, "g": 0.5},
        "u0": u0, "ntst": ntst, "ncol": ncol, "mesh": mesh,
        "options": {"hmax": 0.05, "max_steps": 6, "bounds": [0.5, 1.5]}
    });
    let path = scratch().join("lpc.json");
    std::fs::write(&path, problem.to_string()).unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("bautin_lpc"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let pts = v["branches"][0]["points"].as_array().unwrap();
    assert!(pts.len() > 3, "{v}");
    for p in pts {
        // The orbit at 6 * 3 + 1 points of 2 states, then T and b1; alpha is b2.
        let u = p["u"].as_array().unwrap();
        assert_eq!(u.len(), 2 * 19 + 2);
        assert_eq!(p["mesh"].as_array().unwrap().len(), ntst + 1);
        assert_eq!(p["codim2"].as_array().unwrap().len(), 2);
        let (b1, b2) = (u[39].as_f64().unwrap(), p["alpha"].as_f64().unwrap());
        assert!((b1 + b2 * b2 / 4.0).abs() < 1e-3, "b1 = {b1}, b2 = {b2}");
    }
}
