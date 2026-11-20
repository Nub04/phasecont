//! Runs `pc-cli cont` on the four problem files of `pc-engine` and on bad problem files.
//!
//! The expected special points are hand anchors: the fold of `x' = a + x^2`, the Hopf
//! point of the planar normal form, the pitchfork and Hopf points of the Lorenz system
//! (r = 1 and r = 470/19 with s = 10, b = 8/3) and the Hopf point of the Brusselator.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::{Value, json};

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
    // Only a map run gives the iterate of a branch (SPEC 8y).
    assert!(v["branches"][0].get("iterate").is_none(), "{v}");
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

/// `x' = -x + sqrt(max(a - x - 10, 0))`: the clamp is active, so the root sits at 0 where its
/// slope is infinite, and the tangent of the clamp is zero. The Jacobian is -1 and the branch
/// runs, with the compiled evaluators and with the fallback alike.
#[test]
fn a_clamped_root_does_not_make_the_jacobian_nan() {
    let ode = scratch().join("sqclamp.ode");
    std::fs::write(&ode, "x'=-x+sqrt(max(a-x-10,0))\npar a=1\ndone\n").unwrap();
    let json = scratch().join("sqclamp.json");
    std::fs::write(
        &json,
        r#"{"type":"EP","free":"a","params":{"a":1},"u0":[0],"options":{"max_steps":8}}"#,
    )
    .unwrap();
    for extra in [None, Some("--fallback")] {
        let mut args = vec!["cont", ode.to_str().unwrap(), json.to_str().unwrap()];
        args.extend(extra);
        let o = run(&scratch(), &args);
        assert_eq!(code(&o), 0, "{extra:?}: {}", stderr(&o));
        let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
        let points = v["branches"][0]["points"].as_array().unwrap();
        assert!(points.len() > 2, "{extra:?}: {v}");
        for pt in points {
            near(num(&pt["eigenvalues"][0][0]), -1.0, "eigenvalue");
            near(num(&pt["u"][0]), 0.0, "u");
        }
    }
}

/// A clamped quotient inside `exp`: `x' = -x + exp(-(1 + m) / m)` with `m = max(a - x - 10, 0)`
/// is 0 at the clamp, and the tangent of `1 + m` is zero there over a zero denominator. The
/// zero tangent over zero is 0, not NaN, so the Jacobian is -1 as in the fallback.
#[test]
fn a_clamped_quotient_does_not_make_the_jacobian_nan() {
    let json = scratch().join("qclamp.json");
    std::fs::write(
        &json,
        r#"{"type":"EP","free":"a","params":{"a":1},"u0":[0],"options":{"max_steps":8}}"#,
    )
    .unwrap();
    for (name, rhs) in [
        ("qclamp", "exp(-(1+max(a-x-10,0))/max(a-x-10,0))"),
        ("rclamp", "exp(-1/max(a-x-10,0))"),
    ] {
        let ode = scratch().join(format!("{name}.ode"));
        std::fs::write(&ode, format!("x'=-x+{rhs}\npar a=1\ndone\n")).unwrap();
        for extra in [None, Some("--fallback")] {
            let mut args = vec!["cont", ode.to_str().unwrap(), json.to_str().unwrap()];
            args.extend(extra);
            let o = run(&scratch(), &args);
            assert_eq!(code(&o), 0, "{name} {extra:?}: {}", stderr(&o));
            let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
            let points = v["branches"][0]["points"].as_array().unwrap();
            assert!(points.len() > 2, "{name} {extra:?}: {v}");
            for pt in points {
                near(num(&pt["eigenvalues"][0][0]), -1.0, "eigenvalue");
                near(num(&pt["u"][0]), 0.0, "u");
            }
        }
    }
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
        // LPPD, LPNS, CPC and R1: the planar model has b but no other multipliers, so JSON
        // null (NaN) stands for the three that are not defined.
        let c2 = p["codim2"].as_array().unwrap();
        assert_eq!(c2.len(), 4);
        assert!(
            c2[0].is_null() && c2[1].is_null() && c2[3].is_null(),
            "{c2:?}"
        );
        let (b1, b2) = (u[39].as_f64().unwrap(), p["alpha"].as_f64().unwrap());
        assert!((b1 + b2 * b2 / 4.0).abs() < 1e-3, "b1 = {b1}, b2 = {b2}");
        // b = -2 b2 (1 + b2 / 4), SPEC 8a.
        let b = c2[2].as_f64().unwrap();
        let want = -2.0 * b2 * (1.0 + b2 / 4.0);
        assert!((b - want).abs() < 1e-3 * want.abs(), "b = {b}, want {want}");
    }
}

/// The orbit of Sandstede's model at mu = mut = alpha = 0 on `pts` uniform points over
/// [-period/2, period/2]: w solves w' = (1 - w^2)(1 - a w) / 2 from w(0) = 0 by RK4, and
/// (x, y, z) = (1 - w^2, -w (1 - w^2), 0).
fn sandstede_orbit(a: f64, period: f64, pts: usize) -> Vec<f64> {
    let f = |w: f64| (1.0 - w * w) * (1.0 - a * w) / 2.0;
    (0..pts)
        .flat_map(|k| {
            let t = period * (k as f64 / (pts - 1) as f64 - 0.5);
            let steps = ((t.abs() * 1000.0).ceil() as usize).max(1);
            let h = t / steps as f64;
            let w = (0..steps).fold(0.0, |w: f64, _| {
                let k1 = f(w);
                let k2 = f(w + h / 2.0 * k1);
                let k3 = f(w + h / 2.0 * k2);
                let k4 = f(w + h * k3);
                w + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
            [1.0 - w * w, -w * (1.0 - w * w), 0.0]
        })
        .collect()
}

#[test]
fn a_hom_curve_run_prints_the_orbit_the_saddle_and_fourteen_tests() {
    let (n, ntst, ncol, period) = (3, 60, 4, 20.0);
    let pts = ntst * ncol + 1;
    let mut u0 = sandstede_orbit(-0.25, period, pts);
    u0.push(period);
    let problem = serde_json::json!({
        "type": "HOM", "free": ["mut", "a"], "start": "orbit",
        "params": {"a": -0.25, "b": 1, "c": -2},
        "u0": u0, "ntst": ntst, "ncol": ncol,
        "options": {"hmax": 0.05, "max_newton": 15, "max_steps": 6, "bounds": [-0.26, 0.26]}
    });
    let path = scratch().join("hom.json");
    std::fs::write(&path, problem.to_string()).unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("sandstede"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let pts_json = v["branches"][0]["points"].as_array().unwrap();
    assert!(pts_json.len() > 3, "{v}");
    for p in pts_json {
        // The orbit at 60 * 4 + 1 points of 3 states, then T, the saddle and mut; alpha is a.
        let u = p["u"].as_array().unwrap();
        assert_eq!(u.len(), n * pts + 1 + n + 1);
        assert_eq!(p["mesh"].as_array().unwrap().len(), ntst + 1);
        assert_eq!(p["eigenvalues"].as_array().unwrap().len(), n);
        let c2 = p["codim2"].as_array().unwrap();
        assert_eq!(c2.len(), 14);
        // The key is off, so the two inclination flip tests are null.
        assert!(c2[12].is_null() && c2[13].is_null(), "{c2:?}");
        near(num(&u[n * pts]), period, "T");
        for k in 0..n {
            let x0 = num(&u[n * pts + 1 + k]);
            assert!(x0.abs() < 1e-10, "x0[{k}] = {x0:e}");
        }
        let mut_ = num(&u[n * pts + 1 + n]);
        assert!(mut_.abs() < 1e-6, "mut = {mut_:e}");
    }
}

/// SPEC 8b-3: with `inclination_flips` true the run adds the IFS and IFU columns and finds the
/// IFS zero of the oracle curve (a, b, c, beta) = (-0.25, 1, -3, 1) at alpha = 1.8961.
#[test]
fn a_hom_curve_run_with_inclination_flips_labels_the_ifs_zero() {
    let (ntst, ncol, period) = (160, 4, 20.0);
    let pts = ntst * ncol + 1;
    let mut u0 = sandstede_orbit(-0.25, period, pts);
    u0.push(period);
    let problem = serde_json::json!({
        "type": "HOM", "free": ["mut", "alpha"], "start": "orbit",
        "params": {"a": -0.25, "b": 1, "c": -3, "beta": 1},
        "u0": u0, "ntst": ntst, "ncol": ncol, "inclination_flips": true,
        "options": {"hmax": 0.05, "max_newton": 15, "bounds": [-0.1, 2.1]}
    });
    let path = scratch().join("hom_if.json");
    std::fs::write(&path, problem.to_string()).unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("sandstede"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    for p in v["branches"][0]["points"].as_array().unwrap() {
        let c2 = p["codim2"].as_array().unwrap();
        assert!(c2[12].is_number() && c2[13].is_number(), "{c2:?}");
    }
    let ifs = specials(&v, 0, "IFS");
    assert_eq!(ifs.len(), 1, "{}", v["branches"][0]["specials"]);
    let at = num(&ifs[0]["alpha"]);
    assert!((at - 1.8961).abs() < 5e-3, "IFS at alpha = {at}");
    assert_eq!(v["branches"][0]["specials"].as_array().unwrap().len(), 1);
}

/// A non-HOM run with the key is refused with the parser message.
#[test]
fn inclination_flips_on_a_non_hom_run_is_an_error() {
    let problem = serde_json::json!({
        "type": "EP", "free": "a", "params": {"a": 0.0}, "u0": [0.0, 0.0, 0.0],
        "inclination_flips": true,
        "options": {"max_steps": 2}
    });
    let path = scratch().join("ep_if.json");
    std::fs::write(&path, problem.to_string()).unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("sandstede"), path.to_str().unwrap()],
    );
    assert_ne!(code(&o), 0);
    assert!(
        stderr(&o).contains("inclination_flips: only a HOM run takes this key"),
        "{}",
        stderr(&o)
    );
}

/// A map is refused after the model loads for every type but ITER, EP and the LP, PD and NS
/// curves (SPEC 8t, test 7, and SPEC 8w).
#[test]
fn a_map_model_is_refused_with_exit_one() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus");
    let discrete = scratch().join("logistic_discrete.ode");
    std::fs::write(
        &discrete,
        "par r=2\ninit x=0.2\nx'=r*x*(1-x)\n@ meth=discrete\ndone\n",
    )
    .unwrap();
    let path = scratch().join("map_lp.json");
    std::fs::write(&path, r#"{"type":"H","free":"r","u0":[0.5]}"#).unwrap();
    for file in [corpus.join("logistic_map.ode"), discrete] {
        let o = run(
            &scratch(),
            &["cont", file.to_str().unwrap(), path.to_str().unwrap()],
        );
        assert_eq!(code(&o), 1, "{}", stderr(&o));
        assert!(stdout(&o).is_empty(), "stdout: {}", stdout(&o));
        assert!(
            stderr(&o).contains(pckern::error::MAP_MODEL),
            "{}",
            stderr(&o)
        );
    }
}

/// A map runs an ITER problem and prints its rows (SPEC 8u, test 4).
#[test]
fn a_map_runs_an_iter_problem() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus");
    let path = scratch().join("map_iter.json");
    std::fs::write(
        &path,
        r#"{"type":"ITER","tspan":[0,5],"params":{"r":2.5},"u0":[0.1]}"#,
    )
    .unwrap();
    let o = run(
        &scratch(),
        &[
            "cont",
            corpus.join("logistic_map.ode").to_str().unwrap(),
            path.to_str().unwrap(),
        ],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let points = v["branches"][0]["points"].as_array().unwrap();
    let mut x = 0.1_f64;
    let mut want = vec![(0.0, x)];
    for k in 1..=5 {
        x = (2.5 * x) * (1.0 - x);
        want.push((f64::from(k), x));
    }
    let got: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (p["alpha"].as_f64().unwrap(), p["u"][0].as_f64().unwrap()))
        .collect();
    assert_eq!(got, want);
}

/// `"aux": true` puts the aux values after the state in the `u` of every IVP and ITER point,
/// at the point's own time or iterate number, and leaves the state as it is without the key
/// (SPEC section 10, round E).
#[test]
fn aux_true_adds_the_aux_values_to_ivp_and_iter_points() {
    let ode = scratch().join("aux_osc.ode");
    std::fs::write(
        &ode,
        "par a=0.1\nx'=y\ny'=-x-a*y\naux energy=(x^2+y^2)/2\naux tx=t*x\ninit x=1,y=0\ndone\n",
    )
    .unwrap();
    let map = scratch().join("aux_map.ode");
    std::fs::write(
        &map,
        "par r=2.5\ninit x=0.1\nx(t+1)=r*x*(1-x)\naux dist=x-0.5\naux kx=t*x\ndone\n",
    )
    .unwrap();
    let points = |file: &Path, problem: &str, tag: &str| -> Vec<(f64, Vec<f64>)> {
        let path = scratch().join(format!("aux_{tag}.json"));
        std::fs::write(&path, problem).unwrap();
        let o = run(
            &scratch(),
            &["cont", file.to_str().unwrap(), path.to_str().unwrap()],
        );
        assert_eq!(code(&o), 0, "{tag}: {}", stderr(&o));
        let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
        v["branches"][0]["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let u = p["u"].as_array().unwrap();
                (
                    p["alpha"].as_f64().unwrap(),
                    u.iter().map(|x| x.as_f64().unwrap()).collect(),
                )
            })
            .collect()
    };
    let cases = [
        (&ode, r#""type":"IVP","tspan":[0,4]"#, 2, "ivp"),
        (&map, r#""type":"ITER","tspan":[0,6]"#, 1, "iter"),
    ];
    for (file, head, n, tag) in cases {
        let plain = points(file, &format!("{{{head}}}"), &format!("{tag}_plain"));
        let off = points(
            file,
            &format!(r#"{{{head},"aux":false}}"#),
            &format!("{tag}_off"),
        );
        let with = points(
            file,
            &format!(r#"{{{head},"aux":true}}"#),
            &format!("{tag}_on"),
        );
        assert_eq!(off, plain, "{tag}: aux false is the plain run");
        assert_eq!(with.len(), plain.len(), "{tag}");
        for (i, ((t, u), (t0, u0))) in with.iter().zip(&plain).enumerate() {
            assert_eq!(t.to_bits(), t0.to_bits(), "{tag} row {i}");
            assert_eq!(u0.len(), n, "{tag} row {i}");
            assert_eq!(u.len(), n + 2, "{tag} row {i}");
            assert_eq!(u[..n], u0[..], "{tag} row {i}");
            // The second aux quantity of both models is t * x.
            assert_eq!(u[n + 1].to_bits(), (t * u[0]).to_bits(), "{tag} row {i}");
        }
        // The first aux value: the energy of the oscillator, the distance of the map to 1/2.
        let (_, first) = &with[0];
        let want = if n == 2 { 0.5 } else { 0.1 - 0.5 };
        assert_eq!(first[n].to_bits(), f64::to_bits(want), "{tag}");
    }
}

/// An IVP or ITER point record has the keys of a continuation point less `h`, with constant
/// values in all but `alpha` and `u`, and a run of more rows than the CLI reads at a time
/// prints every row once and in order (SPEC section 8af-1).
#[test]
fn ivp_and_iter_points_have_no_h_and_every_row_is_printed_once() {
    let map = scratch().join("rows_map.ode");
    std::fs::write(&map, "par r=2.5\ninit x=0.1\nx(t+1)=r*x*(1-x)\ndone\n").unwrap();
    let path = scratch().join("rows_iter.json");
    std::fs::write(&path, r#"{"type":"ITER","tspan":[0,9000]}"#).unwrap();
    let o = run(
        &scratch(),
        &["cont", map.to_str().unwrap(), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    // The keys are written in alphabetical order, as for every other point.
    let text = stdout(&o);
    assert!(
        text.contains(r#"{"alpha":0.0,"codim2":[],"eigenvalues":[],"mesh":[],"newton_iters":0,"tests":[null,null,null],"u":[0.1]}"#),
        "{}",
        &text[..text.len().min(300)]
    );
    let v: Value = serde_json::from_str(&text).unwrap();
    let points = v["branches"][0]["points"].as_array().unwrap();
    assert_eq!(points.len(), 9001);
    let keys = [
        "alpha",
        "codim2",
        "eigenvalues",
        "mesh",
        "newton_iters",
        "tests",
        "u",
    ];
    let mut x = 0.1_f64;
    for (k, p) in points.iter().enumerate() {
        let got: Vec<&str> = p.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(got, keys, "row {k}");
        assert_eq!(p["alpha"].as_f64().unwrap(), k as f64, "row {k}");
        assert_eq!(p["u"].as_array().unwrap().len(), 1, "row {k}");
        assert_eq!(
            p["u"][0].as_f64().unwrap().to_bits(),
            x.to_bits(),
            "row {k}"
        );
        assert_eq!(p["codim2"], json!([]));
        assert_eq!(p["eigenvalues"], json!([]));
        assert_eq!(p["mesh"], json!([]));
        assert_eq!(p["newton_iters"], json!(0));
        assert_eq!(p["tests"], json!([null, null, null]));
        x = (2.5 * x) * (1.0 - x);
    }
}

/// A map runs an EP problem: the BP of the logistic map at r = 1, the switch to
/// x = 1 - 1/r and the PD at r = 3 with b = 9. The points equal the rows that the API
/// table gives for the same problem, bit for bit (SPEC 8v, test 9).
#[test]
fn a_map_runs_an_ep_problem_and_matches_the_api() {
    use pc_cli::session::{Engine, Evaluator};
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus");
    let source = corpus.join("logistic_map.ode");
    let body = r#""type":"EP","free":"r","options":{"bounds":[0,4]},"switch":"BP1""#;
    let path = scratch().join("map_ep.json");
    std::fs::write(
        &path,
        format!(r#"{{{body},"params":{{"r":0.5}},"u0":[0.0]}}"#),
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", source.to_str().unwrap(), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let branches = v["branches"].as_array().unwrap();
    assert_eq!(branches.len(), 2, "{v}");
    assert_eq!(
        (&branches[0]["iterate"], &branches[1]["iterate"]),
        (&1.into(), &1.into())
    );
    let bp = specials(&v, 0, "BP");
    assert_eq!(bp.len(), 1, "{v}");
    near(num(&bp[0]["alpha"]), 1.0, "BP alpha");
    let pd = specials(&v, 1, "PD");
    assert_eq!(pd.len(), 1, "{v}");
    near(num(&pd[0]["alpha"]), 3.0, "PD alpha");
    near(num(&pd[0]["u"][0]), 2.0 / 3.0, "PD x");
    let b = num(&pd[0]["coefficient"]);
    assert!((b - 9.0).abs() <= 9e-7, "PD b: got {b:e}");
    assert!(stderr(&o).contains("PD  index"), "{}", stderr(&o));

    let engine = Engine::load().unwrap();
    let text = std::fs::read_to_string(&source).unwrap();
    let m = engine.model(&text, Evaluator::Jit).unwrap();
    let got = m.run(&format!("{{{body}}}"), &[0.0], &[0.5]).unwrap();
    assert_eq!(got.rows.len(), branches.len());
    // A NaN prints as null, and serde_json prints the shortest text that reads back to
    // the same bits.
    let f = |x: &Value| x.as_f64().unwrap_or(f64::NAN);
    for (b, (block, jb)) in got.rows.iter().zip(branches).enumerate() {
        let points = jb["points"].as_array().unwrap();
        assert_eq!(block.rows(), points.len(), "branch {b}");
        assert_eq!(block.cols(), 10, "branch {b}");
        for (r, (row, p)) in block.data().chunks(10).zip(points).enumerate() {
            assert_eq!(p["tests"].as_array().unwrap().len(), 3, "{p}");
            assert!(p["tests"][2].is_null(), "{p}");
            assert_eq!(p["codim2"].as_array().unwrap().len(), 2, "{p}");
            assert!(p["codim2"][1].is_null(), "psi_NS of a 1-D map: {p}");
            let want = [
                f(&p["alpha"]),
                f(&p["u"][0]),
                f(&p["eigenvalues"][0][0]),
                f(&p["eigenvalues"][0][1]),
                f(&p["tests"][0]),
                f(&p["tests"][1]),
                f(&p["codim2"][0]),
                f(&p["codim2"][1]),
                f(&p["h"]),
                f(&p["newton_iters"]),
            ];
            let same = row
                .iter()
                .zip(&want)
                .all(|(a, w)| a.to_bits() == w.to_bits() || (a.is_nan() && w.is_nan()));
            assert!(same, "branch {b} row {r}: api {row:?}, cont {want:?}");
        }
    }
}

/// A PD switch on a map EP run gives branch 0 at iterate 1 and branch 1 at iterate 2, the
/// 2-cycle of the logistic map with its PD at r = 1 + sqrt(6) (SPEC 8y).
#[test]
fn a_map_pd_switch_prints_the_iterate_of_each_branch() {
    let path = scratch().join("map_pd_switch.json");
    std::fs::write(
        &path,
        r#"{"type":"EP","free":"r","options":{"bounds":[2.5,3.6]},"switch":"PD1","params":{"r":2.5},"u0":[0.6]}"#,
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", &model("map_logistic"), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let branches = v["branches"].as_array().unwrap();
    assert_eq!(branches.len(), 2, "{v}");
    assert_eq!(
        (&branches[0]["iterate"], &branches[1]["iterate"]),
        (&1.into(), &2.into())
    );
    let pd = specials(&v, 1, "PD");
    assert_eq!(pd.len(), 1, "{v}");
    let r = num(&pd[0]["alpha"]);
    assert!(
        (r - (1.0 + 6f64.sqrt())).abs() < 1e-8,
        "PD of the 2-cycle at r = {r}"
    );
}

/// Runs a map curve problem through `pc-cli cont` and through the API table and checks
/// that each point equals its ABI row bit for bit (SPEC 8w, test 7). `names` lists the
/// parameters in model order, `tag` names the problem file and `check` sees each point of the
/// CLI output. The row is [a2, u, mu_re, mu_im, codim2, h, newton_iters], where u holds x, a1
/// and, on the NS curve, kappa, and codim2 holds the 4 values of the LP and PD curves or the
/// 8 values of the NS curve (SPEC 8x).
fn map_curve_matches_api(
    tag: &str,
    source: &Path,
    body: &str,
    names: [&str; 2],
    p: [f64; 2],
    u0: &[f64],
    check: impl Fn(&Value),
) {
    use pc_cli::session::{Engine, Evaluator};
    // Each caller gets its own file, since the tests run in parallel.
    let path = scratch().join(format!("map_curve_{tag}.json"));
    let u0_text: Vec<String> = u0.iter().map(|v| format!("{v:?}")).collect();
    std::fs::write(
        &path,
        format!(
            r#"{{{body},"params":{{"{}":{:?},"{}":{:?}}},"u0":[{}]}}"#,
            names[0],
            p[0],
            names[1],
            p[1],
            u0_text.join(",")
        ),
    )
    .unwrap();
    let o = run(
        &scratch(),
        &["cont", source.to_str().unwrap(), path.to_str().unwrap()],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let branches = v["branches"].as_array().unwrap();
    assert_eq!(branches.len(), 1, "{v}");
    let points = branches[0]["points"].as_array().unwrap();
    assert!(points.len() > 5, "{v}");
    points.iter().for_each(&check);

    let engine = Engine::load().unwrap();
    let text = std::fs::read_to_string(source).unwrap();
    let m = engine.model(&text, Evaluator::Jit).unwrap();
    let got = m.run(&format!("{{{body}}}"), u0, &p).unwrap();
    assert_eq!(got.rows.len(), 1);
    let block = &got.rows[0];
    let n = u0.len();
    let nu = points[0]["u"].as_array().unwrap().len();
    let k = points[0]["codim2"].as_array().unwrap().len();
    assert_eq!(k, if nu == n + 2 { 8 } else { 4 }, "{}", points[0]);
    let stride = 1 + nu + 2 * n + k + 2;
    assert_eq!(block.rows(), points.len());
    assert_eq!(block.cols(), stride);
    let f = |x: &Value| x.as_f64().unwrap_or(f64::NAN);
    for (r, (row, p)) in block.data().chunks(stride).zip(points).enumerate() {
        let mut want = vec![f(&p["alpha"])];
        want.extend((0..nu).map(|i| f(&p["u"][i])));
        want.extend((0..n).map(|i| f(&p["eigenvalues"][i][0])));
        want.extend((0..n).map(|i| f(&p["eigenvalues"][i][1])));
        want.extend((0..k).map(|i| f(&p["codim2"][i])));
        want.push(f(&p["h"]));
        want.push(f(&p["newton_iters"]));
        let same = row
            .iter()
            .zip(&want)
            .all(|(a, w)| a.to_bits() == w.to_bits() || (a.is_nan() && w.is_nan()));
        assert!(same, "row {r}: api {row:?}, cont {want:?}");
    }
}

/// A map runs a PD curve of fixed points: the Henon map's PD curve is a = 3 (1 - b)^2 / 4 with
/// x = 2 / (3 (1 - b)).
#[test]
fn a_map_runs_a_pd_curve_and_matches_the_api() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus");
    let (b0, a0) = (0.3_f64, 3.0 * 0.49 / 4.0);
    let x0 = 2.0 / (3.0 * (1.0 - b0));
    map_curve_matches_api(
        "pd",
        &corpus.join("henon_map.ode"),
        r#""type":"PD","free":["a","b"],"options":{"bounds":[0.1,0.6]}"#,
        ["a", "b"],
        [a0, b0],
        &[x0, b0 * x0],
        |p| {
            let b = num(&p["alpha"]);
            let x = 2.0 / (3.0 * (1.0 - b));
            assert_eq!(p["u"].as_array().unwrap().len(), 3, "{p}");
            assert!((num(&p["u"][0]) - x).abs() <= 1e-8, "x at b = {b}: {p}");
            assert!((num(&p["u"][1]) - b * x).abs() <= 1e-8, "y at b = {b}: {p}");
            let a = 3.0 * (1.0 - b).powi(2) / 4.0;
            assert!((num(&p["u"][2]) - a).abs() <= 1e-8, "a at b = {b}: {p}");
        },
    );
}

/// A map runs an LP curve of fixed points: the Henon map's LP curve is a = -(1 - b)^2 / 4
/// with x = 2 / (1 - b).
#[test]
fn a_map_runs_an_lp_curve_and_matches_the_api() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus");
    let (b0, a0) = (0.3_f64, -0.49 / 4.0);
    let x0 = 2.0 / (1.0 - b0);
    map_curve_matches_api(
        "lp",
        &corpus.join("henon_map.ode"),
        r#""type":"LP","free":["a","b"],"options":{"bounds":[0.1,0.6]}"#,
        ["a", "b"],
        [a0, b0],
        &[x0, b0 * x0],
        |p| {
            let b = num(&p["alpha"]);
            let x = 2.0 / (1.0 - b);
            assert_eq!(p["u"].as_array().unwrap().len(), 3, "{p}");
            assert!((num(&p["u"][0]) - x).abs() <= 1e-8, "x at b = {b}: {p}");
            assert!((num(&p["u"][1]) - b * x).abs() <= 1e-8, "y at b = {b}: {p}");
            let a = -(1.0 - b).powi(2) / 4.0;
            assert!((num(&p["u"][2]) - a).abs() <= 1e-8, "a at b = {b}: {p}");
        },
    );
}

/// A map runs an NS curve of fixed points: AUTO's Henon form (x, y) -> (y, a - b x - y^2)
/// has the NS curve b = 1 with x = y = -1 + sqrt(1 + a) and kappa = -y. Free [b, a], so
/// u = (x, y, b, kappa) and alpha = a.
#[test]
fn a_map_runs_an_ns_curve_and_matches_the_api() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-engine/tests/models");
    map_curve_matches_api(
        "ns",
        &models.join("map_henon_auto.ode"),
        r#""type":"NS","free":["b","a"],"options":{"bounds":[0.2,1.5]}"#,
        ["a", "b"],
        [0.69, 1.0],
        &[0.3, 0.3],
        |p| {
            let a = num(&p["alpha"]);
            let y = -1.0 + (1.0 + a).sqrt();
            assert_eq!(p["u"].as_array().unwrap().len(), 4, "{p}");
            assert!((num(&p["u"][0]) - y).abs() <= 1e-8, "x at a = {a}: {p}");
            assert!((num(&p["u"][1]) - y).abs() <= 1e-8, "y at a = {a}: {p}");
            assert!((num(&p["u"][2]) - 1.0).abs() <= 1e-8, "b at a = {a}: {p}");
            assert!((num(&p["u"][3]) + y).abs() <= 1e-8, "kappa at a = {a}: {p}");
        },
    );
}
