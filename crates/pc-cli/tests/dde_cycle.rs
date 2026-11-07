//! Runs `pc-cli dde-cycle` on the model `x' = -x(t - tau) + mu x (1 - x^2 - x(t - tau)^2)`,
//! which has the exact orbit x = sin t, T = 2 pi for tau = pi/2 (SPEC section 8l, test 7).
//! The orbit and Hopf starts must give the numbers of the engine calls. Also runs the bad
//! problem files and the run that does not converge.

mod common;

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use common::{code, run, stderr, stdout};
use pc_engine::{
    DdeCycleSetup, DdeCycleStart, DdeHopfCycle, Evaluator, Options, build_dde, dde_correct_cycle,
    dde_hopf_cycle,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use serde_json::{Value, json};

const TAU: f64 = PI / 2.0;
const TAU_H: f64 = 1.327_237_281_892_678_3;
const OMEGA_H: f64 = 0.953_939_201_416_945_7;
const NTST: usize = 20;
const NCOL: usize = 4;

fn osc_source() -> String {
    format!(
        "par mu=0.3, tau={TAU:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         done\n"
    )
}

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_cycle");
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

/// The start orbit of test 3: a sine with a second harmonic, at the base points.
fn perturbed() -> Vec<f64> {
    let nq = NTST * NCOL;
    (0..=nq)
        .map(|q| {
            let s = q as f64 / nq as f64;
            1.2 * (2.0 * PI * s).sin() + 0.05 * (4.0 * PI * s).sin()
        })
        .collect()
}

/// The engine model of the OSC source, with the JIT evaluator.
fn engine_model() -> pc_engine::DdeModel {
    let mut map = SourceMap::new();
    let top = map.add("osc.ode", osc_source());
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("build_dde")
}

fn newton_options() -> Options {
    Options {
        max_newton: 30,
        ..Options::default()
    }
}

/// A problem of the orbit kind with `profile` and the options of test 3.
fn orbit_problem(profile: &[f64], period: f64) -> Value {
    json!({
        "model": "osc.ode",
        "params": {"mu": 0.3, "tau": TAU},
        "free": "tau",
        "ntst": NTST,
        "ncol": NCOL,
        "orbit": {"profile": profile, "period": period},
        "options": {"max_newton": 30},
    })
}

fn hopf_problem(eps: Value) -> Value {
    json!({
        "model": "osc.ode",
        "params": {"mu": 0.3, "tau": TAU_H},
        "free": "tau",
        "ntst": NTST,
        "ncol": NCOL,
        "hopf": {"x0": [0.0], "omega": OMEGA_H, "eps": eps},
        "options": {"max_newton": 30},
    })
}

/// Runs the command on `problem` and returns the exit code, stdout and stderr.
fn dde_cycle(name: &str, problem: &Value) -> (i32, String, String) {
    write("osc.ode", &osc_source());
    let file = write(name, &problem.to_string());
    let o = run(&scratch(), &["dde-cycle", &file]);
    (code(&o), stdout(&o), stderr(&o))
}

fn doubles(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

#[test]
fn the_orbit_start_matches_correct_cycle_bit_for_bit() {
    let start = perturbed();
    let (status, out, err) = dde_cycle("orbit.json", &orbit_problem(&start, 6.0));
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let mut model = engine_model();
    let setup = DdeCycleSetup {
        params: vec![0.3, TAU],
        free: 1,
        ntst: NTST,
        ncol: NCOL,
    };
    let want = dde_correct_cycle(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: start,
            period: 6.0,
        },
        &newton_options(),
        &AtomicBool::new(false),
    )
    .expect("correct_cycle");
    assert_eq!(num(&v["period"]).to_bits(), want.period.to_bits(), "{out}");
    assert_eq!(num(&v["alpha"]).to_bits(), want.alpha.to_bits(), "{out}");
    assert_eq!(v["iters"].as_u64(), Some(want.iters as u64), "{out}");
    assert_eq!(num(&v["residual"]).to_bits(), want.residual.to_bits());
    assert_eq!(v["ntst"], NTST);
    assert_eq!(v["ncol"], NCOL);
    let got = doubles(&v["profile"]);
    assert_eq!(got.len(), want.profile.len());
    for (i, (a, b)) in got.iter().zip(&want.profile).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "profile[{i}]");
    }
    // The orbit is the exact one: x = sin t, T = 2 pi, and tau stays at pi/2.
    assert!(
        (want.period - 2.0 * PI).abs() <= 1e-8,
        "T = {}",
        want.period
    );
    assert_eq!(want.alpha, TAU);
}

#[test]
fn the_hopf_start_matches_hopf_cycle() {
    let (status, out, err) = dde_cycle("hopf.json", &hopf_problem(json!(0.04)));
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let mut model = engine_model();
    let setup = DdeCycleSetup {
        params: vec![0.3, TAU_H],
        free: 1,
        ntst: NTST,
        ncol: NCOL,
    };
    let want = dde_hopf_cycle(
        &mut model,
        &setup,
        &DdeHopfCycle {
            x: vec![0.0],
            omega: OMEGA_H,
            eps: 0.04,
        },
        &newton_options(),
        &AtomicBool::new(false),
    )
    .expect("hopf_cycle");
    assert_eq!(num(&v["alpha"]).to_bits(), want.alpha.to_bits(), "{out}");
    assert_eq!(num(&v["period"]).to_bits(), want.period.to_bits(), "{out}");
    // The orbit has left the Hopf point: alpha moved by about 0.1476 eps^2 and T by 0.977 eps^2.
    assert!(num(&v["alpha"]) != TAU_H);
    assert!((num(&v["period"]) - 2.0 * PI / OMEGA_H).abs() > 1e-6);
}

#[test]
fn both_or_neither_start_is_a_bad_problem() {
    let mut both = orbit_problem(&perturbed(), 6.0);
    both["hopf"] = json!({"x0": [0.0], "omega": OMEGA_H, "eps": 0.04});
    let (status, out, err) = dde_cycle("both.json", &both);
    assert_eq!(status, 2, "{err}");
    assert!(out.is_empty(), "{out}");
    assert!(err.contains("not both"), "{err}");

    let mut neither = orbit_problem(&perturbed(), 6.0);
    neither.as_object_mut().unwrap().remove("orbit");
    let (status, _, err) = dde_cycle("neither.json", &neither);
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("one of"), "{err}");
}

#[test]
fn an_unknown_key_is_a_bad_problem() {
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["nn"] = json!(3);
    let (status, _, err) = dde_cycle("unknown.json", &p);
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("unknown key \"nn\""), "{err}");

    let mut inner = orbit_problem(&perturbed(), 6.0);
    inner["orbit"]["eps"] = json!(0.1);
    let (status, _, err) = dde_cycle("unknown_inner.json", &inner);
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("unknown key \"eps\""), "{err}");
}

#[test]
fn options_the_corrector_does_not_read_are_refused() {
    for (key, value) in [("hmax", json!(0.1)), ("mesh_adapt", json!(0))] {
        let mut p = orbit_problem(&perturbed(), 6.0);
        p["options"][key] = value;
        let (status, _, err) = dde_cycle("refused_option.json", &p);
        assert_eq!(status, 2, "{key}: {err}");
        assert!(err.contains(&format!("options.{key}")), "{key}: {err}");
    }
}

#[test]
fn a_bad_eps_is_a_bad_problem() {
    // A JSON file has no NaN or infinity, so these are the values a file can hold.
    for eps in [json!(0.0), json!(-0.1), json!("0.04"), json!(null)] {
        let (status, _, err) = dde_cycle("bad_eps.json", &hopf_problem(eps.clone()));
        assert_eq!(status, 2, "eps = {eps}: {err}");
        assert!(err.contains("eps"), "eps = {eps}: {err}");
    }
    let mut missing = hopf_problem(json!(0.04));
    missing["hopf"].as_object_mut().unwrap().remove("eps");
    let (status, _, err) = dde_cycle("no_eps.json", &missing);
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("missing key \"eps\""), "{err}");
}

#[test]
fn a_profile_of_the_wrong_length_is_a_bad_problem() {
    let mut short = perturbed();
    short.pop();
    let (status, _, err) = dde_cycle("short_profile.json", &orbit_problem(&short, 6.0));
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("profile"), "{err}");
    let (status, _, err) = dde_cycle("empty_profile.json", &orbit_problem(&[], 6.0));
    assert_eq!(status, 2, "{err}");
    let (status, _, err) = dde_cycle("x0_length.json", &{
        let mut p = hopf_problem(json!(0.04));
        p["hopf"]["x0"] = json!([0.0, 0.0]);
        p
    });
    assert_eq!(status, 2, "{err}");
}

/// An edit of a problem object.
type Edit = Box<dyn Fn(&mut Value)>;

#[test]
fn bad_names_and_ranges_are_bad_problems() {
    let cases: [(&str, Edit, &str); 6] = [
        ("free_name", Box::new(|p| p["free"] = json!("nope")), "free"),
        (
            "param_name",
            Box::new(|p| p["params"]["nope"] = json!(1.0)),
            "nope",
        ),
        ("ntst_zero", Box::new(|p| p["ntst"] = json!(0)), "ntst"),
        ("ncol_one", Box::new(|p| p["ncol"] = json!(1)), "ncol"),
        ("ncol_eight", Box::new(|p| p["ncol"] = json!(8)), "ncol"),
        (
            "period_zero",
            Box::new(|p| p["orbit"]["period"] = json!(0.0)),
            "period",
        ),
    ];
    for (name, edit, word) in cases {
        let mut p = orbit_problem(&perturbed(), 6.0);
        edit(&mut p);
        let (status, _, err) = dde_cycle(&format!("{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(err.contains(word), "{name}: {err}");
    }
}

#[test]
fn a_run_that_does_not_converge_is_exit_1() {
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["options"]["max_newton"] = json!(1);
    let (status, out, err) = dde_cycle("no_converge.json", &p);
    assert_eq!(status, 1, "{err}");
    assert!(out.is_empty(), "{out}");
    assert!(!err.is_empty());
}

#[test]
fn a_model_without_a_delay_call_is_exit_1() {
    write("plain.ode", "par a=1\nx'=-a*x\ndone\n");
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["model"] = json!("plain.ode");
    p["params"] = json!({"a": 1.0});
    p["free"] = json!("a");
    let (status, _, err) = dde_cycle("plain.json", &p);
    assert_eq!(status, 1, "{err}");
}

#[test]
fn a_missing_or_invalid_file_is_exit_2() {
    let o = run(&scratch(), &["dde-cycle", "no_such_file.json"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let file = write("not_json.json", "{ nope");
    let o = run(&scratch(), &["dde-cycle", &file]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("invalid JSON"), "{}", stderr(&o));
}
