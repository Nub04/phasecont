//! Runs `pc-cli dde-cycle` on the model `x' = -x(t - tau) + mu x (1 - x^2 - x(t - tau)^2)`,
//! which has the exact orbit x = sin t, T = 2 pi for tau = pi/2 (SPEC section 8l, test 7).
//! The orbit and Hopf starts must give the numbers of the engine calls, and so must the
//! `floquet` key (SPEC section 8m, test 9) and the `branch` key (SPEC section 8n, test 8)
//! and the labels of a branch (SPEC section 8o, test 8) and the `adapt` key (SPEC section 8p,
//! test 9).
//! Also runs the bad problem files and the run that does not converge.

mod common;

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use common::{code, run, stderr, stdout};
use pc_engine::{
    DdeCycleBranch, DdeCyclePoint, DdeCycleSetup, DdeCycleStart, DdeFloquetOptions, DdeHopfCycle,
    DdeMeshAdapt, Evaluator, Options, build_dde, dde_continue_cycle, dde_continue_cycle_adapt,
    dde_continue_cycle_floquet, dde_correct_cycle, dde_floquet_multipliers, dde_hopf_cycle,
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

/// The problem of test 3 with the `floquet` key set to `floquet`.
fn floquet_problem(floquet: Value) -> Value {
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["floquet"] = floquet;
    p
}

#[test]
fn the_floquet_key_matches_the_engine_bit_for_bit() {
    let start = perturbed();
    let (status, out, err) = dde_cycle(
        "floquet.json",
        &floquet_problem(json!({"nhist": 16, "count": 4})),
    );
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let mut model = engine_model();
    let mut setup = DdeCycleSetup {
        params: vec![0.3, TAU],
        free: 1,
        ntst: NTST,
        ncol: NCOL,
    };
    let cancel = AtomicBool::new(false);
    let cycle = dde_correct_cycle(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: start,
            period: 6.0,
        },
        &newton_options(),
        &cancel,
    )
    .expect("correct_cycle");
    setup.params[setup.free] = cycle.alpha;
    let want = dde_floquet_multipliers(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: cycle.profile.clone(),
            period: cycle.period,
        },
        &DdeFloquetOptions {
            nhist: 16,
            count: 4,
        },
        &cancel,
    )
    .expect("floquet_multipliers");
    let got = v["multipliers"].as_array().expect("multipliers");
    assert_eq!(got.len(), 4, "{out}");
    for (i, (g, w)) in got.iter().zip(&want.multipliers).enumerate() {
        let pair = doubles(g);
        assert_eq!(pair.len(), 2, "multipliers[{i}]");
        assert_eq!(pair[0].to_bits(), w.0.to_bits(), "re[{i}]");
        assert_eq!(pair[1].to_bits(), w.1.to_bits(), "im[{i}]");
    }
    assert_eq!(
        num(&v["trivial_error"]).to_bits(),
        want.trivial_error.to_bits(),
        "{out}"
    );
    // The corrector's own numbers are still there, and the orbit is the exact one.
    assert_eq!(num(&v["alpha"]).to_bits(), cycle.alpha.to_bits(), "{out}");
    assert_eq!(num(&v["period"]).to_bits(), cycle.period.to_bits(), "{out}");
    assert!(want.trivial_error <= 1e-6, "{}", want.trivial_error);
}

#[test]
fn the_floquet_key_follows_the_parameter_that_a_hopf_start_moves() {
    let mut p = hopf_problem(json!(0.04));
    p["floquet"] = json!({"nhist": 12, "count": 3});
    let (status, out, err) = dde_cycle("floquet_hopf.json", &p);
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let mut model = engine_model();
    let mut setup = DdeCycleSetup {
        params: vec![0.3, TAU_H],
        free: 1,
        ntst: NTST,
        ncol: NCOL,
    };
    let cancel = AtomicBool::new(false);
    let cycle = dde_hopf_cycle(
        &mut model,
        &setup,
        &DdeHopfCycle {
            x: vec![0.0],
            omega: OMEGA_H,
            eps: 0.04,
        },
        &newton_options(),
        &cancel,
    )
    .expect("hopf_cycle");
    // The corrector moved alpha, so the multipliers must be taken at the new value.
    assert!(cycle.alpha != TAU_H);
    setup.params[setup.free] = cycle.alpha;
    let want = dde_floquet_multipliers(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: cycle.profile.clone(),
            period: cycle.period,
        },
        &DdeFloquetOptions {
            nhist: 12,
            count: 3,
        },
        &cancel,
    )
    .expect("floquet_multipliers");
    let got = v["multipliers"].as_array().expect("multipliers");
    assert_eq!(got.len(), 3, "{out}");
    for (i, (g, w)) in got.iter().zip(&want.multipliers).enumerate() {
        let pair = doubles(g);
        assert_eq!(pair[0].to_bits(), w.0.to_bits(), "re[{i}]");
        assert_eq!(pair[1].to_bits(), w.1.to_bits(), "im[{i}]");
    }
}

#[test]
fn without_the_floquet_key_neither_field_is_printed() {
    let (status, out, err) = dde_cycle("no_floquet.json", &orbit_problem(&perturbed(), 6.0));
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    assert!(!keys.contains(&"multipliers"), "{keys:?}");
    assert!(!keys.contains(&"trivial_error"), "{keys:?}");
}

#[test]
fn a_malformed_floquet_key_is_exit_2_and_names_floquet() {
    let bad = [
        json!(true),
        json!(null),
        json!([16, 4]),
        json!("nhist"),
        json!({}),
        json!({"nhist": 16}),
        json!({"count": 4}),
        json!({"nhist": 16, "count": 4, "tol": 1}),
        json!({"nhist": 16.5, "count": 4}),
        json!({"nhist": 16, "count": 2.5}),
        json!({"nhist": 0, "count": 4}),
        json!({"nhist": 16, "count": 0}),
        json!({"nhist": -1, "count": 4}),
        json!({"nhist": 16, "count": "4"}),
    ];
    for (i, floquet) in bad.into_iter().enumerate() {
        let shown = floquet.to_string();
        let (status, out, err) =
            dde_cycle(&format!("floquet_bad{i}.json"), &floquet_problem(floquet));
        assert_eq!(status, 2, "{shown}: {err}");
        assert!(out.is_empty(), "{shown}: {out}");
        assert!(err.contains("floquet"), "{shown}: {err}");
    }
}

#[test]
fn floquet_options_the_model_cannot_take_are_exit_2_and_name_floquet() {
    // OSC has n = 1, so count may be at most nhist + 1, and ncol * (nhist + 1) at most 1000.
    for (i, floquet) in [
        json!({"nhist": 2, "count": 4}),
        json!({"nhist": 300, "count": 1}),
    ]
    .into_iter()
    .enumerate()
    {
        let shown = floquet.to_string();
        let (status, out, err) =
            dde_cycle(&format!("floquet_range{i}.json"), &floquet_problem(floquet));
        assert_eq!(status, 2, "{shown}: {err}");
        assert!(out.is_empty(), "{shown}: {out}");
        assert!(err.contains("floquet"), "{shown}: {err}");
    }
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

#[test]
fn help_lists_the_floquet_key() {
    let o = run(&scratch(), &["dde-cycle", "--help"]);
    assert_eq!(code(&o), 0);
    let text = stdout(&o);
    assert!(text.starts_with("usage: pc-cli dde-cycle"), "{text}");
    for word in ["floquet", "nhist", "count", "multipliers", "trivial_error"] {
        assert!(text.contains(word), "{word} missing from the help text");
    }
}

// ---------------------------------------------------------------------------------------
// The `branch` key (SPEC section 8n, test 8).

/// The exact orbit x = sin t at the base points.
fn sine() -> Vec<f64> {
    let nq = NTST * NCOL;
    (0..=nq)
        .map(|q| (2.0 * PI * q as f64 / nq as f64).sin())
        .collect()
}

/// The problem of the mu-line (test 1 with K = 0, direction +1) with `branch` set to true.
fn mu_line_problem() -> Value {
    json!({
        "model": "osc.ode",
        "params": {"mu": 0.3, "tau": TAU},
        "free": "mu",
        "ntst": NTST,
        "ncol": NCOL,
        "orbit": {"profile": sine(), "period": 2.0 * PI},
        "options": {
            "max_newton": 10, "hmax": 0.2, "max_steps": 200,
            "bounds": [0.1, 0.9], "direction": 1
        },
        "branch": true,
    })
}

/// The problem of test 2: the Hopf cycle at eps = 0.05, then up in tau to pi/2. The corrector
/// of the Hopf start needs `max_newton` 30, and the same options go to the continuation.
fn hopf_branch_problem() -> Value {
    json!({
        "model": "osc.ode",
        "params": {"mu": 0.3, "tau": TAU_H},
        "free": "tau",
        "ntst": NTST,
        "ncol": NCOL,
        "hopf": {"x0": [0.0], "omega": OMEGA_H, "eps": 0.05},
        "options": {
            "max_newton": 30, "hmax": 0.5, "max_steps": 500,
            "bounds": [1.0, PI / 2.0], "direction": 1
        },
        "branch": true,
    })
}

/// Checks that the printed points are those of `want`, bit for bit.
fn same_points(out: &str, want: &DdeCycleBranch, what: &str) {
    let v: Value = serde_json::from_str(out).unwrap();
    assert_eq!(v["ntst"], NTST, "{what}");
    assert_eq!(v["ncol"], NCOL, "{what}");
    let got = v["points"].as_array().expect("points");
    assert_eq!(got.len(), want.points.len(), "{what}: number of points");
    for (i, (g, w)) in got.iter().zip(&want.points).enumerate() {
        assert_eq!(num(&g["alpha"]).to_bits(), w.alpha.to_bits(), "{what} {i}");
        assert_eq!(
            num(&g["period"]).to_bits(),
            w.period.to_bits(),
            "{what} {i}"
        );
        assert_eq!(num(&g["h"]).to_bits(), w.h.to_bits(), "{what} {i}");
        assert_eq!(
            g["iters"].as_u64(),
            Some(w.newton_iters as u64),
            "{what} {i}"
        );
        let p = doubles(&g["profile"]);
        assert_eq!(p.len(), w.profile.len(), "{what} {i}: profile length");
        for (j, (a, b)) in p.iter().zip(&w.profile).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{what} {i} profile[{j}]");
        }
    }
    assert!(v["warnings"].is_array(), "{what}");
}

#[test]
fn the_branch_key_matches_continue_cycle_bit_for_bit() {
    let (status, out, err) = dde_cycle("branch_orbit.json", &mu_line_problem());
    assert_eq!(status, 0, "{err}");
    let mut model = engine_model();
    let setup = DdeCycleSetup {
        params: vec![0.3, TAU],
        free: 0,
        ntst: NTST,
        ncol: NCOL,
    };
    let opts = Options {
        bounds: (0.1, 0.9),
        direction: 1.0,
        hmax: 0.2,
        max_steps: 200,
        max_newton: 10,
        ..Options::default()
    };
    let want = dde_continue_cycle(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: sine(),
            period: 2.0 * PI,
        },
        &opts,
        &AtomicBool::new(false),
    )
    .expect("continue_cycle");
    assert!(want.points.len() > 2, "{} points", want.points.len());
    same_points(&out, &want, "orbit");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["stop"], "bounds", "{out}");
    let last = v["points"].as_array().unwrap().last().unwrap();
    assert!((num(&last["alpha"]) - 0.9).abs() <= 1e-9, "{out}");
}

#[test]
fn the_branch_key_runs_the_hopf_cycle_first() {
    let (status, out, err) = dde_cycle("branch_hopf.json", &hopf_branch_problem());
    assert_eq!(status, 0, "{err}");
    let mut model = engine_model();
    let opts = Options {
        bounds: (1.0, PI / 2.0),
        direction: 1.0,
        hmax: 0.5,
        max_steps: 500,
        max_newton: 30,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let hopf = dde_hopf_cycle(
        &mut model,
        &DdeCycleSetup {
            params: vec![0.3, TAU_H],
            free: 1,
            ntst: NTST,
            ncol: NCOL,
        },
        &DdeHopfCycle {
            x: vec![0.0],
            omega: OMEGA_H,
            eps: 0.05,
        },
        &opts,
        &cancel,
    )
    .expect("hopf_cycle");
    // The continuation starts at the parameter value that the Hopf corrector found.
    let want = dde_continue_cycle(
        &mut model,
        &DdeCycleSetup {
            params: vec![0.3, hopf.alpha],
            free: 1,
            ntst: NTST,
            ncol: NCOL,
        },
        &DdeCycleStart {
            profile: hopf.profile.clone(),
            period: hopf.period,
        },
        &opts,
        &cancel,
    )
    .expect("continue_cycle");
    same_points(&out, &want, "hopf");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["stop"], "bounds", "{out}");
    let first = &v["points"][0];
    assert!((num(&first["alpha"]) - hopf.alpha).abs() <= 1e-9, "{out}");
    let last = v["points"].as_array().unwrap().last().unwrap();
    assert!((num(&last["alpha"]) - PI / 2.0).abs() <= 1e-9, "{out}");
}

#[test]
fn branch_false_and_no_branch_print_the_corrector_output() {
    let mut off = orbit_problem(&perturbed(), 6.0);
    let (status, plain, err) = dde_cycle("no_branch.json", &off);
    assert_eq!(status, 0, "{err}");
    off["branch"] = json!(false);
    let (status, explicit, err) = dde_cycle("branch_false.json", &off);
    assert_eq!(status, 0, "{err}");
    assert_eq!(plain, explicit);
    let v: Value = serde_json::from_str(&plain).unwrap();
    assert!(
        v.get("points").is_none() && v.get("stop").is_none(),
        "{plain}"
    );
    assert!(
        v.get("period").is_some() && v.get("profile").is_some(),
        "{plain}"
    );
    // Without the branch, the options of the corrector still apply.
    off["options"]["hmax"] = json!(0.1);
    let (status, _, err) = dde_cycle("branch_false_hmax.json", &off);
    assert_eq!(status, 2, "{err}");
    assert!(err.contains("options.hmax"), "{err}");
}

#[test]
fn branch_problems_that_cannot_run_are_exit_2() {
    let cases: [(&str, Edit, &str); 5] = [
        (
            "with_mesh_adapt",
            Box::new(|p| p["options"]["mesh_adapt"] = json!(3)),
            "options.mesh_adapt",
        ),
        (
            "with_detect",
            Box::new(|p| p["options"]["detect"] = json!({"lp": true})),
            "options.detect",
        ),
        (
            "branch_string",
            Box::new(|p| p["branch"] = json!("yes")),
            "branch",
        ),
        (
            "branch_null",
            Box::new(|p| p["branch"] = json!(null)),
            "branch",
        ),
        (
            "delay_free_zero_bound",
            Box::new(|p| {
                p["free"] = json!("tau");
                p["options"]["bounds"] = json!([0.0, 2.0]);
            }),
            "bounds",
        ),
    ];
    for (name, edit, word) in cases {
        let mut p = mu_line_problem();
        edit(&mut p);
        let (status, out, err) = dde_cycle(&format!("{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(out.is_empty(), "{name}: {out}");
        assert!(err.contains(word), "{name}: {err}");
    }
}

#[test]
fn a_branch_start_that_does_not_converge_is_exit_1() {
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["options"] = json!({"max_newton": 1, "bounds": [0.1, 2.0]});
    p["branch"] = json!(true);
    let (status, out, err) = dde_cycle("branch_no_converge.json", &p);
    assert_eq!(status, 1, "{err}");
    assert!(out.is_empty(), "{out}");
    assert!(!err.is_empty());
}

#[test]
fn help_lists_the_branch_key() {
    let o = run(&scratch(), &["dde-cycle", "--help"]);
    assert_eq!(code(&o), 0);
    let text = stdout(&o);
    for word in ["branch", "points", "specials", "bounds", "mesh_adapt"] {
        assert!(text.contains(word), "{word} missing from the help text");
    }
}

// ---------------------------------------------------------------------------------------
// The labels of a branch (SPEC section 8o, test 8).

/// The cubic model of section 8o, test 3: a fold of cycles at `a = 0`.
fn lpc_source() -> String {
    format!(
        "par a=0.25, tau={TAU:?}\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         done\n"
    )
}

/// The start of the fold run: the upper cycle `r sin t` at `a = 0.25`, `r^2 = 1.5`.
fn lpc_profile() -> Vec<f64> {
    let nq = NTST * NCOL;
    let r = 1.5_f64.sqrt();
    (0..=nq)
        .map(|q| r * (2.0 * PI * q as f64 / nq as f64).sin())
        .collect()
}

/// The fold run: down in `a` through the fold at `a = 0`, `branch` set to true.
fn lpc_problem() -> Value {
    json!({
        "model": "lpc.ode",
        "params": {"a": 0.25, "tau": TAU},
        "free": "a",
        "ntst": NTST,
        "ncol": NCOL,
        "orbit": {"profile": lpc_profile(), "period": 2.0 * PI},
        "options": {
            "max_newton": 10, "hmax": 0.1, "max_steps": 400,
            "bounds": [-0.5, 0.4], "direction": -1
        },
        "branch": true,
    })
}

/// Runs the command on the fold problem `problem`, which names the model `lpc.ode`.
fn lpc_cycle(name: &str, problem: &Value) -> (i32, String, String) {
    write("lpc.ode", &lpc_source());
    let file = write(name, &problem.to_string());
    let o = run(&scratch(), &["dde-cycle", &file]);
    (code(&o), stdout(&o), stderr(&o))
}

/// The engine run that the fold problem stands for, with `floquet` as given.
fn lpc_engine(floquet: Option<&DdeFloquetOptions>) -> DdeCycleBranch {
    let mut map = SourceMap::new();
    let top = map.add("lpc.ode", lpc_source());
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    let mut model = build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("build_dde");
    let setup = DdeCycleSetup {
        params: vec![0.25, TAU],
        free: 0,
        ntst: NTST,
        ncol: NCOL,
    };
    let opts = Options {
        bounds: (-0.5, 0.4),
        direction: -1.0,
        hmax: 0.1,
        max_steps: 400,
        max_newton: 10,
        ..Options::default()
    };
    dde_continue_cycle_floquet(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: lpc_profile(),
            period: 2.0 * PI,
        },
        &opts,
        floquet,
        &AtomicBool::new(false),
    )
    .expect("continue_cycle_floquet")
}

fn same_bits(got: &Value, want: f64, what: &str) {
    // A NaN prints as null, so it is compared as such.
    if want.is_nan() {
        assert!(got.is_null(), "{what}: {got}");
    } else {
        assert_eq!(num(got).to_bits(), want.to_bits(), "{what}");
    }
}

fn same_multipliers(got: &Value, p: &DdeCyclePoint, what: &str) {
    let pairs = got["multipliers"]
        .as_array()
        .unwrap_or_else(|| panic!("{what}: no multipliers in {got}"));
    assert_eq!(pairs.len(), p.multipliers.len(), "{what}: count");
    for (i, (g, w)) in pairs.iter().zip(&p.multipliers).enumerate() {
        let pair = doubles(g);
        assert_eq!(pair.len(), 2, "{what} multipliers[{i}]");
        assert_eq!(pair[0].to_bits(), w.0.to_bits(), "{what} re[{i}]");
        assert_eq!(pair[1].to_bits(), w.1.to_bits(), "{what} im[{i}]");
    }
    same_bits(&got["trivial_error"], p.trivial_error, what);
}

/// Checks that the printed specials are those of `want`, bit for bit, with the multipliers
/// of the points and of the specials when `floquet` is true.
fn same_specials(out: &str, want: &DdeCycleBranch, floquet: bool, what: &str) {
    let v: Value = serde_json::from_str(out).unwrap();
    let got = v["specials"].as_array().expect("specials");
    assert_eq!(got.len(), want.specials.len(), "{what}: number of specials");
    for (i, (g, w)) in got.iter().zip(&want.specials).enumerate() {
        assert_eq!(g["label"].as_str(), Some(w.label.as_str()), "{what} {i}");
        assert_eq!(g["index"].as_u64(), Some(w.index as u64), "{what} {i}");
        assert_eq!(
            num(&g["alpha"]).to_bits(),
            w.point.alpha.to_bits(),
            "{what} {i}"
        );
        assert_eq!(
            num(&g["period"]).to_bits(),
            w.point.period.to_bits(),
            "{what} {i}"
        );
        let p = doubles(&g["profile"]);
        assert_eq!(p.len(), w.point.profile.len(), "{what} {i}: profile length");
        for (j, (a, b)) in p.iter().zip(&w.point.profile).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{what} {i} profile[{j}]");
        }
        assert_eq!(g.get("multipliers").is_some(), floquet, "{what} {i}");
        if floquet {
            same_multipliers(g, &w.point, &format!("{what} special {i}"));
        }
    }
    let points = v["points"].as_array().expect("points");
    assert_eq!(points.len(), want.points.len(), "{what}: number of points");
    for (i, (g, w)) in points.iter().zip(&want.points).enumerate() {
        assert_eq!(g.get("multipliers").is_some(), floquet, "{what} point {i}");
        assert_eq!(
            g.get("trivial_error").is_some(),
            floquet,
            "{what} point {i}"
        );
        if floquet {
            same_multipliers(g, w, &format!("{what} point {i}"));
        }
    }
}

#[test]
fn a_branch_prints_its_specials_bit_for_bit() {
    let (status, out, err) = lpc_cycle("lpc_branch.json", &lpc_problem());
    assert_eq!(status, 0, "{err}");
    let want = lpc_engine(None);
    let labels: Vec<&str> = want.specials.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["LPC"], "{out}");
    same_points(&out, &want, "lpc");
    same_specials(&out, &want, false, "lpc");
    let v: Value = serde_json::from_str(&out).unwrap();
    let special = &v["specials"][0];
    assert!(num(&special["alpha"]).abs() <= 2e-11, "{out}");
}

#[test]
fn a_branch_with_floquet_prints_the_multipliers_bit_for_bit() {
    let mut p = lpc_problem();
    p["floquet"] = json!({"nhist": 12, "count": 4});
    let (status, out, err) = lpc_cycle("lpc_branch_floquet.json", &p);
    assert_eq!(status, 0, "{err}");
    let want = lpc_engine(Some(&DdeFloquetOptions {
        nhist: 12,
        count: 4,
    }));
    assert!(want.points.len() > 10, "{} points", want.points.len());
    same_points(&out, &want, "lpc floquet");
    same_specials(&out, &want, true, "lpc floquet");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(
        v["specials"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["label"] == "LPC"),
        "{out}"
    );
}

#[test]
fn a_branch_with_a_bad_floquet_is_exit_2_and_names_floquet() {
    let cases = [
        ("nhist_zero", json!({"nhist": 0, "count": 1})),
        ("count_zero", json!({"nhist": 8, "count": 0})),
        ("count_above_dimension", json!({"nhist": 4, "count": 6})),
        ("stage_system_too_big", json!({"nhist": 4000, "count": 1})),
    ];
    for (name, floquet) in cases {
        let mut p = lpc_problem();
        p["floquet"] = floquet;
        let (status, out, err) = lpc_cycle(&format!("lpc_bad_{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(out.is_empty(), "{name}: {out}");
        assert!(err.contains("floquet"), "{name}: {err}");
    }
}

#[test]
fn a_branch_still_refuses_detect_and_mesh_adapt_with_floquet() {
    for (name, key, value) in [
        ("detect", "detect", json!({"lp": true})),
        ("mesh_adapt", "mesh_adapt", json!(3)),
    ] {
        let mut p = lpc_problem();
        p["floquet"] = json!({"nhist": 12, "count": 4});
        p["options"][key] = value;
        let (status, out, err) = lpc_cycle(&format!("lpc_floquet_{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(out.is_empty(), "{name}: {out}");
        assert!(err.contains(&format!("options.{key}")), "{name}: {err}");
    }
}

#[test]
fn a_run_without_branch_prints_no_specials() {
    let mut p = orbit_problem(&perturbed(), 6.0);
    p["floquet"] = json!({"nhist": 8, "count": 2});
    let (status, out, err) = dde_cycle("no_branch_floquet.json", &p);
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(
        v.get("specials").is_none() && v.get("points").is_none(),
        "{out}"
    );
    assert!(v.get("multipliers").is_some(), "{out}");
}

// ---------------------------------------------------------------------------------------
// The NS and PD labels of a branch (SPEC section 8o, test 8), and a bad `floquet` with a
// Hopf start.

/// A model of section 8o other than the fold: its source, its parameters and the run.
struct Case {
    file: &'static str,
    source: String,
    names: &'static [&'static str],
    params: Vec<f64>,
    free: usize,
    components: usize,
    bounds: (f64, f64),
    hmax: f64,
}

impl Case {
    /// Test 1: the oscillator plus `y' = -s y(t - 1)`, `s` from 0.3 up through `pi/2`.
    fn ns() -> Case {
        Case {
            file: "ns.ode",
            source: format!(
                "par mu=0.3, tau={TAU:?}, s=0.3\n\
                 x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
                 y'=-s*delay(y,1)\n\
                 done\n"
            ),
            names: &["mu", "tau", "s"],
            params: vec![0.3, TAU, 0.3],
            free: 2,
            components: 2,
            bounds: (0.1, 2.0),
            hmax: 0.2,
        }
    }

    /// Test 2: the oscillator plus the rotating block, `s` from 0.4 up through `sqrt(0.34)`.
    fn pd() -> Case {
        Case {
            file: "pd.ode",
            source: format!(
                "par mu=0.3, tau={TAU:?}, s=0.4, g=0.3\n\
                 x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
                 y'=s*(-delay(x,tau)*y+x*z)-g*y\n\
                 z'=s*(x*y+delay(x,tau)*z)-g*z\n\
                 done\n"
            ),
            names: &["mu", "tau", "s", "g"],
            params: vec![0.3, TAU, 0.4, 0.3],
            free: 2,
            components: 3,
            bounds: (0.3, 0.8),
            hmax: 0.1,
        }
    }

    /// The exact cycle `x = sin t`, `T = 2 pi`, with the other components zero.
    fn profile(&self) -> Vec<f64> {
        let nq = NTST * NCOL;
        (0..=nq)
            .flat_map(|q| {
                let s = (2.0 * PI * q as f64 / nq as f64).sin();
                (0..self.components).map(move |c| if c == 0 { s } else { 0.0 })
            })
            .collect()
    }

    fn problem(&self, floquet: Option<Value>) -> Value {
        let params: serde_json::Map<String, Value> = self
            .names
            .iter()
            .zip(&self.params)
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();
        let mut p = json!({
            "model": self.file,
            "params": params,
            "free": self.names[self.free],
            "ntst": NTST,
            "ncol": NCOL,
            "orbit": {"profile": self.profile(), "period": 2.0 * PI},
            "options": {
                "max_newton": 10, "hmax": self.hmax, "max_steps": 400,
                "bounds": [self.bounds.0, self.bounds.1], "direction": 1
            },
            "branch": true,
        });
        if let Some(f) = floquet {
            p["floquet"] = f;
        }
        p
    }

    fn cli(&self, name: &str, problem: &Value) -> (i32, String, String) {
        write(self.file, &self.source);
        let file = write(name, &problem.to_string());
        let o = run(&scratch(), &["dde-cycle", &file]);
        (code(&o), stdout(&o), stderr(&o))
    }

    fn engine(&self, floquet: &DdeFloquetOptions) -> DdeCycleBranch {
        let mut map = SourceMap::new();
        let top = map.add(self.file, self.source.clone());
        let l = load(top, &mut map, &mut MapLoader::new());
        assert!(
            l.diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{:?}",
            l.diagnostics
        );
        let mut model = build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("build_dde");
        let setup = DdeCycleSetup {
            params: self.params.clone(),
            free: self.free,
            ntst: NTST,
            ncol: NCOL,
        };
        let opts = Options {
            bounds: self.bounds,
            direction: 1.0,
            hmax: self.hmax,
            max_steps: 400,
            max_newton: 10,
            ..Options::default()
        };
        dde_continue_cycle_floquet(
            &mut model,
            &setup,
            &DdeCycleStart {
                profile: self.profile(),
                period: 2.0 * PI,
            },
            &opts,
            Some(floquet),
            &AtomicBool::new(false),
        )
        .expect("continue_cycle_floquet")
    }
}

/// Runs the case with `floquet`, compares every printed number with the engine call and
/// returns the printed output.
fn labelled_branch(case: &Case, name: &str, label: &str, alpha: f64, tol: f64) {
    let (status, out, err) = case.cli(
        &format!("{name}.json"),
        &case.problem(Some(json!({"nhist": 16, "count": 6}))),
    );
    assert_eq!(status, 0, "{name}: {err}");
    let want = case.engine(&DdeFloquetOptions {
        nhist: 16,
        count: 6,
    });
    let labels: Vec<&str> = want.specials.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, [label], "{name}: {out}");
    same_points(&out, &want, name);
    same_specials(&out, &want, true, name);
    let v: Value = serde_json::from_str(&out).unwrap();
    let got = num(&v["specials"][0]["alpha"]);
    assert!((got - alpha).abs() <= tol, "{name}: {label} at {got}");
}

#[test]
fn a_branch_prints_the_ns_point_bit_for_bit() {
    labelled_branch(&Case::ns(), "ns_branch", "NS", PI / 2.0, 1e-12);
}

#[test]
fn a_branch_prints_the_pd_point_bit_for_bit() {
    labelled_branch(&Case::pd(), "pd_branch", "PD", 0.34_f64.sqrt(), 1e-11);
}

#[test]
fn a_bad_floquet_with_a_hopf_start_is_exit_2_before_the_run() {
    // `max_newton` 1 keeps the Hopf correction from converging, so a check that came after
    // it would give exit 1 with a convergence message.
    for (name, branch) in [("branch", true), ("single", false)] {
        for (i, floquet) in [
            json!({"nhist": 4, "count": 6}),
            json!({"nhist": 4000, "count": 1}),
        ]
        .into_iter()
        .enumerate()
        {
            let mut p = hopf_branch_problem();
            p["floquet"] = floquet.clone();
            p["options"] = json!({"max_newton": 1});
            if !branch {
                p.as_object_mut().unwrap().remove("branch");
            }
            let (status, out, err) = dde_cycle(&format!("hopf_bad_floquet_{name}{i}.json"), &p);
            assert_eq!(status, 2, "{name} {floquet}: {err}");
            assert!(out.is_empty(), "{name} {floquet}: {out}");
            assert!(err.contains("floquet"), "{name} {floquet}: {err}");
        }
    }
    // The same start without the bad floquet does not converge, which is exit 1.
    let mut p = hopf_branch_problem();
    p["options"] = json!({"max_newton": 1});
    let (status, _, err) = dde_cycle("hopf_no_converge.json", &p);
    assert_eq!(status, 1, "{err}");
}

// ---------------------------------------------------------------------------------------
// The `adapt` key (SPEC section 8p, test 9).

fn uniform_mesh() -> Vec<f64> {
    (0..=NTST).map(|j| j as f64 / NTST as f64).collect()
}

/// Checks the printed meshes of the points and the specials against `want`, bit for bit.
fn same_meshes(out: &str, want: &DdeCycleBranch, what: &str) {
    let v: Value = serde_json::from_str(out).unwrap();
    let check = |got: &Value, w: &DdeCyclePoint, at: String| {
        let mesh = doubles(&got["mesh"]);
        assert_eq!(mesh.len(), NTST + 1, "{at}: mesh length");
        assert_eq!(mesh.len(), w.mesh.len(), "{at}: mesh length");
        for (j, (a, b)) in mesh.iter().zip(&w.mesh).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{at} mesh[{j}]");
        }
    };
    for (i, (g, w)) in v["points"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&want.points)
        .enumerate()
    {
        check(g, w, format!("{what} point {i}"));
    }
    for (i, (g, w)) in v["specials"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&want.specials)
        .enumerate()
    {
        check(g, &w.point, format!("{what} special {i}"));
    }
}

/// The mu-line problem with `adapt` and, if given, `floquet`.
fn mu_line_adapt(every: u64, floquet: Option<Value>) -> Value {
    let mut p = mu_line_problem();
    p["adapt"] = json!({"every": every});
    if let Some(f) = floquet {
        p["floquet"] = f;
    }
    p
}

/// The engine run that `mu_line_adapt(every, floquet)` stands for.
fn mu_line_engine(every: usize, floquet: Option<&DdeFloquetOptions>) -> DdeCycleBranch {
    let mut model = engine_model();
    let setup = DdeCycleSetup {
        params: vec![0.3, TAU],
        free: 0,
        ntst: NTST,
        ncol: NCOL,
    };
    let opts = Options {
        bounds: (0.1, 0.9),
        direction: 1.0,
        hmax: 0.2,
        max_steps: 200,
        max_newton: 10,
        ..Options::default()
    };
    dde_continue_cycle_adapt(
        &mut model,
        &setup,
        &DdeCycleStart {
            profile: sine(),
            period: 2.0 * PI,
        },
        &opts,
        floquet,
        &DdeMeshAdapt { every },
        &AtomicBool::new(false),
    )
    .expect("continue_cycle_adapt")
}

#[test]
fn the_adapt_key_matches_the_engine_bit_for_bit() {
    let (status, out, err) = dde_cycle("adapt_orbit.json", &mu_line_adapt(3, None));
    assert_eq!(status, 0, "{err}");
    let want = mu_line_engine(3, None);
    assert!(want.points.len() > 4, "{} points", want.points.len());
    same_points(&out, &want, "adapt");
    same_meshes(&out, &want, "adapt");
    // The first mesh is uniform and a later one is not, so the run did adapt.
    assert_eq!(want.points[0].mesh, uniform_mesh());
    assert!(
        want.points.iter().any(|p| p.mesh != uniform_mesh()),
        "no point has a moved mesh"
    );
}

#[test]
fn the_adapt_key_with_floquet_also_prints_the_multipliers() {
    let floquet = json!({"nhist": 8, "count": 2});
    let (status, out, err) =
        dde_cycle("adapt_orbit_floquet.json", &mu_line_adapt(2, Some(floquet)));
    assert_eq!(status, 0, "{err}");
    let want = mu_line_engine(2, Some(&DdeFloquetOptions { nhist: 8, count: 2 }));
    same_points(&out, &want, "adapt floquet");
    same_specials(&out, &want, true, "adapt floquet");
    same_meshes(&out, &want, "adapt floquet");
}

#[test]
fn the_adapt_key_prints_the_mesh_of_a_special_point() {
    let mut p = lpc_problem();
    p["adapt"] = json!({"every": 4});
    let (status, out, err) = lpc_cycle("adapt_lpc.json", &p);
    assert_eq!(status, 0, "{err}");
    let mut model = {
        let mut map = SourceMap::new();
        let top = map.add("lpc.ode", lpc_source());
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("build_dde")
    };
    let want = dde_continue_cycle_adapt(
        &mut model,
        &DdeCycleSetup {
            params: vec![0.25, TAU],
            free: 0,
            ntst: NTST,
            ncol: NCOL,
        },
        &DdeCycleStart {
            profile: lpc_profile(),
            period: 2.0 * PI,
        },
        &Options {
            bounds: (-0.5, 0.4),
            direction: -1.0,
            hmax: 0.1,
            max_steps: 400,
            max_newton: 10,
            ..Options::default()
        },
        None,
        &DdeMeshAdapt { every: 4 },
        &AtomicBool::new(false),
    )
    .expect("continue_cycle_adapt");
    let labels: Vec<&str> = want.specials.iter().map(|s| s.label.as_str()).collect();
    assert_eq!(labels, ["LPC"], "{out}");
    same_points(&out, &want, "adapt lpc");
    same_specials(&out, &want, false, "adapt lpc");
    same_meshes(&out, &want, "adapt lpc");
}

#[test]
fn without_adapt_no_mesh_is_printed_and_a_late_adapt_changes_nothing_else() {
    let (status, plain, err) = dde_cycle("adapt_off.json", &mu_line_problem());
    assert_eq!(status, 0, "{err}");
    let v: Value = serde_json::from_str(&plain).unwrap();
    for p in v["points"].as_array().unwrap() {
        assert!(p.get("mesh").is_none(), "{p}");
    }
    // A run with a special point prints no mesh on the special either.
    let (status, with_lpc, err) = lpc_cycle("adapt_off_lpc.json", &lpc_problem());
    assert_eq!(status, 0, "{err}");
    let l: Value = serde_json::from_str(&with_lpc).unwrap();
    let specials = l["specials"].as_array().unwrap();
    assert!(!specials.is_empty(), "{with_lpc}");
    for s in specials {
        assert!(s.get("mesh").is_none(), "{s}");
    }
    // `every` so large that no point is ever due: the points are those of the plain run, and
    // only the mesh key is new (SPEC section 8p, test 4).
    let (status, late, err) = dde_cycle("adapt_late.json", &mu_line_adapt(u64::MAX, None));
    assert_eq!(status, 0, "{err}");
    let mut w: Value = serde_json::from_str(&late).unwrap();
    for p in w["points"].as_array_mut().unwrap() {
        assert_eq!(doubles(&p["mesh"]), uniform_mesh());
        p.as_object_mut().unwrap().remove("mesh");
    }
    assert_eq!(w, v);
}

/// The manufactured model STEEP of SPEC section 8p, with the steepness 20 written as a number.
/// Its exact orbit is `y1 = cos t`, `y2 = sin t`, `x = tanh(20 cos t)`, `T = 2 pi`.
fn steep_source() -> String {
    "par c=0.5, tau=1\n\
     x'=-x+c*(delay(x,tau)-tanh(20*delay(y1,tau)))+tanh(20*y1)+20*(1-tanh(20*y1)^2)*(-y2)\n\
     y1'=y1-y2-y1*(y1^2+y2^2)\n\
     y2'=y1+y2-y2*(y1^2+y2^2)\n\
     done\n"
        .to_string()
}

/// The exact STEEP orbit on the uniform mesh, `x`, `y1`, `y2` for each base point.
fn steep_profile() -> Vec<f64> {
    let nq = NTST * NCOL;
    (0..=nq)
        .flat_map(|q| {
            let t = 2.0 * PI * q as f64 / nq as f64;
            [(20.0 * t.cos()).tanh(), t.cos(), t.sin()]
        })
        .collect()
}

/// The problem of test 1 of section 8p: `c` up from 0.5 to 0.7, with `adapt` every point.
fn steep_problem(floquet: Option<Value>) -> Value {
    let mut p = json!({
        "model": "steep.ode",
        "params": {"c": 0.5, "tau": 1.0},
        "free": "c",
        "ntst": NTST,
        "ncol": NCOL,
        "orbit": {"profile": steep_profile(), "period": 2.0 * PI},
        "options": {
            "max_newton": 10, "hmax": 0.1, "max_steps": 60,
            "bounds": [0.3, 0.7], "direction": 1
        },
        "branch": true,
        "adapt": {"every": 1},
    });
    if let Some(f) = floquet {
        p["floquet"] = f;
    }
    p
}

/// The engine run that `steep_problem(floquet)` stands for.
fn steep_engine(floquet: Option<&DdeFloquetOptions>) -> DdeCycleBranch {
    let mut map = SourceMap::new();
    let top = map.add("steep.ode", steep_source());
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    let mut model = build_dde(&l.expanded, &l.info, Evaluator::Jit).expect("build_dde");
    dde_continue_cycle_adapt(
        &mut model,
        &DdeCycleSetup {
            params: vec![0.5, 1.0],
            free: 0,
            ntst: NTST,
            ncol: NCOL,
        },
        &DdeCycleStart {
            profile: steep_profile(),
            period: 2.0 * PI,
        },
        &Options {
            bounds: (0.3, 0.7),
            direction: 1.0,
            hmax: 0.1,
            max_steps: 60,
            max_newton: 10,
            ..Options::default()
        },
        floquet,
        &DdeMeshAdapt { every: 1 },
        &AtomicBool::new(false),
    )
    .expect("continue_cycle_adapt")
}

/// SPEC section 8p, test 9: the problem of test 1 prints the points of `continue_cycle_adapt`
/// bit for bit, without and with `floquet`, and the meshes of the later points have moved.
#[test]
fn the_adapt_key_matches_the_engine_on_the_steep_problem() {
    write("steep.ode", &steep_source());
    for (name, floquet, engine_floquet) in [
        ("adapt_steep.json", None, None),
        (
            "adapt_steep_floquet.json",
            Some(json!({"nhist": 16, "count": 2})),
            Some(DdeFloquetOptions {
                nhist: 16,
                count: 2,
            }),
        ),
    ] {
        let file = write(name, &steep_problem(floquet.clone()).to_string());
        let o = run(&scratch(), &["dde-cycle", &file]);
        let (status, out, err) = (code(&o), stdout(&o), stderr(&o));
        assert_eq!(status, 0, "{name}: {err}");
        let want = steep_engine(engine_floquet.as_ref());
        assert!(
            want.points.len() > 6,
            "{name}: {} points",
            want.points.len()
        );
        same_points(&out, &want, name);
        same_specials(&out, &want, floquet.is_some(), name);
        same_meshes(&out, &want, name);
        assert_eq!(want.points[0].mesh, uniform_mesh(), "{name}");
        let last = want.points.last().unwrap();
        let moved = last
            .mesh
            .iter()
            .zip(uniform_mesh())
            .fold(0.0_f64, |a, (s, u)| a.max((s - u).abs()));
        assert!(moved > 0.03, "{name}: the last mesh moved only {moved}");
    }
}

#[test]
fn adapt_problems_that_cannot_run_are_exit_2_and_name_adapt() {
    let cases: [(&str, Edit); 8] = [
        ("every_zero", Box::new(|p| p["adapt"] = json!({"every": 0}))),
        ("every_missing", Box::new(|p| p["adapt"] = json!({}))),
        (
            "every_fraction",
            Box::new(|p| p["adapt"] = json!({"every": 2.5})),
        ),
        (
            "extra_key",
            Box::new(|p| p["adapt"] = json!({"every": 2, "tol": 1})),
        ),
        ("not_an_object", Box::new(|p| p["adapt"] = json!(3))),
        (
            "without_branch",
            Box::new(|p| {
                p.as_object_mut().unwrap().remove("branch");
            }),
        ),
        ("branch_false", Box::new(|p| p["branch"] = json!(false))),
        ("one_interval", Box::new(|p| p["ntst"] = json!(1))),
    ];
    for (name, edit) in cases {
        let mut p = mu_line_adapt(2, None);
        edit(&mut p);
        let (status, out, err) = dde_cycle(&format!("adapt_bad_{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(out.is_empty(), "{name}: {out}");
        assert!(err.contains("adapt"), "{name}: {err}");
    }
}

#[test]
fn a_bad_adapt_with_a_hopf_start_is_exit_2_before_the_run() {
    // `max_newton` 1 keeps the Hopf correction from converging, so a check that came after
    // it would give exit 1.
    for (name, adapt) in [
        ("zero", json!({"every": 0})),
        ("missing", json!({})),
        ("fraction", json!({"every": 1.5})),
    ] {
        let mut p = hopf_branch_problem();
        p["adapt"] = adapt;
        p["options"] = json!({"max_newton": 1});
        let (status, out, err) = dde_cycle(&format!("hopf_bad_adapt_{name}.json"), &p);
        assert_eq!(status, 2, "{name}: {err}");
        assert!(out.is_empty(), "{name}: {out}");
        assert!(err.contains("adapt"), "{name}: {err}");
    }
    // The same start with a good adapt does not converge, which is exit 1.
    let mut p = hopf_branch_problem();
    p["adapt"] = json!({"every": 2});
    p["options"] = json!({"max_newton": 1});
    let (status, _, err) = dde_cycle("hopf_adapt_no_converge.json", &p);
    assert_eq!(status, 1, "{err}");
}

#[test]
fn the_mesh_adapt_option_is_still_refused_and_points_to_adapt() {
    let mut p = mu_line_problem();
    p["options"]["mesh_adapt"] = json!(3);
    let (status, out, err) = dde_cycle("adapt_mesh_adapt.json", &p);
    assert_eq!(status, 2, "{err}");
    assert!(out.is_empty(), "{out}");
    assert!(err.contains("options.mesh_adapt"), "{err}");
    assert!(err.contains("\"adapt\""), "{err}");
}

#[test]
fn help_lists_the_adapt_key() {
    let o = run(&scratch(), &["dde-cycle", "--help"]);
    assert_eq!(code(&o), 0);
    let text = stdout(&o);
    for word in ["adapt", "every", "mesh"] {
        assert!(text.contains(word), "{word} missing from the help text");
    }
}
