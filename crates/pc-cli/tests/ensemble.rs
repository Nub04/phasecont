//! Runs `pc-cli ensemble` on job files and on bad job files, and `ensemble --bench`.
//!
//! The expected values are closed forms: decay `x' = -a x` gives `x(t) = x0 exp(-a t)`.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::{Value, json};

const DECAY: &str = "par a=1\nx'=-a*x\ndone\n";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("ensemble");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes the model and the job under `name` and returns their paths.
fn files(name: &str, model: &str, job: &Value) -> (String, String) {
    let dir = scratch();
    let m = dir.join(format!("{name}.ode"));
    let j = dir.join(format!("{name}.job.json"));
    std::fs::write(&m, model).unwrap();
    std::fs::write(&j, serde_json::to_string_pretty(job).unwrap()).unwrap();
    (
        m.to_string_lossy().into_owned(),
        j.to_string_lossy().into_owned(),
    )
}

/// A complete job for the decay model: two members, `a = 2`, the state at t = 1.
fn decay_job() -> Value {
    json!({
        "backend": "cpu",
        "precision": "double",
        "t0": 0.0,
        "t1": 1.0,
        "method": {"type": "rk4", "dt": 0.001},
        "x0": [[1.0], [2.0]],
        "params": {"shared": [2.0]},
        "output": "final",
        "bounds": null,
        "max_steps": 1_000_000
    })
}

fn run_job(name: &str, model: &str, job: &Value) -> std::process::Output {
    let (m, j) = files(name, model, job);
    run(&scratch(), &["ensemble", &m, &j])
}

fn ok(name: &str, model: &str, job: &Value) -> Value {
    let o = run_job(name, model, job);
    assert_eq!(code(&o), 0, "{name}: {}", stderr(&o));
    serde_json::from_str(&stdout(&o)).unwrap()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[test]
fn a_decay_ensemble_matches_the_closed_form() {
    let v = ok("decay", DECAY, &decay_job());
    assert_eq!(v["members"], 2);
    assert_eq!(v["n"], 1);
    assert_eq!(v["times"], json!([1.0]));
    assert_eq!(v["status"], json!(["ok", "ok"]));
    let want = (-2.0f64).exp();
    for (m, x0) in [1.0, 2.0].into_iter().enumerate() {
        let got = num(&v["states"][m][0][0]);
        assert!((got - x0 * want).abs() < 1e-9, "member {m}: {got}");
        assert_eq!(num(&v["t_reached"][m]), 1.0);
        assert_eq!(num(&v["accepted"][m]), 1000.0);
        assert_eq!(num(&v["rejected"][m]), 0.0);
    }
    let l = &v["label"];
    assert_eq!(l["backend"], "cpu");
    assert_eq!(l["precision"], "double");
    assert_eq!(l["single_precision"], false);
    assert!(l["device"].as_str().unwrap().contains("rayon"));
    assert!(l["driver"].as_str().unwrap().starts_with("pc-ensemble"));
}

#[test]
fn single_precision_is_labeled_and_less_exact() {
    let mut job = decay_job();
    job["precision"] = json!("single");
    let v = ok("decay_single", DECAY, &job);
    assert_eq!(v["label"]["precision"], "single");
    assert_eq!(v["label"]["single_precision"], true);
    let got = num(&v["states"][0][0][0]);
    let want = (-2.0f64).exp();
    let err = (got - want).abs();
    assert!(err < 1e-5, "{err}");
    assert!(err > 1e-10, "single precision should not reach {err}");
}

#[test]
fn samples_and_per_member_parameters() {
    let mut job = decay_job();
    job["output"] = json!({"samples": [0.5, 1.0]});
    job["params"] = json!({"per_member": [[1.0], [3.0]]});
    let v = ok("decay_samples", DECAY, &job);
    assert_eq!(v["times"], json!([0.5, 1.0]));
    for (m, a) in [1.0f64, 3.0].into_iter().enumerate() {
        let x0 = [1.0, 2.0][m];
        for (s, t) in [0.5f64, 1.0].into_iter().enumerate() {
            let got = num(&v["states"][m][s][0]);
            assert!((got - x0 * (-a * t).exp()).abs() < 1e-9, "{m} {s}: {got}");
        }
    }
}

#[test]
fn the_adaptive_method_runs_with_an_unbounded_step() {
    let mut job = decay_job();
    job["method"] = json!({"type": "dp5", "rtol": 1e-9, "atol": 1e-11, "hmax": null});
    let v = ok("decay_dp5", DECAY, &job);
    let want = 2.0 * (-2.0f64).exp();
    assert!((num(&v["states"][1][0][0]) - want).abs() < 1e-8);
    assert!(num(&v["accepted"][0]) < 100.0);
}

#[test]
fn a_member_that_does_not_finish_is_reported_and_the_run_still_exits_zero() {
    // ln(-1) is NaN: the first member ends nonfinite and its state prints as null.
    let mut job = decay_job();
    job["x0"] = json!([[-1.0], [2.0]]);
    let v = ok("nan", "x'=ln(x)\ndone\n", &{
        let mut j = job;
        j["params"] = json!({"shared": []});
        j
    });
    assert_eq!(v["status"], json!(["nonfinite", "ok"]));
    assert!(v["states"][0][0][0].is_null(), "{}", v["states"]);
    assert!(v["states"][1][0][0].is_number());
}

#[test]
fn a_missing_key_is_named() {
    for key in [
        "backend",
        "precision",
        "t0",
        "t1",
        "method",
        "x0",
        "params",
        "output",
        "bounds",
        "max_steps",
    ] {
        let mut job = decay_job();
        job.as_object_mut().unwrap().remove(key);
        let o = run_job(&format!("missing_{key}"), DECAY, &job);
        assert_eq!(code(&o), 2, "{key}");
        assert!(stderr(&o).contains(key), "{key}: {}", stderr(&o));
        assert_eq!(stdout(&o), "", "{key}");
    }
}

#[test]
fn an_unknown_key_is_named() {
    let mut job = decay_job();
    job["rtol"] = json!(1e-6);
    let o = run_job("unknown_key", DECAY, &job);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("rtol"), "{}", stderr(&o));
}

#[test]
fn wrong_values_name_their_key() {
    let cases: [(&str, Value); 9] = [
        ("precision", json!("half")),
        ("backend", json!("hip")),
        ("t0", json!("zero")),
        ("method", json!({"type": "euler", "dt": 0.1})),
        ("x0", json!([1.0, 2.0])),
        ("x0", json!([[1.0, 2.0]])),
        ("params", json!({"shared": [1.0], "per_member": [[1.0]]})),
        ("output", json!("everything")),
        ("max_steps", json!(-1)),
    ];
    for (i, (key, bad)) in cases.into_iter().enumerate() {
        let mut job = decay_job();
        job[key] = bad;
        let o = run_job(&format!("wrong_{i}"), DECAY, &job);
        assert_eq!(code(&o), 2, "{key}: {}", stderr(&o));
        assert!(stderr(&o).contains(key), "{key}: {}", stderr(&o));
    }
}

#[test]
fn a_job_file_that_is_not_json_is_a_usage_error() {
    let dir = scratch();
    let m = dir.join("notjson.ode");
    let j = dir.join("notjson.job.json");
    std::fs::write(&m, DECAY).unwrap();
    std::fs::write(&j, "{ not json").unwrap();
    let o = run(
        &dir,
        &["ensemble", &m.to_string_lossy(), &j.to_string_lossy()],
    );
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("invalid JSON"), "{}", stderr(&o));
    let o = run(
        &dir,
        &["ensemble", &m.to_string_lossy(), "no-such-file.json"],
    );
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("no-such-file.json"), "{}", stderr(&o));
}

#[test]
fn a_backend_that_cannot_run_has_no_device() {
    // Pick a GPU backend that is not usable in this process: not built, or built but with no
    // driver or device. A machine where every backend works skips this part, loudly.
    let list = pc_ensemble::device::list_devices();
    let down = list.backends.iter().find(|(b, s)| {
        *b != pc_ensemble::device::Backend::Cpu
            && *s != pc_ensemble::device::BackendState::Available
    });
    if let Some((backend, _)) = down {
        let mut job = decay_job();
        job["backend"] = json!(backend.name());
        let o = run_job("down", DECAY, &job);
        assert_eq!(code(&o), 3);
        assert!(stderr(&o).contains(backend.name()), "{}", stderr(&o));
        assert_eq!(stdout(&o), "");
    } else {
        eprintln!("SKIPPED: every GPU backend works here, so none can be missing");
    }
    let mut job = decay_job();
    job["device"] = json!(5);
    let o = run_job("device5", DECAY, &job);
    assert_eq!(code(&o), 3);
}

#[test]
fn the_engine_rejects_a_tolerance_single_precision_cannot_resolve() {
    let mut job = decay_job();
    job["precision"] = json!("single");
    job["method"] = json!({"type": "dp5", "rtol": 1e-12, "atol": 1e-14, "hmax": null});
    let o = run_job("tight", DECAY, &job);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("tolerance too tight"), "{}", stderr(&o));
    assert_eq!(stdout(&o), "");
}

#[test]
fn a_model_that_does_not_load_exits_one() {
    let o = run_job("badmodel", "x'=", &decay_job());
    assert_eq!(code(&o), 1);
}

#[test]
fn the_bench_labels_every_row() {
    let o = run(&scratch(), &["ensemble", "--bench", "--members", "10,20"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert!(v["model"].as_str().unwrap().contains("van der Pol"));
    let rows = v["rows"].as_array().unwrap();
    // The CPU device, both precisions, two sizes.
    assert_eq!(rows.len(), 4, "{v}");
    for r in rows {
        assert_eq!(r["backend"], "cpu");
        assert!(["single", "double"].contains(&r["precision"].as_str().unwrap()));
        assert!([10, 20].contains(&r["members"].as_u64().unwrap()));
        assert!(num(&r["seconds"]) > 0.0);
        assert!(num(&r["member_steps_per_second"]) > 0.0);
        assert!(r["device"].as_str().unwrap().contains("rayon"));
    }
    assert_eq!(v["repeats"], 3);
}

#[test]
fn bad_bench_arguments_are_usage_errors() {
    for args in [
        &["ensemble", "--bench", "--members", "0"][..],
        &["ensemble", "--bench", "--members", "10,x"],
        &["ensemble", "--bench", "--members", ""],
        &["ensemble", "--members", "10"],
        &["ensemble", "--bench", "a.ode"],
        &["ensemble", "a.ode"],
        &["ensemble"],
    ] {
        let o = run(&scratch(), args);
        assert_eq!(code(&o), 2, "{args:?}: {}", stderr(&o));
    }
}
