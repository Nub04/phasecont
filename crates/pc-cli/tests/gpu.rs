//! Runs `pc-cli gpu list` and `pc-cli gpu selftest`. Only the CPU backend exists in this
//! build, so every other backend must report that it has no device.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|e| panic!("not JSON ({e}): {text}"))
}

#[test]
fn list_names_every_backend_and_the_cpu_device() {
    let o = run(&scratch(), &["gpu", "list"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v = json(&stdout(&o));
    let backends = v["backends"].as_array().unwrap();
    let names: Vec<&str> = backends
        .iter()
        .map(|b| b["backend"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["cpu", "cuda", "opencl", "metal"]);
    assert_eq!(backends[0]["state"], "available");
    for b in &backends[1..] {
        assert_eq!(b["state"], "not_built", "{b}");
    }
    let d = &v["devices"][0];
    assert_eq!(d["backend"], "cpu");
    assert_eq!(d["index"], 0);
    assert_eq!(d["fp32"], true);
    assert_eq!(d["fp64"], true);
    assert!(d["name"].as_str().unwrap().contains("rayon"), "{d}");
    assert!(d["driver"].as_str().unwrap().starts_with("pc-ensemble"));
    assert!(d["memory_bytes"].is_null());
}

#[test]
fn the_cpu_selftest_passes_in_both_precisions() {
    for p in ["single", "double"] {
        let o = run(
            &scratch(),
            &["gpu", "selftest", "--backend", "cpu", "--precision", p],
        );
        assert_eq!(code(&o), 0, "{p}: {}", stderr(&o));
        let v = json(&stdout(&o));
        assert_eq!(v["pass"], true, "{v}");
        assert_eq!(v["backend"], "cpu");
        assert_eq!(v["precision"], p);
        assert_eq!(v["device"]["index"], 0);
        assert!(v["device"]["name"].as_str().unwrap().contains("rayon"));
        assert!(v["driver"].as_str().unwrap().starts_with("pc-ensemble"));
        assert_eq!(v["features"]["fp32"], true);
        assert_eq!(v["features"]["fp64"], true);
        // The CPU backend is the reference, so it has nothing to compare bits with.
        assert!(v["bit_exact"].is_null(), "{v}");
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        let tiers = v["tiers"].as_array().unwrap();
        let names: Vec<&str> = tiers.iter().map(|t| t["name"].as_str().unwrap()).collect();
        for want in ["analytic-rk4", "analytic-dp5", "determinism"] {
            assert!(names.contains(&want), "{names:?}");
        }
        for t in tiers {
            assert_eq!(t["pass"], true, "{t}");
            assert!(t["max_error"].as_f64().unwrap() <= t["limit"].as_f64().unwrap());
        }
    }
}

#[test]
fn a_backend_that_is_not_built_has_no_device() {
    for b in ["cuda", "opencl", "metal"] {
        let o = run(
            &scratch(),
            &["gpu", "selftest", "--backend", b, "--precision", "single"],
        );
        assert_eq!(code(&o), 3, "{b}");
        assert!(stderr(&o).contains(b), "{}", stderr(&o));
        // A missing device prints no report that could be read as a pass.
        assert_eq!(stdout(&o), "", "{b}");
    }
}

#[test]
fn a_device_index_past_the_list_has_no_device() {
    let o = run(
        &scratch(),
        &[
            "gpu",
            "selftest",
            "--backend",
            "cpu",
            "--device",
            "1",
            "--precision",
            "double",
        ],
    );
    assert_eq!(code(&o), 3);
    assert!(stderr(&o).contains("lists 1"), "{}", stderr(&o));
    assert_eq!(stdout(&o), "");
}

#[test]
fn the_backend_and_the_precision_are_required() {
    let o = run(&scratch(), &["gpu", "selftest", "--precision", "single"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("--backend"), "{}", stderr(&o));
    let o = run(&scratch(), &["gpu", "selftest", "--backend", "cpu"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("--precision"), "{}", stderr(&o));
}

#[test]
fn an_unknown_backend_or_precision_is_a_usage_error() {
    let o = run(
        &scratch(),
        &[
            "gpu",
            "selftest",
            "--backend",
            "hip",
            "--precision",
            "single",
        ],
    );
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("hip"), "{}", stderr(&o));
    let o = run(
        &scratch(),
        &["gpu", "selftest", "--backend", "cpu", "--precision", "half"],
    );
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("half"), "{}", stderr(&o));
}

#[test]
fn help_prints_and_exits_zero() {
    for args in [&["gpu", "--help"][..], &["gpu", "selftest", "-h"]] {
        let o = run(&scratch(), args);
        assert_eq!(code(&o), 0, "{args:?}");
        assert!(stdout(&o).contains("pc-cli gpu"), "{args:?}");
    }
}
