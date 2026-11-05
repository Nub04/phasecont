//! Finding a device and running a job on it.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::device::{Backend, DeviceInfo, find_device};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleError, Method, Precision, run_cpu};

#[test]
fn the_cpu_device_is_found() {
    let d = find_device(Backend::Cpu, 0).expect("cpu");
    assert_eq!(d.backend, Backend::Cpu);
    assert_eq!(d.index, 0);
}

#[test]
fn a_device_index_past_the_list_is_named() {
    let e = find_device(Backend::Cpu, 3).unwrap_err();
    assert!(e.contains("cpu") && e.contains('3'), "{e}");
    assert!(e.contains("lists 1"), "{e}");
}

#[test]
fn a_backend_that_is_not_built_is_named() {
    for b in [Backend::Cuda, Backend::OpenCl, Backend::Metal] {
        let e = find_device(b, 0).unwrap_err();
        assert!(e.contains(b.name()), "{e}");
        assert!(e.contains("not built"), "{e}");
    }
}

#[test]
fn run_on_the_cpu_matches_run_cpu_bit_for_bit() {
    let m = build_model(VDP);
    let x0 = [1.0, 0.0, 0.5, 0.5];
    for p in [Precision::Single, Precision::Double] {
        let j = job(&m, 2, &x0, &[1.5], 3.0, Method::Rk4 { dt: 0.01 }, p);
        let want = run(&j);
        let dev = find_device(Backend::Cpu, 0).unwrap();
        let got = run_on(&dev, &j, &AtomicBool::new(false)).unwrap();
        assert_eq!(bits(&got.states), bits(&want.states));
        assert_eq!(got.label.backend, "cpu");
        assert_eq!(got.label.precision, p);
    }
}

#[test]
fn run_on_a_device_of_an_unbuilt_backend_is_an_error() {
    let m = build_model(DECAY);
    let j = job(
        &m,
        1,
        &[1.0],
        &[1.0],
        1.0,
        Method::Rk4 { dt: 0.01 },
        Precision::Double,
    );
    let fake = DeviceInfo {
        backend: Backend::Cuda,
        index: 0,
        name: "fake".into(),
        driver: "0".into(),
        fp32: true,
        fp64: true,
        memory_bytes: None,
    };
    let e = run_on(&fake, &j, &AtomicBool::new(false)).unwrap_err();
    let EnsembleError::Unsupported(text) = e else {
        panic!("expected Unsupported, got {e:?}");
    };
    assert!(text.contains("cuda"), "{text}");
}

#[test]
fn run_on_checks_the_job_before_the_backend() {
    // A job the CPU rejects is rejected the same way through run_on.
    let m = build_model(DECAY);
    let j = job(
        &m,
        1,
        &[1.0, 2.0],
        &[1.0],
        1.0,
        Method::Rk4 { dt: 0.01 },
        Precision::Double,
    );
    let want = run_cpu(&j, u64::MAX, &AtomicBool::new(false)).unwrap_err();
    let dev = find_device(Backend::Cpu, 0).unwrap();
    let got = run_on(&dev, &j, &AtomicBool::new(false)).unwrap_err();
    assert_eq!(got, want);
}
