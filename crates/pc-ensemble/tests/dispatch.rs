//! Finding a device and running a job on it.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
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

// Metal is built on macOS only. Every other backend is built everywhere.
#[cfg(not(target_os = "macos"))]
#[test]
fn a_backend_that_is_not_built_is_named() {
    let e = find_device(Backend::Metal, 0).unwrap_err();
    assert!(e.contains("metal"), "{e}");
    assert!(e.contains("not built"), "{e}");
}

#[test]
fn the_cuda_lookup_agrees_with_the_backend_state() {
    // This holds on a machine with a GPU and on one without.
    let list = list_devices();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let count = list
        .devices
        .iter()
        .filter(|d| d.backend == Backend::Cuda)
        .count();
    match state {
        BackendState::Available => {
            assert!(count > 0);
            let d = find_device(Backend::Cuda, 0).expect("device 0");
            assert!(d.fp32 && d.fp64);
            assert!(d.driver.contains("NVRTC"), "{}", d.driver);
            let e = find_device(Backend::Cuda, count).unwrap_err();
            assert!(e.contains(&format!("lists {count}")), "{e}");
        }
        BackendState::Unavailable(why) => {
            assert_eq!(count, 0);
            let e = find_device(Backend::Cuda, 0).unwrap_err();
            assert!(e.contains("cannot start") && e.contains(&why), "{e}");
            skip_gpu("cuda", &why);
        }
        BackendState::NotBuilt => panic!("the cuda backend is built into this binary"),
    }
}

#[test]
fn the_opencl_lookup_agrees_with_the_backend_state() {
    // This holds on a machine with an OpenCL driver and on one without.
    let list = list_devices();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::OpenCl)
        .map(|(_, s)| s.clone())
        .expect("opencl is listed");
    let count = list
        .devices
        .iter()
        .filter(|d| d.backend == Backend::OpenCl)
        .count();
    match state {
        BackendState::Available => {
            assert!(count > 0);
            let d = find_device(Backend::OpenCl, 0).expect("device 0");
            assert!(d.fp32 && !d.name.is_empty());
            let e = find_device(Backend::OpenCl, count).unwrap_err();
            assert!(e.contains(&format!("lists {count}")), "{e}");
        }
        BackendState::Unavailable(why) => {
            assert_eq!(count, 0);
            let e = find_device(Backend::OpenCl, 0).unwrap_err();
            assert!(e.contains("cannot start") && e.contains(&why), "{e}");
            skip_gpu("opencl", &why);
        }
        BackendState::NotBuilt => panic!("the opencl backend is built into this binary"),
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

#[cfg(not(target_os = "macos"))]
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
    let e = run_on(&fake(Backend::Metal, 0), &j, &AtomicBool::new(false)).unwrap_err();
    let EnsembleError::Unsupported(text) = e else {
        panic!("expected Unsupported, got {e:?}");
    };
    assert!(text.contains("metal"), "{text}");
}

#[test]
fn run_on_an_opencl_device_that_is_not_there_is_unavailable() {
    // Device 9999 does not exist on any machine, and a machine with no driver has none at all.
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
    let e = run_on(&fake(Backend::OpenCl, 9999), &j, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(e, EnsembleError::GpuUnavailable(_)), "{e:?}");
}

#[test]
fn the_job_is_checked_before_the_opencl_library_is_touched() {
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
    let got = run_on(&fake(Backend::OpenCl, 0), &j, &AtomicBool::new(false)).unwrap_err();
    assert_eq!(got, want);
}

#[test]
fn run_on_a_cuda_device_that_is_not_there_is_unavailable() {
    // Device 9999 does not exist on any machine, and a machine with no driver has none at all.
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
    let e = run_on(&fake(Backend::Cuda, 9999), &j, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(e, EnsembleError::GpuUnavailable(_)), "{e:?}");
}

#[test]
fn the_job_is_checked_before_the_cuda_library_is_touched() {
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
    let got = run_on(&fake(Backend::Cuda, 0), &j, &AtomicBool::new(false)).unwrap_err();
    assert_eq!(got, want);
}

fn fake(backend: Backend, index: usize) -> DeviceInfo {
    DeviceInfo {
        backend,
        index,
        name: "fake".into(),
        driver: "0".into(),
        fp32: true,
        fp64: true,
        memory_bytes: None,
    }
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
