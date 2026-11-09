//! The kernel cache of the CUDA and OpenCL backends, on a real device when the machine has
//! one. A machine with no device for a backend skips it with a `SKIPPED` line, and
//! `PC_REQUIRE_GPU=<backend>` turns the skip into a failure.
//!
//! The tests of one backend share a process-wide cache and a compile counter, so each
//! backend runs its steps inside a single test.

mod common;

use common::*;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleError, EnsembleResult, Method, Precision};
use std::sync::atomic::AtomicBool;

/// The device to test, or `None` after a skip.
fn device_of(backend: Backend) -> Option<DeviceInfo> {
    let name = backend.name();
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == backend)
        .map(|(_, s)| s.clone())
        .expect("the backend is listed");
    match state {
        BackendState::Available => Some(find_device(backend, 0).expect("device 0")),
        BackendState::Unavailable(why) => {
            skip_gpu(name, &why);
            None
        }
        BackendState::NotBuilt => {
            skip_gpu(name, "not built into this binary");
            None
        }
    }
}

/// How many kernels this process has compiled for `backend`.
fn compiles(backend: Backend) -> usize {
    match backend {
        Backend::Cuda => pc_ensemble::cuda::kernel_compiles(),
        Backend::OpenCl => pc_ensemble::opencl::kernel_compiles(),
        _ => unreachable!("only CUDA and OpenCL cache kernels"),
    }
}

/// How many kernels the kept entry of device `index` holds.
fn kept(backend: Backend, index: usize) -> usize {
    match backend {
        Backend::Cuda => pc_ensemble::cuda::kept_kernels(index),
        Backend::OpenCl => pc_ensemble::opencl::kept_kernels(index),
        _ => unreachable!("only CUDA and OpenCL cache kernels"),
    }
}

fn try_job(
    device: &DeviceInfo,
    src: &str,
    method: Method,
    precision: Precision,
) -> Result<EnsembleResult, EnsembleError> {
    try_job_with(device, src, method, precision, &NO_CANCEL())
}

fn try_job_with(
    device: &DeviceInfo,
    src: &str,
    method: Method,
    precision: Precision,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let model = build_model(src);
    let members = 64;
    let x0 = vec![0.5; members * model.n()];
    let params = vec![1.0; model.n_p()];
    let j = job(&model, members, &x0, &params, 1.0, method, precision);
    run_on(device, &j, cancel)
}

fn run_job(device: &DeviceInfo, src: &str, method: Method, precision: Precision) -> EnsembleResult {
    try_job(device, src, method, precision).expect("run")
}

fn steps(backend: Backend) {
    let Some(device) = device_of(backend) else {
        return;
    };
    let precision = if device.fp64 {
        Precision::Double
    } else {
        Precision::Single
    };
    let method = if precision == Precision::Double {
        DP5_STD
    } else {
        DP5_SINGLE
    };

    // The same kernel twice compiles once, and the results do not change.
    let base = compiles(backend);
    let first = run_job(&device, DECAY, method, precision);
    assert_eq!(compiles(backend), base + 1, "the first run compiles");
    let second = run_job(&device, DECAY, method, precision);
    assert_eq!(compiles(backend), base + 1, "the second run reuses it");
    assert_eq!(
        first.states, second.states,
        "a cached kernel gives the same states"
    );
    assert_eq!(kept(backend, 0), 1);

    // A different model is a different kernel.
    run_job(&device, OSC, method, precision);
    assert_eq!(compiles(backend), base + 2, "a new model compiles");
    run_job(&device, DECAY, method, precision);
    assert_eq!(
        compiles(backend),
        base + 2,
        "the first kernel is still kept"
    );

    // The tolerances are arguments of the launch, not part of the kernel text.
    let looser = Method::Dp5 {
        rtol: 1e-3,
        atol: 1e-5,
        hmax: f64::INFINITY,
    };
    run_job(&device, DECAY, looser, precision);
    assert_eq!(
        compiles(backend),
        base + 2,
        "a new tolerance reuses the kernel"
    );

    // A different precision is a different kernel, even for the same model.
    if device.fp64 {
        run_job(&device, DECAY, DP5_SINGLE, Precision::Single);
        assert_eq!(compiles(backend), base + 3, "a new precision compiles");
        run_job(&device, DECAY, DP5_SINGLE, Precision::Single);
        assert_eq!(compiles(backend), base + 3, "and is then kept");
    }

    // Past the bound the oldest kernel goes, and the count stays at the bound.
    let models = [DECAY, OSC, LOGISTIC, FORCED, VDP, MIXED];
    let methods = [DP5_SINGLE, Method::Rk4 { dt: 0.01 }];
    for m in models {
        for meth in methods {
            run_job(&device, m, meth, Precision::Single);
        }
    }
    assert_eq!(kept(backend, 0), 8, "the cache holds at most 8 kernels");

    cancel_keeps_the_entry(&device, backend);

    #[cfg(feature = "fault-injection")]
    {
        eviction(&device, backend);
        panic_eviction(&device, backend);
    }
}

/// A run that starts with the cancel flag set ends with `Cancelled`. That is not a device
/// error, so the entry stays, and the next run reuses it.
fn cancel_keeps_the_entry(device: &DeviceInfo, backend: Backend) {
    let (method, precision) = if device.fp64 {
        (DP5_STD, Precision::Double)
    } else {
        (DP5_SINGLE, Precision::Single)
    };
    run_job(device, DECAY, method, precision);
    let held = kept(backend, 0);
    assert!(held >= 1, "the run keeps its entry");
    let warm = compiles(backend);
    let flag = AtomicBool::new(true);
    let err = try_job_with(device, DECAY, method, precision, &flag).expect_err("cancelled");
    assert_eq!(err, EnsembleError::Cancelled);
    assert_eq!(kept(backend, 0), held, "a cancel does not remove the entry");
    assert_eq!(compiles(backend), warm, "and reuses the kernel");
    run_job(device, DECAY, method, precision);
    assert_eq!(compiles(backend), warm, "the next run reuses it too");
}

#[test]
fn the_cuda_backend_compiles_a_kernel_once() {
    steps(Backend::Cuda);
}

#[test]
fn the_opencl_backend_compiles_a_kernel_once() {
    steps(Backend::OpenCl);
}

/// A launch the driver rejects ends the run with `gpuLaunch`, the device then keeps nothing,
/// and the next run builds a new context and compiles again. It runs as the last step of
/// `steps`, because the tests of one backend share one cache.
#[cfg(feature = "fault-injection")]
fn eviction(device: &DeviceInfo, backend: Backend) {
    let (method, precision) = if device.fp64 {
        (DP5_STD, Precision::Double)
    } else {
        (DP5_SINGLE, Precision::Single)
    };

    let before = run_job(device, DECAY, method, precision);
    assert!(kept(backend, 0) >= 1, "the run keeps its entry");
    let warm = compiles(backend);

    pc_ensemble::fault::arm_launch_fault();
    let err = try_job(device, DECAY, method, precision).expect_err("the armed launch fails");
    assert!(
        matches!(&err, EnsembleError::GpuLaunch(m) if m.contains("launch")),
        "{err:?}"
    );
    assert_eq!(kept(backend, 0), 0, "the failed run removed the entry");
    assert_eq!(compiles(backend), warm, "the failed run reused the kernel");

    // The flag is one-shot: this run is not armed.
    let after = run_job(device, DECAY, method, precision);
    assert_eq!(compiles(backend), warm + 1, "a new context compiles again");
    assert_eq!(kept(backend, 0), 1, "and keeps its entry");
    assert_eq!(
        before.states, after.states,
        "the new context gives the same states"
    );
}

/// A panic inside the library call ends the run with `gpuLaunch` and removes the entry, as a
/// launch error does. Nothing reaches the caller as a panic.
#[cfg(feature = "fault-injection")]
fn panic_eviction(device: &DeviceInfo, backend: Backend) {
    let (method, precision) = if device.fp64 {
        (DP5_STD, Precision::Double)
    } else {
        (DP5_SINGLE, Precision::Single)
    };
    let before = run_job(device, DECAY, method, precision);
    assert!(kept(backend, 0) >= 1, "the run keeps its entry");
    let warm = compiles(backend);

    pc_ensemble::fault::arm_panic_fault();
    let err = try_job(device, DECAY, method, precision).expect_err("the armed panic fails");
    let library = match backend {
        Backend::Cuda => "panic in cudarc",
        _ => "panic in opencl3",
    };
    assert!(
        matches!(&err, EnsembleError::GpuLaunch(m) if m.contains(library) && m.contains("injected")),
        "{err:?}"
    );
    assert_eq!(kept(backend, 0), 0, "the panicked run removed the entry");
    assert_eq!(
        compiles(backend),
        warm,
        "the panicked run reused the kernel"
    );

    let after = run_job(device, DECAY, method, precision);
    assert_eq!(compiles(backend), warm + 1, "a new context compiles again");
    assert_eq!(kept(backend, 0), 1, "and keeps its entry");
    assert_eq!(
        before.states, after.states,
        "the new context gives the same states"
    );
}
