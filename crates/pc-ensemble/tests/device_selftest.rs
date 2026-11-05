//! The self-test on a real device of each GPU backend, when the machine has one.
//!
//! This is the test that runs kernels. A machine with no device for a backend skips that
//! backend with a `SKIPPED` line, and `PC_REQUIRE_GPU=<backend>` turns the skip into a
//! failure, so a green run cannot hide that nothing ran. `PC_REQUIRE_FP64=<backend>` fails
//! a device that reports no double precision. The PoCL job sets both for `opencl`.
//!
//! The macOS runs are ignored by default. Apple's OpenCL and Metal have not run on a Mac
//! yet, so the blocking `cargo test` job leaves them out, and the non-blocking macOS GPU job
//! runs them with `--include-ignored`.

mod common;

use common::*;
use pc_ensemble::Precision;
use pc_ensemble::device::{Backend, BackendState, find_device, list_devices};
use pc_ensemble::selftest::{Selected, run_selftest};

/// True when the comma list in `var` names `backend`.
fn named_in(var: &str, backend: &str) -> bool {
    std::env::var(var)
        .map(|v| v.split(',').any(|b| b.trim() == backend))
        .unwrap_or(false)
}

/// Runs the self-test on device 0 of `backend`, in every precision the device supports.
fn check(backend: Backend) {
    let name = backend.name();
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == backend)
        .map(|(_, s)| s.clone())
        .expect("the backend is listed");
    let device = match state {
        BackendState::Available => {
            find_device(backend, 0).expect("an available backend has device 0")
        }
        BackendState::Unavailable(why) => return skip_gpu(name, &why),
        BackendState::NotBuilt => return skip_gpu(name, "not built into this binary"),
    };
    eprintln!("{name} device 0: {} ({})", device.name, device.driver);

    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    } else if named_in("PC_REQUIRE_FP64", name) {
        panic!(
            "PC_REQUIRE_FP64 names {name}, but {} reports no double precision",
            device.name
        );
    }
    for p in precisions {
        let r = run_selftest(Selected {
            backend,
            device: 0,
            precision: p,
        })
        .unwrap_or_else(|e| panic!("{name} {} self-test: {e}", p.name()));
        assert!(
            r.pass,
            "{name} {} on {}: bit_exact {:?}, tiers {:#?}",
            p.name(),
            device.name,
            r.bit_exact,
            r.tiers
        );
        assert_eq!(r.label.backend, name);
        assert_eq!(r.label.precision, p);
    }
}

#[test]
fn the_cuda_device_passes_the_selftest() {
    check(Backend::Cuda);
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Apple's OpenCL has not run yet; the macOS GPU job runs it"
)]
fn the_opencl_device_passes_the_selftest() {
    check(Backend::OpenCl);
}

#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "Metal has not run on a Mac yet; the macOS GPU job runs it"
)]
fn the_metal_device_passes_the_selftest() {
    check(Backend::Metal);
}
