//! The CUDA kernel with its step arrays in global memory, on a real device.
//!
//! A machine with no CUDA device skips these tests with a `SKIPPED` line, and
//! `PC_REQUIRE_GPU=cuda` turns the skip into a failure. Both tests run in one function
//! because `force_layout` is a process-wide switch and tests in one binary run in parallel.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::cuda::{force_layout, layout_for};
use pc_ensemble::device::{Backend, BackendState, find_device, list_devices};
use pc_ensemble::kernel::{Integrator, Layout};
use pc_ensemble::run::run_on;
use pc_ensemble::selftest::{Selected, run_selftest};
use pc_ensemble::{Precision, Status};

/// Puts the layout choice back to the size rule, also when a test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_layout(None);
    }
}

#[test]
fn the_global_layout_runs_on_a_cuda_device() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    eprintln!("cuda device 0: {} ({})", device.name, device.driver);
    let _reset = Reset;

    // The self-test under a forced global layout: the bits must still match the CPU.
    force_layout(Some(Layout::Global));
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    }
    for p in precisions {
        let r = run_selftest(Selected {
            backend: Backend::Cuda,
            device: 0,
            precision: p,
        })
        .unwrap_or_else(|e| panic!("global {} self-test: {e}", p.name()));
        assert!(
            r.pass,
            "global {}: bit_exact {:?}, tiers {:#?}",
            p.name(),
            r.bit_exact,
            r.tiers
        );
    }
    force_layout(None);

    // A model whose private arrays are over the launch limit (480 KB in double under
    // Dormand-Prince): the size rule must pick the global layout, and the result must match
    // the CPU bit for bit.
    if !device.fp64 {
        eprintln!("SKIPPED: the big-n run needs double precision");
        return;
    }
    let n = 6000;
    assert_eq!(layout_for(n, Integrator::Dp5, 8), Layout::Global);
    let src = lorenz96(n);
    let m = build_model(&src);
    let members = 4;
    let x0: Vec<f64> = (0..members * n)
        .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
        .collect();
    let j = job(&m, members, &x0, &[8.0], 0.02, DP5_STD, Precision::Double);
    let want = run(&j);
    assert!(want.status.iter().all(|s| *s == Status::Ok));
    let got = run_on(&device, &j, &AtomicBool::new(false)).expect("the big-n run on the device");
    assert_eq!(got.label.backend, "cuda");
    assert_eq!(bits(&want.states), bits(&got.states), "states");
    assert_eq!(want.status, got.status, "status");
    assert_eq!(want.accepted, got.accepted, "accepted");
    assert_eq!(want.rejected, got.rejected, "rejected");
}
