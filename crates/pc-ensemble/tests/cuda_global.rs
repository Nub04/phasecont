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
use pc_ensemble::{Method, Precision, Status};

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

    if !device.fp64 {
        eprintln!("SKIPPED: the big-n runs need double precision");
        return;
    }

    // The same job under the size rule (global) and under a forced private layout, at sizes
    // where private still launches: the two must agree bit for bit in the state, the status
    // and the step counts. Fixed-step RK4 uses only `+ - *`, so it must also match the CPU bit
    // for bit. Dormand-Prince takes powers for its step size, and the CUDA `pow` is not
    // promised to match the CPU's, so against the CPU it only has to agree to a tolerance.
    let rk4 = Method::Rk4 { dt: 0.001 };
    for (name, n, integrator, method) in [
        ("rk4", 5000, Integrator::Rk4, rk4),
        ("dp5", 3500, Integrator::Dp5, DP5_STD),
    ] {
        assert_eq!(
            layout_for(n, integrator, 8),
            Layout::Global,
            "{name} n = {n}"
        );
        let (j_x0, m) = (big_x0(n, 4), build_model(&lorenz96(n)));
        let j = job(&m, 4, &j_x0, &[8.0], 0.02, method, Precision::Double);
        let want = run(&j);
        let global = run_on(&device, &j, &AtomicBool::new(false)).expect("the global run");
        force_layout(Some(Layout::Private));
        let private = run_on(&device, &j, &AtomicBool::new(false)).expect("the private run");
        force_layout(None);
        assert_eq!(global.label.backend, "cuda");
        assert_eq!(bits(&global.states), bits(&private.states), "{name} states");
        assert_eq!(global.status, private.status, "{name} status");
        assert_eq!(global.accepted, private.accepted, "{name} accepted");
        assert_eq!(global.rejected, private.rejected, "{name} rejected");
        assert!(want.status.iter().all(|s| *s == Status::Ok));
        assert_eq!(want.status, global.status, "{name} status against the CPU");
        if integrator == Integrator::Rk4 {
            assert_eq!(
                bits(&want.states),
                bits(&global.states),
                "{name} states against the CPU"
            );
        } else {
            close(&want.states, &global.states, 1e-7, name);
        }
    }

    // A model whose private arrays are over the launch limit (480 KB in double under
    // Dormand-Prince): the size rule must pick the global layout, and the run must finish
    // with the same status as the CPU and a state that agrees to the tolerance.
    let n = 6000;
    assert_eq!(layout_for(n, Integrator::Dp5, 8), Layout::Global);
    let m = build_model(&lorenz96(n));
    let x0 = big_x0(n, 4);
    let j = job(&m, 4, &x0, &[8.0], 0.02, DP5_STD, Precision::Double);
    let want = run(&j);
    assert!(want.status.iter().all(|s| *s == Status::Ok));
    let got = run_on(&device, &j, &AtomicBool::new(false)).expect("the big-n run on the device");
    assert_eq!(got.label.backend, "cuda");
    assert_eq!(want.status, got.status, "status");
    close(&want.states, &got.states, 1e-7, "dp5 n = 6000");
}

/// Initial states of `members` members of an `n`-state model.
fn big_x0(n: usize, members: usize) -> Vec<f64> {
    (0..members * n)
        .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
        .collect()
}

/// Every value of `got` is within `rel` of `want`, relative to `1 + |want|`.
fn close(want: &[f64], got: &[f64], rel: f64, what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    for (k, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(
            (w - g).abs() <= rel * (1.0 + w.abs()),
            "{what}: value {k} is {g:e} on the device and {w:e} on the CPU"
        );
    }
}
