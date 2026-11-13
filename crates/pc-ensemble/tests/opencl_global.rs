//! The OpenCL kernel with its step arrays in global memory, on every OpenCL device.
//!
//! A machine with no OpenCL device skips the test with a `SKIPPED` line, and
//! `PC_REQUIRE_GPU=opencl` turns the skip into a failure. There is one test function because
//! `force_layout` and `force_members` are process-wide switches and tests in one binary run in
//! parallel.
//!
//! On each device: the self-test under a forced global layout (the bits must still match the
//! CPU), the same job under both layouts and in member batches (the bits must agree), a ring
//! of 2000 states under a forced global layout against the CPU, and a ring above the 256 KiB
//! rule that the size rule runs in the global layout. A device whose compiler cannot build the
//! last one must fail the same way in the private layout.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::batch::force_members;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, list_devices};
use pc_ensemble::gpu::layout_by_size;
use pc_ensemble::kernel::{Integrator, Layout};
use pc_ensemble::opencl::force_layout;
use pc_ensemble::run::run_on;
use pc_ensemble::selftest::{Selected, run_selftest};
use pc_ensemble::{EnsembleError, EnsembleResult, Job, Method, Precision, Status};

/// Puts the layout and the member cap back to their defaults, also when a test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_layout(None);
        force_members(None);
    }
}

fn gpu(device: &DeviceInfo, job: &Job) -> EnsembleResult {
    run_on(device, job, &AtomicBool::new(false)).expect("the device run")
}

fn same(a: &EnsembleResult, b: &EnsembleResult, what: &str) {
    assert_eq!(bits(&a.states), bits(&b.states), "{what}: states");
    assert_eq!(a.status, b.status, "{what}: status");
    assert_eq!(a.accepted, b.accepted, "{what}: accepted");
    assert_eq!(a.rejected, b.rejected, "{what}: rejected");
    assert_eq!(bits(&a.t_reached), bits(&b.t_reached), "{what}: t_reached");
}

/// Initial states of `members` members of an `n`-state model.
fn big_x0(n: usize, members: usize) -> Vec<f64> {
    (0..members * n)
        .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
        .collect()
}

#[test]
fn the_global_layout_runs_on_every_opencl_device() {
    let list = list_devices();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::OpenCl)
        .map(|(_, s)| s.clone())
        .expect("opencl is listed");
    match state {
        BackendState::Available => {}
        BackendState::Unavailable(why) => return skip_gpu("opencl", &why),
        BackendState::NotBuilt => return skip_gpu("opencl", "not built into this binary"),
    }
    let devices: Vec<DeviceInfo> = list
        .devices
        .into_iter()
        .filter(|d| d.backend == Backend::OpenCl)
        .collect();
    assert!(
        !devices.is_empty(),
        "opencl is available and lists no device"
    );
    let _reset = Reset;
    // `PC_OPENCL_DEVICE=1` runs one device, to iterate on it.
    let only = std::env::var("PC_OPENCL_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    for device in devices.iter().filter(|d| only.is_none_or(|i| d.index == i)) {
        eprintln!(
            "opencl device {}: {} ({})",
            device.index, device.name, device.driver
        );
        one_device(device);
    }
}

fn one_device(device: &DeviceInfo) {
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    }

    // The self-test under a forced global layout: the bits must still match the CPU.
    eprintln!("  global self-test");
    force_layout(Some(Layout::Global));
    for &p in &precisions {
        let r = run_selftest(Selected {
            backend: Backend::OpenCl,
            device: device.index,
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

    // The same job in both layouts and in member batches: every result has the bits of the
    // private whole run, because the operations and their order are the same.
    let n = 40;
    let members = 21;
    let m = build_model(&lorenz96(n));
    let x0 = big_x0(n, members);
    for &precision in &precisions {
        let dp5 = match precision {
            Precision::Single => DP5_SINGLE,
            Precision::Double => DP5_STD,
        };
        for method in [Method::Rk4 { dt: 0.002 }, dp5] {
            let j = job(&m, members, &x0, &[8.0], 0.2, method, precision);
            let what = format!("{precision:?} {method:?} n = {n}");
            eprintln!("  {what}");
            force_layout(Some(Layout::Private));
            let private = gpu(device, &j);
            force_layout(Some(Layout::Global));
            let global = gpu(device, &j);
            assert_eq!(global.label.backend, "opencl");
            same(
                &global,
                &private,
                &format!("{what}, global against private"),
            );
            for cap in [1, 7] {
                force_members(Some(cap));
                let cut = gpu(device, &j);
                force_members(None);
                same(&cut, &global, &format!("{what}, global batches of {cap}"));
            }
            force_layout(None);
        }
    }

    if !device.fp64 {
        eprintln!("SKIPPED: the big-n run needs double precision");
        return;
    }

    let method = Method::Rk4 { dt: 0.001 };

    // The global layout against the CPU at a size every device compiles (the size rule picks
    // the private layout here, so the layout is forced). Fixed-step RK4 equals the CPU bit
    // for bit.
    let n = 2000;
    eprintln!("  ring n = {n}, global forced");
    let mid = build_model(&ring(n));
    let x0 = big_x0(n, 4);
    let j = job(&mid, 4, &x0, &[], 0.004, method, Precision::Double);
    let want = run(&j);
    force_layout(Some(Layout::Global));
    let got = gpu(device, &j);
    force_layout(None);
    assert!(
        got.status.iter().all(|s| *s == Status::Ok),
        "{:?}",
        got.status
    );
    assert_eq!(
        bits(&want.states),
        bits(&got.states),
        "rk4 n = {n} against the CPU"
    );

    // A model above the 256 KiB rule: the size rule must pick the global layout. A device
    // that compiles it must finish with every member `ok` and the CPU's bits. The Arc 140T
    // does not compile it: its compiler spills the `n` live values of the straight-line
    // right-hand side past its 262144-byte scratch limit, in either layout. That is accepted
    // only when the private layout fails the same way, so it is not a defect of `Global`.
    let n = 5000;
    eprintln!("  big ring n = {n}");
    assert_eq!(layout_by_size(None, n, Integrator::Rk4, 8), Layout::Global);
    let big = build_model(&ring(n));
    let x0 = big_x0(n, 4);
    let j = job(&big, 4, &x0, &[], 0.004, method, Precision::Double);
    match run_on(device, &j, &AtomicBool::new(false)) {
        Ok(got) => {
            let want = run(&j);
            assert!(
                got.status.iter().all(|s| *s == Status::Ok),
                "{:?}",
                got.status
            );
            assert_eq!(
                bits(&want.states),
                bits(&got.states),
                "rk4 n = {n} against the CPU"
            );
        }
        Err(EnsembleError::GpuCompile(why)) if why.contains("total scratch space exceeds") => {
            force_layout(Some(Layout::Private));
            let private = run_on(device, &j, &AtomicBool::new(false));
            force_layout(None);
            assert!(
                matches!(&private, Err(EnsembleError::GpuCompile(m))
                    if m.contains("total scratch space exceeds")),
                "the global layout hit the compiler's scratch limit and the private layout \
                 did not: {private:?}"
            );
            eprintln!("  n = {n}: both layouts exceed the device compiler's scratch limit");
        }
        Err(e) => panic!("big ring n = {n}: {e}"),
    }
}
