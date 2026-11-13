//! Member batches on a real CUDA device.
//!
//! The same job runs whole and in batches of 1, 7 and 64 members, and every result must match
//! the whole run bit for bit: a member does not depend on the others. A machine with no CUDA
//! device skips the test with a `SKIPPED` line, and `PC_REQUIRE_GPU=cuda` turns the skip into
//! a failure. There is one test function because `force_members` and `force_layout` are
//! process-wide switches and tests in one binary run in parallel.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_ensemble::batch::force_members;
use pc_ensemble::cuda::force_layout;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::kernel::Layout;
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleResult, Job, Method, Output, Precision};

/// Lifts both switches, also when a test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_members(None);
        force_layout(None);
    }
}

fn x0s(members: usize) -> Vec<f64> {
    (0..members)
        .flat_map(|i| [0.5 + 0.37 * i as f64, 0.1 * i as f64 - 0.4])
        .collect()
}

fn same(a: &EnsembleResult, b: &EnsembleResult, what: &str) {
    assert_eq!(a.members, b.members, "{what}");
    assert_eq!(bits(&a.times), bits(&b.times), "{what}: times");
    assert_eq!(bits(&a.states), bits(&b.states), "{what}: states");
    assert_eq!(a.status, b.status, "{what}: status");
    assert_eq!(a.accepted, b.accepted, "{what}: accepted");
    assert_eq!(a.rejected, b.rejected, "{what}: rejected");
    assert_eq!(bits(&a.t_reached), bits(&b.t_reached), "{what}: t_reached");
    assert_eq!(a.label, b.label, "{what}: label");
}

fn gpu(device: &DeviceInfo, job: &Job) -> EnsembleResult {
    run_on(device, job, &AtomicBool::new(false)).expect("the device run")
}

/// Runs `job` whole and in batches of 1, 7 and 64, and compares each with the whole run.
fn batches_match(device: &DeviceInfo, job: &Job, what: &str) {
    force_members(None);
    let whole = gpu(device, job);
    assert_eq!(whole.label.backend, "cuda");
    for cap in [1, 7, 64] {
        force_members(Some(cap));
        let cut = gpu(device, job);
        force_members(None);
        same(&cut, &whole, &format!("{what}, batches of {cap}"));
    }
}

#[test]
fn batches_run_on_a_cuda_device_and_give_the_bits_of_one_run() {
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

    // 70 members cut into 70 batches of 1, 10 of 7 and 2 of 35.
    let members = 70;
    let m = build_model(VDP);
    let x0 = x0s(members);
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    }
    for precision in precisions {
        let dp5 = match precision {
            Precision::Single => DP5_SINGLE,
            Precision::Double => DP5_STD,
        };
        for method in [Method::Rk4 { dt: 0.013 }, dp5] {
            let mut j = job(&m, members, &x0, &[1.5], 6.0, method, precision);
            j.output = Output::Samples(vec![1.0, 3.0, 6.0]);
            batches_match(&device, &j, &format!("{precision:?} {method:?}"));
        }
    }

    // Parameters that differ per member follow their members into the batches.
    let mu: Vec<f64> = (0..members).map(|i| 0.2 + 0.05 * i as f64).collect();
    let mut j = job(
        &m,
        members,
        &x0,
        &[0.0],
        5.0,
        Method::Rk4 { dt: 0.01 },
        Precision::Single,
    );
    j.params = pc_ensemble::Params::PerMember(&mu);
    batches_match(&device, &j, "per-member parameters");

    if !device.fp64 {
        eprintln!("SKIPPED: the global layout runs need double precision");
        return;
    }

    // The same under the global scratch layout, whose `work` buffer is the large one, and
    // under a forced private layout on the same job.
    let n = 40;
    let big = build_model(&lorenz96(n));
    let big_x0: Vec<f64> = (0..members * n)
        .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
        .collect();
    for layout in [Layout::Global, Layout::Private] {
        force_layout(Some(layout));
        for method in [Method::Rk4 { dt: 0.002 }, DP5_STD] {
            let j = job(
                &big,
                members,
                &big_x0,
                &[8.0],
                0.2,
                method,
                Precision::Double,
            );
            batches_match(&device, &j, &format!("{layout:?} {method:?} n = {n}"));
        }
        force_layout(None);
    }

    oversized_job_runs_in_batches(&device);
}

/// A job whose buffers together are 10% larger than the card's memory cannot run as one
/// batch, so only the memory budget can make it run. 25 members of the front and of the back
/// of the job are checked against the CPU: fixed-step RK4 has to agree with it bit for bit.
fn oversized_job_runs_in_batches(device: &DeviceInfo) {
    let Some(memory) = device.memory_bytes else {
        return eprintln!("SKIPPED: the device does not report its memory");
    };
    let n = 5000;
    let big = build_model(&lorenz96(n));
    // Per member: `work` 8n + 1 reals under the global layout, `x0` and one sample, n each.
    let per_member = (10 * n as u64 + 1) * 8;
    let members = (memory / per_member * 11 / 10 + 1) as usize;
    let x0: Vec<f64> = (0..members * n)
        .map(|k| 8.0 + 0.001 * ((k % n) as f64) + 1e-7 * (k / n) as f64)
        .collect();
    let method = Method::Rk4 { dt: 0.001 };
    let j = job(&big, members, &x0, &[8.0], 0.004, method, Precision::Double);
    eprintln!("oversized job: {members} members of {n} states, {per_member} bytes each");
    force_members(None);
    let got = gpu(device, &j);
    assert_eq!(got.label.backend, "cuda");
    assert!(got.status.iter().all(|s| *s == pc_ensemble::Status::Ok));
    let edge = 25.min(members);
    for first in [0, members - edge] {
        let part = &x0[first * n..(first + edge) * n];
        let want = run(&job(
            &big,
            edge,
            part,
            &[8.0],
            0.004,
            method,
            Precision::Double,
        ));
        assert_eq!(
            bits(&want.states),
            bits(&got.states[first * n..(first + edge) * n]),
            "members {first}.. of the oversized job"
        );
    }
}
