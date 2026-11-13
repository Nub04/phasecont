//! A job split into member batches gives the bits of one run.
//!
//! The CPU backend is the runner here, so the test needs no device. `tests/cuda_batch.rs`
//! does the same on a CUDA device.

mod common;

use common::*;
use pc_ensemble::batch::run_batched;
use pc_ensemble::{
    EnsembleError, EnsembleResult, Job, Method, Output, Params, Precision, Status, run_cpu,
};
use std::sync::atomic::{AtomicBool, Ordering};

fn x0s(members: usize) -> Vec<f64> {
    (0..members)
        .flat_map(|i| [0.5 + 0.37 * i as f64, 0.1 * i as f64 - 0.4])
        .collect()
}

fn same(a: &EnsembleResult, b: &EnsembleResult, what: &str) {
    assert_eq!(a.n, b.n, "{what}");
    assert_eq!(a.members, b.members, "{what}");
    assert_eq!(bits(&a.times), bits(&b.times), "{what}");
    assert_eq!(bits(&a.states), bits(&b.states), "{what}");
    assert_eq!(a.status, b.status, "{what}");
    assert_eq!(a.accepted, b.accepted, "{what}");
    assert_eq!(a.rejected, b.rejected, "{what}");
    assert_eq!(bits(&a.t_reached), bits(&b.t_reached), "{what}");
    assert_eq!(a.label, b.label, "{what}");
}

fn cpu(job: &Job) -> Result<EnsembleResult, EnsembleError> {
    run_cpu(job, 1 << 16, &NO_CANCEL())
}

fn methods(precision: Precision) -> [Method; 2] {
    let dp5 = match precision {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.013 }, dp5]
}

#[test]
fn batches_give_the_bits_of_one_run() {
    let m = build_model(VDP);
    let members = 23;
    let x0 = x0s(members);
    for precision in [Precision::Single, Precision::Double] {
        for method in methods(precision) {
            let mut j = job(&m, members, &x0, &[1.5], 6.0, method, precision);
            j.output = Output::Samples(vec![1.0, 3.0, 6.0]);
            let whole = cpu(&j).unwrap();
            // 1: a batch per member. 4 and 7: uneven last batch. 11: two batches and a
            // remainder of 1. 22: one batch of 22 and a last of 1. 23 and 100: one batch.
            for size in [1, 4, 7, 11, 22, 23, 100] {
                let cut = run_batched(&j, size, &NO_CANCEL(), cpu).unwrap();
                same(
                    &cut,
                    &whole,
                    &format!("{precision:?} {method:?} size {size}"),
                );
            }
        }
    }
}

#[test]
fn per_member_parameters_follow_their_members() {
    let m = build_model(VDP);
    let members = 17;
    let x0 = x0s(members);
    let mu: Vec<f64> = (0..members).map(|i| 0.2 + 0.31 * i as f64).collect();
    for precision in [Precision::Single, Precision::Double] {
        for method in methods(precision) {
            let mut j = job(&m, members, &x0, &[0.0], 5.0, method, precision);
            j.params = Params::PerMember(&mu);
            let whole = cpu(&j).unwrap();
            for size in [1, 3, 5, 16] {
                let cut = run_batched(&j, size, &NO_CANCEL(), cpu).unwrap();
                same(
                    &cut,
                    &whole,
                    &format!("{precision:?} {method:?} size {size}"),
                );
            }
        }
    }
}

#[test]
fn a_mix_of_statuses_comes_back_in_member_order() {
    // x' = x (1 - x) from x0 < 0 blows up in finite time, so those members stop on the bound.
    let m = build_model(LOGISTIC);
    let members = 13;
    let x0: Vec<f64> = (0..members)
        .map(|i| {
            if i % 3 == 0 {
                -1.0
            } else {
                0.1 * (i + 1) as f64
            }
        })
        .collect();
    for method in [Method::Rk4 { dt: 0.01 }, DP5_STD] {
        let mut j = job(&m, members, &x0, &[], 8.0, method, Precision::Double);
        j.bounds = 1e6;
        let whole = cpu(&j).unwrap();
        assert!(whole.status.contains(&Status::Ok), "{method:?}");
        assert!(whole.status.contains(&Status::Bounds), "{method:?}");
        for size in [1, 2, 5, 12] {
            let cut = run_batched(&j, size, &NO_CANCEL(), cpu).unwrap();
            same(&cut, &whole, &format!("{method:?} size {size}"));
        }
    }
}

#[test]
fn each_batch_gets_its_slice_of_the_inputs() {
    let m = build_model(VDP);
    let members = 10;
    let x0 = x0s(members);
    let mu: Vec<f64> = (0..members).map(|i| 1.0 + i as f64).collect();
    let mut j = job(
        &m,
        members,
        &x0,
        &[0.0],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    j.params = Params::PerMember(&mu);
    let mut seen: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
    run_batched(&j, 4, &NO_CANCEL(), |sub| {
        let p = match sub.params {
            Params::PerMember(p) => p.to_vec(),
            Params::Shared(_) => panic!("a per-member job became shared"),
        };
        seen.push((sub.members, sub.x0.to_vec(), p));
        cpu(sub)
    })
    .unwrap();
    // 10 members at most 4 at a time run as 4, 3 and 3.
    let sizes: Vec<usize> = seen.iter().map(|s| s.0).collect();
    assert_eq!(sizes, [4, 3, 3]);
    let x: Vec<f64> = seen.iter().flat_map(|s| s.1.clone()).collect();
    let p: Vec<f64> = seen.iter().flat_map(|s| s.2.clone()).collect();
    assert_eq!(bits(&x), bits(&x0));
    assert_eq!(bits(&p), bits(&mu));

    // A shared parameter set goes to every batch whole.
    let shared = job(
        &m,
        members,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    let mut lens = Vec::new();
    run_batched(&shared, 4, &NO_CANCEL(), |sub| {
        match sub.params {
            Params::Shared(p) => lens.push(p.len()),
            Params::PerMember(_) => panic!("a shared job became per member"),
        }
        cpu(sub)
    })
    .unwrap();
    assert_eq!(lens, [1, 1, 1]);
}

#[test]
fn a_job_that_fits_runs_once_and_unsliced() {
    let m = build_model(VDP);
    let x0 = x0s(6);
    let j = job(
        &m,
        6,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Single,
    );
    for size in [6, 7, usize::MAX] {
        let mut calls = 0;
        run_batched(&j, size, &NO_CANCEL(), |sub| {
            calls += 1;
            assert_eq!(sub.members, 6);
            cpu(sub)
        })
        .unwrap();
        assert_eq!(calls, 1, "size {size}");
    }
}

#[test]
fn a_batch_size_of_zero_is_a_bad_job() {
    let m = build_model(VDP);
    let x0 = x0s(3);
    let j = job(
        &m,
        3,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    match run_batched(&j, 0, &NO_CANCEL(), cpu) {
        Err(EnsembleError::BadJob(t)) => assert!(t.contains("batch"), "{t}"),
        other => panic!("expected BadJob, got {other:?}"),
    }
}

#[test]
fn an_input_of_the_wrong_length_is_a_bad_job_and_not_a_panic() {
    let m = build_model(VDP);
    let x0 = x0s(3);
    // The job says 5 members and x0 holds 3.
    let j = job(
        &m,
        5,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    assert!(matches!(
        run_batched(&j, 2, &NO_CANCEL(), cpu),
        Err(EnsembleError::BadJob(_))
    ));
}

#[test]
fn cancel_stops_before_the_next_batch() {
    let m = build_model(VDP);
    let x0 = x0s(12);
    let j = job(
        &m,
        12,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );

    // Set before the run: no batch starts.
    let set = AtomicBool::new(true);
    let mut calls = 0;
    let r = run_batched(&j, 4, &set, |sub| {
        calls += 1;
        cpu(sub)
    });
    assert_eq!(r.unwrap_err(), EnsembleError::Cancelled);
    assert_eq!(calls, 0);

    // Set during the first batch: the second never starts.
    let flag = AtomicBool::new(false);
    let mut calls = 0;
    let r = run_batched(&j, 4, &flag, |sub| {
        calls += 1;
        flag.store(true, Ordering::SeqCst);
        cpu(sub)
    });
    assert_eq!(r.unwrap_err(), EnsembleError::Cancelled);
    assert_eq!(calls, 1);
}

#[test]
fn the_first_error_ends_the_run() {
    let m = build_model(VDP);
    let x0 = x0s(12);
    let j = job(
        &m,
        12,
        &x0,
        &[1.5],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    let mut calls = 0;
    let r = run_batched(&j, 4, &NO_CANCEL(), |sub| {
        calls += 1;
        if calls == 2 {
            Err(EnsembleError::GpuLaunch("device lost".into()))
        } else {
            cpu(sub)
        }
    });
    assert_eq!(
        r.unwrap_err(),
        EnsembleError::GpuLaunch("device lost".into())
    );
    assert_eq!(calls, 2);
}
