//! The result of a member does not depend on repeats, on the other members, on how a run is
//! cut into launches or on the number of threads.

mod common;

use common::*;
use pc_ensemble::{EnsembleResult, Job, Method, Output, Params, Precision, run_cpu};
use std::sync::atomic::AtomicBool;

fn x0s(members: usize) -> Vec<f64> {
    (0..members)
        .flat_map(|i| [0.5 + 0.37 * i as f64, 0.1 * i as f64 - 0.4])
        .collect()
}

fn same(a: &EnsembleResult, b: &EnsembleResult) {
    assert_eq!(bits(&a.states), bits(&b.states));
    assert_eq!(a.status, b.status);
    assert_eq!(a.accepted, b.accepted);
    assert_eq!(a.rejected, b.rejected);
    assert_eq!(bits(&a.t_reached), bits(&b.t_reached));
}

fn methods(precision: Precision) -> [Method; 2] {
    let dp5 = match precision {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.013 }, dp5]
}

#[test]
fn repeated_runs_are_identical() {
    let m = build_model(VDP);
    let x0 = x0s(24);
    for precision in [Precision::Single, Precision::Double] {
        for method in methods(precision) {
            let mut j = job(&m, 24, &x0, &[1.5], 9.0, method, precision);
            j.output = Output::Samples(vec![1.0, 4.0, 9.0]);
            same(&run(&j), &run(&j));
        }
    }
}

#[test]
fn a_member_does_not_depend_on_the_others() {
    let m = build_model(VDP);
    let x0 = x0s(16);
    for precision in [Precision::Single, Precision::Double] {
        for method in methods(precision) {
            let all = run(&job(&m, 16, &x0, &[1.5], 9.0, method, precision));
            for i in [0, 5, 15] {
                let one = run(&job(
                    &m,
                    1,
                    &x0[2 * i..2 * i + 2],
                    &[1.5],
                    9.0,
                    method,
                    precision,
                ));
                assert_eq!(bits(all.state(i, 0)), bits(one.state(0, 0)), "member {i}");
                assert_eq!(all.accepted[i], one.accepted[0]);
                assert_eq!(all.rejected[i], one.rejected[0]);
            }
        }
    }
}

#[test]
fn the_launch_length_does_not_change_the_result() {
    let m = build_model(MIXED);
    let x0 = x0s(20);
    for precision in [Precision::Single, Precision::Double] {
        for method in methods(precision) {
            let mut j = job(&m, 20, &x0, &[0.7], 7.0, method, precision);
            j.output = Output::Samples(vec![0.5, 3.0, 7.0]);
            let whole = run_cpu(&j, u64::MAX, &AtomicBool::new(false)).unwrap();
            for k in [1_u64, 2, 7, 64] {
                let cut = run_cpu(&j, k, &AtomicBool::new(false)).unwrap();
                same(&whole, &cut);
            }
        }
    }
}

#[test]
fn the_thread_count_does_not_change_the_result() {
    let m = build_model(VDP);
    let x0 = x0s(32);
    let j = job(&m, 32, &x0, &[1.5], 9.0, DP5_STD, Precision::Double);
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
    };
    let one = pool(1).install(|| run(&j));
    let many = pool(5).install(|| run(&j));
    same(&one, &many);
}

#[test]
fn per_member_parameters_do_not_change_the_cut_invariance() {
    let m = build_model(VDP);
    let x0 = x0s(10);
    let mus: Vec<f64> = (0..10).map(|i| 0.2 + 0.3 * i as f64).collect();
    let mut j: Job = job(&m, 10, &x0, &[], 6.0, DP5_STD, Precision::Double);
    j.params = Params::PerMember(&mus);
    let whole = run(&j);
    let cut = run_cpu(&j, 3, &AtomicBool::new(false)).unwrap();
    same(&whole, &cut);
}
