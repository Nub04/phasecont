//! RK4 on the CPU runs members in groups of lanes (SPEC section 13, `cpu`). A member gives
//! the same bits whatever the group size: every sample, status, time reached and step
//! count matches a run with one lane, with the tape and with a compiled right-hand side, in
//! both precisions, for member counts that are not a multiple of the group, for a lane that
//! stops at any position in its group, for any launch length and any thread count.

mod common;

use common::*;
use pc_ensemble::cpu::{default_lanes, run_cpu_lanes};
use pc_ensemble::job::NativeRhs;
use pc_ensemble::{EnsembleResult, Job, Method, Output, Params, Precision, Status};
use pc_ir::IrModel;
use pc_ir::{Entry, EntryKind};
use pc_jit::JitEntries;
use std::sync::atomic::AtomicBool;

const LANES: [usize; 3] = [2, 4, 8];
const RK4: Method = Method::Rk4 { dt: 0.013 };

fn run_lanes(job: &Job, launch: u64, lanes: usize) -> EnsembleResult {
    run_cpu_lanes(job, launch, &AtomicBool::new(false), lanes).expect("run")
}

fn same(a: &EnsembleResult, b: &EnsembleResult, what: &str) {
    assert_eq!(bits(&a.states), bits(&b.states), "{what}: states");
    assert_eq!(a.status, b.status, "{what}: status");
    assert_eq!(bits(&a.t_reached), bits(&b.t_reached), "{what}: t reached");
    assert_eq!(a.accepted, b.accepted, "{what}: accepted");
    assert_eq!(a.rejected, b.rejected, "{what}: rejected");
}

/// The compiled `F` of `ir` in `precision`, kept alive by the returned entries.
fn native(ir: &IrModel, precision: Precision) -> (JitEntries, NativeRhs) {
    let entry = Entry {
        kind: EntryKind::F,
        outputs: ir.f.clone(),
    };
    let entries = [(EntryKind::F, &entry)];
    let je = match precision {
        Precision::Double => pc_jit::compile(&entries, &ir.dag),
        Precision::Single => pc_jit::compile_f32(&entries, &ir.dag),
    }
    .expect("compile");
    // SAFETY: the `F` entry of `ir` in the precision of the job, alive until `je` drops.
    let f = unsafe {
        match precision {
            Precision::Double => NativeRhs::new(je.plain(&EntryKind::F).expect("entry")),
            Precision::Single => NativeRhs::new_f32(je.plain_f32(&EntryKind::F).expect("entry")),
        }
    };
    (je, f)
}

fn x0s(members: usize, n: usize) -> Vec<f64> {
    (0..members * n)
        .map(|i| 0.9 * ((i * 7 % 23) as f64 / 23.0) - 0.4)
        .collect()
}

/// Every model and precision, with and without the compiled `F`: the job and a label.
fn each_case(mut f: impl FnMut(&Job, &str)) {
    for (name, src, per_member) in [("vdp", VDP, false), ("mixed", MIXED, true)] {
        let m = build_model(src);
        for precision in [Precision::Double, Precision::Single] {
            let (_je, nat) = native(&m, precision);
            for members in [1, 3, 4, 5, 7, 8, 9, 13, 17] {
                let x0 = x0s(members, 2);
                let ps: Vec<f64> = (0..members).map(|i| 0.3 + 0.17 * i as f64).collect();
                for with_native in [false, true] {
                    let mut j = job(&m, members, &x0, &[1.5], 3.0, RK4, precision);
                    if per_member {
                        j.params = Params::PerMember(&ps);
                    }
                    j.output = Output::Samples(vec![0.5, 1.7, 3.0]);
                    if with_native {
                        j.native = Some(nat);
                    }
                    f(
                        &j,
                        &format!("{name} {precision:?} {members} native={with_native}"),
                    );
                }
            }
        }
    }
}

#[test]
fn a_member_has_the_bits_of_one_lane_in_every_group_size() {
    each_case(|j, what| {
        let one = run_lanes(j, u64::MAX, 1);
        for lanes in LANES {
            same(
                &run_lanes(j, u64::MAX, lanes),
                &one,
                &format!("{what} lanes={lanes}"),
            );
        }
    });
}

#[test]
fn the_default_group_gives_the_bits_of_one_lane() {
    each_case(|j, what| {
        same(&run(j), &run_lanes(j, u64::MAX, 1), what);
    });
}

#[test]
fn the_launch_length_does_not_change_a_group() {
    let m = build_model(MIXED);
    let x0 = x0s(11, 2);
    for precision in [Precision::Double, Precision::Single] {
        let (_je, nat) = native(&m, precision);
        let mut j = job(&m, 11, &x0, &[0.7], 2.0, RK4, precision);
        j.output = Output::Samples(vec![0.5, 2.0]);
        j.native = Some(nat);
        let want = run_lanes(&j, u64::MAX, 1);
        for lanes in LANES {
            for launch in [1, 2, 7, 64] {
                same(
                    &run_lanes(&j, launch, lanes),
                    &want,
                    &format!("{precision:?} lanes={lanes} launch={launch}"),
                );
            }
        }
    }
}

#[test]
fn the_thread_count_does_not_change_a_group() {
    let m = build_model(VDP);
    let x0 = x0s(37, 2);
    for precision in [Precision::Double, Precision::Single] {
        let (_je, nat) = native(&m, precision);
        let mut j = job(&m, 37, &x0, &[1.5], 4.0, RK4, precision);
        j.output = Output::Samples(vec![1.0, 4.0]);
        j.native = Some(nat);
        let want = run_lanes(&j, u64::MAX, 1);
        for threads in [1, 2, 5] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for lanes in LANES {
                for launch in [3, u64::MAX] {
                    let got = pool.install(|| run_lanes(&j, launch, lanes));
                    same(
                        &got,
                        &want,
                        &format!("{precision:?} threads={threads} lanes={lanes} launch={launch}"),
                    );
                }
            }
        }
    }
}

/// `x' = -a x` with `a = -2000` for the member at `bad` and `a = 1` for the others: that member
/// leaves the bound or overflows, at every position in its group.
fn one_bad_member(bad: usize, members: usize, bound: f64, launch: u64) {
    let m = build_model(DECAY);
    let x0: Vec<f64> = (0..members).map(|i| 1.0 + 0.1 * i as f64).collect();
    let a: Vec<f64> = (0..members)
        .map(|i| if i == bad { -2000.0 } else { 1.0 })
        .collect();
    for precision in [Precision::Double, Precision::Single] {
        let (_je, nat) = native(&m, precision);
        for with_native in [false, true] {
            let mut j = job(&m, members, &x0, &[], 2.0, RK4, precision);
            j.params = Params::PerMember(&a);
            j.output = Output::Samples(vec![0.5, 1.0, 2.0]);
            j.bounds = bound;
            if with_native {
                j.native = Some(nat);
            }
            let want = run_lanes(&j, launch, 1);
            assert!(want.status[bad] != Status::Ok, "member {bad} must stop");
            for lanes in LANES {
                same(
                    &run_lanes(&j, launch, lanes),
                    &want,
                    &format!(
                        "{precision:?} native={with_native} bad={bad} bound={bound} lanes={lanes} launch={launch}"
                    ),
                );
            }
        }
    }
}

#[test]
fn a_member_past_the_bound_stops_alone_at_every_lane() {
    for bad in 0..11 {
        for launch in [1, 2, 5, u64::MAX] {
            one_bad_member(bad, 11, 1e3, launch);
        }
    }
}

#[test]
fn a_member_that_overflows_stops_alone_at_every_lane() {
    for bad in 0..11 {
        for launch in [1, 2, 5, u64::MAX] {
            one_bad_member(bad, 11, f64::INFINITY, launch);
        }
    }
}

#[test]
fn lanes_that_stop_on_the_same_or_different_steps_match_one_lane() {
    // Members 1 and 2 leave the bound on the same step, member 5 later, member 6 never.
    let m = build_model(DECAY);
    let a = [1.0, -2000.0, -2000.0, 1.0, 0.5, -40.0, 1.0, 2.0, -2000.0];
    let x0: Vec<f64> = (0..a.len()).map(|i| 1.0 + 0.1 * i as f64).collect();
    for precision in [Precision::Double, Precision::Single] {
        let (_je, nat) = native(&m, precision);
        for with_native in [false, true] {
            let mut j = job(&m, a.len(), &x0, &[], 2.0, RK4, precision);
            j.params = Params::PerMember(&a);
            j.output = Output::Samples(vec![0.013, 0.5, 2.0]);
            j.bounds = 1e3;
            if with_native {
                j.native = Some(nat);
            }
            let want = run_lanes(&j, u64::MAX, 1);
            for bad in [1, 2, 5, 8] {
                assert_eq!(want.status[bad], Status::Bounds, "member {bad}");
            }
            for lanes in LANES {
                for launch in [1, 3, u64::MAX] {
                    same(
                        &run_lanes(&j, launch, lanes),
                        &want,
                        &format!(
                            "{precision:?} native={with_native} lanes={lanes} launch={launch}"
                        ),
                    );
                }
            }
        }
    }
}

#[test]
fn final_output_and_dp5_match_one_lane() {
    let m = build_model(VDP);
    let x0 = x0s(9, 2);
    for precision in [Precision::Double, Precision::Single] {
        let (_je, nat) = native(&m, precision);
        let dp5 = match precision {
            Precision::Double => DP5_STD,
            Precision::Single => DP5_SINGLE,
        };
        for method in [RK4, dp5] {
            let mut j = job(&m, 9, &x0, &[1.5], 2.0, method, precision);
            j.native = Some(nat);
            let want = run_lanes(&j, u64::MAX, 1);
            for lanes in LANES {
                for launch in [1, 4, u64::MAX] {
                    same(
                        &run_lanes(&j, launch, lanes),
                        &want,
                        &format!("{precision:?} {method:?} lanes={lanes} launch={launch}"),
                    );
                }
            }
        }
    }
}

#[test]
fn every_lane_reaching_the_step_limit_matches_one_lane() {
    let m = build_model(VDP);
    let x0 = x0s(10, 2);
    for precision in [Precision::Double, Precision::Single] {
        let mut j = job(&m, 10, &x0, &[1.5], 3.0, RK4, precision);
        j.output = Output::Samples(vec![1.0, 3.0]);
        j.max_steps = 100;
        let want = run_lanes(&j, u64::MAX, 1);
        assert!(want.status.iter().all(|&s| s == Status::MaxSteps));
        for lanes in LANES {
            same(
                &run_lanes(&j, 7, lanes),
                &want,
                &format!("{precision:?} {lanes}"),
            );
        }
    }
}

#[test]
fn a_member_past_the_call_cap_stops_alone_in_a_group() {
    // `s(n)` counts down to zero; the member with k = 2e6 passes the frame cap.
    let m = build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n");
    let x0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for bad in 0..6 {
        let k: Vec<f64> = (0..6).map(|i| if i == bad { 2e6 } else { 3.0 }).collect();
        for precision in [Precision::Double, Precision::Single] {
            let mut j = job(&m, 6, &x0, &[], 0.2, Method::Rk4 { dt: 0.05 }, precision);
            j.params = Params::PerMember(&k);
            j.output = Output::Samples(vec![0.1, 0.2]);
            let want = run_lanes(&j, u64::MAX, 1);
            assert_eq!(want.status[bad], Status::RecursionDepth);
            for lanes in LANES {
                same(
                    &run_lanes(&j, u64::MAX, lanes),
                    &want,
                    &format!("{precision:?} bad={bad} lanes={lanes}"),
                );
            }
        }
    }
}

#[test]
fn a_group_size_outside_the_list_is_refused() {
    let m = build_model(DECAY);
    let j = job(&m, 2, &[1.0, 2.0], &[1.0], 1.0, RK4, Precision::Double);
    for lanes in [0, 3, 5, 16] {
        assert!(
            run_cpu_lanes(&j, u64::MAX, &AtomicBool::new(false), lanes).is_err(),
            "lanes={lanes}"
        );
    }
}

#[test]
fn the_default_group_leaves_every_thread_two_groups() {
    // The largest of 4 and 2 with members >= 2 * lanes * threads, else 1 (SPEC section 13).
    for (members, threads, want) in [
        (0, 16, 1),
        (1, 1, 1),
        (3, 1, 1),
        (4, 1, 2),
        (7, 1, 2),
        (8, 1, 4),
        (100_000, 1, 4),
        (8, 16, 1),
        (32, 16, 1),
        (63, 16, 1),
        (64, 16, 2),
        (127, 16, 2),
        (128, 16, 4),
        (100_000, 16, 4),
        (100, 0, 4),
    ] {
        assert_eq!(
            default_lanes(members, threads),
            want,
            "members={members} threads={threads}"
        );
    }
}

#[test]
fn the_default_group_gives_the_bits_of_one_lane_at_every_thread_count() {
    let m = build_model(VDP);
    for members in [3, 8, 16, 33, 64] {
        let x0 = x0s(members, 2);
        for precision in [Precision::Double, Precision::Single] {
            let mut j = job(&m, members, &x0, &[1.5], 2.0, RK4, precision);
            j.output = Output::Samples(vec![1.0, 2.0]);
            let want = run_lanes(&j, u64::MAX, 1);
            for threads in [1, 2, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                same(
                    &pool.install(|| run(&j)),
                    &want,
                    &format!("{precision:?} members={members} threads={threads}"),
                );
            }
        }
    }
}
