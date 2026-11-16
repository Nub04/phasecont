//! The global scratch layout of the CUDA kernel gives the bits of the private layout.
//!
//! The layout moves `y`, `y1`, `yt` and the stage vectors from private arrays to pointers
//! into the member's slice of `work`. Nothing else changes, so a kernel built either way, run
//! as plain C, must match the CPU backend and so match each other. A kernel that put a
//! scratch array on top of a word that a launch hands on, or that read `t` from the wrong
//! offset, shows up here as a different bit. The map method `iter` moves only `y` and `y1`.

mod common;

use common::shim::{Shim, assert_same, check_in};
use common::*;
use pc_ensemble::abi::pack_in;
use pc_ensemble::job::validate;
use pc_ensemble::kernel::{Integrator, Layout, private_reals, work_len};
use pc_ensemble::{Method, Output, Precision, Status, Tape};

const MEMBERS: usize = 4;
const BOTH: [Precision; 2] = [Precision::Single, Precision::Double];

/// Initial states that differ per member and never sit at zero.
fn x0_for(n: usize, members: usize) -> Vec<f64> {
    (0..members * n)
        .map(|k| {
            let (m, i) = (k / n, k % n);
            0.3 + 0.17 * m as f64 - 0.05 * i as f64
        })
        .collect()
}

fn methods(p: Precision) -> [Method; 2] {
    let dp5 = match p {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.01 }, dp5]
}

fn tag(name: &str, p: Precision, m: Method) -> String {
    let kind = match m {
        Method::Rk4 { .. } => "rk4",
        Method::Dp5 { .. } => "dp5",
        Method::Iter => "iter",
    };
    format!("g-{name}-{}-{kind}", p.name())
}

#[test]
fn the_global_layout_matches_the_cpu_bit_for_bit() {
    let cases = [
        ("decay", DECAY.to_owned(), vec![1.3]),
        ("vdp", VDP.to_owned(), vec![1.5]),
        ("mixed", MIXED.to_owned(), vec![0.7]),
        ("l96-8", lorenz96(8), vec![8.0]),
    ];
    for (name, src, params) in &cases {
        let m = build_model(src);
        let x0 = x0_for(m.n(), MEMBERS);
        for p in BOTH {
            for method in methods(p) {
                let j = job(&m, MEMBERS, &x0, params, 0.5, method, p);
                let what = tag(name, p, method);
                let Some(r) = check_in(&what, &j, Layout::Global) else {
                    return;
                };
                assert!(
                    r.status.contains(&Status::Ok),
                    "{what}: no member finished, so the comparison proves little: {:?}",
                    r.status
                );
            }
        }
    }
}

#[test]
fn a_larger_state_gives_the_same_bits_in_both_layouts() {
    // n = 40 puts the scratch far from the words a launch hands on, and several launches of
    // one step each hand `t`, `y` (and for Dormand-Prince `h`, `facold` and `k0`) on.
    let src = lorenz96(40);
    let m = build_model(&src);
    let x0 = x0_for(40, 3);
    for p in BOTH {
        for method in methods(p) {
            let j = job(&m, 3, &x0, &[8.0], 0.2, method, p);
            let what = tag("l96-40", p, method);
            let (Some(private), Some(global)) = (
                Shim::new_in(&format!("{what}-p"), &j, Layout::Private),
                Shim::new_in(&format!("{what}-g"), &j, Layout::Global),
            ) else {
                return;
            };
            for steps in [1, 5, u64::MAX] {
                let a = private.run(&j, steps);
                let b = global.run(&j, steps);
                assert_same(&format!("{what} (steps {steps})"), &a, &b);
            }
        }
    }
}

#[test]
fn a_member_that_stops_early_reports_the_time_from_the_first_word() {
    // `finish` reads `t` at `work[m * work_len]` for a member that did not end `Ok`. The
    // global layout keeps that word first, so the stopped time must match the CPU.
    let m = build_model("x'=x\ndone\n");
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, 3, &[1.0, 2.0, 3.0], &[], 20.0, method, p);
            j.bounds = 100.0;
            let what = tag("bounds", p, method);
            let Some(r) = check_in(&what, &j, Layout::Global) else {
                return;
            };
            assert!(r.status.iter().all(|s| *s == Status::Bounds), "{what}");
            assert!(r.t_reached.iter().all(|&t| t > 0.0 && t < 20.0), "{what}");
        }
    }
}

#[test]
fn the_step_limit_matches_the_cpu_in_the_global_layout() {
    let m = build_model(VDP);
    let x0 = x0_for(2, 2);
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, 2, &x0, &[1.5], 5.0, method, p);
            j.max_steps = 3;
            let what = tag("max-steps", p, method);
            let Some(r) = check_in(&what, &j, Layout::Global) else {
                return;
            };
            assert!(r.status.iter().all(|s| *s == Status::MaxSteps), "{what}");
        }
    }
}

#[test]
fn the_layout_sets_the_work_length() {
    // Global is what a launch hands on plus the scratch: 8n + 1, 12n + 3 and 3n + 1.
    for (n, rk4, dp5, iter) in [(2usize, 17usize, 27usize, 7usize), (40, 321, 483, 121)] {
        assert_eq!(Layout::Private.work_len(n, Integrator::Rk4), n + 1);
        assert_eq!(Layout::Private.work_len(n, Integrator::Dp5), 2 * n + 3);
        assert_eq!(Layout::Private.work_len(n, Integrator::Iter), n + 1);
        assert_eq!(Layout::Global.work_len(n, Integrator::Rk4), rk4);
        assert_eq!(Layout::Global.work_len(n, Integrator::Dp5), dp5);
        assert_eq!(Layout::Global.work_len(n, Integrator::Iter), iter);
        assert_eq!(private_reals(n, Integrator::Iter), 2 * n);
        for i in [Integrator::Rk4, Integrator::Dp5, Integrator::Iter] {
            assert_eq!(
                Layout::Global.work_len(n, i),
                work_len(n, i) + private_reals(n, i)
            );
        }
    }
}

#[test]
fn pack_sizes_the_work_buffer_by_the_layout() {
    let m = build_model(lorenz96(6).as_str());
    let x0 = x0_for(6, 5);
    let tape = Tape::build(&m).expect("tape");
    let map = build_model(&ring_map(6));
    for method in [Method::Rk4 { dt: 0.01 }, DP5_STD, Method::Iter] {
        let (model, par): (_, &[f64]) = if method == Method::Iter {
            (&map, &[])
        } else {
            (&m, &[8.0])
        };
        let j = job(model, 5, &x0, par, 1.0, method, Precision::Double);
        let checked = validate(&j).expect("validate");
        let integrator = pc_ensemble::abi::integrator_of(method);
        for layout in [Layout::Private, Layout::Global] {
            let wl = layout.work_len(6, integrator);
            let p = pack_in::<f64>(&j, &checked, &tape, layout);
            assert_eq!(p.work_len, wl, "{layout:?} {method:?}");
            assert_eq!(p.work.len(), 5 * wl, "{layout:?} {method:?}");
            assert!(p.work.iter().all(|&w| w == 0.0));
        }
        // `pack` stays the private layout.
        let p = pc_ensemble::abi::pack::<f64>(&j, &checked, &tape);
        assert_eq!(p.work_len, work_len(6, integrator));
    }
}

#[test]
fn maps_give_the_same_bits_in_both_layouts() {
    // SPEC 8aa, acceptance test 3, in the global layout; `ring` with n = 40 puts `y1` far
    // from the words a launch hands on, and launches of one iterate each hand `t` and `y` on.
    let ring = ring_map(40);
    let cases: [(&str, &str, &[f64]); 4] = [
        ("logistic", LOGISTIC_MAP, &[3.7]),
        ("henon", HENON_MAP, &[1.4, 0.3]),
        ("shift", SHIFT_MAP, &[]),
        ("ring-40", &ring, &[]),
    ];
    for (name, src, params) in cases {
        let m = build_model(src);
        let x0: Vec<f64> = x0_for(m.n(), MEMBERS).iter().map(|v| 0.1 * v).collect();
        for p in BOTH {
            let mut j = job(&m, MEMBERS, &x0, params, 30.0, Method::Iter, p);
            j.output = Output::Samples(vec![1.0, 7.0, 30.0]);
            let what = tag(name, p, Method::Iter);
            let Some(r) = check_in(&what, &j, Layout::Global) else {
                return;
            };
            assert!(
                r.status.iter().all(|s| *s == Status::Ok),
                "{what}: {:?}",
                r.status
            );
            let (Some(private), Some(global)) = (
                Shim::new_in(&format!("{what}-p"), &j, Layout::Private),
                Shim::new_in(&format!("{what}-g"), &j, Layout::Global),
            ) else {
                return;
            };
            for steps in [1, 5, u64::MAX] {
                let a = private.run(&j, steps);
                let b = global.run(&j, steps);
                assert_same(&format!("{what} (steps {steps})"), &a, &b);
            }
        }
    }
}
