//! The ensemble kernel text, compiled as plain C, gives the bits of the CPU backend.
//!
//! Each case runs the CPU backend once and the kernel with several launch sizes, in single
//! and double precision, for fixed-step RK4, for Dormand-Prince 5(4) and, on maps, for the
//! method `iter`.

mod common;

use common::shim::check;
use common::*;
use pc_ensemble::{Method, Output, Params, Precision, Status};

const ALL_OPS: &str = "par a=0.7, b=2\n\
    x'=sin(x)*a+exp(-y)/(1+x^2)+b*x^5-abs(y)+if(x>y)then(max(x,y))else(min(x,y))\n\
    y'=atan2(x,y)+sqrt(1+y*y)+heav(x-y)+sign(y)+mod(x,1.3)+x^(-3)+t\n\
    done\n";

const MEMBERS: usize = 5;

/// Initial states that differ per member and never sit at zero.
fn x0_for(n: usize, members: usize) -> Vec<f64> {
    (0..members * n)
        .map(|k| {
            let (m, i) = (k / n, k % n);
            0.3 + 0.17 * m as f64 - 0.4 * i as f64
        })
        .collect()
}

fn methods(p: Precision) -> [Method; 2] {
    let dp5 = match p {
        Precision::Single => DP5_SINGLE,
        Precision::Double => DP5_STD,
    };
    [Method::Rk4 { dt: 0.05 }, dp5]
}

const BOTH: [Precision; 2] = [Precision::Single, Precision::Double];

fn tag(name: &str, p: Precision, m: Method) -> String {
    let kind = match m {
        Method::Rk4 { .. } => "rk4",
        Method::Dp5 { .. } => "dp5",
        Method::Iter => "iter",
    };
    format!("{name}-{}-{kind}", p.name())
}

#[test]
fn models_match_the_cpu_bit_for_bit() {
    let cases: [(&str, &str, &[f64]); 5] = [
        ("decay", DECAY, &[1.3]),
        ("osc", OSC, &[]),
        ("vdp", VDP, &[1.5]),
        ("mixed", MIXED, &[0.7]),
        ("allops", ALL_OPS, &[0.7, 2.0]),
    ];
    for (name, src, params) in cases {
        let m = build_model(src);
        let x0 = x0_for(m.n(), MEMBERS);
        for p in BOTH {
            for method in methods(p) {
                let j = job(&m, MEMBERS, &x0, params, 1.0, method, p);
                let what = tag(name, p, method);
                let Some(r) = check(&what, &j) else {
                    return;
                };
                if name != "allops" {
                    assert!(
                        r.status.contains(&Status::Ok),
                        "{what}: no member finished, so the comparison proves little: {:?}",
                        r.status
                    );
                }
            }
        }
    }
}

#[test]
fn samples_match_the_cpu() {
    let m = build_model(VDP);
    let x0 = x0_for(2, MEMBERS);
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, MEMBERS, &x0, &[1.5], 1.0, method, p);
            j.output = Output::Samples(vec![0.25, 0.5, 0.75, 1.0]);
            let what = tag("samples", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            assert_eq!(r.times.len(), 4);
            assert!(r.status.iter().all(|s| *s == Status::Ok), "{what}");
        }
    }
}

#[test]
fn per_member_parameters_match_the_cpu() {
    let m = build_model(DECAY);
    let x0 = vec![1.0; MEMBERS];
    let rates = [0.5, 1.0, 1.5, 2.0, 2.5];
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, MEMBERS, &x0, &[], 1.0, method, p);
            j.params = Params::PerMember(&rates);
            let what = tag("per-member", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            // A faster decay ends lower, so the members really used their own rate.
            let ends: Vec<f64> = (0..MEMBERS).map(|i| r.state(i, 0)[0]).collect();
            assert!(ends.windows(2).all(|w| w[1] < w[0]), "{what}: {ends:?}");
        }
    }
}

#[test]
fn forced_step_rejections_match_the_cpu() {
    // A stiff-ish van der Pol with a tight tolerance makes the controller reject steps.
    let m = build_model(VDP);
    let x0 = x0_for(2, MEMBERS);
    let j = job(&m, MEMBERS, &x0, &[5.0], 10.0, DP5_TIGHT, Precision::Double);
    let Some(r) = check("rejections", &j) else {
        return;
    };
    assert!(r.rejected.iter().any(|&k| k > 0), "{:?}", r.rejected);
}

#[test]
fn a_nan_right_hand_side_matches_the_cpu() {
    let m = build_model("x'=ln(x)\ndone\n");
    let x0 = [-1.0, 2.0];
    for p in BOTH {
        for method in methods(p) {
            let j = job(&m, 2, &x0, &[], 1.0, method, p);
            let what = tag("nan-start", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            assert_eq!(r.status[0], Status::NonFinite, "{what}");
        }
    }
    // The right-hand side turns infinite and then NaN part way through the run.
    let m = build_model("x'=1\ny'=ln(2-x)\ndone\n");
    for p in BOTH {
        for method in methods(p) {
            let j = job(&m, 1, &[0.0, 0.0], &[], 5.0, method, p);
            let what = tag("nan-mid", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            assert_eq!(r.status[0], Status::NonFinite, "{what}");
        }
    }
}

#[test]
fn the_bound_matches_the_cpu() {
    let m = build_model("x'=x\ndone\n");
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, 1, &[1.0], &[], 20.0, method, p);
            j.bounds = 100.0;
            let what = tag("bounds", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            assert_eq!(r.status[0], Status::Bounds, "{what}");
        }
    }
}

#[test]
fn an_overflow_matches_the_cpu_under_a_bound_that_cannot_catch_it() {
    // The state reaches infinity, which no finite bound sees. It must end NONFINITE, and
    // with an infinite bound or one above the largest finite value the fused test has to
    // catch it all the same.
    let m = build_model("x'=x*x*x*x\ndone\n");
    for p in BOTH {
        for method in methods(p) {
            for (label, bounds) in [("inf", f64::INFINITY), ("1e300", 1e300)] {
                let mut j = job(&m, 2, &[1e10, 0.1], &[], 1.0, method, p);
                j.bounds = bounds;
                let what = tag(&format!("overflow-{label}"), p, method);
                let Some(r) = check(&what, &j) else {
                    return;
                };
                // RK4 steps into infinity and stops there. DP5 may end NonFinite or, once it has
                // shrunk the step until it underflows, StepTooSmall, depending on the precision.
                // `check` has already shown that the CPU ends the same way, so the member only
                // has to end with one of those and the quiet member must stay Ok.
                let ok_ends: &[Status] = match method {
                    Method::Rk4 { .. } => &[Status::NonFinite],
                    Method::Dp5 { .. } => &[Status::NonFinite, Status::StepTooSmall],
                    Method::Iter => unreachable!("methods() gives rk4 and dp5"),
                };
                assert!(ok_ends.contains(&r.status[0]), "{what}: {:?}", r.status[0]);
                assert_eq!(r.status[1], Status::Ok, "{what}");
            }
        }
    }
}

#[test]
fn the_step_limit_matches_the_cpu() {
    let m = build_model(DECAY);
    for p in BOTH {
        for method in methods(p) {
            let mut j = job(&m, 1, &[1.0], &[1.0], 1.0, method, p);
            j.method = match method {
                Method::Rk4 { .. } => Method::Rk4 { dt: 0.01 },
                other => other,
            };
            // One step cannot reach t = 1 for either method: RK4 needs 100 and the first
            // adaptive step is at most 100 times the initial guess, which is below 1.
            j.max_steps = 1;
            let what = tag("max-steps", p, method);
            let Some(r) = check(&what, &j) else {
                return;
            };
            assert_eq!(r.status[0], Status::MaxSteps, "{what}");
        }
    }
}

#[test]
fn a_step_below_roundoff_matches_the_cpu() {
    let m = build_model(DECAY);
    for (p, t0) in [(Precision::Single, 1e7), (Precision::Double, 1e15)] {
        let mut j = job(&m, 1, &[1.0], &[1.0], t0 + 1e3, DP5_SINGLE, p);
        j.t0 = t0;
        let what = tag("roundoff", p, DP5_SINGLE);
        let Some(r) = check(&what, &j) else {
            return;
        };
        assert_eq!(r.status[0], Status::StepTooSmall, "{what}");
    }
}

#[test]
fn maps_match_the_cpu_bit_for_bit() {
    // SPEC 8aa, acceptance test 3: the logistic, Henon and x + t maps with samples.
    let cases: [(&str, &str, &[f64]); 4] = [
        ("logistic", LOGISTIC_MAP, &[3.7]),
        ("henon", HENON_MAP, &[1.4, 0.3]),
        ("shift", SHIFT_MAP, &[]),
        ("ring", &ring_map(6), &[]),
    ];
    for (name, src, params) in cases {
        let m = build_model(src);
        let x0: Vec<f64> = x0_for(m.n(), MEMBERS)
            .iter()
            .map(|v| 0.2 + 0.1 * v)
            .collect();
        for p in BOTH {
            let outs = [
                ("final", Output::Final),
                ("samples", Output::Samples(vec![1.0, 7.0, 25.0, 40.0])),
            ];
            for (label, out) in outs {
                let mut j = job(&m, MEMBERS, &x0, params, 40.0, Method::Iter, p);
                j.output = out;
                let what = tag(&format!("{name}-{label}"), p, Method::Iter);
                let Some(r) = check(&what, &j) else {
                    return;
                };
                assert!(
                    r.status.iter().all(|s| *s == Status::Ok),
                    "{what}: {:?}",
                    r.status
                );
                assert!(r.accepted.iter().all(|&a| a == 40), "{what}");
            }
        }
    }
}

#[test]
fn map_stops_match_the_cpu() {
    // An overflow, a bound and the step limit each stop a member of a map part way.
    let square = build_model("x(t+1)=x^2\ndone\n");
    let double = build_model("x(t+1)=2*x\ndone\n");
    for p in BOTH {
        let mut j = job(&square, 2, &[1e10, 0.5], &[], 20.0, Method::Iter, p);
        j.output = Output::Samples(vec![2.0, 10.0, 20.0]);
        let what = tag("map-overflow", p, Method::Iter);
        let Some(r) = check(&what, &j) else {
            return;
        };
        assert_eq!(r.status, [Status::NonFinite, Status::Ok], "{what}");

        let mut j = job(&double, 1, &[1.0], &[], 20.0, Method::Iter, p);
        j.bounds = 100.0;
        let what = tag("map-bounds", p, Method::Iter);
        let Some(r) = check(&what, &j) else {
            return;
        };
        assert_eq!(r.status[0], Status::Bounds, "{what}");
        assert_eq!(r.t_reached[0], 6.0, "{what}");

        let mut j = job(&double, 1, &[1.0], &[], 20.0, Method::Iter, p);
        j.max_steps = 5;
        let what = tag("map-max-steps", p, Method::Iter);
        let Some(r) = check(&what, &j) else {
            return;
        };
        assert_eq!(r.status[0], Status::MaxSteps, "{what}");
        assert_eq!(r.t_reached[0], 5.0, "{what}");
    }
}
