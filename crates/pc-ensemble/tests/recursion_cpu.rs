//! Recursive function calls on the CPU ensemble (SPEC section 8af-5, stage 4a). A member
//! whose right-hand side goes past the frame cap ends with `recursion_depth`, and the other
//! members go on with the same bits as a run without it. Every model is our own.

mod common;

use common::*;
use pc_ensemble::{Method, Output, Params, Precision, Status};

/// `s(n)` counts down to zero, so `s(k) = k` for a whole `k` and the call takes `k + 1`
/// frames. `x' = -s(k) x / k` is `x' = -x` written through the call.
const COUNTED: &str = "par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n";
const PLAIN: &str = "par k=3\nx'=-k*x/k\ndone\n";

/// Past the default cap of 1,000,000 frames.
const DEEP: f64 = 2e6;

#[test]
fn a_member_past_the_cap_ends_alone_and_the_others_keep_their_bits() {
    let calls = build_model(COUNTED);
    let plain = build_model(PLAIN);
    let x0 = [1.0, 2.0, 3.0];
    let k = [3.0, DEEP, 5.0];
    for precision in [Precision::Double, Precision::Single] {
        let single = precision == Precision::Single;
        for method in [
            Method::Rk4 { dt: 0.01 },
            if single { DP5_SINGLE } else { DP5_STD },
        ] {
            let what = format!("{method:?} {precision:?}");
            let mut j = job(&calls, 3, &x0, &[], 1.0, method, precision);
            j.params = Params::PerMember(&k);
            j.output = Output::Samples(vec![0.5, 1.0]);
            let r = run(&j);
            assert_eq!(
                r.status,
                [Status::Ok, Status::RecursionDepth, Status::Ok],
                "{what}"
            );
            assert_eq!(r.t_reached[1], 0.0, "{what}");
            assert_eq!(r.accepted[1], 0, "{what}");
            assert!(
                r.state(1, 0)[0].is_nan() && r.state(1, 1)[0].is_nan(),
                "{what}"
            );
            // The members that finish match the call-free model bit for bit.
            let mut p = job(&plain, 3, &x0, &[], 1.0, method, precision);
            p.params = Params::PerMember(&k);
            p.output = Output::Samples(vec![0.5, 1.0]);
            let want = run(&p);
            for m in [0, 2] {
                for s in 0..2 {
                    assert_eq!(
                        bits(r.state(m, s)),
                        bits(want.state(m, s)),
                        "{what} {m} {s}"
                    );
                }
                assert_eq!(r.accepted[m], want.accepted[m], "{what} {m}");
            }
        }
    }
}

#[test]
fn a_member_that_reaches_the_cap_mid_run_keeps_its_last_accepted_state() {
    // `x' = 1` from 0.4, and the depth of `y'` grows with `x`: past 1,000,000 frames once
    // `x` passes 2/3, so the stage at t = 0.3 fails and the member stops at t = 0.2.
    let m = build_model("s(n)=if(n<1)then(0)else(1+s(n-1))\nx'=1\ny'=s(x*1.5e6)*0\ndone\n");
    let r = run(&job(
        &m,
        1,
        &[0.4, 0.0],
        &[],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    ));
    assert_eq!(r.status, [Status::RecursionDepth]);
    assert_eq!(r.accepted[0], 2);
    assert!((r.t_reached[0] - 0.2).abs() < 1e-12, "{}", r.t_reached[0]);
}

#[test]
fn a_map_member_past_the_cap_ends_alone() {
    let calls =
        build_model("par k=3\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx(t+1)=s(k)*x/k/2\ndone\n");
    let plain = build_model("par k=3\nx(t+1)=k*x/k/2\ndone\n");
    let x0 = [1.0, 2.0];
    let k = [3.0, DEEP];
    let mut j = job(&calls, 2, &x0, &[], 10.0, Method::Iter, Precision::Double);
    j.params = Params::PerMember(&k);
    let r = run(&j);
    assert_eq!(r.status, [Status::Ok, Status::RecursionDepth]);
    assert_eq!(r.t_reached[1], 0.0);
    let mut p = job(&plain, 2, &x0, &[], 10.0, Method::Iter, Precision::Double);
    p.params = Params::PerMember(&k);
    assert_eq!(bits(r.state(0, 0)), bits(run(&p).state(0, 0)));
    assert_eq!(r.state(0, 0)[0], 1.0 / 1024.0);
}

#[test]
fn recursion_depth_has_code_six() {
    assert_eq!(Status::RecursionDepth.code(), 6);
    assert_eq!(Status::RecursionDepth.name(), "recursion_depth");
    assert_eq!(Status::from_code(6), Some(Status::RecursionDepth));
    assert_eq!(Status::from_code(7), None);
}
