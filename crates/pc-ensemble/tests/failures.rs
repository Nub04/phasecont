//! Statuses, rejected jobs, unsupported models and cancel.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common::*;
use pc_ensemble::{EnsembleError, Method, Output, Params, Precision, Status, run_cpu};

fn bad(e: EnsembleError) -> String {
    match e {
        EnsembleError::BadJob(m) => m,
        other => panic!("expected BadJob, got {other:?}"),
    }
}

#[test]
fn a_nan_right_hand_side_ends_one_member_only() {
    // ln(-1) is NaN at the start for the first member.
    let m = build_model("x'=ln(x)\ndone\n");
    let x0 = [-1.0, 2.0];
    for method in [Method::Rk4 { dt: 0.01 }, DP5_STD] {
        let mut j = job(&m, 2, &x0, &[], 1.0, method, Precision::Double);
        j.output = Output::Samples(vec![0.5, 1.0]);
        let r = run(&j);
        assert_eq!(r.status[0], Status::NonFinite, "{method:?}");
        assert_eq!(r.status[1], Status::Ok, "{method:?}");
        assert!(r.state(0, 0)[0].is_nan() && r.state(0, 1)[0].is_nan());
        assert!(r.state(1, 1)[0].is_finite());
        // The healthy member matches a run without the failing one.
        let alone = run(&{
            let mut a = job(&m, 1, &x0[1..], &[], 1.0, method, Precision::Double);
            a.output = Output::Samples(vec![0.5, 1.0]);
            a
        });
        assert_eq!(bits(r.state(1, 1)), bits(alone.state(0, 1)));
    }
}

#[test]
fn growth_past_the_bound_stops_with_bounds() {
    // x' = x^2 from 1 blows up at t = 1.
    let m = build_model("x'=x*x\ndone\n");
    for method in [Method::Rk4 { dt: 0.001 }, DP5_STD] {
        let mut j = job(&m, 1, &[1.0], &[], 2.0, method, Precision::Double);
        j.bounds = 1e3;
        let r = run(&j);
        assert_eq!(r.status[0], Status::Bounds, "{method:?}");
        assert!(r.t_reached[0] < 1.0, "{}", r.t_reached[0]);
        assert!(r.state(0, 0)[0].is_nan());
    }
}

#[test]
fn samples_after_a_failure_stay_nan() {
    let m = build_model("x'=x*x\ndone\n");
    let mut j = job(&m, 1, &[0.5], &[], 4.0, DP5_STD, Precision::Double);
    j.bounds = 1e3;
    // x(t) = 1 / (2 - t), which blows up at t = 2.
    j.output = Output::Samples(vec![1.0, 1.5, 3.0, 4.0]);
    let r = run(&j);
    assert_eq!(r.status[0], Status::Bounds);
    assert!((r.state(0, 0)[0] - 1.0).abs() < 1e-6);
    assert!((r.state(0, 1)[0] - 2.0).abs() < 1e-6);
    assert!(r.state(0, 2)[0].is_nan() && r.state(0, 3)[0].is_nan());
}

#[test]
fn the_step_limit_stops_a_member() {
    let m = build_model(DECAY);
    let mut j = job(
        &m,
        1,
        &[1.0],
        &[1.0],
        10.0,
        Method::Rk4 { dt: 0.001 },
        Precision::Double,
    );
    j.max_steps = 100;
    let r = run(&j);
    assert_eq!(r.status[0], Status::MaxSteps);
    assert_eq!(r.accepted[0], 100);
    assert!((r.t_reached[0] - 0.1).abs() < 1e-12);

    j.method = DP5_STD;
    j.max_steps = 3;
    let r = run(&j);
    assert_eq!(r.status[0], Status::MaxSteps);
    assert_eq!(r.accepted[0], 3);
}

#[test]
fn a_step_below_roundoff_is_reported() {
    // The decay time scale is 1, so the controller asks for steps near 0.1. The step test fails
    // when 0.1 h is below t times the unit roundoff: at t = 1e7 in single precision (spacing 1)
    // and at t = 1e15 in double (spacing 0.125).
    let m = build_model(DECAY);
    for (precision, t0) in [(Precision::Single, 1e7), (Precision::Double, 1e15)] {
        let mut j = job(&m, 1, &[1.0], &[1.0], t0 + 1e3, DP5_SINGLE, precision);
        j.t0 = t0;
        let r = run(&j);
        assert_eq!(r.status[0], Status::StepTooSmall, "{precision:?}");
    }
}

#[test]
fn a_tolerance_below_what_the_precision_resolves_is_an_error() {
    let m = build_model(DECAY);
    let tight = |rtol, precision| {
        let j = job(
            &m,
            1,
            &[1.0],
            &[1.0],
            1.0,
            Method::Dp5 {
                rtol,
                atol: 1e-12,
                hmax: f64::INFINITY,
            },
            precision,
        );
        run_cpu(&j, u64::MAX, &NO_CANCEL())
    };
    assert!(matches!(
        tight(1e-7, Precision::Single),
        Err(EnsembleError::ToleranceTooTight(_))
    ));
    assert!(matches!(
        tight(1e-15, Precision::Double),
        Err(EnsembleError::ToleranceTooTight(_))
    ));
    assert!(tight(1e-5, Precision::Single).is_ok());
    assert!(tight(1e-13, Precision::Double).is_ok());
}

#[test]
fn bad_jobs_are_named() {
    let m = build_model(DECAY);
    let base = || {
        job(
            &m,
            2,
            &[1.0, 2.0],
            &[1.0],
            1.0,
            Method::Rk4 { dt: 0.1 },
            Precision::Double,
        )
    };

    let mut j = base();
    j.t1 = 0.0;
    assert!(bad(run_err(&j)).contains("t1 must be greater"));

    let mut j = base();
    j.members = 0;
    assert!(bad(run_err(&j)).contains("members"));

    let mut j = base();
    j.members = 3;
    assert!(bad(run_err(&j)).contains("x0 has 2 values"));

    let mut j = base();
    j.params = Params::Shared(&[1.0, 2.0]);
    assert!(bad(run_err(&j)).contains("params has 2 values"));

    let mut j = base();
    j.params = Params::PerMember(&[1.0]);
    assert!(bad(run_err(&j)).contains("params has 1 values"));

    let mut j = base();
    j.method = Method::Rk4 { dt: 0.0 };
    assert!(bad(run_err(&j)).contains("dt"));

    let mut j = base();
    j.method = Method::Dp5 {
        rtol: -1.0,
        atol: 1e-9,
        hmax: 1.0,
    };
    assert!(bad(run_err(&j)).contains("rtol"));

    let mut j = base();
    j.bounds = 0.0;
    assert!(bad(run_err(&j)).contains("bounds"));

    let mut j = base();
    j.max_steps = 0;
    assert!(bad(run_err(&j)).contains("max_steps"));

    let mut j = base();
    j.t0 = f64::NAN;
    assert!(bad(run_err(&j)).contains("finite"));

    for s in [
        vec![],
        vec![0.5, 0.5, 1.0],
        vec![0.5, 0.9],
        vec![0.5, 2.0],
        vec![0.0, 1.0],
    ] {
        let mut j = base();
        j.output = Output::Samples(s.clone());
        let msg = bad(run_err(&j));
        assert!(msg.contains("sample"), "{s:?}: {msg}");
    }

    let x_nan = [f64::NAN, 1.0];
    let mut j = base();
    j.x0 = &x_nan;
    assert!(bad(run_err(&j)).contains("x0[0] of member 0"));

    // 1e300 is finite in f64 and infinite in f32.
    let x_big = [1.0, 1e300];
    let mut j = base();
    j.x0 = &x_big;
    j.precision = Precision::Single;
    assert!(bad(run_err(&j)).contains("x0[0] of member 1"));
    j.precision = Precision::Double;
    assert!(run_cpu(&j, u64::MAX, &NO_CANCEL()).is_ok());
}

#[test]
fn samples_that_merge_in_single_precision_are_refused() {
    let m = build_model(DECAY);
    let mut j = job(
        &m,
        1,
        &[1.0],
        &[1.0],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Single,
    );
    j.output = Output::Samples(vec![1.0, 1.0 + 1e-12]);
    // The second sample is beyond t1, so this is refused for that reason in any precision.
    assert!(bad(run_err(&j)).contains("sample"));
    j.t1 = 1.0 + 1e-12;
    assert!(bad(run_err(&j)).contains("single precision"));
    j.precision = Precision::Double;
    assert!(run_cpu(&j, u64::MAX, &NO_CANCEL()).is_ok());
}

#[test]
fn a_construct_the_kernels_cannot_run_is_named() {
    for (src, name) in [
        ("x'=erf(x)\ndone\n", "erf"),
        ("x'=lgamma(x+2)\ndone\n", "lgamma"),
        ("x'=besselj(0,x)\ndone\n", "besselj"),
    ] {
        let m = build_model(src);
        let j = job(
            &m,
            1,
            &[0.5],
            &[],
            1.0,
            Method::Rk4 { dt: 0.1 },
            Precision::Double,
        );
        match run_err(&j) {
            EnsembleError::Unsupported(msg) => {
                assert!(msg.contains(name), "{src}: {msg}");
            }
            other => panic!("{src}: {other:?}"),
        }
    }
}

#[test]
fn a_set_cancel_flag_stops_before_any_work() {
    let m = build_model(DECAY);
    let j = job(
        &m,
        4,
        &[1.0; 4],
        &[1.0],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    let flag = AtomicBool::new(true);
    assert_eq!(
        run_cpu(&j, 10, &flag).unwrap_err(),
        EnsembleError::Cancelled
    );
}

#[test]
fn cancel_during_a_long_run_returns_cancelled() {
    let m = build_model(DECAY);
    let x0 = vec![1.0; 256];
    // About 1e9 steps per member: far longer than the test waits.
    let j = job(
        &m,
        256,
        &x0,
        &[1.0],
        10.0,
        Method::Rk4 { dt: 1e-8 },
        Precision::Double,
    );
    let flag = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let res = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Relaxed);
        });
        run_cpu(&j, 1000, &flag)
    });
    assert_eq!(res.unwrap_err(), EnsembleError::Cancelled);
    assert!(t0.elapsed() < Duration::from_secs(20), "{:?}", t0.elapsed());
}

#[test]
fn zero_steps_per_launch_is_refused() {
    let m = build_model(DECAY);
    let j = job(
        &m,
        1,
        &[1.0],
        &[1.0],
        1.0,
        Method::Rk4 { dt: 0.1 },
        Precision::Double,
    );
    assert!(matches!(
        run_cpu(&j, 0, &NO_CANCEL()),
        Err(EnsembleError::BadJob(_))
    ));
}
