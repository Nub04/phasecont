//! The ensemble map method `iter` (SPEC section 8aa, acceptance tests 1 and 2). Every model
//! is our own, and each test computes the values it expects in the test.

mod common;

use common::*;
use pc_ensemble::job::validate;
use pc_ensemble::{EnsembleError, Method, Output, Precision, Status};

const LOGISTIC_DISCRETE: &str = "par r=2.5\nx'=r*x*(1-x)\n@ meth=discrete\ndone\n";
const SQUARE: &str = "x(t+1)=x^2\ndone\n";
const DOUBLE: &str = "x(t+1)=2*x\ndone\n";

const RK4: Method = Method::Rk4 { dt: 0.1 };

fn bad_text(e: EnsembleError) -> String {
    match e {
        EnsembleError::BadJob(m) => m,
        other => panic!("expected BadJob, got {other:?}"),
    }
}

/// A job on `model` from `t0` to `t1` with the method `iter`.
fn iter_job<'a>(
    model: &'a pc_ir::IrModel,
    x0: &'a [f64],
    params: &'a [f64],
    t0: f64,
    t1: f64,
    precision: Precision,
) -> pc_ensemble::Job<'a> {
    let mut j = job(model, 1, x0, params, t1, Method::Iter, precision);
    j.t0 = t0;
    j
}

#[test]
fn rk4_and_dp5_on_a_map_are_unsupported_and_iter_runs() {
    let want = EnsembleError::Unsupported(
        "the model is a map; rk4 and dp5 integrate an ODE, so run a map with method iter".into(),
    );
    for src in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = build_model(src);
        for method in [RK4, DP5_STD] {
            let j = job(&m, 1, &[0.1], &[2.5], 1.0, method, Precision::Double);
            assert_eq!(validate(&j).unwrap_err(), want, "{src} {method:?}");
        }
        let j = iter_job(&m, &[0.1], &[2.5], 0.0, 10.0, Precision::Double);
        assert_eq!(run(&j).status[0], Status::Ok, "{src}");
    }
}

#[test]
fn iter_on_an_ode_is_a_bad_job() {
    let m = build_model(DECAY);
    let j = iter_job(&m, &[1.0], &[1.0], 0.0, 10.0, Precision::Double);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter iterates a map, and the model is an ODE; use rk4 or dp5"
    );
}

#[test]
fn a_model_of_kind_none_is_unsupported() {
    let mut m = build_model(DECAY);
    m.kind = pc_ode::ModelKind::None;
    let want = EnsembleError::Unsupported("the model has no ODE and no map".into());
    for method in [RK4, DP5_STD, Method::Iter] {
        let j = job(&m, 1, &[1.0], &[1.0], 1.0, method, Precision::Double);
        assert_eq!(validate(&j).unwrap_err(), want, "{method:?}");
    }
}

#[test]
fn iter_needs_whole_number_times() {
    let m = build_model(LOGISTIC_MAP);
    let j = iter_job(&m, &[0.1], &[2.5], 0.5, 10.0, Precision::Double);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter needs whole-number times, got 0.5"
    );
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, 10.5, Precision::Double);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter needs whole-number times, got 10.5"
    );
    let mut j = iter_job(&m, &[0.1], &[2.5], 0.0, 10.0, Precision::Double);
    j.output = Output::Samples(vec![2.5, 10.0]);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter needs whole-number times, got 2.5"
    );
}

#[test]
fn iter_times_have_a_magnitude_limit_per_precision() {
    let m = build_model(LOGISTIC_MAP);
    let two24 = 16_777_216.0_f64;
    let two53 = 9_007_199_254_740_992.0_f64;
    let single = "method iter in single precision needs times of magnitude at most 16777216, got";
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, two24 + 1.0, Precision::Single);
    let e = bad_text(validate(&j).unwrap_err());
    assert_eq!(e, format!("{single} 16777217"));
    let j = iter_job(&m, &[0.1], &[2.5], -two24 - 1.0, 0.0, Precision::Single);
    let e = bad_text(validate(&j).unwrap_err());
    assert_eq!(e, format!("{single} -16777217"));
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, two53 + 2.0, Precision::Double);
    let e = bad_text(validate(&j).unwrap_err());
    assert_eq!(
        e,
        "method iter in double precision needs times of magnitude at most 9007199254740992, got 9007199254740994"
    );
    // The largest single-precision time passes.
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, two24, Precision::Single);
    assert!(validate(&j).is_ok());
}

#[test]
fn iter_has_a_per_segment_iterate_limit_per_precision() {
    let m = build_model(LOGISTIC_MAP);
    let two24 = 16_777_216.0_f64;
    let two32 = 4_294_967_296.0_f64;
    // Both ends are within 2^24, but the one segment makes 2^25 iterates.
    let j = iter_job(&m, &[0.1], &[2.5], -two24, two24, Precision::Single);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter makes 33554432 iterates between -16777216 and 16777216, over the limit of 16777216 per sample interval in single precision"
    );
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, two32, Precision::Double);
    assert_eq!(
        bad_text(validate(&j).unwrap_err()),
        "method iter makes 4294967296 iterates between 0 and 4294967296, over the limit of 4294967295 per sample interval in double precision"
    );
    let j = iter_job(&m, &[0.1], &[2.5], 0.0, two32 - 1.0, Precision::Double);
    assert!(validate(&j).is_ok());
    // The limit is per sample interval.
    let mut j = iter_job(&m, &[0.1], &[2.5], -two24, two24, Precision::Single);
    j.output = Output::Samples(vec![0.0, two24]);
    assert!(validate(&j).is_ok());
}

/// The logistic map r = 2.5 from 0.1, iterated by hand in `f64`.
fn logistic_loop(n: usize) -> Vec<f64> {
    let r = 2.5_f64;
    let mut x = 0.1_f64;
    let mut xs = vec![x];
    for _ in 0..n {
        x = (r * x) * (1.0 - x);
        xs.push(x);
    }
    xs
}

#[test]
fn the_logistic_map_matches_a_hand_loop_bit_for_bit() {
    let xs = logistic_loop(50);
    for src in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = build_model(src);
        let mut j = iter_job(&m, &[0.1], &[2.5], 0.0, 50.0, Precision::Double);
        j.output = Output::Samples(vec![10.0, 20.0, 50.0]);
        let r = run(&j);
        assert_eq!(r.status[0], Status::Ok, "{src}");
        assert_eq!(r.t_reached[0], 50.0);
        assert_eq!((r.accepted[0], r.rejected[0]), (50, 0));
        for (s, k) in [10, 20, 50].into_iter().enumerate() {
            assert_eq!(r.state(0, s)[0].to_bits(), xs[k].to_bits(), "{src} n = {k}");
        }
        let fin = run(&iter_job(&m, &[0.1], &[2.5], 0.0, 50.0, Precision::Double));
        assert_eq!(fin.state(0, 0)[0].to_bits(), xs[50].to_bits(), "{src}");
    }
}

#[test]
fn the_henon_map_matches_a_hand_loop_bit_for_bit() {
    let (a, b) = (1.4_f64, 0.3_f64);
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    let mut xs = vec![[x, y]];
    for _ in 0..100 {
        (x, y) = (1.0 - a * (x * x) + y, b * x);
        xs.push([x, y]);
    }
    let m = build_model(HENON_MAP);
    let par = [a, b];
    let mut j = iter_job(&m, &[0.0, 0.0], &par, 0.0, 100.0, Precision::Double);
    j.output = Output::Samples(vec![10.0, 20.0, 100.0]);
    let r = run(&j);
    assert_eq!(r.status[0], Status::Ok);
    for (s, k) in [10, 20, 100].into_iter().enumerate() {
        assert_eq!(bits(r.state(0, s)), bits(&xs[k]), "n = {k}");
    }
}

#[test]
fn t_inside_the_map_is_the_number_of_the_iterate() {
    let m = build_model(SHIFT_MAP);
    let mut j = iter_job(&m, &[0.0], &[], 0.0, 4.0, Precision::Double);
    j.output = Output::Samples(vec![1.0, 2.0, 3.0, 4.0]);
    let r = run(&j);
    let got: Vec<f64> = (0..4).map(|s| r.state(0, s)[0]).collect();
    assert_eq!(got, [0.0, 1.0, 3.0, 6.0]);
    // From n0 = 5 the iterates map 5, 6, ... so x(9) = 5 + 6 + 7 + 8.
    let r = run(&iter_job(&m, &[0.0], &[], 5.0, 9.0, Precision::Double));
    assert_eq!(r.state(0, 0)[0], 26.0);
    for p in [Precision::Single, Precision::Double] {
        let mut j = iter_job(&m, &[0.0], &[], 0.0, 4.0, p);
        j.output = Output::Samples(vec![1.0, 2.0, 3.0, 4.0]);
        let r = run(&j);
        let got: Vec<f64> = (0..4).map(|s| r.state(0, s)[0]).collect();
        assert_eq!(got, [0.0, 1.0, 3.0, 6.0], "{p:?}");
    }
}

#[test]
fn the_logistic_map_in_single_precision_matches_an_f32_loop() {
    let r = 2.5_f32;
    let mut x = 0.1_f64 as f32;
    let mut xs = vec![x];
    for _ in 0..50 {
        x = (r * x) * (1.0 - x);
        xs.push(x);
    }
    let m = build_model(LOGISTIC_MAP);
    let mut j = iter_job(&m, &[0.1], &[2.5], 0.0, 50.0, Precision::Single);
    j.output = Output::Samples(vec![10.0, 20.0, 50.0]);
    let res = run(&j);
    assert_eq!(res.status[0], Status::Ok);
    for (s, k) in [10, 20, 50].into_iter().enumerate() {
        assert_eq!(
            res.state(0, s)[0].to_bits(),
            f64::from(xs[k]).to_bits(),
            "n = {k}"
        );
    }
}

#[test]
fn an_overflow_stops_with_not_finite_after_the_last_kept_iterate() {
    // 10^(2^8) = 1e256 is finite in double precision and 10^(2^9) is not; in single
    // precision 10^(2^5) = 1e32 is finite and 10^(2^6) is not.
    let m = build_model(SQUARE);
    for (p, kept) in [(Precision::Double, 8u32), (Precision::Single, 5u32)] {
        let mut j = iter_job(&m, &[10.0], &[], 0.0, 20.0, p);
        j.output = Output::Samples(vec![4.0, 10.0, 20.0]);
        let r = run(&j);
        assert_eq!(r.status[0], Status::NonFinite, "{p:?}");
        assert_eq!(r.t_reached[0], f64::from(kept), "{p:?}");
        assert_eq!((r.accepted[0], r.rejected[0]), (kept, 0), "{p:?}");
        assert!(r.state(0, 0)[0].is_finite(), "{p:?}");
        assert!(
            r.state(0, 1)[0].is_nan() && r.state(0, 2)[0].is_nan(),
            "{p:?}"
        );
    }
}

#[test]
fn max_steps_caps_the_iterates_of_the_run() {
    let m = build_model(DOUBLE);
    let mut j = iter_job(&m, &[1.0], &[], 0.0, 10.0, Precision::Double);
    j.max_steps = 5;
    let r = run(&j);
    assert_eq!(r.status[0], Status::MaxSteps);
    assert_eq!((r.t_reached[0], r.accepted[0]), (5.0, 5));
    // A cap of t1 - t0 iterates ends ok.
    j.max_steps = 10;
    let r = run(&j);
    assert_eq!(r.status[0], Status::Ok);
    assert_eq!(r.state(0, 0)[0], 1024.0);
}

#[test]
fn a_bound_stops_the_member_with_bounds() {
    // 2^9 = 512 is within 1e3 and 2^10 = 1024 is not.
    let m = build_model(DOUBLE);
    let mut j = iter_job(&m, &[1.0], &[], 0.0, 20.0, Precision::Double);
    j.bounds = 1e3;
    let r = run(&j);
    assert_eq!(r.status[0], Status::Bounds);
    assert_eq!((r.t_reached[0], r.accepted[0]), (9.0, 9));
    assert!(r.state(0, 0)[0].is_nan());
}
