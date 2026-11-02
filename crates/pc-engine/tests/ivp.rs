//! The initial value problem integrator: accuracy, direction, stops and inputs.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, Evaluator, IvpOptions, IvpSetup, ModelEval, StopReason, build,
    build_allowing_t, integrate,
};
use std::sync::atomic::{AtomicBool, Ordering};

fn model(name: &str, allow_t: bool) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    if allow_t {
        build_allowing_t(&m, &info, Evaluator::Jit).unwrap()
    } else {
        build(&m, &info, Evaluator::Jit).unwrap()
    }
}

fn setup(x0: &[f64], params: &[f64], t0: f64, t1: f64) -> IvpSetup {
    IvpSetup {
        x0: x0.to_vec(),
        params: params.to_vec(),
        t0,
        t1,
    }
}

fn tol(rtol: f64, atol: f64) -> IvpOptions {
    IvpOptions {
        rtol,
        atol,
        ..IvpOptions::default()
    }
}

fn run(m: &mut dyn ModelEval, s: &IvpSetup, o: &IvpOptions) -> Branch {
    integrate(m, s, o, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// The bits of a point's time, state and step, since the NaN test values break `==`.
fn bits(p: &pc_engine::Point) -> Vec<u64> {
    let mut v = vec![p.alpha.to_bits(), p.h.to_bits()];
    v.extend(p.u.iter().map(|x| x.to_bits()));
    v
}

fn last(b: &Branch) -> (f64, &[f64]) {
    let p = b.points.last().unwrap();
    (p.alpha, &p.u)
}

fn err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn rotation_matches_the_exact_solution_and_lands_on_t1() {
    let mut m = model("ivp_rot", false);
    let t1 = 10.0 * std::f64::consts::PI;
    let b = run(
        &mut *m,
        &setup(&[1.0, 0.0], &[], 0.0, t1),
        &tol(1e-10, 1e-12),
    );
    assert_eq!(b.stop, StopReason::End);
    let (t, x) = last(&b);
    assert_eq!(t.to_bits(), t1.to_bits(), "the last point is exactly t1");
    assert!(err(x, &[t1.cos(), -t1.sin()]) < 1e-8, "x = {x:?}");
    // The start point is the data, bit for bit.
    let first = &b.points[0];
    assert_eq!((first.alpha, first.u.as_slice()), (0.0, &[1.0, 0.0][..]));
    assert_eq!(first.h, 0.0);
    // Time increases and every point carries the IVP layout.
    for w in b.points.windows(2) {
        assert!(w[1].alpha > w[0].alpha);
        assert!(w[1].h > 0.0);
    }
    for p in &b.points {
        assert!(p.tangent.is_empty() && p.eigenvalues.is_empty() && p.mesh.is_empty());
        assert!(p.tests.iter().all(|v| v.is_nan()));
    }
}

#[test]
fn error_shrinks_with_the_tolerance() {
    let mut m = model("ivp_rot", false);
    let t1: f64 = 20.0;
    let exact = [t1.cos(), -t1.sin()];
    let mut errs = Vec::new();
    let mut steps = Vec::new();
    for e in [6, 8, 10] {
        let r = 10f64.powi(-e);
        let b = run(
            &mut *m,
            &setup(&[1.0, 0.0], &[], 0.0, t1),
            &tol(r, r * 1e-2),
        );
        errs.push(err(last(&b).1, &exact));
        steps.push(b.points.len());
    }
    assert!(
        errs[0] > 5.0 * errs[1] && errs[1] > 5.0 * errs[2],
        "{errs:?}"
    );
    assert!(steps[0] < steps[1] && steps[1] < steps[2], "{steps:?}");
}

#[test]
fn logistic_growth_matches_the_closed_form() {
    let mut m = model("ivp_logistic", false);
    let (x0, r, t1) = (0.1, 1.5, 10.0);
    let b = run(&mut *m, &setup(&[x0], &[r], 0.0, t1), &tol(1e-10, 1e-12));
    assert_eq!(b.stop, StopReason::End);
    let exact = 1.0 / (1.0 + (1.0 / x0 - 1.0) * (-r * t1).exp());
    assert!((last(&b).1[0] - exact).abs() < 1e-9);
    for p in &b.points {
        let e = 1.0 / (1.0 + (1.0 / x0 - 1.0) * (-r * p.alpha).exp());
        assert!((p.u[0] - e).abs() < 1e-8, "t = {}", p.alpha);
    }
}

#[test]
fn forced_model_reads_time_from_the_integrator() {
    // x' = -x + sin t, x = (sin t - cos t) / 2 + (x0 + 1/2) exp(-t).
    let exact = |x0: f64, t: f64| 0.5 * (t.sin() - t.cos()) + (x0 + 0.5) * (-t).exp();
    let mut m = model("nonautonomous", true);
    let b = run(&mut *m, &setup(&[0.0], &[], 0.0, 12.0), &tol(1e-10, 1e-12));
    assert_eq!(b.stop, StopReason::End);
    for p in &b.points {
        assert!(
            (p.u[0] - exact(0.0, p.alpha)).abs() < 1e-8,
            "t = {}",
            p.alpha
        );
    }
    // A start at t0 = 3 uses t0, not zero.
    let x0 = exact(0.0, 3.0);
    let b = run(&mut *m, &setup(&[x0], &[], 3.0, 9.0), &tol(1e-10, 1e-12));
    assert!((last(&b).1[0] - exact(0.0, 9.0)).abs() < 1e-8);
}

#[test]
fn an_autonomous_run_refuses_a_forced_model() {
    let (m, info) = load("nonautonomous");
    assert!(matches!(
        build(&m, &info, Evaluator::Jit).err(),
        Some(EngineError::NonAutonomous)
    ));
}

#[test]
fn backward_integration_returns_to_the_start() {
    let mut m = model("ivp_rot", false);
    let x0 = [0.3, -0.8];
    let o = tol(1e-11, 1e-13);
    let fwd = run(&mut *m, &setup(&x0, &[], 0.0, 7.0), &o);
    let end = last(&fwd).1.to_vec();
    let back = run(&mut *m, &setup(&end, &[], 7.0, 0.0), &o);
    assert_eq!(back.stop, StopReason::End);
    for w in back.points.windows(2) {
        assert!(w[1].alpha < w[0].alpha, "time decreases");
        assert!(w[1].h < 0.0, "h carries the direction");
    }
    let (t, x) = last(&back);
    assert_eq!(t, 0.0);
    assert!(err(x, &x0) < 1e-8, "x = {x:?}");
}

#[test]
fn backward_forced_model_matches_the_exact_solution() {
    let exact = |t: f64| 0.5 * (t.sin() - t.cos()) + 0.5 * (-t).exp();
    let mut m = model("nonautonomous", true);
    let b = run(
        &mut *m,
        &setup(&[exact(5.0)], &[], 5.0, -2.0),
        &tol(1e-10, 1e-12),
    );
    assert_eq!(b.stop, StopReason::End);
    let (t, x) = last(&b);
    assert_eq!(t, -2.0);
    assert!((x[0] - exact(-2.0)).abs() < 1e-7, "x = {}", x[0]);
}

#[test]
fn lands_on_t1_for_awkward_end_times() {
    let mut m = model("ivp_logistic", false);
    for t1 in [1e-9, 0.3, 1.0 / 3.0, std::f64::consts::E, 123.456, -0.7] {
        let b = run(
            &mut *m,
            &setup(&[0.1], &[1.5], 0.0, t1),
            &IvpOptions::default(),
        );
        assert_eq!(b.stop, StopReason::End, "t1 = {t1}");
        assert_eq!(last(&b).0.to_bits(), t1.to_bits(), "t1 = {t1}");
    }
}

#[test]
fn hmax_caps_every_step() {
    let mut m = model("ivp_logistic", false);
    let o = IvpOptions {
        hmax: 0.01,
        ..IvpOptions::default()
    };
    let b = run(&mut *m, &setup(&[0.1], &[1.5], 0.0, 1.0), &o);
    assert!(b.points.len() >= 101, "{} points", b.points.len());
    assert!(b.points.iter().all(|p| p.h.abs() <= 0.01));
}

#[test]
fn max_steps_stops_with_that_many_accepted_steps() {
    let mut m = model("ivp_rot", false);
    let o = IvpOptions {
        max_steps: 5,
        ..IvpOptions::default()
    };
    let b = run(&mut *m, &setup(&[1.0, 0.0], &[], 0.0, 100.0), &o);
    assert_eq!(b.stop, StopReason::MaxSteps);
    assert_eq!(b.points.len(), 6, "the start plus 5 accepted steps");
    // Reaching t1 on the last allowed step is an End, not a MaxSteps.
    let n = run(
        &mut *m,
        &setup(&[1.0, 0.0], &[], 0.0, 1.0),
        &IvpOptions::default(),
    )
    .points
    .len()
        - 1;
    let o = IvpOptions {
        max_steps: n,
        ..IvpOptions::default()
    };
    let b = run(&mut *m, &setup(&[1.0, 0.0], &[], 0.0, 1.0), &o);
    assert_eq!(b.stop, StopReason::End);
}

#[test]
fn blow_up_stops_before_the_singularity() {
    // x' = x^2 from 1 reaches infinity at t = 1.
    let mut m = model("ivp_square", false);
    let b = run(
        &mut *m,
        &setup(&[1.0], &[], 0.0, 2.0),
        &IvpOptions::default(),
    );
    assert!(
        matches!(
            b.stop,
            StopReason::StepTooSmall | StopReason::Error(EngineError::NotFinite(_))
        ),
        "{:?}",
        b.stop
    );
    // The step size follows the relative error, so the run may cross the pole by a few
    // steps before it stops; it never reaches t = 2.
    let (t, x) = last(&b);
    assert!((t - 1.0).abs() < 1e-6, "t = {t}");
    assert!(x[0].abs() > 1e6, "x = {}", x[0]);
    eprintln!(
        "blow-up: stop {:?}, t = {t:e}, x = {:e}, {} points",
        b.stop,
        x[0],
        b.points.len()
    );
}

#[test]
fn a_non_finite_start_derivative_ends_the_run_at_the_start() {
    let mut m = model("ivp_square", false);
    let b = run(
        &mut *m,
        &setup(&[1e160], &[], 0.0, 1.0),
        &IvpOptions::default(),
    );
    assert!(matches!(
        b.stop,
        StopReason::Error(EngineError::NotFinite(_))
    ));
    assert_eq!(b.points.len(), 1);
}

#[test]
fn an_equilibrium_start_runs_to_t1_unchanged() {
    let mut m = model("ivp_rot", false);
    let b = run(
        &mut *m,
        &setup(&[0.0, 0.0], &[], 0.0, 50.0),
        &IvpOptions::default(),
    );
    assert_eq!(b.stop, StopReason::End);
    assert!(b.points.iter().all(|p| p.u == [0.0, 0.0]));
    assert_eq!(last(&b).0, 50.0);
    assert!(b.points.len() < 100, "{} points", b.points.len());
}

#[test]
fn cancel_stops_between_steps_and_before_the_start() {
    let mut m = model("ivp_rot", false);
    let s = setup(&[1.0, 0.0], &[], 0.0, 1e6);
    let cancel = AtomicBool::new(false);
    let mut n = 0;
    let b = integrate(&mut *m, &s, &IvpOptions::default(), &cancel, &mut |_| {
        n += 1;
        if n == 10 {
            cancel.store(true, Ordering::Relaxed);
        }
    })
    .unwrap();
    assert_eq!(b.stop, StopReason::Cancelled);
    assert_eq!(b.points.len(), 10);
    let pre = AtomicBool::new(true);
    let r = integrate(&mut *m, &s, &IvpOptions::default(), &pre, &mut |_| {});
    assert_eq!(r.err(), Some(EngineError::Cancelled));
}

#[test]
fn points_stream_in_order_and_match_the_branch() {
    let mut m = model("ivp_logistic", false);
    let s = setup(&[0.1], &[1.5], 0.0, 4.0);
    let mut seen = Vec::new();
    let b = integrate(
        &mut *m,
        &s,
        &IvpOptions::default(),
        &AtomicBool::new(false),
        &mut |p| seen.push(p.clone()),
    )
    .unwrap();
    assert_eq!(seen.len(), b.points.len());
    for (a, c) in seen.iter().zip(&b.points) {
        assert_eq!(bits(a), bits(c));
    }
}

#[test]
fn two_runs_agree_bit_for_bit() {
    let mut m = model("ivp_rot", false);
    let s = setup(&[1.0, 0.0], &[], 0.0, 30.0);
    let a = run(&mut *m, &s, &IvpOptions::default());
    let b = run(&mut *m, &s, &IvpOptions::default());
    assert_eq!(a.stop, b.stop);
    assert_eq!(a.points.len(), b.points.len());
    for (p, q) in a.points.iter().zip(&b.points) {
        assert_eq!(bits(p), bits(q));
    }
}

#[test]
fn bad_inputs_are_refused() {
    let mut m = model("ivp_logistic", false);
    let good = setup(&[0.1], &[1.5], 0.0, 1.0);
    let mut go = |s: &IvpSetup, o: &IvpOptions| {
        integrate(&mut *m, s, o, &AtomicBool::new(false), &mut |_| {}).err()
    };
    let d = IvpOptions::default();
    let bad =
        |e: Option<EngineError>| assert!(matches!(e, Some(EngineError::BadProblem(_))), "{e:?}");
    bad(go(&setup(&[0.1, 0.2], &[1.5], 0.0, 1.0), &d));
    bad(go(&setup(&[0.1], &[], 0.0, 1.0), &d));
    bad(go(&setup(&[0.1], &[1.5, 2.0], 0.0, 1.0), &d));
    bad(go(&setup(&[0.1], &[1.5], 1.0, 1.0), &d));
    bad(go(&setup(&[0.1], &[1.5], f64::NAN, 1.0), &d));
    bad(go(&setup(&[0.1], &[1.5], 0.0, f64::INFINITY), &d));
    bad(go(&setup(&[f64::NAN], &[1.5], 0.0, 1.0), &d));
    bad(go(&setup(&[0.1], &[f64::INFINITY], 0.0, 1.0), &d));
    for o in [
        IvpOptions {
            rtol: 0.0,
            ..d.clone()
        },
        IvpOptions {
            rtol: f64::NAN,
            ..d.clone()
        },
        IvpOptions {
            atol: -1.0,
            ..d.clone()
        },
        IvpOptions {
            atol: 0.0,
            ..d.clone()
        },
        IvpOptions {
            hmax: 0.0,
            ..d.clone()
        },
        IvpOptions {
            hmax: f64::NAN,
            ..d.clone()
        },
        IvpOptions {
            max_steps: 0,
            ..d.clone()
        },
    ] {
        bad(go(&good, &o));
    }
}

#[test]
fn defaults_are_the_documented_ones() {
    let d = IvpOptions::default();
    assert_eq!((d.rtol, d.atol, d.max_steps), (1e-8, 1e-10, 100_000));
    assert!(d.hmax.is_infinite());
}
