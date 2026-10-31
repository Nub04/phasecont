//! Background runs: streaming, cancel, drop, switching.

mod common;

use common::load;
use pc_engine::{
    CurveSetup, EngineError, EpSetup, Evaluator, Options, Run, RunKind, RunSpec, RunState,
    SpecialPoint,
};
use std::time::{Duration, Instant};

/// Lorenz from the origin along r up to 30 with a small step: a long run.
fn long_spec() -> RunSpec {
    let (model, info) = load("lorenz");
    let setup = EpSetup::new(1, vec![10.0, 0.5, 8.0 / 3.0], vec![0.0; 3]);
    let opts = Options {
        h0: 1e-4,
        hmax: 1e-4,
        max_steps: 10_000_000,
        bounds: (f64::NEG_INFINITY, 30.0),
        ..Options::default()
    };
    RunSpec::new(model, info, Evaluator::Jit, setup, opts)
}

fn wait_for(run: &Run, mut f: impl FnMut(&pc_engine::Status) -> bool) -> pc_engine::Status {
    let t0 = Instant::now();
    loop {
        let st = run.poll();
        if f(&st) {
            return st;
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "timed out: {st:?}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn poll_sees_points_grow_during_a_long_run() {
    let run = Run::start(long_spec());
    let a = wait_for(&run, |s| s.points.first().is_some_and(|&n| n > 0));
    assert_eq!(a.state, RunState::Running);
    let n1 = a.points[0];
    let mut last = n1;
    let b = wait_for(&run, |s| {
        assert!(s.points[0] >= last, "point count went down");
        last = s.points[0];
        s.points[0] > n1 + 50
    });
    assert_eq!(b.state, RunState::Running);
    assert_eq!(b.specials[0], 0, "specials appear only at the end");
    let pts = run.points(0, 0, 3);
    assert_eq!(pts.len(), 3);
    assert!(run.points(5, 0, 3).is_empty());
    run.cancel();
    let fin = run.join();
    assert!(
        fin.points[0] >= b.points[0],
        "final count below a streamed count"
    );
}

#[test]
fn cancel_ends_in_cancelled_within_a_second() {
    let run = Run::start(long_spec());
    wait_for(&run, |s| s.points.first().is_some_and(|&n| n > 10));
    let t0 = Instant::now();
    run.cancel();
    let st = run.join();
    let dt = t0.elapsed();
    assert_eq!(st.state, RunState::Cancelled);
    assert!(dt < Duration::from_secs(1), "cancel took {dt:?}");
}

#[test]
fn dropping_a_running_run_returns_promptly() {
    let run = Run::start(long_spec());
    wait_for(&run, |s| s.points.first().is_some_and(|&n| n > 10));
    let t0 = Instant::now();
    drop(run);
    let dt = t0.elapsed();
    assert!(dt < Duration::from_secs(1), "drop took {dt:?}");
}

#[test]
fn a_short_run_finishes_done_with_specials() {
    let mut spec = long_spec();
    spec.opts.h0 = 0.05;
    spec.opts.hmax = 0.05;
    spec.opts.bounds = (f64::NEG_INFINITY, 1.5);
    let st = Run::start(spec).join();
    assert_eq!(st.state, RunState::Done);
    assert_eq!(st.points.len(), 1);
    assert_eq!(st.specials[0], 1, "the pitchfork at r = 1");
}

#[test]
fn switching_at_bp1_follows_a_second_branch() {
    let mut spec = long_spec();
    spec.opts.h0 = 0.05;
    spec.opts.hmax = 0.05;
    spec.opts.max_steps = 30;
    spec.opts.bounds = (-5.0, 1.5);
    spec.switch_at = Some("BP1".into());
    let run = Run::start(spec);
    let st = {
        let t0 = Instant::now();
        while run.poll().state == RunState::Running {
            assert!(t0.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(2));
        }
        run.poll()
    };
    assert_eq!(st.state, RunState::Done, "{st:?}");
    assert_eq!(st.points.len(), 2);
    assert!(st.points[1] > 0);
    // The Lorenz nontrivial branch has x = y = +-sqrt(b (r - 1)).
    let pts = run.points(1, 0, usize::MAX);
    assert!(pts.iter().any(|p| p.u[0].abs() > 1e-3), "stayed on x = 0");
}

#[test]
fn switch_at_a_missing_special_fails_and_keeps_the_first_branch() {
    let mut spec = long_spec();
    spec.opts.h0 = 0.05;
    spec.opts.hmax = 0.05;
    spec.opts.bounds = (f64::NEG_INFINITY, 0.8);
    spec.switch_at = Some("BP1".into());
    let run = Run::start(spec);
    let t0 = Instant::now();
    while run.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(2));
    }
    let st = run.poll();
    assert!(matches!(
        st.state,
        RunState::Failed(EngineError::BadProblem(_))
    ));
    assert_eq!(st.points.len(), 1);
    assert!(st.points[0] > 0);
    assert!(!run.points(0, 0, 1).is_empty());
}

#[test]
fn a_bad_model_setup_fails() {
    let mut spec = long_spec();
    spec.setup.u0 = vec![0.0];
    let st = Run::start(spec).join();
    assert!(matches!(st.state, RunState::Failed(_)), "{st:?}");
}

#[test]
fn cancel_right_after_start_ends_cancelled() {
    let run = Run::start(long_spec());
    run.cancel();
    assert_eq!(run.join().state, RunState::Cancelled);
}

#[test]
fn cancel_after_done_leaves_done() {
    let mut spec = long_spec();
    spec.opts.h0 = 0.05;
    spec.opts.hmax = 0.05;
    spec.opts.bounds = (f64::NEG_INFINITY, 0.8);
    let run = Run::start(spec);
    wait_for(&run, |s| s.state != RunState::Running);
    run.cancel();
    assert_eq!(run.join().state, RunState::Done);
}

/// The first special point called `label` on the equilibrium branch of `model` in the
/// parameter `free`, found through a run like any other.
fn ep_special(
    model: &str,
    free: usize,
    params: Vec<f64>,
    u0: Vec<f64>,
    bounds: (f64, f64),
    label: &str,
) -> SpecialPoint {
    let (m, info) = load(model);
    let opts = Options {
        bounds,
        ..Options::default()
    };
    let run = Run::start(RunSpec::new(
        m,
        info,
        Evaluator::Jit,
        EpSetup::new(free, params, u0),
        opts,
    ));
    wait_for(&run, |s| s.state != RunState::Running);
    run.specials(0)
        .into_iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no {label} on the {model} branch"))
}

#[test]
fn an_lp_curve_run_streams_points_and_finds_the_cusp() {
    let lp = ep_special("cusp", 0, vec![0.0, 1.0], vec![-1.0], (-1.0, 1.0), "LP");
    let (m, info) = load("cusp");
    let setup = CurveSetup::new([0, 1], vec![lp.point.alpha, 1.0], lp.point.u.clone());
    let opts = Options {
        bounds: (-0.5, 1.5),
        ..Options::default()
    };
    let spec = RunSpec::curve(m, info, Evaluator::Jit, RunKind::LpCurve(setup), opts);
    let run = Run::start(spec);
    let st = wait_for(&run, |s| s.state != RunState::Running);
    assert_eq!(st.state, RunState::Done);
    assert_eq!(st.points.len(), 1);
    let pts = run.points(0, 0, 1000);
    assert!(pts.len() > 5);
    // On a curve Point.u is (x, b1) and Point.alpha is b2.
    assert!(pts.iter().all(|p| p.u.len() == 2));
    let cp: Vec<_> = run
        .specials(0)
        .into_iter()
        .filter(|s| s.label == "CP")
        .collect();
    assert_eq!(cp.len(), 1);
    assert!(cp[0].extras.refined);
    assert!(cp[0].point.alpha.abs() < 1e-10);
}

#[test]
fn a_hopf_curve_run_follows_the_brusselator_curve() {
    let h = ep_special(
        "brusselator",
        1,
        vec![1.0, 1.2],
        vec![1.0, 1.2],
        (0.5, 4.0),
        "H",
    );
    let (m, info) = load("brusselator");
    let setup = CurveSetup::new([0, 1], vec![1.0, h.point.alpha], h.point.u.clone());
    let opts = Options {
        bounds: (1.5, 5.0),
        ..Options::default()
    };
    let spec = RunSpec::curve(m, info, Evaluator::Jit, RunKind::HopfCurve(setup), opts);
    let run = Run::start(spec);
    let st = wait_for(&run, |s| s.state != RunState::Running);
    assert_eq!(st.state, RunState::Done);
    let pts = run.points(0, 0, 1000);
    assert!(pts.len() > 5);
    // Point.u is (x, y, a, kappa) and Point.alpha is b; the curve is b = 1 + a^2.
    for p in &pts {
        assert_eq!(p.u.len(), 4);
        assert!((p.alpha - (1.0 + p.u[2] * p.u[2])).abs() < 1e-9);
    }
}

#[test]
fn a_curve_run_cannot_switch_branches() {
    let lp = ep_special("cusp", 0, vec![0.0, 1.0], vec![-1.0], (-1.0, 1.0), "LP");
    let (m, info) = load("cusp");
    let setup = CurveSetup::new([0, 1], vec![lp.point.alpha, 1.0], lp.point.u.clone());
    let mut spec = RunSpec::curve(
        m,
        info,
        Evaluator::Jit,
        RunKind::LpCurve(setup),
        Options::default(),
    );
    spec.switch_at = Some("BP1".into());
    let st = Run::start(spec).join();
    assert!(
        matches!(st.state, RunState::Failed(EngineError::BadProblem(_))),
        "{:?}",
        st.state
    );
}
