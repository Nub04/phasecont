//! Background runs: streaming, cancel, drop, switching.

mod common;

use common::load;
use pc_engine::{EngineError, EpSetup, Evaluator, Options, Run, RunSpec, RunState};
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
    let b = wait_for(&run, |s| s.points[0] > n1 + 50);
    assert_eq!(b.state, RunState::Running);
    assert_eq!(b.specials[0], 0, "specials appear only at the end");
    let pts = run.points(0, 0, 3);
    assert_eq!(pts.len(), 3);
    assert!(run.points(5, 0, 3).is_empty());
    run.cancel();
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
