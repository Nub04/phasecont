//! Pseudo-arclength continuation of equilibria: step control, stop reasons and branches
//! of the circle and fold models, each on both evaluators.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, EpSetup, Evaluator, Options, Problem, StopReason, build, continue_branch,
    continue_ep, equilibrium,
};
use std::sync::atomic::{AtomicBool, Ordering};

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn run(
    name: &str,
    which: Evaluator,
    setup: &EpSetup,
    opts: &Options,
) -> Result<Branch, EngineError> {
    let (model, info) = load(name);
    let mut m = build(&model, &info, which).unwrap();
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
}

fn circle_setup() -> EpSetup {
    EpSetup::new(0, vec![0.0], vec![-1.0])
}

fn on_circle(b: &Branch) {
    for (k, p) in b.points.iter().enumerate() {
        let r = (p.u[0] * p.u[0] + p.alpha * p.alpha - 1.0).abs();
        assert!(r <= 1e-10, "point {k} is off the circle by {r:e}");
        let t = &p.tangent;
        let len = (t[0] * t[0] + t[1] * t[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-12, "tangent {k} has length {len}");
    }
}

#[test]
fn circle_goes_round_and_returns() {
    for which in BOTH {
        // With hmax = 1e-3 the points are at most about 1e-3 apart on the circle, so
        // the one nearest the start is within 1e-3 of it (inf-norm of (x, alpha)).
        let opts = Options {
            h0: 1e-3,
            hmax: 1e-3,
            max_steps: 6500,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps, "{which:?}");
        on_circle(&b);
        // Closure: after leaving the start (distance > 0.5), a point comes back within
        // 1e-3 of it. The circle has length 2 pi, so this takes about 6283 steps.
        let near = |p: &pc_engine::Point| (p.u[0] + 1.0).abs().max(p.alpha.abs());
        let left = b.points.iter().position(|p| near(p) > 0.5).unwrap();
        let back = b.points[left..].iter().position(|p| near(p) <= 1e-3);
        assert!(back.is_some(), "{which:?}: the branch did not return");
        let amax = b.points.iter().map(|p| p.alpha).fold(f64::MIN, f64::max);
        let amin = b.points.iter().map(|p| p.alpha).fold(f64::MAX, f64::min);
        assert!(amax > 1.0 - 1e-5 && amin < -1.0 + 1e-5, "{amin} {amax}");
        eprintln!(
            "circle {which:?}: {} points, {} steps, returned at point {}",
            b.points.len(),
            b.points.len() - 1,
            left + back.unwrap()
        );
    }
}

#[test]
fn circle_with_default_steps_stays_on_the_circle() {
    for which in BOTH {
        let opts = Options {
            max_steps: 200,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps);
        assert_eq!(b.points.len(), 201);
        on_circle(&b);
        // One state variable: one eigenvalue, no pairs, so psi_H = 1.
        assert!(b.points.iter().all(|p| p.eigenvalues.len() == 1));
        assert!(b.points.iter().all(|p| p.tests[2] == 1.0));
        assert!(
            b.points
                .iter()
                .all(|p| p.tests[..2].iter().all(|t| t.is_finite()))
        );
        // The first tangent points toward growing alpha.
        assert!(b.points[0].tangent[1] > 0.99);
    }
}

#[test]
fn direction_flips_the_first_tangent() {
    for which in BOTH {
        let opts = Options {
            direction: -1.0,
            max_steps: 3,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert!(b.points[0].tangent[1] < -0.99);
        assert!(b.points[3].alpha < 0.0);
    }
}

#[test]
fn step_grows_by_1_3_up_to_hmax() {
    for which in BOTH {
        // The branch x = 0 of the transcritical model is a straight line.
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let opts = Options {
            h0: 0.01,
            hmax: 0.1,
            max_steps: 12,
            bounds: (f64::NEG_INFINITY, 5.0),
            ..Options::default()
        };
        let b = run("transcritical", which, &setup, &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps);
        let h: Vec<f64> = b.points.iter().map(|p| p.h).collect();
        assert_eq!(h[0], 0.0);
        assert_eq!(h[1], 0.01);
        for k in 1..h.len() - 1 {
            let want = (1.3 * h[k]).min(0.1);
            assert!((h[k + 1] - want).abs() < 1e-15, "step {k}: {h:?}");
        }
        assert_eq!(*h.last().unwrap(), 0.1);
        assert!(b.points[1..].iter().all(|p| p.newton_iters == 1));
    }
}

#[test]
fn step_halves_when_newton_fails() {
    for which in BOTH {
        // One Newton iteration cannot reach 1e-10 from the tangent predictor at h = 0.01
        // on the circle, so the first step is halved until it can.
        let opts = Options {
            max_newton: 1,
            max_steps: 20,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps);
        let h1 = b.points[1].h;
        assert!(h1 < 0.01 / 2.0, "h1 = {h1}");
        let halvings = (0.01 / h1).log2();
        assert!((halvings - halvings.round()).abs() < 1e-12, "{halvings}");
        on_circle(&b);
    }
}

#[test]
fn small_steps_give_step_too_small() {
    for which in BOTH {
        let opts = Options {
            max_newton: 1,
            hmin: 1e-3,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::StepTooSmall);
        assert_eq!(b.points.len(), 1);
    }
}

#[test]
fn max_steps_counts_accepted_steps() {
    for which in BOTH {
        let opts = Options {
            max_steps: 5,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps);
        assert_eq!(b.points.len(), 6);
    }
}

#[test]
fn upper_and_lower_bounds_land_on_alpha() {
    for which in BOTH {
        for (dir, bounds, want) in [
            (1.0, (f64::NEG_INFINITY, 0.5), 0.5),
            (-1.0, (-0.5, f64::INFINITY), -0.5),
        ] {
            let opts = Options {
                direction: dir,
                bounds,
                ..Options::default()
            };
            let b = run("circle", which, &circle_setup(), &opts).unwrap();
            assert_eq!(b.stop, StopReason::Bounds, "{which:?} {dir}");
            let last = b.points.last().unwrap();
            assert!((last.alpha - want).abs() <= 1e-10, "{}", last.alpha);
            assert!(b.points.iter().all(|p| p.alpha >= bounds.0 - 1e-10));
            assert!(b.points.iter().all(|p| p.alpha <= bounds.1 + 1e-10));
            on_circle(&b);
            // The cut-back step is shorter than the step that left the bounds.
            assert!(last.h > 0.0 && last.h <= opts.hmax);
        }
    }
}

#[test]
fn start_on_the_bound_stops_at_once() {
    for which in BOTH {
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.0),
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        assert_eq!(b.stop, StopReason::Bounds);
        assert_eq!(b.points.len(), 1);
    }
}

#[test]
fn cancel_before_and_during_a_run() {
    for which in BOTH {
        let (model, info) = load("circle");
        let mut m = build(&model, &info, which).unwrap();
        // Set after the fifth point: the run keeps those five and stops.
        let cancel = AtomicBool::new(false);
        let mut seen = 0;
        let b = continue_ep(
            m.as_mut(),
            &circle_setup(),
            &Options::default(),
            &cancel,
            &mut |_| {
                seen += 1;
                if seen == 5 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        )
        .unwrap();
        assert_eq!(b.stop, StopReason::Cancelled);
        assert_eq!(b.points.len(), 5);
        // Already set: nothing runs.
        let flag = AtomicBool::new(true);
        let r = continue_ep(
            m.as_mut(),
            &circle_setup(),
            &Options::default(),
            &flag,
            &mut |_| {},
        );
        assert_eq!(r.unwrap_err(), EngineError::Cancelled);
    }
}

#[test]
fn branch_point_hit_exactly_does_not_stop_the_run() {
    for which in BOTH {
        // x = 0 of x' = al x - x^2 meets the transcritical point at al = 0. With h = 0.25
        // the steps land on it exactly, and the corrector matrix there is singular. The run
        // accepts the point, records psi_BP = 0 and goes on (ruling P2-1).
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let opts = Options {
            h0: 0.25,
            hmax: 0.25,
            bounds: (f64::NEG_INFINITY, 1.0),
            ..Options::default()
        };
        let b = run("transcritical", which, &setup, &opts).unwrap();
        assert_eq!(b.stop, StopReason::Bounds, "{which:?}");
        assert_eq!(b.points.len(), 9, "{which:?}");
        assert_eq!(b.points[4].alpha, 0.0);
        assert_eq!(b.points[4].tests[1], 0.0);
        assert!(b.points[3].tests[1] * b.points[5].tests[1] < 0.0);
        // The tangent goes straight through along the old direction.
        assert!(b.points[4].tangent[1] > 0.999);
        assert_eq!(b.points[8].alpha, 1.0);
    }
}

/// The circle x^2 + al^2 = 1 as a hand-written problem whose first corrector solve
/// returns NaN, to check that a NaN update is a failed step and not a converged one.
struct PoisonedCircle {
    poison: usize,
}

impl Problem for PoisonedCircle {
    fn dim(&self) -> usize {
        2
    }
    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        Ok(())
    }
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        out[0] = 2.0 * x[0];
        out[1] = 2.0 * x[1];
        Ok(())
    }
    fn bordered_solve(
        &mut self,
        jac: &[f64],
        border: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
    ) -> Result<f64, EngineError> {
        // Tangent solves have rhs = (0, 1); corrector solves do not.
        if rhs[1] != 1.0 && self.poison > 0 {
            self.poison -= 1;
            sol[0] = f64::NAN;
            sol[1] = 0.0;
            return Ok(1.0);
        }
        let det = jac[0] * border[1] - jac[1] * border[0];
        sol[0] = (rhs[0] * border[1] - jac[1] * rhs[1]) / det;
        sol[1] = (jac[0] * rhs[1] - rhs[0] * border[0]) / det;
        Ok(det)
    }
}

#[test]
fn nan_update_counts_as_a_failed_step() {
    let mut p = PoisonedCircle { poison: 1 };
    let opts = Options {
        h0: 0.02,
        max_steps: 3,
        ..Options::default()
    };
    let b = continue_branch(
        &mut p,
        &[-1.0, 0.0],
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(b.stop, StopReason::MaxSteps);
    // The poisoned first solve failed the step, so it was retried with half the length.
    assert_eq!(b.points[1].h, 0.01);
    assert!(
        b.points
            .iter()
            .all(|q| q.alpha.is_finite() && q.u[0].is_finite())
    );
}

#[test]
fn singular_start_is_an_error() {
    for which in BOTH {
        // The pitchfork origin at al = 0 has a singular state Jacobian.
        let setup = EpSetup::new(0, vec![0.0], vec![0.0]);
        let r = run("pitchfork", which, &setup, &Options::default());
        assert!(matches!(r, Err(EngineError::Singular(_))), "{r:?}");
    }
}

#[test]
fn fold_branch_turns_and_comes_back() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            max_steps: 150,
            ..Options::default()
        };
        let b = run("fold", which, &setup, &opts).unwrap();
        assert_eq!(b.stop, StopReason::MaxSteps);
        for p in &b.points {
            assert!((p.alpha + p.u[0] * p.u[0]).abs() <= 1e-10);
        }
        // alpha = -x^2 peaks at 0 (x = 0); the points near the fold come within 3e-3.
        let amax = b.points.iter().map(|p| p.alpha).fold(f64::MIN, f64::max);
        assert!(amax > -3e-3 && amax <= 1e-10, "{amax}");
        let turn = b.points.iter().position(|p| p.alpha == amax).unwrap();
        assert!(b.points[turn].u[0].abs() < 0.06);
        let end = b.points.last().unwrap();
        assert!(
            end.u[0] > 0.5 && end.alpha < -0.25,
            "{} {}",
            end.u[0],
            end.alpha
        );
        eprintln!(
            "fold {which:?}: {} points, {} steps, fold at point {turn}",
            b.points.len(),
            b.points.len() - 1
        );
    }
}

#[test]
fn equilibrium_converges_and_reports_failure() {
    for which in BOTH {
        let (model, info) = load("circle");
        let mut m = build(&model, &info, which).unwrap();
        let opts = Options::default();
        let u = equilibrium(m.as_mut(), &EpSetup::new(0, vec![0.0], vec![-0.5]), &opts).unwrap();
        assert!((u[0] + 1.0).abs() < 1e-12);
        // x^2 + 3 has no root.
        let r = equilibrium(m.as_mut(), &EpSetup::new(0, vec![2.0], vec![1.0]), &opts);
        assert!(matches!(r, Err(EngineError::NewtonFailed { .. })), "{r:?}");
    }
}

#[test]
fn bad_setups_are_refused() {
    let (model, info) = load("circle");
    let mut m = build(&model, &info, Evaluator::Jit).unwrap();
    let opts = Options::default();
    for setup in [
        EpSetup::new(1, vec![0.0], vec![-1.0]),
        EpSetup::new(0, vec![0.0, 1.0], vec![-1.0]),
        EpSetup::new(0, vec![0.0], vec![-1.0, 0.0]),
    ] {
        let r = equilibrium(m.as_mut(), &setup, &opts);
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
    }
}

/// The circle with a state block whose evaluation fails only at alpha = 0.5, the bound.
struct FailsAtBound {
    error: EngineError,
}

impl Problem for FailsAtBound {
    fn dim(&self) -> usize {
        2
    }
    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        Ok(())
    }
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        out[0] = 2.0 * x[0];
        out[1] = 2.0 * x[1];
        Ok(())
    }
    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        if (x[1] - 0.5).abs() < 1e-9 {
            return Err(self.error.clone());
        }
        Ok(Some(vec![2.0 * x[0]]))
    }
}

#[test]
fn landed_point_is_kept_when_its_tests_fail() {
    let opts = Options {
        bounds: (f64::NEG_INFINITY, 0.5),
        ..Options::default()
    };
    let run = |error: EngineError| {
        continue_branch(
            &mut FailsAtBound { error },
            &[-1.0, 0.0],
            &opts,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .unwrap()
    };
    let b = run(EngineError::NotFinite("JacU".into()));
    assert_eq!(b.stop, StopReason::Bounds);
    let last = b.points.last().unwrap();
    assert!((last.alpha - 0.5).abs() < 1e-10);
    assert!(last.eigenvalues.is_empty() && last.tests.iter().all(|t| t.is_nan()));
    assert_eq!(b.warnings.len(), 1);
    assert!(b.warnings[0].contains("JacU"), "{:?}", b.warnings);
    let c = run(EngineError::Cancelled);
    assert_eq!(c.stop, StopReason::Cancelled);
    assert!((c.points.last().unwrap().alpha - 0.5).abs() < 1e-10);
}

#[test]
fn landing_accepts_the_tol_refine_it_stopped_on() {
    for which in BOTH {
        // The fold branch x = sqrt(-a) runs towards the bound at a = -0.3.
        let setup = EpSetup::new(0, vec![-1.0], vec![1.0]);
        let opts = Options {
            bounds: (-5.0, -0.3),
            tol_refine: 1e-4,
            ..Options::default()
        };
        let b = run("fold", which, &setup, &opts).unwrap();
        assert_eq!(b.stop, StopReason::Bounds, "{which:?}");
        let last = b.points.last().unwrap();
        assert!((last.alpha + 0.3).abs() <= 1e-4, "{}", last.alpha);
    }
}

#[test]
fn infinite_step_options_are_refused() {
    for which in BOTH {
        let opts = Options {
            h0: f64::INFINITY,
            hmax: f64::INFINITY,
            ..Options::default()
        };
        let r = run("circle", which, &circle_setup(), &opts);
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
    }
}

#[test]
fn step_does_not_grow_after_a_slow_corrector() {
    for which in BOTH {
        let opts = Options {
            h0: 0.5,
            hmax: 2.0,
            max_newton: 12,
            max_steps: 12,
            ..Options::default()
        };
        let b = run("circle", which, &circle_setup(), &opts).unwrap();
        let slow: Vec<usize> = (1..b.points.len() - 1)
            .filter(|&k| b.points[k].newton_iters > 3)
            .collect();
        assert!(!slow.is_empty(), "no step needed more than 3 iterations");
        for k in slow {
            assert!(b.points[k + 1].h <= b.points[k].h, "{which:?} step {k}");
        }
    }
}
