//! Iteration of a map x(k+1) = F(x(k), p, t = k) (SPEC section 8u).
//!
//! A run from iterate n0 to iterate n1 returns n1 - n0 + 1 points: point j holds the
//! iterate number n0 + j in `alpha` and the state x(n0 + j) in `u`. Inside F the time t
//! is the number of the iterate that F maps from, so the step from x(k) to x(k+1) sees
//! t = k. `h` is 0 on the start point and 1 on every other point.

use crate::branch::{Branch, Point, StopReason};
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::EntryKind;
use std::sync::atomic::{AtomicBool, Ordering};

/// Start and span of an iteration run.
#[derive(Clone, Debug, PartialEq)]
pub struct IterSetup {
    pub x0: Vec<f64>,
    pub params: Vec<f64>,
    /// Number of the start iterate.
    pub n0: i64,
    /// Number of the last iterate, greater than `n0`.
    pub n1: i64,
}

/// Iterates the map from `setup.x0` at iterate `setup.n0` to iterate `setup.n1`. Every
/// point goes to `on_point` as soon as it exists. A non-finite value of F ends the run
/// with `StopReason::Error` and keeps the points before it. A cancel request is checked
/// before each iterate; a request before the start returns `Err(Cancelled)`.
pub fn iterate(
    model: &mut dyn ModelEval,
    setup: &IterSetup,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let (n, n_p) = (model.n(), model.n_p());
    check(setup, n, n_p)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut branch = Branch {
        points: Vec::new(),
        specials: Vec::new(),
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    };
    let mut push = |branch: &mut Branch, k: i64, x: &[f64], h: f64| {
        let p = Point {
            u: x.to_vec(),
            alpha: k as f64,
            tangent: Vec::new(),
            eigenvalues: Vec::new(),
            tests: [f64::NAN; 3],
            codim2: Vec::new(),
            h,
            newton_iters: 0,
            mesh: Vec::new(),
        };
        on_point(&p);
        branch.points.push(p);
    };
    // The last parameter slot is t, as in the initial value run.
    let mut pbuf = setup.params.clone();
    pbuf.push(0.0);
    let mut x = setup.x0.clone();
    let mut next = vec![0.0; n];
    push(&mut branch, setup.n0, &x, 0.0);
    let mut k = setup.n0;
    branch.stop = loop {
        if k == setup.n1 {
            break StopReason::End;
        }
        if cancel.load(Ordering::Relaxed) {
            break StopReason::Cancelled;
        }
        pbuf[n_p] = k as f64;
        if let Err(e) = model.eval(&EntryKind::F, &x, &pbuf, &[], &mut next) {
            break StopReason::Error(e);
        }
        std::mem::swap(&mut x, &mut next);
        k += 1;
        push(&mut branch, k, &x, 1.0);
    };
    Ok(branch)
}

fn check(setup: &IterSetup, n: usize, n_p: usize) -> Result<(), EngineError> {
    let bad = |m: String| Err(EngineError::BadProblem(m));
    if setup.x0.len() != n {
        return bad(format!(
            "x0 has {} values, the model has {n} states",
            setup.x0.len()
        ));
    }
    if setup.params.len() != n_p {
        return bad(format!(
            "params has {} values, the model has {n_p} parameters",
            setup.params.len()
        ));
    }
    if !setup.x0.iter().chain(&setup.params).all(|v| v.is_finite()) {
        return bad("x0 and params must be finite".into());
    }
    if setup.n1 <= setup.n0 {
        return bad("tspan: the second iterate must be greater than the first".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// x(k+1) = x + t on one state, or a map that fails at a set iterate.
    struct Shift {
        fail_at: Option<f64>,
    }

    impl ModelEval for Shift {
        fn n(&self) -> usize {
            1
        }
        fn n_p(&self) -> usize {
            0
        }
        fn eval(
            &mut self,
            kind: &EntryKind,
            x: &[f64],
            p: &[f64],
            _v: &[f64],
            out: &mut [f64],
        ) -> Result<(), EngineError> {
            assert!(matches!(kind, EntryKind::F));
            if Some(p[0]) == self.fail_at {
                return Err(EngineError::NotFinite("F".into()));
            }
            out[0] = x[0] + p[0];
            Ok(())
        }
    }

    fn setup(n0: i64, n1: i64) -> IterSetup {
        IterSetup {
            x0: vec![0.0],
            params: Vec::new(),
            n0,
            n1,
        }
    }

    fn run(m: &mut Shift, s: &IterSetup) -> Result<Branch, EngineError> {
        iterate(m, s, &AtomicBool::new(false), &mut |_| {})
    }

    #[test]
    fn t_is_the_number_of_the_iterate_mapped_from() {
        let b = run(&mut Shift { fail_at: None }, &setup(0, 4)).unwrap();
        assert_eq!(b.stop, StopReason::End);
        let k: Vec<f64> = b.points.iter().map(|p| p.alpha).collect();
        let x: Vec<f64> = b.points.iter().map(|p| p.u[0]).collect();
        let h: Vec<f64> = b.points.iter().map(|p| p.h).collect();
        assert_eq!(k, [0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x, [0.0, 0.0, 1.0, 3.0, 6.0]);
        assert_eq!(h, [0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_failed_step_keeps_the_points_before_it() {
        let b = run(&mut Shift { fail_at: Some(2.0) }, &setup(0, 4)).unwrap();
        assert!(matches!(
            b.stop,
            StopReason::Error(EngineError::NotFinite(_))
        ));
        assert_eq!(b.points.len(), 3);
        assert_eq!(b.points.last().unwrap().alpha, 2.0);
    }

    #[test]
    fn every_point_is_streamed() {
        let mut seen = Vec::new();
        let b = iterate(
            &mut Shift { fail_at: None },
            &setup(-2, 1),
            &AtomicBool::new(false),
            &mut |p| seen.push(p.alpha),
        )
        .unwrap();
        assert_eq!(seen, [-2.0, -1.0, 0.0, 1.0]);
        assert_eq!(b.points.len(), 4);
    }

    #[test]
    fn a_cancel_before_the_start_is_an_error() {
        let r = iterate(
            &mut Shift { fail_at: None },
            &setup(0, 4),
            &AtomicBool::new(true),
            &mut |_| {},
        );
        assert!(matches!(r, Err(EngineError::Cancelled)));
    }

    #[test]
    fn bad_inputs_are_refused() {
        let mut m = Shift { fail_at: None };
        let mut s = setup(0, 4);
        s.x0 = vec![0.0, 1.0];
        assert!(matches!(run(&mut m, &s), Err(EngineError::BadProblem(_))));
        let mut s = setup(0, 4);
        s.params = vec![1.0];
        assert!(matches!(run(&mut m, &s), Err(EngineError::BadProblem(_))));
        let mut s = setup(0, 4);
        s.x0 = vec![f64::NAN];
        assert!(matches!(run(&mut m, &s), Err(EngineError::BadProblem(_))));
        for (n0, n1) in [(3, 3), (4, 2)] {
            assert!(matches!(
                run(&mut m, &setup(n0, n1)),
                Err(EngineError::BadProblem(_))
            ));
        }
    }
}
