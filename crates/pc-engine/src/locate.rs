//! Locating special points between two accepted points (SPEC section 5).
//!
//! When a test function changes sign strictly between points k and k+1, the zero is found
//! on the step with Illinois regula falsi on s in [0, 1] (Dowell and Jarratt, BIT 11, 1971,
//! the Illinois modification of regula falsi). The trial point at s is the corrector
//! solution on the hyperplane `v_kᵀ(x - x_k) = s h` started from the predictor
//! `x_k + s h v_k`, and the tangent and psi_BP at it use the border `v_k`, as at the
//! accepted point k+1. The values at s = 0 and s = 1 are the stored test values, except
//! for H, where every value is the plain product scaled by its modulus at point k
//! (`psi_h_relative`). A zero found here is not refined; the defining systems of Task 8
//! do that.

use crate::branch::{Extras, Options, Point, SpecialPoint, StopReason};
use crate::error::EngineError;
use crate::palc::{Corrected, correct, tangent_or_kernel};
use crate::problem::Problem;
use crate::tests_fn::{self, BP, H, HopfZero, LP, classify_h_zero};
use std::sync::atomic::AtomicBool;

/// The step from point k, with the data every trial needs.
struct Step<'a> {
    xk: &'a [f64],
    v: &'a [f64],
    h: f64,
}

enum Trial {
    Ok(Point),
    Failed,
    Stop(StopReason),
}

/// Corrects at `s` on the step and evaluates the tests there. A cancel is a `Stop`. A
/// failed corrector, a failed tangent or a NaN value of test `t` is `Failed`, and so is
/// any other error: a locate failure never stops the run.
fn trial(
    p: &mut dyn Problem,
    st: &Step,
    s: f64,
    t: usize,
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> Trial {
    let start: Vec<f64> = st
        .xk
        .iter()
        .zip(st.v)
        .map(|(x, d)| x + s * st.h * d)
        .collect();
    let (x, iters) = match correct(p, st.xk, st.v, s * st.h, start, opts, cancel) {
        Corrected::Done { x, iters } => (x, iters),
        Corrected::Failed => return Trial::Failed,
        Corrected::Stop(StopReason::Cancelled) => return Trial::Stop(StopReason::Cancelled),
        // Any other stop reason is a failure of this trial only (ruling: never stop the run
        // because of a locate failure).
        Corrected::Stop(_) => return Trial::Failed,
    };
    let built = tangent_or_kernel(p, &x, st.v)
        .and_then(|(vn, d)| tests_fn::evaluate(p, &x, &vn, d, opts, warnings).map(|r| (vn, r)));
    match built {
        Ok((vn, r)) if !r.tests[t].is_nan() => {
            let n = x.len() - 1;
            Trial::Ok(Point {
                u: x[..n].to_vec(),
                alpha: x[n],
                tangent: vn,
                eigenvalues: r.eigenvalues,
                tests: r.tests,
                h: s * st.h,
                newton_iters: iters,
            })
        }
        Err(EngineError::Cancelled) => Trial::Stop(StopReason::Cancelled),
        _ => Trial::Failed,
    }
}

/// Label of a zero of test `t` at a point with these eigenvalues; `None` for a zero of
/// psi_H that has no pair behind it.
fn label(t: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
    match t {
        LP => Some("LP"),
        BP => Some("BP"),
        _ => match classify_h_zero(eigs) {
            HopfZero::Hopf => Some("H"),
            HopfZero::NeutralSaddle => Some("NSA"),
            HopfZero::NoPair => None,
        },
    }
}

fn special(label: &str, index: usize, point: Point) -> SpecialPoint {
    SpecialPoint {
        label: label.to_string(),
        index,
        point,
        extras: Extras::default(),
    }
}

/// Finds the zero of test `t` on the step; `fa` and `fb` are its values at the ends, of
/// opposite sign. With `href` set (test H) the trials are scored with `psi_h_relative`
/// against that log reference instead of their stored values. Returns `Ok(None)` after
/// pushing a warning when the search fails.
#[allow(clippy::too_many_arguments)]
fn locate_one(
    p: &mut dyn Problem,
    st: &Step,
    t: usize,
    (fa, fb): (f64, f64),
    href: Option<f64>,
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> Result<Option<(f64, Point)>, StopReason> {
    let (mut lo, mut hi, mut flo, mut fhi) = (0.0, 1.0, fa, fb);
    let mut side = 0i8;
    let mut best: Option<(f64, f64, Point)> = None;
    for _ in 0..opts.max_locate {
        if (hi - lo) * st.h <= 1e-13 {
            break;
        }
        let mut s = (lo * fhi - hi * flo) / (fhi - flo);
        if !(s > lo && s < hi) {
            s = 0.5 * (lo + hi);
        }
        let (s, pt) = match trial(p, st, s, t, opts, cancel, warnings) {
            Trial::Ok(pt) => (s, pt),
            Trial::Stop(r) => return Err(r),
            // One retry at the midpoint of the bracket.
            Trial::Failed => {
                let m = 0.5 * (lo + hi);
                match trial(p, st, m, t, opts, cancel, warnings) {
                    Trial::Ok(pt) => (m, pt),
                    Trial::Stop(r) => return Err(r),
                    Trial::Failed => {
                        break;
                    }
                }
            }
        };
        let fs = match href {
            Some(l) => tests_fn::psi_h_relative(&pt.eigenvalues, l),
            None => pt.tests[t],
        };
        if best.as_ref().is_none_or(|(_, b, _)| fs.abs() < *b) {
            best = Some((s, fs.abs(), pt));
        }
        if fs == 0.0 {
            break;
        }
        if fs * fhi > 0.0 {
            hi = s;
            fhi = fs;
            if side == -1 {
                flo /= 2.0;
            }
            side = -1;
        } else {
            lo = s;
            flo = fs;
            if side == 1 {
                fhi /= 2.0;
            }
            side = 1;
        }
    }
    // A search that ended on a failed trial still returns the best valid trial so far.
    Ok(best.map(|(s, _, pt)| (s, pt)))
}

/// Looks for sign changes of the enabled tests between the last two points of `points`
/// (points k and k+1, with k = len - 2) and appends the special points found, ordered by
/// s, to `specials`. `xk`, `v` and `h` describe the step that reached point k+1: the
/// start, its tangent and the length of the step. Only a cancel is returned as an `Err`;
/// a failed search pushes a warning.
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_step(
    p: &mut dyn Problem,
    xk: &[f64],
    v: &[f64],
    h: f64,
    points: &[Point],
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
    specials: &mut Vec<SpecialPoint>,
) -> Result<(), StopReason> {
    // Trials write to a scratch vector; its entries are merged without repeats.
    let mut scratch = Vec::new();
    let r = detect_inner(
        p,
        xk,
        v,
        h,
        points,
        opts,
        cancel,
        &mut scratch,
        warnings,
        specials,
    );
    for w in scratch {
        if !warnings.contains(&w) {
            warnings.push(w);
        }
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn detect_inner(
    p: &mut dyn Problem,
    xk: &[f64],
    v: &[f64],
    h: f64,
    points: &[Point],
    opts: &Options,
    cancel: &AtomicBool,
    scratch: &mut Vec<String>,
    warnings: &mut Vec<String>,
    specials: &mut Vec<SpecialPoint>,
) -> Result<(), StopReason> {
    let k = match points.len().checked_sub(2) {
        Some(k) => k,
        None => return Ok(()),
    };
    let (pk, pk1) = (&points[k], &points[k + 1]);
    let st = Step { xk, v, h };
    let mut found: Vec<(f64, SpecialPoint)> = Vec::new();
    for t in [LP, BP, H] {
        let (a, b) = (pk.tests[t], pk1.tests[t]);
        // NaN on either side skips the test; a zero at point k was reported before.
        if a.is_nan() || b.is_nan() || a == 0.0 {
            continue;
        }
        if b == 0.0 {
            if let Some(l) = label(t, &pk1.eigenvalues) {
                found.push((1.0, special(l, k + 1, pk1.clone())));
            }
        } else if a * b < 0.0 {
            // The stored psi_H can be flat near its zero when another factor is tiny, so
            // the search on H uses the plain product scaled by its modulus at point k.
            let (ends, href) = if t == H {
                let l = tests_fn::psi_h_log(&pk.eigenvalues);
                let fb = tests_fn::psi_h_relative(&pk1.eigenvalues, l);
                ((a.signum(), fb), Some(l))
            } else {
                ((a, b), None)
            };
            match locate_one(p, &st, t, ends, href, opts, cancel, scratch)? {
                Some((s, pt)) => match label(t, &pt.eigenvalues) {
                    Some(l) => found.push((s, special(l, k + 1, pt))),
                    None => warn(warnings, t, &pt.eigenvalues, k),
                },
                // No point was located, so the end of the step is the nearest guide to
                // which kind of zero (H or NSA) was being looked for.
                None => warn(warnings, t, &pk1.eigenvalues, k),
            }
        }
    }
    found.sort_by(|x, y| x.0.total_cmp(&y.0));
    specials.extend(found.into_iter().map(|(_, sp)| sp));
    Ok(())
}

fn warn(warnings: &mut Vec<String>, t: usize, eigs: &[(f64, f64)], k: usize) {
    let name = label(t, eigs).unwrap_or("H");
    warnings.push(format!(
        "could not locate {name} between points {k} and {}",
        k + 1
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two unknowns, one equation x0 = 0, or an equation that never evaluates.
    struct Dummy {
        broken: bool,
    }
    impl Problem for Dummy {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            if self.broken {
                return Err(EngineError::NotFinite("residual".into()));
            }
            out[0] = x[0];
            Ok(())
        }
        fn jacobian(&mut self, _x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
    }

    fn pt(alpha: f64, tests: [f64; 3]) -> Point {
        Point {
            u: vec![0.0],
            alpha,
            tangent: vec![0.0, 1.0],
            eigenvalues: Vec::new(),
            tests,
            h: 0.1,
            newton_iters: 1,
        }
    }

    fn run(broken: bool, pts: &[Point]) -> (Vec<String>, Vec<SpecialPoint>) {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut Dummy { broken },
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            pts,
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn exact_zero_at_the_end_is_reported_once() {
        let n = f64::NAN;
        let pts = [
            pt(0.0, [n, 1.0, n]),
            pt(0.1, [n, 0.0, n]),
            pt(0.2, [n, -1.0, n]),
        ];
        let (w, sp) = run(false, &pts[..2]);
        assert!(w.is_empty());
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("BP", 1));
        assert_eq!(sp[0].point.alpha, pts[1].alpha);
        assert_eq!(sp[0].point.tests[1], 0.0);
        // The next step starts from zero and reports nothing.
        let (_, sp) = run(false, &pts[1..]);
        assert!(sp.is_empty());
    }

    #[test]
    fn nan_on_either_side_skips_the_test() {
        let n = f64::NAN;
        let (w, sp) = run(false, &[pt(0.0, [1.0, n, n]), pt(0.1, [-1.0, 1.0, n])]);
        assert!(w.is_empty());
        // Only LP is defined on both sides, and its zero is found at s = 0.5.
        assert_eq!(sp.len(), 1);
        assert_eq!(sp[0].label, "LP");
        let (_, sp) = run(false, &[pt(0.0, [n, n, n]), pt(0.1, [n, n, n])]);
        assert!(sp.is_empty());
    }

    #[test]
    fn failed_corrector_gives_a_warning_and_no_special() {
        let n = f64::NAN;
        let (w, sp) = run(true, &[pt(0.0, [1.0, n, n]), pt(0.1, [-1.0, n, n])]);
        assert!(sp.is_empty());
        assert_eq!(
            w,
            vec!["could not locate LP between points 0 and 1".to_string()]
        );
    }
}

#[cfg(test)]
mod failure_tests {
    use super::*;

    struct Failing {
        fail_above: f64,
    }
    impl Problem for Failing {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            if x[1] > self.fail_above {
                return Err(EngineError::BadProblem("model error".into()));
            }
            out[0] = x[0];
            Ok(())
        }
        fn jacobian(&mut self, _x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
    }

    fn pt(tests: [f64; 3]) -> Point {
        Point {
            u: vec![0.0],
            alpha: 0.0,
            tangent: vec![0.0, 1.0],
            eigenvalues: Vec::new(),
            tests,
            h: 0.1,
            newton_iters: 1,
        }
    }

    fn run(fail_above: f64) -> (Vec<String>, Vec<SpecialPoint>) {
        let n = f64::NAN;
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut Failing { fail_above },
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[pt([1.0, n, n]), pt([-1.0, n, n])],
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn a_model_error_at_a_trial_does_not_stop_the_run() {
        // Every trial hits the error: warning, no special, and an Ok result.
        let (w, sp) = run(-1.0);
        assert!(sp.is_empty());
        assert_eq!(
            w,
            vec!["could not locate LP between points 0 and 1".to_string()]
        );
    }

    #[test]
    fn the_best_trial_is_kept_when_a_later_one_fails() {
        // The first trial (s = 0.5, alpha = 0.05) works, the second (alpha = 0.075) fails.
        let (w, sp) = run(0.06);
        assert!(w.is_empty());
        assert_eq!(sp.len(), 1);
        assert!(!sp[0].extras.refined);
        assert!((sp[0].point.alpha - 0.05).abs() < 1e-12);
    }
}

#[cfg(test)]
mod two_zero_tests {
    use super::*;

    /// Two unknowns and the equation x0 = 0. The tangent solve (right-hand side e_2)
    /// returns (1, a - 0.07) with determinant a - 0.03, where a is the alpha of the last
    /// residual call, so psi_BP vanishes at alpha = 0.03 and psi_LP at alpha = 0.07. Other
    /// solves are the identity.
    struct TwoZeros {
        alpha: f64,
    }
    impl Problem for TwoZeros {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            self.alpha = x[1];
            out[0] = x[0];
            Ok(())
        }
        fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            self.alpha = x[1];
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
        fn bordered_solve(
            &mut self,
            _jac: &[f64],
            _border: &[f64],
            rhs: &[f64],
            sol: &mut [f64],
        ) -> Result<f64, EngineError> {
            if rhs == [0.0, 1.0] {
                sol.copy_from_slice(&[1.0, self.alpha - 0.07]);
                Ok(self.alpha - 0.03)
            } else {
                sol.copy_from_slice(rhs);
                Ok(1.0)
            }
        }
    }

    fn pt(alpha: f64) -> Point {
        Point {
            u: vec![0.0],
            alpha,
            tangent: vec![1.0, 0.0],
            eigenvalues: Vec::new(),
            tests: [alpha - 0.07, alpha - 0.03, f64::NAN],
            h: 0.1,
            newton_iters: 1,
        }
    }

    #[test]
    fn two_tests_on_one_step_are_both_located_in_order() {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut TwoZeros { alpha: 0.0 },
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[pt(0.0), pt(0.1)],
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        assert!(w.is_empty(), "{w:?}");
        let names: Vec<&str> = sp.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(names, ["BP", "LP"]);
        assert!(sp.iter().all(|s| s.index == 1 && !s.extras.refined));
        // Tests are ordered LP, BP, H in the loop; the specials are ordered by s.
        assert!(
            (sp[0].point.alpha - 0.03).abs() < 1e-10,
            "{}",
            sp[0].point.alpha
        );
        assert!(
            (sp[1].point.alpha - 0.07).abs() < 1e-10,
            "{}",
            sp[1].point.alpha
        );
        assert!(sp[0].point.h < sp[1].point.h);
    }
}

#[cfg(test)]
mod curved_tests {
    use super::*;

    /// Like the two-zero problem, with det = exp(40 (alpha - 0.01)) - 1, so psi_BP is
    /// convex in alpha with its zero at 0.01. Plain regula falsi crawls on it.
    struct Curved {
        alpha: f64,
    }
    impl Problem for Curved {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            self.alpha = x[1];
            out[0] = x[0];
            Ok(())
        }
        fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            self.alpha = x[1];
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
        fn bordered_solve(
            &mut self,
            _jac: &[f64],
            _border: &[f64],
            rhs: &[f64],
            sol: &mut [f64],
        ) -> Result<f64, EngineError> {
            sol.copy_from_slice(rhs);
            Ok(psi(self.alpha))
        }
    }

    fn psi(alpha: f64) -> f64 {
        (40.0 * (alpha - 0.01)).exp() - 1.0
    }

    fn pt(alpha: f64) -> Point {
        Point {
            u: vec![0.0],
            alpha,
            tangent: vec![0.0, 1.0],
            eigenvalues: Vec::new(),
            tests: [f64::NAN, psi(alpha), f64::NAN],
            h: 0.1,
            newton_iters: 1,
        }
    }

    fn run(max_locate: usize) -> (Vec<String>, Vec<SpecialPoint>) {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        let opts = Options {
            max_locate,
            ..Options::default()
        };
        detect_step(
            &mut Curved { alpha: 0.0 },
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[pt(0.0), pt(0.1)],
            &opts,
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn max_locate_caps_the_trials_and_keeps_the_best() {
        let (w, sp) = run(2);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert!(
            sp[0].point.tests[1].abs() > 1e-3,
            "{}",
            sp[0].point.tests[1]
        );
        // No trials at all is a failed search.
        let (w, sp) = run(0);
        assert!(sp.is_empty());
        assert_eq!(
            w,
            vec!["could not locate BP between points 0 and 1".to_string()]
        );
    }

    #[test]
    fn illinois_halving_converges_on_a_convex_test() {
        // Plain regula falsi keeps one end fixed on a convex function and gains only a
        // fixed factor per trial; with halving 12 trials reach 1e-10.
        let (w, sp) = run(12);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert!(
            (sp[0].point.alpha - 0.01).abs() < 1e-10,
            "{}",
            sp[0].point.alpha
        );
    }
}
