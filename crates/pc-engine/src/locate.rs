//! Locating special points between two accepted points (SPEC section 5).
//!
//! When a test function changes sign strictly between points k and k+1, the zero is found
//! on the step with Illinois regula falsi on s in [0, 1] (Dowell and Jarratt, BIT 11, 1971,
//! the Illinois modification of regula falsi). The trial point at s is the corrector
//! solution on the hyperplane `v_kᵀ(x - x_k) = s h` started from the predictor
//! `x_k + s h v_k`, and the tangent and psi_BP at it use the border `v_k`, as at the
//! accepted point k+1. The values at s = 0 and s = 1 are the stored test values, except
//! for H, where every value is the plain product scaled by its modulus at point k
//! (`psi_h_relative`). A zero of an equilibrium test found here is not refined; the defining
//! systems of Task 8 do that. A zero of a codim-2 test is polished by Newton's method along
//! the curve (`refine_zero`) and flagged `refined` when that converges.

use crate::branch::{Extras, Options, Point, SpecialPoint, StopReason};
use crate::error::EngineError;
use crate::palc::{Corrected, correct, tangent_or_kernel};
use crate::problem::Problem;
use crate::tests_fn::{self, BP, H, HopfZero, LP, classify_h_zero, value_at};
use std::sync::atomic::AtomicBool;

/// A located zero of a codim-2 test counts only if its value is at most this fraction of the
/// smaller end value of the step. A sign change through a pole (the first Lyapunov coefficient
/// at a zero-Hopf point, say) leaves the search at a large value and fails this check.
const POLE_TOL: f64 = 1e-2;

/// Newton steps `refine_zero` takes before it gives up.
const REFINE_ITERS: usize = 8;

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
        Ok((vn, r)) if !r.value(t).is_nan() => {
            let n = x.len() - 1;
            Trial::Ok(Point {
                u: x[..n].to_vec(),
                alpha: x[n],
                tangent: vn,
                eigenvalues: r.eigenvalues,
                tests: r.tests,
                codim2: r.codim2,
                h: s * st.h,
                newton_iters: iters,
                mesh: p.mesh(),
            })
        }
        Err(EngineError::Cancelled) => Trial::Stop(StopReason::Cancelled),
        _ => Trial::Failed,
    }
}

/// Label of a zero of test `t` at a point with these eigenvalues; `None` for a zero of
/// psi_H that has no pair behind it. Tests 0..3 are the equilibrium tests; `3 + i` is codim-2
/// test `i` of the problem, which names it.
fn label(p: &dyn Problem, t: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
    match t {
        LP => Some("LP"),
        BP => Some("BP"),
        H => match classify_h_zero(eigs) {
            HopfZero::Hopf => Some("H"),
            HopfZero::NeutralSaddle => Some("NSA"),
            HopfZero::NoPair => None,
        },
        _ => p.codim2_label(t - 3, eigs),
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
/// against that log reference instead of their stored values. Returns `Ok(None)` when the
/// search fails, and also when a codim-2 test ends at a value above `POLE_TOL` times its
/// smaller end value, since that sign change is a pole and not a zero.
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
            None => value_at(&pt.tests, &pt.codim2, t),
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
    let tol = POLE_TOL * fa.abs().min(fb.abs());
    Ok(best
        .filter(|(_, v, _)| t < 3 || *v <= tol)
        .map(|(s, _, pt)| (s, pt)))
}

/// Polishes the located zero `pt` of codim-2 test `t` with Newton's method along the curve.
/// The point is on the curve already, so the system [F(x); test(x)] = 0 reduces to one
/// unknown, the arclength along the tangent: the slope of the test is a forward difference
/// along the tangent, each step is a corrector solve of that length from the current
/// point, and the run ends when a step is shorter than `tol_x` (scaled by 1 + |x|). The
/// result is `Ok(None)`, and the caller keeps `pt`, when the test is undefined or flat
/// nearby, a step is longer than the step being searched, the corrector fails, the point
/// leaves the step, or the iteration does not converge. Only a cancel is an `Err`.
fn refine_zero(
    p: &mut dyn Problem,
    st: &Step,
    t: usize,
    pt: &Point,
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> Result<Option<Point>, StopReason> {
    // Any error but a cancel only means that this point stays unrefined.
    fn give_up<T>(e: EngineError) -> Result<Option<T>, StopReason> {
        match e {
            EngineError::Cancelled => Err(StopReason::Cancelled),
            _ => Ok(None),
        }
    }
    let mut cur = pt.clone();
    let mut x: Vec<f64> = cur.u.iter().copied().chain([cur.alpha]).collect();
    let (mut tau, mut det) = match tangent_or_kernel(p, &x, st.v) {
        Ok(r) => r,
        Err(e) => return give_up(e),
    };
    for _ in 0..REFINE_ITERS {
        let val = value_at(&cur.tests, &cur.codim2, t);
        if val == 0.0 {
            return Ok(Some(cur));
        }
        let scale = 1.0 + x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let eps = f64::EPSILON.sqrt() * scale;
        let xe: Vec<f64> = x.iter().zip(&tau).map(|(a, d)| a + eps * d).collect();
        let slope = match tests_fn::evaluate(p, &xe, &tau, det, opts, warnings) {
            Ok(r) => (r.value(t) - val) / eps,
            Err(e) => return give_up(e),
        };
        let ds = -val / slope;
        if !ds.is_finite() || ds.abs() > st.h.abs() {
            return Ok(None);
        }
        let start: Vec<f64> = x.iter().zip(&tau).map(|(a, d)| a + ds * d).collect();
        let (xn, iters) = match correct(p, &x, &tau, ds, start, opts, cancel) {
            Corrected::Done { x, iters } => (x, iters),
            Corrected::Stop(StopReason::Cancelled) => return Err(StopReason::Cancelled),
            Corrected::Failed | Corrected::Stop(_) => return Ok(None),
        };
        let (tn, dn) = match tangent_or_kernel(p, &xn, &tau) {
            Ok(r) => r,
            Err(e) => return give_up(e),
        };
        let r = match tests_fn::evaluate(p, &xn, &tn, dn, opts, warnings) {
            Ok(r) if !r.value(t).is_nan() => r,
            Ok(_) => return Ok(None),
            Err(e) => return give_up(e),
        };
        // The step coordinate s h of the new point, which must stay between the end points.
        let along: f64 =
            st.v.iter()
                .zip(xn.iter().zip(st.xk))
                .map(|(v, (a, b))| v * (a - b))
                .sum();
        if along * st.h < 0.0 || along.abs() > st.h.abs() {
            return Ok(None);
        }
        let n = xn.len() - 1;
        cur = Point {
            u: xn[..n].to_vec(),
            alpha: xn[n],
            tangent: tn.clone(),
            eigenvalues: r.eigenvalues,
            tests: r.tests,
            codim2: r.codim2,
            h: along,
            newton_iters: iters,
            mesh: p.mesh(),
        };
        x = xn;
        tau = tn;
        det = dn;
        if ds.abs() <= opts.tol_x * scale {
            return Ok(Some(cur));
        }
    }
    Ok(None)
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
    let n_tests = 3 + pk.codim2.len().min(pk1.codim2.len());
    for t in 0..n_tests {
        let (a, b) = (
            value_at(&pk.tests, &pk.codim2, t),
            value_at(&pk1.tests, &pk1.codim2, t),
        );
        // A test whose value overflowed has no sign change to locate. NaN is the quiet
        // "not defined here", so an infinite value gets a warning instead.
        if a.is_infinite() || b.is_infinite() {
            let name = test_name(&*p, t, &pk1.eigenvalues);
            warnings.push(format!(
                "{name} is not finite between points {k} and {}; the test was skipped there",
                k + 1
            ));
            continue;
        }
        // The first point is the start of the run, so a zero there was not reported before.
        // Later, a zero at point k was reported by the step that reached it. A test that is
        // also zero at the second point, or undefined there, is not a crossing of zero: the
        // coefficient of a model that has none is zero on the whole curve.
        if k == 0
            && a == 0.0
            && !b.is_nan()
            && b != 0.0
            && let Some(l) = label(&*p, t, &pk.eigenvalues)
        {
            found.push((0.0, special(l, 0, pk.clone())));
        }
        // NaN on either side skips the test.
        if a.is_nan() || b.is_nan() || a == 0.0 {
            continue;
        }
        if b == 0.0 {
            if let Some(l) = label(&*p, t, &pk1.eigenvalues) {
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
                Some((s, pt)) => match label(&*p, t, &pt.eigenvalues) {
                    Some(l) => {
                        // A codim-2 point is polished unless that changes what it is.
                        let polished = if t >= 3 {
                            refine_zero(p, &st, t, &pt, opts, cancel, scratch)?
                                .filter(|r| label(&*p, t, &r.eigenvalues) == Some(l))
                        } else {
                            None
                        };
                        let (pt, refined) = match polished {
                            Some(r) => (r, true),
                            None => (pt, false),
                        };
                        let mut sp = special(l, k + 1, pt);
                        sp.extras.refined = refined;
                        found.push((s, sp));
                    }
                    // A codim-2 zero that its problem drops is no bifurcation, and the
                    // search did not fail.
                    None if t >= 3 => {}
                    None => warn(warnings, &*p, t, &pt.eigenvalues, k),
                },
                // A codim-2 value at the start of at most `POLE_TOL` times the value at the
                // second point is a zero at the start by the same measure the pole guard
                // uses. The failed search is no warning then, and the start point gets the
                // label, as it does for an exact zero. Rounding decides which side of zero
                // such a value falls on, and only a sign change reaches this arm.
                None if k == 0 && t >= 3 && a.abs() <= POLE_TOL * b.abs() => {
                    if let Some(l) = label(&*p, t, &pk.eigenvalues) {
                        found.push((0.0, special(l, 0, pk.clone())));
                    }
                }
                // No point was located, so the end of the step is the nearest guide to
                // which kind of zero (H or NSA) was being looked for.
                None => warn(warnings, &*p, t, &pk1.eigenvalues, k),
            }
        }
    }
    found.sort_by(|x, y| x.0.total_cmp(&y.0));
    specials.extend(found.into_iter().map(|(_, sp)| sp));
    Ok(())
}

fn test_name(p: &dyn Problem, t: usize, eigs: &[(f64, f64)]) -> &'static str {
    if t >= 3 {
        p.codim2_labels().get(t - 3).copied().unwrap_or("codim-2")
    } else {
        label(p, t, eigs).unwrap_or("H")
    }
}

fn warn(warnings: &mut Vec<String>, p: &dyn Problem, t: usize, eigs: &[(f64, f64)], k: usize) {
    let name = test_name(p, t, eigs);
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
            codim2: Vec::new(),
            h: 0.1,
            newton_iters: 1,
            mesh: Vec::new(),
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
        // The next step, with the whole run so far, starts from zero and reports nothing.
        let (_, sp) = run(false, &pts);
        assert!(sp.is_empty());
        // Where a run starts on that zero, the start point is labeled instead.
        let (_, sp) = run(false, &pts[1..]);
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("BP", 0));
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
mod codim2_tests {
    use super::*;

    /// Two unknowns, the equation x0 = 0, and one codim-2 test named "ZZ": `x1 - 0.05`, or
    /// `1 / (x1 - 0.05)` with `pole`. With `drop` the label hook rejects every zero. With
    /// `nan_from = Some(n)` the test is NaN from its n-th evaluation on (counting from 1).
    /// With `at_start` the test is `x1 + 3e-14`, which stays above zero on the step and so
    /// models a test that is zero at the start up to rounding.
    #[derive(Default)]
    struct Curve {
        pole: bool,
        drop: bool,
        at_start: bool,
        nan_from: Option<usize>,
        calls: usize,
    }
    fn zero() -> Curve {
        Curve::default()
    }
    impl Problem for Curve {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out[0] = x[0];
            Ok(())
        }
        fn jacobian(&mut self, _x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out.copy_from_slice(&[1.0, 0.0]);
            Ok(())
        }
        fn ep_tests(&self) -> bool {
            false
        }
        fn codim2_labels(&self) -> &'static [&'static str] {
            &["ZZ"]
        }
        fn codim2_tests(
            &mut self,
            x: &[f64],
            _tangent: &[f64],
            _eigs: &[(f64, f64)],
        ) -> Result<Vec<f64>, EngineError> {
            self.calls += 1;
            if self.nan_from.is_some_and(|n| self.calls >= n) {
                return Ok(vec![f64::NAN]);
            }
            let d = if self.at_start {
                x[1] + 3e-14
            } else {
                x[1] - 0.05
            };
            Ok(vec![if self.pole { 1.0 / d } else { d }])
        }
        fn codim2_label(&self, i: usize, _eigs: &[(f64, f64)]) -> Option<&'static str> {
            if self.drop {
                None
            } else {
                self.codim2_labels().get(i).copied()
            }
        }
    }

    fn pt(alpha: f64) -> Point {
        pt_with(alpha, alpha - 0.05)
    }

    fn pt_with(alpha: f64, value: f64) -> Point {
        Point {
            u: vec![0.0],
            alpha,
            tangent: vec![0.0, 1.0],
            eigenvalues: Vec::new(),
            tests: [f64::NAN; 3],
            codim2: vec![value],
            h: 0.1,
            newton_iters: 1,
            mesh: Vec::new(),
        }
    }

    #[test]
    fn a_codim2_zero_is_located_and_labeled() {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut zero(),
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
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("ZZ", 1));
        assert!((sp[0].point.alpha - 0.05).abs() < 1e-10);
        assert!((sp[0].point.codim2[0]).abs() < 1e-10);
        assert!(sp[0].extras.refined);
    }

    /// Runs one step with end values that mislead the search, so one trial stops at about
    /// alpha = 0.0545 and the true zero is at 0.05.
    fn run_capped(p: Curve) -> (Vec<String>, Vec<SpecialPoint>) {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        let opts = Options {
            max_locate: 1,
            ..Options::default()
        };
        detect_step(
            &mut { p },
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[pt_with(0.0, -1.2), pt_with(0.1, 1.0)],
            &opts,
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn newton_polishes_a_zero_the_search_stopped_short_of() {
        let (w, sp) = run_capped(zero());
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert!(sp[0].extras.refined);
        assert!(
            (sp[0].point.alpha - 0.05).abs() < 1e-10,
            "{}",
            sp[0].point.alpha
        );
        assert!(sp[0].point.codim2[0].abs() < 1e-10);
        // The refined point stays inside the step it was found on.
        assert!(sp[0].point.h > 0.0 && sp[0].point.h < 0.1);
    }

    #[test]
    fn an_undefined_test_next_to_the_point_leaves_the_search_result() {
        // The second evaluation is the difference quotient of the refinement.
        let nan = Curve {
            nan_from: Some(2),
            ..zero()
        };
        let (w, sp) = run_capped(nan);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert!(!sp[0].extras.refined);
        assert!((sp[0].point.alpha - 0.05).abs() > 1e-4);
    }

    fn run(mut p: Curve, a: Point, b: Point) -> (Vec<String>, Vec<SpecialPoint>) {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut p,
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[a, b],
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn a_sign_change_through_a_pole_is_not_a_zero() {
        let pole = Curve {
            pole: true,
            ..zero()
        };
        let (w, sp) = run(pole, pt_with(0.0, -20.0), pt_with(0.1, 20.0));
        assert!(sp.is_empty());
        assert_eq!(
            w,
            vec!["could not locate ZZ between points 0 and 1".to_string()]
        );
    }

    #[test]
    fn an_infinite_test_value_is_skipped_with_a_warning() {
        // An overflowed test has no sign change to locate. NaN stays the quiet case.
        for (a, b) in [
            (f64::INFINITY, -1.0),
            (-1.0, f64::NEG_INFINITY),
            (f64::INFINITY, f64::NEG_INFINITY),
        ] {
            let (w, sp) = run(zero(), pt_with(0.0, a), pt_with(0.1, b));
            assert!(sp.is_empty(), "{a} {b}: {sp:?}");
            assert_eq!(
                w,
                vec![
                    "ZZ is not finite between points 0 and 1; the test was skipped there"
                        .to_string()
                ],
                "{a} {b}"
            );
        }
    }

    fn run_from(mut p: Curve, xk: &[f64], points: &[Point]) -> (Vec<String>, Vec<SpecialPoint>) {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        detect_step(
            &mut p,
            xk,
            &[0.0, 1.0],
            0.1,
            points,
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        (w, sp)
    }

    #[test]
    fn a_zero_at_the_first_point_is_labeled_at_point_zero_without_a_warning() {
        // Exactly zero at the start.
        let (w, sp) = run(zero(), pt_with(0.0, 0.0), pt_with(0.1, 0.1));
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("ZZ", 0));
        assert_eq!(sp[0].point.alpha, 0.0);
        // Zero up to rounding at the start, with a sign change to the second point that no
        // search can place: the same label, at the start point.
        let at_start = || Curve {
            at_start: true,
            ..zero()
        };
        let (w, sp) = run(at_start(), pt_with(0.0, -3e-14), pt_with(0.1, 0.1));
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("ZZ", 0));
        assert_eq!(sp[0].point.alpha, 0.0);
        // Rounding that leaves the start on the same side as the second point has no sign
        // change, so nothing is reported in that direction.
        let (w, sp) = run(at_start(), pt_with(0.0, 3e-14), pt_with(0.1, 0.1));
        assert!(w.is_empty() && sp.is_empty(), "{w:?}");
    }

    #[test]
    fn a_dropped_zero_at_the_first_point_gives_no_label() {
        let drop = || Curve {
            drop: true,
            ..zero()
        };
        let (w, sp) = run(drop(), pt_with(0.0, 0.0), pt_with(0.1, 0.1));
        assert!(w.is_empty() && sp.is_empty(), "{w:?}");
        let at_start = Curve {
            at_start: true,
            ..drop()
        };
        let (w, sp) = run(at_start, pt_with(0.0, -3e-14), pt_with(0.1, 0.1));
        assert!(w.is_empty() && sp.is_empty(), "{w:?}");
    }

    #[test]
    fn a_test_that_stays_zero_or_is_undefined_after_the_start_gives_no_start_label() {
        for next in [0.0, f64::NAN] {
            let (w, sp) = run(zero(), pt_with(0.0, 0.0), pt_with(0.1, next));
            assert!(w.is_empty() && sp.is_empty(), "{next}: {w:?}");
        }
    }

    #[test]
    fn a_zero_after_the_first_step_is_not_labeled_twice() {
        // Point 1 has the zero. The step 1 to 2 starts from it and adds nothing.
        let pts = [pt_with(-0.1, -0.1), pt_with(0.0, 0.0), pt_with(0.1, 0.1)];
        let (w, sp) = run_from(zero(), &[0.0, 0.0], &pts);
        assert!(w.is_empty() && sp.is_empty(), "{w:?}");
    }

    #[test]
    fn a_noise_level_value_after_the_first_step_still_warns() {
        let at_start = Curve {
            at_start: true,
            ..zero()
        };
        let pts = [
            pt_with(-0.1, f64::NAN),
            pt_with(0.0, -3e-14),
            pt_with(0.1, 0.1),
        ];
        let (w, sp) = run_from(at_start, &[0.0, 0.0], &pts);
        assert!(sp.is_empty());
        assert_eq!(
            w,
            vec!["could not locate ZZ between points 1 and 2".to_string()]
        );
    }

    #[test]
    fn a_zero_near_the_first_point_that_the_search_places_is_reported() {
        // The zero at alpha = 0.05 lies 1e-3 of a step after the start, which has the value
        // -1e-4 against 0.05 at the second point, inside the `POLE_TOL` ratio. The search
        // places it, so it is a label and not a skipped start.
        let pts = [pt(0.0499), pt(0.1499)];
        let (w, sp) = run_from(zero(), &[0.0, 0.0499], &pts);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(sp.len(), 1);
        assert_eq!((sp[0].label.as_str(), sp[0].index), ("ZZ", 1));
        assert!((sp[0].point.alpha - 0.05).abs() < 1e-10);
    }

    #[test]
    fn a_dropped_codim2_zero_gives_no_special_and_no_warning() {
        let drop = Curve {
            drop: true,
            ..zero()
        };
        let (w, sp) = run(drop, pt(0.0), pt(0.1));
        assert!(w.is_empty() && sp.is_empty(), "{w:?}");
    }

    #[test]
    fn points_without_codim2_values_skip_those_tests() {
        let (mut w, mut sp) = (Vec::new(), Vec::new());
        let mut a = pt(0.0);
        a.codim2.clear();
        detect_step(
            &mut zero(),
            &[0.0, 0.0],
            &[0.0, 1.0],
            0.1,
            &[a, pt(0.1)],
            &Options::default(),
            &AtomicBool::new(false),
            &mut w,
            &mut sp,
        )
        .unwrap();
        assert!(w.is_empty() && sp.is_empty());
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
            codim2: Vec::new(),
            h: 0.1,
            newton_iters: 1,
            mesh: Vec::new(),
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
            codim2: Vec::new(),
            h: 0.1,
            newton_iters: 1,
            mesh: Vec::new(),
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
            codim2: Vec::new(),
            h: 0.1,
            newton_iters: 1,
            mesh: Vec::new(),
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
