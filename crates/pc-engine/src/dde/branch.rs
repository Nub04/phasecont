//! Hopf points along a branch of delay equilibria (SPEC section 8e).
//!
//! The branch is a curve of solutions of `G(x, p) = f(x, x, p) = 0`, continued by
//! `palc::run_branch` on the collapsed model. The roots of the characteristic equation are
//! counted at every point, and a step on which the count changes is bisected with
//! `palc::correct` to bracket each crossing. The method is my own, built from the definition
//! of a local bifurcation of an equilibrium; SPEC section 8e says so.

use super::eq::{Collapsed, check_params, stability};
use super::roots::{CharRoot, RootOptions, RootSet};
use crate::branch::{Options, StopReason};
use crate::ep::{EpProblem, EpSetup, equilibrium as ep_equilibrium};
use crate::error::EngineError;
use crate::model::DdeModel;
use crate::palc::{Corrected, correct, run_branch};
use pc_ir::DelayTime;
use std::sync::atomic::{AtomicBool, Ordering};

/// A pair of roots that crosses with `|Im| <= OMEGA_MIN` is reported as a real crossing.
const OMEGA_MIN: f64 = 1e-8;
/// A bracket is at most this fraction of the step length wide.
const REL_WIDTH: f64 = 1e-12;
/// Halvings of one bracket.
const MAX_HALVINGS: usize = 60;
/// Crossings taken from one step.
const MAX_CROSSINGS: usize = 16;

/// One point of a branch of delay equilibria.
#[derive(Clone, Debug, PartialEq)]
pub struct DdePoint {
    pub alpha: f64,
    pub x: Vec<f64>,
    /// Characteristic roots with `Re > 0`.
    pub unstable: usize,
}

/// A Hopf point, a fold or a branch point on the branch.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeSpecial {
    /// `"H"`, `"LP"` or `"BP"`.
    pub label: &'static str,
    pub alpha: f64,
    pub x: Vec<f64>,
    /// The frequency of an `H`; `None` for `LP` and `BP`.
    pub omega: Option<f64>,
    /// The two points that bracket the step.
    pub between: [usize; 2],
    /// Counts at the two ends of the bracket of an `H`, at the ends of the step for `LP`
    /// and `BP`.
    pub unstable_before: usize,
    pub unstable_after: usize,
}

/// The result of [`continue_equilibrium`].
#[derive(Clone, Debug, PartialEq)]
pub struct DdeBranch {
    pub points: Vec<DdePoint>,
    /// Sorted along the branch: by step, then by distance from the start of the step.
    pub specials: Vec<DdeSpecial>,
    pub stop: StopReason,
    pub warnings: Vec<String>,
}

/// One crossing: the count is `n_lo` at `lo` and `n_hi` at `hi`.
#[derive(Clone, Debug, PartialEq)]
struct Bracket {
    lo: f64,
    hi: f64,
    n_lo: usize,
    n_hi: usize,
}

#[derive(Debug, Default)]
struct Bisection {
    brackets: Vec<Bracket>,
    /// The step had more crossings than `MAX_CROSSINGS`.
    capped: bool,
    /// The value of `h` at which `trial` failed; the search of the step ended there.
    failed_at: Option<f64>,
}

/// Brackets the crossings of one step of length `d`. `trial(h)` is the count at `X(h)`, or
/// `None` when the corrector failed there. The count is `n_lo` at 0 and `n_hi` at `d`.
fn bisect_crossings(
    trial: &mut dyn FnMut(f64) -> Option<usize>,
    d: f64,
    n_lo: usize,
    n_hi: usize,
) -> Bisection {
    let mut out = Bisection::default();
    let mut from = 0.0;
    let mut n_from = n_lo;
    while n_from != n_hi {
        if out.brackets.len() == MAX_CROSSINGS {
            out.capped = true;
            break;
        }
        let (mut lo, mut hi, mut n_at_hi) = (from, d, n_hi);
        for _ in 0..MAX_HALVINGS {
            if hi - lo <= REL_WIDTH * d {
                break;
            }
            let m = 0.5 * (lo + hi);
            match trial(m) {
                None => {
                    out.failed_at = Some(m);
                    return out;
                }
                Some(c) if c == n_from => lo = m,
                Some(c) => {
                    hi = m;
                    n_at_hi = c;
                }
            }
        }
        out.brackets.push(Bracket {
            lo,
            hi,
            n_lo: n_from,
            n_hi: n_at_hi,
        });
        from = hi;
        n_from = n_at_hi;
    }
    out
}

/// Roots with `Re > 0`.
fn count(set: &RootSet) -> usize {
    set.roots.iter().filter(|r| r.re > 0.0).count()
}

/// The root with the smallest positive real part.
fn leading_positive(set: &RootSet) -> Option<CharRoot> {
    set.roots
        .iter()
        .filter(|r| r.re > 0.0)
        .min_by(|a, b| a.re.total_cmp(&b.re))
        .copied()
}

/// Everything a trial needs: the model, the setup and the options of the run.
struct Ctx<'a> {
    m: &'a mut DdeModel,
    setup: &'a EpSetup,
    roots: &'a RootOptions,
    opts: &'a Options,
    cancel: &'a AtomicBool,
    /// Root sets with candidates that did not converge, and the first one.
    unrefined: usize,
    first_unrefined: Option<String>,
}

impl Ctx<'_> {
    /// The roots at the full state `x = (u, alpha)`.
    fn roots_at(&mut self, x: &[f64], what: &str) -> Result<RootSet, EngineError> {
        let n = x.len() - 1;
        let mut p = self.setup.params.clone();
        p[self.setup.free] = x[n];
        let set = stability(self.m, &x[..n], &p, self.roots)?;
        if !set.unrefined.is_empty() {
            self.unrefined += 1;
            if self.first_unrefined.is_none() {
                self.first_unrefined = Some(what.to_string());
            }
        }
        Ok(set)
    }

    /// `X(h)`: the solution of `G = 0` on the hyperplane `v^T (X - xk) = h`, or `None` when
    /// the corrector does not converge there.
    fn correct_at(
        &mut self,
        xk: &[f64],
        v: &[f64],
        h: f64,
    ) -> Result<Option<Vec<f64>>, EngineError> {
        let start: Vec<f64> = xk.iter().zip(v).map(|(a, t)| a + h * t).collect();
        let mut c = Collapsed { m: &mut *self.m };
        let mut problem = EpProblem::new(&mut c, self.setup)?;
        match correct(&mut problem, xk, v, h, start, self.opts, self.cancel) {
            Corrected::Done { x, .. } => Ok(Some(x)),
            Corrected::Failed => Ok(None),
            Corrected::Stop(StopReason::Cancelled) => Err(EngineError::Cancelled),
            Corrected::Stop(StopReason::Error(e)) => Err(e),
            Corrected::Stop(_) => Ok(None),
        }
    }

    fn check_cancel(&self) -> Result<(), EngineError> {
        if self.cancel.load(Ordering::Relaxed) {
            Err(EngineError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// Continues the equilibrium of `model` found from `setup.u0` along the declared parameter
/// `setup.free` and reports the Hopf points, folds and branch points on the way
/// (SPEC section 8e). `setup.params` holds the declared parameters only. `opts.detect_h` is
/// ignored, and `setup.tangent0` and `setup.start_is_special` are not used. A cancelled run
/// is `Err(EngineError::Cancelled)`.
pub fn continue_equilibrium(
    model: &mut DdeModel,
    setup: &EpSetup,
    roots: &RootOptions,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeBranch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    if roots.re_min >= 0.0 || roots.re_min.is_nan() {
        return Err(EngineError::BadProblem(format!(
            "re_min is {}; it must be negative so that every root with a positive real part \
             is searched for",
            roots.re_min
        )));
    }
    check_params(model, &setup.params)?;
    if setup.free >= model.n_user_params {
        return Err(EngineError::BadProblem(format!(
            "parameter index {} is out of range",
            setup.free
        )));
    }
    let free_is_delay = model
        .delays
        .iter()
        .any(|s| matches!(s.time, DelayTime::Param(j) if j == setup.free));
    if free_is_delay && (opts.bounds.0 <= 0.0 || opts.bounds.0.is_nan()) {
        return Err(EngineError::BadProblem(
            "the free parameter is a delay time, so bounds must have a lower bound above 0".into(),
        ));
    }
    let mut opts = opts.clone();
    opts.detect_h = false;

    // The start: Newton on G at the starting value of alpha.
    let u0 = ep_equilibrium(&mut Collapsed { m: &mut *model }, setup, &opts)?;
    let mut start = setup.clone();
    start.u0 = u0.clone();
    let mut x0 = u0;
    x0.push(setup.params[setup.free]);

    let mut ctx = Ctx {
        m: model,
        setup: &start,
        roots,
        opts: &opts,
        cancel,
        unrefined: 0,
        first_unrefined: None,
    };
    // Counting the first point here reports a bad `n` before the run starts.
    let first = ctx.roots_at(&x0, "point 0")?;

    let branch = {
        let mut c = Collapsed { m: &mut *ctx.m };
        let mut problem = EpProblem::new(&mut c, &start)?;
        run_branch(&mut problem, &x0, None, &opts, false, cancel, &mut |_| {})?
    };
    let mut warnings = branch.warnings.clone();

    let states: Vec<Vec<f64>> = branch
        .points
        .iter()
        .map(|p| {
            let mut x = p.u.clone();
            x.push(p.alpha);
            x
        })
        .collect();
    let mut counts = Vec::with_capacity(states.len());
    counts.push(count(&first));
    for (i, x) in states.iter().enumerate().skip(1) {
        ctx.check_cancel()?;
        counts.push(count(&ctx.roots_at(x, &format!("point {i}"))?));
    }

    // (step, distance from the start of the step, special)
    let mut found: Vec<(usize, f64, DdeSpecial)> = Vec::new();
    for sp in &branch.specials {
        let label = match sp.label.as_str() {
            "LP" => "LP",
            "BP" => "BP",
            _ => continue,
        };
        if sp.index == 0 || sp.index >= states.len() {
            continue;
        }
        let k = sp.index - 1;
        let mut x = sp.point.u.clone();
        x.push(sp.point.alpha);
        let d = norm(
            &x.iter()
                .zip(&states[k])
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        found.push((
            k,
            d,
            DdeSpecial {
                label,
                alpha: sp.point.alpha,
                x: sp.point.u.clone(),
                omega: None,
                between: [k, k + 1],
                unstable_before: counts[k],
                unstable_after: counts[k + 1],
            },
        ));
    }

    for k in 0..states.len().saturating_sub(1) {
        let (n0, n1) = (counts[k], counts[k + 1]);
        if n0 == n1 {
            continue;
        }
        ctx.check_cancel()?;
        let (xk, xk1) = (&states[k], &states[k + 1]);
        let diff: Vec<f64> = xk1.iter().zip(xk).map(|(a, b)| a - b).collect();
        let d = norm(&diff);
        if !(d.is_finite() && d > 0.0) {
            continue;
        }
        let v: Vec<f64> = diff.iter().map(|a| a / d).collect();

        let mut err: Option<EngineError> = None;
        let bis = {
            let mut trial = |h: f64| -> Option<usize> {
                let what = format!("a bisection trial in the step from point {k}");
                let run = ctx.correct_at(xk, &v, h).and_then(|x| match x {
                    Some(x) => ctx.roots_at(&x, &what).map(|s| Some(count(&s))),
                    None => Ok(None),
                });
                match run {
                    Ok(c) => c,
                    Err(e) => {
                        err = Some(e);
                        None
                    }
                }
            };
            bisect_crossings(&mut trial, d, n0, n1)
        };
        if let Some(e) = err {
            return Err(e);
        }
        if let Some(h) = bis.failed_at {
            warnings.push(format!(
                "the corrector failed at h = {h:e} inside the step from point {k} to point {}, \
                 so the crossings after that are not located",
                k + 1
            ));
        }
        if bis.capped {
            warnings.push(format!(
                "the step from point {k} to point {} has more than {MAX_CROSSINGS} crossings; \
                 only the first {MAX_CROSSINGS} are reported",
                k + 1
            ));
        }
        let explained = branch
            .specials
            .iter()
            .any(|s| s.index == k + 1 && (s.label == "LP" || s.label == "BP"));
        let mut real_unexplained = false;
        for b in &bis.brackets {
            // The roots at the end with the larger count.
            let h_end = if b.n_hi > b.n_lo { b.hi } else { b.lo };
            let x_end = if h_end <= 0.0 {
                Some(xk.clone())
            } else if h_end >= d {
                Some(xk1.clone())
            } else {
                ctx.correct_at(xk, &v, h_end)?
            };
            let Some(x_end) = x_end else {
                warnings.push(format!(
                    "the corrector failed at the end of a bracket in the step from point {k} to \
                     point {}, so that crossing is not classified",
                    k + 1
                ));
                continue;
            };
            let set = ctx.roots_at(&x_end, "a classification point")?;
            let lead = leading_positive(&set);
            match lead {
                Some(r) if r.im.abs() > OMEGA_MIN => {
                    let n = x_end.len() - 1;
                    found.push((
                        k,
                        0.5 * (b.lo + b.hi),
                        DdeSpecial {
                            label: "H",
                            alpha: x_end[n],
                            x: x_end[..n].to_vec(),
                            omega: Some(r.im.abs()),
                            between: [k, k + 1],
                            unstable_before: b.n_lo,
                            unstable_after: b.n_hi,
                        },
                    ));
                }
                _ => {
                    if !explained {
                        real_unexplained = true;
                    }
                }
            }
        }
        if real_unexplained {
            warnings.push(format!(
                "a real root crosses zero between points {k} and {} and no LP or BP is \
                 reported there",
                k + 1
            ));
        }
    }
    if ctx.unrefined > 0 {
        warnings.push(format!(
            "{} root sets had candidates that did not converge (first: {}); the root count may \
             be too low there, so raise n",
            ctx.unrefined,
            ctx.first_unrefined.as_deref().unwrap_or("unknown")
        ));
    }

    found.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let points = branch
        .points
        .iter()
        .zip(&counts)
        .map(|(p, &unstable)| DdePoint {
            alpha: p.alpha,
            x: p.u.clone(),
            unstable,
        })
        .collect();
    Ok(DdeBranch {
        points,
        specials: found.into_iter().map(|(_, _, s)| s).collect(),
        stop: branch.stop,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count is the number of `jumps` at or below `h`.
    fn jumps_trial(jumps: Vec<f64>) -> impl FnMut(f64) -> Option<usize> {
        move |h| Some(jumps.iter().filter(|&&j| j <= h).count())
    }

    #[test]
    fn brackets_contain_the_jumps_and_are_narrow() {
        let d = 2.0;
        let jumps = vec![0.3 * d, 0.7 * d];
        let mut t = jumps_trial(jumps.clone());
        let b = bisect_crossings(&mut t, d, 0, 2);
        assert!(!b.capped);
        assert_eq!(b.failed_at, None);
        assert_eq!(b.brackets.len(), 2);
        for (br, j) in b.brackets.iter().zip(&jumps) {
            assert!(br.lo < *j && *j <= br.hi, "{br:?} misses {j}");
            assert!(br.hi - br.lo <= REL_WIDTH * d, "{br:?} is wide");
        }
        assert_eq!((b.brackets[0].n_lo, b.brackets[0].n_hi), (0, 1));
        assert_eq!((b.brackets[1].n_lo, b.brackets[1].n_hi), (1, 2));
    }

    #[test]
    fn a_count_that_jumps_by_two_at_once_is_one_crossing() {
        let d = 1.0;
        let mut t = jumps_trial(vec![0.4, 0.4]);
        let b = bisect_crossings(&mut t, d, 0, 2);
        assert_eq!(b.brackets.len(), 1);
        assert_eq!((b.brackets[0].n_lo, b.brackets[0].n_hi), (0, 2));
    }

    #[test]
    fn twenty_jumps_give_sixteen_crossings_and_the_cap_flag() {
        let d = 1.0;
        let jumps: Vec<f64> = (1..=20).map(|i| f64::from(i) / 21.0).collect();
        let mut t = jumps_trial(jumps);
        let b = bisect_crossings(&mut t, d, 0, 20);
        assert_eq!(b.brackets.len(), MAX_CROSSINGS);
        assert!(b.capped);
        assert_eq!(b.failed_at, None);
    }

    #[test]
    fn exactly_sixteen_jumps_are_not_capped() {
        let d = 1.0;
        let jumps: Vec<f64> = (1..=16).map(|i| f64::from(i) / 17.0).collect();
        let mut t = jumps_trial(jumps);
        let b = bisect_crossings(&mut t, d, 0, 16);
        assert_eq!(b.brackets.len(), 16);
        assert!(!b.capped);
    }

    #[test]
    fn a_failed_trial_ends_the_search_and_keeps_earlier_crossings() {
        let d = 1.0;
        let jumps = [0.2, 0.8];
        let mut t = |h: f64| -> Option<usize> {
            if h > 0.5 {
                None
            } else {
                Some(jumps.iter().filter(|&&j| j <= h).count())
            }
        };
        let b = bisect_crossings(&mut t, d, 0, 2);
        assert_eq!(b.brackets.len(), 1, "{b:?}");
        assert!(b.brackets[0].lo < 0.2 && 0.2 <= b.brackets[0].hi);
        let h = b.failed_at.expect("the failure is recorded");
        assert!(h > 0.5 && h <= d, "failed at {h}");
    }

    #[test]
    fn a_failure_in_the_first_search_gives_no_crossing() {
        let mut t = |_: f64| -> Option<usize> { None };
        let b = bisect_crossings(&mut t, 1.0, 0, 1);
        assert!(b.brackets.is_empty());
        assert_eq!(b.failed_at, Some(0.5));
    }

    #[test]
    fn equal_counts_give_nothing() {
        let mut t = |_: f64| -> Option<usize> { panic!("no trial is needed") };
        let b = bisect_crossings(&mut t, 1.0, 3, 3);
        assert!(b.brackets.is_empty() && !b.capped && b.failed_at.is_none());
    }
}
