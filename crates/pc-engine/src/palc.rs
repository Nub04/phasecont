//! Pseudo-arclength continuation with step control (SPEC section 5).
//!
//! The tangent solves the bordered system `[F_x; v_prevᵀ] v = e_N` and is scaled to unit
//! length. A step predicts `x + h v` and corrects with Newton's method on
//! `G(x) = (F(x), vᵀ(x - x_k) - h)`. The scheme follows Keller (1977) and Allgower and
//! Georg (1990), Chapter 2; the step rules and tolerances are those of SPEC section 5.

use crate::branch::{Branch, Options, Point, StopReason};
use crate::error::EngineError;
use crate::locate::detect_step;
use crate::problem::Problem;
use crate::tests_fn;
use pc_linalg::{null_right, singular_values};
use std::sync::atomic::{AtomicBool, Ordering};

/// Maps an error met during a run to the reason the branch stops. Every error that ends a
/// run goes through here, so this is the only place where errors become a `StopReason`.
/// A singular bordered matrix at a branch point is not an error of the run: `correct` accepts
/// a predictor that already solves the system, and `tangent_or_kernel` turns it into a zero
/// of psi_BP. Other singular matrices reach here and stop the run.
pub(crate) fn classify(e: EngineError) -> StopReason {
    match e {
        EngineError::Singular(_) => StopReason::Singular,
        EngineError::Cancelled => StopReason::Cancelled,
        other => StopReason::Error(other),
    }
}

/// Inf-norm that propagates NaN: any NaN element gives NaN, so `norm_inf(v) <= tol` is
/// false for a vector with a NaN in it.
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Outcome of one corrector call.
pub(crate) enum Corrected {
    Done {
        x: Vec<f64>,
        iters: usize,
    },
    /// No convergence (or a non-finite model value); the caller may shorten the step.
    Failed,
    Stop(StopReason),
}

/// Newton on G(x) = 0 for the hyperplane `vᵀ(x - xk) = hh`, starting at `start`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn correct(
    p: &mut dyn Problem,
    xk: &[f64],
    v: &[f64],
    hh: f64,
    start: Vec<f64>,
    opts: &Options,
    cancel: &AtomicBool,
) -> Corrected {
    let n = p.dim();
    let mut x = start;
    let mut f = vec![0.0; n - 1];
    let mut jac = vec![0.0; p.jacobian_len()];
    let mut rhs = vec![0.0; n];
    let mut dx = vec![0.0; n];
    for it in 1..=opts.max_newton {
        if cancel.load(Ordering::Relaxed) {
            return Corrected::Stop(StopReason::Cancelled);
        }
        let step = p
            .residual(&x, &mut f)
            .and_then(|_| p.jacobian(&x, &mut jac));
        if let Err(e) = step {
            return match e {
                EngineError::NotFinite(_) => Corrected::Failed,
                e => Corrected::Stop(classify(e)),
            };
        }
        for i in 0..n - 1 {
            rhs[i] = -f[i];
        }
        let off: f64 = v
            .iter()
            .zip(&x)
            .zip(xk)
            .map(|((a, b), c)| a * (b - c))
            .sum();
        rhs[n - 1] = -(off - hh);
        match p.bordered_solve(&jac, v, &rhs, &mut dx) {
            Ok(_) => {}
            // A predictor that already solves G(x) = 0 to the residual tolerance on a
            // singular matrix is a branch point met exactly (ruling P2-1): accept it.
            Err(EngineError::Singular(_)) if norm_inf(&rhs) <= opts.tol_f => {
                return Corrected::Done { x, iters: it };
            }
            Err(EngineError::NotFinite(_)) => return Corrected::Failed,
            Err(e) => return Corrected::Stop(classify(e)),
        }
        if norm_inf(&dx).is_nan() {
            return Corrected::Failed;
        }
        for i in 0..n {
            x[i] += dx[i];
        }
        if norm_inf(&dx) <= opts.tol_x {
            match p.residual(&x, &mut f) {
                Ok(()) if norm_inf(&f) <= opts.tol_f => return Corrected::Done { x, iters: it },
                Ok(()) => {}
                Err(EngineError::NotFinite(_)) => return Corrected::Failed,
                Err(e) => return Corrected::Stop(classify(e)),
            }
        }
    }
    Corrected::Failed
}

/// Unit tangent at `x` from the bordered system with the given border. Returns the
/// tangent and the determinant of the bordered matrix.
fn tangent(p: &mut dyn Problem, x: &[f64], border: &[f64]) -> Result<(Vec<f64>, f64), EngineError> {
    let n = p.dim();
    let mut jac = vec![0.0; p.jacobian_len()];
    p.jacobian(x, &mut jac)?;
    let mut rhs = vec![0.0; n];
    rhs[n - 1] = 1.0;
    let mut v = vec![0.0; n];
    let det = p.bordered_solve(&jac, border, &rhs, &mut v)?;
    let nrm = dot(&v, &v).sqrt();
    if !nrm.is_finite() || nrm == 0.0 {
        return Err(EngineError::NotFinite("tangent".into()));
    }
    for t in v.iter_mut() {
        *t /= nrm;
    }
    Ok((v, det))
}

/// Like [`tangent`], but a singular bordered matrix is not an error. At a branch point the
/// matrix is singular, which is a zero of psi_BP (ruling P2-1): the determinant is then 0
/// and the tangent comes from [`kernel_tangent`].
pub(crate) fn tangent_or_kernel(
    p: &mut dyn Problem,
    x: &[f64],
    border: &[f64],
) -> Result<(Vec<f64>, f64), EngineError> {
    match tangent(p, x, border) {
        Err(EngineError::Singular(_)) => Ok((kernel_tangent(p, x, border)?, 0.0)),
        r => r,
    }
}

/// Tangent from the null space of the n-1 by n Jacobian at `x`, for a point where the
/// bordered matrix is singular. With a one-dimensional null space (a fold met exactly, or
/// a border orthogonal to it) it is that null vector, signed so that `vᵀ prev > 0`. With
/// a two-dimensional null space (a branch point) it is `prev` projected onto the null
/// space, so the run goes straight through along the old direction.
fn kernel_tangent(p: &mut dyn Problem, x: &[f64], prev: &[f64]) -> Result<Vec<f64>, EngineError> {
    let n = p.dim();
    if p.jacobian_len() != (n - 1) * n {
        return Err(EngineError::Singular(
            "the bordered matrix is singular and this problem has no dense Jacobian".into(),
        ));
    }
    let mut jac = vec![0.0; (n - 1) * n];
    p.jacobian(x, &mut jac)?;
    let s = singular_values(&jac, n - 1, n)?;
    let two = s[n - 2] <= 1e-8 * s[0];
    let z = null_right(&jac, n - 1, n, if two { 2 } else { 1 })?;
    let mut v = vec![0.0; n];
    if two {
        for q in &z {
            let c = dot(q, prev);
            for (vi, qi) in v.iter_mut().zip(q) {
                *vi += c * qi;
            }
        }
    } else {
        v.copy_from_slice(&z[0]);
        if dot(&v, prev) < 0.0 {
            for t in v.iter_mut() {
                *t = -*t;
            }
        }
    }
    let nrm = dot(&v, &v).sqrt();
    if !nrm.is_finite() || nrm == 0.0 {
        return Err(EngineError::NotFinite("tangent".into()));
    }
    for t in v.iter_mut() {
        *t /= nrm;
    }
    Ok(v)
}

/// Builds an accepted point with its eigenvalues and test values. `det` is the determinant
/// of the bordered tangent solve at `x`.
#[allow(clippy::too_many_arguments)]
fn make_point(
    p: &mut dyn Problem,
    x: &[f64],
    v: Vec<f64>,
    det: f64,
    h: f64,
    iters: usize,
    opts: &Options,
    warnings: &mut Vec<String>,
) -> Result<Point, EngineError> {
    let n = x.len() - 1;
    let t = tests_fn::evaluate(p, x, &v, det, opts, warnings)?;
    Ok(Point {
        u: x[..n].to_vec(),
        alpha: x[n],
        tangent: v,
        eigenvalues: t.eigenvalues,
        tests: t.tests,
        codim2: t.codim2,
        h,
        newton_iters: iters,
    })
}

/// A point without test values, for a landing point whose tangent could not be computed.
fn bare_point(x: &[f64], v: Vec<f64>, h: f64, iters: usize) -> Point {
    let n = x.len() - 1;
    Point {
        u: x[..n].to_vec(),
        alpha: x[n],
        tangent: v,
        eigenvalues: Vec::new(),
        tests: [f64::NAN; 3],
        codim2: Vec::new(),
        h,
        newton_iters: iters,
    }
}

/// A point on a bound of alpha, found by a root search along the step.
struct Landed {
    x: Vec<f64>,
    s: f64,
    iters: usize,
}

/// Cuts the step from `xk` back so that alpha lands on `bound`. `sgn` is +1 for an upper
/// bound and -1 for a lower one, so that g(s) = sgn (alpha(s) - bound) is negative inside.
/// Illinois regula falsi on s in [0, 1]; each trial is corrected on the hyperplane
/// `vᵀ(x - xk) = s h` from the predictor point, like the locating of SPEC section 5.
///
/// The best trial (smallest `|g|`) is kept and accepted when `|g| <= opts.tol_refine`. A
/// corrector failure inside the landing returns `NewtonFailed`; it does not shrink `s`.
#[allow(clippy::too_many_arguments)]
fn land_on_bound(
    p: &mut dyn Problem,
    xk: &[f64],
    v: &[f64],
    h: f64,
    x1: &[f64],
    bound: f64,
    sgn: f64,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Landed, StopReason> {
    let n = xk.len();
    let g = |x: &[f64]| sgn * (x[n - 1] - bound);
    let (mut a, mut fa) = (0.0, g(xk));
    let (mut b, mut fb) = (1.0, g(x1));
    let mut side = 0i8;
    let mut best: Option<Landed> = None;
    for _ in 0..opts.max_locate {
        let s = (a * fb - b * fa) / (fb - fa);
        let start: Vec<f64> = xk.iter().zip(v).map(|(x, t)| x + s * h * t).collect();
        let (x, iters) = match correct(p, xk, v, s * h, start, opts, cancel) {
            Corrected::Done { x, iters } => (x, iters),
            Corrected::Failed => {
                return Err(StopReason::Error(EngineError::NewtonFailed {
                    iters: opts.max_newton,
                    residual: f64::NAN,
                }));
            }
            Corrected::Stop(r) => return Err(r),
        };
        let fs = g(&x);
        let hit = fs.abs() <= opts.tol_refine;
        let better = best.as_ref().is_none_or(|l| fs.abs() < g(&l.x).abs());
        if better {
            best = Some(Landed { x, s, iters });
        }
        if hit {
            break;
        }
        if fs * fb > 0.0 {
            b = s;
            fb = fs;
            if side == -1 {
                fa /= 2.0;
            }
            side = -1;
        } else {
            a = s;
            fa = fs;
            if side == 1 {
                fb /= 2.0;
            }
            side = 1;
        }
    }
    match best {
        Some(l) if g(&l.x).abs() <= opts.tol_refine => Ok(l),
        Some(l) => Err(StopReason::Error(EngineError::NewtonFailed {
            iters: opts.max_locate,
            residual: g(&l.x).abs(),
        })),
        None => Err(StopReason::Error(EngineError::BadProblem(
            "max_locate is zero".into(),
        ))),
    }
}

/// Continues the curve of `p` from the solution `x0`. The first tangent has the sign of
/// `opts.direction` in its last component. Errors before the first point exists (bad
/// options, a singular first tangent, an evaluation error at `x0`) are returned; any
/// later problem ends the branch with a [`StopReason`] and keeps the points so far.
pub fn continue_branch(
    p: &mut dyn Problem,
    x0: &[f64],
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    run_branch(p, x0, None, opts, false, cancel, on_point)
}

/// [`continue_branch`] with the flag of `EpSetup::start_is_special`: when set, no special
/// point is looked for on the first step. With `tangent0` the first tangent is that vector
/// (normalized, times the sign of `opts.direction`) instead of the bordered solve; this is
/// how a run leaves a branch point on the other branch.
pub(crate) fn run_branch(
    p: &mut dyn Problem,
    x0: &[f64],
    tangent0: Option<&[f64]>,
    opts: &Options,
    start_is_special: bool,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let n = p.dim();
    if n < 2 || x0.len() != n {
        return Err(EngineError::BadProblem(
            "the start has the wrong length".into(),
        ));
    }
    let finite = opts.h0.is_finite() && opts.hmin.is_finite() && opts.hmax.is_finite();
    if !(finite
        && opts.hmin > 0.0
        && opts.h0 > 0.0
        && opts.hmax >= opts.hmin
        && opts.max_newton >= 1)
    {
        return Err(EngineError::BadProblem("bad step control options".into()));
    }
    let (mut v, det) = match tangent0 {
        Some(t0) => {
            if t0.len() != n {
                return Err(EngineError::BadProblem(
                    "the initial tangent has the wrong length".into(),
                ));
            }
            let nrm = dot(t0, t0).sqrt();
            if !nrm.is_finite() || nrm == 0.0 {
                return Err(EngineError::BadProblem(
                    "the initial tangent is zero or not finite".into(),
                ));
            }
            let s = if opts.direction < 0.0 { -1.0 } else { 1.0 } / nrm;
            let v: Vec<f64> = t0.iter().map(|c| c * s).collect();
            // Only the determinant is wanted; at a branch point it is about 0.
            let (_, det) = tangent_or_kernel(p, x0, &v)?;
            (v, det)
        }
        None => {
            let mut border = vec![0.0; n];
            border[n - 1] = 1.0;
            let (mut v, mut det) = tangent_or_kernel(p, x0, &border)?;
            if v[n - 1] * opts.direction < 0.0 {
                for t in v.iter_mut() {
                    *t = -*t;
                }
                // det[J; w] is linear in the border w, so a flipped tangent flips the
                // determinant.
                det = -det;
            }
            (v, det)
        }
    };
    let mut warnings = Vec::new();
    let mut x = x0.to_vec();
    let first = make_point(p, &x, v.clone(), det, 0.0, 0, opts, &mut warnings)?;
    on_point(&first);
    let mut points = vec![first];
    let mut specials = Vec::new();
    let (lo, hi) = opts.bounds;
    let stop = 'run: {
        if x[n - 1] < lo || x[n - 1] > hi {
            break 'run StopReason::Bounds;
        }
        let mut h = opts.h0.min(opts.hmax);
        loop {
            if cancel.load(Ordering::Relaxed) {
                break 'run StopReason::Cancelled;
            }
            if points.len() > opts.max_steps {
                break 'run StopReason::MaxSteps;
            }
            // Corrector, halving h on failure.
            let (x1, iters) = loop {
                if h < opts.hmin {
                    break 'run StopReason::StepTooSmall;
                }
                let start: Vec<f64> = x.iter().zip(&v).map(|(a, t)| a + h * t).collect();
                match correct(p, &x, &v, h, start, opts, cancel) {
                    Corrected::Done { x, iters } => break (x, iters),
                    Corrected::Failed => h /= 2.0,
                    Corrected::Stop(r) => break 'run r,
                }
            };
            // Bounds: cut the step back so alpha lands on the bound.
            let a1 = x1[n - 1];
            if a1 < lo || a1 > hi {
                let (bound, sgn) = if a1 > hi { (hi, 1.0) } else { (lo, -1.0) };
                if sgn * (x[n - 1] - bound) >= -opts.tol_refine {
                    break 'run StopReason::Bounds;
                }
                match land_on_bound(p, &x, &v, h, &x1, bound, sgn, opts, cancel) {
                    Ok(l) => {
                        // The landed point is kept even when its tangent or tests fail.
                        let built = tangent_or_kernel(p, &l.x, &v).and_then(|(vn, d)| {
                            make_point(p, &l.x, vn, d, l.s * h, l.iters, opts, &mut warnings)
                        });
                        let (pt, reason) = match built {
                            Ok(pt) => (pt, StopReason::Bounds),
                            Err(e) => {
                                let bare = bare_point(&l.x, v.clone(), l.s * h, l.iters);
                                if e == EngineError::Cancelled {
                                    (bare, StopReason::Cancelled)
                                } else {
                                    warnings
                                        .push(format!("tests at the bound point are missing: {e}"));
                                    (bare, StopReason::Bounds)
                                }
                            }
                        };
                        on_point(&pt);
                        points.push(pt);
                        let found = if start_is_special && points.len() == 2 {
                            Ok(())
                        } else {
                            detect_step(
                                p,
                                &x,
                                &v,
                                l.s * h,
                                &points,
                                opts,
                                cancel,
                                &mut warnings,
                                &mut specials,
                            )
                        };
                        break 'run found.err().unwrap_or(reason);
                    }
                    Err(r) => break 'run r,
                }
            }
            let built = tangent_or_kernel(p, &x1, &v)
                .and_then(|(vn, d)| make_point(p, &x1, vn, d, h, iters, opts, &mut warnings));
            let pt = match built {
                Ok(pt) => pt,
                Err(e) => break 'run classify(e),
            };
            on_point(&pt);
            let vn = pt.tangent.clone();
            points.push(pt);
            let skip = start_is_special && points.len() == 2;
            if !skip
                && let Err(r) = detect_step(
                    p,
                    &x,
                    &v,
                    h,
                    &points,
                    opts,
                    cancel,
                    &mut warnings,
                    &mut specials,
                )
            {
                break 'run r;
            }
            x = x1;
            v = vn;
            if iters <= 3 {
                h = (1.3 * h).min(opts.hmax);
            }
        }
    };
    Ok(Branch {
        points,
        specials,
        stop,
        warnings,
    })
}
