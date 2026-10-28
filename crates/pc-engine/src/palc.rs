//! Pseudo-arclength continuation with step control (SPEC section 5).
//!
//! The tangent solves the bordered system `[F_x; v_prevᵀ] v = e_N` and is scaled to unit
//! length. A step predicts `x + h v` and corrects with Newton's method on
//! `G(x) = (F(x), vᵀ(x - x_k) - h)`. The scheme follows Keller (1977) and Allgower and
//! Georg (1990), Chapter 2; the step rules and tolerances are those of SPEC section 5.

use crate::branch::{Branch, Options, Point, StopReason};
use crate::error::EngineError;
use crate::problem::Problem;
use std::sync::atomic::{AtomicBool, Ordering};

/// Maps an error met during a run to the reason the branch stops. Every error that ends a
/// run goes through here, so the treatment of singular matrices lives in one place.
pub(crate) fn classify(e: EngineError) -> StopReason {
    match e {
        EngineError::Singular(_) => StopReason::Singular,
        EngineError::Cancelled => StopReason::Cancelled,
        other => StopReason::Error(other),
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Outcome of one corrector call.
enum Corrected {
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
fn correct(
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
    let mut jac = vec![0.0; (n - 1) * n];
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
            Err(EngineError::NotFinite(_)) => return Corrected::Failed,
            Err(e) => return Corrected::Stop(classify(e)),
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
    let mut jac = vec![0.0; (n - 1) * n];
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

fn make_point(x: &[f64], v: Vec<f64>, h: f64, iters: usize) -> Point {
    let n = x.len() - 1;
    Point {
        u: x[..n].to_vec(),
        alpha: x[n],
        tangent: v,
        eigenvalues: Vec::new(),
        tests: [f64::NAN; 3],
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
        best = Some(Landed { x, s, iters });
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
        Some(l) if g(&l.x).abs() <= 1e-10 => Ok(l),
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
    let n = p.dim();
    if n < 2 || x0.len() != n {
        return Err(EngineError::BadProblem(
            "the start has the wrong length".into(),
        ));
    }
    if !(opts.hmin > 0.0 && opts.h0 > 0.0 && opts.hmax >= opts.hmin && opts.max_newton >= 1) {
        return Err(EngineError::BadProblem("bad step control options".into()));
    }
    let mut border = vec![0.0; n];
    border[n - 1] = 1.0;
    let (mut v, _det) = tangent(p, x0, &border)?;
    if v[n - 1] * opts.direction < 0.0 {
        for t in v.iter_mut() {
            *t = -*t;
        }
    }
    let mut x = x0.to_vec();
    let first = make_point(&x, v.clone(), 0.0, 0);
    on_point(&first);
    let mut points = vec![first];
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
                        let reason = match tangent(p, &l.x, &v) {
                            Ok((vn, _)) => {
                                let pt = make_point(&l.x, vn, l.s * h, l.iters);
                                on_point(&pt);
                                points.push(pt);
                                StopReason::Bounds
                            }
                            Err(e) => classify(e),
                        };
                        break 'run reason;
                    }
                    Err(r) => break 'run r,
                }
            }
            let (vn, _det) = match tangent(p, &x1, &v) {
                Ok(t) => t,
                Err(e) => break 'run classify(e),
            };
            let pt = make_point(&x1, vn.clone(), h, iters);
            on_point(&pt);
            points.push(pt);
            x = x1;
            v = vn;
            if iters <= 3 {
                h = (1.3 * h).min(opts.hmax);
            }
        }
    };
    Ok(Branch {
        points,
        specials: Vec::new(),
        stop,
    })
}
