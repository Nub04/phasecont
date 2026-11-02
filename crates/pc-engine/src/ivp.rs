//! Initial value problems x' = f(t, x), integrated with Dormand-Prince 5(4).
//!
//! The method is the embedded pair of Dormand and Prince (1980) with the first-same-as-last
//! property. The error norm, the step size controller (with the stabilizing exponent
//! beta = 0.04) and the starting step follow Hairer, Norsett and Wanner, Solving Ordinary
//! Differential Equations I, sections II.4 and II.5, as in their `dopri5` code. The tableau
//! is checked by the order tests below, not only by the row sums.
//!
//! A run goes forward or backward from `t0` and lands exactly on `t1`. Time reaches the
//! model as the last entry of `p`, so models that read `t` work (see
//! [`crate::build_allowing_t`]). The branch holds one [`Point`] per accepted step plus the
//! start, with `alpha` the time, `u` the state and `h` the signed step that reached it.

use crate::branch::{Branch, Point, StopReason};
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::EntryKind;
use std::sync::atomic::{AtomicBool, Ordering};

/// Settings of an initial value run.
#[derive(Clone, Debug, PartialEq)]
pub struct IvpOptions {
    /// Relative tolerance of the local error estimate.
    pub rtol: f64,
    /// Absolute tolerance of the local error estimate.
    pub atol: f64,
    /// Accepted steps before the run stops with `MaxSteps`.
    pub max_steps: usize,
    /// Largest step length. Infinite means no cap.
    pub hmax: f64,
}

impl Default for IvpOptions {
    fn default() -> Self {
        IvpOptions {
            rtol: 1e-8,
            atol: 1e-10,
            max_steps: 100_000,
            hmax: f64::INFINITY,
        }
    }
}

/// The data of an initial value problem.
#[derive(Clone, Debug, PartialEq)]
pub struct IvpSetup {
    /// The state at `t0`.
    pub x0: Vec<f64>,
    /// The parameters, `n_p` values. Time is added by the integrator.
    pub params: Vec<f64>,
    pub t0: f64,
    /// The end time. A value below `t0` integrates backward.
    pub t1: f64,
}

/// Nodes c2 to c7 of the pair.
const C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// Rows a21 to a76. The last row is the fifth order solution, which makes the pair FSAL.
const A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// e_i = b_i - bhat_i for the seven stages: the error estimate is h sum e_i k_i.
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// Controller constants of `dopri5`: safety factor, the step may shrink by at most FAC1 and
/// grow by at most FAC2 in one step, and the exponent of the previous error.
const SAFE: f64 = 0.9;
const FAC1: f64 = 0.2;
const FAC2: f64 = 10.0;
const BETA: f64 = 0.04;
/// Unit roundoff the step-too-small test uses, as in `dopri5`.
const UROUND: f64 = 2.3e-16;

/// The right-hand side as a closure over time, state and output.
type Rhs<'a> = dyn FnMut(f64, &[f64], &mut [f64]) -> Result<(), EngineError> + 'a;

/// One Dormand-Prince step from (t, y) with the stage `k[0] = f(t, y)` given. Fills
/// `k[1..7]`, `y1` (the fifth order solution) and returns the scaled error norm. `t_end` is
/// the time to use for the stages at c = 1.
fn step(
    f: &mut Rhs,
    (t, y, h, t_end): (f64, &[f64], f64, f64),
    k: &mut [Vec<f64>; 7],
    ytmp: &mut [f64],
    y1: &mut [f64],
    (rtol, atol): (f64, f64),
) -> Result<f64, EngineError> {
    let n = y.len();
    for s in 1..=5 {
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..s {
                acc += A[s - 1][j] * k[j][i];
            }
            ytmp[i] = y[i] + h * acc;
        }
        let ts = if C[s - 1] == 1.0 {
            t_end
        } else {
            t + C[s - 1] * h
        };
        f(ts, ytmp, &mut k[s])?;
    }
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..6 {
            acc += A[5][j] * k[j][i];
        }
        y1[i] = y[i] + h * acc;
    }
    f(t_end, y1, &mut k[6])?;
    let mut sum = 0.0;
    for i in 0..n {
        let mut e = 0.0;
        for (j, ej) in E.iter().enumerate() {
            e += ej * k[j][i];
        }
        let sk = atol + rtol * y[i].abs().max(y1[i].abs());
        let r = h * e / sk;
        sum += r * r;
    }
    Ok((sum / n as f64).sqrt())
}

/// The starting step of `dopri5` (`HINIT`), signed with the direction.
fn initial_step(
    f: &mut Rhs,
    (t, y, f0): (f64, &[f64], &[f64]),
    (rtol, atol, hmax): (f64, f64, f64),
    posneg: f64,
) -> Result<f64, EngineError> {
    let n = y.len();
    let sk: Vec<f64> = y.iter().map(|v| atol + rtol * v.abs()).collect();
    let dnf: f64 = (0..n).map(|i| (f0[i] / sk[i]).powi(2)).sum();
    let dny: f64 = (0..n).map(|i| (y[i] / sk[i]).powi(2)).sum();
    let h = if dnf <= 1e-10 || dny <= 1e-10 {
        1e-6
    } else {
        (dny / dnf).sqrt() * 0.01
    };
    let h = posneg * h.min(hmax);
    let y1: Vec<f64> = (0..n).map(|i| y[i] + h * f0[i]).collect();
    let mut f1 = vec![0.0; n];
    f(t + h, &y1, &mut f1)?;
    let der2 = (0..n)
        .map(|i| ((f1[i] - f0[i]) / sk[i]).powi(2))
        .sum::<f64>()
        .sqrt()
        / h.abs();
    let der12 = der2.abs().max(dnf.sqrt());
    let h1 = if der12 <= 1e-15 {
        (1e-6_f64).max(h.abs() * 1e-3)
    } else {
        (0.01 / der12).powf(1.0 / 5.0)
    };
    Ok(posneg * (100.0 * h.abs()).min(h1).min(hmax))
}

fn check(setup: &IvpSetup, opts: &IvpOptions, n: usize, n_p: usize) -> Result<(), EngineError> {
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
    if !setup.t0.is_finite() || !setup.t1.is_finite() {
        return bad("tspan must be finite".into());
    }
    if setup.t0 == setup.t1 {
        return bad("tspan has equal ends".into());
    }
    if !(opts.rtol.is_finite() && opts.rtol > 0.0) {
        return bad(format!(
            "rtol must be positive and finite, got {}",
            opts.rtol
        ));
    }
    if !(opts.atol.is_finite() && opts.atol > 0.0) {
        return bad(format!(
            "atol must be positive and finite, got {}",
            opts.atol
        ));
    }
    if opts.hmax.is_nan() || opts.hmax <= 0.0 {
        return bad(format!("hmax must be positive, got {}", opts.hmax));
    }
    if opts.max_steps == 0 {
        return bad("max_steps must be at least 1".into());
    }
    Ok(())
}

/// Integrates x' = f(t, x) from `setup.t0` to `setup.t1`, calling `on_point` for the start
/// and for each accepted step. Returns the branch with the reason the run stopped.
///
/// Input errors and a cancel before the start are `Err`. After that a failure, such as a
/// model entry that gives NaN, ends the run with `StopReason::Error` and keeps the points
/// found so far.
pub fn integrate(
    model: &mut dyn ModelEval,
    setup: &IvpSetup,
    opts: &IvpOptions,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let (n, n_p) = (model.n(), model.n_p());
    check(setup, opts, n, n_p)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut pbuf = setup.params.clone();
    pbuf.push(setup.t0);
    let mut f = |t: f64, x: &[f64], out: &mut [f64]| -> Result<(), EngineError> {
        pbuf[n_p] = t;
        model.eval(&EntryKind::F, x, &pbuf, &[], out)
    };
    let mut branch = Branch {
        points: Vec::new(),
        specials: Vec::new(),
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    };
    let mut push = |branch: &mut Branch, t: f64, x: &[f64], h: f64| {
        let p = Point {
            u: x.to_vec(),
            alpha: t,
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

    let mut x = setup.t0;
    let mut y = setup.x0.clone();
    push(&mut branch, x, &y, 0.0);
    let mut k: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; n]);
    let mut ytmp = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let xend = setup.t1;
    let posneg = (xend - x).signum();
    let tols = (opts.rtol, opts.atol);

    let mut h = match f(x, &y, &mut k[0]).and_then(|()| {
        initial_step(
            &mut f,
            (x, &y, &k[0]),
            (opts.rtol, opts.atol, opts.hmax),
            posneg,
        )
    }) {
        Ok(h) => h,
        Err(e) => {
            branch.stop = StopReason::Error(e);
            return Ok(branch);
        }
    };

    let expo1 = 0.2 - BETA * 0.75;
    let mut facold = 1e-4_f64;
    let mut reject = false;
    let mut last = false;
    let mut accepted = 0usize;
    branch.stop = loop {
        if cancel.load(Ordering::Relaxed) {
            break StopReason::Cancelled;
        }
        // Written as a positive test so that a NaN step size also stops the run.
        let step_ok = 0.1 * h.abs() > x.abs() * UROUND;
        if !step_ok {
            break StopReason::StepTooSmall;
        }
        if (x + 1.01 * h - xend) * posneg > 0.0 {
            h = xend - x;
            last = true;
        }
        let t_end = if last { xend } else { x + h };
        let err = match step(&mut f, (x, &y, h, t_end), &mut k, &mut ytmp, &mut y1, tols) {
            Ok(e) => e,
            Err(e) => break StopReason::Error(e),
        };
        let fac11 = err.powf(expo1);
        let fac = (fac11 / facold.powf(BETA) / SAFE).clamp(1.0 / FAC2, 1.0 / FAC1);
        let mut hnew = h / fac;
        if err <= 1.0 {
            facold = err.max(1e-4);
            accepted += 1;
            x = if last { xend } else { x + h };
            std::mem::swap(&mut y, &mut y1);
            let (first, rest) = k.split_at_mut(1);
            first[0].copy_from_slice(&rest[5]);
            push(&mut branch, x, &y, h);
            if last {
                break StopReason::End;
            }
            if accepted >= opts.max_steps {
                break StopReason::MaxSteps;
            }
            if hnew.abs() > opts.hmax {
                hnew = posneg * opts.hmax;
            }
            if reject {
                hnew = posneg * hnew.abs().min(h.abs());
            }
            reject = false;
            h = hnew;
        } else {
            // The clamp above would give a NaN error a growth factor; shrink instead.
            hnew = h / (1.0 / FAC1).min(fac11 / SAFE);
            reject = true;
            last = false;
            h = hnew;
        }
    };
    Ok(branch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y' = y cos t, y(0) = 1, with y = exp(sin t).
    fn rhs(t: f64, y: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        out[0] = y[0] * t.cos();
        Ok(())
    }

    /// Fixed-step run to `tend` in `steps` steps. With `embedded` the propagated value is
    /// the fourth order one, y1 - h sum e_i k_i, and k1 is evaluated afresh.
    fn fixed(steps: usize, tend: f64, embedded: bool) -> f64 {
        let h = tend / steps as f64;
        let mut y = vec![1.0];
        let mut k: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0]);
        let (mut ytmp, mut y1) = (vec![0.0], vec![0.0]);
        let mut f = rhs;
        for s in 0..steps {
            let t = s as f64 * h;
            f(t, &y, &mut k[0]).unwrap();
            // A huge tolerance keeps the norm finite; only y1 and the k are used.
            step(
                &mut f,
                (t, &y, h, t + h),
                &mut k,
                &mut ytmp,
                &mut y1,
                (1.0, 1.0),
            )
            .unwrap();
            if embedded {
                let e: f64 = E.iter().zip(&k).map(|(ej, kj)| ej * kj[0]).sum();
                y[0] = y1[0] - h * e;
            } else {
                y[0] = y1[0];
            }
        }
        (y[0] - tend.sin().exp()).abs()
    }

    /// Observed orders log2(e(n) / e(2n)) over the step counts `ns`.
    fn orders(ns: &[usize], embedded: bool) -> Vec<f64> {
        let e: Vec<f64> = ns.iter().map(|&n| fixed(n, 2.0, embedded)).collect();
        e.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
    }

    #[test]
    fn the_method_has_order_five() {
        // The orders approach 5 from above on this problem: 5.30, 5.19, ...
        let o = orders(&[8, 16, 32, 64], false);
        assert!(
            o.iter().all(|v| (v - 5.0).abs() < 0.4),
            "observed orders {o:?}"
        );
    }

    #[test]
    fn the_embedded_estimate_has_order_four() {
        // The fourth order solution has a large error constant, so the observed order
        // climbs slowly: 3.1, 3.6, 3.8, 3.9, 3.95. A wrong e_i would settle below 4.
        let o = orders(&[8, 16, 32, 64, 128, 256], true);
        assert!(
            o[0] > 3.0 && o.windows(2).all(|w| w[1] > w[0]),
            "observed orders {o:?}"
        );
        let last = *o.last().unwrap();
        assert!((last - 4.0).abs() < 0.1, "observed orders {o:?}");
    }

    #[test]
    fn the_tableau_is_consistent() {
        for (s, row) in A.iter().enumerate().take(5) {
            let sum: f64 = row.iter().sum();
            assert!((sum - C[s]).abs() < 1e-15, "row {s} sums to {sum}");
        }
        let b: f64 = A[5].iter().sum();
        assert!((b - 1.0).abs() < 1e-15);
        let e: f64 = E.iter().sum();
        assert!(e.abs() < 1e-15, "b and bhat both sum to one, e sums to {e}");
    }
}
