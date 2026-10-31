//! The Hopf curve: equilibria with a pair of eigenvalues `+-i omega`, followed in two
//! parameters.
//!
//! Unknowns `x = (u, a1, kappa, a2)` (n + 3 values) with `kappa = omega^2`, equations `F = 0`
//! and a two-component bordered function `g` of `M = A^2 + kappa I`. At a Hopf point `M` has the
//! eigenvalue 0 on the real eigenplane `E` of the pair, and on `E` it acts as `a I + b J`
//! with `J` the rotation by 90 degrees (`A|E = mu I + omega J`, so `a = mu^2 - omega^2 + kappa`
//! and `b = 2 mu omega`). The bordered system
//!
//! ```text
//! [[M, W], [Vᵀ, 0]] [[X], [G]] = [[0], [I]]      (V, W: n by 2, G: 2 by 2)
//! ```
//!
//! then has `G e1 = 0` exactly when `M|E = 0`, that is `mu = 0` and `kappa = omega^2`. The two
//! equations are `g11 = G[0,0]` and `g21 = G[1,0]`. This is the minimally extended system for
//! a nullity-2 matrix of Govaerts (Numerical Methods for Bifurcations of Dynamical
//! Equilibria, SIAM 2000, Sec. 5.6; Kuznetsov, NBA Lecture 3, 2011) with the second column of
//! `G` left out. The gradient follows from `G_z = -Yᵀ M_z X e1` with
//! `[[Mᵀ, V], [Wᵀ, 0]] [[Y], [H]] = [[0], [I]]`, `M_z = A_z A + A A_z` for the state and the
//! parameters and `M_kappa = I`. The zero set of `g` does not depend on the borders, so
//! they are replaced at each evaluation of the codim-2 tests by an orthonormal basis of the
//! columns of `X` and `Y`; `VᵀX = I` keeps their orientation continuous.
//!
//! On a curve `Point.u` is `(state, a1, kappa)` and `Point.alpha` is `a2`; `Options::bounds`
//! apply to `a2`.
//!
//! Codim-2 tests along the curve (`Point.codim2`, in this order):
//! * `GH`, Bautin: the first Lyapunov coefficient `l1` at `omega = sqrt(kappa)`; NaN for
//!   `kappa <= 0`.
//! * `BT`, Bogdanov-Takens: `kappa`. It changes sign where `omega` goes through 0, which is
//!   where the Hopf curve meets the neutral saddle curve at a BT point.
//! * `ZH`, zero-Hopf: the product of the real parts of the eigenvalues outside the critical
//!   pair. A real eigenvalue crossing zero changes its sign; it is labeled only if the
//!   eigenvalue closest to the imaginary axis outside the pair is real.
//! * `HH`, Hopf-Hopf: `psi_h` of the eigenvalues outside the critical pair, labeled only if
//!   that zero is a conjugate pair (`classify_h_zero`).

use crate::branch::{Branch, Options, Point};
use crate::codim2::{BT_NORMALIZATION, bt_coefficients};
use crate::curve2::{CurveSetup, Model2, dot, norm, start_tangent};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::{
    GH_NORMALIZATION, first_lyapunov_at, second_lyapunov_at, store_zero_hopf, zero_hopf_at,
};
use crate::palc::run_branch;
use crate::problem::Problem;
use crate::tests_fn::{HopfZero, classify_h_zero, critical_pair, psi_h};
use pc_linalg::{Lu, eigenvalues, null_left, null_right};
use std::sync::atomic::{AtomicBool, Ordering};

/// Codim-2 labels of the Hopf curve, in the order of `Point.codim2`.
const LABELS: [&str; 4] = ["GH", "BT", "ZH", "HH"];

/// Relative size of the imaginary part below which an eigenvalue counts as real.
const REAL_TOL: f64 = 1e-8;

/// `a b` for n by n column-major matrices.
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for j in 0..n {
        for k in 0..n {
            let bkj = b[k + n * j];
            for i in 0..n {
                c[i + n * j] += a[i + n * k] * bkj;
            }
        }
    }
    c
}

/// `a v` for an n by n column-major `a`.
fn matvec(a: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (j, vj) in v.iter().enumerate() {
        for i in 0..n {
            out[i] += a[i + n * j] * vj;
        }
    }
    out
}

/// `aᵀ v` for an n by n column-major `a`.
fn matvec_t(a: &[f64], n: usize, v: &[f64]) -> Vec<f64> {
    (0..n).map(|j| dot(&a[n * j..n * (j + 1)], v)).collect()
}

/// `A^2 + kappa I` for the n by n column-major `a`.
fn m_matrix(a: &[f64], n: usize, kappa: f64) -> Vec<f64> {
    let mut m = matmul(a, a, n);
    for i in 0..n {
        m[i + n * i] += kappa;
    }
    m
}

/// Orthonormalizes the two columns of the n by 2 column-major `c` (Gram-Schmidt), keeping the
/// first column's direction.
fn orthonormalize(c: &[f64], n: usize) -> Result<Vec<f64>, EngineError> {
    let (c0, c1) = (&c[..n], &c[n..]);
    let n0 = norm(c0);
    if !(n0.is_finite() && n0 > 0.0) {
        return Err(EngineError::Singular(
            "the border vectors of the Hopf curve".into(),
        ));
    }
    let v0: Vec<f64> = c0.iter().map(|x| x / n0).collect();
    let d = dot(&v0, c1);
    let r: Vec<f64> = c1.iter().zip(&v0).map(|(x, y)| x - d * y).collect();
    let n1 = norm(&r);
    if !(n1.is_finite() && n1 > 0.0) {
        return Err(EngineError::Singular(
            "the border vectors of the Hopf curve are parallel".into(),
        ));
    }
    let mut out = v0;
    out.extend(r.iter().map(|x| x / n1));
    Ok(out)
}

/// The eigenvalues outside the critical pair.
fn outside_pair(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let Some((i, j)) = critical_pair(eigs) else {
        return Vec::new();
    };
    eigs.iter()
        .enumerate()
        .filter(|&(k, _)| k != i && k != j)
        .map(|(_, &e)| e)
        .collect()
}

/// The eigenvalue closest to the imaginary axis is real.
fn nearest_axis_is_real(rest: &[(f64, f64)]) -> bool {
    rest.iter()
        .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()))
        .is_some_and(|e| e.1.abs() <= REAL_TOL * 1.0_f64.max(e.0.hypot(e.1)))
}

/// `ZH` test: the product of the real parts of `rest`; NaN if `rest` is empty.
fn zh_test(rest: &[(f64, f64)]) -> f64 {
    if rest.is_empty() {
        f64::NAN
    } else {
        rest.iter().map(|e| e.0).product()
    }
}

/// The label of codim-2 test `i` at a zero, or `None` for a zero that is no bifurcation.
fn label_for(i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
    match i {
        2 => nearest_axis_is_real(&outside_pair(eigs)).then_some("ZH"),
        3 => (classify_h_zero(&outside_pair(eigs)) == HopfZero::Hopf).then_some("HH"),
        _ => LABELS.get(i).copied(),
    }
}

/// `F = 0, g = 0` over a model with two free parameters.
pub struct HopfCurveProblem<'a> {
    m: Model2<'a>,
    /// n by 2, column-major.
    v_b: Vec<f64>,
    /// n by 2, column-major.
    w_b: Vec<f64>,
    /// The starting unknown `(u, a1, kappa, a2)`.
    x0: Vec<f64>,
}

impl<'a> HopfCurveProblem<'a> {
    /// Checks the setup, takes `kappa` from the critical eigenvalue pair at the start and the
    /// borders from the null spaces of `A^2 + kappa I` there.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &CurveSetup,
    ) -> Result<HopfCurveProblem<'a>, EngineError> {
        let mut m = Model2::new(model, setup)?;
        let n = m.n;
        if n < 2 {
            return Err(EngineError::BadProblem(
                "a Hopf curve needs at least two state variables".into(),
            ));
        }
        let s = setup.start();
        m.set(&s);
        let a = m.jac_u(&s[..n])?;
        let eigs = eigenvalues(&a, n)?;
        let (i, j) = critical_pair(&eigs)
            .filter(|_| classify_h_zero(&eigs) == HopfZero::Hopf)
            .ok_or_else(|| {
                EngineError::BadProblem(
                    "the start is not a Hopf point: A has no complex pair near the imaginary \
                     axis"
                        .into(),
                )
            })?;
        let mu = 0.5 * (eigs[i].0 + eigs[j].0);
        let omega = 0.5 * (eigs[i].1 - eigs[j].1).abs();
        let kappa = omega * omega - mu * mu;
        let mm = m_matrix(&a, n, kappa);
        let v_b = null_right(&mm, n, n, 2)?.concat();
        let w_b = null_left(&mm, n, n, 2)?.concat();
        let mut x0 = s[..n + 1].to_vec();
        x0.push(kappa);
        x0.push(s[n + 1]);
        Ok(HopfCurveProblem { m, v_b, w_b, x0 })
    }

    /// The starting unknown `(u, a1, kappa, a2)`.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    /// The bordered matrix `[[M, W], [Vᵀ, 0]]`, (n + 2) square.
    fn bordered(&self, mm: &[f64]) -> Vec<f64> {
        let n = self.m.n;
        let s = n + 2;
        let mut b = vec![0.0; s * s];
        for j in 0..n {
            b[s * j..s * j + n].copy_from_slice(&mm[n * j..n * (j + 1)]);
            for k in 0..2 {
                b[n + k + s * j] = self.v_b[j + n * k];
            }
        }
        for k in 0..2 {
            b[s * (n + k)..s * (n + k) + n].copy_from_slice(&self.w_b[n * k..n * (k + 1)]);
        }
        b
    }

    /// The columns of `X` (n by 2) and of `Y` (n by 2) of the bordered solves at `mm`.
    fn solves(&self, mm: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
        let n = self.m.n;
        let s = n + 2;
        let lu = Lu::new(&self.bordered(mm), s)?;
        let (mut xs, mut ys) = (Vec::with_capacity(2 * n), Vec::with_capacity(2 * n));
        for k in 0..2 {
            let mut x = vec![0.0; s];
            x[n + k] = 1.0;
            lu.solve(&mut x)?;
            xs.extend_from_slice(&x[..n]);
            let mut y = vec![0.0; s];
            y[n + k] = 1.0;
            lu.solve_transpose(&mut y)?;
            ys.extend_from_slice(&y[..n]);
        }
        Ok((xs, ys))
    }

    /// Adds the normal form coefficients to the codim-2 points of a finished run. A point
    /// whose coefficient cannot be computed keeps `coefficient = None`.
    fn annotate(&mut self, branch: &mut Branch) {
        let n = self.m.n;
        for sp in branch.specials.iter_mut() {
            if !matches!(sp.label.as_str(), "GH" | "BT" | "ZH") {
                continue;
            }
            let mut x = sp.point.u.clone();
            x.push(sp.point.alpha);
            self.m.set_pair(x[n], x[n + 2]);
            let u = &x[..n];
            let kappa = x[n + 1];
            match sp.label.as_str() {
                "GH" => {
                    if kappa > 0.0 {
                        let omega = kappa.sqrt();
                        if let Ok(l2) = second_lyapunov_at(&mut self.m.at(u), omega) {
                            sp.extras.coefficient = Some(l2);
                            sp.extras.omega = Some(omega);
                            sp.extras.normalization = Some(GH_NORMALIZATION.to_string());
                        }
                    }
                }
                "ZH" => {
                    if kappa > 0.0 {
                        let omega = kappa.sqrt();
                        if let Ok(z) = zero_hopf_at(&mut self.m.at(u), omega) {
                            store_zero_hopf(&mut sp.extras, &z, omega);
                        }
                    }
                }
                _ => {
                    if let Ok((a, b)) = bt_coefficients(&mut self.m, u) {
                        sp.extras.coefficient = Some(a);
                        sp.extras.coefficient2 = Some(b);
                        sp.extras.normalization = Some(BT_NORMALIZATION.to_string());
                    }
                }
            }
        }
    }
}

impl Problem for HopfCurveProblem<'_> {
    fn dim(&self) -> usize {
        self.m.n + 3
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        self.m.set_pair(x[n], x[n + 2]);
        let u = &x[..n];
        self.m.f(u, &mut out[..n])?;
        let a = self.m.jac_u(u)?;
        let mm = m_matrix(&a, n, x[n + 1]);
        let s = n + 2;
        let lu = Lu::new(&self.bordered(&mm), s)?;
        let mut xg = vec![0.0; s];
        xg[n] = 1.0;
        lu.solve(&mut xg)?;
        out[n] = xg[n];
        out[n + 1] = xg[n + 1];
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        let rows = n + 2;
        self.m.set_pair(x[n], x[n + 2]);
        let u = &x[..n];
        let a = self.m.jac_u(u)?;
        let fa = self.m.f_a(u)?;
        let mm = m_matrix(&a, n, x[n + 1]);
        let (xs, ys) = self.solves(&mm)?;
        let x0 = &xs[..n];
        let y = [&ys[..n], &ys[n..]];
        let ax0 = matvec(&a, n, x0);
        let aty = [matvec_t(&a, n, y[0]), matvec_t(&a, n, y[1])];
        out.fill(0.0);
        for j in 0..n {
            out[rows * j..rows * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
        }
        out[rows * n..rows * n + n].copy_from_slice(&fa[..n]);
        out[rows * (n + 2)..rows * (n + 2) + n].copy_from_slice(&fa[n..]);
        for col in 0..n + 3 {
            if col == n + 1 {
                for i in 0..2 {
                    out[n + i + rows * col] = -dot(y[i], x0);
                }
            } else {
                // Coordinate k of (u, a1, a2) for the columns of u, a1 and a2.
                let k = if col <= n { col } else { n + 1 };
                let t1 = self.m.a_z_q(u, &ax0, k)?;
                let t2 = self.m.a_z_q(u, x0, k)?;
                for i in 0..2 {
                    out[n + i + rows * col] = -(dot(y[i], &t1) + dot(&aty[i], &t2));
                }
            }
        }
        Ok(())
    }

    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        let n = self.m.n;
        self.m.set_pair(x[n], x[n + 2]);
        Ok(Some(self.m.jac_u(&x[..n])?))
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        &LABELS
    }

    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        let n = self.m.n;
        self.m.set_pair(x[n], x[n + 2]);
        let u = &x[..n];
        let kappa = x[n + 1];
        let gh = if kappa > 0.0 {
            first_lyapunov_at(&mut self.m.at(u), kappa.sqrt()).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let (zh, hh) = zh_hh_tests(kappa, eigs);
        // New borders from the solution of the bordered systems at this point.
        let a = self.m.jac_u(u)?;
        let (xs, ys) = self.solves(&m_matrix(&a, n, kappa))?;
        self.v_b = orthonormalize(&xs, n)?;
        self.w_b = orthonormalize(&ys, n)?;
        Ok(vec![gh, kappa, zh, hh])
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        label_for(i, eigs)
    }
}

/// Follows the Hopf curve through the Hopf point of `setup` in the parameters `setup.free`.
/// `Options::bounds` bound `a2`, the second parameter. The first tangent points to
/// increasing `u[0]` (see `start_tangent`); `Options::direction` -1 reverses it. Codim-2
/// points on the way are listed in `Branch::specials` without refinement.
pub fn continue_hopf_curve(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut problem = HopfCurveProblem::new(model, setup)?;
    let x = problem.start().to_vec();
    let mut f = vec![0.0; x.len() - 1];
    problem.residual(&x, &mut f)?;
    let res = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if res.is_nan() || res > opts.tol_f {
        return Err(EngineError::BadProblem(format!(
            "the start is not on the Hopf curve: the residual is {res:e}; start from a refined \
             Hopf point"
        )));
    }
    let tangent = start_tangent(&mut problem, &x)?;
    let mut branch = run_branch(
        &mut problem,
        &x,
        Some(&tangent),
        opts,
        false,
        cancel,
        on_point,
    )?;
    problem.annotate(&mut branch);
    Ok(branch)
}

/// The ZH and HH test values from the spectrum `eigs` at `kappa`. Past a BT point
/// (`kappa <= 0`) the critical pair is no Hopf pair, so both are NaN there.
fn zh_hh_tests(kappa: f64, eigs: &[(f64, f64)]) -> (f64, f64) {
    if kappa <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let rest = outside_pair(eigs);
    let hh = if rest.len() >= 2 {
        psi_h(&rest)
    } else {
        f64::NAN
    };
    (zh_test(&rest), hh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_pair_finds_the_imaginary_pair() {
        let e = [(-3.0, 0.0), (1e-10, 2.0), (1e-10, -2.0)];
        assert_eq!(critical_pair(&e), Some((1, 2)));
        assert_eq!(critical_pair(&e[..1]), None);
    }

    #[test]
    fn zh_label_needs_a_real_eigenvalue_on_the_axis() {
        // A pair +-2i beside a zero eigenvalue and a stable one is a ZH point.
        let zh = [(0.0, 2.0), (0.0, -2.0), (0.0, 0.0), (-1.0, 0.0)];
        assert_eq!(label_for(2, &zh), Some("ZH"));
        assert_eq!(zh_test(&outside_pair(&zh)), 0.0);
        // A second pair on the axis makes the product vanish too, but it is an HH point.
        let hh = [(0.0, 2.0), (0.0, -2.0), (0.0, 3.0), (0.0, -3.0)];
        assert_eq!(label_for(2, &hh), None);
        assert_eq!(label_for(3, &hh), Some("HH"));
        // A stable real eigenvalue next to the pair gives a nonzero product, so no zero is found.
        let none = [(0.0, 2.0), (0.0, -2.0), (-1.0, 0.0)];
        assert!(zh_test(&outside_pair(&none)) < 0.0);
    }

    #[test]
    fn zh_and_hh_tests_are_off_for_nonpositive_kappa() {
        let e = [(0.0, 2.0), (0.0, -2.0), (0.0, 0.0), (-1.0, 0.0)];
        let (zh, _) = zh_hh_tests(4.0, &e);
        assert_eq!(zh, 0.0);
        for kappa in [0.0, -1.0] {
            let (zh, hh) = zh_hh_tests(kappa, &e);
            assert!(zh.is_nan() && hh.is_nan(), "kappa {kappa}");
        }
    }

    #[test]
    fn hh_label_rejects_a_neutral_saddle_among_the_rest() {
        // Real eigenvalues +-2 outside the pair sum to zero, which is no HH point.
        let ns = [(0.0, 1.0), (0.0, -1.0), (2.0, 0.0), (-2.0, 0.0)];
        assert_eq!(label_for(3, &ns), None);
        // With fewer than two eigenvalues outside the pair there is no HH test.
        let few = [(0.0, 1.0), (0.0, -1.0), (-2.0, 0.0)];
        assert_eq!(label_for(3, &few), None);
        assert!(zh_test(&outside_pair(&few[..2])).is_nan());
    }

    #[test]
    fn orthonormalize_keeps_the_first_direction() {
        let v = orthonormalize(&[3.0, 0.0, 1.0, 2.0], 2).unwrap();
        assert_eq!(v, vec![1.0, 0.0, 0.0, 1.0]);
        assert!(orthonormalize(&[1.0, 0.0, 2.0, 0.0], 2).is_err());
    }
}
