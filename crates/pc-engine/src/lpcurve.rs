//! The LP (fold) curve: equilibria with a zero eigenvalue, followed in two parameters.
//!
//! Unknowns `x = (u, a1, a2)` (n + 2 values), equations `F = 0` and `g = 0`, with `g` the
//! bordered fold function of [`crate::LpSystem`]: `[[A, w_b], [v_bᵀ, 0]] (q; g) = (0; 1)`
//! and `g_z = -wᵀ (A_z q)` with `[[Aᵀ, v_b], [w_bᵀ, 0]] (w; h) = (0; 1)`
//! (Kuznetsov, NBA Lecture 3, 2011; Govaerts, Kuznetsov and Dhooge, SIAM J. Sci. Comput.
//! 27, 2005). The zero set of `g` does not depend on the borders, so they are replaced at
//! every evaluation of the test functions by the unit null vectors found there. That keeps
//! the bordered matrix well conditioned along the curve, and the sign of `q` and `w`
//! continuous, since `v_bᵀ q = 1` and `w_bᵀ w = 1` hold for the old borders.
//!
//! Codim-2 tests along the curve:
//! * `CP`, the cusp: `⟨p, B(q, q)⟩` with unit `q` and `p`. It changes sign with the
//!   quadratic fold coefficient and stays smooth through a Bogdanov-Takens point, where
//!   `⟨p, q⟩` vanishes.
//! * `BT`, Bogdanov-Takens: `⟨p, q⟩` for unit `q` and `p`. It is zero where the zero
//!   eigenvalue is double and changes sign through it.
//! * `ZH`, zero-Hopf: `psi_h` of the eigenvalues without the zero one. It vanishes where a
//!   complex pair crosses the imaginary axis. A zero from two real eigenvalues of opposite
//!   sign (`classify_h_zero`) is not reported.

use crate::branch::{Branch, Options, Point};
use crate::codim2::{
    BT_NORMALIZATION, CUSP_NORMALIZATION, bt_coefficients, cusp_coefficient, without_zero,
};
use crate::curve2::{CurveSetup, Model2, dot, norm, start_tangent};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::{store_zero_hopf, zero_hopf_at};
use crate::palc::run_branch;
use crate::problem::Problem;
use crate::tests_fn::{HopfZero, classify_h_zero, critical_pair, psi_h};
use pc_linalg::{Lu, null_left, null_right};
use std::sync::atomic::{AtomicBool, Ordering};

/// Codim-2 labels of the LP curve, in the order of `Point.codim2`.
const LABELS: [&str; 3] = ["CP", "BT", "ZH"];

/// `F = 0, g = 0` over a model with two free parameters.
pub struct LpCurveProblem<'a> {
    m: Model2<'a>,
    v_b: Vec<f64>,
    w_b: Vec<f64>,
}

impl<'a> LpCurveProblem<'a> {
    /// Checks the setup and takes the borders from the null vectors of A at the start.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &CurveSetup,
    ) -> Result<LpCurveProblem<'a>, EngineError> {
        let mut m = Model2::new(model, setup)?;
        let x = setup.start();
        m.set(&x);
        let n = m.n;
        let a = m.jac_u(&x[..n])?;
        let v_b = null_right(&a, n, n, 1)?.swap_remove(0);
        let w_b = null_left(&a, n, n, 1)?.swap_remove(0);
        Ok(LpCurveProblem { m, v_b, w_b })
    }

    /// The bordered matrix `[[A, w_b], [v_bᵀ, 0]]`, (n + 1) square.
    fn bordered(&self, a: &[f64]) -> Vec<f64> {
        let n = self.m.n;
        let s = n + 1;
        let mut b = vec![0.0; s * s];
        for j in 0..n {
            b[s * j..s * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
            b[n + s * j] = self.v_b[j];
        }
        b[s * n..s * n + n].copy_from_slice(&self.w_b);
        b
    }

    /// Adds the normal form coefficients to the codim-2 points of a finished run. A point
    /// whose coefficient cannot be computed keeps `coefficient = None`.
    fn annotate(&mut self, branch: &mut Branch) {
        let n = self.m.n;
        for sp in branch.specials.iter_mut() {
            if !matches!(sp.label.as_str(), "CP" | "BT" | "ZH") {
                continue;
            }
            let mut x = sp.point.u.clone();
            x.push(sp.point.alpha);
            self.m.set(&x);
            match sp.label.as_str() {
                "CP" => {
                    if let Ok(c) = cusp_coefficient(&mut self.m, &x[..n]) {
                        sp.extras.coefficient = Some(c);
                        sp.extras.normalization = Some(CUSP_NORMALIZATION.to_string());
                    }
                }
                "ZH" => {
                    let rest = without_zero(&sp.point.eigenvalues);
                    if classify_h_zero(&rest) == HopfZero::Hopf
                        && let Some((i, _)) = critical_pair(&rest)
                    {
                        let omega = rest[i].1.abs();
                        if let Ok(z) = zero_hopf_at(&mut self.m.at(&x[..n]), omega) {
                            store_zero_hopf(&mut sp.extras, &z, omega);
                        }
                    }
                }
                _ => {
                    if let Ok((a, b)) = bt_coefficients(&mut self.m, &x[..n]) {
                        sp.extras.coefficient = Some(a);
                        sp.extras.coefficient2 = Some(b);
                        sp.extras.normalization = Some(BT_NORMALIZATION.to_string());
                    }
                }
            }
        }
    }

    /// `(q, g)` and `(w, h)` of the bordered systems at `a`.
    fn solves(&self, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
        let s = self.m.n + 1;
        let lu = Lu::new(&self.bordered(a), s)?;
        let mut qg = vec![0.0; s];
        qg[s - 1] = 1.0;
        lu.solve(&mut qg)?;
        let mut wh = vec![0.0; s];
        wh[s - 1] = 1.0;
        lu.solve_transpose(&mut wh)?;
        Ok((qg, wh))
    }
}

impl Problem for LpCurveProblem<'_> {
    fn dim(&self) -> usize {
        self.m.n + 2
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        self.m.set(x);
        self.m.f(&x[..n], &mut out[..n])?;
        let a = self.m.jac_u(&x[..n])?;
        let s = n + 1;
        let lu = Lu::new(&self.bordered(&a), s)?;
        let mut qg = vec![0.0; s];
        qg[n] = 1.0;
        lu.solve(&mut qg)?;
        out[n] = qg[n];
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        let rows = n + 1;
        self.m.set(x);
        let u = &x[..n];
        let a = self.m.jac_u(u)?;
        let fa = self.m.f_a(u)?;
        let (qg, wh) = self.solves(&a)?;
        let (q, w) = (&qg[..n], &wh[..n]);
        for j in 0..n {
            out[rows * j..rows * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
        }
        for j in 0..2 {
            out[rows * (n + j)..rows * (n + j) + n].copy_from_slice(&fa[n * j..n * (j + 1)]);
        }
        for k in 0..n + 2 {
            let azq = self.m.a_z_q(u, q, k)?;
            out[n + rows * k] = -dot(w, &azq);
        }
        Ok(())
    }

    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        let n = self.m.n;
        self.m.set(x);
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
        self.m.set(x);
        let u = &x[..n];
        let a = self.m.jac_u(u)?;
        let (qg, wh) = self.solves(&a)?;
        let (mut q, mut p) = (qg[..n].to_vec(), wh[..n].to_vec());
        let (nq, np) = (norm(&q), norm(&p));
        if !(nq.is_finite() && np.is_finite() && nq > 0.0 && np > 0.0) {
            return Err(EngineError::Singular(
                "the null vectors of A at a point of the LP curve".into(),
            ));
        }
        q.iter_mut().for_each(|c| *c /= nq);
        p.iter_mut().for_each(|c| *c /= np);
        let bqq = self.m.b(u, &q, &q)?;
        let cp = dot(&p, &bqq);
        let bt = dot(&p, &q);
        let zh = if eigs.is_empty() {
            f64::NAN
        } else {
            psi_h(&without_zero(eigs))
        };
        self.v_b = q;
        self.w_b = p;
        Ok(vec![cp, bt, zh])
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        if i == 2 {
            // psi_h also vanishes where two real eigenvalues sum to zero, which is no ZH.
            (classify_h_zero(&without_zero(eigs)) == HopfZero::Hopf).then_some("ZH")
        } else {
            LABELS.get(i).copied()
        }
    }
}

/// Follows the LP curve through the LP point of `setup` in the parameters `setup.free`.
/// `Options::bounds` bound `a2`, the second parameter. The first tangent points to
/// increasing `u[0]` (see `start_tangent`); `Options::direction` -1 reverses it. Codim-2
/// points on the way are listed in `Branch::specials` without refinement.
pub fn continue_lp_curve(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut problem = LpCurveProblem::new(model, setup)?;
    let x = setup.start();
    let mut f = vec![0.0; x.len() - 1];
    problem.residual(&x, &mut f)?;
    let res = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if res.is_nan() || res > opts.tol_f {
        return Err(EngineError::BadProblem(format!(
            "the start is not on the LP curve: the residual is {res:e}; start from a refined \
             LP point"
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
