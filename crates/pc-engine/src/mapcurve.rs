//! LP, PD and NS curves of fixed points of a map, followed in two parameters (SPEC Section
//! 8w).
//!
//! With `A = F_x(x, a1, a2)` the Jacobian of the map:
//! * LP and PD: unknowns `x = (u, a1, a2)` (n + 2 values), equations `F(u) - u = 0` and the
//!   bordered function `s` of `A + sigma I`, with `sigma = -1` for LP and `+1` for PD:
//!
//!   ```text
//!   [[A + sigma I, w_b], [v_bᵀ, 0]] [[v], [s]] = [[0], [1]]
//!   ```
//!
//!   `s` vanishes exactly where `A` has the multiplier 1 (LP) or -1 (PD). Its derivative is
//!   `s_z = -wᵀ A_z v` with `w` from the transposed system.
//! * NS: unknowns `x = (u, a1, kappa, a2)` (n + 3 values), equations `F(u) - u = 0` and the
//!   first column of the 2 by 2 block `S` of
//!
//!   ```text
//!   [[M, W], [Vᵀ, 0]] [[X], [S]] = [[0], [I]],    M = A^2 - 2 kappa A + I
//!   ```
//!
//!   which vanishes exactly where `A` has the multipliers `e^(+-i theta)` with
//!   `kappa = cos(theta)`. The gradient is `S_z[i, 1] = -Y_iᵀ M_z X_1` with `Y` from the
//!   transposed system, `M_z = A_z A + A A_z - 2 kappa A_z` and `M_kappa = -2 A`.
//!
//! These are equations 2.4 to 2.9 of Govaerts, Khoshsiar Ghaziani, Kuznetsov and Meijer,
//! "Numerical methods for two-parameter local bifurcation analysis of maps", SIAM J. Sci.
//! Comput. 29(6):2644-2667 (2007), with the border update of their p. 2650. The gradients
//! follow from differentiating the bordered systems. The borders are replaced at each
//! evaluation of the codim-2 tests, which leaves the zero sets unchanged.
//!
//! The curves run on the map itself, not on `MapAsRoot`, so `state_jacobian` is `A` and the
//! eigenvalues stored with each point are the multipliers. On a curve `Point.u` is the
//! unknown without its last entry and `Point.alpha` is `a2`; `Options::bounds` bound `a2`.
//!
//! Codim-2 tests (SPEC Section 8x, from Table 2.2 of the same paper). The rest set is the
//! multipliers without the critical ones: 1 on the LP curve, -1 on the PD curve and the two
//! roots of mu^2 - 2 kappa mu + 1 on the NS curve, each removed as the nearest multiplier.
//! * LP: LPPD = Re prod(mu + 1) and LPNS = Re prod(mu_i mu_j - 1) over the rest set,
//!   CP = <w, B(v, v)> and R1 = <w, v> with the unit null vectors v and w.
//! * PD: LPPD = Re prod(mu - 1), PDNS = Re prod(mu_i mu_j - 1), GPD = <w, C(v, v, v) +
//!   3 B(v, h)> / 6 with (I - A) h = B(v, v), and R2 = <w, v>.
//! * NS: R1 to R4 = kappa - 1, kappa + 1, kappa + 1/2 and kappa, then LPNS = Re prod(mu - 1),
//!   PDNS = Re prod(mu + 1) and NSNS = Re prod(mu_i mu_j - 1) over the rest set, and
//!   CH = d / |c1| from the Neimark-Sacker coefficient of `normal::map_ns_core`. The last
//!   four are NaN for |kappa| >= 1.
//!
//! A zero of LPNS, PDNS or NSNS from a real pair of product 1 is a neutral saddle and gets
//! no label.

use crate::branch::{Branch, Options, Point, StopReason};
use crate::curve2::{CurveSetup, Model2, dot, norm, start_tangent};
use crate::cycle::{
    kappa_roots, nearest_pair, nearest_pair_is_complex, pair_product, remove_nearest, rest_product,
};
use crate::error::EngineError;
use crate::hopfcurve::{matmul, matvec, matvec_t, orthonormalize};
use crate::model::ModelEval;
use crate::normal::map_ns_core;
use crate::palc::{Corrected, correct, run_branch};
use crate::problem::Problem;
use pc_linalg::{Lu, eigenvalues, null_left, null_right};
use std::sync::atomic::{AtomicBool, Ordering};

/// The largest residual of a start that the start correction accepts. A PhaseCont value
/// that waits for the owner's sign-off (SPEC Section 8w, `docs/STATUS.md`).
pub const START_RESIDUAL_MAX: f64 = 1e-6;

/// Codim-2 labels of the LP curve of a map, in the order of `Point.codim2`.
pub const MAP_LP_CURVE_LABELS: [&str; 4] = ["LPPD", "LPNS", "CP", "R1"];
/// Codim-2 labels of the PD curve of a map, in the order of `Point.codim2`.
pub const MAP_PD_CURVE_LABELS: [&str; 4] = ["LPPD", "PDNS", "GPD", "R2"];
/// Codim-2 labels of the NS curve of a map, in the order of `Point.codim2`.
pub const MAP_NS_CURVE_LABELS: [&str; 8] = ["R1", "R2", "R3", "R4", "LPNS", "PDNS", "NSNS", "CH"];

/// The index of CH in [`MAP_NS_CURVE_LABELS`].
const CH: usize = 7;
/// The index of NSNS in [`MAP_NS_CURVE_LABELS`].
const NSNS: usize = 6;

/// A sign change of CH gets a label only when its larger end is at least this. On an
/// area-preserving map d is 0 and CH is rounding noise along the whole curve. A PhaseCont
/// value that waits for the owner's sign-off (SPEC Section 8x, ruling (e)).
pub const CH_BRACKET_FLOOR: f64 = 1e-8;

/// `a + sigma I` for the n by n column-major `a`.
fn shifted(a: &[f64], n: usize, sigma: f64) -> Vec<f64> {
    let mut b = a.to_vec();
    for i in 0..n {
        b[i + n * i] += sigma;
    }
    b
}

/// `A^2 - 2 kappa A + I` for the n by n column-major `a`.
fn ns_matrix(a: &[f64], n: usize, kappa: f64) -> Vec<f64> {
    let mut m = matmul(a, a, n);
    for (mi, ai) in m.iter_mut().zip(a) {
        *mi -= 2.0 * kappa * ai;
    }
    for i in 0..n {
        m[i + n * i] += 1.0;
    }
    m
}

/// `v / |v|`, refusing a zero or non-finite norm.
fn unit(v: &[f64], curve: &str) -> Result<Vec<f64>, EngineError> {
    let nv = norm(v);
    if !(nv.is_finite() && nv > 0.0) {
        return Err(EngineError::Singular(format!(
            "the border vectors of the {curve} curve"
        )));
    }
    Ok(v.iter().map(|x| x / nv).collect())
}

/// Whether a zero of NSNS gets a label (SPEC Section 8x, ruling (c)): of the complex
/// conjugate pairs of `mu`, the one whose product is nearest to 1 is taken as the NS pair
/// and removed, and the pair of the rest whose product is nearest to 1 must be complex.
/// With no complex pair, there is no label.
fn nsns_label_ok(mu: &[(f64, f64)]) -> bool {
    let mut best: Option<(f64, usize, usize)> = None;
    for i in 0..mu.len() {
        for j in i + 1..mu.len() {
            let (p, q) = (mu[i], mu[j]);
            let tol = 1e-10 * p.0.hypot(p.1).max(1.0);
            if p.1 != 0.0 && (p.0 - q.0).abs() <= tol && (p.1 + q.1).abs() <= tol {
                // The product of a conjugate pair is |p|^2.
                let d = (p.0 * p.0 + p.1 * p.1 - 1.0).abs();
                if best.is_none_or(|(e, _, _)| d < e) {
                    best = Some((d, i, j));
                }
            }
        }
    }
    let Some((_, i, j)) = best else {
        return false;
    };
    let rest: Vec<(f64, f64)> = mu
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != i && k != j)
        .map(|(_, m)| *m)
        .collect();
    nearest_pair_is_complex(&rest)
}

/// Writes `F(u) - u` to `out`.
fn fixed_point_residual(m: &mut Model2, u: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
    m.f(u, out)?;
    for (o, ui) in out.iter_mut().zip(u) {
        *o -= ui;
    }
    Ok(())
}

/// Writes the rows of `F - x` to the curve Jacobian `out` (`rows` rows, column-major):
/// `A - I` in the state columns, `F_a1` in column `n` and `F_a2` in column `col_a2`.
fn fixed_point_rows(out: &mut [f64], rows: usize, a: &[f64], fa: &[f64], n: usize, col_a2: usize) {
    for j in 0..n {
        out[rows * j..rows * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
        out[j + rows * j] -= 1.0;
    }
    out[rows * n..rows * n + n].copy_from_slice(&fa[..n]);
    out[rows * col_a2..rows * col_a2 + n].copy_from_slice(&fa[n..]);
}

/// The LP curve (`sigma = -1`) or the PD curve (`sigma = +1`) of fixed points of a map.
pub struct MapLpPdProblem<'a> {
    m: Model2<'a>,
    sigma: f64,
    name: &'static str,
    v_b: Vec<f64>,
    w_b: Vec<f64>,
    /// The starting unknown `(u, a1, a2)`.
    x0: Vec<f64>,
}

impl<'a> MapLpPdProblem<'a> {
    /// Checks the setup and takes the borders from the singular vectors of `A + sigma I`
    /// for its smallest singular value at the start. `sigma` is -1 for the LP curve and +1
    /// for the PD curve.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &CurveSetup,
        sigma: f64,
    ) -> Result<MapLpPdProblem<'a>, EngineError> {
        let mut m = Model2::new(model, setup)?;
        let n = m.n;
        let name = if sigma < 0.0 { "LP" } else { "PD" };
        let x0 = setup.start();
        m.set(&x0);
        let a = m.jac_u(&x0[..n])?;
        let b = shifted(&a, n, sigma);
        let v_b = null_right(&b, n, n, 1)?.swap_remove(0);
        let w_b = null_left(&b, n, n, 1)?.swap_remove(0);
        Ok(MapLpPdProblem {
            m,
            sigma,
            name,
            v_b,
            w_b,
            x0,
        })
    }

    /// The starting unknown `(u, a1, a2)`.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    /// The LU factors of `[[A + sigma I, w_b], [v_bᵀ, 0]]`, (n + 1) square.
    fn bordered(&self, a: &[f64]) -> Result<Lu, EngineError> {
        let n = self.m.n;
        let s = n + 1;
        let mut b = vec![0.0; s * s];
        for j in 0..n {
            b[s * j..s * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
            b[j + s * j] += self.sigma;
            b[n + s * j] = self.v_b[j];
        }
        b[s * n..s * n + n].copy_from_slice(&self.w_b);
        Ok(Lu::new(&b, s)?)
    }

    /// `v` and `w` of the bordered solve and its transpose at `a`.
    fn solves(&self, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
        let n = self.m.n;
        let lu = self.bordered(a)?;
        let mut v = vec![0.0; n + 1];
        v[n] = 1.0;
        lu.solve(&mut v)?;
        let mut w = vec![0.0; n + 1];
        w[n] = 1.0;
        lu.solve_transpose(&mut w)?;
        v.truncate(n);
        w.truncate(n);
        Ok((v, w))
    }

    /// The multipliers without the one nearest to `-sigma` (1 on the LP curve, -1 on the
    /// PD curve).
    fn rest(&self, eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut mu = eigs.to_vec();
        remove_nearest(&mut mu, (-self.sigma, 0.0));
        mu
    }

    /// GPD = <w, C(v, v, v) + 3 B(v, h)> / 6 with (I - A) h = B(v, v) and `bvv = B(v, v)`,
    /// NaN where the solve fails or the value is not finite.
    fn gpd(&mut self, u: &[f64], a: &[f64], vh: &[f64], wh: &[f64], bvv: &[f64]) -> f64 {
        let n = self.m.n;
        let mut go = || -> Result<f64, EngineError> {
            // (A - I) h' = B(v, v), so h = -h' solves (I - A) h = B(v, v).
            let lu = Lu::new(&shifted(a, n, -1.0), n)?;
            let mut h = bvv.to_vec();
            lu.solve(&mut h)?;
            h.iter_mut().for_each(|c| *c = -*c);
            let c = self.m.c(u, vh, vh, vh)?;
            let bvh = self.m.b(u, vh, &h)?;
            Ok((dot(wh, &c) + 3.0 * dot(wh, &bvh)) / 6.0)
        };
        match go() {
            Ok(g) if g.is_finite() => g,
            _ => f64::NAN,
        }
    }

    fn labels(&self) -> &'static [&'static str] {
        if self.sigma < 0.0 {
            &MAP_LP_CURVE_LABELS
        } else {
            &MAP_PD_CURVE_LABELS
        }
    }
}

impl Problem for MapLpPdProblem<'_> {
    fn dim(&self) -> usize {
        self.m.n + 2
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        self.m.set(x);
        let u = &x[..n];
        fixed_point_residual(&mut self.m, u, &mut out[..n])?;
        let a = self.m.jac_u(u)?;
        let lu = self.bordered(&a)?;
        let mut v = vec![0.0; n + 1];
        v[n] = 1.0;
        lu.solve(&mut v)?;
        out[n] = v[n];
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        let rows = n + 1;
        self.m.set(x);
        let u = &x[..n];
        let a = self.m.jac_u(u)?;
        let fa = self.m.f_a(u)?;
        let (v, w) = self.solves(&a)?;
        out.fill(0.0);
        fixed_point_rows(out, rows, &a, &fa, n, n + 1);
        // Column k is coordinate k of (u, a1, a2).
        for k in 0..n + 2 {
            let t = self.m.a_z_q(u, &v, k)?;
            out[n + rows * k] = -dot(&w, &t);
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
        self.labels()
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
        let (v, w) = self.solves(&a)?;
        let vh = unit(&v, self.name)?;
        let wh = unit(&w, self.name)?;
        let bvv = self.m.b(u, &vh, &vh)?;
        let rest = self.rest(eigs);
        // LPPD: prod(mu + 1) on the LP curve, prod(mu - 1) on the PD curve.
        let lppd = rest_product(&rest, -self.sigma);
        let ns = pair_product(&rest);
        let third = if self.sigma < 0.0 {
            dot(&wh, &bvv)
        } else {
            self.gpd(u, &a, &vh, &wh, &bvv)
        };
        let r = dot(&wh, &vh);
        self.v_b = vh;
        self.w_b = wh;
        Ok(vec![lppd, ns, third, r])
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        let labels = self.labels();
        if i == 1 {
            // A real pair of product 1 is a neutral saddle, not an NS point.
            nearest_pair_is_complex(&self.rest(eigs)).then_some(labels[1])
        } else {
            labels.get(i).copied()
        }
    }
}

/// The NS curve of fixed points of a map.
pub struct MapNsProblem<'a> {
    m: Model2<'a>,
    /// n by 2, column-major.
    v_b: Vec<f64>,
    /// n by 2, column-major.
    w_b: Vec<f64>,
    /// The starting unknown `(u, a1, kappa, a2)`.
    x0: Vec<f64>,
}

impl<'a> MapNsProblem<'a> {
    /// Checks the setup, takes `kappa` as the real part of the pair of multipliers whose
    /// product is nearest to 1 at the start, and the borders from the singular vectors of
    /// `A^2 - 2 kappa A + I` for its two smallest singular values there.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &CurveSetup,
    ) -> Result<MapNsProblem<'a>, EngineError> {
        if model.n() < 2 {
            return Err(EngineError::BadProblem(
                "an NS curve of a map needs at least two variables".into(),
            ));
        }
        let mut m = Model2::new(model, setup)?;
        let n = m.n;
        let s = setup.start();
        m.set(&s);
        let a = m.jac_u(&s[..n])?;
        let eigs = eigenvalues(&a, n)?;
        let pair = nearest_pair(&eigs).filter(|_| nearest_pair_is_complex(&eigs));
        let Some((i, _)) = pair else {
            return Err(EngineError::BadProblem(
                "the start has no complex pair of multipliers near the unit circle; start the \
                 NS curve from an NS point of a map EP branch"
                    .into(),
            ));
        };
        let kappa = eigs[i].0;
        let mm = ns_matrix(&a, n, kappa);
        let v_b = null_right(&mm, n, n, 2)?.concat();
        let w_b = null_left(&mm, n, n, 2)?.concat();
        let mut x0 = s[..n + 1].to_vec();
        x0.push(kappa);
        x0.push(s[n + 1]);
        Ok(MapNsProblem { m, v_b, w_b, x0 })
    }

    /// The starting unknown `(u, a1, kappa, a2)`.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    /// The LU factors of `[[M, W], [Vᵀ, 0]]`, (n + 2) square.
    fn bordered(&self, mm: &[f64]) -> Result<Lu, EngineError> {
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
        Ok(Lu::new(&b, s)?)
    }

    /// The columns of `X` (n by 2) and of `Y` (n by 2) of the bordered solves at `mm`.
    fn solves(&self, mm: &[f64]) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
        let n = self.m.n;
        let s = n + 2;
        let lu = self.bordered(mm)?;
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
}

impl Problem for MapNsProblem<'_> {
    fn dim(&self) -> usize {
        self.m.n + 3
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.m.n;
        self.m.set_pair(x[n], x[n + 2]);
        let u = &x[..n];
        fixed_point_residual(&mut self.m, u, &mut out[..n])?;
        let a = self.m.jac_u(u)?;
        let lu = self.bordered(&ns_matrix(&a, n, x[n + 1]))?;
        let mut xg = vec![0.0; n + 2];
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
        let kappa = x[n + 1];
        let a = self.m.jac_u(u)?;
        let fa = self.m.f_a(u)?;
        let (xs, ys) = self.solves(&ns_matrix(&a, n, kappa))?;
        let x1 = &xs[..n];
        let y = [&ys[..n], &ys[n..]];
        let ax1 = matvec(&a, n, x1);
        let q: Vec<f64> = ax1
            .iter()
            .zip(x1)
            .map(|(p, r)| p - 2.0 * kappa * r)
            .collect();
        let aty = [matvec_t(&a, n, y[0]), matvec_t(&a, n, y[1])];
        out.fill(0.0);
        fixed_point_rows(out, rows, &a, &fa, n, n + 2);
        for col in 0..n + 3 {
            if col == n + 1 {
                for i in 0..2 {
                    out[n + i + rows * col] = 2.0 * dot(y[i], &ax1);
                }
            } else {
                // Coordinate k of (u, a1, a2) for the columns of u, a1 and a2.
                let k = if col <= n { col } else { n + 1 };
                let t1 = self.m.a_z_q(u, &q, k)?;
                let t2 = self.m.a_z_q(u, x1, k)?;
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
        &MAP_NS_CURVE_LABELS
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
        let a = self.m.jac_u(u)?;
        let (xs, ys) = self.solves(&ns_matrix(&a, n, kappa))?;
        self.v_b = orthonormalize(&xs, n, "NS")?;
        self.w_b = orthonormalize(&ys, n, "NS")?;
        let mut t = vec![kappa - 1.0, kappa + 1.0, kappa + 0.5, kappa];
        if kappa.abs() >= 1.0 {
            t.extend([f64::NAN; 4]);
            return Ok(t);
        }
        let mut rest = eigs.to_vec();
        for r in kappa_roots(kappa) {
            remove_nearest(&mut rest, r);
        }
        t.push(rest_product(&rest, -1.0));
        t.push(rest_product(&rest, 1.0));
        t.push(pair_product(&rest));
        // CH at the multiplier nearest to e^{i theta}, cos theta = kappa.
        let target = (kappa, (1.0 - kappa * kappa).sqrt());
        let dist = |m: &(f64, f64)| (m.0 - target.0).hypot(m.1 - target.1);
        let ch = eigs
            .iter()
            .copied()
            .min_by(|p, q| dist(p).total_cmp(&dist(q)))
            .and_then(|m| {
                let gx = shifted(&a, n, -1.0);
                map_ns_core(&mut self.m.at(u), &a, &gx, m).ok()
            })
            .map_or(f64::NAN, |c| {
                let r = c.c1.0.hypot(c.c1.1);
                if r.is_finite() && r > 0.0 {
                    c.d / r
                } else {
                    f64::NAN
                }
            });
        t.push(ch);
        Ok(t)
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        if i == NSNS {
            nsns_label_ok(eigs).then_some(MAP_NS_CURVE_LABELS[NSNS])
        } else {
            MAP_NS_CURVE_LABELS.get(i).copied()
        }
    }

    fn crossing_ok(&self, i: usize, a: f64, b: f64) -> bool {
        i != CH || a.abs().max(b.abs()) >= CH_BRACKET_FLOOR
    }
}

/// Checks the start `x` of the curve `name`, corrects it at fixed `a2` (the last unknown),
/// and follows the curve from there.
fn follow(
    problem: &mut dyn Problem,
    x: Vec<f64>,
    name: &str,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let mut f = vec![0.0; x.len() - 1];
    problem.residual(&x, &mut f)?;
    // f64::max drops NaN, so the fold keeps a NaN entry by hand.
    let res = f.iter().fold(0.0_f64, |m, v| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v.abs())
        }
    });
    if res.is_nan() || res > START_RESIDUAL_MAX {
        return Err(EngineError::BadProblem(format!(
            "the start is not near the {name} curve: the residual is {res:e}; start from a \
             {name} point of a map EP branch"
        )));
    }
    let mut e_last = vec![0.0; x.len()];
    e_last[x.len() - 1] = 1.0;
    let x = match correct(problem, &x, &e_last, 0.0, x.clone(), opts, cancel) {
        Corrected::Done { x, .. } => x,
        Corrected::Stop(StopReason::Cancelled) => return Err(EngineError::Cancelled),
        Corrected::Stop(StopReason::Error(e)) => return Err(e),
        Corrected::Failed | Corrected::Stop(_) => {
            return Err(EngineError::BadProblem(format!(
                "the start did not converge onto the {name} curve in {} Newton iterations; \
                 start from a {name} point of a map EP branch",
                opts.max_newton
            )));
        }
    };
    let tangent = start_tangent(problem, &x)?;
    run_branch(problem, &x, Some(&tangent), opts, false, cancel, on_point)
}

fn lp_pd(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    sigma: f64,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut problem = MapLpPdProblem::new(model, setup, sigma)?;
    let x = problem.start().to_vec();
    let name = problem.name;
    follow(&mut problem, x, name, opts, cancel, on_point)
}

/// Follows the LP curve of fixed points of the map `model` (a multiplier 1) through the
/// start of `setup` in the parameters `setup.free`. `Options::bounds` bound `a2`. The start
/// may be off the curve by up to `START_RESIDUAL_MAX`; it is corrected at fixed `a2`. The
/// first tangent points to increasing `u[0]` (see `start_tangent`); `Options::direction`
/// -1 reverses it.
pub fn continue_map_lp_curve(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    lp_pd(model, setup, -1.0, opts, cancel, on_point)
}

/// Follows the PD curve of fixed points of the map `model` (a multiplier -1), as
/// `continue_map_lp_curve` does for the LP curve.
pub fn continue_map_pd_curve(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    lp_pd(model, setup, 1.0, opts, cancel, on_point)
}

/// Follows the NS curve of fixed points of the map `model` (multipliers `e^(+-i theta)`),
/// as `continue_map_lp_curve` does for the LP curve. The unknown has `kappa = cos(theta)`
/// after `a1`; it starts at the real part of the pair of multipliers whose product is
/// nearest to 1.
pub fn continue_map_ns_curve(
    model: &mut dyn ModelEval,
    setup: &CurveSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut problem = MapNsProblem::new(model, setup)?;
    let x = problem.start().to_vec();
    follow(&mut problem, x, "NS", opts, cancel, on_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ns_matrix_vanishes_on_a_unit_rotation() {
        // A rotation by theta has M = A^2 - 2 cos(theta) A + I = 0 (Cayley-Hamilton).
        let th: f64 = 0.7;
        let a = [th.cos(), th.sin(), -th.sin(), th.cos()];
        let m = ns_matrix(&a, 2, th.cos());
        assert!(m.iter().all(|v| v.abs() < 1e-15), "{m:?}");
    }

    #[test]
    fn nsns_labels_only_a_second_complex_pair() {
        // SPEC Section 8x, test 1.
        let pair = |r: f64, th: f64| [(r * th.cos(), r * th.sin()), (r * th.cos(), -r * th.sin())];
        let two_pairs: Vec<_> = pair(1.0, 1.0).into_iter().chain(pair(1.0, 2.0)).collect();
        assert!(nsns_label_ok(&two_pairs));
        let saddle: Vec<_> = pair(1.0, 1.0)
            .into_iter()
            .chain([(2.0, 0.0), (0.5, 0.0)])
            .collect();
        assert!(!nsns_label_ok(&saddle));
        // A third pair off the unit circle does not hide the saddle.
        let mut more = saddle.clone();
        more.extend(pair(0.5, 0.3));
        assert!(!nsns_label_ok(&more));
        assert!(!nsns_label_ok(&[(2.0, 0.0), (0.5, 0.0), (0.3, 0.0)]));
        assert!(!nsns_label_ok(&[]));
    }

    #[test]
    fn shifted_adds_to_the_diagonal() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(shifted(&a, 2, -1.0), vec![0.0, 2.0, 3.0, 3.0]);
    }
}
