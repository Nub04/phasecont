//! The NS curve: Neimark-Sacker points of limit cycles in two parameters (SPEC §8).
//!
//! The unknowns are x = (u_0, ..., u_{Nm}, T, a1, kappa, a2). The equations are the
//! collocation, periodicity and phase rows of the cycle problem in (u, T) at (a1, a2), plus
//! G11 = 0 and G21 = 0. Doedel, Govaerts and Kuznetsov (SIAM J. Numer. Anal. 41 (2003))
//! write the NS condition on [0, 2] as v' = T f_x(u) v with v(0) - 2 kappa v(1) + v(2) = 0,
//! which has a nonzero solution exactly when the monodromy M has the multipliers
//! exp(+-i theta) with kappa = cos theta. Here the same problem lives on [0, 1] in the
//! doubled state (v_a, v_b), v_b(t) = v(1 + t), with the boundary rows
//!
//! ```text
//! v_b(0) - v_a(1) = 0,   v_a(0) - 2 kappa v_a(1) + v_b(1) = 0,
//! ```
//!
//! so that (M^2 - 2 kappa M + I) v_a(0) = 0. With A_u the collocation rows in u, the
//! discrete D - T f_x(u), applied to each half, the bordered system is
//!
//! ```text
//! N [V; G] = [0; I],   N = [ A_u ⊕ A_u    0   ]
//!                          [ BC(kappa)    W_b ]
//!                          [ V_bᵀ         0   ]
//! ```
//!
//! with 2 by 2 G, two column borders W_b on the boundary rows and two row borders V_b over
//! the doubled profile. G = 0 at an NS point, and the curve continues the first column
//! G11, G21. With Nᵀ [Y_i; H_i] = e_i, G_i1,x = -Y_iᵀ N_x V_1, where only the collocation
//! rows and the kappa entries of N depend on x.

use super::bordered::{Bordered, orthonormal};
use super::codim2::{NS_CURVE_LABELS, neutral_saddle_gap, ns_curve_label, ns_curve_tests};
use super::continuation::Scaled;
use super::curve::{CycleCurveStart, run_curve};
use super::mesh;
use super::problem::{CycleProblem, Extra};
use crate::branch::{Branch, Options, Point};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::problem::Problem;
use pc_linalg::{CondenseShape, eigenvalues};
use std::sync::atomic::AtomicBool;

/// The NS curve problem over the collocation equations, x = (u, T, a1, kappa, a2).
///
/// As on a cycle branch, the Jacobian lives in the problem, `jacobian_len` is 0 and
/// `jacobian` writes the dense matrix only when it is given a (dim - 1) dim buffer.
pub struct NsProblem<'a> {
    outer: CycleProblem<'a>,
    /// N with k = 2 over the doubled state: the unknowns (V, G) and the V_b rows as border
    /// rows.
    inner: Bordered,
    /// Borders of N over the doubled profile and over the 2n boundary rows.
    v_b: [Vec<f64>; 2],
    w_b: [Vec<f64>; 2],
    /// [V_1; G_1] and [V_2; G_2], the solutions of N s = e for the last two unit vectors.
    /// The second is only solved when the borders change.
    s: Vec<f64>,
    s2: Vec<f64>,
    /// [Y_1; H_1] and [Y_2; H_2].
    y: [Vec<f64>; 2],
    /// The halves of V_1 over the profile and of a Y over the collocation rows.
    sa: Vec<f64>,
    sb: Vec<f64>,
    ya: Vec<f64>,
    yb: Vec<f64>,
    gz: Vec<f64>,
    x0: Vec<f64>,
}

impl<'a> NsProblem<'a> {
    /// Checks the start, takes kappa from the critical pair of multipliers and sets the
    /// borders by two steps of inverse iteration from fixed generic borders, at the start.
    pub fn new(
        model: &'a mut dyn ModelEval,
        start: &CycleCurveStart,
    ) -> Result<NsProblem<'a>, EngineError> {
        let (setup, mesh) = start.setup(model)?;
        let [a1, a2] = start.free;
        let mut x0 = setup.profile.clone();
        x0.extend([setup.period, setup.params[a1], 0.0, setup.params[a2]]);
        let extras = vec![Extra::Param(a1), Extra::Other, Extra::Param(a2)];
        let mut outer = CycleProblem::with_extras(model, &setup, extras)?;
        outer.set_mesh(mesh, &setup.profile);
        let (n, m, ntst) = (outer.n(), outer.basis().m(), outer.ntst());
        let ext = outer.profile_len();
        x0[ext + 2] = critical_kappa(&mut outer, &x0)?;
        let inner = Bordered::new(CondenseShape::new(2 * n, m, ntst, 2)?)?;
        let len = inner.dim();
        let generic = |len: usize| {
            let weyl = |step: f64| -> Vec<f64> {
                (1..=len).map(|q| (q as f64 * step).fract() - 0.5).collect()
            };
            orthonormal(
                &weyl(0.618_033_988_749_894_9),
                &weyl(std::f64::consts::SQRT_2),
            )
            .ok_or_else(|| EngineError::BadProblem("no generic NS borders".into()))
        };
        let rows = ext - n;
        let mut p = NsProblem {
            inner,
            v_b: generic(2 * ext)?,
            w_b: generic(2 * n)?,
            s: vec![0.0; len],
            s2: vec![0.0; len],
            y: [vec![0.0; len], vec![0.0; len]],
            sa: vec![0.0; ext],
            sb: vec![0.0; ext],
            ya: vec![0.0; rows],
            yb: vec![0.0; rows],
            gz: vec![0.0; ext + 4],
            outer,
            x0,
        };
        let x = p.x0.clone();
        for _ in 0..2 {
            p.factor(&x)?;
            p.transpose_solve()?;
            p.take_borders()?;
        }
        p.inner.forget();
        Ok(p)
    }

    /// The start x = (u, T, a1, kappa, a2) on the mesh of the problem.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    /// Assembles the cycle Jacobian at x, factors N and solves N s = e_{2K}, unless that
    /// is already done at this x.
    fn factor(&mut self, x: &[f64]) -> Result<(), EngineError> {
        if self.inner.is_at(x) {
            return Ok(());
        }
        self.inner.forget();
        self.outer.assemble(x)?;
        self.build(x[self.outer.profile_len() + 2]);
        let top = self.inner.dim() - 2;
        let parallel = self.outer.parallel();
        self.inner.factor_solve(x, top, &mut self.s, parallel)
    }

    /// Solves Nᵀ y = e for the last two unit vectors with the factored N.
    fn transpose_solve(&mut self) -> Result<(), EngineError> {
        let top = self.inner.dim() - 2;
        let parallel = self.outer.parallel();
        let [y1, y2] = &mut self.y;
        self.inner.solve_transpose(top, y1, parallel)?;
        self.inner.solve_transpose(top + 1, y2, parallel)
    }

    /// Copies the u columns of the assembled cycle blocks into both halves of N and sets
    /// its boundary and border rows.
    fn build(&mut self, kappa: f64) {
        let (n, m) = (self.outer.n(), self.outer.basis().m());
        let (nm, tcol, n2) = (n * m, n * (m + 1), 2 * n);
        let nm2 = 2 * nm;
        let (outer_size, size) = (nm * (tcol + 4), nm2 * (2 * tcol + 2));
        let src = self.outer.blocks();
        for j in 0..self.outer.ntst() {
            let from = &src[j * outer_size..j * outer_size + nm * tcol];
            let to = &mut self.inner.blocks[j * size..(j + 1) * size];
            to.fill(0.0);
            for i in 0..m {
                for c in 0..n {
                    for l in 0..=m {
                        for c2 in 0..n {
                            let v = from[(i * n + c) + nm * (l * n + c2)];
                            for h in [0, n] {
                                to[(i * n2 + h + c) + nm2 * (l * n2 + h + c2)] = v;
                            }
                        }
                    }
                }
            }
        }
        // Columns 0..2n are the doubled state at t = 0, then 2n..4n at t = 1, then G.
        let bc = &mut self.inner.bc;
        bc.fill(0.0);
        for r in 0..n {
            bc[r + n2 * (n + r)] = 1.0;
            bc[r + n2 * (n2 + r)] = -1.0;
            bc[n + r + n2 * r] = 1.0;
            bc[n + r + n2 * (n2 + r)] = -2.0 * kappa;
            bc[n + r + n2 * (n2 + n + r)] = 1.0;
        }
        for c in 0..n2 {
            bc[c + n2 * (2 * n2)] = self.w_b[0][c];
            bc[c + n2 * (2 * n2 + 1)] = self.w_b[1][c];
        }
        let border = &mut self.inner.border;
        border.fill(0.0);
        for q in 0..self.v_b[0].len() {
            border[2 * q] = self.v_b[0][q];
            border[1 + 2 * q] = self.v_b[1][q];
        }
    }

    /// The new borders from the last solves: V_b from V_1 and V_2, W_b from the boundary
    /// part of Y_1 and Y_2. A pair that is not independent or not finite stays.
    fn take_borders(&mut self) -> Result<(), EngineError> {
        let top = self.inner.dim() - 2;
        let parallel = self.outer.parallel();
        self.inner.solve(top + 1, &mut self.s2, parallel)?;
        if let Some(v) = orthonormal(&self.s[..top], &self.s2[..top]) {
            self.v_b = v;
        }
        let b = top - 2 * self.outer.n();
        if let Some(w) = orthonormal(&self.y[0][b..top], &self.y[1][b..top]) {
            self.w_b = w;
        }
        Ok(())
    }
}

/// The kappa of the complex pair of multipliers at x that lies closest to the unit circle,
/// or `BadProblem` when there is no complex pair.
fn critical_kappa(outer: &mut CycleProblem<'_>, x: &[f64]) -> Result<f64, EngineError> {
    let n = outer.n();
    let eigs = eigenvalues(&outer.monodromy(x)?, n)?;
    let off = |e: &(f64, f64)| (e.0.hypot(e.1) - 1.0).abs();
    eigs.iter()
        .filter(|e| e.1.abs() > 1e-12 * e.0.abs().max(1.0))
        .min_by(|a, b| off(a).total_cmp(&off(b)))
        .map(|e| e.0 / e.0.hypot(e.1))
        .ok_or_else(|| {
            EngineError::BadProblem(
                "the monodromy matrix at the start has no complex pair of multipliers; start \
                 from an NS point"
                    .into(),
            )
        })
}

/// The halves of `v`, `len` blocks of 2n values, into `a` and `b`, len blocks of n each.
fn split(v: &[f64], n: usize, len: usize, a: &mut [f64], b: &mut [f64]) {
    for p in 0..len {
        a[p * n..(p + 1) * n].copy_from_slice(&v[2 * p * n..(2 * p + 1) * n]);
        b[p * n..(p + 1) * n].copy_from_slice(&v[(2 * p + 1) * n..(2 * p + 2) * n]);
    }
}

impl Problem for NsProblem<'_> {
    fn dim(&self) -> usize {
        self.outer.dim()
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.outer.residual(x, out)?;
        self.factor(x)?;
        let (ext, top) = (self.outer.profile_len(), self.inner.dim() - 2);
        out[ext + 1] = self.s[top];
        out[ext + 2] = self.s[top + 1];
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let dim = self.dim();
        if !out.is_empty() && out.len() != (dim - 1) * dim {
            return Err(EngineError::BadProblem("jacobian: wrong length".into()));
        }
        self.factor(x)?;
        self.transpose_solve()?;
        let (n, ext) = (self.outer.n(), self.outer.profile_len());
        let (rows, points) = (ext - n, self.outer.points());
        // The boundary rows of N, and the doubled state at t = 1.
        let (bc, last) = (2 * rows, 2 * (ext - n));
        split(&self.s, n, points, &mut self.sa, &mut self.sb);
        for i in 0..2 {
            let y = &self.y[i];
            split(y, n, points - 1, &mut self.ya, &mut self.yb);
            self.gz.fill(0.0);
            let (ya, yb, sa, sb) = (&self.ya, &self.yb, &self.sa, &self.sb);
            self.outer
                .coll_form_gradient(x, ya, sa, 0.0, &mut self.gz)?;
            self.outer
                .coll_form_gradient(x, yb, sb, 0.0, &mut self.gz)?;
            for g in self.gz.iter_mut() {
                *g = -*g;
            }
            // N_kappa has -2 at (v_a(0) - 2 kappa v_a(1) + v_b(1) row r, v_a(1)_r).
            self.gz[ext + 2] = 2.0
                * (0..n)
                    .map(|r| y[bc + n + r] * self.s[last + r])
                    .sum::<f64>();
            self.outer.set_border_row(1 + i, &self.gz);
        }
        if !out.is_empty() {
            self.outer.write_dense(out);
        }
        Ok(())
    }

    fn jacobian_len(&self) -> usize {
        0
    }

    /// The monodromy matrix, from a new assembly of the cycle blocks.
    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        self.inner.forget();
        self.outer.state_jacobian(x)
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        &NS_CURVE_LABELS
    }

    /// The resonances from kappa, and LPNS and PDNS over the multipliers without the one
    /// at 1 and the pair of kappa.
    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        Ok(ns_curve_tests(x[self.outer.profile_len() + 2], eigs))
    }

    fn codim2_label(&self, i: usize, _eigs: &[(f64, f64)]) -> Option<&'static str> {
        ns_curve_label(i)
    }

    /// Moves the phase reference to x and refreshes the borders there. When N cannot be
    /// solved at x the old borders stay.
    fn accept(&mut self, x: &[f64]) {
        self.inner.forget();
        self.outer.accept(x);
        let _ = self
            .factor(x)
            .and_then(|_| self.transpose_solve())
            .and_then(|_| self.take_borders());
        self.inner.forget();
    }

    fn mesh(&self) -> Vec<f64> {
        self.outer.mesh()
    }

    /// Moves the mesh as on a cycle branch and interpolates V_b onto it. W_b lives on the
    /// boundary rows, so it stays.
    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        let old = self.outer.mesh_points().to_vec();
        if !self.outer.adapt(x, v, failed) {
            return false;
        }
        self.inner.forget();
        let (n, m) = (self.outer.n(), self.outer.basis().m());
        let new = self.outer.mesh_points().to_vec();
        for vb in self.v_b.iter_mut() {
            if let Ok(w) = mesh::interpolate(self.outer.basis(), 2 * n, &old, &new, m, vb) {
                *vb = w;
            }
        }
        true
    }

    fn bordered_solve(
        &mut self,
        jac: &[f64],
        border: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
    ) -> Result<f64, EngineError> {
        self.outer.bordered_solve(jac, border, rhs, sol)
    }
}

/// Follows the NS curve through the NS point of `start` in the parameters `start.free`.
/// `Options::bounds` bound a2. kappa starts from the complex pair of multipliers closest to
/// the unit circle, and the start is first corrected at fixed a2, since a cycle branch
/// locates its NS points on another test function. The first tangent points to growing
/// a2, times `Options::direction`.
pub fn continue_ns_curve(
    model: &mut dyn ModelEval,
    start: &CycleCurveStart,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let mut p = NsProblem::new(model, start)?;
    p.outer.adapt_every = opts.mesh_adapt;
    let x0 = p.x0.clone();
    let (k, points) = (p.outer.profile_len(), p.outer.points());
    let mut branch = run_curve(Scaled::new(p, k, points), x0, opts, cancel, on_point)?;
    neutral_saddle_warning(&mut branch);
    Ok(branch)
}

/// Relative distance above which a root of mu^2 - 2 kappa mu + 1 counts as missing from
/// the multipliers.
const PAIR_TOL: f64 = 1e-6;

/// For |kappa| > 1 the NS equations also hold when the monodromy has only one of the two
/// real roots of mu^2 - 2 kappa mu + 1, so the curve can run through cycles that are no
/// neutral saddles. Warns when points with |kappa| > 1 lack one of the roots.
fn neutral_saddle_warning(branch: &mut Branch) {
    let mut count = 0;
    let mut worst = 0.0_f64;
    for pt in &branch.points {
        let Some(&kappa) = pt.u.last() else { continue };
        if let Some(gap) = neutral_saddle_gap(kappa, &pt.eigenvalues)
            && (gap.is_nan() || gap > PAIR_TOL)
        {
            count += 1;
            worst = worst.max(if gap.is_nan() { f64::INFINITY } else { gap });
        }
    }
    if count > 0 {
        branch.warnings.push(format!(
            "{count} of {} points with |kappa| > 1 lack one of the multipliers mu, 1/mu with \
             mu + 1/mu = 2 kappa (largest relative distance {worst:.3e}); these points have \
             one real multiplier there and are no neutral saddle cycles",
            branch.points.len()
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::StopReason;

    fn point(kappa: f64, eigenvalues: Vec<(f64, f64)>) -> Point {
        Point {
            u: vec![0.0, 1.0, kappa],
            alpha: 0.0,
            tangent: Vec::new(),
            eigenvalues,
            tests: [f64::NAN; 3],
            codim2: Vec::new(),
            h: 0.1,
            newton_iters: 2,
            mesh: Vec::new(),
        }
    }

    fn branch(points: Vec<Point>) -> Branch {
        Branch {
            points,
            specials: Vec::new(),
            stop: StopReason::MaxSteps,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_curve_of_neutral_saddles_has_no_warning() {
        // kappa = 1.25 has the roots 2 and 1/2.
        let mut b = branch(vec![
            point(
                0.5,
                vec![(1.0, 0.0), (0.5, 0.75_f64.sqrt()), (0.5, -0.75_f64.sqrt())],
            ),
            point(1.25, vec![(1.0, 0.0), (2.0, 0.0), (0.5, 0.0), (0.1, 0.0)]),
            point(1.0, Vec::new()),
        ]);
        neutral_saddle_warning(&mut b);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }

    #[test]
    fn points_with_one_real_root_get_a_warning() {
        let mut b = branch(vec![
            point(1.25, vec![(1.0, 0.0), (2.0, 0.0), (0.5, 0.0)]),
            point(1.25, vec![(1.0, 0.0), (2.0, 0.0), (0.3, 0.0)]),
            point(-1.25, vec![(1.0, 0.0), (-2.0, 0.0), (0.9, 0.0)]),
        ]);
        neutral_saddle_warning(&mut b);
        assert_eq!(b.warnings.len(), 1);
        assert!(
            b.warnings[0].starts_with("2 of 3 points"),
            "{}",
            b.warnings[0]
        );
        assert!(b.warnings[0].contains("1.400e0"), "{}", b.warnings[0]);
    }
}
