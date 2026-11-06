//! The LPC curve: folds of limit cycles in two parameters (SPEC §8).
//!
//! The unknowns are x = (u_0, ..., u_{Nm}, T, a1, a2). The equations are the collocation,
//! periodicity and phase rows of the cycle problem in (u, T) at (a1, a2), plus G = 0. G is
//! the last component of the solution of the bordered system
//!
//! ```text
//! N [q; q_T; G] = [0; 1],   N = [ J        [0; w_b] ]
//!                               [ v_bᵀ     0        ]
//! ```
//!
//! where J is the Jacobian of the cycle problem in (u, T) with the phase row of its fixed
//! reference. J is singular exactly at a fold, so G = 0 there (Doedel, Govaerts and
//! Kuznetsov, SIAM J. Numer. Anal. 41 (2003), with this phase row in place of their
//! integral row). The column border w_b sits on the periodicity and phase rows, and the row
//! border v_b covers (u, T). With Nᵀ [y; h] = [0; 1], G_x = -yᵀ N_x [q; q_T; G], and only
//! the collocation rows of N depend on x.

use super::bordered::{Bordered, unit};
use super::codim2::{LPC_CURVE_LABELS, lpc_curve_label, lpc_curve_tests};
use super::continuation::Scaled;
use super::curve::{CycleCurveStart, run_curve};
use super::mesh;
use super::normal::CycleBifurcation;
use super::problem::{CycleProblem, Extra};
use crate::branch::{Branch, Options, Point};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::problem::Problem;
use pc_linalg::CondenseShape;
use std::sync::atomic::AtomicBool;

/// The LPC curve problem over the collocation equations, x = (u, T, a1, a2).
///
/// As on a cycle branch, the Jacobian lives in the problem, `jacobian_len` is 0 and
/// `jacobian` writes the dense matrix only when it is given a (dim - 1) dim buffer.
pub struct LpcProblem<'a> {
    outer: CycleProblem<'a>,
    /// N with k = 2: the unknowns (q, q_T, G), the phase row and the v_b row as border
    /// rows.
    inner: Bordered,
    /// Borders of N over (u, T), and over the periodicity and phase rows.
    v_b: Vec<f64>,
    w_b: Vec<f64>,
    /// [q; q_T; G] and [y; h], K + 2 values each.
    w: Vec<f64>,
    y: Vec<f64>,
    gz: Vec<f64>,
    x0: Vec<f64>,
}

impl<'a> LpcProblem<'a> {
    /// Checks the start and sets the borders by two steps of inverse iteration from a
    /// constant border, at the start.
    pub fn new(
        model: &'a mut dyn ModelEval,
        start: &CycleCurveStart,
    ) -> Result<LpcProblem<'a>, EngineError> {
        let (setup, mesh) = start.setup(model)?;
        let [a1, a2] = start.free;
        let mut x0 = setup.profile.clone();
        x0.extend([setup.period, setup.params[a1], setup.params[a2]]);
        let mut outer =
            CycleProblem::with_extras(model, &setup, vec![Extra::Param(a1), Extra::Param(a2)])?;
        outer.set_mesh(mesh, &setup.profile);
        let (n, m, ntst) = (outer.n(), outer.basis().m(), outer.ntst());
        let inner = Bordered::new(CondenseShape::new(n, m, ntst, 2)?)?;
        let ext = outer.profile_len();
        let len = inner.dim();
        let mut p = LpcProblem {
            inner,
            v_b: vec![1.0 / ((ext + 1) as f64).sqrt(); ext + 1],
            w_b: vec![1.0 / ((n + 1) as f64).sqrt(); n + 1],
            w: vec![0.0; len],
            y: vec![0.0; len],
            gz: vec![0.0; ext + 3],
            outer,
            x0,
        };
        let x = p.x0.clone();
        for _ in 0..2 {
            p.factor(&x)?;
            p.transpose_solve()?;
            p.take_borders();
        }
        p.inner.forget();
        Ok(p)
    }

    /// The start x = (u, T, a1, a2) on the mesh of the problem.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    /// Assembles the cycle Jacobian at x, factors N and solves N w = e, unless that is
    /// already done at this x.
    fn factor(&mut self, x: &[f64]) -> Result<(), EngineError> {
        if self.inner.is_at(x) {
            return Ok(());
        }
        self.inner.forget();
        self.outer.assemble(x)?;
        self.build();
        let last = self.inner.dim() - 1;
        let parallel = self.outer.parallel();
        self.inner.factor_solve(x, last, &mut self.w, parallel)
    }

    /// Solves Nᵀ y = e with the factored N.
    fn transpose_solve(&mut self) -> Result<(), EngineError> {
        let last = self.inner.dim() - 1;
        let parallel = self.outer.parallel();
        self.inner.solve_transpose(last, &mut self.y, parallel)
    }

    /// Copies the (u, T) columns of the assembled cycle blocks into N and sets its
    /// boundary and border rows.
    fn build(&mut self) {
        let (n, m) = (self.outer.n(), self.outer.basis().m());
        let (nm, tcol) = (n * m, n * (m + 1));
        let (outer_size, size) = (nm * (tcol + 3), nm * (tcol + 2));
        let src = self.outer.blocks();
        for j in 0..self.outer.ntst() {
            let to = &mut self.inner.blocks[j * size..(j + 1) * size];
            to[..nm * (tcol + 1)]
                .copy_from_slice(&src[j * outer_size..j * outer_size + nm * (tcol + 1)]);
            to[nm * (tcol + 1)..].fill(0.0);
        }
        let bc = &mut self.inner.bc;
        bc.fill(0.0);
        for c in 0..n {
            bc[c + n * c] = 1.0;
            bc[c + n * (n + c)] = -1.0;
            bc[c + n * (2 * n + 1)] = self.w_b[c];
        }
        let ext = self.outer.profile_len();
        let border = &mut self.inner.border;
        border.fill(0.0);
        for (q, v) in self.outer.phase().iter().enumerate() {
            border[2 * q] = *v;
        }
        border[2 * (ext + 1)] = self.w_b[n];
        for (q, v) in self.v_b.iter().enumerate() {
            border[1 + 2 * q] = *v;
        }
    }

    /// The new borders from the last solves: v_b from (q, q_T), w_b from the periodicity
    /// and phase parts of y. A border that is zero or not finite stays.
    fn take_borders(&mut self) {
        let ext = self.outer.profile_len();
        let n = self.outer.n();
        let top = ext - n;
        if let Some(v) = unit(&self.w[..ext + 1]) {
            self.v_b = v;
        }
        if let Some(v) = unit(&self.y[top..ext + 1]) {
            self.w_b = v;
        }
    }
}

impl Problem for LpcProblem<'_> {
    fn dim(&self) -> usize {
        self.outer.dim()
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.outer.residual(x, out)?;
        self.factor(x)?;
        let ext = self.outer.profile_len();
        out[ext + 1] = self.w[ext + 1];
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let dim = self.dim();
        if !out.is_empty() && out.len() != (dim - 1) * dim {
            return Err(EngineError::BadProblem("jacobian: wrong length".into()));
        }
        self.factor(x)?;
        self.transpose_solve()?;
        let ext = self.outer.profile_len();
        let top = ext - self.outer.n();
        self.gz.fill(0.0);
        let (y, w) = (&self.y[..top], &self.w[..ext]);
        self.outer
            .coll_form_gradient(x, y, w, self.w[ext], &mut self.gz)?;
        for g in self.gz.iter_mut() {
            *g = -*g;
        }
        self.outer.set_border_row(1, &self.gz);
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
        &LPC_CURVE_LABELS
    }

    /// LPPD, LPNS, CPC (the coefficient b) and R1 over the multipliers without the two at 1.
    /// The coefficient is NaN at a point where the normal form is not defined.
    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        let b = self
            .outer
            .coefficient(CycleBifurcation::Lpc, x)
            .unwrap_or(f64::NAN);
        Ok(lpc_curve_tests(eigs, b))
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        lpc_curve_label(i, eigs)
    }

    /// Moves the phase reference to x and refreshes the borders there. When N cannot be
    /// solved at x the old borders stay.
    fn accept(&mut self, x: &[f64]) {
        self.inner.forget();
        self.outer.accept(x);
        if self.factor(x).and_then(|_| self.transpose_solve()).is_ok() {
            self.take_borders();
        }
        self.inner.forget();
    }

    fn mesh(&self) -> Vec<f64> {
        self.outer.mesh()
    }

    /// Moves the mesh as on a cycle branch and interpolates the profile part of v_b onto
    /// it. w_b lives on the periodicity and phase rows, so it stays.
    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        let old = self.outer.mesh_points().to_vec();
        if !self.outer.adapt(x, v, failed) {
            return false;
        }
        self.inner.forget();
        let (n, m, ext) = (
            self.outer.n(),
            self.outer.basis().m(),
            self.outer.profile_len(),
        );
        let new = self.outer.mesh_points().to_vec();
        if let Ok(vb) = mesh::interpolate(self.outer.basis(), n, &old, &new, m, &self.v_b[..ext]) {
            self.v_b[..ext].copy_from_slice(&vb);
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

/// Follows the LPC curve through the LPC point of `start` in the parameters
/// `start.free`. `Options::bounds` bound a2. The start is first corrected at fixed a2,
/// since a cycle branch locates its LPC points on another test function. The first tangent
/// points to growing a2, times `Options::direction`.
pub fn continue_lpc_curve(
    model: &mut dyn ModelEval,
    start: &CycleCurveStart,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let mut p = LpcProblem::new(model, start)?;
    p.outer.adapt_every = opts.mesh_adapt;
    let x0 = p.x0.clone();
    let (k, points) = (p.outer.profile_len(), p.outer.points());
    run_curve(Scaled::new(p, k, points), x0, 2, opts, cancel, on_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build};
    use pc_ode::{FsLoader, SourceMap, load};
    use std::path::Path;

    /// On the Bautin form at a generic point, the transposed solve gives the same G as the
    /// forward one, h = G.
    #[test]
    fn transposed_bordered_solve_gives_the_same_g() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/bautin_lpc.ode");
        let mut map = SourceMap::new();
        let mut loader = FsLoader::new();
        let top = loader.load_top(&mut map, &path).unwrap();
        let loaded = load(top, &mut map, &mut loader);
        let mut m = build(&loaded.expanded, &loaded.info, Evaluator::Jit).unwrap();
        let (ntst, ncol) = (4, 3);
        let pts = ntst * ncol + 1;
        let r = 0.6_f64;
        let orbit: Vec<f64> = (0..pts)
            .flat_map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / (pts - 1) as f64;
                [r * t.cos(), r * t.sin()]
            })
            .collect();
        let start = CycleCurveStart {
            free: [0, 1],
            params: vec![-0.2, 1.0, 0.5],
            orbit,
            period: 5.0,
            mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
            ntst,
            ncol,
        };
        let mut p = LpcProblem::new(m.as_mut(), &start).unwrap();
        let x = p.start().to_vec();
        p.factor(&x).unwrap();
        p.transpose_solve().unwrap();
        let ext = p.outer.profile_len();
        let (g, h) = (p.w[ext + 1], p.y[ext + 1]);
        assert!(g.abs() > 1e-6, "G = {g}");
        assert!(
            (g - h).abs() <= 1e-10 * g.abs().max(1.0),
            "G = {g}, h = {h}"
        );
    }
}
