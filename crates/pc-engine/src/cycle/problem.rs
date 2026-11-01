//! The collocation problem of a periodic orbit (SPEC §7).
//!
//! Time is rescaled to tau in [0, 1], so the orbit solves u' = T f(u, alpha) with
//! u(0) = u(1). The unknowns are x = (u_0, ..., u_{Nm}, T, alpha): the values at the
//! N m + 1 base points of a mesh of N intervals, n each, then the period and the
//! continuation parameter. On interval j, of length h_j, the orbit is the polynomial of
//! degree m through the base points u_{jm}, ..., u_{jm+m}, and it satisfies the ODE at
//! the m Gauss points (orthogonal collocation, de Boor and Swartz 1973; Doedel, Keller
//! and Kernévez 1991). The equations are the n N m collocation conditions, multiplied by
//! h_j, the n periodicity conditions u_0 - u_{Nm} = 0 and the phase condition
//! int_0^1 (u - u_ref)(tau)ᵀ u_ref'(tau) dtau = 0, evaluated with the same Gauss weights.
//! For a periodic u_ref this is the integral condition int uᵀ u_ref' = 0 of AUTO, and the
//! difference form keeps u = u_ref a solution of it when u_ref is periodic only to the
//! Newton tolerance.
//!
//! The linear solves go through the condensation solver of `pc-linalg`, with the phase
//! condition as the first border row and the PALC row as the last.
//!
//! A curve of cycles in two parameters (SPEC §8) uses the same equations with more
//! unknowns after T, listed as [`Extra`]s, and adds its own rows between the phase row
//! and the PALC row.

use super::basis::{CollocationBasis, NCOL_RANGE};
use super::mesh;
use crate::branch::Options;
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::problem::Problem;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{BlockSystem, CondenseShape, Condenser, LinalgError, eigenvalues, eigvec};

/// Where and along which parameter a periodic orbit is computed.
#[derive(Clone, Debug)]
pub struct CycleSetup {
    /// Index of the continuation parameter among the model parameters.
    pub free: usize,
    /// All `n_p` parameter values; the engine appends t = 0.
    pub params: Vec<f64>,
    /// Number of mesh intervals N (NTST), at least 1.
    pub ntst: usize,
    /// Number of collocation points per interval m (NCOL), in [`NCOL_RANGE`].
    pub ncol: usize,
    /// Guess for the orbit at the N m + 1 base points of the uniform mesh, point-major
    /// (u_0, ..., u_{Nm}), n values each. It is also the reference orbit of the phase
    /// condition.
    pub profile: Vec<f64>,
    /// Guess for the period, finite and positive.
    pub period: f64,
}

impl CycleSetup {
    pub fn new(
        free: usize,
        params: Vec<f64>,
        ntst: usize,
        ncol: usize,
        profile: Vec<f64>,
        period: f64,
    ) -> CycleSetup {
        CycleSetup {
            free,
            params,
            ntst,
            ncol,
            profile,
            period,
        }
    }
}

/// A periodic orbit at fixed parameters.
#[derive(Clone, Debug)]
pub struct PeriodicOrbit {
    /// The orbit at the N m + 1 base points, point-major, n values each.
    pub u: Vec<f64>,
    pub period: f64,
    /// The N + 1 mesh points, from 0 to 1.
    pub mesh: Vec<f64>,
    /// Floquet multipliers (re, im), sorted by distance from 1, so the trivial one comes
    /// first.
    pub multipliers: Vec<(f64, f64)>,
    pub newton_iters: usize,
}

/// An unknown of a cycle problem after the period T.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Extra {
    /// The value of the model parameter with this index.
    Param(usize),
    /// An unknown the collocation equations do not depend on, such as kappa on an NS
    /// curve.
    Other,
}

/// The collocation equations of a periodic orbit over a model, with x = (u, T, alpha) and
/// alpha being parameter `free`.
///
/// The Jacobian lives in the problem in condensation layout, so `jacobian_len` is 0.
/// `jacobian` assembles it at x and also writes the dense matrix when it is given a
/// buffer of (dim - 1) dim values. `bordered_solve` uses the matrix of the last
/// `jacobian` call. `state_jacobian` gives the monodromy matrix, whose eigenvalues are
/// the Floquet multipliers.
///
/// Inside the crate the unknowns after T can be any list of [`Extra`]s, x = (u, T,
/// extras), for k = 1 + extras unknowns and k border rows: the phase row, k - 2 rows
/// that a curve sets with [`CycleProblem::set_border_row`], then the PALC row. The
/// `Problem` methods then cover the collocation, periodicity and phase rows only.
pub struct CycleProblem<'a> {
    model: &'a mut dyn ModelEval,
    extras: Vec<Extra>,
    /// Parameters with t = 0 appended; the entries of the `Extra::Param`s are
    /// overwritten from x.
    p: Vec<f64>,
    basis: CollocationBasis,
    ntst: usize,
    mesh: Vec<f64>,
    /// Derivative of the phase condition with respect to the base-point values, and its
    /// value at the reference, so the condition is `phase · u - phase0 = 0`.
    phase: Vec<f64>,
    phase0: f64,
    /// Collocation blocks, boundary rows and the k border rows, in the layout of
    /// [`BlockSystem`].
    blocks: Vec<f64>,
    bc: Vec<f64>,
    border: Vec<f64>,
    condenser: Condenser,
    /// Whether the condenser runs the intervals on the current rayon pool
    /// ([`CondenseShape::parallel_pays`]).
    parallel: bool,
    /// The orbit at one collocation point, and F, F_u and F_p there.
    uz: Vec<f64>,
    f: Vec<f64>,
    jac_u: Vec<f64>,
    jac_p: Vec<f64>,
    /// Accepted points between mesh adaptations, 0 for none (`Options::mesh_adapt`).
    pub(crate) adapt_every: usize,
    /// Accepted points since the last adaptation, and whether the mesh already moved at
    /// the current point.
    since: usize,
    adapted_here: bool,
}

/// A zeroed vector, or an error instead of an abort when the memory is not there.
pub(crate) fn try_zeros(len: Option<usize>) -> Result<Vec<f64>, EngineError> {
    let oom = || EngineError::from(LinalgError::OutOfMemory);
    let len = len.ok_or_else(oom)?;
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| oom())?;
    v.resize(len, 0.0);
    Ok(v)
}

/// Checks the parameters and the mesh sizes against the model and returns the length of
/// the profile, n (N m + 1).
pub(crate) fn check_dims(
    model: &dyn ModelEval,
    free: usize,
    params: &[f64],
    ntst: usize,
    ncol: usize,
) -> Result<usize, EngineError> {
    let n = model.n();
    let bad = |m: String| Err(EngineError::BadProblem(m));
    if n == 0 {
        return bad("the model has no variables".into());
    }
    if params.len() != model.n_p() {
        return bad(format!(
            "expected {} parameters, got {}",
            model.n_p(),
            params.len()
        ));
    }
    if free >= model.n_p() {
        return bad(format!("parameter index {free} is out of range"));
    }
    if ntst == 0 {
        return bad("NTST must be at least 1".into());
    }
    if !NCOL_RANGE.contains(&ncol) {
        return bad(format!(
            "NCOL must be between {} and {}, got {ncol}",
            NCOL_RANGE.start(),
            NCOL_RANGE.end()
        ));
    }
    match ntst
        .checked_mul(ncol)
        .and_then(|k| k.checked_add(1))
        .and_then(|k| k.checked_mul(n))
    {
        None => bad("NTST times NCOL is too large".into()),
        Some(len) => Ok(len),
    }
}

fn check_setup(model: &dyn ModelEval, s: &CycleSetup) -> Result<(), EngineError> {
    let n = model.n();
    let bad = |m: String| Err(EngineError::BadProblem(m));
    let len = check_dims(model, s.free, &s.params, s.ntst, s.ncol)?;
    if len != s.profile.len() {
        return bad(format!(
            "expected {len} profile values (NTST NCOL + 1 points of {n}), got {}",
            s.profile.len()
        ));
    }
    if s.profile.iter().any(|v| !v.is_finite()) {
        return bad("the profile has NaN or infinite values".into());
    }
    if !(s.period.is_finite() && s.period > 0.0) {
        return bad(format!(
            "the period must be finite and positive, got {}",
            s.period
        ));
    }
    Ok(())
}

impl<'a> CycleProblem<'a> {
    /// Checks the setup against the model, on a uniform mesh, with `setup.profile` as the
    /// reference of the phase condition.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &CycleSetup,
    ) -> Result<CycleProblem<'a>, EngineError> {
        let free = setup.free;
        CycleProblem::with_extras(model, setup, vec![Extra::Param(free)])
    }

    /// As [`CycleProblem::new`], with `extras` as the unknowns after T in place of
    /// alpha. `setup.free` is still checked against the model.
    pub(crate) fn with_extras(
        model: &'a mut dyn ModelEval,
        setup: &CycleSetup,
        extras: Vec<Extra>,
    ) -> Result<CycleProblem<'a>, EngineError> {
        check_setup(model, setup)?;
        let (n, n_p) = (model.n(), model.n_p());
        if let Some(i) = extras.iter().find_map(|e| match e {
            Extra::Param(i) if *i >= n_p => Some(*i),
            _ => None,
        }) {
            return Err(EngineError::BadProblem(format!(
                "parameter index {i} is out of range"
            )));
        }
        let (ntst, m) = (setup.ntst, setup.ncol);
        let k = extras.len() + 1;
        let shape = CondenseShape::new(n, m, ntst, k)?;
        let blocks = try_zeros(
            ntst.checked_mul(shape.block_rows())
                .and_then(|k| k.checked_mul(shape.block_cols())),
        )?;
        let mut bc = try_zeros(boundary_len(n, k))?;
        for c in 0..n {
            bc[c + n * c] = 1.0;
            bc[c + n * (n + c)] = -1.0;
        }
        let mut p = setup.params.clone();
        p.push(0.0);
        let mut problem = CycleProblem {
            model,
            extras,
            p,
            basis: CollocationBasis::new(m)?,
            ntst,
            mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
            phase: try_zeros(Some(setup.profile.len()))?,
            phase0: 0.0,
            blocks,
            bc,
            border: try_zeros(shape.dim().checked_mul(k))?,
            condenser: Condenser::new(shape)?,
            parallel: shape.parallel_pays(),
            uz: vec![0.0; n],
            f: vec![0.0; n],
            jac_u: vec![0.0; n * n],
            jac_p: vec![0.0; n * n_p],
            adapt_every: 0,
            since: 0,
            adapted_here: false,
        };
        problem.set_reference(&setup.profile);
        Ok(problem)
    }

    /// Makes `uref`, the orbit at the base points, the reference of the phase condition:
    /// the phase row becomes sum_j sum_i w_i L_l(z_i) h_j u_ref'(z_ji) for base point
    /// jm + l. The h_j of the quadrature cancels the 1/h_j of the derivative.
    pub(crate) fn set_reference(&mut self, uref: &[f64]) {
        let (n, m) = (self.model.n(), self.basis.m());
        let b = &self.basis;
        self.phase.fill(0.0);
        let mut dref = vec![0.0; n];
        for j in 0..self.ntst {
            let at = j * m * n;
            for i in 0..m {
                for (c, d) in dref.iter_mut().enumerate() {
                    *d = (0..=m)
                        .map(|l| b.derivative_at_node(i, l) * uref[at + l * n + c])
                        .sum();
                }
                let w = b.weights()[i];
                for l in 0..=m {
                    let wl = w * b.value_at_node(i, l);
                    let row = &mut self.phase[at + l * n..at + (l + 1) * n];
                    for (r, d) in row.iter_mut().zip(&dref) {
                        *r += wl * d;
                    }
                }
            }
        }
        self.phase0 = self.phase.iter().zip(uref).map(|(a, b)| a * b).sum();
        let k = self.k();
        for (q, v) in self.phase.iter().enumerate() {
            self.border[k * q] = *v;
        }
    }

    /// Number of base points N m + 1.
    pub(crate) fn points(&self) -> usize {
        self.ntst * self.basis.m() + 1
    }

    /// Number of profile values n (N m + 1), the index of T in x.
    pub(crate) fn profile_len(&self) -> usize {
        self.model.n() * self.points()
    }

    /// Number of unknowns after the profile (T and the extras), and of border rows.
    pub(crate) fn k(&self) -> usize {
        self.extras.len() + 1
    }

    /// Sets border row `r`, 1 <= r <= k - 2, to `row`, the derivative of an added
    /// equation with respect to all dim unknowns.
    pub(crate) fn set_border_row(&mut self, r: usize, row: &[f64]) {
        let k = self.k();
        debug_assert!((1..k - 1).contains(&r) && row.len() == self.dim());
        for (q, v) in row.iter().enumerate() {
            self.border[r + k * q] = *v;
        }
    }

    /// Sets the parameters of the `Extra::Param`s from x and returns T.
    fn read_extras(&mut self, x: &[f64]) -> f64 {
        let k = self.profile_len();
        for (i, e) in self.extras.iter().enumerate() {
            if let Extra::Param(a) = e {
                self.p[*a] = x[k + 1 + i];
            }
        }
        x[k]
    }

    /// The orbit at node i of the interval whose base points are `base` into `uz`.
    fn interpolate(&mut self, base: &[f64], i: usize) {
        let n = self.uz.len();
        for c in 0..n {
            self.uz[c] = (0..=self.basis.m())
                .map(|l| self.basis.value_at_node(i, l) * base[l * n + c])
                .sum();
        }
    }

    /// Assembles the collocation blocks at x.
    pub(crate) fn assemble(&mut self, x: &[f64]) -> Result<(), EngineError> {
        let (n, m) = (self.model.n(), self.basis.m());
        let nm = n * m;
        let tcol = n * (m + 1);
        let size = nm * (tcol + self.k());
        let period = self.read_extras(x);
        for j in 0..self.ntst {
            let h = self.mesh[j + 1] - self.mesh[j];
            let base = &x[j * nm..j * nm + tcol];
            for i in 0..m {
                self.interpolate(base, i);
                let (u, p) = (&self.uz, &self.p);
                self.model.eval(&EntryKind::F, u, p, &[], &mut self.f)?;
                self.model
                    .eval(&EntryKind::JacU, u, p, &[], &mut self.jac_u)?;
                self.model
                    .eval(&EntryKind::JacP, u, p, &[], &mut self.jac_p)?;
                let blk = &mut self.blocks[j * size..(j + 1) * size];
                for c in 0..n {
                    let row = i * n + c;
                    for l in 0..=m {
                        let d = self.basis.derivative_at_node(i, l);
                        let v = h * period * self.basis.value_at_node(i, l);
                        for c2 in 0..n {
                            let e = -v * self.jac_u[c + n * c2];
                            blk[row + nm * (l * n + c2)] = if c2 == c { e + d } else { e };
                        }
                    }
                    blk[row + nm * tcol] = -h * self.f[c];
                    for (e, extra) in self.extras.iter().enumerate() {
                        blk[row + nm * (tcol + 1 + e)] = match extra {
                            Extra::Param(a) => -h * period * self.jac_p[a * n + c],
                            Extra::Other => 0.0,
                        };
                    }
                }
            }
        }
        Ok(())
    }

    /// Writes the assembled Jacobian as a dense (dim - 1) by dim matrix: the collocation,
    /// periodicity and phase rows, then border rows 1 to k - 2.
    pub(crate) fn write_dense(&self, out: &mut [f64]) {
        let (n, m) = (self.model.n(), self.basis.m());
        let nm = n * m;
        let tcol = n * (m + 1);
        let k = self.k();
        let size = nm * (tcol + k);
        let dim = self.dim();
        let rows = dim - 1;
        let ext = self.profile_len();
        out.fill(0.0);
        for j in 0..self.ntst {
            let blk = &self.blocks[j * size..(j + 1) * size];
            for q in 0..tcol + k {
                let col = if q < tcol { j * nm + q } else { ext + q - tcol };
                for r in 0..nm {
                    out[j * nm + r + rows * col] = blk[r + nm * q];
                }
            }
        }
        let top = self.ntst * nm;
        let last = (self.points() - 1) * n;
        for q in 0..2 * n + k {
            let col = match q {
                q if q < n => q,
                q if q < 2 * n => last + q - n,
                q => ext + q - 2 * n,
            };
            for r in 0..n {
                out[top + r + rows * col] = self.bc[r + n * q];
            }
        }
        for (q, v) in self.phase.iter().enumerate() {
            out[top + n + rows * q] = *v;
        }
        for r in 1..k - 1 {
            for q in 0..dim {
                out[top + n + r + rows * q] = self.border[r + k * q];
            }
        }
    }

    /// The monodromy matrix at x, n by n, from the condensed collocation blocks.
    pub(super) fn monodromy(&mut self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.assemble(x)?;
        let sys = BlockSystem {
            blocks: &self.blocks,
            bc: &self.bc,
            border: &self.border,
        };
        self.condenser.condense_intervals(&sys, self.parallel)?;
        Ok(self.condenser.monodromy()?)
    }

    /// Adds to `out`, of length dim, the gradient over x of w · (A_u q + A_T q_t), where
    /// A_u and A_T are the derivatives of the collocation rows with respect to the
    /// profile and to T, `w` has one entry per collocation row and `q` is a profile. The
    /// curves of cycles (SPEC §8) take the derivatives of their defining functions from
    /// it. F_uu and F_up come from the model as directional derivatives.
    pub(super) fn coll_form_gradient(
        &mut self,
        x: &[f64],
        w: &[f64],
        q: &[f64],
        q_t: f64,
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        let (n, m, n_p) = (self.model.n(), self.basis.m(), self.model.n_p());
        let (nm, tcol, ext) = (n * m, n * (m + 1), self.profile_len());
        let period = self.read_extras(x);
        let (uu, up) = (
            EntryKind::Dir(vec![Slot::U, Slot::U]),
            EntryKind::Dir(vec![Slot::U, Slot::P]),
        );
        let mut qi = vec![0.0; n];
        let mut v = vec![0.0; 2 * n];
        let mut vp = vec![0.0; n + n_p];
        let mut d2 = vec![0.0; n];
        let mut grad = vec![0.0; n];
        for j in 0..self.ntst {
            let h = self.mesh[j + 1] - self.mesh[j];
            for i in 0..m {
                self.interpolate(&x[j * nm..j * nm + tcol], i);
                for (c, s) in qi.iter_mut().enumerate() {
                    *s = (0..=m)
                        .map(|l| self.basis.value_at_node(i, l) * q[j * nm + l * n + c])
                        .sum();
                }
                let wi = &w[j * nm + i * n..j * nm + (i + 1) * n];
                let (u, p) = (&self.uz, &self.p);
                self.model
                    .eval(&EntryKind::JacU, u, p, &[], &mut self.jac_u)?;
                self.model
                    .eval(&EntryKind::JacP, u, p, &[], &mut self.jac_p)?;
                // The derivative with respect to the orbit at this node, then spread over
                // the base points of the interval.
                v[..n].copy_from_slice(&qi);
                for c2 in 0..n {
                    v[n..].fill(0.0);
                    v[n + c2] = 1.0;
                    self.model.eval(&uu, u, p, &v, &mut d2)?;
                    grad[c2] = (0..n)
                        .map(|c| wi[c] * (period * d2[c] + q_t * self.jac_u[c + n * c2]))
                        .sum::<f64>()
                        * -h;
                }
                for l in 0..=m {
                    let lv = self.basis.value_at_node(i, l);
                    let row = &mut out[j * nm + l * n..j * nm + (l + 1) * n];
                    for (o, g) in row.iter_mut().zip(&grad) {
                        *o += lv * g;
                    }
                }
                let aq = |c: usize| {
                    (0..n)
                        .map(|c2| self.jac_u[c + n * c2] * qi[c2])
                        .sum::<f64>()
                };
                out[ext] -= h * (0..n).map(|c| wi[c] * aq(c)).sum::<f64>();
                for (e, extra) in self.extras.iter().enumerate() {
                    let Extra::Param(a) = *extra else { continue };
                    vp[..n].copy_from_slice(&qi);
                    vp[n..].fill(0.0);
                    vp[n + a] = 1.0;
                    self.model.eval(&up, u, p, &vp, &mut d2)?;
                    out[ext + 1 + e] -= h
                        * (0..n)
                            .map(|c| wi[c] * (period * d2[c] + q_t * self.jac_p[a * n + c]))
                            .sum::<f64>();
                }
            }
        }
        Ok(())
    }

    pub(super) fn n(&self) -> usize {
        self.model.n()
    }

    pub(super) fn basis(&self) -> &CollocationBasis {
        &self.basis
    }

    pub(super) fn ntst(&self) -> usize {
        self.ntst
    }

    /// The N + 1 mesh points, without a copy.
    pub(super) fn mesh_points(&self) -> &[f64] {
        &self.mesh
    }

    /// The phase row over the profile.
    pub(super) fn phase(&self) -> &[f64] {
        &self.phase
    }

    /// The collocation blocks of the last assembly.
    pub(super) fn blocks(&self) -> &[f64] {
        &self.blocks
    }

    pub(super) fn parallel(&self) -> bool {
        self.parallel
    }

    /// Replaces the mesh, with `uref` on the new mesh as the reference of the phase
    /// condition.
    pub(crate) fn set_mesh(&mut self, mesh: Vec<f64>, uref: &[f64]) {
        self.mesh = mesh;
        self.set_reference(uref);
    }

    /// The eigenfunction of the multiplier closest to -1 at x, at the base points, and that
    /// multiplier. It solves the linearized collocation equations with T and the extras held,
    /// from the eigenvector w of the monodromy matrix with 2-norm 1 at tau = 0, so it ends
    /// at mu w. `BadProblem` when that multiplier is not real.
    pub(crate) fn pd_eigenfunction(&mut self, x: &[f64]) -> Result<(Vec<f64>, f64), EngineError> {
        let n = self.model.n();
        let mono = self.monodromy(x)?;
        let eigs = eigenvalues(&mono, n)?;
        let dist = |z: &(f64, f64)| (z.0 + 1.0).hypot(z.1);
        let Some(&(re, im)) = eigs.iter().min_by(|a, b| dist(a).total_cmp(&dist(b))) else {
            return Err(EngineError::BadProblem("the model has no variables".into()));
        };
        if im.abs() > 1e-12 * re.abs().max(1.0) {
            return Err(EngineError::BadProblem(format!(
                "the multiplier closest to -1 is {re} + {im}i, which is not real; start \
                 from a PD point"
            )));
        }
        let (w, _) = eigvec(&mono, n, (re, 0.0))?;
        let (dim, k, ext) = (self.dim(), self.k(), self.profile_len());
        let mut bc = try_zeros(boundary_len(n, k))?;
        for c in 0..n {
            bc[c + n * c] = 1.0;
        }
        let mut border = try_zeros(dim.checked_mul(k))?;
        for r in 0..k {
            border[r + k * (ext + r)] = 1.0;
        }
        let mut rhs = vec![0.0; dim];
        let top = self.ntst * n * self.basis.m();
        rhs[top..top + n].copy_from_slice(&w);
        let mut sol = vec![0.0; dim];
        let sys = BlockSystem {
            blocks: &self.blocks,
            bc: &bc,
            border: &border,
        };
        self.condenser.solve(&sys, &rhs, &mut sol, self.parallel)?;
        sol.truncate(ext);
        Ok((sol, re))
    }
}

/// Length of the n by 2n + k boundary block, `None` when it does not fit in `usize`.
pub(super) fn boundary_len(n: usize, k: usize) -> Option<usize> {
    n.checked_mul(2)
        .and_then(|c| c.checked_add(k))
        .and_then(|c| c.checked_mul(n))
}

impl Problem for CycleProblem<'_> {
    fn dim(&self) -> usize {
        self.profile_len() + self.k()
    }

    /// The collocation, periodicity and phase rows. With k > 2 the rows after them are
    /// left to the curve.
    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let (n, m) = (self.model.n(), self.basis.m());
        let nm = n * m;
        let period = self.read_extras(x);
        for j in 0..self.ntst {
            let h = self.mesh[j + 1] - self.mesh[j];
            let base = &x[j * nm..j * nm + n * (m + 1)];
            for i in 0..m {
                self.interpolate(base, i);
                self.model
                    .eval(&EntryKind::F, &self.uz, &self.p, &[], &mut self.f)?;
                for c in 0..n {
                    let du: f64 = (0..=m)
                        .map(|l| self.basis.derivative_at_node(i, l) * base[l * n + c])
                        .sum();
                    out[j * nm + i * n + c] = du - h * period * self.f[c];
                }
            }
        }
        let top = self.ntst * nm;
        let last = (self.points() - 1) * n;
        for c in 0..n {
            out[top + c] = x[c] - x[last + c];
        }
        let dot: f64 = self.phase.iter().zip(x).map(|(a, b)| a * b).sum();
        out[top + n] = dot - self.phase0;
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.assemble(x)?;
        let dim = self.dim();
        if out.len() == (dim - 1) * dim {
            self.write_dense(out);
        } else if !out.is_empty() {
            return Err(EngineError::BadProblem("jacobian: wrong length".into()));
        }
        Ok(())
    }

    fn jacobian_len(&self) -> usize {
        0
    }

    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        self.monodromy(x).map(Some)
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        &CYCLE_LABELS
    }

    /// LPC is the alpha component of the tangent. PD is the product of mu + 1 and NS the
    /// product of mu_i mu_j - 1 over i < j, both over the nontrivial multipliers (SPEC §7).
    /// Both products are real, because the nontrivial multipliers come in conjugate pairs.
    fn codim2_tests(
        &mut self,
        x: &[f64],
        tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        let lpc = tangent[x.len() - 1];
        if eigs.is_empty() {
            return Ok(vec![lpc, f64::NAN, f64::NAN]);
        }
        let mu = nontrivial(eigs);
        let pd = mu.iter().fold((1.0, 0.0), |p, m| cmul(p, (m.0 + 1.0, m.1)));
        let ns = if mu.len() < 2 {
            f64::NAN
        } else {
            let mut prod = (1.0, 0.0);
            for i in 0..mu.len() {
                for j in i + 1..mu.len() {
                    let q = cmul(mu[i], mu[j]);
                    prod = cmul(prod, (q.0 - 1.0, q.1));
                }
            }
            prod.0
        };
        Ok(vec![lpc, pd.0, ns])
    }

    /// The NS product also vanishes when two real multipliers have product 1, a neutral
    /// saddle cycle. The zero is an NS point only when the pair closest to product 1 is a
    /// complex pair.
    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        if i != 2 {
            return CYCLE_LABELS.get(i).copied();
        }
        let mu = nontrivial(eigs);
        let mut best: Option<(f64, usize, usize)> = None;
        for a in 0..mu.len() {
            for b in a + 1..mu.len() {
                let q = cmul(mu[a], mu[b]);
                let d = (q.0 - 1.0).hypot(q.1);
                if best.is_none_or(|(e, _, _)| d < e) {
                    best = Some((d, a, b));
                }
            }
        }
        let (_, a, b) = best?;
        let (p, q) = (mu[a], mu[b]);
        let tol = 1e-10 * p.0.hypot(p.1).max(1.0);
        let pair = p.1 != 0.0 && (p.0 - q.0).abs() <= tol && (p.1 + q.1).abs() <= tol;
        pair.then_some("NS")
    }

    fn accept(&mut self, x: &[f64]) {
        let k = self.profile_len();
        self.set_reference(&x[..k]);
    }

    /// The N + 1 mesh points.
    fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// Moves the mesh every `adapt_every` accepted points and after a failed corrector,
    /// at most once per point, and interpolates the orbit part of x and v onto it. The
    /// interpolated x becomes the reference of the phase condition.
    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        if !failed {
            self.adapted_here = false;
            self.since += 1;
        }
        let due = if failed {
            !self.adapted_here
        } else {
            self.since >= self.adapt_every
        };
        if self.adapt_every == 0 || !due {
            return false;
        }
        self.since = 0;
        self.adapted_here = true;
        let n = self.model.n();
        let k = self.profile_len();
        let Some(new) = mesh::new_mesh(&self.basis, n, &self.mesh, &x[..k]) else {
            return false;
        };
        let m = self.basis.m();
        // Without the memory for the new orbit, the mesh stays.
        let (Ok(xu), Ok(vu)) = (
            mesh::interpolate(&self.basis, n, &self.mesh, &new, m, &x[..k]),
            mesh::interpolate(&self.basis, n, &self.mesh, &new, m, &v[..k]),
        ) else {
            return false;
        };
        x[..k].copy_from_slice(&xu);
        v[..k].copy_from_slice(&vu);
        self.mesh = new;
        self.set_reference(&x[..k]);
        true
    }

    /// Solves with the condensation solver, with `border` as the last border row. The
    /// determinant is not computed, so the result is NaN.
    fn bordered_solve(
        &mut self,
        _jac: &[f64],
        border: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
    ) -> Result<f64, EngineError> {
        let (dim, k) = (self.dim(), self.k());
        if border.len() != dim || rhs.len() != dim || sol.len() != dim {
            return Err(EngineError::BadProblem(
                "bordered_solve: wrong lengths".into(),
            ));
        }
        for (q, v) in border.iter().enumerate() {
            self.border[k - 1 + k * q] = *v;
        }
        let sys = BlockSystem {
            blocks: &self.blocks,
            bc: &self.bc,
            border: &self.border,
        };
        self.condenser.solve(&sys, rhs, sol, self.parallel)?;
        Ok(f64::NAN)
    }
}

/// Names of the cycle test functions, in the order of `Point::codim2` on a cycle branch.
pub const CYCLE_LABELS: [&str; 3] = ["LPC", "PD", "NS"];

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// The multipliers without the one closest to 1, the trivial multiplier.
fn nontrivial(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let dist = |z: &(f64, f64)| (z.0 - 1.0).hypot(z.1);
    let mut mu = eigs.to_vec();
    if let Some(k) = (0..mu.len()).min_by(|&a, &b| dist(&mu[a]).total_cmp(&dist(&mu[b]))) {
        mu.remove(k);
    }
    mu
}

/// Inf-norm that propagates NaN.
fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

/// Newton's method on the collocation equations at the parameters of `setup`, for the
/// orbit and the period, from `setup.profile` and `setup.period`. Uses the tolerances and
/// the iteration cap of the corrector; gives `NewtonFailed` otherwise. The multipliers
/// are the eigenvalues of the monodromy matrix at the solution.
pub fn periodic_orbit(
    model: &mut dyn ModelEval,
    setup: &CycleSetup,
    opts: &Options,
) -> Result<PeriodicOrbit, EngineError> {
    let n = model.n();
    let mut p = CycleProblem::new(model, setup)?;
    let dim = p.dim();
    let mut x = setup.profile.clone();
    x.push(setup.period);
    x.push(setup.params[setup.free]);
    // The last row fixes alpha: d alpha = 0.
    let mut fix = vec![0.0; dim];
    fix[dim - 1] = 1.0;
    let mut f = vec![0.0; dim - 1];
    let mut rhs = vec![0.0; dim];
    let mut dx = vec![0.0; dim];
    let mut res = f64::INFINITY;
    for it in 1..=opts.max_newton {
        p.residual(&x, &mut f)?;
        p.jacobian(&x, &mut [])?;
        for (r, v) in rhs.iter_mut().zip(&f) {
            *r = -v;
        }
        p.bordered_solve(&[], &fix, &rhs, &mut dx)?;
        for (xi, d) in x.iter_mut().zip(&dx) {
            *xi += d;
        }
        p.residual(&x, &mut f)?;
        res = norm_inf(&f);
        if norm_inf(&dx) <= opts.tol_x && res <= opts.tol_f {
            let mut multipliers = eigenvalues(&p.monodromy(&x)?, n)?;
            let dist = |z: &(f64, f64)| (z.0 - 1.0).hypot(z.1);
            multipliers.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
            x.truncate(dim - 1);
            let period = x.pop().unwrap_or(f64::NAN);
            return Ok(PeriodicOrbit {
                u: x,
                period,
                mesh: p.mesh.clone(),
                multipliers,
                newton_iters: it,
            });
        }
    }
    Err(EngineError::NewtonFailed {
        iters: opts.max_newton,
        residual: res,
    })
}
