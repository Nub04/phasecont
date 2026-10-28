//! Refining LP, BP and Hopf points with their defining systems.
//!
//! A located special point is accurate to the tolerance of the root search. Each system
//! below has the point as an isolated regular solution, so Newton's method on it gives
//! the point to rounding error. All Jacobians are exact: `JacU`, `JacP` and `Dir`
//! entries of the model (no finite differences). The unknown `x = (u, alpha)` holds the
//! state and the free parameter, `A = F_u` and `F_x = [A F_alpha]`.
//!
//! Systems:
//! * [`LpSystem`], the minimally augmented fold system, `F = 0, g = 0`.
//! * [`HopfSystem`], the standard extended Hopf system with `3n + 2` unknowns.
//! * [`BpSystem`], the branch point system with `2n + 2` unknowns.
//!
//! The generic `Problem` trait has no `Dir` entries, so this module works on a
//! `&mut dyn ModelEval` plus the parameter values and the index of the free parameter.

use crate::branch::{Branch, Options, SpecialPoint, StopReason};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::special_coefficient;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{Lu, eigenvalues, eigvec, null_left, null_right};
use std::sync::atomic::{AtomicBool, Ordering};

/// Newton iterations allowed for one refinement.
pub const MAX_REFINE_ITERS: usize = 20;

/// A nonlinear system `G(z) = 0` with square exact Jacobian, column-major.
pub trait DefiningSystem {
    /// Number of unknowns, equal to the number of equations.
    fn dim(&self) -> usize;
    /// Writes `G(z)` to `out` (`dim` values).
    fn residual(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError>;
    /// Writes `dG/dz` to `out` (`dim * dim` values, entry (i, j) at `i + dim * j`).
    fn jacobian(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError>;
}

/// Model access shared by the three systems: parameters with t appended, and helpers for
/// the entries they need.
struct Ctx<'a> {
    model: &'a mut dyn ModelEval,
    n: usize,
    n_p: usize,
    free: usize,
    /// Parameter values with t = 0 appended; entry `free` follows alpha.
    p: Vec<f64>,
}

impl<'a> Ctx<'a> {
    fn new(
        model: &'a mut dyn ModelEval,
        params: &[f64],
        free: usize,
    ) -> Result<Ctx<'a>, EngineError> {
        let (n, n_p) = (model.n(), model.n_p());
        if n == 0 || params.len() != n_p || free >= n_p {
            return Err(EngineError::BadProblem(
                "refinement: parameters or the free index do not fit the model".into(),
            ));
        }
        let mut p = params.to_vec();
        p.push(0.0);
        Ok(Ctx {
            model,
            n,
            n_p,
            free,
            p,
        })
    }

    fn set_alpha(&mut self, alpha: f64) {
        self.p[self.free] = alpha;
    }

    fn f(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut out = vec![0.0; self.n];
        self.model.eval(&EntryKind::F, u, &self.p, &[], &mut out)?;
        Ok(out)
    }

    fn jac_u(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut out = vec![0.0; self.n * self.n];
        self.model
            .eval(&EntryKind::JacU, u, &self.p, &[], &mut out)?;
        Ok(out)
    }

    /// The column of `JacP` for the free parameter.
    fn f_alpha(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut all = vec![0.0; self.n * self.n_p];
        self.model
            .eval(&EntryKind::JacP, u, &self.p, &[], &mut all)?;
        Ok(all[self.free * self.n..(self.free + 1) * self.n].to_vec())
    }

    /// `[A | F_alpha]`, n rows and n + 1 columns.
    fn f_x(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut m = self.jac_u(u)?;
        m.extend(self.f_alpha(u)?);
        Ok(m)
    }

    /// Unit vector of coordinate `k` of x = (u, alpha) inside the `Full` slot (u then all
    /// parameters): u_k for k < n, the free parameter for k = n.
    fn unit_full(&self, k: usize) -> Vec<f64> {
        let mut e = vec![0.0; self.n + self.n_p];
        e[if k < self.n { k } else { self.n + self.free }] = 1.0;
        e
    }

    /// `A_z q` for z = coordinate `k` of x: the entry `Dir([U, Full])` with directions
    /// (q, e_k).
    fn a_z_q(&mut self, u: &[f64], q: &[f64], k: usize) -> Result<Vec<f64>, EngineError> {
        let mut v = q.to_vec();
        v.extend(self.unit_full(k));
        let mut out = vec![0.0; self.n];
        self.model.eval(
            &EntryKind::Dir(vec![Slot::U, Slot::Full]),
            u,
            &self.p,
            &v,
            &mut out,
        )?;
        Ok(out)
    }

    /// `F_xx[e_j, e_k]` for coordinates j and k of x: `Dir([Full, Full])`.
    fn f_xx(&mut self, u: &[f64], j: usize, k: usize) -> Result<Vec<f64>, EngineError> {
        let mut v = self.unit_full(j);
        v.extend(self.unit_full(k));
        let mut out = vec![0.0; self.n];
        self.model.eval(
            &EntryKind::Dir(vec![Slot::Full, Slot::Full]),
            u,
            &self.p,
            &v,
            &mut out,
        )?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------
// LP: minimally augmented fold system

/// The fold (LP) system: unknowns `z = (u, alpha)` (n + 1 values), equations `F = 0` and
/// `g = 0`.
///
/// `g` comes from the bordered system `[[A, w_b], [v_bᵀ, 0]] (q; g) = (0; 1)`. By
/// Cramer's rule `g = det A / det [[A, w_b], [v_bᵀ, 0]]`, so `g = 0` exactly when A is
/// singular. The borders `v_b` (right null vector of A) and `w_b` (left null vector) are
/// taken at the starting point and stay fixed. The gradient is `g_z = -wᵀ (A_z q)` with
/// `[[Aᵀ, v_b], [w_bᵀ, 0]] (w; h) = (0; 1)`.
///
/// Source: Kuznetsov, "NBA Lecture 3: Equilibrium bifurcations of ODEs and their numerical
/// analysis" (Utrecht, 2011), sections "Continuation of LP bifurcation" and "Derivatives
/// of G" (the equilibrium case). The printed sign of eq. (4.5) in Govaerts, Kuznetsov and
/// Dhooge, SIAM J. Sci. Comput. 27 (2005) disagrees with this formula for the cycle case;
/// the tests check `g_z` against central differences.
pub struct LpSystem<'a> {
    ctx: Ctx<'a>,
    v_b: Vec<f64>,
    w_b: Vec<f64>,
    z0: Vec<f64>,
}

impl<'a> LpSystem<'a> {
    /// Builds the system with borders from A at `(u, alpha)`.
    pub fn new(
        model: &'a mut dyn ModelEval,
        params: &[f64],
        free: usize,
        u: &[f64],
        alpha: f64,
    ) -> Result<LpSystem<'a>, EngineError> {
        let mut ctx = Ctx::new(model, params, free)?;
        check_len(u, ctx.n)?;
        ctx.set_alpha(alpha);
        let a = ctx.jac_u(u)?;
        let n = ctx.n;
        let v_b = first(null_right(&a, n, n, 1)?)?;
        let w_b = first(null_left(&a, n, n, 1)?)?;
        let mut z0 = u.to_vec();
        z0.push(alpha);
        Ok(LpSystem { ctx, v_b, w_b, z0 })
    }

    /// The starting point `(u, alpha)`.
    pub fn start(&self) -> &[f64] {
        &self.z0
    }

    /// The bordered matrix `[[A, w_b], [v_bᵀ, 0]]`, (n + 1) square.
    fn bordered(&self, a: &[f64]) -> Vec<f64> {
        let n = self.ctx.n;
        let m = n + 1;
        let mut b = vec![0.0; m * m];
        for j in 0..n {
            for i in 0..n {
                b[i + m * j] = a[i + n * j];
            }
            b[n + m * j] = self.v_b[j];
        }
        for i in 0..n {
            b[i + m * n] = self.w_b[i];
        }
        b
    }
}

impl DefiningSystem for LpSystem<'_> {
    fn dim(&self) -> usize {
        self.ctx.n + 1
    }

    fn residual(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        self.ctx.set_alpha(z[n]);
        let f = self.ctx.f(&z[..n])?;
        let a = self.ctx.jac_u(&z[..n])?;
        let lu = Lu::new(&self.bordered(&a), n + 1)?;
        let mut rhs = vec![0.0; n + 1];
        rhs[n] = 1.0;
        lu.solve(&mut rhs)?;
        out[..n].copy_from_slice(&f);
        out[n] = rhs[n];
        Ok(())
    }

    fn jacobian(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        let m = n + 1;
        self.ctx.set_alpha(z[n]);
        let u = &z[..n];
        let a = self.ctx.jac_u(u)?;
        let fa = self.ctx.f_alpha(u)?;
        let lu = Lu::new(&self.bordered(&a), m)?;
        let mut qg = vec![0.0; m];
        qg[n] = 1.0;
        lu.solve(&mut qg)?;
        let mut wh = vec![0.0; m];
        wh[n] = 1.0;
        lu.solve_transpose(&mut wh)?;
        let q = &qg[..n];
        let w = &wh[..n];
        for j in 0..n {
            for i in 0..n {
                out[i + m * j] = a[i + n * j];
            }
        }
        for i in 0..n {
            out[i + m * n] = fa[i];
        }
        for k in 0..=n {
            let azq = self.ctx.a_z_q(u, q, k)?;
            out[n + m * k] = -w.iter().zip(&azq).map(|(a, b)| a * b).sum::<f64>();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// H: standard extended system

/// The Hopf system: unknowns `z = (u, alpha, q_r, q_i, omega)` (3n + 2 values), equations
///
/// ```text
/// F = 0,  A q_r + omega q_i = 0,  A q_i - omega q_r = 0,  cᵀ q_r = 1,  cᵀ q_i = 0.
/// ```
///
/// Together the middle two say `A (q_r + i q_i) = i omega (q_r + i q_i)`.
///
/// Source: Griewank and Reddien, "The calculation of Hopf points by a direct method",
/// IMA J. Numer. Anal. 3 (1983) 295-303, eq. (3.1), as quoted in Boulle, Farrell and
/// Rognes, "Optimization of Hopf bifurcation points", arXiv:2201.11684, Sec. 2.2,
/// eq. (2.6). Their normalization is `<c,v> = 0, <c,w> = 1`; the one here is the same up
/// to a phase rotation of the eigenvector. The conditions for a nonsingular Jacobian are
/// not taken from a primary source; the tests report its smallest singular value.
pub struct HopfSystem<'a> {
    ctx: Ctx<'a>,
    c: Vec<f64>,
    z0: Vec<f64>,
}

impl<'a> HopfSystem<'a> {
    /// Builds the system near `(u, alpha)` from the eigenvalue `lambda` of A closest to
    /// `+i omega` (`lambda.1 > 0`). `c` is the real part of the unit eigenvector of
    /// `lambda`; the eigenvector is then scaled by a complex factor so that `cᵀq_r = 1`
    /// and `cᵀq_i = 0`.
    pub fn new(
        model: &'a mut dyn ModelEval,
        params: &[f64],
        free: usize,
        u: &[f64],
        alpha: f64,
        lambda: (f64, f64),
    ) -> Result<HopfSystem<'a>, EngineError> {
        let mut ctx = Ctx::new(model, params, free)?;
        let n = ctx.n;
        check_len(u, n)?;
        if lambda.1.is_nan() || lambda.1 <= 0.0 {
            return Err(EngineError::Singular(
                "no eigenvalue with positive imaginary part".into(),
            ));
        }
        ctx.set_alpha(alpha);
        let a = ctx.jac_u(u)?;
        let (re, im) = eigvec(&a, n, lambda)?;
        let c = re.clone();
        let s: f64 = c.iter().zip(&re).map(|(a, b)| a * b).sum();
        let t: f64 = c.iter().zip(&im).map(|(a, b)| a * b).sum();
        // (s + i t) is cᵀ(re + i im); divide the eigenvector by it.
        let d = s * s + t * t;
        let (fr, fi) = (s / d, -t / d);
        let mut z0 = u.to_vec();
        z0.push(alpha);
        z0.extend((0..n).map(|i| re[i] * fr - im[i] * fi));
        z0.extend((0..n).map(|i| re[i] * fi + im[i] * fr));
        z0.push(lambda.1);
        Ok(HopfSystem { ctx, c, z0 })
    }

    /// The starting point `(u, alpha, q_r, q_i, omega)`.
    pub fn start(&self) -> &[f64] {
        &self.z0
    }
}

impl DefiningSystem for HopfSystem<'_> {
    fn dim(&self) -> usize {
        3 * self.ctx.n + 2
    }

    fn residual(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        self.ctx.set_alpha(z[n]);
        let u = &z[..n];
        let qr = &z[n + 1..2 * n + 1];
        let qi = &z[2 * n + 1..3 * n + 1];
        let om = z[3 * n + 1];
        let f = self.ctx.f(u)?;
        let a = self.ctx.jac_u(u)?;
        out[..n].copy_from_slice(&f);
        for i in 0..n {
            let (mut r1, mut r2) = (om * qi[i], -om * qr[i]);
            for j in 0..n {
                r1 += a[i + n * j] * qr[j];
                r2 += a[i + n * j] * qi[j];
            }
            out[n + i] = r1;
            out[2 * n + i] = r2;
        }
        out[3 * n] = dot(&self.c, qr) - 1.0;
        out[3 * n + 1] = dot(&self.c, qi);
        Ok(())
    }

    fn jacobian(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        let dim = 3 * n + 2;
        self.ctx.set_alpha(z[n]);
        let u = &z[..n];
        let qr = &z[n + 1..2 * n + 1];
        let qi = &z[2 * n + 1..3 * n + 1];
        let om = z[3 * n + 1];
        let a = self.ctx.jac_u(u)?;
        let fa = self.ctx.f_alpha(u)?;
        out.fill(0.0);
        let at = |i: usize, j: usize| i + dim * j;
        for i in 0..n {
            for j in 0..n {
                let aij = a[i + n * j];
                out[at(i, j)] = aij;
                out[at(n + i, n + 1 + j)] = aij;
                out[at(2 * n + i, 2 * n + 1 + j)] = aij;
            }
            out[at(i, n)] = fa[i];
            out[at(n + i, 2 * n + 1 + i)] = om;
            out[at(2 * n + i, n + 1 + i)] = -om;
            out[at(n + i, 3 * n + 1)] = qi[i];
            out[at(2 * n + i, 3 * n + 1)] = -qr[i];
            out[at(3 * n, n + 1 + i)] = self.c[i];
            out[at(3 * n + 1, 2 * n + 1 + i)] = self.c[i];
        }
        for k in 0..=n {
            let dr = self.ctx.a_z_q(u, qr, k)?;
            let di = self.ctx.a_z_q(u, qi, k)?;
            for i in 0..n {
                out[at(n + i, k)] = dr[i];
                out[at(2 * n + i, k)] = di[i];
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// BP: branch point system

/// The branch point system: unknowns `z = (u, alpha, b, phi)` (2n + 2 values), equations
///
/// ```text
/// F(x) + b phi = 0,   F_xᵀ phi = 0,   phiᵀ phi - 1 = 0.
/// ```
///
/// `b` is the amount by which a perturbation `F = -b phi` would unfold the point, and is
/// 0 at a branch point. The start is `b = 0` with `phi` the left null vector of `F_x`.
/// The Jacobian block of `F_xᵀ phi` with respect to x has entries `phiᵀ F_xx[e_j, e_k]`.
///
/// Source: derived, not from a primary source. The proof is in
/// `.superpowers/sdd/phase3-pc-engine/sources.md`, item 5: at a simple branch point
/// (`dim ker F_x = 2` and the 2 x 2 matrix `psiᵀ F_xx[tau_i, tau_j]` nonsingular, with
/// the definition from Allgower and Georg, "Introduction to Numerical Continuation
/// Methods", Def. 8.1.11) the Jacobian is nonsingular exactly when that matrix is
/// nonsingular. The tests check the rank numerically and report the smallest singular
/// value.
pub struct BpSystem<'a> {
    ctx: Ctx<'a>,
    z0: Vec<f64>,
}

impl<'a> BpSystem<'a> {
    /// Builds the system with `phi` from the left null vector of `F_x` at `(u, alpha)`.
    pub fn new(
        model: &'a mut dyn ModelEval,
        params: &[f64],
        free: usize,
        u: &[f64],
        alpha: f64,
    ) -> Result<BpSystem<'a>, EngineError> {
        let mut ctx = Ctx::new(model, params, free)?;
        let n = ctx.n;
        check_len(u, n)?;
        ctx.set_alpha(alpha);
        let fx = ctx.f_x(u)?;
        let phi = first(null_left(&fx, n, n + 1, 1)?)?;
        let mut z0 = u.to_vec();
        z0.push(alpha);
        z0.push(0.0);
        z0.extend(phi);
        Ok(BpSystem { ctx, z0 })
    }

    /// The starting point `(u, alpha, b, phi)`.
    pub fn start(&self) -> &[f64] {
        &self.z0
    }
}

impl DefiningSystem for BpSystem<'_> {
    fn dim(&self) -> usize {
        2 * self.ctx.n + 2
    }

    fn residual(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        self.ctx.set_alpha(z[n]);
        let u = &z[..n];
        let b = z[n + 1];
        let phi = &z[n + 2..];
        let f = self.ctx.f(u)?;
        let fx = self.ctx.f_x(u)?;
        for i in 0..n {
            out[i] = f[i] + b * phi[i];
        }
        for k in 0..=n {
            out[n + k] = dot(&fx[k * n..(k + 1) * n], phi);
        }
        out[2 * n + 1] = dot(phi, phi) - 1.0;
        Ok(())
    }

    fn jacobian(&mut self, z: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.ctx.n;
        let dim = 2 * n + 2;
        self.ctx.set_alpha(z[n]);
        let u = &z[..n];
        let b = z[n + 1];
        let phi = &z[n + 2..];
        let fx = self.ctx.f_x(u)?;
        out.fill(0.0);
        let at = |i: usize, j: usize| i + dim * j;
        for k in 0..=n {
            for i in 0..n {
                let v = fx[i + n * k];
                out[at(i, k)] = v;
                out[at(n + k, n + 2 + i)] = v;
            }
        }
        for i in 0..n {
            out[at(i, n + 1)] = phi[i];
            out[at(i, n + 2 + i)] = b;
            out[at(2 * n + 1, n + 2 + i)] = 2.0 * phi[i];
        }
        for j in 0..=n {
            for k in j..=n {
                let v = dot(phi, &self.ctx.f_xx(u, j, k)?);
                out[at(n + j, k)] = v;
                out[at(n + k, j)] = v;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------
// Newton

/// One Newton iteration as logged: inf-norm of the residual before the step and inf-norm
/// of the step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewtonStep {
    pub residual: f64,
    pub step: f64,
}

/// Result of [`newton`].
#[derive(Clone, Debug)]
pub struct NewtonResult {
    pub z: Vec<f64>,
    pub log: Vec<NewtonStep>,
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.max(x.abs())
        }
    })
}

/// Newton's method on `sys` from `z0`. Converged when the step satisfies
/// `‖Δz‖∞ <= tol_refine * max(1, ‖z‖∞)` and the residual at the new point satisfies
/// `‖G‖∞ <= tol_f`. At most [`MAX_REFINE_ITERS`] iterations. The cancel flag is read
/// before each iteration and gives `Cancelled`.
pub fn newton(
    sys: &mut dyn DefiningSystem,
    z0: &[f64],
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<NewtonResult, EngineError> {
    let dim = sys.dim();
    if z0.len() != dim {
        return Err(EngineError::BadProblem(
            "refinement: start has the wrong length".into(),
        ));
    }
    let mut z = z0.to_vec();
    let mut g = vec![0.0; dim];
    let mut jac = vec![0.0; dim * dim];
    let mut log = Vec::new();
    let mut res = f64::INFINITY;
    for _ in 0..MAX_REFINE_ITERS {
        if cancel.load(Ordering::Relaxed) {
            return Err(EngineError::Cancelled);
        }
        sys.residual(&z, &mut g)?;
        let before = norm_inf(&g);
        sys.jacobian(&z, &mut jac)?;
        let lu = Lu::new(&jac, dim)?;
        for gi in g.iter_mut() {
            *gi = -*gi;
        }
        lu.solve(&mut g)?;
        for (zi, d) in z.iter_mut().zip(&g) {
            *zi += d;
        }
        let step = norm_inf(&g);
        log.push(NewtonStep {
            residual: before,
            step,
        });
        if !z.iter().all(|v| v.is_finite()) {
            return Err(EngineError::NotFinite("refinement iterate".into()));
        }
        if step <= opts.tol_refine * norm_inf(&z).max(1.0) {
            sys.residual(&z, &mut g)?;
            res = norm_inf(&g);
            if res <= opts.tol_f {
                return Ok(NewtonResult { z, log });
            }
        }
    }
    Err(EngineError::NewtonFailed {
        iters: MAX_REFINE_ITERS,
        residual: res,
    })
}

// ---------------------------------------------------------------------------------------
// Refining a special point

/// A refined special point.
#[derive(Clone, Debug)]
pub struct Refined {
    pub u: Vec<f64>,
    pub alpha: f64,
    /// Hopf frequency, for H only.
    pub omega: Option<f64>,
    /// Eigenvalues of A at the refined point.
    pub eigenvalues: Vec<(f64, f64)>,
    pub log: Vec<NewtonStep>,
}

/// Refines the special point `sp` with the system of its label. Returns `Ok(None)` for a
/// label without a system (NSA). Errors from the model or the linear algebra are
/// returned as they are; the caller treats all of them except `Cancelled` as a failed
/// refinement.
pub fn refine_special(
    model: &mut dyn ModelEval,
    params: &[f64],
    free: usize,
    sp: &SpecialPoint,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Option<Refined>, EngineError> {
    let n = model.n();
    let (u, alpha) = (&sp.point.u, sp.point.alpha);
    let (z, log, omega) = match sp.label.as_str() {
        "LP" => {
            let mut s = LpSystem::new(model, params, free, u, alpha)?;
            let z0 = s.start().to_vec();
            let r = newton(&mut s, &z0, opts, cancel)?;
            (r.z, r.log, None)
        }
        "BP" => {
            let mut s = BpSystem::new(model, params, free, u, alpha)?;
            let z0 = s.start().to_vec();
            let r = newton(&mut s, &z0, opts, cancel)?;
            (r.z, r.log, None)
        }
        "H" => {
            // The eigenvalue with positive imaginary part closest to the imaginary axis.
            let lambda = sp
                .point
                .eigenvalues
                .iter()
                .filter(|e| e.1 > 0.0)
                .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()))
                .copied()
                .ok_or_else(|| EngineError::Singular("no complex pair at the point".into()))?;
            let mut s = HopfSystem::new(model, params, free, u, alpha, lambda)?;
            let z0 = s.start().to_vec();
            let r = newton(&mut s, &z0, opts, cancel)?;
            let om = r.z[3 * n + 1];
            if om.is_nan() || om <= 0.0 {
                return Err(EngineError::Singular(
                    "refined frequency is not positive".into(),
                ));
            }
            (r.z, r.log, Some(om))
        }
        _ => return Ok(None),
    };
    let u_new = z[..n].to_vec();
    let alpha_new = z[n];
    let mut ctx = Ctx::new(model, params, free)?;
    ctx.set_alpha(alpha_new);
    let a = ctx.jac_u(&u_new)?;
    let eigenvalues = eigenvalues(&a, n)?;
    Ok(Some(Refined {
        u: u_new,
        alpha: alpha_new,
        omega,
        eigenvalues,
        log,
    }))
}

/// Refines every LP, BP and H in `branch.specials`. A refined point takes the Newton
/// `(u, alpha)` and the eigenvalues of A there, and gets `refined = true` (and `omega`
/// for H); its tangent and tests stay those of the located point. A failure keeps the
/// located point and pushes "could not refine <label> near point <index>" to
/// `branch.warnings`. `Cancelled` sets `branch.stop` to `Cancelled` and ends the loop.
pub fn refine_specials(
    model: &mut dyn ModelEval,
    params: &[f64],
    free: usize,
    opts: &Options,
    cancel: &AtomicBool,
    branch: &mut Branch,
) {
    for k in 0..branch.specials.len() {
        let sp = &branch.specials[k];
        match refine_special(model, params, free, sp, opts, cancel) {
            Ok(Some(r)) => {
                let sp = &mut branch.specials[k];
                sp.point.u = r.u;
                sp.point.alpha = r.alpha;
                sp.point.eigenvalues = r.eigenvalues;
                sp.extras.omega = r.omega;
                sp.extras.refined = true;
                // LP and H also get their normal form coefficient at the refined point.
                let sp = &branch.specials[k];
                match special_coefficient(model, params, free, sp) {
                    Ok(Some((c, norm))) => {
                        let ex = &mut branch.specials[k].extras;
                        ex.coefficient = Some(c);
                        ex.normalization = Some(norm.to_string());
                    }
                    Ok(None) => {}
                    Err(EngineError::Cancelled) => {
                        branch.stop = StopReason::Cancelled;
                        return;
                    }
                    Err(_) => {
                        let msg = format!(
                            "could not compute the <a|l1> coefficient near point {}",
                            sp.index
                        );
                        branch.warnings.push(msg);
                    }
                }
            }
            Ok(None) => {}
            Err(EngineError::Cancelled) => {
                branch.stop = StopReason::Cancelled;
                return;
            }
            Err(_) => {
                let msg = format!("could not refine {} near point {}", sp.label, sp.index);
                branch.warnings.push(msg);
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn check_len(u: &[f64], n: usize) -> Result<(), EngineError> {
    if u.len() == n {
        Ok(())
    } else {
        Err(EngineError::BadProblem(format!(
            "expected {n} state values, got {}",
            u.len()
        )))
    }
}

fn first(mut v: Vec<Vec<f64>>) -> Result<Vec<f64>, EngineError> {
    if v.is_empty() {
        Err(EngineError::Singular("no null vector".into()))
    } else {
        Ok(v.swap_remove(0))
    }
}
