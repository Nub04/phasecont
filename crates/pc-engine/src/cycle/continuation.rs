//! Continuation of periodic orbits in one parameter (SPEC §7): the start from a Hopf point
//! or from a guessed orbit, the run on the collocation problem, and the check of the
//! trivial Floquet multiplier.
//!
//! The PALC loop measures arclength in scaled unknowns: the profile values are multiplied
//! by s = 1/sqrt(N m), T and alpha are not. The sum of squares of the scaled profile then
//! approximates the L2 norm of the orbit over [0, 1] (Doedel, Keller and Kernévez 1991),
//! so the step length does not grow with the mesh. The points the caller sees are in the
//! original unknowns.

use super::problem::{CycleProblem, CycleSetup, check_dims};
use crate::branch::{Branch, Options, Point, StopReason};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::palc::{Corrected, correct, run_branch, tangent_or_kernel};
use crate::problem::Problem;
use pc_ir::EntryKind;
use pc_linalg::{eigenvalues, eigvec};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

/// A start from a Hopf point of an equilibrium branch.
#[derive(Clone, Debug)]
pub struct HopfStart {
    /// Index of the continuation parameter among the model parameters.
    pub free: usize,
    /// All `n_p` parameter values at the Hopf point.
    pub params: Vec<f64>,
    /// The equilibrium at the Hopf point.
    pub u0: Vec<f64>,
    /// Number of mesh intervals N (NTST), at least 1.
    pub ntst: usize,
    /// Number of collocation points per interval m (NCOL), in
    /// [`NCOL_RANGE`](super::basis::NCOL_RANGE).
    pub ncol: usize,
}

/// Where a cycle run starts.
#[derive(Clone, Debug)]
pub enum CycleStart {
    /// A guessed orbit, corrected at the fixed parameter value `params[free]` first.
    Orbit(CycleSetup),
    /// A Hopf point, with the first orbit taken from the eigenvector of the Hopf pair.
    Hopf(HopfStart),
}

/// Largest distance of the trivial Floquet multiplier from 1 before a point gets a
/// warning (SPEC §7).
const TRIVIAL_TOL: f64 = 1e-6;

/// Number of times the Hopf start halves the amplitude after a failed correction.
const HOPF_RETRIES: usize = 4;

/// Follows a branch of periodic orbits from `start`. `Point::u` holds the orbit at the
/// N m + 1 base points, point-major, then the period T; `Point::alpha` is the parameter
/// and `Point::eigenvalues` the Floquet multipliers. `Point::codim2` holds the LPC, PD
/// and NS tests of [`CYCLE_LABELS`](super::problem::CYCLE_LABELS); their zeros are listed
/// in `Branch::specials`.
///
/// An orbit start honors `Options::direction` like an equilibrium run. A Hopf start puts
/// the first orbit at amplitude `Options::h0` along the Hopf eigenvector, halving it up to
/// four times when the correction fails, and leaves the Hopf point with growing
/// amplitude: `Options::direction` is ignored there, with a warning when it is negative.
/// A run in which some point has no multiplier within 1e-6 of 1 gets a warning, because
/// the mesh is then too coarse for the multipliers to be trusted.
pub fn continue_cycle(
    model: &mut dyn ModelEval,
    start: &CycleStart,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let mut early = Vec::new();
    let (mut p, x0, tangent, run_opts) = match start {
        CycleStart::Orbit(setup) => {
            let mut p = Scaled::new(CycleProblem::new(model, setup)?);
            let mut x = setup.profile.clone();
            x.push(setup.period);
            x.push(setup.params[setup.free]);
            p.scale(&mut x);
            let x0 = fixed_alpha(&mut p, x, opts, cancel)?;
            (p, x0, None, opts.clone())
        }
        CycleStart::Hopf(h) => {
            let (p, x0, t0) = hopf_start(model, h, opts, cancel, &mut early)?;
            if opts.direction < 0.0 {
                early.push(
                    "the direction is ignored for a start from a Hopf point; the branch \
                     leaves it with growing amplitude"
                        .into(),
                );
            }
            let o = Options {
                direction: 1.0,
                ..opts.clone()
            };
            (p, x0, Some(t0), o)
        }
    };
    let (k, s) = (p.k, p.s);
    let mut seen = |pt: &Point| {
        let mut q = pt.clone();
        unscale_point(&mut q, k, s);
        on_point(&q);
    };
    let mut branch = run_branch(
        &mut p,
        &x0,
        tangent.as_deref(),
        &run_opts,
        false,
        cancel,
        &mut seen,
    )?;
    for pt in branch.points.iter_mut() {
        unscale_point(pt, k, s);
    }
    for sp in branch.specials.iter_mut() {
        unscale_point(&mut sp.point, k, s);
    }
    early.append(&mut branch.warnings);
    branch.warnings = early;
    floquet_warning(&mut branch);
    Ok(branch)
}

/// Corrects a guessed orbit (scaled unknowns) at fixed alpha.
fn fixed_alpha(
    p: &mut Scaled,
    x: Vec<f64>,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<Vec<f64>, EngineError> {
    let dim = x.len();
    let mut fix = vec![0.0; dim];
    fix[dim - 1] = 1.0;
    match correct(p, &x, &fix, 0.0, x.clone(), opts, cancel) {
        Corrected::Done { x, .. } => Ok(x),
        Corrected::Failed => Err(EngineError::NewtonFailed {
            iters: opts.max_newton,
            residual: f64::NAN,
        }),
        Corrected::Stop(r) => Err(stop_error(r)),
    }
}

/// The first orbit next to a Hopf point and the tangent there, in scaled unknowns.
///
/// At the Hopf point A q = i omega q, and the linearized flow has the solutions
/// Re(c q e^{i omega t}). With T = 2 pi / omega and t = T tau, the first orbit is
/// u(tau) = u0 + eps phi(tau), phi = Re(q e^{2 pi i tau}) = q_re cos 2 pi tau - q_im sin
/// 2 pi tau, at the period 2 pi / omega and the Hopf parameter value. The correction keeps
/// the component along phi fixed at eps (the hyperplane through the guess orthogonal to
/// phi), and the tangent is bordered by phi, so the run leaves with growing amplitude.
fn hopf_start<'a>(
    model: &'a mut dyn ModelEval,
    h: &HopfStart,
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> Result<(Scaled<'a>, Vec<f64>, Vec<f64>), EngineError> {
    let n = model.n();
    let len = check_dims(model, h.free, &h.params, h.ntst, h.ncol)?;
    if h.u0.len() != n {
        return Err(EngineError::BadProblem(format!(
            "expected {n} values of the equilibrium, got {}",
            h.u0.len()
        )));
    }
    if h.u0.iter().chain(&h.params).any(|v| !v.is_finite()) {
        return Err(EngineError::BadProblem(
            "the Hopf point has NaN or infinite values".into(),
        ));
    }
    let mut pv = h.params.clone();
    pv.push(0.0);
    let mut a = vec![0.0; n * n];
    model.eval(&EntryKind::JacU, &h.u0, &pv, &[], &mut a)?;
    let eigs = eigenvalues(&a, n)?;
    let Some(&(re, omega)) = eigs
        .iter()
        .filter(|e| e.1 > 0.0)
        .min_by(|x, y| x.0.abs().total_cmp(&y.0.abs()))
    else {
        return Err(EngineError::BadProblem(
            "the Jacobian at the start has no complex pair of eigenvalues; start from a \
             Hopf point"
                .into(),
        ));
    };
    if re.abs() > 1e-6 * omega.max(1.0) {
        warnings.push(format!(
            "the start is not a Hopf point to 1e-6: the closest pair is {re:e} +- {omega} i"
        ));
    }
    let (qr, qi) = eigvec(&a, n, (re, omega))?;
    let nm = len / n - 1;
    let mut phi = Vec::with_capacity(len);
    for kk in 0..=nm {
        let th = 2.0 * PI * kk as f64 / nm as f64;
        let (sn, cs) = th.sin_cos();
        phi.extend((0..n).map(|c| qr[c] * cs - qi[c] * sn));
    }
    let base: Vec<f64> = (0..=nm).flat_map(|_| h.u0.iter().copied()).collect();
    let period = 2.0 * PI / omega;
    let guess =
        |eps: f64| -> Vec<f64> { base.iter().zip(&phi).map(|(u, f)| u + eps * f).collect() };
    let setup = CycleSetup::new(
        h.free,
        h.params.clone(),
        h.ntst,
        h.ncol,
        guess(opts.h0),
        period,
    );
    let mut p = Scaled::new(CycleProblem::new(model, &setup)?);
    // The direction phi in scaled unknowns, as a unit vector.
    let mut dir = phi.clone();
    dir.extend([0.0, 0.0]);
    p.scale(&mut dir);
    let nrm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(nrm.is_finite() && nrm > 0.0) {
        return Err(EngineError::Singular("the Hopf eigenvector is zero".into()));
    }
    for v in dir.iter_mut() {
        *v /= nrm;
    }
    let mut x0 = base.clone();
    x0.push(period);
    x0.push(h.params[h.free]);
    p.scale(&mut x0);
    let mut eps = opts.h0;
    for attempt in 0..=HOPF_RETRIES {
        if attempt > 0 {
            eps /= 2.0;
            p.inner.set_reference(&guess(eps));
        }
        let xg: Vec<f64> = x0.iter().zip(&dir).map(|(x, d)| x + eps * d).collect();
        match correct(&mut p, &xg, &dir, 0.0, xg.clone(), opts, cancel) {
            Corrected::Done { x, .. } => {
                let (t, _) = tangent_or_kernel(&mut p, &x, &dir)?;
                return Ok((p, x, t));
            }
            Corrected::Failed => {}
            Corrected::Stop(r) => return Err(stop_error(r)),
        }
    }
    Err(EngineError::NewtonFailed {
        iters: opts.max_newton,
        residual: f64::NAN,
    })
}

/// The error behind a stop of the corrector.
fn stop_error(r: StopReason) -> EngineError {
    match r {
        StopReason::Cancelled => EngineError::Cancelled,
        StopReason::Error(e) => e,
        StopReason::Singular => EngineError::Singular("the bordered matrix at the start".into()),
        other => EngineError::BadProblem(format!("the start stopped with {other:?}")),
    }
}

/// Undoes the scaling of the profile in a point: the orbit values and the tangent, which
/// is normalized again to unit length.
fn unscale_point(pt: &mut Point, k: usize, s: f64) {
    for v in pt.u[..k].iter_mut() {
        *v /= s;
    }
    if pt.tangent.len() > k {
        for v in pt.tangent[..k].iter_mut() {
            *v /= s;
        }
        let nrm = pt.tangent.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm.is_finite() && nrm > 0.0 {
            for v in pt.tangent.iter_mut() {
                *v /= nrm;
            }
        }
    }
}

/// Adds one warning for the points whose multipliers have none within [`TRIVIAL_TOL`]
/// of 1.
fn floquet_warning(branch: &mut Branch) {
    let (mut count, mut total, mut worst) = (0, 0, 0.0_f64);
    for pt in &branch.points {
        if pt.eigenvalues.is_empty() {
            continue;
        }
        total += 1;
        let d = pt
            .eigenvalues
            .iter()
            .map(|m| (m.0 - 1.0).hypot(m.1))
            .fold(f64::INFINITY, f64::min);
        if d.is_nan() || d > TRIVIAL_TOL {
            count += 1;
            worst = if worst.is_nan() || d.is_nan() {
                f64::NAN
            } else {
                worst.max(d)
            };
        }
    }
    if count > 0 {
        branch.warnings.push(format!(
            "{count} of {total} points have no trivial Floquet multiplier within {TRIVIAL_TOL:e} \
             of 1 (largest distance {worst:.3e}); the mesh is too coarse, increase NTST or NCOL"
        ));
    }
}

/// The collocation problem in scaled unknowns: profile values times `s`, T and alpha as
/// they are.
struct Scaled<'a> {
    inner: CycleProblem<'a>,
    /// Number of profile values.
    k: usize,
    s: f64,
    /// x in the original unknowns.
    xp: Vec<f64>,
    border: Vec<f64>,
}

impl<'a> Scaled<'a> {
    fn new(inner: CycleProblem<'a>) -> Scaled<'a> {
        let dim = inner.dim();
        let k = dim - 2;
        let points = inner.points();
        Scaled {
            inner,
            k,
            s: 1.0 / ((points - 1) as f64).sqrt(),
            xp: vec![0.0; dim],
            border: vec![0.0; dim],
        }
    }

    /// Original unknowns to scaled ones, in place.
    fn scale(&self, x: &mut [f64]) {
        for v in x[..self.k].iter_mut() {
            *v *= self.s;
        }
    }

    /// Scaled `x` to the original unknowns in `self.xp`.
    fn load(&mut self, x: &[f64]) {
        let (k, s) = (self.k, self.s);
        for (i, (o, v)) in self.xp.iter_mut().zip(x).enumerate() {
            *o = if i < k { v / s } else { *v };
        }
    }
}

impl Problem for Scaled<'_> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.load(x);
        self.inner.residual(&self.xp, out)
    }

    /// The dense Jacobian in scaled unknowns is J S^-1: profile columns divided by s.
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.load(x);
        self.inner.jacobian(&self.xp, out)?;
        if !out.is_empty() {
            let rows = self.dim() - 1;
            for v in out[..rows * self.k].iter_mut() {
                *v /= self.s;
            }
        }
        Ok(())
    }

    fn jacobian_len(&self) -> usize {
        0
    }

    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        self.load(x);
        self.inner.state_jacobian(&self.xp)
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        self.inner.codim2_labels()
    }

    fn codim2_tests(
        &mut self,
        x: &[f64],
        tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        self.load(x);
        self.inner.codim2_tests(&self.xp, tangent, eigs)
    }

    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        self.inner.codim2_label(i, eigs)
    }

    fn accept(&mut self, x: &[f64]) {
        self.load(x);
        self.inner.accept(&self.xp);
    }

    /// With y = S z, [J S^-1; bᵀ] y = r is [J; (S b)ᵀ] z = r.
    fn bordered_solve(
        &mut self,
        jac: &[f64],
        border: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
    ) -> Result<f64, EngineError> {
        if border.len() != self.border.len() || sol.len() != self.border.len() {
            return Err(EngineError::BadProblem(
                "bordered_solve: wrong lengths".into(),
            ));
        }
        let (k, s) = (self.k, self.s);
        for (i, (o, b)) in self.border.iter_mut().zip(border).enumerate() {
            *o = if i < k { s * b } else { *b };
        }
        let det = self.inner.bordered_solve(jac, &self.border, rhs, sol)?;
        self.scale(sol);
        Ok(det)
    }
}
