//! Continuation of periodic orbits in one parameter (SPEC §7): the start from a Hopf point,
//! from a PD point or from a guessed orbit, the run on the collocation problem, and the
//! check of the trivial Floquet multiplier.
//!
//! The PALC loop measures arclength in scaled unknowns: the profile values are multiplied
//! by s = 1/sqrt(N m), T and alpha are not. The sum of squares of the scaled profile then
//! approximates the L2 norm of the orbit over [0, 1] (Doedel, Keller and Kernévez 1991),
//! so the step length does not grow with the mesh. The points the caller sees are in the
//! original unknowns.

use super::basis::CollocationBasis;
use super::mesh;
use super::problem::{CycleProblem, CycleSetup, check_dims, try_zeros};
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

/// A start from a PD point of a cycle branch, onto the branch of period about 2T.
#[derive(Clone, Debug)]
pub struct PdStart {
    /// Index of the continuation parameter among the model parameters.
    pub free: usize,
    /// All `n_p` parameter values at the PD point.
    pub params: Vec<f64>,
    /// The orbit at the PD point at the base points of `mesh`, point-major, as in
    /// `Point::u` without the period.
    pub orbit: Vec<f64>,
    /// The period T at the PD point.
    pub period: f64,
    /// The mesh of `orbit`, strictly increasing from 0 to 1.
    pub mesh: Vec<f64>,
    /// Number of mesh intervals N (NTST) of the new branch, at least 1.
    pub ntst: usize,
    /// Number of collocation points per interval m (NCOL) of the new branch, in
    /// [`NCOL_RANGE`](super::basis::NCOL_RANGE).
    pub ncol: usize,
}

impl PdStart {
    /// The start from `point`, a PD special point of a cycle branch in parameter `free`,
    /// with the other parameters from `params`.
    pub fn from_point(
        free: usize,
        params: &[f64],
        point: &Point,
        ntst: usize,
        ncol: usize,
    ) -> PdStart {
        let (orbit, period) = match point.u.split_last() {
            Some((t, orbit)) => (orbit.to_vec(), *t),
            None => (Vec::new(), f64::NAN),
        };
        let mut params = params.to_vec();
        if let Some(a) = params.get_mut(free) {
            *a = point.alpha;
        }
        PdStart {
            free,
            params,
            orbit,
            period,
            mesh: point.mesh.clone(),
            ntst,
            ncol,
        }
    }
}

/// Where a cycle run starts.
#[derive(Clone, Debug)]
pub enum CycleStart {
    /// A guessed orbit, corrected at the fixed parameter value `params[free]` first.
    Orbit(CycleSetup),
    /// A Hopf point, with the first orbit taken from the eigenvector of the Hopf pair.
    Hopf(HopfStart),
    /// A PD point, with the first orbit taken from the orbit there run twice and the
    /// eigenfunction of the multiplier -1.
    Pd(PdStart),
}

/// Largest distance of the trivial Floquet multiplier from 1 before a point gets a
/// warning (SPEC §7).
const TRIVIAL_TOL: f64 = 1e-6;

/// Largest distance of the multiplier closest to -1 from -1 at a PD start before the run
/// gets a warning.
const PD_TOL: f64 = 1e-6;

/// Number of times a start from a Hopf or PD point halves the amplitude after a failed
/// correction.
const RETRIES: usize = 4;

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
/// A PD start runs the orbit at the PD point twice, on NTST intervals of the new mesh, and
/// puts the first orbit at amplitude `Options::h0` along the eigenfunction of the
/// multiplier -1 with the same halving. Both directions there give the same orbits, half
/// a period apart, so `Options::direction` is ignored as at a Hopf point, and the run gets
/// a warning when the multiplier closest to -1 is more than 1e-6 from it. A run in which
/// some point has no multiplier within 1e-6 of 1 gets a warning, because the mesh is then
/// too coarse for the multipliers to be trusted.
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
            let mut p = Scaled::cycle(CycleProblem::new(model, setup)?);
            let mut x = setup.profile.clone();
            x.push(setup.period);
            x.push(setup.params[setup.free]);
            p.scale(&mut x);
            let x0 = fixed_alpha(&mut p, x, opts, cancel)?;
            (p, x0, None, opts.clone())
        }
        CycleStart::Hopf(h) => {
            let (p, x0, t0) = hopf_start(model, h, opts, cancel, &mut early)?;
            let what = "a Hopf point; the branch leaves it with growing amplitude";
            (p, x0, Some(t0), switched(opts, what, &mut early))
        }
        CycleStart::Pd(s) => {
            let (p, x0, t0) = pd_start(model, s, opts, cancel, &mut early)?;
            let what = "a PD point; the two directions give the same orbits, half a period apart";
            (p, x0, Some(t0), switched(opts, what, &mut early))
        }
    };
    p.inner.adapt_every = opts.mesh_adapt;
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
    floquet_warning(&mut branch, 1);
    Ok(branch)
}

/// Corrects a guessed orbit (scaled unknowns) at fixed alpha.
pub(crate) fn fixed_alpha<P: Problem>(
    p: &mut Scaled<P>,
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
) -> Result<Start<CycleProblem<'a>>, EngineError> {
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
    // The sizes come from NTST and NCOL, so these allocations fail with an error instead
    // of aborting, until CycleProblem::new has checked the largest one.
    let nm = len / n - 1;
    let mut phi = try_zeros(Some(len))?;
    for (kk, row) in phi.chunks_exact_mut(n).enumerate() {
        let th = 2.0 * PI * kk as f64 / nm as f64;
        let (sn, cs) = th.sin_cos();
        for (c, v) in row.iter_mut().enumerate() {
            *v = qr[c] * cs - qi[c] * sn;
        }
    }
    let mut base = try_zeros(Some(len))?;
    for row in base.chunks_exact_mut(n) {
        row.copy_from_slice(&h.u0);
    }
    let period = 2.0 * PI / omega;
    let guess = |eps: f64| -> Result<Vec<f64>, EngineError> {
        let mut g = try_zeros(Some(len))?;
        for ((g, u), f) in g.iter_mut().zip(&base).zip(&phi) {
            *g = u + eps * f;
        }
        Ok(g)
    };
    let setup = CycleSetup::new(
        h.free,
        h.params.clone(),
        h.ntst,
        h.ncol,
        guess(opts.h0)?,
        period,
    );
    let p = Scaled::cycle(CycleProblem::new(model, &setup)?);
    let mut dir = phi.clone();
    dir.extend([0.0, 0.0]);
    let dir = unit_direction(&p, dir, "Hopf eigenvector")?;
    let mut x0 = base.clone();
    x0.push(period);
    x0.push(h.params[h.free]);
    p.scale(&mut x0);
    branch_off(p, &x0, &dir, opts, cancel, |q, eps| {
        q.set_reference(&guess(eps)?);
        Ok(())
    })
}

/// The first orbit on the period-doubled branch and the tangent there, in scaled unknowns.
///
/// At the PD point the monodromy matrix has the multiplier -1, and the linearized flow has
/// a solution psi with psi(1) = -psi(0). Run twice, the orbit is a cycle of period 2T on
/// which the doubled branch crosses in a pitchfork, along psi on the first half and -psi
/// on the second. That direction is periodic and orthogonal to the phase condition of the
/// orbit run twice. Both are doubled onto [0, 1] and interpolated onto NTST intervals of
/// NCOL points, whose mesh points are evenly spaced in the index of the doubled mesh. The
/// correction then keeps the component along the direction at eps, as at a Hopf point.
fn pd_start<'a>(
    model: &'a mut dyn ModelEval,
    s: &PdStart,
    opts: &Options,
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> Result<Start<CycleProblem<'a>>, EngineError> {
    let n = model.n();
    let bad = |m: String| Err(EngineError::BadProblem(m));
    check_dims(model, s.free, &s.params, s.ntst, s.ncol)?;
    let nint = s.mesh.len().saturating_sub(1);
    if nint == 0
        || s.mesh[0] != 0.0
        || s.mesh[nint] != 1.0
        || !s.mesh.windows(2).all(|w| w[1] > w[0])
    {
        return bad("the mesh of a PD start must rise strictly from 0 to 1".into());
    }
    let pts = s.orbit.len() / n;
    if !s.orbit.len().is_multiple_of(n) || pts < 2 || !(pts - 1).is_multiple_of(nint) {
        return bad(format!(
            "expected the orbit at N m + 1 points of {n} values on {nint} mesh intervals, got \
             {} values",
            s.orbit.len()
        ));
    }
    if s.params.iter().any(|v| !v.is_finite()) {
        return bad("the PD point has NaN or infinite parameters".into());
    }
    let old_m = (pts - 1) / nint;
    let (psi, mu) = {
        let setup = CycleSetup::new(
            s.free,
            s.params.clone(),
            nint,
            old_m,
            s.orbit.clone(),
            s.period,
        );
        let mut src = CycleProblem::new(&mut *model, &setup)?;
        src.set_mesh(s.mesh.clone(), &s.orbit);
        let mut x = s.orbit.clone();
        x.extend([s.period, s.params[s.free]]);
        src.pd_eigenfunction(&x)?
    };
    if (mu + 1.0).abs() > PD_TOL {
        warnings.push(format!(
            "the start is not a PD point to {PD_TOL:e}: the multiplier closest to -1 is {mu}"
        ));
    }
    let mut doubled: Vec<f64> = s.mesh.iter().map(|t| t / 2.0).collect();
    doubled.extend(s.mesh[1..].iter().map(|t| 0.5 + t / 2.0));
    let mut orbit = s.orbit.clone();
    orbit.extend_from_slice(&s.orbit[n..]);
    let mut psi2 = psi.clone();
    psi2.extend(psi[n..].iter().map(|v| -v));
    // As at a Hopf point, the sizes from NTST and NCOL fail with an error, not an abort.
    let d = 2 * nint;
    let mut new = try_zeros(s.ntst.checked_add(1))?;
    for (i, t) in new.iter_mut().enumerate() {
        let r = i as f64 * d as f64 / s.ntst as f64;
        let j = (r.floor() as usize).min(d - 1);
        *t = doubled[j] + (r - j as f64) * (doubled[j + 1] - doubled[j]);
    }
    new[0] = 0.0;
    new[s.ntst] = 1.0;
    let basis = CollocationBasis::new(old_m)?;
    let u = mesh::interpolate(&basis, n, &doubled, &new, s.ncol, &orbit)?;
    let mut dir = mesh::interpolate(&basis, n, &doubled, &new, s.ncol, &psi2)?;
    let setup = CycleSetup::new(s.free, s.params.clone(), s.ntst, s.ncol, u, 2.0 * s.period);
    let mut inner = CycleProblem::new(model, &setup)?;
    // Moved back out, not cloned, so no copy is made before the check in new.
    let u = setup.profile;
    inner.set_mesh(new, &u);
    let p = Scaled::cycle(inner);
    dir.extend([0.0, 0.0]);
    let dir = unit_direction(&p, dir, "PD eigenfunction")?;
    let mut x0 = u;
    x0.extend([2.0 * s.period, s.params[s.free]]);
    p.scale(&mut x0);
    branch_off(p, &x0, &dir, opts, cancel, |_, _| Ok(()))
}

/// `d`, in the original unknowns, as a unit vector in the scaled unknowns of `p`.
pub(crate) fn unit_direction<P>(
    p: &Scaled<P>,
    mut d: Vec<f64>,
    what: &str,
) -> Result<Vec<f64>, EngineError> {
    p.scale(&mut d);
    let nrm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(nrm.is_finite() && nrm > 0.0) {
        return Err(EngineError::Singular(format!("the {what} is zero")));
    }
    for v in d.iter_mut() {
        *v /= nrm;
    }
    Ok(d)
}

/// Corrects the first orbit at x0 + eps dir, keeping its component along the unit vector
/// `dir` at eps, with eps = `Options::h0` halved up to [`RETRIES`] times when the
/// correction fails. `retry` gets the problem and the new eps before each new attempt.
/// Returns the problem, the orbit and the tangent there, bordered by `dir`.
fn branch_off<'a>(
    mut p: Scaled<CycleProblem<'a>>,
    x0: &[f64],
    dir: &[f64],
    opts: &Options,
    cancel: &AtomicBool,
    mut retry: impl FnMut(&mut CycleProblem<'a>, f64) -> Result<(), EngineError>,
) -> Result<Start<CycleProblem<'a>>, EngineError> {
    let mut eps = opts.h0;
    for attempt in 0..=RETRIES {
        if attempt > 0 {
            eps /= 2.0;
            retry(&mut p.inner, eps)?;
        }
        let xg: Vec<f64> = x0.iter().zip(dir).map(|(x, d)| x + eps * d).collect();
        match correct(&mut p, &xg, dir, 0.0, xg.clone(), opts, cancel) {
            Corrected::Done { x, .. } => {
                let (t, _) = tangent_or_kernel(&mut p, &x, dir)?;
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

/// The options of a run from a point the branch leaves one way only, with a warning when
/// `Options::direction` asks for the other.
pub(crate) fn switched(opts: &Options, what: &str, warnings: &mut Vec<String>) -> Options {
    if opts.direction < 0.0 {
        warnings.push(format!("the direction is ignored for a start from {what}"));
    }
    Options {
        direction: 1.0,
        ..opts.clone()
    }
}

/// The error behind a stop of the corrector.
pub(crate) fn stop_error(r: StopReason) -> EngineError {
    match r {
        StopReason::Cancelled => EngineError::Cancelled,
        StopReason::Error(e) => e,
        StopReason::Singular => EngineError::Singular("the bordered matrix at the start".into()),
        other => EngineError::BadProblem(format!("the start stopped with {other:?}")),
    }
}

/// Undoes the scaling of the profile in a point: the orbit values and the tangent, which
/// is normalized again to unit length.
pub(crate) fn unscale_point(pt: &mut Point, k: usize, s: f64) {
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
/// of 1. On an LPC curve `trivial` is 2: the multiplier 1 is double there, with a Jordan
/// block, so an error e in the monodromy matrix moves the two by about sqrt(e), and their
/// mean, which moves by about e, is checked instead.
pub(crate) fn floquet_warning(branch: &mut Branch, trivial: usize) {
    let dist = |m: &(f64, f64)| (m.0 - 1.0).hypot(m.1);
    let (mut count, mut total, mut worst) = (0, 0, 0.0_f64);
    for pt in &branch.points {
        if pt.eigenvalues.is_empty() {
            continue;
        }
        total += 1;
        let d = if trivial < 2 {
            pt.eigenvalues
                .iter()
                .map(dist)
                .fold(f64::INFINITY, f64::min)
        } else {
            let mut near: Vec<&(f64, f64)> = pt.eigenvalues.iter().collect();
            near.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
            match near[..] {
                [a, b, ..] => ((a.0 + b.0) / 2.0 - 1.0).hypot((a.1 + b.1) / 2.0),
                _ => f64::NAN,
            }
        };
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
        let what = if trivial < 2 {
            "trivial Floquet multiplier"
        } else {
            "double trivial Floquet multiplier (by the mean of the two nearest 1)"
        };
        branch.warnings.push(format!(
            "{count} of {total} points have no {what} within {TRIVIAL_TOL:e} of 1 (largest \
             distance {worst:.3e}); the mesh is too coarse, increase NTST or NCOL"
        ));
    }
}

/// The problem of a run, its first point and the tangent there, in scaled unknowns.
pub(crate) type Start<P> = (Scaled<P>, Vec<f64>, Vec<f64>);

/// A cycle problem in scaled unknowns: the first `k` values (the profile) times `s`, the
/// rest as they are.
pub(crate) struct Scaled<P> {
    pub(crate) inner: P,
    /// Number of profile values.
    pub(crate) k: usize,
    pub(crate) s: f64,
    /// x in the original unknowns.
    xp: Vec<f64>,
    border: Vec<f64>,
}

impl<'a> Scaled<CycleProblem<'a>> {
    /// The scaled P5 cycle problem.
    pub(crate) fn cycle(inner: CycleProblem<'a>) -> Scaled<CycleProblem<'a>> {
        let (k, points) = (inner.profile_len(), inner.points());
        Scaled::new(inner, k, points)
    }
}

impl<P> Scaled<P> {
    /// `inner` with its first `k` unknowns, the profile at `points` points, scaled by
    /// 1 / sqrt(points - 1).
    pub(crate) fn new(inner: P, k: usize, points: usize) -> Scaled<P>
    where
        P: Problem,
    {
        let dim = inner.dim();
        Scaled {
            inner,
            k,
            s: 1.0 / ((points - 1) as f64).sqrt(),
            xp: vec![0.0; dim],
            border: vec![0.0; dim],
        }
    }

    /// Original unknowns to scaled ones, in place.
    pub(crate) fn scale(&self, x: &mut [f64]) {
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

impl<P: Problem> Problem for Scaled<P> {
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

    fn mesh(&self) -> Vec<f64> {
        self.inner.mesh()
    }

    /// The monitor sees the original unknowns. Interpolation is linear, so the scaled
    /// tangent is interpolated as it is and then normalized again.
    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        self.load(x);
        if !self.inner.adapt(&mut self.xp, v, failed) {
            return false;
        }
        x.copy_from_slice(&self.xp);
        self.scale(x);
        let nrm = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        for t in v.iter_mut() {
            *t /= nrm;
        }
        true
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

#[cfg(test)]
mod tests {
    use super::*;

    fn branch_of(mults: &[&[(f64, f64)]]) -> Branch {
        let points = mults
            .iter()
            .map(|m| Point {
                u: vec![0.0],
                alpha: 0.0,
                tangent: Vec::new(),
                eigenvalues: m.to_vec(),
                tests: [f64::NAN; 3],
                codim2: Vec::new(),
                h: 0.1,
                newton_iters: 2,
                mesh: Vec::new(),
            })
            .collect();
        Branch {
            points,
            specials: Vec::new(),
            stop: StopReason::MaxSteps,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_split_double_multiplier_passes_on_its_mean() {
        // 1 +- 2e-3 is what an error of 4e-6 does to a Jordan block at 1.
        let split: &[(f64, f64)] = &[(1.002, 0.0), (0.998, 0.0), (0.3, 0.0)];
        let pair: &[(f64, f64)] = &[(1.0, 2e-3), (1.0, -2e-3), (0.3, 0.0)];
        let mut b = branch_of(&[split, pair]);
        floquet_warning(&mut b, 2);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
        let mut b = branch_of(&[split, pair]);
        floquet_warning(&mut b, 1);
        assert_eq!(b.warnings.len(), 1);
        assert!(b.warnings[0].contains("2 of 2 points"), "{:?}", b.warnings);
    }

    #[test]
    fn a_shifted_double_multiplier_gets_a_warning() {
        let shifted: &[(f64, f64)] = &[(1.003, 0.0), (0.999, 0.0), (0.3, 0.0)];
        let good: &[(f64, f64)] = &[(1.0 + 1e-7, 0.0), (1.0 - 1e-7, 0.0), (0.3, 0.0)];
        let single: &[(f64, f64)] = &[(1.0, 0.0)];
        let mut b = branch_of(&[shifted, good]);
        floquet_warning(&mut b, 2);
        assert_eq!(b.warnings.len(), 1);
        let w = &b.warnings[0];
        assert!(w.contains("1 of 2 points") && w.contains("1.000e-3"), "{w}");
        assert!(w.contains("double"), "{w}");
        // A point without two multipliers cannot have a double one.
        let mut b = branch_of(&[single]);
        floquet_warning(&mut b, 2);
        assert!(b.warnings[0].contains("1 of 1 points") && b.warnings[0].contains("NaN"));
    }
}
