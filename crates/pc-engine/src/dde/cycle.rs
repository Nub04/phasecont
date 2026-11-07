//! Periodic orbits of a delay equation by collocation (SPEC section 8l, slice 1).
//!
//! The orbit is `u(s)`, `s = t / T` in `[0, 1]`, on a mesh of `N` intervals with `m` Gauss
//! points each, as in the ODE cycles of SPEC section 7. The collocation equations are
//!
//! ```text
//! du/ds(c_ji) = T f(u(c_ji), u(s_1'), ..., u(s_S'); alpha),   s_k' = frac(c_ji - tau_k / T)
//! ```
//!
//! with `u(1) = u(0)` and an integral phase condition. A delayed argument lands in another
//! interval, so the Jacobian is dense. The formulation follows the DDE-BIFTOOL manual,
//! section 3.1.1 (arXiv 1406.7144); the code is written from the equations and shares no
//! source with DDE-BIFTOOL.
//!
//! Unknowns: `x = [u at the N m + 1 base points, T, alpha]`, `dim = n (N m + 1) + 2`.
//! Equations: `N m n` collocation rows, `n` periodicity rows and one phase row, `dim - 1`
//! in all, so the solution set is a curve in `(u, T, alpha)`.

use super::eq::{check_params, time_of};
use super::lyap::pair;
use crate::branch::{Options, StopReason};
use crate::cycle::{CollocationBasis, NCOL_RANGE, phase_row, try_zeros};
use crate::error::EngineError;
use crate::model::DdeModel;
use crate::palc::{Corrected, check_step_options, correct, run_branch};
use crate::problem::Problem;
use pc_ir::{DelayTime, EntryKind};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest number of unknowns. The Jacobian is dense, so this bounds its memory at 72 MB.
const MAX_DIM: usize = 3000;

/// What stays fixed along a run: the declared parameters (the value at `free` is only the
/// starting value), which one is free, and the mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycleSetup {
    /// The declared parameters, in declaration order.
    pub params: Vec<f64>,
    /// Index into `params` of the free parameter alpha.
    pub free: usize,
    /// Mesh intervals N, at least 1.
    pub ntst: usize,
    /// Collocation points per interval m, in `NCOL_RANGE`.
    pub ncol: usize,
}

/// A starting orbit for [`correct_cycle`].
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycleStart {
    /// `u` at the `N m + 1` base points, `n` values per point.
    pub profile: Vec<f64>,
    /// The period T, finite and positive.
    pub period: f64,
}

/// A Hopf point of the delay equation to start from.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeHopfCycle {
    /// The equilibrium at the Hopf point, `n` values.
    pub x: Vec<f64>,
    /// The Hopf frequency, finite and positive.
    pub omega: f64,
    /// The amplitude of the first guess, finite and positive. There is no default.
    pub eps: f64,
}

/// A periodic orbit that Newton's method converged to.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycle {
    /// `u` at the base points, `n` values per point.
    pub profile: Vec<f64>,
    pub period: f64,
    /// The value of the free parameter.
    pub alpha: f64,
    /// Newton iterations.
    pub iters: usize,
    /// The inf-norm of the `dim - 1` equations at the result.
    pub residual: f64,
}

/// Checks the setup against the model and returns `n` and `dim`.
pub(super) fn check_setup(m: &DdeModel, s: &DdeCycleSetup) -> Result<(usize, usize), EngineError> {
    check_params(m, &s.params)?;
    if s.params.iter().any(|p| !p.is_finite()) {
        return Err(EngineError::BadProblem(
            "every parameter must be finite".into(),
        ));
    }
    if s.free >= m.n_user_params {
        return Err(EngineError::BadProblem(format!(
            "free = {} but the model has {} parameters",
            s.free, m.n_user_params
        )));
    }
    if s.ntst < 1 {
        return Err(EngineError::BadProblem("ntst must be at least 1".into()));
    }
    if !NCOL_RANGE.contains(&s.ncol) {
        return Err(EngineError::BadProblem(format!(
            "ncol must be between {} and {}, got {}",
            NCOL_RANGE.start(),
            NCOL_RANGE.end(),
            s.ncol
        )));
    }
    let n = m.eval.n();
    let dim = s
        .ntst
        .checked_mul(s.ncol)
        .and_then(|q| q.checked_add(1))
        .and_then(|q| q.checked_mul(n))
        .and_then(|q| q.checked_add(2))
        .filter(|&d| d <= MAX_DIM)
        .ok_or_else(|| {
            EngineError::BadProblem(format!(
                "ntst = {} and ncol = {} give more than {MAX_DIM} unknowns",
                s.ntst, s.ncol
            ))
        })?;
    Ok((n, dim))
}

pub(super) fn check_profile(
    profile: &[f64],
    n: usize,
    s: &DdeCycleSetup,
) -> Result<(), EngineError> {
    let want = n * (s.ntst * s.ncol + 1);
    if profile.len() != want {
        return Err(EngineError::BadProblem(format!(
            "the profile has {} values, expected {want} (n = {n}, ntst = {}, ncol = {})",
            profile.len(),
            s.ntst,
            s.ncol
        )));
    }
    if profile.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::BadProblem(
            "the profile has a NaN or infinite value".into(),
        ));
    }
    Ok(())
}

/// The phase row against `reference` (the profile at the base points), scaled to 2-norm 1,
/// and its offset.
fn reference_row(
    basis: &CollocationBasis,
    n: usize,
    ntst: usize,
    reference: &[f64],
) -> Result<(Vec<f64>, f64), EngineError> {
    let (mut phase, mut phase0) = phase_row(basis, n, ntst, reference);
    let norm = phase.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(EngineError::BadProblem(
            "the phase condition is zero for this reference profile; it must not be constant"
                .into(),
        ));
    }
    phase.iter_mut().for_each(|v| *v /= norm);
    phase0 /= norm;
    Ok((phase, phase0))
}

/// The collocation system of one setup, against a phase reference.
struct DdeCycleProblem<'a> {
    m: &'a mut DdeModel,
    basis: CollocationBasis,
    params: Vec<f64>,
    free: usize,
    n: usize,
    ntst: usize,
    dim: usize,
    /// The phase row, scaled to 2-norm 1, and its offset `phase . uref`.
    phase: Vec<f64>,
    phase0: f64,
}

impl<'a> DdeCycleProblem<'a> {
    fn new(
        m: &'a mut DdeModel,
        s: &DdeCycleSetup,
        n: usize,
        dim: usize,
        reference: &[f64],
    ) -> Result<Self, EngineError> {
        let basis = CollocationBasis::new(s.ncol)?;
        let (phase, phase0) = reference_row(&basis, n, s.ntst, reference)?;
        Ok(DdeCycleProblem {
            m,
            basis,
            params: s.params.clone(),
            free: s.free,
            n,
            ntst: s.ntst,
            dim,
            phase,
            phase0,
        })
    }

    /// Writes the equations to `res` (`dim - 1` values) and the column-major Jacobian to
    /// `jac` (`(dim - 1) dim` values), each when given.
    fn fill(
        &mut self,
        x: &[f64],
        mut res: Option<&mut [f64]>,
        mut jac: Option<&mut [f64]>,
    ) -> Result<(), EngineError> {
        let DdeCycleProblem {
            m,
            basis,
            params,
            free,
            n,
            ntst,
            dim,
            phase,
            phase0,
        } = self;
        let (n, ntst, dim, free) = (*n, *ntst, *dim, *free);
        let rows = dim - 1;
        let mm = basis.m();
        let bad_len = |what: &str| EngineError::BadProblem(format!("wrong length of {what}"));
        if x.len() != dim {
            return Err(bad_len("x"));
        }
        if res.as_ref().is_some_and(|r| r.len() != rows) {
            return Err(bad_len("the residual buffer"));
        }
        if jac.as_ref().is_some_and(|j| j.len() != rows * dim) {
            return Err(bad_len("the Jacobian buffer"));
        }
        let (t, alpha) = (x[dim - 2], x[dim - 1]);
        if !(t.is_finite() && t > 0.0) || !alpha.is_finite() {
            return Err(EngineError::NotFinite("the period or the parameter".into()));
        }
        let mut p = params.clone();
        p[free] = alpha;
        let taus: Vec<f64> = m.delays.iter().map(|d| time_of(d, &p)).collect();
        if taus.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(EngineError::NotFinite("a delay time".into()));
        }
        let n_user = m.n_user_params;
        let n_s = m.delays.len();
        let n_p = m.eval.n_p();
        let mut full = p;
        full.resize(n_user + n_s + 1, 0.0);
        let states: Vec<usize> = m.delays.iter().map(|d| d.state).collect();
        let on_free: Vec<bool> = m
            .delays
            .iter()
            .map(|d| matches!(d.time, DelayTime::Param(j) if j == free))
            .collect();
        let nn = ntst as f64;
        if let Some(j) = jac.as_deref_mut() {
            j.fill(0.0);
        }
        let mut u = vec![0.0; n];
        let mut f = vec![0.0; n];
        let mut ju = vec![0.0; n * n];
        let mut jp = vec![0.0; n * n_p];
        let mut lv = vec![0.0; mm + 1];
        let mut dv = vec![0.0; mm + 1];
        // Per slot: interval, shifted point in it, and the lag value and its slope in s.
        let mut lag_j = vec![0usize; n_s];
        let mut lag_l = vec![vec![0.0; mm + 1]; n_s];
        let mut lag_du = vec![0.0; n_s];
        for j in 0..ntst {
            let at = j * mm * n;
            for i in 0..mm {
                for (c, uc) in u.iter_mut().enumerate() {
                    *uc = (0..=mm)
                        .map(|l| basis.value_at_node(i, l) * x[at + l * n + c])
                        .sum();
                }
                let s = (j as f64 + basis.nodes()[i]) / nn;
                for k in 0..n_s {
                    let sp = (s - taus[k] / t).rem_euclid(1.0);
                    let jp_k = ((nn * sp).floor() as usize).min(ntst - 1);
                    let zeta = nn * sp - jp_k as f64;
                    basis.values(zeta, &mut lv);
                    basis.derivatives(zeta, &mut dv);
                    let base = jp_k * mm * n;
                    let mut y = 0.0;
                    let mut dy = 0.0;
                    for l in 0..=mm {
                        let v = x[base + l * n + states[k]];
                        y += lv[l] * v;
                        dy += dv[l] * v;
                    }
                    lag_j[k] = jp_k;
                    lag_l[k].copy_from_slice(&lv);
                    lag_du[k] = nn * dy;
                    full[n_user + k] = y;
                }
                m.eval.eval(&EntryKind::F, &u, &full, &[], &mut f)?;
                let r0 = (j * mm + i) * n;
                if let Some(r) = &mut res {
                    for c in 0..n {
                        let du: f64 = (0..=mm)
                            .map(|l| basis.derivative_at_node(i, l) * x[at + l * n + c])
                            .sum();
                        r[r0 + c] = nn * du - t * f[c];
                    }
                }
                let Some(jm) = jac.as_deref_mut() else {
                    continue;
                };
                m.eval.eval(&EntryKind::JacU, &u, &full, &[], &mut ju)?;
                m.eval.eval(&EntryKind::JacP, &u, &full, &[], &mut jp)?;
                for c in 0..n {
                    let row = r0 + c;
                    for l in 0..=mm {
                        let (d, v) = (basis.derivative_at_node(i, l), basis.value_at_node(i, l));
                        for c2 in 0..n {
                            let col = at + l * n + c2;
                            jm[row + rows * col] -= t * ju[c + n * c2] * v;
                        }
                        jm[row + rows * (at + l * n + c)] += nn * d;
                    }
                    let mut dt = -f[c];
                    let mut da = -t * jp[c + n * free];
                    for k in 0..n_s {
                        let jk = jp[c + n * (n_user + k)];
                        for (l, ll) in lag_l[k].iter().enumerate().take(mm + 1) {
                            let col = (lag_j[k] * mm + l) * n + states[k];
                            jm[row + rows * col] -= t * jk * ll;
                        }
                        dt -= taus[k] * jk * lag_du[k] / t;
                        if on_free[k] {
                            da += jk * lag_du[k];
                        }
                    }
                    jm[row + rows * (dim - 2)] = dt;
                    jm[row + rows * (dim - 1)] = da;
                }
            }
        }
        let pr = ntst * mm * n;
        for c in 0..n {
            if let Some(r) = &mut res {
                r[pr + c] = x[pr + c] - x[c];
            }
            if let Some(jm) = &mut jac {
                jm[pr + c + rows * (pr + c)] = 1.0;
                jm[pr + c + rows * c] -= 1.0;
            }
        }
        let prow = pr + n;
        if let Some(r) = &mut res {
            r[prow] = phase.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() - *phase0;
        }
        if let Some(jm) = &mut jac {
            for (q, v) in phase.iter().enumerate() {
                jm[prow + rows * q] = *v;
            }
        }
        Ok(())
    }
}

impl Problem for DdeCycleProblem<'_> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.fill(x, Some(out), None)
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.fill(x, None, Some(out))
    }

    /// The tests of an equilibrium branch do not apply to the discretized orbit; the labels
    /// of cycles come in a later slice (SPEC section 8n).
    fn ep_tests(&self) -> bool {
        false
    }

    /// Re-references the phase row to the accepted profile, so that it stays a condition
    /// on the shift of the orbit along the branch. A profile whose row is zero or not
    /// finite keeps the old reference.
    fn accept(&mut self, x: &[f64]) {
        let plen = self.dim - 2;
        if x.len() < plen {
            return;
        }
        if let Ok((phase, phase0)) = reference_row(&self.basis, self.n, self.ntst, &x[..plen]) {
            self.phase = phase;
            self.phase0 = phase0;
        }
    }
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a: f64, x| {
        if a.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            a.max(x.abs())
        }
    })
}

/// The `dim - 1` equations at `x` for the phase reference `reference` (the profile at the
/// base points). `x` is `[u, T, alpha]`; the parameter at `setup.free` is replaced by
/// `alpha`.
pub fn cycle_residual(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    reference: &[f64],
    x: &[f64],
) -> Result<Vec<f64>, EngineError> {
    let (n, dim) = check_setup(model, setup)?;
    check_profile(reference, n, setup)?;
    let mut p = DdeCycleProblem::new(model, setup, n, dim, reference)?;
    let mut out = try_zeros(Some(dim - 1))?;
    p.fill(x, Some(&mut out), None)?;
    Ok(out)
}

/// The column-major `(dim - 1)` by `dim` Jacobian at `x`, for the same arguments as
/// [`cycle_residual`].
pub fn cycle_jacobian(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    reference: &[f64],
    x: &[f64],
) -> Result<Vec<f64>, EngineError> {
    let (n, dim) = check_setup(model, setup)?;
    check_profile(reference, n, setup)?;
    let mut p = DdeCycleProblem::new(model, setup, n, dim, reference)?;
    let mut out = try_zeros((dim - 1).checked_mul(dim))?;
    p.fill(x, None, Some(&mut out))?;
    Ok(out)
}

/// Runs the corrector and turns its outcome into a [`DdeCycle`].
fn run(
    p: &mut DdeCycleProblem,
    xk: &[f64],
    v: &[f64],
    hh: f64,
    start: Vec<f64>,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeCycle, EngineError> {
    let dim = p.dim;
    match correct(p, xk, v, hh, start, opts, cancel) {
        Corrected::Done { x, iters } => {
            let mut res = try_zeros(Some(dim - 1))?;
            p.residual(&x, &mut res)?;
            Ok(DdeCycle {
                profile: x[..dim - 2].to_vec(),
                period: x[dim - 2],
                alpha: x[dim - 1],
                iters,
                residual: norm_inf(&res),
            })
        }
        Corrected::Failed => Err(EngineError::NewtonFailed {
            iters: opts.max_newton,
            residual: f64::NAN,
        }),
        Corrected::Stop(StopReason::Cancelled) => Err(EngineError::Cancelled),
        Corrected::Stop(StopReason::Singular) => Err(EngineError::Singular(
            "the bordered Jacobian of the orbit is singular".into(),
        )),
        Corrected::Stop(StopReason::Error(e)) => Err(e),
        Corrected::Stop(other) => Err(EngineError::BadProblem(format!(
            "the corrector stopped with {other:?}"
        ))),
    }
}

/// Newton's method on the orbit at the fixed parameter `setup.params[setup.free]`, from
/// `start`. The phase condition refers to `start.profile`.
pub fn correct_cycle(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeCycle, EngineError> {
    let (n, dim) = check_setup(model, setup)?;
    check_profile(&start.profile, n, setup)?;
    check_period(start.period)?;
    let mut p = DdeCycleProblem::new(model, setup, n, dim, &start.profile)?;
    let mut x0 = start.profile.clone();
    x0.push(start.period);
    x0.push(setup.params[setup.free]);
    let mut v = vec![0.0; dim];
    v[dim - 1] = 1.0;
    run(&mut p, &x0.clone(), &v, 0.0, x0, opts, cancel)
}

/// The orbit that bifurcates from a Hopf point of an equilibrium, with the parameter
/// free. The first guess is `x + eps Re(q exp(2 pi i s))` with the critical vector `q`,
/// the period is `2 pi / omega` and the parameter is `setup.params[setup.free]`, which
/// must be its value at the Hopf point. Newton runs on the system with the border
/// `phi . (u - x) = eps |phi|`, which fixes the amplitude.
pub fn hopf_cycle(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    hopf: &DdeHopfCycle,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeCycle, EngineError> {
    let (n, dim) = check_setup(model, setup)?;
    if hopf.x.len() != n || hopf.x.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::BadProblem(format!(
            "x must be {n} finite values, the equilibrium at the Hopf point"
        )));
    }
    if !(hopf.omega.is_finite() && hopf.omega > 0.0) {
        return Err(EngineError::BadProblem(format!(
            "omega = {}; it must be finite and positive",
            hopf.omega
        )));
    }
    if !(hopf.eps.is_finite() && hopf.eps > 0.0) {
        return Err(EngineError::BadProblem(format!(
            "eps = {}; it must be finite and positive, there is no default",
            hopf.eps
        )));
    }
    let lin = super::eq::linearize(model, &hopf.x, &setup.params)?;
    let q = pair(&lin, hopf.omega)?.q;
    let nq = setup.ntst * setup.ncol;
    let plen = dim - 2;
    let mut phi = try_zeros(Some(plen))?;
    let mut guess = try_zeros(Some(plen))?;
    for k in 0..=nq {
        let a = 2.0 * PI * k as f64 / nq as f64;
        let (sin, cos) = a.sin_cos();
        for c in 0..n {
            let w = q[c].re * cos - q[c].im * sin;
            phi[k * n + c] = w;
            guess[k * n + c] = hopf.x[c] + hopf.eps * w;
        }
    }
    let norm = phi.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(EngineError::BadProblem(
            "the critical vector gives a zero profile".into(),
        ));
    }
    let mut p = DdeCycleProblem::new(model, setup, n, dim, &guess)?;
    let alpha0 = setup.params[setup.free];
    let t0 = 2.0 * PI / hopf.omega;
    let mut xk = try_zeros(Some(dim))?;
    for k in 0..=nq {
        xk[k * n..(k + 1) * n].copy_from_slice(&hopf.x);
    }
    xk[plen] = t0;
    xk[plen + 1] = alpha0;
    let mut v = phi;
    v.iter_mut().for_each(|e| *e /= norm);
    v.extend([0.0, 0.0]);
    let mut x0 = guess;
    x0.extend([t0, alpha0]);
    run(&mut p, &xk, &v, hopf.eps * norm, x0, opts, cancel)
}

/// One point of a branch of periodic orbits of a delay equation.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCyclePoint {
    /// `u` at the base points, `n` values per point, as in [`DdeCycle`].
    pub profile: Vec<f64>,
    pub period: f64,
    /// The value of the free parameter.
    pub alpha: f64,
    /// The step that reached the point; 0 for the first.
    pub h: f64,
    /// Corrector iterations at the point; 0 for the first.
    pub newton_iters: usize,
}

/// The result of [`continue_cycle`].
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycleBranch {
    pub points: Vec<DdeCyclePoint>,
    pub stop: StopReason,
    pub warnings: Vec<String>,
}

fn check_period(period: f64) -> Result<(), EngineError> {
    if !(period.is_finite() && period > 0.0) {
        return Err(EngineError::BadProblem(format!(
            "the period is {period}; it must be finite and positive"
        )));
    }
    Ok(())
}

/// Continues the orbit from `start` in the parameter `setup.free` by pseudo-arclength
/// continuation of the rows of [`correct_cycle`] on a fixed mesh (SPEC section 8n). `start`
/// is first corrected at the fixed parameter `setup.params[setup.free]`, and the phase
/// reference moves to each accepted profile. Special points are not looked for: `detect_*`
/// and `mesh_adapt` have no effect.
pub fn continue_cycle(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeCycleBranch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let (n, dim) = check_setup(model, setup)?;
    check_profile(&start.profile, n, setup)?;
    check_period(start.period)?;
    let free_is_delay = model
        .delays
        .iter()
        .any(|s| matches!(s.time, DelayTime::Param(j) if j == setup.free));
    if free_is_delay && (opts.bounds.0 <= 0.0 || opts.bounds.0.is_nan()) {
        return Err(EngineError::BadProblem(
            "the free parameter is a delay time, so bounds must have a lower bound above 0".into(),
        ));
    }
    check_step_options(opts)?;
    let mut opts = opts.clone();
    opts.detect_lp = false;
    opts.detect_bp = false;
    opts.detect_h = false;

    let first = correct_cycle(model, setup, start, &opts, cancel)?;
    let mut p = DdeCycleProblem::new(model, setup, n, dim, &first.profile)?;
    let mut x0 = first.profile;
    x0.push(first.period);
    x0.push(first.alpha);
    let branch = run_branch(&mut p, &x0, None, &opts, false, cancel, &mut |_| {})?;
    let points = branch
        .points
        .into_iter()
        .map(|q| DdeCyclePoint {
            profile: q.u[..dim - 2].to_vec(),
            period: q.u[dim - 2],
            alpha: q.alpha,
            h: q.h,
            newton_iters: q.newton_iters,
        })
        .collect();
    Ok(DdeCycleBranch {
        points,
        stop: branch.stop,
        warnings: branch.warnings,
    })
}
