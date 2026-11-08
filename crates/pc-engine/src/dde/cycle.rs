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
use super::floquet::{DdeFloquetOptions, check_floquet, monodromy_on, summarize};
use super::lyap::pair;
use crate::branch::{Options, Point, StopReason};
use crate::cycle::{
    CYCLE_LABELS, CollocationBasis, NCOL_RANGE, interpolate, nearest_pair_is_complex, new_mesh,
    nontrivial, pair_product, phase_row, shifted_product, try_zeros,
};
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

/// Refuses a mesh that is not `ntst + 1` finite values with `mesh[0] = 0`, `mesh[ntst] = 1`
/// and strictly increasing in between.
pub(super) fn check_mesh(mesh: &[f64], ntst: usize) -> Result<(), EngineError> {
    let ok = mesh.len() == ntst + 1
        && mesh[0] == 0.0
        && mesh[ntst] == 1.0
        && mesh.windows(2).all(|w| w[0] < w[1]);
    if ok {
        Ok(())
    } else {
        Err(EngineError::BadProblem(format!(
            "the mesh must be {} increasing values from 0 to 1",
            ntst + 1
        )))
    }
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
    /// The Floquet options of a run that looks for LPC, PD and NS points; `None` leaves the
    /// multipliers out and only the LPC test is defined.
    floquet: Option<DdeFloquetOptions>,
    /// Never set. `state_jacobian` has no cancel argument, so a run is cancelled between
    /// steps and not inside one monodromy matrix.
    never: AtomicBool,
    /// The `ntst + 1` mesh points, `0 = s_0 < ... < s_N = 1` (SPEC section 8p).
    mesh: Vec<f64>,
    /// True while the mesh is the uniform `j / N`. `fill` and the monodromy then take the
    /// formulas of the fixed mesh unchanged. Any other mesh, an explicit uniform one too,
    /// takes the general path.
    uniform: bool,
    /// Accepted points between mesh adaptations; `None` keeps the mesh fixed.
    adapt_every: Option<usize>,
    /// Accepted points since the last adaptation, and whether the current point adapted.
    since: usize,
    adapted_here: bool,
}

/// The interval of `mesh` that holds `sp` in `[0, 1]` and the position of `sp` in it, in
/// `[0, 1]`. A point on a node belongs to the interval that starts there, and 1 belongs to
/// the last interval (SPEC section 8p).
pub(super) fn locate(mesh: &[f64], sp: f64) -> (usize, f64) {
    let ntst = mesh.len() - 1;
    let jp = mesh
        .partition_point(|&s| s <= sp)
        .saturating_sub(1)
        .min(ntst - 1);
    (jp, (sp - mesh[jp]) / (mesh[jp + 1] - mesh[jp]))
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
            floquet: None,
            never: AtomicBool::new(false),
            mesh: (0..=s.ntst).map(|j| j as f64 / s.ntst as f64).collect(),
            uniform: true,
            adapt_every: None,
            since: 0,
            adapted_here: false,
        })
    }

    /// Takes `mesh` (`ntst + 1` increasing values from 0 to 1) and switches to the general
    /// formulas.
    fn set_mesh(&mut self, mesh: Vec<f64>) {
        self.mesh = mesh;
        self.uniform = false;
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
            mesh,
            uniform,
            ..
        } = self;
        let (n, ntst, dim, free, uniform) = (*n, *ntst, *dim, *free, *uniform);
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
        // The factor `1 / delta_j` of `d/ds` on interval `j`; `N` on the uniform mesh.
        let scale = |j: usize| {
            if uniform {
                nn
            } else {
                1.0 / (mesh[j + 1] - mesh[j])
            }
        };
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
            let sc = scale(j);
            for i in 0..mm {
                for (c, uc) in u.iter_mut().enumerate() {
                    *uc = (0..=mm)
                        .map(|l| basis.value_at_node(i, l) * x[at + l * n + c])
                        .sum();
                }
                let s = if uniform {
                    (j as f64 + basis.nodes()[i]) / nn
                } else {
                    mesh[j] + basis.nodes()[i] * (mesh[j + 1] - mesh[j])
                };
                for k in 0..n_s {
                    let sp = (s - taus[k] / t).rem_euclid(1.0);
                    let (jp_k, zeta) = if uniform {
                        let jp_k = ((nn * sp).floor() as usize).min(ntst - 1);
                        (jp_k, nn * sp - jp_k as f64)
                    } else {
                        locate(mesh, sp)
                    };
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
                    lag_du[k] = scale(jp_k) * dy;
                    full[n_user + k] = y;
                }
                m.eval.eval(&EntryKind::F, &u, &full, &[], &mut f)?;
                let r0 = (j * mm + i) * n;
                if let Some(r) = &mut res {
                    for c in 0..n {
                        let du: f64 = (0..=mm)
                            .map(|l| basis.derivative_at_node(i, l) * x[at + l * n + c])
                            .sum();
                        r[r0 + c] = sc * du - t * f[c];
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
                        jm[row + rows * (at + l * n + c)] += sc * d;
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

    /// The tests of an equilibrium branch do not apply to the discretized orbit. The
    /// codim-2 tests below take their place (SPEC section 8o).
    fn ep_tests(&self) -> bool {
        false
    }

    /// The monodromy matrix `Phi(T)` of the history ODE at `x`, whose eigenvalues are the
    /// multipliers (SPEC section 8m). Without Floquet options there is no matrix.
    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        let Some(options) = self.floquet.clone() else {
            return Ok(None);
        };
        if x.len() != self.dim {
            return Err(EngineError::BadProblem("wrong length of x".into()));
        }
        let plen = self.dim - 2;
        let mut params = self.params.clone();
        params[self.free] = x[self.dim - 1];
        let setup = DdeCycleSetup {
            params,
            free: self.free,
            ntst: self.ntst,
            ncol: self.basis.m(),
        };
        let start = DdeCycleStart {
            profile: x[..plen].to_vec(),
            period: x[plen],
        };
        let mesh = (!self.uniform).then_some(self.mesh.as_slice());
        let (phi, _) = monodromy_on(self.m, &setup, &start, &options, mesh, &self.never)?;
        Ok(Some(phi))
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        &CYCLE_LABELS
    }

    /// The same tests as the ODE cycles (SPEC section 7): LPC is the alpha component of the
    /// tangent, PD the product of mu + 1 and NS the product of mu_i mu_j - 1 over i < j, both
    /// over the multipliers without the one closest to 1. Without multipliers PD and NS are
    /// NaN.
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
        Ok(vec![lpc, shifted_product(&mu, 1.0), pair_product(&mu)])
    }

    /// The NS product also vanishes when two real multipliers have product 1; the zero is an
    /// NS point only when the pair closest to product 1 is a complex pair.
    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        if i != 2 {
            return CYCLE_LABELS.get(i).copied();
        }
        nearest_pair_is_complex(&nontrivial(eigs)).then_some("NS")
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

    /// The mesh of the profile, stored with each point of an adapting run; empty when the
    /// run keeps the mesh fixed (SPEC section 8p).
    fn mesh(&self) -> Vec<f64> {
        if self.adapt_every.is_some() {
            self.mesh.clone()
        } else {
            Vec::new()
        }
    }

    /// Moves the mesh every `adapt_every` accepted points and after a failed corrector, at
    /// most once per point, and interpolates the profile part of x and v onto it. T and
    /// alpha stay. The interpolated x becomes the reference of the phase row.
    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        let Some(every) = self.adapt_every else {
            return false;
        };
        if !failed {
            self.adapted_here = false;
            self.since += 1;
        }
        let due = if failed {
            !self.adapted_here
        } else {
            self.since >= every
        };
        if !due {
            return false;
        }
        self.since = 0;
        self.adapted_here = true;
        let k = self.dim - 2;
        if x.len() < self.dim || v.len() < self.dim {
            return false;
        }
        let Some(new) = new_mesh(&self.basis, self.n, &self.mesh, &x[..k], true) else {
            return false;
        };
        let m = self.basis.m();
        // Without the memory for the new profile, the mesh stays.
        let (Ok(xu), Ok(vu)) = (
            interpolate(&self.basis, self.n, &self.mesh, &new, m, &x[..k]),
            interpolate(&self.basis, self.n, &self.mesh, &new, m, &v[..k]),
        ) else {
            return false;
        };
        x[..k].copy_from_slice(&xu);
        v[..k].copy_from_slice(&vu);
        self.set_mesh(new);
        self.accept(x);
        true
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
    /// The largest Floquet multipliers as `(re, im)`, largest modulus first. Empty without
    /// Floquet options, and when the eigenvalue computation did not converge.
    pub multipliers: Vec<(f64, f64)>,
    /// `min |mu - 1|` over all multipliers of the discretized `Phi(T)`; NaN when `multipliers`
    /// is empty.
    pub trivial_error: f64,
    /// The `N + 1` mesh points, from 0 to 1, that `profile` lives on. Empty on a run without
    /// [`DdeMeshAdapt`] (SPEC section 8p).
    pub mesh: Vec<f64>,
}

/// Mesh adaptation for [`continue_cycle_adapt`] (SPEC section 8p).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdeMeshAdapt {
    /// Accepted points between adaptations, at least 1. A failed corrector also adapts the
    /// mesh, at most once per accepted point.
    pub every: usize,
}

/// A special point on a branch of periodic orbits of a delay equation.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycleSpecial {
    /// `"LPC"`, `"PD"` or `"NS"`.
    pub label: String,
    /// The index in `points` of the first point at or after the special point.
    pub index: usize,
    /// The orbit at the special point.
    pub point: DdeCyclePoint,
}

/// The result of [`continue_cycle`].
#[derive(Clone, Debug, PartialEq)]
pub struct DdeCycleBranch {
    pub points: Vec<DdeCyclePoint>,
    /// The LPC, PD and NS points, in the order of the run. Only an LPC is found without
    /// Floquet options.
    pub specials: Vec<DdeCycleSpecial>,
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

/// [`continue_cycle_floquet`] without Floquet options: only fold points (LPC) are found.
pub fn continue_cycle(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeCycleBranch, EngineError> {
    continue_cycle_floquet(model, setup, start, opts, None, cancel)
}

fn cycle_point(q: &Point, dim: usize, count: usize) -> DdeCyclePoint {
    let multipliers = summarize(q.eigenvalues.clone(), count);
    DdeCyclePoint {
        profile: q.u[..dim - 2].to_vec(),
        period: q.u[dim - 2],
        alpha: q.alpha,
        h: q.h,
        newton_iters: q.newton_iters,
        multipliers: multipliers.multipliers,
        trivial_error: multipliers.trivial_error,
        mesh: q.mesh.clone(),
    }
}

/// The checks of `floquet` against the model and the mesh of `setup`, as
/// [`continue_cycle_floquet`] makes them (SPEC section 8o). A caller that corrects the start
/// itself calls this first, so that a bad `floquet` is refused before any work. The message
/// starts with `floquet: `.
pub fn check_cycle_floquet(
    model: &DdeModel,
    setup: &DdeCycleSetup,
    floquet: &DdeFloquetOptions,
) -> Result<(), EngineError> {
    let (n, _) = check_setup(model, setup)?;
    check_floquet_keyed(model, setup, n, floquet)
}

/// `check_floquet` with the key in the message, so a caller can tell the error from the other
/// `BadProblem`s.
fn check_floquet_keyed(
    model: &DdeModel,
    setup: &DdeCycleSetup,
    n: usize,
    floquet: &DdeFloquetOptions,
) -> Result<(), EngineError> {
    check_floquet(model, setup, n, floquet)
        .map(|_| ())
        .map_err(|e| match e {
            EngineError::BadProblem(m) => EngineError::BadProblem(format!("floquet: {m}")),
            other => other,
        })
}

/// Continues the orbit from `start` in the parameter `setup.free` by pseudo-arclength
/// continuation of the rows of [`correct_cycle`] on a fixed mesh (SPEC section 8n). `start`
/// is first corrected at the fixed parameter `setup.params[setup.free]`, and the phase
/// reference moves to each accepted profile.
///
/// The run reports LPC points, folds of the branch, and with `floquet` also PD and NS
/// points from the Floquet multipliers at every point and every locator trial (SPEC
/// section 8o). `detect_*` and `mesh_adapt` have no effect. Without `floquet`, the points
/// carry no multipliers. The mesh stays uniform and the points carry no mesh.
pub fn continue_cycle_floquet(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    floquet: Option<&DdeFloquetOptions>,
    cancel: &AtomicBool,
) -> Result<DdeCycleBranch, EngineError> {
    continue_cycle_on(model, setup, start, opts, floquet, None, cancel)
}

/// [`continue_cycle_floquet`] with the mesh moving along the branch (SPEC section 8p). The
/// start mesh is uniform. Every `adapt.every` accepted points, and after a failed corrector,
/// the mesh is redistributed from the profile and the profile is interpolated onto it. Each
/// point and special carries the mesh of its profile. `opts.mesh_adapt` has no effect.
///
/// The checks of [`continue_cycle_floquet`] come first, then `adapt.every >= 1` and
/// `setup.ntst >= 2`; both give `BadProblem` with a message that starts with `adapt: `.
pub fn continue_cycle_adapt(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    floquet: Option<&DdeFloquetOptions>,
    adapt: &DdeMeshAdapt,
    cancel: &AtomicBool,
) -> Result<DdeCycleBranch, EngineError> {
    continue_cycle_on(model, setup, start, opts, floquet, Some(adapt), cancel)
}

fn continue_cycle_on(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    start: &DdeCycleStart,
    opts: &Options,
    floquet: Option<&DdeFloquetOptions>,
    adapt: Option<&DdeMeshAdapt>,
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
    if let Some(f) = floquet {
        check_floquet_keyed(model, setup, n, f)?;
    }
    if let Some(a) = adapt {
        if a.every == 0 {
            return Err(EngineError::BadProblem(
                "adapt: every must be at least 1".into(),
            ));
        }
        if setup.ntst < 2 {
            return Err(EngineError::BadProblem(
                "adapt: the mesh needs ntst of at least 2".into(),
            ));
        }
    }
    let mut opts = opts.clone();
    opts.detect_lp = false;
    opts.detect_bp = false;
    opts.detect_h = false;

    let first = correct_cycle(model, setup, start, &opts, cancel)?;
    let mut p = DdeCycleProblem::new(model, setup, n, dim, &first.profile)?;
    p.floquet = floquet.cloned();
    p.adapt_every = adapt.map(|a| a.every);
    let count = floquet.map_or(0, |f| f.count);
    let mut x0 = first.profile;
    x0.push(first.period);
    x0.push(first.alpha);
    let branch = run_branch(&mut p, &x0, None, &opts, false, cancel, &mut |_| {})?;
    let points = branch
        .points
        .iter()
        .map(|q| cycle_point(q, dim, count))
        .collect();
    let specials = branch
        .specials
        .iter()
        .map(|sp| DdeCycleSpecial {
            label: sp.label.clone(),
            index: sp.index,
            point: cycle_point(&sp.point, dim, count),
        })
        .collect();
    Ok(DdeCycleBranch {
        points,
        specials,
        stop: branch.stop,
        warnings: branch.warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};
    use std::f64::consts::{FRAC_PI_2, PI};

    const NTST: usize = 20;
    const NCOL: usize = 4;

    /// The manufactured steep model of SPEC section 8p with `G = 20`: the exact orbit is
    /// `y1 = cos t`, `y2 = sin t`, `x = tanh(20 cos t)`, `T = 2 pi`, for every `c` and `tau`.
    const STEEP: &str = "par c=0.5, tau=1\n\
        x'=-x+c*(delay(x,tau)-tanh(20*delay(y1,tau)))+tanh(20*y1)+20*(1-tanh(20*y1)^2)*(-y2)\n\
        y1'=y1-y2-y1*(y1^2+y2^2)\n\
        y2'=y1+y2-y2*(y1^2+y2^2)\n\
        done\n";

    fn build(src: &str) -> DdeModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, Evaluator::Fallback).expect("model builds")
    }

    fn osc() -> DdeModel {
        build(&format!(
            "par mu=0.3, tau={FRAC_PI_2:?}\nx'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\ndone\n"
        ))
    }

    /// Intervals that alternate between `0.5 / N` and `1.5 / N`.
    fn hand_mesh() -> Vec<f64> {
        let mut mesh = vec![0.0];
        for j in 0..NTST {
            let w = if j % 2 == 0 { 0.5 } else { 1.5 };
            mesh.push(mesh[j] + w / NTST as f64);
        }
        mesh[NTST] = 1.0;
        mesh
    }

    fn uniform_mesh() -> Vec<f64> {
        (0..=NTST).map(|j| j as f64 / NTST as f64).collect()
    }

    /// The base points of `mesh`: `s_j + (l / m) (s_{j+1} - s_j)`, then 1.
    fn base_points(mesh: &[f64]) -> Vec<f64> {
        let mut s = Vec::new();
        for j in 0..NTST {
            for l in 0..NCOL {
                s.push(mesh[j] + (l as f64 / NCOL as f64) * (mesh[j + 1] - mesh[j]));
            }
        }
        s.push(1.0);
        s
    }

    fn steep_profile(mesh: &[f64]) -> Vec<f64> {
        base_points(mesh)
            .iter()
            .flat_map(|s| {
                let (sn, cs) = (2.0 * PI * s).sin_cos();
                [(20.0 * cs).tanh(), cs, sn]
            })
            .collect()
    }

    fn sine_profile(mesh: &[f64]) -> Vec<f64> {
        base_points(mesh)
            .iter()
            .map(|s| (2.0 * PI * s).sin())
            .collect()
    }

    fn setup(params: &[f64], free: usize) -> DdeCycleSetup {
        DdeCycleSetup {
            params: params.to_vec(),
            free,
            ntst: NTST,
            ncol: NCOL,
        }
    }

    fn problem<'a>(
        m: &'a mut DdeModel,
        params: &[f64],
        free: usize,
        reference: &[f64],
    ) -> DdeCycleProblem<'a> {
        let st = setup(params, free);
        let (n, dim) = check_setup(m, &st).expect("setup");
        DdeCycleProblem::new(m, &st, n, dim, reference).expect("problem")
    }

    /// A point that is not on the orbit: the exact profile with a smooth bump, `T` and
    /// `alpha` moved off their values.
    fn perturbed(profile: &[f64], alpha: f64) -> Vec<f64> {
        let mut x: Vec<f64> = profile
            .iter()
            .enumerate()
            .map(|(q, v)| v + 0.03 * (0.37 * q as f64 + 0.5).sin())
            .collect();
        x.push(6.1);
        x.push(alpha);
        x
    }

    fn residual_of(p: &mut DdeCycleProblem, x: &[f64]) -> Vec<f64> {
        let mut r = vec![0.0; p.dim - 1];
        p.fill(x, Some(&mut r), None).expect("residual");
        r
    }

    fn jacobian_of(p: &mut DdeCycleProblem, x: &[f64]) -> Vec<f64> {
        let mut j = vec![0.0; (p.dim - 1) * p.dim];
        p.fill(x, None, Some(&mut j)).expect("jacobian");
        j
    }

    /// The largest `|jac - fd| / (1 + |jac|)` over all entries, against central differences.
    fn jacobian_error(p: &mut DdeCycleProblem, x: &[f64]) -> f64 {
        let jac = jacobian_of(p, x);
        let (rows, dim) = (p.dim - 1, p.dim);
        let mut worst: f64 = 0.0;
        for col in 0..dim {
            let h = 1e-6 * (1.0 + x[col].abs());
            let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
            xp[col] += h;
            xm[col] -= h;
            let (rp, rm) = (residual_of(p, &xp), residual_of(p, &xm));
            for row in 0..rows {
                let fd = (rp[row] - rm[row]) / (2.0 * h);
                let a = jac[row + rows * col];
                worst = worst.max((a - fd).abs() / (1.0 + a.abs()));
            }
        }
        worst
    }

    #[test]
    fn jacobian_on_a_hand_made_mesh_matches_central_differences() {
        for (free, alpha) in [(1usize, 1.05), (0usize, 0.55)] {
            let mut m = build(STEEP);
            let reference = steep_profile(&uniform_mesh());
            let mut p = problem(&mut m, &[0.5, 1.0], free, &reference);
            p.set_mesh(hand_mesh());
            let x = perturbed(&steep_profile(&hand_mesh()), alpha);
            let err = jacobian_error(&mut p, &x);
            assert!(err < 1e-6, "free = {free}: relative error {err:e}");
        }
    }

    #[test]
    fn an_explicit_uniform_mesh_agrees_with_the_uniform_path() {
        for (free, alpha) in [(1usize, 1.05), (0usize, 0.55)] {
            let mut m = build(STEEP);
            let reference = steep_profile(&uniform_mesh());
            let x = perturbed(&reference, alpha);
            let mut p = problem(&mut m, &[0.5, 1.0], free, &reference);
            let (r0, j0) = (residual_of(&mut p, &x), jacobian_of(&mut p, &x));
            p.set_mesh(uniform_mesh());
            let (r1, j1) = (residual_of(&mut p, &x), jacobian_of(&mut p, &x));
            for (a, b) in r0.iter().zip(&r1).chain(j0.iter().zip(&j1)) {
                assert!(
                    (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
                    "free = {free}: {a} vs {b}"
                );
            }
        }
    }

    /// The residual of the exact orbit of OSC on the hand-made mesh is small: it is the
    /// interpolation error of the degree-4 pieces on intervals up to 1.5 / N wide.
    #[test]
    fn the_exact_orbit_has_a_small_residual_on_a_hand_made_mesh() {
        let mut m = osc();
        let mesh = hand_mesh();
        let mut x = sine_profile(&mesh);
        x.push(2.0 * PI);
        x.push(0.3);
        let reference = sine_profile(&uniform_mesh());
        let mut p = problem(&mut m, &[0.3, FRAC_PI_2], 0, &reference);
        p.set_mesh(mesh);
        let r = residual_of(&mut p, &x);
        // The collocation rows only. The phase row is not zero here, since its reference
        // holds the values on the uniform mesh.
        let pr = NTST * NCOL;
        let worst = r[..pr].iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(worst < 1.5e-4, "largest residual {worst:e}");
        // The periodicity row is zero for the exact profile: its ends agree.
        assert!(r[pr].abs() < 1e-15, "periodicity row {}", r[pr]);
    }

    #[test]
    fn a_shifted_point_on_a_node_goes_to_the_interval_that_starts_there() {
        let mesh = hand_mesh();
        for j in 0..NTST {
            assert_eq!(locate(&mesh, mesh[j]), (j, 0.0), "node {j}");
        }
        let below = f64::from_bits(mesh[5].to_bits() - 1);
        let (jp, zeta) = locate(&mesh, below);
        assert_eq!(jp, 4);
        assert!(zeta > 1.0 - 1e-12 && zeta <= 1.0, "zeta {zeta}");
        // A value that rounds to 1 after the wrap belongs to the last interval.
        assert_eq!(locate(&mesh, 1.0), (NTST - 1, 1.0));
        let mid = 0.5 * (mesh[7] + mesh[8]);
        let (jp, zeta) = locate(&mesh, mid);
        assert_eq!(jp, 7);
        assert!((zeta - 0.5).abs() < 1e-14, "zeta {zeta}");
    }
}
