//! The HOM curve: orbits homoclinic to a hyperbolic saddle in two parameters (SPEC §8b).
//!
//! The unknowns are x = (u_0, ..., u_{Nm}, T, x0, a1, a2): the orbit at the base points,
//! the truncation time T, the saddle x0 and the two free parameters, a2 being
//! `Point::alpha`. The equations are the collocation rows of the cycle problem without the
//! periodic wrap, the projection boundary rows L_s^T (u(0) - x0) = 0 and
//! L_u^T (u(1) - x0) = 0, the phase row against the previous point, f(x0, p) = 0 and
//! T = T_start (Beyn 1990; Champneys, Kuznetsov and Sandstede 1996).
//!
//! Ported from AUTO-07p, `src/homcont.f90` (BSD-3), at commit
//! 44cc1c42a888179bdbd608a40613caee9cbc4675: the test functions 1 to 12 of PSIHO, the sign
//! rule of the left eigenvectors in EIGHO, and the rotation of the projection bases in
//! PRJCTN. The bases themselves come from the matrix sign function in `pc-linalg`, not
//! from the Schur form HomCont uses.

use super::continuation::Scaled;
use super::curve::{CycleCurveStart, run_curve};
use super::problem::{CycleProblem, CycleSetup, Extra, check_dims};
use crate::branch::{Branch, Options, Point};
use crate::ep::{EpSetup, equilibrium};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::problem::Problem;
use pc_ir::EntryKind;
use pc_linalg::{BasisOf, BasisPair, BasisTracker, eigenvalues, eigvec};
use std::sync::atomic::AtomicBool;

/// Names of the HOM curve test functions, in the order of `Point::codim2` (SPEC §8b).
pub const HOM_CURVE_LABELS: [&str; 12] = [
    "NSA", "BLS", "BLU", "NSF", "NDS", "NDU", "TLS", "TLU", "ZES", "ZEU", "OFS", "OFU",
];

/// An eigenvalue counts as real when its imaginary part is at most this (HomCont's
/// COMPZERO).
const REAL_TOL: f64 = 1e-13;
/// The start needs every eigenvalue of the saddle at least this far from the imaginary
/// axis.
const HYPERBOLIC_TOL: f64 = 1e-8;
/// Relative step of the forward differences of the boundary rows (AUTO's HMACH).
const FD_STEP: f64 = 1e-7;

/// How the start orbit of a HOM curve was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomStartKind {
    /// An orbit on a uniform mesh that leaves the saddle at tau = 0 and returns at 1.
    Orbit,
    /// A point of an LC branch with a long period, on its own mesh.
    Cycle,
}

/// The start of a HOM curve.
#[derive(Clone, Debug)]
pub struct HomStart {
    pub kind: HomStartKind,
    /// Indices of the two free parameters [a1, a2]; a2 is `Point::alpha` on the curve.
    pub free: [usize; 2],
    /// All `n_p` parameter values at the start.
    pub params: Vec<f64>,
    /// The orbit at the base points, point-major, without T.
    pub orbit: Vec<f64>,
    /// T for `Orbit`, the period of the cycle for `Cycle`.
    pub period: f64,
    /// `None` for `Orbit`, whose mesh is uniform with `ntst` intervals of `ncol` points.
    /// The mesh of the cycle for `Cycle`.
    pub mesh: Option<Vec<f64>>,
    /// Number of mesh intervals N (NTST) of the curve.
    pub ntst: usize,
    /// Number of collocation points per interval m (NCOL) of the curve.
    pub ncol: usize,
}

/// The defining system of the HOM curve over the cycle problem with the extras
/// (x0, a1, a2) after T.
pub struct HomProblem<'a> {
    pub(super) outer: CycleProblem<'a>,
    /// The start in the original unknowns.
    pub(super) x0: Vec<f64>,
    n: usize,
    ns: usize,
    free: [usize; 2],
    period0: f64,
    /// The reference bases of the rotation (C_s n by ns, C_u n by n - ns), from the last
    /// accepted point. The orbit rows use the subspaces of A^T.
    tracker: BasisTracker,
    /// The left eigenvectors w_S and w_{S+1} of the last accepted point, for their signs.
    w_prev: [Option<Vec<f64>>; 2],
    /// Set by `accept` when the eigenvalue counts change; every later call fails with it.
    stop: Option<String>,
    f: Vec<f64>,
}

/// The n by n transpose of the column-major `a`.
fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            t[c + n * r] = a[r + n * c];
        }
    }
    t
}

/// Eigenvalues sorted by real part ascending, a complex pair with its positive imaginary
/// part first.
pub(super) fn ascending(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut lam = eigs.to_vec();
    lam.sort_by(|p, q| p.0.total_cmp(&q.0).then(q.1.total_cmp(&p.1)));
    lam
}

pub(super) fn is_real(l: (f64, f64)) -> bool {
    l.1.abs() <= REAL_TOL
}

/// The 12 HOM tests (SPEC §8b) from the eigenvalues `lam` in ascending order with `ns`
/// stable ones. `tls_sign` and `tlu_sign` are the signs of TLS and TLU, and `ofs` and
/// `ofu` the orbit flip tests, each NaN where it is not defined. All tests are NaN when
/// `ns` is 0 or n.
pub(super) fn hom_curve_tests(
    lam: &[(f64, f64)],
    ns: usize,
    tls_sign: f64,
    tlu_sign: f64,
    ofs: f64,
    ofu: f64,
) -> Vec<f64> {
    let n = lam.len();
    let mut t = vec![f64::NAN; HOM_CURVE_LABELS.len()];
    if ns == 0 || ns >= n {
        return t;
    }
    let (s, u, nu) = (ns - 1, ns, n - ns);
    let re = |i: usize| lam[i].0;
    let im = |i: usize| lam[i].1;
    let real = |i: usize| is_real(lam[i]);
    if real(s) && real(u) {
        t[0] = re(s) + re(u);
    }
    if ns >= 2 {
        t[1] = if real(s) {
            (re(s) - re(s - 1)).powi(2)
        } else {
            -(im(s) - im(s - 1)).powi(2)
        };
        t[4] = re(s - 1) + re(s) + re(u);
    }
    if nu >= 2 {
        t[2] = if real(u) {
            (re(u + 1) - re(u)).powi(2)
        } else {
            -(im(u + 1) - im(u)).powi(2)
        };
        t[5] = re(s) + re(u) + re(u + 1);
    }
    if !real(s) || !real(u) {
        t[3] = re(s) + re(u);
    }
    if ns >= 3 {
        t[6] = tls_sign * (re(s) - re(s - 2));
    }
    if nu >= 3 {
        t[7] = tlu_sign * (re(u) - re(u + 2));
    }
    t[8] = re(s);
    t[9] = re(u);
    t[10] = ofs;
    t[11] = ofu;
    t
}

/// Sum of the absolute values.
fn norm1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// The eigenvector of the real eigenvalue `lam` of the n by n `m`, with unit 2-norm. It is
/// the right eigenvector of A for `m` = A and the left one for `m` = A^T. Against `prev`
/// it is flipped when their dot product is negative; without `prev` its component of
/// largest magnitude is positive. `None` when the eigenvector solve fails.
pub(super) fn signed_eigvec(
    m: &[f64],
    n: usize,
    lam: f64,
    prev: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let (mut w, _) = eigvec(m, n, (lam, 0.0)).ok()?;
    let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(nrm > 0.0 && nrm.is_finite()) {
        return None;
    }
    for x in w.iter_mut() {
        *x /= nrm;
    }
    let flip = match prev {
        Some(p) => w.iter().zip(p).map(|(a, b)| a * b).sum::<f64>() < 0.0,
        None => {
            let big = w
                .iter()
                .fold(0.0_f64, |m, x| if x.abs() > m.abs() { *x } else { m });
            big < 0.0
        }
    };
    if flip {
        for x in w.iter_mut() {
            *x = -*x;
        }
    }
    Some(w)
}

/// Index of the mesh point (every `m`-th base point of `orbit`, up to `last`) with the
/// smallest |f| at `p`.
fn slowest_point(
    model: &mut dyn ModelEval,
    orbit: &[f64],
    p: &[f64],
    m: usize,
    last: usize,
) -> Result<usize, EngineError> {
    let n = model.n();
    let mut f = vec![0.0; n];
    let mut best = (0, f64::INFINITY);
    for j in 0..=last {
        let at = j * m * n;
        model.eval(&EntryKind::F, &orbit[at..at + n], p, &[], &mut f)?;
        let nrm = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm < best.1 {
            best = (j, nrm);
        }
    }
    Ok(best.0)
}

/// The start of the "cycle" kind: the cycle shifted in tau so that its mesh point with the
/// smallest |f| sits at tau = 0 and tau = 1.
fn shifted_cycle(
    model: &mut dyn ModelEval,
    start: &HomStart,
) -> Result<CycleCurveStart, EngineError> {
    let bad = |m: String| Err(EngineError::BadProblem(m));
    let Some(mesh) = &start.mesh else {
        return bad("the \"cycle\" start needs the mesh of the cycle".into());
    };
    let n = model.n();
    check_dims(model, start.free[0], &start.params, start.ntst, start.ncol)?;
    let nint = mesh.len().saturating_sub(1);
    if nint == 0 || mesh[0] != 0.0 || mesh[nint] != 1.0 || !mesh.windows(2).all(|w| w[1] > w[0]) {
        return bad("the mesh of the start must rise strictly from 0 to 1".into());
    }
    let pts = start.orbit.len() / n;
    if !start.orbit.len().is_multiple_of(n) || pts < 2 || !(pts - 1).is_multiple_of(nint) {
        return bad(format!(
            "expected the orbit at N m + 1 points of {n} values on {nint} mesh intervals, got \
             {} values",
            start.orbit.len()
        ));
    }
    if start
        .orbit
        .iter()
        .chain(&start.params)
        .any(|v| !v.is_finite())
    {
        return bad("the start has NaN or infinite values".into());
    }
    let m = (pts - 1) / nint;
    let mut p = start.params.clone();
    p.push(0.0);
    let j = slowest_point(model, &start.orbit, &p, m, nint - 1)?;
    let (nm, jm) = (nint * m, j * m);
    let mut orbit = Vec::with_capacity(start.orbit.len());
    for i in 0..=nm {
        let old = (jm + i) % nm;
        orbit.extend_from_slice(&start.orbit[old * n..(old + 1) * n]);
    }
    let t0 = mesh[j];
    let mut shifted: Vec<f64> = (0..=nint)
        .map(|i| {
            let t = mesh[(j + i) % nint] - t0;
            if t < 0.0 { t + 1.0 } else { t }
        })
        .collect();
    shifted[0] = 0.0;
    shifted[nint] = 1.0;
    Ok(CycleCurveStart {
        free: start.free,
        params: start.params.clone(),
        orbit,
        period: start.period,
        mesh: shifted,
        ntst: start.ntst,
        ncol: start.ncol,
    })
}

impl<'a> HomProblem<'a> {
    /// Checks the start, finds the saddle x0 by the Newton solve of the EP problem from the
    /// mesh point with the smallest |f| (with the tolerances of `opts`), counts its stable
    /// and unstable eigenvalues and builds the problem at the start.
    pub fn new(
        model: &'a mut dyn ModelEval,
        start: &HomStart,
        opts: &Options,
    ) -> Result<HomProblem<'a>, EngineError> {
        let bad = |m: String| EngineError::BadProblem(m);
        let curve = match start.kind {
            HomStartKind::Orbit => {
                if start.mesh.is_some() {
                    return Err(bad(
                        "the \"orbit\" start takes no mesh; its orbit is on a uniform mesh".into(),
                    ));
                }
                let nt = start.ntst.max(1);
                CycleCurveStart {
                    free: start.free,
                    params: start.params.clone(),
                    orbit: start.orbit.clone(),
                    period: start.period,
                    mesh: (0..=nt).map(|j| j as f64 / nt as f64).collect(),
                    ntst: start.ntst,
                    ncol: start.ncol,
                }
            }
            HomStartKind::Cycle => shifted_cycle(model, start)?,
        };
        let (setup, mesh): (CycleSetup, Vec<f64>) = curve.setup(model)?;
        let n = model.n();
        let [a1, a2] = start.free;
        let mut p = setup.params.clone();
        p.push(0.0);
        let j = slowest_point(model, &setup.profile, &p, setup.ncol, setup.ntst)?;
        let u0 = setup.profile[j * setup.ncol * n..(j * setup.ncol + 1) * n].to_vec();
        let x0 = match equilibrium(model, &EpSetup::new(a1, setup.params.clone(), u0), opts) {
            Err(EngineError::NewtonFailed { iters, .. }) => {
                return Err(bad(format!(
                    "no equilibrium found in {iters} Newton iterations from the slowest point \
                     of the start orbit; start from an orbit that passes near a saddle"
                )));
            }
            r => r?,
        };
        let mut a = vec![0.0; n * n];
        model.eval(&EntryKind::JacU, &x0, &p, &[], &mut a)?;
        let lam = eigenvalues(&a, n)?;
        if let Some(l) = lam.iter().find(|l| l.0.abs() < HYPERBOLIC_TOL) {
            return Err(bad(format!(
                "the equilibrium of the start is not hyperbolic: eigenvalue {} + {}i",
                l.0, l.1
            )));
        }
        let ns = lam.iter().filter(|l| l.0 < 0.0).count();
        if ns == 0 || ns == n {
            return Err(bad(format!(
                "the equilibrium of the start is not a saddle: {ns} stable and {} unstable \
                 eigenvalues",
                n - ns
            )));
        }
        let mut tracker = BasisTracker::new();
        let first = tracker.bases(BasisOf::Transpose, &a, n, ns)?;
        tracker.set(BasisOf::Transpose, first);
        let mut extras = vec![Extra::Other; n];
        extras.extend([Extra::Param(a1), Extra::Param(a2)]);
        let mut outer = CycleProblem::with_extras(model, &setup, extras)?;
        outer.set_mesh(mesh, &setup.profile);
        let mut x = setup.profile.clone();
        x.push(setup.period);
        x.extend_from_slice(&x0);
        x.extend([setup.params[a1], setup.params[a2]]);
        Ok(HomProblem {
            outer,
            x0: x,
            n,
            ns,
            free: start.free,
            period0: setup.period,
            tracker,
            w_prev: [None, None],
            stop: None,
            f: vec![0.0; n],
        })
    }

    /// The start in the original unknowns.
    pub fn start(&self) -> &[f64] {
        &self.x0
    }

    fn ext(&self) -> usize {
        self.outer.profile_len()
    }

    fn check_stop(&self) -> Result<(), EngineError> {
        match &self.stop {
            Some(m) => Err(EngineError::BadProblem(m.clone())),
            None => Ok(()),
        }
    }

    /// A = f_x(x0, p) at x.
    fn saddle_jacobian(&mut self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        let (n, ext) = (self.n, self.ext());
        let mut a = vec![0.0; n * n];
        self.outer
            .eval_at(x, &EntryKind::JacU, &x[ext + 1..ext + 1 + n], &mut a)?;
        Ok(a)
    }

    /// The bases L_s and L_u at x, rotated against the reference bases.
    fn bases(&mut self, x: &[f64]) -> Result<BasisPair, EngineError> {
        let (n, ns) = (self.n, self.ns);
        let a = self.saddle_jacobian(x)?;
        Ok(self.tracker.bases(BasisOf::Transpose, &a, n, ns)?)
    }

    /// The n boundary rows at x and the bases they use.
    fn bc_rows(&mut self, x: &[f64], out: &mut [f64]) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
        let (n, ns, ext) = (self.n, self.ns, self.ext());
        let (ls, lu) = self.bases(x)?;
        let x0 = &x[ext + 1..ext + 1 + n];
        let last = ext - n;
        for r in 0..ns {
            out[r] = (0..n).map(|c| ls[c + n * r] * (x[c] - x0[c])).sum();
        }
        for r in 0..n - ns {
            out[ns + r] = (0..n).map(|c| lu[c + n * r] * (x[last + c] - x0[c])).sum();
        }
        Ok((ls, lu))
    }

    /// The HOM tests at x from the eigenvalues `eigs` of A.
    fn tests_at(&mut self, x: &[f64], eigs: &[(f64, f64)]) -> Result<Vec<f64>, EngineError> {
        let (n, ns, ext) = (self.n, self.ns, self.ext());
        if eigs.len() != n {
            return Ok(vec![f64::NAN; HOM_CURVE_LABELS.len()]);
        }
        let lam = ascending(eigs);
        let a = self.saddle_jacobian(x)?;
        let at = transpose(&a, n);
        let sign = |i: usize, j: usize| match (eigvec(&a, n, lam[i]), eigvec(&a, n, lam[j])) {
            (Ok(vi), Ok(vj)) => {
                if norm1(&vi.0) > norm1(&vj.0) {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => f64::NAN,
        };
        let nu = n - ns;
        let tls = if ns >= 3 {
            sign(ns - 1, ns - 3)
        } else {
            f64::NAN
        };
        let tlu = if nu >= 3 { sign(ns, ns + 2) } else { f64::NAN };
        let period = x[ext];
        let last = ext - n;
        let mut flip = |w: usize, u: &[f64], l: f64, e: f64| -> Result<f64, EngineError> {
            let Some(wv) = signed_eigvec(&at, n, l, self.w_prev[w].as_deref()) else {
                return Ok(f64::NAN);
            };
            self.outer.eval_at(x, &EntryKind::F, u, &mut self.f)?;
            let dot: f64 = self.f.iter().zip(&wv).map(|(a, b)| a * b).sum();
            Ok(dot * (e * period * l / 2.0).exp())
        };
        let (ls, lu) = (lam[ns - 1], lam[ns]);
        let ofs = if is_real(ls) {
            flip(0, &x[last..last + n], ls.0, -1.0)?
        } else {
            f64::NAN
        };
        let ofu = if is_real(lu) {
            flip(1, &x[..n], lu.0, 1.0)?
        } else {
            f64::NAN
        };
        Ok(hom_curve_tests(&lam, ns, tls, tlu, ofs, ofu))
    }
}

impl Problem for HomProblem<'_> {
    fn dim(&self) -> usize {
        self.outer.dim()
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.check_stop()?;
        let (n, ext) = (self.n, self.ext());
        self.outer.residual(x, out)?;
        let top = ext - n;
        self.bc_rows(x, &mut out[top..top + n])?;
        self.outer
            .eval_at(x, &EntryKind::F, &x[ext + 1..ext + 1 + n], &mut self.f)?;
        out[ext + 1..ext + 1 + n].copy_from_slice(&self.f);
        out[ext + 1 + n] = x[ext] - self.period0;
        Ok(())
    }

    /// The u(0) and u(1) columns of the boundary rows are L_s^T and L_u^T. The x0 and
    /// parameter columns are forward differences with the reference bases held, and the T
    /// column is zero.
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.check_stop()?;
        let dim = self.dim();
        if !out.is_empty() && out.len() != (dim - 1) * dim {
            return Err(EngineError::BadProblem("jacobian: wrong length".into()));
        }
        let (n, ns, ext) = (self.n, self.ns, self.ext());
        self.outer.assemble(x)?;
        let mut g0 = vec![0.0; n];
        let (ls, lu) = self.bc_rows(x, &mut g0)?;
        let cols = n + 2;
        let mut fd = vec![0.0; n * cols];
        let mut xs = x.to_vec();
        let mut g1 = vec![0.0; n];
        for i in 0..cols {
            let q = ext + 1 + i;
            let h = FD_STEP * (1.0 + x[q].abs());
            xs[q] = x[q] + h;
            let r = self.bc_rows(&xs, &mut g1);
            xs[q] = x[q];
            r?;
            for row in 0..n {
                fd[row + n * i] = (g1[row] - g0[row]) / h;
            }
        }
        self.outer.read_extras(x);
        let bc = self.outer.bc_mut();
        bc.fill(0.0);
        for r in 0..ns {
            for c in 0..n {
                bc[r + n * c] = ls[c + n * r];
            }
        }
        for r in 0..n - ns {
            for c in 0..n {
                bc[ns + r + n * (n + c)] = lu[c + n * r];
            }
        }
        for i in 0..cols {
            for row in 0..n {
                bc[row + n * (2 * n + 1 + i)] = fd[row + n * i];
            }
        }
        let a = self.saddle_jacobian(x)?;
        let n_p = self.outer.n_p();
        let mut jp = vec![0.0; n * n_p];
        self.outer
            .eval_at(x, &EntryKind::JacP, &x[ext + 1..ext + 1 + n], &mut jp)?;
        let [a1, a2] = self.free;
        let mut row = vec![0.0; dim];
        for c in 0..n {
            row.fill(0.0);
            for c2 in 0..n {
                row[ext + 1 + c2] = a[c + n * c2];
            }
            row[ext + 1 + n] = jp[a1 * n + c];
            row[ext + 2 + n] = jp[a2 * n + c];
            self.outer.set_border_row(1 + c, &row);
        }
        row.fill(0.0);
        row[ext] = 1.0;
        self.outer.set_border_row(n + 1, &row);
        if !out.is_empty() {
            self.outer.write_dense(out);
        }
        Ok(())
    }

    fn jacobian_len(&self) -> usize {
        0
    }

    /// A = f_x(x0, p), so the eigenvalues of a point are those of the saddle.
    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        self.saddle_jacobian(x).map(Some)
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        &HOM_CURVE_LABELS
    }

    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        self.tests_at(x, eigs)
    }

    /// Makes x the reference of the phase row, checks the eigenvalue counts, and keeps the
    /// bases and the left eigenvectors of x for the next step.
    fn accept(&mut self, x: &[f64]) {
        self.outer.accept(x);
        if self.stop.is_some() {
            return;
        }
        let n = self.n;
        let Ok(a) = self.saddle_jacobian(x) else {
            return;
        };
        let Ok(lam) = eigenvalues(&a, n) else {
            return;
        };
        let ns = lam.iter().filter(|l| l.0 < 0.0).count();
        if ns != self.ns || lam.iter().any(|l| l.0 == 0.0) {
            self.stop = Some(format!(
                "the saddle changed from {} stable and {} unstable eigenvalues to {ns} and \
                 {}; the HOM curve ends here",
                self.ns,
                n - self.ns,
                n - ns
            ));
            return;
        }
        if let Ok(pair) = self.bases(x) {
            self.tracker.set(BasisOf::Transpose, pair);
        }
        let lam = ascending(&lam);
        let at = transpose(&a, n);
        for (w, l) in [(0, lam[ns - 1]), (1, lam[ns])] {
            self.w_prev[w] = if is_real(l) {
                signed_eigvec(&at, n, l.0, self.w_prev[w].as_deref())
            } else {
                None
            };
        }
    }

    fn mesh(&self) -> Vec<f64> {
        self.outer.mesh()
    }

    fn adapt(&mut self, x: &mut [f64], v: &mut [f64], failed: bool) -> bool {
        self.outer.adapt(x, v, failed)
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

/// Follows the HOM curve from `start` in the parameters `start.free`. `Options::bounds`
/// bound a2. The start is first corrected at fixed a2. The first tangent points to growing
/// a2, times `Options::direction`.
pub fn continue_hom_curve(
    model: &mut dyn ModelEval,
    start: &HomStart,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    let mut p = HomProblem::new(model, start, opts)?;
    p.outer.adapt_every = opts.mesh_adapt;
    p.outer.set_periodic(false);
    let x0 = p.x0.clone();
    let (k, points) = (p.outer.profile_len(), p.outer.points());
    let hint = "start from an orbit that passes near a saddle or from a cycle of long period";
    run_curve(
        Scaled::new(p, k, points),
        x0,
        None,
        hint,
        opts,
        cancel,
        on_point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nan_at(t: &[f64]) -> Vec<usize> {
        (0..t.len()).filter(|&i| t[i].is_nan()).collect()
    }

    #[test]
    fn a_real_saddle_has_nsa_and_no_nsf() {
        // -2, a - 1, a + 1 at a = 0.1, as on Sandstede's curve.
        let lam = ascending(&[(1.1, 0.0), (-2.0, 0.0), (-0.9, 0.0)]);
        let t = hom_curve_tests(&lam, 2, f64::NAN, f64::NAN, 0.5, 0.25);
        assert!((t[0] - 0.2).abs() < 1e-15);
        assert!((t[1] - 1.1_f64.powi(2)).abs() < 1e-14);
        assert!((t[4] - (-2.0 - 0.9 + 1.1)).abs() < 1e-15);
        assert_eq!((t[8], t[9], t[10], t[11]), (-0.9, 1.1, 0.5, 0.25));
        assert_eq!(nan_at(&t), vec![2, 3, 5, 6, 7]);
    }

    #[test]
    fn bls_changes_sign_through_a_jordan_block() {
        // c +- sqrt(-delta) with c = -0.4, plus -1 (stable) and 1 (unstable).
        for delta in [-0.01_f64, 0.01] {
            let pair = if delta < 0.0 {
                vec![(-0.4 - (-delta).sqrt(), 0.0), (-0.4 + (-delta).sqrt(), 0.0)]
            } else {
                vec![(-0.4, delta.sqrt()), (-0.4, -delta.sqrt())]
            };
            let mut e = pair.clone();
            e.extend([(-1.0, 0.0), (1.0, 0.0)]);
            let lam = ascending(&e);
            let t = hom_curve_tests(&lam, 3, 1.0, f64::NAN, f64::NAN, f64::NAN);
            assert!(
                (t[1] + 4.0 * delta).abs() < 1e-15,
                "delta {delta}: {}",
                t[1]
            );
            if delta > 0.0 {
                assert!(t[0].is_nan() && (t[3] - 0.6).abs() < 1e-15);
            }
            assert!((t[6] - (lam[2].0 + 1.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn unstable_side_tests_use_the_leading_unstable_pair() {
        let lam = ascending(&[(-1.0, 0.0), (0.5, 2.0), (0.5, -2.0), (3.0, 0.0)]);
        let t = hom_curve_tests(&lam, 1, f64::NAN, -1.0, f64::NAN, f64::NAN);
        assert!(t[0].is_nan() && t[1].is_nan() && t[4].is_nan());
        assert!((t[2] + 16.0).abs() < 1e-14);
        assert!((t[3] + 0.5).abs() < 1e-15);
        assert!((t[5] - 0.0).abs() < 1e-15);
        assert!((t[7] + (0.5 - 3.0)).abs() < 1e-15);
        assert_eq!((t[8], t[9]), (-1.0, 0.5));
    }

    #[test]
    fn no_saddle_gives_only_nan() {
        let lam = ascending(&[(-1.0, 0.0), (-2.0, 0.0)]);
        assert!(
            hom_curve_tests(&lam, 2, 1.0, 1.0, 0.0, 0.0)
                .iter()
                .all(|v| v.is_nan())
        );
    }

    #[test]
    fn eigenvectors_keep_their_sign() {
        // A^T = [[-1, 0], [1, 2]] column-major: left eigenvector of -1 is +-(3, -1)/sqrt 10.
        let at = vec![-1.0, 1.0, 0.0, 2.0];
        let w = signed_eigvec(&at, 2, -1.0, None).unwrap();
        assert!(w[0] > 0.0 && (w[0] * w[0] + w[1] * w[1] - 1.0).abs() < 1e-14);
        let neg: Vec<f64> = w.iter().map(|v| -v).collect();
        let w2 = signed_eigvec(&at, 2, -1.0, Some(&neg)).unwrap();
        assert!(w2[0] < 0.0);
        // A^T w = lambda w.
        let aw = [at[0] * w[0] + at[2] * w[1], at[1] * w[0] + at[3] * w[1]];
        assert!((aw[0] + w[0]).abs() < 1e-14 && (aw[1] + w[1]).abs() < 1e-14);
    }
}
