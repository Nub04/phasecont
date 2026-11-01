//! Curves of cycles in two parameters (SPEC §8): the start shared by the LPC, PD and NS
//! curves and the run on their problems.
//!
//! The unknowns are x = (u_0, ..., u_{Nm}, T, a1, ..., a2): the orbit at the base points,
//! the period, the first free parameter, the unknowns of the curve itself (kappa on an NS
//! curve) and the second free parameter, which is `Point::alpha`. The profile is scaled as
//! on a cycle branch, so the step length does not grow with the mesh.

use super::basis::CollocationBasis;
use super::continuation::{Scaled, fixed_alpha, floquet_warning, unscale_point};
use super::mesh;
use super::problem::{CycleSetup, check_dims, try_zeros};
use crate::branch::{Branch, Options, Point};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::palc::{run_branch, tangent_or_kernel};
use crate::problem::Problem;
use std::sync::atomic::{AtomicBool, Ordering};

/// A start of a curve of cycles at an LPC, PD or NS point of a cycle branch.
#[derive(Clone, Debug)]
pub struct CycleCurveStart {
    /// Indices of the two free parameters [a1, a2] among the model parameters. a1 is the
    /// parameter of the cycle branch the point is on, and a2 is `Point::alpha` on the
    /// curve.
    pub free: [usize; 2],
    /// All `n_p` parameter values at the point.
    pub params: Vec<f64>,
    /// The orbit at the base points of `mesh`, point-major, as in `Point::u` without the
    /// period.
    pub orbit: Vec<f64>,
    /// The period T at the point.
    pub period: f64,
    /// The mesh of `orbit`, strictly increasing from 0 to 1.
    pub mesh: Vec<f64>,
    /// Number of mesh intervals N (NTST) of the curve, at least 1.
    pub ntst: usize,
    /// Number of collocation points per interval m (NCOL) of the curve, in
    /// [`NCOL_RANGE`](super::basis::NCOL_RANGE).
    pub ncol: usize,
}

impl CycleCurveStart {
    /// The start from `point`, a special point of a cycle branch in parameter `free[0]`,
    /// with the other parameters from `params`.
    pub fn from_point(
        free: [usize; 2],
        params: &[f64],
        point: &Point,
        ntst: usize,
        ncol: usize,
    ) -> CycleCurveStart {
        let (orbit, period) = match point.u.split_last() {
            Some((t, orbit)) => (orbit.to_vec(), *t),
            None => (Vec::new(), f64::NAN),
        };
        let mut params = params.to_vec();
        if let Some(a) = params.get_mut(free[0]) {
            *a = point.alpha;
        }
        CycleCurveStart {
            free,
            params,
            orbit,
            period,
            mesh: point.mesh.clone(),
            ntst,
            ncol,
        }
    }

    /// Checks the start against the model and returns the setup on NTST intervals of
    /// NCOL points in parameter a1, and its mesh. The orbit is used as it is when its
    /// mesh has these sizes. Otherwise it is interpolated onto NTST intervals whose mesh
    /// points are evenly spaced in the index of the old mesh.
    pub(super) fn setup(
        &self,
        model: &dyn ModelEval,
    ) -> Result<(CycleSetup, Vec<f64>), EngineError> {
        let n = model.n();
        let bad = |m: String| Err(EngineError::BadProblem(m));
        let [a1, a2] = self.free;
        check_dims(model, a1, &self.params, self.ntst, self.ncol)?;
        if a2 >= model.n_p() {
            return bad(format!("parameter index {a2} is out of range"));
        }
        if a1 == a2 {
            return bad("the two free parameters must differ".into());
        }
        let nint = self.mesh.len().saturating_sub(1);
        if nint == 0
            || self.mesh[0] != 0.0
            || self.mesh[nint] != 1.0
            || !self.mesh.windows(2).all(|w| w[1] > w[0])
        {
            return bad("the mesh of the start must rise strictly from 0 to 1".into());
        }
        let pts = self.orbit.len() / n;
        if !self.orbit.len().is_multiple_of(n) || pts < 2 || !(pts - 1).is_multiple_of(nint) {
            return bad(format!(
                "expected the orbit at N m + 1 points of {n} values on {nint} mesh intervals, \
                 got {} values",
                self.orbit.len()
            ));
        }
        if self
            .orbit
            .iter()
            .chain(&self.params)
            .any(|v| !v.is_finite())
        {
            return bad("the start has NaN or infinite values".into());
        }
        if !(self.period.is_finite() && self.period > 0.0) {
            return bad(format!(
                "the period must be finite and positive, got {}",
                self.period
            ));
        }
        let old_m = (pts - 1) / nint;
        if nint == self.ntst && old_m == self.ncol {
            let setup = CycleSetup::new(
                a1,
                self.params.clone(),
                self.ntst,
                self.ncol,
                self.orbit.clone(),
                self.period,
            );
            return Ok((setup, self.mesh.clone()));
        }
        // The sizes come from NTST and NCOL, so this fails with an error, not an abort.
        let mut new = try_zeros(self.ntst.checked_add(1))?;
        for (i, t) in new.iter_mut().enumerate() {
            let r = i as f64 * nint as f64 / self.ntst as f64;
            let j = (r.floor() as usize).min(nint - 1);
            *t = self.mesh[j] + (r - j as f64) * (self.mesh[j + 1] - self.mesh[j]);
        }
        new[0] = 0.0;
        new[self.ntst] = 1.0;
        let basis = CollocationBasis::new(old_m)?;
        let u = mesh::interpolate(&basis, n, &self.mesh, &new, self.ncol, &self.orbit)?;
        let setup = CycleSetup::new(
            a1,
            self.params.clone(),
            self.ntst,
            self.ncol,
            u,
            self.period,
        );
        Ok((setup, new))
    }
}

/// Runs a curve from `x0`, in the original unknowns, on the scaled problem `p`. The start
/// is corrected at fixed a2 first. A correction that fails, hits a singular matrix or
/// leaves the finite numbers is a start off the curve, such as a special point of another
/// type, and returns `BadProblem`. The first tangent is bordered by the a2 direction, or
/// by the next unknown when that bordered matrix is singular, and points to growing a2,
/// times the sign of `Options::direction`.
pub(super) fn run_curve<P: Problem>(
    mut p: Scaled<P>,
    mut x0: Vec<f64>,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    let dim = p.dim();
    p.scale(&mut x0);
    let hint = "start from a matching special point of a cycle branch";
    let x0 = match fixed_alpha(&mut p, x0, opts, cancel) {
        Err(EngineError::NewtonFailed { iters, .. }) => {
            return Err(EngineError::BadProblem(format!(
                "the start did not converge onto the curve in {iters} Newton iterations; {hint}"
            )));
        }
        Err(e @ (EngineError::Singular(_) | EngineError::NotFinite(_))) => {
            return Err(EngineError::BadProblem(format!(
                "the start did not converge onto the curve ({e}); {hint}"
            )));
        }
        r => r?,
    };
    let mut border = vec![0.0; dim];
    border[dim - 1] = 1.0;
    let mut t = match tangent_or_kernel(&mut p, &x0, &border) {
        Err(EngineError::Singular(_) | EngineError::NotFinite(_)) => {
            border.fill(0.0);
            border[dim - 2] = 1.0;
            tangent_or_kernel(&mut p, &x0, &border)?.0
        }
        r => r?.0,
    };
    if t[dim - 1] < 0.0 {
        for v in t.iter_mut() {
            *v = -*v;
        }
    }
    let (k, s) = (p.k, p.s);
    let mut seen = |pt: &Point| {
        let mut q = pt.clone();
        unscale_point(&mut q, k, s);
        on_point(&q);
    };
    let mut branch = run_branch(&mut p, &x0, Some(&t), opts, false, cancel, &mut seen)?;
    for pt in branch.points.iter_mut() {
        unscale_point(pt, k, s);
    }
    for sp in branch.specials.iter_mut() {
        unscale_point(&mut sp.point, k, s);
    }
    floquet_warning(&mut branch);
    Ok(branch)
}
