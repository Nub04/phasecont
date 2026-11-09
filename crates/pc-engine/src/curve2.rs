//! Shared parts of the two-parameter curve problems (LP and Hopf curves): the model with
//! two free parameters, the curve setup and the start tangent.
//!
//! The unknown of a curve is `x = (u, a1, a2)`: the state, then the two free parameters.
//! The run machinery names the last entry `alpha` and the rest `u`, so on a curve
//! `Point.u` is `(state, a1)` and `Point.alpha` is `a2`, and `Options::bounds` applies to
//! `a2`.

use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::At;
use crate::problem::Problem;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{null_right, singular_values};

/// Relative size of the smallest singular value of the curve Jacobian below which the start
/// counts as singular for the curve.
const RANK_TOL: f64 = 1e-8;

/// Where a codim-1 curve starts: the two free parameters, the model parameters and the
/// state of a point on the curve.
#[derive(Clone, Debug)]
pub struct CurveSetup {
    /// Indices of the two continuation parameters among the model parameters. They must
    /// differ.
    pub free: [usize; 2],
    /// All `n_p` parameter values; the engine appends t = 0. The entries at `free` are the
    /// starting values of the two parameters.
    pub params: Vec<f64>,
    /// State of the starting point, which must lie on the curve (an LP point for the LP
    /// curve) to the residual tolerance of the run.
    pub u0: Vec<f64>,
}

impl CurveSetup {
    pub fn new(free: [usize; 2], params: Vec<f64>, u0: Vec<f64>) -> CurveSetup {
        CurveSetup { free, params, u0 }
    }

    /// The starting unknown `(u0, a1, a2)`.
    pub(crate) fn start(&self) -> Vec<f64> {
        let mut x = self.u0.clone();
        x.push(self.params[self.free[0]]);
        x.push(self.params[self.free[1]]);
        x
    }
}

/// A model with two free parameters.
pub(crate) struct Model2<'a> {
    model: &'a mut dyn ModelEval,
    pub(crate) n: usize,
    n_p: usize,
    free: [usize; 2],
    /// Parameter values with t = 0 appended; the entries at `free` follow x.
    p: Vec<f64>,
}

impl<'a> Model2<'a> {
    pub(crate) fn new(
        model: &'a mut dyn ModelEval,
        setup: &CurveSetup,
    ) -> Result<Model2<'a>, EngineError> {
        let (n, n_p) = (model.n(), model.n_p());
        if n == 0 {
            return Err(EngineError::BadProblem("the model has no variables".into()));
        }
        if setup.params.len() != n_p {
            return Err(EngineError::BadProblem(format!(
                "expected {n_p} parameters, got {}",
                setup.params.len()
            )));
        }
        if setup.free[0] == setup.free[1] || setup.free.iter().any(|&f| f >= n_p) {
            return Err(EngineError::BadProblem(
                "the two curve parameters must be different parameters of the model".into(),
            ));
        }
        if setup.u0.len() != n {
            return Err(EngineError::BadProblem(format!(
                "expected {n} state values, got {}",
                setup.u0.len()
            )));
        }
        let mut p = setup.params.clone();
        p.push(0.0);
        Ok(Model2 {
            model,
            n,
            n_p,
            free: setup.free,
            p,
        })
    }

    /// Sets the two parameters from `x = (u, a1, a2)`.
    pub(crate) fn set(&mut self, x: &[f64]) {
        self.set_pair(x[self.n], x[self.n + 1]);
    }

    /// Sets the two free parameters.
    pub(crate) fn set_pair(&mut self, a1: f64, a2: f64) {
        self.p[self.free[0]] = a1;
        self.p[self.free[1]] = a2;
    }

    /// The model at the state `u` with the current parameters, for the normal form code.
    pub(crate) fn at<'b>(&'b mut self, u: &'b [f64]) -> At<'b> {
        At::new(&mut *self.model, self.n, u, self.p.clone())
    }

    pub(crate) fn f(&mut self, u: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.model.eval(&EntryKind::F, u, &self.p, &[], out)
    }

    /// `A = F_u`, n by n, column-major.
    pub(crate) fn jac_u(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut out = vec![0.0; self.n * self.n];
        self.model
            .eval(&EntryKind::JacU, u, &self.p, &[], &mut out)?;
        Ok(out)
    }

    /// `[F_a1 F_a2]`, n by 2, column-major.
    pub(crate) fn f_a(&mut self, u: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut all = vec![0.0; self.n * self.n_p];
        self.model
            .eval(&EntryKind::JacP, u, &self.p, &[], &mut all)?;
        let n = self.n;
        let mut out = Vec::with_capacity(2 * n);
        for f in self.free {
            out.extend_from_slice(&all[f * n..(f + 1) * n]);
        }
        Ok(out)
    }

    /// Unit vector of coordinate `k` of x inside the `Full` slot (u, then all parameters).
    fn unit_full(&self, k: usize) -> Vec<f64> {
        let mut e = vec![0.0; self.n + self.n_p];
        e[if k < self.n {
            k
        } else {
            self.n + self.free[k - self.n]
        }] = 1.0;
        e
    }

    /// `A_z q` for z = coordinate `k` of x.
    pub(crate) fn a_z_q(
        &mut self,
        u: &[f64],
        q: &[f64],
        k: usize,
    ) -> Result<Vec<f64>, EngineError> {
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

    /// Multilinear form `F_uu[a, b]` in the state directions.
    pub(crate) fn b(&mut self, u: &[f64], a: &[f64], b: &[f64]) -> Result<Vec<f64>, EngineError> {
        let mut v = a.to_vec();
        v.extend_from_slice(b);
        let mut out = vec![0.0; self.n];
        self.model
            .eval(&EntryKind::Dir(vec![Slot::U; 2]), u, &self.p, &v, &mut out)?;
        Ok(out)
    }

    /// Multilinear form `F_uuu[a, b, c]` in the state directions.
    pub(crate) fn c(
        &mut self,
        u: &[f64],
        a: &[f64],
        b: &[f64],
        c: &[f64],
    ) -> Result<Vec<f64>, EngineError> {
        let mut v = a.to_vec();
        v.extend_from_slice(b);
        v.extend_from_slice(c);
        let mut out = vec![0.0; self.n];
        self.model
            .eval(&EntryKind::Dir(vec![Slot::U; 3]), u, &self.p, &v, &mut out)?;
        Ok(out)
    }
}

/// Euclidean norm.
pub(crate) fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// First tangent at the start `x` of a curve: the unit null vector of the `(dim-1)` by `dim`
/// Jacobian, with the sign chosen so that its first component of size above `RANK_TOL` is
/// positive. `Options::direction` -1 reverses it in `run_branch`.
///
/// The columns have the units of the unknowns, so their norms can differ by many orders of
/// magnitude (a delay time of 1e3 beside a rate of 1e-3). The rank test and the null vector
/// use the columns divided by their norms, and the vector is scaled back, so the units of
/// the unknowns do not decide whether a start is regular.
pub(crate) fn start_tangent(p: &mut dyn Problem, x: &[f64]) -> Result<Vec<f64>, EngineError> {
    let m = p.dim() - 1;
    let mut jac = vec![0.0; m * (m + 1)];
    p.jacobian(x, &mut jac)?;
    let mut col = vec![1.0; m + 1];
    for (j, c) in col.iter_mut().enumerate() {
        let nrm = norm(&jac[j * m..(j + 1) * m]);
        if nrm.is_finite() && nrm > 0.0 {
            *c = nrm;
            for v in &mut jac[j * m..(j + 1) * m] {
                *v /= nrm;
            }
        }
    }
    let s = singular_values(&jac, m, m + 1)?;
    if s.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::NotFinite("Jacobian at the start".into()));
    }
    if s[m - 1] <= RANK_TOL * s[0] {
        return Err(EngineError::BadProblem(
            "the curve Jacobian is rank deficient at the start, so the curve has no unique \
             tangent there"
                .into(),
        ));
    }
    let mut t = null_right(&jac, m, m + 1, 1)?.swap_remove(0);
    for (c, k) in t.iter_mut().zip(&col) {
        *c /= k;
    }
    let len = norm(&t);
    for c in t.iter_mut() {
        *c /= len;
    }
    let lead = t
        .iter()
        .copied()
        .find(|c| c.abs() > RANK_TOL)
        .unwrap_or(1.0);
    if lead < 0.0 {
        for c in t.iter_mut() {
            *c = -*c;
        }
    }
    Ok(t)
}
