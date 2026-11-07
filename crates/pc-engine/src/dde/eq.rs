//! Equilibria of a delay equation and the linear equation at one (SPEC section 8d).
//!
//! At an equilibrium every lagged value equals the state, so the hidden parameters of
//! `lower_dde` are set from `x`. The Jacobian of the collapsed field `G(x) = f(x, x, p)`
//! is `A_0` plus the sum of the delay matrices, and `characteristic_roots` takes `A_0`
//! and the delay matrices one by one.

use super::roots::{LinearDde, RootOptions, RootSet, characteristic_roots};
use crate::branch::Options;
use crate::ep::{EpSetup, equilibrium as ep_equilibrium};
use crate::error::EngineError;
use crate::model::{DdeModel, ModelEval};
use pc_ir::{DelaySlot, DelayTime, EntryKind};

/// The delay time of `slot` at the declared parameters `p`.
fn time_of(slot: &DelaySlot, p: &[f64]) -> f64 {
    match slot.time {
        DelayTime::Param(j) => p[j],
        DelayTime::Lit(v) => v,
    }
}

/// Checks the length of `p` and that every delay time given by a parameter is finite and
/// positive.
pub(super) fn check_params(m: &DdeModel, p: &[f64]) -> Result<(), EngineError> {
    if p.len() != m.n_user_params {
        return Err(EngineError::BadProblem(format!(
            "expected {} parameters, got {}",
            m.n_user_params,
            p.len()
        )));
    }
    for s in &m.delays {
        let tau = time_of(s, p);
        if !(tau.is_finite() && tau > 0.0) {
            return Err(EngineError::BadProblem(format!(
                "a delay time is {tau}; it must be finite and positive"
            )));
        }
    }
    Ok(())
}

/// Declared parameters, the lagged values taken from `x`, then `t` = `t`.
fn full_params(m: &DdeModel, x: &[f64], p: &[f64], t: f64) -> Vec<f64> {
    let mut full = Vec::with_capacity(m.n_user_params + m.delays.len() + 1);
    full.extend_from_slice(&p[..m.n_user_params]);
    full.extend(m.delays.iter().map(|s| x[s.state]));
    full.push(t);
    full
}

fn check_state(m: &DdeModel, x: &[f64]) -> Result<(), EngineError> {
    if x.len() != m.eval.n() {
        return Err(EngineError::BadProblem(format!(
            "expected {} state values, got {}",
            m.eval.n(),
            x.len()
        )));
    }
    Ok(())
}

/// The linear delay equation `x' = A_0 x + sum_j A_j x(t - tau_j)` that the model has
/// near `x`, with `p` the declared parameter values. Slots with equal delay times are one
/// term, and the terms go by ascending time. `x` is not checked to be an equilibrium.
pub fn linearize(m: &mut DdeModel, x: &[f64], p: &[f64]) -> Result<LinearDde, EngineError> {
    check_params(m, p)?;
    check_state(m, x)?;
    let n = m.eval.n();
    let n_p = m.eval.n_p();
    let full = full_params(m, x, p, 0.0);
    let mut ju = vec![0.0; n * n];
    let mut jp = vec![0.0; n * n_p];
    m.eval.eval(&EntryKind::JacU, x, &full, &[], &mut ju)?;
    m.eval.eval(&EntryKind::JacP, x, &full, &[], &mut jp)?;
    // The entries are column major and `LinearDde` is row major.
    let mut a0 = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            a0[i * n + k] = ju[i + n * k];
        }
    }
    let times: Vec<f64> = m.delays.iter().map(|s| time_of(s, p)).collect();
    let mut distinct = times.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let mut delays: Vec<(f64, Vec<f64>)> =
        distinct.iter().map(|&t| (t, vec![0.0; n * n])).collect();
    for (s, (slot, tau)) in m.delays.iter().zip(&times).enumerate() {
        let j = distinct
            .binary_search_by(|d| d.total_cmp(tau))
            .expect("every time is in the list");
        let col = m.n_user_params + s;
        for i in 0..n {
            delays[j].1[i * n + slot.state] += jp[i + n * col];
        }
    }
    Ok(LinearDde { dim: n, a0, delays })
}

/// `G(x) = f(x, x, p)` as a model of the declared parameters, so that the equilibrium
/// solver of `ep` runs on it. The Jacobian adds the hidden columns of `JacP` to the
/// columns of the delayed states. Entries other than `F`, `Aux`, `JacU` and `JacP` are
/// not defined on it.
pub(super) struct Collapsed<'a> {
    pub(super) m: &'a mut DdeModel,
}

impl ModelEval for Collapsed<'_> {
    fn n(&self) -> usize {
        self.m.eval.n()
    }

    fn n_p(&self) -> usize {
        self.m.n_user_params
    }

    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        let n_user = self.m.n_user_params;
        if p.len() != n_user + 1 || x.len() != self.m.eval.n() {
            return Err(EngineError::Model(format!(
                "bad length: x has {} values and p has {}",
                x.len(),
                p.len()
            )));
        }
        let full = full_params(self.m, x, &p[..n_user], p[n_user]);
        let n = x.len();
        match kind {
            EntryKind::F | EntryKind::Aux => self.m.eval.eval(kind, x, &full, v, out),
            EntryKind::JacU => {
                let n_p = self.m.eval.n_p();
                let mut jp = vec![0.0; n * n_p];
                self.m.eval.eval(&EntryKind::JacU, x, &full, v, out)?;
                self.m.eval.eval(&EntryKind::JacP, x, &full, v, &mut jp)?;
                for (s, slot) in self.m.delays.iter().enumerate() {
                    for i in 0..n {
                        out[i + n * slot.state] += jp[i + n * (n_user + s)];
                    }
                }
                Ok(())
            }
            EntryKind::JacP => {
                // Column major, so the declared columns come first.
                let n_p = self.m.eval.n_p();
                let mut jp = vec![0.0; n * n_p];
                self.m.eval.eval(&EntryKind::JacP, x, &full, v, &mut jp)?;
                let len = out.len();
                if len != n * n_user {
                    return Err(EngineError::Model(format!(
                        "bad length: out has {len} values, expected {}",
                        n * n_user
                    )));
                }
                out.copy_from_slice(&jp[..len]);
                Ok(())
            }
            EntryKind::Dir(_) => Err(EngineError::Model(
                "directional derivatives are not defined for a delay equation".into(),
            )),
        }
    }
}

/// Newton's method on `G(x) = f(x, x, p) = 0` from `setup.u0`, with the tolerances and the
/// iteration cap of `opts` and `NewtonFailed` otherwise. `setup.params` holds the
/// declared parameters only; `setup.free` is not used.
pub fn equilibrium(
    m: &mut DdeModel,
    setup: &EpSetup,
    opts: &Options,
) -> Result<Vec<f64>, EngineError> {
    check_params(m, &setup.params)?;
    ep_equilibrium(&mut Collapsed { m }, setup, opts)
}

/// The largest absolute entry of `G(x) = f(x, x, p)`, which is NaN when an entry is.
pub fn residual(m: &mut DdeModel, x: &[f64], p: &[f64]) -> Result<f64, EngineError> {
    check_params(m, p)?;
    check_state(m, x)?;
    let full = full_params(m, x, p, 0.0);
    let mut f = vec![0.0; m.eval.n()];
    m.eval.eval(&EntryKind::F, x, &full, &[], &mut f)?;
    Ok(f.iter().fold(0.0, |a: f64, v| {
        if a.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            a.max(v.abs())
        }
    }))
}

/// The characteristic roots of the linear equation at `x` (see [`linearize`]), with the
/// limits of `characteristic_roots`. `x` is not checked to be an equilibrium.
pub fn stability(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    o: &RootOptions,
) -> Result<RootSet, EngineError> {
    let lin = linearize(m, x, p)?;
    characteristic_roots(&lin, o)
}
