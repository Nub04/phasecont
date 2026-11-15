//! Fixed points of a map as roots (SPEC section 8v).
//!
//! A fixed point of `x -> F(x, p)` is a root of `G(x, p) = F(x, p) - x`, so the equilibrium
//! solver, the continuation, the refinement and the LP and BP tests run unchanged on `G`.
//! `G_x = F_x - I` and `G_p = F_p`. A directional entry with one slot subtracts the state
//! part of its direction, and a form with two or more slots is the same for `F` and `G`,
//! since `x` is linear.

use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::{EntryKind, Slot};

/// `G = F - x` over the model of a map `F`.
pub struct MapAsRoot<'a> {
    inner: &'a mut dyn ModelEval,
}

impl<'a> MapAsRoot<'a> {
    pub fn new(inner: &'a mut dyn ModelEval) -> Self {
        MapAsRoot { inner }
    }
}

impl ModelEval for MapAsRoot<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }

    fn n_p(&self) -> usize {
        self.inner.n_p()
    }

    fn map_residual(&self) -> bool {
        true
    }

    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        self.inner.eval(kind, x, p, v, out)?;
        let n = self.inner.n();
        match kind {
            EntryKind::F => {
                for i in 0..n {
                    out[i] -= x[i];
                }
            }
            EntryKind::JacU => {
                for i in 0..n {
                    out[i + n * i] -= 1.0;
                }
            }
            EntryKind::JacP | EntryKind::Aux => {}
            EntryKind::Dir(slots) => match slots.as_slice() {
                // `U` and `Full` both start with the state part of the direction.
                [Slot::U] | [Slot::Full] => {
                    for i in 0..n {
                        out[i] -= v[i];
                    }
                }
                [Slot::P] => {}
                [] | [_, _, ..] => {}
            },
        }
        Ok(())
    }
}
