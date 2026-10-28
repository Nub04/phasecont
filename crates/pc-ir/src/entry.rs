//! Entry kinds shared by DAG entries, the JIT, AOT C code and the f64 fallback.
//!
//! Inputs are `x` (length n), `p` (length n_p + 1, the last slot is `t`) and, for
//! [`EntryKind::Dir`] only, `v`, the slot directions packed in slot order. Outputs:
//! `F` gives n values, `Aux` gives n_aux values, `JacU` gives n*n values with
//! `out[i + n*j] = df_i/du_j`, `JacP` gives n*n_p values in the same layout, and
//! `Dir(slots)` gives n values of `D^k f[(dirs)]` with `k = slots.len()`.

use crate::dag::NodeId;
use crate::diff::{Wrt, d};
use crate::error::{IrError, IrErrorKind};
use crate::lower::IrModel;
use pc_ode::DUMMY_SPAN;

/// What a direction of a [`EntryKind::Dir`] entry covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    /// The state `u` only (length n).
    U,
    /// The parameters without `t` (length n_p).
    P,
    /// `u` followed by the parameters (length n + n_p).
    Full,
}

impl Slot {
    /// Number of entries this slot takes in `v`.
    pub fn len(self, n: usize, n_p: usize) -> usize {
        match self {
            Slot::U => n,
            Slot::P => n_p,
            Slot::Full => n + n_p,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EntryKind {
    F,
    Aux,
    JacU,
    JacP,
    /// 1 to 5 slots.
    Dir(Vec<Slot>),
}

/// A DAG entry: one output node per entry value, in the layout of the contract above.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub kind: EntryKind,
    pub outputs: Vec<NodeId>,
}

impl IrModel {
    /// Builds (or returns the cached) DAG entry for `kind`. `F` and `Aux` reuse the
    /// lowered right sides; `JacU` and `JacP` differentiate them symbolically, column by
    /// column. `Dir` builds the multilinear form (see `multi`) and refuses a slot count
    /// outside 1..=5 with an `Unsupported` error. The error is `TooLarge` when the DAG
    /// passes its node limit.
    pub fn entry(&mut self, kind: &EntryKind) -> Result<Entry, IrError> {
        if let Some(e) = self.entries.get(kind) {
            return Ok(e.clone());
        }
        let cp = self.dag.checkpoint();
        let outputs = match kind {
            EntryKind::F => self.f.clone(),
            EntryKind::Aux => self.aux.clone(),
            EntryKind::JacU => self.jacobian(self.n(), true),
            EntryKind::JacP => self.jacobian(self.n_p(), false),
            EntryKind::Dir(slots) => {
                if slots.is_empty() || slots.len() > 5 {
                    return Err(IrError::new(
                        IrErrorKind::Unsupported,
                        DUMMY_SPAN,
                        "a directional entry takes 1 to 5 slots",
                    ));
                }
                let (n, n_p) = (self.n(), self.n_p());
                crate::multi::build(&mut self.dag, &mut self.diff_memo, &self.f, n, n_p, slots)
            }
        };
        if self.dag.exceeded() {
            // Put the model back as it was before this request: drop the nodes it added,
            // the flag, and the memo entries that point past them. Later requests then
            // see the same DAG as on a model that never got this one.
            self.dag.restore(&cp);
            let keep = cp.len();
            self.diff_memo
                .retain(|&(k, _), v| (k.0 as usize) < keep && (v.0 as usize) < keep);
            return Err(IrError::new(
                IrErrorKind::TooLarge,
                DUMMY_SPAN,
                "the derivative needs more nodes than the limit",
            ));
        }
        let e = Entry {
            kind: kind.clone(),
            outputs,
        };
        self.entries.insert(kind.clone(), e.clone());
        Ok(e)
    }

    /// `n` rows by `cols` columns, column major: `out[i + n*j] = df_i/dw_j`.
    fn jacobian(&mut self, cols: usize, wrt_u: bool) -> Vec<NodeId> {
        let n = self.n();
        let mut out = Vec::with_capacity(n * cols);
        for j in 0..cols {
            let w = if wrt_u {
                Wrt::U(j as u32)
            } else {
                Wrt::P(j as u32)
            };
            for i in 0..n {
                let fi = self.f.get(i).copied();
                let di = match fi {
                    Some(fi) => d(&mut self.dag, &mut self.diff_memo, fi, w),
                    None => self.dag.konst(f64::NAN),
                };
                out.push(di);
            }
        }
        out
    }
}
