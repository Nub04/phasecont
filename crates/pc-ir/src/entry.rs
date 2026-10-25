//! Entry kinds shared by DAG entries, the JIT, AOT C code and the f64 fallback.
//!
//! Inputs are `x` (length n), `p` (length n_p + 1, the last slot is `t`) and, for
//! [`EntryKind::Dir`] only, `v`, the slot directions packed in slot order. Outputs:
//! `F` gives n values, `Aux` gives n_aux values, `JacU` gives n*n values with
//! `out[i + n*j] = df_i/du_j`, `JacP` gives n*n_p values in the same layout, and
//! `Dir(slots)` gives n values of `D^k f[(dirs)]` with `k = slots.len()`.

/// What a direction of a [`EntryKind::Dir`] entry covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryKind {
    F,
    Aux,
    JacU,
    JacP,
    /// 1 to 5 slots.
    Dir(Vec<Slot>),
}
