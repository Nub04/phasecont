//! Lazy compilation of model entries, cached per model.

use crate::{DirFn, JitEntries, JitError, PlainFn, compile};
use pc_ir::{EntryKind, IrModel};
use std::collections::HashMap;

/// A compiled entry point. The pointer is valid only while the [`CompiledModel`] that
/// returned it is alive.
#[derive(Clone, Copy)]
pub enum EntryFn {
    Plain(PlainFn),
    Dir(DirFn),
}

impl EntryFn {
    fn addr(&self) -> usize {
        match self {
            EntryFn::Plain(f) => *f as usize,
            EntryFn::Dir(f) => *f as usize,
        }
    }
}

impl std::fmt::Debug for EntryFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryFn::Plain(_) => write!(f, "Plain({:#x})", self.addr()),
            EntryFn::Dir(_) => write!(f, "Dir({:#x})", self.addr()),
        }
    }
}

impl PartialEq for EntryFn {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other) && self.addr() == other.addr()
    }
}

/// A model with its entries compiled on first use. Each kind gets its own
/// [`JitEntries`]; dropping the model frees all of them.
pub struct CompiledModel {
    pub(crate) ir: IrModel,
    pub(crate) sets: Vec<JitEntries>,
    /// The function and the output length of each compiled kind.
    cache: HashMap<EntryKind, (EntryFn, usize)>,
}

impl CompiledModel {
    pub fn new(ir: IrModel) -> CompiledModel {
        CompiledModel {
            ir,
            sets: Vec::new(),
            cache: HashMap::new(),
        }
    }

    pub fn ir(&self) -> &IrModel {
        &self.ir
    }

    /// Compiles `kind` on the first request and returns the cached function afterwards.
    ///
    /// The returned [`EntryFn`] carries no lifetime: it is valid only while this
    /// `CompiledModel` is alive, and calling it afterwards is undefined behavior. A
    /// failed request leaves the cache as it was.
    pub fn entry(&mut self, kind: &EntryKind) -> Result<EntryFn, JitError> {
        self.entry_with_len(kind).map(|(f, _)| f)
    }

    fn entry_with_len(&mut self, kind: &EntryKind) -> Result<(EntryFn, usize), JitError> {
        if let Some(&hit) = self.cache.get(kind) {
            return Ok(hit);
        }
        let e = self.ir.entry(kind).map_err(JitError::Ir)?;
        let set = compile(&[(kind.clone(), &e)], &self.ir.dag)?;
        let f = match kind {
            EntryKind::Dir(_) => set.dir(kind).map(EntryFn::Dir),
            _ => set.plain(kind).map(EntryFn::Plain),
        };
        let Some(f) = f else {
            return Err(JitError::Cranelift("compiled entry missing".into()));
        };
        let hit = (f, e.outputs.len());
        self.sets.push(set);
        self.cache.insert(kind.clone(), hit);
        Ok(hit)
    }

    /// Runs one entry after checking every length. `p` has n_p + 1 values (time last);
    /// `v` is empty for plain kinds and has the packed slot length for `Dir`.
    pub fn call(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), JitError> {
        let (n, n_p) = (self.ir.n(), self.ir.n_p());
        let (f, out_len) = self.entry_with_len(kind)?;
        let v_len = match kind {
            EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
            _ => 0,
        };
        for (name, got, want) in [
            ("x", x.len(), n),
            ("p", p.len(), n_p + 1),
            ("v", v.len(), v_len),
            ("out", out.len(), out_len),
        ] {
            if got != want {
                return Err(JitError::BadLength(format!(
                    "{name} has {got} values, expected {want}"
                )));
            }
        }
        // SAFETY: `f` came from a JitEntries in `self.sets`, which lives as long as
        // `self`. The lengths were checked above against what the entry reads and
        // writes, and `out` is a unique borrow, so it cannot alias x, p or v.
        unsafe {
            match f {
                EntryFn::Plain(f) => f(x.as_ptr(), p.as_ptr(), out.as_mut_ptr()),
                EntryFn::Dir(f) => f(x.as_ptr(), p.as_ptr(), v.as_ptr(), out.as_mut_ptr()),
            }
        }
        Ok(())
    }
}
