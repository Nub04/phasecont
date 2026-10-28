//! The owner of the JIT code, with a guard that frees the code pages.

use crate::{DirFn, JitError, PlainFn, flags};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;
use pc_ir::EntryKind;

#[cfg(test)]
thread_local! {
    /// Number of `free_memory` calls made by `Drop` on this thread.
    pub(crate) static FREED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Largest frame (bytes below the frame pointer) of any function compiled on this thread.
    pub(crate) static MAX_FRAME: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// When set, `compile` fails right after `JITModule::new`.
    pub(crate) static FAIL_AFTER_NEW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Compiled entries. Dropping it frees the code pages and the table copies, so the
/// function pointers from [`JitEntries::plain`] and [`JitEntries::dir`] must not outlive it.
///
/// `compile` creates this value right after `JITModule::new`, so every later error
/// path drops it and frees the module.
pub struct JitEntries {
    pub(crate) module: Option<JITModule>,
    /// One copy of each table the code reads. The code holds their addresses.
    pub(crate) tables: Vec<Box<[f64]>>,
    pub(crate) fns: Vec<(EntryKind, *const u8)>,
}

impl JitEntries {
    pub(crate) fn new() -> Result<JitEntries, JitError> {
        let isa = flags::host_isa()?;
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        for (name, ptr) in pc_ir::math::symbols() {
            jb.symbol(name, ptr);
        }
        Ok(JitEntries {
            module: Some(JITModule::new(jb)),
            tables: Vec::new(),
            fns: Vec::new(),
        })
    }

    /// The function of an `F`, `Aux`, `JacU` or `JacP` entry, or `None` for another kind
    /// or an entry that was not compiled.
    pub fn plain(&self, kind: &EntryKind) -> Option<PlainFn> {
        if matches!(kind, EntryKind::Dir(_)) {
            return None;
        }
        let (_, code) = self.fns.iter().find(|(k, _)| k == kind)?;
        // SAFETY: plain entries were built with the PlainFn signature and the host's
        // default calling convention.
        Some(unsafe { std::mem::transmute::<*const u8, PlainFn>(*code) })
    }

    /// The function of a `Dir` entry, or `None` for another kind or a missing entry.
    pub fn dir(&self, kind: &EntryKind) -> Option<DirFn> {
        if !matches!(kind, EntryKind::Dir(_)) {
            return None;
        }
        let (_, code) = self.fns.iter().find(|(k, _)| k == kind)?;
        // SAFETY: `Dir` entries were built with the DirFn signature.
        Some(unsafe { std::mem::transmute::<*const u8, DirFn>(*code) })
    }
}

impl Drop for JitEntries {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            #[cfg(test)]
            FREED.with(|c| c.set(c.get() + 1));
            // SAFETY: callers of `plain` and `dir` must not call the returned function
            // pointers after the JitEntries is dropped; this frees their code.
            unsafe { m.free_memory() };
        }
    }
}
