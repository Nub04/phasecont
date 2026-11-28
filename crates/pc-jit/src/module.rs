//! The owner of the JIT code, with a guard that frees the code pages.

use crate::{DirFn, Float, JitError, PlainFn, PlainFnF32, flags};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;
use pc_ir::EntryKind;
use pc_ir::dag::Dag;
use pc_ir::machine::{self, EvalError};
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    /// Largest frame (bytes below the frame pointer) of any function compiled on this thread.
    pub(crate) static MAX_FRAME: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Number of `free_memory` calls made by `Drop` on this thread.
    pub(crate) static FREED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// When set, `compile` fails right after `JITModule::new`.
    pub(crate) static FAIL_AFTER_NEW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// The first error that a call helper met since the last [`clear_call_error`] on this
    /// thread. Compiled code cannot return it, so [`crate::CompiledModel::call`] reads it after
    /// the entry returns.
    static CALL_ERROR: RefCell<Option<EvalError>> = const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Number of times the call helper ran on this thread.
    pub(crate) static HELPER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// When set, the next helper call panics, once.
    pub(crate) static PANIC_IN_HELPER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Forgets an error of an earlier entry call on this thread.
pub(crate) fn clear_call_error() {
    CALL_ERROR.with(|e| *e.borrow_mut() = None);
}

/// The error of the entry call that just ended on this thread, if there was one.
pub(crate) fn take_call_error() -> Option<EvalError> {
    CALL_ERROR.with(|e| e.borrow_mut().take())
}

fn set_call_error(err: EvalError) {
    CALL_ERROR.with(|e| {
        let mut e = e.borrow_mut();
        if e.is_none() {
            *e = Some(err);
        }
    });
}

/// What a call in compiled code needs, owned by the [`JitEntries`] whose code holds its
/// address. The DAG is a snapshot taken at compile time, so later changes to the model's DAG
/// do not reach running code, and it is never written, so any thread may read it.
pub(crate) struct CallImage {
    pub dag: Arc<Dag>,
    /// Lengths of the `x`, `p` and `v` arrays that the entry was compiled for.
    pub nx: usize,
    pub np: usize,
    pub nv: usize,
}

/// A slice of `n` values at `p`, or an empty one when `n` is 0 (the pointer may then be any
/// aligned address).
///
/// # Safety
/// For `n > 0`, `p` must be valid for reads of `n` values.
unsafe fn values<'a>(p: *const f64, n: usize) -> &'a [f64] {
    if n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, n) }
    }
}

/// The helper that compiled code calls for a `Call` node: runs unit `unit` on the `arity`
/// values at `args` and writes its outputs to `out`.
///
/// It returns nothing. On an error, or after an earlier error of the same entry call, every
/// output is NaN and the error waits in [`take_call_error`]. A panic is caught here, since
/// unwinding through compiled code would abort the process.
///
/// # Safety
/// `image` must be the address of a live [`CallImage`], `unit` a unit of its DAG with a body,
/// `args` valid for the unit's arity, `x`, `p` and `v` valid for the lengths in the image and
/// `out` valid for the unit's outputs. Code made by [`crate::compile_with_calls`] guarantees
/// this.
pub(crate) unsafe extern "C" fn pc_jit_call(
    image: *const CallImage,
    unit: u32,
    args: *const f64,
    x: *const f64,
    p: *const f64,
    v: *const f64,
    out: *mut f64,
) {
    // SAFETY: the caller promises that the image is live, and its DAG outlives the call.
    let image = unsafe { &*image };
    let Some((arity, nout)) = image
        .dag
        .unit(unit)
        .map(|u| (u.arity as usize, u.nout as usize))
    else {
        set_call_error(EvalError::Malformed(format!("call of unit {unit}")));
        return;
    };
    // SAFETY: `out` holds the unit's outputs, per the contract above.
    let out = unsafe { std::slice::from_raw_parts_mut(out, nout) };
    if CALL_ERROR.with(|e| e.borrow().is_some()) {
        out.fill(f64::NAN);
        return;
    }
    #[cfg(test)]
    HELPER_CALLS.with(|c| c.set(c.get() + 1));
    // SAFETY: lengths and pointers are as the contract above says.
    let (args, x, p, v) = unsafe {
        (
            values(args, arity),
            values(x, image.nx),
            values(p, image.np),
            values(v, image.nv),
        )
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(test)]
        if PANIC_IN_HELPER.with(|f| f.replace(false)) {
            panic!("forced panic in the call helper");
        }
        machine::run_unit(
            &image.dag,
            unit,
            args,
            x,
            p,
            v,
            out,
            machine::MAX_ACTIVE_CALLS,
        )
    }));
    match r {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            out.fill(f64::NAN);
            set_call_error(e);
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown".into());
            out.fill(f64::NAN);
            set_call_error(EvalError::Malformed(format!(
                "panic in the call helper: {msg}"
            )));
        }
    }
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
    /// The largest frame (bytes below the frame pointer) among these entries.
    pub(crate) frame: u32,
    /// The machine code size, in bytes, summed over these entries.
    pub(crate) code: u32,
    /// The float type of every entry.
    pub(crate) float: Float,
    /// The images that the code of the entries with calls reads. Each sits behind an `Arc`, so
    /// its address stays put while the vector grows.
    pub(crate) images: Vec<Arc<CallImage>>,
}

/// The import name of the call helper, which is not in [`symbols`]: only code with calls
/// imports it.
pub(crate) const CALL_HELPER: &str = "pc_jit_call";

/// The shims that code of this float type imports.
pub(crate) fn symbols(float: Float) -> Vec<(&'static str, *const u8)> {
    match float {
        Float::F64 => pc_ir::math::symbols(),
        Float::F32 => pc_ir::math::symbols_f32(),
    }
}

impl JitEntries {
    pub(crate) fn new(float: Float, calls: bool) -> Result<JitEntries, JitError> {
        let isa = flags::host_isa()?;
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        for (name, ptr) in symbols(float) {
            jb.symbol(name, ptr);
        }
        if calls {
            jb.symbol(CALL_HELPER, pc_jit_call as *const u8);
        }
        Ok(JitEntries {
            module: Some(JITModule::new(jb)),
            tables: Vec::new(),
            fns: Vec::new(),
            frame: 0,
            code: 0,
            float,
            images: Vec::new(),
        })
    }

    /// True when some entry reads a recursive function through the call helper.
    pub fn has_calls(&self) -> bool {
        !self.images.is_empty()
    }

    /// The function of an `F`, `Aux`, `JacU` or `JacP` entry, or `None` for another kind
    /// or an entry that was not compiled.
    /// `None` too for entries compiled in single precision.
    pub fn plain(&self, kind: &EntryKind) -> Option<PlainFn> {
        if self.float != Float::F64 || matches!(kind, EntryKind::Dir(_)) {
            return None;
        }
        let (_, code) = self.fns.iter().find(|(k, _)| k == kind)?;
        // SAFETY: plain entries were built with the PlainFn signature and the host's
        // default calling convention.
        Some(unsafe { std::mem::transmute::<*const u8, PlainFn>(*code) })
    }

    /// The function of a plain entry compiled by [`crate::compile_f32`], or `None` for an
    /// entry of another kind or precision, or one that was not compiled.
    pub fn plain_f32(&self, kind: &EntryKind) -> Option<PlainFnF32> {
        if self.float != Float::F32 || matches!(kind, EntryKind::Dir(_)) {
            return None;
        }
        let (_, code) = self.fns.iter().find(|(k, _)| k == kind)?;
        // SAFETY: single precision entries were built with the PlainFnF32 signature and the
        // host's default calling convention.
        Some(unsafe { std::mem::transmute::<*const u8, PlainFnF32>(*code) })
    }

    /// The largest stack frame, in bytes, of any entry in this value. `max_frame` is a
    /// high-water mark over every compile on the thread; this one belongs to these entries.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// The machine code size, in bytes, summed over the entries in this value.
    pub fn code_size(&self) -> u32 {
        self.code
    }

    /// The function of a `Dir` entry, or `None` for another kind or a missing entry.
    pub fn dir(&self, kind: &EntryKind) -> Option<DirFn> {
        if self.float != Float::F64 || !matches!(kind, EntryKind::Dir(_)) {
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
