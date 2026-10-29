//! The exported `extern "C"` functions behind the `pc_api_v1` table.
//!
//! Every body runs inside `barrier` or `ptr_barrier`, so no panic unwinds into C.
//! The model and run entries are stubs until the tasks that fill them in.

// Every unsafe entry takes raw pointers from the C caller; the contract is in SPEC section 10.
#![allow(clippy::missing_safety_doc)]

use std::ffi::c_char;

use crate::barrier::{barrier, ptr_barrier};
use crate::error::Error;
use crate::registry;

fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("{name} is not implemented yet"))
}

pub unsafe extern "C" fn model_from_ode(
    _text: *const c_char,
    _len: usize,
    _out_model: *mut u64,
    _out_diag: *mut *mut c_char,
) -> i32 {
    barrier(|| Err(not_implemented("model_from_ode")))
}

pub unsafe extern "C" fn model_compile(_model: u64, _mode: *const c_char) -> i32 {
    barrier(|| Err(not_implemented("model_compile")))
}

pub unsafe extern "C" fn model_rhs(
    _model: u64,
    _x: *const f64,
    _nx: usize,
    _p: *const f64,
    _np: usize,
    _out: *mut f64,
    _nout: usize,
) -> i32 {
    barrier(|| Err(not_implemented("model_rhs")))
}

pub unsafe extern "C" fn run_start(
    _model: u64,
    _problem: *const c_char,
    _x0: *const f64,
    _nx: usize,
    _p0: *const f64,
    _np: usize,
    _out_run: *mut u64,
) -> i32 {
    barrier(|| Err(not_implemented("run_start")))
}

pub unsafe extern "C" fn run_poll(_run: u64, _out_json: *mut *mut c_char) -> i32 {
    barrier(|| Err(not_implemented("run_poll")))
}

pub unsafe extern "C" fn run_copy(
    _run: u64,
    _branch: u32,
    _what: u32,
    _from: usize,
    _count: usize,
    _out: *mut f64,
    _cap: usize,
    _written: *mut usize,
) -> i32 {
    barrier(|| Err(not_implemented("run_copy")))
}

pub extern "C" fn run_cancel(_run: u64) -> i32 {
    barrier(|| Err(not_implemented("run_cancel")))
}

pub extern "C" fn free_handle(id: u64) -> i32 {
    barrier(|| registry::remove(id))
}

pub unsafe extern "C" fn model_info(_model: u64, _out_json: *mut *mut c_char) -> i32 {
    barrier(|| Err(not_implemented("model_info")))
}

pub unsafe extern "C" fn model_equilibrium(
    _model: u64,
    _x0: *const f64,
    _nx: usize,
    _p: *const f64,
    _np: usize,
    _opts: *const c_char,
    _out_x: *mut f64,
    _nout_x: usize,
    _out_eig: *mut f64,
    _nout_eig: usize,
) -> i32 {
    barrier(|| Err(not_implemented("model_equilibrium")))
}

/// Test hook. `where_ == 0` panics on the calling thread. `where_ == 1` is for
/// a panic inside a run worker and needs runs, which a later task adds.
///
/// # Safety
/// `run` must be NULL or valid for one `u64` write.
pub unsafe extern "C" fn debug_panic(where_: i32, run: *mut u64) -> i32 {
    barrier(|| match where_ {
        0 => panic!("debug panic"),
        1 => {
            unsafe { crate::barrier::out_ptr(run, "debug_panic: run") }?;
            Err(Error::Internal(
                "debug_panic(1) needs runs; added in Task 7".into(),
            ))
        }
        other => Err(crate::error::bad_arg(format!(
            "debug_panic: where must be 0 or 1, got {other}"
        ))),
    })
}

pub extern "C" fn version() -> *const c_char {
    const VERSION: &str = concat!("pckern ", env!("CARGO_PKG_VERSION"), " abi 1\0");
    ptr_barrier(|| Ok(VERSION.as_ptr().cast::<c_char>()))
}

/// Inserts a dummy entry of kind "debug" so tests can check type confusion
/// and double free. Not part of the C table.
#[doc(hidden)]
pub fn debug_handle() -> u64 {
    registry::insert("debug", ())
}
