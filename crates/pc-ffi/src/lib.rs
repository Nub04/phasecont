//! C ABI for the MEX gateway: `pc_api_v1()`, the handle registry and the panic barrier.
//!
//! The gateway loads this library and calls everything through the table that
//! `pc_api_v1()` returns (`docs/SPEC.md`, section 10). Every export returns an
//! error code and keeps its message for `last_error`. The `json` module holds the
//! problem parser and the diagnostics writer that `pc-cli` and the C ABI share.
//! The C declarations are in `include/pckern.h`; `tests/header.rs` checks them against `PcApiV1`.

use std::ffi::c_char;
use std::panic::catch_unwind;

mod api;
mod barrier;
pub mod error;
pub mod json;
mod model;
pub mod parse;
mod registry;
mod run;

pub use api::{debug_handle, debug_model_panic};

pub const PC_ABI_VERSION: u32 = 1;

/// The function table. Field order and types are ABI v1 and must match
/// `PcApiV1` in `mex/pckern_gateway.c`.
#[repr(C)]
pub struct PcApiV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub model_from_ode:
        unsafe extern "C" fn(*const c_char, usize, *mut u64, *mut *mut c_char) -> i32,
    pub model_compile: unsafe extern "C" fn(u64, *const c_char) -> i32,
    pub model_rhs:
        unsafe extern "C" fn(u64, *const f64, usize, *const f64, usize, *mut f64, usize) -> i32,
    pub run_start: unsafe extern "C" fn(
        u64,
        *const c_char,
        *const f64,
        usize,
        *const f64,
        usize,
        *mut u64,
    ) -> i32,
    pub run_poll: unsafe extern "C" fn(u64, *mut *mut c_char) -> i32,
    pub run_copy:
        unsafe extern "C" fn(u64, u32, u32, usize, usize, *mut f64, usize, *mut usize) -> i32,
    pub run_cancel: extern "C" fn(u64) -> i32,
    pub free_handle: extern "C" fn(u64) -> i32,
    pub free_string: unsafe extern "C" fn(*mut c_char),
    pub last_error: extern "C" fn() -> *const c_char,
    pub model_info: unsafe extern "C" fn(u64, *mut *mut c_char) -> i32,
    pub model_equilibrium: unsafe extern "C" fn(
        u64,
        *const f64,
        usize,
        *const f64,
        usize,
        *const c_char,
        *mut f64,
        usize,
        *mut f64,
        usize,
    ) -> i32,
    pub debug_panic: unsafe extern "C" fn(i32, *mut u64) -> i32,
    pub version: extern "C" fn() -> *const c_char,
    pub model_rhs_t: unsafe extern "C" fn(
        u64,
        f64,
        *const f64,
        usize,
        *const f64,
        usize,
        *mut f64,
        usize,
    ) -> i32,
}

static API: PcApiV1 = PcApiV1 {
    abi_version: PC_ABI_VERSION,
    struct_size: size_of::<PcApiV1>() as u32,
    model_from_ode: api::model_from_ode,
    model_compile: api::model_compile,
    model_rhs: api::model_rhs,
    run_start: api::run_start,
    run_poll: api::run_poll,
    run_copy: api::run_copy,
    run_cancel: api::run_cancel,
    free_handle: api::free_handle,
    free_string: barrier::free_string,
    last_error: barrier::last_error,
    model_info: api::model_info,
    model_equilibrium: api::model_equilibrium,
    debug_panic: api::debug_panic,
    version: api::version,
    model_rhs_t: api::model_rhs_t,
};

/// The only exported symbol. Returns NULL only if a panic was caught.
#[unsafe(no_mangle)]
pub extern "C" fn pc_api_v1() -> *const PcApiV1 {
    catch_unwind(|| &API as *const PcApiV1).unwrap_or(std::ptr::null())
}
