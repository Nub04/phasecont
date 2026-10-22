//! Phase 0 spike library, built as pckern.dll for the MEX gateway and as an
//! rlib for spikecli and the tests.
//!
//! This is not the P7a ABI from `docs/SPEC.md` section 10. The gateway gets
//! one table of function pointers from `pc_api_v1()`. Every function returns
//! an error code and keeps the message for `last_error`, strings that Rust
//! returns are freed with `free_string`, and objects travel as u64 handles.

use std::ffi::c_char;

pub mod ffi;
pub mod registry;
pub mod rng;
pub mod s1;
pub mod s2_jit;
pub mod s3_linalg;
pub mod s4_cuda;

use ffi::{barrier, into_c_string, out_arg, str_arg};

pub const ABI_VERSION: u32 = 1;

/// Signature of a compiled right-hand side or Jacobian (SPEC section 4).
pub type RhsFn = unsafe extern "C" fn(x: *const f64, p: *const f64, out: *mut f64);

/// Field order must match `PcApiV1` in `spikes/mex/pckern_gateway.c`.
#[repr(C)]
pub struct PcApiV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub info: extern "C" fn() -> *const c_char,
    pub last_error: extern "C" fn() -> *const c_char,
    pub free_string: unsafe extern "C" fn(*mut c_char),
    pub scale: unsafe extern "C" fn(*const f64, usize, f64, *mut f64) -> i32,
    pub panic: extern "C" fn() -> i32,
    pub panic_in_worker: extern "C" fn() -> i32,
    pub test_handle: unsafe extern "C" fn(*mut u64) -> i32,
    pub free_handle: extern "C" fn(u64) -> i32,
    pub run_json: unsafe extern "C" fn(*const c_char, *const c_char, *mut *mut c_char) -> i32,
    pub bench_extern:
        unsafe extern "C" fn(Option<RhsFn>, Option<RhsFn>, *const c_char, *mut *mut c_char) -> i32,
    pub s3_build: unsafe extern "C" fn(*const c_char, *mut u64, *mut usize, *mut usize) -> i32,
    pub s3_copy:
        unsafe extern "C" fn(u64, usize, usize, *mut f64, *mut f64, *mut f64, *mut f64) -> i32,
    pub s3_solve:
        unsafe extern "C" fn(u64, usize, *const c_char, *mut f64, *mut *mut c_char) -> i32,
}

static API: PcApiV1 = PcApiV1 {
    abi_version: ABI_VERSION,
    struct_size: size_of::<PcApiV1>() as u32,
    info,
    last_error: ffi::last_error,
    free_string: ffi::free_string,
    scale: s1::scale,
    panic: s1::panic_now,
    panic_in_worker: s1::panic_in_worker,
    test_handle: s1::test_handle,
    free_handle: s1::free_handle,
    run_json,
    bench_extern: s2_jit::bench_extern,
    s3_build: s3_linalg::s3_build,
    s3_copy: s3_linalg::s3_copy,
    s3_solve: s3_linalg::s3_solve,
};

/// The only exported symbol. Returns NULL if a panic is caught. It leaves the
/// `last_error` text alone: without a table the gateway cannot read it, and
/// callers fetch the table right before they read `last_error`.
#[unsafe(no_mangle)]
pub extern "C" fn pc_api_v1() -> *const PcApiV1 {
    ffi::catch_panic(|| &API as *const PcApiV1).unwrap_or(std::ptr::null())
}

/// Build stamp, so the gateway can show which DLL it loaded. Returns NULL if
/// a panic is caught, with the message for `last_error`.
extern "C" fn info() -> *const c_char {
    const INFO: &str = concat!(
        "pckern spike abi=1 target=",
        env!("PCKERN_TARGET"),
        " profile=",
        env!("PCKERN_PROFILE"),
        " built=",
        env!("PCKERN_BUILD_STAMP"),
        "\0"
    );
    ffi::ptr_barrier(|| INFO.as_ptr().cast::<c_char>())
}

/// Runs spike `name` with "key=value" arguments and returns a JSON report
/// through `out`, which the caller frees with `free_string`.
///
/// # Safety
/// `name` and `args` must be NULL or NUL-terminated strings, and `out` must
/// be NULL or valid for one write.
unsafe extern "C" fn run_json(
    name: *const c_char,
    args: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let out = unsafe { out_arg(out, "run_json: output") }?;
        *out = std::ptr::null_mut();
        let name = unsafe { str_arg(name, "run_json: name") }?;
        let args = unsafe { str_arg(args, "run_json: args") }?;
        let report = match name {
            "s2" => s2_jit::run(args)?,
            "s3" => s3_linalg::run(args)?,
            "s4" => s4_cuda::run(args)?,
            other => {
                return Err(ffi::bad_arg(format!(
                    "run_json: unknown spike {other:?}; known: s2, s3, s4"
                )));
            }
        };
        *out = into_c_string(&report);
        Ok(())
    })
}
