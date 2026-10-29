//! The exported `extern "C"` functions behind the `pc_api_v1` table.
//!
//! Every body runs inside `barrier` or `ptr_barrier`, so no panic unwinds into C.
//! The run entries are stubs until the task that fills them in.

// Every unsafe entry takes raw pointers from the C caller; the contract is in SPEC section 10.
#![allow(clippy::missing_safety_doc)]

use std::ffi::c_char;
use std::sync::Arc;

use pc_engine::{EpSetup, Evaluator, Options};
use serde_json::{Value, json};

use crate::barrier::{
    barrier, bytes_arg, f64_arg, f64_out, into_c_string, out_ptr, ptr_barrier, str_arg,
};
use crate::error::{Error, Result, bad_arg};
use crate::json::problem::parse_options;
use crate::model::ModelHandle;
use crate::parse::{MAX_MODEL_TEXT, parse_model};
use crate::registry::{self, Handle};

fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("{name} is not implemented yet"))
}

fn model_handle(id: u64) -> Result<Arc<ModelHandle>> {
    registry::get::<ModelHandle>(id)
}

fn expect_len(what: &str, want: usize, got: usize) -> Result<()> {
    if want == got {
        Ok(())
    } else {
        Err(bad_arg(format!(
            "{what}: expected {want} values, got {got}"
        )))
    }
}

/// Parses a model. `out_model` is 0 and `out_diag` is NULL unless the call
/// sets them. `out_diag` may be NULL when the caller does not want diagnostics;
/// otherwise it gets a JSON string on success and on a model error, which the
/// caller frees with `free_string`.
pub unsafe extern "C" fn model_from_ode(
    text: *const c_char,
    len: usize,
    out_model: *mut u64,
    out_diag: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let out_model = unsafe { out_ptr(out_model, "model") }?;
        *out_model = 0;
        let out_diag = if out_diag.is_null() {
            None
        } else {
            let slot = unsafe { out_ptr(out_diag, "diag_json") }?;
            *slot = std::ptr::null_mut();
            Some(slot)
        };
        if len > MAX_MODEL_TEXT {
            return Err(bad_arg("model text is over 16 MiB"));
        }
        let bytes = unsafe { bytes_arg(text, len, "text") }?;
        let text =
            String::from_utf8(bytes.to_vec()).map_err(|_| bad_arg("model text is not UTF-8"))?;
        let outcome = parse_model(text)?;
        if let Some(slot) = out_diag {
            let s = serde_json::to_string(&outcome.diag)
                .map_err(|e| Error::Internal(format!("diagnostics JSON: {e}")))?;
            *slot = into_c_string(&s);
        }
        let parsed = outcome.result?;
        let handle = ModelHandle::new(parsed.model, parsed.info, parsed.names)?;
        *out_model = registry::insert(handle);
        Ok(())
    })
}

/// Reads the `model_compile` options: NULL or an object with only `evaluator`.
fn evaluator_option(opts: *const c_char) -> Result<Evaluator> {
    if opts.is_null() {
        return Ok(Evaluator::Jit);
    }
    let text = unsafe { str_arg(opts, "options") }?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| bad_arg(format!("options: not valid JSON ({e})")))?;
    let map = v
        .as_object()
        .ok_or_else(|| bad_arg("options: expected a JSON object"))?;
    if let Some(key) = map.keys().find(|k| k.as_str() != "evaluator") {
        return Err(bad_arg(format!(
            "{key}: unknown key; the only key is \"evaluator\""
        )));
    }
    match map.get("evaluator") {
        None => Ok(Evaluator::Jit),
        Some(Value::String(s)) if s == "jit" => Ok(Evaluator::Jit),
        Some(Value::String(s)) if s == "fallback" => Ok(Evaluator::Fallback),
        Some(_) => Err(bad_arg("evaluator: expected \"jit\" or \"fallback\"")),
    }
}

pub unsafe extern "C" fn model_compile(model: u64, opts: *const c_char) -> i32 {
    barrier(|| {
        let h = model_handle(model)?;
        let which = evaluator_option(opts)?;
        h.compile(which)
    })
}

pub unsafe extern "C" fn model_rhs(
    model: u64,
    x: *const f64,
    nx: usize,
    p: *const f64,
    np: usize,
    out: *mut f64,
    nout: usize,
) -> i32 {
    barrier(|| {
        let h = model_handle(model)?;
        expect_len("x", h.n(), nx)?;
        expect_len("p", h.n_p(), np)?;
        expect_len("out", h.n(), nout)?;
        let x = unsafe { f64_arg(x, nx, "x") }?.to_vec();
        let p = unsafe { f64_arg(p, np, "p") }?.to_vec();
        let out = unsafe { f64_out(out, nout, "out") }?;
        out.copy_from_slice(&h.rhs(x, p)?);
        Ok(())
    })
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

pub unsafe extern "C" fn model_info(model: u64, out_json: *mut *mut c_char) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(out_json, "info_json") }?;
        *slot = std::ptr::null_mut();
        let h = model_handle(model)?;
        let evaluator = h.evaluator().map(|e| match e {
            Evaluator::Jit => "jit",
            Evaluator::Fallback => "fallback",
        });
        let names = &h.names;
        let v = json!({
            "n": h.n(),
            "n_p": h.n_p(),
            "variables": names.states,
            "parameters": names.parameters,
            "values": names.par_values,
            "init": names.init,
            "aux": names.aux,
            "autonomous": names.autonomous,
            "evaluator": evaluator,
            "jit_max_frame": h.max_frame(),
        });
        let s =
            serde_json::to_string(&v).map_err(|e| Error::Internal(format!("info JSON: {e}")))?;
        *slot = into_c_string(&s);
        Ok(())
    })
}

/// Reads the options object of `model_equilibrium`; NULL means the defaults.
fn engine_options(opts: *const c_char) -> Result<Options> {
    if opts.is_null() {
        return Ok(Options::default());
    }
    let text = unsafe { str_arg(opts, "options") }?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| Error::BadProblem(format!("options: not valid JSON ({e})")))?;
    parse_options(&v).map_err(Error::BadProblem)
}

/// True when the byte ranges of two non-NULL output buffers share a byte.
fn overlaps(a: *const f64, a_len: usize, b: *const f64, b_len: usize) -> bool {
    if a.is_null() || b.is_null() || a_len == 0 || b_len == 0 {
        return false;
    }
    let (a, b) = (a as usize, b as usize);
    let (a_end, b_end) = (
        a.saturating_add(a_len.saturating_mul(size_of::<f64>())),
        b.saturating_add(b_len.saturating_mul(size_of::<f64>())),
    );
    a < b_end && b < a_end
}

pub unsafe extern "C" fn model_equilibrium(
    model: u64,
    u0: *const f64,
    nu: usize,
    p: *const f64,
    np: usize,
    opts: *const c_char,
    u_out: *mut f64,
    nu_out: usize,
    eig_out: *mut f64,
    neig_out: usize,
) -> i32 {
    barrier(|| {
        let h = model_handle(model)?;
        let n = h.n();
        expect_len("u0", n, nu)?;
        expect_len("p", h.n_p(), np)?;
        expect_len("u_out", n, nu_out)?;
        expect_len("eig_out", 2 * n, neig_out)?;
        if overlaps(u_out, nu_out, eig_out, neig_out) {
            return Err(bad_arg("u_out and eig_out overlap"));
        }
        let u0 = unsafe { f64_arg(u0, nu, "u0") }?.to_vec();
        let p = unsafe { f64_arg(p, np, "p") }?.to_vec();
        let opts = engine_options(opts)?;
        let u_out = unsafe { f64_out(u_out, nu_out, "u_out") }?;
        let eig_out = unsafe { f64_out(eig_out, neig_out, "eig_out") }?;
        let r = h.equilibrium(EpSetup::new(0, p, u0), opts)?;
        u_out.copy_from_slice(&r.u);
        for (i, (re, im)) in r.eig.iter().enumerate() {
            eig_out[i] = *re;
            eig_out[n + i] = *im;
        }
        Ok(())
    })
}

/// Test hook, not in the table: panics on the model thread of `model`.
#[doc(hidden)]
pub fn debug_model_panic(model: u64) -> i32 {
    barrier(|| model_handle(model)?.debug_panic())
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

struct DebugEntry;

impl Handle for DebugEntry {
    const KIND: &'static str = "debug";
}

/// Inserts a dummy entry of kind "debug" so tests can check type confusion
/// and double free. Not part of the C table.
#[doc(hidden)]
pub fn debug_handle() -> u64 {
    registry::insert(DebugEntry)
}
