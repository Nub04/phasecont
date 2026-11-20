//! The exported `extern "C"` functions behind the `pc_api_v1` table.
//!
//! Every body runs inside `barrier` or `ptr_barrier`, so no panic unwinds into C.
//!
// Every unsafe entry takes raw pointers from the C caller; the contract is in SPEC section 10.
#![allow(clippy::missing_safety_doc)]

use std::ffi::c_char;
use std::path::PathBuf;
use std::sync::Arc;

use pc_engine::{EngineError, EpSetup, Evaluator, Options, Run};
use serde_json::{Value, json};

use crate::barrier::{
    barrier, bytes_arg, f64_arg, f64_out, into_c_string, out_ptr, ptr_barrier, str_arg,
};
use crate::ensemble::{self, EnsHandle};
use crate::error::{Error, Result, bad_arg};
use crate::import;
use crate::json::problem::{
    Kind, Mode, map_type_runs, parse_iterate, parse_options, parse_problem,
};
use crate::latex;
use crate::model::{Build, DELAY_MODEL, ModelHandle, NOT_COMPILED};
use crate::parse::{MAX_MODEL_TEXT, parse_model};
use crate::registry::{self, Handle};
use crate::run::{RunHandle, pack_row_aux, poll_json, stride_aux};

fn model_handle(id: u64) -> Result<Arc<ModelHandle>> {
    registry::get::<ModelHandle>(id)
}

/// Length of a buffer of real and imaginary parts of n eigenvalues, 2n, or an error when
/// 2n does not fit in usize.
fn eig_len(n: usize) -> Result<usize> {
    n.checked_mul(2)
        .ok_or_else(|| bad_arg(format!("{n} eigenvalues do not fit in one buffer")))
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
        let out_diag = if out_diag.is_null() {
            None
        } else {
            let slot = unsafe { out_ptr(out_diag, "diag_json") }?;
            *slot = std::ptr::null_mut();
            Some(slot)
        };
        let out_model = unsafe { out_ptr(out_model, "model") }?;
        *out_model = 0;
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
        let handle = ModelHandle::new(parsed.model, parsed.info, parsed.names, parsed.ir)?;
        *out_model = registry::insert(handle);
        Ok(())
    })
}

/// What `model_compile` reads from its options.
struct CompileOptions {
    /// None for `"evaluator":"none"`: write the C file and build no evaluator.
    evaluator: Option<Build>,
    emit_c: Option<PathBuf>,
}

fn string_option<'a>(
    map: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>> {
    match map.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(bad_arg(format!("{key}: expected a string"))),
    }
}

/// Reads the `model_compile` options: NULL or an object with the keys `evaluator`
/// ("jit", "fallback", "aot" or "none"), `aot_lib` (required by "aot") and `emit_c` (a
/// folder). "none" needs `emit_c`: it writes the C file and builds no evaluator, and it
/// leaves the evaluator of an already compiled handle as it is.
fn compile_options(opts: *const c_char) -> Result<CompileOptions> {
    let mut out = CompileOptions {
        evaluator: Some(Build::Jit),
        emit_c: None,
    };
    if opts.is_null() {
        return Ok(out);
    }
    let text = unsafe { str_arg(opts, "options") }?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| bad_arg(format!("options: not valid JSON ({e})")))?;
    let map = v
        .as_object()
        .ok_or_else(|| bad_arg("options: expected a JSON object"))?;
    if let Some(key) = map
        .keys()
        .find(|k| !["evaluator", "aot_lib", "emit_c"].contains(&k.as_str()))
    {
        return Err(bad_arg(format!(
            "{key}: unknown key; the keys are \"evaluator\", \"aot_lib\" and \"emit_c\""
        )));
    }
    let lib = string_option(map, "aot_lib")?;
    out.emit_c = string_option(map, "emit_c")?.map(PathBuf::from);
    out.evaluator = match string_option(map, "evaluator")? {
        None | Some("jit") => Some(Build::Jit),
        Some("fallback") => Some(Build::Fallback),
        Some("aot") => {
            Some(Build::Aot(PathBuf::from(lib.ok_or_else(|| {
                bad_arg("evaluator \"aot\" needs \"aot_lib\"")
            })?)))
        }
        Some("none") => {
            if out.emit_c.is_none() {
                return Err(bad_arg("evaluator \"none\" needs \"emit_c\""));
            }
            None
        }
        Some(_) => {
            return Err(bad_arg(
                "evaluator: expected \"jit\", \"fallback\", \"aot\" or \"none\"",
            ));
        }
    };
    if lib.is_some() && !matches!(out.evaluator, Some(Build::Aot(_))) {
        return Err(bad_arg("aot_lib: only valid with evaluator \"aot\""));
    }
    Ok(out)
}

pub unsafe extern "C" fn model_compile(model: u64, opts: *const c_char) -> i32 {
    barrier(|| {
        let h = model_handle(model)?;
        let o = compile_options(opts)?;
        if let Some(dir) = o.emit_c {
            h.emit_c(dir)?;
        }
        match o.evaluator {
            Some(build) => h.compile(build),
            None => Ok(()),
        }
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
    barrier(|| unsafe { rhs_at(model, None, x, nx, p, np, out, nout) })
}

pub unsafe extern "C" fn model_rhs_t(
    model: u64,
    t: f64,
    x: *const f64,
    nx: usize,
    p: *const f64,
    np: usize,
    out: *mut f64,
    nout: usize,
) -> i32 {
    barrier(|| unsafe { rhs_at(model, Some(t), x, nx, p, np, out, nout) })
}

/// The body of `model_rhs` (`t` is None: time 0, refused for a model that uses t) and
/// `model_rhs_t`.
#[allow(clippy::too_many_arguments)]
unsafe fn rhs_at(
    model: u64,
    t: Option<f64>,
    x: *const f64,
    nx: usize,
    p: *const f64,
    np: usize,
    out: *mut f64,
    nout: usize,
) -> Result<()> {
    let h = model_handle(model)?;
    h.ode_only()?;
    if t.is_none() && !h.names.autonomous {
        return Err(Error::NonAutonomous(
            "the model depends on t; call model_rhs_t".into(),
        ));
    }
    if let Some(t) = t
        && !t.is_finite()
    {
        return Err(bad_arg(format!("t: expected a finite value, got {t}")));
    }
    expect_len("x", h.n(), nx)?;
    expect_len("p", h.n_p(), np)?;
    expect_len("out", h.n(), nout)?;
    let x = unsafe { f64_arg(x, nx, "x") }?.to_vec();
    let p = unsafe { f64_arg(p, np, "p") }?.to_vec();
    let out = unsafe { f64_out(out, nout, "out") }?;
    out.copy_from_slice(&h.rhs(x, p, t.unwrap_or(0.0))?);
    Ok(())
}

pub unsafe extern "C" fn run_start(
    model: u64,
    problem_json: *const c_char,
    u0: *const f64,
    nu: usize,
    p: *const f64,
    np: usize,
    out_run: *mut u64,
) -> i32 {
    barrier(|| {
        let out_run = unsafe { out_ptr(out_run, "run") }?;
        *out_run = 0;
        let h = model_handle(model)?;
        // A map takes ITER and EP, and LP, PD and NS without a cycle key. Every other
        // problem, and a problem that is not JSON, is refused before anything else, so a handle
        // that was not compiled gets the same error (SPEC 8u, 8v, 8w).
        if h.map {
            let taken = unsafe { str_arg(problem_json, "problem_json") }
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(t).ok())
                .is_some_and(|v| map_type_runs(&v));
            if !taken {
                h.not_map()?;
            }
        }
        let evaluator = h.evaluator().ok_or_else(|| bad_arg(NOT_COMPILED))?;
        let text = unsafe { str_arg(problem_json, "problem_json") }?;
        let v: Value = serde_json::from_str(text)
            .map_err(|e| Error::BadProblem(format!("problem_json: {e}")))?;
        // A delay model takes the types "EP", "H" and "LC". A type that names another run is
        // refused with the same text as the ODE-only entries; anything else is left to the
        // parser (SPEC 8s).
        if h.delay && v["type"].as_str().is_some_and(is_other_run_type) {
            return Err(bad_arg(DELAY_MODEL));
        }
        let mut problem = parse_problem(&v, &h.ir, Mode::Abi).map_err(Error::BadProblem)?;
        problem.check_u0_len(h.n(), nu).map_err(bad_arg)?;
        expect_len("params", h.n_p(), np)?;
        problem.setup.u0 = unsafe { f64_arg(u0, nu, "u0") }?.to_vec();
        problem.setup.params = unsafe { f64_arg(p, np, "params") }?.to_vec();
        // Only an IVP run takes the time from the integration, and an ITER run takes the
        // iterate number as t.
        if !h.names.autonomous && !matches!(problem.kind, Kind::Ivp | Kind::Iter) {
            return Err(Error::NonAutonomous(EngineError::NonAutonomous.to_string()));
        }
        let kind = problem.kind;
        // The aux count of the model, or 0 when the key `aux` is off (round E, SPEC 10).
        let naux = problem.n_aux;
        let spec = problem.into_spec(h.model.clone(), h.info.clone(), evaluator);
        let run = Run::start(spec);
        *out_run = registry::insert(RunHandle {
            run,
            kind,
            n: h.n(),
            naux,
        });
        Ok(())
    })
}

/// The run types that a delay model does not take: every `type` of the problem JSON but `"EP"`,
/// `"H"` and `"LC"`.
fn is_other_run_type(t: &str) -> bool {
    matches!(t, "LP" | "LPC" | "PD" | "NS" | "HOM" | "IVP" | "ITER")
}

fn run_handle(id: u64) -> Result<Arc<RunHandle>> {
    registry::get::<RunHandle>(id)
}

pub unsafe extern "C" fn run_poll(run: u64, status_json: *mut *mut c_char) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(status_json, "status_json") }?;
        *slot = std::ptr::null_mut();
        let h = run_handle(run)?;
        let v = poll_json(&h, &h.run.poll())?;
        let s =
            serde_json::to_string(&v).map_err(|e| Error::Internal(format!("status JSON: {e}")))?;
        *slot = into_c_string(&s);
        Ok(())
    })
}

pub unsafe extern "C" fn run_copy(
    run: u64,
    branch: u32,
    what: u32,
    start: usize,
    count: usize,
    buf: *mut f64,
    buf_len: usize,
    out_rows: *mut usize,
) -> i32 {
    barrier(|| {
        let out_rows = unsafe { out_ptr(out_rows, "rows") }?;
        *out_rows = 0;
        let h = run_handle(run)?;
        if what > 1 {
            return Err(bad_arg(format!(
                "what: expected 0 (points) or 1 (specials), got {what}"
            )));
        }
        let branches = h.run.poll().points.len();
        if branch as usize >= branches {
            return Err(bad_arg(format!(
                "branch: the run has {branches} branches, got {branch}"
            )));
        }
        let stride = stride_aux(h.kind, h.n, h.naux);
        let need = count
            .checked_mul(stride)
            .ok_or_else(|| bad_arg(format!("count: {count} rows is too many")))?;
        // An exact length catches a caller that uses the wrong stride.
        if buf_len != need {
            return Err(bad_arg(format!(
                "buf: expected count * stride = {need} values with stride {stride}, got {buf_len}"
            )));
        }
        let buf = unsafe { f64_out(buf, buf_len, "buf") }?;
        if what == 0
            && let Some(rs) = h.run.row_stride(branch as usize)
        {
            // An IVP or ITER branch keeps flat rows of [alpha, x, aux] (SPEC 8af-1).
            if rs != stride {
                return Err(Error::Internal(format!(
                    "run_copy: the engine rows have {rs} values, the handle expects {stride}"
                )));
            }
            *out_rows = h
                .run
                .copy_rows(branch as usize, start, count, buf)
                .map_err(|e| Error::Internal(format!("run_copy: {e}")))?;
            return Ok(());
        }
        let points = if what == 0 {
            h.run.points(branch as usize, start, count)
        } else {
            h.run
                .specials(branch as usize)
                .into_iter()
                .skip(start)
                .take(count)
                .map(|s| s.point)
                .collect()
        };
        for (row, point) in buf.chunks_exact_mut(stride).zip(&points) {
            pack_row_aux(h.kind, point, h.n, h.naux, row);
        }
        *out_rows = points.len().min(count);
        Ok(())
    })
}

pub extern "C" fn run_cancel(run: u64) -> i32 {
    barrier(|| {
        run_handle(run)?.run.cancel();
        Ok(())
    })
}

pub extern "C" fn free_handle(id: u64) -> i32 {
    barrier(|| registry::remove(id))
}

pub unsafe extern "C" fn model_info(model: u64, out_json: *mut *mut c_char) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(out_json, "info_json") }?;
        *slot = std::ptr::null_mut();
        let h = model_handle(model)?;
        let evaluator = h.evaluator();
        let name = evaluator.as_ref().map(|e| match e {
            Evaluator::Jit => "jit",
            Evaluator::Fallback => "fallback",
            Evaluator::Aot { .. } => "aot",
        });
        let aot_lib = match &evaluator {
            Some(Evaluator::Aot { lib, .. }) => Some(lib.to_string_lossy().into_owned()),
            _ => None,
        };
        let c_file = h.c_file().map(|p| p.to_string_lossy().into_owned());
        let names = &h.names;
        let v = json!({
            "n": h.n(),
            "n_p": h.n_p(),
            "delay": h.delay,
            "map": h.map,
            "variables": names.states,
            "parameters": names.parameters,
            "values": names.par_values,
            "init": names.init,
            "aux": names.aux,
            "only": names.only,
            "autonomous": names.autonomous,
            "source_sha256": names.source_sha256,
            "options": names
                .options
                .iter()
                .map(|(k, v)| json!({ "name": k, "value": v }))
                .collect::<Vec<_>>(),
            "evaluator": name,
            "aot_lib": aot_lib,
            "c_file": c_file,
            "jit_max_frame": h.max_frame(),
        });
        let s =
            serde_json::to_string(&v).map_err(|e| Error::Internal(format!("info JSON: {e}")))?;
        *slot = into_c_string(&s);
        Ok(())
    })
}

/// Reads the options object of `model_equilibrium` and its key `iterate` (SPEC 8y), which only
/// a map takes; NULL means the defaults and iterate 1.
fn engine_options(opts: *const c_char, map: bool) -> Result<(Options, usize)> {
    if opts.is_null() {
        return Ok((Options::default(), 1));
    }
    let text = unsafe { str_arg(opts, "options") }?;
    let mut v: Value = serde_json::from_str(text)
        .map_err(|e| Error::BadProblem(format!("options: not valid JSON ({e})")))?;
    let mut iterate = 1;
    if let Some(o) = v.as_object_mut() {
        iterate = parse_iterate(o).map_err(Error::BadProblem)?;
        if o.remove("iterate").is_some() && !map {
            return Err(Error::BadProblem(
                "iterate: only a map takes this key".into(),
            ));
        }
    }
    Ok((parse_options(&v).map_err(Error::BadProblem)?, iterate))
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
        h.ode_only()?;
        let n = h.n();
        if !h.names.autonomous {
            return Err(Error::NonAutonomous(EngineError::NonAutonomous.to_string()));
        }
        expect_len("u0", n, nu)?;
        expect_len("p", h.n_p(), np)?;
        expect_len("u_out", n, nu_out)?;
        expect_len("eig_out", eig_len(n)?, neig_out)?;
        if overlaps(u_out, nu_out, eig_out, neig_out) {
            return Err(bad_arg("u_out and eig_out overlap"));
        }
        let u0 = unsafe { f64_arg(u0, nu, "u0") }?.to_vec();
        let p = unsafe { f64_arg(p, np, "p") }?.to_vec();
        let (opts, iterate) = engine_options(opts, h.map)?;
        let u_out = unsafe { f64_out(u_out, nu_out, "u_out") }?;
        let eig_out = unsafe { f64_out(eig_out, neig_out, "eig_out") }?;
        let r = h.equilibrium(EpSetup::new(0, p, u0), opts, iterate)?;
        u_out.copy_from_slice(&r.u);
        for (i, (re, im)) in r.eig.iter().enumerate() {
            eig_out[i] = *re;
            eig_out[n + i] = *im;
        }
        Ok(())
    })
}

/// Reads the options object of `model_jacobian` (SPEC 8ac): NULL or an object whose only key
/// is `iterate`, which only a map takes. Returns the iterate, 1 by default.
fn jacobian_options(opts: *const c_char, map: bool) -> Result<usize> {
    if opts.is_null() {
        return Ok(1);
    }
    let text = unsafe { str_arg(opts, "options") }?;
    let v: Value = serde_json::from_str(text)
        .map_err(|e| Error::BadProblem(format!("options: not valid JSON ({e})")))?;
    let o = v
        .as_object()
        .ok_or_else(|| Error::BadProblem("options: expected an object".into()))?;
    if let Some(k) = o.keys().find(|k| k.as_str() != "iterate") {
        return Err(Error::BadProblem(format!("options: unknown key `{k}`")));
    }
    if o.contains_key("iterate") && !map {
        return Err(Error::BadProblem(
            "iterate: only a map takes this key".into(),
        ));
    }
    parse_iterate(o).map_err(Error::BadProblem)
}

/// Writes the column-major Jacobian of `F` (or of `F^k` on a map) at `u` to `jac_out`
/// (SPEC 8ac).
///
/// # Safety
/// `u` must be valid for `nu` reads, `p` for `np`, `jac_out` for `njac` writes, and `opts`
/// NULL or NUL-terminated.
pub unsafe extern "C" fn model_jacobian(
    model: u64,
    u: *const f64,
    nu: usize,
    p: *const f64,
    np: usize,
    opts: *const c_char,
    jac_out: *mut f64,
    njac: usize,
) -> i32 {
    barrier(|| {
        let h = model_handle(model)?;
        h.ode_only()?;
        let n = h.n();
        if !h.names.autonomous {
            return Err(Error::NonAutonomous(EngineError::NonAutonomous.to_string()));
        }
        expect_len("u", n, nu)?;
        expect_len("p", h.n_p(), np)?;
        let nn = n
            .checked_mul(n)
            .ok_or_else(|| bad_arg("jac_out: n*n overflows"))?;
        expect_len("jac_out", nn, njac)?;
        let u = unsafe { f64_arg(u, nu, "u") }?.to_vec();
        let p = unsafe { f64_arg(p, np, "p") }?.to_vec();
        let iterate = jacobian_options(opts, h.map)?;
        let out = unsafe { f64_out(jac_out, njac, "jac_out") }?;
        let jac = h.jacobian(u, p, iterate)?;
        out.copy_from_slice(&jac);
        Ok(())
    })
}

/// Translates SBML or a MATLAB function into `.ode` text. `opts` is NULL or a JSON object, and
/// `out_json` is NULL unless the call sets it. A failed import returns `PC_ERR_MODEL` and sets
/// `out_json` to the error object (see `import.rs`); the caller frees the string either way.
///
/// # Safety
/// `bytes` must be valid for `len` bytes, `opts` NULL or NUL-terminated, and `out_json` valid
/// for one pointer write.
pub unsafe extern "C" fn import(
    format: u32,
    bytes: *const c_char,
    len: usize,
    opts: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(out_json, "result_json") }?;
        *slot = std::ptr::null_mut();
        let input = unsafe { bytes_arg(bytes, len, "bytes") }?;
        let opts = if opts.is_null() {
            None
        } else {
            Some(unsafe { str_arg(opts, "options") }?)
        };
        let outcome = import::run(format, input, opts)?;
        *slot = into_c_string(&outcome.json);
        match outcome.failure {
            None => Ok(()),
            Some(message) => Err(Error::Model(message)),
        }
    })
}

/// Renders a figure or table document to LaTeX (`kind` 1 is pgfplots, 2 is a table). On
/// success `out_json` is `{"ok":true,"tex":...,"data_files":[{"name","content"}]}`. A document
/// that `pc-latex` rejects returns `PC_ERR_BAD_PROBLEM` and sets `out_json` to
/// `{"ok":false,"error":{"id","kind","message"}}`. An unknown `kind` or a bad pointer returns
/// `PC_ERR_BAD_ARG` and leaves `out_json` NULL. The caller frees the string either way.
///
/// # Safety
/// `doc_json` must be NUL-terminated and `out_json` valid for one pointer write.
pub unsafe extern "C" fn latex_render(
    kind: u32,
    doc_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(out_json, "result_json") }?;
        *slot = std::ptr::null_mut();
        let doc = unsafe { str_arg(doc_json, "doc_json") }?;
        let outcome = latex::run(kind, doc)?;
        *slot = into_c_string(&outcome.json);
        match outcome.failure {
            None => Ok(()),
            Some(message) => Err(Error::BadProblem(message)),
        }
    })
}

fn ens_handle(id: u64) -> Result<Arc<EnsHandle>> {
    registry::get::<EnsHandle>(id)
}

/// Lists the ensemble backends and their devices as JSON. The caller frees the string.
///
/// # Safety
/// `out_json` must be valid for one pointer write.
pub unsafe extern "C" fn ens_devices(out_json: *mut *mut c_char) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(out_json, "devices_json") }?;
        *slot = std::ptr::null_mut();
        *slot = into_c_string(&ensemble::devices()?);
        Ok(())
    })
}

/// Checks an ensemble job and starts it on a worker thread. `x0` holds `nx / n` members, one
/// after the other, and `params` one set or one per member (the job says which).
///
/// # Safety
/// `job_json` must be NUL-terminated, `x0` and `params` valid for `nx` and `np` reads, and
/// `out_ens` valid for one `u64` write.
pub unsafe extern "C" fn ens_start(
    model: u64,
    job_json: *const c_char,
    x0: *const f64,
    nx: usize,
    params: *const f64,
    np: usize,
    out_ens: *mut u64,
) -> i32 {
    barrier(|| {
        let out_ens = unsafe { out_ptr(out_ens, "ens") }?;
        *out_ens = 0;
        let h = model_handle(model)?;
        h.ode_only()?;
        let text = unsafe { str_arg(job_json, "job_json") }?;
        let x0 = unsafe { f64_arg(x0, nx, "x0") }?;
        let params = unsafe { f64_arg(params, np, "params") }?;
        *out_ens = ensemble::start(&h, text, x0, params)?;
        Ok(())
    })
}

/// Reports `running`, `cancelled` or `done` as JSON. A run that failed returns its error here.
///
/// # Safety
/// `status_json` must be valid for one pointer write.
pub unsafe extern "C" fn ens_poll(ens: u64, status_json: *mut *mut c_char) -> i32 {
    barrier(|| {
        let slot = unsafe { out_ptr(status_json, "status_json") }?;
        *slot = std::ptr::null_mut();
        let h = ens_handle(ens)?;
        let s = serde_json::to_string(&ensemble::poll(&h)?)
            .map_err(|e| Error::Internal(format!("status JSON: {e}")))?;
        *slot = into_c_string(&s);
        Ok(())
    })
}

/// Copies one array of a finished run (`PC_ENS_*`) into `buf`, which must have its exact length.
///
/// # Safety
/// `buf` must be valid for `buf_len` writes.
pub unsafe extern "C" fn ens_copy(ens: u64, what: u32, buf: *mut f64, buf_len: usize) -> i32 {
    barrier(|| {
        let buf = unsafe { f64_out(buf, buf_len, "buf") }?;
        let h = ens_handle(ens)?;
        ensemble::copy(&h, what, buf)
    })
}

pub extern "C" fn ens_cancel(ens: u64) -> i32 {
    barrier(|| {
        let h = ens_handle(ens)?;
        ensemble::cancel(&h);
        Ok(())
    })
}

/// Test hook, not in the table: panics on the model thread of `model`.
#[doc(hidden)]
pub fn debug_model_panic(model: u64) -> i32 {
    barrier(|| model_handle(model)?.debug_panic())
}

/// Test hook. `where_ == 0` panics on the calling thread. `where_ == 1` starts a run whose
/// worker panics and returns its id in `run`.
///
/// # Safety
/// `run` must be NULL or valid for one `u64` write.
pub unsafe extern "C" fn debug_panic(where_: i32, run: *mut u64) -> i32 {
    barrier(|| match where_ {
        0 => panic!("debug panic"),
        1 => {
            let slot = unsafe { out_ptr(run, "debug_panic: run") }?;
            *slot = 0;
            *slot = registry::insert(RunHandle {
                run: Run::start_panicking(),
                kind: Kind::Ep,
                n: 0,
                naux: 0,
            });
            Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_len_checks_for_overflow() {
        assert_eq!(eig_len(0).unwrap(), 0);
        assert_eq!(eig_len(3).unwrap(), 6);
        assert_eq!(eig_len(usize::MAX / 2).unwrap(), usize::MAX - 1);
        assert!(matches!(eig_len(usize::MAX / 2 + 1), Err(Error::BadArg(_))));
        assert!(matches!(eig_len(usize::MAX), Err(Error::BadArg(_))));
    }
}
