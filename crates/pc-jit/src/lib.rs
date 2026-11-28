//! Cranelift lowering of the right-hand side, Jacobians and directional derivatives.
//!
//! [`compile`] turns DAG entries into native functions inside the calling process.
//! Each op follows `pc_ir::eval::eval_op` bit for bit: transcendental functions call
//! the `pc_ir::math` shims, the rest is lowered in line, and no fused multiply-add is
//! used.

pub mod check;
mod compiled;
mod flags;
pub use flags::host_fingerprint;
mod lower;
mod module;

#[cfg(test)]
mod tests;

pub use compiled::{CompiledModel, EntryFn};
pub use module::JitEntries;

/// The largest `frame_to_fp_offset` (bytes) among the functions compiled on the calling
/// thread so far; 0 before any compile.
pub fn max_frame() -> u32 {
    module::MAX_FRAME.with(std::cell::Cell::get)
}

/// Sets the mark that [`max_frame`] reports on the calling thread. A caller that compiles code
/// the mark should not count (the ensemble `F`) reads the mark before and puts it back after.
pub fn set_max_frame(mark: u32) {
    module::MAX_FRAME.with(|m| m.set(mark));
}

use cranelift_codegen::ir::{AbiParam, UserFuncName, types};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_module::{FuncId, Linkage, Module};
use pc_ir::dag::{Dag, NodeId, Op};
use pc_ir::{Entry, EntryKind};
use std::collections::HashMap;
use std::sync::Arc;

/// `F`, `Aux`, `JacU` and `JacP` entries: `(x, p, out)`.
///
/// `x` has n values, `p` has n_p + 1 (time last) and `out` has `outputs.len()` values.
/// `out` must not alias `x` or `p`.
pub type PlainFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64);
/// `Dir` entries: `(x, p, v, out)`.
///
/// `x` has n values, `p` has n_p + 1 (time last), `v` has the packed length of the slots
/// and `out` has `outputs.len()` values. `out` must not alias `x`, `p` or `v`.
pub type DirFn = unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64);
/// A plain entry compiled in single precision: [`PlainFn`] over `f32` arrays.
pub type PlainFnF32 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// The float type of the compiled code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Float {
    /// Double precision, every entry kind. It follows `pc_ir::eval::eval_op`.
    F64,
    /// Single precision, plain entries only. It follows the `f32` tape of pc-ensemble, which
    /// computes every op in `f32` and rounds each constant to `f32` (ties to even). Tables,
    /// `Dir` leaves and the special functions (erf, erfc, lgamma, the Bessel functions and
    /// polygamma) are refused, as that tape refuses them.
    F32,
}

impl Float {
    pub(crate) fn ty(self) -> cranelift_codegen::ir::Type {
        match self {
            Float::F64 => types::F64,
            Float::F32 => types::F32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JitError {
    /// Cranelift refused a setting, a function or the finished module.
    Cranelift(String),
    /// The lowering needs a shim that the module did not import.
    MissingImport(String),
    /// The IR could not build the entry.
    Ir(pc_ir::IrError),
    /// An argument of `CompiledModel::call` has the wrong length.
    BadLength(String),
    /// The entry uses a construct that the JIT does not lower yet, such as a recursive
    /// function call (SPEC 8af-5, stage 2).
    Unsupported(String),
    /// A call of a recursive function stopped the evaluation: the cap on active calls, the
    /// call frames running out of memory, a broken call DAG or a panic in the call helper.
    Eval(pc_ir::machine::EvalError),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Cranelift(m) => write!(f, "cranelift: {m}"),
            JitError::MissingImport(m) => write!(f, "missing import: {m}"),
            JitError::Ir(e) => write!(f, "ir: {e}"),
            JitError::BadLength(m) => write!(f, "bad length: {m}"),
            JitError::Unsupported(m) => write!(f, "not supported by the JIT: {m}"),
            JitError::Eval(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for JitError {}

pub(crate) fn cl<T, E: std::fmt::Display>(what: &str, r: Result<T, E>) -> Result<T, JitError> {
    r.map_err(|e| JitError::Cranelift(format!("{what}: {e}")))
}

/// Compiles each entry to one native function. The code reads copies of the tables it
/// uses, which the result owns, so it stays valid after `dag` is dropped.
pub fn compile(entries: &[(EntryKind, &Entry)], dag: &Dag) -> Result<JitEntries, JitError> {
    compile_as(entries, dag, Float::F64, None)
}

/// [`compile`] in single precision. Only plain entries are allowed; read them with
/// [`JitEntries::plain_f32`].
pub fn compile_f32(entries: &[(EntryKind, &Entry)], dag: &Dag) -> Result<JitEntries, JitError> {
    if let Some((k, _)) = entries.iter().find(|(k, _)| matches!(k, EntryKind::Dir(_))) {
        return Err(JitError::MissingImport(format!(
            "{k:?} in single precision"
        )));
    }
    compile_as(entries, dag, Float::F32, None)
}

/// The sizes of the arrays that an entry reads, which the call helper needs to build slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    /// The number of state variables.
    pub n: usize,
    /// The number of parameters, without the time slot.
    pub n_p: usize,
}

/// [`compile`] for entries that may call recursive functions (SPEC 8af-5). A call becomes a
/// call of a helper that runs the unit on the DAG evaluator, so the code carries a snapshot of
/// `dag` that it owns, and an entry that reaches no call compiles to the same code as with
/// [`compile`]. A helper error (the cap on active calls, memory) does not stop the entry:
/// the call gives NaN and the first error waits for [`CompiledModel::call`], which returns it.
/// A caller that runs the functions of the result directly must read that error itself, so
/// only [`CompiledModel`] uses this. `dims` are the model's `n` and `n_p`.
pub(crate) fn compile_with_calls(
    entries: &[(EntryKind, &Entry)],
    dag: &Dag,
    dims: Dims,
) -> Result<JitEntries, JitError> {
    compile_as(entries, dag, Float::F64, Some(dims))
}

fn compile_as(
    entries: &[(EntryKind, &Entry)],
    dag: &Dag,
    float: Float,
    calls: Option<Dims>,
) -> Result<JitEntries, JitError> {
    let reaches_call = |e: &Entry| e.outputs.iter().any(|&o| dag.has_call(o));
    if calls.is_none() && entries.iter().any(|(_, e)| reaches_call(e)) {
        return Err(JitError::Unsupported(
            "recursive function call; compile does not run them, a CompiledModel does".into(),
        ));
    }
    // The guard exists from here on; any `?` below drops it and frees the module.
    // The call helper is imported only when an entry reaches a call.
    let with_calls = calls.is_some() && entries.iter().any(|(_, e)| reaches_call(e));
    let mut je = JitEntries::new(float, with_calls)?;
    #[cfg(test)]
    if module::FAIL_AFTER_NEW.with(|f| f.get()) {
        return Err(JitError::Cranelift("forced failure".into()));
    }
    let JitEntries {
        module,
        tables,
        fns,
        frame,
        code,
        images,
        ..
    } = &mut je;
    let Some(module) = module.as_mut() else {
        return Err(JitError::Cranelift("module missing".into()));
    };

    let ptr = module.target_config().pointer_type();
    let plain_sig = {
        let mut s = module.make_signature();
        s.params.extend([AbiParam::new(ptr); 3]);
        s
    };
    let dir_sig = {
        let mut s = module.make_signature();
        s.params.extend([AbiParam::new(ptr); 4]);
        s
    };
    let fty = float.ty();
    let mut imports: HashMap<&'static str, FuncId> = HashMap::new();
    for (name, _) in module::symbols(float) {
        let (with_ptr, n) = match name {
            "pc_table" | "pc_table_slope" => (true, 1),
            "pc_atan2" | "pc_pow" | "pc_mod" | "pc_besselj" | "pc_bessely" | "pc_polygamma"
            | "pc_atan2f" | "pc_powf" | "pc_modf" => (false, 2),
            _ => (false, 1),
        };
        let mut sig = module.make_signature();
        if with_ptr {
            sig.params.push(AbiParam::new(ptr));
        }
        sig.params
            .extend(std::iter::repeat_n(AbiParam::new(fty), n));
        sig.returns.push(AbiParam::new(fty));
        let id = cl(name, module.declare_function(name, Linkage::Import, &sig))?;
        imports.insert(name, id);
    }

    // One snapshot of the DAG serves every entry of this compile.
    let snapshot: Option<Arc<Dag>> = with_calls.then(|| Arc::new(dag.clone()));
    let helper_id = if with_calls {
        let mut sig = module.make_signature();
        // image, unit, args, x, p, v, out
        sig.params.push(AbiParam::new(ptr));
        sig.params.push(AbiParam::new(types::I32));
        sig.params.extend([AbiParam::new(ptr); 5]);
        Some(cl(
            module::CALL_HELPER,
            module.declare_function(module::CALL_HELPER, Linkage::Import, &sig),
        )?)
    } else {
        None
    };

    let mut ctx = module.make_context();
    let mut fbc = FunctionBuilderContext::new();
    let mut table_store = lower::Tables::new();
    let mut compiled: Vec<(EntryKind, FuncId)> = Vec::new();
    for (i, (kind, entry)) in entries.iter().enumerate() {
        let has_dir = matches!(kind, EntryKind::Dir(_));
        let sig = if has_dir { &dir_sig } else { &plain_sig };
        let name = format!("pc_entry_{i}");
        let id = cl(&name, module.declare_function(&name, Linkage::Local, sig))?;
        ctx.func.signature = sig.clone();
        ctx.func.name = UserFuncName::user(0, id.as_u32());

        let live = pc_ir::eval::live_set(dag, &entry.outputs);
        let mut refs = HashMap::new();
        for n in lower::imports_needed(dag, &live, float) {
            let Some(&fid) = imports.get(n) else {
                return Err(JitError::MissingImport(n.to_string()));
            };
            refs.insert(n, module.declare_func_in_func(fid, &mut ctx.func));
        }
        let call = match (helper_id, &snapshot, calls) {
            (Some(h), Some(dag), Some(d)) if reaches_call(entry) => {
                let nv = match kind {
                    EntryKind::Dir(slots) => slots.iter().map(|s| s.len(d.n, d.n_p)).sum(),
                    _ => 0,
                };
                let image = Arc::new(module::CallImage {
                    dag: Arc::clone(dag),
                    nx: d.n,
                    np: d.n_p + 1,
                    nv,
                });
                let at: *const module::CallImage = Arc::as_ptr(&image);
                images.push(image);
                Some(lower::CallLower {
                    helper: module.declare_func_in_func(h, &mut ctx.func),
                    image: at,
                })
            }
            _ => None,
        };
        let mut table_ptrs = HashMap::new();
        for (j, &l) in live.iter().enumerate() {
            if !l {
                continue;
            }
            if let Op::Table(t, _) | Op::TableSlope(t, _) = *dag.op(NodeId(j as u32)) {
                table_ptrs.insert(t, table_store.ptr(dag, t));
            }
        }
        lower::build(
            &mut ctx.func,
            &mut fbc,
            lower::Ctx {
                dag,
                outputs: &entry.outputs,
                live: &live,
                refs: &refs,
                table_ptrs: &table_ptrs,
                has_dir,
                derivative: kind.is_derivative(),
                ptr,
                float,
                target: module.target_config(),
                call,
            },
        )?;
        cl(&name, module.define_function(id, &mut ctx))?;
        if let Some(c) = ctx.compiled_code() {
            *code = code.saturating_add(c.code_info().total_size);
        }
        if let Some(fl) = ctx.compiled_code().and_then(|c| c.buffer.frame_layout()) {
            module::MAX_FRAME.with(|m| m.set(m.get().max(fl.frame_to_fp_offset)));
            *frame = (*frame).max(fl.frame_to_fp_offset);
        }
        module.clear_context(&mut ctx);
        compiled.push((kind.clone(), id));
    }
    // A boxed slice keeps its address when the box moves into the guard.
    *tables = table_store.store;
    cl("finalize", module.finalize_definitions())?;
    for (kind, id) in compiled {
        fns.push((kind, module.get_finalized_function(id)));
    }
    Ok(je)
}
