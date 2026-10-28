//! Cranelift lowering of the right-hand side, Jacobians and directional derivatives.
//!
//! [`compile`] turns DAG entries into native functions inside the calling process.
//! Each op follows `pc_ir::eval::eval_op` bit for bit: transcendental functions call
//! the `pc_ir::math` shims, the rest is lowered in line, and no fused multiply-add is
//! used.

mod flags;
mod lower;
mod module;

#[cfg(test)]
mod tests;

pub use module::JitEntries;

use cranelift_codegen::ir::{AbiParam, UserFuncName, types};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_module::{FuncId, Linkage, Module};
use pc_ir::dag::{Dag, NodeId, Op};
use pc_ir::{Entry, EntryKind};
use std::collections::HashMap;

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

#[derive(Debug, Clone, PartialEq)]
pub enum JitError {
    /// Cranelift refused a setting, a function or the finished module.
    Cranelift(String),
    /// The lowering needs a shim that the module did not import.
    MissingImport(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Cranelift(m) => write!(f, "cranelift: {m}"),
            JitError::MissingImport(m) => write!(f, "missing import: {m}"),
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
    // The guard exists from here on; any `?` below drops it and frees the module.
    let mut je = JitEntries::new()?;
    #[cfg(test)]
    if module::FAIL_AFTER_NEW.with(|f| f.get()) {
        return Err(JitError::Cranelift("forced failure".into()));
    }
    let JitEntries {
        module,
        tables,
        fns,
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
    let mut imports: HashMap<&'static str, FuncId> = HashMap::new();
    for (name, _) in pc_ir::math::symbols() {
        let (with_ptr, n) = match name {
            "pc_table" | "pc_table_slope" => (true, 1),
            "pc_atan2" | "pc_pow" | "pc_mod" | "pc_besselj" | "pc_bessely" | "pc_polygamma" => {
                (false, 2)
            }
            _ => (false, 1),
        };
        let mut sig = module.make_signature();
        if with_ptr {
            sig.params.push(AbiParam::new(ptr));
        }
        sig.params
            .extend(std::iter::repeat_n(AbiParam::new(types::F64), n));
        sig.returns.push(AbiParam::new(types::F64));
        let id = cl(name, module.declare_function(name, Linkage::Import, &sig))?;
        imports.insert(name, id);
    }

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

        let live = lower::live_set(dag, &entry.outputs);
        let mut refs = HashMap::new();
        for n in lower::imports_needed(dag, &live) {
            let Some(&fid) = imports.get(n) else {
                return Err(JitError::MissingImport(n.to_string()));
            };
            refs.insert(n, module.declare_func_in_func(fid, &mut ctx.func));
        }
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
                ptr,
                target: module.target_config(),
            },
        )?;
        cl(&name, module.define_function(id, &mut ctx))?;
        #[cfg(test)]
        if let Some(fl) = ctx.compiled_code().and_then(|c| c.buffer.frame_layout()) {
            module::MAX_FRAME.with(|m| m.set(m.get().max(fl.frame_to_fp_offset)));
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
