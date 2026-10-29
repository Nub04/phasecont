//! Evaluating model entries in f64, with the JIT or the interpreter fallback.

use crate::aot::AotEval;
use crate::error::EngineError;
use pc_ir::{EntryKind, Fallback, IrModel, lower};
use pc_jit::CompiledModel;
use pc_ode::{Expr, Model, ModelInfo, StmtKind, SymbolKind};
use std::collections::HashSet;
use std::path::PathBuf;

/// Evaluates the entries of one model. Inputs and outputs follow the contract in
/// `pc_ir::entry`: `x` has `n` values, `p` has `n_p + 1` (time last), `v` is empty except
/// for `Dir` entries, and `out` has the entry's length.
pub trait ModelEval {
    fn n(&self) -> usize;
    fn n_p(&self) -> usize;
    /// Runs one entry. Returns `NotFinite` when any output is NaN or infinite.
    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError>;
}

pub(crate) fn check_finite(kind: &EntryKind, out: &[f64]) -> Result<(), EngineError> {
    if out.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(EngineError::NotFinite(format!("{kind:?}")))
    }
}

/// Expected length of `out` and `v` for an entry.
pub(crate) fn lengths(kind: &EntryKind, n: usize, n_p: usize, n_aux: usize) -> (usize, usize) {
    match kind {
        EntryKind::F => (n, 0),
        EntryKind::Aux => (n_aux, 0),
        EntryKind::JacU => (n * n, 0),
        EntryKind::JacP => (n * n_p, 0),
        EntryKind::Dir(slots) => (n, slots.iter().map(|s| s.len(n, n_p)).sum()),
    }
}

impl ModelEval for CompiledModel {
    fn n(&self) -> usize {
        self.ir().n()
    }

    fn n_p(&self) -> usize {
        self.ir().n_p()
    }

    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        // `call` checks every length and returns `BadLength` on a mismatch.
        self.call(kind, x, p, v, out)?;
        check_finite(kind, out)
    }
}

/// [`Fallback`] with the sizes it was built for, so that lengths can be checked here
/// (the fallback itself fills the output with NaN on a wrong length).
struct FallbackEval {
    fb: Fallback,
    n: usize,
    n_p: usize,
    n_aux: usize,
}

impl ModelEval for FallbackEval {
    fn n(&self) -> usize {
        self.n
    }

    fn n_p(&self) -> usize {
        self.n_p
    }

    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        let (out_len, v_len) = lengths(kind, self.n, self.n_p, self.n_aux);
        for (name, got, want) in [
            ("x", x.len(), self.n),
            ("p", p.len(), self.n_p + 1),
            ("v", v.len(), v_len),
            ("out", out.len(), out_len),
        ] {
            if got != want {
                return Err(EngineError::Model(format!(
                    "bad length: {name} has {got} values, expected {want}"
                )));
            }
        }
        self.fb.eval(kind, x, p, v, out);
        check_finite(kind, out)
    }
}

/// Which evaluator [`build`] makes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Evaluator {
    /// Cranelift-compiled native code.
    Jit,
    /// The f64 interpreter in `pc_ir`.
    Fallback,
    /// A C library from `pc_codegen::emit_c`, built by the host, at this path. The entries
    /// are named by [`crate::entry_symbol`].
    Aot { lib: PathBuf },
}

/// Lowers `model` and returns an evaluator for it.
///
/// Refuses with `NonAutonomous` a model that reads `t`, and with `BadProblem` a model
/// with a formula table whose formula reads a parameter: `lower` tabulates a table once
/// from the declared parameter values, so it would not follow the continuation parameter
/// or an override. The result is not `Send` for `Evaluator::Jit`; build it on the thread
/// that uses it.
pub fn build(
    model: &Model,
    info: &ModelInfo,
    which: Evaluator,
) -> Result<Box<dyn ModelEval>, EngineError> {
    build_checked(model, info, which, true)
}

/// Like [`build`] but accepts a model that reads `t`. The result evaluates `F`, `JacU` and
/// the other entries with time as the last entry of `p` (`p` has `n_p + 1` values). The
/// table check still applies.
pub fn build_allowing_t(
    model: &Model,
    info: &ModelInfo,
    which: Evaluator,
) -> Result<Box<dyn ModelEval>, EngineError> {
    build_checked(model, info, which, false)
}

fn build_checked(
    model: &Model,
    info: &ModelInfo,
    which: Evaluator,
    require_autonomous: bool,
) -> Result<Box<dyn ModelEval>, EngineError> {
    let ir: IrModel = lower(model, info).map_err(|errs| {
        let msgs: Vec<&str> = errs.iter().map(|e| e.message.as_str()).collect();
        EngineError::Model(msgs.join("; "))
    })?;
    if require_autonomous && !ir.autonomous {
        return Err(EngineError::NonAutonomous);
    }
    if let Some(name) = table_reading_a_parameter(model, info) {
        return Err(EngineError::BadProblem(format!(
            "table {name} reads a parameter; tables are tabulated once from the declared values"
        )));
    }
    Ok(match which {
        Evaluator::Jit => Box::new(CompiledModel::new(ir)),
        Evaluator::Aot { lib } => Box::new(AotEval::open(&lib, ir.n(), ir.n_p(), ir.aux.len())?),
        Evaluator::Fallback => {
            let fb = Fallback::new(model, info, &ir);
            Box::new(FallbackEval {
                fb,
                n: ir.n(),
                n_p: ir.n_p(),
                n_aux: ir.aux.len(),
            })
        }
    })
}

/// The name of the first formula table whose formula reads a declared parameter,
/// directly or through a derived parameter, a fixed quantity, a user function or another
/// table.
fn table_reading_a_parameter(model: &Model, info: &ModelInfo) -> Option<String> {
    for stmt in &model.statements {
        if let StmtKind::Table { name, rhs, .. } = &stmt.kind {
            let mut seen = HashSet::new();
            if reads_parameter(rhs, &[], model, info, &mut seen) {
                return Some(name.text.clone());
            }
        }
    }
    None
}

/// True when `e` reads a parameter. `bound` holds the formal parameters of the user
/// function being walked, which shadow declared names. `seen` holds statement indices
/// already walked, which also stops a cycle.
fn reads_parameter(
    e: &Expr,
    bound: &[&str],
    model: &Model,
    info: &ModelInfo,
    seen: &mut HashSet<usize>,
) -> bool {
    match e {
        Expr::Num { .. } | Expr::Index(_) | Expr::Error { .. } => false,
        Expr::Name(id) => {
            if bound.contains(&id.key.as_str()) {
                return false;
            }
            let Some(sym) = info.symbol(&id.key) else {
                return false;
            };
            match sym.kind {
                SymbolKind::Parameter => true,
                SymbolKind::Derived | SymbolKind::Fixed => follow(sym.stmt, model, info, seen),
                _ => false,
            }
        }
        Expr::Unary { operand, .. } => reads_parameter(operand, bound, model, info, seen),
        Expr::Binary { lhs, rhs, .. } => {
            reads_parameter(lhs, bound, model, info, seen)
                || reads_parameter(rhs, bound, model, info, seen)
        }
        Expr::If {
            cond, then, else_, ..
        } => [cond, then, else_]
            .into_iter()
            .any(|x| reads_parameter(x, bound, model, info, seen)),
        Expr::Call { name, args, .. } => {
            if args
                .iter()
                .any(|a| reads_parameter(a, bound, model, info, seen))
            {
                return true;
            }
            match info.symbol(&name.key) {
                Some(sym)
                    if matches!(sym.kind, SymbolKind::Function { .. } | SymbolKind::Table) =>
                {
                    follow(sym.stmt, model, info, seen)
                }
                _ => false,
            }
        }
    }
}

/// Walks the right side of the defining statement `stmt`, once.
fn follow(stmt: usize, model: &Model, info: &ModelInfo, seen: &mut HashSet<usize>) -> bool {
    if !seen.insert(stmt) {
        return false;
    }
    match model.statements.get(stmt).map(|s| &s.kind) {
        Some(StmtKind::Derived { rhs, .. }) | Some(StmtKind::Fixed { rhs, .. }) => {
            reads_parameter(rhs, &[], model, info, seen)
        }
        Some(StmtKind::Table { rhs, .. }) => reads_parameter(rhs, &[], model, info, seen),
        Some(StmtKind::Function { params, rhs, .. }) => {
            let bound: Vec<&str> = params.iter().map(|p| p.key.as_str()).collect();
            reads_parameter(rhs, &bound, model, info, seen)
        }
        _ => false,
    }
}
