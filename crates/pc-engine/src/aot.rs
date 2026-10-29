//! Evaluating model entries with a compiled AOT C library.
//!
//! The library comes from `pc_codegen::emit_c`: it exports `pc_aot_init` and one function
//! per entry, named by [`entry_symbol`]. [`AotEval`] opens it, hands it the `pc-ir` math
//! table so that it calls the same functions as the JIT, and keeps it loaded for as long
//! as the evaluator lives.

use crate::error::EngineError;
use crate::model::{ModelEval, check_finite, lengths};
use libloading::Library;
use pc_ir::math::{MATH, MathTable};
use pc_ir::{EntryKind, Slot};
use std::collections::HashMap;
use std::path::Path;

type PlainFn = unsafe extern "C" fn(*const f64, *const f64, *mut f64);
type DirFn = unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64);

/// An entry function taken from the library.
#[derive(Clone, Copy)]
enum RawFn {
    Plain(PlainFn),
    Dir(DirFn),
}

/// The symbol that the library exports for `kind`: `f`, `aux`, `jac_u`, `jac_p`, and `d_`
/// followed by one letter per slot (`u`, `p`, `f` for `Full`) for a directional entry,
/// for example `d_uu` or `d_fp`. The caller of `emit_c` names the entries with this.
pub fn entry_symbol(kind: &EntryKind) -> String {
    match kind {
        EntryKind::F => "f".to_string(),
        EntryKind::Aux => "aux".to_string(),
        EntryKind::JacU => "jac_u".to_string(),
        EntryKind::JacP => "jac_p".to_string(),
        EntryKind::Dir(slots) => {
            let mut s = String::from("d_");
            for slot in slots {
                s.push(match slot {
                    Slot::U => 'u',
                    Slot::P => 'p',
                    Slot::Full => 'f',
                });
            }
            s
        }
    }
}

/// The entries the engine asks for: the plain ones, `Dir` of orders 1 to 5 with `U`
/// slots (normal forms) and the two mixed ones that the Hopf and branch-point systems use.
/// A library that holds all of them serves every problem type.
pub fn engine_kinds() -> Vec<EntryKind> {
    let mut kinds = vec![
        EntryKind::F,
        EntryKind::Aux,
        EntryKind::JacU,
        EntryKind::JacP,
    ];
    for k in 1..=5 {
        kinds.push(EntryKind::Dir(vec![Slot::U; k]));
    }
    kinds.push(EntryKind::Dir(vec![Slot::U, Slot::Full]));
    kinds.push(EntryKind::Dir(vec![Slot::Full, Slot::Full]));
    kinds
}

/// An AOT library wrapped as a [`ModelEval`].
pub struct AotEval {
    n: usize,
    n_p: usize,
    n_aux: usize,
    fns: HashMap<EntryKind, RawFn>,
    /// Declared last so that it drops last; every `RawFn` points into this library.
    lib: Library,
}

impl AotEval {
    /// Opens `path`, calls `pc_aot_init` with the math table and resolves the entries
    /// that the model sizes call for: `f`, `jac_u`, `aux` when `n_aux > 0` and `jac_p` when
    /// `n_p > 0`. Directional entries are resolved on first use. A library that cannot be
    /// opened, or lacks `pc_aot_init` or a required entry, is `EngineError::Model`.
    pub fn open(path: &Path, n: usize, n_p: usize, n_aux: usize) -> Result<AotEval, EngineError> {
        let bad =
            |what: String| EngineError::Model(format!("AOT library {}: {what}", path.display()));
        // SAFETY: loading runs the initializers of the file. The file is a library that
        // PhaseCont's C emitter wrote and the host compiled; its initializers are empty.
        let lib = unsafe { Library::new(path) }.map_err(|e| bad(format!("cannot open: {e}")))?;
        // SAFETY: `pc_aot_init` is emitted as `void pc_aot_init(const pc_math_t *)`, and
        // `pc_math_t` has the field order of the `#[repr(C)]` `MathTable`.
        unsafe {
            let init: libloading::Symbol<unsafe extern "C" fn(*const MathTable)> = lib
                .get(b"pc_aot_init\0")
                .map_err(|e| bad(format!("no pc_aot_init: {e}")))?;
            // `MATH` is a static, so the pointer the library keeps stays valid.
            init(&MATH);
        }
        let mut ev = AotEval {
            n,
            n_p,
            n_aux,
            fns: HashMap::new(),
            lib,
        };
        let mut need = vec![EntryKind::F, EntryKind::JacU];
        if n_aux > 0 {
            need.push(EntryKind::Aux);
        }
        if n_p > 0 {
            need.push(EntryKind::JacP);
        }
        for kind in need {
            ev.resolve(&kind)?;
        }
        Ok(ev)
    }

    /// Looks up the symbol for `kind` and stores it.
    fn resolve(&mut self, kind: &EntryKind) -> Result<RawFn, EngineError> {
        if let Some(f) = self.fns.get(kind) {
            return Ok(*f);
        }
        let name = entry_symbol(kind);
        let missing = |e: libloading::Error| {
            EngineError::Model(format!("AOT library has no entry {name} for {kind:?}: {e}"))
        };
        let sym = format!("{name}\0");
        let f = if matches!(kind, EntryKind::Dir(_)) {
            // SAFETY: `emit_c` gives a `Dir` entry the signature (x, p, v, out), and the
            // pointer is used only while `self.lib` is alive (it is dropped last).
            let s: libloading::Symbol<DirFn> =
                unsafe { self.lib.get(sym.as_bytes()) }.map_err(missing)?;
            RawFn::Dir(*s)
        } else {
            // SAFETY: `emit_c` gives the other entries the signature (x, p, out); same
            // lifetime argument as above.
            let s: libloading::Symbol<PlainFn> =
                unsafe { self.lib.get(sym.as_bytes()) }.map_err(missing)?;
            RawFn::Plain(*s)
        };
        self.fns.insert(kind.clone(), f);
        Ok(f)
    }

    /// Runs one entry after checking every length, and returns whatever it computed,
    /// NaN and infinity included. [`ModelEval::eval`] adds the finite check.
    pub fn call(
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
        // An empty entry (no aux, no parameters) may be missing from the library.
        if out_len == 0 {
            return Ok(());
        }
        let f = self.resolve(kind)?;
        // SAFETY: the lengths above equal the entry contract, so the C function reads at
        // most `n`, `n_p + 1` and `v_len` values and writes `out_len` values. The library
        // is loaded for as long as `self`.
        unsafe {
            match f {
                RawFn::Plain(f) => f(x.as_ptr(), p.as_ptr(), out.as_mut_ptr()),
                RawFn::Dir(f) => f(x.as_ptr(), p.as_ptr(), v.as_ptr(), out.as_mut_ptr()),
            }
        }
        Ok(())
    }
}

impl ModelEval for AotEval {
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
        self.call(kind, x, p, v, out)?;
        check_finite(kind, out)
    }
}
