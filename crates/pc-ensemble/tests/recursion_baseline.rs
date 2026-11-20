//! The lowering of models without recursion, pinned before recursive functions land (SPEC
//! section 8af-5, P44). For every test model of `pc-ode`, `pc-ir` and `pc-engine`, the golden
//! file records the DAG node count and a hash of its ops after each entry, a hash of the
//! emitted C, the JIT frame size, the tape length and hashes of two GPU kernel texts. A
//! model that does not call a recursive function must keep every value.
//!
//! `PC_BLESS=1 cargo test -p pc-ensemble --test recursion_baseline` rewrites the golden
//! file; read the diff before you commit it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use pc_codegen::fnv1a64;
use pc_codegen::kernel::{Dialect, Scalar};
use pc_ensemble::Tape;
use pc_ensemble::kernel::{Integrator, kernel_source};
use pc_ir::{EntryKind, IrModel, Slot};
use pc_ode::{FsLoader, Severity, SourceMap, load};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn models() -> Vec<PathBuf> {
    let mut v = Vec::new();
    for dir in [
        "pc-ode/tests/corpus",
        "pc-ir/tests/models",
        "pc-engine/tests/models",
        "pc-ensemble/tests/models",
    ] {
        let d = root().join(dir);
        let mut here: Vec<PathBuf> = std::fs::read_dir(&d)
            .unwrap_or_else(|e| panic!("{}: {e}", d.display()))
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|x| x == "ode"))
            .collect();
        here.sort();
        v.extend(here);
    }
    v
}

fn lower(path: &Path) -> Result<IrModel, String> {
    let mut map = SourceMap::new();
    let top = FsLoader::new()
        .load_top(&mut map, path)
        .map_err(|e| format!("read {e}"))?;
    let l = load(top, &mut map, &mut FsLoader::new());
    if l.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err("diagnostics".into());
    }
    pc_ir::lower(&l.expanded, &l.info).map_err(|e| format!("lower {e:?}"))
}

fn ops_hash(m: &IrModel) -> u64 {
    let mut s = String::new();
    for i in 0..m.dag.len() {
        let _ = writeln!(s, "{:?}", m.dag.op(pc_ir::dag::NodeId(i as u32)));
    }
    for t in &m.dag.tables {
        for v in t {
            let _ = write!(s, "{:016x} ", v.to_bits());
        }
        s.push('\n');
    }
    fnv1a64(s.as_bytes())
}

fn line(path: &Path) -> String {
    let name = path
        .strip_prefix(root())
        .unwrap_or(path)
        .display()
        .to_string();
    let mut out = format!("{name}:");
    let mut m = match lower(path) {
        Ok(m) => m,
        Err(e) => {
            // Only the first word: the debug text of an error is not part of the baseline.
            let w = e.split([' ', '(', '{']).next().unwrap_or("").to_string();
            return format!("{out} skipped ({w})");
        }
    };
    let mut kinds = vec![("f", EntryKind::F)];
    if !m.aux.is_empty() {
        kinds.push(("aux", EntryKind::Aux));
    }
    kinds.push(("jacu", EntryKind::JacU));
    kinds.push(("jacp", EntryKind::JacP));
    kinds.push(("uu", EntryKind::Dir(vec![Slot::U, Slot::U])));
    kinds.push(("up", EntryKind::Dir(vec![Slot::U, Slot::P])));
    kinds.push(("uuu", EntryKind::Dir(vec![Slot::U, Slot::U, Slot::U])));
    let mut built = Vec::new();
    for (tag, k) in &kinds {
        match m.entry(k) {
            Ok(e) => {
                let _ = write!(out, " {tag}={}/{:016x}", m.dag.len(), ops_hash(&m));
                built.push((tag.to_string(), e));
            }
            Err(e) => {
                let w = format!("{e:?}");
                let w = w.split([' ', '(', '{']).next().unwrap_or("");
                let _ = write!(out, " {tag}=err:{w}");
            }
        }
    }
    let refs: Vec<(String, &pc_ir::Entry)> = built.iter().map(|(t, e)| (t.clone(), e)).collect();
    match pc_codegen::emit_c(&m, &refs) {
        Ok(c) => {
            let _ = write!(out, " c={:016x}", fnv1a64(c.as_bytes()));
        }
        Err(_) => out.push_str(" c=err"),
    }
    let plain: Vec<(EntryKind, &pc_ir::Entry)> = built
        .iter()
        .filter(|(_, e)| !matches!(e.kind, EntryKind::Dir(_)))
        .map(|(_, e)| (e.kind.clone(), e))
        .collect();
    match pc_jit::compile(&plain, &m.dag) {
        Ok(j) => {
            let _ = write!(out, " jit={}", j.frame());
        }
        Err(_) => out.push_str(" jit=err"),
    }
    match Tape::build(&m) {
        Ok(t) => {
            let _ = write!(out, " tape={}", t.len());
            for (i, s, d, tag) in [
                (Integrator::Rk4, Scalar::F64, Dialect::Cuda, "cu"),
                (Integrator::Dp5, Scalar::F32, Dialect::OpenCl, "cl"),
            ] {
                match kernel_source(&t, i, s, d) {
                    Ok(k) => {
                        let _ = write!(out, " {tag}={:016x}", fnv1a64(k.as_bytes()));
                    }
                    Err(_) => {
                        let _ = write!(out, " {tag}=err");
                    }
                }
            }
        }
        Err(_) => out.push_str(" tape=err"),
    }
    out
}

#[test]
fn models_without_recursion_keep_their_lowering() {
    let mut text = String::new();
    for p in models() {
        text.push_str(&line(&p));
        text.push('\n');
    }
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/recursion_baseline.golden");
    if std::env::var_os("PC_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(&golden, &text).expect("write golden");
        return;
    }
    let want = std::fs::read_to_string(&golden)
        .expect("golden file is missing (run with PC_BLESS=1 to create it)");
    for (i, (w, g)) in want.lines().zip(text.lines()).enumerate() {
        assert_eq!(w, g, "line {} differs", i + 1);
    }
    assert_eq!(want.lines().count(), text.lines().count(), "model count");
}
