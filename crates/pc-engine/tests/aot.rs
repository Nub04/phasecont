//! The AOT evaluator against the JIT on every model of the corpus, bit for bit, and the
//! refusals of a missing or wrong library. The library is built with gcc; the tests print
//! a note and return when it is missing.

mod common;

use pc_codegen::emit_c;
use pc_engine::{AotEval, EngineError, Evaluator, ModelEval, build, engine_kinds, entry_symbol};
use pc_ir::check::{Rng, draw_point};
use pc_ir::{Entry, EntryKind, IrModel, lower};
use pc_jit::CompiledModel;
use pc_ode::{FsLoader, Loaded, Severity, SourceMap, load};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The flags of `crates/pc-codegen/tests/common/mod.rs`.
const FLAGS: [&str; 7] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-std=c99",
    "-Wall",
    "-Wextra",
];

fn gcc_version() -> Option<String> {
    let o = Command::new("gcc").arg("--version").output().ok()?;
    if !o.status.success() {
        return None;
    }
    String::from_utf8_lossy(&o.stdout)
        .lines()
        .next()
        .map(|s| s.trim().to_string())
}

/// A temp folder removed on drop.
struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Temp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "pc-engine-aot-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        Temp(dir)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Compiles `c` to a shared library inside `tmp` and returns its path.
fn compile(tmp: &Temp, c: &str) -> Result<PathBuf, String> {
    let src = tmp.0.join("model.c");
    let so = tmp.0.join(if cfg!(windows) {
        "model.dll"
    } else {
        "model.so"
    });
    std::fs::write(&src, c).map_err(|e| format!("write c: {e}"))?;
    let out = Command::new("gcc")
        .args(FLAGS)
        .arg("-Werror")
        .arg("-o")
        .arg(&so)
        .arg(&src)
        .output()
        .map_err(|e| format!("run gcc: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "gcc failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(so)
}

fn dir(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Every `*.ode` file directly in the two corpus folders, sorted.
fn model_files() -> Vec<(String, PathBuf)> {
    let mut all = Vec::new();
    for (tag, rel) in [
        ("models", "../pc-ir/tests/models"),
        ("corpus", "../pc-ode/tests/corpus"),
    ] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir(rel))
            .unwrap_or_else(|e| panic!("cannot read {rel}: {e}"))
            .map(|e| e.expect("directory entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "ode"))
            .collect();
        files.sort();
        for f in files {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            all.push((format!("{tag}/{name}"), f));
        }
    }
    all
}

fn load_path(path: &Path) -> Option<Loaded> {
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let top = loader.load_top(&mut map, path).ok()?;
    let l = load(top, &mut map, &mut loader);
    l.diagnostics
        .iter()
        .all(|d| d.severity != Severity::Error)
        .then_some(l)
}

/// Builds the library for `ir` with every entry the engine uses.
fn emit_all(ir: &mut IrModel) -> String {
    let kinds = engine_kinds();
    let entries: Vec<Entry> = kinds.iter().map(|k| ir.entry(k).expect("entry")).collect();
    let named: Vec<(String, &Entry)> = kinds.iter().map(entry_symbol).zip(entries.iter()).collect();
    emit_c(ir, &named)
}

/// Same bits, or both NaN.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn aot_matches_jit_on_the_corpus() {
    let Some(v) = gcc_version() else {
        println!("skip: gcc not found, AOT evaluator test not run");
        return;
    };
    println!("compiler: {v}");
    println!("flags: gcc {} -Werror", FLAGS.join(" "));
    let (mut models, mut values) = (0usize, 0usize);
    for (name, path) in model_files() {
        let Some(loaded) = load_path(&path) else {
            continue;
        };
        let Ok(mut ir) = lower(&loaded.expanded, &loaded.info) else {
            continue;
        };
        let c = emit_all(&mut ir);
        let tmp = Temp::new("corpus");
        let so = compile(&tmp, &c).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (n, n_p, n_aux) = (ir.n(), ir.n_p(), ir.aux.len());
        let mut aot = AotEval::open(&so, n, n_p, n_aux).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((aot.n(), aot.n_p()), (n, n_p), "{name}");
        let mut jit = CompiledModel::new(ir);
        let mut rng = Rng(11);
        for kind in engine_kinds() {
            let (out_len, v_len) = match &kind {
                EntryKind::F => (n, 0),
                EntryKind::Aux => (n_aux, 0),
                EntryKind::JacU => (n * n, 0),
                EntryKind::JacP => (n * n_p, 0),
                EntryKind::Dir(s) => (n, s.iter().map(|s| s.len(n, n_p)).sum()),
            };
            for _ in 0..6 {
                let (x, p) = draw_point(jit.ir(), &mut rng, 1.0);
                let v: Vec<f64> = (0..v_len).map(|_| rng.range(-1.0, 1.0)).collect();
                let mut want = vec![f64::MAX; out_len];
                jit.call(&kind, &x, &p, &v, &mut want).expect("jit call");
                let mut got = vec![f64::MIN; out_len];
                aot.call(&kind, &x, &p, &v, &mut got).expect("aot call");
                for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                    values += 1;
                    assert!(
                        same(*a, *b),
                        "{name} {kind:?}[{i}]: aot {a:e} ({:#x}) jit {b:e} ({:#x}), x={x:?} p={p:?}",
                        a.to_bits(),
                        b.to_bits()
                    );
                }
                // The trait entry reports NotFinite exactly when an output is not finite.
                let mut out = vec![0.0; out_len];
                let r = aot.eval(&kind, &x, &p, &v, &mut out);
                assert_eq!(
                    r.is_err(),
                    got.iter().any(|g| !g.is_finite()),
                    "{name} {kind:?}"
                );
            }
        }
        models += 1;
    }
    assert_eq!(models, 28, "models compared");
    println!("aot evaluator: {models} models, {values} values, 0 ulp");
}

#[test]
fn build_with_aot_runs_a_model() {
    if gcc_version().is_none() {
        println!("skip: gcc not found, AOT build test not run");
        return;
    }
    let (model, info) = common::load("lorenz");
    let mut ir = lower(&model, &info).expect("lower");
    let tmp = Temp::new("build");
    let so = compile(&tmp, &emit_all(&mut ir)).expect("compile");
    let mut aot = build(&model, &info, Evaluator::Aot { lib: so }).expect("build aot");
    let mut jit = build(&model, &info, Evaluator::Jit).expect("build jit");
    assert_eq!((aot.n(), aot.n_p()), (jit.n(), jit.n_p()));
    let x = [0.3, -0.7, 1.9];
    let p = [10.0, 28.0, 8.0 / 3.0, 0.0];
    let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
    aot.eval(&EntryKind::F, &x, &p, &[], &mut a).unwrap();
    jit.eval(&EntryKind::F, &x, &p, &[], &mut b).unwrap();
    for i in 0..3 {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "F[{i}]");
    }
    // A wrong length is an error, not a read past the slice.
    let r = aot.eval(&EntryKind::F, &x[..2], &p, &[], &mut a);
    assert!(matches!(r, Err(EngineError::Model(_))), "{r:?}");
    // The library path is a plain build argument: a missing one is a model error.
    let r = build(
        &model,
        &info,
        Evaluator::Aot {
            lib: PathBuf::from("/no/such/dir/model.so"),
        },
    );
    assert!(matches!(r, Err(EngineError::Model(_))));
}

#[test]
fn missing_library_is_a_model_error() {
    let r = AotEval::open(Path::new("/no/such/dir/model.so"), 1, 1, 0);
    match r {
        Err(EngineError::Model(m)) => assert!(m.contains("model.so"), "{m}"),
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!("opened a missing library"),
    }
}

#[test]
fn library_without_init_is_a_model_error() {
    if gcc_version().is_none() {
        println!("skip: gcc not found, AOT refusal test not run");
        return;
    }
    let tmp = Temp::new("noinit");
    let so = compile(
        &tmp,
        "void f(const double *x, const double *p, double *out) { out[0] = x[0] + p[0]; }\n",
    )
    .expect("compile");
    match AotEval::open(&so, 1, 1, 0) {
        Err(EngineError::Model(m)) => assert!(m.contains("pc_aot_init"), "{m}"),
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!("opened a library without pc_aot_init"),
    }
}

#[test]
fn library_without_an_entry_is_a_model_error() {
    if gcc_version().is_none() {
        println!("skip: gcc not found, AOT refusal test not run");
        return;
    }
    // Has pc_aot_init and f, but no jac_u.
    let tmp = Temp::new("noentry");
    let c = "struct t; void pc_aot_init(const struct t *m) { (void)m; }\n\
             void f(const double *x, const double *p, double *out) { out[0] = x[0] + p[0]; }\n";
    let so = compile(&tmp, c).expect("compile");
    match AotEval::open(&so, 1, 0, 0) {
        Err(EngineError::Model(m)) => assert!(m.contains("jac_u"), "{m}"),
        Err(e) => panic!("wrong error {e:?}"),
        Ok(_) => panic!("opened a library without jac_u"),
    }
}

#[test]
fn missing_directional_entry_is_an_error_at_first_use() {
    if gcc_version().is_none() {
        println!("skip: gcc not found, AOT refusal test not run");
        return;
    }
    let (model, info) = common::load("fold");
    let mut ir = lower(&model, &info).expect("lower");
    let kinds = [EntryKind::F, EntryKind::JacU, EntryKind::JacP];
    let entries: Vec<Entry> = kinds.iter().map(|k| ir.entry(k).unwrap()).collect();
    let named: Vec<(String, &Entry)> = kinds.iter().map(entry_symbol).zip(&entries).collect();
    let tmp = Temp::new("nodir");
    let so = compile(&tmp, &emit_c(&ir, &named)).expect("compile");
    let mut aot = AotEval::open(&so, ir.n(), ir.n_p(), ir.aux.len()).expect("open");
    let kind = EntryKind::Dir(vec![pc_ir::Slot::U]);
    let mut out = [0.0];
    let r = aot.eval(&kind, &[0.5], &[1.0, 0.0], &[1.0], &mut out);
    assert!(matches!(r, Err(EngineError::Model(_))), "{r:?}");
}
