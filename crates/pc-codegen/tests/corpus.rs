//! `crates/pc-ir/tests/models/` and every Phase 1 corpus model in
//! `crates/pc-ode/tests/corpus/` that `lower` accepts.
//!
//! Per accepted model:
//! 1. `check_model`: the DAG entries against the f64 dual-number fallback (the section 3
//!    tolerance) and against central finite differences. On a model that is not smooth
//!    the finite-difference rows (labels ending in `(FD)`) are printed but not counted,
//!    since a kink between the two sides of a difference makes them unreliable.
//! 2. The JIT against the DAG evaluator, bit for bit (a NaN matches any NaN).
//! 3. The JIT against the fallback: `F` and `Aux` within 4 ulp (SPEC section 4), derivatives
//!    within the section 3 tolerance.
//! 4. The AOT C code, compiled with gcc, against the JIT, within 4 ulp. Skipped with a
//!    note when gcc is missing.
//!
//! The refused corpus models are listed with their `IrErrorKind`, and the test asserts
//! the refused set exactly, so a newly refused or newly accepted model is noticed.
//! Run with `-- --nocapture` to see the table.

mod common;

use common::{gcc_version, record_compiler, try_compile_c, ulp_dist};
use pc_codegen::emit_c;
use pc_ir::check::{Options, Rng, check_model, draw_point};
use pc_ir::{Entry, EntryKind, Fallback, IrErrorKind, IrModel, lower};
use pc_jit::CompiledModel;
use pc_jit::check::{RowKind, jit_kinds, jit_rows};
use pc_ode::{FsLoader, Loaded, Severity, SourceMap, load};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Random points per entry and the generator seed. Fixed, so a failure reproduces with
/// `pc-cli check-derivs <file> --points 6 --seed 1`.
const POINTS: usize = 6;
const SEED: u64 = 1;

/// How many of the models in the two folders `lower` accepts. A new model file, or a model
/// that starts or stops being refused, changes this and the test says so.
const EXPECTED_ACCEPTED: usize = 28;

/// The models this test adds under `crates/pc-ir/tests/models/`.
const REQUIRED_MODELS: [&str; 9] = [
    "bautin.ode",
    "bogdanov_takens.ode",
    "brusselator.ode",
    "cusp.ode",
    "fold.ode",
    "hopf.ode",
    "lorenz.ode",
    "nonsmooth.ode",
    "table.ode",
];

/// The corpus models `lower` refuses, with the set of distinct error kinds (compared as a
/// sorted set, so the order of the errors in a model does not matter).
const EXPECTED_REFUSED: [(&str, &[IrErrorKind]); 2] = [
    ("corpus/stochastic.ode", &[IrErrorKind::SimulationOnly]),
    (
        "corpus/unsupported.ode",
        &[IrErrorKind::Delay, IrErrorKind::Unsupported],
    ),
];

fn dir(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// `(display name, path)` of every `*.ode` file directly in the two folders, sorted.
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

fn load_path(path: &Path) -> Result<Loaded, String> {
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let top = loader
        .load_top(&mut map, path)
        .map_err(|e| format!("cannot read {e}"))?;
    Ok(load(top, &mut map, &mut loader))
}

/// What the AOT comparison did.
enum AotResult {
    /// gcc is missing.
    Skipped,
    /// The largest distance in ulp over `values` finite and non-finite outputs.
    Compared { max_ulp: u64, values: usize },
}

struct Stats {
    n: usize,
    n_p: usize,
    smooth: bool,
    /// Counted rows: `check_model` rows (FD rows only on a smooth model) plus JIT rows.
    counted: usize,
    /// Counted rows by check: fallback/FD, JIT vs DAG, JIT vs fallback.
    by_check: [usize; 3],
    /// Largest ratio of the exact rows and of the FD rows of `check_model`, with labels.
    worst_exact: (f64, String),
    worst_fd: (f64, String),
    aot: AotResult,
    failures: Vec<String>,
}

enum Status {
    Accepted(Box<Stats>),
    Refused(Vec<IrErrorKind>, String),
    /// The file did not load; always a test failure.
    LoadFailed(String),
}

struct ModelResult {
    name: String,
    status: Status,
    time: Duration,
}

/// Compiles the F, JacU, JacP, ... entries of `loaded` to C, builds the JIT of the same
/// model and compares them at random points. Returns the largest ulp distance and the
/// number of values compared, and pushes one line per failing entry onto `failures`.
fn aot_vs_jit(
    name: &str,
    loaded: &Loaded,
    hw: f64,
    failures: &mut Vec<String>,
) -> Result<(u64, usize), String> {
    let mut ir = lower(&loaded.expanded, &loaded.info).map_err(|e| format!("{e:?}"))?;
    let kinds = jit_kinds(&ir);
    let mut entries: Vec<Entry> = Vec::new();
    for (kind, label) in &kinds {
        entries.push(
            ir.entry(kind)
                .map_err(|e| format!("entry {label} not built: {e}"))?,
        );
    }
    let c = {
        let named: Vec<(String, &Entry)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("e{i}"), e))
            .collect();
        emit_c(&ir, &named).map_err(|e| e.to_string())?
    };
    let tag = name.replace(['/', '.'], "_");
    let aot = try_compile_c("gcc", &tag, &c)?;
    let mut jit = CompiledModel::new(ir);
    let mut rng = Rng(SEED);
    let (mut max_ulp, mut values) = (0_u64, 0_usize);
    for (i, ((kind, label), entry)) in kinds.iter().zip(&entries).enumerate() {
        let (n, n_p) = (jit.ir().n(), jit.ir().n_p());
        let v_len = match kind {
            EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
            _ => 0,
        };
        let is_dir = matches!(kind, EntryKind::Dir(_));
        let mut worst: Option<String> = None;
        let mut kind_max = 0_u64;
        for _ in 0..POINTS {
            let (x, p) = draw_point(jit.ir(), &mut rng, hw);
            let v: Vec<f64> = (0..v_len).map(|_| rng.range(-1.0, 1.0)).collect();
            let mut want = vec![f64::NAN; entry.outputs.len()];
            jit.call(kind, &x, &p, &v, &mut want)
                .map_err(|e| format!("JIT call {label}: {e}"))?;
            let mut got = vec![f64::NAN; entry.outputs.len()];
            aot.call(&format!("e{i}"), &x, &p, is_dir.then_some(&v[..]), &mut got);
            for (j, (a, b)) in got.iter().zip(&want).enumerate() {
                values += 1;
                let d = ulp_dist(*a, *b);
                if d > kind_max {
                    kind_max = d;
                    if d > 4 {
                        worst = Some(format!(
                            "{name}: {label} (AOT vs JIT) output {j}: aot {a:e}, jit {b:e}, \
                             {d} ulp, x = {x:?}, p = {p:?}"
                        ));
                    }
                }
            }
        }
        max_ulp = max_ulp.max(kind_max.min(1 << 62));
        failures.extend(worst);
    }
    Ok((max_ulp, values))
}

fn run_model(name: &str, path: &Path, have_gcc: bool) -> ModelResult {
    let start = Instant::now();
    let status = run_model_inner(name, path, have_gcc);
    ModelResult {
        name: name.to_string(),
        status,
        time: start.elapsed(),
    }
}

fn run_model_inner(name: &str, path: &Path, have_gcc: bool) -> Status {
    let loaded = match load_path(path) {
        Ok(l) => l,
        Err(e) => return Status::LoadFailed(e),
    };
    if loaded
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        return Status::LoadFailed("the model has load errors".into());
    }
    let mut ir: IrModel = match lower(&loaded.expanded, &loaded.info) {
        Ok(ir) => ir,
        Err(errors) => {
            let mut kinds: Vec<IrErrorKind> = Vec::new();
            for e in &errors {
                if !kinds.contains(&e.kind) {
                    kinds.push(e.kind);
                }
            }
            let first = errors
                .first()
                .map_or_else(String::new, |e| e.message.clone());
            return Status::Refused(kinds, format!("{} error(s), first: {first}", errors.len()));
        }
    };
    let fb = Fallback::new(&loaded.expanded, &loaded.info, &ir);
    let opt = Options::new(POINTS, SEED);
    let (n, n_p, smooth) = (ir.n(), ir.n_p(), ir.smooth);
    let report = check_model(&mut ir, &fb, POINTS, SEED);
    let mut st = Stats {
        n,
        n_p,
        smooth,
        counted: 0,
        by_check: [0; 3],
        worst_exact: (0.0, "-".into()),
        worst_fd: (0.0, "-".into()),
        aot: AotResult::Skipped,
        failures: Vec::new(),
    };
    for e in &report.entries {
        let is_fd = e.label.ends_with("(FD)");
        let worst = if is_fd {
            &mut st.worst_fd
        } else {
            &mut st.worst_exact
        };
        if e.ratio > worst.0 || worst.1 == "-" {
            *worst = (e.ratio, e.label.clone());
        }
        // The FD rows of a model with kinks are information only.
        if is_fd && !smooth {
            continue;
        }
        st.counted += 1;
        st.by_check[0] += 1;
        if !e.pass {
            st.failures.push(format!("{name}: {e}"));
        }
    }
    for r in jit_rows(ir, &fb, POINTS, SEED, opt.half_width) {
        st.counted += 1;
        st.by_check[match r.kind {
            RowKind::JitVsDag => 1,
            RowKind::JitVsFallback => 2,
        }] += 1;
        if !r.pass {
            st.failures.push(format!("{name}: {}", r.text));
        }
    }
    if have_gcc {
        match aot_vs_jit(name, &loaded, opt.half_width, &mut st.failures) {
            Ok((max_ulp, values)) => st.aot = AotResult::Compared { max_ulp, values },
            Err(e) => st.failures.push(format!("{name}: AOT: {e}")),
        }
    }
    Status::Accepted(Box::new(st))
}

/// Runs the models on a few threads, each with a large stack.
fn run_all(files: &[(String, PathBuf)], have_gcc: bool) -> Vec<ModelResult> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, ModelResult)>> = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .clamp(1, 8)
        .min(files.len().max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let worker = std::thread::Builder::new()
                .stack_size(32 << 20)
                .spawn_scoped(s, || {
                    loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        let Some((name, path)) = files.get(i) else {
                            break;
                        };
                        let r = run_model(name, path, have_gcc);
                        if let Ok(mut g) = results.lock() {
                            g.push((i, r));
                        }
                    }
                });
            if let Err(e) = worker {
                panic!("cannot start a worker thread: {e}");
            }
        }
    });
    let mut all = results.into_inner().unwrap_or_default();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, r)| r).collect()
}

/// The distinct error kinds as sorted names, so two lists compare as sets.
fn kind_set(kinds: &[IrErrorKind]) -> Vec<String> {
    let mut names: Vec<String> = kinds.iter().map(|k| format!("{k:?}")).collect();
    names.sort();
    names.dedup();
    names
}

fn table(results: &[ModelResult]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:<30} {:>2} {:>3} {:<6} {:>6}  {:<30} {:<30} {:>8} {:>7}",
        "model",
        "n",
        "n_p",
        "smooth",
        "checks",
        "worst exact ratio",
        "worst FD ratio",
        "AOT ulp",
        "time"
    );
    for r in results {
        match &r.status {
            Status::Accepted(st) => {
                let aot = match st.aot {
                    AotResult::Skipped => "skipped".to_string(),
                    AotResult::Compared { max_ulp, .. } => max_ulp.to_string(),
                };
                let fd = if st.smooth {
                    format!("{:.2e} {}", st.worst_fd.0, st.worst_fd.1)
                } else {
                    format!("({:.2e} {})", st.worst_fd.0, st.worst_fd.1)
                };
                let _ = writeln!(
                    s,
                    "{:<30} {:>2} {:>3} {:<6} {:>6}  {:<30} {:<30} {:>8} {:>6.1}s",
                    r.name,
                    st.n,
                    st.n_p,
                    st.smooth,
                    st.counted,
                    format!("{:.2e} {}", st.worst_exact.0, st.worst_exact.1),
                    fd,
                    aot,
                    r.time.as_secs_f64()
                );
            }
            Status::Refused(kinds, msg) => {
                let _ = writeln!(s, "{:<30} refused: {kinds:?} ({msg})", r.name);
            }
            Status::LoadFailed(msg) => {
                let _ = writeln!(s, "{:<30} LOAD FAILED: {msg}", r.name);
            }
        }
    }
    s
}

#[test]
fn section_3_and_4_checks_pass_on_the_model_set() {
    let files = model_files();
    // The new model files are all there.
    let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    let mut problems: Vec<String> = Vec::new();
    for m in REQUIRED_MODELS {
        if !names.contains(&format!("models/{m}").as_str()) {
            problems.push(format!("models/{m} is missing"));
        }
    }
    let gcc = gcc_version();
    match &gcc {
        Some(v) => {
            record_compiler(v);
            println!("AOT compiler: {v}");
        }
        None => println!("skip: gcc not found, the AOT check is not run"),
    }
    let started = Instant::now();
    let results = run_all(&files, gcc.is_some());
    let wall = started.elapsed();
    println!("{POINTS} points per entry, seed {SEED}");
    print!("{}", table(&results));

    // The refused set must be exactly the expected one.
    let refused: BTreeMap<&str, Vec<String>> = results
        .iter()
        .filter_map(|r| match &r.status {
            Status::Refused(kinds, _) => Some((r.name.as_str(), kind_set(kinds))),
            _ => None,
        })
        .collect();
    let expected: BTreeMap<&str, Vec<String>> = EXPECTED_REFUSED
        .into_iter()
        .map(|(n, k)| (n, kind_set(k)))
        .collect();
    if refused != expected {
        problems.push(format!(
            "refused models changed: got {refused:?}, expected {expected:?}"
        ));
    }
    let mut totals = [0_usize; 3];
    let (mut accepted, mut aot_values, mut aot_models) = (0, 0, 0);
    for r in &results {
        match &r.status {
            Status::Accepted(st) => {
                accepted += 1;
                for (t, c) in totals.iter_mut().zip(st.by_check) {
                    *t += c;
                }
                if let AotResult::Compared { values, .. } = st.aot {
                    aot_models += 1;
                    aot_values += values;
                }
                problems.extend(st.failures.iter().cloned());
            }
            Status::LoadFailed(msg) => problems.push(format!("{}: load failed: {msg}", r.name)),
            Status::Refused(..) => {}
        }
    }
    println!(
        "{accepted} models accepted, {} refused, wall time {:.1}s",
        refused.len(),
        wall.as_secs_f64()
    );
    println!(
        "counted rows: {} DAG vs fallback and FD, {} JIT vs DAG, {} JIT vs fallback; \
         AOT vs JIT: {aot_models} models, {aot_values} values",
        totals[0], totals[1], totals[2]
    );
    for (name, kinds) in &refused {
        println!("refused: {name}: {kinds:?}");
    }
    assert!(
        problems.is_empty(),
        "{} problem(s):\n{}",
        problems.len(),
        problems.join("\n")
    );
    assert_eq!(accepted, EXPECTED_ACCEPTED, "accepted model count");
}
