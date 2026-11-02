//! A parsed model and the thread that owns its evaluator.
//!
//! The JIT evaluator is not `Send`, so it is built and used only on the model
//! thread. The MATLAB side sends owned requests over a channel and waits for
//! the reply, which also serializes calls from several threads.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use pc_engine::{
    EpSetup, Evaluator, ModelEval, Options, build_allowing_t, engine_kinds, entry_symbol,
    equilibrium,
};
use pc_ir::{EntryKind, IrModel};
use pc_ode::{Model, ModelInfo};

use crate::barrier::guarded;
use crate::error::{Error, Result, bad_arg};
use crate::registry::Handle;

/// Stack of the model thread. Cranelift and the lowering recurse deeply.
pub(crate) const MODEL_STACK: usize = 8 * 1024 * 1024;

/// What `model_info` reports from the lowered model.
#[derive(Clone, Debug)]
pub(crate) struct Names {
    pub states: Vec<String>,
    pub parameters: Vec<String>,
    pub aux: Vec<String>,
    pub init: Vec<f64>,
    pub par_values: Vec<f64>,
    pub autonomous: bool,
    /// SHA-256 of the exact source bytes, as 64 lower-case hex digits.
    pub source_sha256: String,
    /// The `@` options, see `parse::option_pairs`.
    pub options: Vec<(String, String)>,
}

enum Request {
    Compile(Evaluator),
    EmitC(PathBuf),
    Rhs { x: Vec<f64>, p: Vec<f64>, t: f64 },
    Equilibrium { setup: EpSetup, opts: Options },
    Panic,
}

/// The answer to an equilibrium request: the solution and its eigenvalues
/// as (re, im) in the order `pc_linalg::eigenvalues` gives.
pub(crate) struct Equilibrium {
    pub u: Vec<f64>,
    pub eig: Vec<(f64, f64)>,
}

enum Reply {
    Compiled,
    CFile(PathBuf),
    Rhs(Vec<f64>),
    Equilibrium(Equilibrium),
}

enum Msg {
    Call(Box<Request>, Sender<Result<Reply>>),
    Stop,
}

pub(crate) struct ModelHandle {
    /// The expanded model, cloned for each run.
    pub model: Model,
    pub info: ModelInfo,
    /// The lowered model. A run reads its names and defaults to check a problem.
    pub ir: IrModel,
    pub names: Names,
    /// The evaluator the model thread holds, with the library path for an AOT one.
    /// Written by that thread after a successful compile.
    evaluator: Arc<Mutex<Option<Evaluator>>>,
    /// The C file of the last successful `emit_c`.
    c_file: Mutex<Option<PathBuf>>,
    /// Written by the model thread after every request.
    max_frame: Arc<AtomicU32>,
    tx: Mutex<Option<Sender<Msg>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Handle for ModelHandle {
    const KIND: &'static str = "model";
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing panics while one of these locks is held.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn gone() -> Error {
    Error::Internal("model thread is gone".into())
}

impl ModelHandle {
    /// Starts the model thread. It owns a copy of the model.
    pub fn new(model: Model, info: ModelInfo, names: Names, ir: IrModel) -> Result<ModelHandle> {
        let (tx, rx) = mpsc::channel();
        let max_frame = Arc::new(AtomicU32::new(0));
        let evaluator = Arc::new(Mutex::new(None));
        let ctx = Context {
            model: model.clone(),
            info: info.clone(),
            warm_x: names.init.clone(),
            warm_p: names.par_values.clone(),
            max_frame: Arc::clone(&max_frame),
            evaluator: Arc::clone(&evaluator),
        };
        let thread = std::thread::Builder::new()
            .name("pckern-model".into())
            .stack_size(MODEL_STACK)
            .spawn(move || serve(ctx, rx))
            .map_err(|e| Error::Internal(format!("cannot start the model thread: {e}")))?;
        Ok(ModelHandle {
            model,
            info,
            ir,
            names,
            evaluator,
            c_file: Mutex::new(None),
            max_frame,
            tx: Mutex::new(Some(tx)),
            thread: Mutex::new(Some(thread)),
        })
    }

    pub fn n(&self) -> usize {
        self.names.states.len()
    }

    pub fn n_p(&self) -> usize {
        self.names.parameters.len()
    }

    /// The evaluator of the last successful compile, or None.
    pub fn evaluator(&self) -> Option<Evaluator> {
        lock(&self.evaluator).clone()
    }

    /// The C file written by the last successful `emit_c`, or None.
    pub fn c_file(&self) -> Option<PathBuf> {
        lock(&self.c_file).clone()
    }

    /// The largest JIT frame the model thread has compiled, 0 before any.
    pub fn max_frame(&self) -> u32 {
        self.max_frame.load(Ordering::Relaxed)
    }

    fn call(&self, req: Request) -> Result<Reply> {
        let (rtx, rrx) = mpsc::channel();
        {
            let tx = lock(&self.tx);
            let tx = tx.as_ref().ok_or_else(gone)?;
            tx.send(Msg::Call(Box::new(req), rtx)).map_err(|_| gone())?;
        }
        rrx.recv().map_err(|_| gone())?
    }

    pub fn compile(&self, which: Evaluator) -> Result<()> {
        match self.call(Request::Compile(which))? {
            Reply::Compiled => Ok(()),
            _ => Err(Error::Internal("unexpected reply to compile".into())),
        }
    }

    /// Writes the C library source for every engine entry into `dir` and records the path.
    pub fn emit_c(&self, dir: PathBuf) -> Result<PathBuf> {
        match self.call(Request::EmitC(dir))? {
            Reply::CFile(path) => {
                *lock(&self.c_file) = Some(path.clone());
                Ok(path)
            }
            _ => Err(Error::Internal("unexpected reply to emit_c".into())),
        }
    }

    pub fn rhs(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Vec<f64>> {
        match self.call(Request::Rhs { x, p, t })? {
            Reply::Rhs(out) => Ok(out),
            _ => Err(Error::Internal("unexpected reply to rhs".into())),
        }
    }

    pub fn equilibrium(&self, setup: EpSetup, opts: Options) -> Result<Equilibrium> {
        match self.call(Request::Equilibrium { setup, opts })? {
            Reply::Equilibrium(e) => Ok(e),
            _ => Err(Error::Internal("unexpected reply to equilibrium".into())),
        }
    }

    /// Test hook: panics on the model thread.
    pub fn debug_panic(&self) -> Result<()> {
        self.call(Request::Panic).map(|_| ())
    }
}

impl Drop for ModelHandle {
    fn drop(&mut self) {
        if let Some(tx) = lock(&self.tx).take() {
            let _ = tx.send(Msg::Stop);
        }
        if let Some(t) = lock(&self.thread).take() {
            // Every request ran inside `guarded`, so a join error has nothing to add.
            let _ = t.join();
        }
    }
}

/// The model thread. It ends on `Stop` or when the channel closes.
fn serve(ctx: Context, rx: Receiver<Msg>) {
    let mut ev: Option<Box<dyn ModelEval>> = None;
    while let Ok(Msg::Call(req, reply)) = rx.recv() {
        let r = guarded(|| handle(*req, &ctx, &mut ev));
        // The JIT compiles each entry on first use, so the frame can grow after any request.
        ctx.max_frame.store(pc_jit::max_frame(), Ordering::Relaxed);
        // The caller may have gone; nothing to do then.
        let _ = reply.send(r);
    }
}

/// The text of the error for a model that has no evaluator yet.
pub(crate) const NOT_COMPILED: &str = "model is not compiled; call model_compile first";

// A run reads the `IrModel` on the calling thread.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<IrModel>();
};

fn compiled(ev: &mut Option<Box<dyn ModelEval>>) -> Result<&mut dyn ModelEval> {
    match ev {
        Some(b) => Ok(&mut **b),
        None => Err(bad_arg(NOT_COMPILED)),
    }
}

/// What the model thread owns besides the evaluator.
struct Context {
    model: Model,
    info: ModelInfo,
    /// The declared start and parameters, used to compile `F` at once.
    warm_x: Vec<f64>,
    warm_p: Vec<f64>,
    max_frame: Arc<AtomicU32>,
    evaluator: Arc<Mutex<Option<Evaluator>>>,
}

fn handle(req: Request, ctx: &Context, ev: &mut Option<Box<dyn ModelEval>>) -> Result<Reply> {
    match req {
        Request::Compile(which) => {
            let mut built = build_allowing_t(&ctx.model, &ctx.info, which.clone())?;
            // The JIT compiles an entry on its first call. Run `F` once so that the
            // compile happens here and `jit_max_frame` is set. The value does not matter,
            // and a NaN at the declared start is not an error.
            let mut p = ctx.warm_p.clone();
            p.push(0.0);
            let mut out = vec![0.0; built.n()];
            let _ = built.eval(&EntryKind::F, &ctx.warm_x, &p, &[], &mut out);
            *ev = Some(built);
            *lock(&ctx.evaluator) = Some(which);
            Ok(Reply::Compiled)
        }
        Request::EmitC(dir) => Ok(Reply::CFile(write_c(&ctx.model, &ctx.info, &dir)?)),
        Request::Rhs { x, mut p, t } => {
            let ev = compiled(ev)?;
            p.push(t);
            let mut out = vec![0.0; ev.n()];
            ev.eval(&EntryKind::F, &x, &p, &[], &mut out)?;
            Ok(Reply::Rhs(out))
        }
        Request::Equilibrium { setup, opts } => {
            let ev = compiled(ev)?;
            let u = equilibrium(ev, &setup, &opts)?;
            let n = ev.n();
            let mut p = setup.params.clone();
            p.push(0.0);
            let mut jac = vec![0.0; n * n];
            ev.eval(&EntryKind::JacU, &u, &p, &[], &mut jac)?;
            // `JacU` is column-major (`out[i + n*j] = df_i/du_j`), as `eigenvalues` expects.
            let eig = pc_linalg::eigenvalues(&jac, n)
                .map_err(|e| Error::Numeric(format!("eigenvalues: {e}")))?;
            Ok(Reply::Equilibrium(Equilibrium { u, eig }))
        }
        Request::Panic => panic!("debug model panic"),
    }
}

/// FNV-1a, 64 bit. It names the C file, so it must not change between releases.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Lowers the model, emits C for every entry in `engine_kinds` and writes it to
/// `<dir>/pc_model_<hash>.c`, creating `dir` if it is missing.
fn write_c(model: &Model, info: &ModelInfo, dir: &Path) -> Result<PathBuf> {
    let mut ir = pc_ir::lower(model, info).map_err(|errs| {
        let msgs: Vec<&str> = errs.iter().map(|e| e.message.as_str()).collect();
        Error::Model(msgs.join("; "))
    })?;
    let kinds = engine_kinds();
    let mut entries = Vec::with_capacity(kinds.len());
    for k in &kinds {
        entries.push(
            ir.entry(k)
                .map_err(|e| Error::Model(format!("emit_c: {e}")))?,
        );
    }
    let named: Vec<(String, &pc_ir::Entry)> =
        kinds.iter().map(entry_symbol).zip(entries.iter()).collect();
    let text =
        pc_codegen::emit_c(&ir, &named).map_err(|e| Error::Internal(format!("emit_c: {e}")))?;
    let path = dir.join(format!("pc_model_{:016x}.c", fnv1a64(text.as_bytes())));
    std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(&path, &text))
        .map_err(|e| bad_arg(format!("emit_c: cannot write {}: {e}", path.display())))?;
    Ok(path)
}
