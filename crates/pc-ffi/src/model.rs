//! A parsed model and the thread that owns its evaluator.
//!
//! The JIT evaluator is not `Send`, so it is built and used only on the model
//! thread. The MATLAB side sends owned requests over a channel and waits for
//! the reply, which also serializes calls from several threads.

use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use pc_engine::{EpSetup, Evaluator, ModelEval, Options, build, equilibrium};
use pc_ir::EntryKind;
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
}

enum Request {
    Compile(Evaluator),
    Rhs { x: Vec<f64>, p: Vec<f64> },
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
    Rhs(Vec<f64>),
    Equilibrium(Equilibrium),
}

enum Msg {
    Call(Box<Request>, Sender<Result<Reply>>),
    Stop,
}

pub(crate) struct ModelHandle {
    /// The expanded model, cloned for each run (Task 7).
    #[allow(dead_code)]
    pub model: Model,
    #[allow(dead_code)]
    pub info: ModelInfo,
    pub names: Names,
    /// The evaluator the model thread holds: 0 none, 1 jit, 2 fallback. Written by that thread.
    evaluator: Arc<AtomicU8>,
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
    pub fn new(model: Model, info: ModelInfo, names: Names) -> Result<ModelHandle> {
        let (tx, rx) = mpsc::channel();
        let max_frame = Arc::new(AtomicU32::new(0));
        let evaluator = Arc::new(AtomicU8::new(0));
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
            names,
            evaluator,
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
        match self.evaluator.load(Ordering::Relaxed) {
            1 => Some(Evaluator::Jit),
            2 => Some(Evaluator::Fallback),
            _ => None,
        }
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

    pub fn rhs(&self, x: Vec<f64>, p: Vec<f64>) -> Result<Vec<f64>> {
        match self.call(Request::Rhs { x, p })? {
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

fn compiled(ev: &mut Option<Box<dyn ModelEval>>) -> Result<&mut dyn ModelEval> {
    match ev {
        Some(b) => Ok(&mut **b),
        None => Err(bad_arg("model is not compiled; call model_compile first")),
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
    evaluator: Arc<AtomicU8>,
}

fn handle(req: Request, ctx: &Context, ev: &mut Option<Box<dyn ModelEval>>) -> Result<Reply> {
    match req {
        Request::Compile(which) => {
            let mut built = build(&ctx.model, &ctx.info, which)?;
            // The JIT compiles an entry on its first call. Run `F` once so that the
            // compile happens here and `jit_max_frame` is set. The value does not matter,
            // and a NaN at the declared start is not an error.
            let mut p = ctx.warm_p.clone();
            p.push(0.0);
            let mut out = vec![0.0; built.n()];
            let _ = built.eval(&EntryKind::F, &ctx.warm_x, &p, &[], &mut out);
            *ev = Some(built);
            let code = if which == Evaluator::Jit { 1 } else { 2 };
            ctx.evaluator.store(code, Ordering::Relaxed);
            Ok(Reply::Compiled)
        }
        Request::Rhs { x, mut p } => {
            let ev = compiled(ev)?;
            p.push(0.0);
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
