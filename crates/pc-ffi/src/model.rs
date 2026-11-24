//! A parsed model and the thread that owns its evaluator.
//!
//! The JIT evaluator is not `Send`, so it is built and used only on the model
//! thread. The MATLAB side sends owned requests over a channel and waits for
//! the reply, which also serializes calls from several threads.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use pc_engine::{
    EngineError, EpSetup, Evaluator, IteratedMap, MapAsRoot, ModelEval, Options, build_allowing_t,
    build_dde, check_tables, engine_kinds, entry_symbol, equilibrium,
};
use pc_ensemble::Precision;
use pc_ensemble::job::{NATIVE_FRAME_LIMIT, NativeRhs};
use pc_ir::{Entry, EntryKind, IrModel};
use pc_jit::JitEntries;
use pc_ode::{Model, ModelInfo, ModelKind};

use crate::barrier::guarded;
use crate::error::{Error, MAP_MODEL, Result, bad_arg};
use crate::registry::Handle;

/// Stack of the model thread. Cranelift and the lowering recurse deeply.
pub(crate) const MODEL_STACK: usize = 8 * 1024 * 1024;

/// What `model_info` reports from the lowered model.
#[derive(Clone, Debug)]
pub(crate) struct Names {
    pub states: Vec<String>,
    pub parameters: Vec<String>,
    pub aux: Vec<String>,
    /// The names of the model's `only` list in the order of the file, empty without one.
    pub only: Vec<String>,
    pub init: Vec<f64>,
    pub par_values: Vec<f64>,
    pub autonomous: bool,
    /// SHA-256 of the exact source bytes, as 64 lower-case hex digits.
    pub source_sha256: String,
    /// The `@` options, see `parse::option_pairs`.
    pub options: Vec<(String, String)>,
}

/// The evaluator a `compile` request asks for. An AOT one gets its load tag on the model
/// thread, from the C text of this model.
pub(crate) enum Build {
    Jit,
    Fallback,
    Aot(PathBuf),
}

enum Request {
    Compile(Build),
    EmitC(PathBuf),
    Rhs {
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    },
    Equilibrium {
        setup: EpSetup,
        opts: Options,
        /// The k of `F^k` for a map (SPEC 8y); 1 for an ODE.
        iterate: usize,
    },
    /// The Jacobian of `F^iterate` at `u` (SPEC 8ac); `iterate` is 1 for an ODE.
    Jacobian {
        u: Vec<f64>,
        p: Vec<f64>,
        iterate: usize,
    },
    /// `F` at each column of `x`, `n` values a column, with the code of `Rhs` (SPEC 8ai-1b).
    RhsMany {
        x: Vec<f64>,
        p: Vec<f64>,
        t: f64,
    },
    /// The Jacobian at each column of `u`, with the code of `Jacobian` (SPEC 8ai-1b).
    JacobianMany {
        u: Vec<f64>,
        p: Vec<f64>,
        iterate: usize,
    },
    /// The solve of `Equilibrium` from each `n`-value column of `u0`, one after another
    /// (SPEC 8ai-2 with one worker).
    EquilibriumMany {
        u0: Vec<f64>,
        p: Vec<f64>,
        opts: Options,
        iterate: usize,
        hook: StartHook,
    },
    /// Compile `F` for the ensemble workers in this precision. The code stays in the model
    /// thread's slot for that precision.
    EnsembleF(Precision),
    Panic,
}

/// The answer to an equilibrium request: the solution and its eigenvalues
/// as (re, im) in the order `pc_linalg::eigenvalues` gives.
pub(crate) struct Equilibrium {
    pub u: Vec<f64>,
    pub eig: Vec<(f64, f64)>,
}

/// The answer to a many-column request (SPEC 8ai-1b): the values of the columns before the
/// first error, in column order, and that error with its 0-based column.
pub(crate) struct Many {
    pub out: Vec<f64>,
    pub failed: Option<(usize, Error)>,
}

/// The outcome of one start of `model_equilibrium_many` (SPEC 8ai-2).
pub(crate) enum Start {
    /// Converged: the bits of `model_equilibrium` from this start, and `max |G(u)|`.
    Done(Equilibrium, f64),
    /// A numeric error, with the residual the SPEC gives for it (NaN, the last Newton
    /// residual, or infinity when Newton took no step).
    Failed(f64),
}

/// The least number of starts one worker of `model_equilibrium_many` gets, the M of SPEC
/// 8ai-2. On the bistable test ODE a start costs about 2 us and building a JIT evaluator
/// for a worker about 0.5 ms, so a worker pays off from a few hundred starts; STATUS has
/// the timing.
pub(crate) const MIN_STARTS_PER_WORKER: usize = 500;

/// The number of workers for `k` starts: min(T, ceil(k / M)), at least 1 (SPEC 8ai-2).
pub(crate) fn equilibrium_workers(k: usize, threads: usize) -> usize {
    threads.min(k.div_ceil(MIN_STARTS_PER_WORKER)).max(1)
}

/// Test hook of `model_equilibrium_many`: a panic or a model error at chosen starts.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct StartHook {
    pub panic_at: Option<usize>,
    pub error_at: Vec<usize>,
}

#[cfg(test)]
impl StartHook {
    fn check(&self, j: usize) -> Result<()> {
        if self.panic_at == Some(j) {
            panic!("debug start panic at {j}");
        }
        if self.error_at.contains(&j) {
            return Err(Error::Model(format!("debug start error at {j}")));
        }
        Ok(())
    }
}

/// Outside tests the hook is empty and does nothing.
#[cfg(not(test))]
#[derive(Clone, Default)]
pub(crate) struct StartHook;

#[cfg(not(test))]
impl StartHook {
    #[inline(always)]
    fn check(&self, _j: usize) -> Result<()> {
        Ok(())
    }
}

enum Reply {
    Compiled,
    CFile(PathBuf),
    Rhs(Vec<f64>),
    Equilibrium(Equilibrium),
    Jacobian(Vec<f64>),
    Many(Many),
    Starts(Vec<Start>),
    EnsembleF(Option<NativeRhs>),
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
    pub ir: Arc<IrModel>,
    pub names: Names,
    /// True for a model with `delay(...)` calls (SPEC 8s). Only an "EP" run takes one.
    pub delay: bool,
    /// True for a map (`x(t+1)=...`, or an ODE file with `@ meth=discrete`; SPEC 8t). Only
    /// ITER and EP runs and `model_equilibrium` take one (SPEC 8u, 8v). A delay model is never
    /// a map, since lowering refuses a delay in a map.
    pub map: bool,
    /// The compiled `F` of the ensemble workers, one per precision (see [`slot_of`]), each
    /// from the first CPU run of that precision. The outer `None` means not tried yet, the
    /// inner one that the model has none (a delay model, an `F` with a frame over
    /// [`NATIVE_FRAME_LIMIT`], or a compile error). The code lives in a slot of the model
    /// thread, so it lives as long as this handle, and a new `model_compile` does not touch it.
    native: Mutex<[Option<Option<NativeRhs>>; 2]>,
    /// Test hook: when set, the model thread panics on the next ensemble `F` request.
    #[cfg(test)]
    panic_in_native: Arc<std::sync::atomic::AtomicBool>,
    /// Test hook: what `equilibrium_many` does at chosen starts.
    #[cfg(test)]
    pub(crate) start_hook: Mutex<StartHook>,
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
        let delay = !ir.delays.is_empty();
        let map = info.kind == ModelKind::Map;
        let ir = Arc::new(ir);
        #[cfg(test)]
        let panic_in_native = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ctx = Context {
            model: model.clone(),
            info: info.clone(),
            ir: Arc::clone(&ir),
            delay,
            warm_x: names.init.clone(),
            warm_p: names.par_values.clone(),
            max_frame: Arc::clone(&max_frame),
            evaluator: Arc::clone(&evaluator),
            tag: Cell::new(None),
            #[cfg(test)]
            panic_in_native: Arc::clone(&panic_in_native),
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
            delay,
            map,
            native: Mutex::new([None, None]),
            #[cfg(test)]
            panic_in_native,
            #[cfg(test)]
            start_hook: Mutex::new(StartHook::default()),
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

    /// The error for an entry that has no delay form. Call it first in such an entry.
    pub fn ode_only(&self) -> Result<()> {
        if self.delay {
            return Err(bad_arg(DELAY_MODEL));
        }
        Ok(())
    }

    /// The error for an entry that computes a result a map does not have yet (SPEC 8t, 8v).
    pub fn not_map(&self) -> Result<()> {
        if self.map {
            return Err(Error::UnsupportedModel(MAP_MODEL.into()));
        }
        Ok(())
    }

    pub fn compile(&self, which: Build) -> Result<()> {
        match self.call(Request::Compile(which))? {
            Reply::Compiled => Ok(()),
            _ => Err(Error::Internal("unexpected reply to compile".into())),
        }
    }

    /// Writes the C library source for every engine entry into `dir` and records the path.
    pub fn emit_c(&self, dir: PathBuf) -> Result<PathBuf> {
        self.ode_only()?;
        match self.call(Request::EmitC(dir))? {
            Reply::CFile(path) => {
                *lock(&self.c_file) = Some(path.clone());
                Ok(path)
            }
            _ => Err(Error::Internal("unexpected reply to emit_c".into())),
        }
    }

    pub fn rhs(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Vec<f64>> {
        self.ode_only()?;
        match self.call(Request::Rhs { x, p, t })? {
            Reply::Rhs(out) => Ok(out),
            _ => Err(Error::Internal("unexpected reply to rhs".into())),
        }
    }

    pub fn equilibrium(
        &self,
        setup: EpSetup,
        opts: Options,
        iterate: usize,
    ) -> Result<Equilibrium> {
        self.ode_only()?;
        match self.call(Request::Equilibrium {
            setup,
            opts,
            iterate,
        })? {
            Reply::Equilibrium(e) => Ok(e),
            _ => Err(Error::Internal("unexpected reply to equilibrium".into())),
        }
    }

    /// The column-major Jacobian of `F^iterate` at `u` with parameters `p` and time 0, the
    /// matrix [`Self::equilibrium`] takes its eigenvalues from (SPEC 8ac).
    pub fn jacobian(&self, u: Vec<f64>, p: Vec<f64>, iterate: usize) -> Result<Vec<f64>> {
        self.ode_only()?;
        match self.call(Request::Jacobian { u, p, iterate })? {
            Reply::Jacobian(j) => Ok(j),
            _ => Err(Error::Internal("unexpected reply to jacobian".into())),
        }
    }

    /// [`Self::rhs`] at each `n`-value column of `x` in one request to the model thread, so
    /// each column has the bits of a single call (SPEC 8ai-1b). Stops at the first error.
    pub fn rhs_many(&self, x: Vec<f64>, p: Vec<f64>, t: f64) -> Result<Many> {
        self.ode_only()?;
        match self.call(Request::RhsMany { x, p, t })? {
            Reply::Many(m) => Ok(m),
            _ => Err(Error::Internal("unexpected reply to rhs_many".into())),
        }
    }

    /// [`Self::jacobian`] at each `n`-value column of `u` in one request to the model thread,
    /// so each `n*n` block has the bits of a single call (SPEC 8ai-1b). Stops at the first
    /// error.
    pub fn jacobian_many(&self, u: Vec<f64>, p: Vec<f64>, iterate: usize) -> Result<Many> {
        self.ode_only()?;
        match self.call(Request::JacobianMany { u, p, iterate })? {
            Reply::Many(m) => Ok(m),
            _ => Err(Error::Internal("unexpected reply to jacobian_many".into())),
        }
    }

    /// [`Self::equilibrium`] from each `n`-value column of `u0` (SPEC 8ai-2). With
    /// `workers` 1 the starts run on the model thread. With more, each of
    /// `min(workers, k)` scoped threads builds its own evaluator of the kind the model
    /// thread holds and takes the next start from a shared counter, so every start runs
    /// the same code as a single call. A start's numeric error is a [`Start::Failed`]; any
    /// other error fails the call, with the error of the lowest such start.
    pub fn equilibrium_many(
        &self,
        u0: Vec<f64>,
        p: Vec<f64>,
        opts: Options,
        iterate: usize,
        workers: usize,
    ) -> Result<Vec<Start>> {
        self.ode_only()?;
        let n = self.n();
        if n == 0 || !u0.len().is_multiple_of(n) {
            return Err(Error::Internal(
                "equilibrium_many: u0 is not whole columns".into(),
            ));
        }
        let k = u0.len() / n;
        let hook = self.start_hook();
        if workers <= 1 || k <= 1 {
            return match self.call(Request::EquilibriumMany {
                u0,
                p,
                opts,
                iterate,
                hook,
            })? {
                Reply::Starts(s) => Ok(s),
                _ => Err(Error::Internal(
                    "unexpected reply to equilibrium_many".into(),
                )),
            };
        }
        let which = self.evaluator().ok_or_else(|| bad_arg(NOT_COMPILED))?;
        let next = AtomicUsize::new(0);
        // The lowest start with an error of the call so far; workers take no start above it.
        let first_fatal = AtomicUsize::new(usize::MAX);
        let worker = || {
            let mut ev: Option<Box<dyn ModelEval>> = None;
            let mut done = Vec::new();
            loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= k || j > first_fatal.load(Ordering::Relaxed) {
                    return (done, None);
                }
                let r = guarded(|| {
                    hook.check(j)?;
                    if ev.is_none() {
                        ev = Some(build_allowing_t(&self.model, &self.info, which.clone())?);
                    }
                    let ev = ev.as_deref_mut().expect("built above");
                    one_start(ev, self.map, &u0[j * n..(j + 1) * n], &p, &opts, iterate)
                });
                match r {
                    Ok(s) => done.push((j, s)),
                    Err(e) => {
                        first_fatal.fetch_min(j, Ordering::Relaxed);
                        return (done, Some((j, e)));
                    }
                }
            }
        };
        let w = workers.min(k);
        let mut parts = Vec::with_capacity(w);
        let mut spawn_err = None;
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(w);
            for _ in 0..w {
                match std::thread::Builder::new()
                    .name("pckern-equilibria".into())
                    .stack_size(MODEL_STACK)
                    .spawn_scoped(s, worker)
                {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        // The running workers stop after their current start.
                        first_fatal.store(0, Ordering::Relaxed);
                        spawn_err = Some(Error::Internal(format!(
                            "cannot start an equilibrium worker: {e}"
                        )));
                        break;
                    }
                }
            }
            for h in handles {
                // Every start runs inside the panic barrier, so a join error is a bug here.
                parts.push(h.join().map_err(|_| {
                    Error::Internal("an equilibrium worker panicked outside the barrier".into())
                }));
            }
        });
        if let Some(e) = spawn_err {
            return Err(e);
        }
        let mut starts: Vec<Option<Start>> = (0..k).map(|_| None).collect();
        let mut fatal: Option<(usize, Error)> = None;
        for part in parts {
            let (done, err) = part?;
            for (j, s) in done {
                starts[j] = Some(s);
            }
            if let Some((j, e)) = err
                && fatal.as_ref().is_none_or(|(i, _)| j < *i)
            {
                fatal = Some((j, e));
            }
        }
        if let Some((_, e)) = fatal {
            return Err(e);
        }
        starts
            .into_iter()
            .map(|s| {
                s.ok_or_else(|| Error::Internal("equilibrium_many: a start did not run".into()))
            })
            .collect()
    }

    #[cfg(test)]
    fn start_hook(&self) -> StartHook {
        lock(&self.start_hook).clone()
    }

    #[cfg(not(test))]
    fn start_hook(&self) -> StartHook {
        StartHook
    }

    /// The compiled `F` in `precision` for the ensemble workers, or `None` when the model has
    /// none (a delay model, an oversized frame, a compile error) and the run uses the tape.
    /// The first call for a precision compiles on the model thread, 0.1 to 7 ms for the models
    /// measured; later calls return the stored value.
    ///
    /// The returned function stays valid while `self` lives. A run holds an `Arc` of this
    /// handle, so it does.
    pub fn native_rhs(&self, precision: Precision) -> Result<Option<NativeRhs>> {
        let mut slots = lock(&self.native);
        let slot = &mut slots[slot_of(precision)];
        if let Some(done) = *slot {
            return Ok(done);
        }
        let got = match self.call(Request::EnsembleF(precision))? {
            Reply::EnsembleF(f) => f,
            _ => return Err(Error::Internal("unexpected reply to ensemble F".into())),
        };
        *slot = Some(got);
        Ok(got)
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
    // Declared after `ev`, so it drops first. Ensemble workers call into it only while an
    // `Arc<ModelHandle>` is alive, and the handle's `Drop` joins this thread. One slot per
    // precision, indexed by `slot_of`.
    let mut native: [Option<JitEntries>; 2] = [None, None];
    while let Ok(Msg::Call(req, reply)) = rx.recv() {
        let r = guarded(|| handle(*req, &ctx, &mut ev, &mut native));
        // The JIT compiles each entry on first use, so the frame can grow after any request.
        ctx.max_frame.store(pc_jit::max_frame(), Ordering::Relaxed);
        // The caller may have gone; nothing to do then.
        let _ = reply.send(r);
    }
}

/// The text of the error for an entry that a delay model cannot use.
pub(crate) const DELAY_MODEL: &str = "the model has delay terms, and this entry is for ODE models";

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

/// The column-major Jacobian of `it` at `u` with parameters `params` and time 0. Both
/// `model_equilibrium` and `model_jacobian` call it, so they see the same bits (SPEC 8ac).
fn jacobian_at(it: &mut IteratedMap<'_>, u: &[f64], params: &[f64]) -> Result<Vec<f64>> {
    let n = it.n();
    let mut p = params.to_vec();
    p.push(0.0);
    let mut jac = vec![0.0; n * n];
    // `JacU` is column-major (`out[i + n*j] = df_i/du_j`), as `eigenvalues` expects.
    it.eval(&EntryKind::JacU, u, &p, &[], &mut jac)?;
    Ok(jac)
}

/// The Newton solve of `model_equilibrium` through `it`: `F(u) = 0` on an ODE and
/// `F^k(u) - u = 0` on a map (SPEC 8v, 8y). Every equilibrium entry calls it, so a start
/// of `model_equilibrium_many` has the bits of a single call (SPEC 8ai-2).
fn newton(
    it: &mut IteratedMap<'_>,
    map: bool,
    setup: &EpSetup,
    opts: &Options,
) -> std::result::Result<Vec<f64>, EngineError> {
    if map {
        equilibrium(&mut MapAsRoot::new(it), setup, opts)
    } else {
        equilibrium(it, setup, opts)
    }
}

/// The eigenvalues of the Jacobian of `it` at `u`: the eigenvalues of an equilibrium, or
/// the multipliers of a fixed point of `F^k`.
fn eigen_at(it: &mut IteratedMap<'_>, u: &[f64], params: &[f64]) -> Result<Vec<(f64, f64)>> {
    let jac = jacobian_at(it, u, params)?;
    pc_linalg::eigenvalues(&jac, it.n()).map_err(|e| Error::Numeric(format!("eigenvalues: {e}")))
}

/// `max_i |G_i(u)|` for the `G` that [`newton`] solves, with the same evaluation as its
/// last residual.
fn residual_at(it: &mut IteratedMap<'_>, map: bool, u: &[f64], params: &[f64]) -> Result<f64> {
    let mut p = params.to_vec();
    p.push(0.0);
    let mut g = vec![0.0; it.n()];
    if map {
        MapAsRoot::new(it).eval(&EntryKind::F, u, &p, &[], &mut g)?;
    } else {
        it.eval(&EntryKind::F, u, &p, &[], &mut g)?;
    }
    Ok(g.iter().fold(0.0, |m: f64, v| m.max(v.abs())))
}

/// One start of `model_equilibrium_many` (SPEC 8ai-2). A start with NaN or infinity in
/// `u0` fails before any Newton step. A numeric error is a failed start; any other error
/// is the error of the call.
fn one_start(
    ev: &mut dyn ModelEval,
    map: bool,
    u0: &[f64],
    p: &[f64],
    opts: &Options,
    iterate: usize,
) -> Result<Start> {
    if !u0.iter().all(|v| v.is_finite()) {
        return Ok(Start::Failed(f64::NAN));
    }
    let failed = |e: Error| match e {
        Error::Numeric(_) => Ok(Start::Failed(f64::NAN)),
        e => Err(e),
    };
    let mut it = IteratedMap::new(ev, iterate);
    let setup = EpSetup::new(0, p.to_vec(), u0.to_vec());
    let u = match newton(&mut it, map, &setup, opts) {
        Ok(u) => u,
        Err(EngineError::NewtonFailed { residual, .. }) => return Ok(Start::Failed(residual)),
        Err(e) => return failed(e.into()),
    };
    let eig = match eigen_at(&mut it, &u, p) {
        Ok(eig) => eig,
        Err(e) => return failed(e),
    };
    let res = match residual_at(&mut it, map, &u, p) {
        Ok(r) => r,
        Err(e) => return failed(e),
    };
    Ok(Start::Done(Equilibrium { u, eig }, res))
}

/// What the model thread owns besides the evaluator.
struct Context {
    model: Model,
    info: ModelInfo,
    /// A delay model compiles through `build_dde`.
    delay: bool,
    /// The declared start and parameters, used to compile `F` at once.
    warm_x: Vec<f64>,
    warm_p: Vec<f64>,
    max_frame: Arc<AtomicU32>,
    evaluator: Arc<Mutex<Option<Evaluator>>>,
    /// The lowered model, for the ensemble `F`.
    ir: Arc<IrModel>,
    /// The `pc_aot_tag` of this model's C text, once `emit_c` or an AOT compile has made it.
    tag: Cell<Option<u64>>,
    #[cfg(test)]
    panic_in_native: Arc<std::sync::atomic::AtomicBool>,
}

/// The index of the ensemble `F` slot of a precision.
fn slot_of(precision: Precision) -> usize {
    match precision {
        Precision::Double => 0,
        Precision::Single => 1,
    }
}

/// Compiles `F` of the lowered model in `precision` into `slot` and returns a function for
/// ensemble workers of that precision. Gives `None`, and leaves `slot` empty, for a delay
/// model, for a frame over `limit` bytes and when Cranelift refuses the model (in single
/// precision it refuses the ops the `f32` tape refuses). A run then uses the tape, which
/// gives the same bits.
fn compile_native(
    ctx: &Context,
    slot: &mut Option<JitEntries>,
    limit: u32,
    precision: Precision,
) -> Option<NativeRhs> {
    if ctx.delay {
        return None;
    }
    // A second call must not replace the code that an earlier call handed out.
    if slot.is_none() {
        let entry = Entry {
            kind: EntryKind::F,
            outputs: ctx.ir.f.clone(),
        };
        // `jit_max_frame` reports the frames of the model's own evaluator, so the compile for
        // the ensemble puts the mark back.
        let mark = pc_jit::max_frame();
        let entries = [(EntryKind::F, &entry)];
        let built = match precision {
            Precision::Double => pc_jit::compile(&entries, &ctx.ir.dag),
            Precision::Single => pc_jit::compile_f32(&entries, &ctx.ir.dag),
        };
        pc_jit::set_max_frame(mark);
        let je = built.ok()?;
        let has_f = match precision {
            Precision::Double => je.plain(&EntryKind::F).is_some(),
            Precision::Single => je.plain_f32(&EntryKind::F).is_some(),
        };
        if je.frame() > limit || !has_f {
            return None;
        }
        *slot = Some(je);
    }
    let je = slot.as_ref()?;
    // SAFETY: `f` is the `F` entry of this model with `n` outputs, built by the JIT in the
    // precision of the run, which gives the bits of the tape of that precision (the 0 ulp
    // tests in pc-ensemble) and keeps no state, so threads may call it at once. Its frame was
    // within `limit` when it was stored. `slot` keeps the code until the model thread ends,
    // and the handle joins that thread only after every run is done.
    let f = match precision {
        Precision::Double => unsafe { NativeRhs::new(je.plain(&EntryKind::F)?) },
        Precision::Single => unsafe { NativeRhs::new_f32(je.plain_f32(&EntryKind::F)?) },
    };
    Some(f)
}

impl Context {
    /// The tag a library built from this model's C text reports.
    fn aot_tag(&self) -> Result<u64> {
        if let Some(t) = self.tag.get() {
            return Ok(t);
        }
        let t = emit_model(&self.model, &self.info)?.tag;
        self.tag.set(Some(t));
        Ok(t)
    }
}

fn handle(
    req: Request,
    ctx: &Context,
    ev: &mut Option<Box<dyn ModelEval>>,
    native: &mut [Option<JitEntries>; 2],
) -> Result<Reply> {
    match req {
        Request::EnsembleF(precision) => {
            #[cfg(test)]
            if ctx.panic_in_native.load(Ordering::Relaxed) {
                panic!("debug ensemble F panic");
            }
            Ok(Reply::EnsembleF(compile_native(
                ctx,
                &mut native[slot_of(precision)],
                NATIVE_FRAME_LIMIT,
                precision,
            )))
        }
        Request::Compile(build) => {
            let mut p = ctx.warm_p.clone();
            let which = match build {
                Build::Jit => Evaluator::Jit,
                Build::Fallback => Evaluator::Fallback,
                // `build_dde` refuses an AOT evaluator before it reads the tag.
                Build::Aot(lib) => Evaluator::Aot {
                    tag: if ctx.delay { 0 } else { ctx.aot_tag()? },
                    lib,
                },
            };
            let mut built = if ctx.delay {
                // The hidden delay slots follow the declared parameters, then `t`.
                let dm = build_dde(&ctx.model, &ctx.info, which.clone())?;
                p.resize(p.len() + dm.delays.len(), 0.0);
                dm.eval
            } else {
                build_allowing_t(&ctx.model, &ctx.info, which.clone())?
            };
            // The JIT compiles an entry on its first call. Run `F` once so that the
            // compile happens here and `jit_max_frame` is set. The value does not matter,
            // and a NaN at the declared start is not an error. A recursion cap hit at the
            // declared start is not one either: the parameters can change before a run, and
            // the run reports the cap. A model error is one, since the model cannot run: the
            // JIT could not compile an entry, or a recursive call failed in the call helper.
            p.push(0.0);
            let mut out = vec![0.0; built.n()];
            if let Err(e @ pc_engine::EngineError::Model(_)) =
                built.eval(&EntryKind::F, &ctx.warm_x, &p, &[], &mut out)
            {
                return Err(e.into());
            }
            *ev = Some(built);
            *lock(&ctx.evaluator) = Some(which);
            Ok(Reply::Compiled)
        }
        Request::EmitC(dir) => {
            // `evaluator: "none"` builds nothing that would make this check.
            check_tables(&ctx.model, &ctx.info)?;
            let (path, tag) = write_c(&ctx.model, &ctx.info, &dir)?;
            ctx.tag.set(Some(tag));
            Ok(Reply::CFile(path))
        }
        Request::Rhs { x, mut p, t } => {
            let ev = compiled(ev)?;
            p.push(t);
            let mut out = vec![0.0; ev.n()];
            ev.eval(&EntryKind::F, &x, &p, &[], &mut out)?;
            Ok(Reply::Rhs(out))
        }
        Request::Equilibrium {
            setup,
            opts,
            iterate,
        } => {
            let ev = compiled(ev)?;
            // A map solves G = F^k - x = 0, and the eigenvalues of the Jacobian of F^k at the
            // fixed point are its multipliers (SPEC 8v and 8y). k = 1 evaluates F itself, and
            // an ODE always has k = 1.
            let mut it = IteratedMap::new(ev, iterate);
            let u = newton(&mut it, ctx.info.kind == ModelKind::Map, &setup, &opts)?;
            let eig = eigen_at(&mut it, &u, &setup.params)?;
            Ok(Reply::Equilibrium(Equilibrium { u, eig }))
        }
        Request::EquilibriumMany {
            u0,
            p,
            opts,
            iterate,
            hook,
        } => {
            let ev = compiled(ev)?;
            let n = ev.n();
            if n == 0 || !u0.len().is_multiple_of(n) {
                return Err(Error::Internal(
                    "equilibrium_many: u0 is not whole columns".into(),
                ));
            }
            let map = ctx.info.kind == ModelKind::Map;
            let mut out = Vec::with_capacity(u0.len() / n);
            for (j, c) in u0.chunks_exact(n).enumerate() {
                hook.check(j)?;
                out.push(one_start(ev, map, c, &p, &opts, iterate)?);
            }
            Ok(Reply::Starts(out))
        }
        Request::Jacobian { u, p, iterate } => {
            let ev = compiled(ev)?;
            let mut it = IteratedMap::new(ev, iterate);
            Ok(Reply::Jacobian(jacobian_at(&mut it, &u, &p)?))
        }
        Request::RhsMany { x, mut p, t } => {
            let ev = compiled(ev)?;
            let n = ev.n();
            if n == 0 || !x.len().is_multiple_of(n) {
                return Err(Error::Internal("rhs_many: x is not whole columns".into()));
            }
            p.push(t);
            let mut out = vec![0.0; x.len()];
            for (j, (xc, oc)) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)).enumerate() {
                if let Err(e) = ev.eval(&EntryKind::F, xc, &p, &[], oc) {
                    out.truncate(j * n);
                    return Ok(Reply::Many(Many {
                        out,
                        failed: Some((j, e.into())),
                    }));
                }
            }
            Ok(Reply::Many(Many { out, failed: None }))
        }
        Request::JacobianMany { u, p, iterate } => {
            let ev = compiled(ev)?;
            let mut it = IteratedMap::new(ev, iterate);
            let n = it.n();
            if n == 0 || !u.len().is_multiple_of(n) {
                return Err(Error::Internal(
                    "jacobian_many: u is not whole columns".into(),
                ));
            }
            let mut p = p;
            p.push(0.0);
            let nn = n * n;
            let mut out = vec![0.0; u.len() * n];
            for (j, (uc, oc)) in u.chunks_exact(n).zip(out.chunks_exact_mut(nn)).enumerate() {
                // The same `JacU` call as `jacobian_at`, so each block has its bits.
                if let Err(e) = it.eval(&EntryKind::JacU, uc, &p, &[], oc) {
                    out.truncate(j * nn);
                    return Ok(Reply::Many(Many {
                        out,
                        failed: Some((j, e.into())),
                    }));
                }
            }
            Ok(Reply::Many(Many { out, failed: None }))
        }
        Request::Panic => panic!("debug model panic"),
    }
}

/// Lowers the model and emits C for every entry in `engine_kinds`, with the load tag.
fn emit_model(model: &Model, info: &ModelInfo) -> Result<pc_codegen::EmittedC> {
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
    // Every refusal of the emitter is a property of the model: two names with one C
    // identifier, a reserved identifier, or a recursive call (SPEC 8af-5).
    pc_codegen::emit_c_tagged(&ir, &named).map_err(|e| Error::Model(format!("emit_c: {e}")))
}

/// Writes the C text of the model to `<dir>/pc_model_<hash>.c`, creating `dir` if it is
/// missing. The file name hashes the text with FNV-1a 64, which must not change between
/// releases. Returns the path and the load tag.
fn write_c(model: &Model, info: &ModelInfo, dir: &Path) -> Result<(PathBuf, u64)> {
    let c = emit_model(model, info)?;
    let path = dir.join(format!(
        "pc_model_{:016x}.c",
        pc_codegen::fnv1a64(c.text.as_bytes())
    ));
    std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(&path, &c.text))
        .map_err(|e| bad_arg(format!("emit_c: cannot write {}: {e}", path.display())))?;
    Ok((path, c.tag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_model;
    use crate::registry;
    use pc_ensemble::device::{Backend, find_device};
    use pc_ensemble::job::{Method, Output, Params};
    use pc_ensemble::run::run_on;
    use pc_ensemble::{EnsembleResult, Job, Precision};
    use std::sync::atomic::AtomicBool;

    const VDP: &str = "par mu=1.5\ninit x=2\ninit y=0\nx'=y\ny'=mu*(1-x*x)*y-x\ndone\n";
    const HUTCH: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

    fn parsed(text: &str) -> crate::parse::Parsed {
        parse_model(text.into()).unwrap().result.unwrap()
    }

    fn handle_of(text: &str) -> ModelHandle {
        let p = parsed(text);
        ModelHandle::new(p.model, p.info, p.names, p.ir).unwrap()
    }

    fn context_of(text: &str) -> Context {
        let p = parsed(text);
        Context {
            delay: !p.ir.delays.is_empty(),
            warm_x: p.names.init.clone(),
            warm_p: p.names.par_values.clone(),
            max_frame: Arc::new(AtomicU32::new(0)),
            evaluator: Arc::new(Mutex::new(None)),
            ir: Arc::new(p.ir),
            model: p.model,
            info: p.info,
            tag: Cell::new(None),
            panic_in_native: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A function pointer has no `PartialEq` here, so compare the address it prints.
    fn address(f: Option<NativeRhs>) -> String {
        format!("{:?}", f.expect("a native function"))
    }

    fn same_bits(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    fn run(h: &ModelHandle, method: Method, native: Option<NativeRhs>) -> EnsembleResult {
        run_in(h, method, native, Precision::Double)
    }

    fn run_in(
        h: &ModelHandle,
        method: Method,
        native: Option<NativeRhs>,
        precision: Precision,
    ) -> EnsembleResult {
        let members = 7;
        let x0: Vec<f64> = (0..members)
            .flat_map(|k| [1.0 + 0.3 * k as f64, 0.1 * k as f64])
            .collect();
        let p = [1.5];
        let job = Job {
            model: &h.ir,
            members,
            x0: &x0,
            params: Params::Shared(&p),
            t0: 0.0,
            t1: 6.0,
            output: Output::Samples(vec![1.0, 2.5, 6.0]),
            method,
            bounds: f64::INFINITY,
            max_steps: 1_000_000,
            precision,
            math: pc_ensemble::Math::Exact,
            native,
            stack: pc_ensemble::StackLayout::Member,
        };
        let device = find_device(Backend::Cpu, 0).unwrap();
        run_on(&device, &job, &AtomicBool::new(false)).unwrap()
    }

    #[test]
    fn a_frame_over_the_limit_gives_no_native_function() {
        // A small model has no stack frame, so the guard needs a model that has one.
        let ode: String = (0..40)
            .map(|i| {
                let (a, b, c) = ((i + 39) % 40, (i + 1) % 40, (i + 38) % 40);
                format!("x{i}'=(x{b}-x{c})*x{a}-x{i}+8\ninit x{i}=1\n")
            })
            .collect::<String>()
            + "done\n";
        let ctx = context_of(&ode);
        let mut slot = None;
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Double).is_some());
        let frame = slot.as_ref().unwrap().frame();
        assert!(frame > 0, "the 40-state model should need a stack frame");
        let mut refused = None;
        assert!(compile_native(&ctx, &mut refused, frame - 1, Precision::Double).is_none());
        assert!(refused.is_none(), "a refused compile must not keep code");
        assert!(compile_native(&ctx, &mut refused, frame, Precision::Double).is_some());
        assert!(refused.is_some());
    }

    #[test]
    fn a_delay_model_gets_no_native_function() {
        let ctx = context_of(HUTCH);
        let mut slot = None;
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Double).is_none());
        assert!(slot.is_none());
        assert!(
            handle_of(HUTCH)
                .native_rhs(Precision::Double)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_second_compile_keeps_the_first_code() {
        let ctx = context_of(VDP);
        let mut slot = None;
        let a = address(compile_native(
            &ctx,
            &mut slot,
            NATIVE_FRAME_LIMIT,
            Precision::Double,
        ));
        let b = address(compile_native(
            &ctx,
            &mut slot,
            NATIVE_FRAME_LIMIT,
            Precision::Double,
        ));
        assert_eq!(a, b);
    }

    #[test]
    fn the_handle_compiles_once_and_a_recompile_leaves_the_code_alone() {
        let h = handle_of(VDP);
        let first = h.native_rhs(Precision::Double).unwrap();
        h.compile(Build::Jit).unwrap();
        h.compile(Build::Jit).unwrap();
        let second = h.native_rhs(Precision::Double).unwrap();
        assert_eq!(address(first), address(second));
        // The code from before the recompiles still runs, and still gives the tape's bits.
        let method = Method::Rk4 { dt: 0.01 };
        let with = run(&h, method, first);
        let without = run(&h, method, None);
        assert!(same_bits(&with.states, &without.states));
    }

    #[test]
    fn a_run_with_the_compiled_f_has_the_bits_of_the_tape() {
        let h = handle_of(VDP);
        let native = h.native_rhs(Precision::Double).unwrap();
        assert!(native.is_some(), "an ODE model gets a native function");
        for method in [
            Method::Rk4 { dt: 0.01 },
            Method::Dp5 {
                rtol: 1e-6,
                atol: 1e-9,
                hmax: f64::INFINITY,
            },
        ] {
            let with = run(&h, method, native);
            let without = run(&h, method, None);
            assert!(same_bits(&with.states, &without.states), "{method:?}");
            assert_eq!(with.status, without.status, "{method:?}");
            assert!(same_bits(&with.t_reached, &without.t_reached), "{method:?}");
            assert_eq!(with.accepted, without.accepted, "{method:?}");
            assert_eq!(with.rejected, without.rejected, "{method:?}");
        }
    }

    /// Forty coupled states, so the compiled `F` has a stack frame.
    fn lorenz96_40() -> String {
        (0..40)
            .map(|i| {
                let (a, b, c) = ((i + 39) % 40, (i + 1) % 40, (i + 38) % 40);
                format!("x{i}'=(x{b}-x{c})*x{a}-x{i}+8\ninit x{i}=1\n")
            })
            .collect::<String>()
            + "done\n"
    }

    /// Uses `t`, two parameters, `x^3` and `y^2` (powi), `sin`, `cos`, `exp` and a branch.
    const WIDE: &str = "par a=1.2, b=0.4\ninit x=0.5\ninit y=0.1\n\
x'=y-a*x^3+b*sin(2*t)\n\
y'=if(x>0)then(-x*y^2+cos(y))else(x^2*y-exp(-y*y))-0.1*y\n\
done\n";

    /// Seven members with their own `(a, b)`, on the model `WIDE`.
    fn run_wide(
        h: &ModelHandle,
        method: Method,
        native: Option<NativeRhs>,
        precision: Precision,
    ) -> EnsembleResult {
        let members = 7;
        let x0: Vec<f64> = (0..members)
            .flat_map(|k| [0.5 - 0.2 * k as f64, 0.1 * k as f64])
            .collect();
        let p: Vec<f64> = (0..members)
            .flat_map(|k| [0.8 + 0.15 * k as f64, 0.1 + 0.1 * k as f64])
            .collect();
        let job = Job {
            model: &h.ir,
            members,
            x0: &x0,
            params: Params::PerMember(&p),
            t0: 0.0,
            t1: 6.0,
            output: Output::Samples(vec![1.0, 2.5, 6.0]),
            method,
            bounds: f64::INFINITY,
            max_steps: 1_000_000,
            precision,
            math: pc_ensemble::Math::Exact,
            native,
            stack: pc_ensemble::StackLayout::Member,
        };
        let device = find_device(Backend::Cpu, 0).unwrap();
        run_on(&device, &job, &AtomicBool::new(false)).unwrap()
    }

    #[test]
    fn a_wide_model_with_t_per_member_parameters_and_a_branch_has_the_bits_of_the_tape() {
        let h = handle_of(WIDE);
        let native = h.native_rhs(Precision::Double).unwrap();
        assert!(native.is_some(), "an ODE model gets a native function");
        for method in [
            Method::Rk4 { dt: 0.01 },
            Method::Dp5 {
                rtol: 1e-6,
                atol: 1e-9,
                hmax: f64::INFINITY,
            },
        ] {
            let with = run_wide(&h, method, native, Precision::Double);
            let without = run_wide(&h, method, None, Precision::Double);
            assert!(same_bits(&with.states, &without.states), "{method:?}");
            assert_eq!(with.status, without.status, "{method:?}");
            assert!(same_bits(&with.t_reached, &without.t_reached), "{method:?}");
            assert_eq!(with.accepted, without.accepted, "{method:?}");
            assert_eq!(with.rejected, without.rejected, "{method:?}");
        }
    }

    const JOB: &str = r#"{"backend":"cpu","precision":"double","t0":0,"t1":1,"method":{"type":"rk4","dt":0.01},"output":"final","bounds":null,"max_steps":100000,"params":"shared"}"#;

    #[test]
    fn a_model_the_tape_refuses_is_not_compiled() {
        // The JIT takes `erf`, the ensemble tape does not.
        let h = Arc::new(handle_of("par k=1\ninit x=1\nx'=-k*erf(x)\ndone\n"));
        let id = crate::ensemble::start(&h, JOB, &[1.0], &[1.0]).expect("the run starts");
        assert!(
            lock(&h.native).iter().all(Option::is_none),
            "a refused model must not be compiled"
        );
        registry::remove(id).unwrap();
    }

    #[test]
    fn a_model_the_tape_takes_is_compiled_by_the_first_double_cpu_run() {
        let h = Arc::new(handle_of(VDP));
        assert!(lock(&h.native).iter().all(Option::is_none));
        let id = crate::ensemble::start(&h, JOB, &[1.0, 0.0], &[1.5]).unwrap();
        assert!(matches!(
            lock(&h.native)[slot_of(Precision::Double)],
            Some(Some(_))
        ));
        assert!(
            lock(&h.native)[slot_of(Precision::Single)].is_none(),
            "a double run compiles only the double F"
        );
        registry::remove(id).unwrap();
    }

    #[test]
    fn the_ensemble_compile_does_not_raise_the_reported_jit_frame() {
        let h = handle_of(&lorenz96_40());
        let before = h.max_frame();
        assert!(h.native_rhs(Precision::Double).unwrap().is_some());
        assert_eq!(h.max_frame(), before);
        // A compile on this thread leaves the mark where it was, too.
        let ctx = context_of(&lorenz96_40());
        let mut slot = None;
        let mark = pc_jit::max_frame();
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Double).is_some());
        assert!(slot.as_ref().unwrap().frame() > 0);
        assert_eq!(pc_jit::max_frame(), mark);
    }

    /// The two DP5 settings and RK4, with tolerances the single-precision check accepts.
    fn methods_of(precision: Precision) -> [Method; 2] {
        let (rtol, atol) = match precision {
            Precision::Double => (1e-6, 1e-9),
            Precision::Single => (1e-4, 1e-6),
        };
        [
            Method::Rk4 { dt: 0.01 },
            Method::Dp5 {
                rtol,
                atol,
                hmax: f64::INFINITY,
            },
        ]
    }

    fn assert_same_run(with: &EnsembleResult, without: &EnsembleResult, what: &str) {
        assert!(same_bits(&with.states, &without.states), "{what}");
        assert_eq!(with.status, without.status, "{what}");
        assert!(same_bits(&with.t_reached, &without.t_reached), "{what}");
        assert_eq!(with.accepted, without.accepted, "{what}");
        assert_eq!(with.rejected, without.rejected, "{what}");
    }

    #[test]
    fn a_single_run_with_the_compiled_f_has_the_bits_of_the_f32_tape() {
        let h = handle_of(VDP);
        let native = h.native_rhs(Precision::Single).unwrap();
        assert_eq!(
            native.map(NativeRhs::precision),
            Some(Precision::Single),
            "an ODE model gets a single-precision function"
        );
        for method in methods_of(Precision::Single) {
            let with = run_in(&h, method, native, Precision::Single);
            let without = run_in(&h, method, None, Precision::Single);
            assert_same_run(&with, &without, &format!("{method:?}"));
        }
        let h = handle_of(WIDE);
        let native = h.native_rhs(Precision::Single).unwrap();
        assert!(native.is_some());
        for method in methods_of(Precision::Single) {
            let with = run_wide(&h, method, native, Precision::Single);
            let without = run_wide(&h, method, None, Precision::Single);
            assert_same_run(&with, &without, &format!("wide {method:?}"));
        }
    }

    #[test]
    fn each_precision_has_its_own_compiled_f() {
        let h = handle_of(VDP);
        let single = h.native_rhs(Precision::Single).unwrap();
        let double = h.native_rhs(Precision::Double).unwrap();
        assert_eq!(single.map(NativeRhs::precision), Some(Precision::Single));
        assert_eq!(double.map(NativeRhs::precision), Some(Precision::Double));
        assert_ne!(address(single), address(double));
        // Later calls return the stored code of each precision.
        assert_eq!(
            address(single),
            address(h.native_rhs(Precision::Single).unwrap())
        );
        assert_eq!(
            address(double),
            address(h.native_rhs(Precision::Double).unwrap())
        );
        // A function of the wrong precision is ignored, so the run keeps the tape's bits.
        let method = Method::Rk4 { dt: 0.01 };
        let crossed = run_in(&h, method, double, Precision::Single);
        let tape = run_in(&h, method, None, Precision::Single);
        assert_same_run(&crossed, &tape, "double F on a single run");
    }

    #[test]
    fn single_precision_refuses_what_the_f32_tape_refuses() {
        // `erf` is in the double JIT but not in the f32 one.
        let ctx = context_of("par k=1\ninit x=1\nx'=-k*erf(x)\ndone\n");
        let mut slot = None;
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Single).is_none());
        assert!(slot.is_none(), "a refused compile must not keep code");
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Double).is_some());
        // The frame guard holds in single precision too.
        let ctx = context_of(&lorenz96_40());
        let mut slot = None;
        assert!(compile_native(&ctx, &mut slot, NATIVE_FRAME_LIMIT, Precision::Single).is_some());
        let frame = slot.as_ref().unwrap().frame();
        assert!(frame > 0, "the 40-state model should need a stack frame");
        let mut refused = None;
        assert!(compile_native(&ctx, &mut refused, frame - 1, Precision::Single).is_none());
        assert!(refused.is_none());
    }

    const JOB_SINGLE: &str = r#"{"backend":"cpu","precision":"single","t0":0,"t1":1,"method":{"type":"rk4","dt":0.01},"output":"final","bounds":null,"max_steps":100000,"params":"shared"}"#;

    #[test]
    fn a_model_the_tape_takes_is_compiled_by_the_first_single_cpu_run() {
        let h = Arc::new(handle_of(VDP));
        let id = crate::ensemble::start(&h, JOB_SINGLE, &[1.0, 0.0], &[1.5]).unwrap();
        assert!(matches!(
            lock(&h.native)[slot_of(Precision::Single)],
            Some(Some(_))
        ));
        assert!(
            lock(&h.native)[slot_of(Precision::Double)].is_none(),
            "a single run compiles only the single F"
        );
        registry::remove(id).unwrap();
    }

    #[test]
    fn a_panic_while_compiling_the_ensemble_f_fails_the_start() {
        for job in [JOB, JOB_SINGLE] {
            let h = Arc::new(handle_of(VDP));
            h.panic_in_native.store(true, Ordering::Relaxed);
            let e = crate::ensemble::start(&h, job, &[1.0, 0.0], &[1.5])
                .expect_err("the start must fail");
            assert!(matches!(e, Error::Panic(_)), "{job}: {e}");
            assert!(e.to_string().contains("debug ensemble F panic"), "{e}");
            assert!(
                lock(&h.native).iter().all(Option::is_none),
                "a failed compile stores nothing"
            );
            // The model thread survives the panic, and the next start compiles.
            h.panic_in_native.store(false, Ordering::Relaxed);
            let id = crate::ensemble::start(&h, job, &[1.0, 0.0], &[1.5]).unwrap();
            registry::remove(id).unwrap();
        }
    }

    /// Columns of a grid on [-2, 2]^2, `k` of them.
    fn grid(k: usize) -> Vec<f64> {
        (0..k)
            .flat_map(|j| {
                let s = j as f64 / k.max(1) as f64;
                [-2.0 + 4.0 * s, 2.0 - 3.7 * s * s]
            })
            .collect()
    }

    #[test]
    fn rhs_many_has_the_bits_of_single_calls() {
        // SPEC 8ai-1b, test 1.
        let h = handle_of(VDP);
        h.compile(Build::Jit).unwrap();
        let x = grid(37);
        let m = h.rhs_many(x.clone(), vec![1.5], 0.25).unwrap();
        assert!(m.failed.is_none());
        let single: Vec<f64> = x
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| h.rhs(c.to_vec(), vec![1.5], 0.25).unwrap())
            .collect();
        assert!(same_bits(&m.out, &single));
    }

    #[test]
    fn rhs_many_passes_t_to_every_column() {
        // SPEC 8ai-1b, test 1, on a model that depends on t, so a wrong or dropped t
        // changes the bits.
        let h = handle_of("par r=2\nx'=r*x*(1-x)+t\ny'=-y*t\ndone\n");
        h.compile(Build::Jit).unwrap();
        let x = grid(37);
        for t in [0.25, 0.75] {
            let m = h.rhs_many(x.clone(), vec![2.0], t).unwrap();
            assert!(m.failed.is_none());
            let single: Vec<f64> = x
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|c| h.rhs(c.to_vec(), vec![2.0], t).unwrap())
                .collect();
            assert!(same_bits(&m.out, &single), "t = {t}");
        }
        let a = h.rhs_many(x.clone(), vec![2.0], 0.25).unwrap();
        let b = h.rhs_many(x, vec![2.0], 0.75).unwrap();
        assert!(!same_bits(&a.out, &b.out));
    }

    #[test]
    fn jacobian_many_has_the_bits_of_single_calls() {
        // SPEC 8ai-1b, test 1, for an ODE and for a map at iterate 3.
        let h = handle_of(VDP);
        h.compile(Build::Jit).unwrap();
        let u = grid(37);
        let m = h.jacobian_many(u.clone(), vec![1.5], 1).unwrap();
        assert!(m.failed.is_none());
        let single: Vec<f64> = u
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| h.jacobian(c.to_vec(), vec![1.5], 1).unwrap())
            .collect();
        assert!(same_bits(&m.out, &single));

        let h = handle_of("par a=1.4, b=0.3\nx(t+1)=1-a*x*x+y\ny(t+1)=b*x\ndone\n");
        h.compile(Build::Jit).unwrap();
        let m = h.jacobian_many(u.clone(), vec![1.4, 0.3], 3).unwrap();
        assert!(m.failed.is_none());
        let single: Vec<f64> = u
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| h.jacobian(c.to_vec(), vec![1.4, 0.3], 3).unwrap())
            .collect();
        assert!(same_bits(&m.out, &single));
    }

    #[test]
    fn many_calls_stop_at_the_first_refused_column() {
        // SPEC 8ai-1b, test 2: sqrt(x+1) is NaN at x = -2 and its slope is not finite at
        // x = -1. The columns before the refused one come back with their single-call
        // bits, and the error is the single call's.
        let h = handle_of("x'=sqrt(x+1)\ny'=-y\ndone\n");
        h.compile(Build::Jit).unwrap();
        let x = vec![0.0, 0.0, 1.0, 1.0, -2.0, 0.0, 2.0, 3.0];
        let m = h.rhs_many(x.clone(), vec![], 0.0).unwrap();
        let (j, e) = m.failed.expect("column 2 is refused");
        assert_eq!(j, 2);
        let single = h.rhs(x[4..6].to_vec(), vec![], 0.0).unwrap_err();
        assert_eq!(e.to_string(), single.to_string());
        assert_eq!(e.code(), single.code());
        let first: Vec<f64> = x[..4]
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| h.rhs(c.to_vec(), vec![], 0.0).unwrap())
            .collect();
        assert!(same_bits(&m.out, &first));

        let u = vec![0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 2.0, 3.0];
        let m = h.jacobian_many(u.clone(), vec![], 1).unwrap();
        let (j, e) = m.failed.expect("column 2 is refused");
        assert_eq!(j, 2);
        let single = h.jacobian(u[4..6].to_vec(), vec![], 1).unwrap_err();
        assert_eq!(e.to_string(), single.to_string());
        assert_eq!(e.code(), single.code());
        let first: Vec<f64> = u[..4]
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| h.jacobian(c.to_vec(), vec![], 1).unwrap())
            .collect();
        assert!(same_bits(&m.out, &first));
    }

    const BISTABLE: &str = "x'=x-x^3\ny'=-y\ndone\n";
    const HENON: &str = "par a=1.4, b=0.3\nx(t+1)=1-a*x*x+y\ny(t+1)=b*x\ndone\n";

    fn newton_opts(max_newton: usize) -> Options {
        Options {
            max_newton,
            ..Options::default()
        }
    }

    /// Columns of a 2-D grid on [-1.5, 1.5]^2, `side` by `side`.
    fn box_starts(side: usize) -> Vec<f64> {
        let at = |i: usize| -1.5 + 3.0 * i as f64 / (side - 1) as f64;
        (0..side * side)
            .flat_map(|j| [at(j % side), at(j / side)])
            .collect()
    }

    fn eig_bits(e: &[(f64, f64)]) -> Vec<f64> {
        e.iter().flat_map(|&(re, im)| [re, im]).collect()
    }

    /// `max_i |G_i(u)|` through `model_rhs`: `F(u)` on an ODE, `F^k(u) - u` on a map.
    fn residual_by_rhs(h: &ModelHandle, u: &[f64], p: &[f64], map: bool, k: usize) -> f64 {
        let g = if map {
            let mut v = u.to_vec();
            for _ in 0..k {
                v = h.rhs(v, p.to_vec(), 0.0).unwrap();
            }
            v.iter().zip(u).map(|(a, b)| a - b).collect()
        } else {
            h.rhs(u.to_vec(), p.to_vec(), 0.0).unwrap()
        };
        g.iter().fold(0.0, |m: f64, v| m.max(v.abs()))
    }

    /// Checks every start of `got` against a single `equilibrium` call from its column.
    /// Returns the number of converged starts.
    #[allow(clippy::too_many_arguments)]
    fn assert_starts_match_single_calls(
        h: &ModelHandle,
        got: &[Start],
        u0: &[f64],
        p: &[f64],
        opts: &Options,
        map: bool,
        k: usize,
        what: &str,
    ) -> usize {
        let n = h.ir.n();
        assert_eq!(got.len(), u0.len() / n, "{what}");
        let mut done = 0;
        for (j, (s, c)) in got.iter().zip(u0.chunks_exact(n)).enumerate() {
            let single = h.equilibrium(EpSetup::new(0, p.to_vec(), c.to_vec()), opts.clone(), k);
            match (s, single) {
                (Start::Done(e, res), Ok(one)) => {
                    assert!(same_bits(&e.u, &one.u), "{what}: u at start {j}");
                    assert!(
                        same_bits(&eig_bits(&e.eig), &eig_bits(&one.eig)),
                        "{what}: eigenvalues at start {j}"
                    );
                    let by_rhs = residual_by_rhs(h, &e.u, p, map, k);
                    assert_eq!(
                        res.to_bits(),
                        by_rhs.to_bits(),
                        "{what}: residual at start {j}"
                    );
                    done += 1;
                }
                (Start::Failed(_), Err(e)) => {
                    assert_eq!(
                        e.code(),
                        Error::Numeric(String::new()).code(),
                        "{what}: {j}"
                    );
                }
                (Start::Done(..), Err(e)) => panic!("{what}: start {j} converged, single: {e}"),
                (Start::Failed(r), Ok(_)) => panic!("{what}: start {j} failed ({r}), single ok"),
            }
        }
        done
    }

    #[test]
    fn equilibrium_many_has_the_bits_of_single_calls() {
        // SPEC 8ai-2, test 1, for the JIT and the fallback (the AOT case runs through the
        // C ABI in tests/equilibria.rs).
        let u0 = box_starts(5);
        let k = u0.len() / 2;
        let cases: [(&str, Vec<f64>, usize); 3] = [
            (BISTABLE, vec![], 1),
            (HENON, vec![1.4, 0.3], 1),
            (HENON, vec![1.4, 0.3], 2),
        ];
        for (build, name) in [(Build::Jit, "jit"), (Build::Fallback, "fallback")] {
            for (text, p, it) in &cases {
                let h = handle_of(text);
                h.compile(match build {
                    Build::Jit => Build::Jit,
                    _ => Build::Fallback,
                })
                .unwrap();
                let map = h.map;
                let opts = newton_opts(50);
                for w in [1, 2, 3, k + 1] {
                    let got = h
                        .equilibrium_many(u0.clone(), p.clone(), opts.clone(), *it, w)
                        .unwrap();
                    let what = format!("{name}, iterate {it}, map {map}, W = {w}");
                    let done =
                        assert_starts_match_single_calls(&h, &got, &u0, p, &opts, map, *it, &what);
                    assert!(done >= 3, "{what}: only {done} starts converged");
                }
            }
        }
    }

    #[test]
    fn failed_starts_follow_the_residual_rule() {
        // SPEC 8ai-2, test 2. The Jacobian of x' = x^2 - 1 is singular at x = 0.
        let h = handle_of("x'=x*x-1\ny'=-y\ndone\n");
        h.compile(Build::Jit).unwrap();
        let u0 = vec![
            2.0,
            0.3,
            0.0,
            0.5,
            f64::NAN,
            0.0,
            0.5,
            f64::INFINITY,
            -3.0,
            1.0,
        ];
        let opts = newton_opts(50);
        for w in [1, 2, 3] {
            let got = h
                .equilibrium_many(u0.clone(), vec![], opts.clone(), 1, w)
                .unwrap();
            let done = assert_starts_match_single_calls(
                &h,
                &got,
                &u0,
                &[],
                &opts,
                false,
                1,
                &format!("W = {w}"),
            );
            assert_eq!(done, 2, "W = {w}");
            for (j, s) in got.iter().enumerate().take(4).skip(1) {
                match s {
                    Start::Failed(r) => assert!(r.is_nan(), "W = {w}, start {j}: {r}"),
                    Start::Done(..) => panic!("W = {w}: start {j} converged"),
                }
            }
        }

        // Newton does not converge in 2 steps from x = 100: the residual is its last one,
        // and with no step at all it is infinity.
        let far = vec![100.0, 1.0, 2.0, 0.3];
        for w in [1, 2] {
            let got = h
                .equilibrium_many(far.clone(), vec![], newton_opts(2), 1, w)
                .unwrap();
            let Start::Failed(r) = got[0] else {
                panic!("W = {w}: the far start converged")
            };
            let single = h
                .equilibrium(
                    EpSetup::new(0, vec![], far[..2].to_vec()),
                    newton_opts(2),
                    1,
                )
                .err()
                .expect("the single far call converged");
            assert!(r.is_finite() && r > 1.0, "W = {w}: {r}");
            assert!(single.to_string().contains("Newton"), "{single}");
            let got = h
                .equilibrium_many(far.clone(), vec![], newton_opts(0), 1, w)
                .unwrap();
            for s in &got {
                match s {
                    Start::Failed(r) => assert_eq!(*r, f64::INFINITY, "W = {w}"),
                    Start::Done(..) => panic!("W = {w}: converged with no Newton step"),
                }
            }
        }
    }

    #[test]
    fn the_error_of_the_call_comes_from_the_lowest_start() {
        // SPEC 8ai-2, test 3.
        let h = handle_of(BISTABLE);
        h.compile(Build::Jit).unwrap();
        let u0 = box_starts(4);
        let cases = [
            (None, vec![7, 3], "debug start error at 3"),
            (Some(5), vec![9, 2], "debug start error at 2"),
            (Some(4), vec![11], "debug start panic at 4"),
            (Some(1), vec![], "debug start panic at 1"),
        ];
        for (panic_at, error_at, want) in cases {
            *h.start_hook.lock().unwrap() = StartHook {
                panic_at,
                error_at: error_at.clone(),
            };
            for w in [1, 2, 3] {
                let e = match h.equilibrium_many(u0.clone(), vec![], newton_opts(50), 1, w) {
                    Ok(_) => panic!("W = {w}: no error for {want}"),
                    Err(e) => e,
                };
                assert!(e.to_string().contains(want), "W = {w}: {e}, want {want}");
            }
        }
        // The handle still works after a panic in a start.
        *h.start_hook.lock().unwrap() = StartHook::default();
        for w in [1, 3] {
            assert!(
                h.equilibrium_many(u0.clone(), vec![], newton_opts(50), 1, w)
                    .is_ok()
            );
        }
    }

    #[test]
    fn the_worker_count_follows_the_spec() {
        // W = min(T, ceil(k / M)), at least 1 (SPEC 8ai-2).
        let m = MIN_STARTS_PER_WORKER;
        assert_eq!(equilibrium_workers(0, 16), 1);
        assert_eq!(equilibrium_workers(1, 16), 1);
        assert_eq!(equilibrium_workers(m, 16), 1);
        assert_eq!(equilibrium_workers(m + 1, 16), 2);
        assert_eq!(equilibrium_workers(1000 * m, 16), 16);
        assert_eq!(equilibrium_workers(1000 * m, 0), 1);
    }

    /// Timing for SPEC 8ai-1b: K = 40401 single calls against one many-column call.
    /// Run with `cargo test --release -p pc-ffi many_call_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn many_call_timing() {
        use std::time::Instant;
        let h = handle_of(VDP);
        h.compile(Build::Jit).unwrap();
        let k = 40401;
        let u = grid(k);
        for _ in 0..2 {
            let t0 = Instant::now();
            for c in u.as_chunks::<2>().0 {
                std::hint::black_box(h.jacobian(c.to_vec(), vec![1.5], 1).unwrap());
            }
            let single = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            std::hint::black_box(h.jacobian_many(u.clone(), vec![1.5], 1).unwrap());
            let many = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            for c in u.as_chunks::<2>().0 {
                std::hint::black_box(h.rhs(c.to_vec(), vec![1.5], 0.0).unwrap());
            }
            let rsingle = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            std::hint::black_box(h.rhs_many(u.clone(), vec![1.5], 0.0).unwrap());
            let rmany = t0.elapsed().as_secs_f64();
            let us = 1e6 / k as f64;
            println!(
                "jacobian single {:.3} us/pt, many {:.3} us/pt; rhs single {:.3} us/pt, many {:.3} us/pt",
                single * us,
                many * us,
                rsingle * us,
                rmany * us
            );
        }
    }
}
