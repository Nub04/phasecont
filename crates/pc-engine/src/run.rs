//! Continuations on a worker thread, with poll and cancel.
//!
//! [`Run::start`] spawns a thread that builds the model, follows the branch and, when
//! asked, switches at a BP and follows the second branch. The caller polls the shared
//! state without blocking, reads points as they arrive and can cancel. This is the shape
//! the C ABI wraps. The body runs under `catch_unwind`, and a panic ends in
//! [`RunState::Panicked`].
//!
//! Points reach the shared state as they are accepted. When a branch finishes, the
//! `Branch` that `continue_ep` returns (refined specials, stop reason) replaces the
//! streamed one, so specials are visible only after that. While a branch is running its
//! stored `stop` is a placeholder and is not meaningful.

use crate::branch::{Branch, Options, Point, SpecialPoint, StopReason};
use crate::ep::{EpSetup, continue_ep};
use crate::error::EngineError;
use crate::model::{Evaluator, build};
use crate::panic::{install_hook, mark_worker, payload_message, take_last_panic};
use crate::switch::{special_index, switch_bp};
use pc_ode::{Model, ModelInfo};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// What to run. The worker lowers and compiles the model on its own thread.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub model: Model,
    pub info: ModelInfo,
    pub evaluator: Evaluator,
    pub setup: EpSetup,
    pub opts: Options,
    /// Name of a special point of the first branch, for example `"BP1"`. When set, the
    /// run switches there and follows the second branch.
    pub switch_at: Option<String>,
    /// Test hook: wraps the evaluator in one that panics in `eval`.
    #[cfg(test)]
    pub(crate) panic_in_eval: bool,
}

impl RunSpec {
    /// A spec without a switch.
    pub fn new(
        model: Model,
        info: ModelInfo,
        evaluator: Evaluator,
        setup: EpSetup,
        opts: Options,
    ) -> RunSpec {
        RunSpec {
            model,
            info,
            evaluator,
            setup,
            opts,
            switch_at: None,
            #[cfg(test)]
            panic_in_eval: false,
        }
    }
}

/// Where a run stands.
#[derive(Clone, Debug, PartialEq)]
pub enum RunState {
    Running,
    Done,
    Failed(EngineError),
    Cancelled,
    Panicked { message: String, location: String },
}

/// A snapshot of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Status {
    pub state: RunState,
    /// Points so far, per branch.
    pub points: Vec<usize>,
    /// Special points so far, per branch.
    pub specials: Vec<usize>,
}

struct Inner {
    branches: Vec<Branch>,
    state: RunState,
}

struct Shared {
    inner: Mutex<Inner>,
    cancel: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A poisoned lock only means a panic elsewhere; the data is still consistent
        // because every update is a single push or assignment.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A running or finished continuation.
pub struct Run {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

fn placeholder() -> Branch {
    Branch {
        points: Vec::new(),
        specials: Vec::new(),
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    }
}

#[cfg(test)]
struct PanicEval(Box<dyn crate::model::ModelEval>);

#[cfg(test)]
impl crate::model::ModelEval for PanicEval {
    fn n(&self) -> usize {
        self.0.n()
    }
    fn n_p(&self) -> usize {
        self.0.n_p()
    }
    fn eval(
        &mut self,
        _kind: &pc_ir::EntryKind,
        _x: &[f64],
        _p: &[f64],
        _v: &[f64],
        _out: &mut [f64],
    ) -> Result<(), EngineError> {
        panic!("{}", String::from("eval blew up on purpose"));
    }
}

/// Follows one branch and stores the result as branch `idx`. Returns the branch.
fn follow(
    shared: &Shared,
    model: &mut dyn crate::model::ModelEval,
    setup: &EpSetup,
    opts: &Options,
    idx: usize,
) -> Result<Branch, EngineError> {
    shared.lock().branches.push(placeholder());
    let b = continue_ep(model, setup, opts, &shared.cancel, &mut |p: &Point| {
        shared.lock().branches[idx].points.push(p.clone());
    });
    if let Ok(branch) = &b {
        shared.lock().branches[idx] = branch.clone();
    }
    b
}

fn body(shared: &Shared, spec: RunSpec) -> RunState {
    let model = match build(&spec.model, &spec.info, spec.evaluator) {
        Ok(m) => m,
        Err(e) => return finish(shared, Err(e)),
    };
    #[cfg(test)]
    let model: Box<dyn crate::model::ModelEval> = if spec.panic_in_eval {
        Box::new(PanicEval(model))
    } else {
        model
    };
    let mut model = model;
    let first = match follow(shared, model.as_mut(), &spec.setup, &spec.opts, 0) {
        Ok(b) => b,
        Err(e) => return finish(shared, Err(e)),
    };
    let Some(name) = &spec.switch_at else {
        return finish(shared, Ok(&first));
    };
    if shared.cancel.load(Ordering::Relaxed) || first.stop == StopReason::Cancelled {
        return finish(shared, Ok(&first));
    }
    let Some(k) = special_index(&first, name) else {
        return RunState::Failed(EngineError::BadProblem(format!(
            "no special point {name} to switch at"
        )));
    };
    let next = match switch_bp(model.as_mut(), &spec.setup, &first, k, &spec.opts) {
        Ok(s) => s,
        Err(e) => return RunState::Failed(e),
    };
    match follow(shared, model.as_mut(), &next, &spec.opts, 1) {
        Ok(b) => finish(shared, Ok(&b)),
        Err(e) => finish(shared, Err(e)),
    }
}

/// Final state from the result of the last branch (R15c).
fn finish(shared: &Shared, r: Result<&Branch, EngineError>) -> RunState {
    let cancelled = shared.cancel.load(Ordering::Relaxed);
    match r {
        Ok(b) if cancelled || b.stop == StopReason::Cancelled => RunState::Cancelled,
        Ok(_) => RunState::Done,
        Err(EngineError::Cancelled) => RunState::Cancelled,
        Err(e) => RunState::Failed(e),
    }
}

fn worker(shared: Arc<Shared>, spec: RunSpec) {
    mark_worker();
    let state = match catch_unwind(AssertUnwindSafe(|| body(&shared, spec))) {
        Ok(s) => s,
        Err(payload) => {
            let (message, location) = take_last_panic()
                .unwrap_or_else(|| (payload_message(payload.as_ref()), "unknown".to_string()));
            RunState::Panicked { message, location }
        }
    };
    shared.lock().state = state;
}

impl Run {
    /// Spawns the worker thread (named "pc-engine-run", 8 MiB stack) and returns at once.
    /// If the thread cannot be spawned the run is `Failed` from the start.
    pub fn start(spec: RunSpec) -> Run {
        install_hook();
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                branches: Vec::new(),
                state: RunState::Running,
            }),
            cancel: AtomicBool::new(false),
        });
        let s2 = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("pc-engine-run".into())
            .stack_size(8 << 20)
            .spawn(move || worker(s2, spec));
        let handle = match spawned {
            Ok(h) => Some(h),
            Err(e) => {
                shared.lock().state =
                    RunState::Failed(EngineError::BadProblem(format!("cannot spawn worker: {e}")));
                None
            }
        };
        Run { shared, handle }
    }

    /// A snapshot of the run. Does not block on the worker.
    pub fn poll(&self) -> Status {
        let g = self.shared.lock();
        Status {
            state: g.state.clone(),
            points: g.branches.iter().map(|b| b.points.len()).collect(),
            specials: g.branches.iter().map(|b| b.specials.len()).collect(),
        }
    }

    /// Up to `count` points of `branch` from index `start`; empty when out of range.
    pub fn points(&self, branch: usize, start: usize, count: usize) -> Vec<Point> {
        let g = self.shared.lock();
        match g.branches.get(branch) {
            Some(b) => b.points.iter().skip(start).take(count).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// The special points of `branch`; empty until the branch has finished.
    pub fn specials(&self, branch: usize) -> Vec<SpecialPoint> {
        let g = self.shared.lock();
        g.branches
            .get(branch)
            .map(|b| b.specials.clone())
            .unwrap_or_default()
    }

    /// Asks the worker to stop. Returns at once.
    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    /// Waits for the worker and returns the final status.
    pub fn join(mut self) -> Status {
        self.wait();
        self.poll()
    }

    fn wait(&mut self) {
        if let Some(h) = self.handle.take() {
            // The body catches panics itself; an error here would mean the thread
            // died outside it, and the state would still say Running.
            if h.join().is_err() {
                let mut g = self.shared.lock();
                if g.state == RunState::Running {
                    g.state = RunState::Panicked {
                        message: "worker thread died".into(),
                        location: "unknown".into(),
                    };
                }
            }
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.cancel();
        self.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ode::{FsLoader, Severity, SourceMap, load};
    use std::path::Path;

    fn lorenz_spec() -> RunSpec {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/lorenz.ode");
        let mut map = SourceMap::new();
        let mut loader = FsLoader::new();
        let top = loader.load_top(&mut map, &path).unwrap();
        let loaded = load(top, &mut map, &mut loader);
        assert!(
            loaded
                .diagnostics
                .iter()
                .all(|d| d.severity != Severity::Error)
        );
        let setup = EpSetup::new(1, vec![10.0, 0.5, 8.0 / 3.0], vec![0.0, 0.0, 0.0]);
        let opts = Options {
            h0: 0.01,
            hmax: 0.01,
            max_steps: 1000,
            bounds: (f64::NEG_INFINITY, 2.0),
            ..Options::default()
        };
        RunSpec::new(loaded.expanded, loaded.info, Evaluator::Jit, setup, opts)
    }

    #[test]
    fn a_panic_in_eval_becomes_panicked_and_later_runs_work() {
        let mut spec = lorenz_spec();
        spec.panic_in_eval = true;
        let st = Run::start(spec).join();
        match st.state {
            RunState::Panicked { message, location } => {
                assert_eq!(message, "eval blew up on purpose");
                assert!(location.contains("src/run.rs:"), "{location}");
                let mut parts = location.rsplitn(3, ':');
                assert!(parts.next().unwrap().parse::<u32>().is_ok());
                assert!(parts.next().unwrap().parse::<u32>().is_ok());
            }
            other => panic!("expected Panicked, got {other:?}"),
        }
        // The same binary keeps running, and a second run succeeds.
        let st = Run::start(lorenz_spec()).join();
        assert_eq!(st.state, RunState::Done);
        assert!(st.points[0] > 10);
    }

    #[test]
    fn str_and_string_payloads_give_their_text() {
        let a = catch_unwind(|| panic!("static text")).unwrap_err();
        assert_eq!(payload_message(a.as_ref()), "static text");
        let b = catch_unwind(|| panic!("{}", String::from("owned text"))).unwrap_err();
        assert_eq!(payload_message(b.as_ref()), "owned text");
    }
}
