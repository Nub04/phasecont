//! Continuations on a worker thread, with poll and cancel.
//!
//! [`Run::start`] spawns a thread that builds the model, follows the branch and, when
//! asked, switches at a BP and follows the second branch. The caller polls the shared
//! state without blocking, reads points as they arrive and can cancel. This is the shape
//! the C ABI wraps. The body runs under `catch_unwind`, and a panic ends in
//! [`RunState::Panicked`].
//!
//! Points reach the shared state as they are accepted. When a branch finishes, the
//! `Branch` that the continuation returns (refined specials, stop reason) replaces the
//! streamed one, so specials are visible only after that. While a branch is running its
//! stored `stop` is a placeholder and is not meaningful.

use crate::branch::{Branch, Extras, Options, Point, SpecialPoint, StopReason};
use crate::curve2::CurveSetup;
use crate::cycle::{
    CycleCurveStart, CycleStart, HomStart, continue_cycle, continue_hom_curve, continue_lpc_curve,
    continue_ns_curve, continue_pd_curve,
};
use crate::dde::{
    DdeBranch, DdeHopfBranch, HopfExtras, HopfSetup, RootOptions,
    continue_equilibrium as dde_continue_equilibrium, continue_hopf_with as dde_continue_hopf_with,
};
use crate::ep::{EpSetup, continue_ep};
use crate::error::EngineError;
use crate::hopfcurve::continue_hopf_curve;
use crate::ivp::{IvpOptions, IvpSetup, integrate};
use crate::lpcurve::continue_lp_curve;
use crate::model::{Evaluator, build, build_allowing_t, build_dde};
use crate::panic::{clear_last_panic, install_hook, mark_worker, payload_message, take_last_panic};
use crate::switch::{special_index, switch_bp};
use pc_ode::{Model, ModelInfo};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// What a run follows.
#[derive(Clone, Debug)]
pub enum RunKind {
    /// An equilibrium branch from `RunSpec::setup`.
    Ep,
    /// The LP curve through an LP point, in two parameters.
    LpCurve(CurveSetup),
    /// The Hopf curve through a Hopf point, in two parameters.
    HopfCurve(CurveSetup),
    /// A branch of periodic orbits in one parameter.
    Cycle(CycleStart),
    /// The LPC curve through an LPC point of a cycle branch, in two parameters.
    LpcCurve(CycleCurveStart),
    /// The PD curve through a PD point of a cycle branch, in two parameters.
    PdCurve(CycleCurveStart),
    /// The NS curve through an NS point of a cycle branch, in two parameters.
    NsCurve(CycleCurveStart),
    /// The HOM curve of orbits homoclinic to a saddle, in two parameters.
    HomCurve(HomStart),
    /// The equilibrium branch of a delay equation from `RunSpec::setup`, with the characteristic
    /// roots counted by `RootOptions` (SPEC 8s). The continuation runs once and the branch
    /// appears when it ends, so a poll sees no points before then.
    DdeEq(RootOptions),
    /// The Hopf curve of a delay equilibrium in the two parameters `free`, from the start
    /// `setup.u0` and `setup.params` with the frequency `omega`, and the additions of `extras`
    /// (SPEC 8s, slice 2). Like `DdeEq` it runs once and the branch appears when it ends.
    DdeHopf {
        free: [usize; 2],
        omega: f64,
        extras: HopfExtras,
    },
    /// An initial value run over `tspan` from the state `setup.u0` with the parameters
    /// `setup.params`. The model may depend on t.
    Ivp { tspan: [f64; 2], opts: IvpOptions },
}

/// What to run. The worker lowers and compiles the model on its own thread.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub model: Model,
    pub info: ModelInfo,
    pub evaluator: Evaluator,
    pub kind: RunKind,
    /// Start of an `Ep` run. A curve run ignores it.
    pub setup: EpSetup,
    pub opts: Options,
    /// Name of a special point of the first branch, for example `"BP1"`. When set, the
    /// run switches there and follows the second branch. Only an `Ep` run can switch.
    pub switch_at: Option<String>,
    /// Test hook: wraps the evaluator in one that panics in `eval` after this many
    /// successful calls.
    #[cfg(test)]
    pub(crate) panic_after: Option<usize>,
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
            kind: RunKind::Ep,
            setup,
            opts,
            switch_at: None,
            #[cfg(test)]
            panic_after: None,
        }
    }

    /// A spec for a run that starts from `kind` alone, which is any kind but `Ep`.
    /// With `Ep` the run fails with `BadProblem` because there is no equilibrium start.
    pub fn curve(
        model: Model,
        info: ModelInfo,
        evaluator: Evaluator,
        kind: RunKind,
        opts: Options,
    ) -> RunSpec {
        let mut spec = RunSpec::new(
            model,
            info,
            evaluator,
            EpSetup::new(0, Vec::new(), Vec::new()),
            opts,
        );
        spec.kind = kind;
        spec
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
struct PanicEval {
    inner: Box<dyn crate::model::ModelEval>,
    left: usize,
}

#[cfg(test)]
impl crate::model::ModelEval for PanicEval {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn n_p(&self) -> usize {
        self.inner.n_p()
    }
    fn eval(
        &mut self,
        kind: &pc_ir::EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        if self.left == 0 {
            panic!("{}", String::from("eval blew up on purpose"));
        }
        self.left -= 1;
        self.inner.eval(kind, x, p, v, out)
    }
}

/// Follows one branch and stores the result as branch `idx`. Returns the branch.
fn follow(
    shared: &Shared,
    model: &mut dyn crate::model::ModelEval,
    kind: &RunKind,
    setup: &EpSetup,
    opts: &Options,
    idx: usize,
) -> Result<Branch, EngineError> {
    shared.lock().branches.push(placeholder());
    let on_point = &mut |p: &Point| {
        shared.lock().branches[idx].points.push(p.clone());
    };
    let b = match kind {
        RunKind::Ep => continue_ep(model, setup, opts, &shared.cancel, on_point),
        RunKind::LpCurve(cs) => continue_lp_curve(model, cs, opts, &shared.cancel, on_point),
        RunKind::HopfCurve(cs) => continue_hopf_curve(model, cs, opts, &shared.cancel, on_point),
        RunKind::Cycle(start) => continue_cycle(model, start, opts, &shared.cancel, on_point),
        RunKind::LpcCurve(cs) => continue_lpc_curve(model, cs, opts, &shared.cancel, on_point),
        RunKind::PdCurve(cs) => continue_pd_curve(model, cs, opts, &shared.cancel, on_point),
        RunKind::NsCurve(cs) => continue_ns_curve(model, cs, opts, &shared.cancel, on_point),
        RunKind::HomCurve(hs) => continue_hom_curve(model, hs, opts, &shared.cancel, on_point),
        // `body` sends a delay run to `body_dde` before it builds an ODE model.
        RunKind::DdeEq(_) | RunKind::DdeHopf { .. } => Err(EngineError::BadProblem(
            "a delay run has no ODE model to follow".into(),
        )),
        RunKind::Ivp { tspan, opts } => {
            let ivp = IvpSetup {
                x0: setup.u0.clone(),
                params: setup.params.clone(),
                t0: tspan[0],
                t1: tspan[1],
            };
            integrate(model, &ivp, opts, &shared.cancel, on_point)
        }
    };
    match &b {
        Ok(branch) => shared.lock().branches[idx] = branch.clone(),
        // Keep the streamed points and say that the branch did not finish.
        Err(e) => shared.lock().branches[idx].stop = StopReason::Error(e.clone()),
    }
    b
}

fn body(shared: &Shared, spec: RunSpec) -> RunState {
    if spec.switch_at.is_some() && !matches!(spec.kind, RunKind::Ep) {
        return finish(
            shared,
            Err(EngineError::BadProblem(
                "only an equilibrium run can switch branches".into(),
            )),
        );
    }
    match spec.kind {
        RunKind::DdeEq(roots) => return body_dde(shared, spec, roots),
        RunKind::DdeHopf {
            free,
            omega,
            extras,
        } => return body_dde_hopf(shared, spec, free, omega, extras),
        _ => {}
    }
    // Only an initial value run may depend on t.
    let built = if matches!(spec.kind, RunKind::Ivp { .. }) {
        build_allowing_t(&spec.model, &spec.info, spec.evaluator)
    } else {
        build(&spec.model, &spec.info, spec.evaluator)
    };
    let model = match built {
        Ok(m) => m,
        Err(e) => return finish(shared, Err(e)),
    };
    #[cfg(test)]
    let model: Box<dyn crate::model::ModelEval> = match spec.panic_after {
        Some(left) => Box::new(PanicEval { inner: model, left }),
        None => model,
    };
    let mut model = model;
    let first = match follow(
        shared,
        model.as_mut(),
        &spec.kind,
        &spec.setup,
        &spec.opts,
        0,
    ) {
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
        return finish(
            shared,
            Err(EngineError::BadProblem(format!(
                "no special point {name} to switch at"
            ))),
        );
    };
    let next = match switch_bp(model.as_mut(), &spec.setup, &first, k, &spec.opts) {
        Ok(s) => s,
        Err(e) => return finish(shared, Err(e)),
    };
    match follow(shared, model.as_mut(), &RunKind::Ep, &next, &spec.opts, 1) {
        Ok(b) => finish(shared, Ok(&b)),
        Err(e) => finish(shared, Err(e)),
    }
}

/// Follows the equilibrium branch of a delay equation (SPEC 8s). The engine returns the branch in
/// one piece, so branch 0 stays empty until the end and a cancelled run keeps no points.
fn body_dde(shared: &Shared, spec: RunSpec, roots: RootOptions) -> RunState {
    let mut model = match build_dde(&spec.model, &spec.info, spec.evaluator) {
        Ok(m) => m,
        Err(e) => return finish(shared, Err(e)),
    };
    shared.lock().branches.push(placeholder());
    match dde_continue_equilibrium(&mut model, &spec.setup, &roots, &spec.opts, &shared.cancel) {
        Ok(b) => {
            let branch = dde_branch(b);
            shared.lock().branches[0] = branch.clone();
            finish(shared, Ok(&branch))
        }
        Err(e) => {
            shared.lock().branches[0].stop = match &e {
                EngineError::Cancelled => StopReason::Cancelled,
                other => StopReason::Error(other.clone()),
            };
            finish(shared, Err(e))
        }
    }
}

/// Follows the Hopf curve of a delay equilibrium (SPEC 8s, slice 2). Like `body_dde` it stores
/// the branch when the engine returns it. A flag set at the call keeps no points. A flag set
/// during the run keeps the points the engine found, with stop `Cancelled`.
fn body_dde_hopf(
    shared: &Shared,
    spec: RunSpec,
    free: [usize; 2],
    omega: f64,
    extras: HopfExtras,
) -> RunState {
    let mut model = match build_dde(&spec.model, &spec.info, spec.evaluator) {
        Ok(m) => m,
        Err(e) => return finish(shared, Err(e)),
    };
    shared.lock().branches.push(placeholder());
    let setup = HopfSetup {
        free,
        params: spec.setup.params.clone(),
        x0: spec.setup.u0.clone(),
        omega,
    };
    match dde_continue_hopf_with(&mut model, &setup, &spec.opts, &shared.cancel, &extras) {
        Ok(b) => {
            let branch = dde_hopf_branch(b);
            shared.lock().branches[0] = branch.clone();
            finish(shared, Ok(&branch))
        }
        Err(e) => {
            shared.lock().branches[0].stop = match &e {
                EngineError::Cancelled => StopReason::Cancelled,
                other => StopReason::Error(other.clone()),
            };
            finish(shared, Err(e))
        }
    }
}

/// A delay Hopf curve as a [`Branch`]. `u` is `[x, alpha1, kappa]`, `alpha` is the second free
/// parameter and `codim2` is `[l1, hh]`, NaN where the engine has `None`. A special has the row
/// of its point with both NaN. There is no tangent, eigenvalue, mesh, step length or iteration
/// count. `omega2` of an `HH` special goes in `extras.values`, followed by its coefficients
/// (SPEC 8s, slice 3) for a `ZH` or `HH` special of a run with that option.
fn dde_hopf_branch(b: DdeHopfBranch) -> Branch {
    let point = |alpha: [f64; 2], x: &[f64], kappa: f64, l1: f64, hh: f64| {
        let mut u = x.to_vec();
        u.push(alpha[0]);
        u.push(kappa);
        Point {
            u,
            alpha: alpha[1],
            tangent: Vec::new(),
            eigenvalues: Vec::new(),
            tests: [f64::NAN; 3],
            codim2: vec![l1, hh],
            h: 0.0,
            newton_iters: 0,
            mesh: Vec::new(),
        }
    };
    Branch {
        points: b
            .points
            .iter()
            .map(|p| {
                point(
                    p.alpha,
                    &p.x,
                    p.kappa,
                    p.l1.unwrap_or(f64::NAN),
                    p.hh.unwrap_or(f64::NAN),
                )
            })
            .collect(),
        specials: b
            .specials
            .iter()
            .map(|s| {
                let mut extras = Extras {
                    omega: (s.kappa > 0.0).then(|| s.kappa.sqrt()),
                    refined: false,
                    ..Extras::default()
                };
                if s.label == "GH" {
                    extras.coefficient = s.l2;
                    extras.normalization = Some(DDE_GH_NORMALIZATION.to_string());
                }
                if let Some(w2) = s.omega2 {
                    extras.values.push(("omega2".to_string(), w2));
                }
                extras
                    .values
                    .extend(s.coefficients.iter().map(|(n, v)| (n.to_string(), *v)));
                SpecialPoint {
                    label: s.label.to_string(),
                    index: s.index,
                    point: point(s.alpha, &s.x, s.kappa, f64::NAN, f64::NAN),
                    extras,
                }
            })
            .collect(),
        stop: b.stop,
        warnings: b.warnings,
    }
}

/// The normalization of the second Lyapunov coefficient of a delay Hopf curve (SPEC 8g-2).
pub const DDE_GH_NORMALIZATION: &str = "|q|=1, p Delta'(i omega) q=1";

/// A delay equilibrium branch as a [`Branch`]. The root count of a point goes in `tests[0]` as
/// a float, with the other two tests NaN, and there is no tangent, eigenvalue, step length or
/// iteration count. A special point carries both counts in `extras.values`.
fn dde_branch(b: DdeBranch) -> Branch {
    let point = |alpha: f64, x: Vec<f64>, count: usize| Point {
        u: x,
        alpha,
        tangent: Vec::new(),
        eigenvalues: Vec::new(),
        tests: [count as f64, f64::NAN, f64::NAN],
        codim2: Vec::new(),
        h: 0.0,
        newton_iters: 0,
        mesh: Vec::new(),
    };
    Branch {
        points: b
            .points
            .into_iter()
            .map(|p| point(p.alpha, p.x, p.unstable))
            .collect(),
        specials: b
            .specials
            .into_iter()
            .map(|s| SpecialPoint {
                label: s.label.to_string(),
                index: s.between[1],
                point: point(s.alpha, s.x, s.unstable_after),
                extras: Extras {
                    omega: s.omega,
                    values: vec![
                        ("unstable_before".to_string(), s.unstable_before as f64),
                        ("unstable_after".to_string(), s.unstable_after as f64),
                    ],
                    refined: false,
                    ..Extras::default()
                },
            })
            .collect(),
        stop: b.stop,
        warnings: b.warnings,
    }
}

/// Final state from the result of the last branch (R15c).
fn finish(shared: &Shared, r: Result<&Branch, EngineError>) -> RunState {
    let cancelled = shared.cancel.load(Ordering::Relaxed);
    match r {
        _ if cancelled => RunState::Cancelled,
        Ok(b) if b.stop == StopReason::Cancelled => RunState::Cancelled,
        Ok(_) => RunState::Done,
        Err(EngineError::Cancelled) => RunState::Cancelled,
        Err(e) => RunState::Failed(e),
    }
}

fn worker(shared: Arc<Shared>, run_body: impl FnOnce(&Shared) -> RunState) {
    mark_worker();
    clear_last_panic();
    let state = match catch_unwind(AssertUnwindSafe(|| run_body(&shared))) {
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
    /// Installs the panic hook on first use. That replaces the process-wide hook and chains
    /// to the previous one for every thread that is not an engine worker.
    ///
    /// Spawns the worker thread (named "pc-engine-run", 8 MiB stack) and returns at once.
    /// If the thread cannot be spawned the run is `Failed` from the start.
    pub fn start(spec: RunSpec) -> Run {
        Run::spawn(move |shared| body(shared, spec))
    }

    /// Starts a run whose body panics at once, through the same worker wrapper as
    /// `start`. For the FFI panic tests only.
    #[doc(hidden)]
    pub fn start_panicking() -> Run {
        Run::spawn(|_| panic!("start_panicking"))
    }

    fn spawn(run_body: impl FnOnce(&Shared) -> RunState + Send + 'static) -> Run {
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
            .spawn(move || worker(s2, run_body));
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
        // Read the finished flag first: the worker sets the state before it ends, so a
        // finished worker with a state of Running never wrote one.
        let ended = self.handle.as_ref().is_some_and(|h| h.is_finished());
        let g = self.shared.lock();
        let state = if ended && g.state == RunState::Running {
            RunState::Panicked {
                message: "worker ended without a state".into(),
                location: "unknown".into(),
            }
        } else {
            g.state.clone()
        };
        Status {
            state,
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

    /// Why `branch` stopped. After a failed branch this is `StopReason::Error`. While the
    /// branch is running it is a placeholder. `None` when the branch does not exist.
    pub fn stop(&self, branch: usize) -> Option<StopReason> {
        self.shared
            .lock()
            .branches
            .get(branch)
            .map(|b| b.stop.clone())
    }

    /// The warnings of `branch`; empty when the branch does not exist.
    pub fn warnings(&self, branch: usize) -> Vec<String> {
        self.shared
            .lock()
            .branches
            .get(branch)
            .map(|b| b.warnings.clone())
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
            let died = h.join().is_err();
            let mut g = self.shared.lock();
            if g.state == RunState::Running {
                g.state = RunState::Panicked {
                    message: if died {
                        "worker thread died".into()
                    } else {
                        "worker ended without a state".into()
                    },
                    location: "unknown".into(),
                };
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
    use crate::cycle::{CYCLE_LABELS, HopfStart};
    use pc_ode::{FsLoader, Severity, SourceMap, load};
    use std::path::Path;

    fn test_model(name: &str) -> (Model, ModelInfo) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/models")
            .join(name);
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
        (loaded.expanded, loaded.info)
    }

    fn lorenz_spec() -> RunSpec {
        let (model, info) = test_model("lorenz.ode");
        let setup = EpSetup::new(1, vec![10.0, 0.5, 8.0 / 3.0], vec![0.0, 0.0, 0.0]);
        let opts = Options {
            h0: 0.01,
            hmax: 0.01,
            max_steps: 1000,
            bounds: (f64::NEG_INFINITY, 2.0),
            ..Options::default()
        };
        RunSpec::new(model, info, Evaluator::Jit, setup, opts)
    }

    #[test]
    fn a_cycle_run_follows_the_orbits_from_a_hopf_point() {
        let (model, info) = test_model("hopf_c.ode");
        let (ntst, ncol) = (10, 3);
        let start = CycleStart::Hopf(HopfStart {
            free: 0,
            params: vec![0.0, -1.0],
            u0: vec![0.0, 0.0],
            ntst,
            ncol,
        });
        let opts = Options {
            hmax: 0.05,
            max_steps: 40,
            bounds: (f64::NEG_INFINITY, 0.2),
            ..Options::default()
        };
        let spec = RunSpec::curve(model, info, Evaluator::Jit, RunKind::Cycle(start), opts);
        let run = Run::start(spec);
        while run.poll().state == RunState::Running {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let st = run.poll();
        assert_eq!(st.state, RunState::Done);
        assert!(st.points[0] > 3, "{:?}", st.points);
        for p in run.points(0, 0, usize::MAX) {
            assert_eq!(p.u.len(), 2 * (ntst * ncol + 1) + 1);
            assert_eq!(p.mesh.len(), ntst + 1);
            assert_eq!(p.eigenvalues.len(), 2);
            assert_eq!(p.codim2.len(), CYCLE_LABELS.len());
        }
    }

    #[test]
    fn an_lpc_curve_run_follows_the_bautin_fold() {
        // On bautin_lpc.ode the LPC curve is b1 = -b2^2 / 4 with r^2 = b2 / 2, and the
        // period is 2 pi / (1 + g r^2).
        let (model, info) = test_model("bautin_lpc.ode");
        let (ntst, ncol) = (8, 3);
        let r = 0.5_f64.sqrt();
        let pts = ntst * ncol + 1;
        let orbit = (0..pts)
            .flat_map(|k| {
                let t = std::f64::consts::TAU * k as f64 / (pts - 1) as f64;
                [r * t.cos(), r * t.sin()]
            })
            .collect();
        let start = CycleCurveStart {
            free: [0, 1],
            params: vec![-0.25, 1.0, 0.5],
            orbit,
            period: std::f64::consts::TAU / (1.0 + 0.5 * r * r),
            mesh: (0..=ntst).map(|j| j as f64 / ntst as f64).collect(),
            ntst,
            ncol,
        };
        let opts = Options {
            max_newton: 15,
            hmax: 0.05,
            max_steps: 30,
            bounds: (0.8, 1.2),
            ..Options::default()
        };
        let spec = RunSpec::curve(model, info, Evaluator::Jit, RunKind::LpcCurve(start), opts);
        let run = Run::start(spec);
        while run.poll().state == RunState::Running {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let st = run.poll();
        assert_eq!(st.state, RunState::Done);
        assert!(st.points[0] > 3, "{:?}", st.points);
        for p in run.points(0, 0, usize::MAX) {
            // u = (profile, T, a1) and alpha = a2.
            assert_eq!(p.u.len(), 2 * (ntst * ncol + 1) + 2);
            assert_eq!(p.mesh.len(), ntst + 1);
            assert_eq!(p.eigenvalues.len(), 2);
            assert_eq!(p.codim2.len(), crate::cycle::LPC_CURVE_LABELS.len());
            let b1 = p.u[p.u.len() - 1];
            assert!((b1 + p.alpha * p.alpha / 4.0).abs() < 1e-3, "b1 {b1}");
        }
    }

    #[test]
    fn a_panic_in_eval_becomes_panicked_and_later_runs_work() {
        let mut spec = lorenz_spec();
        spec.panic_after = Some(0);
        let st = Run::start(spec).join();
        match st.state {
            RunState::Panicked { message, location } => {
                assert_eq!(message, "eval blew up on purpose");
                assert!(
                    location.replace('\\', "/").contains("src/run.rs:"),
                    "{location}"
                );
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
    fn start_panicking_ends_panicked_with_a_location() {
        let st = Run::start_panicking().join();
        match st.state {
            RunState::Panicked { message, location } => {
                assert!(!message.is_empty());
                assert_ne!(location, "unknown");
            }
            other => panic!("expected Panicked, got {other:?}"),
        }
    }

    #[test]
    fn finish_maps_cancel_per_r15c() {
        let sh = Shared {
            inner: Mutex::new(Inner {
                branches: Vec::new(),
                state: RunState::Running,
            }),
            cancel: AtomicBool::new(false),
        };
        assert_eq!(
            finish(&sh, Err(EngineError::Cancelled)),
            RunState::Cancelled
        );
        let other = EngineError::BadProblem("x".into());
        assert_eq!(
            finish(&sh, Err(other.clone())),
            RunState::Failed(other.clone())
        );
        sh.cancel.store(true, Ordering::Relaxed);
        assert_eq!(finish(&sh, Err(other)), RunState::Cancelled);
    }

    #[test]
    fn poll_after_panicked_keeps_the_streamed_points() {
        let mut spec = lorenz_spec();
        // Enough successful evaluations for a few points to stream before the panic.
        spec.panic_after = Some(40);
        let run = Run::start(spec);
        let t0 = std::time::Instant::now();
        while run.poll().state == RunState::Running {
            assert!(t0.elapsed().as_secs() < 30);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let a = run.poll();
        let b = run.poll();
        assert!(matches!(a.state, RunState::Panicked { .. }));
        assert_eq!(a, b);
        let streamed = a.points.first().copied().unwrap_or(0);
        assert!(streamed > 0, "no point streamed before the panic");
        assert_eq!(run.points(0, 0, 10).len(), streamed.min(10));
        assert!(run.specials(0).is_empty());
    }

    #[test]
    fn a_worker_that_ends_without_a_state_is_reported() {
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                branches: Vec::new(),
                state: RunState::Running,
            }),
            cancel: AtomicBool::new(false),
        });
        let handle = std::thread::spawn(|| {});
        let run = Run {
            shared,
            handle: Some(handle),
        };
        let t0 = std::time::Instant::now();
        loop {
            let st = run.poll();
            if let RunState::Panicked { message, .. } = &st.state {
                assert_eq!(message, "worker ended without a state");
                break;
            }
            assert!(t0.elapsed().as_secs() < 30, "{st:?}");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let st = run.join();
        assert!(matches!(st.state, RunState::Panicked { .. }));
    }

    #[test]
    fn warnings_come_from_the_branch_and_are_empty_for_a_missing_one() {
        let mut b = placeholder();
        b.warnings.push("first".into());
        let shared = Arc::new(Shared {
            inner: Mutex::new(Inner {
                branches: vec![b, placeholder()],
                state: RunState::Done,
            }),
            cancel: AtomicBool::new(false),
        });
        let run = Run {
            shared,
            handle: None,
        };
        assert_eq!(run.warnings(0), ["first"]);
        assert!(run.warnings(1).is_empty());
        assert!(run.warnings(2).is_empty());
    }

    #[test]
    fn str_and_string_payloads_give_their_text() {
        let a = catch_unwind(|| panic!("static text")).unwrap_err();
        assert_eq!(payload_message(a.as_ref()), "static text");
        let b = catch_unwind(|| panic!("{}", String::from("owned text"))).unwrap_err();
        assert_eq!(payload_message(b.as_ref()), "owned text");
    }

    #[test]
    fn a_delay_branch_keeps_the_root_count_in_the_first_test() {
        let b = dde_branch(DdeBranch {
            points: vec![
                crate::dde::DdePoint {
                    alpha: 0.5,
                    x: vec![1.0],
                    unstable: 0,
                },
                crate::dde::DdePoint {
                    alpha: 0.7,
                    x: vec![1.5],
                    unstable: 2,
                },
            ],
            specials: vec![crate::dde::DdeSpecial {
                label: "H",
                alpha: 0.6,
                x: vec![1.25],
                omega: Some(1.5),
                between: [0, 1],
                unstable_before: 0,
                unstable_after: 2,
            }],
            stop: StopReason::Bounds,
            warnings: vec!["w".to_string()],
        });
        assert_eq!(b.points.len(), 2);
        assert_eq!(b.points[1].tests[0], 2.0);
        assert!(b.points[1].tests[1].is_nan() && b.points[1].tests[2].is_nan());
        assert!(b.points[0].tangent.is_empty() && b.points[0].eigenvalues.is_empty());
        let s = &b.specials[0];
        assert_eq!(
            (s.label.as_str(), s.index, s.extras.omega),
            ("H", 1, Some(1.5))
        );
        assert_eq!(s.point.tests[0], 2.0);
        assert_eq!(
            s.extras.values,
            [
                ("unstable_before".to_string(), 0.0),
                ("unstable_after".to_string(), 2.0)
            ]
        );
        assert_eq!(b.stop, StopReason::Bounds);
        assert_eq!(b.warnings, ["w"]);
    }

    #[test]
    fn a_delay_hopf_branch_has_two_codim2_values_and_no_ode_data() {
        use crate::dde::{DdeHopfPoint, DdeHopfSpecial};
        let p = |a1: f64, a2: f64, kappa: f64, l1: Option<f64>, hh: Option<f64>| DdeHopfPoint {
            alpha: [a1, a2],
            x: vec![0.5, -0.5],
            kappa,
            omega: (kappa > 0.0).then(|| kappa.sqrt()),
            l1,
            hh,
        };
        let s = |label: &'static str, kappa: f64, l2: Option<f64>, omega2: Option<f64>| {
            DdeHopfSpecial {
                label,
                alpha: [1.5, 0.25],
                x: vec![0.5, -0.5],
                kappa,
                index: 1,
                l2,
                omega2,
                coefficients: Vec::new(),
            }
        };
        let b = dde_hopf_branch(DdeHopfBranch {
            points: vec![
                p(1.0, 0.0, 2.25, Some(-0.5), None),
                p(1.2, 0.1, 2.25, None, Some(0.75)),
            ],
            specials: vec![
                s("GH", 2.25, Some(0.125), None),
                s("HH", 4.0, None, Some(3.0)),
                s("BT", -1.0, None, None),
            ],
            stop: StopReason::Bounds,
            warnings: vec!["w".to_string()],
        });
        // u = [x, alpha1, kappa]; alpha is the second parameter; codim2 is [l1, hh].
        assert_eq!(b.points[0].u, [0.5, -0.5, 1.0, 2.25]);
        assert_eq!(b.points[0].alpha, 0.0);
        assert_eq!(b.points[0].codim2[0], -0.5);
        assert!(b.points[0].codim2[1].is_nan());
        assert!(b.points[1].codim2[0].is_nan());
        assert_eq!(b.points[1].codim2[1], 0.75);
        assert!(b.points[0].tangent.is_empty() && b.points[0].eigenvalues.is_empty());
        assert!(b.points[0].mesh.is_empty());
        assert!(b.points[0].tests.iter().all(|t| t.is_nan()));
        // GH: l2 as the coefficient and the normalization text of SPEC 8g-2.
        let gh = &b.specials[0];
        assert_eq!((gh.label.as_str(), gh.index), ("GH", 1));
        assert_eq!(gh.extras.omega, Some(1.5));
        assert_eq!(gh.extras.coefficient, Some(0.125));
        assert_eq!(
            gh.extras.normalization.as_deref(),
            Some("|q|=1, p Delta'(i omega) q=1")
        );
        assert!(!gh.extras.refined);
        assert!(gh.point.codim2.iter().all(|c| c.is_nan()));
        assert_eq!(gh.point.u, [0.5, -0.5, 1.5, 2.25]);
        assert_eq!(gh.point.alpha, 0.25);
        // HH: omega2 in the values, no coefficient, no normalization.
        let hh = &b.specials[1];
        assert_eq!(hh.extras.omega, Some(2.0));
        assert_eq!(hh.extras.values, [("omega2".to_string(), 3.0)]);
        assert!(hh.extras.coefficient.is_none() && hh.extras.normalization.is_none());
        // BT: kappa < 0 has no omega.
        assert_eq!(b.specials[2].extras.omega, None);
        assert_eq!(b.stop, StopReason::Bounds);
        assert_eq!(b.warnings, ["w"]);
    }
}
