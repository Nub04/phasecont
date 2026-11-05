//! An ensemble run behind an opaque handle.
//!
//! `ens_start` checks the job on the calling thread, so a bad job is an error at once, and
//! then starts one worker thread that runs the whole ensemble. The worker writes its outcome
//! into a slot that `ens_poll` and `ens_copy` read. A panic on the worker is caught there and
//! stored as the outcome, so it reaches MATLAB through the next poll or copy (rule 7).
//! Freeing the handle sets the cancel flag and joins the worker.

use std::result::Result as StdResult;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use pc_ensemble::device::{find_device, list_devices};
use pc_ensemble::run::run_on;
use pc_ensemble::{
    EnsembleError, EnsembleResult, Job, Label, Params, Precision, Status, job::validate,
};
use pc_ir::IrModel;
use serde_json::{Value, json};

use crate::error::{Error, Result, bad_arg};
use crate::json::ensemble::{
    HEAD_KEYS, Head, devices_json, object, only_keys, parse_head, required, string,
};
use crate::model::{MODEL_STACK, ModelHandle};
use crate::registry::{self, Handle};

/// `ens_copy` selector: the states, `members * samples * n` values, member-major, then
/// sample, then state.
pub const PC_ENS_STATES: u32 = 0;
/// `ens_copy` selector: the output times, `samples` values.
pub const PC_ENS_TIMES: u32 = 1;
/// `ens_copy` selector: the status code of each member (see [`Status`]), `members` values.
pub const PC_ENS_STATUS: u32 = 2;
/// `ens_copy` selector: the time each member reached, `members` values.
pub const PC_ENS_T_REACHED: u32 = 3;
/// `ens_copy` selector: the accepted steps of each member, `members` values.
pub const PC_ENS_ACCEPTED: u32 = 4;
/// `ens_copy` selector: the rejected steps of each member, `members` values.
pub const PC_ENS_REJECTED: u32 = 5;

/// How far a worker got.
enum Slot {
    Running,
    Done(Arc<EnsembleResult>),
    Cancelled,
    Failed(Error),
}

pub(crate) struct EnsHandle {
    cancel: Arc<AtomicBool>,
    slot: Arc<Mutex<Slot>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Handle for EnsHandle {
    const KIND: &'static str = "ensemble";
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing panics while one of these locks is held.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl EnsHandle {
    /// Runs `work` on a new thread. A panic in `work` ends the run as `Failed`.
    fn spawn<F>(work: F) -> Result<EnsHandle>
    where
        F: FnOnce(&AtomicBool) -> StdResult<EnsembleResult, EnsembleError> + Send + 'static,
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(Mutex::new(Slot::Running));
        let (c, s) = (Arc::clone(&cancel), Arc::clone(&slot));
        let thread = std::thread::Builder::new()
            .name("pckern-ensemble".into())
            .stack_size(MODEL_STACK)
            .spawn(move || {
                let outcome = match crate::barrier::guarded(|| Ok(work(&c))) {
                    Ok(Ok(result)) => Slot::Done(Arc::new(result)),
                    Ok(Err(EnsembleError::Cancelled)) => Slot::Cancelled,
                    Ok(Err(e)) => Slot::Failed(Error::from(e)),
                    Err(panic) => Slot::Failed(panic),
                };
                *lock(&s) = outcome;
            })
            .map_err(|e| Error::Internal(format!("cannot start the ensemble thread: {e}")))?;
        Ok(EnsHandle {
            cancel,
            slot,
            thread: Mutex::new(Some(thread)),
        })
    }
}

impl Drop for EnsHandle {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(t) = lock(&self.thread).take() {
            // The worker runs inside `guarded`, so a join error has nothing to add.
            let _ = t.join();
        }
    }
}

/// Everything in `ens_start`'s job text that is not the head: where the parameters come from.
fn parse_job(text: &str) -> StdResult<(Head, bool), Error> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| Error::BadProblem(format!("job_json: {e}")))?;
    let read = || -> StdResult<(Head, bool), String> {
        let m = object(&v, "job")?;
        let mut allowed = HEAD_KEYS.to_vec();
        allowed.push("params");
        only_keys(m, &allowed, "")?;
        let head = parse_head(m)?;
        let mode = string(required(m, "params", "")?, "params")?;
        match mode {
            "shared" => Ok((head, false)),
            "per_member" => Ok((head, true)),
            other => Err(format!("params must be shared or per_member, got {other}")),
        }
    };
    read().map_err(Error::BadProblem)
}

fn job_of<'a>(
    ir: &'a IrModel,
    head: &Head,
    members: usize,
    x0: &'a [f64],
    params: &'a [f64],
    per_member: bool,
) -> Job<'a> {
    Job {
        model: ir,
        members,
        x0,
        params: if per_member {
            Params::PerMember(params)
        } else {
            Params::Shared(params)
        },
        t0: head.t0,
        t1: head.t1,
        output: head.output.clone(),
        method: head.method,
        bounds: head.bounds,
        max_steps: head.max_steps,
        precision: head.precision,
    }
}

/// Checks the job, then starts the worker and returns the id of the new handle.
pub(crate) fn start(
    model: &Arc<ModelHandle>,
    job_text: &str,
    x0: &[f64],
    params: &[f64],
) -> Result<u64> {
    let (head, per_member) = parse_job(job_text)?;
    let n = model.n();
    if n == 0 {
        return Err(EnsembleError::BadJob("the model has no states".into()).into());
    }
    if !x0.len().is_multiple_of(n) {
        return Err(EnsembleError::BadJob(format!(
            "x0 has {} values, which is not a multiple of {n}, the number of states",
            x0.len()
        ))
        .into());
    }
    let members = x0.len() / n;
    if members == 0 {
        return Err(EnsembleError::BadJob("x0 must hold at least one member".into()).into());
    }
    validate(&job_of(&model.ir, &head, members, x0, params, per_member))?;
    let device = find_device(head.backend, head.device).map_err(Error::GpuUnavailable)?;

    let (model, x0, params) = (Arc::clone(model), x0.to_vec(), params.to_vec());
    let handle = EnsHandle::spawn(move |cancel| {
        let job = job_of(&model.ir, &head, members, &x0, &params, per_member);
        run_on(&device, &job, cancel)
    })?;
    Ok(registry::insert(handle))
}

fn label_json(l: &Label) -> Value {
    json!({
        "backend": l.backend,
        "device": l.device,
        "driver": l.driver,
        "precision": l.precision.name(),
        "single_precision": l.single_precision(),
    })
}

fn done_json(r: &EnsembleResult) -> Value {
    let count = |s: Status| r.status.iter().filter(|&&x| x == s).count();
    json!({
        "state": "done",
        "members": r.members,
        "n": r.n,
        "samples": r.times.len(),
        "counts": {
            "ok": count(Status::Ok),
            "bounds": count(Status::Bounds),
            "step_too_small": count(Status::StepTooSmall),
            "max_steps": count(Status::MaxSteps),
            "nonfinite": count(Status::NonFinite),
        },
        "label": label_json(&r.label),
    })
}

/// The state of the run as the JSON `ens_poll` returns. A worker that failed gives its error.
pub(crate) fn poll(h: &EnsHandle) -> Result<Value> {
    match &*lock(&h.slot) {
        Slot::Running => Ok(json!({"state": "running"})),
        Slot::Cancelled => Ok(json!({"state": "cancelled"})),
        Slot::Done(r) => Ok(done_json(r)),
        Slot::Failed(e) => Err(e.clone()),
    }
}

/// Asks the worker to stop at its next check. Cancelling twice, or after the end, is fine.
pub(crate) fn cancel(h: &EnsHandle) {
    h.cancel.store(true, Ordering::Relaxed);
}

/// Writes one array of a finished run into `out`, which must have exactly its length.
pub(crate) fn copy(h: &EnsHandle, what: u32, out: &mut [f64]) -> Result<()> {
    let r = match &*lock(&h.slot) {
        Slot::Running => return Err(bad_arg("the ensemble run has not finished")),
        Slot::Cancelled => return Err(bad_arg("the ensemble run was cancelled")),
        Slot::Failed(e) => return Err(e.clone()),
        Slot::Done(r) => Arc::clone(r),
    };
    let (name, want) = match what {
        PC_ENS_STATES => ("states", r.states.len()),
        PC_ENS_TIMES => ("times", r.times.len()),
        PC_ENS_STATUS => ("status", r.members),
        PC_ENS_T_REACHED => ("t_reached", r.members),
        PC_ENS_ACCEPTED => ("accepted", r.members),
        PC_ENS_REJECTED => ("rejected", r.members),
        other => return Err(bad_arg(format!("unknown ens_copy selector {other}"))),
    };
    if out.len() != want {
        return Err(bad_arg(format!(
            "{name}: expected {want} values, got {}",
            out.len()
        )));
    }
    match what {
        PC_ENS_STATES => out.copy_from_slice(&r.states),
        PC_ENS_TIMES => out.copy_from_slice(&r.times),
        PC_ENS_STATUS => fill(out, r.status.iter().map(|s| f64::from(s.code()))),
        PC_ENS_T_REACHED => out.copy_from_slice(&r.t_reached),
        PC_ENS_ACCEPTED => fill(out, r.accepted.iter().map(|&k| f64::from(k))),
        _ => fill(out, r.rejected.iter().map(|&k| f64::from(k))),
    }
    Ok(())
}

fn fill(out: &mut [f64], values: impl Iterator<Item = f64>) {
    for (slot, v) in out.iter_mut().zip(values) {
        *slot = v;
    }
}

/// The backends and devices as the JSON `ens_devices` returns.
pub(crate) fn devices() -> Result<String> {
    serde_json::to_string(&devices_json(&list_devices()))
        .map_err(|e| Error::Internal(format!("devices JSON: {e}")))
}

/// Test hook, not in the table: inserts an ensemble whose worker panics, and returns its id.
#[doc(hidden)]
pub fn debug_ensemble_panic() -> u64 {
    let work = |_: &AtomicBool| -> StdResult<EnsembleResult, EnsembleError> {
        panic!("debug ensemble panic")
    };
    EnsHandle::spawn(work).map_or(0, registry::insert)
}

// The precision enum is only named here so a rename in pc-ensemble breaks this file loudly.
const _: [Precision; 2] = [Precision::Single, Precision::Double];
