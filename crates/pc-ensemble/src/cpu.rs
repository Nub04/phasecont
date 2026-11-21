//! The CPU backend: every member on a rayon thread, in `f32` or `f64`.
//!
//! The run is cut into launches of at most `steps_per_launch` step attempts per member, the
//! same way a GPU run is. Members never read each other's state, so the cut changes nothing
//! in the result; the tests check that bit for bit. The cancel flag is read before each
//! launch and before each member inside a launch, so after a cancel every thread finishes at
//! most the member it is on. A cancelled run returns no result, so a member that a cancel
//! skipped changes nothing.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::job::{EnsembleError, EnsembleResult, Job, Label, Precision, Status, validate};
use crate::real::Real;
use crate::tape::Tape;
use crate::traj::{Ctx, Traj};

/// Runs `job` on the CPU. `steps_per_launch` is at least 1; pass `u64::MAX` for one launch.
pub fn run_cpu(
    job: &Job,
    steps_per_launch: u64,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    if steps_per_launch == 0 {
        return Err(EnsembleError::BadJob(
            "steps per launch must be at least 1".into(),
        ));
    }
    match job.precision {
        Precision::Single => run_as::<f32>(job, steps_per_launch, cancel),
        Precision::Double => run_as::<f64>(job, steps_per_launch, cancel),
    }
}

fn run_as<T: Real>(
    job: &Job,
    steps_per_launch: u64,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let checked = validate(job)?;
    let tape = Tape::build(job.model)?;
    let mut ctx = Ctx::<T>::new(job, &checked, &tape);
    ctx.cancel = Some(cancel);
    let (work_len, out_len) = (ctx.work_len(), ctx.out_len());
    let members = job.members;

    let mut work = vec![T::ZERO; members * work_len];
    let mut out = vec![T::NAN; members * out_len];
    let mut trajs: Vec<Traj<T>> = (0..members).map(Traj::new).collect();

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(EnsembleError::Cancelled);
        }
        trajs
            .par_iter_mut()
            .zip(work.par_chunks_mut(work_len))
            .zip(out.par_chunks_mut(out_len))
            .for_each_init(
                || ctx.rhs.slab(),
                |vals, ((tr, w), o)| {
                    if !cancel.load(Ordering::Relaxed) {
                        tr.advance(&ctx, w, o, vals, steps_per_launch);
                    }
                },
            );
        // A cancel inside a long evaluation of a tape with calls fails that evaluation, so
        // the member reads `recursion_depth`; the flag is read again before that is taken as
        // a finished run.
        if cancel.load(Ordering::Relaxed) {
            return Err(EnsembleError::Cancelled);
        }
        if trajs.iter().all(|t| t.status != Status::Running) {
            break;
        }
    }

    let t_end = *checked.times.last().expect("validate keeps one sample");
    Ok(EnsembleResult {
        n: checked.n,
        members,
        states: T::into_f64_vec(out),
        status: trajs.iter().map(|t| t.status).collect(),
        t_reached: trajs
            .iter()
            .map(|t| {
                if t.status == Status::Ok {
                    t_end
                } else {
                    t.t.to_f64()
                }
            })
            .collect(),
        accepted: trajs.iter().map(|t| t.accepted).collect(),
        rejected: trajs.iter().map(|t| t.rejected).collect(),
        times: checked.times,
        label: Label {
            backend: "cpu".into(),
            device: format!("rayon, {} threads", rayon::current_num_threads()),
            driver: format!("pc-ensemble {}", env!("CARGO_PKG_VERSION")),
            precision: job.precision,
            math: job.math,
        },
    })
}
