//! The CPU backend: every member on a rayon thread, in `f32` or `f64`.
//!
//! The run is cut into launches of at most `steps_per_launch` step attempts per member, the
//! same way a GPU run is. Members never read each other's state, so the cut changes nothing
//! in the result; the tests check that bit for bit. The cancel flag is read before each
//! launch and before each member inside a launch, so after a cancel every thread finishes at
//! most the member it is on. A cancelled run returns no result, so a member that a cancel
//! skipped changes nothing.
//!
//! RK4 runs up to `CPU_LANES` consecutive members per rayon task, stage by stage across the
//! lanes, and the cancel flag is then read before each group (SPEC section 13, `cpu`). A small
//! ensemble gets smaller groups, so that every thread still gets work ([`default_lanes`]).

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::job::{EnsembleError, EnsembleResult, Job, Label, Precision, Status, validate};
use crate::real::Real;
use crate::tape::Tape;
use crate::traj::{Ctx, Kind, Traj};

/// The largest number of RK4 members one rayon task runs together.
pub const CPU_LANES: usize = 4;

/// The RK4 group size [`run_cpu`] uses for `members` on `threads` threads: the largest W of
/// `CPU_LANES` and 2 with `members >= 2 * W * threads`, so every thread gets at least two
/// groups, and 1 otherwise. A thread count of 0 counts as 1. The size changes no bit.
pub fn default_lanes(members: usize, threads: usize) -> usize {
    let threads = threads.max(1);
    [CPU_LANES, 2]
        .into_iter()
        .find(|&w| members >= 2 * w * threads)
        .unwrap_or(1)
}

/// The size of the CPU thread pool: `RAYON_NUM_THREADS`, or the number of logical cores.
/// Other crates size their own worker sets from it (SPEC 8ai-2).
pub fn cpu_threads() -> usize {
    rayon::current_num_threads()
}

/// Runs `job` on the CPU. `steps_per_launch` is at least 1; pass `u64::MAX` for one launch.
pub fn run_cpu(
    job: &Job,
    steps_per_launch: u64,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let lanes = default_lanes(job.members, rayon::current_num_threads());
    run_cpu_lanes(job, steps_per_launch, cancel, lanes)
}

/// [`run_cpu`] with RK4 groups of `lanes` members, 1, 2, 4 or 8; 1 runs one member at a
/// time. The group size changes no bit of the result. For tests and benches.
#[doc(hidden)]
pub fn run_cpu_lanes(
    job: &Job,
    steps_per_launch: u64,
    cancel: &AtomicBool,
    lanes: usize,
) -> Result<EnsembleResult, EnsembleError> {
    if steps_per_launch == 0 {
        return Err(EnsembleError::BadJob(
            "steps per launch must be at least 1".into(),
        ));
    }
    if !matches!(lanes, 1 | 2 | 4 | 8) {
        return Err(EnsembleError::BadJob(format!(
            "lanes must be 1, 2, 4 or 8, not {lanes}"
        )));
    }
    match job.precision {
        Precision::Single => run_as::<f32>(job, steps_per_launch, cancel, lanes),
        Precision::Double => run_as::<f64>(job, steps_per_launch, cancel, lanes),
    }
}

/// One launch of every member, `W` consecutive members per rayon task.
fn launch_groups<T: Real, const W: usize>(
    ctx: &Ctx<T>,
    trajs: &mut [Traj<T>],
    work: &mut [T],
    out: &mut [T],
    steps_per_launch: u64,
    cancel: &AtomicBool,
) {
    let (work_len, out_len) = (ctx.work_len(), ctx.out_len());
    trajs
        .par_chunks_mut(W)
        .zip(work.par_chunks_mut(W * work_len))
        .zip(out.par_chunks_mut(W * out_len))
        .for_each_init(
            || {
                let slabs = std::array::from_fn::<_, W, _>(|_| ctx.rhs.slab());
                (slabs, vec![T::ZERO; 7 * W * ctx.n])
            },
            |(slabs, g), ((trs, w), o)| {
                if !cancel.load(Ordering::Relaxed) {
                    Traj::advance_group(trs, ctx, w, o, slabs, g, steps_per_launch);
                }
            },
        );
}

fn run_as<T: Real>(
    job: &Job,
    steps_per_launch: u64,
    cancel: &AtomicBool,
    lanes: usize,
) -> Result<EnsembleResult, EnsembleError> {
    let checked = validate(job)?;
    let tape = Tape::build(job.model)?;
    let mut ctx = Ctx::<T>::new(job, &checked, &tape);
    ctx.cancel = Some(cancel);
    let (work_len, out_len) = (ctx.work_len(), ctx.out_len());
    let members = job.members;
    let rk4 = matches!(ctx.kind, Kind::Rk4(_));

    let mut work = vec![T::ZERO; members * work_len];
    let mut out = vec![T::NAN; members * out_len];
    let mut trajs: Vec<Traj<T>> = (0..members).map(Traj::new).collect();

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(EnsembleError::Cancelled);
        }
        let (w, o, s) = (&mut work[..], &mut out[..], steps_per_launch);
        match lanes {
            2 if rk4 => launch_groups::<T, 2>(&ctx, &mut trajs, w, o, s, cancel),
            4 if rk4 => launch_groups::<T, 4>(&ctx, &mut trajs, w, o, s, cancel),
            8 if rk4 => launch_groups::<T, 8>(&ctx, &mut trajs, w, o, s, cancel),
            _ => trajs
                .par_iter_mut()
                .zip(w.par_chunks_mut(work_len))
                .zip(o.par_chunks_mut(out_len))
                .for_each_init(
                    || ctx.rhs.slab(),
                    |vals, ((tr, w), o)| {
                        if !cancel.load(Ordering::Relaxed) {
                            tr.advance(&ctx, w, o, vals, s);
                        }
                    },
                ),
        }
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
            // The CPU keeps no GPU call stack: it names the layout asked for and runs none.
            stack: job.stack,
            stack_ran: crate::job::StackRan::Member,
            sorted: false,
        },
    })
}
