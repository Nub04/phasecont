//! Member order and stack layout choice for the `auto` stack layout (SPEC section 8af-5).
//!
//! A kernel with calls runs fastest with the warp stack layout when the 32 lanes of a warp
//! recurse to nearly the same depth, and up to about six times slower than the member layout
//! when they do not. `auto` evaluates the right-hand side of a sample of the members once on
//! the CPU at `(x0, t0)`, records how deep it went, sorts all members by the frames they
//! entered when the sample is out of order, and bounds from the depths the ratio of warp time to member time.
//! The GPU run checks the bounds against the depths a probe kernel reports and, when they
//! leave the choice open, times both layouts.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use rayon::prelude::*;

use crate::job::{EnsembleError, EnsembleResult, Job, Method, Params, Precision, Status};
use crate::kernel::CTL;
use crate::real::Real;
use crate::tape::{CpuTape, KERNEL_FRAMES, Tape, kernel_stride};

/// How deep one right-hand side of a member recursed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Depth {
    /// The most frames active at once.
    pub peak: u32,
    /// The frames entered.
    pub calls: u64,
}

/// Lanes of one warp, the group the bounds look at.
pub const WARP: usize = 32;

/// The widest spread of peak depth inside a single-chain group (every member enters as many
/// frames as its peak) that still counts as narrow, in both precisions.
pub const NARROW_SPREAD: u32 = 16;

/// The narrowest spread of peak depth of a single-chain group that counts as wide.
pub fn wide_spread(precision: Precision) -> u32 {
    match precision {
        Precision::Double => 128,
        Precision::Single => 512,
    }
}

/// The class of one warp group of depths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Narrow,
    Band,
    Wide,
    /// Recursion other than one chain in which no lane enters more than [`SHALLOW_CALLS`]
    /// frames.
    TreeShallow,
    /// Deeper recursion other than one chain whose lanes differ by at most
    /// [`tree_narrow_ratio`] in frames entered.
    TreeNarrow,
    /// Any other recursion that is not one chain.
    TreeBand,
}

/// The most frames a lane of a shallow tree group enters (`fib(7)`).
pub const SHALLOW_CALLS: u64 = 41;

/// The largest ratio of the most to the fewest frames entered by the lanes of a tree group
/// that still counts as narrow.
pub fn tree_narrow_ratio(precision: Precision) -> u64 {
    match precision {
        Precision::Double => 3,
        Precision::Single => 5,
    }
}

/// The lowest and the highest ratio of warp time to member time measured on the RTX 5060 for
/// a group of this class (SPEC section 8af-5, table of 2026-11-21).
pub fn ratio_row(group: Group, precision: Precision) -> (f64, f64) {
    match (group, precision) {
        (Group::Narrow, Precision::Double) => (0.15, 0.44),
        (Group::Band, Precision::Double) => (0.39, 2.26),
        (Group::Wide, Precision::Double) => (1.64, 3.12),
        (Group::Narrow, Precision::Single) => (0.15, 0.67),
        (Group::Band, Precision::Single) => (0.44, 4.04),
        (Group::Wide, Precision::Single) => (1.54, 5.79),
        (Group::TreeShallow, Precision::Double) => (0.71, 0.99),
        (Group::TreeNarrow, Precision::Double) => (0.24, 0.74),
        (Group::TreeBand, Precision::Double) => (0.77, 1.22),
        (Group::TreeShallow, Precision::Single) => (0.51, 1.01),
        (Group::TreeNarrow, Precision::Single) => (0.12, 0.65),
        (Group::TreeBand, Precision::Single) => (0.34, 1.22),
    }
}

/// `auto` runs warp when the upper bound of the ratio is at most this.
pub const CHOOSE_RATIO: f64 = 0.9;
/// `auto` runs member when the lower bound of the ratio is at least this.
pub const FALLBACK_RATIO: f64 = 1.0;

/// The probe runs again once the launches since the last probe have taken this many times as
/// long as that probe.
pub const PROBE_EVERY: u32 = 20;
/// The shortest timed launch when `auto` times both layouts.
pub const TIMED_MIN: Duration = Duration::from_millis(10);
/// Member wins a timing when it takes less than this share of the warp time per step.
pub const SWITCH_MARGIN: f64 = 0.9;

/// The depth of every member's right-hand side at `(x0, t0)`, in the job's precision, with
/// the kernel's frame cap. `None` for a tape without calls. A member that passes the cap
/// reports the depth where it stopped, which is the cap.
pub fn depth_keys(
    tape: &Tape,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<Option<Vec<Depth>>, EnsembleError> {
    if tape.units.is_empty() {
        return Ok(None);
    }
    let all: Vec<usize> = (0..job.members).collect();
    keys_of(tape, job, &all, cancel).map(Some)
}

/// The keys of the members in `which`, in that order.
fn keys_of(
    tape: &Tape,
    job: &Job,
    which: &[usize],
    cancel: &AtomicBool,
) -> Result<Vec<Depth>, EnsembleError> {
    match job.precision {
        Precision::Double => keys_as::<f64>(tape, job, which, cancel),
        Precision::Single => keys_as::<f32>(tape, job, which, cancel),
    }
}

fn keys_as<T: Real>(
    tape: &Tape,
    job: &Job,
    which: &[usize],
    cancel: &AtomicBool,
) -> Result<Vec<Depth>, EnsembleError> {
    let (n, n_p) = (tape.n, tape.n_p);
    let mut cpu = CpuTape::<T>::new(tape);
    cpu.set_max_calls(KERNEL_FRAMES);
    let t0 = T::from_f64(job.t0);
    let keys: Vec<Depth> = which
        .par_iter()
        .map_init(
            || {
                (
                    cpu.slab(),
                    vec![T::ZERO; n],
                    vec![T::ZERO; n_p],
                    vec![T::ZERO; n],
                )
            },
            |(slab, x, p, out), &m| {
                if cancel.load(Ordering::Relaxed) {
                    return Depth::default();
                }
                for (d, &s) in x.iter_mut().zip(&job.x0[m * n..(m + 1) * n]) {
                    *d = T::from_f64(s);
                }
                let src = match job.params {
                    Params::Shared(v) => &v[..n_p],
                    Params::PerMember(v) => &v[m * n_p..(m + 1) * n_p],
                };
                for (d, &s) in p.iter_mut().zip(src) {
                    *d = T::from_f64(s);
                }
                cpu.bind(p, slab);
                cpu.eval_xt_with(x, t0, slab, out, Some(cancel));
                let (calls, peak) = slab.depth().expect("a tape with calls");
                Depth { peak, calls }
            },
        )
        .collect();
    if cancel.load(Ordering::Relaxed) {
        return Err(EnsembleError::Cancelled);
    }
    Ok(keys)
}

/// Every how many groups of [`WARP`] members the key pass samples one: all of them below 16
/// groups, then at least 8, and at most 1 in 16.
pub fn sample_stride(groups: usize) -> usize {
    if groups < 16 {
        1
    } else if groups < 128 {
        groups / 8
    } else {
        16
    }
}

/// The members of the sampled groups, in the caller's order.
pub fn sampled_members(members: usize) -> Vec<usize> {
    let s = sample_stride(members.div_ceil(WARP));
    (0..members)
        .filter(|m| (m / WARP).is_multiple_of(s))
        .collect()
}

/// The sort key of a member: the frames it entered, then its peak depth. A lane's time in
/// either layout follows the frames it enters, so a warp of members with similar keys
/// diverges little. On a single chain the two numbers are equal, so the order is the order
/// of the peak, but a tree beside a chain (`fib(n) + s(k)`) sorted by peak puts members with
/// a deep chain and a short tree next to members with a short chain and a wide tree.
pub fn sort_key(d: &Depth) -> (u64, u32) {
    (d.calls, d.peak)
}

/// True when `keys` are in nondecreasing order of [`sort_key`].
pub fn in_order(keys: &[Depth]) -> bool {
    keys.windows(2).all(|w| sort_key(&w[0]) <= sort_key(&w[1]))
}

/// The stable order that sorts `keys` by [`sort_key`], or `None` when they are in that
/// order already.
pub fn sort_order(keys: &[Depth]) -> Option<Vec<u32>> {
    if in_order(keys) {
        return None;
    }
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_by_key(|&m| sort_key(&keys[m as usize]));
    Some(order)
}

/// The class of one warp group. A single chain (every lane enters as many frames as its
/// peak) is narrow when its peaks differ by at most [`NARROW_SPREAD`], wide from
/// [`wide_spread`] on and in the band between. Any other recursion is a tree group: shallow
/// when no lane enters more than [`SHALLOW_CALLS`] frames, narrow when the most frames a lane
/// enters are at most [`tree_narrow_ratio`] times the fewest, and in the tree band otherwise.
/// A tree group is never wide. An empty group is narrow.
pub fn classify(group: &[Depth], precision: Precision) -> Group {
    if group.is_empty() {
        return Group::Narrow;
    }
    if group.iter().all(|d| d.calls == u64::from(d.peak)) {
        let (lo, hi) = group.iter().fold((u32::MAX, 0), |(lo, hi), d| {
            (lo.min(d.peak), hi.max(d.peak))
        });
        match hi - lo {
            s if s <= NARROW_SPREAD => Group::Narrow,
            s if s >= wide_spread(precision) => Group::Wide,
            _ => Group::Band,
        }
    } else {
        let (lo, hi) = group.iter().fold((u64::MAX, 0), |(lo, hi), d| {
            (lo.min(d.calls), hi.max(d.calls))
        });
        if hi <= SHALLOW_CALLS {
            Group::TreeShallow
        } else if hi <= lo.saturating_mul(tree_narrow_ratio(precision)) {
            Group::TreeNarrow
        } else {
            Group::TreeBand
        }
    }
}

/// The lower and the upper bound of warp time over member time for these groups: the means
/// of the bottom and the top of each group's row of [`ratio_row`], weighted by the most
/// frames a lane of the group enters. `None` when no lane enters a frame, so the layouts cost
/// the same.
pub fn bounds<'a>(
    groups: impl IntoIterator<Item = &'a [Depth]>,
    precision: Precision,
) -> Option<(f64, f64)> {
    let (mut lo, mut hi, mut den) = (0.0f64, 0.0f64, 0.0f64);
    for g in groups {
        let w = g.iter().map(|d| d.calls).max().unwrap_or(0) as f64;
        let (l, h) = ratio_row(classify(g, precision), precision);
        lo += w * l;
        hi += w * h;
        den += w;
    }
    (den > 0.0).then(|| (lo / den, hi / den))
}

/// The fewest right-hand sides a member evaluates before `auto` computes keys: a shorter run
/// costs less than the CPU pass that sorts it, so it runs member (SPEC section 8af-5).
pub const FLOOR_RHS: f64 = 20.0;

/// How many times the compile cost of the kernels it lacks the member kernel of a model must
/// run before `auto` takes its keys and builds the warp and probe kernels (SPEC section 8af-5,
/// compile gate). With 10 a model whose calls end just after the gate stays within 1.1 times
/// `member`.
pub const REPAY: f64 = 10.0;

/// The factor the gate uses, [`REPAY`] unless a test set another one.
static REPAY_NOW: AtomicU64 = AtomicU64::new(REPAY.to_bits());

/// Sets the factor of the compile gate for this process, so a test can pass the gate on the
/// second call. Not for use outside tests.
#[doc(hidden)]
pub fn set_repay_factor(factor: f64) {
    REPAY_NOW.store(factor.to_bits(), Ordering::SeqCst);
}

/// True when the member kernel has run long enough for `auto` to build the `missing` warp
/// and probe kernels: its launches took `spent` in this process, at least `factor` times the
/// estimated compile time of the missing kernels, 2 · `nvrtc` each, with `nvrtc` the NVRTC
/// time of the member kernel. With nothing missing the gate passes.
pub fn gate_passes_with(factor: f64, spent: Duration, nvrtc: Duration, missing: usize) -> bool {
    spent.as_secs_f64() >= factor * missing as f64 * 2.0 * nvrtc.as_secs_f64()
}

/// [`gate_passes_with`] at the factor of this process.
pub fn gate_passes(spent: Duration, nvrtc: Duration, missing: usize) -> bool {
    let factor = f64::from_bits(REPAY_NOW.load(Ordering::SeqCst));
    gate_passes_with(factor, spent, nvrtc, missing)
}

/// What `auto` does with a job on CUDA.
#[derive(Clone, Debug, PartialEq)]
pub enum AutoPlan {
    /// Run the member layout in the job's order.
    Member,
    /// Run in `order` (`None` keeps the job's order), choosing per batch from `keys`, which
    /// are in the run order. Without a full pass only the sampled groups have keys.
    Keys {
        keys: Vec<Option<Depth>>,
        order: Option<Vec<u32>>,
    },
}

/// True when the run is too short for the keys to pay off: RK4 below [`FLOOR_RHS`] right-hand
/// sides (four per step) or a map below that many iterates. DP5 has no fixed step count, so it
/// always computes keys.
pub fn below_floor(job: &Job) -> bool {
    let span = job.t1 - job.t0;
    match job.method {
        Method::Rk4 { dt } => 4.0 * (span / dt).ceil() < FLOOR_RHS,
        Method::Iter => span < FLOOR_RHS,
        Method::Dp5 { .. } => false,
    }
}

/// True when `auto` runs `job` as member without keys whatever the kernels kept: a tape
/// without calls, fewer than [`WARP`] members, a run below the floor or a stack that would
/// pass 2^31 reals in a group of [`WARP`]. Such a run skips the compile gate.
pub fn member_only(tape: &Tape, job: &Job) -> bool {
    let stride = kernel_stride(tape);
    tape.units.is_empty()
        || job.members < WARP
        || below_floor(job)
        || stride.saturating_mul(KERNEL_FRAMES).saturating_mul(WARP) >= 1 << 31
}

/// The plan of `auto` for `job` on CUDA. A tape without calls, a job of fewer than [`WARP`]
/// members, a run below the floor and a stack that would pass 2^31 reals in a group of
/// [`WARP`] run member without keys. Every other job gets the keys of the sampled groups
/// ([`sampled_members`]); when they are out of order it gets the keys of every member and the
/// order that sorts them.
pub fn plan_auto(tape: &Tape, job: &Job, cancel: &AtomicBool) -> Result<AutoPlan, EnsembleError> {
    plan_with(tape, job, |which| keys_of(tape, job, which, cancel))
}

/// [`plan_auto`] with the keys of the members in `which` from `keys_of`, in that order. The
/// CUDA backend passes the keys of its key kernel, which runs the same right-hand side as the
/// kernels of the run.
pub fn plan_with(
    tape: &Tape,
    job: &Job,
    mut keys_of: impl FnMut(&[usize]) -> Result<Vec<Depth>, EnsembleError>,
) -> Result<AutoPlan, EnsembleError> {
    if member_only(tape, job) {
        return Ok(AutoPlan::Member);
    }
    let sample = sampled_members(job.members);
    let sampled = keys_of(&sample)?;
    let mut keys: Vec<Option<Depth>> = vec![None; job.members];
    for (&m, &k) in sample.iter().zip(&sampled) {
        keys[m] = Some(k);
    }
    if in_order(&sampled) {
        return Ok(AutoPlan::Keys { keys, order: None });
    }
    // The rest of the members, then the order of all of them.
    let rest: Vec<usize> = (0..job.members).filter(|&m| keys[m].is_none()).collect();
    for (&m, k) in rest.iter().zip(keys_of(&rest)?) {
        keys[m] = Some(k);
    }
    let all: Vec<Depth> = keys.iter().map(|k| k.expect("every key")).collect();
    let order = sort_order(&all);
    let keys = match &order {
        Some(o) => o.iter().map(|&m| Some(all[m as usize])).collect(),
        None => keys,
    };
    Ok(AutoPlan::Keys { keys, order })
}

/// Copies of the initial states and per-member parameters of `job` in `order`. Shared
/// parameters need no copy, so the second value is `None` for them.
pub fn permute(job: &Job, order: &[u32]) -> (Vec<f64>, Option<Vec<f64>>) {
    let n = job.model.n();
    let gather = |v: &[f64], k: usize| -> Vec<f64> {
        let mut out = Vec::with_capacity(order.len() * k);
        for &m in order {
            let m = m as usize;
            out.extend_from_slice(&v[m * k..(m + 1) * k]);
        }
        out
    };
    let params = match job.params {
        Params::Shared(_) => None,
        Params::PerMember(p) => Some(gather(p, job.model.n_p())),
    };
    (gather(job.x0, n), params)
}

/// Puts the members of a result that ran in `order` back in the job's order: member `i` of
/// `r` is member `order[i]` of the job.
pub fn unpermute(r: EnsembleResult, order: &[u32]) -> EnsembleResult {
    let block = r.times.len() * r.n;
    let mut states = vec![0.0; r.states.len()];
    let mut status = vec![Status::Ok; r.members];
    let mut t_reached = vec![0.0; r.members];
    let mut accepted = vec![0; r.members];
    let mut rejected = vec![0; r.members];
    for (i, &m) in order.iter().enumerate() {
        let m = m as usize;
        states[m * block..(m + 1) * block].copy_from_slice(&r.states[i * block..(i + 1) * block]);
        status[m] = r.status[i];
        t_reached[m] = r.t_reached[i];
        accepted[m] = r.accepted[i];
        rejected[m] = r.rejected[i];
    }
    EnsembleResult {
        states,
        status,
        t_reached,
        accepted,
        rejected,
        ..r
    }
}

/// How an `auto` batch starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// The warp kernel, with the probe.
    Warp,
    /// The member kernel for the whole batch.
    Member,
    /// The warp kernel, timing both layouts after the first launch.
    Time,
}

/// The choice for bounds `b` of the ratio: warp when the upper bound is at most
/// [`CHOOSE_RATIO`], member when the lower bound is at least [`FALLBACK_RATIO`] or no member
/// enters a frame, and timing otherwise.
pub fn choice(b: Option<(f64, f64)>) -> Choice {
    match b {
        None => Choice::Member,
        Some((_, hi)) if hi <= CHOOSE_RATIO => Choice::Warp,
        Some((lo, _)) if lo >= FALLBACK_RATIO => Choice::Member,
        Some(_) => Choice::Time,
    }
}

/// The choice for these groups: `None` when no lane enters a frame, warp when every group
/// whose lanes enter frames is a shallow tree (warp then costs at most 1.01 times member and
/// a timing would cost more), and [`choice`] of the [`bounds`] otherwise.
pub fn verdict(groups: &[Vec<Depth>], precision: Precision) -> Option<Choice> {
    let deep: Vec<&[Depth]> = groups
        .iter()
        .filter(|g| g.iter().any(|d| d.calls > 0))
        .map(Vec::as_slice)
        .collect();
    if deep.is_empty() {
        return None;
    }
    if deep
        .iter()
        .all(|g| classify(g, precision) == Group::TreeShallow)
    {
        return Some(Choice::Warp);
    }
    Some(choice(bounds(deep, precision)))
}

/// The groups of [`WARP`] members of `keys[start..end]` that have keys, cut where the groups
/// of the whole run are cut, so that a sampled group stays whole.
fn known_groups(keys: &[Option<Depth>], start: usize, end: usize) -> Vec<Vec<Depth>> {
    let mut out = Vec::new();
    let mut a = start;
    while a < end {
        let b = ((a / WARP + 1) * WARP).min(end);
        let g: Vec<Depth> = keys[a..b].iter().flatten().copied().collect();
        if !g.is_empty() {
            out.push(g);
        }
        a = b;
    }
    out
}

/// The choice for the batch of the members `start..end` of the run, from their keys. A batch
/// of fewer than [`WARP`] members runs member. A batch without a sampled group takes the
/// bounds of every sampled group of the run, since those stand for the run.
pub fn choose(keys: &[Option<Depth>], start: usize, end: usize, precision: Precision) -> Choice {
    if end - start < WARP {
        return Choice::Member;
    }
    let mut groups = known_groups(keys, start, end);
    if groups.is_empty() {
        groups = known_groups(keys, 0, keys.len());
    }
    verdict(&groups, precision).unwrap_or(Choice::Member)
}

/// What a batch on warp does after a probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfterProbe {
    /// Keep running warp.
    Stay,
    /// Time both layouts.
    Time,
    /// Run the member kernel from now on.
    Member,
}

/// The verdict of the probe depths `dep` (the peak and the calls of each member, two words per
/// member) for the members still running in `ctl`, grouped by the warp each runs in: member
/// when the [`verdict`] is member, timing when it is timing and the batch has not `timed`
/// yet, and warp otherwise.
pub fn after_probe(
    dep: &[u32],
    ctl: &[u32],
    members: usize,
    precision: Precision,
    timed: bool,
) -> AfterProbe {
    let running = u32::from(Status::Running.code());
    let groups: Vec<Vec<Depth>> = (0..members.div_ceil(WARP))
        .map(|w| {
            (w * WARP..((w + 1) * WARP).min(members))
                .filter(|&m| ctl[m * CTL] == running)
                .map(|m| Depth {
                    peak: dep[2 * m],
                    calls: u64::from(dep[2 * m + 1]),
                })
                .collect::<Vec<_>>()
        })
        .filter(|g| !g.is_empty())
        .collect();
    match verdict(&groups, precision) {
        Some(Choice::Member) => AfterProbe::Member,
        Some(Choice::Time) if !timed => AfterProbe::Time,
        _ => AfterProbe::Stay,
    }
}

/// True when the launches since the last probe, which took `since`, have taken
/// [`PROBE_EVERY`] times as long as that probe, which took `probe`.
pub fn probe_due(since: Duration, probe: Duration) -> bool {
    since >= probe * PROBE_EVERY
}

/// The step attempts of a timed launch, given that `steps` took `took`: enough for
/// [`TIMED_MIN`], and at least one.
pub fn timed_steps(steps: u64, took: Duration) -> u64 {
    // A launch too short for the clock grows to the cap.
    let want = (steps as f64 * TIMED_MIN.as_secs_f64() / took.as_secs_f64()).ceil();
    (want.min(crate::gpu::MAX_STEPS_PER_LAUNCH as f64) as u64).max(1)
}

/// How far from [`SWITCH_MARGIN`], as a factor either way, a ratio of short timed launches
/// must be for the timing to decide without growing them.
pub const CLEAR_FACTOR: f64 = 1.5;

/// What a timing does with a pair of timed launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Judge {
    /// Go back to warp.
    Warp,
    /// Run member for the rest of the batch.
    Member,
    /// Run the timed launches again for this many step attempts each.
    Grow(u64),
}

/// The verdict on a warp launch and a member launch, each `(steps, time)`, where the member
/// time is the mean of the member launches before and after the warp one (SPEC section
/// 8af-5, timing both layouts): [`member_wins`] decides when both took at least
/// [`TIMED_MIN`], when the ratio of member to warp time per step is at most
/// [`SWITCH_MARGIN`] / [`CLEAR_FACTOR`] or at least [`SWITCH_MARGIN`] * [`CLEAR_FACTOR`], or
/// when the launches cannot grow; otherwise both run again for steps that would take the
/// faster one [`TIMED_MIN`].
pub fn judge(warp: (u64, Duration), member: (u64, Duration)) -> Judge {
    let decide = || match member_wins(warp, member) {
        true => Judge::Member,
        false => Judge::Warp,
    };
    if warp.1 >= TIMED_MIN && member.1 >= TIMED_MIN {
        return decide();
    }
    let per = |(n, t): (u64, Duration)| t.as_secs_f64() / n.max(1) as f64;
    let (w, m) = (per(warp), per(member));
    if m <= SWITCH_MARGIN / CLEAR_FACTOR * w || m >= SWITCH_MARGIN * CLEAR_FACTOR * w {
        return decide();
    }
    let n = warp.0.max(member.0);
    let fast = if m < w { member } else { warp };
    match timed_steps(fast.0, fast.1) {
        k if k > n => Judge::Grow(k),
        _ => decide(),
    }
}

/// True when the member launch, `(steps, time)`, took less than [`SWITCH_MARGIN`] of the warp
/// launch per step.
pub fn member_wins(warp: (u64, Duration), member: (u64, Duration)) -> bool {
    let per = |(n, t): (u64, Duration)| t.as_secs_f64() / n.max(1) as f64;
    per(member) < SWITCH_MARGIN * per(warp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_compile_gate_passes_at_ten_times_the_missing_compiles() {
        let ms = Duration::from_millis;
        let t = ms(90);
        // Nothing missing: the gate passes at once.
        assert!(gate_passes_with(REPAY, Duration::ZERO, t, 0));
        // One missing kernel costs 2 · 90 ms, so the gate needs 1.8 s of member launches.
        assert!(!gate_passes_with(REPAY, ms(1799), t, 1));
        assert!(gate_passes_with(REPAY, ms(1800), t, 1));
        // Two missing kernels need 3.6 s.
        assert!(!gate_passes_with(REPAY, ms(3599), t, 2));
        assert!(gate_passes_with(REPAY, ms(3600), t, 2));
        // A factor of 0 passes the gate as soon as the member kernel is kept.
        assert!(gate_passes_with(0.0, Duration::ZERO, t, 2));
    }

    fn chain(peaks: &[u32]) -> Vec<Depth> {
        peaks
            .iter()
            .map(|&p| Depth {
                peak: p,
                calls: u64::from(p),
            })
            .collect()
    }

    #[test]
    fn sorted_keys_need_no_order_and_the_order_is_stable() {
        assert_eq!(sort_order(&chain(&[1, 1, 2, 5])), None);
        let k = chain(&[3, 1, 3, 1, 2]);
        assert_eq!(sort_order(&k), Some(vec![1, 3, 4, 0, 2]));
        // Frames entered first, then peak.
        let t = vec![
            Depth { peak: 2, calls: 9 },
            Depth { peak: 2, calls: 3 },
            Depth { peak: 1, calls: 50 },
            Depth { peak: 5, calls: 9 },
            Depth { peak: 4, calls: 9 },
        ];
        assert_eq!(sort_order(&t), Some(vec![1, 0, 4, 3, 2]));
        // A deep chain with few frames comes before a shallow tree with many.
        let t = vec![
            Depth {
                peak: 13,
                calls: 1219,
            },
            Depth {
                peak: 201,
                calls: 205,
            },
        ];
        assert!(!in_order(&t));
        assert_eq!(sort_order(&t), Some(vec![1, 0]));
    }

    #[test]
    fn a_chain_is_narrow_to_16_and_wide_from_the_precision_spread() {
        for p in [Precision::Double, Precision::Single] {
            let at =
                |spread: u32| -> Vec<u32> { (0..32).map(|i| 2048 + spread * i / 31).collect() };
            assert_eq!(classify(&chain(&at(16)), p), Group::Narrow);
            assert_eq!(classify(&chain(&at(17)), p), Group::Band);
            assert_eq!(classify(&chain(&at(wide_spread(p) - 1)), p), Group::Band);
            assert_eq!(classify(&chain(&at(wide_spread(p))), p), Group::Wide);
            assert_eq!(classify(&[], p), Group::Narrow);
        }
        // A spread of 200 is wide in double and in the band in single.
        let s: Vec<u32> = (0..32).map(|i| 200 * i / 31).collect();
        assert_eq!(classify(&chain(&s), Precision::Double), Group::Wide);
        assert_eq!(classify(&chain(&s), Precision::Single), Group::Band);
    }

    /// The keys of `fib(n)` for each `n`: peak n, calls 2 fib(n + 1) - 1.
    fn fib_keys(ns: &[u32]) -> Vec<Depth> {
        let fib = |n: u32| -> u64 { (0..n).fold((0u64, 1u64), |(a, b), _| (b, a + b)).0 };
        ns.iter()
            .map(|&n| Depth {
                peak: n,
                calls: 2 * fib(n + 1) - 1,
            })
            .collect()
    }

    #[test]
    fn a_tree_is_shallow_to_41_calls_then_narrow_to_the_precision_ratio() {
        let (d, s) = (Precision::Double, Precision::Single);
        // fib(7) enters 41 frames, fib(8) 67.
        assert_eq!(fib_keys(&[7])[0].calls, 41);
        for p in [d, s] {
            assert_eq!(classify(&fib_keys(&[4; 32]), p), Group::TreeShallow);
            // fib(1) enters one frame at peak 1: a chain.
            assert_eq!(classify(&fib_keys(&[1; 32]), p), Group::Narrow);
            let mut g = fib_keys(&[4; 32]);
            g[0] = fib_keys(&[7])[0];
            assert_eq!(classify(&g, p), Group::TreeShallow);
            g[0] = fib_keys(&[8])[0];
            // 67 calls against 9: past both ratios.
            assert_eq!(classify(&g, p), Group::TreeBand);
            assert_eq!(classify(&fib_keys(&[17; 32]), p), Group::TreeNarrow);
        }
        // Calls 67 to 177 (ratio 2.6) are narrow in both; 67 to 287 (4.3) only in single;
        // 67 to 465 (6.9) in neither.
        let one = |top: u32| {
            let mut g = fib_keys(&[8; 32]);
            g[0] = fib_keys(&[top])[0];
            g
        };
        assert_eq!(classify(&one(10), d), Group::TreeNarrow);
        assert_eq!(classify(&one(10), s), Group::TreeNarrow);
        assert_eq!(classify(&one(11), d), Group::TreeBand);
        assert_eq!(classify(&one(11), s), Group::TreeNarrow);
        assert_eq!(classify(&one(12), s), Group::TreeBand);
        // The cutoff sits exactly at the ratio: 3 and 5 times the fewest calls.
        let pair = |lo: u64, hi: u64| {
            let mut g = vec![Depth { peak: 5, calls: lo }; 32];
            g[3].calls = hi;
            g
        };
        assert_eq!(classify(&pair(50, 150), d), Group::TreeNarrow);
        assert_eq!(classify(&pair(50, 151), d), Group::TreeBand);
        assert_eq!(classify(&pair(50, 250), s), Group::TreeNarrow);
        assert_eq!(classify(&pair(50, 251), s), Group::TreeBand);
        // A lane with no frames among tree lanes past the shallow limit is in the band.
        let mut g = fib_keys(&[17; 32]);
        g[9] = Depth::default();
        assert_eq!(classify(&g, d), Group::TreeBand);
    }

    #[test]
    fn the_bounds_weight_groups_by_their_deepest_lane() {
        let d = Precision::Double;
        let zero = chain(&[0; 32]);
        assert_eq!(bounds([zero.as_slice()], d), None);
        let narrow = chain(&[100; 32]);
        assert_eq!(bounds([narrow.as_slice()], d), Some((0.15, 0.44)));
        // A wide group of depths 0 to 310 weighs 310 against the narrow group's 100.
        let wide: Vec<Depth> = chain(&(0..32).map(|i| 10 * i).collect::<Vec<_>>());
        let (lo, hi) = bounds([narrow.as_slice(), wide.as_slice()], d).expect("frames");
        let lo_want = (100.0 * 0.15 + 310.0 * 1.64) / 410.0;
        let hi_want = (100.0 * 0.44 + 310.0 * 3.12) / 410.0;
        assert!((lo - lo_want).abs() < 1e-15 && (hi - hi_want).abs() < 1e-15);
    }

    #[test]
    fn the_choice_is_warp_member_or_timing_by_the_bounds() {
        assert_eq!(choice(None), Choice::Member);
        assert_eq!(choice(Some((0.15, 0.44))), Choice::Warp);
        assert_eq!(choice(Some((0.5, 0.9))), Choice::Warp);
        assert_eq!(choice(Some((1.0, 3.0))), Choice::Member);
        assert_eq!(choice(Some((0.39, 2.26))), Choice::Time);
    }

    #[test]
    fn a_batch_takes_its_own_sampled_groups_or_else_those_of_the_run() {
        let d = Precision::Double;
        let some = |v: Vec<Depth>| v.into_iter().map(Some).collect::<Vec<_>>();
        let mut keys = some(chain(&[100; 64]));
        assert_eq!(choose(&keys, 0, 64, d), Choice::Warp);
        assert_eq!(choose(&keys, 0, 31, d), Choice::Member);
        // The second group is wide but unsampled, so the batch of it alone takes the run's
        // sampled first group.
        let wide: Vec<u32> = (0..32).map(|i| 100 + 10 * i).collect();
        keys.splice(32..64, some(chain(&wide)));
        assert_eq!(choose(&keys, 32, 64, d), Choice::Member);
        for k in &mut keys[32..64] {
            *k = None;
        }
        assert_eq!(choose(&keys, 32, 64, d), Choice::Warp);
        // A batch cut inside a group still sees the whole sampled part of that group.
        let keys = some(chain(
            &(0..64)
                .map(|i| if i < 40 { 100 } else { 3000 })
                .collect::<Vec<_>>(),
        ));
        assert_eq!(choose(&keys, 0, 40, d), Choice::Warp);
        assert_eq!(choose(&keys, 40, 64, d), Choice::Member);
        // No lane enters a frame: member.
        let keys = some(chain(&[0; 64]));
        assert_eq!(choose(&keys, 0, 64, d), Choice::Member);
        // Shallow trees alone run warp although their row reaches 0.99 and 1.01, in both
        // precisions; with a tree band group beside them the batch times.
        for p in [d, Precision::Single] {
            let mut keys = some(fib_keys(&[4; 64]));
            assert_eq!(choose(&keys, 0, 64, p), Choice::Warp);
            keys.splice(
                32..64,
                some(fib_keys(&(0..32).map(|i| 8 + i % 9).collect::<Vec<_>>())),
            );
            assert_eq!(choose(&keys, 0, 64, p), Choice::Time);
        }
    }

    #[test]
    fn the_tree_rows_follow_the_table() {
        let (d, s) = (Precision::Double, Precision::Single);
        assert_eq!(ratio_row(Group::TreeShallow, d), (0.71, 0.99));
        assert_eq!(ratio_row(Group::TreeShallow, s), (0.51, 1.01));
        assert_eq!(ratio_row(Group::TreeNarrow, d), (0.24, 0.74));
        assert_eq!(ratio_row(Group::TreeNarrow, s), (0.12, 0.65));
        assert_eq!(ratio_row(Group::TreeBand, d), (0.77, 1.22));
        assert_eq!(ratio_row(Group::TreeBand, s), (0.34, 1.22));
        // Narrow trees run warp without timing; no tree row reaches the member side.
        for p in [d, s] {
            let g = fib_keys(&[12; 32]);
            assert_eq!(choice(bounds([g.as_slice()], p)), Choice::Warp);
            for t in [Group::TreeShallow, Group::TreeNarrow, Group::TreeBand] {
                assert!(ratio_row(t, p).0 < FALLBACK_RATIO);
            }
        }
    }

    #[test]
    fn sampling_takes_every_group_below_16_then_at_least_8_and_at_most_1_in_16() {
        assert_eq!(sample_stride(1), 1);
        assert_eq!(sample_stride(15), 1);
        assert_eq!(sample_stride(16), 2);
        assert_eq!(sample_stride(127), 15);
        assert_eq!(sample_stride(128), 16);
        assert_eq!(sample_stride(1 << 20), 16);
        assert_eq!(sampled_members(40), (0..40).collect::<Vec<_>>());
        // 16 groups: every second one.
        let s = sampled_members(16 * 32 - 5);
        assert_eq!(s.len(), 8 * 32);
        assert!(s.iter().all(|m| (m / 32) % 2 == 0));
        assert!(in_order(&chain(&[1, 1, 2])));
        assert!(!in_order(&chain(&[2, 1])));
        assert!(in_order(&[]));
    }

    #[test]
    fn the_probe_looks_only_at_running_members() {
        let d = Precision::Double;
        let running = u32::from(Status::Running.code());
        let ok = u32::from(Status::Ok.code());
        let members = 64;
        let mut ctl = vec![0u32; members * CTL];
        let mut dep = vec![0u32; 2 * members];
        for m in 0..members {
            ctl[m * CTL] = running;
            // A narrow first group and a wide second one.
            let peak = if m < 32 { 100 } else { 30 * m as u32 };
            dep[2 * m] = peak;
            dep[2 * m + 1] = peak;
        }
        // Weights 100 and 1890 put the lower bound past 1.
        assert_eq!(
            after_probe(&dep, &ctl, members, d, false),
            AfterProbe::Member
        );
        // With the wide group finished, the rest is narrow.
        for m in 32..members {
            ctl[m * CTL] = ok;
        }
        assert_eq!(after_probe(&dep, &ctl, members, d, false), AfterProbe::Stay);
        // A band group asks for timing once.
        for m in 32..members {
            ctl[m * CTL] = running;
            dep[2 * m] = 100 + 2 * (m as u32 - 32);
            dep[2 * m + 1] = dep[2 * m];
        }
        for m in 0..32 {
            ctl[m * CTL] = ok;
        }
        assert_eq!(after_probe(&dep, &ctl, members, d, false), AfterProbe::Time);
        assert_eq!(after_probe(&dep, &ctl, members, d, true), AfterProbe::Stay);
        // Shallow trees stay on warp with no timing, and a narrow tree beside them too.
        for m in 0..members {
            ctl[m * CTL] = running;
            dep[2 * m] = 3;
            dep[2 * m + 1] = 5;
        }
        assert_eq!(after_probe(&dep, &ctl, members, d, false), AfterProbe::Stay);
        for m in 32..members {
            dep[2 * m] = 12;
            dep[2 * m + 1] = 465;
        }
        assert_eq!(after_probe(&dep, &ctl, members, d, false), AfterProbe::Stay);
        // No running member: nothing to judge.
        for m in 0..members {
            ctl[m * CTL] = ok;
        }
        assert_eq!(after_probe(&dep, &ctl, members, d, false), AfterProbe::Stay);
    }

    #[test]
    fn probes_timed_launches_and_the_switch_follow_their_rules() {
        let ms = Duration::from_millis;
        assert!(!probe_due(ms(19), ms(1)));
        assert!(probe_due(ms(20), ms(1)));
        assert_eq!(timed_steps(16, ms(1)), 160);
        assert_eq!(timed_steps(16, ms(20)), 8);
        assert_eq!(timed_steps(1, ms(500)), 1);
        assert_eq!(
            timed_steps(1, Duration::ZERO),
            crate::gpu::MAX_STEPS_PER_LAUNCH
        );
        assert!(member_wins((10, ms(100)), (20, ms(179))));
        assert!(!member_wins((10, ms(100)), (20, ms(181))));
        assert!(!member_wins((10, ms(100)), (5, ms(50))));
    }

    #[test]
    fn a_timing_decides_on_long_or_clear_samples_and_grows_short_close_ones() {
        let ms = Duration::from_millis;
        let cap = crate::gpu::MAX_STEPS_PER_LAUNCH;
        // Both at least 10 ms: the margin of 0.9 decides.
        assert_eq!(judge((100, ms(20)), (100, ms(16))), Judge::Member);
        assert_eq!(judge((100, ms(20)), (100, ms(19))), Judge::Warp);
        assert_eq!(judge((100, ms(20)), (200, ms(35))), Judge::Member);
        // Short but clear: r at most 0.6 or at least 1.35.
        assert_eq!(
            judge((10, ms(5)), (10, Duration::from_micros(2900))),
            Judge::Member
        );
        assert_eq!(
            judge((10, ms(5)), (10, Duration::from_micros(3100))),
            Judge::Grow(33)
        );
        assert_eq!(
            judge((10, ms(4)), (10, Duration::from_micros(5500))),
            Judge::Warp
        );
        assert_eq!(
            judge((10, ms(4)), (10, Duration::from_micros(5300))),
            Judge::Grow(25)
        );
        // Short and close: grow so that the faster launch would take 10 ms.
        assert_eq!(judge((10, ms(4)), (10, ms(5))), Judge::Grow(25));
        assert_eq!(judge((10, ms(5)), (10, ms(4))), Judge::Grow(25));
        assert_eq!(judge((10, ms(2)), (10, ms(20))), Judge::Warp);
        // One long and one short launch still counts as short.
        assert_eq!(judge((10, ms(12)), (10, ms(9))), Judge::Grow(12));
        // At the cap, or when growing would not add a step, the pair decides as it is.
        assert_eq!(
            judge((cap, ms(1)), (cap, Duration::from_micros(800))),
            Judge::Member
        );
        assert_eq!(judge((cap, ms(1)), (cap, ms(1))), Judge::Warp);
    }

    #[test]
    fn unpermute_inverts_the_order() {
        use crate::job::{Label, Math, StackLayout, StackRan};
        let order = vec![2u32, 0, 1];
        let r = EnsembleResult {
            n: 1,
            members: 3,
            times: vec![0.0, 1.0],
            // Member i of the run holds 10 * order[i] and 10 * order[i] + 1.
            states: vec![20.0, 21.0, 0.0, 1.0, 10.0, 11.0],
            status: vec![Status::Bounds, Status::Ok, Status::NonFinite],
            t_reached: vec![0.5, 1.0, 0.25],
            accepted: vec![7, 8, 9],
            rejected: vec![1, 2, 3],
            label: Label {
                backend: "cuda".into(),
                device: String::new(),
                driver: String::new(),
                precision: Precision::Double,
                math: Math::Exact,
                stack: StackLayout::Auto,
                stack_ran: StackRan::Warp,
                sorted: true,
            },
        };
        let u = unpermute(r, &order);
        assert_eq!(u.states, vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        assert_eq!(
            u.status,
            vec![Status::Ok, Status::NonFinite, Status::Bounds]
        );
        assert_eq!(u.t_reached, vec![1.0, 0.25, 0.5]);
        assert_eq!(u.accepted, vec![8, 9, 7]);
        assert_eq!(u.rejected, vec![2, 3, 1]);
    }
}
