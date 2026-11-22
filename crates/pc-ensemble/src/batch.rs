//! Splits a job into member batches that fit the device memory, and joins the results.
//!
//! A GPU run holds a few buffers per member (`x0`, the parameters when they differ per
//! member, `work`, `out` and `ctl`) and a few that do not depend on the member count. The
//! `work` buffer is the large one: with the global scratch layout it takes `12 n + 3` reals
//! per member for Dormand-Prince. [`plan_batches`] turns a memory budget into the most members
//! one batch can hold, [`run_batched`] runs the job in that many members at a time through the
//! backend's single-run code and joins the results in member order. A member does not depend
//! on the other members, so the joined result has the bits of one run.
//!
//! A kernel with calls also takes a stack of 4,096 frames per member (SPEC section 8af-5). It
//! is a member buffer like the others, outside the 256 KiB rule of the scratch layout, so a
//! deep model runs in smaller batches, and a member whose stack does not fit at all is a
//! [`EnsembleError::GpuLaunch`] that names the function and its frame.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::job::{EnsembleError, EnsembleResult, Job, Params};
use crate::kernel::{CTL, Stack};

/// The bytes a run takes on the device: [`Costs::per_member`] for each member and
/// [`Costs::fixed`] whatever the member count is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Costs {
    /// Bytes one member takes in each buffer that grows with the member count.
    pub per_member: [(&'static str, u64); 6],
    /// Bytes of the buffers that do not (`rc`, `segm` and a shared parameter set).
    pub fixed: u64,
    /// The stack of a kernel with calls, for the error when it does not fit.
    pub stack: Option<StackCost>,
}

/// The stack of one member in bytes: `frames` frames of `frame_bytes` for `function`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackCost {
    pub function: String,
    pub frame_bytes: u64,
    pub frames: u64,
}

impl StackCost {
    fn text(&self) -> String {
        format!(
            "the recursive function {} needs a stack of {} frames of {} bytes, {} bytes for \
             one member",
            self.function,
            self.frames,
            self.frame_bytes,
            self.frames.saturating_mul(self.frame_bytes)
        )
    }
}

impl Costs {
    /// The costs of a run of `n` states and `n_p` parameters with `ns` output times and
    /// `work_len` values of `work` per member, in a scalar of `real_bytes` bytes.
    /// `per_member_params` is true when every member has its own parameter set, and `stack`
    /// is the stack of a kernel with calls.
    pub fn new(
        n: usize,
        n_p: usize,
        per_member_params: bool,
        ns: usize,
        work_len: usize,
        stack: Option<&Stack>,
        real_bytes: usize,
    ) -> Costs {
        let t = real_bytes as u64;
        let reals = |count: usize| (count as u64).saturating_mul(t);
        let par = reals(n_p);
        Costs {
            per_member: [
                ("x0", reals(n)),
                ("par", if per_member_params { par } else { 0 }),
                ("work", reals(work_len)),
                ("out", reals(ns.saturating_mul(n))),
                ("ctl", (CTL as u64) * 4),
                ("stack", stack.map_or(0, |s| reals(s.reals()))),
            ],
            fixed: reals(ns.saturating_mul(4).saturating_add(5))
                .saturating_add((ns as u64).saturating_mul(8))
                .saturating_add(if per_member_params { 0 } else { par }),
            stack: stack.map(|s| StackCost {
                function: s.function.clone(),
                frame_bytes: reals(s.frame_reals),
                frames: s.frames as u64,
            }),
        }
    }

    /// The bytes of one member over all buffers.
    pub fn member(&self) -> u64 {
        self.per_member
            .iter()
            .fold(0, |sum, (_, b)| sum.saturating_add(*b))
    }
}

/// The most members one batch can hold.
///
/// A batch takes `fixed` bytes plus [`Costs::member`] bytes per member, and that has to fit in
/// `budget`. No single buffer of a batch may exceed `max_alloc` either. A `budget` or a
/// `max_alloc` of 0 turns that limit off. The result is at least 1 and at most `members`. A
/// member that does not fit is a [`EnsembleError::BadJob`] that names the limit, or a
/// [`EnsembleError::GpuLaunch`] that names the function when the member has a stack.
pub fn plan_batches(
    members: usize,
    costs: &Costs,
    budget: u64,
    max_alloc: u64,
) -> Result<usize, EnsembleError> {
    let mut most = members.max(1) as u64;
    let each = costs.member();
    if budget > 0 && each > 0 {
        let room = budget.saturating_sub(costs.fixed);
        let fits = room / each;
        if fits == 0 {
            let need = format!(
                "one member needs {each} bytes on the device, and {} bytes are left of a budget of \
                 {budget} bytes after {} bytes that do not depend on the members",
                room, costs.fixed
            );
            return Err(match &costs.stack {
                Some(s) => EnsembleError::GpuLaunch(format!("{}: {need}", s.text())),
                None => EnsembleError::BadJob(need),
            });
        }
        most = most.min(fits);
    }
    if max_alloc > 0 {
        for &(name, bytes) in &costs.per_member {
            if bytes == 0 {
                continue;
            }
            let fits = max_alloc / bytes;
            if fits == 0 {
                let need = format!(
                    "the {name} buffer needs {bytes} bytes for one member and the device allows \
                     {max_alloc} bytes in one buffer"
                );
                return Err(match (&costs.stack, name) {
                    (Some(s), "stack") => EnsembleError::GpuLaunch(format!("{}: {need}", s.text())),
                    _ => EnsembleError::BadJob(need),
                });
            }
            most = most.min(fits);
        }
    }
    Ok(most as usize)
}

/// [`plan_batches`] for a stack layout that pads the stack buffer of a batch to whole groups
/// of `group` members ([`StackLayout::buffer_reals`](crate::job::StackLayout::buffer_reals)).
/// The padding of up to `group - 1` stacks comes out of the budget, and the padded buffer must
/// fit in one allocation. A `group` of 1 is plain [`plan_batches`].
fn plan_grouped(
    members: usize,
    mut costs: Costs,
    budget: u64,
    max_alloc: u64,
    group: u64,
) -> Result<usize, EnsembleError> {
    let group = group.max(1);
    let stack_bytes = costs.per_member[5].1;
    costs.fixed = costs
        .fixed
        .saturating_add((group - 1).saturating_mul(stack_bytes));
    let planned = plan_batches(members, &costs, budget, max_alloc)?;
    if group == 1 || stack_bytes == 0 || max_alloc == 0 {
        return Ok(planned);
    }
    // A batch of k members fits in one allocation when k, rounded up to a whole group, does.
    let whole = (max_alloc / stack_bytes) / group * group;
    if whole == 0 {
        return Err(EnsembleError::GpuLaunch(format!(
            "{}: the warp stack layout needs {} bytes for a group of {group} members and the \
             device allows {max_alloc} bytes in one buffer; use stack member",
            costs.stack.as_ref().map_or(String::new(), StackCost::text),
            stack_bytes.saturating_mul(group)
        )));
    }
    Ok(planned.min(usize::try_from(whole).unwrap_or(usize::MAX)))
}

/// The members per batch for a GPU run, from the memory the backend can use (`budget`, from
/// its free-memory query or a fraction of its total) and the largest buffer it allows.
/// [`force_members`] lowers the result.
pub fn members_per_batch(
    job: &Job,
    ns: usize,
    work_len: usize,
    stack: Option<&Stack>,
    real_bytes: usize,
    budget: Option<u64>,
    max_alloc: u64,
) -> Result<usize, EnsembleError> {
    let costs = Costs::new(
        job.model.n(),
        job.model.n_p(),
        matches!(job.params, Params::PerMember(_)),
        ns,
        work_len,
        stack,
        real_bytes,
    );
    let planned = plan_grouped(
        job.members,
        costs,
        budget.unwrap_or(0),
        max_alloc,
        job.stack.group() as u64,
    )?;
    Ok(match forced_members() {
        Some(k) => planned.min(k),
        None => planned,
    })
}

/// The budget of a backend that cannot ask for its free memory (OpenCL and Metal): half of the
/// device memory. Another process or the display may hold part of it, so the fraction is a
/// margin and not a tuned value.
pub fn half_budget(memory: u64) -> u64 {
    memory / 2
}

/// The cap a test sets on the members per batch: 0 for none.
static FORCED: AtomicUsize = AtomicUsize::new(0);

/// Caps the members per batch of every GPU run in this process, or lifts the cap with `None`
/// (or 0). The device tests use it to run one job whole and in small batches. It is not part
/// of the API.
#[doc(hidden)]
pub fn force_members(cap: Option<usize>) {
    FORCED.store(cap.unwrap_or(0), Ordering::SeqCst);
}

fn forced_members() -> Option<usize> {
    match FORCED.load(Ordering::SeqCst) {
        0 => None,
        k => Some(k),
    }
}

/// Cuts `members` into the fewest ranges of at most `most` members. The sizes differ by at
/// most one and the larger ones come first.
pub fn batch_ranges(members: usize, most: usize) -> Vec<Range<usize>> {
    let most = most.max(1);
    let count = members.div_ceil(most).max(1);
    let (base, extra) = (members / count, members % count);
    let mut at = 0;
    (0..count)
        .map(|i| {
            let len = base + usize::from(i < extra);
            let range = at..at + len;
            at += len;
            range
        })
        .collect()
}

fn bad(m: String) -> EnsembleError {
    EnsembleError::BadJob(m)
}

/// Runs `job` in batches of at most `size` members and joins the results.
///
/// `run_one` runs one sub-job, which has the same model, times, method and precision as
/// `job` and a slice of its members. A job of `size` members or fewer goes to `run_one` as it
/// is. The cancel flag is read before each batch, and a set flag gives
/// [`EnsembleError::Cancelled`]. The first error ends the run. The label of the result is the
/// one of the first batch, with `stack_ran` joined over every batch.
pub fn run_batched(
    job: &Job,
    size: usize,
    cancel: &AtomicBool,
    mut run_one: impl FnMut(&Job) -> Result<EnsembleResult, EnsembleError>,
) -> Result<EnsembleResult, EnsembleError> {
    if size == 0 {
        return Err(bad("the batch size must be at least 1".into()));
    }
    if size >= job.members {
        return run_one(job);
    }
    let (n, n_p) = (job.model.n(), job.model.n_p());
    if job.x0.len() != job.members.saturating_mul(n) {
        return Err(bad(format!(
            "x0 has {} values, expected {} members x {n} states",
            job.x0.len(),
            job.members
        )));
    }
    if let Params::PerMember(p) = job.params
        && p.len() != job.members.saturating_mul(n_p)
    {
        return Err(bad(format!(
            "params has {} values, expected {} members x {n_p} parameters",
            p.len(),
            job.members
        )));
    }
    let mut sub = job.clone();
    let mut joined: Option<EnsembleResult> = None;
    for r in batch_ranges(job.members, size) {
        if cancel.load(Ordering::Relaxed) {
            return Err(EnsembleError::Cancelled);
        }
        sub.members = r.len();
        sub.x0 = &job.x0[r.start * n..r.end * n];
        if let Params::PerMember(p) = job.params {
            sub.params = Params::PerMember(&p[r.start * n_p..r.end * n_p]);
        }
        let part = run_one(&sub)?;
        if part.members != r.len() || part.n != n {
            return Err(bad(format!(
                "a batch of {} members of {n} states came back with {} members of {} states",
                r.len(),
                part.members,
                part.n
            )));
        }
        match &mut joined {
            None => {
                let whole = |len: usize| job.members.saturating_mul(len);
                let mut all = EnsembleResult {
                    members: job.members,
                    states: Vec::with_capacity(whole(part.times.len() * n)),
                    status: Vec::with_capacity(job.members),
                    t_reached: Vec::with_capacity(job.members),
                    accepted: Vec::with_capacity(job.members),
                    rejected: Vec::with_capacity(job.members),
                    ..part.clone_head()
                };
                all.append(part);
                joined = Some(all);
            }
            Some(all) => all.append(part),
        }
    }
    joined.ok_or_else(|| bad("the job has no members".into()))
}

impl EnsembleResult {
    /// A copy with no member data, for the first batch of a join.
    fn clone_head(&self) -> EnsembleResult {
        EnsembleResult {
            n: self.n,
            members: 0,
            times: self.times.clone(),
            states: Vec::new(),
            status: Vec::new(),
            t_reached: Vec::new(),
            accepted: Vec::new(),
            rejected: Vec::new(),
            label: self.label.clone(),
        }
    }

    /// Adds the members of `part` after the members already here.
    fn append(&mut self, part: EnsembleResult) {
        // `auto` chooses the layout per batch, so the layouts that ran add up.
        self.label.stack_ran = self.label.stack_ran.join(part.label.stack_ran);
        self.states.extend(part.states);
        self.status.extend(part.status);
        self.t_reached.extend(part.t_reached);
        self.accepted.extend(part.accepted);
        self.rejected.extend(part.rejected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn costs(per: [u64; 5], fixed: u64) -> Costs {
        Costs {
            per_member: [
                ("x0", per[0]),
                ("par", per[1]),
                ("work", per[2]),
                ("out", per[3]),
                ("ctl", per[4]),
                ("stack", 0),
            ],
            fixed,
            stack: None,
        }
    }

    /// The costs above plus a stack of `frames` frames of `frame` bytes for `fib`.
    fn with_stack(per: [u64; 5], fixed: u64, frames: u64, frame: u64) -> Costs {
        let mut c = costs(per, fixed);
        c.per_member[5] = ("stack", frames * frame);
        c.stack = Some(StackCost {
            function: "fib".into(),
            frame_bytes: frame,
            frames,
        });
        c
    }

    fn launch_text(e: EnsembleError) -> String {
        match e {
            EnsembleError::GpuLaunch(t) => t,
            other => panic!("expected GpuLaunch, got {other:?}"),
        }
    }

    fn text(e: EnsembleError) -> String {
        match e {
            EnsembleError::BadJob(t) => t,
            other => panic!("expected BadJob, got {other:?}"),
        }
    }

    #[test]
    fn a_group_of_one_plans_like_plan_batches() {
        let c = with_stack([16, 0, 100, 0, 0], 50, 4096, 24);
        for (budget, max_alloc) in [(0, 0), (1 << 26, 0), (1 << 30, 1 << 22)] {
            assert_eq!(
                plan_grouped(1000, c.clone(), budget, max_alloc, 1),
                plan_batches(1000, &c, budget, max_alloc)
            );
        }
    }

    #[test]
    fn the_warp_padding_comes_out_of_the_budget() {
        // 116 + 4096 * 24 bytes a member; the padding of 31 stacks is fixed.
        let stack = 4096 * 24;
        let each = 116 + stack;
        let c = with_stack([16, 0, 100, 0, 0], 50, 4096, 24);
        let budget = 50 + 31 * stack + 40 * each;
        assert_eq!(plan_grouped(1000, c.clone(), budget, 0, 32), Ok(40));
        assert_eq!(plan_grouped(1000, c.clone(), budget - 1, 0, 32), Ok(39));
        // The plain layout gets the padding as members.
        assert!(plan_grouped(1000, c, budget, 0, 1).unwrap() > 40);
    }

    #[test]
    fn the_padded_warp_stack_fits_one_allocation() {
        let stack: u64 = 4096 * 24;
        let c = with_stack([16, 0, 100, 0, 0], 0, 4096, 24);
        // 70 stacks fit in one buffer, so 64 members (two whole groups) do when padded.
        assert_eq!(plan_grouped(1000, c.clone(), 0, 70 * stack, 32), Ok(64));
        assert_eq!(plan_grouped(1000, c.clone(), 0, 70 * stack, 1), Ok(70));
        // A batch of fewer members than one group still pads to a whole group.
        assert_eq!(plan_grouped(10, c.clone(), 0, 32 * stack, 32), Ok(10));
        let e = launch_text(plan_grouped(1000, c, 0, 31 * stack, 32).unwrap_err());
        assert!(e.contains("warp stack layout"), "{e}");
        assert!(e.contains("use stack member"), "{e}");
        assert!(e.contains("fib"), "{e}");
    }

    #[test]
    fn a_run_that_fits_is_one_batch() {
        let c = costs([16, 0, 100, 0, 0], 50);
        assert_eq!(plan_batches(100, &c, 1_000_000, 0), Ok(100));
        assert_eq!(plan_batches(100, &c, 0, 0), Ok(100));
    }

    #[test]
    fn the_budget_after_the_fixed_buffers_sets_the_size() {
        // 116 bytes a member, 50 fixed: (1050 - 50) / 116 = 8.
        let c = costs([16, 0, 100, 0, 0], 50);
        assert_eq!(plan_batches(100, &c, 1050, 0), Ok(8));
        // One byte short of nine members.
        assert_eq!(plan_batches(100, &c, 50 + 9 * 116 - 1, 0), Ok(8));
        assert_eq!(plan_batches(100, &c, 50 + 9 * 116, 0), Ok(9));
    }

    #[test]
    fn no_buffer_of_a_batch_exceeds_the_largest_allocation() {
        let c = costs([16, 0, 100, 0, 0], 0);
        assert_eq!(plan_batches(100, &c, 0, 1000), Ok(10));
        // The budget allows 20 and the largest buffer 10.
        assert_eq!(plan_batches(100, &c, 20 * 116, 1000), Ok(10));
        // The largest buffer allows 62 and the budget 5.
        assert_eq!(plan_batches(100, &c, 5 * 116, 1000), Ok(5));
    }

    #[test]
    fn one_member_that_does_not_fit_is_a_bad_job_that_names_the_limit() {
        let c = costs([16, 0, 100, 0, 0], 50);
        let t = text(plan_batches(100, &c, 150, 0).unwrap_err());
        assert!(t.contains("one member needs 116"), "{t}");
        assert!(t.contains("150"), "{t}");
        // The fixed buffers alone are over the budget.
        let t = text(plan_batches(100, &c, 40, 0).unwrap_err());
        assert!(t.contains("one member"), "{t}");
        let big = costs([16, 0, 2000, 0, 0], 0);
        let t = text(plan_batches(100, &big, 0, 1000).unwrap_err());
        assert!(
            t.contains("work") && t.contains("1000") && t.contains("2000"),
            "{t}"
        );
    }

    #[test]
    fn sizes_at_the_edge_of_u64_do_not_overflow() {
        let c = costs([u64::MAX, u64::MAX, u64::MAX, 0, 0], u64::MAX);
        let s = with_stack([1, 0, 0, 0, 0], 0, u64::MAX, 1);
        assert_eq!(plan_batches(10, &s, u64::MAX, 0), Ok(1));
        assert!(plan_batches(10, &c, u64::MAX, 0).is_err());
        assert!(plan_batches(10, &c, 0, u64::MAX).is_ok());
        assert_eq!(plan_batches(0, &c, 0, 0), Ok(1));
    }

    #[test]
    fn costs_follow_the_buffers_of_a_run() {
        // 2 states, 1 parameter, 3 output times, 5 work values, 8-byte reals.
        let own = Costs::new(2, 1, true, 3, 5, None, 8);
        assert_eq!(
            own.per_member.map(|(_, b)| b),
            [16, 8, 40, 48, (CTL as u64) * 4, 0]
        );
        assert_eq!(own.fixed, (5 + 12) * 8 + 3 * 8);
        let shared = Costs::new(2, 1, false, 3, 5, None, 8);
        assert_eq!(shared.per_member[1], ("par", 0));
        assert_eq!(shared.fixed, own.fixed + 8);
        assert_eq!(own.member(), 16 + 8 + 40 + 48 + (CTL as u64) * 4);
    }

    #[test]
    fn the_stack_is_a_member_buffer_that_shrinks_the_batch() {
        // 116 bytes a member without the stack, 4096 frames of 24 bytes with it.
        let c = with_stack([16, 0, 100, 0, 0], 50, 4096, 24);
        assert_eq!(c.member(), 116 + 4096 * 24);
        assert_eq!(plan_batches(100, &c, 50 + 7 * c.member(), 0), Ok(7));
        assert_eq!(plan_batches(100, &c, 50 + 7 * c.member() - 1, 0), Ok(6));
        // The largest buffer bounds the stack buffer as it bounds the others.
        assert_eq!(plan_batches(100, &c, 0, 3 * 4096 * 24), Ok(3));
    }

    #[test]
    fn a_stack_that_does_not_fit_one_member_is_a_launch_error_that_names_the_function() {
        let c = with_stack([16, 0, 100, 0, 0], 50, 4096, 24);
        let t = launch_text(plan_batches(100, &c, 50 + 1000, 0).unwrap_err());
        for want in ["fib", "4096 frames", "24 bytes", "98304", "1050"] {
            assert!(t.contains(want), "{want}: {t}");
        }
        let t = launch_text(plan_batches(100, &c, 0, 4096 * 24 - 1).unwrap_err());
        for want in ["fib", "24 bytes", "98303"] {
            assert!(t.contains(want), "{want}: {t}");
        }
        // A member too big for its other buffers is still a bad job.
        let big = with_stack([16, 0, 2000, 0, 0], 0, 4096, 24);
        assert!(matches!(
            plan_batches(100, &big, 0, 1000),
            Err(EnsembleError::BadJob(_))
        ));
    }

    #[test]
    fn costs_hold_the_stack_of_a_kernel_with_calls() {
        let stack = Stack {
            function: "fib".into(),
            frame_reals: 3,
            frames: 4096,
        };
        let c = Costs::new(2, 1, true, 3, 5, Some(&stack), 8);
        assert_eq!(c.per_member[5], ("stack", 4096 * 3 * 8));
        assert_eq!(
            c.stack,
            Some(StackCost {
                function: "fib".into(),
                frame_bytes: 24,
                frames: 4096
            })
        );
        let plain = Costs::new(2, 1, true, 3, 5, None, 8);
        assert_eq!(plain.per_member[5], ("stack", 0));
        assert_eq!(plain.stack, None);
        assert_eq!(c.member(), plain.member() + 4096 * 24);
    }

    #[test]
    fn a_device_without_a_free_memory_query_gets_half_of_its_memory() {
        assert_eq!(half_budget(0), 0);
        assert_eq!(half_budget(7), 3);
        assert_eq!(half_budget(16 << 30), 8 << 30);
    }

    #[test]
    fn batch_ranges_cover_the_members_in_even_parts() {
        assert_eq!(batch_ranges(10, 4), [0..4, 4..7, 7..10]);
        assert_eq!(batch_ranges(10, 6), [0..5, 5..10]);
        for (members, most) in [(7, 7), (7, 100), (1, 5)] {
            assert_eq!(batch_ranges(members, most), vec![0..members]);
        }
        assert_eq!(batch_ranges(3, 1), [0..1, 1..2, 2..3]);
        for members in 1..60 {
            for most in 1..70 {
                let r = batch_ranges(members, most);
                assert_eq!(r.len(), members.div_ceil(most), "{members} {most}");
                assert_eq!(r[0].start, 0);
                assert_eq!(r.last().unwrap().end, members);
                for w in r.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                    assert!(w[0].len() >= w[1].len() && w[0].len() - w[1].len() <= 1);
                }
                assert!(r.iter().all(|x| x.len() <= most && !x.is_empty()));
            }
        }
    }
}
