//! The flat buffers a kernel reads and writes, and how a finished run turns back into an
//! [`EnsembleResult`].
//!
//! Every backend (the C shim of the tests, CUDA, OpenCL, Metal) gets the same buffers in the
//! same layout, rounded to the scalar type of the run:
//!
//! * `rc`, `5 + 4 * ns` values: `t0`, the bound, `rtol`, `atol`, `hmax`, then the `ns` output
//!   times and, for RK4, the start, end and length of each segment. The slots of the method
//!   a job does not use are zero.
//! * `segm`, `ns` counts: steps of each RK4 segment, zero for the adaptive method.
//! * `x0`, `members * n` states and `par`, the parameters (one set shared by every member,
//!   or `members * n_p` values with `pstride = n_p`).
//! * `work`, `work_len(n)` values per member, `out`, `ns * n` values per member that start
//!   as NaN, and `ctl`, [`CTL`] words per member.
//!
//! Control words of a member: status, started, accepted, rejected, next sample, step in the
//! segment, reject flag and NaN-error flag.

use crate::job::{Checked, EnsembleError, EnsembleResult, Job, Label, Method, Params, Status};
use crate::kernel::{CTL, work_len};
use crate::real::Real;
use crate::tape::Tape;
use crate::traj::{Ctx, Kind};

/// Buffers of one run.
pub struct Packed<T: Real> {
    pub n: usize,
    pub members: usize,
    /// Number of output times.
    pub ns: usize,
    pub rc: Vec<T>,
    pub segm: Vec<u64>,
    pub x0: Vec<T>,
    pub par: Vec<T>,
    /// Distance in `par` between the parameters of two members: 0 when shared.
    pub pstride: usize,
    pub work: Vec<T>,
    pub out: Vec<T>,
    pub ctl: Vec<u32>,
    pub max_steps: u32,
    /// The output times as given in the job.
    pub times: Vec<f64>,
}

/// Rounds `job` to `T` and lays it out for a kernel. The job must have passed `validate`.
pub fn pack<T: Real>(job: &Job, checked: &Checked, tape: &Tape) -> Packed<T> {
    let ctx = Ctx::<T>::new(job, checked, tape);
    let (n, members, ns) = (checked.n, job.members, ctx.times.len());
    let zeros = |len: usize| vec![T::ZERO; len];

    let (rtol, atol, hmax, seg_a, seg_b, seg_h, segm) = match &ctx.kind {
        Kind::Rk4(segs) => (
            T::ZERO,
            T::ZERO,
            T::ZERO,
            segs.iter().map(|s| s.a).collect(),
            segs.iter().map(|s| s.b).collect(),
            segs.iter().map(|s| s.h).collect(),
            segs.iter().map(|s| s.m).collect(),
        ),
        Kind::Dp5(p) => (
            p.rtol,
            p.atol,
            p.hmax,
            zeros(ns),
            zeros(ns),
            zeros(ns),
            vec![0; ns],
        ),
    };
    let mut rc = vec![ctx.t0, ctx.bounds, rtol, atol, hmax];
    rc.extend_from_slice(&ctx.times);
    rc.extend(seg_a);
    rc.extend(seg_b);
    rc.extend(seg_h);
    debug_assert_eq!(rc.len(), 5 + 4 * ns);

    let (src, pstride) = match job.params {
        Params::Shared(p) => (p, 0),
        Params::PerMember(p) => (p, ctx.n_p),
    };
    let mut par: Vec<T> = src.iter().map(|&v| T::from_f64(v)).collect();
    if par.is_empty() {
        // A kernel argument must not be empty, even when the model has no parameters.
        par.push(T::ZERO);
    }

    let mut ctl = vec![0u32; members * CTL];
    for m in 0..members {
        ctl[m * CTL] = u32::from(Status::Running.code());
    }
    Packed {
        n,
        members,
        ns,
        rc,
        segm,
        x0: job.x0.iter().map(|&v| T::from_f64(v)).collect(),
        par,
        pstride,
        work: zeros(members * work_len(n)),
        out: vec![T::NAN; members * ns * n],
        ctl,
        max_steps: job.max_steps,
        times: checked.times.clone(),
    }
}

impl<T: Real> Packed<T> {
    /// True when no member is still running.
    pub fn done(&self) -> bool {
        let running = u32::from(Status::Running.code());
        (0..self.members).all(|m| self.ctl[m * CTL] != running)
    }

    /// Turns the buffers of a finished run into a result.
    pub fn finish(self, label: Label) -> Result<EnsembleResult, EnsembleError> {
        let wl = work_len(self.n);
        let t_end = *self.times.last().expect("a job has at least one sample");
        let mut status = Vec::with_capacity(self.members);
        for m in 0..self.members {
            let code = self.ctl[m * CTL];
            match Status::from_code(code) {
                Some(Status::Running) | None => {
                    return Err(EnsembleError::BadJob(format!(
                        "member {m} ended with status code {code}, which no finished run has"
                    )));
                }
                Some(s) => status.push(s),
            }
        }
        let t_reached = (0..self.members)
            .map(|m| {
                if status[m] == Status::Ok {
                    t_end
                } else {
                    self.work[m * wl + 10 * self.n].to_f64()
                }
            })
            .collect();
        Ok(EnsembleResult {
            n: self.n,
            members: self.members,
            states: self.out.iter().map(|v| v.to_f64()).collect(),
            accepted: (0..self.members).map(|m| self.ctl[m * CTL + 2]).collect(),
            rejected: (0..self.members).map(|m| self.ctl[m * CTL + 3]).collect(),
            status,
            t_reached,
            times: self.times,
            label,
        })
    }
}

/// The integrator a job asks for.
pub fn integrator_of(method: Method) -> crate::kernel::Integrator {
    match method {
        Method::Rk4 { .. } => crate::kernel::Integrator::Rk4,
        Method::Dp5 { .. } => crate::kernel::Integrator::Dp5,
    }
}
