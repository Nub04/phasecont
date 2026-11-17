//! The flat buffers a kernel reads and writes, and how a finished run turns back into an
//! [`EnsembleResult`].
//!
//! Every backend (the C shim of the tests, CUDA, OpenCL, Metal) gets the same buffers in the
//! same layout, rounded to the scalar type of the run:
//!
//! * `rc`, `5 + 4 * ns` values: `t0`, the bound, `rtol`, `atol`, `hmax`, then the `ns` output
//!   times and, for RK4 and the iteration of a map, the start, end and length of each
//!   segment (the length is 1 for a map). The slots of the method
//!   a job does not use are zero. The bound is capped at the largest finite value of the
//!   scalar type, so an infinite bound reads as that value and `|y| <= bound` fails for NaN and
//!   for infinity alike.
//! * `segm`, `ns` counts: steps of each RK4 segment or iterates of each map segment, zero for
//!   the adaptive method.
//! * `x0`, `members * n` states and `par`, the parameters (one set shared by every member,
//!   or `members * n_p` values with `pstride = n_p`).
//! * `work`, `work_len(n, integrator)` values per member (more under the global scratch
//!   layout, see below), `out`, `ns * n` values per member
//!   that start as NaN, and `ctl`, [`CTL`] words per member.
//!
//! Control words of a member: status, started, accepted, rejected, next sample, step in the
//! segment, reject flag and NaN-error flag.
//!
//! `work` holds only what a launch hands to the next one. The scratch of a step (the trial
//! state and the stage vectors) is private to the thread and does not appear in it, except
//! under [`Layout::Global`], where the same scratch follows those words in the member's
//! slice (`Layout::work_len` values) and the words a launch hands on stay where they are. The
//! member's time `t` is the first value in both layouts, then for RK4 and a map the state `y`;
//! for DP5
//! the step size `h`, the step error memory `facold`, the state `y` and the first stage `k0`
//! (the last stage of the previous step).

use crate::job::{Checked, EnsembleError, EnsembleResult, Job, Label, Method, Params, Status};
use crate::kernel::{CTL, Integrator, Layout};
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
    /// Values per member in `work`: [`work_len`] of the integrator the run uses, or
    /// [`Layout::work_len`] when the scratch is global.
    pub work_len: usize,
    pub out: Vec<T>,
    pub ctl: Vec<u32>,
    pub max_steps: u32,
    /// The output times as given in the job.
    pub times: Vec<f64>,
}

/// Rounds `job` to `T` and lays it out for a kernel with private scratch. The job must have
/// passed `validate`.
pub fn pack<T: Real>(job: &Job, checked: &Checked, tape: &Tape) -> Packed<T> {
    pack_in(job, checked, tape, Layout::Private)
}

/// [`pack`] with the `work` buffer sized for the scratch layout the kernel was built with.
pub fn pack_in<T: Real>(job: &Job, checked: &Checked, tape: &Tape, layout: Layout) -> Packed<T> {
    let ctx = Ctx::<T>::new(job, checked, tape);
    let (n, members, ns) = (checked.n, job.members, ctx.times.len());
    let zeros = |len: usize| vec![T::ZERO; len];

    let (rtol, atol, hmax, seg_a, seg_b, seg_h, segm) = match &ctx.kind {
        Kind::Rk4(segs) | Kind::Iter(segs) => (
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
    let mut rc = vec![ctx.t0, ctx.lim, rtol, atol, hmax];
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

    let wl = layout.work_len(n, integrator_of(job.method));
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
        work: zeros(members * wl),
        work_len: wl,
        out: vec![T::NAN; members * ns * n],
        ctl,
        max_steps: job.max_steps,
        times: checked.times.clone(),
    }
}

/// True when no member of the first `members` is still running in `ctl`.
pub fn ctl_done(ctl: &[u32], members: usize) -> bool {
    let running = u32::from(Status::Running.code());
    (0..members).all(|m| ctl[m * CTL] != running)
}

impl<T: Real> Packed<T> {
    /// True when no member is still running.
    pub fn done(&self) -> bool {
        ctl_done(&self.ctl, self.members)
    }

    /// True when [`Packed::finish`] reads the `work` buffer: some member did not end `Ok`.
    ///
    /// An `Ok` member reached `t1`, and `finish` reports the last output time for it without
    /// reading `work`, so a run where every member is `Ok` can skip the download of `work`.
    pub fn needs_work(&self) -> bool {
        let ok = u32::from(Status::Ok.code());
        (0..self.members).any(|m| self.ctl[m * CTL] != ok)
    }

    /// Turns the buffers of a finished run into a result.
    pub fn finish(self, label: Label) -> Result<EnsembleResult, EnsembleError> {
        let wl = self.work_len;
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
                    self.work[m * wl].to_f64()
                }
            })
            .collect();
        Ok(EnsembleResult {
            n: self.n,
            members: self.members,
            states: T::into_f64_vec(self.out),
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
pub fn integrator_of(method: Method) -> Integrator {
    match method {
        Method::Rk4 { .. } => Integrator::Rk4,
        Method::Dp5 { .. } => Integrator::Dp5,
        Method::Iter => Integrator::Iter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Precision;
    use crate::kernel::work_len;

    /// Two members of one state, two output times, `ctl` and `work` as a device would leave them.
    fn finished(codes: [Status; 2], t_reached: [f64; 2]) -> Packed<f64> {
        let (n, members) = (1, 2);
        let wl = work_len(n, Integrator::Rk4);
        let mut ctl = vec![0u32; members * CTL];
        let mut work = vec![0.0; members * wl];
        for m in 0..members {
            ctl[m * CTL] = u32::from(codes[m].code());
            work[m * wl] = t_reached[m];
        }
        Packed {
            n,
            members,
            ns: 2,
            rc: Vec::new(),
            segm: Vec::new(),
            x0: vec![0.0; members * n],
            par: vec![0.0],
            pstride: 0,
            work,
            work_len: wl,
            out: vec![1.0; members * 2 * n],
            ctl,
            max_steps: 100,
            times: vec![0.5, 2.0],
        }
    }

    fn label() -> Label {
        Label {
            backend: "cpu".into(),
            device: "test".into(),
            driver: String::new(),
            precision: Precision::Double,
            math: crate::job::Math::Exact,
        }
    }

    #[test]
    fn the_work_buffer_holds_only_what_a_launch_hands_on() {
        // RK4: t and y. DP5: t, h, facold, y and the first stage.
        assert_eq!(work_len(2, Integrator::Rk4), 3);
        assert_eq!(work_len(2, Integrator::Dp5), 7);
        assert_eq!(work_len(40, Integrator::Rk4), 41);
        assert_eq!(work_len(40, Integrator::Dp5), 83);
    }

    #[test]
    fn a_run_of_ok_members_needs_no_work_buffer() {
        let mut p = finished([Status::Ok, Status::Ok], [2.0, 2.0]);
        assert!(!p.needs_work());
        // `finish` must not read `work` then: poison it and compare with the clean result.
        let clean = finished([Status::Ok, Status::Ok], [2.0, 2.0])
            .finish(label())
            .unwrap();
        p.work.iter_mut().for_each(|w| *w = f64::NAN);
        let r = p.finish(label()).unwrap();
        assert_eq!(r.t_reached, clean.t_reached);
        assert!(r.t_reached.iter().all(|&t| t.to_bits() == 2.0f64.to_bits()));
        assert_eq!(r.states, clean.states);
    }

    #[test]
    fn a_member_that_stopped_early_needs_the_work_buffer() {
        for bad in [
            Status::Bounds,
            Status::StepTooSmall,
            Status::MaxSteps,
            Status::NonFinite,
        ] {
            let p = finished([Status::Ok, bad], [2.0, 0.75]);
            assert!(p.needs_work(), "{bad:?}");
            let r = p.finish(label()).unwrap();
            assert_eq!(r.t_reached, vec![2.0, 0.75], "{bad:?}");
        }
    }

    #[test]
    fn a_member_still_running_needs_the_work_buffer() {
        let mut p = finished([Status::Ok, Status::Ok], [2.0, 2.0]);
        p.ctl[CTL] = u32::from(Status::Running.code());
        assert!(p.needs_work());
    }
}
