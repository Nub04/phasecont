//! One trajectory as a state machine that can stop and resume between steps.
//!
//! Everything a member needs between two step attempts lives in [`Traj`] and in its slice of
//! the workspace, so a run can be cut into launches of any length and give the same bits.
//! A GPU kernel keeps the same state in registers and global memory and relaunches the same
//! way.
//!
//! Fixed step RK4 follows the textbook scheme with the operation order written out in
//! [`Traj::rk4`]. The adaptive method is the Dormand-Prince 5(4) pair of
//! `pc_engine::integrate` (same tableau, controller and starting step, from Hairer, Norsett
//! and Wanner, Solving Ordinary Differential Equations I, II.4 and II.5), generalized in two
//! ways: it works in `f32` as well as `f64`, and it shortens a step to land on each output
//! time.
//!
//! Workspace of a member, `10 n + n_p + 1` values in this order: `y`, `y1`, `yt`, the seven
//! stages `k0` to `k6`, and the parameters followed by the time.

use std::sync::atomic::AtomicBool;

use crate::job::{Checked, Job, Method, NativeRhs, Params, Status};
use crate::real::Real;
use crate::tape::{CpuTape, Slab, Tape};

/// Nodes c2 to c7 of the pair.
const C: [f64; 6] = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// Rows a21 to a76. The last row is the fifth order solution, which makes the pair FSAL.
const A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// e_i = b_i - bhat_i for the seven stages: the error estimate is h sum e_i k_i.
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

const SAFE: f64 = 0.9;
const FAC1: f64 = 0.2;
const FAC2: f64 = 10.0;
const BETA: f64 = 0.04;

/// All the constants of the integrators rounded to the run's scalar type.
#[derive(Clone, Debug)]
pub struct Coef<T: Real> {
    pub c: [T; 6],
    pub a: [[T; 6]; 6],
    pub e: [T; 7],
    pub safe: T,
    pub inv_fac1: T,
    pub inv_fac2: T,
    pub beta: T,
    pub expo1: T,
    pub facold0: T,
    pub facold_min: T,
    pub uround: T,
    pub tenth: T,
    pub c101: T,
    pub c1e10: T,
    pub c1e6: T,
    pub c1e3: T,
    pub c1e15: T,
    pub c001: T,
    pub c100: T,
    pub fifth: T,
    pub half: T,
    pub two: T,
    pub six: T,
}

impl<T: Real> Coef<T> {
    pub fn new() -> Self {
        let r = T::from_f64;
        Coef {
            c: C.map(r),
            a: A.map(|row| row.map(r)),
            e: E.map(r),
            safe: r(SAFE),
            inv_fac1: r(1.0 / FAC1),
            inv_fac2: r(1.0 / FAC2),
            beta: r(BETA),
            expo1: r(0.2 - BETA * 0.75),
            facold0: r(1e-4),
            facold_min: r(1e-4),
            uround: r(T::UROUND_F64),
            tenth: r(0.1),
            c101: r(1.01),
            c1e10: r(1e-10),
            c1e6: r(1e-6),
            c1e3: r(1e-3),
            c1e15: r(1e-15),
            c001: r(0.01),
            c100: r(100.0),
            fifth: r(1.0 / 5.0),
            half: r(0.5),
            two: r(2.0),
            six: r(6.0),
        }
    }
}

impl<T: Real> Default for Coef<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// An RK4 segment: `m` equal steps of length `h` from `a` to `b`. `half = h * 0.5` and
/// `h6 = h / 6` are computed once here, with the same operations a step would use.
#[derive(Clone, Copy, Debug)]
pub struct Seg<T: Real> {
    pub a: T,
    pub b: T,
    pub h: T,
    pub half: T,
    pub h6: T,
    pub m: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Dp5Par<T: Real> {
    pub rtol: T,
    pub atol: T,
    pub hmax: T,
}

#[derive(Clone, Debug)]
pub enum Kind<T: Real> {
    Rk4(Vec<Seg<T>>),
    Dp5(Dp5Par<T>),
    /// One segment per sample interval with `m = b - a` iterates and `h = 1`. `half` and
    /// `h6` are zero and unused.
    Iter(Vec<Seg<T>>),
}

/// What every member of a run shares.
pub struct Ctx<'a, T: Real> {
    /// The right-hand side, lowered once for `T`.
    pub rhs: CpuTape<T>,
    /// A compiled right-hand side, set only for a job that carries one of precision `T`.
    pub native: Option<NativeRhs>,
    pub n: usize,
    pub n_p: usize,
    pub x0: &'a [f64],
    pub params: Params<'a>,
    pub t0: T,
    pub bounds: T,
    /// `bounds` capped at the largest finite value. A state passes the step check when every
    /// `|y| <= lim`, and that test is false for NaN and infinity even when `bounds` is
    /// infinite, so one pass catches both failures.
    pub lim: T,
    pub max_steps: u32,
    /// The output times rounded to `T`.
    pub times: Vec<T>,
    pub kind: Kind<T>,
    pub co: Coef<T>,
    /// The cancel flag of the run, read inside a long evaluation of a tape with calls. None
    /// from [`Ctx::new`].
    pub cancel: Option<&'a AtomicBool>,
}

impl<'a, T: Real> Ctx<'a, T> {
    /// Rounds the job to `T`. The job must have passed `validate`.
    pub fn new(job: &Job<'a>, checked: &Checked, tape: &Tape) -> Self {
        let times: Vec<T> = checked.times.iter().map(|&s| T::from_f64(s)).collect();
        let t0 = T::from_f64(job.t0);
        let co = Coef::new();
        let kind = match job.method {
            Method::Rk4 { dt } => {
                let mut segs = Vec::with_capacity(times.len());
                let (mut a64, mut a) = (job.t0, t0);
                for (&b64, &b) in checked.times.iter().zip(&times) {
                    let m = ((b64 - a64) / dt).ceil().max(1.0) as u64;
                    let h = (b - a) / T::from_f64(m as f64);
                    let (half, h6) = (h * co.half, h / co.six);
                    segs.push(Seg {
                        a,
                        b,
                        h,
                        half,
                        h6,
                        m,
                    });
                    (a64, a) = (b64, b);
                }
                Kind::Rk4(segs)
            }
            Method::Dp5 { rtol, atol, hmax } => Kind::Dp5(Dp5Par {
                rtol: T::from_f64(rtol),
                atol: T::from_f64(atol),
                hmax: T::from_f64(hmax),
            }),
            Method::Iter => {
                let mut segs = Vec::with_capacity(times.len());
                let (mut a64, mut a) = (job.t0, t0);
                for (&b64, &b) in checked.times.iter().zip(&times) {
                    // validate checked that both ends are whole numbers with b > a and
                    // b - a at most u32::MAX, so the difference is exact.
                    let m = (b64 - a64) as u64;
                    segs.push(Seg {
                        a,
                        b,
                        h: T::ONE,
                        half: T::ZERO,
                        h6: T::ZERO,
                        m,
                    });
                    (a64, a) = (b64, b);
                }
                Kind::Iter(segs)
            }
        };
        let bounds = T::from_f64(job.bounds);
        Ctx {
            rhs: CpuTape::new(tape),
            // A function of the other precision would only fall back to the tape on each call.
            native: job.native.filter(|&f| T::native(f).is_some()),
            n: checked.n,
            n_p: checked.n_p,
            x0: job.x0,
            params: job.params,
            t0,
            bounds,
            lim: bounds.fmin(T::MAX),
            max_steps: job.max_steps,
            times,
            kind,
            co,
            cancel: None,
        }
    }

    /// Workspace length of one member.
    pub fn work_len(&self) -> usize {
        10 * self.n + self.n_p + 1
    }

    /// Output length of one member.
    pub fn out_len(&self) -> usize {
        self.times.len() * self.n
    }
}

/// The slices a member works on.
pub struct Work<'a, T: Real> {
    y: &'a mut [T],
    y1: &'a mut [T],
    yt: &'a mut [T],
    k: &'a mut [T],
    p: &'a mut [T],
    vals: &'a mut Slab<T>,
    out: &'a mut [T],
}

impl<'a, T: Real> Work<'a, T> {
    /// Splits a member's workspace `w` of `Ctx::work_len` values for `n` states.
    fn new(n: usize, w: &'a mut [T], out: &'a mut [T], vals: &'a mut Slab<T>) -> Self {
        let (y, rest) = w.split_at_mut(n);
        let (y1, rest) = rest.split_at_mut(n);
        let (yt, rest) = rest.split_at_mut(n);
        let (k, p) = rest.split_at_mut(7 * n);
        Work {
            y,
            y1,
            yt,
            k,
            p,
            vals,
            out,
        }
    }
}

/// The state of one member between step attempts.
#[derive(Clone, Copy, Debug)]
pub struct Traj<T: Real> {
    member: usize,
    pub status: Status,
    started: bool,
    /// Time of the last accepted point.
    pub t: T,
    pub accepted: u32,
    pub rejected: u32,
    /// Next output time (RK4: the current segment).
    si: u32,
    /// RK4: steps taken in the current segment.
    step_in_seg: u32,
    /// DP5: the step to try next.
    h: T,
    facold: T,
    reject: bool,
    /// The error estimate of the last attempt was NaN.
    nan_err: bool,
}

fn all_finite<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn clamp<T: Real>(x: T, lo: T, hi: T) -> T {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

impl<T: Real> Traj<T> {
    pub fn new(member: usize) -> Self {
        Traj {
            member,
            status: Status::Running,
            started: false,
            t: T::ZERO,
            accepted: 0,
            rejected: 0,
            si: 0,
            step_in_seg: 0,
            h: T::ZERO,
            facold: T::ZERO,
            reject: false,
            nan_err: false,
        }
    }

    /// Runs up to `budget` step attempts and returns true when the member is finished.
    /// `vals` is scratch from `ctx.rhs.slab()`; `w` is the member's workspace and `out` its
    /// output slice, `times.len() * n` values that start as NaN.
    pub fn advance(
        &mut self,
        ctx: &Ctx<T>,
        w: &mut [T],
        out: &mut [T],
        vals: &mut Slab<T>,
        budget: u64,
    ) -> bool {
        if self.status != Status::Running {
            return true;
        }
        let mut wk = Work::new(ctx.n, w, out, vals);
        self.load(ctx, &mut wk);
        if self.status == Status::Running {
            match &ctx.kind {
                Kind::Rk4(segs) => self.rk4(ctx, segs, &mut wk, budget),
                Kind::Dp5(par) => self.dp5(ctx, par, &mut wk, budget),
                Kind::Iter(segs) => self.iter(ctx, segs, &mut wk, budget),
            }
        }
        self.status != Status::Running
    }

    /// Binds the member's parameters to `wk.vals`, or starts the member on its first call.
    fn load(&mut self, ctx: &Ctx<T>, wk: &mut Work<T>) {
        if self.started {
            ctx.rhs.bind(wk.p, wk.vals);
        } else {
            self.start(ctx, wk);
        }
    }

    /// Runs up to `budget` step attempts for each member of `trs`, a group of `W`
    /// consecutive members or fewer at the end of the run (SPEC section 13, `cpu`). `w` and
    /// `out` hold the workspaces and output slices of the group in member order, and
    /// `slabs` gives each lane its own slab. A full RK4 group whose members are all running
    /// and at the same step runs stage by stage across the lanes; every other case runs one
    /// member at a time. Either way a member does the same operations in the same order on
    /// its own data as it would alone, so its bits do not depend on the group.
    pub fn advance_group<const W: usize>(
        trs: &mut [Traj<T>],
        ctx: &Ctx<T>,
        w: &mut [T],
        out: &mut [T],
        slabs: &mut [Slab<T>; W],
        g: &mut [T],
        budget: u64,
    ) {
        let (wl, ol) = (ctx.work_len(), ctx.out_len());
        let mut ws = w.chunks_exact_mut(wl);
        let mut os = out.chunks_exact_mut(ol);
        let one_by_one = |trs: &mut [Traj<T>],
                          ws: &mut std::slice::ChunksExactMut<'_, T>,
                          os: &mut std::slice::ChunksExactMut<'_, T>,
                          slabs: &mut [Slab<T>; W],
                          budget: u64| {
            for ((tr, (w, o)), vals) in trs.iter_mut().zip(ws.zip(os)).zip(slabs.iter_mut()) {
                tr.advance(ctx, w, o, vals, budget);
            }
        };
        let Kind::Rk4(segs) = &ctx.kind else {
            return one_by_one(trs, &mut ws, &mut os, slabs, budget);
        };
        if trs.len() != W {
            return one_by_one(trs, &mut ws, &mut os, slabs, budget);
        }
        let group: &mut [Traj<T>; W] = trs.try_into().expect("the length is W");
        if group.iter().any(|t| t.status != Status::Running) {
            return one_by_one(trs, &mut ws, &mut os, slabs, budget);
        }
        let mut sl = slabs.iter_mut();
        let mut wks: [Work<T>; W] = std::array::from_fn(|_| {
            let (Some(w), Some(o), Some(vals)) = (ws.next(), os.next(), sl.next()) else {
                unreachable!("a full group has W workspaces, outputs and slabs");
            };
            Work::new(ctx.n, w, o, vals)
        });
        for (tr, wk) in group.iter_mut().zip(wks.iter_mut()) {
            tr.load(ctx, wk);
        }
        let first = group[0];
        let in_step = group.iter().all(|t| {
            t.status == Status::Running
                && (t.si, t.step_in_seg, t.accepted)
                    == (first.si, first.step_in_seg, first.accepted)
        });
        let left = if in_step {
            Traj::rk4_lanes(group, ctx, segs, &mut wks, g, budget)
        } else {
            budget
        };
        if left == 0 {
            return;
        }
        for (tr, wk) in group.iter_mut().zip(wks.iter_mut()) {
            if tr.status == Status::Running {
                tr.rk4(ctx, segs, wk, left);
            }
        }
    }

    /// RK4 on the `W` lanes of a group that are all running and at the same step, stage by
    /// stage across the lanes. Each lane does the operations of [`Traj::rk4`] in its order.
    /// It stops after the first step at which a lane stops and returns the budget left for
    /// the other lanes, which then go on one at a time.
    fn rk4_lanes<const W: usize>(
        trs: &mut [Traj<T>; W],
        ctx: &Ctx<T>,
        segs: &[Seg<T>],
        wks: &mut [Work<T>; W],
        g: &mut [T],
        budget: u64,
    ) -> u64 {
        let n = ctx.n;
        let wn = W * n;
        // The group's states and stages side by side, member-major, so that each pass below
        // is one loop over `W n` values.
        let (y, rest) = g[..7 * wn].split_at_mut(wn);
        let (yt, rest) = rest.split_at_mut(wn);
        let (k1, rest) = rest.split_at_mut(wn);
        let (k2, rest) = rest.split_at_mut(wn);
        let (k3, rest) = rest.split_at_mut(wn);
        let (k4, y1) = rest.split_at_mut(wn);
        for (c, wk) in y.chunks_exact_mut(n).zip(wks.iter()) {
            c.copy_from_slice(wk.y);
        }
        Traj::rk4_lanes_in(trs, ctx, segs, wks, [y, yt, k1, k2, k3, k4, y1], budget)
    }

    /// The loop of [`Traj::rk4_lanes`] on the group buffers `y, yt, k1, k2, k3, k4, y1`. The
    /// buffers `y` and `y1` trade places after a step that every lane accepts, and each lane's
    /// state goes back to its workspace on return.
    fn rk4_lanes_in<const W: usize>(
        trs: &mut [Traj<T>; W],
        ctx: &Ctx<T>,
        segs: &[Seg<T>],
        wks: &mut [Work<T>; W],
        bufs: [&mut [T]; 7],
        mut budget: u64,
    ) -> u64 {
        let (n, two) = (ctx.n, ctx.co.two);
        let [mut y, yt, k1, k2, k3, k4, mut y1] = bufs;
        let call = ctx.native.and_then(|f| ctx.rhs.native_call(f));
        // `k = f(t, x)` for every lane, each with its own slab.
        let stage = |wks: &mut [Work<T>; W], x: &[T], t: T, k: &mut [T]| match call {
            Some(c) => {
                for ((wk, x), k) in wks
                    .iter_mut()
                    .zip(x.chunks_exact(n))
                    .zip(k.chunks_exact_mut(n))
                {
                    c.call(x, t, wk.vals, k);
                }
            }
            None => {
                for ((wk, x), k) in wks
                    .iter_mut()
                    .zip(x.chunks_exact(n))
                    .zip(k.chunks_exact_mut(n))
                {
                    eval_at(ctx, x, t, wk.vals, k);
                }
            }
        };
        while budget > 0 {
            budget -= 1;
            let (si, i) = (trs[0].si, trs[0].step_in_seg);
            let seg = &segs[si as usize];
            let last_in_seg = u64::from(i) + 1 == seg.m;
            let ts = seg.a + T::from_u32(i) * seg.h;
            let t_next = if last_in_seg {
                seg.b
            } else {
                seg.a + T::from_u32(i + 1) * seg.h
            };
            let (h, half, h6) = (seg.h, seg.half, seg.h6);
            stage(wks, y, ts, k1);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k1.iter()) {
                *t = a + half * k;
            }
            stage(wks, yt, ts + half, k2);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k2.iter()) {
                *t = a + half * k;
            }
            stage(wks, yt, ts + half, k3);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k3.iter()) {
                *t = a + h * k;
            }
            stage(wks, yt, t_next, k4);
            for (((((o, &a), &q1), &q2), &q3), &q4) in y1
                .iter_mut()
                .zip(y.iter())
                .zip(k1.iter())
                .zip(k2.iter())
                .zip(k3.iter())
                .zip(k4.iter())
            {
                let mut s = q1 + two * q2;
                s = s + two * q3;
                s = s + q4;
                *o = a + h6 * s;
            }
            let mut going = true;
            let mut took = [true; W];
            for (((tr, wk), y1), took) in trs
                .iter_mut()
                .zip(wks.iter_mut())
                .zip(y1.chunks_exact(n))
                .zip(took.iter_mut())
            {
                match tr.accept(ctx, y1, wk.vals, wk.out, t_next, last_in_seg) {
                    Some(running) => going &= running,
                    None => (*took, going) = (false, false),
                }
            }
            if !going {
                for (((wk, y), y1), took) in wks
                    .iter_mut()
                    .zip(y.chunks_exact(n))
                    .zip(y1.chunks_exact(n))
                    .zip(took)
                {
                    wk.y.copy_from_slice(if took { y1 } else { y });
                }
                return budget;
            }
            std::mem::swap(&mut y, &mut y1);
        }
        for (wk, y) in wks.iter_mut().zip(y.chunks_exact(n)) {
            wk.y.copy_from_slice(y);
        }
        budget
    }

    /// Ends a step whose new state is in `y1`: stops the member when the right-hand side
    /// went past the frame cap or `y1` is rejected, and otherwise accepts `y1` at `t_next`,
    /// writes a sample at the end of a segment and checks the step limit. Returns true while
    /// the member is still running.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn end_step(
        &mut self,
        ctx: &Ctx<T>,
        y: &mut [T],
        y1: &mut [T],
        vals: &Slab<T>,
        out: &mut [T],
        t_next: T,
        last_in_seg: bool,
    ) -> bool {
        match self.accept(ctx, y1, vals, out, t_next, last_in_seg) {
            None => false,
            Some(running) => {
                y.swap_with_slice(y1);
                running
            }
        }
    }

    /// [`Traj::end_step`] without the swap: `None` when the step is rejected and the member
    /// stops on its old state, otherwise whether the member, now on `y1`, still runs.
    #[inline]
    fn accept(
        &mut self,
        ctx: &Ctx<T>,
        y1: &[T],
        vals: &Slab<T>,
        out: &mut [T],
        t_next: T,
        last_in_seg: bool,
    ) -> Option<bool> {
        if self.too_deep(vals) || self.reject_state(ctx, y1) {
            return None;
        }
        self.t = t_next;
        self.accepted += 1;
        self.step_in_seg += 1;
        if last_in_seg {
            self.finish_sample(out, y1);
            self.step_in_seg = 0;
            if self.si as usize == ctx.times.len() {
                self.status = Status::Ok;
                return Some(false);
            }
        }
        if self.accepted >= ctx.max_steps {
            self.status = Status::MaxSteps;
            return Some(false);
        }
        Some(true)
    }

    /// Loads the member's inputs and, for the adaptive method, the first stage and the
    /// starting step.
    fn start(&mut self, ctx: &Ctx<T>, wk: &mut Work<T>) {
        let (n, n_p) = (ctx.n, ctx.n_p);
        self.started = true;
        self.t = ctx.t0;
        let m = self.member;
        for i in 0..n {
            wk.y[i] = T::from_f64(ctx.x0[m * n + i]);
        }
        let src = match ctx.params {
            Params::Shared(p) => p,
            Params::PerMember(p) => &p[m * n_p..(m + 1) * n_p],
        };
        for (dst, &s) in wk.p[..n_p].iter_mut().zip(src) {
            *dst = T::from_f64(s);
        }
        ctx.rhs.bind(wk.p, wk.vals);
        let Kind::Dp5(par) = &ctx.kind else {
            return;
        };
        eval_at(ctx, wk.y, ctx.t0, wk.vals, &mut wk.k[..n]);
        if self.too_deep(wk.vals) {
            return;
        }
        if !all_finite(&wk.k[..n]) {
            self.status = Status::NonFinite;
            return;
        }
        let h = initial_step(ctx, par, wk);
        if self.too_deep(wk.vals) {
            return;
        }
        if !(h.is_finite() && h > T::ZERO) {
            self.status = Status::NonFinite;
            return;
        }
        self.h = h;
        self.facold = ctx.co.facold0;
    }

    fn finish_sample(&mut self, out: &mut [T], y: &[T]) {
        let n = y.len();
        let at = self.si as usize * n;
        out[at..at + n].copy_from_slice(y);
        self.si += 1;
    }

    /// Classical RK4 on equal steps. With `k1` to `k4` the stages and `h6 = h / 6`:
    /// `y1 = y + h6 * (((k1 + 2 k2) + 2 k3) + k4)`. Only `+ - * /` appear, so a kernel
    /// without fused multiply-add reproduces the bits.
    fn rk4(&mut self, ctx: &Ctx<T>, segs: &[Seg<T>], wk: &mut Work<T>, mut budget: u64) {
        let (n, co) = (ctx.n, &ctx.co);
        let two = co.two;
        let (y, y1, yt) = (&mut *wk.y, &mut *wk.y1, &mut *wk.yt);
        let (vals, out) = (&mut *wk.vals, &mut *wk.out);
        let mut ks = wk.k.chunks_exact_mut(n);
        let (Some(k1), Some(k2), Some(k3), Some(k4)) = (ks.next(), ks.next(), ks.next(), ks.next())
        else {
            unreachable!("the workspace holds seven stages");
        };
        while budget > 0 {
            budget -= 1;
            let seg = &segs[self.si as usize];
            let i = self.step_in_seg;
            let last_in_seg = u64::from(i) + 1 == seg.m;
            let ts = seg.a + T::from_u32(i) * seg.h;
            let t_next = if last_in_seg {
                seg.b
            } else {
                seg.a + T::from_u32(i + 1) * seg.h
            };
            let (h, half, h6) = (seg.h, seg.half, seg.h6);
            eval_at(ctx, y, ts, vals, k1);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k1.iter()) {
                *t = a + half * k;
            }
            eval_at(ctx, yt, ts + half, vals, k2);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k2.iter()) {
                *t = a + half * k;
            }
            eval_at(ctx, yt, ts + half, vals, k3);
            for ((t, &a), &k) in yt.iter_mut().zip(y.iter()).zip(k3.iter()) {
                *t = a + h * k;
            }
            eval_at(ctx, yt, t_next, vals, k4);
            for j in 0..n {
                let mut s = k1[j] + two * k2[j];
                s = s + two * k3[j];
                s = s + k4[j];
                y1[j] = y[j] + h6 * s;
            }
            if !self.end_step(ctx, y, y1, vals, out, t_next, last_in_seg) {
                return;
            }
        }
    }

    /// Iteration of a map (SPEC section 8aa): `y1 = F(y, p, ts)` with `ts = a + i` for the
    /// iterate `i` of the segment, then the same state check as RK4.
    fn iter(&mut self, ctx: &Ctx<T>, segs: &[Seg<T>], wk: &mut Work<T>, mut budget: u64) {
        let (y, y1) = (&mut *wk.y, &mut *wk.y1);
        let (vals, out) = (&mut *wk.vals, &mut *wk.out);
        while budget > 0 {
            budget -= 1;
            let seg = &segs[self.si as usize];
            let i = self.step_in_seg;
            let last_in_seg = u64::from(i) + 1 == seg.m;
            let ts = seg.a + T::from_u32(i);
            let t_next = if last_in_seg {
                seg.b
            } else {
                seg.a + T::from_u32(i + 1)
            };
            eval_at(ctx, y, ts, vals, y1);
            if !self.end_step(ctx, y, y1, vals, out, t_next, last_in_seg) {
                return;
            }
        }
    }

    /// Ends the member with [`Status::RecursionDepth`] when a right-hand side since the
    /// last bind went past the frame cap. Only a tape with calls can fail, and a failed
    /// slab stays failed until the next bind, so one test after each step covers every stage.
    fn too_deep(&mut self, vals: &Slab<T>) -> bool {
        if vals.failed() {
            self.status = Status::RecursionDepth;
            return true;
        }
        false
    }

    /// Sets the status and returns true when the new state `y1` is not finite or leaves the
    /// bounds. The member then keeps its last accepted state and time.
    /// One pass without early exit tests the whole state against `lim`. Only when it fails
    /// does the slow path look again, so that NONFINITE still comes before BOUNDS.
    fn reject_state(&mut self, ctx: &Ctx<T>, y1: &[T]) -> bool {
        let lim = ctx.lim;
        // Written as a positive test so that NaN fails it.
        let ok = y1.iter().fold(true, |acc, &v| acc & (v.abs() <= lim));
        if ok {
            return false;
        }
        if !all_finite(y1) {
            self.status = Status::NonFinite;
            return true;
        }
        if y1.iter().any(|v| v.abs() > ctx.bounds) {
            self.status = Status::Bounds;
            return true;
        }
        false
    }

    /// Dormand-Prince 5(4) with the controller of `dopri5`.
    fn dp5(&mut self, ctx: &Ctx<T>, par: &Dp5Par<T>, wk: &mut Work<T>, mut budget: u64) {
        let (n, co) = (ctx.n, &ctx.co);
        while budget > 0 {
            budget -= 1;
            // Written as a positive test so that a NaN step size also stops the run.
            let step_ok = co.tenth * self.h.abs() > self.t.abs() * co.uround;
            if !step_ok {
                self.status = if self.nan_err {
                    Status::NonFinite
                } else {
                    Status::StepTooSmall
                };
                return;
            }
            // Shorten the step to land on the next output time.
            let s = ctx.times[self.si as usize];
            let h_free = self.h;
            let mut landing = false;
            if self.t + co.c101 * self.h > s {
                self.h = s - self.t;
                landing = true;
            }
            let last = landing && self.si as usize + 1 == ctx.times.len();
            let t_end = if landing { s } else { self.t + self.h };
            let err = dp5_step(ctx, par, wk, self.t, self.h, t_end);
            if self.too_deep(wk.vals) {
                return;
            }
            let fac11 = err.powf(co.expo1);
            if err <= T::ONE {
                if self.reject_state(ctx, wk.y1) {
                    return;
                }
                // Only an accepted step uses facold^beta, so a rejected one skips the power.
                let fac = clamp(
                    fac11 / self.facold.powf(co.beta) / co.safe,
                    co.inv_fac2,
                    co.inv_fac1,
                );
                let mut hnew = self.h / fac;
                self.facold = err.fmax(co.facold_min);
                self.accepted += 1;
                self.t = t_end;
                self.nan_err = false;
                wk.y.swap_with_slice(wk.y1);
                let (k0, rest) = wk.k.split_at_mut(n);
                k0.copy_from_slice(&rest[5 * n..6 * n]);
                if landing {
                    self.finish_sample(wk.out, wk.y);
                    if last {
                        self.status = Status::Ok;
                        return;
                    }
                }
                if self.accepted >= ctx.max_steps {
                    self.status = Status::MaxSteps;
                    return;
                }
                if hnew.abs() > par.hmax {
                    hnew = par.hmax;
                }
                if self.reject {
                    hnew = hnew.abs().fmin(self.h.abs());
                }
                // A step cut short to land on an output time says little about the size
                // the controller wants, so the next step starts from the size it had.
                if landing && self.h < h_free && hnew < h_free {
                    hnew = h_free;
                }
                self.reject = false;
                self.h = hnew;
            } else {
                // The clamp above would give a NaN error a growth factor; shrink instead.
                self.h = self.h / co.inv_fac1.fmin(fac11 / co.safe);
                self.reject = true;
                self.rejected += 1;
                self.nan_err = err.is_nan();
            }
        }
    }
}

fn eval_at<T: Real>(ctx: &Ctx<T>, x: &[T], t: T, vals: &mut Slab<T>, out: &mut [T]) {
    match ctx.native {
        Some(f) => ctx.rhs.eval_native(f, x, t, vals, out),
        None => ctx.rhs.eval_xt_with(x, t, vals, out, ctx.cancel),
    }
}

/// One Dormand-Prince step from `(t, y)` with `k0 = f(t, y)` given. Fills `k1` to `k6` and
/// `y1`, and returns the scaled error norm. `t_end` is the time of the stages at c = 1.
fn dp5_step<T: Real>(ctx: &Ctx<T>, par: &Dp5Par<T>, wk: &mut Work<T>, t: T, h: T, t_end: T) -> T {
    let (n, co) = (ctx.n, &ctx.co);
    let (y, y1, yt) = (&*wk.y, &mut *wk.y1, &mut *wk.yt);
    let (vals, k) = (&mut *wk.vals, &mut *wk.k);
    for s in 1..=5 {
        let (done, rest) = k.split_at_mut(s * n);
        let c = &co.a[s - 1][..s];
        for (i, (v, &yi)) in yt.iter_mut().zip(y).enumerate() {
            *v = yi + h * stage_sum(c, done, n, i);
        }
        let ts = if co.c[s - 1] == T::ONE {
            t_end
        } else {
            t + co.c[s - 1] * h
        };
        eval_at(ctx, yt, ts, vals, &mut rest[..n]);
    }
    let (done, rest) = k.split_at_mut(6 * n);
    for (i, (v, &yi)) in y1.iter_mut().zip(y).enumerate() {
        *v = yi + h * stage_sum(&co.a[5], done, n, i);
    }
    eval_at(ctx, y1, t_end, vals, &mut rest[..n]);
    let mut sum = T::ZERO;
    for (i, (&yi, &y1i)) in y.iter().zip(y1.iter()).enumerate() {
        let e = stage_sum(&co.e, k, n, i);
        let sk = par.atol + par.rtol * yi.abs().fmax(y1i.abs());
        let r = h * e / sk;
        sum = sum + r * r;
    }
    (sum / T::from_u32(n as u32)).sqrt()
}

/// The sum over `j` of `c[j] * k[j * n + i]`, started from zero with the stages added in
/// order, zero coefficients included. Every DP5 sum goes through here, so each element sees
/// the same operations in the same order. The sum stays in a register for the whole element;
/// a pass over the whole state per stage was slower at small `n`.
fn stage_sum<T: Real>(c: &[T], k: &[T], n: usize, i: usize) -> T {
    let mut acc = T::ZERO;
    for (j, &cj) in c.iter().enumerate() {
        acc = acc + cj * k[j * n + i];
    }
    acc
}

/// The starting step of `dopri5` (`HINIT`). `k0` holds `f(t0, y0)`.
fn initial_step<T: Real>(ctx: &Ctx<T>, par: &Dp5Par<T>, wk: &mut Work<T>) -> T {
    let (n, co) = (ctx.n, &ctx.co);
    let sk = |y: T| par.atol + par.rtol * y.abs();
    let mut dnf = T::ZERO;
    let mut dny = T::ZERO;
    for i in 0..n {
        let s = sk(wk.y[i]);
        let a = wk.k[i] / s;
        dnf = dnf + a * a;
        let b = wk.y[i] / s;
        dny = dny + b * b;
    }
    let h = if dnf <= co.c1e10 || dny <= co.c1e10 {
        co.c1e6
    } else {
        (dny / dnf).sqrt() * co.c001
    };
    let h = h.fmin(par.hmax);
    for i in 0..n {
        wk.yt[i] = wk.y[i] + h * wk.k[i];
    }
    let t0 = ctx.t0;
    {
        let (k0, rest) = wk.k.split_at_mut(n);
        let _ = k0;
        eval_at(ctx, wk.yt, t0 + h, wk.vals, &mut rest[..n]);
    }
    let mut s2 = T::ZERO;
    for i in 0..n {
        let d = (wk.k[n + i] - wk.k[i]) / sk(wk.y[i]);
        s2 = s2 + d * d;
    }
    let der2 = s2.sqrt() / h.abs();
    let der12 = der2.abs().fmax(dnf.sqrt());
    let h1 = if der12 <= co.c1e15 {
        co.c1e6.fmax(h.abs() * co.c1e3)
    } else {
        (co.c001 / der12).powf(co.fifth)
    };
    (co.c100 * h.abs()).fmin(h1).fmin(par.hmax)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tableau_is_consistent() {
        for (s, row) in A.iter().enumerate().take(5) {
            let sum: f64 = row.iter().sum();
            assert!((sum - C[s]).abs() < 1e-15, "row {s} sums to {sum}");
        }
        let b: f64 = A[5].iter().sum();
        assert!((b - 1.0).abs() < 1e-15);
        let e: f64 = E.iter().sum();
        assert!(e.abs() < 1e-15, "b and bhat both sum to one, e sums to {e}");
    }

    #[test]
    fn single_precision_constants_round_to_nearest() {
        let c32 = Coef::<f32>::new();
        assert_eq!(c32.expo1, 0.17_f32);
        assert_eq!(c32.inv_fac1, 5.0_f32);
        assert_eq!(c32.inv_fac2, 0.1_f32);
        let c64 = Coef::<f64>::new();
        assert_eq!(c64.uround, 2.3e-16);
    }
}
