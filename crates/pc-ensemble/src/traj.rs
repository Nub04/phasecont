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

use crate::job::{Checked, Job, Method, Params, Status};
use crate::real::Real;
use crate::tape::Tape;

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

/// An RK4 segment: `m` equal steps of length `h` from `a` to `b`.
#[derive(Clone, Copy, Debug)]
pub struct Seg<T: Real> {
    pub a: T,
    pub b: T,
    pub h: T,
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
}

/// What every member of a run shares.
pub struct Ctx<'a, T: Real> {
    pub tape: &'a Tape,
    pub n: usize,
    pub n_p: usize,
    pub x0: &'a [f64],
    pub params: Params<'a>,
    pub t0: T,
    pub bounds: T,
    pub max_steps: u32,
    /// The output times rounded to `T`.
    pub times: Vec<T>,
    pub kind: Kind<T>,
    pub co: Coef<T>,
}

impl<'a, T: Real> Ctx<'a, T> {
    /// Rounds the job to `T`. The job must have passed `validate`.
    pub fn new(job: &Job<'a>, checked: &Checked, tape: &'a Tape) -> Self {
        let times: Vec<T> = checked.times.iter().map(|&s| T::from_f64(s)).collect();
        let t0 = T::from_f64(job.t0);
        let kind = match job.method {
            Method::Rk4 { dt } => {
                let mut segs = Vec::with_capacity(times.len());
                let (mut a64, mut a) = (job.t0, t0);
                for (&b64, &b) in checked.times.iter().zip(&times) {
                    let m = ((b64 - a64) / dt).ceil().max(1.0) as u64;
                    let h = (b - a) / T::from_f64(m as f64);
                    segs.push(Seg { a, b, h, m });
                    (a64, a) = (b64, b);
                }
                Kind::Rk4(segs)
            }
            Method::Dp5 { rtol, atol, hmax } => Kind::Dp5(Dp5Par {
                rtol: T::from_f64(rtol),
                atol: T::from_f64(atol),
                hmax: T::from_f64(hmax),
            }),
        };
        Ctx {
            tape,
            n: checked.n,
            n_p: checked.n_p,
            x0: job.x0,
            params: job.params,
            t0,
            bounds: T::from_f64(job.bounds),
            max_steps: job.max_steps,
            times,
            kind,
            co: Coef::new(),
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
    vals: &'a mut [T],
    out: &'a mut [T],
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
    /// `vals` is scratch of at least `tape.len()` values; `w` is the member's workspace and
    /// `out` its output slice, `times.len() * n` values that start as NaN.
    pub fn advance(
        &mut self,
        ctx: &Ctx<T>,
        w: &mut [T],
        out: &mut [T],
        vals: &mut [T],
        budget: u64,
    ) -> bool {
        if self.status != Status::Running {
            return true;
        }
        let n = ctx.n;
        let (y, rest) = w.split_at_mut(n);
        let (y1, rest) = rest.split_at_mut(n);
        let (yt, rest) = rest.split_at_mut(n);
        let (k, p) = rest.split_at_mut(7 * n);
        let mut wk = Work {
            y,
            y1,
            yt,
            k,
            p,
            vals,
            out,
        };
        if !self.started {
            self.start(ctx, &mut wk);
        }
        if self.status == Status::Running {
            match &ctx.kind {
                Kind::Rk4(segs) => self.rk4(ctx, segs, &mut wk, budget),
                Kind::Dp5(par) => self.dp5(ctx, par, &mut wk, budget),
            }
        }
        self.status != Status::Running
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
        wk.p[n_p] = ctx.t0;
        let Kind::Dp5(par) = &ctx.kind else {
            return;
        };
        eval_at(ctx, wk.y, wk.p, ctx.t0, wk.vals, &mut wk.k[..n]);
        if !all_finite(&wk.k[..n]) {
            self.status = Status::NonFinite;
            return;
        }
        let h = initial_step(ctx, par, wk);
        if !(h.is_finite() && h > T::ZERO) {
            self.status = Status::NonFinite;
            return;
        }
        self.h = h;
        self.facold = ctx.co.facold0;
    }

    fn finish_sample(&mut self, ctx: &Ctx<T>, wk: &mut Work<T>) {
        let n = ctx.n;
        let at = self.si as usize * n;
        wk.out[at..at + n].copy_from_slice(wk.y);
        self.si += 1;
    }

    /// Classical RK4 on equal steps. With `k1` to `k4` the stages and `h6 = h / 6`:
    /// `y1 = y + h6 * (((k1 + 2 k2) + 2 k3) + k4)`. Only `+ - * /` appear, so a kernel
    /// without fused multiply-add reproduces the bits.
    fn rk4(&mut self, ctx: &Ctx<T>, segs: &[Seg<T>], wk: &mut Work<T>, mut budget: u64) {
        let (n, co) = (ctx.n, &ctx.co);
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
            let h = seg.h;
            let half = h * co.half;
            eval_at(ctx, wk.y, wk.p, ts, wk.vals, &mut wk.k[..n]);
            for j in 0..n {
                wk.yt[j] = wk.y[j] + half * wk.k[j];
            }
            eval_at(ctx, wk.yt, wk.p, ts + half, wk.vals, &mut wk.k[n..2 * n]);
            for j in 0..n {
                wk.yt[j] = wk.y[j] + half * wk.k[n + j];
            }
            eval_at(
                ctx,
                wk.yt,
                wk.p,
                ts + half,
                wk.vals,
                &mut wk.k[2 * n..3 * n],
            );
            for j in 0..n {
                wk.yt[j] = wk.y[j] + h * wk.k[2 * n + j];
            }
            eval_at(ctx, wk.yt, wk.p, t_next, wk.vals, &mut wk.k[3 * n..4 * n]);
            let h6 = h / co.six;
            for j in 0..n {
                let mut s = wk.k[j] + co.two * wk.k[n + j];
                s = s + co.two * wk.k[2 * n + j];
                s = s + wk.k[3 * n + j];
                wk.y1[j] = wk.y[j] + h6 * s;
            }
            if self.reject_state(ctx, wk) {
                return;
            }
            wk.y.swap_with_slice(wk.y1);
            self.t = t_next;
            self.accepted += 1;
            self.step_in_seg += 1;
            if last_in_seg {
                self.finish_sample(ctx, wk);
                self.step_in_seg = 0;
                if self.si as usize == ctx.times.len() {
                    self.status = Status::Ok;
                    return;
                }
            }
            if self.accepted >= ctx.max_steps {
                self.status = Status::MaxSteps;
                return;
            }
        }
    }

    /// Sets the status and returns true when the new state `y1` is not finite or leaves the
    /// bounds. The member then keeps its last accepted state and time.
    fn reject_state(&mut self, ctx: &Ctx<T>, wk: &Work<T>) -> bool {
        if !all_finite(wk.y1) {
            self.status = Status::NonFinite;
            return true;
        }
        if wk.y1.iter().any(|v| v.abs() > ctx.bounds) {
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
            let fac11 = err.powf(co.expo1);
            let fac = clamp(
                fac11 / self.facold.powf(co.beta) / co.safe,
                co.inv_fac2,
                co.inv_fac1,
            );
            let mut hnew = self.h / fac;
            if err <= T::ONE {
                if self.reject_state(ctx, wk) {
                    return;
                }
                self.facold = err.fmax(co.facold_min);
                self.accepted += 1;
                self.t = t_end;
                self.nan_err = false;
                wk.y.swap_with_slice(wk.y1);
                let (k0, rest) = wk.k.split_at_mut(n);
                k0.copy_from_slice(&rest[5 * n..6 * n]);
                if landing {
                    self.finish_sample(ctx, wk);
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
                hnew = self.h / co.inv_fac1.fmin(fac11 / co.safe);
                self.reject = true;
                self.rejected += 1;
                self.nan_err = err.is_nan();
                self.h = hnew;
            }
        }
    }
}

fn eval_at<T: Real>(ctx: &Ctx<T>, x: &[T], p: &mut [T], t: T, vals: &mut [T], out: &mut [T]) {
    p[ctx.n_p] = t;
    ctx.tape.eval(x, p, vals, out);
}

/// One Dormand-Prince step from `(t, y)` with `k0 = f(t, y)` given. Fills `k1` to `k6` and
/// `y1`, and returns the scaled error norm. `t_end` is the time of the stages at c = 1.
fn dp5_step<T: Real>(ctx: &Ctx<T>, par: &Dp5Par<T>, wk: &mut Work<T>, t: T, h: T, t_end: T) -> T {
    let (n, co) = (ctx.n, &ctx.co);
    for s in 1..=5 {
        for i in 0..n {
            let mut acc = T::ZERO;
            for j in 0..s {
                acc = acc + co.a[s - 1][j] * wk.k[j * n + i];
            }
            wk.yt[i] = wk.y[i] + h * acc;
        }
        let ts = if co.c[s - 1] == T::ONE {
            t_end
        } else {
            t + co.c[s - 1] * h
        };
        eval_at(ctx, wk.yt, wk.p, ts, wk.vals, &mut wk.k[s * n..(s + 1) * n]);
    }
    for i in 0..n {
        let mut acc = T::ZERO;
        for j in 0..6 {
            acc = acc + co.a[5][j] * wk.k[j * n + i];
        }
        wk.y1[i] = wk.y[i] + h * acc;
    }
    eval_at(ctx, wk.y1, wk.p, t_end, wk.vals, &mut wk.k[6 * n..7 * n]);
    let mut sum = T::ZERO;
    for i in 0..n {
        let mut e = T::ZERO;
        for (j, ej) in co.e.iter().enumerate() {
            e = e + *ej * wk.k[j * n + i];
        }
        let sk = par.atol + par.rtol * wk.y[i].abs().fmax(wk.y1[i].abs());
        let r = h * e / sk;
        sum = sum + r * r;
    }
    (sum / T::from_u32(n as u32)).sqrt()
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
        eval_at(ctx, wk.yt, wk.p, t0 + h, wk.vals, &mut rest[..n]);
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
