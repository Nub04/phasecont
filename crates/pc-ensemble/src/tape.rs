//! The right-hand side of a model as a flat tape, evaluated in `f32` or `f64`.
//!
//! The tape holds the nodes reachable from `ir.f`, renumbered from zero, in the order
//! `pc_ir::eval::eval` visits them. Each op follows `pc_ir::eval::eval_op`, with the scalar
//! type generic. The kernel emitters walk the same tape, so the CPU loop and a GPU kernel
//! run the same operations in the same order.
//!
//! Constructs the kernels cannot lower are refused when the tape is built, by name:
//! tables, `erf`, `erfc`, `lgamma`, Bessel functions and `polygamma`.
//!
//! The CPU loop runs a [`CpuTape`], the same tape lowered once per run for one scalar type:
//! constants decoded to `T` up front, integer powers written out as the multiply chain of
//! [`Op::PowI`], and every operand a slot of one scratch slab. Each value goes through the
//! same operations in the same order as [`Tape::eval`], so the two give the same bits.

use std::collections::HashMap;

use pc_codegen::kernel::unsupported_op;
use pc_ir::IrModel;
use pc_ir::dag::{Cmp, Fn1, Fn2, NodeId, Op};
use pc_ir::eval::live_set;

use crate::job::{EnsembleError, NativeRhs};
use crate::real::Real;

#[derive(Clone, Debug)]
pub struct Tape {
    pub n: usize,
    pub n_p: usize,
    /// Ops in evaluation order. A child's id is the index of its slot.
    pub ops: Vec<Op>,
    /// The slot of each state's right-hand side.
    pub outputs: Vec<u32>,
}

fn remap(op: &Op, m: &[u32]) -> Op {
    let r = |id: NodeId| NodeId(m[id.0 as usize]);
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => *op,
        Op::Neg(a) => Op::Neg(r(a)),
        Op::Add(a, b) => Op::Add(r(a), r(b)),
        Op::Sub(a, b) => Op::Sub(r(a), r(b)),
        Op::Mul(a, b) => Op::Mul(r(a), r(b)),
        Op::Div(a, b) => Op::Div(r(a), r(b)),
        Op::PowI(a, n) => Op::PowI(r(a), n),
        Op::Call1(f, a) => Op::Call1(f, r(a)),
        Op::Call2(f, a, b) => Op::Call2(f, r(a), r(b)),
        Op::Cmp(c, a, b) => Op::Cmp(c, r(a), r(b)),
        Op::And(a, b) => Op::And(r(a), r(b)),
        Op::Or(a, b) => Op::Or(r(a), r(b)),
        Op::Not(a) => Op::Not(r(a)),
        Op::Select(c, a, b) => Op::Select(r(c), r(a), r(b)),
        Op::Table(t, a) => Op::Table(t, r(a)),
        Op::TableSlope(t, a) => Op::TableSlope(t, r(a)),
    }
}

impl Tape {
    /// Builds the tape of `ir.f`, or names the constructs that cannot run.
    pub fn build(ir: &IrModel) -> Result<Tape, EnsembleError> {
        let live = live_set(&ir.dag, &ir.f);
        let mut map = vec![u32::MAX; ir.dag.len()];
        let mut ops = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        for (i, &is_live) in live.iter().enumerate() {
            if !is_live {
                continue;
            }
            let op = ir.dag.op(NodeId(i as u32));
            if let Some(name) = unsupported_op(op) {
                if !names.contains(&name) {
                    names.push(name);
                }
                continue;
            }
            map[i] = ops.len() as u32;
            ops.push(remap(op, &map));
        }
        if !names.is_empty() {
            return Err(EnsembleError::Unsupported(format!(
                "the right-hand side uses {}, which the ensemble engine cannot run",
                names.join(", ")
            )));
        }
        let mut outputs = Vec::with_capacity(ir.f.len());
        for id in &ir.f {
            match map.get(id.0 as usize) {
                Some(&s) if s != u32::MAX => outputs.push(s),
                _ => {
                    return Err(EnsembleError::BadJob(
                        "a right-hand side is outside the model's expression graph".into(),
                    ));
                }
            }
        }
        Ok(Tape {
            n: ir.n(),
            n_p: ir.n_p(),
            ops,
            outputs,
        })
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Writes `f(t, x)` to `out`. `p` holds the `n_p` parameters followed by `t`. `vals` is
    /// scratch of at least [`Tape::len`] values.
    pub fn eval<T: Real>(&self, x: &[T], p: &[T], vals: &mut [T], out: &mut [T]) {
        for (i, op) in self.ops.iter().enumerate() {
            vals[i] = eval_op(op, vals, x, p);
        }
        for (o, &s) in out.iter_mut().zip(&self.outputs) {
            *o = vals[s as usize];
        }
    }
}

fn b2f<T: Real>(b: bool) -> T {
    if b { T::ONE } else { T::ZERO }
}

/// Integer power in the order documented on [`Op::PowI`].
fn powi<T: Real>(x: T, n: i32) -> T {
    let mut m = n.unsigned_abs();
    let mut base = x;
    let mut acc: Option<T> = None;
    if m == 0 {
        return T::ONE;
    }
    loop {
        if m & 1 == 1 {
            acc = Some(match acc {
                None => base,
                Some(a) => a * base,
            });
        }
        m >>= 1;
        if m == 0 {
            break;
        }
        base = base * base;
    }
    let r = acc.unwrap_or(T::ONE);
    if n < 0 { T::ONE / r } else { r }
}

fn call1<T: Real>(f: Fn1, x: T) -> T {
    match f {
        Fn1::Sin => x.sin(),
        Fn1::Cos => x.cos(),
        Fn1::Tan => x.tan(),
        Fn1::Asin => x.asin(),
        Fn1::Acos => x.acos(),
        Fn1::Atan => x.atan(),
        Fn1::Sinh => x.sinh(),
        Fn1::Cosh => x.cosh(),
        Fn1::Tanh => x.tanh(),
        Fn1::Exp => x.exp(),
        Fn1::Ln => x.ln(),
        Fn1::Log10 => x.log10(),
        Fn1::Sqrt => x.sqrt(),
        Fn1::Abs => x.abs(),
        Fn1::Floor => x.floor(),
        Fn1::Ceil => x.ceil(),
        Fn1::Heav => {
            if x >= T::ZERO {
                T::ONE
            } else {
                T::ZERO
            }
        }
        Fn1::Sign => {
            if x > T::ZERO {
                T::ONE
            } else if x < T::ZERO {
                -T::ONE
            } else if x == T::ZERO {
                T::ZERO
            } else {
                T::NAN
            }
        }
        // Refused when the tape is built.
        Fn1::Erf | Fn1::Erfc | Fn1::Lgamma => T::NAN,
    }
}

fn call2<T: Real>(f: Fn2, x: T, y: T) -> T {
    match f {
        Fn2::Atan2 => x.atan2(y),
        Fn2::Pow => x.powf(y),
        Fn2::Mod => {
            let r = x.fmod(y);
            if r < T::ZERO { r + y } else { r }
        }
        Fn2::Min => {
            if x < y {
                x
            } else {
                y
            }
        }
        Fn2::Max => {
            if x > y {
                x
            } else {
                y
            }
        }
        Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma => T::NAN,
    }
}

fn eval_op<T: Real>(op: &Op, v: &[T], x: &[T], p: &[T]) -> T {
    let a = |id: NodeId| v[id.0 as usize];
    match *op {
        Op::Const(b) => T::from_f64(f64::from_bits(b)),
        Op::Var(i) => x[i as usize],
        Op::Param(j) => p[j as usize],
        Op::Neg(u) => -a(u),
        Op::Add(u, w) => a(u) + a(w),
        Op::Sub(u, w) => a(u) - a(w),
        Op::Mul(u, w) => a(u) * a(w),
        Op::Div(u, w) => a(u) / a(w),
        Op::PowI(u, n) => powi(a(u), n),
        Op::Call1(f, u) => call1(f, a(u)),
        Op::Call2(f, u, w) => call2(f, a(u), a(w)),
        Op::Cmp(c, u, w) => cmp(c, a(u), a(w)),
        Op::And(u, w) => and(a(u), a(w)),
        Op::Or(u, w) => or(a(u), a(w)),
        Op::Not(u) => not(a(u)),
        Op::Select(c, u, w) => select(a(c), a(u), a(w)),
        // Refused when the tape is built.
        Op::Dir(_) | Op::Table(..) | Op::TableSlope(..) => T::NAN,
    }
}

fn cmp<T: Real>(c: Cmp, l: T, r: T) -> T {
    b2f(match c {
        Cmp::Lt => l < r,
        Cmp::Le => l <= r,
        Cmp::Gt => l > r,
        Cmp::Ge => l >= r,
        Cmp::Eq => l == r,
    })
}

fn and<T: Real>(l: T, r: T) -> T {
    b2f(l != T::ZERO && r != T::ZERO)
}

fn or<T: Real>(l: T, r: T) -> T {
    b2f(l != T::ZERO || r != T::ZERO)
}

fn not<T: Real>(x: T) -> T {
    b2f(x == T::ZERO)
}

fn select<T: Real>(c: T, u: T, w: T) -> T {
    if c != T::ZERO { u } else { w }
}

/// One instruction of a [`CpuTape`]. Operands are slab slots; instruction `k` writes slot
/// `tb + k`, where `tb` is the first temporary.
#[derive(Clone, Copy, Debug)]
enum Ins {
    Neg(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Call1(Fn1, u32),
    Call2(Fn2, u32, u32),
    Cmp(Cmp, u32, u32),
    And(u32, u32),
    Or(u32, u32),
    Not(u32),
    Select(u32, u32, u32),
}

/// Runs one instruction.
///
/// # Safety
/// Every operand slot of `ins` must be below `s.len()`. [`CpuTape::new`] checks each operand
/// against the slab layout, and [`CpuTape::eval_xt`] checks the slab length.
unsafe fn run<T: Real>(ins: Ins, s: &[T]) -> T {
    // SAFETY: the caller promises every operand slot is in range.
    let a = |i: u32| unsafe { *s.get_unchecked(i as usize) };
    match ins {
        Ins::Neg(u) => -a(u),
        Ins::Add(u, w) => a(u) + a(w),
        Ins::Sub(u, w) => a(u) - a(w),
        Ins::Mul(u, w) => a(u) * a(w),
        Ins::Div(u, w) => a(u) / a(w),
        Ins::Call1(f, u) => call1(f, a(u)),
        Ins::Call2(f, u, w) => call2(f, a(u), a(w)),
        Ins::Cmp(c, u, w) => cmp(c, a(u), a(w)),
        Ins::And(u, w) => and(a(u), a(w)),
        Ins::Or(u, w) => or(a(u), a(w)),
        Ins::Not(u) => not(a(u)),
        Ins::Select(c, u, w) => select(a(c), a(u), a(w)),
    }
}

/// Copies `src` to `dst`, which have the same length, one element at a time. The state vector
/// is a few values for most models, and a call into `memmove` costs more than the copy. The
/// volatile store keeps the compiler from turning the loop back into that call.
fn copy_states<T: Real>(dst: &mut [T], src: &[T]) {
    assert_eq!(dst.len(), src.len());
    for (d, &v) in dst.iter_mut().zip(src) {
        // SAFETY: `d` is a valid, aligned, exclusive reference.
        unsafe { std::ptr::write_volatile(d, v) };
    }
}

/// A [`Tape`] lowered for the CPU loop in one scalar type.
///
/// A slab holds, in order, the constants, the `n` states, the `n_p` parameters and `t`,
/// then one temporary per instruction. Constants, states and parameters need no
/// instruction: an op that reads one reads its slot.
#[derive(Clone, Debug)]
pub struct CpuTape<T: Real> {
    n: usize,
    /// `n_p + 1`: the parameters and `t`.
    np1: usize,
    consts: Vec<T>,
    ins: Vec<Ins>,
    /// The slot of each state's right-hand side.
    outputs: Vec<u32>,
}

/// Scratch of a [`CpuTape`], made by [`CpuTape::slab`] with the constants in place. A slab
/// belongs to the tape that made it.
#[derive(Clone, Debug)]
pub struct Slab<T: Real>(Vec<T>);

/// The constants of a tape, each bit pattern once, in order of first use.
#[derive(Default)]
struct Consts {
    bits: Vec<u64>,
    at: HashMap<u64, u32>,
}

impl Consts {
    fn add(&mut self, b: u64) {
        let next = self.bits.len() as u32;
        if let std::collections::hash_map::Entry::Vacant(e) = self.at.entry(b) {
            e.insert(next);
            self.bits.push(b);
        }
    }
}

const ONE_BITS: u64 = 0x3ff0_0000_0000_0000;

struct Lower {
    ins: Vec<Ins>,
    tb: u32,
}

impl Lower {
    fn emit(&mut self, i: Ins) -> u32 {
        self.ins.push(i);
        self.tb + self.ins.len() as u32 - 1
    }

    /// `x^k` as the chain [`powi`] computes, one instruction per multiply or divide. `one`
    /// is the slot of 1, present whenever `k <= 0`.
    fn powi(&mut self, x: u32, k: i32, one: Option<u32>) -> u32 {
        let mut m = k.unsigned_abs();
        if m == 0 {
            return one.expect("a power k <= 0 adds the constant 1");
        }
        let mut base = x;
        let mut acc: Option<u32> = None;
        loop {
            if m & 1 == 1 {
                acc = Some(match acc {
                    None => base,
                    Some(a) => self.emit(Ins::Mul(a, base)),
                });
            }
            m >>= 1;
            if m == 0 {
                break;
            }
            base = self.emit(Ins::Mul(base, base));
        }
        let r = acc.expect("m > 0 has a set bit");
        if k < 0 {
            let one = one.expect("a power k <= 0 adds the constant 1");
            self.emit(Ins::Div(one, r))
        } else {
            r
        }
    }
}

impl<T: Real> CpuTape<T> {
    /// Lowers `tape`. Panics when an op reads a slot that is not computed yet, or a state or
    /// parameter the tape does not have; [`Tape::build`] never makes such a tape.
    pub fn new(tape: &Tape) -> Self {
        let (n, np1) = (tape.n, tape.n_p + 1);
        let nan_bits = f64::NAN.to_bits();
        let mut consts = Consts::default();
        for op in &tape.ops {
            match *op {
                Op::Const(b) => consts.add(b),
                Op::PowI(_, k) if k <= 0 => consts.add(ONE_BITS),
                Op::Dir(_) | Op::Table(..) | Op::TableSlope(..) => consts.add(nan_bits),
                _ => {}
            }
        }
        let nc = consts.bits.len();
        let one = consts.at.get(&ONE_BITS).copied();
        let mut lo = Lower {
            ins: Vec::new(),
            tb: (nc + n + np1) as u32,
        };
        let mut slot: Vec<u32> = Vec::with_capacity(tape.ops.len());
        for (i, op) in tape.ops.iter().enumerate() {
            let s = {
                let c = |id: NodeId| {
                    let j = id.0 as usize;
                    assert!(j < i, "op {i} reads slot {j} before it is computed");
                    slot[j]
                };
                match *op {
                    Op::Const(b) => consts.at[&b],
                    Op::Var(v) => {
                        assert!((v as usize) < n, "op {i} reads state {v} of {n}");
                        (nc + v as usize) as u32
                    }
                    Op::Param(j) => {
                        assert!((j as usize) < np1, "op {i} reads parameter {j} of {np1}");
                        (nc + n + j as usize) as u32
                    }
                    Op::Dir(_) | Op::Table(..) | Op::TableSlope(..) => consts.at[&nan_bits],
                    Op::Neg(a) => lo.emit(Ins::Neg(c(a))),
                    Op::Add(a, b) => lo.emit(Ins::Add(c(a), c(b))),
                    Op::Sub(a, b) => lo.emit(Ins::Sub(c(a), c(b))),
                    Op::Mul(a, b) => lo.emit(Ins::Mul(c(a), c(b))),
                    Op::Div(a, b) => lo.emit(Ins::Div(c(a), c(b))),
                    Op::PowI(a, k) => lo.powi(c(a), k, one),
                    Op::Call1(f, a) => lo.emit(Ins::Call1(f, c(a))),
                    Op::Call2(f, a, b) => lo.emit(Ins::Call2(f, c(a), c(b))),
                    Op::Cmp(k, a, b) => lo.emit(Ins::Cmp(k, c(a), c(b))),
                    Op::And(a, b) => lo.emit(Ins::And(c(a), c(b))),
                    Op::Or(a, b) => lo.emit(Ins::Or(c(a), c(b))),
                    Op::Not(a) => lo.emit(Ins::Not(c(a))),
                    Op::Select(k, a, b) => lo.emit(Ins::Select(c(k), c(a), c(b))),
                }
            };
            slot.push(s);
        }
        let outputs = tape
            .outputs
            .iter()
            .map(|&o| {
                assert!(
                    (o as usize) < slot.len(),
                    "output slot {o} is not on the tape"
                );
                slot[o as usize]
            })
            .collect();
        CpuTape {
            n,
            np1,
            consts: consts
                .bits
                .iter()
                .map(|&b| T::from_f64(f64::from_bits(b)))
                .collect(),
            ins: lo.ins,
            outputs,
        }
    }

    /// A slab for this tape, with the constants in place.
    pub fn slab(&self) -> Slab<T> {
        let mut v = self.consts.clone();
        v.resize(self.slab_len(), T::ZERO);
        Slab(v)
    }

    fn slab_len(&self) -> usize {
        self.consts.len() + self.n + self.np1 + self.ins.len()
    }

    /// Loads the parameters into `slab`, once for a run of evaluations. `p` holds the `n_p`
    /// parameters, and may hold `t` after them, which this ignores.
    pub fn bind(&self, p: &[T], slab: &mut Slab<T>) {
        let ps = self.consts.len() + self.n;
        let np = self.np1 - 1;
        slab.0[ps..ps + np].copy_from_slice(&p[..np]);
    }

    /// Writes `f(t, x)` to `out`, with the parameters of the last [`CpuTape::bind`] on this
    /// slab. `slab` must come from [`CpuTape::slab`] of this tape.
    pub fn eval_xt(&self, x: &[T], t: T, slab: &mut Slab<T>, out: &mut [T]) {
        let s = slab.0.as_mut_slice();
        assert_eq!(s.len(), self.slab_len(), "the slab is from another tape");
        let xs = self.consts.len();
        let ps = xs + self.n;
        let tb = ps + self.np1;
        copy_states(&mut s[xs..ps], &x[..self.n]);
        s[tb - 1] = t;
        for (k, &ins) in self.ins.iter().enumerate() {
            // SAFETY: `new` checked every operand slot against the slab layout, and the
            // assert above made `s` that long. `tb + k` is below `s.len()` for the same reason.
            unsafe {
                let v = run(ins, s);
                *s.get_unchecked_mut(tb + k) = v;
            }
        }
        for (o, &i) in out.iter_mut().zip(&self.outputs) {
            // SAFETY: `new` checked every output slot against the instruction count.
            *o = unsafe { *s.get_unchecked(i as usize) };
        }
    }

    /// Writes `f(t, x)` to `out`. `p` holds the `n_p` parameters followed by `t`. `slab`
    /// must come from [`CpuTape::slab`] of this tape.
    pub fn eval(&self, x: &[T], p: &[T], slab: &mut Slab<T>, out: &mut [T]) {
        self.bind(p, slab);
        self.eval_xt(x, p[self.np1 - 1], slab, out);
    }

    /// [`CpuTape::eval_xt`] through a compiled `F`. The function reads `x` from the caller's
    /// slice and the parameters and `t` from the slab, where [`CpuTape::bind`] put them, so the
    /// slab's temporaries are not touched. Single precision runs [`CpuTape::eval_xt`] instead.
    ///
    /// Panics when `x` has fewer than `n` values or `out` fewer than the tape's outputs, before
    /// the call.
    #[inline]
    pub fn eval_native(&self, f: NativeRhs, x: &[T], t: T, slab: &mut Slab<T>, out: &mut [T]) {
        assert_eq!(
            slab.0.len(),
            self.slab_len(),
            "the slab is from another tape"
        );
        let (Some(xv), Some(ov)) = (T::as_f64s(x), T::as_f64s_mut(out)) else {
            return self.eval_xt(x, t, slab, out);
        };
        assert!(
            xv.len() >= self.n,
            "native rhs: {} states, the model has {}",
            xv.len(),
            self.n
        );
        assert!(
            ov.len() >= self.outputs.len(),
            "native rhs: room for {} outputs, the model has {}",
            ov.len(),
            self.outputs.len()
        );
        let Some(sv) = T::as_f64s_mut(slab.0.as_mut_slice()) else {
            unreachable!("x and out were f64, so the slab is");
        };
        let ps = self.consts.len() + self.n;
        sv[ps + self.np1 - 1] = t.to_f64();
        // SAFETY: `xv` has `n` values and `sv[ps..]` the `n_p` parameters and `t`, in the order
        // the function reads them; `ov` has room for every output. They are three distinct
        // slices, so `out` does not alias an input. The contract of `NativeRhs::new` covers the
        // rest.
        unsafe { f.call(xv.as_ptr(), sv[ps..].as_ptr(), ov.as_mut_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ir::dag::Dag;
    use pc_ir::eval::eval;
    use pc_ir::{Entry, EntryKind};
    use proptest::prelude::*;

    fn model(src: &str) -> IrModel {
        use pc_ode::{MapLoader, Severity, SourceMap, load};
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
        pc_ir::lower(&l.expanded, &l.info).expect("lower")
    }

    /// The f64 tape gives the same bits as the DAG interpreter.
    #[test]
    fn matches_the_interpreter_bit_for_bit() {
        let ir = model(
            "par a=0.7, b=2\n\
             x'=sin(x)*a+exp(-y)/(1+x^2)+b*x^5-abs(y)+if(x>y)then(max(x,y))else(min(x,y))\n\
             y'=atan2(x,y)+sqrt(1+y*y)+heav(x-y)+sign(y)+mod(x,1.3)+x^(-3)+t\n\
             done\n",
        );
        let tape = Tape::build(&ir).expect("tape");
        let mut vals = vec![0.0; tape.len()];
        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        for (x0, x1, t) in [(0.3, -1.2, 0.0), (-2.5, 4.0, 1.5), (0.0, 0.0, 2.0)] {
            let x = [x0, x1];
            let p = [0.7, 2.0, t];
            let mut got = [0.0; 2];
            tape.eval::<f64>(&x, &p, &mut vals, &mut got);
            let mut fast = [0.0; 2];
            cpu.eval(&x, &p, &mut slab, &mut fast);
            let mut want = [0.0; 2];
            eval(&ir.dag, &ir.f, &x, &p, &[], &mut want);
            for i in 0..2 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "state {i} at {x:?}: {} vs {}",
                    got[i],
                    want[i]
                );
                assert_eq!(
                    fast[i].to_bits(),
                    want[i].to_bits(),
                    "CpuTape, state {i} at {x:?}: {} vs {}",
                    fast[i],
                    want[i]
                );
            }
        }
    }

    /// Bits of `v` widened to `f64`, with every NaN the same: NaN payloads are not part of
    /// the result.
    fn bits<T: Real>(v: T) -> u64 {
        let f = v.to_f64();
        if f.is_nan() { u64::MAX } else { f.to_bits() }
    }

    /// Runs `tape` and its [`CpuTape`] in `T`. The slab first sees other inputs, so stale
    /// temporaries from an earlier call would show.
    fn check<T: Real>(tape: &Tape, x: &[f64], p: &[f64]) -> Result<(), String> {
        let x: Vec<T> = x.iter().map(|&v| T::from_f64(v)).collect();
        let p: Vec<T> = p.iter().map(|&v| T::from_f64(v)).collect();
        let mut vals = vec![T::ZERO; tape.len()];
        let mut want = vec![T::ZERO; tape.n];
        tape.eval(&x, &p, &mut vals, &mut want);
        let cpu = CpuTape::<T>::new(tape);
        let mut slab = cpu.slab();
        let xg: Vec<T> = x.iter().map(|&v| v + T::ONE).collect();
        let pg: Vec<T> = p.iter().map(|&v| v + T::ONE).collect();
        let mut got = vec![T::ZERO; tape.n];
        cpu.eval(&xg, &pg, &mut slab, &mut got);
        cpu.eval(&x, &p, &mut slab, &mut got);
        for i in 0..tape.n {
            if bits(got[i]) != bits(want[i]) {
                return Err(format!(
                    "{} output {i}: {:e} vs {:e}",
                    T::NAME,
                    got[i].to_f64(),
                    want[i].to_f64()
                ));
            }
        }
        Ok(())
    }

    const FN1: [Fn1; 21] = [
        Fn1::Sin,
        Fn1::Cos,
        Fn1::Tan,
        Fn1::Asin,
        Fn1::Acos,
        Fn1::Atan,
        Fn1::Sinh,
        Fn1::Cosh,
        Fn1::Tanh,
        Fn1::Exp,
        Fn1::Ln,
        Fn1::Log10,
        Fn1::Sqrt,
        Fn1::Abs,
        Fn1::Erf,
        Fn1::Erfc,
        Fn1::Lgamma,
        Fn1::Floor,
        Fn1::Ceil,
        Fn1::Heav,
        Fn1::Sign,
    ];
    const FN2: [Fn2; 8] = [
        Fn2::Atan2,
        Fn2::Pow,
        Fn2::Mod,
        Fn2::BesselJ,
        Fn2::BesselY,
        Fn2::Polygamma,
        Fn2::Min,
        Fn2::Max,
    ];
    const CMP: [Cmp; 5] = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq];
    const SPECIAL: [f64; 13] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -3.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        1e300,
        -1e-310,
    ];

    /// Kind, three child or index picks, a constant and a power, for one op.
    type RawOp = (u8, u32, u32, u32, f64, i32);

    /// A tape from raw picks. Every child id points back, every leaf is in range.
    fn build(n: usize, n_p: usize, raw: &[RawOp], outs: &[u32]) -> Tape {
        let mut ops = Vec::with_capacity(raw.len());
        for (i, &(kind, a, b, c, v, k)) in raw.iter().enumerate() {
            let i = i as u32;
            let id = |x: u32| NodeId(x % i);
            let op = if i == 0 || (kind & 0x0f) < 3 {
                match b % 7 {
                    0 | 1 => Op::Const(v.to_bits()),
                    2 | 3 => Op::Var(a % n as u32),
                    4 | 5 => Op::Param(a % (n_p as u32 + 1)),
                    _ => Op::Dir(0),
                }
            } else {
                match (kind >> 4) % 13 {
                    0 => Op::Neg(id(a)),
                    1 => Op::Add(id(a), id(b)),
                    2 => Op::Sub(id(a), id(b)),
                    3 => Op::Mul(id(a), id(b)),
                    4 => Op::Div(id(a), id(b)),
                    5 => Op::PowI(id(a), k),
                    6 => Op::Call1(FN1[b as usize % FN1.len()], id(a)),
                    7 => Op::Call2(FN2[c as usize % FN2.len()], id(a), id(b)),
                    8 => Op::Cmp(CMP[c as usize % CMP.len()], id(a), id(b)),
                    9 => Op::And(id(a), id(b)),
                    10 => Op::Or(id(a), id(b)),
                    11 => Op::Not(id(a)),
                    _ => Op::Select(id(a), id(b), id(c)),
                }
            };
            ops.push(op);
        }
        let outputs = outs.iter().map(|&o| o % ops.len() as u32).collect();
        Tape {
            n,
            n_p,
            ops,
            outputs,
        }
    }

    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![
            -4.0f64..4.0,
            any::<f64>(),
            proptest::sample::select(SPECIAL.to_vec()),
        ]
    }

    fn case() -> impl Strategy<Value = (Tape, Vec<f64>, Vec<f64>)> {
        let raw = (
            any::<u8>(),
            any::<u32>(),
            any::<u32>(),
            any::<u32>(),
            value(),
            -18i32..=18,
        );
        (1usize..4, 0usize..3, proptest::collection::vec(raw, 1..40)).prop_flat_map(
            |(n, n_p, raw)| {
                (
                    proptest::collection::vec(any::<u32>(), n),
                    proptest::collection::vec(value(), n),
                    proptest::collection::vec(value(), n_p + 1),
                )
                    .prop_map(move |(outs, x, p)| (build(n, n_p, &raw, &outs), x, p))
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// The lowered tape gives the bits of the tape, op by op, in both precisions.
        #[test]
        fn cpu_tape_matches_the_tape_in_both_precisions((tape, x, p) in case()) {
            prop_assert_eq!(check::<f64>(&tape, &x, &p), Ok(()));
            prop_assert_eq!(check::<f32>(&tape, &x, &p), Ok(()));
        }
    }

    /// A [`Dag`] built from raw picks through its own constructors, and the [`Tape`] over all
    /// its nodes in node order, so a node id is its slot. The constructors fold and
    /// simplify; the tape and the JIT both start from the result.
    ///
    /// Only ops that `Tape::build` accepts: it refuses `Dir`, tables and the special
    /// functions, the `Tape` evaluates them as NaN and the JIT would call the real function.
    fn dag_case(n: usize, n_p: usize, raw: &[RawOp], outs: &[u32]) -> (Dag, Tape) {
        let node = NodeId(0);
        let fn1: Vec<Fn1> = FN1
            .iter()
            .copied()
            .filter(|&f| unsupported_op(&Op::Call1(f, node)).is_none())
            .collect();
        let fn2: Vec<Fn2> = FN2
            .iter()
            .copied()
            .filter(|&f| unsupported_op(&Op::Call2(f, node, node)).is_none())
            .collect();
        let mut dag = Dag::new(4096);
        let mut ids: Vec<NodeId> = Vec::new();
        for (i, &(kind, a, b, c, v, k)) in raw.iter().enumerate() {
            let id = if i == 0 || (kind & 0x0f) < 3 {
                match b % 6 {
                    0 | 1 => dag.konst(v),
                    2 | 3 => dag.var(a % n as u32),
                    _ => dag.param(a % (n_p as u32 + 1)),
                }
            } else {
                let pick = |x: u32| ids[x as usize % ids.len()];
                let (x, y, z) = (pick(a), pick(b), pick(c));
                match (kind >> 4) % 13 {
                    0 => dag.neg(x),
                    1 => dag.add(x, y),
                    2 => dag.sub(x, y),
                    3 => dag.mul(x, y),
                    4 => dag.div(x, y),
                    5 => dag.powi(x, k),
                    6 => dag.call1(fn1[b as usize % fn1.len()], x),
                    7 => dag.call2(fn2[c as usize % fn2.len()], x, y),
                    8 => dag.cmp(CMP[c as usize % CMP.len()], x, y),
                    9 => dag.and(x, y),
                    10 => dag.or(x, y),
                    11 => dag.not(x),
                    _ => dag.select(x, y, z),
                }
            };
            ids.push(id);
        }
        let ops = (0..dag.len()).map(|i| *dag.op(NodeId(i as u32))).collect();
        let outputs = outs
            .iter()
            .map(|&o| ids[o as usize % ids.len()].0)
            .collect();
        let tape = Tape {
            n,
            n_p,
            ops,
            outputs,
        };
        (dag, tape)
    }

    /// Compiles the outputs of `tape` as an `F` entry of `dag` and runs it through
    /// [`CpuTape::eval_native`], against [`CpuTape::eval_xt`] and [`Tape::eval`]. The slab first
    /// sees other inputs, so a stale temporary or parameter would show.
    fn check_native(dag: &Dag, tape: &Tape, x: &[f64], p: &[f64]) -> Result<(), String> {
        let entry = Entry {
            kind: EntryKind::F,
            outputs: tape.outputs.iter().map(|&o| NodeId(o)).collect(),
        };
        let jit = pc_jit::compile(&[(EntryKind::F, &entry)], dag).map_err(|e| e.to_string())?;
        let f = jit.plain(&EntryKind::F).ok_or("no F entry")?;
        // SAFETY: the entry is the `F` of `dag` with `tape.n` outputs, and the JIT and the
        // tape give the same bits (this test); it is called on one thread, and `jit` outlives
        // every call below.
        let native = unsafe { NativeRhs::new(f) };
        let cpu = CpuTape::<f64>::new(tape);
        let mut slab = cpu.slab();
        let t = p[tape.n_p];

        let mut vals = vec![0.0; tape.len()];
        let mut interp = vec![0.0; tape.n];
        tape.eval(x, p, &mut vals, &mut interp);

        let mut by_tape = vec![0.0; tape.n];
        cpu.eval(x, p, &mut slab, &mut by_tape);

        let (xg, pg): (Vec<f64>, Vec<f64>) = (
            x.iter().map(|v| v + 1.0).collect(),
            p.iter().map(|v| v + 1.0).collect(),
        );
        let mut got = vec![f64::MAX; tape.n];
        cpu.bind(&pg, &mut slab);
        cpu.eval_native(native, &xg, pg[tape.n_p], &mut slab, &mut got);
        cpu.bind(p, &mut slab);
        cpu.eval_native(native, x, t, &mut slab, &mut got);
        for i in 0..tape.n {
            for (what, want) in [("CpuTape", by_tape[i]), ("Tape", interp[i])] {
                if bits(got[i]) != bits(want) {
                    return Err(format!("output {i}: jit {:e} vs {what} {:e}", got[i], want));
                }
            }
        }
        Ok(())
    }

    /// `(n, n_p, raw ops, output picks, x, p)`. A `Dag` has no `Debug`, so the case is built in
    /// the test body.
    type NativeIn = (usize, usize, Vec<RawOp>, Vec<u32>, Vec<f64>, Vec<f64>);

    fn native_case() -> impl Strategy<Value = NativeIn> {
        let raw = (
            any::<u8>(),
            any::<u32>(),
            any::<u32>(),
            any::<u32>(),
            value(),
            -18i32..=18,
        );
        (1usize..4, 0usize..3, proptest::collection::vec(raw, 1..40)).prop_flat_map(
            |(n, n_p, raw)| {
                (
                    Just(n),
                    Just(n_p),
                    Just(raw),
                    proptest::collection::vec(any::<u32>(), n),
                    proptest::collection::vec(value(), n),
                    proptest::collection::vec(value(), n_p + 1),
                )
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        /// The compiled `F` gives the bits of `CpuTape` and `Tape`, for every op the tape takes
        /// (SPEC 13, the `cpu` backend). Slow: each case compiles a module.
        #[test]
        fn the_jit_matches_cpu_tape_bit_for_bit((n, n_p, raw, outs, x, p) in native_case()) {
            let (dag, tape) = dag_case(n, n_p, &raw, &outs);
            prop_assert_eq!(check_native(&dag, &tape, &x, &p), Ok(()));
        }
    }

    /// The compiled `F` of a model with `if`, `min`, `max`, integer powers and `t`.
    #[test]
    fn the_jit_runs_a_model_like_the_tape() {
        let ir = model(
            "par a=0.7, b=2\n\
             x'=sin(x)*a+exp(-y)/(1+x^2)+b*x^5-abs(y)+if(x>y)then(max(x,y))else(min(x,y))\n\
             y'=atan2(x,y)+sqrt(1+y*y)+heav(x-y)+sign(y)+mod(x,1.3)+x^(-3)+t\n\
             done\n",
        );
        let tape = Tape::build(&ir).expect("tape");
        let entry = Entry {
            kind: EntryKind::F,
            outputs: ir.f.clone(),
        };
        let jit = pc_jit::compile(&[(EntryKind::F, &entry)], &ir.dag).expect("compile");
        let f = jit.plain(&EntryKind::F).expect("F");
        // SAFETY: the `F` of `ir` with two outputs, on one thread, and `jit` outlives the loop.
        let native = unsafe { NativeRhs::new(f) };
        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        cpu.bind(&[0.7, 2.0, 0.0], &mut slab);
        for (x, y, t) in [
            (0.3, -1.2, 0.0),
            (2.0, 2.0, 5.5),
            (-0.0, 0.0, -1.0),
            (1e5, -1e5, 1.0),
        ] {
            let (mut want, mut got) = ([0.0; 2], [0.0; 2]);
            cpu.eval_xt(&[x, y], t, &mut slab, &mut want);
            cpu.eval_native(native, &[x, y], t, &mut slab, &mut got);
            assert_eq!(bits(got[0]), bits(want[0]), "x at ({x},{y},{t})");
            assert_eq!(bits(got[1]), bits(want[1]), "y at ({x},{y},{t})");
        }
    }

    /// Single precision never calls the compiled function.
    #[test]
    fn single_precision_ignores_the_native_function() {
        unsafe extern "C" fn poison(_: *const f64, _: *const f64, out: *mut f64) {
            // SAFETY: the caller gave one output.
            unsafe { *out = 12345.0 };
        }
        let ir = model("x'=sin(x)+1\ndone\n");
        let tape = Tape::build(&ir).expect("tape");
        // SAFETY: never called in single precision, which is what this test checks.
        let native = unsafe { NativeRhs::new(poison) };
        let cpu = CpuTape::<f32>::new(&tape);
        let mut slab = cpu.slab();
        cpu.bind(&[0.0], &mut slab);
        let (mut want, mut got) = ([0.0f32], [0.0f32]);
        cpu.eval_xt(&[0.5], 0.0, &mut slab, &mut want);
        cpu.eval_native(native, &[0.5], 0.0, &mut slab, &mut got);
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }

    /// A state or output slice that is too short is a panic, not a read past the end.
    #[test]
    #[should_panic(expected = "native")]
    fn a_short_state_slice_is_refused_before_the_call() {
        unsafe extern "C" fn never(_: *const f64, _: *const f64, _: *mut f64) {
            std::process::abort();
        }
        let ir = model("x'=x\ny'=y\ndone\n");
        let tape = Tape::build(&ir).expect("tape");
        // SAFETY: refused before the call, which is what this test checks.
        let native = unsafe { NativeRhs::new(never) };
        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        cpu.bind(&[0.0], &mut slab);
        let mut out = [0.0; 2];
        cpu.eval_native(native, &[1.0], 0.0, &mut slab, &mut out);
    }

    /// The written-out chain of `PowI` is the chain of [`powi`], including `k` of 0, 1 and
    /// the ends of `i32`.
    #[test]
    fn integer_powers_take_the_same_multiply_chain() {
        let xs = [
            0.7,
            -1.3,
            0.0,
            -0.0,
            3e-160,
            1e155,
            f64::INFINITY,
            f64::NAN,
            5e-324,
            1.0000001,
        ];
        for k in (-33..=33).chain([i32::MIN, i32::MIN + 1, i32::MAX]) {
            let tape = Tape {
                n: 1,
                n_p: 0,
                ops: vec![Op::Var(0), Op::PowI(NodeId(0), k)],
                outputs: vec![1],
            };
            let cpu = CpuTape::<f64>::new(&tape);
            let mut slab = cpu.slab();
            for x in xs {
                let mut got = [0.0];
                cpu.eval(&[x], &[0.0], &mut slab, &mut got);
                let want = powi(x, k);
                assert_eq!(
                    bits(got[0]),
                    bits(want),
                    "{x:e}^{k}: {:e} vs {want:e}",
                    got[0]
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "before it is computed")]
    fn a_forward_reference_is_refused() {
        let tape = Tape {
            n: 1,
            n_p: 0,
            ops: vec![Op::Neg(NodeId(1)), Op::Var(0)],
            outputs: vec![0],
        };
        let _ = CpuTape::<f64>::new(&tape);
    }

    #[test]
    fn single_precision_is_close_to_double() {
        let ir = model("x'=sin(x)+exp(-x*x)\ndone\n");
        let tape = Tape::build(&ir).expect("tape");
        let mut v32 = vec![0.0f32; tape.len()];
        let mut v64 = vec![0.0f64; tape.len()];
        let (mut o32, mut o64) = ([0.0f32], [0.0f64]);
        tape.eval::<f32>(&[0.7], &[0.0], &mut v32, &mut o32);
        tape.eval::<f64>(&[0.7], &[0.0], &mut v64, &mut o64);
        assert!((f64::from(o32[0]) - o64[0]).abs() < 1e-6);
    }

    #[test]
    fn refuses_constructs_by_name() {
        for (src, name) in [
            ("x'=erf(x)\ndone\n", "erf"),
            ("x'=lgamma(x)+1\ndone\n", "lgamma"),
            ("x'=besselj(1,x)\ndone\n", "besselj"),
        ] {
            let ir = model(src);
            match Tape::build(&ir) {
                Err(EnsembleError::Unsupported(m)) => {
                    assert!(m.contains(name), "{m}");
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }
}
