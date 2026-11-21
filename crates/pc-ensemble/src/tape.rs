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

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use std::fmt::Write as _;

use pc_codegen::kernel::{KernelError, Scalar, float_literal, op_expr, unsupported_op};
use pc_ir::IrModel;
use pc_ir::dag::{Cmp, Fn1, Fn2, NodeId, Op};
use pc_ir::eval::live_set;

use crate::job::{EnsembleError, NativeRhs, StackLayout};

const SEVERAL_OUTPUTS: &str = "function with several outputs";
use crate::real::Real;

#[derive(Clone, Debug)]
pub struct Tape {
    pub n: usize,
    pub n_p: usize,
    /// Ops in evaluation order. A child's id is the index of its slot.
    pub ops: Vec<Op>,
    /// The slot of each state's right-hand side.
    pub outputs: Vec<u32>,
    /// The argument lists of the [`Op::Call`] ops, as tape slots: `Call(u, l)` reads list
    /// `l`. Empty for a tape without calls.
    pub args: Vec<Vec<u32>>,
    /// The functions the calls reach: `Call(u, l)` calls unit `u`. Empty for a tape without
    /// calls, which is the only kind the GPU kernels and [`Tape::eval`] run.
    pub units: Vec<TapeUnit>,
}

/// A function of a [`Tape`] with calls (SPEC section 8af-5).
#[derive(Clone, Debug)]
pub struct TapeUnit {
    /// The function name, for messages.
    pub name: String,
    pub arity: u32,
    /// The slot of the body's value. The body reads its formals through [`Op::Arg`].
    pub out: u32,
}

fn remap(op: &Op, m: &[u32]) -> Op {
    let r = |id: NodeId| NodeId(m[id.0 as usize]);
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => *op,
        // `Tape::build` remaps a call through `build_calls` and refuses `Out`.
        Op::Arg(_) | Op::Call(..) | Op::Out(..) => *op,
        Op::If(c, a, b) => Op::If(r(c), r(a), r(b)),
        Op::Neg(a) => Op::Neg(r(a)),
        Op::Add(a, b) => Op::Add(r(a), r(b)),
        Op::Sub(a, b) => Op::Sub(r(a), r(b)),
        Op::Mul(a, b) => Op::Mul(r(a), r(b)),
        Op::ZMul(a, b) => Op::ZMul(r(a), r(b)),
        Op::Div(a, b) => Op::Div(r(a), r(b)),
        Op::ZDiv(a, b) => Op::ZDiv(r(a), r(b)),
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
    ///
    /// A right-hand side that calls a recursive function gets the bodies of the functions
    /// it reaches on the same tape, with [`Tape::units`] and [`Tape::args`] filled in. The
    /// CPU loop and the GPU kernels run such a tape.
    pub fn build(ir: &IrModel) -> Result<Tape, EnsembleError> {
        if ir.f.iter().any(|&id| ir.dag.has_call(id)) {
            return Self::build_calls(ir);
        }
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
            args: Vec::new(),
            units: Vec::new(),
        })
    }

    /// [`Tape::build`] for a right-hand side with calls: the nodes reachable from `ir.f`
    /// and from the body of every function a reachable call names, in id order.
    fn build_calls(ir: &IrModel) -> Result<Tape, EnsembleError> {
        let dag = &ir.dag;
        let outside = || {
            EnsembleError::BadJob(
                "a right-hand side is outside the model's expression graph".into(),
            )
        };
        if ir.f.iter().any(|id| id.0 as usize >= dag.len()) {
            return Err(outside());
        }
        let mut live = vec![false; dag.len()];
        let mut seen: Vec<bool> = vec![false; dag.units().len()];
        let mut names: Vec<&'static str> = Vec::new();
        let mut work: Vec<NodeId> = ir.f.clone();
        while let Some(id) = work.pop() {
            let i = id.0 as usize;
            if i >= dag.len() {
                return Err(outside());
            }
            if std::mem::replace(&mut live[i], true) {
                continue;
            }
            let op = dag.op(id);
            if let Op::Call(u, l) = *op {
                let unit = dag.unit(u).ok_or_else(|| {
                    EnsembleError::BadJob("a call names a function the model does not have".into())
                })?;
                let given = dag.args(l).len();
                if given != unit.arity as usize {
                    return Err(EnsembleError::BadJob(format!(
                        "a call passes {given} arguments to function {}, which takes {}",
                        unit.name, unit.arity
                    )));
                }
                if !std::mem::replace(&mut seen[u as usize], true) {
                    if unit.nout != 1 && !names.contains(&SEVERAL_OUTPUTS) {
                        names.push(SEVERAL_OUTPUTS);
                    }
                    let body = unit.outputs().ok_or_else(|| {
                        EnsembleError::BadJob(format!("function {} has no body", unit.name))
                    })?;
                    work.extend_from_slice(body);
                }
            }
            dag.operands(op, &mut |c| work.push(c));
        }
        for (i, _) in live.iter().enumerate().filter(|(_, l)| **l) {
            let name = match dag.op(NodeId(i as u32)) {
                Op::Arg(_) | Op::Call(..) | Op::If(..) => None,
                Op::Out(..) => Some(SEVERAL_OUTPUTS),
                op => unsupported_op(op),
            };
            if let Some(name) = name
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
        if !names.is_empty() {
            return Err(EnsembleError::Unsupported(format!(
                "the right-hand side uses {}, which the ensemble engine cannot run",
                names.join(", ")
            )));
        }
        // Units keep their order in the DAG.
        let mut unit_at = vec![u32::MAX; seen.len()];
        let mut units_src: Vec<u32> = Vec::new();
        for (u, _) in seen.iter().enumerate().filter(|(_, s)| **s) {
            unit_at[u] = units_src.len() as u32;
            units_src.push(u as u32);
        }
        let mut map = vec![u32::MAX; dag.len()];
        let mut ops = Vec::new();
        let mut args: Vec<Vec<u32>> = Vec::new();
        for (i, _) in live.iter().enumerate().filter(|(_, l)| **l) {
            let op = dag.op(NodeId(i as u32));
            let new = match *op {
                Op::Call(u, l) => {
                    let list = dag.args(l).iter().map(|a| map[a.0 as usize]).collect();
                    args.push(list);
                    Op::Call(unit_at[u as usize], (args.len() - 1) as u32)
                }
                _ => remap(op, &map),
            };
            map[i] = ops.len() as u32;
            ops.push(new);
        }
        let units = units_src
            .iter()
            .map(|&u| {
                let unit = &dag.units()[u as usize];
                let root = unit.outputs().map_or(u32::MAX, |b| map[b[0].0 as usize]);
                TapeUnit {
                    name: unit.name.clone(),
                    arity: unit.arity,
                    out: root,
                }
            })
            .collect();
        let outputs = ir.f.iter().map(|id| map[id.0 as usize]).collect();
        let tape = Tape {
            n: ir.n(),
            n_p: ir.n_p(),
            ops,
            outputs,
            args,
            units,
        };
        tape.check_contexts()?;
        Ok(tape)
    }

    /// Checks that each formal is read inside a function that has it: an [`Op::Arg`] that
    /// the right-hand side reaches outside every body, or a formal past the arity of the
    /// function that reads it, is a malformed job.
    fn check_contexts(&self) -> Result<(), EnsembleError> {
        let check = |roots: &[u32], arity: Option<&TapeUnit>| -> Result<(), EnsembleError> {
            let mark = self.reach(roots);
            for (i, _) in mark.iter().enumerate().filter(|(_, m)| **m) {
                if let Op::Arg(k) = self.ops[i] {
                    match arity {
                        None => {
                            return Err(EnsembleError::BadJob(
                                "a right-hand side reads a function argument outside a function"
                                    .into(),
                            ));
                        }
                        Some(u) if k >= u.arity => {
                            return Err(EnsembleError::BadJob(format!(
                                "function {} reads argument {} of {}",
                                u.name,
                                k + 1,
                                u.arity
                            )));
                        }
                        Some(_) => {}
                    }
                }
            }
            Ok(())
        };
        check(&self.outputs, None)?;
        for u in &self.units {
            check(&[u.out], Some(u))?;
        }
        Ok(())
    }

    /// Calls `f` on every operand of `op`, the arguments of a call included. The body of a
    /// called function is not an operand.
    pub fn operands(&self, op: &Op, f: &mut dyn FnMut(u32)) {
        match *op {
            Op::Call(_, l) => {
                for &a in &self.args[l as usize] {
                    f(a);
                }
            }
            _ => pc_ir::eval::children(op, &mut |c| f(c.0)),
        }
    }

    /// Marks the ops one frame evaluates for `roots`: those reachable through operands, call
    /// arguments included, without entering a called body.
    pub fn reach(&self, roots: &[u32]) -> Vec<bool> {
        let mut mark = vec![false; self.ops.len()];
        for &r in roots {
            mark[r as usize] = true;
        }
        for i in (0..self.ops.len()).rev() {
            if mark[i] {
                self.operands(&self.ops[i], &mut |c| mark[c as usize] = true);
            }
        }
        mark
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Writes `f(t, x)` to `out`. `p` holds the `n_p` parameters followed by `t`. `vals` is
    /// scratch of at least [`Tape::len`] values. Only a tape without calls runs here; a tape
    /// with calls runs through [`CpuTape`].
    pub fn eval<T: Real>(&self, x: &[T], p: &[T], vals: &mut [T], out: &mut [T]) {
        assert!(
            self.units.is_empty(),
            "Tape::eval runs only a tape without calls"
        );
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
        Op::ZMul(u, w) => zmul(a(u), a(w)),
        Op::Div(u, w) => a(u) / a(w),
        Op::ZDiv(u, w) => zdiv(a(u), a(w)),
        Op::PowI(u, n) => powi(a(u), n),
        Op::Call1(f, u) => call1(f, a(u)),
        Op::Call2(f, u, w) => call2(f, a(u), a(w)),
        Op::Cmp(c, u, w) => cmp(c, a(u), a(w)),
        Op::And(u, w) => and(a(u), a(w)),
        Op::Or(u, w) => or(a(u), a(w)),
        Op::Not(u) => not(a(u)),
        Op::Select(c, u, w) => select(a(c), a(u), a(w)),
        // Refused when the tape is built.
        Op::Dir(_)
        | Op::Table(..)
        | Op::TableSlope(..)
        | Op::Arg(_)
        | Op::Call(..)
        | Op::If(..)
        | Op::Out(..) => T::NAN,
    }
}

/// `+0` when `t == 0` (either sign), else `t * p`: the rule of `Op::ZMul`.
fn zmul<T: Real>(t: T, p: T) -> T {
    if t == T::ZERO { T::ZERO } else { t * p }
}

/// `+0` when `n == 0` (either sign), else `n / d`: the rule of `Op::ZDiv`.
fn zdiv<T: Real>(n: T, d: T) -> T {
    if n == T::ZERO { T::ZERO } else { n / d }
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
    ZMul(u32, u32),
    Div(u32, u32),
    ZDiv(u32, u32),
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
    apply(ins, |i: u32| unsafe { *s.get_unchecked(i as usize) })
}

/// Runs one instruction with its operands read through `a`.
#[inline(always)]
fn apply<T: Real>(ins: Ins, a: impl Fn(u32) -> T) -> T {
    match ins {
        Ins::Neg(u) => -a(u),
        Ins::Add(u, w) => a(u) + a(w),
        Ins::Sub(u, w) => a(u) - a(w),
        Ins::Mul(u, w) => a(u) * a(w),
        Ins::ZMul(u, w) => zmul(a(u), a(w)),
        Ins::Div(u, w) => a(u) / a(w),
        Ins::ZDiv(u, w) => zdiv(a(u), a(w)),
        Ins::Call1(f, u) => call1(f, a(u)),
        Ins::Call2(f, u, w) => call2(f, a(u), a(w)),
        Ins::Cmp(c, u, w) => cmp(c, a(u), a(w)),
        Ins::And(u, w) => and(a(u), a(w)),
        Ins::Or(u, w) => or(a(u), a(w)),
        Ins::Not(u) => not(a(u)),
        Ins::Select(c, u, w) => select(a(c), a(u), a(w)),
    }
}

/// The instruction of an arithmetic op, with each operand mapped through `s` in order, or
/// `None` for an op that is not one instruction.
fn lower_op(op: &Op, s: &mut dyn FnMut(NodeId) -> u32) -> Option<Ins> {
    Some(match *op {
        Op::Neg(a) => Ins::Neg(s(a)),
        Op::Add(a, b) => Ins::Add(s(a), s(b)),
        Op::Sub(a, b) => Ins::Sub(s(a), s(b)),
        Op::Mul(a, b) => Ins::Mul(s(a), s(b)),
        Op::ZMul(a, b) => Ins::ZMul(s(a), s(b)),
        Op::Div(a, b) => Ins::Div(s(a), s(b)),
        Op::ZDiv(a, b) => Ins::ZDiv(s(a), s(b)),
        Op::Call1(f, a) => Ins::Call1(f, s(a)),
        Op::Call2(f, a, b) => Ins::Call2(f, s(a), s(b)),
        Op::Cmp(k, a, b) => Ins::Cmp(k, s(a), s(b)),
        Op::And(a, b) => Ins::And(s(a), s(b)),
        Op::Or(a, b) => Ins::Or(s(a), s(b)),
        Op::Not(a) => Ins::Not(s(a)),
        Op::Select(k, a, b) => Ins::Select(s(k), s(a), s(b)),
        _ => return None,
    })
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
    /// The call program of a tape with calls; `ins` is then its global prologue.
    prog: Option<Box<Prog>>,
}

/// Scratch of a [`CpuTape`], made by [`CpuTape::slab`] with the constants in place. A slab
/// belongs to the tape that made it.
#[derive(Clone, Debug)]
pub struct Slab<T: Real> {
    v: Vec<T>,
    /// The stacks of a tape with calls.
    calls: Option<Box<CallState>>,
    /// Set when an evaluation since the last [`CpuTape::bind`] passed the call limit or ran
    /// out of memory for frames. The right-hand sides are then NaN.
    failed: bool,
}

impl<T: Real> Slab<T> {
    /// Whether an evaluation since the last [`CpuTape::bind`] passed the call limit or ran
    /// out of memory for its frames. The member then stops with status `recursionDepth`.
    pub fn failed(&self) -> bool {
        self.failed
    }

    /// The frames entered and the most frames active at once in the last evaluation of a
    /// tape with calls, or `None` for a tape without calls. An evaluation that failed reports
    /// the counts at the point it stopped.
    pub fn depth(&self) -> Option<(u64, u32)> {
        self.calls.as_ref().map(|c| (c.calls, c.depth))
    }
}

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

thread_local! {
    /// The frame cap a test set for the call tapes this thread builds: 0 for the default.
    static FORCED_CAP: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` with every call tape that this thread builds capped at `cap` frames instead of
/// `MAX_ACTIVE_CALLS`, then restores the cap, even when `f` panics. The GPU shim tests use it
/// to compare a kernel with the CPU at the GPU cap of 4,096 frames. It is not part of the API.
#[doc(hidden)]
pub fn with_max_calls<R>(cap: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED_CAP.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FORCED_CAP.with(|c| c.replace(cap)));
    f()
}

impl<T: Real> CpuTape<T> {
    /// Lowers `tape`. Panics when an op reads a slot that is not computed yet, or a state or
    /// parameter the tape does not have; [`Tape::build`] never makes such a tape.
    pub fn new(tape: &Tape) -> Self {
        if !tape.units.is_empty() {
            return Self::new_calls(tape);
        }
        let (n, np1) = (tape.n, tape.n_p + 1);
        let nan_bits = f64::NAN.to_bits();
        let mut consts = Consts::default();
        for op in &tape.ops {
            match *op {
                Op::Const(b) => consts.add(b),
                Op::PowI(_, k) if k <= 0 => consts.add(ONE_BITS),
                Op::Dir(_)
                | Op::Table(..)
                | Op::TableSlope(..)
                | Op::Arg(_)
                | Op::Call(..)
                | Op::If(..)
                | Op::Out(..) => consts.add(nan_bits),
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
                    Op::Dir(_)
                    | Op::Table(..)
                    | Op::TableSlope(..)
                    | Op::Arg(_)
                    | Op::Call(..)
                    | Op::If(..)
                    | Op::Out(..) => consts.at[&nan_bits],
                    Op::PowI(a, k) => lo.powi(c(a), k, one),
                    _ => {
                        let ins = lower_op(op, &mut |id| c(id)).expect("an arithmetic op");
                        lo.emit(ins)
                    }
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
            prog: None,
        }
    }

    /// A slab for this tape, with the constants in place.
    pub fn slab(&self) -> Slab<T> {
        let mut v = self.consts.clone();
        v.resize(self.slab_len(), T::ZERO);
        Slab {
            v,
            calls: self.prog.as_ref().map(|p| {
                Box::new(CallState {
                    flags: vec![0; p.nflag_global],
                    frames: Vec::new(),
                    gosub: Vec::new(),
                    calls: 0,
                    depth: 0,
                    #[cfg(test)]
                    entered: 0,
                    #[cfg(test)]
                    peak: 0,
                })
            }),
            failed: false,
        }
    }

    fn slab_len(&self) -> usize {
        match &self.prog {
            Some(p) => p.glen,
            None => self.consts.len() + self.n + self.np1 + self.ins.len(),
        }
    }

    /// Sets the most frames a right-hand side of a tape with calls may hold at once: the
    /// kernel cap for the depth keys, or a small cap in tests of the limit. A tape without
    /// calls ignores it.
    pub(crate) fn set_max_calls(&mut self, cap: usize) {
        if let Some(p) = &mut self.prog {
            p.cap = cap;
        }
    }

    /// Loads the parameters into `slab`, once for a run of evaluations. `p` holds the `n_p`
    /// parameters, and may hold `t` after them, which this ignores.
    pub fn bind(&self, p: &[T], slab: &mut Slab<T>) {
        let ps = self.consts.len() + self.n;
        let np = self.np1 - 1;
        slab.v[ps..ps + np].copy_from_slice(&p[..np]);
        slab.failed = false;
    }

    /// Writes `f(t, x)` to `out`, with the parameters of the last [`CpuTape::bind`] on this
    /// slab. `slab` must come from [`CpuTape::slab`] of this tape.
    pub fn eval_xt(&self, x: &[T], t: T, slab: &mut Slab<T>, out: &mut [T]) {
        self.eval_xt_with(x, t, slab, out, None);
    }

    /// [`CpuTape::eval_xt`] that also stops a tape with calls soon after `cancel` is set,
    /// once every `CANCEL_POLL` frames, and leaves the slab failed as the call limit does.
    /// The caller tells the two apart by reading the flag. A tape without calls ignores it.
    #[inline]
    pub fn eval_xt_with(
        &self,
        x: &[T],
        t: T,
        slab: &mut Slab<T>,
        out: &mut [T],
        cancel: Option<&AtomicBool>,
    ) {
        if let Some(prog) = &self.prog {
            return self.eval_calls(prog, x, t, slab, out, cancel);
        }
        let s = slab.v.as_mut_slice();
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
    /// slab's temporaries are not touched. A function of the other precision is not called;
    /// [`CpuTape::eval_xt`] runs instead.
    ///
    /// Panics when `x` has fewer than `n` values or `out` fewer than the tape's outputs, before
    /// the call.
    #[inline]
    pub fn eval_native(&self, f: NativeRhs, x: &[T], t: T, slab: &mut Slab<T>, out: &mut [T]) {
        assert_eq!(
            slab.v.len(),
            self.slab_len(),
            "the slab is from another tape"
        );
        let Some(g) = T::native(f).filter(|_| self.prog.is_none()) else {
            return self.eval_xt(x, t, slab, out);
        };
        assert!(
            x.len() >= self.n,
            "native rhs: {} states, the model has {}",
            x.len(),
            self.n
        );
        assert!(
            out.len() >= self.outputs.len(),
            "native rhs: room for {} outputs, the model has {}",
            out.len(),
            self.outputs.len()
        );
        let ps = self.consts.len() + self.n;
        let sv = &mut slab.v[ps..];
        sv[self.np1 - 1] = t;
        // SAFETY: `x` has `n` values and `sv` the `n_p` parameters and `t`, in the order the
        // function reads them; `out` has room for every output. They are three distinct
        // slices, so `out` does not alias an input. The contract of `NativeRhs::new` (or
        // `new_f32`, which `T::native` matched to `T`) covers the rest.
        unsafe { g(x.as_ptr(), sv.as_ptr(), out.as_mut_ptr()) };
    }
}

// The call program of a tape with calls (SPEC section 8af-5, stage 4a).
//
// Ops without a call below them run eagerly: those that do not read a formal once per
// right-hand side in the global prologue (`CpuTape::ins`), and those that do once per frame
// at the entry of their function. An op with a call below it runs only when the evaluation
// reaches it, in operand order, and at most once: per right-hand side when it reads no
// formal, per frame otherwise. That is the order and the memo of `pc_ir::machine::run`, so
// the two make the same calls at the same depths and reach the same values.
//
// A slot or flag operand with the `LOCAL` bit set is relative to the current frame.

/// Marks a slot or flag that is relative to the current frame.
const LOCAL: u32 = 1 << 31;

#[inline(always)]
fn res(o: u32, base: usize) -> usize {
    (o & !LOCAL) as usize + (base & 0usize.wrapping_sub((o >> 31) as usize))
}

/// One step of a [`Prog`]. Jump targets are step indices.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// Writes the instruction's value to the slot.
    Op(u32, Ins),
    Move(u32, u32),
    Jmp(u32),
    /// Jumps when the slot holds zero. NaN does not jump, the way `If` takes its first arm.
    JmpIfZero(u32, u32),
    /// Runs the subroutine at the target unless the flag is set.
    Sub(u32, u32),
    /// Sets the flag and returns from a subroutine.
    Rts(u32),
    /// Calls the unit with the arguments at `pool[lo..]`, and writes its value to the slot.
    Invoke(u32, u32, u32),
    /// Returns the slot's value from a unit.
    Ret(u32),
    End,
}

#[derive(Clone, Debug)]
struct ProgUnit {
    entry: u32,
    arity: u32,
    /// Slots of a frame: the formals, then the frame's own values.
    flen: u32,
    nflags: u32,
}

#[derive(Clone, Debug)]
struct Prog {
    steps: Vec<Step>,
    /// The argument slots of each `Invoke`.
    pool: Vec<u32>,
    units: Vec<ProgUnit>,
    entry: u32,
    nflag_global: usize,
    /// Slots outside every frame: the constants, states, parameters, `t`, the prologue and
    /// the values that read no formal.
    glen: usize,
    /// The most frames one right-hand side may hold at once.
    cap: usize,
}

/// One active call: where it returns, the slot its value goes to, and the caller's bases.
#[derive(Clone, Copy, Debug)]
struct Frame {
    ret: u32,
    dst: u32,
    fb: u32,
    flb: u32,
}

/// The stacks of a [`Slab`] for a tape with calls. They keep their capacity between
/// evaluations, so a deep member allocates once.
#[derive(Clone, Debug)]
struct CallState {
    flags: Vec<u8>,
    frames: Vec<Frame>,
    gosub: Vec<u32>,
    /// Frames entered in this evaluation, for the cancel poll and the depth keys.
    calls: u64,
    /// The most frames active at once in this evaluation, for the depth keys.
    depth: u32,
    /// Frames entered and the most active at once, for the tests against `EvalStats`.
    #[cfg(test)]
    entered: u64,
    #[cfg(test)]
    peak: usize,
}

/// Why an evaluation stopped.
#[derive(Debug, PartialEq)]
enum Halt {
    Depth,
    Memory,
    Cancelled,
}

/// An evaluation reads the cancel flag once every this many frames it enters, about 1 ms of
/// `sum` frames in release, so a branching recursion such as `fib(60)` stops soon after a
/// cancel. A tape without calls never reads it.
const CANCEL_POLL: u64 = 1 << 16;

/// The emitter's view of one function body, or of the global code when `local` is false.
struct Cx {
    local: bool,
    /// The frame slot of each op that reads a formal.
    slot: Vec<u32>,
    /// References to each op from the lazy ops of this body, and from its root.
    refs: Vec<u32>,
    flag: Vec<u32>,
    label: Vec<u32>,
    nslot: u32,
    nflag: u32,
    queue: Vec<u32>,
}

impl Cx {
    fn global() -> Self {
        Cx {
            local: false,
            slot: Vec::new(),
            refs: Vec::new(),
            flag: Vec::new(),
            label: Vec::new(),
            nslot: 0,
            nflag: 0,
            queue: Vec::new(),
        }
    }
}

struct Emit<'a> {
    tape: &'a Tape,
    /// Ops with a call below them, which run only when reached.
    lazy: Vec<bool>,
    /// Ops that read a formal, which run once per frame.
    argdep: Vec<bool>,
    gslot: Vec<u32>,
    gflag: Vec<u32>,
    glabel: Vec<u32>,
    gqueue: Vec<u32>,
    glen: u32,
    nflag_global: u32,
    one: u32,
    steps: Vec<Step>,
    labels: Vec<u32>,
    pool: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl Emit<'_> {
    fn label(&mut self) -> u32 {
        self.labels.push(NONE);
        (self.labels.len() - 1) as u32
    }

    fn bind(&mut self, l: u32) {
        self.labels[l as usize] = self.steps.len() as u32;
    }

    fn fresh(&mut self, cx: &mut Cx) -> u32 {
        if cx.local {
            cx.nslot += 1;
            assert!(cx.nslot < LOCAL, "a frame of more than 2^31 values");
            LOCAL | (cx.nslot - 1)
        } else {
            self.glen += 1;
            assert!(self.glen < LOCAL, "more than 2^31 values");
            self.glen - 1
        }
    }

    fn op_into(&mut self, cx: &mut Cx, ins: Ins) -> u32 {
        let d = self.fresh(cx);
        self.steps.push(Step::Op(d, ins));
        d
    }

    /// `x^k` as the chain of [`powi`], one step per multiply or divide.
    fn powi(&mut self, cx: &mut Cx, x: u32, k: i32) -> u32 {
        let mut m = k.unsigned_abs();
        if m == 0 {
            return self.one;
        }
        let mut base = x;
        let mut acc: Option<u32> = None;
        loop {
            if m & 1 == 1 {
                acc = Some(match acc {
                    None => base,
                    Some(a) => self.op_into(cx, Ins::Mul(a, base)),
                });
            }
            m >>= 1;
            if m == 0 {
                break;
            }
            base = self.op_into(cx, Ins::Mul(base, base));
        }
        let r = acc.expect("m > 0 has a set bit");
        if k < 0 {
            let one = self.one;
            self.op_into(cx, Ins::Div(one, r))
        } else {
            r
        }
    }

    /// The slot that holds op `c` once the steps emitted here have run.
    fn use_(&mut self, cx: &mut Cx, c: u32) -> u32 {
        let i = c as usize;
        if !self.lazy[i] {
            return match self.tape.ops[i] {
                Op::Arg(k) => LOCAL | k,
                _ if self.argdep[i] => cx.slot[i],
                _ => self.gslot[i],
            };
        }
        if !self.argdep[i] {
            if self.glabel[i] == NONE {
                self.gslot[i] = self.glen;
                self.glen += 1;
                self.gflag[i] = self.nflag_global;
                self.nflag_global += 1;
                self.glabel[i] = self.label();
                self.gqueue.push(c);
            }
            self.steps.push(Step::Sub(self.gflag[i], self.glabel[i]));
            return self.gslot[i];
        }
        if cx.refs[i] > 1 {
            if cx.label[i] == NONE {
                cx.slot[i] = self.fresh(cx);
                cx.flag[i] = LOCAL | cx.nflag;
                cx.nflag += 1;
                cx.label[i] = self.label();
                cx.queue.push(c);
            }
            self.steps.push(Step::Sub(cx.flag[i], cx.label[i]));
            return cx.slot[i];
        }
        let home = self.fresh(cx);
        cx.slot[i] = home;
        self.compute(cx, c, home);
        home
    }

    /// Emits the steps that compute lazy op `c` into `home`.
    fn compute(&mut self, cx: &mut Cx, c: u32, home: u32) {
        let op = self.tape.ops[c as usize];
        match op {
            Op::Call(u, l) => {
                let list = &self.tape.args[l as usize];
                let mut slots = Vec::with_capacity(list.len());
                for j in 0..list.len() {
                    let a = self.tape.args[l as usize][j];
                    slots.push(self.use_(cx, a));
                }
                let lo = self.pool.len() as u32;
                self.pool.extend_from_slice(&slots);
                self.steps.push(Step::Invoke(u, lo, home));
            }
            Op::If(k, a, b) => {
                let sk = self.use_(cx, k.0);
                let (lb, le) = (self.label(), self.label());
                self.steps.push(Step::JmpIfZero(sk, lb));
                let sa = self.use_(cx, a.0);
                self.steps.push(Step::Move(home, sa));
                self.steps.push(Step::Jmp(le));
                self.bind(lb);
                let sb = self.use_(cx, b.0);
                self.steps.push(Step::Move(home, sb));
                self.bind(le);
            }
            Op::PowI(a, k) => {
                let sa = self.use_(cx, a.0);
                let r = self.powi(cx, sa, k);
                self.steps.push(Step::Move(home, r));
            }
            _ => {
                let ins = lower_op(&op, &mut |id| self.use_(cx, id.0))
                    .expect("a lazy op is a call, an If, a power or an instruction");
                self.steps.push(Step::Op(home, ins));
            }
        }
    }

    /// Emits the subroutines queued in `cx`.
    fn drain(&mut self, cx: &mut Cx) {
        while let Some(c) = cx.queue.pop() {
            let i = c as usize;
            self.bind(cx.label[i]);
            let home = cx.slot[i];
            self.compute(cx, c, home);
            self.steps.push(Step::Rts(cx.flag[i]));
        }
    }

    /// Emits the global subroutines queued so far.
    fn drain_global(&mut self) {
        let mut g = Cx::global();
        while let Some(c) = self.gqueue.pop() {
            let i = c as usize;
            self.bind(self.glabel[i]);
            let home = self.gslot[i];
            self.compute(&mut g, c, home);
            self.steps.push(Step::Rts(self.gflag[i]));
        }
    }

    /// Emits the body of `unit`: the frame's eager ops in id order, then its root.
    fn unit(&mut self, unit: &TapeUnit) -> ProgUnit {
        let tape = self.tape;
        let len = tape.ops.len();
        let mark = tape.reach(&[unit.out]);
        let mut cx = Cx {
            local: true,
            slot: vec![NONE; len],
            refs: vec![0; len],
            flag: vec![NONE; len],
            label: vec![NONE; len],
            nslot: unit.arity,
            nflag: 0,
            queue: Vec::new(),
        };
        for i in (0..len).filter(|&i| mark[i] && self.lazy[i] && self.argdep[i]) {
            tape.operands(&tape.ops[i], &mut |o| cx.refs[o as usize] += 1);
        }
        cx.refs[unit.out as usize] += 1;
        let entry = self.steps.len() as u32;
        for (i, &reached) in mark.iter().enumerate().take(len) {
            if !reached || self.lazy[i] || !self.argdep[i] {
                continue;
            }
            let op = tape.ops[i];
            cx.slot[i] = match op {
                Op::Arg(k) => LOCAL | k,
                Op::PowI(a, k) => {
                    let sa = self.use_(&mut cx, a.0);
                    self.powi(&mut cx, sa, k)
                }
                _ => {
                    let ins = lower_op(&op, &mut |id| self.use_(&mut cx, id.0))
                        .expect("an op that reads a formal is a formal, a power or an instruction");
                    self.op_into(&mut cx, ins)
                }
            };
        }
        let root = self.use_(&mut cx, unit.out);
        self.steps.push(Step::Ret(root));
        self.drain(&mut cx);
        ProgUnit {
            entry,
            arity: unit.arity,
            flen: cx.nslot,
            nflags: cx.nflag,
        }
    }
}

impl<T: Real> CpuTape<T> {
    /// [`CpuTape::new`] for a tape with calls.
    fn new_calls(tape: &Tape) -> Self {
        let (n, np1) = (tape.n, tape.n_p + 1);
        let len = tape.ops.len();
        let mut lazy = vec![false; len];
        let mut argdep = vec![false; len];
        for (i, op) in tape.ops.iter().enumerate() {
            let (mut l, mut a) = (matches!(op, Op::Call(..)), matches!(op, Op::Arg(_)));
            tape.operands(op, &mut |c| {
                assert!(
                    (c as usize) < i,
                    "op {i} reads slot {c} before it is computed"
                );
                l |= lazy[c as usize];
                a |= argdep[c as usize];
            });
            lazy[i] = l;
            argdep[i] = a;
        }
        let mut consts = Consts::default();
        consts.add(ONE_BITS);
        for op in &tape.ops {
            if let Op::Const(b) = *op {
                consts.add(b);
            }
        }
        let nc = consts.bits.len();
        let mut lo = Lower {
            ins: Vec::new(),
            tb: (nc + n + np1) as u32,
        };
        let mut gslot = vec![NONE; len];
        for (i, op) in tape.ops.iter().enumerate() {
            if lazy[i] || argdep[i] {
                continue;
            }
            let c = |id: NodeId| gslot[id.0 as usize];
            gslot[i] = match *op {
                Op::Const(b) => consts.at[&b],
                Op::Var(v) => {
                    assert!((v as usize) < n, "op {i} reads state {v} of {n}");
                    (nc + v as usize) as u32
                }
                Op::Param(j) => {
                    assert!((j as usize) < np1, "op {i} reads parameter {j} of {np1}");
                    (nc + n + j as usize) as u32
                }
                Op::PowI(a, k) => lo.powi(c(a), k, Some(consts.at[&ONE_BITS])),
                _ => {
                    let ins = lower_op(op, &mut |id| c(id))
                        .expect("Tape::build keeps only ops the CPU tape runs");
                    lo.emit(ins)
                }
            };
        }
        let glen = lo.tb + lo.ins.len() as u32;
        let mut e = Emit {
            tape,
            lazy,
            argdep,
            gslot,
            gflag: vec![NONE; len],
            glabel: vec![NONE; len],
            gqueue: Vec::new(),
            glen,
            nflag_global: 0,
            one: consts.at[&ONE_BITS],
            steps: Vec::new(),
            labels: Vec::new(),
            pool: Vec::new(),
        };
        let mut g = Cx::global();
        let outputs: Vec<u32> = tape.outputs.iter().map(|&o| e.use_(&mut g, o)).collect();
        e.steps.push(Step::End);
        let units: Vec<ProgUnit> = tape.units.iter().map(|u| e.unit(u)).collect();
        e.drain_global();
        let labels = e.labels;
        for s in &mut e.steps {
            match s {
                Step::Jmp(t) | Step::JmpIfZero(_, t) | Step::Sub(_, t) => {
                    *t = labels[*t as usize];
                    debug_assert_ne!(*t, NONE, "a jump to an unbound label");
                }
                _ => {}
            }
        }
        let prog = Prog {
            steps: e.steps,
            pool: e.pool,
            units,
            entry: 0,
            nflag_global: e.nflag_global as usize,
            glen: e.glen as usize,
            cap: match FORCED_CAP.with(Cell::get) {
                0 => pc_ir::machine::MAX_ACTIVE_CALLS,
                k => k,
            },
        };
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
            prog: Some(Box::new(prog)),
        }
    }

    /// [`CpuTape::eval_xt`] for a tape with calls. After a failed evaluation the slab stays
    /// failed, and every right-hand side is NaN, until the next [`CpuTape::bind`].
    fn eval_calls(
        &self,
        prog: &Prog,
        x: &[T],
        t: T,
        slab: &mut Slab<T>,
        out: &mut [T],
        cancel: Option<&AtomicBool>,
    ) {
        assert_eq!(slab.v.len(), prog.glen, "the slab is from another tape");
        let cs = slab.calls.as_mut().expect("a slab of a tape with calls");
        if !slab.failed {
            let s = slab.v.as_mut_slice();
            let xs = self.consts.len();
            let ps = xs + self.n;
            let tb = ps + self.np1;
            copy_states(&mut s[xs..ps], &x[..self.n]);
            s[tb - 1] = t;
            for (k, &ins) in self.ins.iter().enumerate() {
                // SAFETY: as in `eval_xt`: `new_calls` checked every operand of the prologue,
                // and the assert above made `s` the length of the global slots.
                unsafe {
                    let v = run(ins, s);
                    *s.get_unchecked_mut(tb + k) = v;
                }
            }
            cs.flags[..prog.nflag_global].fill(0);
            cs.calls = 0;
            cs.depth = 0;
            if exec(prog, &mut slab.v, cs, cancel).is_err() {
                slab.v.truncate(prog.glen);
                cs.flags.truncate(prog.nflag_global);
                cs.frames.clear();
                cs.gosub.clear();
                slab.failed = true;
            }
        }
        for (o, &i) in out.iter_mut().zip(&self.outputs) {
            *o = if slab.failed {
                T::NAN
            } else {
                slab.v[i as usize]
            };
        }
    }
}

/// Runs `prog` on the global slots in `v`.
fn exec<T: Real>(
    prog: &Prog,
    v: &mut Vec<T>,
    cs: &mut CallState,
    cancel: Option<&AtomicBool>,
) -> Result<(), Halt> {
    let steps = &prog.steps;
    let (mut pc, mut fb, mut flb) = (prog.entry as usize, 0usize, 0usize);
    loop {
        match steps[pc] {
            Step::Op(d, ins) => {
                let x = apply(ins, |o| v[res(o, fb)]);
                v[res(d, fb)] = x;
                pc += 1;
            }
            Step::Move(d, s) => {
                v[res(d, fb)] = v[res(s, fb)];
                pc += 1;
            }
            Step::Jmp(t) => pc = t as usize,
            Step::JmpIfZero(c, t) => {
                pc = if v[res(c, fb)] == T::ZERO {
                    t as usize
                } else {
                    pc + 1
                };
            }
            Step::Sub(f, t) => {
                if cs.flags[res(f, flb)] != 0 {
                    pc += 1;
                } else {
                    cs.gosub.try_reserve(1).map_err(|_| Halt::Memory)?;
                    cs.gosub.push(pc as u32 + 1);
                    pc = t as usize;
                }
            }
            Step::Rts(f) => {
                cs.flags[res(f, flb)] = 1;
                pc = cs
                    .gosub
                    .pop()
                    .expect("a subroutine returns where it was called")
                    as usize;
            }
            Step::Invoke(u, lo, d) => {
                if cs.frames.len() >= prog.cap {
                    return Err(Halt::Depth);
                }
                cs.calls += 1;
                if cs.calls.is_multiple_of(CANCEL_POLL)
                    && cancel.is_some_and(|c| c.load(Ordering::Relaxed))
                {
                    return Err(Halt::Cancelled);
                }
                let unit = &prog.units[u as usize];
                let nfb = v.len();
                let flen = unit.flen as usize;
                if nfb + flen > LOCAL as usize {
                    return Err(Halt::Memory);
                }
                v.try_reserve(flen).map_err(|_| Halt::Memory)?;
                cs.flags
                    .try_reserve(unit.nflags as usize)
                    .map_err(|_| Halt::Memory)?;
                cs.frames.try_reserve(1).map_err(|_| Halt::Memory)?;
                let lo = lo as usize;
                for &a in &prog.pool[lo..lo + unit.arity as usize] {
                    let x = v[res(a, fb)];
                    v.push(x);
                }
                v.resize(nfb + flen, T::ZERO);
                let nflb = cs.flags.len();
                cs.flags.resize(nflb + unit.nflags as usize, 0);
                cs.depth = cs.depth.max(cs.frames.len() as u32 + 1);
                #[cfg(test)]
                {
                    cs.entered += 1;
                    cs.peak = cs.peak.max(cs.frames.len() + 1);
                }
                cs.frames.push(Frame {
                    ret: pc as u32 + 1,
                    dst: res(d, fb) as u32,
                    fb: fb as u32,
                    flb: flb as u32,
                });
                fb = nfb;
                flb = nflb;
                pc = unit.entry as usize;
            }
            Step::Ret(s) => {
                let x = v[res(s, fb)];
                let f = cs.frames.pop().expect("a unit returns to its caller");
                v.truncate(fb);
                cs.flags.truncate(flb);
                fb = f.fb as usize;
                flb = f.flb as usize;
                v[f.dst as usize] = x;
                pc = f.ret as usize;
            }
            Step::End => return Ok(()),
        }
    }
}

// The kernel text of a call tape (SPEC section 8af-5, stage 4b). OpenCL C 1.2 and the Metal
// Shading Language have no recursion, and a CUDA thread's own stack is small, so every
// dialect runs the steps of the same `Prog` the CPU runs: one `switch` on the step index
// inside a loop, with the frames of each member on a stack in global memory. A frame is
// `stride` reals: the return label, the formals and frame values, then the frame's flags
// from the top down. The labels fit in a `float`, so the frames hold them as reals.

/// Frames on the per-member stack of a GPU kernel. A call past it ends the member with
/// `recursion_depth`, as a CPU call past `MAX_ACTIVE_CALLS` does.
pub const KERNEL_FRAMES: usize = 4096;

/// The instruction of `ins` as an op whose node ids are its operand slots, `LOCAL` bit
/// included, for [`op_expr`].
fn ins_op(ins: Ins) -> Op {
    let s = NodeId;
    match ins {
        Ins::Neg(a) => Op::Neg(s(a)),
        Ins::Add(a, b) => Op::Add(s(a), s(b)),
        Ins::Sub(a, b) => Op::Sub(s(a), s(b)),
        Ins::Mul(a, b) => Op::Mul(s(a), s(b)),
        Ins::ZMul(a, b) => Op::ZMul(s(a), s(b)),
        Ins::Div(a, b) => Op::Div(s(a), s(b)),
        Ins::ZDiv(a, b) => Op::ZDiv(s(a), s(b)),
        Ins::Call1(f, a) => Op::Call1(f, s(a)),
        Ins::Call2(f, a, b) => Op::Call2(f, s(a), s(b)),
        Ins::Cmp(c, a, b) => Op::Cmp(c, s(a), s(b)),
        Ins::And(a, b) => Op::And(s(a), s(b)),
        Ins::Or(a, b) => Op::Or(s(a), s(b)),
        Ins::Not(a) => Op::Not(s(a)),
        Ins::Select(c, a, b) => Op::Select(s(c), s(a), s(b)),
    }
}

/// Reals per frame of the kernel stack of `tape`: the largest frame of its functions, or 0
/// for a tape without calls.
#[doc(hidden)]
pub fn kernel_stride(tape: &Tape) -> usize {
    kernel_frame(tape).map_or(0, |(_, reals)| reals)
}

/// The function of `tape` with the largest kernel frame and the reals of that frame, or
/// `None` for a tape without calls. The first such function wins a tie.
#[doc(hidden)]
pub fn kernel_frame(tape: &Tape) -> Option<(String, usize)> {
    if tape.units.is_empty() {
        return None;
    }
    let prog = CpuTape::<f64>::new_calls(tape)
        .prog
        .expect("a call program");
    let mut best: Option<(usize, usize)> = None;
    for (i, u) in prog.units.iter().enumerate() {
        let reals = 1 + u.flen as usize + u.nflags as usize;
        if best.is_none_or(|(_, r)| reals > r) {
            best = Some((i, reals));
        }
    }
    best.map(|(i, reals)| (tape.units[i].name.clone(), reals))
}

fn stride_of(prog: &Prog) -> usize {
    prog.units
        .iter()
        .map(|u| 1 + u.flen as usize + u.nflags as usize)
        .max()
        .unwrap_or(1)
}

/// The kernel function `pc_rhs` of a tape with calls, and the reals per frame of its stack.
/// It takes the arguments of [`emit_rhs`](pc_codegen::kernel::emit_rhs), then the member's
/// stack and its error word: a call past [`KERNEL_FRAMES`] sets the word, and every
/// right-hand side is NaN from then on.
#[doc(hidden)]
pub fn emit_calls_rhs(
    tape: &Tape,
    scalar: Scalar,
    stack: StackLayout,
) -> Result<(String, usize), KernelError> {
    let mut names: Vec<&'static str> = Vec::new();
    for op in &tape.ops {
        if matches!(op, Op::Arg(_) | Op::Call(..) | Op::If(..) | Op::Out(..)) {
            continue;
        }
        if let Some(n) = unsupported_op(op)
            && !names.contains(&n)
        {
            names.push(n);
        }
    }
    if !names.is_empty() {
        return Err(KernelError::Unsupported(names.join(", ")));
    }
    let cpu = CpuTape::<f64>::new_calls(tape);
    let prog = cpu.prog.as_deref().expect("a call program");
    let stride = stride_of(prog);
    let ninvoke = prog
        .steps
        .iter()
        .filter(|s| matches!(s, Step::Invoke(..)))
        .count();
    // Labels are exact in a `float`, and a stack index fits in 31 bits.
    if prog.steps.len() + ninvoke >= 1 << 24 || stride * KERNEL_FRAMES >= 1 << 31 {
        return Err(KernelError::Unsupported(
            "recursive functions too large for a kernel".into(),
        ));
    }
    // The warp layout scales every slot index by the group of 32 members.
    if stack == StackLayout::Warp && stride * KERNEL_FRAMES * 32 >= 1 << 31 {
        return Err(KernelError::Unsupported(
            "recursive functions too large for the warp stack layout; use stack member".into(),
        ));
    }
    // Slot `e` of this member's stack. `stk` already points at the member (member layout) or
    // at its lane of the group (warp layout, where slot e of the 32 lanes sits side by side).
    let at = |e: String| -> String {
        match stack {
            StackLayout::Member => format!("stk[{e}]"),
            StackLayout::Warp => format!("stk[({e}) * 32u]"),
        }
    };
    let (n, n_p) = (tape.n, tape.n_p);
    let nc = cpu.consts.len();
    let tb = nc + n + n_p + 1;
    let name = |o: u32| -> String {
        if o & LOCAL != 0 {
            return at(format!("fb + {}u", 1 + (o & !LOCAL)));
        }
        let o = o as usize;
        if o < nc {
            format!("c{o}")
        } else if o < nc + n {
            format!("x[{}]", o - nc)
        } else if o < tb {
            match o - nc - n {
                j if j == n_p => "t".to_string(),
                j => format!("p[{j}]"),
            }
        } else {
            format!("g{o}")
        }
    };
    let flag = |f: u32| -> String {
        if f & LOCAL != 0 {
            at(format!("fb + {}u", stride as u32 - 1 - (f & !LOCAL)))
        } else {
            format!("gf{f}")
        }
    };
    let mut b = String::new();
    b.push_str(
        "PC_FN void pc_rhs(pc_real t, PC_SCR const pc_real *x, PC_IN pc_real *p, PC_SCR pc_real *out,\n\
         \x20                 PC_GLOBAL pc_real *stk, PC_PRIV pc_uint *cerr) {\n",
    );
    b.push_str("    (void)t;\n    (void)x;\n    (void)p;\n");
    b.push_str("    if (*cerr != 0u) {\n");
    for k in 0..n {
        let _ = writeln!(b, "        out[{k}] = PC_NAN;");
    }
    b.push_str("        return;\n    }\n");
    for (k, &c) in cpu.consts.iter().enumerate() {
        let lit = float_literal(c, scalar);
        let _ = writeln!(b, "    const pc_real c{k} = {lit};\n    (void)c{k};");
    }
    for (k, &ins) in cpu.ins.iter().enumerate() {
        let id = format!("g{}", tb + k);
        let expr = op_expr(
            &ins_op(ins),
            &mut |a| name(a.0),
            n_p,
            scalar,
            &format!("q{k}"),
            &mut b,
        );
        let _ = writeln!(b, "    const pc_real {id} = {expr};\n    (void){id};");
    }
    for k in tb + cpu.ins.len()..prog.glen {
        let _ = writeln!(b, "    pc_real g{k} = PC_ZERO;\n    (void)g{k};");
    }
    for f in 0..prog.nflag_global {
        let _ = writeln!(b, "    int gf{f} = 0;\n    (void)gf{f};");
    }
    b.push_str(
        "    pc_real rv = PC_ZERO;\n    pc_uint depth = 0u, fb = 0u, nfb = 0u;\n    int pc = 0;\n\
         \x20   (void)rv;\n    (void)depth;\n    (void)fb;\n    (void)nfb;\n",
    );
    // A case label at every step a jump, a return or a resume reaches.
    let mut label = vec![false; prog.steps.len()];
    label[prog.entry as usize] = true;
    for u in &prog.units {
        label[u.entry as usize] = true;
    }
    for (s, step) in prog.steps.iter().enumerate() {
        match *step {
            Step::Jmp(t) | Step::JmpIfZero(_, t) => label[t as usize] = true,
            Step::Sub(_, t) => {
                label[t as usize] = true;
                if s + 1 < label.len() {
                    label[s + 1] = true;
                }
            }
            _ => {}
        }
    }
    b.push_str("    while (pc >= 0) {\n        switch (pc) {\n");
    let mut resume = prog.steps.len();
    for (s, step) in prog.steps.iter().enumerate() {
        if label[s] {
            if s > 0 {
                b.push_str("        /* fallthrough */\n");
            }
            let _ = writeln!(b, "        case {s}:");
        }
        b.push_str("        {\n");
        match *step {
            Step::Op(d, ins) => {
                let mut pre = String::new();
                let expr = op_expr(
                    &ins_op(ins),
                    &mut |a| name(a.0),
                    n_p,
                    scalar,
                    &format!("m{s}"),
                    &mut pre,
                );
                for l in pre.lines() {
                    let _ = writeln!(b, "        {l}");
                }
                let _ = writeln!(b, "            {} = {expr};", name(d));
            }
            Step::Move(d, a) => {
                let _ = writeln!(b, "            {} = {};", name(d), name(a));
            }
            Step::Jmp(t) => {
                let _ = writeln!(b, "            pc = {t};\n            continue;");
            }
            Step::JmpIfZero(c, t) => {
                let _ = writeln!(
                    b,
                    "            if ({} == PC_ZERO) {{\n                pc = {t};\n                continue;\n            }}",
                    name(c)
                );
            }
            Step::Sub(f, t) if f & LOCAL == 0 => {
                // The flag holds 0 before the subroutine runs, its return label while it
                // runs and -1 after. A subroutine that reaches itself through a call never
                // ends on the CPU either, which stops at its frame cap.
                let g = flag(f);
                let _ = writeln!(
                    b,
                    "            if ({g} == 0) {{\n                {g} = {};\n                pc = {t};\n                continue;\n            }}\n\
                     \x20           if ({g} > 0) {{\n                *cerr = 1u;\n                pc = -1;\n                continue;\n            }}",
                    s + 1
                );
            }
            Step::Sub(f, t) => {
                let g = flag(f);
                let _ = writeln!(
                    b,
                    "            if ({g} == PC_ZERO) {{\n                {g} = (pc_real){};\n                pc = {t};\n                continue;\n            }}",
                    s + 1
                );
            }
            Step::Rts(f) if f & LOCAL == 0 => {
                let g = flag(f);
                let _ = writeln!(
                    b,
                    "            pc = {g};\n            {g} = -1;\n            continue;"
                );
            }
            Step::Rts(f) => {
                let g = flag(f);
                let _ = writeln!(
                    b,
                    "            pc = (int){g};\n            {g} = -PC_ONE;\n            continue;"
                );
            }
            Step::Invoke(u, lo, d) => {
                let unit = &prog.units[u as usize];
                let _ = writeln!(
                    b,
                    "            if (depth >= {KERNEL_FRAMES}u) {{\n                *cerr = 1u;\n                pc = -1;\n                continue;\n            }}\n\
                     \x20           nfb = depth * {stride}u;"
                );
                let lo = lo as usize;
                for (j, &a) in prog.pool[lo..lo + unit.arity as usize].iter().enumerate() {
                    let _ = writeln!(
                        b,
                        "            {} = {};",
                        at(format!("nfb + {}u", 1 + j)),
                        name(a)
                    );
                }
                for f in 0..unit.nflags {
                    let _ = writeln!(
                        b,
                        "            {} = PC_ZERO;",
                        at(format!("nfb + {}u", stride as u32 - 1 - f))
                    );
                }
                let _ = writeln!(
                    b,
                    "            {} = (pc_real){resume};\n            fb = nfb;\n            depth++;\n\
                     \x20           pc = {};\n            continue;\n        }}\n        /* fallthrough */\n\
                     \x20       case {resume}:\n        {{\n            {} = rv;",
                    at("nfb".into()),
                    unit.entry,
                    name(d)
                );
                resume += 1;
            }
            Step::Ret(a) => {
                let _ = writeln!(
                    b,
                    "            rv = {};\n            pc = (int){};\n            depth--;\n\
                     \x20           fb -= {stride}u;\n            continue;",
                    name(a),
                    at("fb".into())
                );
            }
            Step::End => {
                b.push_str("            pc = -1;\n            continue;\n");
            }
        }
        b.push_str("        }\n");
    }
    b.push_str(
        "        /* fallthrough */\n        default:\n            *cerr = 1u;\n            pc = -1;\n\
         \x20           continue;\n        }\n    }\n",
    );
    b.push_str("    if (*cerr != 0u) {\n");
    for k in 0..n {
        let _ = writeln!(b, "        out[{k}] = PC_NAN;");
    }
    b.push_str("    } else {\n");
    for (k, &o) in cpu.outputs.iter().enumerate() {
        let _ = writeln!(b, "        out[{k}] = {};", name(o));
    }
    b.push_str("    }\n}\n");
    Ok((b, stride))
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

    // Stage 4a (SPEC 8af-5): the CPU tape runs recursive functions. `pc_ir::machine::run` is
    // the oracle: the same bits, the same frames entered, the same peak depth and the same
    // depth error.

    /// Runs the call tape of `ir` and the machine at `x`, `p` (with `t` last) under `cap`.
    /// Returns the tape's outputs, or None when the machine stopped on its cap.
    fn calls_vs_machine(ir: &IrModel, x: &[f64], p: &[f64], cap: usize) -> Option<Vec<f64>> {
        let tape = Tape::build(ir).expect("tape");
        assert!(!tape.units.is_empty(), "the model has calls");
        let mut cpu = CpuTape::<f64>::new(&tape);
        cpu.set_max_calls(cap);
        let mut slab = cpu.slab();
        let mut got = vec![0.0; ir.f.len()];
        cpu.eval(x, p, &mut slab, &mut got);
        let mut want = vec![0.0; ir.f.len()];
        let r = pc_ir::machine::run(&ir.dag, &ir.f, x, p, &[], &mut want, None, cap);
        let cs = slab.calls.as_ref().expect("call state");
        match r {
            Ok(stats) => {
                assert!(
                    !slab.failed(),
                    "the tape failed where the machine ran: {x:?} {p:?}"
                );
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                        "output {i} at x={x:?} p={p:?}: tape {g:?} machine {w:?}"
                    );
                }
                assert_eq!(cs.entered, stats.calls, "frames entered at {x:?} {p:?}");
                assert_eq!(cs.peak, stats.max_depth, "peak depth at {x:?} {p:?}");
                Some(got)
            }
            Err(pc_ir::machine::EvalError::Depth { .. }) => {
                assert!(
                    slab.failed(),
                    "the machine hit its cap and the tape did not"
                );
                assert!(got.iter().all(|g| g.is_nan()));
                None
            }
            Err(e) => panic!("machine: {e:?}"),
        }
    }

    const CALL_DEFS: &str = "par k=3, a=0.5\n\
        pw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\n\
        fx(n)=if(n<1)then(1)else(x*fx(n-1))\n\
        fact(n)=if(n<=1)then(1)else(n*fact(n-1))\n\
        ev(n)=if(n==0)then(1)else(od(n-1))\n\
        od(n)=if(n==0)then(0)else(ev(n-1))\n\
        h(n)=if(n<1)then(0)else(if(n>5)then(h(n-1))else(1+h(n-1)))\n\
        g(n)=(if(n>0)then(g(n-1))else(0))+(if(n>5)then(g(n-1))else(1))\n\
        sm(n)=if(n<1)then(0)else(n+sm(n-1))\n\
        gl(n)=if(n<1)then(0)else(sm(k)+gl(n-1))\n\
        inc(n)=if(n>100)then(inc(n-1))else(n+1)\n\
        dbl(n)=if(n>100)then(dbl(n-1))else(n*2)\n\
        fib(n)=if(n<2)then(n)else(fib(n-1)+fib(n-2))\n";

    const CALL_RHS: &[&str] = &[
        "pw(x,k)",
        "fx(k)",
        "fact(k)",
        "ev(k)+od(k)",
        "h(k)",
        "g(k)",
        "sm(k)*a",
        "gl(k)",
        "inc(k)+dbl(k)",
        "inc(inc(k))",
        "pw(x,fact(k))",
        "if(k<0)then(0)else(fx(k))",
        "if(a)then(sm(k))else(fact(k))",
        "(if(k<0)then(0)else(fx(k)))+fx(k)",
        "if(k<0)then(0)else(if(k>5)then(fx(k)*2)else(fx(k)+1))",
        "fx(if(k<0)then(0)else(sm(k)))",
        "(if(k<2)then(sm(k))else(fx(k)))*(if(k>4)then(fx(k))else(sm(k)))",
        "fib(k)+sin(x)*a",
    ];

    fn call_model(rhs: &str) -> IrModel {
        model(&format!("{CALL_DEFS}x'={rhs}\ny'=x-y\ndone\n"))
    }

    /// The corpus of the stage 2 tests agrees with the machine bit for bit, in its frames and
    /// in its peak depth, at several states and parameters, including a NaN state and a NaN
    /// condition, and under a tight cap.
    #[test]
    fn call_tapes_match_the_machine() {
        let pts: &[(f64, f64, f64)] = &[
            (0.3, 3.0, 0.5),
            (-1.7, 0.0, 0.0),
            (2.0, 6.0, -0.25),
            (0.9, 9.0, f64::NAN),
            (f64::NAN, 4.0, 1.0),
            (1.1, -1.0, 2.0),
            (0.5, 2.5, 1.0),
        ];
        for rhs in CALL_RHS {
            let ir = call_model(rhs);
            for &(x, k, a) in pts {
                for cap in [1_000_000, 8, 3, 1, 0] {
                    calls_vs_machine(&ir, &[x, 0.1], &[k, a, 0.0], cap);
                }
            }
        }
    }

    /// The f32 tape gives the closed forms where they are exact in single precision.
    #[test]
    fn call_tapes_reach_the_closed_forms_in_single_precision() {
        let ir = call_model("fact(k)+0*pw(x,k)");
        let tape = Tape::build(&ir).expect("tape");
        let cpu = CpuTape::<f32>::new(&tape);
        let mut slab = cpu.slab();
        let mut out = [0f32; 2];
        let mut want = 1f32;
        for n in 1..=10u8 {
            want *= f32::from(n);
            cpu.eval(&[0.5, 0.0], &[f32::from(n), 0.0, 0.0], &mut slab, &mut out);
            assert_eq!(out[0], want, "fact({n})");
        }
        for (rhs, k, want) in [
            ("sm(k)", 100f32, 5050f32),
            ("ev(k)", 7.0, 0.0),
            ("od(k)", 7.0, 1.0),
            ("h(k)", 9.0, 5.0),
            ("fib(k)", 20.0, 6765.0),
            ("pw(x,k)", 3.0, 0.125),
        ] {
            let tape = Tape::build(&call_model(rhs)).expect("tape");
            let cpu = CpuTape::<f32>::new(&tape);
            let mut slab = cpu.slab();
            cpu.eval(&[0.5, 0.0], &[k, 0.0, 0.0], &mut slab, &mut out);
            assert_eq!(out[0], want, "{rhs} at k={k}");
        }
    }

    /// A call that reaches itself with the same arguments outside any formal never ends: the
    /// cap stops it, as it stops the machine.
    #[test]
    fn a_self_reentrant_global_call_hits_the_cap() {
        let ir = model("par k=1\nf(n)=n+f(k)\nx'=f(k)\ndone\n");
        assert!(calls_vs_machine(&ir, &[0.0], &[1.0, 0.0], 50).is_none());
    }

    /// The cap counts active frames, and the entry has none: `sm(n)` takes `n + 1` frames.
    #[test]
    fn the_cap_is_the_number_of_active_frames() {
        let ir = call_model("sm(k)");
        for cap in 0..6 {
            for k in 0..7 {
                let r = calls_vs_machine(&ir, &[0.0, 0.0], &[f64::from(k), 0.0, 0.0], cap);
                assert_eq!(r.is_some(), (k as usize) < cap, "k={k} cap={cap}");
            }
        }
    }

    /// The default cap is `MAX_ACTIVE_CALLS`: `sm(999_999)` runs and `sm(1_000_000)` does not.
    #[test]
    fn the_default_cap_is_a_million_frames() {
        let tape = Tape::build(&call_model("sm(k)")).expect("tape");
        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        let mut out = [0.0; 2];
        cpu.eval(&[0.0, 0.0], &[999_999.0, 0.0, 0.0], &mut slab, &mut out);
        assert!(!slab.failed());
        assert_eq!(out[0], 999_999.0 * 1_000_000.0 / 2.0);
        cpu.eval(&[0.0, 0.0], &[1_000_000.0, 0.0, 0.0], &mut slab, &mut out);
        assert!(slab.failed());
        assert!(out[0].is_nan() && out[1].is_nan());
    }

    /// Two calls in one body each enter their own frames; a cap below the depth stops them.
    #[test]
    fn fib_matches_the_machine_under_a_cap() {
        let ir = call_model("fib(k)");
        for k in 0..16 {
            for cap in [4, 9, 100] {
                calls_vs_machine(&ir, &[0.0, 0.0], &[f64::from(k), 0.0, 0.0], cap);
            }
        }
    }

    /// `g(n)` names `g(n-1)` twice, so it runs once per frame and `g(60)` takes 61 frames,
    /// not 2^60.
    #[test]
    fn a_call_named_twice_in_a_body_runs_once() {
        let ir = call_model("g(k)");
        let got = calls_vs_machine(&ir, &[0.0, 0.0], &[60.0, 0.0, 0.0], 1000).expect("runs");
        // g(n) = g(n-1) + 1 up to g(5) = 6, then doubles.
        assert_eq!(got[0], 6.0 * 2f64.powi(55));
    }

    /// A chain of 24 `If`s that each name the level below in both arms emits a program
    /// linear in the depth of the chain. Written out as text the chain would double at each
    /// level, so it is built on the DAG.
    #[test]
    fn a_deep_if_chain_stays_linear() {
        use pc_ir::dag::Cmp;
        let mut ir = call_model("sm(k)");
        let d = &mut ir.dag;
        let u = d.add_unit("chain", 1, 1);
        let n = d.arg(0);
        let one = d.konst(1.0);
        let m = d.sub(n, one);
        let mut e = d.call(u, &[m]);
        for i in 0..24 {
            let ki = d.konst(f64::from(i));
            let c = d.cmp(Cmp::Gt, n, ki);
            let e1 = d.add(e, one);
            e = d.select(c, e, e1);
        }
        let zero = d.konst(0.0);
        let stop = d.cmp(Cmp::Lt, n, one);
        let body = d.select(stop, zero, e);
        assert!(d.set_body(u, &[body]));
        let k = d.param(0);
        ir.f[0] = d.call(u, &[k]);
        let tape = Tape::build(&ir).expect("tape");
        let cpu = CpuTape::<f64>::new(&tape);
        let steps = cpu.prog.as_ref().expect("prog").steps.len();
        assert!(steps < 40 * 24, "{steps} steps for 24 levels");
        for k in [0.0, 3.0, 30.0] {
            calls_vs_machine(&ir, &[0.0, 0.0], &[k, 0.0, 0.0], 1000);
        }
    }

    /// A set cancel flag stops an evaluation that enters more than `CANCEL_POLL` frames, at
    /// any depth, and leaves the slab failed like the cap; a clear flag changes no bit, and an
    /// evaluation with fewer frames never reads the flag.
    #[test]
    fn a_set_cancel_flag_stops_a_long_evaluation() {
        use std::sync::atomic::AtomicBool;
        let tape = Tape::build(&call_model("fib(k)")).expect("tape");
        let cpu = CpuTape::<f64>::new(&tape);
        let (clear, set) = (AtomicBool::new(false), AtomicBool::new(true));
        let mut slab = cpu.slab();
        let (mut want, mut out) = ([0.0; 2], [0.0; 2]);
        // fib(25) enters 242,785 frames, at most 25 at once.
        let p = [25.0, 0.0, 0.0];
        cpu.eval(&[0.0, 0.0], &p, &mut slab, &mut want);
        assert_eq!(want[0], 75025.0);
        cpu.bind(&p, &mut slab);
        cpu.eval_xt_with(&[0.0, 0.0], 0.0, &mut slab, &mut out, Some(&clear));
        assert!(!slab.failed());
        assert_eq!(out.map(f64::to_bits), want.map(f64::to_bits));
        cpu.eval_xt_with(&[0.0, 0.0], 0.0, &mut slab, &mut out, Some(&set));
        assert!(slab.failed() && out[0].is_nan());
        let cs = slab.calls.as_ref().expect("call state");
        assert!(cs.frames.is_empty() && cs.gosub.is_empty());
        // fib(10) enters 177 frames, fewer than one poll interval.
        cpu.bind(&[10.0, 0.0, 0.0], &mut slab);
        cpu.eval_xt_with(&[0.0, 0.0], 0.0, &mut slab, &mut out, Some(&set));
        assert!(!slab.failed());
        assert_eq!(out[0], 55.0);
    }

    /// A failed slab gives NaN until the next bind, and then runs again.
    #[test]
    fn a_slab_runs_again_after_the_cap() {
        let tape = Tape::build(&call_model("sm(k)")).expect("tape");
        let mut cpu = CpuTape::<f64>::new(&tape);
        cpu.set_max_calls(10);
        let mut slab = cpu.slab();
        let mut out = [0.0; 2];
        cpu.eval(&[0.0, 0.0], &[20.0, 0.0, 0.0], &mut slab, &mut out);
        assert!(slab.failed() && out[0].is_nan());
        cpu.eval_xt(&[0.0, 0.0], 0.0, &mut slab, &mut out);
        assert!(
            slab.failed() && out[0].is_nan(),
            "failed until the next bind"
        );
        cpu.eval(&[0.0, 0.0], &[4.0, 0.0, 0.0], &mut slab, &mut out);
        assert!(!slab.failed());
        assert_eq!(out, [10.0, 0.0]);
        let cs = slab.calls.as_ref().expect("call state");
        assert!(cs.frames.is_empty() && cs.gosub.is_empty());
        assert_eq!(slab.v.len(), cpu.prog.as_ref().expect("prog").glen);
    }

    /// A call that is never reached enters no frame, even with a cap of zero.
    #[test]
    fn an_untaken_arm_makes_no_call() {
        let ir = call_model("if(k<0)then(0)else(fx(k))");
        let got = calls_vs_machine(&ir, &[2.0, 0.0], &[-1.0, 0.0, 0.0], 0).expect("runs");
        assert_eq!(got[0], 0.0);
        assert!(calls_vs_machine(&ir, &[2.0, 0.0], &[1.0, 0.0, 0.0], 0).is_none());
    }

    /// A call inside a body that reads no formal runs once per right-hand side.
    #[test]
    fn a_global_call_in_a_body_runs_once() {
        let ir = call_model("gl(k)");
        let tape = Tape::build(&ir).expect("tape");
        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        let mut out = [0.0; 2];
        cpu.eval(&[0.0, 0.0], &[50.0, 0.0, 0.0], &mut slab, &mut out);
        assert_eq!(out[0], 50.0 * 1275.0);
        // 51 frames of gl and 51 of sm(50), not 50 * 51.
        assert_eq!(slab.calls.as_ref().expect("call state").entered, 102);
    }

    /// The tape refuses a call whose argument count does not match its function.
    #[test]
    fn a_call_with_the_wrong_arity_is_refused() {
        let mut ir = call_model("inc(k)");
        let u = ir.dag.add_unit("two", 2, 1);
        let a = ir.dag.arg(0);
        assert!(ir.dag.set_body(u, &[a]));
        let x = ir.dag.var(0);
        ir.f[0] = ir.dag.call(u, &[x]);
        let Err(EnsembleError::BadJob(m)) = Tape::build(&ir) else {
            panic!("not refused");
        };
        assert!(m.contains("passes 1 arguments to function two"), "{m}");
    }

    /// A function argument read outside a function is refused.
    #[test]
    fn a_formal_outside_a_function_is_refused() {
        let mut ir = call_model("sm(k)");
        let a = ir.dag.arg(0);
        let c = ir.f[0];
        ir.f[1] = ir.dag.add(a, c);
        let Err(EnsembleError::BadJob(m)) = Tape::build(&ir) else {
            panic!("not refused");
        };
        assert!(m.contains("outside a function"), "{m}");
    }

    /// Random bodies over a guarded recursion agree with the machine.
    fn body_expr(depth: u32) -> BoxedStrategy<String> {
        let leaf = prop_oneof![
            Just("@N".to_string()),
            Just("x".to_string()),
            Just("a".to_string()),
            Just("1.5".to_string()),
            Just("@P".to_string()),
            Just("@Q".to_string()),
            Just("@R".to_string()),
        ];
        if depth == 0 {
            return leaf.boxed();
        }
        let sub = body_expr(depth - 1);
        prop_oneof![
            leaf,
            (sub.clone(), sub.clone(), 0..4u8).prop_map(|(l, r, o)| {
                let op = ["+", "-", "*", "/"][o as usize];
                format!("({l}){op}({r})")
            }),
            sub.clone().prop_map(|e| format!("sin({e})")),
            (sub.clone(), sub.clone(), sub)
                .prop_map(|(c, l, r)| { format!("if(({c})>0)then({l})else({r})") }),
        ]
        .boxed()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        #[test]
        fn random_call_tapes_match_the_machine(
            body in body_expr(3),
            gbody in body_expr(2),
            rhs in body_expr(2),
            k in -1.0..7.0f64,
            a in -2.0..2.0f64,
            x in -2.0..2.0f64,
            cap in prop_oneof![Just(1_000usize), 0..6usize],
        ) {
            // `@N` is the formal and `@P`, `@Q`, `@R` are calls. `f` runs down to n < 1, and
            // `g` calls `f` but not itself.
            let sub = |e: &str, t: [&str; 4]| {
                e.replace("@N", t[0]).replace("@P", t[1]).replace("@Q", t[2]).replace("@R", t[3])
            };
            let body = sub(&body, ["n", "f(n-1)", "f(n-2)", "g(a)"]);
            let gbody = sub(&gbody, ["m", "f(m)", "f(a)", "f(x)"]);
            let rhs = sub(&rhs, ["k", "f(k)", "f(x)", "g(a)"]);
            let src = format!(
                "par k=3, a=0.5\nf(n)=if(n<1)then(n*x)else({body})\ng(m)={gbody}\nx'={rhs}\ndone\n"
            );
            let ir = model(&src);
            prop_assume!(ir.f.iter().any(|&id| ir.dag.has_call(id)));
            calls_vs_machine(&ir, &[x], &[k.round(), a, 0.0], cap);
        }
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
                match (kind >> 4) % 15 {
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
                    12 => Op::Select(id(a), id(b), id(c)),
                    13 => Op::ZMul(id(a), id(b)),
                    _ => Op::ZDiv(id(a), id(b)),
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
            args: Vec::new(),
            units: Vec::new(),
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
                match (kind >> 4) % 15 {
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
                    12 => dag.select(x, y, z),
                    13 => dag.zmul(x, y),
                    _ => dag.zdiv(x, y),
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
            args: Vec::new(),
            units: Vec::new(),
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

    /// Compiles the outputs of `tape` as a single-precision `F` entry of `dag` and calls it
    /// against [`CpuTape::<f32>::eval`]. The function is called on other inputs first, so a
    /// value kept from an earlier call would show.
    fn check_native_f32(dag: &Dag, tape: &Tape, x: &[f64], p: &[f64]) -> Result<(), String> {
        let entry = Entry {
            kind: EntryKind::F,
            outputs: tape.outputs.iter().map(|&o| NodeId(o)).collect(),
        };
        let jit = pc_jit::compile_f32(&[(EntryKind::F, &entry)], dag).map_err(|e| e.to_string())?;
        let f = jit.plain_f32(&EntryKind::F).ok_or("no F entry")?;
        let x: Vec<f32> = x.iter().map(|&v| f32::from_f64(v)).collect();
        let p: Vec<f32> = p.iter().map(|&v| f32::from_f64(v)).collect();
        let cpu = CpuTape::<f32>::new(tape);
        let mut slab = cpu.slab();
        let mut want = vec![0.0f32; tape.n];
        cpu.eval(&x, &p, &mut slab, &mut want);

        // SAFETY: `f` is the `F` of `dag` with `tape.n` outputs, and the JIT and the `f32` tape
        // give the same bits (this test); it is called on one thread, and `jit` outlives every
        // call below.
        let native = unsafe { NativeRhs::new_f32(f) };
        let xg: Vec<f32> = x.iter().map(|v| v + 1.0).collect();
        let pg: Vec<f32> = p.iter().map(|v| v + 1.0).collect();
        let mut got = vec![f32::MAX; tape.n];
        cpu.bind(&pg, &mut slab);
        cpu.eval_native(native, &xg, pg[tape.n_p], &mut slab, &mut got);
        cpu.bind(&p, &mut slab);
        cpu.eval_native(native, &x, p[tape.n_p], &mut slab, &mut got);
        for i in 0..tape.n {
            if bits(got[i]) != bits(want[i]) {
                return Err(format!(
                    "output {i}: jit {:e} vs CpuTape {:e}",
                    got[i], want[i]
                ));
            }
        }
        Ok(())
    }

    /// Values that sit at the edges of `f32`: subnormals, the largest finite value, values
    /// that overflow or underflow when narrowed, and halfway cases of the narrowing.
    const SPECIAL_F32: [f64; 10] = [
        1.401298464324817e-45,
        -1.1754942106924411e-38,
        3.4028234663852886e38,
        -3.4028235677973366e38,
        1e39,
        1e-46,
        1.0000000596046448,
        1.0000001788139343,
        -16777217.0,
        0.1,
    ];

    fn value_f32() -> impl Strategy<Value = f64> {
        prop_oneof![
            -4.0f64..4.0,
            any::<f64>(),
            any::<f32>().prop_map(f64::from),
            proptest::sample::select(SPECIAL.to_vec()),
            proptest::sample::select(SPECIAL_F32.to_vec()),
        ]
    }

    fn native_case_f32() -> impl Strategy<Value = NativeIn> {
        let raw = (
            any::<u8>(),
            any::<u32>(),
            any::<u32>(),
            any::<u32>(),
            value_f32(),
            -18i32..=18,
        );
        (1usize..4, 0usize..3, proptest::collection::vec(raw, 1..40)).prop_flat_map(
            |(n, n_p, raw)| {
                (
                    Just(n),
                    Just(n_p),
                    Just(raw),
                    proptest::collection::vec(any::<u32>(), n),
                    proptest::collection::vec(value_f32(), n),
                    proptest::collection::vec(value_f32(), n_p + 1),
                )
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1024))]

        /// The single-precision `F` gives the bits of `CpuTape<f32>` for every op the tape
        /// takes. Slow: each case compiles a module.
        #[test]
        fn the_f32_jit_matches_cpu_tape_bit_for_bit((n, n_p, raw, outs, x, p) in native_case_f32()) {
            let (dag, tape) = dag_case(n, n_p, &raw, &outs);
            prop_assert_eq!(check_native_f32(&dag, &tape, &x, &p), Ok(()));
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

    /// A native function of the other precision is not called: each tape runs its own
    /// interpreter instead.
    #[test]
    fn a_native_function_of_the_other_precision_is_not_called() {
        unsafe extern "C" fn poison64(_: *const f64, _: *const f64, out: *mut f64) {
            // SAFETY: the caller gave one output.
            unsafe { *out = 12345.0 };
        }
        unsafe extern "C" fn poison32(_: *const f32, _: *const f32, out: *mut f32) {
            // SAFETY: the caller gave one output.
            unsafe { *out = 12345.0 };
        }
        let ir = model("x'=sin(x)+1\ndone\n");
        let tape = Tape::build(&ir).expect("tape");
        // SAFETY: neither is called, which is what this test checks.
        let (double, single) = unsafe { (NativeRhs::new(poison64), NativeRhs::new_f32(poison32)) };

        let cpu = CpuTape::<f32>::new(&tape);
        let mut slab = cpu.slab();
        cpu.bind(&[0.0], &mut slab);
        let (mut want, mut got) = ([0.0f32], [0.0f32]);
        cpu.eval_xt(&[0.5], 0.0, &mut slab, &mut want);
        cpu.eval_native(double, &[0.5], 0.0, &mut slab, &mut got);
        assert_eq!(got[0].to_bits(), want[0].to_bits());

        let cpu = CpuTape::<f64>::new(&tape);
        let mut slab = cpu.slab();
        cpu.bind(&[0.0], &mut slab);
        let (mut want, mut got) = ([0.0f64], [0.0f64]);
        cpu.eval_xt(&[0.5], 0.0, &mut slab, &mut want);
        cpu.eval_native(single, &[0.5], 0.0, &mut slab, &mut got);
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

    /// `ZMul(t, p)` and `ZDiv(t, p)` are +0 for a zero `t` of either sign, whatever `p` is,
    /// and `t * p` or `t / p` otherwise, in the interpreter and the compiled tape, in both
    /// precisions.
    #[test]
    fn a_zero_first_operand_hides_a_non_finite_second_operand() {
        fn run<T: Real>(vals: &[f64]) {
            let tape = Tape {
                n: 2,
                n_p: 0,
                ops: vec![
                    Op::Var(0),
                    Op::Var(1),
                    Op::ZMul(NodeId(0), NodeId(1)),
                    Op::ZDiv(NodeId(0), NodeId(1)),
                ],
                outputs: vec![2, 3],
                args: Vec::new(),
                units: Vec::new(),
            };
            let cpu = CpuTape::<T>::new(&tape);
            let mut slab = cpu.slab();
            for &t in vals {
                for &p in vals {
                    let (tt, pp) = (T::from_f64(t), T::from_f64(p));
                    let want = if tt == T::ZERO {
                        [T::ZERO, T::ZERO]
                    } else {
                        [tt * pp, tt / pp]
                    };
                    let mut got = [T::ZERO; 2];
                    cpu.eval(&[tt, pp], &[T::ZERO], &mut slab, &mut got);
                    let mut by_tape = [T::ZERO; 2];
                    let mut scratch = vec![T::ZERO; tape.len()];
                    tape.eval(&[tt, pp], &[T::ZERO], &mut scratch, &mut by_tape);
                    for k in 0..2 {
                        assert_eq!(bits(got[k]), bits(want[k]), "{} {k} {t:e} {p:e}", T::NAME);
                        assert_eq!(
                            bits(by_tape[k]),
                            bits(want[k]),
                            "{} {k} {t:e} {p:e}",
                            T::NAME
                        );
                        if t == 0.0 {
                            assert_eq!(bits(got[k]), bits(T::ZERO), "zero sign");
                        }
                    }
                }
            }
        }
        let vals = [
            f64::NAN,
            0.0,
            -0.0,
            1.0,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-40,
        ];
        run::<f64>(&vals);
        run::<f32>(&vals);
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
                args: Vec::new(),
                units: Vec::new(),
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
            args: Vec::new(),
            units: Vec::new(),
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
