//! The frame machine: evaluates a DAG whose outputs reach a [`Op::Call`] (SPEC section
//! 8af-5).
//!
//! The machine keeps its work on the heap: a stack of tasks, a stack of call frames and one
//! stack of frame slots. The native stack does not grow with the recursion depth, so a run
//! at the cap of [`MAX_ACTIVE_CALLS`] works on a thread with a small stack.
//!
//! A node that does not depend on a formal argument ([`Dag::arg_dep`] false) has one value
//! for the whole evaluation and is stored once, in the global memo, even when a call body
//! reads it. A node that depends on a formal is stored in a slot of the frame of the call
//! that reads it ([`crate::dag::Unit::local`]), so each frame evaluates it at most once: a
//! call node named by both arms of an `If`, or by both operands of an add, runs once.
//!
//! `If` is lazy: the machine evaluates the condition, then only the arm it takes. A NaN
//! condition takes the first arm, as [`Op::Select`] does. Every other op evaluates all of
//! its operands, with the scalar rules of [`crate::eval::eval_op`], so an output that
//! reaches no call gets the bits of the eager evaluator.
//!
//! With error bounds asked for, each value carries the running bound of
//! [`crate::eval::eval_err`]. A call adds no rounding of its own: a formal takes the bound of
//! its argument, and a call takes the bound of the body output.

use std::collections::HashMap;

use crate::dag::{Dag, NodeId, Op};
use crate::eval::{children, err_op, eval_op};

/// Most call frames that may be active at once. Entering one more fails with
/// [`EvalError::Depth`].
pub const MAX_ACTIVE_CALLS: usize = 1_000_000;

/// What one evaluation did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// Call frames entered.
    pub calls: u64,
    /// Most call frames active at once.
    pub max_depth: usize,
}

/// Why an evaluation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    /// Entering a call of `func` would make more than `cap` calls active at once.
    Depth { func: String, cap: usize },
    /// A stack of the machine could not grow.
    OutOfMemory,
    /// The DAG breaks a rule of the call ops: a call of a unit without a body or with the
    /// wrong number of arguments, or a formal read outside a call.
    Malformed(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Depth { func, cap } => write!(
                f,
                "a call of {func} would make more than {cap} calls active at once"
            ),
            EvalError::OutOfMemory => write!(f, "out of memory for the call frames"),
            EvalError::Malformed(m) => write!(f, "malformed call DAG: {m}"),
        }
    }
}

impl std::error::Error for EvalError {}

fn malformed(m: &str) -> EvalError {
    EvalError::Malformed(m.to_string())
}

fn grow<T>(v: &mut Vec<T>, n: usize) -> Result<(), EvalError> {
    v.try_reserve(n).map_err(|_| EvalError::OutOfMemory)
}

/// [`run`] at the cap of [`MAX_ACTIVE_CALLS`], without error bounds. Outputs that reach no
/// call go through the eager evaluator. On an error every `out` slot is NaN.
pub fn try_eval(
    dag: &Dag,
    outputs: &[NodeId],
    x: &[f64],
    p: &[f64],
    v: &[f64],
    out: &mut [f64],
) -> Result<EvalStats, EvalError> {
    if !needs_machine(dag, outputs) {
        crate::eval::eval_eager(dag, outputs, x, p, v, out);
        return Ok(EvalStats::default());
    }
    run(dag, outputs, x, p, v, out, None, MAX_ACTIVE_CALLS)
}

/// [`try_eval`] that also writes the running error bound of each output into `err`.
pub fn try_eval_err(
    dag: &Dag,
    outputs: &[NodeId],
    x: &[f64],
    p: &[f64],
    v: &[f64],
    out: &mut [f64],
    err: &mut [f64],
) -> Result<EvalStats, EvalError> {
    if !needs_machine(dag, outputs) {
        crate::eval::eval_err_eager(dag, outputs, x, p, v, out, err);
        return Ok(EvalStats::default());
    }
    run(dag, outputs, x, p, v, out, Some(err), MAX_ACTIVE_CALLS)
}

/// Evaluates `outputs` with the frame machine, at most `cap` calls active at once, and
/// writes `out[i] = value(outputs[i])` and, when `err` is given, `err[i]` = its bound.
/// Extra slots and output ids outside the DAG are NaN. On an error every slot is NaN.
#[allow(clippy::too_many_arguments)]
pub fn run(
    dag: &Dag,
    outputs: &[NodeId],
    x: &[f64],
    p: &[f64],
    v: &[f64],
    out: &mut [f64],
    mut err: Option<&mut [f64]>,
    cap: usize,
) -> Result<EvalStats, EvalError> {
    out.fill(f64::NAN);
    if let Some(e) = err.as_deref_mut() {
        e.fill(f64::NAN);
    }
    let mut m = Machine::new(dag, x, p, v, cap, err.is_some())?;
    let mut vals = Vec::new();
    grow(&mut vals, outputs.len())?;
    for &o in outputs {
        if o.0 as usize >= dag.len() {
            vals.push((f64::NAN, f64::NAN));
            continue;
        }
        m.push(Task::Eval(o))?;
        m.drain()?;
        vals.push(m.read(o)?);
    }
    for (slot, &(val, _)) in out.iter_mut().zip(&vals) {
        *slot = val;
    }
    if let Some(e) = err {
        for (slot, &(_, b)) in e.iter_mut().zip(&vals) {
            *slot = b;
        }
    }
    Ok(m.stats)
}

/// Whether an output reaches a call, or reads a formal (which only the machine reports as
/// malformed).
fn needs_machine(dag: &Dag, outputs: &[NodeId]) -> bool {
    outputs.iter().any(|&o| dag.has_call(o) || dag.arg_dep(o))
}

#[derive(Debug, Clone, Copy)]
enum Task {
    /// Evaluate a node unless its value is stored.
    Eval(NodeId),
    /// Apply a node's op to the stored values of its operands.
    Compute(NodeId),
    /// The condition of an `If` is stored: evaluate the arm it takes.
    IfPick(NodeId),
    /// The arm (second field) of an `If` is stored: copy it into the `If`.
    IfFinish(NodeId, NodeId),
    /// The arguments of a call are stored: enter its frame.
    CallEnter(NodeId),
    /// The body outputs of a call are stored: leave its frame.
    CallExit(NodeId),
}

#[derive(Debug, Clone, Copy)]
struct Frame {
    unit: u32,
    /// First slot of the frame in the slot stack.
    base: usize,
}

/// Values, error bounds and set flags of a run of nodes or slots. The bounds are kept only
/// when asked for.
struct Store {
    val: Vec<f64>,
    err: Vec<f64>,
    set: Vec<u64>,
    want_err: bool,
}

impl Store {
    fn new(want_err: bool) -> Store {
        Store {
            val: Vec::new(),
            err: Vec::new(),
            set: Vec::new(),
            want_err,
        }
    }

    fn len(&self) -> usize {
        self.val.len()
    }

    /// Appends `n` unset entries.
    fn extend(&mut self, n: usize) -> Result<(), EvalError> {
        let len = self.val.len();
        let new = len.checked_add(n).ok_or(EvalError::OutOfMemory)?;
        grow(&mut self.val, n)?;
        self.val.resize(new, f64::NAN);
        if self.want_err {
            grow(&mut self.err, n)?;
            self.err.resize(new, f64::NAN);
        }
        let words = new.div_ceil(64);
        if words > self.set.len() {
            let more = words - self.set.len();
            grow(&mut self.set, more)?;
            self.set.resize(words, 0);
        }
        for i in len..new {
            self.set[i / 64] &= !(1u64 << (i % 64));
        }
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        self.val.truncate(len);
        self.err.truncate(len);
    }

    fn is_set(&self, i: usize) -> bool {
        i < self.val.len() && self.set[i / 64] & (1u64 << (i % 64)) != 0
    }

    fn get(&self, i: usize) -> (f64, f64) {
        let e = if self.want_err { self.err[i] } else { f64::NAN };
        (self.val[i], e)
    }

    fn put(&mut self, i: usize, (v, e): (f64, f64)) -> Result<(), EvalError> {
        if i >= self.val.len() {
            return Err(malformed("a value is stored outside its frame"));
        }
        self.val[i] = v;
        if self.want_err {
            self.err[i] = e;
        }
        self.set[i / 64] |= 1u64 << (i % 64);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Loc {
    Global(usize),
    Slot(usize),
}

struct Machine<'a> {
    dag: &'a Dag,
    x: &'a [f64],
    p: &'a [f64],
    v: &'a [f64],
    cap: usize,
    want_err: bool,
    global: Store,
    slots: Store,
    frames: Vec<Frame>,
    tasks: Vec<Task>,
    /// Outputs of global calls of units with more than one output, read by `Out`.
    outs: HashMap<NodeId, Vec<(f64, f64)>>,
    scratch: Vec<(f64, f64)>,
    stats: EvalStats,
}

impl<'a> Machine<'a> {
    fn new(
        dag: &'a Dag,
        x: &'a [f64],
        p: &'a [f64],
        v: &'a [f64],
        cap: usize,
        want_err: bool,
    ) -> Result<Machine<'a>, EvalError> {
        let mut global = Store::new(want_err);
        global.extend(dag.len())?;
        Ok(Machine {
            dag,
            x,
            p,
            v,
            cap,
            want_err,
            global,
            slots: Store::new(want_err),
            frames: Vec::new(),
            tasks: Vec::new(),
            outs: HashMap::new(),
            scratch: Vec::new(),
            stats: EvalStats::default(),
        })
    }

    fn push(&mut self, t: Task) -> Result<(), EvalError> {
        grow(&mut self.tasks, 1)?;
        self.tasks.push(t);
        Ok(())
    }

    /// Where node `id` is stored in the current frame.
    fn loc(&self, id: NodeId) -> Result<Loc, EvalError> {
        let i = id.0 as usize;
        if i >= self.dag.len() {
            return Err(malformed("an operand outside the DAG"));
        }
        if !self.dag.arg_dep(id) {
            return Ok(Loc::Global(i));
        }
        let f = self
            .frames
            .last()
            .ok_or_else(|| malformed("a formal argument is read outside a call"))?;
        let s = self
            .dag
            .unit(f.unit)
            .and_then(|u| u.local(id))
            .ok_or_else(|| malformed("a node of a call body has no frame slot"))?;
        Ok(Loc::Slot(f.base + s as usize))
    }

    fn is_set(&self, l: Loc) -> bool {
        match l {
            Loc::Global(i) => self.global.is_set(i),
            Loc::Slot(i) => self.slots.is_set(i),
        }
    }

    fn put(&mut self, l: Loc, ve: (f64, f64)) -> Result<(), EvalError> {
        match l {
            Loc::Global(i) => self.global.put(i, ve),
            Loc::Slot(i) => self.slots.put(i, ve),
        }
    }

    /// The stored value and bound of `id` in the current frame.
    fn read(&self, id: NodeId) -> Result<(f64, f64), EvalError> {
        let l = self.loc(id)?;
        if !self.is_set(l) {
            return Err(malformed("a node is read before it is evaluated"));
        }
        Ok(match l {
            Loc::Global(i) => self.global.get(i),
            Loc::Slot(i) => self.slots.get(i),
        })
    }

    fn drain(&mut self) -> Result<(), EvalError> {
        while let Some(t) = self.tasks.pop() {
            self.step(t)?;
        }
        Ok(())
    }

    fn step(&mut self, t: Task) -> Result<(), EvalError> {
        let dag = self.dag;
        match t {
            Task::Eval(id) => {
                let l = self.loc(id)?;
                if self.is_set(l) {
                    return Ok(());
                }
                match *dag.op(id) {
                    // A formal the frame did not bind: its index is past the arity.
                    Op::Arg(_) => return Err(malformed("a formal past the arity of its unit")),
                    Op::If(c, _, _) => {
                        self.push(Task::IfPick(id))?;
                        self.push(Task::Eval(c))?;
                    }
                    Op::Call(_, list) => {
                        self.push(Task::CallEnter(id))?;
                        for &a in dag.args(list).iter().rev() {
                            self.push(Task::Eval(a))?;
                        }
                    }
                    Op::Out(c, _) => {
                        // A local `Out` has the slot its call's exit fills; a global one
                        // copies from the outputs of its global call.
                        if let Loc::Global(_) = l {
                            self.push(Task::Compute(id))?;
                        }
                        self.push(Task::Eval(c))?;
                    }
                    op => {
                        self.push(Task::Compute(id))?;
                        let (kids, nk) = kids(&op);
                        for &c in kids[..nk].iter().rev() {
                            self.push(Task::Eval(c))?;
                        }
                    }
                }
            }
            Task::Compute(id) => self.compute(id)?,
            Task::IfPick(id) => {
                let Op::If(c, a, b) = *dag.op(id) else {
                    return Err(malformed("IfPick on a node that is not an If"));
                };
                let (cv, _) = self.read(c)?;
                let arm = if cv != 0.0 { a } else { b };
                self.push(Task::IfFinish(id, arm))?;
                self.push(Task::Eval(arm))?;
            }
            Task::IfFinish(id, arm) => {
                let ve = self.read(arm)?;
                let l = self.loc(id)?;
                self.put(l, ve)?;
            }
            Task::CallEnter(id) => self.enter(id)?,
            Task::CallExit(id) => self.exit(id)?,
        }
        Ok(())
    }

    fn compute(&mut self, id: NodeId) -> Result<(), EvalError> {
        let op = *self.dag.op(id);
        if let Op::Out(c, j) = op {
            let ve = self
                .outs
                .get(&c)
                .and_then(|o| o.get(j as usize))
                .copied()
                .ok_or_else(|| malformed("Out reads an output its unit does not have"))?;
            let l = self.loc(id)?;
            return self.put(l, ve);
        }
        if matches!(op, Op::Arg(_) | Op::Call(..) | Op::If(..)) {
            return Err(malformed("Compute on a call op"));
        }
        let (ids, nk) = kids(&op);
        let mut vals = [(f64::NAN, f64::NAN); 3];
        for (slot, &c) in vals.iter_mut().zip(&ids[..nk]) {
            *slot = self.read(c)?;
        }
        let find = |c: NodeId| ids[..nk].iter().position(|&k| k == c).map(|i| vals[i]);
        let val = |c: NodeId| find(c).map_or(f64::NAN, |ve| ve.0);
        let r = eval_op(&op, &self.dag.tables, &val, self.x, self.p, self.v);
        let e = if self.want_err {
            let er = |c: NodeId| find(c).map_or(f64::NAN, |ve| ve.1);
            err_op(self.dag, &op, r, &val, &er)
        } else {
            f64::NAN
        };
        let l = self.loc(id)?;
        self.put(l, (r, e))
    }

    fn enter(&mut self, id: NodeId) -> Result<(), EvalError> {
        let dag = self.dag;
        let Op::Call(u, list) = *dag.op(id) else {
            return Err(malformed("CallEnter on a node that is not a call"));
        };
        let unit = dag
            .unit(u)
            .ok_or_else(|| malformed("a call of an unknown unit"))?;
        let outs = unit
            .outputs()
            .ok_or_else(|| EvalError::Malformed(format!("unit {} has no body", unit.name)))?;
        let args = dag.args(list);
        if args.len() != unit.arity as usize {
            return Err(EvalError::Malformed(format!(
                "a call of {} with {} arguments, not {}",
                unit.name,
                args.len(),
                unit.arity
            )));
        }
        self.scratch.clear();
        grow(&mut self.scratch, args.len())?;
        for &a in args {
            let ve = self.read(a)?;
            self.scratch.push(ve);
        }
        if self.frames.len() >= self.cap {
            return Err(EvalError::Depth {
                func: unit.name.clone(),
                cap: self.cap,
            });
        }
        let base = self.slots.len();
        self.slots.extend(unit.nlocal() as usize)?;
        grow(&mut self.frames, 1)?;
        self.frames.push(Frame { unit: u, base });
        self.stats.calls += 1;
        self.stats.max_depth = self.stats.max_depth.max(self.frames.len());
        for k in 0..unit.arity {
            if let Some(s) = unit.arg_slot(k) {
                let ve = self.scratch[k as usize];
                self.slots.put(base + s as usize, ve)?;
            }
        }
        self.push(Task::CallExit(id))?;
        for &o in outs.iter().rev() {
            self.push(Task::Eval(o))?;
        }
        Ok(())
    }

    fn exit(&mut self, id: NodeId) -> Result<(), EvalError> {
        let dag = self.dag;
        let f = *self
            .frames
            .last()
            .ok_or_else(|| malformed("CallExit without a frame"))?;
        let outs = dag
            .unit(f.unit)
            .and_then(|u| u.outputs())
            .ok_or_else(|| malformed("a frame of a unit without a body"))?;
        // The body outputs are read in the callee's frame, then written in the caller's.
        self.scratch.clear();
        grow(&mut self.scratch, outs.len())?;
        for &o in outs {
            let ve = self.read(o)?;
            self.scratch.push(ve);
        }
        self.frames.pop();
        self.slots.truncate(f.base);
        let first = *self
            .scratch
            .first()
            .ok_or_else(|| malformed("a call of a unit without outputs"))?;
        match self.loc(id)? {
            Loc::Slot(s) => {
                for j in 0..self.scratch.len() {
                    let ve = self.scratch[j];
                    self.slots.put(s + j, ve)?;
                }
            }
            l @ Loc::Global(_) => {
                self.put(l, first)?;
                if self.scratch.len() > 1 {
                    self.outs.insert(id, self.scratch.clone());
                }
            }
        }
        Ok(())
    }
}

/// The children of `op` ([`children`]), at most three, and their count.
fn kids(op: &Op) -> ([NodeId; 3], usize) {
    let mut ids = [NodeId(0); 3];
    let mut n = 0;
    children(op, &mut |c| {
        if let Some(s) = ids.get_mut(n) {
            *s = c;
        }
        n += 1;
    });
    (ids, n.min(3))
}
