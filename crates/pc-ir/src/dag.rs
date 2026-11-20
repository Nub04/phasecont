//! Hash-consed expression DAG with local folding.
//!
//! Every constructor folds first and then looks the node up in a map, so equal
//! structures share one [`NodeId`]. Children always have smaller ids than their parents.
//! Folding never reorders or reassociates operands.

use crate::diff::Activity;
use crate::eval::eval_op;
use std::collections::HashMap;
use std::sync::Arc;

/// Default node limit for a model.
pub const MAX_NODES: usize = 1_000_000;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fn1 {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
    Erf,
    Erfc,
    Lgamma,
    Floor,
    Ceil,
    Heav,
    Sign,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fn2 {
    Atan2,
    Pow,
    Mod,
    BesselJ,
    BesselY,
    Polygamma,
    Min,
    Max,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// f64 bits; nodes hash and compare on bits, so +0.0 and -0.0 are distinct.
    Const(u64),
    /// `u[i]`.
    Var(u32),
    /// `p[j]`; `j == n_p` is time `t`.
    Param(u32),
    /// `v[k]`, the packed direction vector of a directional entry.
    Dir(u32),
    Neg(NodeId),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    /// `ZMul(t, p)` is `+0.0` when `t == 0.0` (either sign), else exactly `t * p`. A NaN `t`
    /// is not zero. The derivative rules use it for `tangent * partial` when the partial
    /// can be infinite or NaN at a finite input: a tangent that is zero at run time then
    /// contributes `+0` instead of `0 * inf = NaN`, as in the fallback interpreter.
    ///
    /// Every evaluator copies this rule bit for bit: compare `t == 0.0`, then select
    /// between the constant `+0.0` and the product `t * p`.
    ZMul(NodeId, NodeId),
    Div(NodeId, NodeId),
    /// `ZDiv(n, d)` is `+0.0` when `n == 0.0` (either sign), else exactly `n / d`. A NaN `n`
    /// is not zero. The derivative rules use it for `tangent / denominator`: a tangent that
    /// is zero at run time over a denominator that is also zero (a clamped quotient inside
    /// `exp`, for example) then contributes `+0` instead of `0 / 0 = NaN`, as in the
    /// fallback interpreter, which skips a zero tangent.
    ///
    /// Every evaluator copies this rule bit for bit, as for [`Op::ZMul`].
    ZDiv(NodeId, NodeId),
    /// Integer power with `2 <= |n| <= 16` (the constructor folds n = 0 and n = 1 and sends
    /// larger |n| to `Pow`).
    ///
    /// Exact evaluation order, which every evaluator must copy bit for bit. Let
    /// `m = |n|`, `base = x`, and `acc` start empty:
    /// ```text
    /// loop {
    ///     if m & 1 == 1 { acc = if acc is empty { base } else { acc * base }; }
    ///     m >>= 1;
    ///     if m == 0 { break; }
    ///     base = base * base;
    /// }
    /// r = acc;                       // so n = 1 gives x exactly
    /// result = if n < 0 { 1.0 / r } else { r };
    /// ```
    /// The accumulator is seeded with the first set bit's power, not with 1.0. The product
    /// is `acc * base` (accumulator on the left). Example: n = 5 gives `x * (x2 * x2)`,
    /// where `x2 = x * x`.
    PowI(NodeId, i32),
    Call1(Fn1, NodeId),
    Call2(Fn2, NodeId, NodeId),
    Cmp(Cmp, NodeId, NodeId),
    And(NodeId, NodeId),
    Or(NodeId, NodeId),
    Not(NodeId),
    /// `cond != 0 ? a : b`; a NaN condition picks `a`.
    Select(NodeId, NodeId, NodeId),
    /// Table id into [`Dag::tables`], then the argument.
    Table(u32, NodeId),
    TableSlope(u32, NodeId),
    // The four ops of recursive functions (SPEC section 8af-5). They come last so the
    // `Debug` text of every other op, which the recursion baseline hashes, stays the same.
    /// Formal `k` of the unit whose body is being evaluated. Bodies of different units share
    /// these leaves, since every unit reads its own frame.
    Arg(u32),
    /// `Call(unit, args)`: a call of unit `unit` (an index into [`Dag::units`]) with the
    /// argument list `args` (an index into the argument lists, see [`Dag::args`]). Its
    /// value is output 0 of the unit.
    Call(u32, u32),
    /// `cond != 0 ? a : b` that evaluates `cond` and then only the taken arm; a NaN
    /// condition takes `a`, as in [`Op::Select`]. The builder makes it instead of a
    /// `Select` exactly when an arm reaches a [`Op::Call`].
    If(NodeId, NodeId, NodeId),
    /// Output `j >= 1` of the `Call` node `call`, for a unit with several outputs.
    Out(NodeId, u32),
}

/// A callable unit: a recursive function, or a derivative of one built by `diff` or
/// `multi`. Its body is the list of output roots, set by [`Dag::set_body`].
#[derive(Clone, Debug)]
pub struct Unit {
    /// The function name, for the depth cap message.
    pub name: String,
    pub arity: u32,
    pub nout: u32,
    body: Option<Body>,
}

const NO_SLOT: u32 = u32::MAX;

#[derive(Clone, Debug)]
struct Body {
    outputs: Vec<NodeId>,
    /// The frame slot of every node of the body that depends on an `Arg`. A `Call` node
    /// takes `nout` consecutive slots, one per output of its unit, and `Out(call, j)` maps
    /// to slot `j` of its call. Dense over ids `lo..lo + locals.len()`, with `NO_SLOT` for
    /// ids that have no slot, so a lookup is one bounds check.
    lo: u32,
    locals: Vec<u32>,
    nlocal: u32,
    /// The slot of `Arg(k)`, or `None` when the body does not read formal `k`.
    arg_slots: Vec<Option<u32>>,
}

impl Unit {
    /// The output roots, or `None` before the body is set.
    pub fn outputs(&self) -> Option<&[NodeId]> {
        self.body.as_ref().map(|b| b.outputs.as_slice())
    }

    /// The frame slot of node `id`, when `id` depends on an `Arg` and is in the body.
    pub fn local(&self, id: NodeId) -> Option<u32> {
        let b = self.body.as_ref()?;
        let i = id.0.checked_sub(b.lo)? as usize;
        b.locals.get(i).copied().filter(|&s| s != NO_SLOT)
    }

    /// The number of frame slots of one call.
    pub fn nlocal(&self) -> u32 {
        self.body.as_ref().map_or(0, |b| b.nlocal)
    }

    /// The frame slot of formal `k`, when the body reads it.
    pub fn arg_slot(&self, k: u32) -> Option<u32> {
        self.body
            .as_ref()
            .and_then(|b| b.arg_slots.get(k as usize).copied().flatten())
    }
}

/// The size of a [`Dag`] at one moment, from [`Dag::checkpoint`].
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    nodes: usize,
    tables: usize,
    exceeded: bool,
    arglists: usize,
    units: usize,
    derived_log: usize,
}

impl Checkpoint {
    /// The node count at the checkpoint: ids below it stay valid after a restore.
    pub fn len(&self) -> usize {
        self.nodes
    }

    pub fn is_empty(&self) -> bool {
        self.nodes == 0
    }
}

pub struct Dag {
    nodes: Vec<Op>,
    /// `nonfinite[i]` is [`Dag::may_be_nonfinite`] of node `i`, set when the node is added.
    nonfinite: Vec<bool>,
    map: HashMap<Op, NodeId>,
    pub tables: Vec<Vec<f64>>,
    cap: usize,
    exceeded: bool,
    /// `has_call[i]`: node `i` is a `Call` or `Out` or has an operand that has one.
    has_call: Vec<bool>,
    /// `arg_dep[i]`: node `i` is an `Arg` or has an operand that depends on one.
    arg_dep: Vec<bool>,
    arglists: Vec<Vec<NodeId>>,
    arg_map: HashMap<Vec<NodeId>, u32>,
    units: Vec<Unit>,
    /// Derived units (tangent and jet units) by the key their builder gives them.
    derived: HashMap<Vec<u32>, u32>,
    /// Each [`Dag::set_derived`] in order, with the value its key had before, so
    /// [`Dag::restore`] can undo the entries set after a checkpoint.
    derived_log: Vec<(Vec<u32>, Option<u32>)>,
    /// The activity of each unit whose closure is complete (see `diff::activity`). Every
    /// restore clears it.
    activity: HashMap<u32, Arc<Activity>>,
    /// How many times a derivative was refused, and the reason of the last refusal.
    refused_hits: u64,
    refused_last: Option<NoDerivative>,
}

/// Why a derivative could not be built (SPEC section 8af-5). The builder then puts a NaN
/// placeholder in the derivative and records the reason on the DAG, and
/// [`crate::IrModel::entry`] turns it into an error, so the NaN never reaches a caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoDerivative {
    /// The derivative calls this unit, whose body is not set.
    MissingBody(u32),
    /// The derivative goes through a call of this unit, which has more than one output.
    SeveralOutputs(u32),
    /// The derivative meets a node with no rule: a formal outside its body, an unknown
    /// unit, or a broken internal invariant.
    NoRule,
}

impl NoDerivative {
    fn unit(self) -> Option<u32> {
        match self {
            NoDerivative::MissingBody(u) | NoDerivative::SeveralOutputs(u) => Some(u),
            NoDerivative::NoRule => None,
        }
    }
}

impl Dag {
    pub fn new(cap: usize) -> Self {
        Dag {
            nodes: Vec::new(),
            nonfinite: Vec::new(),
            map: HashMap::new(),
            tables: Vec::new(),
            cap,
            exceeded: false,
            has_call: Vec::new(),
            arg_dep: Vec::new(),
            arglists: Vec::new(),
            arg_map: HashMap::new(),
            units: Vec::new(),
            derived: HashMap::new(),
            derived_log: Vec::new(),
            activity: HashMap::new(),
            refused_hits: 0,
            refused_last: None,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Records the current size so [`Dag::restore`] can drop everything added later.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            nodes: self.nodes.len(),
            tables: self.tables.len(),
            exceeded: self.exceeded,
            arglists: self.arglists.len(),
            units: self.units.len(),
            derived_log: self.derived_log.len(),
        }
    }

    /// Drops every node, table, argument list and unit added since `cp` and resets the
    /// limit flag to its value then. A unit made before `cp` whose body reads a dropped node
    /// loses its body, and every derived entry set after `cp` is undone, so an overwritten
    /// entry gets its old value back. Ids
    /// returned after `cp` are invalid afterwards, and a caller that keeps a memo keyed on
    /// node ids must drop its entries past [`Checkpoint::len`].
    pub fn restore(&mut self, cp: &Checkpoint) {
        if cp.nodes < self.nodes.len() {
            for op in self.nodes.drain(cp.nodes..) {
                self.map.remove(&op);
            }
            self.nonfinite.truncate(cp.nodes);
            self.has_call.truncate(cp.nodes);
            self.arg_dep.truncate(cp.nodes);
        }
        if cp.arglists < self.arglists.len() {
            for l in self.arglists.drain(cp.arglists..) {
                self.arg_map.remove(&l);
            }
        }
        if cp.units < self.units.len() {
            self.units.truncate(cp.units);
        }
        // Undo in reverse order, so a key set twice ends at its value before `cp`.
        while self.derived_log.len() > cp.derived_log {
            let (key, old) = self.derived_log.pop().expect("log is longer than cp");
            match old {
                Some(u) => self.derived.insert(key, u),
                None => self.derived.remove(&key),
            };
        }
        let n = cp.nodes;
        for u in &mut self.units {
            if u.body
                .as_ref()
                .is_some_and(|b| b.outputs.iter().any(|o| o.0 as usize >= n))
            {
                u.body = None;
            }
        }
        self.tables.truncate(cp.tables);
        self.exceeded = cp.exceeded;
        self.activity.clear();
        let gone = |w: NoDerivative| w.unit().is_some_and(|u| u as usize >= self.units.len());
        if self.refused_last.is_some_and(gone) {
            self.refused_last = None;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// True once a node was added past the cap. The DAG keeps working after that.
    pub fn exceeded(&self) -> bool {
        self.exceeded
    }

    /// The node limit: adding a new node while the DAG already holds this many sets
    /// [`Dag::exceeded`].
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Changes the node limit. Nodes already in the DAG stay. The new limit applies to the
    /// nodes added from now on, so a limit at or below `len()` makes the next new node
    /// exceed it. The default is `MAX_NODES`; this exists so a test can reach the
    /// `TooLarge` path without a model of a million nodes.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
    }

    /// The op of `id`. An id from another DAG may be out of range; that returns `Const(NaN)`
    /// rather than panicking.
    pub fn op(&self, id: NodeId) -> &Op {
        const NAN_CONST: Op = Op::Const(0x7ff8_0000_0000_0000);
        self.nodes.get(id.0 as usize).unwrap_or(&NAN_CONST)
    }

    /// Stores a table block (layout in `math::table`) and returns its id.
    pub fn add_table(&mut self, data: Vec<f64>) -> u32 {
        self.tables.push(data);
        (self.tables.len() - 1) as u32
    }

    fn intern(&mut self, op: Op) -> NodeId {
        if let Some(&id) = self.map.get(&op) {
            return id;
        }
        if self.nodes.len() >= self.cap {
            self.exceeded = true;
        }
        let id = NodeId(self.nodes.len() as u32);
        let nonfinite = self.value_may_be_nonfinite(&op);
        let (mut has_call, mut arg_dep) = match op {
            Op::Call(..) | Op::Out(..) => (true, false),
            Op::Arg(_) => (false, true),
            _ => (false, false),
        };
        self.operands(&op, &mut |c| {
            has_call |= self.has_call.get(c.0 as usize).copied().unwrap_or(false);
            arg_dep |= self.arg_dep.get(c.0 as usize).copied().unwrap_or(false);
        });
        self.nodes.push(op);
        self.nonfinite.push(nonfinite);
        self.has_call.push(has_call);
        self.arg_dep.push(arg_dep);
        self.map.insert(op, id);
        id
    }

    /// Calls `f` on every operand of `op`, including the arguments of a [`Op::Call`],
    /// which [`crate::eval::children`] does not visit. The body of a called unit is not an
    /// operand: a walker that needs it goes through [`Dag::unit`].
    pub fn operands(&self, op: &Op, f: &mut dyn FnMut(NodeId)) {
        match *op {
            Op::Call(_, l) => {
                for &a in self.args(l) {
                    f(a);
                }
            }
            _ => crate::eval::children(op, f),
        }
    }

    /// True when node `id` is a [`Op::Call`] or [`Op::Out`] or reaches one through its
    /// operands. An id from another DAG gives false.
    pub fn has_call(&self, id: NodeId) -> bool {
        self.has_call.get(id.0 as usize).copied().unwrap_or(false)
    }

    /// True when node `id` is an [`Op::Arg`] or reaches one through its operands, so its
    /// value differs from one call frame to the next.
    pub fn arg_dep(&self, id: NodeId) -> bool {
        self.arg_dep.get(id.0 as usize).copied().unwrap_or(false)
    }

    /// The argument list `l`, or an empty list for an unknown index.
    pub fn args(&self, l: u32) -> &[NodeId] {
        self.arglists.get(l as usize).map_or(&[], |v| v.as_slice())
    }

    /// The units, in the order they were added.
    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    /// Unit `u`, or `None` for an unknown index.
    pub fn unit(&self, u: u32) -> Option<&Unit> {
        self.units.get(u as usize)
    }

    /// Adds a unit of `arity` formals and `nout` outputs with no body yet, and returns its
    /// index. `nout` is at least 1.
    pub fn add_unit(&mut self, name: &str, arity: u32, nout: u32) -> u32 {
        self.units.push(Unit {
            name: name.to_string(),
            arity,
            nout: nout.max(1),
            body: None,
        });
        (self.units.len() - 1) as u32
    }

    /// The derived unit stored under `key`, if any.
    pub fn derived(&self, key: &[u32]) -> Option<u32> {
        self.derived.get(key).copied()
    }

    /// Stores derived unit `u` under `key`. [`Dag::restore`] undoes it if it came after the
    /// checkpoint.
    pub fn set_derived(&mut self, key: Vec<u32>, u: u32) {
        let old = self.derived.insert(key.clone(), u);
        self.derived_log.push((key, old));
    }

    /// The unit without a body that the last refused derivative met, if its body is still
    /// unset: a call into a function whose body, or the body of a function it calls, is
    /// missing has no derivative (SPEC section 8af-5).
    pub fn missing_body(&self) -> Option<u32> {
        match self.refused_last {
            Some(NoDerivative::MissingBody(u)) => Some(u),
            _ => None,
        }
    }

    /// The reason of the last refused derivative (see [`NoDerivative`]).
    pub fn refusal(&self) -> Option<NoDerivative> {
        self.refused_last
    }

    pub(crate) fn note_refusal(&mut self, why: NoDerivative) {
        self.refused_hits += 1;
        self.refused_last = Some(why);
    }

    /// How many derivatives were refused so far; a builder compares it before and after.
    pub(crate) fn refusal_hits(&self) -> u64 {
        self.refused_hits
    }

    pub(crate) fn cached_activity(&self, u: u32) -> Option<Arc<Activity>> {
        self.activity.get(&u).cloned()
    }

    pub(crate) fn cache_activity(&mut self, u: u32, a: Arc<Activity>) {
        self.activity.insert(u, a);
    }

    /// Sets the body of unit `u` to the output roots `outputs`, one per output of the unit,
    /// and assigns a frame slot to every node of the body that depends on an `Arg`. A `Call`
    /// node takes one slot per output of its unit, and `Out(call, j)` reads slot `j` of its
    /// call instead of taking one. Returns false, and leaves the unit as it
    /// was, for an unknown unit, a unit that has a body already, a wrong number of outputs
    /// or an id outside the DAG. A body is set once: derivatives, tangent units and cached
    /// entries built from it would go stale if it changed. A body that a
    /// [`Dag::restore`] dropped can be set again.
    #[must_use = "a body that is already set is not replaced"]
    pub fn set_body(&mut self, u: u32, outputs: &[NodeId]) -> bool {
        let Some(unit) = self.units.get(u as usize) else {
            return false;
        };
        if unit.body.is_some()
            || outputs.len() != unit.nout as usize
            || outputs.iter().any(|o| o.0 as usize >= self.nodes.len())
        {
            return false;
        }
        let arity = unit.arity as usize;
        // A first body cannot change a cached activity, since a closure that meets a unit
        // without a body is never cached.
        if self.refused_last == Some(NoDerivative::MissingBody(u)) {
            self.refused_last = None;
        }
        // Every node of a body has a smaller id than the largest output, so a walk down from
        // the outputs in decreasing id order meets each local once.
        let mut seen = std::collections::HashSet::new();
        let mut stack: Vec<NodeId> = outputs
            .iter()
            .copied()
            .filter(|&o| self.arg_dep(o))
            .collect();
        let mut found = Vec::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            found.push(id);
            let op = *self.op(id);
            self.operands(&op, &mut |c| {
                if self.arg_dep(c) && !seen.contains(&c) {
                    stack.push(c);
                }
            });
        }
        found.sort_unstable();
        let lo = found.first().map_or(0, |n| n.0);
        let span = found.last().map_or(0, |n| (n.0 - lo) as usize + 1);
        let mut locals = vec![NO_SLOT; span];
        let at = |id: NodeId| (id.0 - lo) as usize;
        let mut arg_slots = vec![None; arity];
        let mut next = 0u32;
        for id in found {
            match *self.op(id) {
                Op::Arg(k) => {
                    if let Some(s) = arg_slots.get_mut(k as usize) {
                        *s = Some(next);
                    }
                    locals[at(id)] = next;
                    next += 1;
                }
                Op::Call(c, _) => {
                    locals[at(id)] = next;
                    next += self.units.get(c as usize).map_or(1, |x| x.nout);
                }
                // `Out(call, j)` reads slot `j` of its call, which has a smaller id and so
                // already has its slots.
                Op::Out(c, j) => {
                    if let Some(&base) =
                        c.0.checked_sub(lo)
                            .and_then(|i| locals.get(i as usize))
                            .filter(|&&b| b != NO_SLOT)
                    {
                        locals[at(id)] = base + j;
                    }
                }
                _ => {
                    locals[at(id)] = next;
                    next += 1;
                }
            }
        }
        self.units[u as usize].body = Some(Body {
            outputs: outputs.to_vec(),
            lo,
            locals,
            nlocal: next,
            arg_slots,
        });
        true
    }

    /// Formal `k` of the unit whose body is being built.
    pub fn arg(&mut self, k: u32) -> NodeId {
        self.intern(Op::Arg(k))
    }

    /// A call of unit `u` with the arguments `args`; its value is output 0. A call is never
    /// folded, not even with constant arguments, since its value comes from the frame
    /// machine.
    pub fn call(&mut self, u: u32, args: &[NodeId]) -> NodeId {
        let l = match self.arg_map.get(args) {
            Some(&l) => l,
            None => {
                self.arglists.push(args.to_vec());
                let l = (self.arglists.len() - 1) as u32;
                self.arg_map.insert(args.to_vec(), l);
                l
            }
        };
        self.intern(Op::Call(u, l))
    }

    /// Output `j` of the call node `call`. Output 0 is the call node itself.
    pub fn out(&mut self, call: NodeId, j: u32) -> NodeId {
        if j == 0 {
            return call;
        }
        self.intern(Op::Out(call, j))
    }

    /// True when node `id` may be infinite or NaN at finite inputs. The derivative rules use
    /// it to decide whether a product term needs a [`Op::ZMul`]: a tangent that is zero at
    /// run time times a factor that is infinite is `0 * inf = NaN`, where the interpreter
    /// skips the zero tangent. The flag is computed once, when the node is added.
    ///
    /// It is true for a constant that is not finite, a division by anything but a nonzero
    /// finite constant, a call whose partial can be infinite or NaN (the list of
    /// `diff::partial_may_be_nonfinite`, which here also stands for its value) and a negative
    /// integer power. A sum, product, selection or other call is flagged when an operand is.
    /// Variables, parameters, directions, comparisons and the `floor`, `ceil`, `heav` and
    /// `sign` of a finite argument are not. The overflow of `exp`, `sinh`, `cosh` or a
    /// product of large finite numbers is not tracked, so a polynomial or an `exp` of
    /// finite arguments carries no flag and its product terms stay plain products.
    ///
    /// An id from another DAG is out of range and is flagged.
    pub fn may_be_nonfinite(&self, id: NodeId) -> bool {
        self.nonfinite.get(id.0 as usize).copied().unwrap_or(true)
    }

    fn value_may_be_nonfinite(&self, op: &Op) -> bool {
        let nf = |id: NodeId| self.may_be_nonfinite(id);
        match *op {
            Op::Const(b) => !f64::from_bits(b).is_finite(),
            Op::Var(_) | Op::Param(_) | Op::Dir(_) => false,
            Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) => false,
            Op::Neg(a) => nf(a),
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::ZMul(a, b) => nf(a) || nf(b),
            Op::Div(a, b) | Op::ZDiv(a, b) => match self.cval(b) {
                Some(c) if c != 0.0 && c.is_finite() => nf(a),
                _ => true,
            },
            Op::Select(_, a, b) => nf(a) || nf(b),
            // A finite argument gives a finite value for these four.
            Op::Call1(Fn1::Floor | Fn1::Ceil | Fn1::Heav | Fn1::Sign, a) => nf(a),
            Op::Call1(_, a) => crate::diff::partial_may_be_nonfinite(self, op) || nf(a),
            Op::PowI(a, _) => crate::diff::partial_may_be_nonfinite(self, op) || nf(a),
            Op::Table(_, a) | Op::TableSlope(_, a) => nf(a),
            Op::Call2(_, a, b) => crate::diff::partial_may_be_nonfinite(self, op) || nf(a) || nf(b),
            // Nothing is known about a formal or the value of a call.
            Op::Arg(_) | Op::Call(..) | Op::Out(..) => true,
            Op::If(_, a, b) => nf(a) || nf(b),
        }
    }

    fn cval(&self, id: NodeId) -> Option<f64> {
        match self.op(id) {
            Op::Const(b) => Some(f64::from_bits(*b)),
            _ => None,
        }
    }

    fn is_zero(&self, id: NodeId) -> bool {
        self.cval(id) == Some(0.0)
    }

    fn is_one(&self, id: NodeId) -> bool {
        self.cval(id) == Some(1.0)
    }

    /// Evaluates `op` when every operand is a constant, using the interpreter's code.
    fn fold(&mut self, op: Op, args: &[NodeId]) -> Option<NodeId> {
        if !args.iter().all(|&a| self.cval(a).is_some()) {
            return None;
        }
        let r = {
            let get = |id: NodeId| self.cval(id).unwrap_or(f64::NAN);
            eval_op(&op, &self.tables, &get, &[], &[], &[])
        };
        Some(self.konst(r))
    }

    pub fn konst(&mut self, x: f64) -> NodeId {
        self.intern(Op::Const(x.to_bits()))
    }

    pub fn var(&mut self, i: u32) -> NodeId {
        self.intern(Op::Var(i))
    }

    pub fn param(&mut self, j: u32) -> NodeId {
        self.intern(Op::Param(j))
    }

    pub fn dir(&mut self, k: u32) -> NodeId {
        self.intern(Op::Dir(k))
    }

    pub fn neg(&mut self, a: NodeId) -> NodeId {
        let op = Op::Neg(a);
        if let Some(c) = self.fold(op, &[a]) {
            return c;
        }
        if let Op::Neg(inner) = *self.op(a) {
            return inner;
        }
        self.intern(op)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Add(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        if self.is_zero(b) {
            return a;
        }
        if self.is_zero(a) {
            return b;
        }
        self.intern(op)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Sub(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        if self.is_zero(b) {
            return a;
        }
        if self.is_zero(a) {
            return self.neg(b);
        }
        // `f(n) - f(n)` stays, so a depth cap error inside the call is not folded away.
        if a == b && !self.has_call(a) {
            return self.konst(0.0);
        }
        self.intern(op)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Mul(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        // A zero factor folds to +0 unless the other factor reaches a call, whose depth cap
        // error must not be folded away.
        if (self.is_zero(a) && !self.has_call(b)) || (self.is_zero(b) && !self.has_call(a)) {
            return self.konst(0.0);
        }
        if self.is_one(b) {
            return a;
        }
        if self.is_one(a) {
            return b;
        }
        self.intern(op)
    }

    /// `t * p` with a zero `t` giving `+0.0` (see [`Op::ZMul`]). A constant zero `t` folds to
    /// `+0.0` whatever `p` is, and so does a constant zero `p`, as in [`Dag::mul`]. A nonzero
    /// finite constant `t` cannot be zero at run time, so the node becomes an ordinary
    /// `mul`. An infinite or NaN constant `t` stays a `ZMul` node. `ZMul` and `Mul` never
    /// merge, since they differ when `t` is zero and `p` is not finite.
    pub fn zmul(&mut self, t: NodeId, p: NodeId) -> NodeId {
        let op = Op::ZMul(t, p);
        if let Some(c) = self.fold(op, &[t, p]) {
            return c;
        }
        // A static zero factor folds to +0 whatever `t` is, as in `mul`.
        if self.is_zero(p) && !self.has_call(t) {
            return self.konst(0.0);
        }
        match self.cval(t) {
            Some(0.0) if !self.has_call(p) => return self.konst(0.0),
            Some(0.0) => {}
            Some(c) if c.is_finite() => return self.mul(t, p),
            _ => {}
        }
        self.intern(op)
    }

    /// `n / d` with a zero `n` giving `+0.0` (see [`Op::ZDiv`]). A constant zero `n` folds to
    /// `+0.0` whatever `d` is. A `ZDiv` never becomes a `Div` here, not even for `d = 1` or
    /// a finite constant `d`, since a `-0.0` numerator gives `-0.0` through `Div` and `+0.0`
    /// through `ZDiv`. `ZDiv` and `Div` never merge for the same reason.
    pub fn zdiv(&mut self, n: NodeId, d: NodeId) -> NodeId {
        let op = Op::ZDiv(n, d);
        if let Some(c) = self.fold(op, &[n, d]) {
            return c;
        }
        if self.is_zero(n) && !self.has_call(d) {
            return self.konst(0.0);
        }
        self.intern(op)
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Div(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        if self.is_one(b) {
            return a;
        }
        self.intern(op)
    }

    /// `a^n` by repeated multiplication (see [`Op::PowI`]). For `|n| > 16` this returns a
    /// `Call2(Pow, a, Const(n as f64))` node instead.
    pub fn powi(&mut self, a: NodeId, n: i32) -> NodeId {
        if n == 0 && self.has_call(a) {
            // `a^0` is 1, but the call in `a` still runs so its depth cap error shows.
            let e = self.konst(0.0);
            return self.call2(Fn2::Pow, a, e);
        }
        if n == 0 {
            return self.konst(1.0);
        }
        if n == 1 {
            return a;
        }
        if n.unsigned_abs() > 16 {
            let e = self.konst(f64::from(n));
            return self.call2(Fn2::Pow, a, e);
        }
        let op = Op::PowI(a, n);
        if let Some(c) = self.fold(op, &[a]) {
            return c;
        }
        self.intern(op)
    }

    pub fn call1(&mut self, f: Fn1, a: NodeId) -> NodeId {
        let op = Op::Call1(f, a);
        if let Some(c) = self.fold(op, &[a]) {
            return c;
        }
        self.intern(op)
    }

    pub fn call2(&mut self, f: Fn2, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Call2(f, a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        self.intern(op)
    }

    pub fn cmp(&mut self, c: Cmp, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Cmp(c, a, b);
        if let Some(r) = self.fold(op, &[a, b]) {
            return r;
        }
        self.intern(op)
    }

    pub fn and(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::And(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        self.intern(op)
    }

    pub fn or(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Or(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        self.intern(op)
    }

    pub fn not(&mut self, a: NodeId) -> NodeId {
        let op = Op::Not(a);
        if let Some(c) = self.fold(op, &[a]) {
            return c;
        }
        self.intern(op)
    }

    /// `op` with each operand `c` replaced by `map(c)`, built through the folding
    /// constructors, so a rebuilt node folds as a freshly built one would. The arguments of
    /// a call are mapped too. An `If` goes through [`Dag::select`], which makes an `If`
    /// again exactly when an arm reaches a call.
    pub fn rebuild(&mut self, op: &Op, map: &dyn Fn(NodeId) -> NodeId) -> NodeId {
        match *op {
            Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) | Op::Arg(_) => self.intern(*op),
            Op::Neg(a) => self.neg(map(a)),
            Op::Add(a, b) => self.add(map(a), map(b)),
            Op::Sub(a, b) => self.sub(map(a), map(b)),
            Op::Mul(a, b) => self.mul(map(a), map(b)),
            Op::ZMul(a, b) => self.zmul(map(a), map(b)),
            Op::Div(a, b) => self.div(map(a), map(b)),
            Op::ZDiv(a, b) => self.zdiv(map(a), map(b)),
            Op::PowI(a, k) => self.powi(map(a), k),
            Op::Call1(f, a) => self.call1(f, map(a)),
            Op::Call2(f, a, b) => self.call2(f, map(a), map(b)),
            Op::Cmp(k, a, b) => self.cmp(k, map(a), map(b)),
            Op::And(a, b) => self.and(map(a), map(b)),
            Op::Or(a, b) => self.or(map(a), map(b)),
            Op::Not(a) => self.not(map(a)),
            Op::Select(c, a, b) | Op::If(c, a, b) => self.select(map(c), map(a), map(b)),
            Op::Table(t, a) => self.table(t, map(a)),
            Op::TableSlope(t, a) => self.table_slope(t, map(a)),
            Op::Call(u, l) => {
                let args: Vec<NodeId> = self.args(l).iter().map(|&a| map(a)).collect();
                self.call(u, &args)
            }
            Op::Out(c, j) => self.out(map(c), j),
        }
    }

    pub fn select(&mut self, c: NodeId, a: NodeId, b: NodeId) -> NodeId {
        if let Some(cv) = self.cval(c) {
            // NaN != 0 is true, so a NaN condition picks `a`.
            return if cv != 0.0 { a } else { b };
        }
        if a == b && !self.has_call(c) {
            return a;
        }
        // An arm that reaches a call is evaluated only when it is taken.
        if self.has_call(a) || self.has_call(b) {
            return self.intern(Op::If(c, a, b));
        }
        self.intern(Op::Select(c, a, b))
    }

    pub fn table(&mut self, id: u32, x: NodeId) -> NodeId {
        let op = Op::Table(id, x);
        if let Some(c) = self.fold(op, &[x]) {
            return c;
        }
        self.intern(op)
    }

    pub fn table_slope(&mut self, id: u32, x: NodeId) -> NodeId {
        let op = Op::TableSlope(id, x);
        if let Some(c) = self.fold(op, &[x]) {
            return c;
        }
        self.intern(op)
    }
}
