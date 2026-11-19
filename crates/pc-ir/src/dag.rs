//! Hash-consed expression DAG with local folding.
//!
//! Every constructor folds first and then looks the node up in a map, so equal
//! structures share one [`NodeId`]. Children always have smaller ids than their parents.
//! Folding never reorders or reassociates operands.

use crate::eval::eval_op;
use std::collections::HashMap;

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
}

/// The size of a [`Dag`] at one moment, from [`Dag::checkpoint`].
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint {
    nodes: usize,
    tables: usize,
    exceeded: bool,
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
        }
    }

    /// Drops every node and table added since `cp` and resets the limit flag to its value
    /// then. Ids returned after `cp` are invalid afterwards, and a caller that keeps a memo
    /// keyed on node ids must drop its entries past [`Checkpoint::len`].
    pub fn restore(&mut self, cp: &Checkpoint) {
        if cp.nodes < self.nodes.len() {
            for op in self.nodes.drain(cp.nodes..) {
                self.map.remove(&op);
            }
            self.nonfinite.truncate(cp.nodes);
        }
        self.tables.truncate(cp.tables);
        self.exceeded = cp.exceeded;
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
        self.nodes.push(op);
        self.nonfinite.push(nonfinite);
        self.map.insert(op, id);
        id
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
        if a == b {
            return self.konst(0.0);
        }
        self.intern(op)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let op = Op::Mul(a, b);
        if let Some(c) = self.fold(op, &[a, b]) {
            return c;
        }
        if self.is_zero(a) || self.is_zero(b) {
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
        if self.is_zero(p) {
            return self.konst(0.0);
        }
        match self.cval(t) {
            Some(0.0) => return self.konst(0.0),
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
        if self.is_zero(n) {
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

    pub fn select(&mut self, c: NodeId, a: NodeId, b: NodeId) -> NodeId {
        if let Some(cv) = self.cval(c) {
            // NaN != 0 is true, so a NaN condition picks `a`.
            return if cv != 0.0 { a } else { b };
        }
        if a == b {
            return a;
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
