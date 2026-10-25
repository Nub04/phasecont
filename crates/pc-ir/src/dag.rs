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
    Div(NodeId, NodeId),
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

pub struct Dag {
    nodes: Vec<Op>,
    map: HashMap<Op, NodeId>,
    pub tables: Vec<Vec<f64>>,
    cap: usize,
    exceeded: bool,
}

impl Dag {
    pub fn new(cap: usize) -> Self {
        Dag {
            nodes: Vec::new(),
            map: HashMap::new(),
            tables: Vec::new(),
            cap,
            exceeded: false,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// True once a node was added past the cap. The DAG keeps working after that.
    pub fn exceeded(&self) -> bool {
        self.exceeded
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
        self.nodes.push(op);
        self.map.insert(op, id);
        id
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
