//! Multilinear derivative forms `D^k f[v1..vk]`, k = 1..=5, by subset expansion.
//!
//! Every node `a` gets components `a_S` for the subsets `S` of the k slots (a bitmask over
//! k bits, so 2^k entries; `a_0` is the node itself). Component `a_S` is the mixed
//! derivative of the node along the directions of the slots in `S`, each slot taken once.
//! The rules are the multilinear ones:
//!
//! * `Var(i)` and `Param(j)` have a singleton component `dir(..)` for every slot that
//!   covers them. Time, constants and everything else have none.
//! * Add, Sub, Neg act componentwise. Mul is `sum_{T <= S} a_T b_{S-T}`. Div solves
//!   `q b = a` for `q_S`. Select, Min and Max select componentwise.
//! * A call `g(a)` or `g(a, b)` follows Faa di Bruno: the sum over set partitions of `S`
//!   with each block given to one argument, of the partial `d^i_a d^j_b g` times the
//!   product of the block components.
//!
//! Route for the partials: `g` is built once on two placeholder `Var` nodes (indices
//! `u32::MAX` and `u32::MAX - 1`, which no model uses), differentiated with [`d`] as often
//! as needed, and the result is rebuilt on the real argument nodes by substitution. An
//! argument with no components stays as itself in `g`, so a constant `m` in
//! `polygamma(m, x)` is still seen by the derivative rule. Components that fold to the
//! constant 0 never reach a product, so zero directions cost nothing.
//!
//! A component that is zero at run time must contribute +0 even where its partial is
//! infinite or NaN at a finite input, as the interpreter's `Dual::term` does. So where
//! [`partial_may_be_nonfinite`] says the partial can be non-finite, the product of the
//! partial and the block components is a `ZMul` that tests the block product first. The
//! `ZMul` node gets the product rule with `ZMul` terms, and the Div recurrence uses `ZMul` for
//! each term of `q b = a`. A term `a_T b_{S-T}` of a `Mul` is a `ZMul` too when the other
//! factor can be non-finite (see [`Dag::may_be_nonfinite`]). The interpreter nests one `Dual` per
//! direction with slot 0 outermost, so the tangent is the factor that holds the lowest slot of
//! `S`, and it comes first. The final quotient `num / b_0` is a `ZDiv`: a numerator that is zero
//! at run time over a zero `b_0` gives +0, as the interpreter does by skipping a zero tangent.
//!
//! `pow(x, y)`: the interpreter skips every term that carries a `y` derivative when
//! `pow(x, y)` is 0 (it would meet `0 * ln 0`), so those terms are wrapped in a select on
//! `pow(x, y) == 0`.
//!
//! A call `g(args)` of a user function goes through a jet unit of `g` (SPEC section 8af-5):
//! a unit that takes the arguments and the nonzero components of its active formals and
//! returns the value (output 0) and every component `S` (output `S`). Its body is the body
//! of `g` built with these rules, each call inside it replaced by its own jet call, so one
//! frame runs each recursive call once. The value of the jet call replaces the primal call
//! everywhere in the form, so an entry makes one call per frame.

use crate::dag::{Cmp, Dag, Fn1, Fn2, NoDerivative, NodeId, Op};
use crate::diff::{
    Activity, Wrt, call_act, d, diff_children, out_refusal, partial_may_be_nonfinite, push_active,
};
use crate::entry::Slot;
use std::collections::{HashMap, HashSet};

const PA: u32 = u32::MAX;
const PB: u32 = u32::MAX - 1;

/// The derived-map tag of a jet unit (the tangent units of `diff.rs` use 0).
const JET_TAG: u32 = 1;

type Memo = HashMap<(NodeId, Wrt), NodeId>;

/// Output nodes of the order-`slots.len()` form: `n` entries, `c_full` of each `f[i]`.
/// Stops early when the DAG passes its cap (the caller reports `TooLarge`); then the
/// outputs are NaN constants. `slots` must hold 1 to 5 entries.
pub(crate) fn build(
    dag: &mut Dag,
    memo: &mut Memo,
    f: &[NodeId],
    n: usize,
    n_p: usize,
    slots: &[Slot],
) -> Vec<NodeId> {
    let mut b = Builder::new(dag, memo, n, n_p, slots);
    b.run(f)
}

struct Builder<'a> {
    dag: &'a mut Dag,
    memo: &'a mut Memo,
    n: usize,
    n_p: usize,
    slots: Vec<Slot>,
    off: Vec<usize>,
    /// `2^k`.
    size: usize,
    zero: NodeId,
    comps: HashMap<NodeId, Vec<NodeId>>,
    /// Inside a jet body, the components of each formal; `None` at the top level.
    args: Option<Vec<Vec<NodeId>>>,
    /// The rebuilt value of each node that reaches a call: a needed call is replaced by its
    /// jet call, whose output 0 is the value, so the value and the components share a call.
    prim: HashMap<NodeId, NodeId>,
}

/// State for one call node: argument components, the placeholder expression and caches.
struct Faa {
    args: Vec<NodeId>,
    comps: Vec<Vec<NodeId>>,
    active: Vec<bool>,
    g: NodeId,
    raw: HashMap<(usize, usize), NodeId>,
    done: HashMap<(usize, usize), NodeId>,
    sub: HashMap<NodeId, NodeId>,
    sums: [NodeId; 2],
    /// The partials of this call can be infinite or NaN (at a finite input, or because an
    /// argument is flagged by [`Dag::may_be_nonfinite`]), so a product that is zero at run
    /// time must give +0 (see [`Op::ZMul`]).
    guard: bool,
}

impl<'a> Builder<'a> {
    fn new(
        dag: &'a mut Dag,
        memo: &'a mut Memo,
        n: usize,
        n_p: usize,
        slots: &[Slot],
    ) -> Builder<'a> {
        let mut off = Vec::with_capacity(slots.len());
        let mut at = 0;
        for s in slots {
            off.push(at);
            at += s.len(n, n_p);
        }
        let zero = dag.konst(0.0);
        Builder {
            dag,
            memo,
            n,
            n_p,
            slots: slots.to_vec(),
            off,
            size: 1 << slots.len(),
            zero,
            comps: HashMap::new(),
            args: None,
            prim: HashMap::new(),
        }
    }

    /// The rebuilt value of `id` (itself when it reaches no call).
    fn p(&self, id: NodeId) -> NodeId {
        self.prim.get(&id).copied().unwrap_or(id)
    }

    fn is_zero(&self, id: NodeId) -> bool {
        matches!(self.dag.op(id), Op::Const(b) if f64::from_bits(*b) == 0.0)
    }

    fn inert(&self, c: &[NodeId]) -> bool {
        c.iter().skip(1).all(|&x| self.is_zero(x))
    }

    /// Components of the nodes the rules read from `roots`, and the rebuilt values of the
    /// nodes that reach a call, in id order (an operand has a smaller id than its user).
    fn walk(&mut self, roots: &[NodeId]) {
        // The nodes whose components the rules read. `diff_children` does not visit the
        // arguments of a call, so the active ones are pushed here.
        let mut needed = HashSet::new();
        let mut stack: Vec<NodeId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !needed.insert(id) {
                continue;
            }
            let op = *self.dag.op(id);
            match op {
                Op::Call(g, l) => {
                    if let Some(a) = call_act(self.dag, g) {
                        push_active(self.dag, &a, l, &mut stack);
                    }
                }
                Op::Out(..) | Op::Arg(_) => {}
                _ => diff_children(&op, &mut |c| stack.push(c)),
            }
        }
        // The nodes whose value changes when the needed calls become jet calls.
        let mut order: Vec<NodeId> = needed.iter().copied().collect();
        let mut seen = HashSet::new();
        let mut stack: Vec<NodeId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !self.dag.has_call(id) || !seen.insert(id) {
                continue;
            }
            if !needed.contains(&id) {
                order.push(id);
            }
            let op = *self.dag.op(id);
            self.dag.operands(&op, &mut |c| stack.push(c));
        }
        order.sort_unstable();
        for id in order {
            let need = needed.contains(&id);
            let op = *self.dag.op(id);
            if let Op::Call(g, l) = op {
                self.call(id, g, l, need);
            } else {
                if self.dag.has_call(id) {
                    let prim = &self.prim;
                    let p = self
                        .dag
                        .rebuild(&op, &|c| prim.get(&c).copied().unwrap_or(c));
                    self.prim.insert(id, p);
                }
                if need {
                    let c = self.node(id);
                    self.comps.insert(id, c);
                }
            }
            if self.dag.exceeded() {
                break;
            }
        }
    }

    /// The call `id = g(args)` with argument list `l`. A needed call whose active formals
    /// have nonzero components, or whose unit reads a direction, becomes a jet call; any
    /// other call stays a primal call on the rebuilt arguments.
    fn call(&mut self, id: NodeId, g: u32, l: u32, need: bool) {
        let args: Vec<NodeId> = self.dag.args(l).iter().map(|&a| self.p(a)).collect();
        let act = if need { call_act(self.dag, g) } else { None };
        let Some(a) = act else {
            let p = self.dag.call(g, &args);
            self.prim.insert(id, p);
            if need {
                // `call_act` recorded the refusal, so `entry` returns an error.
                let nan = self.dag.konst(f64::NAN);
                let mut c = vec![nan; self.size];
                c[0] = p;
                self.comps.insert(id, c);
            }
            return;
        };
        let l_args: Vec<NodeId> = self.dag.args(l).to_vec();
        let mut masks = Vec::with_capacity(args.len());
        let mut extra = Vec::new();
        for (i, &x) in l_args.iter().enumerate() {
            let mut m = 0u32;
            if a.active(i) {
                let cx = self.get(x);
                for (s, &cs) in cx.iter().enumerate().skip(1) {
                    if !self.is_zero(cs) {
                        m |= 1 << (s - 1);
                        extra.push(cs);
                    }
                }
            }
            masks.push(m);
        }
        let cover = self.slot_cover(&a);
        if masks.iter().all(|&m| m == 0) && cover == 0 {
            let p = self.dag.call(g, &args);
            self.prim.insert(id, p);
            self.comps.insert(id, self.leaf(p));
            return;
        }
        let closure = self.closure(&masks, cover);
        let t = self.jet_unit(g, &masks, cover, &closure);
        let mut v = args;
        v.extend(extra);
        let jc = self.dag.call(t, &v);
        self.prim.insert(id, jc);
        let mut c = self.leaf(jc);
        for (s, x) in c.iter_mut().enumerate().skip(1) {
            if closure[s] {
                *x = self.dag.out(jc, s as u32);
            }
        }
        self.comps.insert(id, c);
    }

    /// Bit `s` is set when slot `s` covers a state or a parameter that a unit with
    /// activity `a` reads on its derivative edges.
    fn slot_cover(&self, a: &Activity) -> u32 {
        let mut cover = 0;
        for (s, slot) in self.slots.iter().enumerate() {
            let hit = a.reads.iter().any(|w| match (*w, slot) {
                (Wrt::U(v), Slot::U | Slot::Full) => (v as usize) < self.n,
                (Wrt::P(p), Slot::P | Slot::Full) => (p as usize) < self.n_p,
                _ => false,
            });
            if hit {
                cover |= 1 << s;
            }
        }
        cover
    }

    /// `closure[S]` when the output component `S` of a jet call can be nonzero: `S` is a
    /// disjoint union of atoms, the subsets in some formal mask (bit `S - 1` of `masks[i]`)
    /// and the single slots in `cover`. Every rule builds a nonzero component only from
    /// nonzero components on disjoint subsets, so any other output is 0.
    fn closure(&self, masks: &[u32], cover: u32) -> Vec<bool> {
        let mut atom = vec![false; self.size];
        for &m in masks {
            for (s, x) in atom.iter_mut().enumerate().skip(1) {
                *x |= m & (1 << (s - 1)) != 0;
            }
        }
        for s in 0..self.slots.len() {
            if cover & (1 << s) != 0 {
                atom[1 << s] = true;
            }
        }
        let mut reach = vec![false; self.size];
        reach[0] = true;
        for s in 1..self.size {
            let low = s.isolate_lowest_one();
            // Subsets of s that hold its lowest bit.
            let rest = s ^ low;
            let mut t = rest;
            loop {
                let block = low | t;
                if atom[block] && reach[s ^ block] {
                    reach[s] = true;
                    break;
                }
                if t == 0 {
                    break;
                }
                t = (t - 1) & rest;
            }
        }
        reach
    }

    /// The jet unit of `g` for the formal masks `masks` and the covering slots `cover`.
    /// The key holds `g`, the order, whether a slot covers a read (and then the kind of
    /// every slot, since the body reads the directions) and the masks. The unit is
    /// registered before its body is built, so a recursive call inside the body finds it.
    fn jet_unit(&mut self, g: u32, masks: &[u32], cover: u32, closure: &[bool]) -> u32 {
        let k = self.slots.len() as u32;
        let mut key = vec![JET_TAG, g, k, u32::from(cover != 0)];
        if cover != 0 {
            key.extend(self.slots.iter().map(|s| match s {
                Slot::U => 1,
                Slot::P => 2,
                Slot::Full => 3,
            }));
        }
        key.extend_from_slice(masks);
        if let Some(t) = self.dag.derived(&key) {
            return t;
        }
        let unit = self.dag.unit(g).expect("call_act checked the unit");
        let (name, arity) = (unit.name.clone(), unit.arity);
        let out0 = unit.outputs().expect("call_act checked the body")[0];
        let extra: u32 = masks.iter().map(|m| m.count_ones()).sum();
        let t = self.dag.add_unit(&name, arity + extra, self.size as u32);
        self.dag.set_derived(key, t);

        // Formal i: value `Arg(i)`, then its nonzero components in ascending `S`.
        let mut fcomps = Vec::with_capacity(arity as usize);
        let mut next = arity;
        for (i, &m) in masks.iter().enumerate() {
            let mut c = vec![self.zero; self.size];
            c[0] = self.dag.arg(i as u32);
            for (s, x) in c.iter_mut().enumerate().skip(1) {
                if m & (1 << (s - 1)) != 0 {
                    *x = self.dag.arg(next);
                    next += 1;
                }
            }
            fcomps.push(c);
        }
        let mut body = Builder {
            dag: &mut *self.dag,
            memo: &mut *self.memo,
            n: self.n,
            n_p: self.n_p,
            slots: self.slots.clone(),
            off: self.off.clone(),
            size: self.size,
            zero: self.zero,
            comps: HashMap::new(),
            args: Some(fcomps),
            prim: HashMap::new(),
        };
        body.walk(&[out0]);
        if body.dag.exceeded() {
            return t;
        }
        let c = body.get(out0);
        let outs: Vec<NodeId> = c
            .iter()
            .enumerate()
            .map(|(s, &x)| if s == 0 || closure[s] { x } else { body.zero })
            .collect();
        let ok = body.dag.set_body(t, &outs);
        debug_assert!(ok, "a jet body has 2^k outputs and in-range formals");
        t
    }

    fn run(&mut self, f: &[NodeId]) -> Vec<NodeId> {
        self.walk(f);
        let top = self.size - 1;
        let mut out = Vec::with_capacity(f.len());
        for &fi in f {
            let r = self.comps.get(&fi).and_then(|c| c.get(top)).copied();
            out.push(match r {
                Some(r) if !self.dag.exceeded() => r,
                // Past the node limit `entry` reports `TooLarge`; otherwise every root has
                // components, so a missing one is a broken invariant and is refused.
                _ => {
                    if !self.dag.exceeded() {
                        self.dag.note_refusal(NoDerivative::NoRule);
                    }
                    self.dag.konst(f64::NAN)
                }
            });
        }
        out
    }

    /// Components of `id`, or an inert vector when the node was not processed.
    fn get(&self, id: NodeId) -> Vec<NodeId> {
        match self.comps.get(&id) {
            Some(c) => c.clone(),
            None => self.leaf(self.p(id)),
        }
    }

    /// `[id, 0, 0, ...]`.
    fn leaf(&self, id: NodeId) -> Vec<NodeId> {
        let mut c = vec![self.zero; self.size];
        c[0] = id;
        c
    }

    fn node(&mut self, id: NodeId) -> Vec<NodeId> {
        let op = *self.dag.op(id);
        let mut c = self.leaf(self.p(id));
        match op {
            Op::Var(i) => {
                for s in 0..self.slots.len() {
                    if matches!(self.slots[s], Slot::U | Slot::Full) && (i as usize) < self.n {
                        c[1 << s] = self.dag.dir((self.off[s] + i as usize) as u32);
                    }
                }
            }
            Op::Param(j) => {
                for s in 0..self.slots.len() {
                    let j = j as usize;
                    if j >= self.n_p {
                        break;
                    }
                    let at = match self.slots[s] {
                        Slot::U => None,
                        Slot::P => Some(self.off[s] + j),
                        Slot::Full => Some(self.off[s] + self.n + j),
                    };
                    if let Some(at) = at {
                        c[1 << s] = self.dag.dir(at as u32);
                    }
                }
            }
            Op::Neg(a) => {
                let a = self.get(a);
                for s in 1..self.size {
                    c[s] = self.dag.neg(a[s]);
                }
            }
            Op::Add(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    c[s] = self.dag.add(a[s], b[s]);
                }
            }
            Op::Sub(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    c[s] = self.dag.sub(a[s], b[s]);
                }
            }
            Op::Mul(na, nb) => {
                let (a, b) = (self.get(na), self.get(nb));
                let (fa, fb) = (
                    self.dag.may_be_nonfinite(self.p(na)),
                    self.dag.may_be_nonfinite(self.p(nb)),
                );
                for s in 1..self.size {
                    let mut acc = self.zero;
                    let mut t = s;
                    // The interpreter nests one `Dual` per direction with slot 0 outermost,
                    // and a `Dual` product tests the tangent of its own level for zero. For
                    // the slots in `s` that level is the lowest set bit. The factor of a term
                    // that holds that bit is therefore the guarded first operand of the `ZMul`.
                    let low = s.isolate_lowest_one();
                    loop {
                        let (at, bu) = (a[t], b[s ^ t]);
                        let m = if t & low != 0 {
                            self.product_term(at, bu, fb)
                        } else {
                            self.product_term(bu, at, fa)
                        };
                        acc = self.dag.add(acc, m);
                        if t == 0 {
                            break;
                        }
                        t = (t - 1) & s;
                    }
                    c[s] = acc;
                }
            }
            // The same sum as `Mul`, each term guarded by its first factor.
            Op::ZMul(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    let mut acc = self.zero;
                    let mut t = s;
                    loop {
                        let m = self.dag.zmul(a[t], b[s ^ t]);
                        acc = self.dag.add(acc, m);
                        if t == 0 {
                            break;
                        }
                        t = (t - 1) & s;
                    }
                    c[s] = acc;
                }
            }
            // `ZDiv` is the same recurrence: its tangent components are those of `Div`.
            Op::Div(a, b) | Op::ZDiv(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    let mut sum = self.zero;
                    // Proper subsets of s, empty one first.
                    let mut t = (s - 1) & s;
                    loop {
                        // A zero tangent component adds +0 even where the quotient is infinite.
                        let m = self.dag.zmul(b[s ^ t], c[t]);
                        sum = self.dag.add(sum, m);
                        if t == 0 {
                            break;
                        }
                        t = (t - 1) & s;
                    }
                    let num = self.dag.sub(a[s], sum);
                    // A zero numerator over a zero `b[0]` is +0 as in the interpreter, which
                    // skips a zero tangent. `ZDiv` replaces the `Div`, so it costs no node.
                    c[s] = if self.is_zero(num) {
                        self.zero
                    } else {
                        self.dag.zdiv(num, b[0])
                    };
                }
            }
            Op::Select(cond, a, b) | Op::If(cond, a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                let cond = self.p(cond);
                for s in 1..self.size {
                    c[s] = self.dag.select(cond, a[s], b[s]);
                }
            }
            Op::Call2(f @ (Fn2::Min | Fn2::Max), a, b) => {
                let (ca, cb) = (self.get(a), self.get(b));
                let rel = if f == Fn2::Min { Cmp::Lt } else { Cmp::Gt };
                let cond = self.dag.cmp(rel, self.p(a), self.p(b));
                for s in 1..self.size {
                    c[s] = self.dag.select(cond, ca[s], cb[s]);
                }
            }
            Op::Call1(Fn1::Floor | Fn1::Ceil | Fn1::Heav | Fn1::Sign, _) => {}
            Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) | Op::TableSlope(..) => {}
            Op::Call1(_, a) | Op::PowI(a, _) | Op::Table(_, a) => {
                self.faa(id, op, &[a], &mut c);
            }
            Op::Call2(_, a, b) => self.faa(id, op, &[a, b], &mut c),
            Op::Const(_) | Op::Dir(_) => {}
            // A formal inside a jet body takes the components the call passes in.
            Op::Arg(i) if self.args.is_some() => {
                let a = self.args.as_ref().expect("checked");
                return a.get(i as usize).cloned().unwrap_or_else(|| self.leaf(id));
            }
            // `walk` handles every call. A formal outside a body has no rule, and an output
            // `j > 0` belongs to a unit with several outputs; each is refused, so `entry`
            // returns an error instead of this NaN.
            Op::Arg(_) | Op::Call(..) | Op::Out(..) => {
                let why = match op {
                    Op::Out(call, _) => out_refusal(self.dag, call),
                    _ => NoDerivative::NoRule,
                };
                self.dag.note_refusal(why);
                let nan = self.dag.konst(f64::NAN);
                for x in c.iter_mut().skip(1) {
                    *x = nan;
                }
            }
        }
        c
    }

    /// `tangent * other` for one term of a product, with `other_flag` the
    /// [`Dag::may_be_nonfinite`] of the node that `other` is a component of. A component of a
    /// node that is not flagged is a polynomial of finite values, so only a flagged node can
    /// have an infinite or NaN component. A zero `tangent` then gives +0, as in the
    /// interpreter, and a plain `mul` is used otherwise.
    fn product_term(&mut self, tangent: NodeId, other: NodeId, other_flag: bool) -> NodeId {
        if other_flag {
            self.dag.zmul(tangent, other)
        } else {
            self.dag.mul(tangent, other)
        }
    }

    /// Faa di Bruno for the call `op` (node `id`) with argument nodes `args`.
    fn faa(&mut self, id: NodeId, op: Op, args: &[NodeId], c: &mut [NodeId]) {
        let comps: Vec<Vec<NodeId>> = args.iter().map(|&a| self.get(a)).collect();
        let active: Vec<bool> = comps.iter().map(|x| !self.inert(x)).collect();
        if !active.iter().any(|&x| x) {
            return;
        }
        let place = |dag: &mut Dag, j: usize| dag.var(if j == 0 { PA } else { PB });
        let ga: Vec<NodeId> = (0..args.len())
            .map(|j| {
                if active[j] {
                    place(self.dag, j)
                } else {
                    self.p(args[j])
                }
            })
            .collect();
        let g = match op {
            Op::Call1(f, _) => self.dag.call1(f, ga[0]),
            Op::PowI(_, k) => self.dag.powi(ga[0], k),
            Op::Table(t, _) => self.dag.table(t, ga[0]),
            Op::Call2(f, ..) => self.dag.call2(f, ga[0], ga[1]),
            _ => return,
        };
        let mut sub = HashMap::new();
        for (j, &a) in args.iter().enumerate() {
            if active[j] {
                let pa = self.p(a);
                sub.insert(place(self.dag, j), pa);
            }
        }
        // The partials are finite for a finite argument unless `partial_may_be_nonfinite` says
        // otherwise, but an argument that can be infinite or NaN gives such a partial for any
        // call (`cosh(1/m)` has the partial `sinh(1/m)`).
        let guard = partial_may_be_nonfinite(self.dag, &op)
            || args.iter().any(|&a| self.dag.may_be_nonfinite(self.p(a)));
        let mut ctx = Faa {
            args: args.to_vec(),
            comps,
            active,
            g,
            raw: HashMap::new(),
            done: HashMap::new(),
            sub,
            sums: [self.zero; 2],
            guard,
        };
        // In a jet body an inactive argument can read a formal, as the exponent `e` of
        // `x^e` does. A formal never depends on the placeholders, so its partial with
        // respect to one is 0; without this seed `d` refuses `Arg`, which has no rule outside
        // a body. Only jet bodies have formals, so a model without recursion is unchanged.
        if let Some(n) = self.args.as_ref().map(Vec::len) {
            let z = self.zero;
            for k in 0..n {
                let a = self.dag.arg(k as u32);
                for w in [PA, PB] {
                    self.memo.entry((a, Wrt::U(w))).or_insert(z);
                }
            }
        }
        let is_pow = matches!(op, Op::Call2(Fn2::Pow, ..));
        for (s, slot) in c.iter_mut().enumerate().skip(1) {
            ctx.sums = [self.zero; 2];
            let one = self.dag.konst(1.0);
            self.dfs(&mut ctx, s as u32, one, 0, 0);
            let [s0, s1] = ctx.sums;
            let s1 = if is_pow && !self.is_zero(s1) {
                let z = self.zero;
                let pid = self.p(id);
                let cond = self.dag.cmp(Cmp::Eq, pid, z);
                self.dag.select(cond, z, s1)
            } else {
                s1
            };
            *slot = self.dag.add(s0, s1);
            if self.dag.exceeded() {
                return;
            }
        }
    }

    /// Distributes the elements of `rest` over blocks, each block given to one argument.
    fn dfs(&mut self, ctx: &mut Faa, rest: u32, prod: NodeId, i: usize, j: usize) {
        if rest == 0 {
            let p = self.partial(ctx, i, j);
            if self.is_zero(p) {
                return;
            }
            // A zero tangent product contributes +0 even where the partial is infinite.
            let t = if ctx.guard {
                self.dag.zmul(prod, p)
            } else {
                self.dag.mul(p, prod)
            };
            let at = usize::from(j > 0);
            ctx.sums[at] = self.dag.add(ctx.sums[at], t);
            return;
        }
        if self.dag.exceeded() {
            return;
        }
        let low = rest.isolate_lowest_one();
        let others = rest ^ low;
        let mut t = others;
        loop {
            let block = (low | t) as usize;
            for arg in 0..ctx.args.len() {
                if !ctx.active[arg] {
                    continue;
                }
                let comp = ctx.comps[arg][block];
                if self.is_zero(comp) {
                    continue;
                }
                let np = self.dag.mul(prod, comp);
                let (ni, nj) = if arg == 0 { (i + 1, j) } else { (i, j + 1) };
                self.dfs(ctx, rest ^ block as u32, np, ni, nj);
            }
            if t == 0 {
                break;
            }
            t = (t - 1) & others;
        }
    }

    /// `d^i_a d^j_b g` evaluated at the real arguments.
    fn partial(&mut self, ctx: &mut Faa, i: usize, j: usize) -> NodeId {
        if let Some(&p) = ctx.done.get(&(i, j)) {
            return p;
        }
        let r = self.raw(ctx, i, j);
        let p = subst(self.dag, r, &mut ctx.sub);
        ctx.done.insert((i, j), p);
        p
    }

    fn raw(&mut self, ctx: &mut Faa, i: usize, j: usize) -> NodeId {
        if let Some(&r) = ctx.raw.get(&(i, j)) {
            return r;
        }
        let r = if i == 0 && j == 0 {
            ctx.g
        } else if i > 0 {
            let prev = self.raw(ctx, i - 1, j);
            d(self.dag, self.memo, prev, Wrt::U(PA))
        } else {
            let prev = self.raw(ctx, i, j - 1);
            d(self.dag, self.memo, prev, Wrt::U(PB))
        };
        ctx.raw.insert((i, j), r);
        r
    }
}

/// Rebuilds `node` with the nodes in `sub` replaced, through the folding constructors.
/// The expressions are partials of one call, so their depth is small.
fn subst(dag: &mut Dag, node: NodeId, sub: &mut HashMap<NodeId, NodeId>) -> NodeId {
    if let Some(&r) = sub.get(&node) {
        return r;
    }
    let op = *dag.op(node);
    let r = match op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => node,
        Op::Neg(a) => {
            let a = subst(dag, a, sub);
            dag.neg(a)
        }
        Op::Add(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.add(a, b)
        }
        Op::Sub(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.sub(a, b)
        }
        Op::Mul(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.mul(a, b)
        }
        Op::ZMul(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.zmul(a, b)
        }
        Op::Div(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.div(a, b)
        }
        Op::ZDiv(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.zdiv(a, b)
        }
        Op::PowI(a, k) => {
            let a = subst(dag, a, sub);
            dag.powi(a, k)
        }
        Op::Call1(f, a) => {
            let a = subst(dag, a, sub);
            dag.call1(f, a)
        }
        Op::Call2(f, a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.call2(f, a, b)
        }
        Op::Cmp(k, a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.cmp(k, a, b)
        }
        Op::And(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.and(a, b)
        }
        Op::Or(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.or(a, b)
        }
        Op::Not(a) => {
            let a = subst(dag, a, sub);
            dag.not(a)
        }
        Op::Select(c, a, b) => {
            let (c, a, b) = (subst(dag, c, sub), subst(dag, a, sub), subst(dag, b, sub));
            dag.select(c, a, b)
        }
        Op::Table(t, a) => {
            let a = subst(dag, a, sub);
            dag.table(t, a)
        }
        Op::TableSlope(t, a) => {
            let a = subst(dag, a, sub);
            dag.table_slope(t, a)
        }
        Op::If(c, a, b) => {
            let (c, a, b) = (subst(dag, c, sub), subst(dag, a, sub), subst(dag, b, sub));
            dag.select(c, a, b)
        }
        Op::Arg(_) => node,
        Op::Call(u, l) => {
            let args: Vec<NodeId> = dag.args(l).to_vec();
            let args: Vec<NodeId> = args.into_iter().map(|a| subst(dag, a, sub)).collect();
            dag.call(u, &args)
        }
        Op::Out(c, j) => {
            let c = subst(dag, c, sub);
            dag.out(c, j)
        }
    };
    sub.insert(node, r);
    r
}

#[cfg(test)]
#[path = "multi_tests.rs"]
mod tests;
