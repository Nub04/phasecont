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
//! `pow(x, y)`: the interpreter skips every term that carries a `y` derivative when
//! `pow(x, y)` is 0 (it would meet `0 * ln 0`), so those terms are wrapped in a select on
//! `pow(x, y) == 0`.

use crate::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use crate::diff::{Wrt, d, diff_children};
use crate::entry::Slot;
use std::collections::HashMap;

const PA: u32 = u32::MAX;
const PB: u32 = u32::MAX - 1;

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
        }
    }

    fn is_zero(&self, id: NodeId) -> bool {
        matches!(self.dag.op(id), Op::Const(b) if f64::from_bits(*b) == 0.0)
    }

    fn inert(&self, c: &[NodeId]) -> bool {
        c.iter().skip(1).all(|&x| self.is_zero(x))
    }

    fn run(&mut self, f: &[NodeId]) -> Vec<NodeId> {
        // Reachable nodes through the children the rules read, in id order.
        let mut seen = std::collections::HashSet::new();
        let mut stack: Vec<NodeId> = f.to_vec();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let op = *self.dag.op(id);
            diff_children(&op, &mut |c| stack.push(c));
        }
        let mut order: Vec<NodeId> = seen.into_iter().collect();
        order.sort_unstable();
        for id in order {
            let c = self.node(id);
            self.comps.insert(id, c);
            if self.dag.exceeded() {
                break;
            }
        }
        let top = self.size - 1;
        let mut out = Vec::with_capacity(f.len());
        for &fi in f {
            let r = self.comps.get(&fi).and_then(|c| c.get(top)).copied();
            out.push(match r {
                Some(r) if !self.dag.exceeded() => r,
                _ => self.dag.konst(f64::NAN),
            });
        }
        out
    }

    /// Components of `id`, or an inert vector when the node was not processed.
    fn get(&self, id: NodeId) -> Vec<NodeId> {
        match self.comps.get(&id) {
            Some(c) => c.clone(),
            None => self.leaf(id),
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
        let mut c = self.leaf(id);
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
            Op::Mul(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    let mut acc = self.zero;
                    let mut t = s;
                    loop {
                        let m = self.dag.mul(a[t], b[s ^ t]);
                        acc = self.dag.add(acc, m);
                        if t == 0 {
                            break;
                        }
                        t = (t - 1) & s;
                    }
                    c[s] = acc;
                }
            }
            Op::Div(a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    let mut sum = self.zero;
                    // Proper subsets of s, empty one first.
                    let mut t = (s - 1) & s;
                    loop {
                        let m = self.dag.mul(c[t], b[s ^ t]);
                        sum = self.dag.add(sum, m);
                        if t == 0 {
                            break;
                        }
                        t = (t - 1) & s;
                    }
                    let num = self.dag.sub(a[s], sum);
                    c[s] = if self.is_zero(num) {
                        self.zero
                    } else {
                        self.dag.div(num, b[0])
                    };
                }
            }
            Op::Select(cond, a, b) => {
                let (a, b) = (self.get(a), self.get(b));
                for s in 1..self.size {
                    c[s] = self.dag.select(cond, a[s], b[s]);
                }
            }
            Op::Call2(f @ (Fn2::Min | Fn2::Max), a, b) => {
                let (ca, cb) = (self.get(a), self.get(b));
                let rel = if f == Fn2::Min { Cmp::Lt } else { Cmp::Gt };
                let cond = self.dag.cmp(rel, a, b);
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
        }
        c
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
                    args[j]
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
                sub.insert(place(self.dag, j), a);
            }
        }
        let mut ctx = Faa {
            args: args.to_vec(),
            comps,
            active,
            g,
            raw: HashMap::new(),
            done: HashMap::new(),
            sub,
            sums: [self.zero; 2],
        };
        let is_pow = matches!(op, Op::Call2(Fn2::Pow, ..));
        for (s, slot) in c.iter_mut().enumerate().skip(1) {
            ctx.sums = [self.zero; 2];
            let one = self.dag.konst(1.0);
            self.dfs(&mut ctx, s as u32, one, 0, 0);
            let [s0, s1] = ctx.sums;
            let s1 = if is_pow && !self.is_zero(s1) {
                let z = self.zero;
                let cond = self.dag.cmp(Cmp::Eq, id, z);
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
            let t = self.dag.mul(p, prod);
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
        Op::Div(a, b) => {
            let (a, b) = (subst(dag, a, sub), subst(dag, b, sub));
            dag.div(a, b)
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
    };
    sub.insert(node, r);
    r
}

#[cfg(test)]
#[path = "multi_tests.rs"]
mod tests;
