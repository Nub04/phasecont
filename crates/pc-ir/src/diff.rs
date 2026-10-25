//! First-order symbolic differentiation on the hash-consed DAG.
//!
//! The rules copy the `Dual` rules of `interp.rs` formula for formula and branch for
//! branch. Every derivative is built through the folding constructors of [`Dag`], so a
//! zero tangent folds away and a term with a zero tangent is never built (the same as the
//! interpreter, which skips a zero tangent so it never meets an infinite partial). One
//! difference remains: the interpreter tests the tangent at run time and the DAG tests it
//! at build time, so a symbolic tangent that happens to be 0.0 at a point still multiplies
//! an infinite or NaN partial there.
//!
//! `polygamma(m, x)` differentiates to `polygamma(m + 1, x)` for `m` in 0..=3. The math
//! layer supports `m` up to 4 only, so a derivative that would need `m = 5` (the first
//! derivative of `polygamma(4, x)`, the sixth derivative of `lgamma`) is a NaN constant.
//! A non-constant `m` gives a NaN constant too.

use crate::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use crate::eval::children;
use std::collections::{HashMap, HashSet};

/// The variable a derivative is taken with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Wrt {
    /// State `u[i]`.
    U(u32),
    /// Parameter `p[j]`; `j` is below `n_p`, so the time slot is never a variable.
    P(u32),
}

fn is_zero(dag: &Dag, id: NodeId) -> bool {
    matches!(dag.op(id), Op::Const(b) if f64::from_bits(*b) == 0.0)
}

fn cval(dag: &Dag, id: NodeId) -> Option<f64> {
    match dag.op(id) {
        Op::Const(b) => Some(f64::from_bits(*b)),
        _ => None,
    }
}

/// `tangent * partial()`, or the zero constant without building `partial` when the tangent
/// is the zero constant.
fn term(dag: &mut Dag, tangent: NodeId, partial: impl FnOnce(&mut Dag) -> NodeId) -> NodeId {
    if is_zero(dag, tangent) {
        return dag.konst(0.0);
    }
    let p = partial(dag);
    dag.mul(tangent, p)
}

/// The children of `op` whose derivatives the rule for `op` reads.
fn diff_children(op: &Op, f: &mut dyn FnMut(NodeId)) {
    match *op {
        Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) | Op::TableSlope(..) => {}
        Op::Call1(Fn1::Floor | Fn1::Ceil | Fn1::Heav | Fn1::Sign, _) => {}
        Op::Select(_, a, b) => {
            f(a);
            f(b);
        }
        Op::Call2(Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma, _, x) => f(x),
        _ => children(op, f),
    }
}

/// `d node / d wrt`. `memo` caches results per `(node, wrt)` and may be shared across
/// calls on the same DAG. Walks the needed nodes in id order, so deep DAGs do not recurse.
pub fn d(
    dag: &mut Dag,
    memo: &mut HashMap<(NodeId, Wrt), NodeId>,
    node: NodeId,
    wrt: Wrt,
) -> NodeId {
    if let Some(&r) = memo.get(&(node, wrt)) {
        return r;
    }
    let mut seen = HashSet::new();
    let mut stack = vec![node];
    while let Some(id) = stack.pop() {
        if memo.contains_key(&(id, wrt)) || !seen.insert(id) {
            continue;
        }
        let op = *dag.op(id);
        diff_children(&op, &mut |c| stack.push(c));
    }
    let mut order: Vec<NodeId> = seen.into_iter().collect();
    order.sort_unstable();
    for id in order {
        let r = rule(dag, memo, id, wrt);
        memo.insert((id, wrt), r);
    }
    match memo.get(&(node, wrt)) {
        Some(&r) => r,
        None => dag.konst(f64::NAN),
    }
}

fn rule(dag: &mut Dag, memo: &HashMap<(NodeId, Wrt), NodeId>, node: NodeId, wrt: Wrt) -> NodeId {
    let mut nan = None;
    let mut dc = |dag: &mut Dag, c: NodeId| match memo.get(&(c, wrt)) {
        Some(&r) => r,
        None => *nan.get_or_insert_with(|| dag.konst(f64::NAN)),
    };
    let op = *dag.op(node);
    match op {
        Op::Const(_) | Op::Dir(_) => dag.konst(0.0),
        Op::Var(i) => dag.konst(if wrt == Wrt::U(i) { 1.0 } else { 0.0 }),
        Op::Param(j) => dag.konst(if wrt == Wrt::P(j) { 1.0 } else { 0.0 }),
        Op::Neg(a) => {
            let da = dc(dag, a);
            dag.neg(da)
        }
        Op::Add(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            dag.add(da, db)
        }
        Op::Sub(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            dag.sub(da, db)
        }
        Op::Mul(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            let t1 = term(dag, da, |_| b);
            let t2 = term(dag, db, |_| a);
            dag.add(t1, t2)
        }
        Op::Div(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            let ta = if is_zero(dag, da) {
                dag.konst(0.0)
            } else {
                dag.div(da, b)
            };
            // The quotient is `node` itself.
            let tb = term(dag, db, |g| {
                let q = g.div(a, b);
                let qb = g.div(q, b);
                g.neg(qb)
            });
            dag.add(ta, tb)
        }
        Op::PowI(a, n) => {
            let da = dc(dag, a);
            term(dag, da, |g| {
                let c = g.konst(f64::from(n));
                let p = g.powi(a, n - 1);
                g.mul(c, p)
            })
        }
        Op::Call1(f, a) => {
            let da = dc(dag, a);
            call1_rule(dag, f, node, a, da)
        }
        Op::Call2(f, a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            call2_rule(dag, f, node, (a, da), (b, db))
        }
        Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) | Op::TableSlope(..) => dag.konst(0.0),
        Op::Select(c, a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            dag.select(c, da, db)
        }
        Op::Table(id, a) => {
            let da = dc(dag, a);
            term(dag, da, |g| g.table_slope(id, a))
        }
    }
}

/// `2/sqrt(pi) * exp(-x^2)`, the derivative of `erf` (DLMF 7.10).
fn erf_slope(g: &mut Dag, x: NodeId) -> NodeId {
    let c = g.konst(std::f64::consts::FRAC_2_SQRT_PI);
    let xx = g.mul(x, x);
    let nx = g.neg(xx);
    let e = g.call1(Fn1::Exp, nx);
    g.mul(c, e)
}

/// `1 / sqrt(1 - x^2)`.
fn inv_sqrt_1m_sq(g: &mut Dag, x: NodeId) -> NodeId {
    let one = g.konst(1.0);
    let xx = g.mul(x, x);
    let s = g.sub(one, xx);
    let r = g.call1(Fn1::Sqrt, s);
    g.div(one, r)
}

fn call1_rule(dag: &mut Dag, f: Fn1, node: NodeId, a: NodeId, da: NodeId) -> NodeId {
    match f {
        Fn1::Sin => term(dag, da, |g| g.call1(Fn1::Cos, a)),
        Fn1::Cos => term(dag, da, |g| {
            let s = g.call1(Fn1::Sin, a);
            g.neg(s)
        }),
        Fn1::Tan => term(dag, da, |g| {
            let one = g.konst(1.0);
            let tt = g.mul(node, node);
            g.add(one, tt)
        }),
        Fn1::Asin => term(dag, da, |g| inv_sqrt_1m_sq(g, a)),
        Fn1::Acos => term(dag, da, |g| {
            let r = inv_sqrt_1m_sq(g, a);
            g.neg(r)
        }),
        Fn1::Atan => term(dag, da, |g| {
            let one = g.konst(1.0);
            let aa = g.mul(a, a);
            let s = g.add(one, aa);
            g.div(one, s)
        }),
        Fn1::Sinh => term(dag, da, |g| g.call1(Fn1::Cosh, a)),
        Fn1::Cosh => term(dag, da, |g| g.call1(Fn1::Sinh, a)),
        Fn1::Tanh => term(dag, da, |g| {
            let one = g.konst(1.0);
            let tt = g.mul(node, node);
            g.sub(one, tt)
        }),
        Fn1::Exp => term(dag, da, |_| node),
        Fn1::Ln => term(dag, da, |g| {
            let one = g.konst(1.0);
            g.div(one, a)
        }),
        Fn1::Log10 => term(dag, da, |g| {
            let one = g.konst(1.0);
            let c = g.konst(std::f64::consts::LN_10);
            let s = g.mul(a, c);
            g.div(one, s)
        }),
        Fn1::Sqrt => term(dag, da, |g| {
            let one = g.konst(1.0);
            let two = g.konst(2.0);
            let s = g.mul(two, node);
            g.div(one, s)
        }),
        Fn1::Abs => term(dag, da, |g| g.call1(Fn1::Sign, a)),
        Fn1::Erf => term(dag, da, |g| erf_slope(g, a)),
        Fn1::Erfc => term(dag, da, |g| {
            let s = erf_slope(g, a);
            g.neg(s)
        }),
        Fn1::Lgamma => term(dag, da, |g| {
            let m = g.konst(0.0);
            g.call2(Fn2::Polygamma, m, a)
        }),
        Fn1::Floor | Fn1::Ceil | Fn1::Heav | Fn1::Sign => dag.konst(0.0),
    }
}

fn call2_rule(
    dag: &mut Dag,
    f: Fn2,
    node: NodeId,
    (a, da): (NodeId, NodeId),
    (b, db): (NodeId, NodeId),
) -> NodeId {
    match f {
        // atan2(y, x) with y = a and x = b.
        Fn2::Atan2 => {
            let dy = term(dag, da, |g| {
                let d = sum_squares(g, a, b);
                g.div(b, d)
            });
            let dx = term(dag, db, |g| {
                let d = sum_squares(g, a, b);
                let q = g.div(a, d);
                g.neg(q)
            });
            dag.add(dy, dx)
        }
        Fn2::Pow => {
            let dx = term(dag, da, |g| {
                let one = g.konst(1.0);
                let ym1 = g.sub(b, one);
                let p = g.call2(Fn2::Pow, a, ym1);
                g.mul(b, p)
            });
            // A zero result contributes nothing, so `0 * ln(0)` is never formed.
            let dy = if is_zero(dag, db) {
                dag.konst(0.0)
            } else {
                let ln = dag.call1(Fn1::Ln, a);
                let r = dag.mul(node, ln);
                let t = dag.mul(db, r);
                let z = dag.konst(0.0);
                let c = dag.cmp(Cmp::Eq, node, z);
                dag.select(c, z, t)
            };
            dag.add(dx, dy)
        }
        Fn2::Mod => {
            let dy = term(dag, db, |g| {
                let num = g.sub(a, node);
                let q = g.div(num, b);
                g.neg(q)
            });
            dag.add(da, dy)
        }
        Fn2::BesselJ | Fn2::BesselY => term(dag, db, |g| {
            let one = g.konst(1.0);
            let nm = g.sub(a, one);
            let np = g.add(a, one);
            let jm = g.call2(f, nm, b);
            let jp = g.call2(f, np, b);
            let diff = g.sub(jm, jp);
            let half = g.konst(0.5);
            g.mul(diff, half)
        }),
        Fn2::Polygamma => term(dag, db, |g| {
            let next = match cval(g, a) {
                Some(m) if m == 0.0 || m == 1.0 || m == 2.0 || m == 3.0 => Some(m + 1.0),
                _ => None,
            };
            match next {
                Some(m) => {
                    let mc = g.konst(m);
                    g.call2(Fn2::Polygamma, mc, b)
                }
                None => g.konst(f64::NAN),
            }
        }),
        Fn2::Min => {
            let c = dag.cmp(Cmp::Lt, a, b);
            dag.select(c, da, db)
        }
        Fn2::Max => {
            let c = dag.cmp(Cmp::Gt, a, b);
            dag.select(c, da, db)
        }
    }
}

/// `y*y + x*x`, the denominator of the `atan2` partials.
fn sum_squares(g: &mut Dag, y: NodeId, x: NodeId) -> NodeId {
    let yy = g.mul(y, y);
    let xx = g.mul(x, x);
    g.add(yy, xx)
}

#[cfg(test)]
#[path = "diff_tests.rs"]
mod tests;
