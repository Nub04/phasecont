//! First-order symbolic differentiation on the hash-consed DAG.
//!
//! The rules copy the `Dual` rules of `interp.rs` formula for formula and branch for
//! branch. Every derivative is built through the folding constructors of [`Dag`], so a
//! zero tangent folds away and a term with a zero tangent is never built (the same as the
//! interpreter, which skips a zero tangent so it never meets an infinite partial).
//!
//! The interpreter tests the tangent at run time and the DAG tests it at build time. A
//! symbolic tangent can be 0.0 at a point (a clamp such as `max(a - x, 0)` is flat there),
//! so a rule whose partial can be infinite or NaN at a finite input builds `ZMul(tangent,
//! partial)` instead of `Mul` (see [`zterm`]). A zero tangent then contributes `+0`, as in
//! the interpreter. The first term of the quotient rule, `tangent / denominator`, is a
//! `ZDiv` (see [`quotient_term`]): a zero tangent over a zero denominator, as in a clamped
//! `max(m, 0)` inside `exp(-1 / m)`, is +0 and not `0 / 0`. The calls whose partials are
//! finite wherever their argument is finite are `sin`, `cos`, `tan`, `atan`, `sinh`, `cosh`,
//! `tanh`, `exp`, `abs` (through `sign`), `erf`, `erfc`, an integer power `n >= 1`, table
//! slopes and `pow` with a constant integer exponent `>= 1`. Their term is a plain `Mul`
//! unless [`Dag::may_be_nonfinite`] flags the argument (see [`call_term`]), because an
//! infinite argument gives an infinite or NaN partial: the derivative of
//! `cosh(1 / max(a - x - 10, 0))` along a zero tangent is +0 in the interpreter. The
//! product rule multiplies each tangent by the other operand of the model, which can be
//! infinite as well: `exp(-c * (1 / m))` with a clamped `c` and a zero `m` has a finite
//! model value, since `exp(-inf)` is 0, and a zero tangent of `c` times the infinite `1 / m`
//! is NaN. A product term is therefore a `ZMul(tangent, other)` when [`Dag::may_be_nonfinite`]
//! flags the other operand (see [`product_term`]), and a plain `Mul` otherwise, so a model
//! without a division, a singular function or a non-finite constant gets no new guard. A zero
//! tangent times an overflowed `exp` or `sinh` is not guarded. A guarded rule also maps a true
//! singularity, such as the derivative of `sqrt(u^2)` at `u = 0`, to 0 instead of NaN.
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

/// `ZMul(tangent, partial())`: the term of a rule whose partial can be infinite or NaN at a
/// finite input. The zero constant tangent never builds `partial`. A partial that is a
/// finite constant cannot poison the product, so it stays a plain `mul`.
fn zterm(dag: &mut Dag, tangent: NodeId, partial: impl FnOnce(&mut Dag) -> NodeId) -> NodeId {
    if is_zero(dag, tangent) {
        return dag.konst(0.0);
    }
    let p = partial(dag);
    match cval(dag, p) {
        Some(c) if c.is_finite() => dag.mul(tangent, p),
        _ => dag.zmul(tangent, p),
    }
}

/// `tangent * other`, one term of the product rule: `other` is an operand of the model, not
/// a partial. A tangent that is zero at run time must give +0 where `other` is infinite or
/// NaN, as the interpreter skips a zero tangent, so the term is a `ZMul` when
/// [`Dag::may_be_nonfinite`] flags `other` and a plain `mul` when it does not. The zero
/// constant tangent never builds the term.
pub(crate) fn product_term(dag: &mut Dag, tangent: NodeId, other: NodeId) -> NodeId {
    if dag.may_be_nonfinite(other) {
        zterm(dag, tangent, |_| other)
    } else {
        term(dag, tangent, |_| other)
    }
}

/// `tangent * partial()` for a call whose partial is finite for every finite argument: the
/// term is a `ZMul` when [`Dag::may_be_nonfinite`] flags the argument `arg`, because an
/// infinite or NaN argument gives an infinite or NaN partial (`cosh(1/m)` has the partial
/// `sinh(1/m)`), and a zero tangent must still give +0. A plain `mul` otherwise, so a model
/// whose arguments are never flagged gets no new guard.
fn call_term(
    dag: &mut Dag,
    tangent: NodeId,
    arg: NodeId,
    partial: impl FnOnce(&mut Dag) -> NodeId,
) -> NodeId {
    if dag.may_be_nonfinite(arg) {
        zterm(dag, tangent, partial)
    } else {
        term(dag, tangent, partial)
    }
}

/// `tangent / denominator`, the first term of the quotient rule. A tangent that is zero at
/// run time over a zero denominator must give +0, not `0 / 0`, as the interpreter skips a
/// zero tangent, so the term is a `ZDiv` unless the denominator is a nonzero finite
/// constant (then a plain `div` has the same value for every tangent but a `-0.0`). The
/// zero constant tangent never builds the term.
fn quotient_term(dag: &mut Dag, tangent: NodeId, denom: NodeId) -> NodeId {
    if is_zero(dag, tangent) {
        return dag.konst(0.0);
    }
    match cval(dag, denom) {
        Some(c) if c != 0.0 && c.is_finite() => dag.div(tangent, denom),
        _ => dag.zdiv(tangent, denom),
    }
}

/// Whether a derivative of the call `op` has a partial that can be infinite or NaN at a
/// finite input, so that a zero tangent must not multiply it. This is the list of rules in
/// [`rule`] that use [`zterm`], for the multilinear forms of `multi.rs`, which multiply the
/// partials of every order by a product of tangents. A `pow` with a constant integer
/// exponent `>= 1` has finite partials of every order.
pub(crate) fn partial_may_be_nonfinite(dag: &Dag, op: &Op) -> bool {
    match *op {
        Op::Call1(f, _) => !matches!(
            f,
            Fn1::Sin
                | Fn1::Cos
                | Fn1::Tan
                | Fn1::Atan
                | Fn1::Sinh
                | Fn1::Cosh
                | Fn1::Tanh
                | Fn1::Exp
                | Fn1::Abs
                | Fn1::Erf
                | Fn1::Erfc
        ),
        Op::PowI(_, n) => n < 0,
        Op::Table(..) => false,
        Op::Call2(Fn2::Pow, _, b) => {
            !matches!(cval(dag, b), Some(c) if c >= 1.0 && c.fract() == 0.0 && c.is_finite())
        }
        Op::Call2(Fn2::Min | Fn2::Max, ..) => false,
        Op::Call2(..) => true,
        _ => false,
    }
}

/// The children of `op` whose derivatives the rule for `op` reads.
pub(crate) fn diff_children(op: &Op, f: &mut dyn FnMut(NodeId)) {
    match *op {
        Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) | Op::TableSlope(..) => {}
        Op::Call1(Fn1::Floor | Fn1::Ceil | Fn1::Heav | Fn1::Sign, _) => {}
        Op::Select(_, a, b) | Op::If(_, a, b) => {
            f(a);
            f(b);
        }
        Op::Call2(Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma, _, x) => f(x),
        _ => children(op, f),
    }
}

/// `d node / d wrt`. `memo` caches results per `(node, wrt)` and may be shared across
/// calls on the same DAG. Walks the needed nodes in id order, so deep DAGs do not recurse.
///
/// A call to a function `f` differentiates through a tangent unit of `f` (SPEC section
/// 8af-5): a unit that takes the arguments and the tangents of the active formals (those
/// whose tangent is not the constant 0) and returns the value and the derivative. Its body
/// is the body of `f` with each call inside it replaced by the matching tangent call, whose
/// output 0 is the value, so a frame runs each recursive call once. A call whose argument
/// tangents are all the constant 0, into a function that does not read `wrt` through any
/// chain of calls, has derivative 0 and makes no call. The units are cached in the derived
/// map of the DAG, so a second derivative request reuses them.
pub fn d(
    dag: &mut Dag,
    memo: &mut HashMap<(NodeId, Wrt), NodeId>,
    node: NodeId,
    wrt: Wrt,
) -> NodeId {
    if let Some(&r) = memo.get(&(node, wrt)) {
        return r;
    }
    let mut cx = Cx::default();
    let mut seen = HashSet::new();
    let mut stack = vec![node];
    while let Some(id) = stack.pop() {
        if memo.contains_key(&(id, wrt)) || !seen.insert(id) {
            continue;
        }
        let op = *dag.op(id);
        match op {
            // `diff_children` does not visit the arguments of a call.
            Op::Call(_, l) => stack.extend_from_slice(dag.args(l)),
            Op::Out(..) | Op::Arg(_) => {}
            _ => diff_children(&op, &mut |c| stack.push(c)),
        }
    }
    let mut order: Vec<NodeId> = seen.into_iter().collect();
    order.sort_unstable();
    for id in order {
        let r = match *dag.op(id) {
            Op::Call(g, l) => {
                let args = dag.args(l).to_vec();
                let targs: Vec<NodeId> = args
                    .iter()
                    .map(|&a| memo.get(&(a, wrt)).copied().unwrap_or(NodeId(u32::MAX)))
                    .collect();
                if targs.iter().any(|t| t.0 == u32::MAX) {
                    dag.konst(f64::NAN)
                } else {
                    // At the top level the primal call stays in the value, so an entry that
                    // needs the value and the derivative makes about 2(n + 1) calls.
                    call_tangent(dag, memo, &mut cx, g, &args, &targs, wrt).1
                }
            }
            // An output j > 0 of a call and a formal outside a unit have no tangent here.
            Op::Out(..) | Op::Arg(_) => dag.konst(f64::NAN),
            _ => rule(dag, &|c| memo.get(&(c, wrt)).copied(), id, wrt),
        };
        memo.insert((id, wrt), r);
    }
    match memo.get(&(node, wrt)) {
        Some(&r) => r,
        None => dag.konst(f64::NAN),
    }
}

/// The derived-map tag of a tangent unit; the jet units of `multi.rs` use other tags.
const TANGENT_TAG: u32 = 0;

/// State of one [`d`] request: whether a function reads a variable through its calls.
#[derive(Default)]
struct Cx {
    reads: HashMap<(u32, Wrt), bool>,
}

/// Whether the body of unit `g`, or of a unit it calls through any chain of calls, reads
/// `w`. A unit without a body counts as reading it. The unit graph can be cyclic, so the
/// walk keeps a visited set of units.
fn reads(dag: &Dag, cx: &mut Cx, g: u32, w: Wrt) -> bool {
    if let Some(&r) = cx.reads.get(&(g, w)) {
        return r;
    }
    let mut units = vec![g];
    let mut seen_units = HashSet::from([g]);
    let mut seen = HashSet::new();
    let mut found = false;
    'units: while let Some(u) = units.pop() {
        let Some(outs) = dag.unit(u).and_then(|x| x.outputs()) else {
            found = true;
            break;
        };
        let mut stack = outs.to_vec();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let op = *dag.op(id);
            match op {
                Op::Var(i) if w == Wrt::U(i) => {
                    found = true;
                    break 'units;
                }
                Op::Param(j) if w == Wrt::P(j) => {
                    found = true;
                    break 'units;
                }
                Op::Call(h, _) if seen_units.insert(h) => units.push(h),
                _ => {}
            }
            dag.operands(&op, &mut |c| stack.push(c));
        }
    }
    cx.reads.insert((g, w), found);
    found
}

/// The value and the derivative along `wrt` of the call `g(args)`, given the tangents
/// `targs` of the arguments: `(tc, out(tc, 1))` for the tangent call `tc`, whose output 0
/// is the value, or the primal call and the constant 0 when no tangent is active and `g`
/// does not read `wrt`. A unit without a body or with more than one output gets the
/// primal call and a NaN derivative.
fn call_tangent(
    dag: &mut Dag,
    memo: &mut HashMap<(NodeId, Wrt), NodeId>,
    cx: &mut Cx,
    g: u32,
    args: &[NodeId],
    targs: &[NodeId],
    wrt: Wrt,
) -> (NodeId, NodeId) {
    let ok = dag
        .unit(g)
        .is_some_and(|u| u.nout == 1 && u.outputs().is_some());
    if !ok {
        let c = dag.call(g, args);
        return (c, dag.konst(f64::NAN));
    }
    let mask: Vec<u32> = (0..targs.len() as u32)
        .filter(|&i| !is_zero(dag, targs[i as usize]))
        .collect();
    let r = reads(dag, cx, g, wrt);
    if mask.is_empty() && !r {
        let c = dag.call(g, args);
        return (c, dag.konst(0.0));
    }
    let t = tangent_unit(dag, memo, cx, g, wrt, r, &mask);
    let mut a = args.to_vec();
    a.extend(mask.iter().map(|&i| targs[i as usize]));
    let tc = dag.call(t, &a);
    (tc, dag.out(tc, 1))
}

/// The tangent unit of `f` for the active formals `mask`, along `wrt` when `reads_w` and
/// with no variable otherwise (a unit that reads no variable is shared by every `wrt`).
/// The unit is registered before its body is built, so a recursive call inside the body
/// finds it.
#[allow(clippy::too_many_arguments)]
fn tangent_unit(
    dag: &mut Dag,
    memo: &mut HashMap<(NodeId, Wrt), NodeId>,
    cx: &mut Cx,
    f: u32,
    wrt: Wrt,
    reads_w: bool,
    mask: &[u32],
) -> u32 {
    let (wtag, widx) = match (reads_w, wrt) {
        (false, _) => (0, 0),
        (true, Wrt::U(i)) => (1, i),
        (true, Wrt::P(j)) => (2, j),
    };
    let mut key = vec![TANGENT_TAG, f, wtag, widx];
    key.extend_from_slice(mask);
    if let Some(t) = dag.derived(&key) {
        return t;
    }
    let unit = dag.unit(f).expect("call_tangent checked the unit");
    let (name, arity) = (unit.name.clone(), unit.arity);
    let out0 = unit.outputs().expect("call_tangent checked the body")[0];
    let t = dag.add_unit(&name, arity + mask.len() as u32, 2);
    dag.set_derived(key, t);

    // The calls whose tangent the derivative reads: the walk of `d` from the output, plus
    // the arguments of each call, inside the nodes that depend on a formal or a call. A call
    // reached only through a condition stays a primal call.
    let local = |dag: &Dag, id: NodeId| dag.arg_dep(id) || dag.has_call(id);
    let mut needed = HashSet::new();
    let mut stack = vec![out0];
    while let Some(id) = stack.pop() {
        if !local(dag, id) || !needed.insert(id) {
            continue;
        }
        let op = *dag.op(id);
        match op {
            Op::Call(_, l) => stack.extend_from_slice(dag.args(l)),
            Op::Out(c, _) => stack.push(c),
            Op::Arg(_) => {}
            _ => diff_children(&op, &mut |c| stack.push(c)),
        }
    }
    // A node without a call rebuilds to itself, so only the nodes that reach a call are
    // rebuilt, in id order (an operand has a smaller id than its user).
    let mut with_call = HashSet::new();
    let mut stack = vec![out0];
    while let Some(id) = stack.pop() {
        if !dag.has_call(id) || !with_call.insert(id) {
            continue;
        }
        let op = *dag.op(id);
        dag.operands(&op, &mut |c| stack.push(c));
    }
    let mut order: Vec<NodeId> = with_call.into_iter().collect();
    order.sort_unstable();

    let w = reads_w.then_some(wrt);
    let mut tan: HashMap<NodeId, NodeId> = HashMap::new();
    for k in 0..arity {
        let a = dag.arg(k);
        let ta = match mask.iter().position(|&m| m == k) {
            Some(r) => dag.arg(arity + r as u32),
            None => dag.konst(0.0),
        };
        tan.insert(a, ta);
    }
    let mut prim: HashMap<NodeId, NodeId> = HashMap::new();
    for id in order {
        let op = *dag.op(id);
        let p = match op {
            Op::Call(g, l) => {
                let args: Vec<NodeId> = dag
                    .args(l)
                    .iter()
                    .map(|a| prim.get(a).copied().unwrap_or(*a))
                    .collect();
                if needed.contains(&id) {
                    let targs: Vec<NodeId> = args
                        .iter()
                        .map(|&a| local_tangent(dag, memo, &mut tan, a, w))
                        .collect();
                    let (p, tp) = call_tangent(dag, memo, cx, g, &args, &targs, wrt);
                    tan.insert(p, tp);
                    p
                } else {
                    dag.call(g, &args)
                }
            }
            _ => dag.rebuild(&op, &|c| prim.get(&c).copied().unwrap_or(c)),
        };
        prim.insert(id, p);
    }
    let p0 = prim.get(&out0).copied().unwrap_or(out0);
    let t0 = local_tangent(dag, memo, &mut tan, p0, w);
    let ok = dag.set_body(t, &[p0, t0]);
    debug_assert!(ok, "a tangent body has two outputs and in-range formals");
    t
}

/// The tangent of `node` inside a unit body, with the tangents of the formals and of the
/// tangent calls already in `tan`. A node that reads no formal and no call is the same in
/// every frame, so its tangent comes from [`d`] along `w`, or is 0 for a unit that reads no
/// variable. A call or a call output missing from `tan` has a NaN tangent.
fn local_tangent(
    dag: &mut Dag,
    memo: &mut HashMap<(NodeId, Wrt), NodeId>,
    tan: &mut HashMap<NodeId, NodeId>,
    node: NodeId,
    w: Option<Wrt>,
) -> NodeId {
    if let Some(&r) = tan.get(&node) {
        return r;
    }
    let mut seen = HashSet::new();
    let mut stack = vec![node];
    while let Some(id) = stack.pop() {
        if tan.contains_key(&id) || !seen.insert(id) {
            continue;
        }
        if !dag.arg_dep(id) && !dag.has_call(id) {
            continue;
        }
        let op = *dag.op(id);
        if !matches!(op, Op::Call(..) | Op::Out(..) | Op::Arg(_)) {
            diff_children(&op, &mut |c| stack.push(c));
        }
    }
    let mut order: Vec<NodeId> = seen.into_iter().collect();
    order.sort_unstable();
    // `rule` reads `wrt` only for a state or a parameter, which are leaves handled here.
    let wrt = w.unwrap_or(Wrt::U(u32::MAX));
    for id in order {
        let r = if !dag.arg_dep(id) && !dag.has_call(id) {
            match w {
                Some(w) => d(dag, memo, id, w),
                None => dag.konst(0.0),
            }
        } else {
            match *dag.op(id) {
                Op::Call(..) | Op::Out(..) | Op::Arg(_) => dag.konst(f64::NAN),
                _ => rule(dag, &|c| tan.get(&c).copied(), id, wrt),
            }
        };
        tan.insert(id, r);
    }
    tan[&node]
}

/// The first-order rule for `node`, reading the tangents of its children through `look`.
fn rule(dag: &mut Dag, look: &dyn Fn(NodeId) -> Option<NodeId>, node: NodeId, wrt: Wrt) -> NodeId {
    let mut nan = None;
    let mut dc = |dag: &mut Dag, c: NodeId| match look(c) {
        Some(r) => r,
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
            let t1 = product_term(dag, da, b);
            let t2 = product_term(dag, db, a);
            dag.add(t1, t2)
        }
        // `ZMul(t, p)` is `t * p` with a zero `t` giving +0, so each term of the product
        // rule is guarded by its own tangent factor. A zero `t` makes the second term +0,
        // and a zero `da` makes the first one +0, even where `p`, `dp` or `da` is infinite.
        Op::ZMul(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            let t1 = zterm(dag, da, |_| b);
            let t2 = if is_zero(dag, db) {
                dag.konst(0.0)
            } else {
                dag.zmul(a, db)
            };
            dag.add(t1, t2)
        }
        Op::Div(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            let ta = quotient_term(dag, da, b);
            // The quotient is `node` itself.
            let tb = zterm(dag, db, |g| {
                let q = g.div(a, b);
                let qb = g.div(q, b);
                g.neg(qb)
            });
            dag.add(ta, tb)
        }
        // `ZDiv(n, d)` is `n / d` with a zero `n` giving +0, so the quotient rule of `Div`
        // applies with each term guarded by its own tangent: `ZDiv(dn, d)` is +0 for a zero
        // `dn`, and the second term is `ZMul(dd, -(q / d))`, with `q` the node itself. A
        // `-(q / d)` that is infinite or NaN (a zero `d`) is hidden by a zero `dd`.
        Op::ZDiv(a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            let ta = quotient_term(dag, da, b);
            let tb = zterm(dag, db, |g| {
                let qb = g.div(node, b);
                g.neg(qb)
            });
            dag.add(ta, tb)
        }
        Op::PowI(a, n) => {
            let da = dc(dag, a);
            // `n < 0` has an infinite partial at 0; `n >= 2` is finite wherever `a` is.
            let slope = |g: &mut Dag| {
                let c = g.konst(f64::from(n));
                let p = g.powi(a, n - 1);
                g.mul(c, p)
            };
            if n < 0 {
                zterm(dag, da, slope)
            } else {
                call_term(dag, da, a, slope)
            }
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
        // `select` makes an `If` again when an arm of the derivative reaches a call.
        Op::Select(c, a, b) | Op::If(c, a, b) => {
            let (da, db) = (dc(dag, a), dc(dag, b));
            dag.select(c, da, db)
        }
        Op::Table(id, a) => {
            let da = dc(dag, a);
            call_term(dag, da, a, |g| g.table_slope(id, a))
        }
        // `d` and `local_tangent` handle formals and calls before they reach a rule.
        Op::Arg(_) | Op::Call(..) | Op::Out(..) => dag.konst(f64::NAN),
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
        Fn1::Sin => call_term(dag, da, a, |g| g.call1(Fn1::Cos, a)),
        Fn1::Cos => call_term(dag, da, a, |g| {
            let s = g.call1(Fn1::Sin, a);
            g.neg(s)
        }),
        Fn1::Tan => call_term(dag, da, a, |g| {
            let one = g.konst(1.0);
            let tt = g.mul(node, node);
            g.add(one, tt)
        }),
        Fn1::Asin => zterm(dag, da, |g| inv_sqrt_1m_sq(g, a)),
        Fn1::Acos => zterm(dag, da, |g| {
            let r = inv_sqrt_1m_sq(g, a);
            g.neg(r)
        }),
        Fn1::Atan => call_term(dag, da, a, |g| {
            let one = g.konst(1.0);
            let aa = g.mul(a, a);
            let s = g.add(one, aa);
            g.div(one, s)
        }),
        Fn1::Sinh => call_term(dag, da, a, |g| g.call1(Fn1::Cosh, a)),
        Fn1::Cosh => call_term(dag, da, a, |g| g.call1(Fn1::Sinh, a)),
        Fn1::Tanh => call_term(dag, da, a, |g| {
            let one = g.konst(1.0);
            let tt = g.mul(node, node);
            g.sub(one, tt)
        }),
        Fn1::Exp => call_term(dag, da, a, |_| node),
        Fn1::Ln => zterm(dag, da, |g| {
            let one = g.konst(1.0);
            g.div(one, a)
        }),
        Fn1::Log10 => zterm(dag, da, |g| {
            let one = g.konst(1.0);
            let c = g.konst(std::f64::consts::LN_10);
            let s = g.mul(a, c);
            g.div(one, s)
        }),
        Fn1::Sqrt => zterm(dag, da, |g| {
            let one = g.konst(1.0);
            let two = g.konst(2.0);
            let s = g.mul(two, node);
            g.div(one, s)
        }),
        Fn1::Abs => call_term(dag, da, a, |g| g.call1(Fn1::Sign, a)),
        Fn1::Erf => call_term(dag, da, a, |g| erf_slope(g, a)),
        Fn1::Erfc => call_term(dag, da, a, |g| {
            let s = erf_slope(g, a);
            g.neg(s)
        }),
        Fn1::Lgamma => zterm(dag, da, |g| {
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
            let dy = zterm(dag, da, |g| {
                let d = sum_squares(g, a, b);
                g.div(b, d)
            });
            let dx = zterm(dag, db, |g| {
                let d = sum_squares(g, a, b);
                let q = g.div(a, d);
                g.neg(q)
            });
            dag.add(dy, dx)
        }
        Fn2::Pow => {
            // `b * a^(b-1)` is finite for finite `a` only when `b` is a constant integer
            // `>= 1`; otherwise `a = 0` gives an infinite or NaN partial.
            let slope = |g: &mut Dag| {
                let one = g.konst(1.0);
                let ym1 = g.sub(b, one);
                let p = g.call2(Fn2::Pow, a, ym1);
                g.mul(b, p)
            };
            let finite_slope = !partial_may_be_nonfinite(dag, &Op::Call2(Fn2::Pow, a, b));
            let dx = if finite_slope {
                call_term(dag, da, a, slope)
            } else {
                zterm(dag, da, slope)
            };
            // A zero result contributes nothing, so `0 * ln(0)` is never formed.
            let dy = if is_zero(dag, db) {
                dag.konst(0.0)
            } else {
                let ln = dag.call1(Fn1::Ln, a);
                let r = dag.mul(node, ln);
                // `r` is infinite where `a` is 0 or the power overflows, and `db` can be 0.
                let t = dag.zmul(db, r);
                let z = dag.konst(0.0);
                let c = dag.cmp(Cmp::Eq, node, z);
                dag.select(c, z, t)
            };
            dag.add(dx, dy)
        }
        Fn2::Mod => {
            let dy = zterm(dag, db, |g| {
                let num = g.sub(a, node);
                let q = g.div(num, b);
                g.neg(q)
            });
            dag.add(da, dy)
        }
        // DLMF 10.6.1 for the order k = trunc(a) that `besselj` and `bessely` use. The order
        // has no tangent, since a truncated order is piecewise constant.
        Fn2::BesselJ | Fn2::BesselY => zterm(dag, db, |g| {
            let one = g.konst(1.0);
            let k = trunc(g, a);
            let nm = g.sub(k, one);
            let np = g.add(k, one);
            let jm = g.call2(f, nm, b);
            let jp = g.call2(f, np, b);
            let diff = g.sub(jm, jp);
            let half = g.konst(0.5);
            g.mul(diff, half)
        }),
        Fn2::Polygamma => zterm(dag, db, |g| {
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

/// `a` truncated toward zero: `floor(a)` for `a >= 0`, else `ceil(a)` (so NaN stays NaN).
/// A constant folds when the node is built.
fn trunc(g: &mut Dag, a: NodeId) -> NodeId {
    let zero = g.konst(0.0);
    let nonneg = g.cmp(Cmp::Ge, a, zero);
    let down = g.call1(Fn1::Floor, a);
    let up = g.call1(Fn1::Ceil, a);
    g.select(nonneg, down, up)
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
