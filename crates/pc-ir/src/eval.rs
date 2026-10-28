//! Interpreter for the DAG. [`eval_op`] is the single per-op rule set; constant folding in
//! `dag.rs` calls it too, so folded constants carry the bits the interpreter would produce.

use crate::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use crate::interp::{Dual, Scalar};
use crate::math;

fn b2f(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

fn call1(f: Fn1, x: f64) -> f64 {
    match f {
        Fn1::Sin => math::sin(x),
        Fn1::Cos => math::cos(x),
        Fn1::Tan => math::tan(x),
        Fn1::Asin => math::asin(x),
        Fn1::Acos => math::acos(x),
        Fn1::Atan => math::atan(x),
        Fn1::Sinh => math::sinh(x),
        Fn1::Cosh => math::cosh(x),
        Fn1::Tanh => math::tanh(x),
        Fn1::Exp => math::exp(x),
        Fn1::Ln => math::ln(x),
        Fn1::Log10 => math::log10(x),
        Fn1::Sqrt => math::sqrt(x),
        Fn1::Abs => x.abs(),
        Fn1::Erf => math::erf(x),
        Fn1::Erfc => math::erfc(x),
        Fn1::Lgamma => math::lgamma(x),
        Fn1::Floor => math::floor(x),
        Fn1::Ceil => math::ceil(x),
        Fn1::Heav => math::heav(x),
        Fn1::Sign => math::sign(x),
    }
}

fn call2(f: Fn2, x: f64, y: f64) -> f64 {
    match f {
        Fn2::Atan2 => math::atan2(x, y),
        Fn2::Pow => math::pow(x, y),
        Fn2::Mod => math::pc_mod(x, y),
        Fn2::BesselJ => math::besselj(x, y),
        Fn2::BesselY => math::bessely(x, y),
        Fn2::Polygamma => math::polygamma(x, y),
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
    }
}

/// Integer power in the order documented on [`Op::PowI`].
fn powi(x: f64, n: i32) -> f64 {
    let mut m = n.unsigned_abs();
    if m == 0 {
        return 1.0;
    }
    let mut base = x;
    let mut acc: Option<f64> = None;
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
        base *= base;
    }
    let r = acc.unwrap_or(1.0);
    if n < 0 { 1.0 / r } else { r }
}

/// Value of one op. `arg` returns the value of a child node; leaves read `x`, `p` and `v`
/// (a missing entry gives NaN). Unknown table ids give NaN.
pub fn eval_op(
    op: &Op,
    tables: &[Vec<f64>],
    arg: &dyn Fn(NodeId) -> f64,
    x: &[f64],
    p: &[f64],
    v: &[f64],
) -> f64 {
    let leaf = |s: &[f64], i: u32| s.get(i as usize).copied().unwrap_or(f64::NAN);
    let table = |id: u32| tables.get(id as usize).map(Vec::as_slice).unwrap_or(&[]);
    match *op {
        Op::Const(b) => f64::from_bits(b),
        Op::Var(i) => leaf(x, i),
        Op::Param(j) => leaf(p, j),
        Op::Dir(k) => leaf(v, k),
        Op::Neg(a) => -arg(a),
        Op::Add(a, b) => arg(a) + arg(b),
        Op::Sub(a, b) => arg(a) - arg(b),
        Op::Mul(a, b) => arg(a) * arg(b),
        Op::Div(a, b) => arg(a) / arg(b),
        Op::PowI(a, n) => powi(arg(a), n),
        Op::Call1(f, a) => call1(f, arg(a)),
        Op::Call2(f, a, b) => call2(f, arg(a), arg(b)),
        Op::Cmp(c, a, b) => {
            let (l, r) = (arg(a), arg(b));
            b2f(match c {
                Cmp::Lt => l < r,
                Cmp::Le => l <= r,
                Cmp::Gt => l > r,
                Cmp::Ge => l >= r,
                Cmp::Eq => l == r,
            })
        }
        Op::And(a, b) => b2f(arg(a) != 0.0 && arg(b) != 0.0),
        Op::Or(a, b) => b2f(arg(a) != 0.0 || arg(b) != 0.0),
        Op::Not(a) => b2f(arg(a) == 0.0),
        Op::Select(c, a, b) => {
            if arg(c) != 0.0 {
                arg(a)
            } else {
                arg(b)
            }
        }
        Op::Table(id, a) => math::table(table(id), arg(a)),
        Op::TableSlope(id, a) => math::table_slope(table(id), arg(a)),
    }
}

/// Calls `f` on each child of `op`, in operand order.
pub fn children(op: &Op, f: &mut dyn FnMut(NodeId)) {
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => {}
        Op::Neg(a)
        | Op::PowI(a, _)
        | Op::Call1(_, a)
        | Op::Not(a)
        | Op::Table(_, a)
        | Op::TableSlope(_, a) => f(a),
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::Call2(_, a, b)
        | Op::Cmp(_, a, b)
        | Op::And(a, b)
        | Op::Or(a, b) => {
            f(a);
            f(b);
        }
        Op::Select(c, a, b) => {
            f(c);
            f(a);
            f(b);
        }
    }
}

/// Marks the nodes reachable from `outputs`. An output id outside the DAG is ignored.
pub fn live_set(dag: &Dag, outputs: &[NodeId]) -> Vec<bool> {
    let mut live = vec![false; dag.len()];
    for o in outputs {
        if let Some(l) = live.get_mut(o.0 as usize) {
            *l = true;
        }
    }
    for i in (0..dag.len()).rev() {
        if live[i] {
            children(dag.op(NodeId(i as u32)), &mut |c| {
                if let Some(l) = live.get_mut(c.0 as usize) {
                    *l = true;
                }
            });
        }
    }
    live
}

/// Evaluates every node reachable from `outputs` in increasing id order and writes
/// `out[i] = value(outputs[i])`. `p` carries time at index `n_p`. Select evaluates both
/// branches (no short circuit). Extra `out` slots and out-of-range outputs are NaN.
pub fn eval(dag: &Dag, outputs: &[NodeId], x: &[f64], p: &[f64], v: &[f64], out: &mut [f64]) {
    let n = dag.len();
    let live = live_set(dag, outputs);
    let mut vals = vec![f64::NAN; n];
    for i in 0..n {
        if live[i] {
            let r = {
                let get = |id: NodeId| vals.get(id.0 as usize).copied().unwrap_or(f64::NAN);
                eval_op(dag.op(NodeId(i as u32)), &dag.tables, &get, x, p, v)
            };
            vals[i] = r;
        }
    }
    for (slot, o) in out.iter_mut().zip(outputs) {
        *slot = vals.get(o.0 as usize).copied().unwrap_or(f64::NAN);
    }
}

/// `c * e`, except that an exact input (`e == 0`) contributes nothing even where `c` is
/// infinite or NaN, as a zero tangent does in [`Dual`].
fn mulz(c: f64, e: f64) -> f64 {
    if e == 0.0 { 0.0 } else { c * e }
}

/// `f` on a dual number, for the exact first derivative of a one-argument builtin.
fn dual1(f: Fn1, x: Dual<f64>) -> Dual<f64> {
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
        Fn1::Erf => x.erf(),
        Fn1::Erfc => x.erfc(),
        Fn1::Lgamma => x.lgamma(),
        Fn1::Floor => x.floor(),
        Fn1::Ceil => x.ceil(),
        Fn1::Heav => x.heav(),
        Fn1::Sign => x.sign(),
    }
}

/// `|f'(a)|`, with the interpreter's derivative rules (the active branch at a kink).
fn slope1(f: Fn1, a: f64) -> f64 {
    dual1(f, Dual { re: a, eps: 1.0 }).eps.abs()
}

/// `(|d_a f|, |d_b f|)` of a two-argument builtin, with the interpreter's rules. The order
/// argument of the Bessel functions and of `polygamma` is not differentiated.
fn slope2(f: Fn2, a: f64, b: f64) -> (f64, f64) {
    let seed = |ea: f64, eb: f64| (Dual { re: a, eps: ea }, Dual { re: b, eps: eb });
    let both = |g: fn(Dual<f64>, Dual<f64>) -> Dual<f64>| {
        let (x, y) = seed(1.0, 0.0);
        let da = g(x, y).eps.abs();
        let (x, y) = seed(0.0, 1.0);
        (da, g(x, y).eps.abs())
    };
    match f {
        Fn2::Atan2 => both(|x, y| x.atan2(y)),
        Fn2::Pow => both(|x, y| x.pow(y)),
        Fn2::Mod => both(|x, y| x.modulo(y)),
        Fn2::BesselJ => both(|x, y| x.besselj(y)),
        Fn2::BesselY => both(|x, y| x.bessely(y)),
        Fn2::Min => both(|x, y| x.min(y)),
        Fn2::Max => both(|x, y| x.max(y)),
        Fn2::Polygamma => (0.0, math::polygamma(a + 1.0, b).abs()),
    }
}

/// The running error bound of one op, given its computed value `r`. `val` and `err` give
/// the value and the bound of a child. See [`eval_err`].
fn err_op(
    dag: &Dag,
    op: &Op,
    r: f64,
    val: &dyn Fn(NodeId) -> f64,
    err: &dyn Fn(NodeId) -> f64,
) -> f64 {
    let ar = r.abs();
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => 0.0,
        Op::Neg(a) => err(a),
        Op::Add(a, b) | Op::Sub(a, b) => err(a) + err(b) + ar,
        Op::Mul(a, b) => mulz(val(a).abs(), err(b)) + mulz(val(b).abs(), err(a)) + ar,
        Op::Div(a, b) => (err(a) + mulz(ar, err(b))) / val(b).abs() + ar,
        Op::PowI(a, n) => {
            let (av, ea) = (val(a).abs(), err(a));
            // Binary powering compounds by the `Mul` rule: `|n| - 1` roundings of size `|r|`,
            // plus one for the reciprocal of a negative power.
            let mut e =
                (f64::from(n.unsigned_abs()) - 1.0).max(0.0) * ar + if n < 0 { ar } else { 0.0 };
            if ea != 0.0 {
                // The relative error of the base is multiplied by |n|.
                e += f64::from(n.unsigned_abs()) * ar * ea / av;
            } else if av == 0.0 {
                return 0.0;
            }
            e
        }
        Op::Call1(f, a) => {
            let a_val = val(a);
            mulz(slope1(f, a_val), err(a)) + 2.0 * ar
        }
        Op::Call2(f, a, b) => {
            let (da, db) = slope2(f, val(a), val(b));
            mulz(da, err(a)) + mulz(db, err(b)) + 2.0 * ar
        }
        Op::Cmp(..) | Op::And(..) | Op::Or(..) | Op::Not(_) | Op::TableSlope(..) => 0.0,
        Op::Select(c, a, b) => {
            if val(c) != 0.0 {
                err(a)
            } else {
                err(b)
            }
        }
        Op::Table(id, a) => {
            let t = dag
                .tables
                .get(id as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            mulz(math::table_slope(t, val(a)).abs(), err(a)) + ar
        }
    }
}

/// Like [`eval`], and also writes `err[i]`, a running error bound for `out[i]`
/// (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 3.3).
///
/// With `u = eps / 2`, the rounding error of the computed value is bounded to first order
/// in `u` by `u * err[i]`. Each rounded operation adds `|result|`, and the bounds of the
/// operands pass through the exact partial derivatives of the operation. Leaves are exact.
/// A bound that is NaN or infinite says nothing (no extra tolerance). A builtin call adds
/// `2 |result|`, which allows the shim one ulp of error.
pub fn eval_err(
    dag: &Dag,
    outputs: &[NodeId],
    x: &[f64],
    p: &[f64],
    v: &[f64],
    out: &mut [f64],
    err: &mut [f64],
) {
    let n = dag.len();
    let live = live_set(dag, outputs);
    let mut vals = vec![f64::NAN; n];
    let mut errs = vec![f64::NAN; n];
    for i in 0..n {
        if live[i] {
            let op = dag.op(NodeId(i as u32));
            let val = |id: NodeId| vals.get(id.0 as usize).copied().unwrap_or(f64::NAN);
            let r = eval_op(op, &dag.tables, &val, x, p, v);
            let e = {
                let er = |id: NodeId| errs.get(id.0 as usize).copied().unwrap_or(f64::NAN);
                err_op(dag, op, r, &val, &er)
            };
            vals[i] = r;
            errs[i] = e;
        }
    }
    for (slot, o) in out.iter_mut().zip(outputs) {
        *slot = vals.get(o.0 as usize).copied().unwrap_or(f64::NAN);
    }
    for (slot, o) in err.iter_mut().zip(outputs) {
        *slot = errs.get(o.0 as usize).copied().unwrap_or(f64::NAN);
    }
}

#[cfg(test)]
#[path = "eval_tests.rs"]
mod tests;
