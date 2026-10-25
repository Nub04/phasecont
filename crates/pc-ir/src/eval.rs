//! Interpreter for the DAG. [`eval_op`] is the single per-op rule set; constant folding in
//! `dag.rs` calls it too, so folded constants carry the bits the interpreter would produce.

use crate::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
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

pub(crate) fn children(op: &Op, f: &mut dyn FnMut(NodeId)) {
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

/// Evaluates every node reachable from `outputs` in increasing id order and writes
/// `out[i] = value(outputs[i])`. `p` carries time at index `n_p`. Select evaluates both
/// branches (no short circuit). Extra `out` slots and out-of-range outputs are NaN.
pub fn eval(dag: &Dag, outputs: &[NodeId], x: &[f64], p: &[f64], v: &[f64], out: &mut [f64]) {
    let n = dag.len();
    let mut live = vec![false; n];
    for o in outputs {
        if let Some(l) = live.get_mut(o.0 as usize) {
            *l = true;
        }
    }
    for i in (0..n).rev() {
        if live[i] {
            children(dag.op(NodeId(i as u32)), &mut |c| {
                if let Some(l) = live.get_mut(c.0 as usize) {
                    *l = true;
                }
            });
        }
    }
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
