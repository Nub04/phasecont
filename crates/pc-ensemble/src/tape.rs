//! The right-hand side of a model as a flat tape, evaluated in `f32` or `f64`.
//!
//! The tape holds the nodes reachable from `ir.f`, renumbered from zero, in the order
//! `pc_ir::eval::eval` visits them. Each op follows `pc_ir::eval::eval_op`, with the scalar
//! type generic. The kernel emitters walk the same tape, so the CPU loop and a GPU kernel
//! run the same operations in the same order.
//!
//! Constructs the kernels cannot lower are refused when the tape is built, by name:
//! tables, `erf`, `erfc`, `lgamma`, Bessel functions and `polygamma`.

use pc_ir::IrModel;
use pc_ir::dag::{Cmp, Fn1, Fn2, NodeId, Op};
use pc_ir::eval::live_set;

use crate::job::EnsembleError;
use crate::real::Real;

#[derive(Clone, Debug)]
pub struct Tape {
    pub n: usize,
    pub n_p: usize,
    /// Ops in evaluation order. A child's id is the index of its slot.
    pub ops: Vec<Op>,
    /// The slot of each state's right-hand side.
    pub outputs: Vec<u32>,
}

fn unsupported_name(op: &Op) -> Option<&'static str> {
    match *op {
        Op::Dir(_) => Some("directional derivative input"),
        Op::Table(..) => Some("table"),
        Op::TableSlope(..) => Some("table slope"),
        Op::Call1(f, _) => match f {
            Fn1::Erf => Some("erf"),
            Fn1::Erfc => Some("erfc"),
            Fn1::Lgamma => Some("lgamma"),
            _ => None,
        },
        Op::Call2(f, ..) => match f {
            Fn2::BesselJ => Some("besselj"),
            Fn2::BesselY => Some("bessely"),
            Fn2::Polygamma => Some("polygamma"),
            _ => None,
        },
        _ => None,
    }
}

fn remap(op: &Op, m: &[u32]) -> Op {
    let r = |id: NodeId| NodeId(m[id.0 as usize]);
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => *op,
        Op::Neg(a) => Op::Neg(r(a)),
        Op::Add(a, b) => Op::Add(r(a), r(b)),
        Op::Sub(a, b) => Op::Sub(r(a), r(b)),
        Op::Mul(a, b) => Op::Mul(r(a), r(b)),
        Op::Div(a, b) => Op::Div(r(a), r(b)),
        Op::PowI(a, n) => Op::PowI(r(a), n),
        Op::Call1(f, a) => Op::Call1(f, r(a)),
        Op::Call2(f, a, b) => Op::Call2(f, r(a), r(b)),
        Op::Cmp(c, a, b) => Op::Cmp(c, r(a), r(b)),
        Op::And(a, b) => Op::And(r(a), r(b)),
        Op::Or(a, b) => Op::Or(r(a), r(b)),
        Op::Not(a) => Op::Not(r(a)),
        Op::Select(c, a, b) => Op::Select(r(c), r(a), r(b)),
        Op::Table(t, a) => Op::Table(t, r(a)),
        Op::TableSlope(t, a) => Op::TableSlope(t, r(a)),
    }
}

impl Tape {
    /// Builds the tape of `ir.f`, or names the constructs that cannot run.
    pub fn build(ir: &IrModel) -> Result<Tape, EnsembleError> {
        let live = live_set(&ir.dag, &ir.f);
        let mut map = vec![u32::MAX; ir.dag.len()];
        let mut ops = Vec::new();
        let mut names: Vec<&'static str> = Vec::new();
        for (i, &is_live) in live.iter().enumerate() {
            if !is_live {
                continue;
            }
            let op = ir.dag.op(NodeId(i as u32));
            if let Some(name) = unsupported_name(op) {
                if !names.contains(&name) {
                    names.push(name);
                }
                continue;
            }
            map[i] = ops.len() as u32;
            ops.push(remap(op, &map));
        }
        if !names.is_empty() {
            return Err(EnsembleError::Unsupported(format!(
                "the right-hand side uses {}, which the ensemble engine cannot run",
                names.join(", ")
            )));
        }
        let mut outputs = Vec::with_capacity(ir.f.len());
        for id in &ir.f {
            match map.get(id.0 as usize) {
                Some(&s) if s != u32::MAX => outputs.push(s),
                _ => {
                    return Err(EnsembleError::BadJob(
                        "a right-hand side is outside the model's expression graph".into(),
                    ));
                }
            }
        }
        Ok(Tape {
            n: ir.n(),
            n_p: ir.n_p(),
            ops,
            outputs,
        })
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Writes `f(t, x)` to `out`. `p` holds the `n_p` parameters followed by `t`. `vals` is
    /// scratch of at least [`Tape::len`] values.
    pub fn eval<T: Real>(&self, x: &[T], p: &[T], vals: &mut [T], out: &mut [T]) {
        for (i, op) in self.ops.iter().enumerate() {
            vals[i] = eval_op(op, vals, x, p);
        }
        for (o, &s) in out.iter_mut().zip(&self.outputs) {
            *o = vals[s as usize];
        }
    }
}

fn b2f<T: Real>(b: bool) -> T {
    if b { T::ONE } else { T::ZERO }
}

/// Integer power in the order documented on [`Op::PowI`].
fn powi<T: Real>(x: T, n: i32) -> T {
    let mut m = n.unsigned_abs();
    let mut base = x;
    let mut acc: Option<T> = None;
    if m == 0 {
        return T::ONE;
    }
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
        base = base * base;
    }
    let r = acc.unwrap_or(T::ONE);
    if n < 0 { T::ONE / r } else { r }
}

fn call1<T: Real>(f: Fn1, x: T) -> T {
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
        Fn1::Floor => x.floor(),
        Fn1::Ceil => x.ceil(),
        Fn1::Heav => {
            if x >= T::ZERO {
                T::ONE
            } else {
                T::ZERO
            }
        }
        Fn1::Sign => {
            if x > T::ZERO {
                T::ONE
            } else if x < T::ZERO {
                -T::ONE
            } else if x == T::ZERO {
                T::ZERO
            } else {
                T::NAN
            }
        }
        // Refused when the tape is built.
        Fn1::Erf | Fn1::Erfc | Fn1::Lgamma => T::NAN,
    }
}

fn call2<T: Real>(f: Fn2, x: T, y: T) -> T {
    match f {
        Fn2::Atan2 => x.atan2(y),
        Fn2::Pow => x.powf(y),
        Fn2::Mod => {
            let r = x.fmod(y);
            if r < T::ZERO { r + y } else { r }
        }
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
        Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma => T::NAN,
    }
}

fn eval_op<T: Real>(op: &Op, v: &[T], x: &[T], p: &[T]) -> T {
    let a = |id: NodeId| v[id.0 as usize];
    match *op {
        Op::Const(b) => T::from_f64(f64::from_bits(b)),
        Op::Var(i) => x[i as usize],
        Op::Param(j) => p[j as usize],
        Op::Neg(u) => -a(u),
        Op::Add(u, w) => a(u) + a(w),
        Op::Sub(u, w) => a(u) - a(w),
        Op::Mul(u, w) => a(u) * a(w),
        Op::Div(u, w) => a(u) / a(w),
        Op::PowI(u, n) => powi(a(u), n),
        Op::Call1(f, u) => call1(f, a(u)),
        Op::Call2(f, u, w) => call2(f, a(u), a(w)),
        Op::Cmp(c, u, w) => {
            let (l, r) = (a(u), a(w));
            b2f(match c {
                Cmp::Lt => l < r,
                Cmp::Le => l <= r,
                Cmp::Gt => l > r,
                Cmp::Ge => l >= r,
                Cmp::Eq => l == r,
            })
        }
        Op::And(u, w) => b2f(a(u) != T::ZERO && a(w) != T::ZERO),
        Op::Or(u, w) => b2f(a(u) != T::ZERO || a(w) != T::ZERO),
        Op::Not(u) => b2f(a(u) == T::ZERO),
        Op::Select(c, u, w) => {
            if a(c) != T::ZERO {
                a(u)
            } else {
                a(w)
            }
        }
        // Refused when the tape is built.
        Op::Dir(_) | Op::Table(..) | Op::TableSlope(..) => T::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ir::eval::eval;

    fn model(src: &str) -> IrModel {
        use pc_ode::{MapLoader, Severity, SourceMap, load};
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
        pc_ir::lower(&l.expanded, &l.info).expect("lower")
    }

    /// The f64 tape gives the same bits as the DAG interpreter.
    #[test]
    fn matches_the_interpreter_bit_for_bit() {
        let ir = model(
            "par a=0.7, b=2\n\
             x'=sin(x)*a+exp(-y)/(1+x^2)+b*x^5-abs(y)+if(x>y)then(max(x,y))else(min(x,y))\n\
             y'=atan2(x,y)+sqrt(1+y*y)+heav(x-y)+sign(y)+mod(x,1.3)+x^(-3)+t\n\
             done\n",
        );
        let tape = Tape::build(&ir).expect("tape");
        let mut vals = vec![0.0; tape.len()];
        for (x0, x1, t) in [(0.3, -1.2, 0.0), (-2.5, 4.0, 1.5), (0.0, 0.0, 2.0)] {
            let x = [x0, x1];
            let p = [0.7, 2.0, t];
            let mut got = [0.0; 2];
            tape.eval::<f64>(&x, &p, &mut vals, &mut got);
            let mut want = [0.0; 2];
            eval(&ir.dag, &ir.f, &x, &p, &[], &mut want);
            for i in 0..2 {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "state {i} at {x:?}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn single_precision_is_close_to_double() {
        let ir = model("x'=sin(x)+exp(-x*x)\ndone\n");
        let tape = Tape::build(&ir).expect("tape");
        let mut v32 = vec![0.0f32; tape.len()];
        let mut v64 = vec![0.0f64; tape.len()];
        let (mut o32, mut o64) = ([0.0f32], [0.0f64]);
        tape.eval::<f32>(&[0.7], &[0.0], &mut v32, &mut o32);
        tape.eval::<f64>(&[0.7], &[0.0], &mut v64, &mut o64);
        assert!((f64::from(o32[0]) - o64[0]).abs() < 1e-6);
    }

    #[test]
    fn refuses_constructs_by_name() {
        for (src, name) in [
            ("x'=erf(x)\ndone\n", "erf"),
            ("x'=lgamma(x)+1\ndone\n", "lgamma"),
            ("x'=besselj(1,x)\ndone\n", "besselj"),
        ] {
            let ir = model(src);
            match Tape::build(&ir) {
                Err(EnsembleError::Unsupported(m)) => {
                    assert!(m.contains(name), "{m}");
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }
}
