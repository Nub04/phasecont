use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use pc_ir::eval::eval;

fn run(d: &Dag, id: NodeId, x: &[f64], p: &[f64], v: &[f64]) -> f64 {
    let mut out = [0.0];
    eval(d, &[id], x, p, v, &mut out);
    out[0]
}

fn cval(d: &Dag, id: NodeId) -> Option<u64> {
    match d.op(id) {
        Op::Const(b) => Some(*b),
        _ => None,
    }
}

#[test]
fn hash_consing_one_node_per_structure() {
    let mut d = Dag::new(100);
    let x = d.var(0);
    let y = d.var(1);
    assert_eq!(d.var(0), x);
    let a = d.add(x, y);
    let b = d.add(x, y);
    assert_eq!(a, b);
    let n = d.len();
    let _ = d.add(x, y);
    assert_eq!(d.len(), n);
    // no operand reordering
    assert_ne!(d.add(x, y), d.add(y, x));
    assert_eq!(d.konst(2.5), d.konst(2.5));
    assert_ne!(d.konst(0.0), d.konst(-0.0));
    assert_eq!(d.konst(f64::NAN), d.konst(f64::NAN));
    assert_ne!(d.param(0), d.dir(0));
}

#[test]
fn mul_rules() {
    let mut d = Dag::new(100);
    let x = d.var(0);
    for z in [0.0, -0.0] {
        let c = d.konst(z);
        let r1 = d.mul(x, c);
        let r2 = d.mul(c, x);
        assert_eq!(cval(&d, r1), Some(0.0f64.to_bits()));
        assert_eq!(cval(&d, r2), Some(0.0f64.to_bits()));
    }
    let one = d.konst(1.0);
    assert_eq!(d.mul(x, one), x);
    assert_eq!(d.mul(one, x), x);
    let two = d.konst(2.0);
    {
        let r = d.mul(x, two);
        assert!(matches!(d.op(r), Op::Mul(_, _)));
    }
}

#[test]
fn add_sub_div_rules() {
    let mut d = Dag::new(100);
    let x = d.var(0);
    let y = d.var(1);
    for z in [0.0, -0.0] {
        let c = d.konst(z);
        assert_eq!(d.add(x, c), x);
        assert_eq!(d.add(c, x), x);
        assert_eq!(d.sub(x, c), x);
        let r = d.sub(c, x);
        assert_eq!(r, d.neg(x));
        assert!(matches!(d.op(r), Op::Neg(_)));
    }
    let one = d.konst(1.0);
    assert_eq!(d.div(x, one), x);
    {
        let r = d.div(one, x);
        assert!(matches!(d.op(r), Op::Div(_, _)));
    }
    let s = d.sub(x, x);
    assert_eq!(cval(&d, s), Some(0.0f64.to_bits()));
    // same value, different node ids: not folded
    let a = d.add(x, y);
    let b = d.add(y, x);
    {
        let r = d.sub(a, b);
        assert!(matches!(d.op(r), Op::Sub(_, _)));
    }
}

#[test]
fn neg_neg_and_powi_and_select() {
    let mut d = Dag::new(100);
    let x = d.var(0);
    let y = d.var(1);
    let n = d.neg(x);
    assert_eq!(d.neg(n), x);
    let p0 = d.powi(x, 0);
    assert_eq!(cval(&d, p0), Some(1.0f64.to_bits()));
    assert_eq!(d.powi(x, 1), x);
    {
        let r = d.powi(x, 3);
        assert!(matches!(d.op(r), Op::PowI(_, 3)));
    }
    // |n| > 16 becomes Pow with a constant exponent
    let big = d.powi(x, 17);
    match d.op(big) {
        Op::Call2(Fn2::Pow, a, e) => {
            assert_eq!(*a, x);
            assert_eq!(cval(&d, *e), Some(17.0f64.to_bits()));
        }
        o => panic!("{o:?}"),
    }
    let neg = d.powi(x, -40);
    assert!(matches!(d.op(neg), Op::Call2(Fn2::Pow, _, _)));
    // select
    let c1 = d.konst(2.0);
    let c0 = d.konst(0.0);
    let cn = d.konst(f64::NAN);
    assert_eq!(d.select(c1, x, y), x);
    assert_eq!(d.select(c0, x, y), y);
    assert_eq!(d.select(cn, x, y), x);
    let cond = d.cmp(Cmp::Lt, x, y);
    assert_eq!(d.select(cond, x, x), x);
    {
        let r = d.select(cond, x, y);
        assert!(matches!(d.op(r), Op::Select(_, _, _)));
    }
}

/// Builds each op over variables and over constants with the same values; the live
/// evaluation of the variable form must equal the bits the folded constant carries.
#[test]
fn constant_folding_matches_eval() {
    let vals = [
        (0.7, -2.3),
        (-1.5, 0.25),
        (3.0, 3.0),
        (0.0, 5.0),
        (1.0, -4.0),
        (2.0, f64::INFINITY),
    ];
    let f1 = [
        Fn1::Sin,
        Fn1::Cos,
        Fn1::Tan,
        Fn1::Asin,
        Fn1::Acos,
        Fn1::Atan,
        Fn1::Sinh,
        Fn1::Cosh,
        Fn1::Tanh,
        Fn1::Exp,
        Fn1::Ln,
        Fn1::Log10,
        Fn1::Sqrt,
        Fn1::Abs,
        Fn1::Erf,
        Fn1::Erfc,
        Fn1::Lgamma,
        Fn1::Floor,
        Fn1::Ceil,
        Fn1::Heav,
        Fn1::Sign,
    ];
    let f2 = [
        Fn2::Atan2,
        Fn2::Pow,
        Fn2::Mod,
        Fn2::BesselJ,
        Fn2::BesselY,
        Fn2::Polygamma,
        Fn2::Min,
        Fn2::Max,
    ];
    let cmps = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq];
    for (av, bv) in vals {
        let mut d = Dag::new(10_000);
        let (x, y) = (d.var(0), d.var(1));
        let (a, b) = (d.konst(av), d.konst(bv));
        let t = d.add_table(vec![3.0, 0.0, 1.0, 1.0, 3.0, 2.0]);
        let mut pairs: Vec<(NodeId, NodeId)> = vec![
            (d.add(x, y), d.add(a, b)),
            (d.sub(x, y), d.sub(a, b)),
            (d.mul(x, y), d.mul(a, b)),
            (d.div(x, y), d.div(a, b)),
            (d.neg(x), d.neg(a)),
            (d.not(x), d.not(a)),
            (d.and(x, y), d.and(a, b)),
            (d.or(x, y), d.or(a, b)),
            (d.table(t, x), d.table(t, a)),
            (d.table_slope(t, x), d.table_slope(t, a)),
        ];
        for n in -16..=16 {
            pairs.push((d.powi(x, n), d.powi(a, n)));
        }
        for f in f1 {
            pairs.push((d.call1(f, x), d.call1(f, a)));
        }
        for f in f2 {
            pairs.push((d.call2(f, x, y), d.call2(f, a, b)));
        }
        for c in cmps {
            pairs.push((d.cmp(c, x, y), d.cmp(c, a, b)));
        }
        for (live, folded) in pairs {
            let bits = cval(&d, folded).unwrap_or_else(|| panic!("not folded: {:?}", d.op(folded)));
            let got = run(&d, live, &[av, bv], &[], &[]);
            assert_eq!(got.to_bits(), bits, "{:?} at ({av}, {bv})", d.op(live));
        }
    }
}

#[test]
fn cap_flag() {
    let mut d = Dag::new(3);
    assert!(!d.exceeded());
    let a = d.var(0);
    let b = d.var(1);
    let c = d.var(2);
    assert!(!d.exceeded());
    let e = d.add(a, b);
    assert!(d.exceeded());
    assert_eq!(d.len(), 4);
    let _ = (c, e);
    assert_eq!(pc_ir::dag::MAX_NODES, 1_000_000);
}

#[test]
fn eval_leaves_and_semantics() {
    let mut d = Dag::new(100);
    let x = d.var(0);
    let p = d.param(1);
    let v = d.dir(0);
    let s = d.add(x, p);
    let s = d.mul(s, v);
    assert_eq!(run(&d, s, &[2.0], &[0.0, 3.0], &[4.0]), 20.0);
    let nan = d.konst(f64::NAN);
    let a = d.and(nan, x);
    assert_eq!(run(&d, a, &[1.0], &[], &[]), 1.0);
    let n = d.not(x);
    assert_eq!(run(&d, n, &[0.0], &[], &[]), 1.0);
    let c = d.cmp(Cmp::Lt, nan, x);
    assert_eq!(run(&d, c, &[1.0], &[], &[]), 0.0);
    let mn = d.call2(Fn2::Min, x, p);
    assert_eq!(run(&d, mn, &[2.0], &[0.0, 1.0], &[]), 1.0);
    // multiple outputs
    let mut out = [0.0; 2];
    eval(&d, &[x, p], &[5.0], &[0.0, 6.0], &[], &mut out);
    assert_eq!(out, [5.0, 6.0]);
}

/// Reference for the documented PowI order: squarings s[k] = x^(2^k), then the set bits
/// of |n| multiplied low to high into an accumulator seeded by the lowest set bit.
fn powi_ref(x: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let m = n.unsigned_abs();
    let mut sq = vec![x];
    for k in 1..5 {
        let prev = sq[k - 1];
        sq.push(prev * prev);
    }
    let mut acc: Option<f64> = None;
    for (k, s) in sq.iter().enumerate() {
        if m >> k & 1 == 1 {
            acc = Some(match acc {
                None => *s,
                Some(a) => a * *s,
            });
        }
    }
    let r = acc.unwrap();
    if n < 0 { 1.0 / r } else { r }
}

#[test]
fn powi_order_matches_reference() {
    let mut d = Dag::new(1000);
    let x = d.var(0);
    for n in -16..=16 {
        let id = d.powi(x, n);
        for xv in [1.1f64, -0.93, 2.4417, 0.3333333333333333, 1.0000001] {
            let got = run(&d, id, &[xv], &[], &[]);
            assert_eq!(got.to_bits(), powi_ref(xv, n).to_bits(), "n={n} x={xv}");
        }
    }
}
