use super::*;
use crate::dag::MAX_NODES;

/// Value and running error bound of `out` at `x = (x0, x1, x2)`.
fn run(dag: &Dag, out: NodeId, x: [f64; 3]) -> (f64, f64) {
    let (mut v, mut e) = ([f64::NAN], [f64::NAN]);
    eval_err(dag, &[out], &x, &[], &[], &mut v, &mut e);
    (v[0], e[0])
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
}

#[test]
fn values_match_eval_and_leaves_are_exact() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b) = (d.var(0), d.var(1));
    let s = d.call1(Fn1::Sin, a);
    let m = d.mul(s, b);
    let q = d.div(m, a);
    let mut want = [0.0];
    eval(&d, &[q], &[0.7, 1.3, 0.0], &[], &[], &mut want);
    let (got, e) = run(&d, q, [0.7, 1.3, 0.0]);
    assert_eq!(got.to_bits(), want[0].to_bits());
    assert!(e > 0.0 && e.is_finite());
    assert_eq!(run(&d, a, [0.7, 0.0, 0.0]).1, 0.0);
    let k = d.konst(2.5);
    assert_eq!(run(&d, k, [0.0; 3]).1, 0.0);
}

#[test]
fn cancellation_in_a_sum_keeps_the_size_of_the_terms() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b, c) = (d.var(0), d.var(1), d.var(2));
    let t = d.add(a, b);
    let r = d.sub(t, c);
    let (x0, x1, x2) = (0.1, 0.2, 0.3);
    let (v, e) = run(&d, r, [x0, x1, x2]);
    let tv = x0 + x1;
    // E(t) = |t|; E(r) = E(t) + 0 + |r|. The result is tiny and the bound is not.
    assert!(near(e, tv.abs() + v.abs()), "{e:e}");
    assert!(v.abs() < 1e-16 && e > 0.3);
    // A sum of exact inputs: E is just the size of the result.
    let (v, e) = run(&d, t, [x0, x1, x2]);
    assert_eq!(e, v.abs());
}

#[test]
fn product_quotient_and_negation() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b, c) = (d.var(0), d.var(1), d.var(2));
    let ab = d.mul(a, b);
    let abc = d.mul(ab, c);
    let (v, e) = run(&d, abc, [2.0, 3.0, 5.0]);
    // E(ab) = |ab| = 6; E(abc) = |c| E(ab) + |abc| = 30 + 30.
    assert_eq!((v, e), (30.0, 60.0));
    let q = d.div(ab, c);
    let (v, e) = run(&d, q, [2.0, 3.0, 5.0]);
    // E = (E(ab) + |q| * 0) / |c| + |q|.
    assert!(near(e, 6.0 / 5.0 + v.abs()), "{e:e}");
    let n = d.neg(ab);
    let (v, e) = run(&d, n, [2.0, 3.0, 5.0]);
    assert_eq!((v, e), (-6.0, 6.0));
}

#[test]
fn integer_powers() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b) = (d.var(0), d.var(1));
    // x^3: two roundings; x^-2: one rounding and the reciprocal.
    let p3 = d.powi(a, 3);
    let (v, e) = run(&d, p3, [2.0, 0.0, 0.0]);
    assert_eq!((v, e), (8.0, 16.0));
    let pm2 = d.powi(a, -2);
    let (v, e) = run(&d, pm2, [2.0, 0.0, 0.0]);
    assert_eq!((v, e), (0.25, 0.5));
    // (a b)^3 with E(ab) = |ab|: 3 |r| E/|ab| + 2 |r| = 5 |r|.
    let ab = d.mul(a, b);
    let q = d.powi(ab, 3);
    let (v, e) = run(&d, q, [2.0, 1.5, 0.0]);
    assert!(near(e, 5.0 * v.abs()), "{e:e}");
    // The roundings compound: x^n has (|n| - 1) |r|, so x^5 has 4 |r| and x^8 has 7 |r|.
    let p5 = d.powi(a, 5);
    assert_eq!(run(&d, p5, [2.0, 0.0, 0.0]), (32.0, 4.0 * 32.0));
    let p8 = d.powi(a, 8);
    assert_eq!(run(&d, p8, [2.0, 0.0, 0.0]), (256.0, 7.0 * 256.0));
    // x^-4: three roundings and the reciprocal, 4 |r| in all.
    let pm4 = d.powi(a, -4);
    assert_eq!(run(&d, pm4, [2.0, 0.0, 0.0]), (0.0625, 4.0 * 0.0625));
    let p1 = d.powi(a, 1);
    assert_eq!(run(&d, p1, [2.0, 0.0, 0.0]), (2.0, 0.0));
    // An exact zero base has no error to scale; a zero base that carries one has no bound.
    assert_eq!(run(&d, p3, [0.0, 0.0, 0.0]).1, 0.0);
    let c = d.var(2);
    let ca = d.mul(c, a);
    let z = d.sub(ab, ca);
    let pz = d.powi(z, 3);
    let (v, e) = run(&d, pz, [2.0, 3.0, 3.0]);
    assert_eq!(v, 0.0);
    assert!(!e.is_finite(), "{e:e}");
}

#[test]
fn a_call_passes_the_error_through_the_derivative() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b) = (d.var(0), d.var(1));
    let ab = d.mul(a, b);
    let ex = d.call1(Fn1::Exp, ab);
    let (x0, x1) = (0.5, 1.5);
    let (v, e) = run(&d, ex, [x0, x1, 0.0]);
    let arg = x0 * x1;
    // |exp'(arg)| E(ab) + 2 |r| with E(ab) = |arg|.
    assert!(near(e, v * arg + 2.0 * v), "{e:e}");
    // An exact argument adds only the 2 |r| of the call.
    let ea = d.call1(Fn1::Exp, a);
    let (v, e) = run(&d, ea, [x0, 0.0, 0.0]);
    assert!(near(e, 2.0 * v));
    // Two arguments: atan2(a b, a) uses both partials.
    let at = d.call2(Fn2::Atan2, ab, a);
    let (v, e) = run(&d, at, [x0, x1, 0.0]);
    let (y, xx) = (arg, x0);
    let den = y * y + xx * xx;
    let want = (xx / den).abs() * arg + 2.0 * v.abs();
    assert!(near(e, want), "{e:e} {want:e}");
}

#[test]
fn select_takes_the_bound_of_the_taken_branch() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b, c) = (d.var(0), d.var(1), d.var(2));
    let ab = d.mul(a, b);
    let cond = d.cmp(Cmp::Lt, a, c);
    let sel = d.select(cond, ab, a);
    // a < c picks the product, whose bound is |ab|; otherwise the exact leaf.
    let (v, e) = run(&d, sel, [1.0, 3.0, 2.0]);
    assert_eq!((v, e), (3.0, 3.0));
    let (v, e) = run(&d, sel, [4.0, 3.0, 2.0]);
    assert_eq!((v, e), (4.0, 0.0));
    // Comparisons and logic carry no bound.
    assert_eq!(run(&d, cond, [1.0, 3.0, 2.0]).1, 0.0);
}

#[test]
fn tables_use_the_slope() {
    let mut d = Dag::new(MAX_NODES);
    // Three points 0, 1, 2 on [0, 2] with values 0, 4, 6: slopes 4 and 2.
    let id = d.add_table(vec![3.0, 0.0, 2.0, 0.0, 4.0, 6.0]);
    let (a, b) = (d.var(0), d.var(1));
    let ab = d.mul(a, b);
    let t = d.table(id, ab);
    let (v, e) = run(&d, t, [0.5, 3.0, 0.0]);
    // ab = 1.5 is in the second segment (slope 2): |slope| E(ab) + |r|.
    assert!(near(e, 2.0 * 1.5 + v.abs()), "{v} {e}");
    let ts = d.table_slope(id, ab);
    assert_eq!(run(&d, ts, [0.5, 3.0, 0.0]).1, 0.0);
}

#[test]
fn a_non_finite_derivative_gives_a_non_finite_bound() {
    let mut d = Dag::new(MAX_NODES);
    let (a, b) = (d.var(0), d.var(1));
    // acos'(1) is infinite, and the argument 0.5 * 2 carries an error bound.
    let ab = d.mul(a, b);
    let ac = d.call1(Fn1::Acos, ab);
    let (v, e) = run(&d, ac, [0.5, 2.0, 0.0]);
    assert_eq!(v, 0.0);
    assert!(!e.is_finite(), "{e:e}");
    // An exact argument has no error to carry the infinity, so only the call's 2 |r|.
    let ac1 = d.call1(Fn1::Acos, a);
    let (_, e) = run(&d, ac1, [1.0, 0.0, 0.0]);
    assert_eq!(e, 0.0);
    // Division by zero: no finite bound.
    let q = d.div(a, b);
    let (_, e) = run(&d, q, [1.0, 0.0, 0.0]);
    assert!(!e.is_finite());
    // A missing leaf reads as NaN, and so does the bound of any sum with it.
    let s = d.add(a, b);
    let (mut v, mut e) = ([0.0], [0.0]);
    eval_err(&d, &[s], &[], &[], &[], &mut v, &mut e);
    assert!(v[0].is_nan() && e[0].is_nan());
}
