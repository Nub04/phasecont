//! Zero tangents against infinite partials.
//!
//! A tangent that is zero at run time, times a partial that is infinite or NaN at a finite
//! input, must contribute +0, as it does in the fallback interpreter (`Dual::term` skips a
//! zero tangent). `Op::ZMul(t, p)` is that product: +0.0 when `t == 0.0`, else `t * p`.

use pc_ir::check::check_model;
use pc_ir::dag::{Dag, NodeId, Op};
use pc_ir::diff::{Wrt, d};
use pc_ir::eval::eval;
use pc_ir::{Fallback, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use proptest::prelude::*;
use std::collections::HashMap;

const SPECIALS: [f64; 10] = [
    0.0,
    -0.0,
    1.0,
    -2.5,
    1e-310,
    1e300,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MIN_POSITIVE,
];

fn run(d: &Dag, id: NodeId, x: &[f64]) -> f64 {
    let mut out = [f64::MAX];
    eval(d, &[id], x, &[0.0], &[], &mut out);
    out[0]
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
fn zmul_is_plus_zero_for_a_zero_first_operand_and_a_product_otherwise() {
    let mut g = Dag::new(100);
    let (t, p) = (g.var(0), g.var(1));
    let z = g.zmul(t, p);
    assert!(matches!(g.op(z), Op::ZMul(a, b) if *a == t && *b == p));
    for &tv in &SPECIALS {
        for &pv in &SPECIALS {
            let want = if tv == 0.0 { 0.0 } else { tv * pv };
            let got = run(&g, z, &[tv, pv]);
            assert!(
                same(got, want),
                "zmul({tv:?}, {pv:?}) = {got:?} ({:016x}), want {want:?}",
                got.to_bits()
            );
        }
    }
}

#[test]
fn a_zero_in_the_second_operand_is_not_guarded() {
    let mut g = Dag::new(100);
    let (t, p) = (g.var(0), g.var(1));
    let z = g.zmul(t, p);
    assert!(run(&g, z, &[f64::INFINITY, 0.0]).is_nan());
    assert!(run(&g, z, &[f64::NAN, 0.0]).is_nan());
}

#[test]
fn zmul_folds() {
    let mut g = Dag::new(100);
    let p = g.var(1);
    // A static zero of either sign gives the +0 constant, whatever the partial is.
    let (pz, nz) = (g.konst(0.0), g.konst(-0.0));
    let plus = g.konst(0.0);
    assert_eq!(g.zmul(pz, p), plus);
    assert_eq!(g.zmul(nz, p), plus);
    // Both operands constant: the node is evaluated with the same rule.
    let (c0, inf, two, three) = (
        g.konst(-0.0),
        g.konst(f64::INFINITY),
        g.konst(2.0),
        g.konst(3.0),
    );
    assert_eq!(g.zmul(c0, inf), plus);
    let six = g.konst(6.0);
    assert_eq!(g.zmul(two, three), six);
    let nan = g.konst(f64::NAN);
    assert_eq!(g.zmul(nan, three), nan);
    // A nonzero finite constant tangent is an ordinary product.
    let m = g.mul(two, p);
    assert_eq!(g.zmul(two, p), m);
    assert!(matches!(g.op(m), Op::Mul(..)));
    // A static zero factor folds to +0, as `x * 0` does in `Dag::mul`. Without it, a
    // sparse component list would keep a node per dead term and the entries grow.
    let t = g.var(0);
    let (p_zero, p_nzero) = (g.konst(0.0), g.konst(-0.0));
    assert_eq!(g.zmul(t, p_zero), plus);
    assert_eq!(g.zmul(t, p_nzero), plus);
    // A nonconstant tangent, and an infinite or NaN constant one, stay ZMul nodes.
    let z = g.zmul(t, p);
    assert!(matches!(g.op(z), Op::ZMul(..)));
    let z = g.zmul(inf, p);
    assert!(matches!(g.op(z), Op::ZMul(..)));
    let z = g.zmul(nan, p);
    assert!(matches!(g.op(z), Op::ZMul(..)));
}

#[test]
fn zmul_and_mul_are_different_nodes() {
    let mut g = Dag::new(100);
    let (t, p) = (g.var(0), g.var(1));
    let z = g.zmul(t, p);
    let m = g.mul(t, p);
    assert_ne!(z, m);
    assert_eq!(g.zmul(t, p), z);
    assert_eq!(g.mul(t, p), m);
}

#[test]
fn the_derivative_of_zmul_guards_each_term_and_drops_static_zeros() {
    let mut g = Dag::new(100);
    let (t, p) = (g.var(0), g.var(1));
    let z = g.zmul(t, p);
    let mut memo = HashMap::new();
    // d/du0: ZMul(1, p) is p; the term with dp = 0 is dropped.
    let d0 = d(&mut g, &mut memo, z, Wrt::U(0));
    assert_eq!(d0, p);
    // d/du1: the term with dt = 0 is dropped; the other is ZMul(t, 1).
    let d1 = d(&mut g, &mut memo, z, Wrt::U(1));
    let one = g.konst(1.0);
    let want = g.zmul(t, one);
    assert_eq!(d1, want);
    assert!(same(run(&g, d1, &[-0.0, 3.0]), 0.0));
    assert_eq!(run(&g, d1, &[-0.0, 3.0]).to_bits(), 0.0f64.to_bits());
    // Both operands depend on u0: p = t * t. The product rule gives
    // ZMul(1, t*t) + ZMul(t, dp), and a zero t with an infinite dp stays 0.
    let q = g.call1(pc_ir::dag::Fn1::Sqrt, t);
    let zq = g.zmul(t, q);
    let dq = d(&mut g, &mut memo, zq, Wrt::U(0));
    // At t = 0: sqrt'(0) is infinite, but the tangent of t is the constant 1, so
    // ZMul(t, dq_inner) = 0 and ZMul(1, sqrt(0)) = 0.
    assert_eq!(run(&g, dq, &[0.0, 0.0]).to_bits(), 0.0f64.to_bits());
}

fn build(src: &str) -> (pc_ir::IrModel, Fallback) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    let fb = Fallback::new(&l.expanded, &l.info, &ir);
    (ir, fb)
}

/// `m` is 0 at every point the check draws (x and y within 0.5 of 0, a within 10% of 1).
const M: &str = "max(a-x-10,0)";

/// Right-hand sides whose value is finite and whose argument sits on a pole, branch point or
/// domain boundary of the outer function, with a zero tangent at run time.
fn templates() -> Vec<(&'static str, String)> {
    let m = M;
    vec![
        ("sqrt", format!("sqrt({m})")),
        ("fractional power", format!("{m}^0.5")),
        (
            "fractional powers 1.5 and 0.25",
            format!("{m}^1.5+{m}^0.25"),
        ),
        ("power with a parameter exponent", format!("{m}^q")),
        ("asin at 1", format!("asin(1-{m})")),
        ("acos at 1", format!("acos(1-{m})")),
        ("exp of ln", format!("exp(ln({m}))")),
        ("exp of log10", format!("exp(log10({m}))")),
        ("exp of lgamma", format!("exp(-lgamma({m}))")),
        ("exp of a reciprocal", format!("exp(-1/{m})")),
        ("exp of a negative power", format!("exp(-{m}^(-2))")),
        ("atan2 at the origin", format!("atan2({m},{m})")),
        ("sqrt of a product", format!("sqrt({m}*{m})")),
        (
            "sqrt of heav times u",
            "sqrt(heav(a-x-10)*(a-x-10))".to_string(),
        ),
    ]
}

#[test]
fn clamped_arguments_give_the_fallback_derivatives() {
    for (what, rhs) in templates() {
        let src = format!("par a=1, q=0.5\ninit x=0, y=0\nx'=-x+{rhs}\ny'=-y+x\ndone\n");
        let (mut ir, fb) = build(&src);
        let report = check_model(&mut ir, &fb, 4, 3);
        let bad: Vec<_> = report
            .failures()
            .map(|e| format!("{} (ratio {:e}, note {:?})", e.label, e.ratio, e.note))
            .collect();
        assert!(bad.is_empty(), "{what}: {bad:?}");
    }
}

/// The same check with the singular function inside the second equation and the clamp
/// coefficients drawn at random.
fn clamp(c1: i32, c2: i32) -> String {
    // |c1 x + c2 y| <= 4 on the sample box, so a - 12 + that stays below 0.
    format!("max(a-12+({c1})*x+({c2})*y,0)")
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 40, ..ProptestConfig::default() })]

    #[test]
    fn random_clamps_inside_singular_functions_give_the_fallback_derivatives(
        which in 0usize..9,
        c1 in -2i32..=2,
        c2 in -2i32..=2,
        seed in 1u64..1000,
    ) {
        let m = clamp(c1, c2);
        let rhs = match which {
            0 => format!("sqrt({m})"),
            1 => format!("{m}^0.5"),
            2 => format!("asin(1-{m})"),
            3 => format!("acos(1-{m})"),
            4 => format!("exp(ln({m}))"),
            5 => format!("exp(-1/{m})"),
            6 => format!("atan2({m},{m})+sqrt({m})*y"),
            7 => format!("{m}^q"),
            _ => format!("exp(log10({m}))+sqrt({m})"),
        };
        let src = format!("par a=1, q=0.5\ninit x=0, y=0\nx'=-x+{rhs}\ny'=-y+x*y\ndone\n");
        let (mut ir, fb) = build(&src);
        let report = check_model(&mut ir, &fb, 3, seed);
        let bad: Vec<_> = report.failures().map(|e| e.label.clone()).collect();
        prop_assert!(bad.is_empty(), "{rhs}: {bad:?}");
    }
}
