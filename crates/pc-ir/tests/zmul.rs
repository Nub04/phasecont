//! Zero tangents against infinite partials.
//!
//! A tangent that is zero at run time, times a partial that is infinite or NaN at a finite
//! input, must contribute +0, as it does in the fallback interpreter (`Dual::term` skips a
//! zero tangent). `Op::ZMul(t, p)` is that product: +0.0 when `t == 0.0`, else `t * p`.

use pc_ir::check::check_model;
use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use pc_ir::diff::{Wrt, d};
use pc_ir::eval::{eval, eval_entry};
use pc_ir::{EntryKind, Fallback, IrModel, Slot, lower};
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

#[test]
fn zdiv_is_plus_zero_for_a_zero_numerator_and_a_quotient_otherwise() {
    let mut g = Dag::new(100);
    let (n, dd) = (g.var(0), g.var(1));
    let z = g.zdiv(n, dd);
    assert!(matches!(g.op(z), Op::ZDiv(a, b) if *a == n && *b == dd));
    for &nv in &SPECIALS {
        for &dv in &SPECIALS {
            let want = if nv == 0.0 { 0.0 } else { nv / dv };
            let got = run(&g, z, &[nv, dv]);
            assert!(
                same(got, want),
                "zdiv({nv:?}, {dv:?}) = {got:?} ({:016x}), want {want:?}",
                got.to_bits()
            );
        }
    }
}

#[test]
fn a_zero_in_the_denominator_is_not_guarded_for_a_nonzero_numerator() {
    let mut g = Dag::new(100);
    let (n, dd) = (g.var(0), g.var(1));
    let z = g.zdiv(n, dd);
    assert_eq!(run(&g, z, &[1.0, 0.0]), f64::INFINITY);
    assert_eq!(run(&g, z, &[-1.0, -0.0]), f64::INFINITY);
    assert!(run(&g, z, &[f64::NAN, 0.0]).is_nan());
}

#[test]
fn zdiv_folds() {
    let mut g = Dag::new(100);
    let (n, dd) = (g.var(0), g.var(1));
    let plus = g.konst(0.0);
    // A static zero numerator of either sign gives +0, whatever the denominator is.
    let (pz, nz) = (g.konst(0.0), g.konst(-0.0));
    assert_eq!(g.zdiv(pz, dd), plus);
    assert_eq!(g.zdiv(nz, dd), plus);
    // Both operands constant: evaluated with the same rule.
    let (zero, one, two, six) = (g.konst(-0.0), g.konst(1.0), g.konst(2.0), g.konst(6.0));
    assert_eq!(g.zdiv(zero, zero), plus);
    let three = g.konst(3.0);
    assert_eq!(g.zdiv(six, two), three);
    let ninf = g.konst(f64::NEG_INFINITY);
    assert_eq!(g.zdiv(one, zero), ninf);
    // A nonconstant numerator stays a ZDiv node, whatever the denominator, including 1 and
    // a finite constant: `-0.0 / c` is `-0.0` but `ZDiv(-0.0, c)` is `+0.0`.
    for den in [dd, one, two, zero] {
        let z = g.zdiv(n, den);
        assert!(matches!(g.op(z), Op::ZDiv(..)), "{den:?}");
    }
    // ZDiv and Div never merge.
    let q = g.div(n, dd);
    let z = g.zdiv(n, dd);
    assert_ne!(q, z);
    assert_eq!(g.zdiv(n, dd), z);
    assert_eq!(g.div(n, dd), q);
}

#[test]
fn the_derivative_of_zdiv_guards_each_term() {
    let mut g = Dag::new(100);
    let (n, dd) = (g.var(0), g.var(1));
    let z = g.zdiv(n, dd);
    let mut memo = HashMap::new();
    // d/du0: ZDiv(1, d) is 1 / d; the term with dd = 0 is dropped.
    let d0 = d(&mut g, &mut memo, z, Wrt::U(0));
    let one = g.konst(1.0);
    let want = g.zdiv(one, dd);
    assert_eq!(d0, want);
    // d/du1: ZMul(1, -(q / d)), the term with dn = 0 is dropped.
    let d1 = d(&mut g, &mut memo, z, Wrt::U(1));
    assert_eq!(run(&g, d1, &[6.0, 3.0]), -(6.0 / 3.0 / 3.0));
    // A zero numerator and a zero denominator: the derivative in the numerator direction
    // is +0 (a zero tangent), where the plain quotient is 1 / 0 = inf. Here the tangent
    // is the constant 1, so the term is 1 / d, which is not guarded.
    assert_eq!(run(&g, d0, &[0.0, 0.0]), f64::INFINITY);
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
        ("exp of a negative power", format!("exp(-{m}^(-2))")),
        ("exp of a reciprocal", format!("exp(-1/{m})")),
        ("exp of a quotient", format!("exp(-(1+{m})/{m})")),
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
        which in 0usize..11,
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
            5 => format!("exp(-{m}^(-2))"),
            6 => format!("atan2({m},{m})+sqrt({m})*y"),
            7 => format!("{m}^q"),
            8 => format!("exp(-1/{m})"),
            9 => format!("exp(-(1+{m})/{m})*y"),
            _ => format!("exp(log10({m}))+sqrt({m})"),
        };
        let src = format!("par a=1, q=0.5\ninit x=0, y=0\nx'=-x+{rhs}\ny'=-y+x*y\ndone\n");
        let (mut ir, fb) = build(&src);
        let report = check_model(&mut ir, &fb, 3, seed);
        let bad: Vec<_> = report.failures().map(|e| e.label.clone()).collect();
        prop_assert!(bad.is_empty(), "{rhs}: {bad:?}");
    }
}

// ---- Product terms against a non-finite other factor ----------------------------------

/// A clamp whose tangent is zero at run time: `max(1-x, 2)` is 2 for every x above -1.
const CL: &str = "max(1-x,2)";

/// The same clamp on y.
const CLY: &str = "max(1-y,2)";

/// Right-hand sides inside `exp`, so the model value is finite (`exp(-inf)` is 0), with a
/// clamped factor times a factor that is infinite at `m = 0`. The product rule gives
/// `tangent(CL) * other + CL * tangent(other)`: a plain product has `0 * inf = NaN` where the
/// fallback skips the zero tangent. Each form has both operand orders, because the
/// guard is on the tangent and the tangent is the first or the second factor.
fn product_forms() -> Vec<(&'static str, String)> {
    let m = M;
    vec![
        ("clamp times a reciprocal", format!("exp(-{CL}*(1/{m}))")),
        ("reciprocal times a clamp", format!("exp(-(1/{m})*{CL})")),
        ("clamp times a log", format!("exp({CL}*ln({m}))")),
        ("log times a clamp", format!("exp(ln({m})*{CL})")),
        (
            "clamp on y times a reciprocal",
            format!("exp(-{CLY}*(1/{m}))"),
        ),
        (
            "two clamps and a quotient",
            format!("exp(-{CL}*{CLY}*(1/{m}))"),
        ),
        (
            "clamp times a log of a power",
            format!("exp({CL}*ln({m}^3))"),
        ),
        (
            "clamp times a negative power",
            format!("exp(-{CL}*{m}^(-1))"),
        ),
    ]
}

fn product_kinds() -> Vec<(&'static str, EntryKind)> {
    use Slot::{Full, P, U};
    vec![
        ("JacU", EntryKind::JacU),
        ("JacP", EntryKind::JacP),
        ("Dir[U]", EntryKind::Dir(vec![U])),
        ("Dir[P]", EntryKind::Dir(vec![P])),
        ("Dir[Full]", EntryKind::Dir(vec![Full])),
        ("Dir[U,U]", EntryKind::Dir(vec![U, U])),
        ("Dir[U,P]", EntryKind::Dir(vec![U, P])),
        ("Dir[U,U,U]", EntryKind::Dir(vec![U, U, U])),
        ("Dir[Full,U,U]", EntryKind::Dir(vec![Full, U, U])),
        ("Dir[U,U,U,U]", EntryKind::Dir(vec![U, U, U, U])),
    ]
}

/// States x and y with the clamp of `CL` and `CLY` active (|x|, |y| < 1) and `m = 0`
/// (a - x - 10 < 0), then the parameter a and t.
const PRODUCT_POINTS: [([f64; 2], [f64; 2]); 6] = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.3, 0.2], [1.0, 0.0]),
    ([-0.4, 0.5], [0.95, 0.0]),
    ([0.5, -0.3], [1.1, 0.0]),
    ([-0.0, 0.0], [1.0, 0.0]),
    ([0.1, -0.6], [0.5, 0.0]),
];

/// Directions with zero entries at run time, as the derivative of a clamp has.
const PRODUCT_DIRS: [[f64; 8]; 3] = [
    [1.0, 0.0, -0.5, 0.25, 2.0, 1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0, 0.0, -1.0, 0.5, 2.0, 0.0],
    [0.75, -1.5, 0.0, 0.0, 1.0, -0.25, 1.0, 1.0],
];

fn packed(kind: &EntryKind, n: usize, n_p: usize) -> usize {
    match kind {
        EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
        _ => 0,
    }
}

#[test]
fn a_clamped_factor_times_a_non_finite_one_gives_the_fallback_derivatives_bit_for_bit() {
    let mut bad = Vec::new();
    let mut compared = 0usize;
    for (what, rhs) in product_forms() {
        let src = format!("par a=1\ninit x=0, y=0\nx'=-x+{rhs}\ny'=-y+x\ndone\n");
        let (mut ir, fb) = build(&src);
        let (n, n_p) = (ir.n(), ir.n_p());
        // The model value is finite at every point: the infinite factor sits inside exp.
        let ef = ir.entry(&EntryKind::F).expect("F");
        for (x, a) in PRODUCT_POINTS {
            let mut f = [f64::MAX; 2];
            eval_entry(&ir.dag, &ef, &x, &a, &[], &mut f);
            assert!(
                f.iter().all(|v| v.is_finite()),
                "{what}: F is {f:?} at {x:?}"
            );
        }
        for (name, kind) in product_kinds() {
            let entry = ir.entry(&kind).expect("entry");
            let v_len = packed(&kind, n, n_p);
            for (x, a) in PRODUCT_POINTS {
                for (di, dir) in PRODUCT_DIRS.iter().enumerate() {
                    if v_len == 0 && di > 0 {
                        continue;
                    }
                    let v = &dir[..v_len];
                    let mut got = vec![f64::MAX; entry.outputs.len()];
                    eval_entry(&ir.dag, &entry, &x, &a, v, &mut got);
                    let mut want = vec![f64::MAX; entry.outputs.len()];
                    fb.eval(&kind, &x, &a, v, &mut want);
                    for i in 0..got.len() {
                        compared += 1;
                        if !same(got[i], want[i]) {
                            bad.push(format!(
                                "{what} {name} x={x:?} a={:?} dir {di} out {i}: dag {:?} \
                                 ({:016x}) fallback {:?} ({:016x})",
                                a[0],
                                got[i],
                                got[i].to_bits(),
                                want[i],
                                want[i].to_bits()
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 2_000, "only {compared} values compared");
    assert!(
        bad.is_empty(),
        "{} of {compared} values differ, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(10)]
    );
}

// ---- The flag that decides where a product term is guarded -----------------------------

#[test]
fn the_nonfinite_flag_follows_the_value_of_a_node() {
    let mut g = Dag::new(1000);
    let (x, y) = (g.var(0), g.var(1));
    let (one, two, zero) = (g.konst(1.0), g.konst(2.0), g.konst(0.0));
    let (inf, nan, three) = (g.konst(f64::INFINITY), g.konst(f64::NAN), g.konst(3.0));
    let p = g.param(0);
    let mut yes = Vec::new();
    let mut no = Vec::new();
    // Inputs, finite constants and finite arithmetic.
    no.extend([
        x,
        p,
        one,
        zero,
        g.neg(x),
        g.mul(x, y),
        g.add(x, y),
        g.sub(x, two),
    ]);
    no.push(g.div(x, two));
    let xy = g.mul(x, y);
    no.push(g.div(xy, three));
    // A non-finite constant, a division by anything else, a negative power.
    yes.extend([
        inf,
        nan,
        g.div(one, x),
        g.div(one, zero),
        g.div(x, inf),
        g.div(x, nan),
    ]);
    yes.push(g.powi(x, -2));
    // Functions with a pole, a branch point or a boundary.
    for f in [
        Fn1::Ln,
        Fn1::Log10,
        Fn1::Sqrt,
        Fn1::Asin,
        Fn1::Acos,
        Fn1::Lgamma,
    ] {
        yes.push(g.call1(f, x));
    }
    // Functions that are finite at every finite argument. Overflow is not tracked.
    for f in [
        Fn1::Sin,
        Fn1::Cos,
        Fn1::Tan,
        Fn1::Atan,
        Fn1::Sinh,
        Fn1::Cosh,
        Fn1::Tanh,
        Fn1::Exp,
        Fn1::Abs,
        Fn1::Erf,
        Fn1::Erfc,
        Fn1::Floor,
        Fn1::Ceil,
        Fn1::Heav,
        Fn1::Sign,
    ] {
        no.push(g.call1(f, x));
    }
    no.push(g.powi(x, 3));
    no.push(g.call2(Fn2::Max, x, y));
    no.push(g.call2(Fn2::Min, x, two));
    no.push(g.call2(Fn2::Pow, x, three));
    yes.push(g.call2(Fn2::Pow, x, y));
    let half = g.konst(0.5);
    yes.push(g.call2(Fn2::Pow, x, half));
    yes.push(g.call2(Fn2::Atan2, x, y));
    // A comparison is 0 or 1 whatever its operands are.
    let lnx = g.call1(Fn1::Ln, x);
    no.push(g.cmp(Cmp::Lt, lnx, y));
    // The flag spreads to every node that reads a flagged one, except through a comparison
    // and a finite function of the four that are finite at a finite argument.
    let rec = g.div(one, x);
    yes.extend([
        g.neg(rec),
        g.mul(y, rec),
        g.add(rec, y),
        g.sub(y, rec),
        g.call1(Fn1::Sin, rec),
        g.call1(Fn1::Floor, rec),
        g.powi(rec, 3),
        g.call2(Fn2::Max, y, rec),
        g.div(rec, two),
    ]);
    let c = g.cmp(Cmp::Gt, x, y);
    yes.push(g.select(c, rec, y));
    no.push(g.select(c, x, y));
    for id in yes {
        assert!(g.may_be_nonfinite(id), "{:?} should be flagged", g.op(id));
    }
    for id in no {
        assert!(
            !g.may_be_nonfinite(id),
            "{:?} should not be flagged",
            g.op(id)
        );
    }
    // An id from another DAG is out of range and is flagged.
    assert!(g.may_be_nonfinite(NodeId(u32::MAX)));
}

#[test]
fn the_flag_follows_a_checkpoint_and_a_restore() {
    let mut g = Dag::new(1000);
    let x = g.var(0);
    let cp = g.checkpoint();
    let ln = g.call1(Fn1::Ln, x);
    assert!(g.may_be_nonfinite(ln));
    g.restore(&cp);
    // The next node takes the id of the dropped one, and gets its own flag.
    let sin = g.call1(Fn1::Sin, x);
    assert_eq!(sin, ln);
    assert!(!g.may_be_nonfinite(sin));
    assert!(g.may_be_nonfinite(NodeId(sin.0 + 1)));
}

fn guard_nodes(ir: &IrModel) -> usize {
    (0..ir.dag.len())
        .filter(|&i| matches!(ir.dag.op(NodeId(i as u32)), Op::ZMul(..) | Op::ZDiv(..)))
        .count()
}

/// Models with no division, singular function or non-finite constant get no guard in any
/// derivative entry: the product terms stay plain products. `exp`, `sin`, an integer power
/// and a clamp do not set the flag.
#[test]
fn models_without_a_pole_get_no_guard_nodes() {
    let models = [
        "x'=-x^3+a*x*y+y*max(1-x,2)\ny'=x*y-min(x,0)*max(y,0)",
        "x'=a*x*(1-x)-x*y*max(x,0)\ny'=-y+x^2*y",
        "x'=exp(-x)*sin(y)*max(1-x,2)+a\ny'=tanh(x)*cos(y)-y^4",
        "x'=abs(x)*y-flr(y)*x\ny'=heav(x-y)*x*y+sign(y)",
    ];
    use Slot::{Full, P, U};
    let kinds = [
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![U]),
        EntryKind::Dir(vec![U, U]),
        EntryKind::Dir(vec![Full, U, U]),
        EntryKind::Dir(vec![U, P, U, U]),
    ];
    for body in models {
        let (mut ir, _) = build(&format!("par a=1\ninit x=0, y=0\n{body}\ndone\n"));
        for k in &kinds {
            ir.entry(k).expect("entry");
        }
        assert_eq!(guard_nodes(&ir), 0, "{body}");
    }
    // A reciprocal of a state does set the flag: the same products get a guard.
    let (mut ir, _) = build("par a=1\ninit x=0, y=0\nx'=y*max(1-x,2)/x\ny'=-y\ndone\n");
    for k in &kinds {
        ir.entry(k).expect("entry");
    }
    assert!(guard_nodes(&ir) > 0);
}
