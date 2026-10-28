use super::*;
use crate::dag::{Cmp, Fn1, Fn2};
use crate::entry::EntryKind;
use crate::eval::eval;
use crate::interp::Fallback;
use crate::lower::{IrModel, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn setup(src: &str) -> (Fallback, IrModel) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    (Fallback::new(&l.expanded, &l.info, &ir), ir)
}

fn ulp(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    f64::from_bits(x.abs().to_bits() + 1) - x.abs()
}

/// Section 3 tolerance for one output vector; `want` is the reference.
fn within(got: &[f64], want: &[f64]) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("length {} vs {}", got.len(), want.len()));
    }
    let s = want
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a == b {
            continue;
        }
        let diff = (a - b).abs();
        let m = a.abs().max(b.abs());
        if diff <= 4.0 * ulp(m) || diff <= 1e-13 * s {
            continue;
        }
        return Err(format!("entry {i}: {a:?} vs {b:?} (scale {s:e})"));
    }
    Ok(())
}

fn run(ir: &IrModel, outs: &[NodeId], x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; outs.len()];
    eval(&ir.dag, outs, x, p, &[], &mut out);
    out
}

/// JacU and JacP of the DAG entries against the fallback at each `(x, p)`.
fn check_model(src: &str, points: &[(&[f64], &[f64])]) {
    let (fb, mut ir) = setup(src);
    let (n, n_p) = (ir.n(), ir.n_p());
    for (kind, len) in [(EntryKind::JacU, n * n), (EntryKind::JacP, n * n_p)] {
        let entry = match ir.entry(&kind) {
            Ok(e) => e,
            Err(e) => panic!("entry failed: {e}"),
        };
        assert_eq!(entry.outputs.len(), len);
        for (x, p) in points {
            let got = run(&ir, &entry.outputs, x, p);
            let mut want = vec![0.0; len];
            fb.eval(&kind, x, p, &[], &mut want);
            if let Err(m) = within(&got, &want) {
                panic!("{kind:?} at {x:?} {p:?}: {m}");
            }
        }
    }
}

#[test]
fn jac_builtins_smooth() {
    check_model(
        "x'=sin(x)*cos(y)+tan(y/3)+asin(x/4)+acos(y/4)+atan(x*y)+sinh(x)+cosh(y)+tanh(x*y)\n\
         y'=exp(x)*ln(1+y*y)+log10(2+x*x)+sqrt(1+x*x)+abs(x-y)+erf(x)+erfc(y)+lgamma(2+x*x)\
         +heav(x)+sign(y)+flr(x)+ceil(y)\n\
         done\n",
        &[
            (&[0.3, 0.7], &[0.0]),
            (&[-0.8, 1.1], &[0.0]),
            (&[1.7, -0.4], &[0.0]),
        ],
    );
}

#[test]
fn jac_two_argument_builtins_and_branches() {
    check_model(
        "par a=2, b=0.5\n\
         x'=atan2(y,x)+mod(x*7,a+1)+min(x,y)+max(x,b*y)+besselj(2,x)+bessely(1,abs(x)+0.5)\n\
         y'=if(x>y)then(x*a)else(y*b)+if(x<0.5&y>0)then(sin(y))else(x*x)\n\
         done\n",
        &[
            (&[0.3, 0.7], &[2.0, 0.5, 0.0]),
            (&[-0.8, 1.1], &[2.0, 0.5, 0.0]),
            (&[1.7, -0.4], &[1.3, 0.9, 0.0]),
        ],
    );
}

#[test]
fn jac_pow_forms() {
    check_model(
        "par a=2, b=0.5\n\
         x'=(1+x*x)^1.5+(1+y*y)^a+2^x+x^3+y^(-2)+(1+x*x)^(0.5+y*y)+x^20\n\
         y'=(2+x)^b*y+(3+y)**a\n\
         done\n",
        &[
            (&[0.3, 0.7], &[2.0, 0.5, 0.0]),
            (&[1.2, 1.1], &[1.5, 0.8, 0.0]),
        ],
    );
}

#[test]
fn jac_tables() {
    check_model(
        "par a=1.5\ntable h % 9 -2 2 sin(2*t)\n\
         x'=h(x)+h(y)*x\ny'=-y+a*x*h(x*y)\ndone\n",
        &[(&[0.31, 0.77], &[1.5, 0.0]), (&[-0.83, 1.13], &[1.5, 0.0])],
    );
}

#[test]
fn jac_non_autonomous_has_no_time_column() {
    let src = "par a=1, w=2\nx'=-x+a*sin(w*t)*y\ny'=x*cos(t)+a*y\ndone\n";
    check_model(src, &[(&[0.4, -0.9], &[1.0, 2.0, 0.37])]);
    let (_fb, mut ir) = setup(src);
    assert!(!ir.autonomous);
    let e = ir.entry(&EntryKind::JacP).unwrap();
    // n * n_p = 4: no column for t.
    assert_eq!(e.outputs.len(), 4);
    let got = run(&ir, &e.outputs, &[0.4, -0.9], &[1.0, 2.0, 0.37]);
    let (s, c) = ((2.0_f64 * 0.37).sin(), (2.0_f64 * 0.37).cos());
    // columns: d/da = [sin(wt) y, y], d/dw = [a cos(wt) t y, 0]
    let want = [s * -0.9, -0.9, c * 0.37 * -0.9, 0.0];
    within(&got, &want).unwrap();
}

#[test]
fn jac_hand_derived_predator_prey() {
    let (_fb, mut ir) = setup("par a=1.5, b=0.8\nx'=a*x-b*x*y\ny'=-y+b*x*y\ndone\n");
    let (x, y, a, b) = (0.6_f64, 1.9_f64, 1.5_f64, 0.8_f64);
    let ju = ir.entry(&EntryKind::JacU).unwrap();
    let got = run(&ir, &ju.outputs, &[x, y], &[a, b, 0.0]);
    within(&got, &[a - b * y, b * y, -b * x, -1.0 + b * x]).unwrap();
    let jp = ir.entry(&EntryKind::JacP).unwrap();
    let got = run(&ir, &jp.outputs, &[x, y], &[a, b, 0.0]);
    within(&got, &[x, 0.0, -x * y, x * y]).unwrap();
}

#[test]
fn entry_f_aux_cached_and_empty_dir_refused() {
    let (_fb, mut ir) = setup("x'=-x*y\ny'=x\naux s=x+y\ndone\n");
    let f = ir.entry(&EntryKind::F).unwrap();
    assert_eq!(f.outputs, ir.f);
    let aux = ir.entry(&EntryKind::Aux).unwrap();
    assert_eq!(aux.outputs, ir.aux);
    let j1 = ir.entry(&EntryKind::JacU).unwrap();
    let len = ir.dag.len();
    let j2 = ir.entry(&EntryKind::JacU).unwrap();
    assert_eq!(j1, j2);
    assert_eq!(ir.dag.len(), len);
    let e = ir.entry(&EntryKind::Dir(vec![])).unwrap_err();
    assert_eq!(e.kind, crate::IrErrorKind::Unsupported);
}

#[test]
fn entry_reports_too_large() {
    let src = "x'=sin(x)*cos(x)+x*x\ndone\n";
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let full = lower(&l.expanded, &l.info).unwrap().dag.len();
    // A cap equal to the lowered size lets F through; JacU then crosses it inside `entry`.
    let mut ir = match crate::lower::lower_with_cap(&l.expanded, &l.info, full) {
        Ok(m) => m,
        Err(e) => panic!("lowering at the exact size must pass: {e:?}"),
    };
    assert!(!ir.dag.exceeded());
    assert!(ir.entry(&EntryKind::F).is_ok());
    let e = ir.entry(&EntryKind::JacU).unwrap_err();
    assert_eq!(e.kind, crate::IrErrorKind::TooLarge);
}

// ---- rules on the DAG directly ----

fn val(dag: &Dag, id: NodeId, x: &[f64], p: &[f64]) -> f64 {
    let mut out = [0.0];
    eval(dag, &[id], x, p, &[], &mut out);
    out[0]
}

fn const_bits(dag: &Dag, id: NodeId) -> Option<u64> {
    match dag.op(id) {
        Op::Const(b) => Some(*b),
        _ => None,
    }
}

#[test]
fn leaves_and_constants() {
    let mut dag = Dag::new(1000);
    let mut memo = HashMap::new();
    let (x, y) = (dag.var(0), dag.var(1));
    let (p0, t) = (dag.param(0), dag.param(2));
    let c = dag.konst(3.5);
    let zero = dag.konst(0.0);
    let one = dag.konst(1.0);
    assert_eq!(d(&mut dag, &mut memo, c, Wrt::U(0)), zero);
    assert_eq!(d(&mut dag, &mut memo, x, Wrt::U(0)), one);
    assert_eq!(d(&mut dag, &mut memo, x, Wrt::U(1)), zero);
    assert_eq!(d(&mut dag, &mut memo, y, Wrt::U(1)), one);
    assert_eq!(d(&mut dag, &mut memo, p0, Wrt::P(0)), one);
    assert_eq!(d(&mut dag, &mut memo, p0, Wrt::U(0)), zero);
    // t is Param(n_p) = Param(2) here; no P(k) with k < n_p selects it.
    for k in 0..2 {
        assert_eq!(d(&mut dag, &mut memo, t, Wrt::P(k)), zero);
    }
}

#[test]
fn pow_without_exponent_tangent_has_no_select() {
    let mut dag = Dag::new(1000);
    let mut memo = HashMap::new();
    let x = dag.var(0);
    let e = dag.konst(2.5);
    let pw = dag.call2(Fn2::Pow, x, e);
    let dp = d(&mut dag, &mut memo, pw, Wrt::U(0));
    assert!(!(0..dag.len()).any(|i| matches!(dag.op(NodeId(i as u32)), Op::Select(..))));
    let want = 2.5 * 1.7_f64.powf(1.5);
    assert!((val(&dag, dp, &[1.7], &[0.0]) - want).abs() <= 4.0 * ulp(want));
}

#[test]
fn pow_exponent_tangent_is_zero_where_the_power_is_zero() {
    let mut dag = Dag::new(1000);
    let mut memo = HashMap::new();
    let (x, y) = (dag.var(0), dag.var(1));
    let pw = dag.call2(Fn2::Pow, x, y);
    let dy = d(&mut dag, &mut memo, pw, Wrt::U(1));
    assert!(matches!(dag.op(dy), Op::Select(..)));
    assert_eq!(val(&dag, dy, &[0.0, 2.0], &[0.0]), 0.0);
    let want = 3.0_f64.powf(2.0) * 3.0_f64.ln();
    assert!((val(&dag, dy, &[3.0, 2.0], &[0.0]) - want).abs() <= 8.0 * ulp(want));
    // Base 2, exponent tangent only.
    let two = dag.konst(2.0);
    let p2 = dag.call2(Fn2::Pow, two, y);
    let d2 = d(&mut dag, &mut memo, p2, Wrt::U(0));
    assert_eq!(const_bits(&dag, d2), Some(0.0_f64.to_bits()));
}

#[test]
fn polygamma_lgamma_and_order_limit() {
    let mut dag = Dag::new(1000);
    let mut memo = HashMap::new();
    let x = dag.var(0);
    let lg = dag.call1(Fn1::Lgamma, x);
    let dl = d(&mut dag, &mut memo, lg, Wrt::U(0));
    let m0 = dag.konst(0.0);
    let psi0 = dag.call2(Fn2::Polygamma, m0, x);
    assert_eq!(dl, psi0);
    let d2 = d(&mut dag, &mut memo, psi0, Wrt::U(0));
    let m1 = dag.konst(1.0);
    assert_eq!(d2, dag.call2(Fn2::Polygamma, m1, x));
    assert!((val(&dag, dl, &[1.0], &[0.0]) + std::f64::consts::EULER_GAMMA).abs() < 1e-14);
    let m4 = dag.konst(4.0);
    let psi4 = dag.call2(Fn2::Polygamma, m4, x);
    let d4 = d(&mut dag, &mut memo, psi4, Wrt::U(0));
    assert!(val(&dag, d4, &[1.5], &[0.0]).is_nan());
    assert!(const_bits(&dag, d4).is_some());
}

#[test]
fn zero_rules_and_branch_rules() {
    let mut dag = Dag::new(1000);
    let mut memo = HashMap::new();
    let (x, y) = (dag.var(0), dag.var(1));
    let zero = dag.konst(0.0);
    for f in [Fn1::Floor, Fn1::Ceil, Fn1::Heav, Fn1::Sign] {
        let n = dag.call1(f, x);
        assert_eq!(d(&mut dag, &mut memo, n, Wrt::U(0)), zero);
    }
    let lt = dag.cmp(Cmp::Lt, x, y);
    assert_eq!(d(&mut dag, &mut memo, lt, Wrt::U(0)), zero);
    // min follows the branch chosen by x < y; ties pick y.
    let mn = dag.call2(Fn2::Min, x, y);
    let dmn = d(&mut dag, &mut memo, mn, Wrt::U(0));
    assert_eq!(val(&dag, dmn, &[1.0, 2.0], &[0.0]), 1.0);
    assert_eq!(val(&dag, dmn, &[3.0, 2.0], &[0.0]), 0.0);
    assert_eq!(val(&dag, dmn, &[2.0, 2.0], &[0.0]), 0.0);
    let mx = dag.call2(Fn2::Max, x, y);
    let dmx = d(&mut dag, &mut memo, mx, Wrt::U(0));
    assert_eq!(val(&dag, dmx, &[3.0, 2.0], &[0.0]), 1.0);
    assert_eq!(val(&dag, dmx, &[1.0, 2.0], &[0.0]), 0.0);
    // abs' = sign, table_slope' = 0, Bessel order tangent 0.
    let ab = dag.call1(Fn1::Abs, x);
    let dab = d(&mut dag, &mut memo, ab, Wrt::U(0));
    assert_eq!(val(&dag, dab, &[-2.0], &[0.0]), -1.0);
    assert_eq!(val(&dag, dab, &[0.0], &[0.0]), 0.0);
    let tid = dag.add_table(vec![3.0, 0.0, 2.0, 0.0, 1.0, 4.0]);
    let ts = dag.table_slope(tid, x);
    assert_eq!(d(&mut dag, &mut memo, ts, Wrt::U(0)), zero);
    let tb = dag.table(tid, x);
    let dtb = d(&mut dag, &mut memo, tb, Wrt::U(0));
    assert_eq!(dtb, ts);
    let one = dag.konst(1.0);
    let bj = dag.call2(Fn2::BesselJ, x, one);
    assert_eq!(d(&mut dag, &mut memo, bj, Wrt::U(0)), zero);
}

#[test]
fn deep_chain_does_not_recurse() {
    let mut dag = Dag::new(10_000_000);
    let mut memo = HashMap::new();
    let mut n = dag.var(0);
    for _ in 0..200_000 {
        n = dag.call1(Fn1::Sin, n);
    }
    let r = d(&mut dag, &mut memo, n, Wrt::U(0));
    assert!(val(&dag, r, &[0.3], &[0.0]).is_finite());
}

// ---- Bessel x-derivative with the order truncated toward zero (DLMF 10.6.1) ----

const BESSEL_ORDERS: [f64; 8] = [0.5, -0.5, 0.9, -0.9, 1.5, -1.5, 2.0, 0.0];
const BESSEL_XS: [f64; 4] = [0.7, 1.3, 2.9, 5.1];

/// Central difference of `fb`'s `F` entry (one state) in `x`, with the parameters `p`.
fn fd_in_x(fb: &Fallback, x: f64, p: &[f64]) -> f64 {
    let h = 1e-5 * x.abs().max(1.0);
    let at = |x: f64| {
        let mut o = [0.0];
        fb.eval(&EntryKind::F, &[x], p, &[], &mut o);
        o[0]
    };
    (at(x + h) - at(x - h)) / (2.0 * h)
}

fn assert_fd(what: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-7 * want.abs().max(1.0),
        "{what}: {got:?} vs finite difference {want:?}"
    );
}

#[test]
fn bessel_x_derivative_with_a_literal_order() {
    for name in ["besselj", "bessely"] {
        for k in BESSEL_ORDERS {
            let src = format!("x'={name}({k:?},x)\ndone\n");
            let (fb, mut ir) = setup(&src);
            let e = ir.entry(&EntryKind::JacU).unwrap();
            for x in BESSEL_XS {
                let want = fd_in_x(&fb, x, &[0.0]);
                let dag = run(&ir, &e.outputs, &[x], &[0.0])[0];
                assert_fd(&format!("{name}({k}, {x}) DAG"), dag, want);
                let mut o = [0.0];
                fb.eval(&EntryKind::JacU, &[x], &[0.0], &[], &mut o);
                assert_fd(&format!("{name}({k}, {x}) Fallback"), o[0], want);
            }
        }
    }
}

#[test]
fn bessel_x_derivative_with_the_order_from_a_parameter() {
    for name in ["besselj", "bessely"] {
        let src = format!("par a=0.5\nx'={name}(a,x)\ndone\n");
        let (fb, mut ir) = setup(&src);
        let e = ir.entry(&EntryKind::JacU).unwrap();
        for k in BESSEL_ORDERS {
            let p = [k, 0.0];
            for x in BESSEL_XS {
                let want = fd_in_x(&fb, x, &p);
                let dag = run(&ir, &e.outputs, &[x], &p)[0];
                assert_fd(&format!("{name}(a={k}, {x}) DAG"), dag, want);
                let mut o = [0.0];
                fb.eval(&EntryKind::JacU, &[x], &p, &[], &mut o);
                assert_fd(&format!("{name}(a={k}, {x}) Fallback"), o[0], want);
            }
        }
    }
}

#[test]
fn bessel_order_derivative_stays_zero_and_bad_orders_stay_nan() {
    // d/da of besselj(a, x) is 0 for every a (the order is truncated).
    let (_fb, mut ir) = setup("par a=0.5\nx'=besselj(a,x)\ndone\n");
    let e = ir.entry(&EntryKind::JacP).unwrap();
    assert_eq!(run(&ir, &e.outputs, &[1.3], &[0.5, 0.0]), [0.0]);
    let e = ir.entry(&EntryKind::JacU).unwrap();
    for bad in [f64::NAN, f64::INFINITY, 3e9, -3e9] {
        assert!(
            run(&ir, &e.outputs, &[1.3], &[bad, 0.0])[0].is_nan(),
            "{bad}"
        );
    }
}

#[test]
fn bessel_second_derivative_with_a_fractional_order() {
    // besselj(0.5, x) is J_0(x), and J_0'' = -J_0 - J_0'/x = -J_0 + J_1/x.
    let (_fb, mut ir) = setup("x'=besselj(0.5,x)\ndone\n");
    let e = ir
        .entry(&EntryKind::Dir(vec![
            crate::entry::Slot::U,
            crate::entry::Slot::U,
        ]))
        .unwrap();
    let x = 1.3_f64;
    let mut out = [0.0];
    eval(&ir.dag, &e.outputs, &[x], &[0.0], &[1.0, 1.0], &mut out);
    let want = -libm::j0(x) + libm::j1(x) / x;
    assert!((out[0] - want).abs() < 1e-13, "{} vs {want}", out[0]);
}

// ---- a failed entry leaves the model as it was ----

const CAP_SRC: &str = "par a=1.3\nx'=sin(x)*cos(y)+a*x*y\ny'=exp(x*y)-tanh(a*y)\ndone\n";

fn lower_capped(src: &str, cap: usize) -> IrModel {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    match crate::lower::lower_with_cap(&l.expanded, &l.info, cap) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}

/// Builds `F`, then `small` on a model without a tight cap and returns its node count and the
/// values of both at a point.
fn fresh_reference(small: &EntryKind, x: &[f64], p: &[f64]) -> (usize, Vec<f64>, Vec<f64>) {
    let mut ir = lower_capped(CAP_SRC, crate::dag::MAX_NODES);
    let f = ir.entry(&EntryKind::F).unwrap();
    let s = ir.entry(small).unwrap();
    let v = [0.3, -0.6, 0.9, 0.2, -0.4, 0.8];
    let n_v = if matches!(small, EntryKind::Dir(_)) {
        2
    } else {
        0
    };
    let mut fo = vec![0.0; f.outputs.len()];
    eval(&ir.dag, &f.outputs, x, p, &v[..n_v], &mut fo);
    let mut so = vec![0.0; s.outputs.len()];
    eval(&ir.dag, &s.outputs, x, p, &v[..n_v], &mut so);
    (ir.dag.len(), fo, so)
}

fn too_large_then_small_entries_still_build(small: EntryKind) {
    let (x, p) = ([0.4, -0.7], [1.3, 0.0]);
    let (cap, want_f, want_s) = fresh_reference(&small, &x, &p);
    let big = EntryKind::Dir(vec![crate::entry::Slot::U; 5]);
    {
        let mut ir = lower_capped(CAP_SRC, crate::dag::MAX_NODES);
        ir.entry(&big).unwrap();
        assert!(
            ir.dag.len() > cap,
            "the big entry must not fit under the cap"
        );
    }
    let mut ir = lower_capped(CAP_SRC, cap);
    let before = ir.dag.len();
    for _ in 0..2 {
        let e = ir.entry(&big).unwrap_err();
        assert_eq!(e.kind, crate::IrErrorKind::TooLarge);
        assert!(
            !ir.dag.exceeded(),
            "the failed request must not stay flagged"
        );
        assert_eq!(
            ir.dag.len(),
            before,
            "the failed request must leave no nodes"
        );
    }
    let f = ir.entry(&EntryKind::F).expect("F after a failed request");
    let s = ir
        .entry(&small)
        .expect("small entry after a failed request");
    assert_eq!(ir.dag.len(), cap, "same nodes as a model that never failed");
    let v = [0.3, -0.6, 0.9, 0.2, -0.4, 0.8];
    let n_v = if matches!(small, EntryKind::Dir(_)) {
        2
    } else {
        0
    };
    let mut fo = vec![0.0; f.outputs.len()];
    eval(&ir.dag, &f.outputs, &x, &p, &v[..n_v], &mut fo);
    let mut so = vec![0.0; s.outputs.len()];
    eval(&ir.dag, &s.outputs, &x, &p, &v[..n_v], &mut so);
    assert_eq!(fo, want_f);
    assert_eq!(so, want_s);
}

#[test]
fn a_too_large_dir_does_not_poison_jacu() {
    too_large_then_small_entries_still_build(EntryKind::JacU);
}

#[test]
fn a_too_large_dir_does_not_poison_a_small_dir() {
    too_large_then_small_entries_still_build(EntryKind::Dir(vec![crate::entry::Slot::U]));
}

#[test]
fn a_too_large_jacu_does_not_poison_f() {
    let src = "x'=sin(x)*cos(x)+x*x\ndone\n";
    let full = {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        lower(&l.expanded, &l.info).unwrap().dag.len()
    };
    let mut ir = lower_capped(src, full);
    assert_eq!(
        ir.entry(&EntryKind::JacU).unwrap_err().kind,
        crate::IrErrorKind::TooLarge
    );
    assert!(!ir.dag.exceeded());
    assert_eq!(ir.dag.len(), full);
    assert!(ir.entry(&EntryKind::F).is_ok());
}
