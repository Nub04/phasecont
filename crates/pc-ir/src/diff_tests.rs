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
fn entry_f_aux_cached_and_dir_refused() {
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
    let e = ir.entry(&EntryKind::Dir(vec![crate::Slot::U])).unwrap_err();
    assert_eq!(e.kind, crate::IrErrorKind::Unsupported);
}

#[test]
fn entry_reports_too_large() {
    let mut ir = setup("x'=sin(x)*cos(x)+x*x\ndone\n").1;
    ir.dag = Dag::new(0);
    // The old ids point into the empty DAG; the derivative still has to add nodes.
    ir.f = vec![ir.dag.var(0)];
    let sin = ir.dag.call1(Fn1::Sin, ir.f[0]);
    ir.f = vec![sin];
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
