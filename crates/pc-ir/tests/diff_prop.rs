//! Property tests for the symbolic derivatives. Random expressions are written as `.ode`
//! text, loaded and lowered; `d` is compared with the interpreter (`Fallback`) and
//! checked structurally on the DAG.

use pc_ir::check::{close, close_err, close_ulps, scale};
use pc_ir::dag::{Dag, NodeId, Op};
use pc_ir::diff::{Wrt, d};
use pc_ir::eval::{eval, eval_err};
use pc_ir::{EntryKind, Fallback, IrModel, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use proptest::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
enum G {
    C(f64),
    X,
    Y,
    A,
    Add(Box<G>, Box<G>),
    Sub(Box<G>, Box<G>),
    Mul(Box<G>, Box<G>),
    /// a / (1 + b*b)
    Div(Box<G>, Box<G>),
    Neg(Box<G>),
    Sin(Box<G>),
    Cos(Box<G>),
    Tanh(Box<G>),
    Atan(Box<G>),
    Exp(Box<G>),
    /// sqrt(1 + g*g)
    Sqrt(Box<G>),
    /// ln(1 + g*g)
    Ln(Box<G>),
    Abs(Box<G>),
    Sq(Box<G>),
    Cube(Box<G>),
    Min(Box<G>, Box<G>),
    Max(Box<G>, Box<G>),
    /// if (a > b) then (c) else (e)
    If(Box<G>, Box<G>, Box<G>, Box<G>),
    /// (1 + a*a) ^ c
    Pow(Box<G>, f64),
    /// (1 + a*a) ^ (0.5 + b*b)
    PowE(Box<G>, Box<G>),
}

fn num(v: f64) -> String {
    format!("{v:?}")
}

fn text(g: &G) -> String {
    let b = |g: &G| text(g);
    match g {
        G::C(v) => num(*v),
        G::X => "x".into(),
        G::Y => "y".into(),
        G::A => "a".into(),
        G::Add(p, q) => format!("({}+{})", b(p), b(q)),
        G::Sub(p, q) => format!("({}-{})", b(p), b(q)),
        G::Mul(p, q) => format!("({}*{})", b(p), b(q)),
        G::Div(p, q) => format!("({}/(1+{}*{}))", b(p), b(q), b(q)),
        G::Neg(p) => format!("(-{})", b(p)),
        G::Sin(p) => format!("sin({})", b(p)),
        G::Cos(p) => format!("cos({})", b(p)),
        G::Tanh(p) => format!("tanh({})", b(p)),
        G::Atan(p) => format!("atan({})", b(p)),
        G::Exp(p) => format!("exp({})", b(p)),
        G::Sqrt(p) => format!("sqrt(1+{}*{})", b(p), b(p)),
        G::Ln(p) => format!("ln(1+{}*{})", b(p), b(p)),
        G::Abs(p) => format!("abs({})", b(p)),
        G::Sq(p) => format!("({})^2", b(p)),
        G::Cube(p) => format!("({})^3", b(p)),
        G::Min(p, q) => format!("min({},{})", b(p), b(q)),
        G::Max(p, q) => format!("max({},{})", b(p), b(q)),
        G::If(p, q, r, s) => format!("if({}>{})then({})else({})", b(p), b(q), b(r), b(s)),
        G::Pow(p, c) => format!("(1+{}*{})^({})", b(p), b(p), num(*c)),
        G::PowE(p, q) => format!("(1+{}*{})^(0.5+{}*{})", b(p), b(p), b(q), b(q)),
    }
}

fn g_strategy() -> impl Strategy<Value = G> {
    let leaf = prop_oneof![
        prop::sample::select(vec![0.0, 1.0, 2.0, 0.5, 1.5, 3.0, 0.25]).prop_map(G::C),
        Just(G::X),
        Just(G::Y),
        Just(G::A),
    ];
    leaf.prop_recursive(4, 24, 2, |inner| {
        let i = || inner.clone().prop_map(Box::new);
        prop_oneof![
            (i(), i()).prop_map(|(a, b)| G::Add(a, b)),
            (i(), i()).prop_map(|(a, b)| G::Sub(a, b)),
            (i(), i()).prop_map(|(a, b)| G::Mul(a, b)),
            (i(), i()).prop_map(|(a, b)| G::Div(a, b)),
            i().prop_map(G::Neg),
            i().prop_map(G::Sin),
            i().prop_map(G::Cos),
            i().prop_map(G::Tanh),
            i().prop_map(G::Atan),
            i().prop_map(G::Exp),
            i().prop_map(G::Sqrt),
            i().prop_map(G::Ln),
            i().prop_map(G::Abs),
            i().prop_map(G::Sq),
            i().prop_map(G::Cube),
            (i(), i()).prop_map(|(a, b)| G::Min(a, b)),
            (i(), i()).prop_map(|(a, b)| G::Max(a, b)),
            (i(), i(), i(), i()).prop_map(|(a, b, c, e)| G::If(a, b, c, e)),
            (i(), prop::sample::select(vec![0.5, 1.5, 2.5, -1.5])).prop_map(|(a, c)| G::Pow(a, c)),
            (i(), i()).prop_map(|(a, b)| G::PowE(a, b)),
        ]
    })
}

fn build(f0: &G, f1: &G) -> Result<(Fallback, IrModel), String> {
    let src = format!("par a=0.7\nx'={}\ny'={}\ndone\n", text(f0), text(f1));
    let mut map = SourceMap::new();
    let top = map.add("t.ode", &src);
    let l = load(top, &mut map, &mut MapLoader::new());
    if l.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(format!("check error in {src}"));
    }
    let ir = lower(&l.expanded, &l.info).map_err(|e| format!("lower error {e:?} in {src}"))?;
    Ok((Fallback::new(&l.expanded, &l.info, &ir), ir))
}

/// `pc_ir::check::close` (SPEC section 3 without a running error bound) on every value.
fn all_close(got: &[f64], want: &[f64]) -> bool {
    let s = scale(want);
    got.iter().zip(want).all(|(&a, &b)| close(a, b, s))
}

fn run(ir: &IrModel, outs: &[NodeId], x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; outs.len()];
    eval(&ir.dag, outs, x, p, &[], &mut out);
    out
}

/// Value and running error bound of each output.
fn run_err(ir: &IrModel, outs: &[NodeId], x: &[f64], p: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut v, mut e) = (vec![0.0; outs.len()], vec![0.0; outs.len()]);
    eval_err(&ir.dag, outs, x, p, &[], &mut v, &mut e);
    (v, e)
}

/// Rebuilds every node of `src` in a fresh DAG, in id order, through the folding
/// constructors. Every node is already canonical, so each one must come back at its own id.
fn rebuild(src: &Dag) -> Result<Dag, String> {
    let mut dst = Dag::new(usize::MAX);
    dst.tables = src.tables.clone();
    for i in 0..src.len() {
        let new = match *src.op(NodeId(i as u32)) {
            Op::Const(b) => dst.konst(f64::from_bits(b)),
            Op::Var(k) => dst.var(k),
            Op::Param(k) => dst.param(k),
            Op::Dir(k) => dst.dir(k),
            Op::Neg(a) => dst.neg(a),
            Op::Add(a, b) => dst.add(a, b),
            Op::Sub(a, b) => dst.sub(a, b),
            Op::Mul(a, b) => dst.mul(a, b),
            Op::Div(a, b) => dst.div(a, b),
            Op::PowI(a, n) => dst.powi(a, n),
            Op::Call1(f, a) => dst.call1(f, a),
            Op::Call2(f, a, b) => dst.call2(f, a, b),
            Op::Cmp(c, a, b) => dst.cmp(c, a, b),
            Op::And(a, b) => dst.and(a, b),
            Op::Or(a, b) => dst.or(a, b),
            Op::Not(a) => dst.not(a),
            Op::Select(c, a, b) => dst.select(c, a, b),
            Op::Table(t, x) => dst.table(t, x),
            Op::TableSlope(t, x) => dst.table_slope(t, x),
        };
        if new != NodeId(i as u32) {
            return Err(format!(
                "node {i} {:?} came back as {new:?} ({:?})",
                src.op(NodeId(i as u32)),
                dst.op(new)
            ));
        }
    }
    Ok(dst)
}

const BIG: f64 = 1e8;

fn small(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite() && x.abs() < BIG)
}

/// 300 cases, or the number in `PROPTEST_CASES`. `ProptestConfig::with_cases` would ignore the
/// variable. The `prop_assume!(small(..))` calls reject cases with huge or non-finite values, so
/// the reject budget scales with the case count (the default of 1024 stops a large run).
fn config() -> ProptestConfig {
    let cases: u32 = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    ProptestConfig {
        max_global_rejects: cases.saturating_mul(100).max(1024),
        ..ProptestConfig::with_cases(cases)
    }
}

proptest! {
    #![proptest_config(config())]

    /// JacU and JacP of the DAG against the interpreter's dual-number columns.
    #[test]
    fn jacobians_match_the_interpreter(
        f0 in g_strategy(),
        f1 in g_strategy(),
        x in -2.0..2.0f64,
        y in -2.0..2.0f64,
        a in 0.1..1.5f64,
    ) {
        let (fb, mut ir) = build(&f0, &f1).map_err(TestCaseError::fail)?;
        let (xs, ps) = ([x, y], [a, 0.0]);
        for (kind, len) in [(EntryKind::JacU, 4), (EntryKind::JacP, 2)] {
            let e = ir.entry(&kind).map_err(|e| TestCaseError::fail(e.to_string()))?;
            prop_assert_eq!(e.outputs.len(), len);
            let (got, err) = run_err(&ir, &e.outputs, &xs, &ps);
            let mut want = vec![0.0; len];
            fb.eval(&kind, &xs, &ps, &[], &mut want);
            prop_assume!(small(&want) && small(&got));
            let s = scale(&want);
            let ok = got
                .iter()
                .zip(&want)
                .zip(&err)
                .all(|((&a, &b), &e)| close_err(a, b, s, e));
            prop_assert!(ok, "{:?}: {:?} vs {:?}", kind, got, want);
        }
    }

    /// `F` of the DAG against the interpreter: 4 ulp (SPEC section 4), a NaN matches any
    /// NaN, and `+0` equals `-0`.
    #[test]
    fn f_matches_the_interpreter_within_4_ulp(
        f0 in g_strategy(),
        f1 in g_strategy(),
        x in -2.0..2.0f64,
        y in -2.0..2.0f64,
        a in 0.1..1.5f64,
    ) {
        let (fb, mut ir) = build(&f0, &f1).map_err(TestCaseError::fail)?;
        let (xs, ps) = ([x, y], [a, 0.0]);
        let e = ir.entry(&EntryKind::F).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let got = run(&ir, &e.outputs, &xs, &ps);
        let mut want = vec![0.0; 2];
        fb.eval(&EntryKind::F, &xs, &ps, &[], &mut want);
        prop_assume!(small(&want) && small(&got));
        let ok = got.iter().zip(&want).all(|(&a, &b)| close_ulps(a, b));
        prop_assert!(ok, "F: {:?} vs {:?}", got, want);
    }

    /// Building the nodes of a model again through the folding constructors changes nothing:
    /// each node comes back at its own id, and the outputs are bit for bit the same.
    #[test]
    fn a_second_folding_pass_changes_nothing(
        f0 in g_strategy(),
        f1 in g_strategy(),
        x in -2.0..2.0f64,
        y in -2.0..2.0f64,
    ) {
        let (_fb, mut ir) = build(&f0, &f1).map_err(TestCaseError::fail)?;
        let mut outs = ir.f.clone();
        for kind in [EntryKind::JacU, EntryKind::JacP] {
            let e = ir.entry(&kind).map_err(|e| TestCaseError::fail(e.to_string()))?;
            outs.extend(e.outputs);
        }
        let again = rebuild(&ir.dag).map_err(TestCaseError::fail)?;
        prop_assert_eq!(again.len(), ir.dag.len());
        let (xs, ps) = ([x, y], [0.7, 0.0]);
        let (mut a, mut b) = (vec![0.0; outs.len()], vec![0.0; outs.len()]);
        eval(&ir.dag, &outs, &xs, &ps, &[], &mut a);
        eval(&again, &outs, &xs, &ps, &[], &mut b);
        let same = a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()));
        prop_assert!(same, "{:?} vs {:?}", a, b);
    }

    /// d(const) = 0, d(x)/dx = 1, linearity (structural), product rule (numeric), and one
    /// node per structure when the derivative is built twice.
    #[test]
    fn structural_rules(
        f0 in g_strategy(),
        f1 in g_strategy(),
        c in prop::sample::select(vec![0.0, 1.0, -2.0, 0.5, 3.0]),
        x in -2.0..2.0f64,
        y in -2.0..2.0f64,
    ) {
        let (fb, mut ir) = build(&f0, &f1).map_err(TestCaseError::fail)?;
        let (g0, g1) = (ir.f[0], ir.f[1]);
        let mut memo = HashMap::new();
        let w = Wrt::U(0);
        let zero = ir.dag.konst(0.0);
        let one = ir.dag.konst(1.0);
        let k = ir.dag.konst(c);
        let vx = ir.dag.var(0);
        prop_assert_eq!(d(&mut ir.dag, &mut memo, k, w), zero);
        prop_assert_eq!(d(&mut ir.dag, &mut memo, vx, w), one);

        let (d0, d1) = (d(&mut ir.dag, &mut memo, g0, w), d(&mut ir.dag, &mut memo, g1, w));
        // Linearity: d(f + g) = df + dg and d(c * f) = df * c, as one shared node.
        let sum = ir.dag.add(g0, g1);
        let dsum = d(&mut ir.dag, &mut memo, sum, w);
        let want_sum = ir.dag.add(d0, d1);
        // Constants may differ only in the sign of zero (neg(0) is -0.0, -0.0 + 0.0 is +0.0).
        let same_const = |dag: &pc_ir::dag::Dag, a: NodeId, b: NodeId| match (dag.op(a), dag.op(b)) {
            (Op::Const(p), Op::Const(q)) => f64::from_bits(*p) == f64::from_bits(*q),
            _ => false,
        };
        prop_assert!(dsum == want_sum || same_const(&ir.dag, dsum, want_sum));
        let scaled = ir.dag.mul(k, g0);
        let dscaled = d(&mut ir.dag, &mut memo, scaled, w);
        let want_scaled = ir.dag.mul(d0, k);
        prop_assert!(dscaled == want_scaled || same_const(&ir.dag, dscaled, want_scaled));

        // Product rule against the interpreter: the DAG derivative of f0*f1 against
        // J0j*f1 + f0*J1j, with f and the Jacobian taken from Fallback (Dual<f64>).
        let prod = ir.dag.mul(g0, g1);
        let dprod = [Wrt::U(0), Wrt::U(1)].map(|wj| d(&mut ir.dag, &mut memo, prod, wj));
        let (xs, ps) = ([x, y], [0.7, 0.0]);
        let got = run(&ir, &dprod, &xs, &ps);
        let (mut fv, mut jac) = ([0.0; 2], [0.0; 4]);
        fb.eval(&EntryKind::F, &xs, &ps, &[], &mut fv);
        fb.eval(&EntryKind::JacU, &xs, &ps, &[], &mut jac);
        let want = [
            jac[0] * fv[1] + fv[0] * jac[1],
            jac[2] * fv[1] + fv[0] * jac[3],
        ];
        prop_assume!(small(&want) && small(&got) && small(&fv) && small(&jac));
        prop_assert!(all_close(&got, &want), "{:?} vs {:?}", got, want);

        // Building again with a fresh memo adds no node and returns the same ids.
        let len = ir.dag.len();
        let mut memo2 = HashMap::new();
        prop_assert_eq!(d(&mut ir.dag, &mut memo2, g0, w), d0);
        prop_assert_eq!(d(&mut ir.dag, &mut memo2, sum, w), dsum);
        prop_assert_eq!(ir.dag.len(), len);
    }
}
