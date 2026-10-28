use crate::{DirFn, PlainFn, compile, module};
use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId};
use pc_ir::eval::eval;
use pc_ir::{Entry, EntryKind, Slot, lower};
use proptest::prelude::*;

/// Bitwise equality where every NaN equals every NaN.
fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn bits_list(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:?}({:016x})", x.to_bits()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn entry(outputs: Vec<NodeId>) -> Entry {
    Entry {
        kind: EntryKind::F,
        outputs,
    }
}

/// Compiles `outputs` as an `F` entry and compares with the interpreter on every point.
fn check_points(dag: &Dag, outputs: Vec<NodeId>, points: &[Vec<f64>], what: &str) {
    let e = entry(outputs.clone());
    let jit = compile(&[(EntryKind::F, &e)], dag).expect("compile");
    let f: PlainFn = jit.plain(&EntryKind::F).expect("plain fn");
    for x in points {
        let p = [0.5, 0.25];
        let mut want = vec![f64::NAN; outputs.len()];
        eval(dag, &outputs, x, &p, &[], &mut want);
        let mut got = vec![f64::MAX; outputs.len()];
        // SAFETY: x, p and got have the lengths the entry reads and writes.
        unsafe { f(x.as_ptr(), p.as_ptr(), got.as_mut_ptr()) };
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                same(*a, *b),
                "{what}: output {k} at x=[{}]: jit {a:?} vs eval {b:?}",
                bits_list(x)
            );
        }
    }
}

fn specials() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        2.5,
        -2.5,
        3.0,
        7.25,
        10.0,
        0.1,
        -0.1,
        1e-310,
        -1e-310,
        1e-300,
        1e10,
        1e308,
        -1e308,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

/// A small deterministic generator for sample points.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

const FN1: [Fn1; 21] = [
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

const FN2: [Fn2; 8] = [
    Fn2::Atan2,
    Fn2::Pow,
    Fn2::Mod,
    Fn2::BesselJ,
    Fn2::BesselY,
    Fn2::Polygamma,
    Fn2::Min,
    Fn2::Max,
];

#[test]
fn every_fn1_matches_eval_on_edge_values_and_random_points() {
    let mut dag = Dag::new(1000);
    let x = dag.var(0);
    let outs: Vec<NodeId> = FN1.iter().map(|&f| dag.call1(f, x)).collect();
    let mut pts: Vec<Vec<f64>> = specials().into_iter().map(|v| vec![v]).collect();
    let mut r = Lcg(1);
    for _ in 0..2000 {
        let s = [1.0, 10.0, 100.0][(r.next() * 3.0) as usize];
        pts.push(vec![r.range(-s, s)]);
    }
    for (f, o) in FN1.iter().zip(&outs) {
        check_points(&dag, vec![*o], &pts, &format!("{f:?}"));
    }
    check_points(&dag, outs, &pts, "all Fn1 in one entry");
}

#[test]
fn heav_and_sign_edge_values() {
    let mut dag = Dag::new(100);
    let x = dag.var(0);
    let h = dag.call1(Fn1::Heav, x);
    let s = dag.call1(Fn1::Sign, x);
    let e = entry(vec![h, s]);
    let jit = compile(&[(EntryKind::F, &e)], &dag).expect("compile");
    let f = jit.plain(&EntryKind::F).expect("fn");
    let run = |x: f64| {
        let mut out = [9.0; 2];
        // SAFETY: lengths match the entry.
        unsafe { f(&x, [0.0].as_ptr(), out.as_mut_ptr()) };
        out
    };
    let [h0, s0] = run(0.0);
    assert_eq!((h0.to_bits(), s0.to_bits()), (1.0f64.to_bits(), 0));
    let [hn, sn] = run(-0.0);
    assert_eq!((hn.to_bits(), sn.to_bits()), (1.0f64.to_bits(), 0));
    let [hnan, snan] = run(f64::NAN);
    assert_eq!(hnan.to_bits(), 0);
    assert_eq!(snan.to_bits(), f64::NAN.to_bits());
    assert_eq!(run(-3.0), [0.0, -1.0]);
    assert_eq!(run(f64::INFINITY), [1.0, 1.0]);
    assert_eq!(run(f64::NEG_INFINITY), [0.0, -1.0]);
}

#[test]
fn every_fn2_matches_eval() {
    let mut dag = Dag::new(1000);
    let (a, b) = (dag.var(0), dag.var(1));
    // Bessel and polygamma take an order; keep it small so libm stays fast.
    let orders = [0.0, -0.0, 1.0, 2.0, 3.0, -1.0, 0.5, 5.0, f64::NAN, 2.9];
    let sp = specials();
    for &f in &FN2 {
        let o = dag.call2(f, a, b);
        let small = matches!(f, Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma);
        let firsts: &[f64] = if small { &orders } else { &sp };
        let mut pts = Vec::new();
        for &u in firsts {
            for &v in &sp {
                pts.push(vec![u, v]);
            }
        }
        let mut r = Lcg(7);
        for _ in 0..500 {
            let u = if small {
                (r.next() * 6.0).floor()
            } else {
                r.range(-20.0, 20.0)
            };
            pts.push(vec![u, r.range(-20.0, 20.0)]);
        }
        check_points(&dag, vec![o], &pts, &format!("{f:?}"));
    }
}

#[test]
fn powi_for_every_exponent() {
    let mut dag = Dag::new(1000);
    let x = dag.var(0);
    let mut outs = Vec::new();
    for n in -16..=16 {
        if n == 0 || n == 1 {
            continue;
        }
        outs.push(dag.powi(x, n));
    }
    let mut pts: Vec<Vec<f64>> = specials().into_iter().map(|v| vec![v]).collect();
    let mut r = Lcg(3);
    for _ in 0..1000 {
        pts.push(vec![r.range(-3.0, 3.0)]);
    }
    for o in &outs {
        check_points(&dag, vec![*o], &pts, &format!("{:?}", dag.op(*o)));
    }
}

#[test]
fn arithmetic_matches_eval() {
    let mut dag = Dag::new(1000);
    let (a, b) = (dag.var(0), dag.var(1));
    let outs = vec![
        dag.neg(a),
        dag.add(a, b),
        dag.sub(a, b),
        dag.mul(a, b),
        dag.div(a, b),
    ];
    let sp = specials();
    let mut pts = Vec::new();
    for &u in &sp {
        for &v in &sp {
            pts.push(vec![u, v]);
        }
    }
    check_points(&dag, outs, &pts, "arithmetic");
}

#[test]
fn comparisons_and_logic_with_nan_and_signed_zero() {
    let mut dag = Dag::new(1000);
    let (a, b, c) = (dag.var(0), dag.var(1), dag.var(2));
    let mut outs = Vec::new();
    for k in [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq] {
        outs.push(dag.cmp(k, a, b));
    }
    outs.push(dag.and(a, b));
    outs.push(dag.or(a, b));
    outs.push(dag.not(a));
    outs.push(dag.select(c, a, b));
    outs.push(dag.select(a, b, c));
    let vals = [
        f64::NAN,
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e-310,
    ];
    let mut pts = Vec::new();
    for &u in &vals {
        for &v in &vals {
            for &w in &vals {
                pts.push(vec![u, v, w]);
            }
        }
    }
    check_points(&dag, outs.clone(), &pts, "cmp/logic/select");
    for o in outs {
        check_points(&dag, vec![o], &pts, &format!("{:?}", dag.op(o)));
    }
}

#[test]
fn tables_survive_dropping_the_dag() {
    let mut dag = Dag::new(100);
    let t = dag.add_table(vec![5.0, -1.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]);
    let bad_short = dag.add_table(vec![5.0, 0.0, 1.0]);
    let bad_empty = dag.add_table(vec![]);
    let bad_n = dag.add_table(vec![1.0, 0.0, 1.0, 0.0]);
    let bad_nan = dag.add_table(vec![f64::NAN, 0.0, 1.0, 0.0, 1.0]);
    let x = dag.var(0);
    let outs = vec![
        dag.table(t, x),
        dag.table_slope(t, x),
        dag.table(bad_short, x),
        dag.table_slope(bad_empty, x),
        dag.table(77, x),
        dag.table(bad_n, x),
        dag.table_slope(bad_nan, x),
    ];
    let e = entry(outs.clone());
    let pts: Vec<f64> = vec![
        -2.0,
        -1.0,
        -0.7,
        0.0,
        0.3,
        0.5,
        1.0,
        1.4,
        f64::NAN,
        f64::INFINITY,
        -0.0,
    ];
    let want: Vec<Vec<f64>> = pts
        .iter()
        .map(|&v| {
            let mut w = vec![0.0; outs.len()];
            eval(&dag, &outs, &[v], &[0.0], &[], &mut w);
            w
        })
        .collect();
    let jit = compile(&[(EntryKind::F, &e)], &dag).expect("compile");
    drop(dag);
    // Overwrite freed memory so a stale pointer would show.
    let junk: Vec<Vec<f64>> = (0..64).map(|_| vec![f64::NAN; 16]).collect();
    let f = jit.plain(&EntryKind::F).expect("fn");
    for (v, w) in pts.iter().zip(&want) {
        let mut got = vec![0.0; outs.len()];
        // SAFETY: lengths match the entry.
        unsafe { f(v, [0.0].as_ptr(), got.as_mut_ptr()) };
        for k in 0..got.len() {
            assert!(
                same(got[k], w[k]),
                "x={v:?} out {k}: {} vs {}",
                got[k],
                w[k]
            );
        }
    }
    drop(junk);
    // Malformed blocks give NaN from both evaluators (outputs 2 and up).
    for w in &want {
        assert!((2..outs.len()).all(|k| w[k].is_nan()), "{w:?}");
    }
}

#[test]
fn empty_and_out_of_range_outputs() {
    let mut dag = Dag::new(100);
    let x = dag.var(0);
    let d = dag.dir(0);
    let k = dag.konst(2.5);
    let e0 = entry(vec![]);
    let e1 = entry(vec![k, x, d, NodeId(9999), x]);
    let jit = compile(&[(EntryKind::Aux, &e0), (EntryKind::F, &e1)], &dag).expect("compile");
    let f0 = jit.plain(&EntryKind::Aux).expect("aux");
    let mut out = [7.0; 2];
    // SAFETY: the empty entry writes nothing.
    unsafe { f0([1.0].as_ptr(), [0.0].as_ptr(), out.as_mut_ptr()) };
    assert_eq!(out, [7.0, 7.0]);
    let f1 = jit.plain(&EntryKind::F).expect("f");
    let mut out = [0.0; 5];
    // SAFETY: lengths match.
    unsafe { f1([4.0].as_ptr(), [0.0].as_ptr(), out.as_mut_ptr()) };
    assert_eq!(out[0], 2.5);
    assert_eq!(out[1], 4.0);
    assert!(out[2].is_nan(), "Dir in a plain entry reads as NaN");
    assert!(out[3].is_nan());
    assert_eq!(out[4], 4.0);
    // Wrong accessor kinds give None.
    assert!(jit.dir(&EntryKind::F).is_none());
    assert!(jit.plain(&EntryKind::JacU).is_none());
    assert!(jit.plain(&EntryKind::Dir(vec![Slot::U])).is_none());
}

#[test]
fn a_frame_over_4096_bytes_runs() {
    // 1500 products stay live until the final sum, so most spill to the stack.
    let mut dag = Dag::new(10_000);
    let x = dag.var(0);
    let prods: Vec<NodeId> = (1..=1500)
        .map(|i| {
            let c = dag.konst(i as f64 + 0.5);
            dag.mul(x, c)
        })
        .collect();
    let mut acc = prods[0];
    for &p in &prods[1..] {
        acc = dag.add(acc, p);
    }
    // Add the products again in reverse so all of them are live late.
    for &p in prods.iter().rev() {
        acc = dag.sub(acc, p);
    }
    let pts: Vec<Vec<f64>> = [0.3, -1.7, 2.0, f64::NAN, 1e200]
        .iter()
        .map(|&v| vec![v])
        .collect();
    module::MAX_FRAME.with(|m| m.set(0));
    check_points(&dag, vec![acc], &pts, "large frame");
    let frame = module::MAX_FRAME.with(|m| m.get());
    assert!(frame > 4096, "frame is only {frame} bytes");
}

#[test]
fn a_forced_error_after_module_creation_frees_the_module() {
    let dag = Dag::new(10);
    let before = module::FREED.with(|c| c.get());
    module::FAIL_AFTER_NEW.with(|f| f.set(true));
    let r = compile(&[], &dag);
    module::FAIL_AFTER_NEW.with(|f| f.set(false));
    assert!(r.is_err());
    assert_eq!(module::FREED.with(|c| c.get()), before + 1);
    let ok = compile(&[], &dag).expect("compile");
    drop(ok);
    assert_eq!(module::FREED.with(|c| c.get()), before + 2);
}

fn build_model(src: &str) -> lower::IrModel {
    use pc_ode::{MapLoader, Severity, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    match pc_ir::lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}

#[test]
fn model_entries_match_eval() {
    let src = "par a=1.3, b=0.7\ninit x=0.4, y=0.3\n\
        aux s=x*y+sin(a)\n\
        x'=a*sin(x*y)-x^3+exp(-y)/(1+x^2)+max(x,b*y)\n\
        y'=b*x-y+tanh(a*y)+sqrt(1+x*x)+x^20+atan2(y,x)\ndone\n";
    let mut ir = build_model(src);
    let (n, n_p) = (ir.n(), ir.n_p());
    let dir_kinds = [
        EntryKind::Dir(vec![Slot::U]),
        EntryKind::Dir(vec![Slot::U, Slot::U]),
        EntryKind::Dir(vec![Slot::Full, Slot::P]),
        EntryKind::Dir(vec![Slot::U, Slot::P, Slot::Full]),
        EntryKind::Dir(vec![Slot::U, Slot::U, Slot::U, Slot::U, Slot::U]),
    ];
    let mut kinds = vec![
        EntryKind::F,
        EntryKind::Aux,
        EntryKind::JacU,
        EntryKind::JacP,
    ];
    kinds.extend(dir_kinds);
    let es: Vec<Entry> = kinds.iter().map(|k| ir.entry(k).expect("entry")).collect();
    let pairs: Vec<(EntryKind, &Entry)> = kinds.iter().cloned().zip(es.iter()).collect();
    let jit = compile(&pairs, &ir.dag).expect("compile");
    let mut r = Lcg(42);
    for _ in 0..50 {
        let x: Vec<f64> = (0..n).map(|_| r.range(-2.0, 2.0)).collect();
        let p: Vec<f64> = (0..=n_p).map(|_| r.range(0.1, 2.0)).collect();
        for (kind, e) in &pairs {
            let mut want = vec![f64::NAN; e.outputs.len()];
            let mut got = vec![f64::MAX; e.outputs.len()];
            if let EntryKind::Dir(slots) = kind {
                let len: usize = slots.iter().map(|s| s.len(n, n_p)).sum();
                let v: Vec<f64> = (0..len).map(|_| r.range(-1.0, 1.0)).collect();
                eval(&ir.dag, &e.outputs, &x, &p, &v, &mut want);
                let f: DirFn = jit.dir(kind).expect("dir fn");
                // SAFETY: x has n, p has n_p + 1, v has len and got has n values.
                unsafe { f(x.as_ptr(), p.as_ptr(), v.as_ptr(), got.as_mut_ptr()) };
            } else {
                eval(&ir.dag, &e.outputs, &x, &p, &[], &mut want);
                let f: PlainFn = jit.plain(kind).expect("plain fn");
                // SAFETY: lengths follow the entry contract.
                unsafe { f(x.as_ptr(), p.as_ptr(), got.as_mut_ptr()) };
            }
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(same(*a, *b), "{kind:?} out {k}: {a:?} vs {b:?}");
            }
        }
    }
}

// ---- random expressions ----

#[derive(Clone, Debug)]
enum Ex {
    Var(u32),
    Param(u32),
    Const(f64),
    Neg(Box<Ex>),
    Bin(u8, Box<Ex>, Box<Ex>),
    Un(usize, Box<Ex>),
    Two(usize, Box<Ex>, Box<Ex>),
    Pow(i32, Box<Ex>),
    Cmp(u8, Box<Ex>, Box<Ex>),
    Logic(u8, Box<Ex>, Box<Ex>),
    Not(Box<Ex>),
    Sel(Box<Ex>, Box<Ex>, Box<Ex>),
}

fn ex_strategy() -> impl Strategy<Value = Ex> {
    let consts = prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(1.0),
        Just(-1.0),
        Just(2.0),
        Just(0.5),
        Just(f64::NAN),
        Just(f64::INFINITY),
        -10.0..10.0f64,
    ];
    let leaf = prop_oneof![
        (0u32..3).prop_map(Ex::Var),
        (0u32..3).prop_map(Ex::Param),
        consts.prop_map(Ex::Const),
    ];
    leaf.prop_recursive(6, 64, 3, |inner| {
        let b = || inner.clone().prop_map(Box::new);
        prop_oneof![
            b().prop_map(Ex::Neg),
            (0u8..4, b(), b()).prop_map(|(o, l, r)| Ex::Bin(o, l, r)),
            (0usize..FN1.len(), b()).prop_map(|(f, a)| Ex::Un(f, a)),
            // Bessel and polygamma stay out of the random pool; their grids are above.
            (0usize..3, b(), b()).prop_map(|(f, l, r)| Ex::Two(f, l, r)),
            (-16i32..=16, b()).prop_map(|(n, a)| Ex::Pow(n, a)),
            (0u8..5, b(), b()).prop_map(|(o, l, r)| Ex::Cmp(o, l, r)),
            (0u8..2, b(), b()).prop_map(|(o, l, r)| Ex::Logic(o, l, r)),
            b().prop_map(Ex::Not),
            (b(), b(), b()).prop_map(|(c, l, r)| Ex::Sel(c, l, r)),
        ]
    })
}

fn build_ex(dag: &mut Dag, e: &Ex) -> NodeId {
    match e {
        Ex::Var(i) => dag.var(*i),
        Ex::Param(j) => dag.param(*j),
        Ex::Const(c) => dag.konst(*c),
        Ex::Neg(a) => {
            let a = build_ex(dag, a);
            dag.neg(a)
        }
        Ex::Bin(o, l, r) => {
            let (l, r) = (build_ex(dag, l), build_ex(dag, r));
            match o {
                0 => dag.add(l, r),
                1 => dag.sub(l, r),
                2 => dag.mul(l, r),
                _ => dag.div(l, r),
            }
        }
        Ex::Un(f, a) => {
            let a = build_ex(dag, a);
            dag.call1(FN1[*f], a)
        }
        Ex::Two(f, l, r) => {
            let (l, r) = (build_ex(dag, l), build_ex(dag, r));
            let f = [Fn2::Min, Fn2::Max, Fn2::Pow][*f % 3];
            dag.call2(f, l, r)
        }
        Ex::Pow(n, a) => {
            let a = build_ex(dag, a);
            dag.powi(a, *n)
        }
        Ex::Cmp(o, l, r) => {
            let (l, r) = (build_ex(dag, l), build_ex(dag, r));
            let c = [Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge, Cmp::Eq][*o as usize % 5];
            dag.cmp(c, l, r)
        }
        Ex::Logic(o, l, r) => {
            let (l, r) = (build_ex(dag, l), build_ex(dag, r));
            if *o == 0 { dag.and(l, r) } else { dag.or(l, r) }
        }
        Ex::Not(a) => {
            let a = build_ex(dag, a);
            dag.not(a)
        }
        Ex::Sel(c, l, r) => {
            let (c, l, r) = (build_ex(dag, c), build_ex(dag, l), build_ex(dag, r));
            dag.select(c, l, r)
        }
    }
}

fn point() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(-0.0),
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        -5.0..5.0f64,
        -1e3..1e3f64,
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn random_expressions_match_eval(
        e in ex_strategy(),
        pts in proptest::collection::vec(proptest::collection::vec(point(), 3), 8),
    ) {
        let mut dag = Dag::new(100_000);
        let o = build_ex(&mut dag, &e);
        let en = entry(vec![o]);
        let jit = compile(&[(EntryKind::F, &en)], &dag).expect("compile");
        let f = jit.plain(&EntryKind::F).expect("fn");
        for x in &pts {
            let p = [x[2], x[0], 0.75];
            let mut want = [0.0];
            eval(&dag, &[o], x, &p, &[], &mut want);
            let mut got = [1.0];
            // SAFETY: x has 3, p has 3 and out has 1 element.
            unsafe { f(x.as_ptr(), p.as_ptr(), got.as_mut_ptr()) };
            prop_assert!(same(got[0], want[0]), "{:?} at {:?}: jit {:?} vs eval {:?}", e, x, got[0], want[0]);
        }
    }
}

// ---- CompiledModel ----

mod compiled {
    use crate::{CompiledModel, EntryFn, JitError, module};
    use pc_ir::check::{close, scale, ulp};
    use pc_ir::{EntryKind, Fallback, Slot};
    use pc_ode::{MapLoader, Severity, SourceMap, load};

    use super::Lcg;

    fn setup(src: &str) -> (CompiledModel, Fallback) {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        assert!(
            l.diagnostics.iter().all(|d| d.severity != Severity::Error),
            "{:?}",
            l.diagnostics
        );
        let ir = match pc_ir::lower(&l.expanded, &l.info) {
            Ok(m) => m,
            Err(e) => panic!("lower failed: {e:?}"),
        };
        let fb = Fallback::new(&l.expanded, &l.info, &ir);
        (CompiledModel::new(ir), fb)
    }

    const SIMPLE: &str = "par a=1.3, b=0.7\ninit x=0.4, y=0.3\naux s=x*y+a\n\
        x'=a*x*y-x^3+b*y^2-x\ny'=x^2*y-y^4+a*b*x\ndone\n";
    /// Nonautonomous with parameters and a trig/exp mix.
    const NONAUTO: &str = "par a=1.3, b=0.7, w=2\ninit x=0.4, y=0.3\naux s=x*y+sin(a*t)\n\
        x'=a*sin(w*t)*x-x^3+b*y^2+exp(-y)/(1+x^2)\ny'=b*cos(t)-y+tanh(a*y)+x*y\ndone\n";
    /// Aux and a lookup table.
    const TABLES: &str = "par a=2, c=0.5\ntable h % 5 -2 2 t^2+sin(t)\ninit x=0.5, y=0.2\n\
        aux s=h(x)+c*y\nx'=-a*x+h(y)+c\ny'=x-y^3+h(x)*c\ndone\n";

    fn ulp_close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 4.0 * ulp(a.abs().max(b.abs()))
    }

    fn all_kinds() -> Vec<EntryKind> {
        let mut k = vec![
            EntryKind::F,
            EntryKind::Aux,
            EntryKind::JacU,
            EntryKind::JacP,
            EntryKind::Dir(vec![Slot::U]),
            EntryKind::Dir(vec![Slot::U, Slot::U]),
            EntryKind::Dir(vec![Slot::Full, Slot::P]),
            EntryKind::Dir(vec![Slot::U, Slot::P, Slot::Full]),
            EntryKind::Dir(vec![Slot::P, Slot::U, Slot::U, Slot::Full]),
            EntryKind::Dir(vec![Slot::U; 5]),
        ];
        k.shrink_to_fit();
        k
    }

    fn vs_fallback(src: &str, seed: u64) {
        let (mut cm, fb) = setup(src);
        let (n, n_p) = (cm.ir().n(), cm.ir().n_p());
        let mut r = Lcg(seed);
        for _ in 0..8 {
            let x: Vec<f64> = (0..n).map(|_| r.range(-1.5, 1.5)).collect();
            let mut p: Vec<f64> = (0..n_p).map(|_| r.range(0.3, 2.0)).collect();
            p.push(r.range(0.0, 2.0));
            for kind in all_kinds() {
                if matches!(kind, EntryKind::Aux) && cm.ir().aux.is_empty() {
                    continue;
                }
                let len = match &kind {
                    EntryKind::Dir(s) => s.iter().map(|s| s.len(n, n_p)).sum(),
                    _ => 0,
                };
                let v: Vec<f64> = (0..len).map(|_| r.range(-1.0, 1.0)).collect();
                let out_len = cm.ir.entry(&kind).expect("entry").outputs.len();
                let mut got = vec![f64::MAX; out_len];
                let mut want = vec![f64::NAN; out_len];
                cm.call(&kind, &x, &p, &v, &mut got).expect("call");
                fb.eval(&kind, &x, &p, &v, &mut want);
                let s = scale(&want);
                for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                    let ok = if matches!(kind, EntryKind::F) {
                        ulp_close(*a, *b)
                    } else {
                        close(*a, *b, s)
                    };
                    assert!(
                        ok,
                        "{kind:?} out {k} at x={x:?} p={p:?}: jit {a:?} vs fb {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn jit_matches_fallback_on_a_simple_model() {
        vs_fallback(SIMPLE, 1);
    }

    #[test]
    fn jit_matches_fallback_on_a_nonautonomous_model() {
        vs_fallback(NONAUTO, 2);
    }

    #[test]
    fn jit_matches_fallback_with_aux_and_tables() {
        vs_fallback(TABLES, 3);
    }

    #[test]
    fn the_second_request_returns_the_same_pointer() {
        let (mut cm, _) = setup(SIMPLE);
        let a = cm.entry(&EntryKind::F).expect("F");
        let sets = cm.sets.len();
        let b = cm.entry(&EntryKind::F).expect("F again");
        assert_eq!(a, b);
        assert!(matches!(a, EntryFn::Plain(_)));
        assert_eq!(cm.sets.len(), sets, "no recompilation");
        let d = EntryKind::Dir(vec![Slot::U, Slot::P]);
        let c = cm.entry(&d).expect("Dir");
        assert!(matches!(c, EntryFn::Dir(_)));
        assert_eq!(c, cm.entry(&d).expect("Dir again"));
    }

    #[test]
    fn many_kinds_get_distinct_functions() {
        let (mut cm, _) = setup(SIMPLE);
        let kinds = all_kinds();
        let mut fns = Vec::new();
        for k in &kinds {
            fns.push(cm.entry(k).expect("entry"));
        }
        assert_eq!(cm.sets.len(), kinds.len());
        for i in 0..fns.len() {
            for j in i + 1..fns.len() {
                assert_ne!(fns[i], fns[j], "{:?} vs {:?}", kinds[i], kinds[j]);
            }
        }
        for (k, f) in kinds.iter().zip(&fns) {
            assert_eq!(*f, cm.entry(k).expect("again"));
        }
    }

    #[test]
    fn dropping_the_model_frees_every_module() {
        let before = module::FREED.with(|c| c.get());
        let (mut cm, _) = setup(SIMPLE);
        let kinds = all_kinds();
        for k in &kinds {
            cm.entry(k).expect("entry");
        }
        assert_eq!(module::FREED.with(|c| c.get()), before);
        drop(cm);
        assert_eq!(module::FREED.with(|c| c.get()), before + kinds.len());
    }

    #[test]
    fn a_failed_compile_keeps_the_cache() {
        let (mut cm, _) = setup(SIMPLE);
        let f = cm.entry(&EntryKind::F).expect("F");
        let before = module::FREED.with(|c| c.get());
        module::FAIL_AFTER_NEW.with(|x| x.set(true));
        let r = cm.entry(&EntryKind::JacU);
        module::FAIL_AFTER_NEW.with(|x| x.set(false));
        assert!(matches!(r, Err(JitError::Cranelift(_))));
        assert_eq!(module::FREED.with(|c| c.get()), before + 1);
        assert_eq!(cm.entry(&EntryKind::F).expect("F"), f);
        assert_eq!(cm.sets.len(), 1);
        // The failed kind compiles later, and the old function still works.
        cm.entry(&EntryKind::JacU).expect("retry");
        let mut out = [0.0; 2];
        cm.call(&EntryKind::F, &[0.4, 0.3], &[1.3, 0.7, 0.0], &[], &mut out)
            .expect("call");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn an_ir_error_is_reported_and_leaves_the_cache() {
        let (mut cm, _) = setup(SIMPLE);
        cm.entry(&EntryKind::F).expect("F");
        let r = cm.entry(&EntryKind::Dir(vec![]));
        assert!(matches!(r, Err(JitError::Ir(_))), "{r:?}");
        assert_eq!(cm.sets.len(), 1);
    }

    #[test]
    fn call_checks_every_length() {
        let (mut cm, _) = setup(SIMPLE);
        let (x, p) = ([0.4, 0.3], [1.3, 0.7, 0.0]);
        let mut out = [0.0; 2];
        let bad = |r: Result<(), JitError>, name: &str, msg: &str| match r {
            Err(JitError::BadLength(m)) => {
                assert!(m.starts_with(name) && m.contains(msg), "{m}");
            }
            other => panic!("expected BadLength({name}), got {other:?}"),
        };
        let f = EntryKind::F;
        bad(
            cm.call(&f, &x[..1], &p, &[], &mut out),
            "x",
            "has 1 values, expected 2",
        );
        bad(
            cm.call(&f, &x, &p[..2], &[], &mut out),
            "p",
            "has 2 values, expected 3",
        );
        bad(
            cm.call(&f, &x, &p, &[1.0], &mut out),
            "v",
            "has 1 values, expected 0",
        );
        bad(
            cm.call(&f, &x, &p, &[], &mut out[..1]),
            "out",
            "has 1 values, expected 2",
        );
        // Dir [U, P] needs v of 2 + 2 = 4.
        let d = EntryKind::Dir(vec![Slot::U, Slot::P]);
        bad(
            cm.call(&d, &x, &p, &[0.0; 3], &mut out),
            "v",
            "has 3 values, expected 4",
        );
        bad(
            cm.call(&d, &x, &p, &[], &mut out),
            "v",
            "has 0 values, expected 4",
        );
        bad(
            cm.call(&d, &x, &p, &[0.0; 4], &mut out[..0]),
            "out",
            "has 0 values, expected 2",
        );
        cm.call(&d, &x, &p, &[0.0; 4], &mut out)
            .expect("Dir happy path");
        assert_eq!(out, [0.0, 0.0]);
        cm.call(&f, &x, &p, &[], &mut out).expect("F happy path");
        assert!(out[0] != 0.0 || out[1] != 0.0);
    }
}

#[test]
fn bessel_derivative_with_truncated_orders_matches_eval_and_finite_differences() {
    let src = "par a=0.5\nx'=besselj(a,x)+bessely(-1.5,x+3)+besselj(0.9,x)\ndone\n";
    let mut ir = build_model(src);
    let kinds = [
        EntryKind::F,
        EntryKind::JacU,
        EntryKind::Dir(vec![Slot::U, Slot::U]),
    ];
    let es: Vec<Entry> = kinds.iter().map(|k| ir.entry(k).expect("entry")).collect();
    let pairs: Vec<(EntryKind, &Entry)> = kinds.iter().cloned().zip(es.iter()).collect();
    let jit = compile(&pairs, &ir.dag).expect("compile");
    let f: PlainFn = jit.plain(&EntryKind::F).expect("F");
    let ju: PlainFn = jit.plain(&EntryKind::JacU).expect("JacU");
    let d2: DirFn = jit.dir(&kinds[2]).expect("Dir");
    for a in [0.5, -0.5, 0.9, -0.9, 1.5, -1.5, 2.0, 0.0] {
        for x in [0.7, 1.3, 2.9] {
            let p = [a, 0.0];
            let (mut got, mut want) = ([0.0], [0.0]);
            // SAFETY: x has 1 value, p has 2 and out has 1.
            unsafe { ju([x].as_ptr(), p.as_ptr(), got.as_mut_ptr()) };
            eval(&ir.dag, &es[1].outputs, &[x], &p, &[], &mut want);
            assert!(
                same(got[0], want[0]),
                "JacU a={a} x={x}: {got:?} vs {want:?}"
            );
            let h = 1e-5;
            let at = |x: f64| {
                let mut o = [0.0];
                // SAFETY: as above.
                unsafe { f([x].as_ptr(), p.as_ptr(), o.as_mut_ptr()) };
                o[0]
            };
            let fd = (at(x + h) - at(x - h)) / (2.0 * h);
            assert!(
                (got[0] - fd).abs() <= 1e-7 * fd.abs().max(1.0),
                "JacU a={a} x={x}: jit {} vs fd {fd}",
                got[0]
            );
            let v = [1.0, 1.0];
            // SAFETY: v has the packed length of two U slots.
            unsafe { d2([x].as_ptr(), p.as_ptr(), v.as_ptr(), got.as_mut_ptr()) };
            eval(&ir.dag, &es[2].outputs, &[x], &p, &v, &mut want);
            assert!(
                same(got[0], want[0]),
                "Dir a={a} x={x}: {got:?} vs {want:?}"
            );
        }
    }
}
