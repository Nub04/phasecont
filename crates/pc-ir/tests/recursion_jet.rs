//! Jet units of recursive functions (SPEC section 8af-5): directional entries of order 1 to
//! 5 through calls, their values, their call counts in the frame machine, and a bitwise
//! comparison with the same body inlined to a fixed depth.

use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId};
use pc_ir::eval::try_eval_entry;
use pc_ir::{EntryKind, IrErrorKind, IrModel, Slot, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

/// A model with one state `x`, parameters `a` and `m`, and a DAG that holds 100,000
/// nodes. The right-hand side is replaced by each test.
fn base() -> IrModel {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", "par a=0.7,m=4\ninit x=0.9\nx'=a*x+m\ndone\n");
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
    let ir = lower(&l.expanded, &l.info).expect("lower");
    assert_eq!((ir.n(), ir.n_p()), (1, 2));
    ir
}

/// `pw(x, n) = if(n<1)then(1)else(x*pw(x, n-1))`.
fn pw(g: &mut Dag) -> u32 {
    let u = g.add_unit("pw", 2, 1);
    let (x, n) = (g.arg(0), g.arg(1));
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(u, &[x, m]);
    let e = g.mul(x, r);
    let body = g.select(c, one, e);
    assert!(g.set_body(u, &[body]));
    u
}

/// `f(n) = if(n<1)then(1)else(x*f(n-1))`, whose body reads the state `x`.
fn fx(g: &mut Dag) -> u32 {
    let u = g.add_unit("f", 1, 1);
    let n = g.arg(0);
    let x = g.var(0);
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(u, &[m]);
    let e = g.mul(x, r);
    let body = g.select(c, one, e);
    assert!(g.set_body(u, &[body]));
    u
}

/// Evaluates the Dir entry for `slots` and returns its value and the calls it made.
fn dir(ir: &mut IrModel, slots: &[Slot], x: f64, p: [f64; 2], v: &[f64]) -> (f64, u64) {
    let e = match ir.entry(&EntryKind::Dir(slots.to_vec())) {
        Ok(e) => e,
        Err(e) => panic!("entry {slots:?}: {e}"),
    };
    let mut out = [f64::NAN];
    let st = try_eval_entry(&ir.dag, &e, &[x], &p, v, &mut out).expect("evaluates");
    (out[0], st.calls)
}

/// The length of `v` for `slots` with one state and two parameters.
fn vlen(slots: &[Slot]) -> usize {
    slots.iter().map(|s| s.len(1, 2)).sum()
}

/// The second derivative of `x' = pw(x, n)` along `[U, U]` is `n (n-1) x^(n-2)` with
/// n + 1 calls; the first and third derivatives make n + 1 calls too. At `x = 0.5` every
/// partial result is a small integer times a power of two, so the values are exact.
#[test]
fn dir_of_pw_gives_closed_forms_with_n_plus_1_calls() {
    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (x, m) = (ir.dag.var(0), ir.dag.param(1));
    ir.f[0] = ir.dag.call(u, &[x, m]);
    for n in [0u32, 1, 2, 3, 7, 100, 1000] {
        let nf = f64::from(n);
        let xp = |k: i32| 0.5f64.powi(n as i32 - k);
        let (d1, c1) = dir(&mut ir, &[Slot::U], 0.5, [0.7, nf], &[1.0]);
        let (d2, c2) = dir(&mut ir, &[Slot::U, Slot::U], 0.5, [0.7, nf], &[1.0, 1.0]);
        let w1 = if n >= 1 { nf * xp(1) } else { 0.0 };
        let w2 = if n >= 2 { nf * (nf - 1.0) * xp(2) } else { 0.0 };
        assert_eq!(d1, w1, "first derivative at n = {n}");
        assert_eq!(d2, w2, "second derivative at n = {n}");
        assert_eq!(c1, u64::from(n) + 1, "first derivative calls at n = {n}");
        assert_eq!(c2, u64::from(n) + 1, "second derivative calls at n = {n}");
    }
    let (d3, c3) = dir(&mut ir, &[Slot::U; 3], 0.5, [0.7, 100.0], &[1.0; 3]);
    assert_eq!(d3, 100.0 * 99.0 * 98.0 * 0.5f64.powi(97));
    assert_eq!(c3, 101);
    // Multilinear in the directions.
    let (d, _) = dir(&mut ir, &[Slot::U, Slot::U], 0.5, [0.7, 10.0], &[2.0, -3.0]);
    assert_eq!(d, -6.0 * 90.0 * 0.5f64.powi(8));
}

/// A body that reads the state: `x' = f(n)` gives `n (n-1) x^(n-2)` along `[U, U]`.
#[test]
fn dir_of_a_body_that_reads_a_state() {
    let mut ir = base();
    let u = fx(&mut ir.dag);
    let m = ir.dag.param(1);
    ir.f[0] = ir.dag.call(u, &[m]);
    for n in [0u32, 1, 2, 5, 100, 1000] {
        let nf = f64::from(n);
        let (d2, c) = dir(&mut ir, &[Slot::U, Slot::U], 0.5, [0.7, nf], &[1.0, 1.0]);
        let w = if n >= 2 {
            nf * (nf - 1.0) * 0.5f64.powi(n as i32 - 2)
        } else {
            0.0
        };
        assert_eq!(d2, w, "n = {n}");
        assert_eq!(c, u64::from(n) + 1, "calls at n = {n}");
    }
    // Along P the call reads no covered variable and its argument has no component: zero
    // with no call.
    let (d, c) = dir(&mut ir, &[Slot::P], 0.5, [0.7, 100.0], &[1.0, 1.0]);
    assert_eq!((d, c), (0.0, 0));
}

/// The value and its components come from one frame: `x' = x * pw(x, n)` along `[U]`
/// makes n + 1 calls, not 2(n + 1), since the top level is rebuilt on the jet call.
#[test]
fn top_level_value_and_components_share_one_call() {
    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (x, m) = (ir.dag.var(0), ir.dag.param(1));
    let c = ir.dag.call(u, &[x, m]);
    ir.f[0] = ir.dag.mul(x, c);
    for n in [100u32, 1000] {
        let nf = f64::from(n);
        let (d, calls) = dir(&mut ir, &[Slot::U], 0.5, [0.7, nf], &[1.0]);
        assert_eq!(d, (nf + 1.0) * 0.5f64.powi(n as i32), "n = {n}");
        assert_eq!(calls, u64::from(n) + 1, "calls at n = {n}");
    }
}

/// A call in a condition that is also on a derivative path, and calls under `exp` and
/// `pow`, still make n + 1 calls.
#[test]
fn call_in_a_condition_and_under_exp_and_pow() {
    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (x, m) = (ir.dag.var(0), ir.dag.param(1));
    let c = ir.dag.call(u, &[x, m]);
    let tenth = ir.dag.konst(0.1);
    let cond = ir.dag.cmp(Cmp::Gt, c, tenth);
    ir.f[0] = ir.dag.select(cond, c, x);
    let (d, calls) = dir(&mut ir, &[Slot::U, Slot::U], 1.0, [0.7, 100.0], &[1.0, 1.0]);
    assert_eq!((d, calls), (9900.0, 101));

    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (x, m) = (ir.dag.var(0), ir.dag.param(1));
    let c = ir.dag.call(u, &[x, m]);
    ir.f[0] = ir.dag.call1(Fn1::Exp, c);
    // d2/dx2 exp(x^n) at x = 1 is e (n (n-1) + n^2).
    let (d, calls) = dir(&mut ir, &[Slot::U, Slot::U], 1.0, [0.7, 10.0], &[1.0, 1.0]);
    let want = std::f64::consts::E * 190.0;
    assert!(
        (d - want).abs() <= 4.0 * f64::EPSILON * want,
        "{d} vs {want}"
    );
    assert_eq!(calls, 11);

    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (x, m) = (ir.dag.var(0), ir.dag.param(1));
    let c = ir.dag.call(u, &[x, m]);
    let a = ir.dag.param(0);
    ir.f[0] = ir.dag.call2(Fn2::Pow, c, a);
    // pow(x^n, a) = x^(n a); along [U, P] at x = 1 the mixed derivative is n.
    let (d, calls) = dir(
        &mut ir,
        &[Slot::U, Slot::P],
        1.0,
        [0.7, 10.0],
        &[1.0, 1.0, 0.0],
    );
    assert!((d - 10.0).abs() <= 8.0 * f64::EPSILON * 10.0, "{d}");
    assert_eq!(calls, 11);
}

/// A call whose arguments have no components and whose body reads no covered variable
/// has derivative 0, makes no call and registers no jet unit.
#[test]
fn zero_masks_give_zero_without_a_unit() {
    let mut ir = base();
    let u = pw(&mut ir.dag);
    let (a, m) = (ir.dag.param(0), ir.dag.param(1));
    ir.f[0] = ir.dag.call(u, &[a, m]);
    let units = ir.dag.units().len();
    let (d, c) = dir(&mut ir, &[Slot::U, Slot::U], 0.5, [0.7, 10.0], &[1.0, 1.0]);
    assert_eq!((d, c), (0.0, 0));
    assert_eq!(ir.dag.units().len(), units, "no jet unit for a zero call");
    // Along P the first formal moves: d pw(a, m) / da = m a^(m-1), and the counter m,
    // whose argument also moves, gets no component.
    let (d, c) = dir(&mut ir, &[Slot::P], 0.5, [0.5, 10.0], &[1.0, 1.0]);
    assert_eq!((d, c), (10.0 * 0.5f64.powi(9), 11));
}

/// `q(x, y, n) = if(n<1)then(y*x)else(sin(x)*q(x, y, n-1) + y/x)`.
fn q(g: &mut Dag) -> u32 {
    let u = g.add_unit("q", 3, 1);
    let (x, y, n) = (g.arg(0), g.arg(1), g.arg(2));
    let body = q_body(g, x, y, n, |g, x, y, m| g.call(u, &[x, y, m]));
    assert!(g.set_body(u, &[body]));
    u
}

fn q_body(
    g: &mut Dag,
    x: NodeId,
    y: NodeId,
    n: NodeId,
    rec: impl FnOnce(&mut Dag, NodeId, NodeId, NodeId) -> NodeId,
) -> NodeId {
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let base = g.mul(y, x);
    let m = g.sub(n, one);
    let r = rec(g, x, y, m);
    let s = g.call1(Fn1::Sin, x);
    let t = g.mul(s, r);
    let w = g.div(y, x);
    let e = g.add(t, w);
    g.select(c, base, e)
}

/// `q` inlined `depth` levels deep, with 0 below the last level.
fn q_inline(g: &mut Dag, x: NodeId, y: NodeId, n: NodeId, depth: u32) -> NodeId {
    if depth == 0 {
        return g.konst(0.0);
    }
    q_body(g, x, y, n, |g, x, y, m| q_inline(g, x, y, m, depth - 1))
}

/// The jet units give the same bits as the existing builder on the body inlined eight
/// levels deep, at generic directions, for orders 1 to 5 and mixed slots.
#[test]
fn jet_units_match_the_inlined_body_bitwise() {
    let mut call = base();
    let u = q(&mut call.dag);
    let (x, a, m) = (call.dag.var(0), call.dag.param(0), call.dag.param(1));
    call.f[0] = call.dag.call(u, &[x, a, m]);
    let mut inl = base();
    let (x, a, m) = (inl.dag.var(0), inl.dag.param(0), inl.dag.param(1));
    inl.f[0] = q_inline(&mut inl.dag, x, a, m, 8);
    let slot_sets: [&[Slot]; 10] = [
        &[Slot::U],
        &[Slot::P],
        &[Slot::Full],
        &[Slot::U, Slot::P],
        &[Slot::U, Slot::U],
        &[Slot::P, Slot::Full],
        &[Slot::U, Slot::P, Slot::Full],
        &[Slot::Full; 4],
        &[Slot::U, Slot::P, Slot::U, Slot::P, Slot::Full],
        &[Slot::U; 5],
    ];
    let vals = [
        0.3, -0.7, 1.1, 0.45, -1.3, 0.9, 0.2, -0.55, 1.7, -0.25, 0.65, 1.4, -0.85, 0.35,
    ];
    for slots in slot_sets {
        let v = &vals[..vlen(slots)];
        for n in [0.0, 1.0, 3.0, 6.0] {
            let (want, _) = dir(&mut inl, slots, 0.9, [0.7, n], v);
            let (got, calls) = dir(&mut call, slots, 0.9, [0.7, n], v);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{slots:?} at n = {n}: {got:e} vs {want:e}"
            );
            assert!(want.is_finite());
            assert_eq!(calls, n as u64 + 1, "{slots:?} at n = {n}");
        }
    }
}

/// A directional entry that reaches a function without a body is refused, leaves
/// nothing behind, and builds once the body is set.
#[test]
fn dir_through_a_missing_body_is_refused_then_builds() {
    let mut ir = base();
    let mb = ir.dag.add_unit("mb", 1, 1);
    let x = ir.dag.var(0);
    ir.f[0] = ir.dag.call(mb, &[x]);
    let units = ir.dag.units().len();
    for slots in [vec![Slot::U], vec![Slot::U, Slot::P], vec![Slot::Full; 3]] {
        for _ in 0..2 {
            let e = ir
                .entry(&EntryKind::Dir(slots.clone()))
                .expect_err("no NaN form");
            assert_eq!(e.kind, IrErrorKind::Unsupported);
            assert!(e.message.contains("`mb`"), "{}", e.message);
            assert_eq!(ir.dag.units().len(), units, "rolled back");
        }
    }
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(mb, &[sq]));
    let (d, c) = dir(&mut ir, &[Slot::U, Slot::U], 3.0, [0.7, 1.0], &[1.0, 1.0]);
    assert_eq!((d, c), (2.0, 1));
    let (d, _) = dir(&mut ir, &[Slot::U], 3.0, [0.7, 1.0], &[1.0]);
    assert_eq!(d, 6.0);
}

/// A directional entry through a function with two outputs is refused and names it.
#[test]
fn dir_through_several_outputs_is_refused() {
    let mut ir = base();
    let m2 = ir.dag.add_unit("m2", 1, 2);
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(m2, &[sq, a]));
    let x = ir.dag.var(0);
    let c = ir.dag.call(m2, &[x]);
    for f in [c, ir.dag.out(c, 1)] {
        ir.f[0] = f;
        for _ in 0..2 {
            let e = ir
                .entry(&EntryKind::Dir(vec![Slot::U, Slot::U]))
                .expect_err("refused");
            assert_eq!(e.kind, IrErrorKind::Unsupported);
            assert!(e.message.contains("`m2`"), "{}", e.message);
            assert!(e.message.contains("more than one output"), "{}", e.message);
        }
    }
}
