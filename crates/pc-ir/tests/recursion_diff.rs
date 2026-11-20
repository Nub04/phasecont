//! Tangent units of recursive functions (SPEC section 8af-5): first derivatives of calls
//! through `diff::d`, their values, and their call counts in the frame machine. The units
//! are built with the DAG API, since the lowering of `.ode` recursion comes later.

use pc_ir::dag::{Cmp, Dag, NodeId, Op};
use pc_ir::diff::{Wrt, d};
use pc_ir::machine::try_eval;
use std::collections::HashMap;

/// `pw(x, n) = if(n<1)then(1)else(x*pw(x, n-1))`; returns the unit.
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

/// `f(n) = if(n<1)then(1)else(x*f(n-1))`, whose body reads the state `x = u[0]`.
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

/// Evaluates `o` at the states `x` and parameters `p`; returns the value and the calls.
fn run(g: &Dag, o: NodeId, x: &[f64], p: &[f64]) -> (f64, u64) {
    let mut out = [f64::NAN];
    let st = try_eval(g, &[o], x, p, &[], &mut out).expect("evaluates");
    (out[0], st.calls)
}

fn is_zero(g: &Dag, id: NodeId) -> bool {
    matches!(g.op(id), Op::Const(b) if f64::from_bits(*b) == 0.0)
}

/// The JacU entry of `x' = pw(x, n)` is `n x^(n-1)` with n + 1 calls. At `x = 0.5` every
/// partial result `k 2^-(k-1)` is exact, so the value is compared exactly.
#[test]
fn jacu_of_pw_makes_n_plus_1_calls() {
    let mut g = Dag::new(100_000);
    let u = pw(&mut g);
    let (x, n) = (g.var(0), g.param(0));
    let o = g.call(u, &[x, n]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    for n in [0u32, 1, 2, 7, 100, 1000] {
        let nf = f64::from(n);
        let (v, calls) = run(&g, dx, &[0.5], &[nf]);
        let want = if n == 0 {
            0.0
        } else {
            nf * 0.5f64.powi(n as i32 - 1)
        };
        assert_eq!(v, want, "d pw / dx at n = {n}");
        assert_eq!(calls, u64::from(n) + 1, "calls at n = {n}");
    }
    let (v, _) = run(&g, dx, &[2.0], &[10.0]);
    assert_eq!(v, 10.0 * 512.0);
}

/// The value of the tangent call matches the dual-number recursion `v_k = x v_{k-1}`,
/// `d_k = v_{k-1} + d_{k-1} x` bit for bit at a point where rounding happens.
#[test]
fn jacu_of_pw_matches_the_dual_recursion_bitwise() {
    let mut g = Dag::new(100_000);
    let u = pw(&mut g);
    let (x, n) = (g.var(0), g.param(0));
    let o = g.call(u, &[x, n]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    let xv = 1.1f64;
    let (mut v, mut dv) = (1.0f64, 0.0f64);
    for _ in 0..37 {
        let nd = v + dv * xv;
        v *= xv;
        dv = nd;
    }
    let (got, _) = run(&g, dx, &[xv], &[37.0]);
    assert_eq!(got.to_bits(), dv.to_bits());
}

/// A body that reads a state: the JacU of `x' = f(n)` is `n x^(n-1)` with n + 1 calls,
/// though the argument of the call does not depend on `x`.
#[test]
fn jacu_of_a_body_that_reads_a_state() {
    let mut g = Dag::new(100_000);
    let u = fx(&mut g);
    let n = g.param(0);
    let o = g.call(u, &[n]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    for n in [0u32, 1, 5, 100, 1000] {
        let nf = f64::from(n);
        let (v, calls) = run(&g, o, &[0.5], &[nf]);
        assert_eq!(v, 0.5f64.powi(n as i32), "f at n = {n}");
        assert_eq!(calls, u64::from(n) + 1);
        let (v, calls) = run(&g, dx, &[0.5], &[nf]);
        let want = if n == 0 {
            0.0
        } else {
            nf * 0.5f64.powi(n as i32 - 1)
        };
        assert_eq!(v, want, "d f / dx at n = {n}");
        assert_eq!(calls, u64::from(n) + 1, "calls at n = {n}");
    }
}

/// A call whose argument tangents are all the constant 0, into a component that does not
/// read `w`, has derivative 0 and builds no unit.
#[test]
fn zero_tangents_and_no_read_give_zero_without_a_unit() {
    let mut g = Dag::new(10_000);
    let u = fx(&mut g);
    let n = g.param(0);
    let o = g.call(u, &[n]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let d1 = d(&mut g, &mut memo, o, Wrt::U(1));
    assert!(is_zero(&g, d1));
    assert_eq!(g.units().len(), units);
    let upw = pw(&mut g);
    let x1 = g.var(1);
    let o2 = g.call(upw, &[x1, n]);
    let units = g.units().len();
    let d2 = d(&mut g, &mut memo, o2, Wrt::U(0));
    assert!(is_zero(&g, d2));
    assert_eq!(g.units().len(), units);
}

/// `x' = if(k<0)then(0)else(f(k))` at k = -1: the derivative is 0 with no call, since the
/// tangent `If` is as lazy as the value.
#[test]
fn untaken_arm_of_the_derivative_makes_no_call() {
    let mut g = Dag::new(10_000);
    let u = fx(&mut g);
    let k = g.param(0);
    let zero = g.konst(0.0);
    let c = g.cmp(Cmp::Lt, k, zero);
    let r = g.call(u, &[k]);
    let o = g.select(c, zero, r);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(run(&g, dx, &[0.5], &[-1.0]), (0.0, 0));
    assert_eq!(run(&g, dx, &[0.5], &[3.0]), (0.75, 4));
}

/// `h(a, b) = if(b<1)then(a)else(b*h(a, b-1))`, so `h(a, b) = a b!`; `x' = h(x, 2) +
/// h(2, x)` uses two tangent units, one per set of active formals.
#[test]
fn two_active_sets_of_one_function() {
    let mut g = Dag::new(10_000);
    let u = g.add_unit("h", 2, 1);
    let (a, b) = (g.arg(0), g.arg(1));
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, b, one);
    let m = g.sub(b, one);
    let r = g.call(u, &[a, m]);
    let e = g.mul(b, r);
    let body = g.select(c, a, e);
    assert!(g.set_body(u, &[body]));
    let x = g.var(0);
    let two = g.konst(2.0);
    let h1 = g.call(u, &[x, two]);
    let h2 = g.call(u, &[two, x]);
    let o = g.add(h1, h2);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 2);
    // h(3, 2) = 6 and h(2, 3) = 12. d h(x, 2)/dx = 2. d h(2, b)/db follows
    // d_k = h(2, k-1) + k d_{k-1} from d_0 = 0: 2, 6, 22.
    assert_eq!(run(&g, o, &[3.0], &[]).0, 18.0);
    let (v, calls) = run(&g, dx, &[3.0], &[]);
    assert_eq!(v, 24.0);
    assert_eq!(calls, 3 + 4);
    // A second derivative request reuses the units.
    let dx2 = d(&mut g, &mut memo, h1, Wrt::U(0));
    assert_eq!(g.units().len(), units + 2);
    assert_eq!(run(&g, dx2, &[3.0], &[]).0, 2.0);
}

/// Two functions of different arity share the node `Arg(0)*Arg(0)`. Their tangent units
/// read the tangent of formal 0 from different slots (formal 1 of `p`, formal 2 of `q`),
/// so the rebuilt bodies must not share a derivative of that node.
#[test]
fn shared_body_node_of_two_arities() {
    // p(2.5) = 0.5^2 and q(2.5, 2) = 2.5^2, so d/dx = 2 * 0.5 + 3 * 2 * 2.5.
    let (mut g, cp, cq) = shared();
    let three = g.konst(3.0);
    let t = g.mul(three, cq);
    let o = g.add(cp, t);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(run(&g, o, &[2.5], &[]).0, 0.25 + 3.0 * 6.25);
    assert_eq!(run(&g, dx, &[2.5], &[]).0, 1.0 + 15.0);
    // In the other order, on a fresh DAG, the tangent body of q is built first.
    let (mut g, cp, cq) = shared();
    let mut memo = HashMap::new();
    let dq = d(&mut g, &mut memo, cq, Wrt::U(0));
    let dp = d(&mut g, &mut memo, cp, Wrt::U(0));
    assert_eq!(run(&g, dq, &[2.5], &[]).0, 5.0);
    assert_eq!(run(&g, dp, &[2.5], &[]).0, 1.0);
}

/// `p(a) = if(a<1)then(a*a)else(p(a-1))` and `q(a, b) = if(b<1)then(a*a)else(q(a, b-1))`
/// share the body node `Arg0 * Arg0`. Returns the DAG and the calls `p(x)`, `q(x, 2)`.
fn shared() -> (Dag, NodeId, NodeId) {
    let mut g = Dag::new(10_000);
    let p = g.add_unit("p", 1, 1);
    let q = g.add_unit("q", 2, 1);
    let a = g.arg(0);
    let b = g.arg(1);
    let one = g.konst(1.0);
    let sq = g.mul(a, a);
    let ca = g.cmp(Cmp::Lt, a, one);
    let am = g.sub(a, one);
    let rp = g.call(p, &[am]);
    let bp = g.select(ca, sq, rp);
    assert!(g.set_body(p, &[bp]));
    let cb = g.cmp(Cmp::Lt, b, one);
    let bm = g.sub(b, one);
    let rq = g.call(q, &[a, bm]);
    let bq = g.select(cb, sq, rq);
    assert!(g.set_body(q, &[bq]));
    let x = g.var(0);
    let two = g.konst(2.0);
    let cp = g.call(p, &[x]);
    let cq = g.call(q, &[x, two]);
    (g, cp, cq)
}

/// A call that only feeds a condition gets no tangent unit: in `h(a, n) =
/// if(cnt(n)<2)then(a)else(a*a)`, `cnt(n) = if(n<1)then(0)else(1+cnt(n-1))` stays a primal
/// call inside the tangent body of `h`.
#[test]
fn call_in_a_condition_has_no_tangent_unit() {
    let mut g = Dag::new(10_000);
    let cnt = g.add_unit("cnt", 1, 1);
    let n = g.arg(0);
    let (zero, one, two) = (g.konst(0.0), g.konst(1.0), g.konst(2.0));
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(cnt, &[m]);
    let s = g.add(one, r);
    let bc = g.select(c, zero, s);
    assert!(g.set_body(cnt, &[bc]));
    let h = g.add_unit("h", 2, 1);
    let (a, n) = (g.arg(0), g.arg(1));
    let k = g.call(cnt, &[n]);
    let ck = g.cmp(Cmp::Lt, k, two);
    let sq = g.mul(a, a);
    let bh = g.select(ck, a, sq);
    assert!(g.set_body(h, &[bh]));
    let (x, p) = (g.var(0), g.param(0));
    let o = g.call(h, &[x, p]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 1, "only h gets a tangent unit");
    // One call of the tangent of h, then n + 1 calls of cnt.
    assert_eq!(run(&g, dx, &[3.0], &[1.0]), (1.0, 3));
    assert_eq!(run(&g, dx, &[3.0], &[5.0]), (6.0, 7));
}

/// A node cap hit while the tangent units are built leaves the DAG flagged; restoring the
/// checkpoint drops the half-built units and their derived entries, and a rebuild with a
/// fresh memo and a larger cap gives the right derivative.
#[test]
fn cap_hit_while_building_a_tangent_unit_restores_cleanly() {
    let mut g = Dag::new(100_000);
    let u = pw(&mut g);
    let (x, n) = (g.var(0), g.param(0));
    let o = g.call(u, &[x, n]);
    let units = g.units().len();
    let cap = g.cap();
    let cp = g.checkpoint();
    g.set_cap(g.len() + 2);
    let mut memo = HashMap::new();
    let _ = d(&mut g, &mut memo, o, Wrt::U(0));
    assert!(g.exceeded(), "the cap is hit");
    g.restore(&cp);
    g.set_cap(cap);
    assert!(!g.exceeded());
    assert_eq!(g.units().len(), units);
    // pw reads no state and only its formal 0 has a tangent: key [tag, unit, 0, 0, mask].
    let key = [0, u, 0, 0, 0];
    assert_eq!(g.derived(&key), None);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert!(!g.exceeded());
    assert_eq!(g.derived(&key), Some(units as u32));
    assert_eq!(run(&g, dx, &[0.5], &[7.0]), (7.0 * 0.5f64.powi(6), 8));
}

/// Mutual recursion: `ev(n) = if(n<1)then(x)else(od(n-1))`, `od(n) = if(n<1)then(0)else(
/// ev(n-1))`, so `ev(n)` is `x` for even n and 0 for odd n, and so is its derivative.
#[test]
fn mutual_recursion_tangent() {
    let mut g = Dag::new(10_000);
    let ev = g.add_unit("ev", 1, 1);
    let od = g.add_unit("od", 1, 1);
    let n = g.arg(0);
    let x = g.var(0);
    let (zero, one) = (g.konst(0.0), g.konst(1.0));
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let ro = g.call(od, &[m]);
    let be = g.select(c, x, ro);
    assert!(g.set_body(ev, &[be]));
    let re = g.call(ev, &[m]);
    let bo = g.select(c, zero, re);
    assert!(g.set_body(od, &[bo]));
    let k = g.param(0);
    let o = g.call(ev, &[k]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    for n in 0..8u32 {
        let want = if n % 2 == 0 { 1.0 } else { 0.0 };
        let (v, calls) = run(&g, dx, &[4.0], &[f64::from(n)]);
        assert_eq!(v, want, "n = {n}");
        assert_eq!(calls, u64::from(n) + 1);
    }
}

/// The JacP column of a call: `pw(x, n)` with `x = p[0]`.
#[test]
fn jacp_of_pw() {
    let mut g = Dag::new(10_000);
    let u = pw(&mut g);
    let (x, n) = (g.param(0), g.param(1));
    let o = g.call(u, &[x, n]);
    let mut memo = HashMap::new();
    let dp = d(&mut g, &mut memo, o, Wrt::P(0));
    let (v, calls) = run(&g, dp, &[], &[2.0, 5.0]);
    assert_eq!(v, 5.0 * 16.0);
    assert_eq!(calls, 6);
    let dn = d(&mut g, &mut memo, o, Wrt::P(1));
    assert_eq!(run(&g, dn, &[], &[2.0, 5.0]).0, 0.0);
}
