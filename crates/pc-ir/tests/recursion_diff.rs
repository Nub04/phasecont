//! Tangent units of recursive functions (SPEC section 8af-5): first derivatives of calls
//! through `diff::d`, their values, and their call counts in the frame machine. The units
//! are built with the DAG API, since the lowering of `.ode` recursion comes later.

use pc_ir::dag::{Cmp, Dag, Fn1, NodeId, Op};
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

/// A rule that reads its own node (exp) inside a recursive body: `e(n) =
/// if(n<1)then(1)else(exp(x*e(n-1)))`. The derivative follows the dual recursion
/// `d_k = (v_{k-1} + d_{k-1} x) v_k` with n + 1 calls, so the exp of each frame is the
/// value output of its tangent call and is not evaluated again.
#[test]
fn exp_of_a_recursive_call_reuses_the_frame_value() {
    let mut g = Dag::new(100_000);
    let u = g.add_unit("e", 1, 1);
    let n = g.arg(0);
    let x = g.var(0);
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(u, &[m]);
    let xr = g.mul(x, r);
    let e = g.call1(Fn1::Exp, xr);
    let body = g.select(c, one, e);
    assert!(g.set_body(u, &[body]));
    let k = g.param(0);
    let o = g.call(u, &[k]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    let xv = 0.1f64;
    for n in [1u32, 5, 100] {
        let (mut v, mut dv) = (1.0f64, 0.0f64);
        for _ in 0..n {
            let nv = (xv * v).exp();
            dv = (v + dv * xv) * nv;
            v = nv;
        }
        let (got, calls) = run(&g, dx, &[xv], &[f64::from(n)]);
        assert_eq!(got.to_bits(), dv.to_bits(), "d e / dx at n = {n}");
        assert_eq!(calls, u64::from(n) + 1, "calls at n = {n}");
    }
}

/// The active set changes down the recursion: in `s(a, b, n) =
/// if(n<1)then(a)else(s(b, a, n-1))` with only `a` active, the inner call has only `b`
/// active, and the call after that has `a` again. Two units, n + 1 calls.
#[test]
fn active_set_that_alternates_down_the_recursion() {
    let mut g = Dag::new(10_000);
    let s = g.add_unit("s", 3, 1);
    let (a, b, n) = (g.arg(0), g.arg(1), g.arg(2));
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(s, &[b, a, m]);
    let body = g.select(c, a, r);
    assert!(g.set_body(s, &[body]));
    let (x, two, k) = (g.var(0), g.konst(2.0), g.param(0));
    let o = g.call(s, &[x, two, k]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 2);
    for n in 0u32..6 {
        let want = if n % 2 == 0 { 1.0 } else { 0.0 };
        assert_eq!(
            run(&g, dx, &[3.0], &[f64::from(n)]),
            (want, u64::from(n) + 1)
        );
    }
}

/// Two calls in one body: `fb(x, n) = if(n<2)then(x)else(fb(x, n-1)+fb(x, n-2))` is x
/// times the number of leaves, and its derivative makes exactly the primal number of calls.
#[test]
fn two_calls_in_one_body() {
    let mut g = Dag::new(10_000);
    let f = g.add_unit("fb", 2, 1);
    let (a, n) = (g.arg(0), g.arg(1));
    let (one, two) = (g.konst(1.0), g.konst(2.0));
    let c = g.cmp(Cmp::Lt, n, two);
    let m1 = g.sub(n, one);
    let m2 = g.sub(n, two);
    let r1 = g.call(f, &[a, m1]);
    let r2 = g.call(f, &[a, m2]);
    let sm = g.add(r1, r2);
    let body = g.select(c, a, sm);
    assert!(g.set_body(f, &[body]));
    let (x, k) = (g.var(0), g.param(0));
    let o = g.call(f, &[x, k]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    let (v, primal) = run(&g, o, &[1.5], &[10.0]);
    assert_eq!(v, 1.5 * 89.0);
    assert_eq!(run(&g, dx, &[1.5], &[10.0]), (89.0, primal));
}

/// A recursive call inside the argument of another call: `g(x, n) =
/// if(n<1)then(x)else(pw(g(x, n-1), 2))` is `x^(2^n)`. At x = 0.5 and n = 3 the
/// derivative `8 x^7` is exact, and the tangent makes the primal number of calls.
#[test]
fn call_inside_a_call_argument() {
    let mut g = Dag::new(10_000);
    let p = pw(&mut g);
    let q = g.add_unit("g", 2, 1);
    let (a, n) = (g.arg(0), g.arg(1));
    let (one, two) = (g.konst(1.0), g.konst(2.0));
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(q, &[a, m]);
    let sq = g.call(p, &[r, two]);
    let body = g.select(c, a, sq);
    assert!(g.set_body(q, &[body]));
    let (x, k) = (g.var(0), g.param(0));
    let o = g.call(q, &[x, k]);
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    let (v, primal) = run(&g, o, &[0.5], &[3.0]);
    assert_eq!((v, primal), (0.5f64.powi(8), 4 + 3 * 3));
    assert_eq!(run(&g, dx, &[0.5], &[3.0]), (8.0 * 0.5f64.powi(7), primal));
}

/// `Wrt::P` with a body that reads the parameter: `f(n) = if(n<1)then(1)else(p*f(n-1))`
/// gives `n p^(n-1)` with n + 1 calls.
#[test]
fn jacp_of_a_body_that_reads_the_parameter() {
    let mut g = Dag::new(10_000);
    let u = g.add_unit("f", 1, 1);
    let n = g.arg(0);
    let pp = g.param(0);
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let r = g.call(u, &[m]);
    let e = g.mul(pp, r);
    let body = g.select(c, one, e);
    assert!(g.set_body(u, &[body]));
    let k = g.param(1);
    let o = g.call(u, &[k]);
    let mut memo = HashMap::new();
    let dp = d(&mut g, &mut memo, o, Wrt::P(0));
    let units = g.units().len();
    let dk = d(&mut g, &mut memo, o, Wrt::P(1));
    assert_eq!(g.units().len(), units, "d/dn adds no unit");
    for n in [0u32, 1, 6, 100] {
        let want = if n == 0 {
            0.0
        } else {
            f64::from(n) * 0.5f64.powi(n as i32 - 1)
        };
        assert_eq!(
            run(&g, dp, &[], &[0.5, f64::from(n)]),
            (want, u64::from(n) + 1)
        );
    }
    // n only feeds the conditions and the counter, so its formal is inactive and the body
    // does not read p[1]: d/dn is the constant 0 with no call (SPEC 8af-5).
    assert!(is_zero(&g, dk));
}

/// A counter formal is inactive even when its argument moves: in `pw(x, x)` the second
/// formal only feeds the condition and the counter, so the derivative treats n as fixed,
/// `n x^(n-1)`, and the tangent unit is the one of `pw(x, n)`, with only formal 0.
#[test]
fn counter_formal_gets_no_tangent_even_when_its_argument_moves() {
    let mut g = Dag::new(10_000);
    let u = pw(&mut g);
    let x = g.var(0);
    let o = g.call(u, &[x, x]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 1);
    assert_eq!(g.derived(&[0, u, 0, 0, 0]), Some(units as u32));
    assert_eq!(run(&g, dx, &[3.0], &[]), (27.0, 4));
}

/// A state read only in a condition: `cz(n) = if(x<n)then(1)else(1+cz(n-1))` is piecewise
/// constant in x, so its derivative along x is the constant 0 with no unit.
#[test]
fn state_read_only_in_a_condition_gives_zero_without_a_unit() {
    let mut g = Dag::new(10_000);
    let u = g.add_unit("cz", 1, 1);
    let n = g.arg(0);
    let x = g.var(0);
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, x, n);
    let m = g.sub(n, one);
    let r = g.call(u, &[m]);
    let s = g.add(one, r);
    let body = g.select(c, one, s);
    assert!(g.set_body(u, &[body]));
    let k = g.param(0);
    let o = g.call(u, &[k]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert!(is_zero(&g, dx));
    assert_eq!(g.units().len(), units);
}

/// `h(a) = if(a<1)then(3)else(h(a-1))`, recursive and active in its formal.
fn h3(g: &mut Dag) -> u32 {
    let u = g.add_unit("h", 1, 1);
    let a = g.arg(0);
    let (one, three) = (g.konst(1.0), g.konst(3.0));
    let c = g.cmp(Cmp::Lt, a, one);
    let m = g.sub(a, one);
    let r = g.call(u, &[m]);
    let body = g.select(c, three, r);
    assert!(g.set_body(u, &[body]));
    u
}

/// `pw(x, h(x))`: the argument of the counter formal is a recursive call. Its tangent is
/// never built, so `h` runs primal only and gets no tangent unit, at the top level and
/// inside a unit body. At x = 2.5, h(x) = 3 after 3 calls and pw makes 4.
#[test]
fn call_in_a_counter_argument_runs_primal_only() {
    let mut g = Dag::new(10_000);
    let p = pw(&mut g);
    let h = h3(&mut g);
    let x = g.var(0);
    let hx = g.call(h, &[x]);
    let o = g.call(p, &[x, hx]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 1, "only pw gets a tangent unit");
    assert_eq!(run(&g, dx, &[2.5], &[]), (18.75, 4 + 3));
    assert_eq!(memo.get(&(hx, Wrt::U(0))), None, "no tangent of h(x)");

    // The same inside a unit: f(a) = pw(a, h(a)).
    let f = g.add_unit("f", 1, 1);
    let a = g.arg(0);
    let ha = g.call(h, &[a]);
    let body = g.call(p, &[a, ha]);
    assert!(g.set_body(f, &[body]));
    let o = g.call(f, &[x]);
    let units = g.units().len();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 1, "only f gets a new tangent unit");
    assert_eq!(run(&g, dx, &[2.5], &[]), (18.75, 1 + 4 + 3));
}

/// Activity through mutual recursion and call arguments: `a(s, n) =
/// if(n<1)then(s)else(b(n-1, s))` and `b(m, s) = a(s, m)` swap the argument order, so s is
/// active in formal 0 of `a` and formal 1 of `b`, and the counters are inactive in both.
#[test]
fn activity_through_mutual_recursion_and_call_arguments() {
    let mut g = Dag::new(10_000);
    let ua = g.add_unit("a", 2, 1);
    let ub = g.add_unit("b", 2, 1);
    let (s, n) = (g.arg(0), g.arg(1));
    let one = g.konst(1.0);
    let c = g.cmp(Cmp::Lt, n, one);
    let m = g.sub(n, one);
    let rb = g.call(ub, &[m, s]);
    let ba = g.select(c, s, rb);
    assert!(g.set_body(ua, &[ba]));
    let (m0, s1) = (g.arg(0), g.arg(1));
    let bb = g.call(ua, &[s1, m0]);
    assert!(g.set_body(ub, &[bb]));
    let (x, k) = (g.var(0), g.param(0));
    let o = g.call(ua, &[x, k]);
    let units = g.units().len();
    let mut memo = HashMap::new();
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(g.units().len(), units + 2);
    assert_eq!(g.derived(&[0, ua, 0, 0, 0]), Some(units as u32));
    assert_eq!(g.derived(&[0, ub, 0, 0, 1]), Some(units as u32 + 1));
    for kv in 0u32..5 {
        assert_eq!(
            run(&g, dx, &[1.5], &[f64::from(kv)]),
            (1.0, 2 * u64::from(kv) + 1)
        );
    }
    // With x in the counter slot the derivative is the constant 0.
    let o2 = g.call(ua, &[k, x]);
    let units = g.units().len();
    let d2 = d(&mut g, &mut memo, o2, Wrt::U(0));
    assert!(is_zero(&g, d2));
    assert_eq!(g.units().len(), units);
}

/// A call into a unit with no body, directly or through a unit that calls it, has a NaN
/// derivative, builds no tangent unit and leaves no memo entry, and the DAG names the
/// unit. Once the body is set, the same request gives the right derivative.
#[test]
fn missing_body_gives_nan_without_caching_it() {
    let mut g = Dag::new(10_000);
    let mb = g.add_unit("mb", 1, 1);
    let f = g.add_unit("f", 1, 1);
    let a = g.arg(0);
    let fb = g.call(mb, &[a]);
    assert!(g.set_body(f, &[fb]));
    let x = g.var(0);
    let o = g.call(f, &[x]);
    let direct = g.call(mb, &[x]);
    assert_eq!(g.missing_body(), None);
    let units = g.units().len();
    let mut memo = HashMap::new();
    for node in [o, direct] {
        let dx = d(&mut g, &mut memo, node, Wrt::U(0));
        assert!(matches!(g.op(dx), Op::Const(b) if f64::from_bits(*b).is_nan()));
        assert_eq!(g.units().len(), units, "no tangent unit");
        assert_eq!(memo.get(&(node, Wrt::U(0))), None, "no memo entry");
        assert_eq!(g.missing_body(), Some(mb));
    }
    let sq = g.mul(a, a);
    assert!(g.set_body(mb, &[sq]));
    assert_eq!(g.missing_body(), None);
    let dx = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(run(&g, dx, &[1.5], &[]), (3.0, 2));
}

/// The activity of a unit is recomputed after its body changes, whether by a second
/// `set_body` or by a restore that drops the body.
#[test]
fn activity_follows_a_new_body() {
    let mut g = Dag::new(10_000);
    let k = g.add_unit("k", 1, 1);
    let a = g.arg(0);
    assert!(g.set_body(k, &[a]));
    let p = g.param(0);
    let o = g.call(k, &[p]);
    let mut memo = HashMap::new();
    let d0 = d(&mut g, &mut memo, o, Wrt::U(0));
    assert!(is_zero(&g, d0), "k(p) = p does not depend on x");
    // k(a) = x a now reads x.
    let x = g.var(0);
    let xa = g.mul(x, a);
    assert!(g.set_body(k, &[xa]));
    let mut memo = HashMap::new();
    let d1 = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(run(&g, d1, &[2.0], &[0.5]), (0.5, 1));

    // Restore drops a body that reads a node added after the checkpoint.
    let mut g = Dag::new(10_000);
    let k = g.add_unit("k", 1, 1);
    let a = g.arg(0);
    let p = g.param(0);
    let o = g.call(k, &[p]);
    let cp = g.checkpoint();
    let x = g.var(0);
    let xa = g.mul(x, a);
    assert!(g.set_body(k, &[xa]));
    let mut memo = HashMap::new();
    let d1 = d(&mut g, &mut memo, o, Wrt::U(0));
    assert_eq!(run(&g, d1, &[2.0], &[0.5]), (0.5, 1));
    g.restore(&cp);
    assert!(g.set_body(k, &[a]));
    let mut memo = HashMap::new();
    let d0 = d(&mut g, &mut memo, o, Wrt::U(0));
    assert!(is_zero(&g, d0));
}
