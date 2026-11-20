//! Frame machine tests for recursive functions (SPEC section 8af-5): the closed-form
//! oracles of the section, call counts, the lazy `If`, the depth cap, error bounds through
//! calls and the errors of a malformed DAG. The units are built with the DAG API, since the
//! lowering of `.ode` recursion comes later.

use pc_ir::dag::{Cmp, Dag, Fn1, NodeId};
use pc_ir::eval::{eval, eval_err};
use pc_ir::machine::{EvalError, MAX_ACTIVE_CALLS, run, try_eval, try_eval_err};

/// Builds `if(n<1)then(base)else(step(n, f(n-1)))` as the body of unit `u`.
fn chain(d: &mut Dag, u: u32, base: f64, step: fn(&mut Dag, NodeId, NodeId) -> NodeId) {
    let n = d.arg(0);
    let one = d.konst(1.0);
    let b = d.konst(base);
    let c = d.cmp(Cmp::Lt, n, one);
    let m = d.sub(n, one);
    let r = d.call(u, &[m]);
    let e = step(d, n, r);
    let body = d.select(c, b, e);
    assert!(d.set_body(u, &[body]));
}

/// `fact(n) = if(n<1)then(1)else(n*fact(n-1))`; returns the unit.
fn fact(d: &mut Dag) -> u32 {
    let u = d.add_unit("fact", 1, 1);
    chain(d, u, 1.0, |d, n, r| d.mul(n, r));
    u
}

/// `sum(n) = if(n<1)then(0)else(n+sum(n-1))`; returns the unit.
fn sum(d: &mut Dag) -> u32 {
    let u = d.add_unit("sum", 1, 1);
    chain(d, u, 0.0, |d, n, r| d.add(n, r));
    u
}

/// A DAG with one unit built by `build` and the output `unit(x0)`.
fn top(build: fn(&mut Dag) -> u32) -> (Dag, NodeId) {
    let mut d = Dag::new(10_000);
    let u = build(&mut d);
    let x = d.var(0);
    let o = d.call(u, &[x]);
    (d, o)
}

fn value(d: &Dag, o: NodeId, x: f64) -> (f64, u64) {
    let mut out = [f64::NAN];
    let st = try_eval(d, &[o], &[x], &[], &[], &mut out).expect("evaluates");
    (out[0], st.calls)
}

#[test]
fn fact_is_exact_up_to_22() {
    let (d, o) = top(fact);
    let mut want: u128 = 1;
    for n in 0..=22u32 {
        if n > 0 {
            want *= u128::from(n);
        }
        let (got, calls) = value(&d, o, f64::from(n));
        assert_eq!(got, want as f64, "fact({n})");
        assert_eq!(calls, u64::from(n) + 1, "calls of fact({n})");
    }
}

#[test]
fn depth_equal_to_the_cap_passes_and_one_more_fails() {
    let (d, o) = top(sum);
    let mut out = [f64::NAN];
    // sum(4) makes 5 active calls.
    let st = run(&d, &[o], &[4.0], &[], &[], &mut out, None, 5).expect("5 calls fit");
    assert_eq!(out[0], 10.0);
    assert_eq!(st.max_depth, 5);
    let e = run(&d, &[o], &[5.0], &[], &[], &mut out, None, 5).unwrap_err();
    assert_eq!(
        e,
        EvalError::Depth {
            func: "sum".into(),
            cap: 5
        }
    );
}

/// The two runs at the cap of the SPEC, on a thread with a 256 KiB stack: the machine keeps
/// its frames on the heap, so the native stack does not grow with the depth.
#[test]
fn sum_at_the_cap_on_a_small_stack() {
    let h = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let (d, o) = top(sum);
            let mut out = [f64::NAN];
            let st = try_eval(&d, &[o], &[999_999.0], &[], &[], &mut out).expect("fits");
            assert_eq!(out[0], 499_999_500_000.0);
            assert_eq!(st.max_depth, MAX_ACTIVE_CALLS);
            assert_eq!(st.calls, 1_000_000);
            let e = try_eval(&d, &[o], &[1_000_000.0], &[], &[], &mut out).unwrap_err();
            assert!(matches!(e, EvalError::Depth { cap, .. } if cap == MAX_ACTIVE_CALLS));
        })
        .expect("spawns");
    h.join().expect("no panic");
}

/// `g(n) = (if(n>0)then(g(n-1))else(0)) + (if(n>5)then(g(n-1))else(1))`: both arms name the
/// same call node, which the frame memo evaluates once.
#[test]
fn shared_call_in_two_arms_runs_once_per_frame() {
    let mut d = Dag::new(1000);
    let u = d.add_unit("g", 1, 1);
    let n = d.arg(0);
    let (zero, one, five) = (d.konst(0.0), d.konst(1.0), d.konst(5.0));
    let m = d.sub(n, one);
    let r = d.call(u, &[m]);
    let c1 = d.cmp(Cmp::Gt, n, zero);
    let a = d.select(c1, r, zero);
    let c2 = d.cmp(Cmp::Gt, n, five);
    let b = d.select(c2, r, one);
    let body = d.add(a, b);
    assert!(d.set_body(u, &[body]));
    let x = d.var(0);
    let o = d.call(u, &[x]);
    for k in 0..=30u32 {
        let want = if k <= 5 {
            f64::from(k + 1)
        } else {
            6.0 * 2f64.powi(k as i32 - 5)
        };
        let (got, calls) = value(&d, o, f64::from(k));
        assert_eq!(got, want, "g({k})");
        assert_eq!(calls, u64::from(k) + 1, "calls of g({k})");
    }
}

/// `x' = if(k<0)then(0)else(f(k))` at k = -1 gives 0 without a call.
#[test]
fn untaken_arm_makes_no_call() {
    let mut d = Dag::new(1000);
    let u = fact(&mut d);
    let k = d.var(0);
    let zero = d.konst(0.0);
    let c = d.cmp(Cmp::Lt, k, zero);
    let r = d.call(u, &[k]);
    let o = d.select(c, zero, r);
    assert_eq!(value(&d, o, -1.0), (0.0, 0));
    assert_eq!(value(&d, o, 3.0), (6.0, 4));
}

/// A NaN condition takes the first arm.
#[test]
fn nan_condition_takes_the_first_arm() {
    let mut d = Dag::new(1000);
    let u = fact(&mut d);
    let k = d.var(0);
    let two = d.konst(2.0);
    let r = d.call(u, &[two]);
    let o = d.select(k, r, k);
    assert_eq!(value(&d, o, f64::NAN), (2.0, 3));
    assert_eq!(value(&d, o, 0.0), (0.0, 0));
}

/// `h(n)=if(n<1)then(0)else(if(n>5)then(h(n-1))else(1+h(n-1)))` gives min(n, 5).
#[test]
fn nested_if_gives_min_of_n_and_5() {
    let mut d = Dag::new(1000);
    let u = d.add_unit("h", 1, 1);
    let n = d.arg(0);
    let (zero, one, five) = (d.konst(0.0), d.konst(1.0), d.konst(5.0));
    let m = d.sub(n, one);
    let r = d.call(u, &[m]);
    let r1 = d.add(one, r);
    let c2 = d.cmp(Cmp::Gt, n, five);
    let inner = d.select(c2, r, r1);
    let c1 = d.cmp(Cmp::Lt, n, one);
    let body = d.select(c1, zero, inner);
    assert!(d.set_body(u, &[body]));
    let x = d.var(0);
    let o = d.call(u, &[x]);
    for k in 0..=40u32 {
        let (got, calls) = value(&d, o, f64::from(k));
        assert_eq!(got, f64::from(k.min(5)), "h({k})");
        assert_eq!(calls, u64::from(k) + 1);
    }
}

/// `ev(n)=if(n<1)then(1)else(od(n-1))`, `od(n)=if(n<1)then(0)else(ev(n-1))`.
#[test]
fn mutual_recursion_gives_parity() {
    let mut d = Dag::new(1000);
    let ev = d.add_unit("ev", 1, 1);
    let od = d.add_unit("od", 1, 1);
    for (u, other, base) in [(ev, od, 1.0), (od, ev, 0.0)] {
        let n = d.arg(0);
        let one = d.konst(1.0);
        let b = d.konst(base);
        let c = d.cmp(Cmp::Lt, n, one);
        let m = d.sub(n, one);
        let r = d.call(other, &[m]);
        let body = d.select(c, b, r);
        assert!(d.set_body(u, &[body]));
    }
    let x = d.var(0);
    let o = d.call(ev, &[x]);
    for k in 0..=25u32 {
        let (got, calls) = value(&d, o, f64::from(k));
        assert_eq!(got, f64::from(u32::from(k % 2 == 0)), "ev({k})");
        assert_eq!(calls, u64::from(k) + 1);
    }
}

/// A unit with two outputs: `p(n) = (fib(n), fib(n+1))`, built as
/// `if(n<1)then(0, 1)else(b, a+b)` with `(a, b) = p(n-1)`. `Out(call, 1)` reads the second.
#[test]
fn two_output_unit_reads_its_second_output() {
    let mut d = Dag::new(1000);
    let u = d.add_unit("p", 1, 2);
    let n = d.arg(0);
    let (zero, one) = (d.konst(0.0), d.konst(1.0));
    let c = d.cmp(Cmp::Lt, n, one);
    let m = d.sub(n, one);
    let a = d.call(u, &[m]);
    let b = d.out(a, 1);
    let s = d.add(a, b);
    let o0 = d.select(c, zero, b);
    let o1 = d.select(c, one, s);
    assert!(d.set_body(u, &[o0, o1]));
    let x = d.var(0);
    let top = d.call(u, &[x]);
    let top1 = d.out(top, 1);
    let mut out = [f64::NAN; 2];
    let st = try_eval(&d, &[top, top1], &[10.0], &[], &[], &mut out).expect("evaluates");
    assert_eq!(out, [55.0, 89.0]);
    assert_eq!(st.calls, 11);
    // The second output alone runs the same calls.
    let mut one_out = [f64::NAN];
    let st = try_eval(&d, &[top1], &[10.0], &[], &[], &mut one_out).expect("evaluates");
    assert_eq!(one_out[0], 89.0);
    assert_eq!(st.calls, 11);
}

/// `f(n) = if(n<1)then(0)else(f(n-1)+k(3))` with `k(m) = 2m`: the call `k(3)` has constant
/// arguments, so it is a global node and runs once for the whole evaluation.
#[test]
fn constant_argument_call_in_a_body_runs_once() {
    let mut d = Dag::new(1000);
    let k = d.add_unit("k", 1, 1);
    let m = d.arg(0);
    let two = d.konst(2.0);
    let km = d.mul(two, m);
    assert!(d.set_body(k, &[km]));
    let f = d.add_unit("f", 1, 1);
    let three = d.konst(3.0);
    let k3 = d.call(k, &[three]);
    chain_with(&mut d, f, k3);
    let x = d.var(0);
    let o = d.call(f, &[x]);
    assert_eq!(value(&d, o, 7.0), (42.0, 8 + 1));
}

fn chain_with(d: &mut Dag, u: u32, k3: NodeId) {
    let n = d.arg(0);
    let (zero, one) = (d.konst(0.0), d.konst(1.0));
    let c = d.cmp(Cmp::Lt, n, one);
    let m = d.sub(n, one);
    let r = d.call(u, &[m]);
    let e = d.add(r, k3);
    let body = d.select(c, zero, e);
    assert!(d.set_body(u, &[body]));
}

/// Two calls of one unit with different arguments in one body are separate frames.
#[test]
fn two_calls_with_different_arguments() {
    // t(a, b) = a*b + a, evaluated at (x0, x1) and (x1, x0).
    let mut d = Dag::new(1000);
    let t = d.add_unit("t", 2, 1);
    let (a, b) = (d.arg(0), d.arg(1));
    let ab = d.mul(a, b);
    let body = d.add(ab, a);
    assert!(d.set_body(t, &[body]));
    let (x0, x1) = (d.var(0), d.var(1));
    let c1 = d.call(t, &[x0, x1]);
    let c2 = d.call(t, &[x1, x0]);
    let o = d.sub(c1, c2);
    let mut out = [f64::NAN];
    let st = try_eval(&d, &[o], &[3.0, 5.0], &[], &[], &mut out).expect("evaluates");
    assert_eq!(out[0], (15.0 + 3.0) - (15.0 + 5.0));
    assert_eq!(st.calls, 2);
}

#[test]
fn malformed_dags_give_an_error_not_nan() {
    let mut out = [0.0];
    // A call of a unit without a body.
    let mut d = Dag::new(100);
    let u = d.add_unit("nobody", 1, 1);
    let x = d.var(0);
    let c = d.call(u, &[x]);
    let e = try_eval(&d, &[c], &[1.0], &[], &[], &mut out).unwrap_err();
    assert!(matches!(e, EvalError::Malformed(_)), "{e:?}");
    // A call with the wrong number of arguments.
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 2, 1);
    let a = d.arg(0);
    assert!(d.set_body(u, &[a]));
    let x = d.var(0);
    let c = d.call(u, &[x]);
    let e = try_eval(&d, &[c], &[1.0], &[], &[], &mut out).unwrap_err();
    assert!(matches!(e, EvalError::Malformed(_)), "{e:?}");
    // A formal read outside any call.
    let e = try_eval(&d, &[a], &[1.0], &[], &[], &mut out).unwrap_err();
    assert!(matches!(e, EvalError::Malformed(_)), "{e:?}");
    // A call of an unknown unit.
    let mut d = Dag::new(100);
    let x = d.var(0);
    let c = d.call(7, &[x]);
    let e = try_eval(&d, &[c], &[1.0], &[], &[], &mut out).unwrap_err();
    assert!(matches!(e, EvalError::Malformed(_)), "{e:?}");
}

/// The infallible `eval` and `eval_err` write NaN where the machine fails, and never panic.
#[test]
fn infallible_eval_writes_nan_on_error() {
    let (d, o) = top(sum);
    let mut out = [0.0];
    eval(&d, &[o], &[3.0], &[], &[], &mut out);
    assert_eq!(out[0], 6.0);
    let mut err = [0.0];
    eval_err(&d, &[o], &[3.0], &[], &[], &mut out, &mut err);
    assert_eq!(out[0], 6.0);
    assert!(err[0].is_finite());
    eval(&d, &[o], &[1_000_000.0], &[], &[], &mut out);
    assert!(out[0].is_nan());
    eval_err(&d, &[o], &[1_000_000.0], &[], &[], &mut out, &mut err);
    assert!(out[0].is_nan() && err[0].is_nan());
}

/// An output without a call next to one with a call gets the bits and the bound of the
/// eager evaluator.
#[test]
fn plain_output_next_to_a_call_matches_the_eager_path() {
    let mut d = Dag::new(1000);
    let u = fact(&mut d);
    let (x0, x1) = (d.var(0), d.var(1));
    let c = d.call(u, &[x0]);
    let s = d.call1(Fn1::Sin, x1);
    let m = d.mul(x0, x1);
    let plain = d.add(m, s);
    let x = [5.0, 0.7];
    let (mut want, mut want_err) = ([f64::NAN], [f64::NAN]);
    eval_err(&d, &[plain], &x, &[], &[], &mut want, &mut want_err);
    let (mut got, mut got_err) = ([f64::NAN; 2], [f64::NAN; 2]);
    try_eval_err(&d, &[c, plain], &x, &[], &[], &mut got, &mut got_err).expect("evaluates");
    assert_eq!(got[0], 120.0);
    assert_eq!(got[1].to_bits(), want[0].to_bits());
    assert_eq!(got_err[1].to_bits(), want_err[0].to_bits());
    let mut both = [f64::NAN; 2];
    eval(&d, &[c, plain], &x, &[], &[], &mut both);
    assert_eq!(both[1].to_bits(), want[0].to_bits());
}

/// A call is transparent to the error bound: `t(a, b) = a*b + a` at `(x0+x1, x1)` has the
/// bound of the same expression written out.
#[test]
fn error_bound_passes_through_a_call() {
    let mut d = Dag::new(1000);
    let t = d.add_unit("t", 2, 1);
    let (a, b) = (d.arg(0), d.arg(1));
    let ab = d.mul(a, b);
    let body = d.add(ab, a);
    assert!(d.set_body(t, &[body]));
    let (x0, x1) = (d.var(0), d.var(1));
    let s = d.add(x0, x1);
    let c = d.call(t, &[s, x1]);
    let sm = d.mul(s, x1);
    let inline = d.add(sm, s);
    let x = [0.1, 0.7];
    let (mut want, mut want_err) = ([f64::NAN], [f64::NAN]);
    eval_err(&d, &[inline], &x, &[], &[], &mut want, &mut want_err);
    let (mut got, mut got_err) = ([f64::NAN], [f64::NAN]);
    try_eval_err(&d, &[c], &x, &[], &[], &mut got, &mut got_err).expect("evaluates");
    assert_eq!(got[0].to_bits(), want[0].to_bits());
    assert_eq!(got_err[0].to_bits(), want_err[0].to_bits());
    assert!(got_err[0] > 0.0);
}

/// The bound of a lazy `If` is the bound of the arm it takes.
#[test]
fn error_bound_of_if_is_its_taken_arm() {
    let mut d = Dag::new(1000);
    let u = fact(&mut d);
    let (x0, x1) = (d.var(0), d.var(1));
    let zero = d.konst(0.0);
    let c = d.cmp(Cmp::Lt, x0, zero);
    let r = d.call(u, &[x0]);
    let arm = d.add(x1, x1);
    let o = d.select(c, arm, r);
    let (mut out, mut err) = ([f64::NAN], [f64::NAN]);
    try_eval_err(&d, &[o], &[-1.0, 0.3], &[], &[], &mut out, &mut err).expect("evaluates");
    assert_eq!(out[0], 0.6);
    assert_eq!(err[0], 0.6);
    // fact(3) = 3*(2*(1*1)): each product is exact, and the bound counts one rounding of each.
    try_eval_err(&d, &[o], &[3.0, 0.3], &[], &[], &mut out, &mut err).expect("evaluates");
    assert_eq!(out[0], 6.0);
    assert!(err[0] > 0.0 && err[0].is_finite());
}

/// Output slots past the outputs and output ids outside the DAG are NaN, as in `eval`.
#[test]
fn extra_and_unknown_outputs_are_nan() {
    let (d, o) = top(fact);
    let mut out = [0.0; 3];
    try_eval(&d, &[o, NodeId(99_999)], &[3.0], &[], &[], &mut out).expect("evaluates");
    assert_eq!(out[0], 6.0);
    assert!(out[1].is_nan() && out[2].is_nan());
}
