//! A derivative entry of a model that calls a function whose body is not set is an
//! error, never a silent NaN (SPEC section 8af-5), and the error leaves nothing cached.

use pc_ir::dag::Cmp;
use pc_ir::eval::eval;
use pc_ir::{EntryKind, IrErrorKind, IrModel, Slot, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

/// `x' = mb(x)` with `mb` a unit of arity 1 that has no body yet.
fn model() -> (IrModel, u32) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", "par a=2\ninit x=3\nx'=a*x\ndone\n");
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
    let mut ir = lower(&l.expanded, &l.info).expect("lower");
    let mb = ir.dag.add_unit("mb", 1, 1);
    let x = ir.dag.var(0);
    ir.f[0] = ir.dag.call(mb, &[x]);
    (ir, mb)
}

#[test]
fn jacobian_of_a_call_without_a_body_is_an_error() {
    let (mut ir, mb) = model();
    for kind in [
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![Slot::U, Slot::U]),
    ] {
        let e = ir
            .entry(&kind)
            .expect_err("a missing body must not give NaN");
        assert_eq!(e.kind, IrErrorKind::Unsupported);
        assert!(e.message.contains("`mb`"), "{}", e.message);
    }
    // The value entry needs no derivative and builds.
    assert!(ir.entry(&EntryKind::F).is_ok());

    // Once the body is set, the same request gives the derivative: nothing stale was kept.
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(mb, &[sq]));
    let j = ir.entry(&EntryKind::JacU).expect("body is set");
    let mut out = [f64::NAN];
    eval(&ir.dag, &j.outputs, &[3.0], &[0.0], &[], &mut out);
    assert_eq!(out[0], 6.0);
}

/// A directional entry of a model with a call builds through the jet units once the body
/// is set: `mb(x) = x*x` gives `2 x` along `[U]` and 2 along `[U, U]`.
#[test]
fn directional_entry_with_a_call_builds() {
    let (mut ir, mb) = model();
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(mb, &[sq]));
    for (slots, want) in [(vec![Slot::U], 6.0), (vec![Slot::U, Slot::U], 2.0)] {
        let e = ir.entry(&EntryKind::Dir(slots)).expect("jet units");
        let mut out = [f64::NAN];
        eval(&ir.dag, &e.outputs, &[3.0], &[0.0], &[1.0, 1.0], &mut out);
        assert_eq!(out[0], want);
    }
}

/// A body cannot be replaced: a second `set_body` returns false and keeps the first body,
/// so a derivative built from it stays right.
#[test]
fn a_body_is_set_once() {
    let (mut ir, mb) = model();
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(mb, &[sq]));
    let j = ir.entry(&EntryKind::JacU).expect("body is set");
    let cube = ir.dag.mul(sq, a);
    assert!(!ir.dag.set_body(mb, &[cube]), "a second body is refused");
    let j2 = ir.entry(&EntryKind::JacU).expect("cached");
    assert_eq!(j, j2);
    let mut out = [f64::NAN];
    eval(&ir.dag, &j2.outputs, &[3.0], &[0.0], &[], &mut out);
    assert_eq!(out[0], 6.0);
}

/// `m2`, a unit of arity 1 with two outputs `[a*a, a]`, and the model of [`model`].
fn with_m2(body: bool) -> (IrModel, u32) {
    let (mut ir, _) = model();
    let m2 = ir.dag.add_unit("m2", 1, 2);
    if body {
        let a = ir.dag.arg(0);
        let sq = ir.dag.mul(a, a);
        assert!(ir.dag.set_body(m2, &[sq, a]));
    }
    (ir, m2)
}

/// Each derivative entry is refused, twice, with an error that names `m2`.
fn refused_twice(ir: &mut IrModel, what: &str) {
    for kind in [
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![Slot::U]),
        EntryKind::Dir(vec![Slot::U, Slot::P]),
    ] {
        for _ in 0..2 {
            let e = ir.entry(&kind).expect_err(what);
            assert_eq!(e.kind, IrErrorKind::Unsupported, "{what}");
            assert!(e.message.contains("`m2`"), "{what}: {}", e.message);
            assert!(e.message.contains("more than one output"), "{}", e.message);
        }
    }
    assert!(ir.entry(&EntryKind::F).is_ok(), "{what}");
}

#[test]
fn a_call_with_several_outputs_is_refused_not_nan() {
    for body in [false, true] {
        let (mut ir, m2) = with_m2(body);
        let x = ir.dag.var(0);
        ir.f[0] = ir.dag.call(m2, &[x]);
        refused_twice(&mut ir, "direct call");
    }
    // Output 1 of the call.
    let (mut ir, m2) = with_m2(true);
    let x = ir.dag.var(0);
    let c = ir.dag.call(m2, &[x]);
    ir.f[0] = ir.dag.out(c, 1);
    refused_twice(&mut ir, "out node");
    // Inside the body of a unit with one output that the model calls.
    let (mut ir, m2) = with_m2(true);
    let k = ir.dag.add_unit("k", 1, 1);
    let a = ir.dag.arg(0);
    let c = ir.dag.call(m2, &[a]);
    let s = ir.dag.add(c, a);
    assert!(ir.dag.set_body(k, &[s]));
    let x = ir.dag.var(0);
    ir.f[0] = ir.dag.call(k, &[x]);
    refused_twice(&mut ir, "call inside a body");
}

/// A call with several outputs that feeds only a condition has no derivative to take, so
/// the Jacobian builds, both at the top level and inside a body.
#[test]
fn a_call_with_several_outputs_in_a_condition_is_fine() {
    let (mut ir, m2) = with_m2(true);
    let k = ir.dag.add_unit("k", 1, 1);
    let a = ir.dag.arg(0);
    let c = ir.dag.call(m2, &[a]);
    let zero = ir.dag.konst(0.0);
    let cond = ir.dag.cmp(Cmp::Gt, c, zero);
    let sq = ir.dag.mul(a, a);
    let body = ir.dag.select(cond, sq, a);
    assert!(ir.dag.set_body(k, &[body]));
    let x = ir.dag.var(0);
    let top = ir.dag.call(m2, &[x]);
    let tcond = ir.dag.cmp(Cmp::Gt, top, zero);
    let kx = ir.dag.call(k, &[x]);
    ir.f[0] = ir.dag.select(tcond, kx, x);
    let j = ir
        .entry(&EntryKind::JacU)
        .expect("no derivative through m2");
    let mut out = [f64::NAN];
    eval(&ir.dag, &j.outputs, &[3.0], &[0.0], &[], &mut out);
    assert_eq!(out[0], 6.0);
    assert!(ir.entry(&EntryKind::JacP).is_ok());
}

/// A formal outside a body has no derivative rule: an error, not a NaN entry.
#[test]
fn a_formal_outside_a_body_is_refused() {
    let (mut ir, _) = model();
    let a = ir.dag.arg(0);
    let x = ir.dag.var(0);
    ir.f[0] = ir.dag.mul(a, x);
    for kind in [
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![Slot::U]),
    ] {
        for _ in 0..2 {
            let e = ir.entry(&kind).expect_err("no NaN entry");
            assert_eq!(e.kind, IrErrorKind::Unsupported);
        }
    }
}
