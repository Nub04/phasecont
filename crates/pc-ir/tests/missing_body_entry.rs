//! A derivative entry of a model that calls a function whose body is not set is an
//! error, never a silent NaN (SPEC section 8af-5), and the error leaves nothing cached.

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
    for kind in [EntryKind::JacU, EntryKind::JacP] {
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

/// Until the jet units exist, a directional entry of a model with a call is refused, with
/// or without a body, instead of giving a NaN form.
#[test]
fn directional_entry_with_a_call_is_refused_not_nan() {
    let (mut ir, mb) = model();
    let a = ir.dag.arg(0);
    let sq = ir.dag.mul(a, a);
    assert!(ir.dag.set_body(mb, &[sq]));
    for slots in [vec![Slot::U], vec![Slot::U, Slot::P]] {
        let e = ir.entry(&EntryKind::Dir(slots)).expect_err("no NaN form");
        assert_eq!(e.kind, IrErrorKind::Unsupported);
        assert!(e.message.contains("user function"), "{}", e.message);
    }
}
