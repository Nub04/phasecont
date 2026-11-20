//! DAG builder tests for the ops of recursive functions (SPEC section 8af-5): `Arg`,
//! `Call`, `If` and `Out`, their flags, the folds that must keep a call, unit bodies and
//! their frame slots, and `restore`.

use pc_ir::dag::{Cmp, Dag, Fn2, NodeId, Op};

fn is_const(d: &Dag, id: NodeId, x: f64) -> bool {
    matches!(d.op(id), Op::Const(b) if *b == x.to_bits())
}

/// `fact(n) = if(n<1)then(1)else(n*fact(n-1))` as unit 0 of `d`; returns the unit and the
/// body output.
fn fact(d: &mut Dag) -> (u32, NodeId) {
    let u = d.add_unit("fact", 1, 1);
    let n = d.arg(0);
    let one = d.konst(1.0);
    let c = d.cmp(Cmp::Lt, n, one);
    let m = d.sub(n, one);
    let r = d.call(u, &[m]);
    let e = d.mul(n, r);
    let body = d.select(c, one, e);
    assert!(d.set_body(u, &[body]));
    (u, body)
}

#[test]
fn arg_and_call_are_hash_consed() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let a0 = d.arg(0);
    assert_eq!(d.arg(0), a0);
    assert_ne!(d.arg(1), a0);
    let x = d.var(0);
    let c1 = d.call(u, &[x]);
    let n = d.len();
    let c2 = d.call(u, &[x]);
    assert_eq!(c1, c2);
    assert_eq!(d.len(), n);
    // The same argument list is stored once and shared by calls of different units.
    let v = d.add_unit("g", 1, 1);
    let c3 = d.call(v, &[x]);
    let (Op::Call(_, l1), Op::Call(_, l3)) = (*d.op(c1), *d.op(c3)) else {
        panic!("not calls");
    };
    assert_eq!(l1, l3);
    assert_eq!(d.args(l1), &[x]);
    // A different list makes a different call.
    let y = d.var(1);
    assert_ne!(d.call(u, &[y]), c1);
    // An unknown list index reads as empty.
    assert!(d.args(999).is_empty());
}

#[test]
fn call_with_constant_args_is_never_folded() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let k = d.konst(3.0);
    let c = d.call(u, &[k]);
    assert!(matches!(d.op(c), Op::Call(..)));
    // Arithmetic on it does not fold either.
    let s = d.add(c, k);
    assert!(matches!(d.op(s), Op::Add(..)));
}

#[test]
fn out_zero_is_the_call_and_out_j_is_a_node() {
    let mut d = Dag::new(100);
    let u = d.add_unit("t", 1, 2);
    let x = d.var(0);
    let c = d.call(u, &[x]);
    assert_eq!(d.out(c, 0), c);
    let o1 = d.out(c, 1);
    assert_eq!(*d.op(o1), Op::Out(c, 1));
    assert_eq!(d.out(c, 1), o1);
    assert!(d.has_call(o1));
}

#[test]
fn has_call_and_arg_dep_flags() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let x = d.var(0);
    let a = d.arg(0);
    let k = d.konst(2.0);
    assert!(!d.has_call(x) && !d.arg_dep(x));
    assert!(!d.has_call(a) && d.arg_dep(a));
    let ax = d.add(a, x);
    assert!(d.arg_dep(ax) && !d.has_call(ax));
    // A call with constant arguments has a call but no Arg.
    let ck = d.call(u, &[k]);
    assert!(d.has_call(ck) && !d.arg_dep(ck));
    // Flags flow through call arguments.
    let ca = d.call(u, &[ax]);
    assert!(d.has_call(ca) && d.arg_dep(ca));
    let s = d.mul(ck, x);
    assert!(d.has_call(s) && !d.arg_dep(s));
    // Out of an Arg-dependent call is Arg-dependent.
    let v = d.add_unit("t", 1, 2);
    let cv = d.call(v, &[a]);
    let o = d.out(cv, 1);
    assert!(d.has_call(o) && d.arg_dep(o));
    // Unknown ids give false.
    assert!(!d.has_call(NodeId(9999)) && !d.arg_dep(NodeId(9999)));
}

#[test]
fn select_makes_if_only_when_an_arm_calls() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let x = d.var(0);
    let y = d.var(1);
    let c = d.cmp(Cmp::Lt, x, y);
    let s = d.select(c, x, y);
    assert!(matches!(d.op(s), Op::Select(..)));
    let f = d.call(u, &[x]);
    let i = d.select(c, x, f);
    assert_eq!(*d.op(i), Op::If(c, x, f));
    assert!(d.has_call(i));
    let j = d.select(c, f, y);
    assert!(matches!(d.op(j), Op::If(..)));
    // A call only in the condition keeps a Select: the condition is always evaluated.
    let fc = d.cmp(Cmp::Lt, f, y);
    let k = d.select(fc, x, y);
    assert!(matches!(d.op(k), Op::Select(..)));
    assert!(d.has_call(k));
    // A constant condition picks an arm, call or not.
    let one = d.konst(1.0);
    let zero = d.konst(0.0);
    assert_eq!(d.select(one, f, y), f);
    assert_eq!(d.select(zero, f, y), y);
    let nan = d.konst(f64::NAN);
    assert_eq!(d.select(nan, f, y), f);
}

#[test]
fn select_same_arms_keeps_a_calling_condition() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let x = d.var(0);
    let y = d.var(1);
    let c = d.cmp(Cmp::Lt, x, y);
    assert_eq!(d.select(c, x, x), x);
    let f = d.call(u, &[x]);
    let fc = d.cmp(Cmp::Lt, f, y);
    let s = d.select(fc, x, x);
    assert_ne!(s, x);
    assert!(d.has_call(s));
    // Same calling arms with a plain condition: the call runs either way.
    assert_eq!(d.select(c, f, f), f);
}

#[test]
fn folds_keep_a_call() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let x = d.var(0);
    let f = d.call(u, &[x]);
    let zero = d.konst(0.0);
    // sub: x - x folds, f - f stays.
    let xx = d.sub(x, x);
    assert!(is_const(&d, xx, 0.0));
    let ff = d.sub(f, f);
    assert!(matches!(d.op(ff), Op::Sub(..)));
    // mul: 0 * x folds, 0 * f and f * 0 stay.
    let zx = d.mul(zero, x);
    assert!(is_const(&d, zx, 0.0));
    let zf = d.mul(zero, f);
    assert!(matches!(d.op(zf), Op::Mul(..)));
    let fz = d.mul(f, zero);
    assert!(matches!(d.op(fz), Op::Mul(..)));
    // zmul: a zero p folds unless t calls; a zero t folds unless p calls.
    let z1 = d.zmul(x, zero);
    assert!(is_const(&d, z1, 0.0));
    let z2 = d.zmul(f, zero);
    assert!(matches!(d.op(z2), Op::ZMul(..)));
    let z3 = d.zmul(zero, x);
    assert!(is_const(&d, z3, 0.0));
    let z4 = d.zmul(zero, f);
    assert!(matches!(d.op(z4), Op::ZMul(..)));
    // zdiv: a zero numerator folds unless the denominator calls.
    let q1 = d.zdiv(zero, x);
    assert!(is_const(&d, q1, 0.0));
    let q2 = d.zdiv(zero, f);
    assert!(matches!(d.op(q2), Op::ZDiv(..)));
    // powi n = 0: 1 for a plain base, pow(f, 0) for a calling one.
    let p1 = d.powi(x, 0);
    assert!(is_const(&d, p1, 1.0));
    let p2 = d.powi(f, 0);
    let Op::Call2(Fn2::Pow, b, e) = *d.op(p2) else {
        panic!("powi(f, 0) is not pow");
    };
    assert_eq!(b, f);
    assert!(is_const(&d, e, 0.0));
}

#[test]
fn set_body_slots_for_fact() {
    let mut d = Dag::new(100);
    let (u, body) = fact(&mut d);
    let n = d.arg(0);
    let one = d.konst(1.0);
    let unit = d.unit(u).unwrap();
    assert_eq!(unit.name, "fact");
    assert_eq!(unit.outputs(), Some(&[body][..]));
    // Locals: n, n<1, n-1, fact(n-1), n*fact(n-1), the If. Constant 1 is not a local.
    assert_eq!(unit.nlocal(), 6);
    assert_eq!(unit.arg_slot(0), unit.local(n));
    assert!(unit.arg_slot(1).is_none());
    assert!(unit.local(one).is_none());
    assert!(unit.local(body).is_some());
    // Slots are distinct and below nlocal.
    let mut slots = std::collections::HashSet::new();
    for id in 0..d.len() as u32 {
        if let Some(s) = unit.local(NodeId(id)) {
            assert!(s < unit.nlocal());
            assert!(slots.insert(s), "slot {s} used twice");
        }
    }
    assert_eq!(slots.len(), 6);
}

#[test]
fn set_body_call_takes_nout_slots_and_out_reads_them() {
    let mut d = Dag::new(100);
    // t has two outputs; g calls t and uses both.
    let t = d.add_unit("t", 1, 2);
    let g = d.add_unit("g", 1, 1);
    let a = d.arg(0);
    let c = d.call(t, &[a]);
    let o1 = d.out(c, 1);
    let s = d.add(c, o1);
    assert!(d.set_body(g, &[s]));
    let unit = d.unit(g).unwrap();
    let ca = unit.local(c).unwrap();
    assert_eq!(unit.local(o1), Some(ca + 1));
    // Arg 1 + call 2 + sum 1; Out takes no slot of its own.
    assert_eq!(unit.nlocal(), 4);
    let sa = unit.local(s).unwrap();
    let aa = unit.local(a).unwrap();
    let mut all = [aa, ca, ca + 1, sa];
    all.sort_unstable();
    assert_eq!(all, [0, 1, 2, 3]);
}

#[test]
fn set_body_rejects_bad_input() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let a = d.arg(0);
    assert!(!d.set_body(7, &[a]), "unknown unit");
    assert!(!d.set_body(u, &[a, a]), "wrong output count");
    assert!(!d.set_body(u, &[]), "no output");
    assert!(!d.set_body(u, &[NodeId(9999)]), "id outside the DAG");
    assert!(d.unit(u).unwrap().outputs().is_none());
    // A body that does not read its formal is still valid.
    let k = d.konst(5.0);
    assert!(d.set_body(u, &[k]));
    let unit = d.unit(u).unwrap();
    assert_eq!(unit.nlocal(), 0);
    assert!(unit.arg_slot(0).is_none());
}

#[test]
fn units_of_different_arity_share_arg_nodes() {
    // f(a) = a*a and g(a, b) = a*a + b share the node Arg(0)*Arg(0); each body gets its own
    // slots for it.
    let mut d = Dag::new(100);
    let f = d.add_unit("f", 1, 1);
    let g = d.add_unit("g", 2, 1);
    let a = d.arg(0);
    let b = d.arg(1);
    let sq = d.mul(a, a);
    assert!(d.set_body(f, &[sq]));
    let gb = d.add(sq, b);
    assert!(d.set_body(g, &[gb]));
    let uf = d.unit(f).unwrap();
    let ug = d.unit(g).unwrap();
    assert_eq!(uf.nlocal(), 2);
    assert_eq!(ug.nlocal(), 4);
    assert!(uf.local(sq).is_some() && ug.local(sq).is_some());
    assert!(uf.local(b).is_none());
    assert!(uf.arg_slot(1).is_none());
    assert!(ug.arg_slot(1).is_some());
}

#[test]
fn restore_drops_arglists_units_derived_and_bodies() {
    let mut d = Dag::new(1000);
    let (f, _) = fact(&mut d);
    let x = d.var(0);
    let cx = d.call(f, &[x]);
    d.set_derived(vec![f, 1], f);
    let cp = d.checkpoint();
    let nargs = {
        let Op::Call(_, l) = *d.op(cx) else { panic!() };
        l
    };
    // After the checkpoint: a new unit, a derived entry for it, a new list, a body on an
    // old unit that reads a new node, and a derived entry for an old unit.
    let g = d.add_unit("g", 1, 1);
    d.set_derived(vec![g, 0], g);
    d.set_derived(vec![f, 2], f);
    // A key that names the new unit but maps to an old one, and an overwrite of an entry
    // made before the checkpoint.
    d.set_derived(vec![g, 3], f);
    d.set_derived(vec![f, 1], g);
    let y = d.var(1);
    let cy = d.call(f, &[y]);
    let h = d.add_unit("h", 1, 1);
    assert!(d.set_body(h, &[cy]));
    d.restore(&cp);
    assert_eq!(d.units().len(), 1);
    // The restore undoes every derived entry set after the checkpoint, whatever unit it
    // names or maps to, and puts back the entry an overwrite replaced.
    assert_eq!(d.derived(&[f, 1]), Some(f));
    assert_eq!(d.derived(&[g, 0]), None);
    assert_eq!(d.derived(&[f, 2]), None);
    assert_eq!(d.derived(&[g, 3]), None);
    // A unit made after the restore reuses the index of `g` and finds no stale entry.
    let g2 = d.add_unit("g2", 1, 1);
    assert_eq!(g2, g);
    assert_eq!(d.derived(&[g2, 3]), None);
    assert!(d.unit(f).unwrap().outputs().is_some());
    // The list of the dropped call is gone; rebuilding it gets the next index again.
    let y2 = d.var(1);
    let cy2 = d.call(f, &[y2]);
    let Op::Call(_, l2) = *d.op(cy2) else {
        panic!()
    };
    assert_eq!(l2, nargs + 1);
    assert_eq!(d.args(l2), &[y2]);
    // The old call and its list survive.
    assert_eq!(d.call(f, &[x]), cx);
}

#[test]
fn restore_clears_an_old_unit_body_that_reads_a_dropped_node() {
    let mut d = Dag::new(1000);
    let u = d.add_unit("f", 1, 1);
    let cp = d.checkpoint();
    let a = d.arg(0);
    let k = d.konst(3.0);
    let s = d.add(a, k);
    assert!(d.set_body(u, &[s]));
    d.restore(&cp);
    assert_eq!(d.units().len(), 1);
    assert!(d.unit(u).unwrap().outputs().is_none());
    assert_eq!(d.unit(u).unwrap().nlocal(), 0);
}

#[test]
fn restore_after_exceeding_mid_build() {
    // The cap is hit partway through building a body; restore puts the DAG, the lists and
    // the units back, and the flag with them.
    let mut d = Dag::new(1000);
    let x = d.var(0);
    let cp = d.checkpoint();
    d.set_cap(d.len() + 3);
    let u = d.add_unit("f", 1, 1);
    let a = d.arg(0);
    let one = d.konst(1.0);
    let m = d.sub(a, one);
    let c = d.call(u, &[m]);
    let e = d.mul(a, c);
    let _ = d.set_body(u, &[e]);
    assert!(d.exceeded());
    d.restore(&cp);
    d.set_cap(1000);
    assert!(!d.exceeded());
    assert_eq!(d.len(), cp.len());
    assert!(d.units().is_empty());
    assert_eq!(d.var(0), x);
    // A rebuild after the restore works and reuses the freed indices.
    let u2 = d.add_unit("f", 1, 1);
    assert_eq!(u2, u);
    let a2 = d.arg(0);
    let c2 = d.call(u2, &[a2]);
    let Op::Call(_, l) = *d.op(c2) else { panic!() };
    assert_eq!(l, 0);
    assert!(!d.exceeded());
}

#[test]
fn value_may_be_nonfinite_for_calls() {
    let mut d = Dag::new(100);
    let u = d.add_unit("f", 1, 1);
    let x = d.var(0);
    let y = d.var(1);
    let a = d.arg(0);
    let f = d.call(u, &[x]);
    assert!(d.may_be_nonfinite(a));
    assert!(d.may_be_nonfinite(f));
    let c = d.cmp(Cmp::Lt, x, y);
    let i = d.select(c, x, f);
    assert!(d.may_be_nonfinite(i));
}
