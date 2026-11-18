//! The sign of a zero in a derivative entry.
//!
//! The DAG folds zeros it can see when it is built (`add(a, 0)` is `a`, `sub(0, b)` is
//! `neg(b)`, `neg(0)` is -0), and the f64 fallback sees only the values of one run. A
//! derivative that is a zero then comes out as -0 in one and +0 in the other. Both now
//! store `out + 0.0` for every derivative entry (`JacU`, `JacP`, `Dir`), which turns -0 into
//! +0 and leaves every other value as it is. `F` and `Aux` are not touched.

use pc_ir::eval::{eval_entry, eval_entry_err};
use pc_ir::{EntryKind, Fallback, IrModel, Slot, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn build(src: &str) -> (IrModel, Fallback) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    let fb = Fallback::new(&l.expanded, &l.info, &ir);
    (ir, fb)
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Our own right-hand sides. Each one has a clamp, a `neg`, a subtraction from zero, an
/// addition of zero, a product or an integer power that meets a clamp at x = 0 or y = 0.
/// The values and the partials are finite at every point below, so a difference is a
/// difference in the sign of a zero (or a real bug), never a NaN.
const RHS: [(&str, &str); 16] = [
    ("-(max(x,0)^2)", "x*max(y,0)"),
    ("-max(x,0)*y+a", "min(x,0)*max(y,0)-x"),
    ("(max(x,0)+0)*y", "-(min(x,0)*min(y,0))"),
    ("max(x,0)^3-y", "-max(x-y,0)^2*a"),
    ("max(x,0)-max(y,0)", "0-max(x,0)*y"),
    ("-(x*max(y,0)*max(x,0))", "a*min(max(x,0),max(y,0))"),
    ("-(max(x,0)^2)*y^2+max(y,0)^3", "max(x,0)^2*max(y,0)"),
    ("(max(x,0)-0)*(0-min(y,0))", "abs(x)*max(y,0)"),
    ("-(heav(x)*y)", "min(x,y)*max(x,y)"),
    ("0-(max(x,0)+max(y,0))", "-(a*max(x,0)^2)-y"),
    ("max(x,0)*min(y,0)*a", "-(max(x+y,0)^2)"),
    ("-max(0,x)^3+0", "min(0,y)^2-max(0,x)"),
    ("-(max(x,0)*max(x,0)*max(y,0))", "(0-max(y,0))*(0-max(x,0))"),
    ("max(x*a,0)-min(y*a,0)", "-(max(x,0)^2+min(y,0)^2)"),
    ("-(max(x,0)^4)+x*y", "0-max(y,0)^2*x"),
    ("-(min(x,0)^3)*max(y,0)", "max(x-1,0)*y-max(y-1,0)*x"),
];

fn kinds() -> Vec<(&'static str, EntryKind)> {
    use Slot::{Full, P, U};
    vec![
        ("JacU", EntryKind::JacU),
        ("JacP", EntryKind::JacP),
        ("Dir[U]", EntryKind::Dir(vec![U])),
        ("Dir[P]", EntryKind::Dir(vec![P])),
        ("Dir[Full]", EntryKind::Dir(vec![Full])),
        ("Dir[U,U]", EntryKind::Dir(vec![U, U])),
        ("Dir[U,P]", EntryKind::Dir(vec![U, P])),
        ("Dir[U,U,U]", EntryKind::Dir(vec![U, U, U])),
        ("Dir[Full,U,U]", EntryKind::Dir(vec![Full, U, U])),
        ("Dir[U,U,U,U]", EntryKind::Dir(vec![U, U, U, U])),
    ]
}

/// States x and y, one parameter a, then t. Zeros of both signs are among the inputs.
const POINTS: [([f64; 2], [f64; 2]); 8] = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 0.5], [1.0, 0.0]),
    ([-0.5, 0.0], [1.0, 0.0]),
    ([0.3, 0.0], [0.5, 0.0]),
    ([0.0, -0.7], [0.0, 0.0]),
    ([0.4, -0.2], [1.0, 0.0]),
    ([-0.0, 0.0], [-1.0, 0.0]),
    ([1.5, 2.5], [2.0, 0.0]),
];

/// Direction vectors: some entries are zero at run time, so a Dual skips them.
const DIRS: [[f64; 10]; 3] = [
    [1.0, 0.0, -0.5, 0.25, 2.0, 1.0, 0.0, -1.0, 0.5, 1.0],
    [0.0, 1.0, 1.0, 0.0, -1.0, 0.5, 2.0, 0.0, 1.0, -0.5],
    [0.75, -1.5, 0.0, 0.0, 1.0, -0.25, 1.0, 1.0, 0.0, 0.0],
];

fn packed_len(kind: &EntryKind, n: usize, n_p: usize) -> usize {
    match kind {
        EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
        _ => 0,
    }
}

#[test]
fn derivative_entries_of_clamped_models_match_the_fallback_bit_for_bit() {
    let mut bad = Vec::new();
    let mut compared = 0usize;
    for (k, (f0, f1)) in RHS.iter().enumerate() {
        let src = format!("par a=1\ninit x=0, y=0\nx'={f0}\ny'={f1}\ndone\n");
        let (mut ir, fb) = build(&src);
        let (n, n_p) = (ir.n(), ir.n_p());
        for (name, kind) in kinds() {
            let entry = ir.entry(&kind).expect("entry");
            let v_len = packed_len(&kind, n, n_p);
            for (pt, (x, a)) in POINTS.iter().enumerate() {
                for (d, dir) in DIRS.iter().enumerate() {
                    if v_len == 0 && d > 0 {
                        continue;
                    }
                    let v = &dir[..v_len];
                    let p = [a[0], a[1]];
                    let mut got = vec![f64::MAX; entry.outputs.len()];
                    eval_entry(&ir.dag, &entry, x, &p, v, &mut got);
                    let mut want = vec![f64::MAX; entry.outputs.len()];
                    fb.eval(&kind, x, &p, v, &mut want);
                    // Same values from the entry that also tracks an error bound.
                    let mut got_err = vec![f64::MAX; entry.outputs.len()];
                    let mut err = vec![f64::MAX; entry.outputs.len()];
                    eval_entry_err(&ir.dag, &entry, x, &p, v, &mut got_err, &mut err);
                    for i in 0..got.len() {
                        compared += 1;
                        if !same(got[i], want[i]) || !same(got[i], got_err[i]) {
                            bad.push(format!(
                                "rhs {k} {name} pt {pt} dir {d} out {i}: dag {:016x} err {:016x} \
                                 fallback {:016x} ({:?} vs {:?})",
                                got[i].to_bits(),
                                got_err[i].to_bits(),
                                want[i].to_bits(),
                                got[i],
                                want[i]
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 5_000, "only {compared} values compared");
    assert!(
        bad.is_empty(),
        "{} of {compared} values differ, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(12)]
    );
}

#[test]
fn a_negative_zero_derivative_is_stored_as_plus_zero() {
    // d/dx of -(max(x,0)^2) at x = 0: the clamp is active, the derivative is a zero.
    let (mut ir, fb) = build("par a=1\ninit x=0\nx'=-(max(x,0)^2)\ndone\n");
    for kind in [
        EntryKind::JacU,
        EntryKind::Dir(vec![Slot::U]),
        EntryKind::Dir(vec![Slot::U, Slot::U]),
    ] {
        let e = ir.entry(&kind).unwrap();
        let v: Vec<f64> = vec![1.0; packed_len(&kind, 1, 1)];
        let mut got = [f64::MAX];
        eval_entry(&ir.dag, &e, &[0.0], &[1.0, 0.0], &v, &mut got);
        assert_eq!(got[0].to_bits(), 0.0f64.to_bits(), "{kind:?} dag");
        let mut want = [f64::MAX];
        fb.eval(&kind, &[0.0], &[1.0, 0.0], &v, &mut want);
        assert_eq!(want[0].to_bits(), 0.0f64.to_bits(), "{kind:?} fallback");
    }
}

#[test]
fn f_and_aux_keep_a_negative_zero() {
    let (mut ir, fb) = build("par a=1\ninit x=0\naux w=-max(x,0)\nx'=-max(x,0)\ndone\n");
    for kind in [EntryKind::F, EntryKind::Aux] {
        let e = ir.entry(&kind).unwrap();
        let mut got = [f64::MAX];
        eval_entry(&ir.dag, &e, &[0.0], &[1.0, 0.0], &[], &mut got);
        assert_eq!(got[0].to_bits(), (-0.0f64).to_bits(), "{kind:?} dag");
        let mut want = [f64::MAX];
        fb.eval(&kind, &[0.0], &[1.0, 0.0], &[], &mut want);
        assert_eq!(want[0].to_bits(), (-0.0f64).to_bits(), "{kind:?} fallback");
    }
}

#[test]
fn normalizing_changes_no_other_value() {
    // A nonzero value, an infinity and a NaN pass through `out + 0.0` unchanged. The model
    // is the identity map x -> x scaled by a, and its derivative entry is a.
    let (mut ir, _) = build("par a=1\ninit x=0\nx'=a*x\ndone\n");
    let e = ir.entry(&EntryKind::JacU).unwrap();
    for a in [
        2.5,
        -3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e-310,
        f64::NAN,
    ] {
        let mut got = [f64::MAX];
        eval_entry(&ir.dag, &e, &[0.5], &[a, 0.0], &[], &mut got);
        assert!(same(got[0], a), "{a:?} became {:?}", got[0]);
    }
}
