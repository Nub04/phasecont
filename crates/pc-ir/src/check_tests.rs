use super::*;
use crate::lower::lower;
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn setup(src: &str) -> (Fallback, IrModel) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    (Fallback::new(&l.expanded, &l.info, &ir), ir)
}

fn low_order(points: usize, seed: u64, max_order: usize) -> Options {
    Options {
        max_order,
        ..Options::new(points, seed)
    }
}

#[test]
fn ulp_is_the_spacing() {
    assert_eq!(ulp(1.0), f64::EPSILON);
    assert_eq!(ulp(-1.0), f64::EPSILON);
    assert_eq!(ulp(2.0), 2.0 * f64::EPSILON);
    assert!(ulp(f64::NAN).is_nan());
    assert!(ulp(f64::INFINITY).is_nan());
    assert!(ulp(0.0) > 0.0);
}

#[test]
fn scale_skips_non_finite_entries() {
    assert_eq!(scale(&[]), 0.0);
    assert_eq!(scale(&[1.0, -3.0, f64::NAN, f64::INFINITY, 2.0]), 3.0);
}

#[test]
fn close_follows_the_spec() {
    assert!(close(f64::NAN, f64::NAN, 1.0));
    assert!(!close(f64::NAN, 1.0, 1.0));
    assert!(close(f64::INFINITY, f64::INFINITY, 1.0));
    assert!(!close(f64::INFINITY, 1.0, 1e300));
    assert!(close(1.0, 1.0 + 4.0 * f64::EPSILON, 0.0));
    assert!(!close(1.0, 1.0 + 8.0 * f64::EPSILON, 0.0));
    // The scale term: 1e-13 of s, however small the values themselves.
    assert!(close(1e-20, 9e-14, 1.0));
    assert!(!close(1e-20, 2e-13, 1.0));
    assert!(!close(1e-20, 9e-14, 0.0));
    assert!(close(5.0, 5.0 + 3e-13, 4.0));
}

#[test]
fn within_is_relative_to_the_scale() {
    assert!(within(1.0, 1.0 + 1e-7, 1.0, 1e-6));
    assert!(!within(1.0, 1.0 + 1e-5, 1.0, 1e-6));
    assert!(within(f64::NAN, f64::NAN, 1.0, 1e-6));
}

#[test]
fn rng_is_deterministic_and_in_range() {
    let (mut a, mut b) = (Rng(7), Rng(7));
    for _ in 0..100 {
        let (x, y) = (a.unit(), b.unit());
        assert_eq!(x, y);
        assert!((0.0..1.0).contains(&x));
    }
    assert_ne!(Rng(7).next_u64(), Rng(8).next_u64());
    // First SplitMix64 output for seed 0.
    assert_eq!(Rng(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    for _ in 0..100 {
        let r = a.range(-2.0, 3.0);
        assert!((-2.0..3.0).contains(&r));
    }
}

#[test]
fn slot_mix_counts() {
    let counts: Vec<usize> = (0..=5).map(|k| slot_mixes(k).len()).collect();
    assert_eq!(counts, [1, 3, 9, 27, 81, 243]);
}

#[test]
fn a_model_with_aux_passes_every_check() {
    let (fb, mut ir) = setup(
        "par a=1.3, b=0.7\ninit x=0.4, y=0.6\n\
         x'=a*x*y-x^3+b*y^2-x\ny'=x^2*y-y^4+a*b*x\naux s=x*y+a\ndone\n",
    );
    let rep = check_model(&mut ir, &fb, 2, 3);
    assert!(rep.passed(), "{rep}");
    for label in [
        "F",
        "Aux",
        "JacU",
        "JacP",
        "Dir[U,U,U]",
        "Dir[Full,P,U,U,P]",
        "A (FD)",
        "f_alpha (FD)",
        "B (FD)",
        "C (FD)",
    ] {
        assert!(rep.find(label).is_some(), "missing {label}\n{rep}");
    }
    // F, Aux, JacU, JacP, 363 directional entries and 4 finite-difference rows.
    assert_eq!(rep.entries.len(), 4 + 363 + 4);
}

#[test]
fn zero_parameters_and_zero_aux_do_not_panic() {
    let (fb, mut ir) = setup("init x=0.5\nx'=-x^3+sin(x)\ndone\n");
    assert_eq!((ir.n_p(), ir.aux.len()), (0, 0));
    let rep = check_model(&mut ir, &fb, 2, 1);
    assert!(rep.passed(), "{rep}");
    assert!(rep.find("Aux").is_none() && rep.find("JacP").is_none());
    assert!(rep.find("f_alpha (FD)").is_none());
    // Only the all-U mixes exist without parameters: one per order.
    let dirs = rep
        .entries
        .iter()
        .filter(|e| e.label.starts_with("Dir["))
        .count();
    assert_eq!(dirs, 5);
}

#[test]
fn a_wrong_reference_is_reported_with_its_point() {
    let (_fb, mut ir) = setup("init x=0.5\nx'=2*x\ndone\n");
    let (fb_other, _) = setup("init x=0.5\nx'=3*x\ndone\n");
    let rep = check_model_with(&mut ir, &fb_other, &low_order(2, 5, 1));
    assert!(!rep.passed());
    let f = rep.find("F").unwrap();
    assert!(!f.pass && f.ratio > 1.0 && f.worst_err > 0.0);
    assert_eq!(f.x.len(), 1);
    assert_eq!(f.p.len(), 1);
    // The finite differences use only the DAG, so they still pass.
    assert!(rep.find("A (FD)").unwrap().pass);
    let text = rep.to_string();
    assert!(text.contains("FAIL") && text.contains("at x ="), "{text}");
}

#[test]
fn a_wrong_derivative_fails_the_finite_difference_check() {
    let (fb, mut ir) = setup("par a=1.5\ninit x=0.5\nx'=a*x^3\ndone\n");
    let e = ir.entry(&EntryKind::JacU).unwrap();
    let wrong = ir.dag.konst(1.0);
    ir.entries.insert(
        EntryKind::JacU,
        crate::entry::Entry {
            kind: EntryKind::JacU,
            outputs: vec![wrong; e.outputs.len()],
        },
    );
    let rep = check_model_with(&mut ir, &fb, &low_order(2, 5, 1));
    assert!(!rep.find("A (FD)").unwrap().pass, "{rep}");
    assert!(!rep.find("JacU").unwrap().pass, "{rep}");
    assert!(rep.find("F").unwrap().pass);
}

#[test]
fn display_lists_every_entry() {
    let (fb, mut ir) = setup("init x=0.5\nx'=-x\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(1, 1, 2));
    let text = rep.to_string();
    assert!(
        text.contains("Dir[U,U]") && text.contains("C (FD)"),
        "{text}"
    );
    let tail = format!("all {} checks pass", rep.entries.len());
    assert!(text.ends_with(&tail), "{text}");
}

#[test]
fn close_has_an_absolute_floor() {
    assert!(close(0.0, 2e-17, 1e-20));
    assert!(close(0.0, ABS_FLOOR, 1e-20));
    assert!(!close(0.0, 2e-15, 1e-20));
}
