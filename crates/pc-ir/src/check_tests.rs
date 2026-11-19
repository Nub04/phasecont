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

#[test]
fn a_finite_error_bound_adds_a_pass_condition() {
    // 1e-3 apart with scale 1: far outside 4 ulp, 1e-13 and 1e-15.
    let (a, b) = (1.0, 1.0 + 1e-3);
    assert!(!close(a, b, 1.0));
    // E = 1e13 allows 8 * eps * 1e13 = 8 * 2.2e-16 * 1e13 = 1.8e-2.
    assert!(close_err(a, b, 1.0, 1e13));
    // E = 1e11 allows 1.8e-4, not enough.
    assert!(!close_err(a, b, 1.0, 1e11));
    let m = allowed_diff(a, b, 1.0, Some(1e13));
    assert!(
        (m - ERR_FACTOR * f64::EPSILON * 1e13).abs() <= 1e-18,
        "{m:e}"
    );
}

#[test]
fn a_non_finite_error_bound_adds_nothing() {
    let (a, b) = (1.0, 1.0 + 1e-3);
    for e in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!close_err(a, b, 1.0, e), "{e}");
        assert_eq!(
            allowed_diff(a, b, 1.0, Some(e)),
            allowed_diff(a, b, 1.0, None)
        );
    }
    // The old conditions still apply with a non-finite bound.
    assert!(close_err(1.0, 1.0 + 1e-15, 1.0, f64::NAN));
    assert!(close_err(f64::NAN, f64::NAN, 1.0, f64::INFINITY));
}

#[test]
fn close_pins_the_conditions_without_a_bound() {
    // 4 ulp passes, 5 ulp does not (scale 0, so only the ulp term and the floor apply).
    // Near 8 the spacing is 8 eps, so 4 ulp is 32 eps, well above the absolute floor.
    let u = ulp(8.0);
    assert_eq!(u, 8.0 * f64::EPSILON);
    assert!(close(8.0, 8.0 + 4.0 * u, 0.0));
    assert!(!close(8.0, 8.0 + 5.0 * u, 0.0));
    assert!(close(8.0 + 4.0 * u, 8.0, 0.0));
    // 1e-13 of the scale, exactly and just above.
    for s in [1.0, 37.5, 1e6] {
        assert!(close(0.0, 1e-13 * s, s), "{s}");
        assert!(!close(0.0, 1.000_001e-13 * s, s), "{s}");
    }
    // The absolute floor, with a scale too small to matter.
    assert!(close(0.0, 1e-15, 1e-20));
    assert!(!close(0.0, 1.000_001e-15, 1e-20));
    // The allowed difference is the largest of the three terms, and `None` adds no fourth.
    assert_eq!(allowed_diff(0.0, 0.0, 1e-20, None), ABS_FLOOR);
    assert_eq!(allowed_diff(0.0, 0.0, 10.0, None), 1e-13 * 10.0);
    assert_eq!(allowed_diff(8.0, 8.0, 0.0, None), 4.0 * u);
}

#[test]
fn a_vector_that_passes_only_through_the_bound_has_a_ratio_below_one() {
    // The model computes x' = (x + 1e6) - 1e6, which loses about 1e-10 to rounding. The
    // reference is x' = x. The error is far above 1e-13 s and 4 ulp, so only the running
    // error bound (E about 1e6, so 8 eps E about 2e-9) lets the row pass.
    let (_, mut ir) = setup("init x=0.5\nx'=(x+1000000)-1000000\ndone\n");
    let (fb, _) = setup("init x=0.5\nx'=x\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(8, 3, 1));
    let f = rep.find("F").unwrap();
    assert!(f.pass, "{rep}");
    assert!(f.ratio <= 1.0 && f.ratio > 0.0, "{}", f.ratio);
    assert!(
        f.worst_err > 100.0 * 1e-13 * f.scale,
        "the old rule would fail: err {:e}, scale {:e}",
        f.worst_err,
        f.scale
    );
}

#[test]
fn close_ulps_is_four_ulp_with_nan_and_zero_rules() {
    assert!(close_ulps(1.0, 1.0));
    assert!(close_ulps(1.0, 1.0 + 4.0 * f64::EPSILON));
    assert!(!close_ulps(1.0, 1.0 + 5.0 * f64::EPSILON));
    // The spacing is taken at the larger magnitude, as in SPEC section 3.
    assert!(close_ulps(2.0, 2.0 + 4.0 * ulp(2.0)));
    assert!(!close_ulps(2.0, 2.0 + 5.0 * ulp(2.0)));
    // A NaN matches any NaN and nothing else.
    assert!(close_ulps(f64::NAN, -f64::NAN));
    assert!(!close_ulps(f64::NAN, 1.0));
    assert!(!close_ulps(1.0, f64::NAN));
    // Zeros of either sign are equal, and a tiny value is not close to a normal one.
    assert!(close_ulps(0.0, -0.0));
    assert!(!close_ulps(1e-300, 2e-300));
    // Infinities match only themselves.
    assert!(close_ulps(f64::INFINITY, f64::INFINITY));
    assert!(!close_ulps(f64::INFINITY, f64::NEG_INFINITY));
    assert!(!close_ulps(f64::INFINITY, f64::MAX));
    // Unlike the section 3 rule, no scale or absolute floor widens it.
    assert!(close(1e-14, 2e-14, 1.0));
    assert!(!close_ulps(1e-14, 2e-14));
}

/// A model whose derivative entries pass the node limit while `F` still fits.
fn setup_capped(src: &str) -> (Fallback, IrModel) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let full = lower(&l.expanded, &l.info).unwrap().dag.len();
    let ir = crate::lower::lower_with_cap(&l.expanded, &l.info, full).unwrap();
    (Fallback::new(&l.expanded, &l.info, &ir), ir)
}

#[test]
fn an_entry_over_the_node_limit_is_skipped_and_does_not_fail() {
    let (fb, mut ir) = setup_capped("x'=sin(x)*cos(x)+x*x\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(2, 1, 2));
    let text = rep.to_string();
    assert!(rep.passed(), "{text}");
    assert_eq!(rep.failures().count(), 0, "{text}");
    for label in ["JacU", "Dir[U]", "Dir[U,U]", "FD"] {
        let r = rep.find(label).unwrap_or_else(|| panic!("{label}: {text}"));
        assert!(r.pass, "{label}: {text}");
        let n = r.note.as_deref().unwrap_or("");
        assert!(
            n.contains("entry not built") && n.contains("TooLarge"),
            "{text}"
        );
    }
    assert!(text.contains("SKIP"), "{text}");
    assert!(!text.contains("FAIL"), "{text}");
    assert!(rep.find("F").unwrap().pass);
}

#[test]
fn an_entry_that_cannot_be_built_for_another_reason_still_fails() {
    use crate::error::{IrError, IrErrorKind};
    let e = IrError::new(IrErrorKind::Unsupported, pc_ode::DUMMY_SPAN, "no rule");
    let r = EntryReport::unbuilt("Dir[U]".into(), &e);
    assert!(!r.pass && !r.skip, "{r}");
    assert!(r.to_string().contains("FAIL"), "{r}");
    let e = IrError::new(IrErrorKind::TooLarge, pc_ode::DUMMY_SPAN, "too big");
    let r = EntryReport::unbuilt("Dir[U]".into(), &e);
    assert!(r.pass && r.skip, "{r}");
    assert!(r.to_string().contains("SKIP"), "{r}");
}

const NAN_HALF: &str = "par g=0.5, q=2.5
init x=0,y=0
x'=g*y^q
y'=-y
done
";

#[test]
fn sample_points_with_a_non_finite_model_value_are_redrawn_in_every_row() {
    let (fb, mut ir) = setup(NAN_HALF);
    let rep = check_model_with(&mut ir, &fb, &low_order(40, 1, 3));
    let text = rep.to_string();
    assert!(rep.passed(), "{text}");
    // Every row, the comparisons with the fallback and the finite differences alike, ran
    // on 40 usable points, and drew more than 40 to get them.
    for r in &rep.entries {
        assert_eq!(r.points, 40, "{}: {text}", r.label);
        assert!(r.skipped_points > 0, "{}: {text}", r.label);
        assert!(r.compared > 0, "{}: {text}", r.label);
        let line = r.to_string();
        let drawn = 40 + r.skipped_points;
        let want = format!("(40 usable of {drawn} drawn");
        assert!(line.contains(&want), "{line}");
    }
}

#[test]
fn a_row_draws_at_most_ten_times_the_requested_points() {
    let (fb, mut ir) = setup("par q=2.5\ninit x=0\nx'=(0-1-x*x)^q\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(5, 1, 1));
    for r in &rep.entries {
        assert!(
            r.points == 0 && r.skipped_points == 50,
            "{}: {rep}",
            r.label
        );
        assert!(!r.pass && r.compared == 0, "{}: {rep}", r.label);
        let line = r.to_string();
        assert!(line.contains("no usable point"), "{line}");
        assert!(line.contains("all 50 drawn points"), "{line}");
    }
    assert!(!rep.passed());
}

#[test]
fn the_draws_of_a_model_that_is_finite_everywhere_are_the_old_ones() {
    // `usable_points` draws `x`, `p`, then each direction vector, in this order and from the
    // one generator, so the first `points` draws are exactly what the checks drew before
    // points could be dropped.
    let (fb, mut ir) = setup("par a=1.5\ninit x=0.5,y=0.2\nx'=a*x*y\ny'=-y\ndone\n");
    ir.entry(&EntryKind::F).unwrap();
    let opt = low_order(6, 7, 1);
    let (found, drawn) = usable_points(&ir, &fb, &mut Rng(7), &opt, &[2, 3]);
    assert_eq!((found.len(), drawn), (6, 6));
    let mut rng = Rng(7);
    for s in &found {
        let (x, p) = draw_point(&ir, &mut rng, opt.half_width);
        let d0: Vec<f64> = (0..2).map(|_| rng.range(-1.0, 1.0)).collect();
        let d1: Vec<f64> = (0..3).map(|_| rng.range(-1.0, 1.0)).collect();
        assert_eq!((&s.x, &s.p, &s.dirs), (&x, &p, &vec![d0, d1]));
    }
}

#[test]
fn a_redraw_takes_the_next_draw_and_keeps_the_order_of_the_stream() {
    let (fb, mut ir) = setup(NAN_HALF);
    ir.entry(&EntryKind::F).unwrap();
    let opt = low_order(30, 3, 1);
    let (found, drawn) = usable_points(&ir, &fb, &mut Rng(3), &opt, &[2]);
    assert_eq!(found.len(), 30);
    assert!(drawn > 30 && drawn <= 300, "{drawn}");
    // The same call again gives the same points.
    let (again, drawn2) = usable_points(&ir, &fb, &mut Rng(3), &opt, &[2]);
    assert_eq!(drawn, drawn2);
    for (a, b) in found.iter().zip(&again) {
        assert_eq!((&a.x, &a.p, &a.dirs), (&b.x, &b.p, &b.dirs));
    }
    // Every kept point has a finite model value in the fallback, every dropped one does not.
    // Here the DAG and the fallback agree everywhere, so the DAG's value is not finite at
    // exactly the dropped points too.
    let mut rng = Rng(3);
    let mut kept = 0;
    for _ in 0..drawn {
        let (x, p) = draw_point(&ir, &mut rng, opt.half_width);
        let _dir: Vec<f64> = (0..2).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut want = vec![f64::NAN; 2];
        fb.eval(&EntryKind::F, &x, &p, &[], &mut want);
        let finite = want.iter().all(|y| y.is_finite());
        let dag_finite = run(&ir, &EntryKind::F, &x, &p, &[])
            .iter()
            .all(|y| y.is_finite());
        assert_eq!(finite, dag_finite);
        if finite {
            assert_eq!(found[kept].x, x);
            kept += 1;
        }
    }
    assert_eq!(kept, 30);
}

#[test]
fn a_model_with_finite_values_everywhere_skips_no_point() {
    let (fb, mut ir) = setup("par a=1.5\ninit x=0.5\nx'=a*x^3\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(5, 1, 3));
    assert!(rep.passed(), "{rep}");
    assert!(rep.entries.iter().all(|e| e.skipped_points == 0 && !e.skip));
    assert!(!rep.to_string().contains("skipped"), "{rep}");
}

#[test]
fn a_finite_model_value_with_a_non_finite_derivative_is_compared_and_fails() {
    // x is drawn in [0.5, 1.5), so exp(2000*x) overflows at every sample point
    // and on both sides of every difference. f = atan(inf) = pi/2 is finite,
    // while the exact derivative is 2000*inf * 1/(1+inf) = inf * 0 = NaN. Only
    // a non-finite f skips a point, so the A row compares every point and fails.
    let (fb, mut ir) = setup("init x=1\nx'=atan(exp(2000*x))\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(5, 1, 1));
    let a = rep.find("A (FD)").unwrap();
    assert!(!a.pass && a.points == 5 && a.skipped_points == 0, "{rep}");
    assert!(!rep.passed());
}

/// `F` of the DAG and of the fallback at `x`, with `t = 0`. The tests below use it to check
/// that the two evaluators differ where a test says they do.
fn f_pair(ir: &mut IrModel, fb: &Fallback, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    ir.entry(&EntryKind::F).unwrap();
    let p = [0.0];
    let got = run(ir, &EntryKind::F, x, &p, &[]);
    let mut want = vec![f64::NAN; got.len()];
    fb.eval(&EntryKind::F, x, &p, &[], &mut want);
    (got, want)
}

/// The `F` row of a check of `dag_src` against the fallback of `fb_src`, as in
/// `a_wrong_reference_is_reported_with_its_point`.
fn f_row_of_pair(dag_src: &str, fb_src: &str, points: usize, seed: u64) -> (Report, EntryReport) {
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let rep = check_model_with(&mut ir, &fb, &low_order(points, seed, 1));
    let f = rep.find("F").unwrap().clone();
    (rep, f)
}

#[test]
fn a_dag_value_that_is_nan_where_the_fallback_is_finite_fails_the_f_row() {
    // The DAG's F is x + 1e-300 * sqrt(x), NaN for x < 0. The fallback's F is x. The two
    // agree for x >= 0 (the extra term is far below an ulp) and disagree for x < 0, where
    // x is drawn in [-0.5, 0.5).
    let dag_src = "init x=0\nx'=x+1e-300*sqrt(x)\ndone\n";
    let fb_src = "init x=0\nx'=x\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[-0.25]);
    assert!(got[0].is_nan() && want[0] == -0.25, "{got:?} {want:?}");
    let (got, want) = f_pair(&mut ir, &fb, &[0.25]);
    assert_eq!(got, want);

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 20, 1);
    // No draw is dropped: the point where only the DAG is NaN is kept and compared.
    assert_eq!((f.points, f.skipped_points), (20, 0), "{rep}");
    assert!(
        !f.pass && f.compared == 20 && f.ratio.is_infinite(),
        "{rep}"
    );
    assert!(f.x[0] < 0.0, "{rep}");
    assert!(!rep.passed());
}

#[test]
fn a_dag_value_that_is_finite_where_the_fallback_is_nan_fails_the_f_row() {
    let dag_src = "init x=0\nx'=x\ndone\n";
    let fb_src = "init x=0\nx'=x+1e-300*sqrt(x)\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[-0.25]);
    assert!(got[0] == -0.25 && want[0].is_nan(), "{got:?} {want:?}");

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 20, 1);
    assert_eq!((f.points, f.skipped_points), (20, 0), "{rep}");
    assert!(
        !f.pass && f.compared == 20 && f.ratio.is_infinite(),
        "{rep}"
    );
    assert!(f.x[0] < 0.0, "{rep}");
}

#[test]
fn infinities_of_opposite_sign_fail_the_f_row_and_are_compared() {
    // exp(1000 + x) overflows at every sample point: +inf in the DAG, -inf in the fallback.
    let dag_src = "init x=0\nx'=exp(1000+x)\ndone\n";
    let fb_src = "init x=0\nx'=-exp(1000+x)\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[0.1]);
    assert_eq!((got[0], want[0]), (f64::INFINITY, f64::NEG_INFINITY));

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 12, 1);
    // The row compared the 12 values and found them different: it did not skip every draw
    // and fail for having nothing to compare.
    assert_eq!(
        (f.points, f.skipped_points, f.compared),
        (12, 0, 12),
        "{rep}"
    );
    assert!(!f.pass && f.ratio.is_infinite(), "{rep}");
    let line = f.to_string();
    assert!(!line.contains("nothing was compared"), "{line}");
}

#[test]
fn an_infinity_in_the_dag_against_nan_in_the_fallback_fails_the_f_row() {
    // The fallback computes inf - inf, which is NaN, where the DAG gives +inf.
    let dag_src = "init x=0\nx'=exp(1000+x)\ndone\n";
    let fb_src = "init x=0\nx'=exp(1000+x)-exp(1001+x)\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[0.1]);
    assert!(
        got[0] == f64::INFINITY && want[0].is_nan(),
        "{got:?} {want:?}"
    );

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 12, 1);
    assert_eq!(
        (f.points, f.skipped_points, f.compared),
        (12, 0, 12),
        "{rep}"
    );
    assert!(!f.pass && f.ratio.is_infinite(), "{rep}");
    let line = f.to_string();
    assert!(!line.contains("nothing was compared"), "{line}");
}

#[test]
fn a_draw_that_is_nan_in_the_dag_and_in_the_fallback_is_skipped_and_counted() {
    // sqrt(x) in the DAG and x^0.5 in the fallback are NaN for x < 0 and agree within a few
    // ulps for x >= 0. The draws with x < 0 are dropped, no row fails, and the rows count
    // what they dropped.
    let dag_src = "init x=0\nx'=sqrt(x)\ndone\n";
    let fb_src = "init x=0\nx'=x^0.5\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[-0.25]);
    assert!(got[0].is_nan() && want[0].is_nan(), "{got:?} {want:?}");
    let (got, want) = f_pair(&mut ir, &fb, &[0.25]);
    assert_eq!((got[0], want[0]), (0.5, 0.5));

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 20, 1);
    assert!(f.pass, "{rep}");
    assert_eq!((f.points, f.compared), (20, 20), "{rep}");
    assert!(f.skipped_points > 0, "{rep}");
    let drawn = 20 + f.skipped_points;
    let line = f.to_string();
    assert!(
        line.contains(&format!("(20 usable of {drawn} drawn")),
        "{line}"
    );
}

#[test]
fn a_nan_shared_by_both_evaluators_does_not_hide_a_finite_mismatch_at_another_element() {
    // Element 0 is NaN on both sides for x < 0. Element 1 is 2y in the DAG and y in the
    // fallback, which differ wherever y is not tiny. The draw is kept, so the F row sees
    // the mismatch at every point, those with x < 0 included.
    let dag_src = "init x=0,y=0.5\nx'=sqrt(x)\ny'=2*y\ndone\n";
    let fb_src = "init x=0,y=0.5\nx'=sqrt(x)\ny'=y\ndone\n";
    let (_, mut ir) = setup(dag_src);
    let (fb, _) = setup(fb_src);
    let (got, want) = f_pair(&mut ir, &fb, &[-0.25, 0.5]);
    assert!(got[0].is_nan() && want[0].is_nan(), "{got:?} {want:?}");
    assert_eq!((got[1], want[1]), (1.0, 0.5));

    let (rep, f) = f_row_of_pair(dag_src, fb_src, 20, 1);
    assert_eq!((f.points, f.skipped_points), (20, 0), "{rep}");
    assert!(!f.pass && f.ratio > 1.0 && f.ratio.is_finite(), "{rep}");
}

#[test]
fn the_draws_are_the_same_every_run_when_the_two_evaluators_disagree() {
    let dag_src = "init x=0\nx'=x+1e-300*sqrt(x)\ndone\n";
    let fb_src = "init x=0\nx'=x\ndone\n";
    let (r1, f1) = f_row_of_pair(dag_src, fb_src, 15, 4);
    let (r2, f2) = f_row_of_pair(dag_src, fb_src, 15, 4);
    assert_eq!(r1.to_string(), r2.to_string());
    assert_eq!((f1.x, f1.p), (f2.x, f2.p));
}

#[test]
fn usable_points_keeps_every_draw_where_the_two_evaluators_disagree() {
    // The DAG is NaN for x < 0 and the fallback is finite there. The draws are the first
    // ones of the stream, none is replaced, and the stream is the one a model that agrees
    // with its fallback gives.
    let (_, mut ir) = setup("init x=0\nx'=x+1e-300*sqrt(x)\ndone\n");
    let (fb, _) = setup("init x=0\nx'=x\ndone\n");
    ir.entry(&EntryKind::F).unwrap();
    let opt = low_order(25, 9, 1);
    let (found, drawn) = usable_points(&ir, &fb, &mut Rng(9), &opt, &[1]);
    assert_eq!((found.len(), drawn), (25, 25));
    assert!(found.iter().any(|s| s.x[0] < 0.0));
    let mut rng = Rng(9);
    for s in &found {
        let (x, p) = draw_point(&ir, &mut rng, opt.half_width);
        let d: Vec<f64> = vec![rng.range(-1.0, 1.0)];
        assert_eq!((&s.x, &s.p, &s.dirs), (&x, &p, &vec![d]));
    }
}

#[test]
fn usable_points_drops_nothing_when_the_f_entry_is_not_built() {
    // Without an F entry no point can be tested against it, as before.
    let (_, ir) = setup(NAN_HALF);
    let (fb, _) = setup(NAN_HALF);
    let opt = low_order(10, 3, 1);
    let (found, drawn) = usable_points(&ir, &fb, &mut Rng(3), &opt, &[2]);
    assert_eq!((found.len(), drawn), (10, 10));
}

#[test]
fn the_node_limit_defaults_to_max_nodes_and_a_hook_lowers_it() {
    let (_, mut ir) = setup("init x=0.5\nx'=sin(x)*cos(x)+x*x\ndone\n");
    assert_eq!(ir.node_limit(), crate::dag::MAX_NODES);
    assert!(ir.entry(&EntryKind::JacU).is_ok());
    // A limit at the current size leaves no room for a new node. An entry built before the
    // change stays available, a new one is refused as too large.
    let size = ir.dag.len();
    ir.set_node_limit(size);
    assert_eq!(ir.node_limit(), size);
    assert!(ir.entry(&EntryKind::JacU).is_ok());
    let e = ir
        .entry(&EntryKind::Dir(vec![Slot::U, Slot::U]))
        .unwrap_err();
    assert_eq!(e.kind, IrErrorKind::TooLarge);
    // The refused request left the DAG as it was, and the limit can be raised again.
    assert_eq!(ir.dag.len(), size);
    assert!(!ir.dag.exceeded());
    ir.set_node_limit(crate::dag::MAX_NODES);
    assert!(ir.entry(&EntryKind::Dir(vec![Slot::U, Slot::U])).is_ok());
}

#[test]
fn a_lowered_node_limit_gives_skip_rows_and_no_failure() {
    // Lowered normally, then limited through the hook: no entry beyond F can be built.
    let (fb, mut ir) = setup("par a=1.5\ninit x=0.5,y=0.2\nx'=a*sin(x)*y\ny'=cos(x)-y*y\ndone\n");
    ir.set_node_limit(ir.dag.len());
    let rep = check_model_with(&mut ir, &fb, &low_order(3, 1, 2));
    let text = rep.to_string();
    assert!(rep.passed(), "{text}");
    assert_eq!(rep.failures().count(), 0, "{text}");
    let skipped: Vec<&str> = rep
        .entries
        .iter()
        .filter(|e| e.skip)
        .map(|e| e.label.as_str())
        .collect();
    for label in ["JacU", "JacP", "Dir[U]", "Dir[U,U]", "FD"] {
        assert!(skipped.contains(&label), "{label} not skipped: {text}");
    }
    assert!(!rep.find("F").unwrap().skip && rep.find("F").unwrap().pass);
    assert!(text.contains("SKIP") && !text.contains("FAIL"), "{text}");
    assert!(text.contains("entry too large"), "{text}");
}

/// Our own probe model for the rounding allowance. `df/dq` is `y^q ln(y) / (y^q + 1)^2`,
/// about 1e-10 at the sample points with small `y`, while `|f|` is about 0.2, so the
/// difference `f(q + h) - f(q - h)` is a few ulp of `f`: rounding, not a wrong derivative.
/// `y^q` with `y < 0` and a non-integer `q` is NaN, so half of the drawn points have no
/// finite model value.
const POWNEG: &str = "# own probe model
par q=4
init x=0,y=0
x'=-x+y^q/(y^q+1)
y'=-y
done
";

#[test]
fn powneg_passes_every_row_on_seeds_1_to_60() {
    let mut failing = Vec::new();
    for seed in 1..=60 {
        let (fb, mut ir) = setup(POWNEG);
        let rep = check_model_with(&mut ir, &fb, &low_order(20, seed, 1));
        if !rep.passed() {
            let rows: Vec<String> = rep.failures().map(|e| e.to_string()).collect();
            failing.push(format!("seed {seed}: {}", rows.join(" | ")));
        }
    }
    assert!(failing.is_empty(), "{}", failing.join("\n"));
}

/// A model whose derivative entries all have a value of order one and curvature, so a
/// relative error of 1e-3 in one entry is far above every allowance.
const CURVED: &str = "par a=1.5, b=0.8
init x=0.5,y=0.2
x'=a*sin(x)*y+b*x*x
y'=cos(x)-y*y+a*b*x
done
";

/// Replaces the outputs of `kind` by `factor` times the true outputs.
fn scale_entry(ir: &mut IrModel, kind: &EntryKind, factor: f64) {
    let e = ir.entry(kind).unwrap();
    let k = ir.dag.konst(factor);
    let outputs = e.outputs.iter().map(|&o| ir.dag.mul(o, k)).collect();
    ir.entries.insert(
        kind.clone(),
        crate::entry::Entry {
            kind: kind.clone(),
            outputs,
        },
    );
}

#[test]
fn an_entry_off_by_one_in_a_thousand_fails_its_finite_difference_row() {
    let cases = [
        (EntryKind::JacU, "A (FD)"),
        (EntryKind::JacP, "f_alpha (FD)"),
        (EntryKind::Dir(vec![Slot::U, Slot::U]), "B (FD)"),
        (EntryKind::Dir(vec![Slot::U, Slot::U, Slot::U]), "C (FD)"),
    ];
    for (kind, row) in cases {
        for seed in 1..=3 {
            let (fb, mut ir) = setup(CURVED);
            scale_entry(&mut ir, &kind, 1.001);
            let rep = check_model_with(&mut ir, &fb, &low_order(10, seed, 1));
            let r = rep.find(row).unwrap();
            assert!(!r.pass && r.ratio > 1.0, "{row} seed {seed}: {rep}");
        }
    }
}

#[test]
fn the_unchanged_curved_model_passes_every_finite_difference_row() {
    for seed in 1..=10 {
        let (fb, mut ir) = setup(CURVED);
        let rep = check_model_with(&mut ir, &fb, &low_order(10, seed, 1));
        assert!(rep.passed(), "seed {seed}: {rep}");
    }
}

#[test]
fn a_wrong_jac_p_fails_on_powneg_even_with_the_rounding_allowance() {
    for seed in 1..=5 {
        let (fb, mut ir) = setup(POWNEG);
        let e = ir.entry(&EntryKind::JacP).unwrap();
        let wrong = ir.dag.konst(1.0);
        ir.entries.insert(
            EntryKind::JacP,
            crate::entry::Entry {
                kind: EntryKind::JacP,
                outputs: vec![wrong; e.outputs.len()],
            },
        );
        let rep = check_model_with(&mut ir, &fb, &low_order(20, seed, 1));
        let r = rep.find("f_alpha (FD)").unwrap();
        assert!(!r.pass && r.ratio > 1.0, "seed {seed}: {rep}");
    }
}

#[test]
fn an_element_that_is_nan_on_both_sides_is_agreement_and_not_a_compared_value() {
    assert_eq!(element(f64::NAN, f64::NAN, 1.0), None);
    assert_eq!(element(2.0, 2.0, 1.0), Some((0.0, 0.0)));
    // Equal infinities agree; their difference would be NaN.
    assert_eq!(element(f64::INFINITY, f64::INFINITY, 1.0), Some((0.0, 0.0)));
    let (err, r) = element(f64::NAN, 1.0, 1.0).unwrap();
    assert!(err.is_nan() && r.is_infinite());
    assert_eq!(element(1.0, 1.5, 0.25), Some((0.5, 2.0)));

    let mut rep = EntryReport::new("t".into());
    rep.record([None, None].into_iter(), 1.0, &[0.5], &[0.2]);
    assert_eq!((rep.points, rep.compared), (1, 0));
    assert_eq!(rep.worst_err, 0.0);
    assert!(rep.pass && rep.x.is_empty());
    // A NaN pair beside a compared element leaves the worst error to the compared one.
    rep.record(
        [None, Some((1e-3, 0.5)), None].into_iter(),
        2.0,
        &[0.7],
        &[0.4],
    );
    assert_eq!((rep.points, rep.compared), (2, 1));
    assert_eq!((rep.worst_err, rep.ratio, rep.scale), (1e-3, 0.5, 2.0));
    assert_eq!(
        (rep.x.as_slice(), rep.p.as_slice()),
        (&[0.7][..], &[0.4][..])
    );
    // A later, smaller ratio does not replace it; a larger one does, and a ratio above 1 fails.
    rep.record([Some((5e-4, 0.25))].into_iter(), 3.0, &[0.1], &[0.1]);
    assert_eq!((rep.worst_err, rep.compared), (1e-3, 2));
    rep.record([Some((4.0, 8.0))].into_iter(), 4.0, &[0.9], &[0.9]);
    assert_eq!((rep.worst_err, rep.ratio, rep.pass), (4.0, 8.0, false));
}

#[test]
fn the_first_compared_element_sets_the_worst_error_even_when_it_is_zero() {
    let mut rep = EntryReport::new("t".into());
    rep.record([Some((0.0, 0.0))].into_iter(), 1.0, &[0.5], &[0.2]);
    assert_eq!(
        (rep.compared, rep.worst_err, rep.x.as_slice()),
        (1, 0.0, &[0.5][..])
    );
    let rep = rep.finish();
    assert!(rep.pass && rep.note.is_none());
}

#[test]
fn a_row_with_zero_compared_values_fails_and_says_nothing_was_compared() {
    // f = atan(exp(2000 q)) is pi/2, finite. Its derivative in q is 2000 exp(2000 q)
    // / (1 + exp(4000 q)) = inf / inf in the DAG and in the fallback alike: NaN on both
    // sides at every point, so the JacP row has points and compares no value.
    let (fb, mut ir) = setup("par q=1\ninit x=1\nx'=atan(exp(2000*q))\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(5, 1, 1));
    let r = rep.find("JacP").unwrap();
    let line = r.to_string();
    assert_eq!(
        (r.points, r.compared, r.skipped_points),
        (5, 0, 0),
        "{line}"
    );
    assert!(!r.pass && !r.skip, "{line}");
    assert!(line.contains("FAIL") && !line.contains("pass"), "{line}");
    assert!(line.contains("nothing was compared"), "{line}");
    assert!(!line.contains("inf"), "{line}");
    assert_eq!(r.worst_err, 0.0);
    assert!(!rep.passed());
    // The rows that did compare values are unchanged.
    assert!(rep.find("F").unwrap().pass && rep.find("F").unwrap().compared == 5);
}

#[test]
fn powneg_jacp_compares_values_and_never_prints_worst_inf() {
    // Half of the drawn points have y < 0 and a non-integer q, where y^q and its derivative
    // are NaN on both sides. Before the redraw and the compared count, the JacP row of this
    // model printed "pass  worst inf  s 0  ratio 0" at such points.
    for seed in 1..=20 {
        let (fb, mut ir) = setup(POWNEG);
        let rep = check_model_with(&mut ir, &fb, &low_order(20, seed, 1));
        let r = rep.find("JacP").unwrap();
        let line = r.to_string();
        assert!(
            r.pass && r.points == 20 && r.compared == 40,
            "seed {seed}: {line}"
        );
        assert!(
            r.worst_err.is_finite() && r.scale > 0.0,
            "seed {seed}: {line}"
        );
        assert!(line.contains("20 points, 40 values"), "seed {seed}: {line}");
        assert!(!rep.to_string().contains("worst inf"), "seed {seed}: {rep}");
    }
}

#[test]
fn every_row_prints_how_many_values_it_compared() {
    let (fb, mut ir) = setup(CURVED);
    let rep = check_model_with(&mut ir, &fb, &low_order(7, 2, 1));
    let text = rep.to_string();
    assert!(rep.passed(), "{text}");
    // Two states and two parameters: F has 2 values per point, JacU 4, JacP 4, Dir[U] 2.
    for (label, per_point) in [
        ("F", 2),
        ("JacU", 4),
        ("JacP", 4),
        ("Dir[U]", 2),
        ("A (FD)", 4),
        ("f_alpha (FD)", 4),
        ("B (FD)", 2),
        ("C (FD)", 2),
    ] {
        let r = rep.find(label).unwrap();
        assert_eq!(r.compared, 7 * per_point, "{label}: {text}");
        let want = format!("7 points, {} values", 7 * per_point);
        assert!(r.to_string().contains(&want), "{label}: {text}");
    }
}

#[test]
fn a_zero_point_request_compares_nothing_and_fails() {
    let (fb, mut ir) = setup("init x=0.5\nx'=-x\ndone\n");
    let rep = check_model_with(&mut ir, &fb, &low_order(0, 1, 1));
    assert!(!rep.passed(), "{rep}");
    let f = rep.find("F").unwrap();
    assert!(f.to_string().contains("no point was drawn"), "{f}");
}

/// B and C are about 1e-8 of A here, so the finite differences of `A v` and `B(v, w)` cancel
/// to eight digits. The rounding of `A v` alone, over the step, is above `FD_TOL_HIGHER * |B|`
/// at some points; the allowance has to come from the `eval_err` bounds and not from a share
/// of `|A v|`.
const TINY_CURVATURE: &str = "init x=0.5,y=0.2
x'=x+1e-8*x*x*y
y'=-y+1e-8*x*y*y
done
";

#[test]
fn a_model_with_tiny_second_and_third_derivatives_passes_every_finite_difference_row() {
    for seed in 1..=30 {
        let (fb, mut ir) = setup(TINY_CURVATURE);
        let rep = check_model_with(&mut ir, &fb, &low_order(20, seed, 3));
        assert!(rep.passed(), "seed {seed}: {rep}");
        for row in ["A (FD)", "B (FD)", "C (FD)"] {
            let r = rep.find(row).unwrap();
            assert!(
                r.compared == 20 * if row == "A (FD)" { 4 } else { 2 },
                "{r}"
            );
        }
    }
}

#[test]
fn a_wrong_b_on_a_model_with_tiny_curvature_still_fails() {
    // B is off by 10 percent, which is 1e-9 of A v: far below any share of |A v|, so only a
    // bound that follows the size of B itself can see it.
    for seed in 1..=5 {
        let (fb, mut ir) = setup(TINY_CURVATURE);
        scale_entry(&mut ir, &EntryKind::Dir(vec![Slot::U, Slot::U]), 1.1);
        let rep = check_model_with(&mut ir, &fb, &low_order(20, seed, 1));
        let r = rep.find("B (FD)").unwrap();
        assert!(!r.pass && r.ratio > 1.0, "seed {seed}: {rep}");
    }
}

/// sin(40000 x): the step is `h = eps^(1/3)`, so `k h = 0.24` and the truncation error of the
/// Richardson value, `(k h)^4 / 480` of the derivative, is about 7 times `FD_TOL_FIRST`.
/// Only the correction term `|D(h) - D(h/2)| / 3` lets the exact derivative pass.
const STEEP: &str = "init x=0.5
x'=sin(40000*x)
done
";

#[test]
fn a_steep_model_passes_through_the_richardson_correction_term() {
    for seed in 1..=5 {
        let (fb, mut ir) = setup(STEEP);
        let rep = check_model_with(&mut ir, &fb, &low_order(10, seed, 1));
        let r = rep.find("A (FD)").unwrap();
        assert!(r.pass && r.compared == 10, "seed {seed}: {r}");
        // The exact derivative needs the term: the ratio is far above 1e-3 of the tolerance.
        assert!(r.ratio > 1e-3, "seed {seed}: {r}");
    }
}

#[test]
fn a_ten_percent_error_on_a_steep_model_still_fails() {
    for seed in 1..=5 {
        let (fb, mut ir) = setup(STEEP);
        scale_entry(&mut ir, &EntryKind::JacU, 1.1);
        let rep = check_model_with(&mut ir, &fb, &low_order(10, seed, 1));
        let r = rep.find("A (FD)").unwrap();
        assert!(!r.pass && r.ratio > 1.0, "seed {seed}: {r}");
    }
}
