//! Hopf points along a branch of delay equilibria (SPEC section 8e, tests 1 to 8).
//!
//! The counts below come from an argument-principle count of the characteristic function on
//! the contour Re = 1e-9, |Im| <= R, run in a script that shares no code with the engine.
//! Hutchinson, tau = 1: N = 0 at r = 1.2 to 1.4, N = 2 at pi/2 + 0.1 and 2.0, N = 4 at 8.5.
//! Hayes: N = 0 at tau = 0.909 and N = 2 at 1.509. Sigmoid, tau = 2: N = 0 at a = 6.91 and
//! N = 2 at 7.91. Fold: N = 0 at x = -0.1 and -0.2, N = 1 at x = -0.3, -0.4 and -0.5.
//! Pitchfork at x = 0: N = 0 at mu = 0.3 and 0.45, N = 1 at 0.55 and 0.8.

use pc_engine::{
    DdeBranch, DdeModel, EngineError, EpSetup, Evaluator, Options, RootOptions, StopReason,
    build_dde, dde_continue_equilibrium,
};
use pc_ode::{MapLoader, Model, ModelInfo, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const HUTCH: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";
const HAYES: &str = "par tau=1\nx'=-x-2*delay(x,tau)\ndone\n";
const SIGMOID: &str = "par a=7, tau=2\nx'=-x+a/(1+delay(x,tau)^2)\ndone\n";
const FOLD: &str = "par mu=0\nx'=mu-x*x-0.5*delay(x,1)\ndone\n";
const PITCH: &str = "par mu=0\nx'=mu*x-x*x*x-0.5*delay(x,1)\ndone\n";

fn loaded(src: &str) -> (Model, ModelInfo) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    (l.expanded, l.info)
}

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let (m, i) = loaded(src);
    match build_dde(&m, &i, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

fn both() -> [Evaluator; 2] {
    [Evaluator::Jit, Evaluator::Fallback]
}

/// One run: the free parameter is the first one and its start value is `params[0]`.
fn run(
    src: &str,
    which: Evaluator,
    params: &[f64],
    x0: f64,
    ro: RootOptions,
    opts: &Options,
) -> Result<DdeBranch, EngineError> {
    let mut m = dde(src, which);
    let setup = EpSetup::new(0, params.to_vec(), vec![x0]);
    dde_continue_equilibrium(&mut m, &setup, &ro, opts, &AtomicBool::new(false))
}

fn ok(
    src: &str,
    which: Evaluator,
    params: &[f64],
    x0: f64,
    ro: RootOptions,
    opts: &Options,
) -> DdeBranch {
    match run(src, which, params, x0, ro, opts) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn ro(n: usize) -> RootOptions {
    RootOptions { n, re_min: -0.5 }
}

fn opts(hmax: f64, bounds: (f64, f64), direction: f64) -> Options {
    Options {
        hmax,
        bounds,
        direction,
        ..Options::default()
    }
}

/// Every point with alpha below `star` has `below` roots in the right half-plane and every
/// point above has `above`.
fn counts_split(b: &DdeBranch, star: f64, below: usize, above: usize) {
    let (mut lo, mut hi) = (0, 0);
    for p in &b.points {
        if p.alpha < star {
            lo += 1;
            assert_eq!(p.unstable, below, "alpha = {}", p.alpha);
        } else {
            hi += 1;
            assert_eq!(p.unstable, above, "alpha = {}", p.alpha);
        }
    }
    assert!(lo > 0 && hi > 0, "branch covers one side only: {lo} {hi}");
}

fn labels(b: &DdeBranch) -> Vec<&'static str> {
    b.specials.iter().map(|s| s.label).collect()
}

#[test]
fn hutchinson_has_one_hopf_point_at_r_pi_over_two() {
    for which in both() {
        let o = opts(0.05, (1.2, 2.0), 1.0);
        let b = ok(HUTCH, which, &[1.3, 1.0], 1.0, ro(30), &o);
        assert_eq!(labels(&b), ["H"], "{:?}", b.specials);
        let h = &b.specials[0];
        assert!((h.alpha - PI / 2.0).abs() <= 1e-9, "alpha = {}", h.alpha);
        assert!(
            (h.omega.unwrap() - PI / 2.0).abs() <= 1e-9,
            "omega = {:?}",
            h.omega
        );
        assert!((h.x[0] - 1.0).abs() <= 1e-10, "x = {:?}", h.x);
        assert_eq!((h.unstable_before, h.unstable_after), (0, 2));
        assert_eq!(h.between[1], h.between[0] + 1);
        counts_split(&b, PI / 2.0, 0, 2);
        let last = b.points.last().unwrap();
        assert!((last.alpha - 2.0).abs() <= 1e-12, "last = {}", last.alpha);
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn hayes_loses_stability_at_the_delay_two_pi_over_three_root_three() {
    for which in both() {
        let o = opts(0.1, (0.9, 1.6), 1.0); // hmax 0.1 is the default
        let b = ok(HAYES, which, &[1.0], 0.0, ro(30), &o);
        assert_eq!(labels(&b), ["H"], "{:?}", b.specials);
        let star = 2.0 * PI / (3.0 * 3f64.sqrt());
        let h = &b.specials[0];
        assert!((h.alpha - star).abs() <= 1e-9, "tau = {}", h.alpha);
        assert!((h.omega.unwrap() - 3f64.sqrt()).abs() <= 1e-9);
        counts_split(&b, star, 0, 2);
        assert!(b.points.iter().all(|p| p.x[0].abs() < 1e-12));
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn the_sigmoid_model_has_its_hopf_point_at_the_known_gain() {
    for which in both() {
        let o = opts(0.1, (6.9, 7.9), 1.0);
        let b = ok(SIGMOID, which, &[7.0, 2.0], 1.7, ro(30), &o);
        assert_eq!(labels(&b), ["H"], "{:?}", b.specials);
        let h = &b.specials[0];
        assert!(
            (h.alpha - 7.409583945668119).abs() <= 1e-9,
            "a = {}",
            h.alpha
        );
        assert!((h.x[0] - 1.779031616618799).abs() <= 1e-9, "x = {:?}", h.x);
        assert!((h.omega.unwrap() - 1.144464864051702).abs() <= 1e-9);
        counts_split(&b, 7.409583945668119, 0, 2);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn a_fold_reports_the_limit_point_and_the_count_changes_there() {
    for which in both() {
        let o = Options {
            bounds: (-0.07, -0.03),
            direction: -1.0,
            ..Options::default()
        };
        let b = ok(FOLD, which, &[-0.04], -0.1, ro(30), &o);
        assert_eq!(labels(&b), ["LP"], "{:?}", b.specials);
        let s = &b.specials[0];
        assert!((s.alpha + 0.0625).abs() <= 1e-8, "mu = {}", s.alpha);
        assert!((s.x[0] + 0.25).abs() <= 1e-8, "x = {:?}", s.x);
        assert_eq!((s.unstable_before, s.unstable_after), (0, 1));
        assert!(s.omega.is_none());
        for p in &b.points {
            let want = if p.x[0] > -0.25 { 0 } else { 1 };
            assert_eq!(p.unstable, want, "x = {}", p.x[0]);
        }
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn a_pitchfork_reports_the_branch_point_on_the_trivial_branch() {
    for which in both() {
        let o = opts(0.1, (0.3, 0.8), 1.0);
        let b = ok(PITCH, which, &[0.35], 0.0, ro(30), &o);
        assert_eq!(labels(&b), ["BP"], "{:?}", b.specials);
        let s = &b.specials[0];
        assert!((s.alpha - 0.5).abs() <= 1e-8, "mu = {}", s.alpha);
        assert_eq!((s.unstable_before, s.unstable_after), (0, 1));
        counts_split(&b, 0.5, 0, 1);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn a_stable_range_gives_no_special_and_no_warning() {
    for which in both() {
        let o = opts(0.05, (0.5, 1.4), 1.0);
        let b = ok(HUTCH, which, &[1.0, 1.0], 1.0, ro(30), &o);
        assert!(b.specials.is_empty(), "{:?}", b.specials);
        assert!(b.points.iter().all(|p| p.unstable == 0));
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn a_real_crossing_with_no_lp_reported_gets_a_warning() {
    for which in both() {
        let o = Options {
            bounds: (-0.07, -0.03),
            direction: -1.0,
            detect_lp: false,
            ..Options::default()
        };
        let b = ok(FOLD, which, &[-0.04], -0.1, ro(30), &o);
        assert!(b.specials.is_empty(), "{:?}", b.specials);
        assert_eq!(b.points.first().unwrap().unstable, 0);
        assert_eq!(b.points.last().unwrap().unstable, 1);
        assert_eq!(b.warnings.len(), 1, "{:?}", b.warnings);
        assert!(
            b.warnings[0].contains("a real root crosses zero"),
            "{:?}",
            b.warnings
        );
    }
}

#[test]
fn one_step_over_two_crossings_finds_both() {
    for which in both() {
        let o = Options {
            h0: 8.0,
            hmax: 8.0,
            bounds: (1.2, 8.5),
            direction: 1.0,
            ..Options::default()
        };
        let b = ok(HUTCH, which, &[1.3, 1.0], 1.0, ro(40), &o);
        assert_eq!(b.points.len(), 2, "{:?}", b.points);
        assert!((b.points[1].alpha - 8.5).abs() <= 1e-12);
        assert_eq!(b.points[1].unstable, 4);
        assert_eq!(labels(&b), ["H", "H"], "{:?}", b.specials);
        let (a, c) = (&b.specials[0], &b.specials[1]);
        assert!((a.alpha - PI / 2.0).abs() <= 1e-8, "{}", a.alpha);
        assert!((c.alpha - 5.0 * PI / 2.0).abs() <= 1e-8, "{}", c.alpha);
        assert!((a.omega.unwrap() - PI / 2.0).abs() <= 1e-8);
        assert!((c.omega.unwrap() - 5.0 * PI / 2.0).abs() <= 1e-8);
        assert_eq!((a.unstable_before, a.unstable_after), (0, 2));
        assert_eq!((c.unstable_before, c.unstable_after), (2, 4));
        assert_eq!(a.between, [0, 1]);
        assert_eq!(c.between, [0, 1]);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    }
}

#[test]
fn a_non_negative_re_min_is_a_bad_problem() {
    for re_min in [0.0, 0.5] {
        let o = opts(0.05, (1.2, 2.0), 1.0);
        let r = RootOptions { n: 30, re_min };
        match run(HUTCH, Evaluator::Jit, &[1.3, 1.0], 1.0, r, &o) {
            Err(EngineError::BadProblem(m)) => assert!(m.contains("re_min"), "{m}"),
            other => panic!("expected BadProblem, got {other:?}"),
        }
    }
}

#[test]
fn a_free_delay_time_needs_a_positive_lower_bound() {
    // Hayes with tau as the free parameter and the default bounds (-inf, inf).
    let o = Options::default();
    match run(HAYES, Evaluator::Jit, &[1.0], 0.0, ro(30), &o) {
        Err(EngineError::BadProblem(m)) => assert!(m.contains("delay time"), "{m}"),
        other => panic!("expected BadProblem, got {other:?}"),
    }
}

#[test]
fn a_cancelled_run_returns_cancelled() {
    let mut m = dde(HUTCH, Evaluator::Jit);
    let setup = EpSetup::new(0, vec![1.3, 1.0], vec![1.0]);
    let o = opts(0.05, (1.2, 2.0), 1.0);
    let cancel = AtomicBool::new(true);
    let b = dde_continue_equilibrium(&mut m, &setup, &ro(30), &o, &cancel);
    match b {
        Ok(b) => assert!(matches!(b.stop, StopReason::Cancelled), "{:?}", b.stop),
        Err(EngineError::Cancelled) => {}
        Err(e) => panic!("unexpected error {e}"),
    }
}

/// With `t = alpha t'` Hutchinson's Hopf point is `r = alpha pi/2` with `omega = alpha pi/2`.
/// At `alpha = 1e-9` that frequency is below the absolute 1e-8 the classification used before
/// it followed the size of the matrices (SPEC section 8e), and the crossing is still an `H`.
#[test]
fn a_hopf_point_at_a_tiny_time_scale_is_still_an_h() {
    let a = 1e-9;
    for which in both() {
        let o = Options {
            h0: 0.01 * a,
            hmin: 1e-8 * a,
            hmax: 0.05 * a,
            bounds: (1.2 * a, 2.0 * a),
            direction: 1.0,
            tol_x: 1e-12 * a,
            tol_f: 1e-12 * a,
            tol_refine: 1e-12 * a,
            ..Options::default()
        };
        let b = ok(HUTCH, which, &[1.3 * a, 1.0 / a], 1.0, ro(30), &o);
        assert_eq!(labels(&b), ["H"], "{:?} {:?}", b.specials, b.warnings);
        let h = &b.specials[0];
        assert!(
            (h.alpha / a - PI / 2.0).abs() <= 1e-6,
            "alpha = {}",
            h.alpha
        );
        assert!(
            (h.omega.unwrap() / a - PI / 2.0).abs() <= 1e-6,
            "omega = {:?}",
            h.omega
        );
        // The root finder's absolute tolerances put candidates unconverged at this scale. That
        // is a warning about the count, so it is the only warning allowed.
        assert!(
            b.warnings.iter().all(|w| w.contains("did not converge")),
            "{:?}",
            b.warnings
        );
    }
}
