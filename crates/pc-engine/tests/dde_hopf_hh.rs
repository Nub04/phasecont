//! The HH label on a Hopf curve of delay equilibria (SPEC section 8k).
//!
//! The coupled model `x' = c y - a x(t-1)`, `y' = -d x - b y(t-1.5)` has the origin as its
//! equilibrium for every `(a, b)`. Its characteristic function is
//! `(l + a e^{-l})(l + b e^{-1.5 l}) + c d`. The numbers below were computed with numpy by
//! Newton's method on that determinant, at 16 digits (SPEC section 8k).
//!
//! With `c = d = 0` the two blocks decouple: the Hopf curve is `a = pi/2` and the second pair
//! solves `l + b e^{-1.5 l} = 0`, which is on the axis at `b = pi/3` with `omega2 = pi/3`.

use pc_engine::{
    CharRoot, DdeHopfBranch, DdeHopfExtras, DdeHopfSetup, DdeModel, EngineError, Evaluator,
    Options, RootOptions, RootSet, StopReason, build_dde, characteristic_roots, dde_continue_hopf,
    dde_continue_hopf_with, dde_hopf_pairs, dde_linearize,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const COUPLED: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)\n\
    y'=-d*x-b*delay(y,1.5)\n\
    done\n";
const DECOUPLED: &str = "par a=1.4, b=1.2, c=0, d=0\n\
    x'=c*y-a*delay(x,1)\n\
    y'=-d*x-b*delay(y,1.5)\n\
    done\n";
// The same linearization as COUPLED, so the Hopf curve and the HH point are the same, with
// quadratic terms so that l1 is not zero by construction.
const COUPLED_NL: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)+x*x+x*y\n\
    y'=-d*x-b*delay(y,1.5)\n\
    done\n";
const BT: &str = "par a=-2, tau=1, b=1, c=-0.25\nx'=c-a*x-b*delay(x,tau)-x*x\ndone\n";
const ZH: &str = "par a=0, tau=1, b=1, d=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y+d*x*x\ndone\n";

const B_STAR: f64 = 1.184_419_463_185_973_2;
const A_STAR: f64 = 1.454_526_952_022_688;
const OMEGA1_STAR: f64 = 1.516_027_941_313_437_7;
const OMEGA2_STAR: f64 = 1.130_821_068_412_625_6;

/// `(b, a, omega1, Re l2, Im l2)` on the Hopf curve of the coupled model.
const TABLE: [(f64, f64, f64, f64, f64); 7] = [
    (
        0.5,
        1.478_574_856_082_080_8,
        1.555_678_137_532_598_8,
        -0.391_651_422_073_654_44,
        0.795_843_710_611_601_3,
    ),
    (
        0.8,
        1.460_779_746_665_151_3,
        1.539_791_694_360_651_3,
        -0.182_394_645_982_381_53,
        0.990_817_463_171_413_5,
    ),
    (
        1.0,
        1.453_487_170_712_646_7,
        1.526_966_524_941_607_6,
        -0.080_891_234_631_346_08,
        1.071_617_858_367_675_4,
    ),
    (
        1.1,
        1.452_883_435_032_108,
        1.520_707_184_400_439_6,
        -0.035_994_324_821_699_71,
        1.105_192_730_266_564,
    ),
    (
        1.19,
        1.454_710_660_283_154_5,
        1.515_747_995_473_345_3,
        0.002_330_686_837_183_361_5,
        1.132_432_280_917_528_5,
    ),
    (
        1.25,
        1.457_278_714_504_184_3,
        1.513_025_660_156_989,
        0.027_062_503_671_456_562,
        1.149_113_138_987_745_1,
    ),
    (
        1.4,
        1.467_973_480_442_997_1,
        1.508_960_397_573_654_3,
        0.086_541_847_359_830_78,
        1.185_660_081_667_308_4,
    ),
];

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    match build_dde(&l.expanded, &l.info, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

fn both() -> [Evaluator; 2] {
    [Evaluator::Jit, Evaluator::Fallback]
}

fn opts(hmax: f64, bounds: (f64, f64), direction: f64) -> Options {
    Options {
        hmax,
        bounds,
        direction,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        ..Options::default()
    }
}

fn setup(free: [usize; 2], params: &[f64], x0: &[f64], omega: f64) -> DdeHopfSetup {
    DdeHopfSetup {
        free,
        params: params.to_vec(),
        x0: x0.to_vec(),
        omega,
    }
}

fn ro() -> RootOptions {
    RootOptions {
        n: 30,
        re_min: -3.0,
    }
}

fn extras(lyapunov: bool) -> DdeHopfExtras {
    DdeHopfExtras {
        lyapunov,
        hh: Some(ro()),
    }
}

fn coupled_setup(b: f64, omega: f64) -> DdeHopfSetup {
    setup([0, 1], &[1.46, b, 1.0, 0.1], &[0.001, 0.001], omega)
}

fn decoupled_setup() -> DdeHopfSetup {
    setup([0, 1], &[1.57, 0.8, 0.0, 0.0], &[0.0, 0.0], 1.55)
}

fn run_with(
    src: &str,
    which: Evaluator,
    s: &DdeHopfSetup,
    o: &Options,
    e: &DdeHopfExtras,
) -> DdeHopfBranch {
    let mut m = dde(src, which);
    match dde_continue_hopf_with(&mut m, s, o, &AtomicBool::new(false), e) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn run_plain(src: &str, which: Evaluator, s: &DdeHopfSetup, o: &Options) -> DdeHopfBranch {
    let mut m = dde(src, which);
    match dde_continue_hopf(&mut m, s, o, &AtomicBool::new(false)) {
        Ok(b) => b,
        Err(e) => panic!("run failed: {e}"),
    }
}

fn err_with(src: &str, s: &DdeHopfSetup, o: &Options, e: &DdeHopfExtras) -> EngineError {
    let mut m = dde(src, Evaluator::Fallback);
    match dde_continue_hopf_with(&mut m, s, o, &AtomicBool::new(false), e) {
        Ok(_) => panic!("the run was expected to fail"),
        Err(e) => e,
    }
}

fn labels(b: &DdeHopfBranch) -> Vec<&'static str> {
    b.specials.iter().map(|s| s.label).collect()
}

fn root_set(roots: &[(f64, f64)]) -> RootSet {
    let mut all = Vec::new();
    for &(re, im) in roots {
        all.push(CharRoot {
            re,
            im,
            residual: 0.0,
        });
        if im != 0.0 {
            all.push(CharRoot {
                re,
                im: -im,
                residual: 0.0,
            });
        }
    }
    RootSet {
        n: 30,
        roots: all,
        unrefined: Vec::new(),
    }
}

fn is_bad_problem(e: &EngineError) -> bool {
    matches!(e, EngineError::BadProblem(_))
}

fn forward_opts() -> Options {
    opts(0.05, (0.5, 1.4), 1.0)
}

// Test 1.
#[test]
fn hopf_pairs_on_hand_made_root_sets() {
    let w = 1.5;
    let tol = 1e-6 * (1.0 + w);
    // One critical root, others around it. A real root with a larger real part and a root
    // with Im < 0 (its conjugate) are not taken.
    let set = root_set(&[(0.0, w), (-0.5, 0.9), (-0.2, 3.0), (0.3, 0.0), (-1.0, 2.0)]);
    let p = dde_hopf_pairs(&set, w);
    assert_eq!(p.critical, 1);
    let s = p.second.expect("a second root");
    assert_eq!((s.re, s.im), (-0.2, 3.0));
    // Inside and outside the distance.
    let inside = root_set(&[(0.0, w + 0.5 * tol), (-0.4, 1.0)]);
    assert_eq!(dde_hopf_pairs(&inside, w).critical, 1);
    let outside = root_set(&[(0.0, w + 2.0 * tol), (-0.4, 1.0)]);
    let p = dde_hopf_pairs(&outside, w);
    assert_eq!(p.critical, 0);
    // The root off the axis by 2 tol is a second root, and it is the largest.
    assert_eq!(p.second.expect("second").im, w + 2.0 * tol);
    // Two roots within the distance.
    let two = root_set(&[(0.0, w), (0.0, w + 0.1 * tol), (-0.4, 1.0)]);
    assert_eq!(dde_hopf_pairs(&two, w).critical, 2);
    // Empty and real-only sets.
    for set in [root_set(&[]), root_set(&[(-1.0, 0.0), (0.5, 0.0)])] {
        let p = dde_hopf_pairs(&set, w);
        assert_eq!(p.critical, 0);
        assert!(p.second.is_none());
    }
    // The critical root only.
    let only = root_set(&[(0.0, w)]);
    let p = dde_hopf_pairs(&only, w);
    assert_eq!(p.critical, 1);
    assert!(p.second.is_none());
    // Equal real parts: the larger imaginary part.
    let tie = root_set(&[(0.0, w), (-0.3, 1.0), (-0.3, 2.0)]);
    let s = dde_hopf_pairs(&tie, w).second.expect("second");
    assert_eq!((s.re, s.im), (-0.3, 2.0));
}

// Test 2.
#[test]
fn the_roots_at_the_oracle_points_match_the_table() {
    for which in both() {
        let mut m = dde(COUPLED, which);
        for (b, a, w1, re2, im2) in TABLE {
            let lin = dde_linearize(&mut m, &[0.0, 0.0], &[a, b, 1.0, 0.1]).unwrap();
            let set = characteristic_roots(&lin, &ro()).unwrap();
            let p = dde_hopf_pairs(&set, w1);
            assert_eq!(p.critical, 1, "b = {b}");
            let crit = set
                .roots
                .iter()
                .filter(|r| r.im > 0.0)
                .map(|r| ((r.re).hypot(r.im - w1), r))
                .fold((f64::INFINITY, None), |acc, (d, r)| {
                    if d < acc.0 { (d, Some(r)) } else { acc }
                });
            assert!(crit.0 < 1e-9, "b = {b}: critical root off by {}", crit.0);
            let s = p.second.expect("second root");
            assert!((s.re - re2).abs() < 1e-9, "b = {b}: re {} vs {re2}", s.re);
            assert!((s.im - im2).abs() < 1e-9, "b = {b}: im {} vs {im2}", s.im);
        }
    }
    // Decoupled: the second pair of the y block.
    let table = [
        (0.5, -0.344_970_363_686_372_6, 0.764_662_957_019_079),
        (1.0, -0.021_855_823_943_715_07, 1.033_095_882_233_439_4),
        (1.2, 0.064_809_958_365_866_62, 1.086_902_617_617_715),
    ];
    let mut m = dde(DECOUPLED, Evaluator::Jit);
    for (b, re2, im2) in table {
        let lin = dde_linearize(&mut m, &[0.0, 0.0], &[PI / 2.0, b, 0.0, 0.0]).unwrap();
        let set = characteristic_roots(&lin, &ro()).unwrap();
        let p = dde_hopf_pairs(&set, PI / 2.0);
        assert_eq!(p.critical, 1, "decoupled b = {b}");
        let s = p.second.expect("second root");
        assert!((s.re - re2).abs() < 1e-9, "b = {b}: re {} vs {re2}", s.re);
        assert!((s.im - im2).abs() < 1e-9, "b = {b}: im {} vs {im2}", s.im);
    }
}

// Test 3.
#[test]
fn the_coupled_curve_has_one_hh_point_at_the_oracle_values() {
    for which in both() {
        let b = run_with(
            COUPLED,
            which.clone(),
            &coupled_setup(0.8, 1.54),
            &forward_opts(),
            &extras(false),
        );
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        assert_eq!(labels(&b), ["HH"], "{:?}", b.specials);
        assert!(
            (b.points[0].alpha[0] - TABLE[1].1).abs() < 1e-9,
            "start a = {}",
            b.points[0].alpha[0]
        );
        let sp = &b.specials[0];
        assert!((sp.alpha[1] - B_STAR).abs() < 1e-6, "b = {}", sp.alpha[1]);
        assert!((sp.alpha[0] - A_STAR).abs() < 1e-6, "a = {}", sp.alpha[0]);
        assert!(
            (sp.kappa - OMEGA1_STAR * OMEGA1_STAR).abs() < 1e-6,
            "kappa = {}",
            sp.kappa
        );
        let w2 = sp.omega2.expect("omega2 at an HH special");
        assert!((w2 - OMEGA2_STAR).abs() < 1e-6, "omega2 = {w2}");
        assert!(sp.x.iter().all(|v| v.abs() <= 1e-8), "x = {:?}", sp.x);
        // The second pair is on the axis at the special.
        let mut m = dde(COUPLED, which);
        let lin = dde_linearize(&mut m, &sp.x, &[sp.alpha[0], sp.alpha[1], 1.0, 0.1]).unwrap();
        let set = characteristic_roots(&lin, &ro()).unwrap();
        let s = dde_hopf_pairs(&set, sp.kappa.sqrt()).second.unwrap();
        assert!(s.re.abs() < 1e-6, "Re l2 at the special = {}", s.re);
        // Every point has hh, negative before the special and positive from it on.
        for (i, p) in b.points.iter().enumerate() {
            let h = p.hh.unwrap_or_else(|| panic!("point {i} has no hh"));
            if i < sp.index {
                assert!(h < 0.0, "point {i}: hh = {h}");
            } else {
                assert!(h > 0.0, "point {i}: hh = {h}");
            }
        }
    }
}

// Test 4.
#[test]
fn the_other_direction_has_no_hh_and_negative_h_everywhere() {
    let o = opts(0.05, (0.5, 1.4), -1.0);
    let b = run_with(
        COUPLED,
        Evaluator::Jit,
        &coupled_setup(0.8, 1.54),
        &o,
        &extras(false),
    );
    assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
    assert!(b.specials.is_empty(), "{:?}", b.specials);
    let last = b.points.last().unwrap();
    assert!((last.alpha[1] - 0.5).abs() < 1e-8, "b = {}", last.alpha[1]);
    assert!(
        (last.alpha[0] - TABLE[0].1).abs() < 1e-8,
        "a = {}",
        last.alpha[0]
    );
    for p in &b.points {
        assert!(p.hh.expect("hh") < 0.0, "hh = {:?}", p.hh);
    }
}

// Test 5.
#[test]
fn the_decoupled_curve_has_its_hh_point_at_b_pi_over_three() {
    for which in both() {
        let b = run_with(
            DECOUPLED,
            which,
            &decoupled_setup(),
            &opts(0.05, (0.5, 1.2), 1.0),
            &extras(false),
        );
        assert!(matches!(b.stop, StopReason::Bounds), "{:?}", b.stop);
        for p in &b.points {
            assert!((p.alpha[0] - PI / 2.0).abs() < 1e-8, "a = {}", p.alpha[0]);
            assert!(
                (p.kappa - PI * PI / 4.0).abs() < 1e-8,
                "kappa = {}",
                p.kappa
            );
        }
        assert_eq!(labels(&b), ["HH"], "{:?}", b.specials);
        let sp = &b.specials[0];
        assert!((sp.alpha[1] - PI / 3.0).abs() < 1e-6, "b = {}", sp.alpha[1]);
        assert!((sp.alpha[0] - PI / 2.0).abs() < 1e-8, "a = {}", sp.alpha[0]);
        let w2 = sp.omega2.expect("omega2");
        assert!((w2 - PI / 3.0).abs() < 1e-6, "omega2 = {w2}");
    }
}

// Test 6.
#[test]
fn a_good_degree_gives_no_hh_warning() {
    let runs = [
        (COUPLED, coupled_setup(0.8, 1.54), forward_opts()),
        (
            COUPLED,
            coupled_setup(0.8, 1.54),
            opts(0.05, (0.5, 1.4), -1.0),
        ),
        (DECOUPLED, decoupled_setup(), opts(0.05, (0.5, 1.2), 1.0)),
    ];
    for (src, s, o) in runs {
        let b = run_with(src, Evaluator::Jit, &s, &o, &extras(false));
        let hh: Vec<_> = b.warnings.iter().filter(|w| w.contains("hh")).collect();
        assert!(hh.is_empty(), "{hh:?}");
    }
}

// Test 7.
#[test]
fn bt_and_zh_runs_are_unchanged_and_have_no_hh() {
    let cases = [
        (
            BT,
            setup(
                [0, 1],
                &[-2.24, 1.361_344_425_034_588_3, 1.0, -0.25],
                &[1.001],
                0.97,
            ),
            opts(0.05, (0.8, 1.5), -1.0),
            "BT",
        ),
        (
            ZH,
            setup(
                [0, 1],
                &[0.07, 1.407_136_057_451_117_4, 1.0, 1.0],
                &[0.001, -0.249],
                0.98,
            ),
            opts(0.05, (1.3, 2.0), 1.0),
            "ZH",
        ),
    ];
    for (src, s, o, want) in cases {
        for which in both() {
            let on = run_with(src, which.clone(), &s, &o, &extras(false));
            let off = run_plain(src, which, &s, &o);
            assert_eq!(labels(&on), [want], "{:?}", on.specials);
            assert_eq!(labels(&off), [want]);
            assert_eq!(on.points.len(), off.points.len());
            for (p, q) in on.points.iter().zip(&off.points) {
                assert_eq!((&p.alpha, &p.x, p.kappa), (&q.alpha, &q.x, q.kappa));
                if p.kappa <= 0.0 {
                    assert!(p.hh.is_none(), "{want}: hh at kappa = {}", p.kappa);
                }
            }
            assert_eq!(on.specials.len(), off.specials.len());
            for (p, q) in on.specials.iter().zip(&off.specials) {
                assert_eq!(
                    (&p.alpha, &p.x, p.kappa, p.index),
                    (&q.alpha, &q.x, q.kappa, q.index)
                );
                assert!(p.omega2.is_none());
            }
            assert_eq!(on.warnings, off.warnings);
        }
    }
}

// Test 8.
#[test]
fn off_equals_absent() {
    let (s, o) = (coupled_setup(0.8, 1.54), forward_opts());
    let plain = run_plain(COUPLED, Evaluator::Jit, &s, &o);
    assert!(plain.specials.is_empty(), "{:?}", plain.specials);
    assert!(plain.points.iter().all(|p| p.hh.is_none()));
    let off = DdeHopfExtras {
        lyapunov: false,
        hh: None,
    };
    let none = run_with(COUPLED, Evaluator::Jit, &s, &o, &off);
    let on = run_with(COUPLED, Evaluator::Jit, &s, &o, &extras(false));
    assert_eq!(none.points.len(), plain.points.len());
    assert_eq!(on.points.len(), plain.points.len());
    for ((p, q), r) in plain.points.iter().zip(&none.points).zip(&on.points) {
        assert_eq!((&p.alpha, &p.x, p.kappa), (&q.alpha, &q.x, q.kappa));
        assert_eq!((&p.alpha, &p.x, p.kappa), (&r.alpha, &r.x, r.kappa));
        assert!(q.hh.is_none());
    }
    assert_eq!(none.warnings, plain.warnings);
    assert!(none.specials.is_empty());
}

// Test 9.
#[test]
fn hh_with_lyapunov_finds_the_same_point() {
    let (s, o) = (coupled_setup(0.8, 1.54), forward_opts());
    let base = run_with(COUPLED, Evaluator::Jit, &s, &o, &extras(false));
    let b = run_with(COUPLED_NL, Evaluator::Jit, &s, &o, &extras(true));
    let hh: Vec<_> = b.specials.iter().filter(|s| s.label == "HH").collect();
    assert_eq!(hh.len(), 1, "{:?}", b.specials);
    assert!(
        b.specials
            .iter()
            .all(|s| s.label == "HH" || s.label == "GH"),
        "{:?}",
        b.specials
    );
    assert!(
        (hh[0].alpha[1] - base.specials[0].alpha[1]).abs() < 1e-8,
        "{} vs {}",
        hh[0].alpha[1],
        base.specials[0].alpha[1]
    );
    let w2 = hh[0].omega2.expect("omega2");
    assert!((w2 - OMEGA2_STAR).abs() < 1e-6, "omega2 = {w2}");
    for p in &b.points {
        assert!(p.hh.is_some(), "hh missing");
        assert!(p.l1.is_some(), "l1 missing at kappa = {}", p.kappa);
    }
}

// Test 10.
#[test]
fn option_errors_are_bad_problems() {
    let (s, o) = (coupled_setup(0.8, 1.54), forward_opts());
    let with = |n: usize, re_min: f64| DdeHopfExtras {
        lyapunov: false,
        hh: Some(RootOptions { n, re_min }),
    };
    for re_min in [0.0, 0.5, f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
        let e = err_with(COUPLED, &s, &o, &with(30, re_min));
        assert!(is_bad_problem(&e), "re_min {re_min}: {e:?}");
        assert!(e.to_string().contains("re_min"), "re_min {re_min}: {e}");
    }
    for n in [0, 2000] {
        let e = err_with(COUPLED, &s, &o, &with(n, -3.0));
        assert!(is_bad_problem(&e), "n {n}: {e:?}");
    }
    // A degree that leaves no root at i omega at the start.
    let e = err_with(COUPLED, &s, &o, &with(1, -3.0));
    assert!(is_bad_problem(&e), "n 1: {e:?}");
    assert!(e.to_string().contains("hh"), "n 1: {e}");
}
