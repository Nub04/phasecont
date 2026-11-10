//! The coefficients at the ZH and HH specials of a delay Hopf curve (SPEC section 8s, slice 3).
//!
//! The numbers come from `zero_hopf` and `hopf_hopf` (SPEC sections 8i and 8j), so the tests
//! compare the values of a run with a direct call of those two functions on the same special,
//! bit for bit, and then check the published points: the ZH model of section 8f test 5 at
//! `(a, tau) = (0, pi/2)` (section 8i test 7) and the coupled model of section 8k.

use pc_engine::{
    DdeHopfBranch, DdeHopfExtras, DdeHopfSetup, DdeHopfSpecial, DdeModel, DdeZeroHopf, EngineError,
    Evaluator, Options, RootOptions, build_dde, dde_coefficient_warning, dde_continue_hopf_with,
    dde_hopf_hopf, dde_special_coefficients, dde_zero_hopf,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const ZH: &str = "par a=0, tau=1, b=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";
const BT: &str = "par a=-2, tau=1, b=1, c=-0.25\nx'=c-a*x-b*delay(x,tau)-x*x\ndone\n";
// The coupled model of section 8k with quadratic and cubic terms, so that the coefficients are
// not all zero.
const COUPLED_NL: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)+x*x+x*y-x*x*x\n\
    y'=-d*x-b*delay(y,1.5)+y*y*y+x*y*y\n\
    done\n";

const A_STAR: f64 = 1.454_526_952_022_688;
const B_STAR: f64 = 1.184_419_463_185_973_2;
const OMEGA1_STAR: f64 = 1.516_027_941_313_437_7;
const OMEGA2_STAR: f64 = 1.130_821_068_412_625_6;

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

// Section 8f test 5.
fn zh_setup() -> DdeHopfSetup {
    setup(
        [0, 1],
        &[0.07, 1.4071360574511174, 1.0],
        &[0.001, -0.249],
        0.98,
    )
}

fn zh_opts() -> Options {
    opts(0.05, (1.3, 2.0), 1.0)
}

// Section 8k, slice 2 test 3.
fn coupled_setup() -> DdeHopfSetup {
    setup([0, 1], &[1.46, 0.8, 1.0, 0.1], &[0.001, 0.001], 1.54)
}

fn coupled_opts() -> Options {
    opts(0.05, (0.5, 1.4), 1.0)
}

fn ro() -> RootOptions {
    RootOptions {
        n: 30,
        re_min: -3.0,
    }
}

fn extras(coefficients: bool, hh: bool) -> DdeHopfExtras {
    DdeHopfExtras {
        lyapunov: false,
        hh: hh.then(ro),
        coefficients,
    }
}

fn run(
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

/// The declared parameters of a special: the free values set over the setup's parameters.
fn declared(s: &DdeHopfSetup, sp: &DdeHopfSpecial) -> Vec<f64> {
    let mut p = s.params.clone();
    p[s.free[0]] = sp.alpha[0];
    p[s.free[1]] = sp.alpha[1];
    p
}

fn value(sp: &DdeHopfSpecial, name: &str) -> f64 {
    sp.coefficients
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no value {name} in {:?}", sp.coefficients))
        .1
}

fn names(sp: &DdeHopfSpecial) -> Vec<&'static str> {
    sp.coefficients.iter().map(|(n, _)| *n).collect()
}

fn pair(out: &mut Vec<(&'static str, f64)>, re: &'static str, im: &'static str, z: [f64; 2]) {
    out.push((re, z[0]));
    out.push((im, z[1]));
}

/// The values of SPEC slice 3 for a ZH special, written out from the result of `zero_hopf`.
fn zh_expected(z: &DdeZeroHopf) -> Vec<(&'static str, f64)> {
    let mut v = Vec::new();
    pair(&mut v, "g200_re", "g200_im", z.g200);
    pair(&mut v, "g110_re", "g110_im", z.g110);
    pair(&mut v, "g011_re", "g011_im", z.g011);
    pair(&mut v, "g300_re", "g300_im", z.g300);
    pair(&mut v, "g111_re", "g111_im", z.g111);
    pair(&mut v, "g210_re", "g210_im", z.g210);
    pair(&mut v, "g021_re", "g021_im", z.g021);
    v.push(("s_product", z.s_product));
    v.push(("s", f64::from(z.s)));
    if let Some(t) = z.theta {
        v.push(("theta", t));
    }
    if let Some(e) = z.e {
        v.push(("e", e));
    }
    v.push(("null_residual_zero", z.null_residual_zero));
    v.push(("null_residual_hopf", z.null_residual_hopf));
    v.push(("equilibrium_residual", z.equilibrium_residual));
    v
}

fn same_bits(a: &[(&'static str, f64)], b: &[(&'static str, f64)]) {
    assert_eq!(a.len(), b.len(), "{a:?} against {b:?}");
    for ((na, va), (nb, vb)) in a.iter().zip(b) {
        assert_eq!(na, nb);
        assert_eq!(va.to_bits(), vb.to_bits(), "{na}: {va} against {vb}");
    }
}

fn same_curve(a: &DdeHopfBranch, b: &DdeHopfBranch) {
    same_points_and_specials(a, b);
    assert_eq!(a.warnings, b.warnings);
    assert_eq!(a.stop, b.stop);
}

/// `same_curve` without the warnings and the stop reason.
fn same_points_and_specials(a: &DdeHopfBranch, b: &DdeHopfBranch) {
    assert_eq!(a.points.len(), b.points.len());
    for (p, q) in a.points.iter().zip(&b.points) {
        for (u, v) in p.alpha.iter().zip(&q.alpha) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
        for (u, v) in p.x.iter().zip(&q.x) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
        assert_eq!(p.kappa.to_bits(), q.kappa.to_bits());
    }
    assert_eq!(a.specials.len(), b.specials.len());
    for (s, t) in a.specials.iter().zip(&b.specials) {
        assert_eq!((s.label, s.index), (t.label, t.index));
        for (u, v) in s.alpha.iter().zip(&t.alpha) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
        for (u, v) in s.x.iter().zip(&t.x) {
            assert_eq!(u.to_bits(), v.to_bits());
        }
        assert_eq!(s.kappa.to_bits(), t.kappa.to_bits());
        assert_eq!(s.omega2.map(f64::to_bits), t.omega2.map(f64::to_bits));
    }
}

// Test 1.
#[test]
fn zh_values_equal_the_direct_call_and_the_closed_form() {
    for which in both() {
        let s = zh_setup();
        let b = run(ZH, which.clone(), &s, &zh_opts(), &extras(true, false));
        assert_eq!(b.specials.len(), 1, "{:?}", b.specials);
        let sp = &b.specials[0];
        assert_eq!(sp.label, "ZH");
        let mut m = dde(ZH, which.clone());
        let z = dde_zero_hopf(&mut m, &sp.x, &declared(&s, sp), sp.kappa.sqrt())
            .expect("zero_hopf at the special");
        let want = zh_expected(&z);
        same_bits(&sp.coefficients, &want);
        // The names and their order, theta present and e absent at this point.
        assert_eq!(names(sp).first(), Some(&"g200_re"));
        assert!(names(sp).contains(&"theta"));
        assert!(!names(sp).contains(&"e"), "{:?}", names(sp));
        // Section 8i test 7, at the located point.
        assert!((value(sp, "g200_re") + 1.0).abs() < 1e-6);
        assert!(value(sp, "g200_im").abs() < 1e-6);
        assert!((value(sp, "g110_re") + 0.288_400_439_142_001).abs() < 1e-6);
        assert!((value(sp, "g110_im") - 0.453_018_350_450_290_2).abs() < 1e-6);
        assert!(value(sp, "g011_re").abs() < 1e-6);
        assert!(value(sp, "s_product").abs() < 1e-6);
        assert!((value(sp, "theta") - 0.288_400_439_142_001).abs() < 1e-6);
        // The same run without the option: the curve and the specials do not change.
        let off = run(ZH, which, &s, &zh_opts(), &extras(false, false));
        same_curve(&b, &off);
        assert!(off.specials.iter().all(|sp| sp.coefficients.is_empty()));
        assert!(b.warnings.iter().all(|w| !w.starts_with("coefficients")));
    }
}

// Test 2.
#[test]
fn hh_values_equal_the_direct_call_and_the_oracle() {
    for which in both() {
        let s = coupled_setup();
        let b = run(
            COUPLED_NL,
            which.clone(),
            &s,
            &coupled_opts(),
            &extras(true, true),
        );
        assert_eq!(b.specials.len(), 1, "{:?}", b.specials);
        let sp = &b.specials[0];
        assert_eq!(sp.label, "HH");
        assert!((sp.alpha[0] - A_STAR).abs() < 1e-6);
        assert!((sp.alpha[1] - B_STAR).abs() < 1e-6);
        let w1 = sp.kappa.sqrt();
        let w2 = sp.omega2.expect("omega2 at the special");
        assert!((w1 - OMEGA1_STAR).abs() < 1e-6);
        assert!((w2 - OMEGA2_STAR).abs() < 1e-6);
        let mut m = dde(COUPLED_NL, which.clone());
        let h = dde_hopf_hopf(&mut m, &sp.x, &declared(&s, sp), w1.max(w2), w1.min(w2))
            .expect("hopf_hopf at the special");
        let mut want = vec![("omega_large", h.omega1), ("omega_small", h.omega2)];
        pair(&mut want, "g2100_re", "g2100_im", h.g2100);
        pair(&mut want, "g1011_re", "g1011_im", h.g1011);
        pair(&mut want, "g1110_re", "g1110_im", h.g1110);
        pair(&mut want, "g0021_re", "g0021_im", h.g0021);
        if let Some(t) = h.theta {
            want.push(("theta", t));
        }
        if let Some(d) = h.delta {
            want.push(("delta", d));
        }
        want.push(("product", h.product));
        want.push(("sign", f64::from(h.sign)));
        want.push(("null_residual_1", h.null_residual_1));
        want.push(("null_residual_2", h.null_residual_2));
        want.push(("equilibrium_residual", h.equilibrium_residual));
        same_bits(&sp.coefficients, &want);
        assert!((value(sp, "omega_large") - OMEGA1_STAR).abs() < 1e-6);
        assert!((value(sp, "omega_small") - OMEGA2_STAR).abs() < 1e-6);
        // The same run without the option.
        let off = run(COUPLED_NL, which, &s, &coupled_opts(), &extras(false, true));
        same_curve(&b, &off);
        assert!(off.specials[0].coefficients.is_empty());
    }
}

// Test 2, the second half: without `roots` there is no HH special and no HH values.
#[test]
fn without_roots_there_is_no_hh_special() {
    let b = run(
        COUPLED_NL,
        Evaluator::Fallback,
        &coupled_setup(),
        &coupled_opts(),
        &extras(true, false),
    );
    assert!(b.specials.is_empty(), "{:?}", b.specials);
}

// BT has no coefficients (omega -> 0 there).
#[test]
fn bt_special_gets_no_coefficients() {
    let s = setup(
        [0, 1],
        &[-2.24, 1.3613444250345883, 1.0, -0.25],
        &[1.001],
        0.97,
    );
    let o = opts(0.05, (0.8, 1.5), -1.0);
    let b = run(BT, Evaluator::Fallback, &s, &o, &extras(true, false));
    assert_eq!(b.specials.len(), 1, "{:?}", b.specials);
    assert_eq!(b.specials[0].label, "BT");
    assert!(b.specials[0].coefficients.is_empty());
    assert!(b.warnings.iter().all(|w| !w.starts_with("coefficients")));
}

fn hand_made(label: &'static str, x: Vec<f64>, kappa: f64, omega2: Option<f64>) -> DdeHopfSpecial {
    DdeHopfSpecial {
        label,
        alpha: [0.0, PI / 2.0],
        x,
        kappa,
        index: 3,
        l2: None,
        omega2,
        coefficients: Vec::new(),
    }
}

// Test 3.
#[test]
fn the_larger_frequency_comes_first() {
    let mut m = dde(COUPLED_NL, Evaluator::Fallback);
    let p = vec![A_STAR, B_STAR, 1.0, 0.1];
    let x = vec![0.0, 0.0];
    let a = hand_made(
        "HH",
        x.clone(),
        OMEGA1_STAR * OMEGA1_STAR,
        Some(OMEGA2_STAR),
    );
    let b = hand_made(
        "HH",
        x.clone(),
        OMEGA2_STAR * OMEGA2_STAR,
        Some(OMEGA1_STAR),
    );
    let va = dde_special_coefficients(&mut m, &a, &p).expect("sqrt(kappa) larger");
    let vb = dde_special_coefficients(&mut m, &b, &p).expect("sqrt(kappa) smaller");
    for v in [&va, &vb] {
        let get = |n: &str| v.iter().find(|(m, _)| *m == n).expect("value").1;
        assert!((get("omega_large") - OMEGA1_STAR).abs() < 1e-12);
        assert!((get("omega_small") - OMEGA2_STAR).abs() < 1e-12);
    }
    // The two orders give the same values, down to rounding of the square root and its square.
    assert_eq!(va.len(), vb.len());
    for ((n, u), (_, w)) in va.iter().zip(&vb) {
        assert!((u - w).abs() <= 1e-9 * (1.0 + u.abs()), "{n}: {u} {w}");
    }
    // Equal frequencies: hopf_hopf refuses, and the error names the order.
    let c = hand_made("HH", x, OMEGA1_STAR * OMEGA1_STAR, Some(OMEGA1_STAR));
    match dde_special_coefficients(&mut m, &c, &p) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.contains("must be greater than omega2"), "{msg}");
        }
        other => panic!("expected BadProblem, got {other:?}"),
    }
    // No second frequency.
    let d = hand_made("HH", vec![0.0, 0.0], OMEGA1_STAR * OMEGA1_STAR, None);
    match dde_special_coefficients(&mut m, &d, &p) {
        Err(EngineError::BadProblem(msg)) => {
            assert_eq!(msg, "omega2 of the special is not available");
        }
        other => panic!("expected BadProblem, got {other:?}"),
    }
}

// Test 4.
#[test]
fn a_refused_special_gets_the_warning_text() {
    let mut m = dde(ZH, Evaluator::Fallback);
    // y = 0.5: Delta(0) is regular, so there is no zero root (the equilibrium residual is
    // not checked).
    let sp = hand_made("ZH", vec![0.0, 0.5], 1.0, None);
    let p = vec![0.0, PI / 2.0, 1.0];
    let err = match dde_special_coefficients(&mut m, &sp, &p) {
        Err(e) => e,
        Ok(v) => panic!("expected a refusal, got {v:?}"),
    };
    assert!(matches!(err, EngineError::BadProblem(_)), "{err:?}");
    assert!(err.to_string().contains("Delta(0)"), "{err}");
    let w = dde_coefficient_warning(&sp, &err);
    assert_eq!(
        w,
        format!("coefficients: the ZH special at index 3 has none: {err}")
    );
}

// The labels GH and BT are not asked for coefficients.
#[test]
fn only_zh_and_hh_are_computed() {
    let mut m = dde(ZH, Evaluator::Fallback);
    let p = vec![0.0, PI / 2.0, 1.0];
    for label in ["BT", "GH"] {
        let sp = hand_made(label, vec![0.0, 0.0], 1.0, None);
        assert!(
            dde_special_coefficients(&mut m, &sp, &p).is_err(),
            "{label}"
        );
    }
}

// The ZH model with a decoupled oscillator u' = w v, v' = -w u, whose roots are +-i w for every
// parameter value. At the ZH point the Hopf frequency is b = 1, so w = 2 puts a root at
// 2 i omega: Delta(2 i omega) is singular and `zero_hopf` refuses there. The curve itself does
// not notice the oscillator, since its block of Delta(i omega) is regular for omega != w.
const ZH_OSC: &str = "par a=0, tau=1, b=1, w=2\n\
x'=-(a+y)*x-b*delay(x,tau)\n\
y'=a-y*y\n\
u'=w*v\n\
v'=-w*u\n\
done\n";

fn osc_setup(w: f64) -> DdeHopfSetup {
    setup(
        [0, 1],
        &[0.07, 1.4071360574511174, 1.0, w],
        &[0.001, -0.249, 0.0, 0.0],
        0.98,
    )
}

// Test 5: a refusal inside a whole run is a branch warning, not a failed run.
#[test]
fn a_refused_special_in_a_run_becomes_a_branch_warning() {
    for which in both() {
        let s = osc_setup(2.0);
        let on = run(ZH_OSC, which.clone(), &s, &zh_opts(), &extras(true, false));
        assert_eq!(on.specials.len(), 1, "{:?}", on.specials);
        let sp = &on.specials[0];
        assert_eq!(sp.label, "ZH");
        assert!(sp.coefficients.is_empty(), "{:?}", sp.coefficients);
        let ours: Vec<&String> = on
            .warnings
            .iter()
            .filter(|w| w.starts_with("coefficients"))
            .collect();
        assert_eq!(ours.len(), 1, "{:?}", on.warnings);
        let prefix = format!(
            "coefficients: the ZH special at index {} has none: ",
            sp.index
        );
        assert!(ours[0].starts_with(&prefix), "{}", ours[0]);
        assert!(ours[0].contains("Delta(2 i omega)"), "{}", ours[0]);
        // The curve and the special are those of a run without the option, and the other
        // warnings and the stop reason do not change.
        let off = run(ZH_OSC, which, &s, &zh_opts(), &extras(false, false));
        same_points_and_specials(&on, &off);
        assert_eq!(on.stop, off.stop);
        assert!(off.warnings.iter().all(|w| !w.starts_with("coefficients")));
        let others: Vec<&String> = on
            .warnings
            .iter()
            .filter(|w| !w.starts_with("coefficients"))
            .collect();
        assert_eq!(others, off.warnings.iter().collect::<Vec<_>>());
    }
}

// Test 6: the same model with the oscillator moved off 2 omega gets its coefficients and no
// warning, so the refusal above comes from the resonance and not from the extra variables.
#[test]
fn the_same_run_without_the_resonance_has_coefficients() {
    for which in both() {
        let s = osc_setup(3.0);
        let b = run(ZH_OSC, which, &s, &zh_opts(), &extras(true, false));
        assert_eq!(b.specials.len(), 1, "{:?}", b.specials);
        let sp = &b.specials[0];
        assert!(!sp.coefficients.is_empty());
        assert!(
            b.warnings.iter().all(|w| !w.starts_with("coefficients")),
            "{:?}",
            b.warnings
        );
        // The extra block does not touch the published ZH values (section 8i test 7).
        assert!((value(sp, "g200_re") + 1.0).abs() < 1e-6);
    }
}

// The active control system of Bosschaert, Janssens and Kuznetsov (arXiv 1903.08276 v1, Sec. 8.3),
// the model of SPEC section 8j test 1. Its Hopf-Hopf point is at (zeta, tau) = (ZETA, TAU).
const ACS: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427\n\
    s'=tau*y\n\
    y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1)+0.1*delay(s,1)^3)\n\
    done\n";
const ACS_ZETA: f64 = -0.016_224_370_430_552_95;
const ACS_TAU: f64 = 5.898_023_025_699_427;
const ACS_W1: f64 = 7.644_926_613_449_257_5;
const ACS_W2: f64 = 4.527_486_915_496_89;

// Test 2, the independent half: the Hopf curve of the published model, run through its own
// Hopf-Hopf point. The values come from a numpy script written for SPEC section 8j and from the
// paper (four printed digits). Neither is the engine's own output, unlike the direct call above.
#[test]
fn hh_values_on_the_published_model_match_the_numpy_values_and_the_paper() {
    // (name, numpy value, printed value, unit of the last printed digit)
    let want: [(&str, f64, f64, f64); 11] = [
        ("g2100_re", -0.091_452_514_113_883_53, -0.0915, 1e-4),
        ("g2100_im", 0.121_439_974_396_385_96, 0.1214, 1e-4),
        ("g1011_re", -0.308_448_621_979_563_9, -0.3084, 1e-4),
        ("g1011_im", 0.409_589_535_276_780_66, 0.4096, 1e-4),
        ("g1110_re", 0.215_070_731_513_221_98, 0.2151, 1e-4),
        ("g1110_im", 0.387_604_330_232_565_2, 0.3876, 1e-4),
        ("g0021_re", 0.181_346_219_418_256_54, 0.1813, 1e-4),
        ("g0021_im", 0.326_825_409_590_890_07, 0.3268, 1e-4),
        ("theta", -1.700_882_560_270_852_2, -1.7009, 1e-4),
        ("delta", -2.351_720_273_599_037, -2.3517, 1e-4),
        ("product", -0.016_584_567_690_847_526, -0.0166, 1e-4),
    ];
    for which in both() {
        // A Hopf point of the larger-frequency curve at tau = 5.8, found with a Newton solve on
        // the characteristic equation in a scratch script.
        let s = setup(
            [0, 1],
            &[-0.061_838_555_573_730_6, 5.8],
            &[0.0, 0.0],
            7.468_270_805_042_674,
        );
        let b = run(
            ACS,
            which,
            &s,
            &opts(0.05, (5.5, 6.3), 1.0),
            &extras(true, true),
        );
        let hh: Vec<_> = b.specials.iter().filter(|sp| sp.label == "HH").collect();
        assert_eq!(hh.len(), 1, "{:?}", b.specials);
        let sp = hh[0];
        assert!((sp.alpha[0] - ACS_ZETA).abs() < 1e-9, "{:?}", sp.alpha);
        assert!((sp.alpha[1] - ACS_TAU).abs() < 1e-9, "{:?}", sp.alpha);
        assert!((value(sp, "omega_large") - ACS_W1).abs() < 1e-9);
        assert!((value(sp, "omega_small") - ACS_W2).abs() < 1e-9);
        for (name, numpy, printed, unit) in want {
            let got = value(sp, name);
            // The special is located to the last digits of the parameters, and the run and the
            // numpy script agree to about 1e-15 relative. The bound leaves room for a libm that
            // rounds differently.
            assert!(
                (got - numpy).abs() <= 1e-10 * numpy.abs(),
                "{name}: {got} vs numpy {numpy}"
            );
            assert!(
                (got - printed).abs() <= 0.5 * unit * (1.0 + 1e-9),
                "{name}: {got} vs printed {printed}"
            );
        }
        assert_eq!(value(sp, "sign"), -1.0);
    }
}
