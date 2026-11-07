//! The critical normal-form coefficients at a Hopf-Hopf point of a delay equilibrium (SPEC
//! section 8j, acceptance tests 1 to 4 and 6).
//!
//! The reference values come from numpy scripts written for the section: one with the formulas
//! of the section, and one that solves the homological equation in the eigenbasis (ODE limit
//! only). The active control system values are also the published ones (Bosschaert, Janssens
//! and Kuznetsov, arXiv 1903.08276 v1, Sec. 8.3).

use pc_engine::{DdeHopfHopf, DdeModel, EngineError, Evaluator, build_dde, dde_hopf_hopf};
use pc_ode::{MapLoader, Severity, SourceMap, load};

const ZETA: f64 = -0.016_224_370_430_552_95;
const TAU: f64 = 5.898_023_025_699_427;
const W1: f64 = 7.644_926_613_449_257_5;
const W2: f64 = 4.527_486_915_496_89;

/// The active control system of the paper's Section 8.3 (SPEC section 8j, tests 1 and 2).
const ACS: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427\n\
                   s'=tau*y\n\
                   y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1)+0.1*delay(s,1)^3)\n\
                   done\n";

/// The same system after `s = sigma + al sigma^2`, written to third order.
const ACS_ALPHA: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427, al=0.3\n\
     s'=tau*y-2*al*tau*s*y+4*al^2*tau*s^2*y\n\
     y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1))-tau*al*s^2\
     -tau*al*0.1*delay(s,1)^2+tau*0.1*delay(s,1)^3\n\
     done\n";

/// `ACS_ALPHA` with the quadratic terms removed.
const ACS_ALPHA_NO_B: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427, al=0.3\n\
     s'=tau*y+4*al^2*tau*s^2*y\n\
     y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1))+tau*0.1*delay(s,1)^3\n\
     done\n";

struct Expect {
    g2100: [f64; 2],
    g1011: [f64; 2],
    g1110: [f64; 2],
    g0021: [f64; 2],
    theta: f64,
    delta: f64,
    product: f64,
}

fn acs_expect() -> Expect {
    Expect {
        g2100: [-0.091_452_514_113_883_53, 0.121_439_974_396_385_96],
        g1011: [-0.308_448_621_979_563_9, 0.409_589_535_276_780_66],
        g1110: [0.215_070_731_513_221_98, 0.387_604_330_232_565_2],
        g0021: [0.181_346_219_418_256_54, 0.326_825_409_590_890_07],
        theta: -1.700_882_560_270_852_2,
        delta: -2.351_720_273_599_037,
        product: -0.016_584_567_690_847_526,
    }
}

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

fn hh(
    src: &str,
    which: Evaluator,
    x: &[f64],
    p: &[f64],
    w1: f64,
    w2: f64,
) -> Result<DdeHopfHopf, EngineError> {
    dde_hopf_hopf(&mut dde(src, which), x, p, w1, w2)
}

fn ok(src: &str, which: Evaluator, x: &[f64], p: &[f64], w1: f64, w2: f64) -> DdeHopfHopf {
    match hh(src, which, x, p, w1, w2) {
        Ok(r) => r,
        Err(e) => panic!("hopf_hopf failed: {e}"),
    }
}

fn near(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs(),
        "{what}: {got} vs {want}"
    );
}

/// Both parts of `got` against `want`, each relative to its own size.
fn near2(got: [f64; 2], want: [f64; 2], rel: f64, what: &str) {
    for k in 0..2 {
        let tol = rel * want[k].abs();
        assert!(
            (got[k] - want[k]).abs() <= tol,
            "{what}[{k}]: {} vs {}",
            got[k],
            want[k]
        );
    }
}

/// `got` is within half a unit of the last printed digit.
fn printed(got: f64, printed: f64, unit: f64, what: &str) {
    assert!(
        (got - printed).abs() <= 0.5 * unit * (1.0 + 1e-9),
        "{what}: {got} vs printed {printed}"
    );
}

fn check_values(r: &DdeHopfHopf, e: &Expect, rel_g: f64, rel_n: f64) {
    near2(r.g2100, e.g2100, rel_g, "g2100");
    near2(r.g1011, e.g1011, rel_g, "g1011");
    near2(r.g1110, e.g1110, rel_g, "g1110");
    near2(r.g0021, e.g0021, rel_g, "g0021");
    near(r.theta.expect("theta"), e.theta, rel_n, "theta");
    near(r.delta.expect("delta"), e.delta, rel_n, "delta");
    near(r.product, e.product, rel_n, "product");
    assert_eq!(r.sign, if e.product > 0.0 { 1 } else { -1 });
}

fn cdist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[test]
fn the_published_point_matches_the_numpy_oracle_and_the_paper() {
    let e = acs_expect();
    for which in both() {
        let r = ok(ACS, which, &[0.0, 0.0], &[ZETA, TAU], W1, W2);
        check_values(&r, &e, 1e-8, 1e-8);
        near(r.omega1, W1, 1e-15, "omega1");
        near(r.omega2, W2, 1e-15, "omega2");
        assert!(r.null_residual_1 < 1e-10, "{}", r.null_residual_1);
        assert!(r.null_residual_2 < 1e-10, "{}", r.null_residual_2);
        assert_eq!(r.equilibrium_residual, 0.0);
        printed(r.g2100[0], -0.0915, 1e-4, "Re g2100");
        printed(r.g2100[1], 0.1214, 1e-4, "Im g2100");
        printed(r.g1011[0], -0.3084, 1e-4, "Re g1011");
        printed(r.g1011[1], 0.4096, 1e-4, "Im g1011");
        printed(r.g1110[0], 0.2151, 1e-4, "Re g1110");
        printed(r.g1110[1], 0.3876, 1e-4, "Im g1110");
        printed(r.g0021[0], 0.1813, 1e-4, "Re g0021");
        printed(r.g0021[1], 0.3268, 1e-4, "Im g0021");
        printed(r.theta.unwrap(), -1.7009, 1e-4, "theta");
        printed(r.delta.unwrap(), -2.3517, 1e-4, "delta");
        printed(r.product, -0.0166, 1e-4, "product");
        // The difficult type: the product is negative.
        assert_eq!(r.sign, -1);
    }
}

#[test]
fn a_change_of_coordinates_keeps_the_four_coefficients() {
    let e = acs_expect();
    for al in [0.3, -0.7] {
        for which in both() {
            let r = ok(
                ACS_ALPHA,
                which.clone(),
                &[0.0, 0.0],
                &[ZETA, TAU, al],
                W1,
                W2,
            );
            for (got, want, name) in [
                (r.g2100, e.g2100, "g2100"),
                (r.g1011, e.g1011, "g1011"),
                (r.g1110, e.g1110, "g1110"),
                (r.g0021, e.g0021, "g0021"),
            ] {
                let rel = cdist(got, want) / want[0].hypot(want[1]);
                assert!(
                    rel <= 1e-9,
                    "al = {al}, {name}: {got:?} vs {want:?} ({rel:e})"
                );
            }
            near(r.theta.unwrap(), e.theta, 1e-8, "theta");
            near(r.delta.unwrap(), e.delta, 1e-8, "delta");
            // The test is not vacuous: with the quadratic terms removed the coefficients move.
            let off = ok(ACS_ALPHA_NO_B, which, &[0.0, 0.0], &[ZETA, TAU, al], W1, W2);
            for (got, want, name) in [(off.g2100, e.g2100, "g2100"), (off.g0021, e.g0021, "g0021")]
            {
                let rel = cdist(got, want) / want[0].hypot(want[1]);
                assert!(rel >= 0.1, "al = {al}, {name} without B moved by {rel:e}");
            }
        }
    }
}

/// The A matrix of the ODE limit, to 16 digits (SPEC section 8j, test 3).
const ODE_A: [[f64; 4]; 4] = [
    [
        0.442_799_678_197_908_3,
        -1.240_124_698_310_538_7,
        0.829_666_130_329_847,
        -0.821_902_654_867_256_7,
    ],
    [
        3.556_234_915_526_951,
        0.591_029_766_693_483_3,
        0.243_724_859_211_585_04,
        -1.111_061_946_902_655,
    ],
    [
        -1.437_731_295_253_419,
        -0.070_876_910_699_919_53,
        -0.196_218_825_422_365_24,
        -0.736_283_185_840_708_1,
    ],
    [
        0.962_107_803_700_724_1,
        -0.383_226_065_969_428_7,
        1.286_846_339_501_206_9,
        -0.837_610_619_469_026_5,
    ],
];

fn ode_limit_source() -> String {
    let lin = |i: usize| {
        format!(
            "({:?})*x0+({:?})*x1+({:?})*x2+({:?})*x3",
            ODE_A[i][0], ODE_A[i][1], ODE_A[i][2], ODE_A[i][3]
        )
    };
    format!(
        "x0'={}+0.5*x0^2-0.8*x0*x1+0.3*x1*x2+0.4*x2*x3\
         +0.3*x0^3-0.4*x0*x1*x2+0.2*x2^2*x3\n\
         x1'={}+0.7*x1^2+0.4*x0*x2-0.6*x2^2+0.25*x0*x3\
         +0.5*x1^3+0.2*x0^2*x1-0.3*x1*x3^2\n\
         x2'={}-0.5*x0*x1+0.9*x1*x3+0.2*x0^2-0.3*x3^2\
         -0.6*x2^3+0.35*x0*x2^2+0.25*x0*x1*x3\n\
         x3'={}+0.6*x0*x3-0.35*x1*x2+0.45*x2*x3\
         +0.4*x3^3-0.2*x1^2*x2+0.3*x0*x2*x3\n\
         w'=-2*w-delay(w,1)\ndone\n",
        lin(0),
        lin(1),
        lin(2),
        lin(3)
    )
}

#[test]
fn the_ode_limit_matches_the_homological_equation() {
    let e = Expect {
        g2100: [0.249_794_419_171_067_65, -0.089_248_578_844_897_73],
        g1011: [0.452_999_071_036_476_2, -0.098_274_043_191_440_84],
        g1110: [0.144_076_054_623_692_36, 1.037_403_007_139_733_3],
        g0021: [-0.357_841_205_569_087_2, -0.439_319_101_566_813_3],
        theta: -1.265_922_045_830_513_4,
        delta: 0.576_778_516_917_242_2,
        product: -0.089_386_736_100_604_76,
    };
    let src = ode_limit_source();
    for which in both() {
        let r = ok(&src, which, &[0.0; 5], &[], 2.3, 1.0);
        check_values(&r, &e, 1e-9, 1e-9);
        assert!(r.null_residual_1 < 1e-10, "{}", r.null_residual_1);
        assert!(r.null_residual_2 < 1e-10, "{}", r.null_residual_2);
        assert_eq!(r.equilibrium_residual, 0.0);
    }
}

/// Two oscillators coupled through the radii, and the decoupled `w` that gives the model a delay.
fn closed_source(a: f64, b: f64, c: f64, d: f64, w1: f64, w2: f64) -> String {
    let r = |p: f64, q: f64| format!("({p:?}*(x0^2+x1^2)+({q:?})*(x2^2+x3^2))");
    let first = r(a, b);
    let second = r(c, d);
    format!(
        "x0'=-({w1:?})*x1+x0*{first}\nx1'=({w1:?})*x0+x1*{first}\n\
         x2'=-({w2:?})*x3+x2*{second}\nx3'=({w2:?})*x2+x3*{second}\n\
         w'=-2*w-delay(w,1)\ndone\n"
    )
}

#[test]
fn the_closed_form_point_gives_twice_the_radius_coefficients() {
    let (a, b, c, d) = (0.7, -0.4, 0.55, -0.9);
    let src = closed_source(a, b, c, d, 2.3, 1.0);
    for which in both() {
        let r = ok(&src, which, &[0.0; 5], &[], 2.3, 1.0);
        for (g, want, name) in [
            (r.g2100, 2.0 * a, "g2100"),
            (r.g1011, 2.0 * b, "g1011"),
            (r.g1110, 2.0 * c, "g1110"),
            (r.g0021, 2.0 * d, "g0021"),
        ] {
            near(g[0], want, 1e-12, name);
            assert!(g[1].abs() < 1e-12, "{name} has imaginary part {}", g[1]);
        }
        near(r.theta.unwrap(), 4.0 / 9.0, 1e-12, "theta");
        near(r.delta.unwrap(), 11.0 / 14.0, 1e-12, "delta");
        near(r.product, -2.52, 1e-12, "product");
        assert_eq!(r.sign, -1);
    }
}

/// Oscillators `u' = -w v, v' = w u` at the given frequencies, `extra` states (each line of
/// `extra` ends in a newline), and the decoupled `w` with the delay time `tt` (a parameter, so
/// that a bad delay time can be passed).
fn osc_source(freqs: &[f64], extra: &str) -> String {
    let mut s = String::from("par tt=1\n");
    for (k, w) in freqs.iter().enumerate() {
        s += &format!("u{k}'=-({w:?})*v{k}\nv{k}'=({w:?})*u{k}\n");
    }
    s += extra;
    s += "w'=-2*w-delay(w,tt)\ndone\n";
    s
}

/// Two states of an oscillator at `f`, named `e0` and `e1`.
fn extra_osc(f: f64) -> String {
    format!("e0'=-({f:?})*e1\ne1'=({f:?})*e0\n")
}

/// A Jordan block at `i f`: four states `j0` to `j3`.
fn jordan(f: f64) -> String {
    format!("j0'=-({f:?})*j1+j2\nj1'=({f:?})*j0+j3\nj2'=-({f:?})*j3\nj3'=({f:?})*j2\n")
}

fn expect_bad(r: Result<DdeHopfHopf, EngineError>, what: &str, needle: Option<&str>) {
    match r {
        Err(EngineError::BadProblem(msg)) => {
            if let Some(n) = needle {
                assert!(msg.contains(n), "{what}: message {msg:?} lacks {n:?}");
            }
        }
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    }
}

fn expect_singular(r: Result<DdeHopfHopf, EngineError>, what: &str, needle: &str, absent: &str) {
    match r {
        Err(EngineError::Singular(msg)) => {
            assert!(
                msg.contains(needle),
                "{what}: message {msg:?} lacks {needle:?}"
            );
            assert!(
                !msg.contains(absent),
                "{what}: message {msg:?} should not contain {absent:?}"
            );
        }
        other => panic!("{what}: expected Singular, got {other:?}"),
    }
}

#[test]
fn bad_inputs_are_refused_as_bad_problems() {
    let src = osc_source(&[2.3, 1.0], "");
    let x = [0.0; 5];
    for which in both() {
        let go = |x: &[f64], p: &[f64], w1: f64, w2: f64| hh(&src, which.clone(), x, p, w1, w2);
        expect_bad(go(&x[..4], &[1.0], 2.3, 1.0), "short x", None);
        expect_bad(go(&x, &[], 2.3, 1.0), "short p", None);
        expect_bad(go(&x, &[1.0, 2.0], 2.3, 1.0), "long p", None);
        for w in [0.0, f64::NAN, 1e-9, -1.0, f64::INFINITY] {
            expect_bad(
                go(&x, &[1.0], 2.3, w),
                &format!("omega2 {w}"),
                Some("omega2"),
            );
            expect_bad(
                go(&x, &[1.0], w, 1.0),
                &format!("omega1 {w}"),
                Some("omega1"),
            );
        }
        for tau in [0.0, -1.0, f64::NAN] {
            expect_bad(go(&x, &[tau], 2.3, 1.0), &format!("tt {tau}"), None);
        }
    }
}

#[test]
fn frequencies_in_the_wrong_order_are_refused_and_not_sorted() {
    let src = osc_source(&[2.3, 1.0], "");
    for which in both() {
        for (w1, w2) in [(1.0, 2.3), (1.0, 1.0)] {
            expect_bad(
                hh(&src, which.clone(), &[0.0; 5], &[1.0], w1, w2),
                &format!("({w1}, {w2})"),
                Some("greater than"),
            );
        }
    }
}

#[test]
fn a_point_without_a_zero_root_names_delta_zero() {
    let src = osc_source(&[2.3, 1.0], "z'=0*z\n");
    for which in both() {
        expect_singular(
            hh(&src, which, &[0.0; 6], &[1.0], 2.3, 1.0),
            "z' = 0*z",
            "Delta(0)",
            "Delta(2",
        );
    }
}

#[test]
fn a_frequency_that_is_not_a_hopf_frequency_names_its_delta() {
    let src = osc_source(&[2.3, 1.0], "");
    for which in both() {
        expect_bad(
            hh(&src, which.clone(), &[0.0; 5], &[1.0], 2.31, 1.0),
            "omega1 = 2.31",
            Some("Delta(i omega1)"),
        );
        expect_bad(
            hh(&src, which, &[0.0; 5], &[1.0], 2.3, 1.01),
            "omega2 = 1.01",
            Some("Delta(i omega2)"),
        );
    }
}

#[test]
fn a_jordan_block_at_either_root_names_that_root() {
    // The block replaces the oscillator at that frequency, so the root has one eigenvector
    // and a generalized one.
    let at_first = osc_source(&[1.0], &jordan(2.3));
    let at_second = osc_source(&[2.3], &jordan(1.0));
    for which in both() {
        expect_singular(
            hh(&at_first, which.clone(), &[0.0; 7], &[1.0], 2.3, 1.0),
            "Jordan block at omega1",
            "omega1",
            "omega2",
        );
        expect_singular(
            hh(&at_second, which, &[0.0; 7], &[1.0], 2.3, 1.0),
            "Jordan block at omega2",
            "omega2",
            "omega1",
        );
    }
}

#[test]
fn each_second_order_resonance_names_its_delta() {
    let cases = [
        (4.6, "Delta(2 i omega1)"),
        (2.0, "Delta(2 i omega2)"),
        (3.3, "Delta(i (omega1 + omega2))"),
        (1.3, "Delta(i (omega1 - omega2))"),
    ];
    for (f, name) in cases {
        let src = osc_source(&[2.3, 1.0], &extra_osc(f));
        for which in both() {
            match hh(&src, which, &[0.0; 7], &[1.0], 2.3, 1.0) {
                Err(EngineError::Singular(msg)) => {
                    assert!(
                        msg.contains(name),
                        "extra oscillator at {f}: message {msg:?}"
                    );
                }
                other => panic!("extra oscillator at {f}: expected Singular, got {other:?}"),
            }
        }
    }
}

#[test]
fn an_exact_one_to_two_resonance_names_the_first_singular_delta() {
    // Delta(2 i omega2) and Delta(i (omega1 - omega2)) are both singular here.
    let src = osc_source(&[2.0, 1.0], "");
    for which in both() {
        expect_singular(
            hh(&src, which, &[0.0; 5], &[1.0], 2.0, 1.0),
            "1:2",
            "Delta(2 i omega2)",
            "Delta(i (omega1 - omega2))",
        );
    }
}

#[test]
fn a_positive_product_gives_sign_plus_one() {
    // Same closed form with d = +0.9: g0021 = 1.8, so the product is 1.4 * 1.8 = 2.52 and
    // theta = -0.8 / 1.8.
    let src = closed_source(0.7, -0.4, 0.55, 0.9, 2.3, 1.0);
    for which in both() {
        let r = ok(&src, which, &[0.0; 5], &[], 2.3, 1.0);
        near(r.g0021[0], 1.8, 1e-12, "g0021");
        near(r.theta.unwrap(), -4.0 / 9.0, 1e-12, "theta");
        near(r.delta.unwrap(), 11.0 / 14.0, 1e-12, "delta");
        near(r.product, 2.52, 1e-12, "product");
        assert_eq!(r.sign, 1);
    }
}

#[test]
fn the_equilibrium_residual_reports_a_point_that_is_not_an_equilibrium() {
    // `w` is decoupled and linear, so moving it off 0 leaves the four coefficients alone and
    // makes f(w) = -2 w - w = -0.3 at w = 0.1. The point is not polished.
    let src = closed_source(0.7, -0.4, 0.55, -0.9, 2.3, 1.0);
    for which in both() {
        let at_zero = ok(&src, which.clone(), &[0.0; 5], &[], 2.3, 1.0);
        let off = ok(&src, which, &[0.0, 0.0, 0.0, 0.0, 0.1], &[], 2.3, 1.0);
        near(off.equilibrium_residual, 0.3, 1e-12, "equilibrium_residual");
        assert_eq!(at_zero.equilibrium_residual, 0.0);
        for (a, b, name) in [
            (off.g2100, at_zero.g2100, "g2100"),
            (off.g1011, at_zero.g1011, "g1011"),
            (off.g1110, at_zero.g1110, "g1110"),
            (off.g0021, at_zero.g0021, "g0021"),
        ] {
            for k in 0..2 {
                assert!((a[k] - b[k]).abs() <= 1e-12, "{name}[{k}]: {a:?} vs {b:?}");
            }
        }
    }
}

#[test]
fn a_semisimple_double_root_is_refused_and_names_that_root() {
    // A second oscillator at the same frequency gives a null space of dimension 2, and the
    // pairing of a null vector with its left partner is not 0, so only the dimension test
    // sees it.
    let at_first = osc_source(&[2.3, 1.0], &extra_osc(2.3));
    let at_second = osc_source(&[2.3, 1.0], &extra_osc(1.0));
    for which in both() {
        expect_singular(
            hh(&at_first, which.clone(), &[0.0; 7], &[1.0], 2.3, 1.0),
            "second oscillator at omega1",
            "null space of dimension 2 or more",
            "omega2",
        );
        expect_singular(
            hh(&at_second, which, &[0.0; 7], &[1.0], 2.3, 1.0),
            "second oscillator at omega2",
            "null space of dimension 2 or more",
            "omega1",
        );
    }
}

#[test]
fn the_checks_of_one_root_run_in_the_spec_order() {
    // Jordan block at omega1 and a double root at omega2: the pairing of omega1 comes first.
    let jordan_then_double = osc_source(&[1.0], &format!("{}{}", jordan(2.3), extra_osc(1.0)));
    // Double root at omega1 and a Jordan block at omega2: the dimension test of omega1 comes
    // first.
    let double_then_jordan = osc_source(&[2.3], &format!("{}{}", extra_osc(2.3), jordan(1.0)));
    // A double root at omega1 and an omega2 that is not a Hopf frequency: both residuals are
    // checked before any multiplicity or pairing test.
    let double_and_off = osc_source(&[2.3, 1.0], &extra_osc(2.3));
    for which in both() {
        expect_singular(
            hh(
                &jordan_then_double,
                which.clone(),
                &[0.0; 9],
                &[1.0],
                2.3,
                1.0,
            ),
            "Jordan at omega1, double at omega2",
            "omega1",
            "omega2",
        );
        expect_singular(
            hh(
                &double_then_jordan,
                which.clone(),
                &[0.0; 9],
                &[1.0],
                2.3,
                1.0,
            ),
            "double at omega1, Jordan at omega2",
            "null space of dimension 2 or more",
            "omega2",
        );
        expect_bad(
            hh(&double_and_off, which, &[0.0; 7], &[1.0], 2.3, 1.5),
            "double at omega1, omega2 off",
            Some("omega2"),
        );
    }
}
