//! The critical normal-form coefficients at a zero-Hopf point of a delay equilibrium (SPEC
//! section 8i, acceptance tests 1 to 4, 7 and 8).
//!
//! The reference values come from two numpy scripts written for the section: one with the
//! bordered formulas of Lemma 4, and one that solves the homological equation in the
//! eigenbasis (ODE limit only). The Hindmarsh-Rose values are also the published ones
//! (Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1, Sec. 6.2.1).

use pc_engine::{DdeModel, DdeZeroHopf, EngineError, Evaluator, build_dde, dde_zero_hopf};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::FRAC_PI_2;

/// The Hindmarsh-Rose neuron with a delay in the quadratic term. `sl` is the paper's `S`.
const HR: &str = "par a=1, b=3, c=1, d=5, chi=-1.6, r=0.001, sl=-0.57452592, iapp=0, tau=1\n\
                  x'=y-a*x^3+b*delay(x,tau)^2-c*z+iapp\n\
                  y'=c-d*x^2-y\n\
                  z'=r*(sl*(x-chi)-z)\ndone\n";

struct Hr {
    p: [f64; 9],
    x: [f64; 3],
    omega: f64,
    expect: Expect,
}

struct Expect {
    g200: [f64; 2],
    g110: [f64; 2],
    g011: [f64; 2],
    g300: [f64; 2],
    g111: [f64; 2],
    g210: [f64; 2],
    g021: [f64; 2],
    s_product: f64,
    theta: f64,
    e: f64,
}

fn point1() -> Hr {
    Hr {
        p: [
            1.0,
            3.0,
            1.0,
            5.0,
            -1.6,
            0.001,
            -0.57452592,
            -1.957_934_372_224_000_1,
            5.768_830_919_756_731,
        ],
        x: [
            0.130_799_999_999_999_94,
            0.914_456_800_000_000_1,
            -0.994_389_462_336_000_1,
        ],
        omega: 1.007_916_143_551_579_8,
        expect: Expect {
            g200: [-0.002_370_511_705_141_935_4, 0.0],
            g110: [0.329_575_777_314_695_7, 0.700_582_959_211_202_2],
            g011: [-0.007_798_607_170_293_79, 0.0],
            g300: [0.010_646_551_403_411_44, 0.0],
            g111: [-0.014_622_249_811_153_01, 0.0],
            g210: [-2.776_405_821_689_472_5, -3.180_644_918_483_814_4],
            g021: [-0.374_515_574_007_687_87, -2.475_382_718_221_687],
            s_product: 1.848_668_958_098_525_5e-5,
            theta: -139.031_491_217_615_47,
            e: 15.694_064_382_557_59,
        },
    }
}

fn point2() -> Hr {
    Hr {
        p: [
            1.0,
            3.0,
            1.0,
            5.0,
            -1.6,
            1.4,
            -8.0,
            -18.849_042_475_527_19,
            0.940_246_941_050_084_6,
        ],
        x: [
            1.097_167_540_709_727,
            -5.018_883_061_935_152,
            -21.577_340_325_677_817,
        ],
        omega: 5.604_243_934_764_924,
        expect: Expect {
            g200: [0.201_133_569_688_555_72, 0.0],
            g110: [-0.031_555_114_715_046_5, -0.078_392_520_201_704_37],
            g011: [8.800_187_670_315_616, 0.0],
            g300: [-0.202_773_836_953_084_54, 0.0],
            g111: [-9.518_981_756_536_286, 0.0],
            g210: [0.025_797_202_214_236_054, 0.047_465_229_337_271_556],
            g021: [1.245_718_635_098_787_4, 2.192_692_461_061_300_3],
            s_product: 1.770_013_160_059_794_8,
            theta: -0.156_886_365_433_218_66,
            e: -0.037_793_674_456_654_1,
        },
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

fn zh(
    src: &str,
    which: Evaluator,
    x: &[f64],
    p: &[f64],
    omega: f64,
) -> Result<DdeZeroHopf, EngineError> {
    dde_zero_hopf(&mut dde(src, which), x, p, omega)
}

fn ok(src: &str, which: Evaluator, x: &[f64], p: &[f64], omega: f64) -> DdeZeroHopf {
    match zh(src, which, x, p, omega) {
        Ok(r) => r,
        Err(e) => panic!("zero_hopf failed: {e}"),
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

/// `got` rounds to `printed` at `digits` decimals, that is within half a unit of the last digit.
fn printed(got: f64, printed: f64, unit: f64, what: &str) {
    assert!(
        (got - printed).abs() <= 0.5 * unit * (1.0 + 1e-9),
        "{what}: {got} vs printed {printed}"
    );
}

/// The imaginary part of a coefficient that is real by theory.
fn real_by_theory(g: [f64; 2], what: &str) {
    assert!(
        g[1].abs() <= 1e-12 * g[0].abs(),
        "{what} has imaginary part {} next to {}",
        g[1],
        g[0]
    );
}

fn check_values(r: &DdeZeroHopf, e: &Expect, rel_g: f64, rel_n: f64) {
    near(r.g200[0], e.g200[0], rel_g, "g200");
    near2(r.g110, e.g110, rel_g, "g110");
    near(r.g011[0], e.g011[0], rel_g, "g011");
    near(r.g300[0], e.g300[0], rel_g, "g300");
    near(r.g111[0], e.g111[0], rel_g, "g111");
    near2(r.g210, e.g210, rel_g, "g210");
    near2(r.g021, e.g021, rel_g, "g021");
    near(r.s_product, e.s_product, rel_n, "s_product");
    near(r.theta.expect("theta"), e.theta, rel_n, "theta");
    near(r.e.expect("e"), e.e, rel_n, "e");
    assert_eq!(r.s, if e.s_product > 0.0 { 1 } else { -1 });
    for (g, name) in [
        (r.g200, "g200"),
        (r.g011, "g011"),
        (r.g300, "g300"),
        (r.g111, "g111"),
    ] {
        real_by_theory(g, name);
    }
}

#[test]
fn hindmarsh_rose_point_1_matches_the_numpy_oracle_and_the_paper() {
    let h = point1();
    for which in both() {
        let r = ok(HR, which, &h.x, &h.p, h.omega);
        check_values(&r, &h.expect, 1e-8, 1e-8);
        near(r.omega, h.omega, 1e-15, "omega");
        assert!(r.null_residual_zero < 1e-10, "{}", r.null_residual_zero);
        assert!(r.null_residual_hopf < 1e-10, "{}", r.null_residual_hopf);
        // The point is the published one, to the digits of the paper.
        printed(r.g200[0], -0.0024, 1e-4, "g200");
        printed(r.g110[0], 0.3296, 1e-4, "Re g110");
        printed(r.g110[1], 0.7006, 1e-4, "Im g110");
        printed(r.g011[0], -0.0078, 1e-4, "g011");
        printed(r.g300[0], 0.0106, 1e-4, "g300");
        printed(r.g111[0], -0.0146, 1e-4, "g111");
        printed(r.g210[0], -2.7764, 1e-4, "Re g210");
        printed(r.g210[1], -3.1806, 1e-4, "Im g210");
        printed(r.g021[0], -0.3745, 1e-4, "Re g021");
        printed(r.g021[1], -2.4754, 1e-4, "Im g021");
        printed(r.s_product, 1.8487e-5, 1e-9, "s");
        printed(r.theta.unwrap(), -139.0315, 1e-4, "theta");
        printed(r.e.unwrap(), 15.6941, 1e-4, "e");
    }
}

#[test]
fn hindmarsh_rose_point_2_matches_the_numpy_oracle_and_the_paper() {
    let h = point2();
    for which in both() {
        let r = ok(HR, which, &h.x, &h.p, h.omega);
        check_values(&r, &h.expect, 1e-8, 1e-8);
        assert!(r.null_residual_zero < 1e-10, "{}", r.null_residual_zero);
        assert!(r.null_residual_hopf < 1e-10, "{}", r.null_residual_hopf);
        printed(r.s_product, 1.770013, 1e-6, "s");
        printed(r.theta.unwrap(), -0.156886, 1e-6, "theta");
        printed(r.e.unwrap(), -0.037794, 1e-6, "e");
    }
}

/// The A matrix of the ODE limit, to 16 digits, and the polynomial terms of SPEC test 3.
const ODE_A: [[f64; 3]; 3] = [
    [
        -0.279_141_104_294_478_5,
        -0.590_184_049_079_754_6,
        -0.095_705_521_472_392_64,
    ],
    [
        0.073_773_006_134_969_33,
        -0.271_165_644_171_779_24,
        -1.256_134_969_325_153_6,
    ],
    [
        0.630_061_349_693_251_4,
        1.443_558_282_208_589,
        0.550_306_748_466_257_6,
    ],
];

fn ode_limit_source() -> String {
    let lin = |i: usize| {
        format!(
            "({:?})*x0+({:?})*x1+({:?})*x2",
            ODE_A[i][0], ODE_A[i][1], ODE_A[i][2]
        )
    };
    format!(
        "x0'={}+0.5*x0^2-0.8*x0*x1+0.3*x1*x2+0.3*x0^3-0.4*x0*x1*x2\n\
         x1'={}+0.7*x1^2+0.4*x0*x2-0.6*x2^2+0.5*x1^3+0.2*x0^2*x1\n\
         x2'={}-0.5*x0*x1+0.9*x1*x2+0.2*x0^2-0.6*x2^3+0.35*x0*x2^2\n\
         w'=-2*w-delay(w,1)\ndone\n",
        lin(0),
        lin(1),
        lin(2)
    )
}

#[test]
fn the_ode_limit_matches_the_homological_equation() {
    let e = Expect {
        g200: [0.924_610_316_151_116_8, 0.0],
        g110: [-0.814_028_849_034_943_9, -0.009_644_784_401_850_182],
        g011: [-0.031_624_874_642_161_88, 0.0],
        g300: [-0.022_260_876_063_725_51, 0.0],
        g111: [-0.186_963_830_327_294_16, 0.0],
        g210: [0.391_713_405_114_732_3, 1.323_425_661_808_935_7],
        g021: [0.004_070_414_517_577_045, 0.208_949_050_381_805_36],
        s_product: -0.029_240_685_341_128_735,
        theta: -0.880_402_083_791_914_3,
        e: -1.820_143_422_321_809_8,
    };
    let src = ode_limit_source();
    for which in both() {
        let r = ok(&src, which, &[0.0; 4], &[], 1.3);
        check_values(&r, &e, 1e-9, 1e-9);
        assert!(r.null_residual_zero < 1e-10, "{}", r.null_residual_zero);
        assert!(r.null_residual_hopf < 1e-10, "{}", r.null_residual_hopf);
        assert_eq!(r.equilibrium_residual, 0.0);
    }
}

/// The Hindmarsh-Rose model in `s` with `x = x* + s + al s^2`.
fn shifted_hr(x_star: f64) -> String {
    let big_x = format!("({x_star:?}+s+al*s^2)");
    let big_xd = format!("({x_star:?}+delay(s,tau)+al*delay(s,tau)^2)");
    format!(
        "par a=1, b=3, c=1, d=5, chi=-1.6, r=0.001, sl=-0.57452592, iapp=0, tau=1, al=0.3\n\
         s'=(y-a*{big_x}^3+b*{big_xd}^2-c*z+iapp)/(1+2*al*s)\n\
         y'=c-d*{big_x}^2-y\n\
         z'=r*(sl*({big_x}-chi)-z)\ndone\n"
    )
}

#[test]
fn a_change_of_coordinates_keeps_the_invariants_and_moves_the_others() {
    for h in [point1(), point2()] {
        let src = shifted_hr(h.x[0]);
        let x = [0.0, h.x[1], h.x[2]];
        for al in [0.3, -0.7] {
            let mut p = h.p.to_vec();
            p.push(al);
            for which in both() {
                let r = ok(&src, which, &x, &p, h.omega);
                let e = &h.expect;
                near(r.g200[0], e.g200[0], 1e-9, "g200");
                near2(r.g110, e.g110, 1e-9, "g110");
                near(r.g011[0], e.g011[0], 1e-9, "g011");
                near(r.g300[0], e.g300[0], 1e-9, "g300");
                near(r.s_product, e.s_product, 1e-8, "s_product");
                near(r.theta.unwrap(), e.theta, 1e-8, "theta");
                near(r.e.unwrap(), e.e, 1e-8, "e");
                // The test is not vacuous: the three non-invariants move.
                let moved = |got: [f64; 2], want: [f64; 2]| {
                    let d = ((got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2)).sqrt();
                    d / want[0].hypot(want[1])
                };
                let m210 = moved(r.g210, e.g210);
                let m111 = moved(r.g111, e.g111);
                let m021 = moved(r.g021, e.g021);
                assert!(
                    m210.max(m111).max(m021) >= 1e-3,
                    "al = {al}: g210, g111, g021 moved by {m210:e}, {m111:e}, {m021:e}"
                );
            }
        }
    }
}

/// The model of section 8f test 5: a zero-Hopf point at `(a, tau, b) = (0, pi/2, 1)`.
const CLOSED: &str = "par a=0, tau=1, b=1\n\
                      x'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";

#[test]
fn the_closed_form_point_has_g011_zero_and_no_e() {
    for which in both() {
        let r = ok(CLOSED, which, &[0.0, 0.0], &[0.0, FRAC_PI_2, 1.0], 1.0);
        assert_eq!(r.g200[1], 0.0);
        assert!((r.g200[0] + 1.0).abs() < 1e-12, "{:?}", r.g200);
        let g110 = [-0.288_400_439_142_001, 0.453_018_350_450_290_2];
        assert!((r.g110[0] - g110[0]).abs() < 1e-12, "{:?}", r.g110);
        assert!((r.g110[1] - g110[1]).abs() < 1e-12, "{:?}", r.g110);
        assert_eq!(r.g011, [0.0, 0.0]);
        assert!(r.s_product.abs() < 1e-12);
        assert_eq!(r.s, 0);
        assert!((r.theta.expect("theta") - 0.288_400_439_142_001).abs() < 1e-12);
        assert_eq!(r.e, None);
    }
}

fn expect_bad(r: Result<DdeZeroHopf, EngineError>, what: &str, needle: Option<&str>) {
    match r {
        Err(EngineError::BadProblem(msg)) => {
            if let Some(n) = needle {
                assert!(msg.contains(n), "{what}: message {msg:?} lacks {n:?}");
            }
        }
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    }
}

#[test]
fn bad_inputs_are_refused_as_bad_problems() {
    let h = point1();
    let closed_p = [0.0, FRAC_PI_2, 1.0];
    for which in both() {
        expect_bad(
            zh(HR, which.clone(), &h.x[..2], &h.p, h.omega),
            "short x",
            None,
        );
        expect_bad(
            zh(HR, which.clone(), &h.x, &h.p[..8], h.omega),
            "short p",
            None,
        );
        for w in [0.0, f64::NAN, 1e-9, -1.0, f64::INFINITY] {
            expect_bad(
                zh(CLOSED, which.clone(), &[0.0, 0.0], &closed_p, w),
                &format!("omega {w}"),
                Some("omega"),
            );
        }
        for tau in [0.0, -1.0, f64::NAN] {
            expect_bad(
                zh(CLOSED, which.clone(), &[0.0, 0.0], &[0.0, tau, 1.0], 1.0),
                &format!("tau {tau}"),
                None,
            );
        }
    }
}

#[test]
fn a_point_that_is_not_a_zero_root_names_delta_zero() {
    for which in both() {
        // Point 1 with sl = 0.5 and r = 0.5. Row 3 of Delta(0) is r (-sl, 0, 1), so r
        // factors out of det Delta(0) and changing r alone keeps it singular.
        let h = point1();
        let mut p = h.p;
        p[5] = 0.5;
        p[6] = 0.5;
        expect_bad(
            zh(HR, which.clone(), &h.x, &p, h.omega),
            "sl = 0.5, r = 0.5",
            Some("Delta(0)"),
        );
        // Wright's equation: Delta(0) = a.
        let wright = "par a=1.5707963267948966\nu'=-a*delay(u,1)*(1+u)\ndone\n";
        expect_bad(
            zh(wright, which, &[0.0], &[FRAC_PI_2], FRAC_PI_2),
            "Wright",
            Some("Delta(0)"),
        );
    }
}

#[test]
fn a_frequency_that_is_not_a_hopf_frequency_names_delta_i_omega() {
    let h = point1();
    for which in both() {
        expect_bad(
            zh(HR, which.clone(), &h.x, &h.p, 0.9),
            "omega 0.9",
            Some("Delta(i omega)"),
        );
        // Point 1 with r = 0.01 keeps Delta(0) singular and only spoils Delta(i omega),
        // by a relative residual of about 6e-4. The message must not name Delta(0).
        let mut p = h.p;
        p[5] = 0.01;
        match zh(HR, which, &h.x, &p, h.omega) {
            Err(EngineError::BadProblem(msg)) => {
                assert!(msg.contains("Delta(i omega)"), "message {msg:?}");
                assert!(!msg.contains("Delta(0)"), "message {msg:?}");
            }
            other => panic!("r = 0.01: expected BadProblem, got {other:?}"),
        }
    }
}

#[test]
fn a_jordan_block_at_zero_is_singular() {
    let src = "x'=y\ny'=0\nu'=-v\nv'=u\nw'=-2*w-delay(w,1)\ndone\n";
    for which in both() {
        match zh(src, which, &[0.0; 5], &[], 1.0) {
            Err(EngineError::Singular(msg)) => {
                assert!(msg.contains("root 0 "), "message {msg:?}");
            }
            other => panic!("expected Singular, got {other:?}"),
        }
    }
}

/// Expects `Singular` whose text has every phrase of `has` and none of `lacks`.
fn expect_singular(r: Result<DdeZeroHopf, EngineError>, what: &str, has: &[&str], lacks: &[&str]) {
    match r {
        Err(EngineError::Singular(msg)) => {
            for h in has {
                assert!(msg.contains(h), "{what}: message {msg:?} lacks {h:?}");
            }
            for l in lacks {
                assert!(!msg.contains(l), "{what}: message {msg:?} has {l:?}");
            }
        }
        other => panic!("{what}: expected Singular, got {other:?}"),
    }
}

const DIM: &str = "null space of dimension 2 or more";

#[test]
fn a_semisimple_double_zero_root_is_refused_and_names_delta_zero() {
    // a' = 0 and b' = 0: Delta(0) has a null space of dimension 2 and Delta' = I, so the
    // pairing at 0 is not 0.
    let src = "a'=0\nb'=0\nu'=-v\nv'=u\nw'=-2*w-delay(w,1)\ndone\n";
    for which in both() {
        expect_singular(
            zh(src, which, &[0.0; 5], &[], 1.0),
            "double zero root",
            &[DIM, "Delta(0)"],
            &[],
        );
    }
}

#[test]
fn a_semisimple_double_hopf_root_is_refused_and_names_delta_i_omega() {
    let src = "a'=0\nu'=-v\nv'=u\ne0'=-e1\ne1'=e0\nw'=-2*w-delay(w,1)\ndone\n";
    for which in both() {
        expect_singular(
            zh(src, which, &[0.0; 6], &[], 1.0),
            "double Hopf root",
            &[DIM, "Delta(i omega)"],
            &[],
        );
    }
}

#[test]
fn the_tests_of_the_two_roots_run_in_the_spec_order() {
    // A Jordan block at zero and a double Hopf root: the pairing at 0 comes before the
    // dimension test at i omega.
    let jordan_then_double = "x'=y\ny'=0\nu'=-v\nv'=u\ne0'=-e1\ne1'=e0\nw'=-2*w-delay(w,1)\ndone\n";
    // A double zero root and a Jordan block at i omega: the dimension test at 0 comes first.
    let double_then_jordan = "a'=0\nb'=0\nj0'=-j1+j2\nj1'=j0+j3\nj2'=-j3\nj3'=j2\n\
                              w'=-2*w-delay(w,1)\ndone\n";
    for which in both() {
        expect_singular(
            zh(jordan_then_double, which.clone(), &[0.0; 7], &[], 1.0),
            "Jordan at 0, double at i omega",
            &["root 0 ", "not simple"],
            &["null space"],
        );
        expect_singular(
            zh(double_then_jordan, which, &[0.0; 7], &[], 1.0),
            "double at 0, Jordan at i omega",
            &[DIM, "Delta(0)"],
            &[],
        );
    }
}

#[test]
fn a_one_to_two_resonance_names_delta_two_i_omega() {
    let src = "par a=0, tau=1, b=1\n\
               x'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\n\
               u'=-2*v\nv'=2*u\ndone\n";
    for which in both() {
        match zh(src, which, &[0.0; 4], &[0.0, FRAC_PI_2, 1.0], 1.0) {
            Err(EngineError::Singular(msg)) => {
                assert!(msg.contains("Delta(2 i omega)"), "message {msg:?}");
            }
            other => panic!("expected Singular, got {other:?}"),
        }
    }
}
