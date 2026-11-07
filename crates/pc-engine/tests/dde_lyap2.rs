//! The second Lyapunov coefficient at a generalized Hopf point of a delay equilibrium (SPEC
//! section 8h, acceptance tests 1 to 4, 8 and 10).
//!
//! The reference values come from an independent numpy implementation of the formulas of
//! section 8h, which uses the closed-form `Delta`, `Delta'` and `Delta''` of the models and
//! the derivative forms written out by hand. The FHN value is also the published one,
//! `l2 = -15.6733` (Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1, Sec. 7).

use pc_engine::{DdeLyapunov2, DdeModel, EngineError, Evaluator, build_dde, dde_lyapunov2};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::{FRAC_PI_2, PI};

/// `u' = -a u(t-1)(1+u) + b u^3 + c u^5`. At `a = omega = pi/2` the origin has a Hopf pair
/// for every `(b, c)`, and `l1` is linear in `b`.
const WRIGHT_Q: &str = "par a=1.5707963267948966, b=0.3887604649348081, c=0\n\
                        u'=-a*delay(u,1)*(1+u)+b*u^3+c*u^5\ndone\n";
/// The same equation after `u = x + al x^2`.
const WRIGHT_X: &str = "par a=1.5707963267948966, b=0.3887604649348081, c=0, al=0.3\n\
                        x'=(-a*(delay(x,1)+al*delay(x,1)^2)*(1+x+al*x^2)+b*(x+al*x^2)^3\
                        +c*(x+al*x^2)^5)/(1+2*al*x)\ndone\n";
/// Wright's equation with a second delay.
const TWO_DELAY: &str = "par a=1.2554671180351438, d=0.3, b=0.1825032938841381, c=0\n\
                         u'=-a*delay(u,1)*(1+u)-d*delay(u,2)*(1+u)+b*u^3+c*u^5\ndone\n";
const TWO_OMEGA: f64 = 1.3525223386535317;
/// A Hopf pair in polar form, `z' = (i w + a r + b r^2 + i e r + i g r^2) z` with
/// `r = |z|^2`, next to a decoupled delay block.
const POLAR: &str = "par w=0.7, a=0, b=0.37, e=0, g=0\n\
                     x'=-w*y+a*x*(x^2+y^2)+b*x*(x^2+y^2)^2-e*y*(x^2+y^2)-g*y*(x^2+y^2)^2\n\
                     y'=w*x+a*y*(x^2+y^2)+b*y*(x^2+y^2)^2+e*x*(x^2+y^2)+g*x*(x^2+y^2)^2\n\
                     z'=-2*z-delay(z,1)\ndone\n";
const FHN: &str = "par b=0.9, eps=0.08, c=2.0528, d=-3.2135, tau=1.7722, beta=1.9, \
                   alpha=-1.042941538463503\n\
                   u1'=-u1^3/3+(c+alpha)*u1^2+d*u1-u2+2*beta*tanh(delay(u1,tau))\n\
                   u2'=eps*(u1-b*u2)\ndone\n";
/// The Hopf frequency of the published parameters, rounded to four digits.
const FHN_OMEGA: f64 = 0.07199637889093541;
/// The generalized Hopf point on the Hopf curve: `Delta(i omega)` is singular to 1e-17 and
/// `l1 = 0` to 1e-15. The published `beta = 1.9` and `alpha = -1.0429` are this point rounded.
const FHN_GH_BETA: f64 = 1.899_972_376_314_735_6;
const FHN_GH_ALPHA: f64 = -1.042_940_901_474_694_4;
const FHN_GH_OMEGA: f64 = 0.072_000_433_494_719_7;

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

fn l2_of(
    src: &str,
    which: Evaluator,
    x: &[f64],
    p: &[f64],
    omega: f64,
) -> Result<DdeLyapunov2, EngineError> {
    dde_lyapunov2(&mut dde(src, which), x, p, omega)
}

fn ok(src: &str, which: Evaluator, x: &[f64], p: &[f64], omega: f64) -> DdeLyapunov2 {
    match l2_of(src, which, x, p, omega) {
        Ok(r) => r,
        Err(e) => panic!("lyapunov2 failed: {e}"),
    }
}

fn near(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs(),
        "{what}: {got} vs {want}"
    );
}

fn wright_p(b: f64, c: f64) -> [f64; 3] {
    [FRAC_PI_2, b, c]
}

const B_STAR: f64 = 0.3887604649348081;

#[test]
fn wright_cubic_and_quintic_match_the_numpy_oracle() {
    // c = 0: c2 = -0.15676488850186443 - 1.9754621004742934 i. c = 1 adds the quintic term.
    let cases = [
        (
            0.0,
            [-0.156_764_888_501_864_43, -1.975_462_100_474_293_4],
            -0.099_799_627_633_286_2,
        ),
        (
            1.0,
            [2.727_239_502_918_144_4, -6.505_645_604_977_195],
            1.736_214_591_539_624_8,
        ),
    ];
    for which in both() {
        for (c, c2, l2) in cases {
            let r = ok(
                WRIGHT_Q,
                which.clone(),
                &[0.0],
                &wright_p(B_STAR, c),
                FRAC_PI_2,
            );
            assert!(r.l1.abs() < 1e-12, "l1 = {}", r.l1);
            assert!(r.c1[0].abs() < 1e-12, "{:?}", r.c1);
            near(r.c1[1], -0.942_477_796_076_938, 1e-9, "Im c1");
            near(r.c2[0], c2[0], 1e-9, "Re c2");
            near(r.c2[1], c2[1], 1e-9, "Im c2");
            near(r.l2, l2, 1e-9, "l2");
            assert!((r.l2 - r.c2[0] / r.omega).abs() < 1e-15);
            assert!(r.null_residual < 1e-12);
            assert_eq!(r.equilibrium_residual, 0.0);
        }
    }
}

#[test]
fn fhn_at_the_generalized_hopf_point_matches_the_published_value() {
    let p = [
        0.9,
        0.08,
        2.0528,
        -3.2135,
        1.7722,
        FHN_GH_BETA,
        FHN_GH_ALPHA,
    ];
    for which in both() {
        let r = ok(FHN, which, &[0.0, 0.0], &p, FHN_GH_OMEGA);
        assert!(r.l1.abs() < 1e-12, "l1 = {}", r.l1);
        assert!(r.null_residual < 1e-12, "{}", r.null_residual);
        near(r.l2, -15.673_336_803_892_045, 1e-9, "l2");
        near(r.c2[0], -1.128_487_044_188_971_8, 1e-9, "Re c2");
        near(r.c2[1], 3.624_629_668_654_813, 1e-9, "Im c2");
        near(r.c1[1], -0.180_883_688_290_307_54, 1e-9, "Im c1");
        // The published value has six digits.
        near(r.l2, -15.6733, 1e-5, "published");
    }
}

#[test]
fn fhn_near_the_generalized_hopf_point_has_a_small_l1() {
    // alpha = -1.0429, the rounded value of the paper. The point is not exactly a GH point,
    // so l1 is small and nonzero, and l2 stays close to its value there.
    let p = [0.9, 0.08, 2.0528, -3.2135, 1.7722, 1.9, -1.0429];
    for which in both() {
        let r = ok(FHN, which, &[0.0, 0.0], &p, FHN_OMEGA);
        near(r.l1, 2.219_449_063_997_193_5e-4, 1e-6, "l1");
        near(r.l2, -15.674_792_445_917_703, 1e-6, "l2");
    }
}

#[test]
fn polar_form_gives_the_closed_form() {
    // e = g = 0: c1 = 0 and c2 = 4 b, so l2 = 4 b / w.
    // e = 0.3, g = 0.2: c1 = 2 i e = 0.6 i and c2 = 4 b + 4 i g. A nonzero Im c1 is what the
    // bordered solve for H2100 has to get right.
    let b = 0.37;
    for which in both() {
        let r = ok(
            POLAR,
            which.clone(),
            &[0.0; 3],
            &[0.7, 0.0, b, 0.0, 0.0],
            0.7,
        );
        assert!(r.c1[0].abs() < 1e-12 && r.c1[1].abs() < 1e-12, "{:?}", r.c1);
        near(r.c2[0], 4.0 * b, 1e-12, "Re c2");
        assert!(r.c2[1].abs() < 1e-12, "Im c2 = {}", r.c2[1]);
        near(r.l2, 4.0 * b / 0.7, 1e-12, "l2");
        let r = ok(POLAR, which, &[0.0; 3], &[0.7, 0.0, b, 0.3, 0.2], 0.7);
        assert!(r.c1[0].abs() < 1e-12, "{:?}", r.c1);
        near(r.c1[1], 0.6, 1e-12, "Im c1");
        near(r.c2[0], 4.0 * b, 1e-12, "Re c2");
        near(r.c2[1], 0.8, 1e-12, "Im c2");
        near(r.l2, 4.0 * b / 0.7, 1e-12, "l2");
    }
}

#[test]
fn a_change_of_coordinates_keeps_re_c2_and_changes_im_c2() {
    // u = x + al x^2 is a coordinate change that fixes the origin and its linearization. At a
    // GH point Re c2 is invariant, and Im c2 is not.
    let want = [
        (0.0, -0.156_764_888_501_864_43, -0.099_799_627_633_286_2),
        (1.0, 2.727_239_502_918_144_4, 1.736_214_591_539_624_8),
    ];
    let wright_im = [-1.975_462_100_474_293_4, -6.505_645_604_977_195];
    // SPEC test 4, c = 0: Im c2 at al = 0.3 and al = -0.7.
    let shifted_im = [-1.409_975_422_828_130_3, 4.621_882_472_064_265];
    for which in both() {
        for (k, (c, re_c2, l2)) in want.into_iter().enumerate() {
            let mut ims = Vec::new();
            for al in [0.3, -0.7] {
                let p = [FRAC_PI_2, B_STAR, c, al];
                let r = ok(WRIGHT_X, which.clone(), &[0.0], &p, FRAC_PI_2);
                assert!(r.l1.abs() < 1e-12, "al = {al}: l1 = {}", r.l1);
                near(r.c2[0], re_c2, 1e-9, "Re c2");
                near(r.l2, l2, 1e-9, "l2");
                ims.push(r.c2[1]);
            }
            assert!((ims[0] - ims[1]).abs() > 1.0, "{ims:?}");
            if k == 0 {
                near(ims[0], shifted_im[0], 1e-9, "Im c2 at al = 0.3");
                near(ims[1], shifted_im[1], 1e-9, "Im c2 at al = -0.7");
            }
            for im in ims {
                assert!((im - wright_im[k]).abs() > 0.1, "{im}");
            }
        }
    }
}

#[test]
fn two_delays_match_the_numpy_oracle() {
    let cases = [
        (
            0.0,
            [-0.044_536_466_250_489_86, -0.666_842_784_133_601_3],
            -0.032_928_451_514_395_67,
        ),
        (
            1.0,
            [3.297_054_800_005_045, -4.553_605_974_334_239],
            2.437_708_203_242_944,
        ),
    ];
    for which in both() {
        for (c, c2, l2) in cases {
            let p = [1.255_467_118_035_143_8, 0.3, 0.182_503_293_884_138_1, c];
            let r = ok(TWO_DELAY, which.clone(), &[0.0], &p, TWO_OMEGA);
            assert!(r.l1.abs() < 1e-12, "l1 = {}", r.l1);
            near(r.c1[1], -0.545_216_829_472_681_2, 1e-9, "Im c1");
            near(r.c2[0], c2[0], 1e-9, "Re c2");
            near(r.c2[1], c2[1], 1e-9, "Im c2");
            near(r.l2, l2, 1e-9, "l2");
        }
    }
}

fn is_bad(r: &Result<DdeLyapunov2, EngineError>) -> bool {
    matches!(r, Err(EngineError::BadProblem(_)))
}

fn is_singular(r: &Result<DdeLyapunov2, EngineError>, what: &str) -> bool {
    matches!(r, Err(EngineError::Singular(s)) if s.contains(what))
}

#[test]
fn bad_arguments_are_bad_problems() {
    let p = wright_p(B_STAR, 1.0);
    for which in both() {
        let mut m = dde(WRIGHT_Q, which);
        assert!(
            is_bad(&dde_lyapunov2(&mut m, &[0.0], &p, 1.2)),
            "omega = 1.2"
        );
        assert!(is_bad(&dde_lyapunov2(&mut m, &[0.0, 0.0], &p, FRAC_PI_2)));
        assert!(is_bad(&dde_lyapunov2(&mut m, &[], &p, FRAC_PI_2)));
        assert!(is_bad(&dde_lyapunov2(&mut m, &[0.0], &[], FRAC_PI_2)));
        assert!(is_bad(&dde_lyapunov2(
            &mut m,
            &[0.0],
            &[1.0, 2.0, 3.0, 4.0],
            FRAC_PI_2
        )));
        for w in [0.0, f64::NAN, 1e-9, -1.0, f64::INFINITY] {
            assert!(is_bad(&dde_lyapunov2(&mut m, &[0.0], &p, w)), "omega = {w}");
        }
    }
}

#[test]
fn a_zero_root_is_singular() {
    let src = "par a=0, b=1, tau=1.5707963267948966\n\
               x'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";
    for which in both() {
        let r = l2_of(src, which, &[0.0, 0.0], &[0.0, 1.0, FRAC_PI_2], 1.0);
        assert!(is_singular(&r, "Delta(0)"), "{r:?}");
    }
}

#[test]
fn a_one_to_two_resonance_is_singular() {
    let src = "x1'=-y1+x1^3\ny1'=x1\nx2'=-2*y2\ny2'=2*x2\nz'=-2*z-delay(z,1)\ndone\n";
    for which in both() {
        let r = l2_of(src, which, &[0.0; 5], &[], 1.0);
        assert!(is_singular(&r, "Delta(2 i omega)"), "{r:?}");
    }
}

#[test]
fn a_one_to_three_resonance_is_singular() {
    // The delay block has Delta_z(3 i omega) = i (3 pi/2 - 4.71238898038469), which is zero
    // up to rounding. Delta(0), Delta(i omega) and Delta(2 i omega) are regular.
    let src = "par a=1.5707963267948966, b=0.3887604649348081, c=1\n\
               u'=-a*delay(u,1)*(1+u)+b*u^3+c*u^5\n\
               z'=4.71238898038469*delay(z,1)\ndone\n";
    for which in both() {
        let r = l2_of(
            src,
            which,
            &[0.0, 0.0],
            &[FRAC_PI_2, B_STAR, 1.0],
            FRAC_PI_2,
        );
        assert!(is_singular(&r, "Delta(3 i omega)"), "{r:?}");
    }
}

#[test]
fn a_nonzero_l1_is_not_refused() {
    // At b = 0 the first Lyapunov coefficient is -0.2141..., so the point is not a GH point.
    // The function still returns the numbers; l2 is only meaningful where l1 = 0.
    for which in both() {
        let r = ok(WRIGHT_Q, which, &[0.0], &wright_p(0.0, 0.0), FRAC_PI_2);
        near(r.l1, -0.214_130_922_441_773_92, 1e-9, "l1");
        assert!(r.l2.is_finite() && r.c2.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn b_star_is_the_zero_of_l1_along_the_b_line() {
    // l1 = L10 + slope b with L10 = -0.2141... at b = 0 and slope = 6 / (pi (1 + pi^2/4)).
    let slope = 6.0 / (PI * (1.0 + PI * PI / 4.0));
    assert!((-(-0.214_130_922_441_773_92) / slope - B_STAR).abs() < 1e-15);
}
