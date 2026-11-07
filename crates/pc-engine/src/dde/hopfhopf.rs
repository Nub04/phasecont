//! The critical normal-form coefficients at a Hopf-Hopf point of a delay equilibrium (SPEC
//! section 8j).
//!
//! The formulas are those of Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1,
//! Section 6.3.1. The linear part and the forms are those of `lyap.rs` and `lyap2.rs`. There
//! are two pairs of null vectors, `(q1, p1)` of `Delta(i omega1)` and `(q2, p2)` of
//! `Delta(i omega2)`, with `omega1 > omega2`. No quadratic monomial is resonant, so all six
//! quadratic vectors are pure exponentials `exp(i sigma theta) Delta(i sigma)^-1 eta` and there
//! is no bordered system. The four cubic coefficients are the monomial coefficients of
//!
//! ```text
//! z1' = i omega1 z1 + g2100 z1 |z1|^2 + g1011 z1 |z2|^2
//! z2' = i omega2 z2 + g1110 z2 |z1|^2 + g0021 z2 |z2|^2
//! ```
//!
//! `theta = Re g1011 / Re g0021` and `delta = Re g1110 / Re g2100` are the numbers that
//! Section 8.3 of the paper and its supplement S3.2 use to sort the example point into a case.
//! The function does not compute the predictors and does not say which of the twelve
//! unfolding cases the point has.
//!
//! The scale is `|q1|_2 = |q2|_2 = 1`. The paper states no scale. Multiplying `q1` by `d1`
//! multiplies `g2100` and `g1110` by `|d1|^2`, and multiplying `q2` by `d2` multiplies `g1011`
//! and `g0021` by `|d2|^2`. So `theta` and `delta` do not depend on the scale, while the four
//! coefficients and the product do.

use super::eq::{check_params, check_state, linearize, residual};
use super::lyap::{
    C, NULL_RESIDUAL_MAX, OMEGA_MIN, PAIRING_MIN, RESONANCE_MIN, bilinear, check_regular, delta,
    embed, frobenius_real, matvec, norm2, solve, transpose, unstack,
};
use super::lyap2::{Ctx, add, conj_all};
use super::roots::LinearDde;
use crate::error::EngineError;
use crate::model::DdeModel;
use pc_linalg::{null_right, singular_values};

/// The four coefficients and the classification numbers at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfHopf {
    pub omega1: f64,
    pub omega2: f64,
    /// Each coefficient is `[re, im]`.
    pub g2100: [f64; 2],
    pub g1011: [f64; 2],
    pub g1110: [f64; 2],
    pub g0021: [f64; 2],
    /// `Re g1011 / Re g0021`, or `None` when `Re g0021 = 0`.
    pub theta: Option<f64>,
    /// `Re g1110 / Re g2100`, or `None` when `Re g2100 = 0`.
    pub delta: Option<f64>,
    /// `(Re g2100)(Re g0021)`. A negative value is the difficult type.
    pub product: f64,
    /// The sign of `product`, with 0 for exactly 0.
    pub sign: i8,
    /// `|Delta(i omega1) q1| / (omega1 + sum_j |M_j|_F)`.
    pub null_residual_1: f64,
    /// `|Delta(i omega2) q2| / (omega2 + sum_j |M_j|_F)`.
    pub null_residual_2: f64,
    /// The largest absolute entry of `f(x, ..., x, p)`.
    pub equilibrium_residual: f64,
}

/// The Hopf-Hopf coefficients of the point `(x, p, omega1, omega2)`, where `p` holds the
/// declared parameters and `omega1 > omega2`. `x`, `omega1` and `omega2` are not polished.
pub fn hopf_hopf(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega1: f64,
    omega2: f64,
) -> Result<HopfHopf, EngineError> {
    hopf_hopf_scaled(m, x, p, omega1, omega2, C::new(1.0, 0.0), C::new(1.0, 0.0))
}

/// `q1` is multiplied by `d1` and `p1` divided by it, and `q2` by `d2` and `p2` divided by it.
/// SPEC section 8j lists how the coefficients scale.
fn hopf_hopf_scaled(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega1: f64,
    omega2: f64,
    d1: C,
    d2: C,
) -> Result<HopfHopf, EngineError> {
    for (name, w) in [("omega1", omega1), ("omega2", omega2)] {
        if !(w.is_finite() && w > OMEGA_MIN) {
            return Err(EngineError::BadProblem(format!(
                "{name} is {w}; it must be finite and above {OMEGA_MIN:e}"
            )));
        }
    }
    if omega1 <= omega2 {
        return Err(EngineError::BadProblem(format!(
            "omega1 = {omega1} must be greater than omega2 = {omega2}; the frequencies are not \
             sorted for you"
        )));
    }
    check_params(m, p)?;
    check_state(m, x)?;
    let lin = linearize(m, x, p)?;
    let n = lin.dim;
    check_regular(&delta(&lin, 0.0, 0), n, "Delta(0)")?;
    let mut v = vectors(&lin, omega1, omega2)?;
    for (sigma, name) in [
        (2.0 * omega1, "Delta(2 i omega1)"),
        (2.0 * omega2, "Delta(2 i omega2)"),
        (omega1 + omega2, "Delta(i (omega1 + omega2))"),
        (omega1 - omega2, "Delta(i (omega1 - omega2))"),
    ] {
        check_regular(&delta(&lin, sigma, 0), n, name)?;
    }
    v.q1.iter_mut().for_each(|c| *c = *c * d1);
    v.p1.iter_mut().for_each(|c| *c = c.div(d1));
    v.q2.iter_mut().for_each(|c| *c = *c * d2);
    v.p2.iter_mut().for_each(|c| *c = c.div(d2));
    let mut cx = Ctx::new(m, x, p);
    let g = compute(&mut cx, &lin, &v, omega1, omega2)?;
    let Classes {
        product,
        sign,
        theta,
        delta,
    } = classify(&g);
    let pair = |c: C| [c.re, c.im];
    Ok(HopfHopf {
        omega1,
        omega2,
        g2100: pair(g.g2100),
        g1011: pair(g.g1011),
        g1110: pair(g.g1110),
        g0021: pair(g.g0021),
        theta,
        delta,
        product,
        sign,
        null_residual_1: v.null_1,
        null_residual_2: v.null_2,
        equilibrium_residual: residual(m, x, p)?,
    })
}

/// The two pairs of null vectors.
struct Vectors {
    q1: Vec<C>,
    p1: Vec<C>,
    q2: Vec<C>,
    p2: Vec<C>,
    null_1: f64,
    null_2: f64,
}

/// One unit right null vector of `Delta(i omega)`, the left one before the scaling, the relative
/// residual, the pairing `p Delta' q` and the ratio of the second smallest to the largest
/// singular value of `Delta(i omega)`, which is infinite when `n = 1`.
struct Raw {
    q: Vec<C>,
    pt: Vec<C>,
    null: f64,
    pairing: C,
    second_ratio: f64,
}

fn raw_pair(lin: &LinearDde, omega: f64, norms: f64) -> Result<Raw, EngineError> {
    let n = lin.dim;
    let d = delta(lin, omega, 0);
    let dp = delta(lin, omega, 1);
    // The real embedding repeats each complex singular value, so the second smallest complex
    // one is `s[2n - 3]`.
    let second_ratio = if n >= 2 {
        let s = singular_values(&embed(&d, n), 2 * n, 2 * n)?;
        s[2 * n - 3] / s[0]
    } else {
        f64::INFINITY
    };
    let q = unstack(&null_right(&embed(&d, n), 2 * n, 2 * n, 1)?.remove(0), n);
    let null = norm2(&matvec(&d, n, &q)) / (omega + norms);
    let pt = unstack(
        &null_right(&embed(&transpose(&d, n), n), 2 * n, 2 * n, 1)?.remove(0),
        n,
    );
    let pairing = bilinear(&pt, &matvec(&dp, n, &q));
    Ok(Raw {
        q,
        pt,
        null,
        pairing,
        second_ratio,
    })
}

/// `q1`, `p1`, `q2` and `p2` of SPEC section 8j, scaled so that `|q1|_2 = |q2|_2 = 1` and
/// `p_k Delta'(i omega_k) q_k = 1`. The errors come in the order of the SPEC: the residual of
/// `Delta(i omega1)`, the residual of `Delta(i omega2)`, and then, for `omega1` and for
/// `omega2`, a null space of dimension 2 or more and a pairing that is too small.
fn vectors(lin: &LinearDde, omega1: f64, omega2: f64) -> Result<Vectors, EngineError> {
    let norms = frobenius_real(&lin.a0)
        + lin
            .delays
            .iter()
            .map(|(_, a)| frobenius_real(a))
            .sum::<f64>();
    let r1 = raw_pair(lin, omega1, norms)?;
    let r2 = raw_pair(lin, omega2, norms)?;
    for (r, omega, name) in [(&r1, omega1, "omega1"), (&r2, omega2, "omega2")] {
        if r.null.is_nan() || r.null > NULL_RESIDUAL_MAX {
            return Err(EngineError::BadProblem(format!(
                "{name} = {omega} is not a Hopf frequency of this equation: the relative \
                 residual of Delta(i {name}) q is {:e}, above {NULL_RESIDUAL_MAX:e}",
                r.null
            )));
        }
    }
    for (r, name) in [(&r1, "omega1"), (&r2, "omega2")] {
        if r.second_ratio.is_nan() || r.second_ratio < RESONANCE_MIN {
            return Err(EngineError::Singular(format!(
                "Delta(i {name}) has a null space of dimension 2 or more (second smallest over \
                 largest singular value {:e}); the root i {name} is not simple and the \
                 coefficients would depend on the choice of null vector",
                r.second_ratio
            )));
        }
        if r.pairing.abs().is_nan() || r.pairing.abs() < PAIRING_MIN {
            return Err(EngineError::Singular(format!(
                "|p Delta'(i {name}) q| = {:e}; the root i {name} is not simple",
                r.pairing.abs()
            )));
        }
    }
    let p1 = r1.pt.iter().map(|c| c.div(r1.pairing)).collect();
    let p2 = r2.pt.iter().map(|c| c.div(r2.pairing)).collect();
    Ok(Vectors {
        q1: r1.q,
        p1,
        q2: r2.q,
        p2,
        null_1: r1.null,
        null_2: r2.null,
    })
}

struct Coefficients {
    g2100: C,
    g1011: C,
    g1110: C,
    g0021: C,
}

/// The numbers derived from the real parts of the four coefficients.
struct Classes {
    product: f64,
    sign: i8,
    theta: Option<f64>,
    delta: Option<f64>,
}

/// `product = Re g2100 Re g0021` and its sign, `theta = Re g1011 / Re g0021` and
/// `delta = Re g1110 / Re g2100`. Each ratio is `None` exactly when its own denominator is 0,
/// including `-0.0`.
fn classify(g: &Coefficients) -> Classes {
    let product = g.g2100.re * g.g0021.re;
    let sign = if product > 0.0 {
        1
    } else if product < 0.0 {
        -1
    } else {
        0
    };
    Classes {
        product,
        sign,
        theta: (g.g0021.re != 0.0).then(|| g.g1011.re / g.g0021.re),
        delta: (g.g2100.re != 0.0).then(|| g.g1110.re / g.g2100.re),
    }
}

/// `exp(i sigma theta) Delta(i sigma)^-1 eta` as a direction of the lowered model.
fn exponential(cx: &Ctx, lin: &LinearDde, sigma: f64, eta: &[C]) -> Result<Vec<C>, EngineError> {
    let h = solve(&delta(lin, sigma, 0), lin.dim, eta)?;
    Ok(cx.dir(&h, sigma))
}

/// The four coefficients for the given vectors. The caller has checked that `Delta(0)`,
/// `Delta(2 i omega1)`, `Delta(2 i omega2)`, `Delta(i (omega1 + omega2))` and
/// `Delta(i (omega1 - omega2))` are regular.
fn compute(
    cx: &mut Ctx,
    lin: &LinearDde,
    v: &Vectors,
    omega1: f64,
    omega2: f64,
) -> Result<Coefficients, EngineError> {
    let n = lin.dim;
    let f1 = cx.dir(&v.q1, omega1);
    let f2 = cx.dir(&v.q2, omega2);
    let f1b = conj_all(&f1);
    let f2b = conj_all(&f2);

    let eta = cx.b(&f1, &f1b)?;
    let h110000 = exponential(cx, lin, 0.0, &eta)?;
    let eta = cx.b(&f1, &f1)?;
    let h200000 = exponential(cx, lin, 2.0 * omega1, &eta)?;
    let eta = cx.b(&f1, &f2)?;
    let h101000 = exponential(cx, lin, omega1 + omega2, &eta)?;
    let eta = cx.b(&f2, &f2b)?;
    let h001100 = exponential(cx, lin, 0.0, &eta)?;
    let eta = cx.b(&f1, &f2b)?;
    let h100100 = exponential(cx, lin, omega1 - omega2, &eta)?;
    let h100100b = conj_all(&h100100);
    let eta = cx.b(&f2, &f2)?;
    let h002000 = exponential(cx, lin, 2.0 * omega2, &eta)?;

    let mut t = vec![C::ZERO; n];
    add(&mut t, 2.0, &cx.b(&f1, &h110000)?);
    add(&mut t, 1.0, &cx.b(&f1b, &h200000)?);
    add(&mut t, 1.0, &cx.c(&f1, &f1, &f1b)?);
    let g2100 = bilinear(&v.p1, &t).scale(0.5);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 1.0, &cx.b(&f2b, &h101000)?);
    add(&mut t, 1.0, &cx.b(&f1, &h001100)?);
    add(&mut t, 1.0, &cx.b(&f2, &h100100)?);
    add(&mut t, 1.0, &cx.c(&f1, &f2, &f2b)?);
    let g1011 = bilinear(&v.p1, &t);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 1.0, &cx.b(&f1b, &h101000)?);
    add(&mut t, 1.0, &cx.b(&f1, &h100100b)?);
    add(&mut t, 1.0, &cx.b(&f2, &h110000)?);
    add(&mut t, 1.0, &cx.c(&f1, &f1b, &f2)?);
    let g1110 = bilinear(&v.p2, &t);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 2.0, &cx.b(&f2, &h001100)?);
    add(&mut t, 1.0, &cx.b(&f2b, &h002000)?);
    add(&mut t, 1.0, &cx.c(&f2, &f2, &f2b)?);
    let g0021 = bilinear(&v.p2, &t).scale(0.5);

    Ok(Coefficients {
        g2100,
        g1011,
        g1110,
        g0021,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};

    fn build_dde_model(src: &str, ev: Evaluator) -> DdeModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, ev).expect("model builds")
    }

    fn both() -> [Evaluator; 2] {
        [Evaluator::Jit, Evaluator::Fallback]
    }

    const ACS: &str = "par zeta=-0.01622437043055295, tau=5.898023025699427\n\
                       s'=tau*y\n\
                       y'=tau*(-s-2*zeta*y)+tau*(-0.1*delay(s,1)-0.52*delay(y,1)+0.1*delay(s,1)^3)\n\
                       done\n";

    const CLOSED: &str = "x0'=-2.3*x1+x0*(0.7*(x0^2+x1^2)+(-0.4)*(x2^2+x3^2))\n\
                          x1'=2.3*x0+x1*(0.7*(x0^2+x1^2)+(-0.4)*(x2^2+x3^2))\n\
                          x2'=-1.0*x3+x2*(0.55*(x0^2+x1^2)+(-0.9)*(x2^2+x3^2))\n\
                          x3'=1.0*x2+x3*(0.55*(x0^2+x1^2)+(-0.9)*(x2^2+x3^2))\n\
                          w'=-2*w-delay(w,1)\ndone\n";

    struct Point {
        src: &'static str,
        x: Vec<f64>,
        p: Vec<f64>,
        omega1: f64,
        omega2: f64,
    }

    fn points() -> [Point; 2] {
        [
            Point {
                src: ACS,
                x: vec![0.0, 0.0],
                p: vec![-0.016_224_370_430_552_95, 5.898_023_025_699_427],
                omega1: 7.644_926_613_449_257_5,
                omega2: 4.527_486_915_496_89,
            },
            Point {
                src: CLOSED,
                x: vec![0.0; 5],
                p: vec![],
                omega1: 2.3,
                omega2: 1.0,
            },
        ]
    }

    fn near(a: C, b: C, what: &str) {
        assert!(
            (a - b).abs() <= 1e-11 * b.abs().max(1e-300),
            "{what}: {a:?} against {b:?}"
        );
    }

    #[test]
    fn the_coefficients_scale_with_the_vectors() {
        let cases = [
            (C::new(0.5, 0.0), C::new(2.0, 0.0)),
            (C::new(0.0, 2.0), C::new(0.5, 0.0)),
            (C::new(1.0, 0.0), C::expi(0.7)),
            (C::expi(1.1), C::new(0.0, 3.0)),
            (C::new(-1.0, 0.0), C::new(1.0, 0.0)),
        ];
        let from = |g: [f64; 2]| C::new(g[0], g[1]);
        for pt in points() {
            for ev in both() {
                let mut m = build_dde_model(pt.src, ev);
                let base = hopf_hopf(&mut m, &pt.x, &pt.p, pt.omega1, pt.omega2).expect("base");
                for (d1, d2) in cases {
                    let r = hopf_hopf_scaled(&mut m, &pt.x, &pt.p, pt.omega1, pt.omega2, d1, d2)
                        .expect("scaled");
                    let w1 = d1.abs() * d1.abs();
                    let w2 = d2.abs() * d2.abs();
                    for (name, b, s, weight) in [
                        ("g2100", base.g2100, r.g2100, w1),
                        ("g1011", base.g1011, r.g1011, w2),
                        ("g1110", base.g1110, r.g1110, w1),
                        ("g0021", base.g0021, r.g0021, w2),
                    ] {
                        near(
                            from(s),
                            from(b).scale(weight),
                            &format!("{name}, d1 = {d1:?}, d2 = {d2:?}"),
                        );
                    }
                    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-11 * b.abs();
                    assert!(rel(r.theta.expect("theta"), base.theta.expect("theta")));
                    assert!(rel(r.delta.expect("delta"), base.delta.expect("delta")));
                    assert!(rel(r.product, base.product * w1 * w2), "product");
                    assert_eq!(r.sign, base.sign);
                }
            }
        }
    }

    fn coefficients(re: [f64; 4]) -> Coefficients {
        // The imaginary parts are nonzero and never enter the classification.
        Coefficients {
            g2100: C::new(re[0], 0.5),
            g1011: C::new(re[1], -0.25),
            g1110: C::new(re[2], 2.0),
            g0021: C::new(re[3], -1.0),
        }
    }

    #[test]
    fn the_sign_is_the_sign_of_the_product_of_the_two_radius_terms() {
        let c = classify(&coefficients([2.0, 1.0, 1.0, 3.0]));
        assert_eq!((c.product, c.sign), (6.0, 1));
        let c = classify(&coefficients([-2.0, 1.0, 1.0, 3.0]));
        assert_eq!((c.product, c.sign), (-6.0, -1));
        let c = classify(&coefficients([-2.0, 1.0, 1.0, -3.0]));
        assert_eq!((c.product, c.sign), (6.0, 1));
        // Either factor being 0 gives sign 0, and so does a negative zero.
        for re in [
            [0.0, 1.0, 1.0, 3.0],
            [2.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 1.0, -0.0],
        ] {
            let c = classify(&coefficients(re));
            assert_eq!(c.sign, 0, "{re:?}");
            assert_eq!(c.product, 0.0, "{re:?}");
        }
    }

    #[test]
    fn theta_and_delta_each_need_only_their_own_denominator() {
        // theta = Re g1011 / Re g0021, delta = Re g1110 / Re g2100.
        let c = classify(&coefficients([2.0, 1.5, 4.0, 3.0]));
        assert_eq!((c.theta, c.delta), (Some(0.5), Some(2.0)));
        // Only Re g0021 = 0: theta is None, delta is still defined.
        let c = classify(&coefficients([2.0, 1.5, 4.0, 0.0]));
        assert_eq!((c.theta, c.delta), (None, Some(2.0)));
        // Only Re g2100 = 0: delta is None, theta is still defined.
        let c = classify(&coefficients([0.0, 1.5, 4.0, 3.0]));
        assert_eq!((c.theta, c.delta), (Some(0.5), None));
        // A negative zero is a zero.
        let c = classify(&coefficients([-0.0, 1.5, 4.0, -0.0]));
        assert_eq!((c.theta, c.delta), (None, None));
        // A zero numerator is a value, not a missing one.
        let c = classify(&coefficients([2.0, 0.0, 0.0, 3.0]));
        assert_eq!((c.theta, c.delta), (Some(0.0), Some(0.0)));
    }

    #[test]
    fn a_zero_denominator_gives_none() {
        // Re g0021 = 0 and Re g2100 = 0: two oscillators with no radial coupling.
        let src = "x0'=-2.3*x1\nx1'=2.3*x0\nx2'=-1.0*x3\nx3'=1.0*x2\nw'=-2*w-delay(w,1)\ndone\n";
        for ev in both() {
            let mut m = build_dde_model(src, ev);
            let r = hopf_hopf(&mut m, &[0.0; 5], &[], 2.3, 1.0).expect("coefficients");
            assert_eq!(r.theta, None);
            assert_eq!(r.delta, None);
            assert_eq!(r.sign, 0);
            assert_eq!(r.product, 0.0);
        }
    }

    /// The coefficient matrix of the delay-free test system, the same as in
    /// `tests/dde_hopfhopf.rs`.
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

    /// The system of `ODE_A` with its quadratic and cubic terms. `extra` is appended before
    /// `done`, so a caller can add the decoupled delay variable that makes it a delay model.
    fn ode_source(extra: &str) -> String {
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
             +0.4*x3^3-0.2*x1^2*x2+0.3*x0*x2*x3\n{extra}done\n",
            lin(0),
            lin(1),
            lin(2),
            lin(3)
        )
    }

    /// With no delay term the coefficients are those of the ODE routine of `normal.rs`, which
    /// computes them with real eigenvector solves and shifted matrices, a separate route from
    /// this function. The delay variable `w` only makes the model a delay model. The ODE
    /// routine returns Kuznetsov's G, which carries the 1/2! of the terms `z1^2 conj(z1)` and
    /// `z2^2 conj(z2)`, so G2100 and G0021 are halved to the monomial coefficients used here.
    #[test]
    fn it_agrees_with_the_ode_routine_at_a_delay_free_point() {
        use crate::model::build;
        use crate::normal::{At, DoubleHopf, double_hopf_at};
        let (omega1, omega2) = (2.3, 1.0);
        let mut map = SourceMap::new();
        let top = map.add("ode.ode", ode_source(""));
        let l = load(top, &mut map, &mut MapLoader::new());
        let mut ode = build(&l.expanded, &l.info, Evaluator::Jit).expect("ode model builds");
        let u = [0.0; 4];
        let big_g = {
            let mut at = At::new(ode.as_mut(), 4, &u, vec![0.0]);
            double_hopf_at(&mut at, omega1, omega2).expect("ode coefficients")
        };
        let half = |g: (f64, f64)| (g.0 / 2.0, g.1 / 2.0);
        let want = DoubleHopf {
            g2100: half(big_g.g2100),
            g1011: big_g.g1011,
            g1110: big_g.g1110,
            g0021: half(big_g.g0021),
        };
        let delay_src = ode_source("w'=-2*w-delay(w,1)\n");
        for ev in both() {
            let mut m = build_dde_model(&delay_src, ev);
            let r = hopf_hopf(&mut m, &[0.0; 5], &[], omega1, omega2).expect("coefficients");
            for (name, got, w) in [
                ("g2100", r.g2100, want.g2100),
                ("g1011", r.g1011, want.g1011),
                ("g1110", r.g1110, want.g1110),
                ("g0021", r.g0021, want.g0021),
            ] {
                let d = ((got[0] - w.0).powi(2) + (got[1] - w.1).powi(2)).sqrt();
                let size = w.0.hypot(w.1);
                assert!(d <= 1e-9 * size, "{name}: {got:?} against {w:?}");
            }
            let theta = want.g1011.0 / want.g0021.0;
            let delta = want.g1110.0 / want.g2100.0;
            let rel = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs();
            assert!(rel(r.theta.expect("theta"), theta), "theta");
            assert!(rel(r.delta.expect("delta"), delta), "delta");
        }
    }
}
