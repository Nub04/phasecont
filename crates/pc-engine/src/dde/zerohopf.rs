//! The critical normal-form coefficients at a zero-Hopf point of a delay equilibrium (SPEC
//! section 8i).
//!
//! The formulas are those of Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1,
//! Section 6.2.1, with Lemma 4 (equation 27) for the vectors that are not pure exponentials.
//! The linear part and the forms are those of `lyap.rs` and `lyap2.rs`. There are two pairs of
//! null vectors, `(q0, p0)` of the real matrix `Delta(0)` and `(q1, p1)` of `Delta(i omega)`,
//! and the coefficients are
//!
//! ```text
//! g200 = (1/2) p0 B(phi0, phi0)    g110 = p1 B(phi0, phi1)    g011 = p0 B(phi1, conj phi1)
//! ```
//!
//! and four cubic ones, which need `H20000`, `H11000`, `H01100` (each a bordered solve, with a
//! `theta` term) and `H02000` (a pure exponential). `s_product`, `theta` and `e` are the
//! numbers of Section 6.2.1 that classify the unfolding. The function does not compute the
//! predictors and does not say which scenario the point has.
//!
//! The scale is `|q0|_2 = |q1|_2 = 1`, and the sign of `q0` is the one SPEC section 8i fixes.
//! The paper states neither, so `g200`, `g110` and `g011` match its values only up to that
//! sign.

use super::eq::{check_params, check_state, linearize, residual};
use super::lyap::{
    C, NULL_RESIDUAL_MAX, PAIRING_MIN, bilinear, check_omega_finite, check_omega_floor,
    check_regular, check_simple_kernel, delta, embed, frobenius_real, matvec, norm2, second_ratio,
    solve, transpose, unstack,
};
use super::lyap2::{Ctx, add, conj_all};
use super::roots::LinearDde;
use crate::error::EngineError;
use crate::model::DdeModel;
use pc_linalg::{null_right, singular_values};

/// The coefficients and the classification numbers at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZeroHopf {
    pub omega: f64,
    /// Each coefficient is `[re, im]`. `g200`, `g011`, `g300` and `g111` are real by theory.
    pub g200: [f64; 2],
    pub g110: [f64; 2],
    pub g011: [f64; 2],
    pub g300: [f64; 2],
    pub g111: [f64; 2],
    pub g210: [f64; 2],
    pub g021: [f64; 2],
    /// `Re (g200 g011)`.
    pub s_product: f64,
    /// The sign of `s_product`, with 0 for exactly 0.
    pub s: i8,
    /// `Re g110 / Re g200`, or `None` when `Re g200 = 0`.
    pub theta: Option<f64>,
    /// The number `e` of Section 6.2.1, or `None` when it is not finite.
    pub e: Option<f64>,
    /// `|Delta(0) q0| / sum_j |M_j|_F`.
    pub null_residual_zero: f64,
    /// `|Delta(i omega) q1| / (omega + sum_j |M_j|_F)`.
    pub null_residual_hopf: f64,
    /// The largest absolute entry of `f(x, ..., x, p)`.
    pub equilibrium_residual: f64,
}

/// The zero-Hopf coefficients of the point `(x, p, omega)`, where `p` holds the declared
/// parameters. `x` and `omega` are not polished.
pub fn zero_hopf(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
) -> Result<ZeroHopf, EngineError> {
    zero_hopf_scaled(m, x, p, omega, 1.0, C::new(1.0, 0.0))
}

/// `q0` is multiplied by `s0` (real) and `p0` divided by it, and `q1` by `s1` and `p1` divided
/// by it. SPEC section 8i lists how the coefficients scale.
fn zero_hopf_scaled(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
    s0: f64,
    s1: C,
) -> Result<ZeroHopf, EngineError> {
    check_omega_finite(omega, "omega")?;
    check_params(m, p)?;
    check_state(m, x)?;
    let lin = linearize(m, x, p)?;
    check_omega_floor(&lin, omega, "omega")?;
    let n = lin.dim;
    let mut v = vectors(&lin, omega)?;
    check_regular(&delta(&lin, 2.0 * omega, 0), n, "Delta(2 i omega)")?;
    v.q0.iter_mut().for_each(|c| *c = c.scale(s0));
    v.p0.iter_mut().for_each(|c| *c = c.scale(1.0 / s0));
    v.q1.iter_mut().for_each(|c| *c = *c * s1);
    v.p1.iter_mut().for_each(|c| *c = c.div(s1));
    let mut cx = Ctx::new(m, x, p);
    let g = compute(&mut cx, &lin, &v, omega)?;
    let product = (g.g200 * g.g011).re;
    let s = if product > 0.0 {
        1
    } else if product < 0.0 {
        -1
    } else {
        0
    };
    let theta = (g.g200.re != 0.0).then(|| g.g110.re / g.g200.re);
    let e = {
        let inner = C::new(g.g021.re, 0.0).div(g.g011) - g.g300.div(g.g200).scale(1.5)
            + g.g111.div(g.g011.scale(2.0));
        let t = g.g210 + g.g110 * inner - (g.g021 * g.g200).div(g.g011);
        t.re.is_finite().then_some(t.re)
    };
    let pair = |c: C| [c.re, c.im];
    Ok(ZeroHopf {
        omega,
        g200: pair(g.g200),
        g110: pair(g.g110),
        g011: pair(g.g011),
        g300: pair(g.g300),
        g111: pair(g.g111),
        g210: pair(g.g210),
        g021: pair(g.g021),
        s_product: product,
        s,
        theta,
        e,
        null_residual_zero: v.null_zero,
        null_residual_hopf: v.null_hopf,
        equilibrium_residual: residual(m, x, p)?,
    })
}

/// The two pairs of null vectors. `q0` and `p0` are real and are stored with zero imaginary
/// parts.
struct Vectors {
    q0: Vec<C>,
    p0: Vec<C>,
    q1: Vec<C>,
    p1: Vec<C>,
    null_zero: f64,
    null_hopf: f64,
}

/// `q0`, `p0`, `q1` and `p1` of SPEC section 8i, scaled so that `|q0|_2 = |q1|_2 = 1` and
/// `p0 Delta'(0) q0 = p1 Delta'(i omega) q1 = 1`. The errors come in the order of the SPEC:
/// the residual of `Delta(0)`, the residual of `Delta(i omega)`, then for the root 0 and for the
/// root `i omega` a null space of dimension 2 or more and a pairing that is too small.
fn vectors(lin: &LinearDde, omega: f64) -> Result<Vectors, EngineError> {
    let n = lin.dim;
    let norms = frobenius_real(&lin.a0)
        + lin
            .delays
            .iter()
            .map(|(_, a)| frobenius_real(a))
            .sum::<f64>();

    // The real pair, from the smallest singular vectors of the real matrix Delta(0).
    let d0 = delta(lin, 0.0, 0);
    let d0p = delta(lin, 0.0, 1);
    let mut col = vec![0.0; n * n];
    let mut col_t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            col[i + n * j] = d0[i * n + j].re;
            col_t[j + n * i] = d0[i * n + j].re;
        }
    }
    // The real matrix repeats nothing, so the second smallest singular value is `s[n - 2]`.
    let second_zero = if n >= 2 {
        let s = singular_values(&col, n, n)?;
        s[n - 2] / s[0]
    } else {
        f64::INFINITY
    };
    let mut q0r = null_right(&col, n, n, 1)?.remove(0);
    let threshold = 1.0 / (2.0 * (n as f64).sqrt());
    if q0r
        .iter()
        .find(|v| v.abs() >= threshold)
        .is_some_and(|v| *v < 0.0)
    {
        q0r.iter_mut().for_each(|v| *v = -*v);
    }
    let q0: Vec<C> = q0r.iter().map(|v| C::new(*v, 0.0)).collect();
    let null_zero = norm2(&matvec(&d0, n, &q0)) / norms;
    let pt0: Vec<C> = null_right(&col_t, n, n, 1)?
        .remove(0)
        .iter()
        .map(|v| C::new(*v, 0.0))
        .collect();
    let s0 = bilinear(&pt0, &matvec(&d0p, n, &q0)).re;

    // The complex pair, as in `lyap::pair`.
    let d1 = delta(lin, omega, 0);
    let d1p = delta(lin, omega, 1);
    let second_hopf = second_ratio(&d1, n)?;
    let q1 = unstack(&null_right(&embed(&d1, n), 2 * n, 2 * n, 1)?.remove(0), n);
    let null_hopf = norm2(&matvec(&d1, n, &q1)) / (omega + norms);
    let pt1 = unstack(
        &null_right(&embed(&transpose(&d1, n), n), 2 * n, 2 * n, 1)?.remove(0),
        n,
    );
    let s1 = bilinear(&pt1, &matvec(&d1p, n, &q1));

    if null_zero.is_nan() || null_zero > NULL_RESIDUAL_MAX {
        return Err(EngineError::BadProblem(format!(
            "this point has no zero root: the relative residual of Delta(0) q0 is \
             {null_zero:e}, above {NULL_RESIDUAL_MAX:e}"
        )));
    }
    if null_hopf.is_nan() || null_hopf > NULL_RESIDUAL_MAX {
        return Err(EngineError::BadProblem(format!(
            "omega = {omega} is not a Hopf frequency of this equation: the relative residual \
             of Delta(i omega) q1 is {null_hopf:e}, above {NULL_RESIDUAL_MAX:e}"
        )));
    }
    check_simple_kernel(second_zero, "Delta(0)", "0")?;
    if s0.is_nan() || s0.abs() < PAIRING_MIN {
        return Err(EngineError::Singular(format!(
            "|p0 Delta'(0) q0| = {:e}; the root 0 is not simple",
            s0.abs()
        )));
    }
    check_simple_kernel(second_hopf, "Delta(i omega)", "i omega")?;
    if s1.abs().is_nan() || s1.abs() < PAIRING_MIN {
        return Err(EngineError::Singular(format!(
            "|p1 Delta'(i omega) q1| = {:e}; the root i omega is not simple",
            s1.abs()
        )));
    }
    Ok(Vectors {
        q0,
        p0: pt0.iter().map(|c| c.scale(1.0 / s0)).collect(),
        q1,
        p1: pt1.iter().map(|c| c.div(s1)).collect(),
        null_zero,
        null_hopf,
    })
}

/// One solution of Lemma 4 for the root `i sigma`:
/// `H(theta) = exp(i sigma theta)(v0 - kappa theta q)`.
struct Bordered {
    sigma: f64,
    q: Vec<C>,
    #[cfg(test)]
    pv: Vec<C>,
    kappa: C,
    v0: Vec<C>,
    #[cfg(test)]
    gamma: C,
}

impl Bordered {
    /// The direction of `H` in the variables of the lowered model.
    fn dir(&self, cx: &Ctx) -> Vec<C> {
        cx.poly(&self.v0, self.sigma, self.kappa.scale(-1.0), &self.q)
    }
}

/// The bordered solve of SPEC section 8i, for a root `i sigma` with the pair `(q, pv)`, the
/// vector `eta` and `kappa = -pv eta`:
///
/// ```text
/// [[Delta, p^H], [q^H, 0]] (xi, s) = (eta + kappa Delta' q, 0)
/// gamma = -p Delta' xi + (1/2) kappa p Delta'' q,   v0 = xi + gamma q
/// ```
fn bordered(
    lin: &LinearDde,
    sigma: f64,
    q: &[C],
    pv: &[C],
    eta: &[C],
    kappa: C,
) -> Result<Bordered, EngineError> {
    let n = lin.dim;
    let d = delta(lin, sigma, 0);
    let d1 = delta(lin, sigma, 1);
    let d2 = delta(lin, sigma, 2);
    let d1q = matvec(&d1, n, q);
    let k = n + 1;
    let mut mat = vec![C::ZERO; k * k];
    for i in 0..n {
        for j in 0..n {
            mat[i * k + j] = d[i * n + j];
        }
        mat[i * k + n] = pv[i].conj();
        mat[n * k + i] = q[i].conj();
    }
    let mut rhs = vec![C::ZERO; k];
    for i in 0..n {
        rhs[i] = eta[i] + kappa * d1q[i];
    }
    let sol = solve(&mat, k, &rhs)?;
    let xi = &sol[..n];
    let pd1xi = bilinear(pv, &matvec(&d1, n, xi));
    let pd2q = bilinear(pv, &matvec(&d2, n, q));
    let gamma = (kappa * pd2q).scale(0.5) - pd1xi;
    let v0 = xi.iter().zip(q).map(|(a, b)| *a + gamma * *b).collect();
    Ok(Bordered {
        sigma,
        q: q.to_vec(),
        #[cfg(test)]
        pv: pv.to_vec(),
        kappa,
        v0,
        #[cfg(test)]
        gamma,
    })
}

struct Coefficients {
    g200: C,
    g110: C,
    g011: C,
    g300: C,
    g111: C,
    g210: C,
    g021: C,
    #[cfg(test)]
    h20000: Bordered,
    #[cfg(test)]
    h11000: Bordered,
    #[cfg(test)]
    h01100: Bordered,
}

/// The seven coefficients for the given vectors. The caller has checked that `Delta(2 i omega)`
/// is regular.
fn compute(
    cx: &mut Ctx,
    lin: &LinearDde,
    v: &Vectors,
    omega: f64,
) -> Result<Coefficients, EngineError> {
    let n = lin.dim;
    let phi0 = cx.dir(&v.q0, 0.0);
    let phi1 = cx.dir(&v.q1, omega);
    let phi1b = conj_all(&phi1);
    let b00 = cx.b(&phi0, &phi0)?;
    let b01 = cx.b(&phi0, &phi1)?;
    let b11b = cx.b(&phi1, &phi1b)?;
    let g200 = bilinear(&v.p0, &b00).scale(0.5);
    let g110 = bilinear(&v.p1, &b01);
    let g011 = bilinear(&v.p0, &b11b);

    let h20000 = bordered(lin, 0.0, &v.q0, &v.p0, &b00, g200.scale(-2.0))?;
    let h11000 = bordered(lin, omega, &v.q1, &v.p1, &b01, g110.scale(-1.0))?;
    let h01100 = bordered(lin, 0.0, &v.q0, &v.p0, &b11b, g011.scale(-1.0))?;
    let d20000 = h20000.dir(cx);
    let d11000 = h11000.dir(cx);
    let d10100 = conj_all(&d11000);
    let d01100 = h01100.dir(cx);
    let h02000 = solve(&delta(lin, 2.0 * omega, 0), n, &cx.b(&phi1, &phi1)?)?;
    let d02000 = cx.dir(&h02000, 2.0 * omega);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 3.0, &cx.b(&phi0, &d20000)?);
    add(&mut t, 1.0, &cx.c(&phi0, &phi0, &phi0)?);
    let g300 = bilinear(&v.p0, &t).scale(1.0 / 6.0);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 1.0, &cx.b(&phi0, &d01100)?);
    add(&mut t, 1.0, &cx.b(&phi1, &d10100)?);
    add(&mut t, 1.0, &cx.b(&phi1b, &d11000)?);
    add(&mut t, 1.0, &cx.c(&phi0, &phi1, &phi1b)?);
    let g111 = bilinear(&v.p0, &t);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 2.0, &cx.b(&phi0, &d11000)?);
    add(&mut t, 1.0, &cx.b(&phi1, &d20000)?);
    add(&mut t, 1.0, &cx.c(&phi0, &phi0, &phi1)?);
    let g210 = bilinear(&v.p1, &t).scale(0.5);

    let mut t = vec![C::ZERO; n];
    add(&mut t, 2.0, &cx.b(&phi1, &d01100)?);
    add(&mut t, 1.0, &cx.b(&phi1b, &d02000)?);
    add(&mut t, 1.0, &cx.c(&phi1, &phi1, &phi1b)?);
    let g021 = bilinear(&v.p1, &t).scale(0.5);

    Ok(Coefficients {
        g200,
        g110,
        g011,
        g300,
        g111,
        g210,
        g021,
        #[cfg(test)]
        h20000,
        #[cfg(test)]
        h11000,
        #[cfg(test)]
        h01100,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};
    use std::f64::consts::PI;

    fn build(src: &str, ev: Evaluator) -> DdeModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, ev).expect("model builds")
    }

    fn both() -> [Evaluator; 2] {
        [Evaluator::Jit, Evaluator::Fallback]
    }

    const HR: &str = "par a=1, b=3, c=1, d=5, chi=-1.6, r=0.001, sl=-0.57452592, iapp=0, tau=1\n\
                      x'=y-a*x^3+b*delay(x,tau)^2-c*z+iapp\n\
                      y'=c-d*x^2-y\n\
                      z'=r*(sl*(x-chi)-z)\ndone\n";

    struct Point {
        p: [f64; 9],
        x: [f64; 3],
        omega: f64,
    }

    fn points() -> [Point; 2] {
        [
            Point {
                p: [
                    1.0,
                    3.0,
                    1.0,
                    5.0,
                    -1.6,
                    0.001,
                    -0.574_525_92,
                    -1.957_934_372_224_000_1,
                    5.768_830_919_756_731,
                ],
                x: [
                    0.130_799_999_999_999_94,
                    0.914_456_800_000_000_1,
                    -0.994_389_462_336_000_1,
                ],
                omega: 1.007_916_143_551_579_8,
            },
            Point {
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
            },
        ]
    }

    /// Nodes and weights of the `m`-point Gauss-Legendre rule on `[-1, 1]`.
    fn gauss_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
        let mf = m as f64;
        let mut nodes = vec![0.0; m];
        let mut weights = vec![0.0; m];
        for i in 0..m {
            let mut z = (PI * (i as f64 + 0.75) / (mf + 0.5)).cos();
            let legendre = |z: f64| {
                let (mut p0, mut p1) = (1.0, z);
                for k in 2..=m {
                    let kf = k as f64;
                    let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                    p0 = p1;
                    p1 = p2;
                }
                (p1, mf * (z * p1 - p0) / (z * z - 1.0))
            };
            for _ in 0..100 {
                let (p, dp) = legendre(z);
                let dz = p / dp;
                z -= dz;
                if dz.abs() < 1e-16 {
                    break;
                }
            }
            let (_, dp) = legendre(z);
            nodes[i] = z;
            weights[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        }
        (nodes, weights)
    }

    #[test]
    fn the_quadrature_rule_integrates_a_polynomial_exactly() {
        let (x, w) = gauss_legendre(30);
        let sum: f64 = w.iter().sum();
        assert!((sum - 2.0).abs() < 1e-14);
        let m4: f64 = x.iter().zip(&w).map(|(a, b)| a.powi(4) * b).sum();
        assert!((m4 - 0.4).abs() < 1e-14);
    }

    /// The Hale pairing `<psi, H>` of `psi(s) = p exp(-i sigma s)` with
    /// `H(theta) = exp(i sigma theta)(v0 - kappa theta q)`, the integrals over `[-tau_j, 0]`
    /// done by a 30-point Gauss-Legendre rule:
    /// `psi(0) H(0) + sum_j int psi(xi + tau_j) M_j H(xi) d xi`.
    fn hale(lin: &LinearDde, b: &Bordered, v0: &[C]) -> C {
        let n = lin.dim;
        let (nodes, weights) = gauss_legendre(30);
        let mut total = bilinear(&b.pv, v0);
        for (tau, aj) in &lin.delays {
            let tau = *tau;
            let mut sum = C::ZERO;
            for (x, w) in nodes.iter().zip(&weights) {
                let t = tau * (x - 1.0) / 2.0;
                let lead = C::expi(b.sigma * t);
                let psi = C::expi(-b.sigma * (t + tau));
                let phi: Vec<C> = (0..n)
                    .map(|i| lead * (v0[i] + (b.kappa * b.q[i]).scale(-t)))
                    .collect();
                let mut aphi = vec![C::ZERO; n];
                for i in 0..n {
                    for j in 0..n {
                        aphi[i] = aphi[i] + phi[j].scale(aj[i * n + j]);
                    }
                }
                sum = sum + (psi * bilinear(&b.pv, &aphi)).scale(w * tau / 2.0);
            }
            total = total + sum;
        }
        total
    }

    fn dist(a: C, b: C) -> f64 {
        (a - b).abs()
    }

    #[test]
    fn the_bordered_vectors_are_orthogonal_to_the_adjoint_eigenfunction_and_gamma_matters() {
        let mut large = 0;
        for pt in points() {
            for ev in both() {
                let mut m = build(HR, ev);
                let lin = linearize(&mut m, &pt.x, &pt.p).expect("linearize");
                let v = vectors(&lin, pt.omega).expect("vectors");
                let mut cx = Ctx::new(&mut m, &pt.x, &pt.p);
                let c = compute(&mut cx, &lin, &v, pt.omega).expect("coefficients");
                for b in [&c.h20000, &c.h11000, &c.h01100] {
                    let n = lin.dim;
                    let zero = hale(&lin, b, &b.v0);
                    assert!(zero.abs() < 1e-10, "pairing {zero:?}");
                    // Closed form: p Delta' v0 - (1/2) kappa p Delta'' q.
                    let d1 = delta(&lin, b.sigma, 1);
                    let d2 = delta(&lin, b.sigma, 2);
                    let closed = bilinear(&b.pv, &matvec(&d1, n, &b.v0))
                        - (b.kappa * bilinear(&b.pv, &matvec(&d2, n, &b.q))).scale(0.5);
                    assert!(dist(closed, zero) < 1e-10, "{closed:?} {zero:?}");
                    // Without gamma the pairing is -gamma, since p Delta' q = 1.
                    let xi: Vec<C> =
                        b.v0.iter()
                            .zip(&b.q)
                            .map(|(v0, q)| *v0 - b.gamma * *q)
                            .collect();
                    let control = hale(&lin, b, &xi);
                    assert!(
                        (control.abs() - b.gamma.abs()).abs() < 1e-10,
                        "{control:?} vs {:?}",
                        b.gamma
                    );
                    if b.gamma.abs() > 1e-3 {
                        large += 1;
                    }
                }
            }
        }
        // Two points, two evaluators and three vectors each.
        assert_eq!(large, 12, "gamma is above 1e-3 in {large} of 12 cases");
    }

    fn near(a: C, b: C, what: &str) {
        assert!(
            dist(a, b) <= 1e-11 * b.abs().max(1e-300),
            "{what}: {a:?} against {b:?}"
        );
    }

    #[test]
    fn the_coefficients_scale_with_the_vectors() {
        let cases = [
            (0.5, C::new(2.0, 0.0)),
            (2.0, C::new(0.0, 0.5)),
            (-1.0, C::new(1.0, 0.0)),
            (1.0, C::expi(0.7)),
        ];
        let from = |g: [f64; 2]| C::new(g[0], g[1]);
        for pt in points() {
            for ev in both() {
                let mut m = build(HR, ev);
                let base = zero_hopf(&mut m, &pt.x, &pt.p, pt.omega).expect("base");
                for (c, d) in cases {
                    let r = zero_hopf_scaled(&mut m, &pt.x, &pt.p, pt.omega, c, d).expect("scaled");
                    let d2 = d.abs() * d.abs();
                    let w = [
                        ("g200", base.g200, r.g200, c),
                        ("g110", base.g110, r.g110, c),
                        ("g011", base.g011, r.g011, d2 / c),
                        ("g300", base.g300, r.g300, c * c),
                        ("g111", base.g111, r.g111, d2),
                        ("g210", base.g210, r.g210, c * c),
                        ("g021", base.g021, r.g021, d2),
                    ];
                    for (name, b, s, weight) in w {
                        near(from(s), from(b).scale(weight), &format!("{name}, c = {c}"));
                    }
                    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-11 * b.abs();
                    assert!(rel(r.s_product, base.s_product * d2), "s_product");
                    assert!(
                        rel(r.theta.expect("theta"), base.theta.expect("theta")),
                        "theta"
                    );
                    assert!(
                        rel(r.e.expect("e"), base.e.expect("e") * c * c),
                        "e at c = {c}"
                    );
                    assert_eq!(r.s, base.s);
                }
            }
        }
    }
}
