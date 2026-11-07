//! The second Lyapunov coefficient at a generalized Hopf point of a delay equilibrium (SPEC
//! section 8h).
//!
//! The formulas are those of Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1,
//! Sections 2.5, 6.1.1 and 7, with Lemma 4 (equation 27) for `H2100`. The linear part and
//! the forms are those of `lyap.rs`. This file adds three things:
//!
//! - `Delta''` and `Delta(3 i omega)`, for `H2100` and `H3000`;
//! - directions of the form `exp(s theta)(v + c theta w)`, which `H2100` and `H3100` are,
//!   with the explicit delayed value `exp(-s tau)(v - c tau w)` in each hidden slot;
//! - the forms of order 4 and 5.
//!
//! `H2100` is solved with the bordered system `[[Delta(i omega), p^H], [q^H, 0]]`, which is
//! nonsingular for a simple root even where `p q = 0`. The vector `v0` does not depend on
//! the border, because the coefficient `gamma` makes the Hale pairing of `H2100` with the
//! adjoint eigenfunction zero.
//!
//! `c2`, and so `l2 = Re c2 / omega`, is a normal-form coefficient only where `Re c1 = 0`.
//! The function does not refuse any other point.

use super::eq::{check_params, check_state, full_params, linearize, residual, time_of};
use super::lyap::{
    C, OMEGA_MIN, Q_NORM, bilinear, check_regular, delta, direction, form, matvec, pair, solve,
};
use super::roots::LinearDde;
use crate::error::EngineError;
use crate::model::DdeModel;

/// The coefficients at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lyapunov2 {
    pub omega: f64,
    /// `[Re c1, Im c1]` at `|q| = Q_NORM`.
    pub c1: [f64; 2],
    /// `[Re c2, Im c2]` at `|q| = Q_NORM`. The imaginary part has no invariant meaning.
    pub c2: [f64; 2],
    /// `Re c1 / omega`. The point is a generalized Hopf point where this is zero.
    pub l1: f64,
    /// `Re c2 / omega`, meaningful only where `l1` is zero.
    pub l2: f64,
    /// As in `Lyapunov`.
    pub null_residual: f64,
    /// As in `Lyapunov`.
    pub equilibrium_residual: f64,
}

/// The first and second Lyapunov coefficients of the point `(x, p, omega)`, where `p` holds
/// the declared parameters. `x` and `omega` are not polished.
pub fn lyapunov2(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
) -> Result<Lyapunov2, EngineError> {
    lyapunov2_scaled(m, x, p, omega, C::new(Q_NORM, 0.0))
}

/// The vectors `q` and `p` are multiplied by `s` and divided by `s`. A real `s` scales `c2`
/// with `s^4`, and a phase leaves it unchanged.
fn lyapunov2_scaled(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
    s: C,
) -> Result<Lyapunov2, EngineError> {
    if !(omega.is_finite() && omega > OMEGA_MIN) {
        return Err(EngineError::BadProblem(format!(
            "omega is {omega}; it must be finite and above {OMEGA_MIN:e}"
        )));
    }
    check_params(m, p)?;
    check_state(m, x)?;
    let lin = linearize(m, x, p)?;
    let n = lin.dim;
    let pr = pair(&lin, omega)?;
    let q: Vec<C> = pr.q.iter().map(|c| *c * s).collect();
    let pv: Vec<C> = pr.p.iter().map(|c| c.div(s)).collect();
    let mut cx = Ctx::new(m, x, p);
    let f = first_order(&mut cx, &lin, &q, &pv, omega)?;
    let c1 = f.c1;
    let phi = &f.phi;
    let phib = conj_all(phi);
    let d2000 = &f.d2000;
    let d2000b = conj_all(d2000);
    let d1100 = &f.d1100;
    let d3000 = &f.d3000;

    let (v0, _gamma) = h2100_vectors(&lin, &f.f21, c1, &q, &pv, omega)?;
    let d2100 = cx.poly(&v0, omega, c1.scale(2.0), &q);
    let d2100b = conj_all(&d2100);

    // H2200, a constant.
    let mut r = vec![C::ZERO; n];
    add(&mut r, 2.0, &cx.b(&phib, &d2100)?);
    add(&mut r, 2.0, &cx.b(phi, &d2100b)?);
    add(&mut r, 1.0, &cx.b(&d2000b, d2000)?);
    add(&mut r, 2.0, &cx.b(d1100, d1100)?);
    add(&mut r, 1.0, &cx.c(phi, phi, &d2000b)?);
    add(&mut r, 4.0, &cx.c(phi, &phib, d1100)?);
    add(&mut r, 1.0, &cx.c(&phib, &phib, d2000)?);
    add(&mut r, 1.0, &cx.d(phi, phi, &phib, &phib)?);
    let h2200 = solve(&delta(&lin, 0.0, 0), n, &r)?;
    let d2200 = cx.dir(&h2200, 0.0);

    // H3100 = exp(2 i omega theta)(v31 + 6 c1 theta h2000).
    let mut f31 = vec![C::ZERO; n];
    add(&mut f31, 1.0, &cx.b(&phib, d3000)?);
    add(&mut f31, 3.0, &cx.b(phi, &d2100)?);
    add(&mut f31, 3.0, &cx.b(d1100, d2000)?);
    add(&mut f31, 3.0, &cx.c(phi, &phib, d2000)?);
    add(&mut f31, 3.0, &cx.c(phi, phi, d1100)?);
    add(&mut f31, 1.0, &cx.d(phi, phi, phi, &phib)?);
    let h2000 = &d2000[..n];
    let shift = matvec(&delta(&lin, 2.0 * omega, 1), n, h2000);
    let six_c1 = c1.scale(6.0);
    let rhs: Vec<C> = (0..n).map(|i| f31[i] - six_c1 * shift[i]).collect();
    let v31 = solve(&delta(&lin, 2.0 * omega, 0), n, &rhs)?;
    let d3100 = cx.poly(&v31, 2.0 * omega, six_c1, h2000);

    // c2, in the order of SPEC section 8h.
    let mut t = vec![C::ZERO; n];
    add(&mut t, 2.0, &cx.b(&phib, &d3100)?);
    add(&mut t, 1.0, &cx.b(&d2000b, d3000)?);
    add(&mut t, 3.0, &cx.b(phi, &d2200)?);
    add(&mut t, 6.0, &cx.b(d1100, &d2100)?);
    add(&mut t, 3.0, &cx.b(&d2100b, d2000)?);
    add(&mut t, 1.0, &cx.c(&phib, &phib, d3000)?);
    add(&mut t, 6.0, &cx.c(phi, &phib, &d2100)?);
    add(&mut t, 6.0, &cx.c(&phib, d1100, d2000)?);
    add(&mut t, 3.0, &cx.c(phi, &d2000b, d2000)?);
    add(&mut t, 3.0, &cx.c(phi, phi, &d2100b)?);
    add(&mut t, 6.0, &cx.c(phi, d1100, d1100)?);
    add(&mut t, 3.0, &cx.d(phi, &phib, &phib, d2000)?);
    add(&mut t, 6.0, &cx.d(phi, phi, &phib, d1100)?);
    add(&mut t, 1.0, &cx.d(phi, phi, phi, &d2000b)?);
    add(&mut t, 1.0, &cx.e(phi, phi, phi, &phib, &phib)?);
    let c2 = bilinear(&pv, &t).scale(1.0 / 12.0);

    Ok(Lyapunov2 {
        omega,
        c1: [c1.re, c1.im],
        c2: [c2.re, c2.im],
        l1: c1.re / omega,
        l2: c2.re / omega,
        null_residual: pr.null_residual,
        equilibrium_residual: residual(m, x, p)?,
    })
}

/// The model, the point and the delay times, for evaluating the forms.
struct Ctx<'a> {
    m: &'a mut DdeModel,
    x: &'a [f64],
    full: Vec<f64>,
    times: Vec<f64>,
}

impl<'a> Ctx<'a> {
    fn new(m: &'a mut DdeModel, x: &'a [f64], p: &[f64]) -> Ctx<'a> {
        let full = full_params(m, x, p, 0.0);
        let times = m.delays.iter().map(|s| time_of(s, p)).collect();
        Ctx { m, x, full, times }
    }

    /// The direction of `exp(i sigma theta) h`.
    fn dir(&self, h: &[C], sigma: f64) -> Vec<C> {
        direction(&*self.m, &self.times, h, sigma)
    }

    /// The direction of `exp(i sigma theta)(v + c theta w)`.
    fn poly(&self, v: &[C], sigma: f64, c: C, w: &[C]) -> Vec<C> {
        direction_poly(&*self.m, &self.times, v, sigma, c, w)
    }

    fn b(&mut self, a: &[C], b: &[C]) -> Result<Vec<C>, EngineError> {
        form(self.m, self.x, &self.full, &[a, b])
    }

    fn c(&mut self, a: &[C], b: &[C], c: &[C]) -> Result<Vec<C>, EngineError> {
        form(self.m, self.x, &self.full, &[a, b, c])
    }

    fn d(&mut self, a: &[C], b: &[C], c: &[C], d: &[C]) -> Result<Vec<C>, EngineError> {
        form(self.m, self.x, &self.full, &[a, b, c, d])
    }

    fn e(&mut self, a: &[C], b: &[C], c: &[C], d: &[C], e: &[C]) -> Result<Vec<C>, EngineError> {
        form(self.m, self.x, &self.full, &[a, b, c, d, e])
    }
}

/// The direction of `exp(i sigma theta)(v + c theta w)` in the variables of the lowered
/// model. The value at `theta = 0` is `v`, and the value at `theta = -tau` is
/// `exp(-i sigma tau)(v - c tau w)`, which goes into the hidden slot of each delay.
fn direction_poly(m: &DdeModel, times: &[f64], v: &[C], sigma: f64, c: C, w: &[C]) -> Vec<C> {
    let n = v.len();
    let mut d = vec![C::ZERO; n + m.eval.n_p()];
    d[..n].copy_from_slice(v);
    for (k, s) in m.delays.iter().enumerate() {
        let lagged = v[s.state] - (c * w[s.state]).scale(times[k]);
        d[n + m.n_user_params + k] = lagged * C::expi(-sigma * times[k]);
    }
    d
}

fn conj_all(v: &[C]) -> Vec<C> {
    v.iter().map(|c| c.conj()).collect()
}

/// `acc += coef * v`.
fn add(acc: &mut [C], coef: f64, v: &[C]) {
    for (a, x) in acc.iter_mut().zip(v) {
        *a = *a + x.scale(coef);
    }
}

/// The directions that come from the exponential eigenfunction alone.
struct First {
    phi: Vec<C>,
    d2000: Vec<C>,
    d1100: Vec<C>,
    d3000: Vec<C>,
    f21: Vec<C>,
    c1: C,
}

/// `H2000`, `H1100`, `H3000`, `F21` and `c1`, after the checks of `Delta(0)`,
/// `Delta(2 i omega)` and `Delta(3 i omega)`, in that order.
fn first_order(
    cx: &mut Ctx,
    lin: &LinearDde,
    q: &[C],
    pv: &[C],
    omega: f64,
) -> Result<First, EngineError> {
    let n = lin.dim;
    let d0 = delta(lin, 0.0, 0);
    let d2 = delta(lin, 2.0 * omega, 0);
    let d3 = delta(lin, 3.0 * omega, 0);
    check_regular(&d0, n, "Delta(0)")?;
    check_regular(&d2, n, "Delta(2 i omega)")?;
    check_regular(&d3, n, "Delta(3 i omega)")?;
    let phi = cx.dir(q, omega);
    let phib = conj_all(&phi);
    let h2000 = solve(&d2, n, &cx.b(&phi, &phi)?)?;
    let h1100 = solve(&d0, n, &cx.b(&phi, &phib)?)?;
    let d2000 = cx.dir(&h2000, 2.0 * omega);
    let d1100 = cx.dir(&h1100, 0.0);
    let mut r3 = vec![C::ZERO; n];
    add(&mut r3, 3.0, &cx.b(&phi, &d2000)?);
    add(&mut r3, 1.0, &cx.c(&phi, &phi, &phi)?);
    let h3000 = solve(&d3, n, &r3)?;
    let d3000 = cx.dir(&h3000, 3.0 * omega);
    let mut f21 = vec![C::ZERO; n];
    add(&mut f21, 1.0, &cx.b(&phib, &d2000)?);
    add(&mut f21, 2.0, &cx.b(&phi, &d1100)?);
    add(&mut f21, 1.0, &cx.c(&phi, &phi, &phib)?);
    let c1 = bilinear(pv, &f21).scale(0.5);
    Ok(First {
        phi,
        d2000,
        d1100,
        d3000,
        f21,
        c1,
    })
}

/// `v0` of `H2100(theta) = exp(i omega theta)(v0 + 2 c1 theta q)` and the `gamma` that
/// fixes it, with `kappa = -2 c1`:
///
/// ```text
/// [[Delta(i omega), p^H], [q^H, 0]] (xi, s) = (F21 + kappa Delta'(i omega) q, 0)
/// gamma = -p Delta'(i omega) xi + (1/2) kappa p Delta''(i omega) q,   v0 = xi + gamma q
/// ```
fn h2100_vectors(
    lin: &LinearDde,
    f21: &[C],
    c1: C,
    q: &[C],
    pv: &[C],
    omega: f64,
) -> Result<(Vec<C>, C), EngineError> {
    let n = lin.dim;
    let d = delta(lin, omega, 0);
    let d1 = delta(lin, omega, 1);
    let d2 = delta(lin, omega, 2);
    let kappa = c1.scale(-2.0);
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
        rhs[i] = f21[i] + kappa * d1q[i];
    }
    let sol = solve(&mat, k, &rhs)?;
    let xi = &sol[..n];
    let pd1xi = bilinear(pv, &matvec(&d1, n, xi));
    let pd2q = bilinear(pv, &matvec(&d2, n, q));
    let gamma = (kappa * pd2q).scale(0.5) - pd1xi;
    let v0 = xi.iter().zip(q).map(|(a, b)| *a + gamma * *b).collect();
    Ok((v0, gamma))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn build(src: &str, ev: Evaluator) -> DdeModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, ev).expect("model builds")
    }

    fn both() -> [Evaluator; 2] {
        [Evaluator::Jit, Evaluator::Fallback]
    }

    const WRIGHT: &str = "par a=1.5707963267948966, b=0.3887604649348081, c=1\n\
                          u'=-a*delay(u,1)*(1+u)+b*u^3+c*u^5\ndone\n";
    const WRIGHT_P: [f64; 3] = [FRAC_PI_2, 0.388_760_464_934_808_1, 1.0];
    const TWO_DELAY: &str = "par a=1.2554671180351438, d=0.3, b=0.1825032938841381, c=1\n\
                             u'=-a*delay(u,1)*(1+u)-d*delay(u,2)*(1+u)+b*u^3+c*u^5\ndone\n";
    const TWO_P: [f64; 4] = [1.255_467_118_035_143_8, 0.3, 0.182_503_293_884_138_1, 1.0];
    const TWO_OMEGA: f64 = 1.352_522_338_653_531_7;
    /// FHN on the Hopf curve at the generalized Hopf point (SPEC section 8h, test 2).
    const FHN: &str = "par b=0.9, eps=0.08, c=2.0528, d=-3.2135, tau=1.7722, beta=1.9, \
                       alpha=-1.0429\n\
                       u1'=-u1^3/3+(c+alpha)*u1^2+d*u1-u2+2*beta*tanh(delay(u1,tau))\n\
                       u2'=eps*(u1-b*u2)\ndone\n";
    const FHN_BETA: f64 = 1.899_972_376_314_735_6;
    const FHN_P: [f64; 7] = [
        0.9,
        0.08,
        2.0528,
        -3.2135,
        1.7722,
        FHN_BETA,
        -1.042_940_901_474_694_4,
    ];
    const FHN_OMEGA: f64 = 0.072_000_433_494_719_7;

    fn cdist(a: C, b: C) -> f64 {
        (a - b).abs()
    }

    /// The coefficients of `lyapunov2_scaled` at scale `s`.
    fn scaled(src: &str, ev: Evaluator, p: &[f64], omega: f64, s: C) -> Lyapunov2 {
        let n = if src.contains("u1'") { 2 } else { 1 };
        let mut m = build(src, ev);
        lyapunov2_scaled(&mut m, &vec![0.0; n], p, omega, s).expect("coefficients")
    }

    #[test]
    fn a_scale_multiplies_c2_by_its_fourth_power_and_a_phase_changes_nothing() {
        let cases: [(&str, &[f64], f64); 2] =
            [(WRIGHT, &WRIGHT_P, FRAC_PI_2), (FHN, &FHN_P, FHN_OMEGA)];
        for (src, p, omega) in cases {
            for ev in both() {
                let base = scaled(src, ev.clone(), p, omega, C::new(1.0, 0.0));
                let c2 = C::new(base.c2[0], base.c2[1]);
                let c1 = C::new(base.c1[0], base.c1[1]);
                assert!(c2.abs() > 1e-3, "c2 = {c2:?}");
                for k in [0.5_f64, 2.0] {
                    let r = scaled(src, ev.clone(), p, omega, C::new(k, 0.0));
                    let got = C::new(r.c2[0], r.c2[1]);
                    let err = cdist(got, c2.scale(k.powi(4)));
                    assert!(
                        err <= 1e-11 * c2.abs() * k.powi(4),
                        "k = {k}: {got:?} {c2:?}"
                    );
                    let got1 = C::new(r.c1[0], r.c1[1]);
                    let err1 = cdist(got1, c1.scale(k * k));
                    assert!(err1 <= 1e-11 * c1.abs().max(1e-3) * k * k, "c1, k = {k}");
                    assert!(
                        (r.l2 - base.l2 * k.powi(4)).abs() <= 1e-11 * base.l2.abs() * k.powi(4)
                    );
                }
                for theta in [0.7_f64, -2.1] {
                    let r = scaled(src, ev.clone(), p, omega, C::expi(theta));
                    let got = C::new(r.c2[0], r.c2[1]);
                    assert!(cdist(got, c2) <= 1e-11 * c2.abs(), "theta = {theta}");
                    let got1 = C::new(r.c1[0], r.c1[1]);
                    assert!(
                        cdist(got1, c1) <= 1e-11 * c1.abs().max(1e-3),
                        "c1, theta = {theta}"
                    );
                }
                // A complex scale acts as its modulus to the fourth power.
                let s = C::expi(0.7).scale(2.0);
                let r = scaled(src, ev.clone(), p, omega, s);
                let got = C::new(r.c2[0], r.c2[1]);
                assert!(cdist(got, c2.scale(16.0)) <= 1e-11 * c2.abs() * 16.0);
            }
        }
    }

    /// Nodes and weights of the Gauss-Legendre rule with `m` points on `[-1, 1]`, by Newton's
    /// method on the Legendre polynomial.
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

    /// The vectors of one point that the pairing test needs.
    struct Point {
        lin: LinearDde,
        q: Vec<C>,
        pv: Vec<C>,
        c1: C,
        omega: f64,
        v0: Vec<C>,
        gamma: C,
    }

    fn point(src: &str, ev: Evaluator, x: &[f64], p: &[f64], omega: f64) -> Point {
        let mut m = build(src, ev);
        let lin = linearize(&mut m, x, p).expect("linearize");
        let pr = pair(&lin, omega).expect("pair");
        let mut cx = Ctx::new(&mut m, x, p);
        let f = first_order(&mut cx, &lin, &pr.q, &pr.p, omega).expect("first order");
        let (v0, gamma) =
            h2100_vectors(&lin, &f.f21, f.c1, &pr.q, &pr.p, omega).expect("h2100 vectors");
        Point {
            lin,
            q: pr.q,
            pv: pr.p,
            c1: f.c1,
            omega,
            v0,
            gamma,
        }
    }

    /// The Hale pairing `<psi, phi>` of `psi(s) = p exp(-i omega s)` with
    /// `phi(theta) = exp(i omega theta)(v0 + 2 c1 theta q)`, with the integrals over
    /// `[-tau_j, 0]` done by a 30-point Gauss-Legendre rule:
    /// `psi(0) phi(0) + sum_j int psi(xi + tau_j) A_j phi(xi) d xi`.
    fn hale(pt: &Point, v0: &[C]) -> C {
        let n = pt.lin.dim;
        let (nodes, weights) = gauss_legendre(30);
        let mut total = bilinear(&pt.pv, v0);
        for (tau, aj) in &pt.lin.delays {
            let tau = *tau;
            let mut sum = C::ZERO;
            for (x, w) in nodes.iter().zip(&weights) {
                let t = tau * (x - 1.0) / 2.0;
                let lead = C::expi(pt.omega * t);
                let psi = C::expi(-pt.omega * (t + tau));
                let phi: Vec<C> = (0..n)
                    .map(|i| lead * (v0[i] + (pt.c1 * pt.q[i]).scale(2.0 * t)))
                    .collect();
                let mut aphi = vec![C::ZERO; n];
                for i in 0..n {
                    for j in 0..n {
                        aphi[i] = aphi[i] + phi[j].scale(aj[i * n + j]);
                    }
                }
                sum = sum + (psi * bilinear(&pt.pv, &aphi)).scale(w * tau / 2.0);
            }
            total = total + sum;
        }
        total
    }

    #[test]
    fn h2100_is_orthogonal_to_the_adjoint_eigenfunction_and_gamma_matters() {
        let points: [(&str, &[f64], f64); 3] = [
            (WRIGHT, &WRIGHT_P, FRAC_PI_2),
            (FHN, &FHN_P, FHN_OMEGA),
            (TWO_DELAY, &TWO_P, TWO_OMEGA),
        ];
        let mut large = 0;
        for (src, p, omega) in points {
            let n = if src.contains("u1'") { 2 } else { 1 };
            for ev in both() {
                let pt = point(src, ev, &vec![0.0; n], p, omega);
                let zero = hale(&pt, &pt.v0);
                assert!(zero.abs() < 1e-10, "pairing {zero:?}");
                // Closed form: p Delta' v0 + c1 p Delta'' q.
                let d1 = delta(&pt.lin, pt.omega, 1);
                let d2 = delta(&pt.lin, pt.omega, 2);
                let closed = bilinear(&pt.pv, &matvec(&d1, n, &pt.v0))
                    + pt.c1 * bilinear(&pt.pv, &matvec(&d2, n, &pt.q));
                assert!(cdist(closed, zero) < 1e-10, "{closed:?} {zero:?}");
                // Without gamma the pairing is -gamma, since p Delta' q = 1.
                let xi: Vec<C> = pt
                    .v0
                    .iter()
                    .zip(&pt.q)
                    .map(|(v, q)| *v - pt.gamma * *q)
                    .collect();
                let control = hale(&pt, &xi);
                assert!(
                    (control.abs() - pt.gamma.abs()).abs() < 1e-10,
                    "{control:?} vs {:?}",
                    pt.gamma
                );
                if pt.gamma.abs() > 1e-3 {
                    large += 1;
                }
            }
        }
        // Both evaluators at three points: gamma is large at two of them.
        assert!(large >= 4, "large gamma at {large} of 6 runs");
    }

    #[test]
    fn delta_second_derivative_matches_a_difference_of_the_first() {
        let src = "x'=-x+0.3*y+0.5*delay(x,1)-0.4*delay(y,2)\n\
                   y'=0.2*x-0.7*y+0.6*delay(y,1)+0.1*delay(x,2)\ndone\n";
        let mut m = build(src, Evaluator::Fallback);
        let lin = linearize(&mut m, &[0.0, 0.0], &[]).expect("linearize");
        let (sigma, h) = (0.9, 1e-5);
        let exact = delta(&lin, sigma, 2);
        let up = delta(&lin, sigma + h, 1);
        let down = delta(&lin, sigma - h, 1);
        for k in 0..4 {
            // Delta''(i sigma) = -i d/d sigma Delta'(i sigma).
            let d = (up[k] - down[k]).scale(1.0 / (2.0 * h));
            let fd = C::new(d.im, -d.re);
            assert!(exact[k].abs() > 1e-3);
            assert!(cdist(exact[k], fd) <= 1e-8 * exact[k].abs(), "entry {k}");
        }
    }

    fn arg(n_slots: usize, hidden: usize, u: f64, v: f64, w: f64) -> Vec<C> {
        let mut d = vec![C::ZERO; n_slots];
        d[0] = C::new(u, w);
        d[hidden] = C::new(v, -w);
        d
    }

    #[test]
    fn forms_of_order_2_to_5_match_the_closed_form() {
        // f = exp(a x + b x(t-1)) - 1, so D^r f[h_1..h_r] = e^(a x + b x_d) prod(a h_k + b hd_k).
        let src = "par a=0.8, b=-0.5\nx'=exp(a*x+b*delay(x,1))-1\ndone\n";
        let (a, b) = (0.8, -0.5);
        let x = [0.3];
        let p = [a, b];
        for ev in both() {
            let mut m = build(src, ev);
            let len = 1 + m.eval.n_p();
            let hidden = 1 + m.n_user_params;
            let full = full_params(&m, &x, &p, 0.0);
            let scale = ((a + b) * x[0]).exp();
            for r in 2..=5 {
                for set in 0..4 {
                    let args: Vec<Vec<C>> = (0..r)
                        .map(|k| {
                            let u = ((k * 3 + set * 5 + 1) % 7) as f64 / 4.0 - 0.6;
                            let v = ((k * 5 + set * 3 + 2) % 9) as f64 / 5.0 - 0.7;
                            // The last set is complex.
                            let w = if set == 3 {
                                0.25 * (k as f64 + 1.0)
                            } else {
                                0.0
                            };
                            arg(len, hidden, u, v, w)
                        })
                        .collect();
                    let refs: Vec<&[C]> = args.iter().map(|v| v.as_slice()).collect();
                    let got = form(&mut m, &x, &full, &refs).expect("form");
                    let mut want = C::new(scale, 0.0);
                    for h in &args {
                        want = want * (h[0].scale(a) + h[hidden].scale(b));
                    }
                    assert!(
                        cdist(got[0], want) <= 1e-12 * want.abs().max(1e-3),
                        "r = {r}, set {set}: {:?} vs {want:?}",
                        got[0]
                    );
                }
            }
        }
    }

    #[test]
    fn fhn_forms_have_the_published_first_row_derivatives() {
        // u1' = -u1^3/3 + (c + alpha) u1^2 + d u1 - u2 + 2 beta tanh(u1(t - tau)): the third,
        // fourth and fifth derivatives of tanh at 0 are -2, 0 and 16.
        let x = [0.0, 0.0];
        for ev in both() {
            let mut m = build(FHN, ev);
            let len = 2 + m.eval.n_p();
            let hidden = 2 + m.n_user_params;
            let full = full_params(&m, &x, &FHN_P, 0.0);
            let unit = |slot: usize| {
                let mut d = vec![C::ZERO; len];
                d[slot] = C::new(1.0, 0.0);
                d
            };
            let (u1, dl) = (unit(0), unit(hidden));
            let mut run = |args: &[&[C]]| form(&mut m, &x, &full, args).expect("form");
            let near = |got: C, want: f64, what: &str| {
                assert!(
                    (got.re - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{what}"
                );
                assert!(got.im.abs() < 1e-12, "{what}");
            };
            let c_alpha = FHN_P[2] + FHN_P[6];
            near(run(&[&u1, &u1])[0], 2.0 * c_alpha, "u1 u1");
            near(run(&[&dl, &dl])[0], 0.0, "d d");
            near(run(&[&u1, &u1, &u1])[0], -2.0, "u1 u1 u1");
            near(run(&[&dl, &dl, &dl])[0], -4.0 * FHN_BETA, "d d d");
            near(run(&[&u1, &u1, &dl])[0], 0.0, "u1 u1 d");
            near(run(&[&dl, &dl, &dl, &dl])[0], 0.0, "d d d d");
            near(
                run(&[&dl, &dl, &dl, &dl, &dl])[0],
                32.0 * FHN_BETA,
                "d d d d d",
            );
            near(run(&[&u1, &u1, &u1, &u1])[0], 0.0, "u1 u1 u1 u1");
            // The second row is linear.
            let second = [run(&[&u1, &u1])[1], run(&[&dl, &dl, &dl])[1]];
            for v in second {
                assert!(v.abs() < 1e-14, "second row {v:?}");
            }
        }
    }
}
