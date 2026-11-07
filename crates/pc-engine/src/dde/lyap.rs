//! The first Lyapunov coefficient at a Hopf point of a delay equilibrium (SPEC section 8g).
//!
//! The formulas are those of Bosschaert, Janssens and Kuznetsov, arXiv 1903.08276 v1,
//! Sections 2.5, 6.1, 6.1.1 and 7. With `Delta(z) = zI - sum_j M_j exp(-z tau_j)`, the
//! right and left null vectors `q` and `p` of `Delta(i omega)` satisfy
//! `p Delta'(i omega) q = 1` and `|q| = 1`, and `phi(theta) = exp(i omega theta) q`. Then
//!
//! ```text
//! H2000 = Delta(2 i omega)^-1 B(phi, phi)
//! H1100 = Delta(0)^-1 B(phi, conj phi)
//! c1    = (1/2) p ( B(conj phi, H2000) + 2 B(phi, H1100) + C(phi, phi, conj phi) )
//! l1    = Re c1 / omega
//! ```
//!
//! `B` and `C` are the second and third derivatives of the right side with respect to the
//! state and the lagged states. The step `l1 = Re c1 / omega` is not in the paper. The
//! scale `|q| = 1` is the one DDE-BIFTOOL uses, and `c1` depends on it, so only the sign of
//! `l1` and the equation `l1 = 0` do not.
//!
//! The model evaluates the forms through `Dir(vec![Slot::Full; r])`. The lowered model has
//! one hidden parameter per (state, delay time) pair, and the direction of
//! `exp(s theta) h` has `h` in the state and `exp(-s tau_k) h[state_k]` in hidden slot `k`.
//! A complex argument is split into its real and imaginary parts, so that a form of `r`
//! complex arguments is a sum of `2^r` real evaluations.

use super::eq::{check_params, check_state, full_params, linearize, residual, time_of};
use super::roots::LinearDde;
use crate::error::EngineError;
use crate::model::DdeModel;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{Lu, null_right, singular_values};
use std::ops::{Add, Mul, Sub};

/// Norm of `q`. The coefficient scales with the square of it.
pub const Q_NORM: f64 = 1.0;
/// The largest relative residual `|Delta(i omega) q| / |Delta(i omega)|_F` that is read as
/// a Hopf frequency.
pub const NULL_RESIDUAL_MAX: f64 = 1e-4;
/// The smallest `|p Delta' q|` before scaling, with `|p| = |q| = 1`.
pub const PAIRING_MIN: f64 = 1e-8;
/// The smallest ratio of the extreme singular values of `Delta(0)` and `Delta(2 i omega)`.
pub const RESONANCE_MIN: f64 = 1e-8;
const OMEGA_MIN: f64 = 1e-8;

/// The coefficient at one Hopf point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lyapunov {
    pub omega: f64,
    /// `[Re c1, Im c1]` at `|q| = Q_NORM`.
    pub c1: [f64; 2],
    /// `Re c1 / omega`. Negative is supercritical and positive is subcritical.
    pub l1: f64,
    /// `|Delta(i omega) q| / s` for the unit vector `q`, with
    /// `s = omega + |A_0|_F + sum_j |A_j|_F`.
    pub null_residual: f64,
    /// The largest absolute entry of `f(x, ..., x, p)`.
    pub equilibrium_residual: f64,
}

/// The first Lyapunov coefficient of the Hopf point `(x, p, omega)`, where `p` holds the
/// declared parameters. `x` and `omega` are not polished.
pub fn lyapunov(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
) -> Result<Lyapunov, EngineError> {
    lyapunov_scaled(m, x, p, omega, Q_NORM)
}

fn lyapunov_scaled(
    m: &mut DdeModel,
    x: &[f64],
    p: &[f64],
    omega: f64,
    q_norm: f64,
) -> Result<Lyapunov, EngineError> {
    if !(omega.is_finite() && omega > OMEGA_MIN) {
        return Err(EngineError::BadProblem(format!(
            "omega is {omega}; it must be finite and above {OMEGA_MIN:e}"
        )));
    }
    check_params(m, p)?;
    check_state(m, x)?;
    let lin = linearize(m, x, p)?;
    let pr = pair(&lin, omega)?;
    let q: Vec<C> = pr.q.iter().map(|c| c.scale(q_norm)).collect();
    let pv: Vec<C> = pr.p.iter().map(|c| c.scale(1.0 / q_norm)).collect();
    let c1 = coefficient(m, &lin, x, p, omega, &q, &pv)?;
    Ok(Lyapunov {
        omega,
        c1: [c1.re, c1.im],
        l1: c1.re / omega,
        null_residual: pr.null_residual,
        equilibrium_residual: residual(m, x, p)?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };

    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }

    /// `exp(i theta)`.
    fn expi(theta: f64) -> C {
        C::new(theta.cos(), theta.sin())
    }

    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }

    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn div(self, d: C) -> C {
        let den = d.re * d.re + d.im * d.im;
        C::new(
            (self.re * d.re + self.im * d.im) / den,
            (self.im * d.re - self.re * d.im) / den,
        )
    }
}

impl Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// `Delta(i sigma)`, or `Delta'(i sigma)` when `deriv`, as an n-by-n row-major matrix.
fn delta(lin: &LinearDde, sigma: f64, deriv: bool) -> Vec<C> {
    let n = lin.dim;
    let mut d = vec![C::ZERO; n * n];
    for i in 0..n {
        d[i * n + i] = if deriv {
            C::new(1.0, 0.0)
        } else {
            C::new(0.0, sigma)
        };
    }
    let terms = std::iter::once((0.0, &lin.a0)).chain(lin.delays.iter().map(|(t, a)| (*t, a)));
    for (tau, mat) in terms {
        // exp(-i sigma tau)
        let e = C::expi(-sigma * tau);
        let w = if deriv { e.scale(tau) } else { e.scale(-1.0) };
        for (dst, a) in d.iter_mut().zip(mat.iter()) {
            *dst = *dst + w.scale(*a);
        }
    }
    d
}

/// The real `2n`-by-`2n` column-major matrix `[[Re, -Im], [Im, Re]]` of a complex matrix.
fn embed(c: &[C], n: usize) -> Vec<f64> {
    let m = 2 * n;
    let mut e = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..n {
            let z = c[i * n + j];
            e[i + m * j] = z.re;
            e[i + m * (j + n)] = -z.im;
            e[(i + n) + m * j] = z.im;
            e[(i + n) + m * (j + n)] = z.re;
        }
    }
    e
}

fn transpose(c: &[C], n: usize) -> Vec<C> {
    let mut t = vec![C::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = c[i * n + j];
        }
    }
    t
}

fn matvec(c: &[C], n: usize, v: &[C]) -> Vec<C> {
    (0..n)
        .map(|i| (0..n).fold(C::ZERO, |s, j| s + c[i * n + j] * v[j]))
        .collect()
}

/// `sum_i a_i b_i`, with no conjugate.
fn bilinear(a: &[C], b: &[C]) -> C {
    a.iter().zip(b).fold(C::ZERO, |s, (x, y)| s + *x * *y)
}

fn norm2(v: &[C]) -> f64 {
    v.iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum::<f64>()
        .sqrt()
}

fn frobenius_real(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn unstack(v: &[f64], n: usize) -> Vec<C> {
    (0..n).map(|i| C::new(v[i], v[n + i])).collect()
}

/// `c^-1 rhs`, through the real embedding.
fn solve(c: &[C], n: usize, rhs: &[C]) -> Result<Vec<C>, EngineError> {
    let lu = Lu::new(&embed(c, n), 2 * n)?;
    let mut b = vec![0.0; 2 * n];
    for (i, z) in rhs.iter().enumerate() {
        b[i] = z.re;
        b[n + i] = z.im;
    }
    lu.solve(&mut b)?;
    Ok(unstack(&b, n))
}

struct Pair {
    q: Vec<C>,
    p: Vec<C>,
    null_residual: f64,
}

/// The unit right null vector `q` of `Delta(i omega)` and the row vector `p` with
/// `p Delta = 0` and `p Delta' q = 1`.
fn pair(lin: &LinearDde, omega: f64) -> Result<Pair, EngineError> {
    let n = lin.dim;
    let d = delta(lin, omega, false);
    let dp = delta(lin, omega, true);
    let right = null_right(&embed(&d, n), 2 * n, 2 * n, 1)?.remove(0);
    let q = unstack(&right, n);
    let scale = omega
        + frobenius_real(&lin.a0)
        + lin
            .delays
            .iter()
            .map(|(_, a)| frobenius_real(a))
            .sum::<f64>();
    let null_residual = norm2(&matvec(&d, n, &q)) / scale;
    if null_residual.is_nan() || null_residual > NULL_RESIDUAL_MAX {
        return Err(EngineError::BadProblem(format!(
            "omega = {omega} is not a Hopf frequency of this equation: the relative residual \
             of Delta(i omega) q is {null_residual:e}, above {NULL_RESIDUAL_MAX:e}"
        )));
    }
    let dt = transpose(&d, n);
    let left = null_right(&embed(&dt, n), 2 * n, 2 * n, 1)?.remove(0);
    let pt = unstack(&left, n);
    let s = bilinear(&pt, &matvec(&dp, n, &q));
    if s.abs().is_nan() || s.abs() < PAIRING_MIN {
        return Err(EngineError::Singular(format!(
            "|p Delta'(i omega) q| = {:e}; the root i omega is not simple",
            s.abs()
        )));
    }
    let p = pt.iter().map(|c| c.div(s)).collect();
    Ok(Pair {
        q,
        p,
        null_residual,
    })
}

/// Fails when the extreme singular values of the embedding of `d` have a ratio below
/// `RESONANCE_MIN`.
fn check_regular(d: &[C], n: usize, what: &str) -> Result<(), EngineError> {
    let s = singular_values(&embed(d, n), 2 * n, 2 * n)?;
    let ratio = s[2 * n - 1] / s[0];
    if ratio.is_nan() || ratio < RESONANCE_MIN {
        return Err(EngineError::Singular(format!(
            "{what} is singular (smallest over largest singular value {ratio:e}), so the \
             coefficient is not defined"
        )));
    }
    Ok(())
}

/// The direction of `exp(i sigma theta) h` in the variables of the lowered model: `h` in
/// the state, then the hidden slots, with zeros for the declared parameters.
fn direction(m: &DdeModel, times: &[f64], h: &[C], sigma: f64) -> Vec<C> {
    let n = h.len();
    let mut d = vec![C::ZERO; n + m.eval.n_p()];
    d[..n].copy_from_slice(h);
    for (k, s) in m.delays.iter().enumerate() {
        // exp(-i sigma tau_k)
        d[n + m.n_user_params + k] = h[s.state] * C::expi(-sigma * times[k]);
    }
    d
}

/// The multilinear form `D^r f[args]` at `(x, full)` for complex directions, where
/// `r = args.len()` is 2 or 3.
fn form(m: &mut DdeModel, x: &[f64], full: &[f64], args: &[&[C]]) -> Result<Vec<C>, EngineError> {
    let r = args.len();
    let n = m.eval.n();
    let len = n + m.eval.n_p();
    let kind = EntryKind::Dir(vec![Slot::Full; r]);
    let mut out = vec![C::ZERO; n];
    let mut v = vec![0.0; r * len];
    let mut o = vec![0.0; n];
    for mask in 0..(1usize << r) {
        let mut imag = 0;
        for (k, a) in args.iter().enumerate() {
            let use_im = (mask >> k) & 1 == 1;
            imag += usize::from(use_im);
            for (i, c) in a.iter().enumerate() {
                v[k * len + i] = if use_im { c.im } else { c.re };
            }
        }
        m.eval.eval(&kind, x, full, &v, &mut o)?;
        // i^imag
        let f = match imag % 4 {
            0 => C::new(1.0, 0.0),
            1 => C::new(0.0, 1.0),
            2 => C::new(-1.0, 0.0),
            _ => C::new(0.0, -1.0),
        };
        for (dst, val) in out.iter_mut().zip(&o) {
            *dst = *dst + f.scale(*val);
        }
    }
    Ok(out)
}

/// `c1` for the given vectors. The caller supplies `q` and `p` with `p Delta' q = 1`.
fn coefficient(
    m: &mut DdeModel,
    lin: &LinearDde,
    x: &[f64],
    p_decl: &[f64],
    omega: f64,
    q: &[C],
    pv: &[C],
) -> Result<C, EngineError> {
    let n = lin.dim;
    let d0 = delta(lin, 0.0, false);
    let d2 = delta(lin, 2.0 * omega, false);
    check_regular(&d0, n, "Delta(0)")?;
    check_regular(&d2, n, "Delta(2 i omega)")?;
    let full = full_params(m, x, p_decl, 0.0);
    let times: Vec<f64> = m.delays.iter().map(|s| time_of(s, p_decl)).collect();
    let qbar: Vec<C> = q.iter().map(|c| c.conj()).collect();
    let phi = direction(m, &times, q, omega);
    let phib = direction(m, &times, &qbar, -omega);
    let b_pp = form(m, x, &full, &[&phi, &phi])?;
    let b_pb = form(m, x, &full, &[&phi, &phib])?;
    let h2000 = solve(&d2, n, &b_pp)?;
    let h1100 = solve(&d0, n, &b_pb)?;
    let dir_h2000 = direction(m, &times, &h2000, 2.0 * omega);
    let dir_h1100 = direction(m, &times, &h1100, 0.0);
    let t1 = form(m, x, &full, &[&phib, &dir_h2000])?;
    let t2 = form(m, x, &full, &[&phi, &dir_h1100])?;
    let t3 = form(m, x, &full, &[&phi, &phi, &phib])?;
    let w: Vec<C> = (0..n).map(|i| t1[i] + t2[i].scale(2.0) + t3[i]).collect();
    Ok(bilinear(pv, &w).scale(0.5))
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

    fn model(src: &str) -> DdeModel {
        build(src, Evaluator::Fallback)
    }

    const WRIGHT: &str = "par a=1.5707963267948966\nu'=-a*delay(u,1)*(1+u)\ndone\n";
    const POLAR: &str = "par w=1.3, a=0.37\nx'=-w*y+a*x*(x^2+y^2)\ny'=w*x+a*y*(x^2+y^2)\n\
                         z'=-2*z-delay(z,1)\ndone\n";
    const FHN: &str = "par b=0.9, eps=0.08, c=2.0528, d=-3.2135, tau=1.7722, beta=1.9, \
                       alpha=-0.9710\n\
                       u1'=-u1^3/3+(c+alpha)*u1^2+d*u1-u2+2*beta*tanh(delay(u1,tau))\n\
                       u2'=eps*(u1-b*u2)\ndone\n";
    const FHN_P: [f64; 7] = [0.9, 0.08, 2.0528, -3.2135, 1.7722, 1.9, -0.9710];
    const FHN_OMEGA: f64 = 0.07199637889093541;

    fn wright_l1() -> f64 {
        2.0 * (2.0 - 3.0 * PI) / (5.0 * (4.0 + PI * PI))
    }

    fn is_bad(r: &Result<Lyapunov, EngineError>) -> bool {
        matches!(r, Err(EngineError::BadProblem(_)))
    }

    fn is_singular(r: &Result<Lyapunov, EngineError>, what: &str) -> bool {
        matches!(r, Err(EngineError::Singular(s)) if s.contains(what))
    }

    fn wright_checks(m: &mut DdeModel) {
        let r = lyapunov(m, &[0.0], &[FRAC_PI_2], FRAC_PI_2).expect("coefficient");
        assert!((r.l1 - wright_l1()).abs() < 1e-9, "l1 = {}", r.l1);
        assert!((r.c1[0] - r.omega * r.l1).abs() < 1e-9);
        assert!((r.c1[0] + 0.33635606642474136).abs() < 1e-9, "{:?}", r.c1);
        assert!((r.c1[1] + 0.41413092244177396).abs() < 1e-9, "{:?}", r.c1);
        assert!(r.null_residual < 1e-12);
        assert_eq!(r.equilibrium_residual, 0.0);
    }

    #[test]
    fn wright_matches_the_closed_form() {
        wright_checks(&mut model(WRIGHT));
    }

    #[test]
    fn wright_matches_the_closed_form_with_the_jit() {
        wright_checks(&mut build(WRIGHT, Evaluator::Jit));
    }

    #[test]
    fn wright_pair_is_normalized() {
        let mut m = model(WRIGHT);
        let lin = linearize(&mut m, &[0.0], &[FRAC_PI_2]).unwrap();
        let pr = pair(&lin, FRAC_PI_2).unwrap();
        let d = delta(&lin, FRAC_PI_2, false);
        let dp = delta(&lin, FRAC_PI_2, true);
        let pd: Vec<C> = matvec(&transpose(&d, 1), 1, &pr.p);
        assert!(norm2(&pd) < 1e-12);
        let s = bilinear(&pr.p, &matvec(&dp, 1, &pr.q));
        assert!((s - C::new(1.0, 0.0)).abs() < 1e-12);
        assert!(norm2(&matvec(&d, 1, &pr.q)) < 1e-12);
        assert!((norm2(&pr.q) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn polar_with_a_decoupled_delay_block() {
        let mut m = model(POLAR);
        let r = lyapunov(&mut m, &[0.0; 3], &[1.3, 0.37], 1.3).expect("coefficient");
        assert!((r.c1[0] - 0.74).abs() < 1e-12, "{:?}", r.c1);
        assert!(r.c1[1].abs() < 1e-12, "{:?}", r.c1);
        assert!((r.l1 - 0.5692307692307692).abs() < 1e-12, "{}", r.l1);
    }

    fn fhn_l1(alpha: f64, omega: f64) -> f64 {
        let mut m = model(FHN);
        let mut p = FHN_P;
        p[6] = alpha;
        lyapunov(&mut m, &[0.0, 0.0], &p, omega)
            .expect("coefficient")
            .l1
    }

    #[test]
    fn fhn_matches_the_numpy_value_and_the_published_one() {
        let l1 = fhn_l1(-0.9710, FHN_OMEGA);
        assert!((l1 - 0.398075831491134).abs() < 1e-8, "l1 = {l1}");
        assert!((l1 - 0.3980).abs() < 1e-3 * 0.3980);
        let mut m = model(FHN);
        let r = lyapunov(&mut m, &[0.0, 0.0], &FHN_P, FHN_OMEGA).unwrap();
        assert!(r.null_residual < 1e-4);
    }

    #[test]
    fn fhn_changes_sign_in_alpha() {
        let w = 0.07199638;
        assert!(fhn_l1(-1.0430, w) < 0.0);
        assert!(fhn_l1(-1.0428, w) > 0.0);
        let (mut lo, mut hi) = (-1.0430, -1.0428);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if fhn_l1(mid, w) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!((0.5 * (lo + hi) + 1.042941555598).abs() < 1e-6, "{lo} {hi}");
    }

    fn rel(a: C, b: C) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn the_scale_of_q_enters_as_a_square() {
        for (src, x, p, w) in [
            (WRIGHT, vec![0.0], vec![FRAC_PI_2], FRAC_PI_2),
            (FHN, vec![0.0, 0.0], FHN_P.to_vec(), FHN_OMEGA),
        ] {
            let mut m = model(src);
            let base = lyapunov(&mut m, &x, &p, w).unwrap();
            let b = C::new(base.c1[0], base.c1[1]);
            for s in [0.5, 2.0] {
                let r = lyapunov_scaled(&mut m, &x, &p, w, s).unwrap();
                let c = C::new(r.c1[0], r.c1[1]);
                assert!(rel(c, b.scale(s * s)) < 1e-12, "s = {s}");
            }
        }
    }

    #[test]
    fn a_phase_of_q_does_not_change_c1() {
        for (src, x, p, w) in [
            (WRIGHT, vec![0.0], vec![FRAC_PI_2], FRAC_PI_2),
            (FHN, vec![0.0, 0.0], FHN_P.to_vec(), FHN_OMEGA),
        ] {
            let mut m = model(src);
            let lin = linearize(&mut m, &x, &p).unwrap();
            let pr = pair(&lin, w).unwrap();
            let base = coefficient(&mut m, &lin, &x, &p, w, &pr.q, &pr.p).unwrap();
            for theta in [0.7, -2.1] {
                let e = C::expi(theta);
                let q: Vec<C> = pr.q.iter().map(|c| *c * e).collect();
                let pv: Vec<C> = pr.p.iter().map(|c| *c * e.conj()).collect();
                let c = coefficient(&mut m, &lin, &x, &p, w, &q, &pv).unwrap();
                assert!(rel(c, base) < 1e-12, "theta = {theta}");
            }
        }
    }

    /// `F` at `(x + s d_x, full + s d_slots)`.
    fn shifted(m: &mut DdeModel, x: &[f64], full: &[f64], d: &[f64], s: f64) -> Vec<f64> {
        let n = x.len();
        let xs: Vec<f64> = (0..n).map(|i| x[i] + s * d[i]).collect();
        let mut ps = full.to_vec();
        for (j, v) in d[n..].iter().enumerate() {
            ps[j] += s * v;
        }
        let mut out = vec![0.0; n];
        m.eval.eval(&EntryKind::F, &xs, &ps, &[], &mut out).unwrap();
        out
    }

    #[test]
    fn the_forms_match_finite_differences() {
        let mut m = model(FHN);
        let x = [0.0, 0.0];
        let full = full_params(&m, &x, &FHN_P, 0.0);
        let times: Vec<f64> = m.delays.iter().map(|s| time_of(s, &FHN_P)).collect();
        let h = [C::new(0.7, 0.0), C::new(-0.4, 0.0)];
        let d = direction(&m, &times, &h, 0.0);
        let dr: Vec<f64> = d.iter().map(|c| c.re).collect();
        assert!(d.iter().all(|c| c.im == 0.0));
        let s = 1e-3;
        let (g0, g1, gm1) = (
            shifted(&mut m, &x, &full, &dr, 0.0),
            shifted(&mut m, &x, &full, &dr, s),
            shifted(&mut m, &x, &full, &dr, -s),
        );
        let (g2, gm2) = (
            shifted(&mut m, &x, &full, &dr, 2.0 * s),
            shifted(&mut m, &x, &full, &dr, -2.0 * s),
        );
        let b = form(&mut m, &x, &full, &[&d, &d]).unwrap();
        let c = form(&mut m, &x, &full, &[&d, &d, &d]).unwrap();
        for i in 0..2 {
            let b_fd = (g1[i] - 2.0 * g0[i] + gm1[i]) / (s * s);
            let c_fd = (g2[i] - 2.0 * g1[i] + 2.0 * gm1[i] - gm2[i]) / (2.0 * s * s * s);
            assert!(b[i].im.abs() < 1e-14 && c[i].im.abs() < 1e-14);
            assert!((b[i].re - b_fd).abs() < 1e-5 * (1.0 + b_fd.abs()), "B {i}");
            assert!((c[i].re - c_fd).abs() < 1e-5 * (1.0 + c_fd.abs()), "C {i}");
        }
        // The cubic term of the first row is -u1^3/3, so C[0] has a part -2 h1^3.
        assert!(c[0].re.abs() > 1e-3);
    }

    #[test]
    fn the_second_form_is_symmetric_and_real_on_a_conjugate_pair() {
        let mut m = model(FHN);
        let x = [0.0, 0.0];
        let full = full_params(&m, &x, &FHN_P, 0.0);
        let times: Vec<f64> = m.delays.iter().map(|s| time_of(s, &FHN_P)).collect();
        let a = direction(&m, &times, &[C::new(0.3, 0.2), C::new(-0.5, 0.1)], 0.9);
        let b = direction(&m, &times, &[C::new(0.1, -0.4), C::new(0.6, 0.7)], -0.3);
        let ab = form(&mut m, &x, &full, &[&a, &b]).unwrap();
        let ba = form(&mut m, &x, &full, &[&b, &a]).unwrap();
        for i in 0..2 {
            assert!((ab[i] - ba[i]).abs() < 1e-14 * (1.0 + ab[i].abs()));
        }
        let lin = linearize(&mut m, &x, &FHN_P).unwrap();
        let pr = pair(&lin, FHN_OMEGA).unwrap();
        let qbar: Vec<C> = pr.q.iter().map(|c| c.conj()).collect();
        let phi = direction(&m, &times, &pr.q, FHN_OMEGA);
        let phib = direction(&m, &times, &qbar, -FHN_OMEGA);
        let b_pb = form(&mut m, &x, &full, &[&phi, &phib]).unwrap();
        assert!(b_pb.iter().all(|c| c.im.abs() < 1e-14), "{b_pb:?}");
    }

    #[test]
    fn wright_has_no_cubic_term() {
        let mut m = model(WRIGHT);
        let full = full_params(&m, &[0.0], &[FRAC_PI_2], 0.0);
        let times = [1.0];
        let lin = linearize(&mut m, &[0.0], &[FRAC_PI_2]).unwrap();
        let pr = pair(&lin, FRAC_PI_2).unwrap();
        let qbar: Vec<C> = pr.q.iter().map(|c| c.conj()).collect();
        let phi = direction(&m, &times, &pr.q, FRAC_PI_2);
        let phib = direction(&m, &times, &qbar, -FRAC_PI_2);
        let c = form(&mut m, &[0.0], &full, &[&phi, &phi, &phib]).unwrap();
        assert!(c[0].abs() < 1e-15, "{:?}", c[0]);
    }

    #[test]
    fn bad_arguments_are_bad_problems() {
        let mut m = model(WRIGHT);
        let a = [FRAC_PI_2];
        assert!(is_bad(&lyapunov(&mut m, &[0.0], &a, 1.2)), "omega = 1.2");
        assert!(is_bad(&lyapunov(&mut m, &[0.0, 0.0], &a, FRAC_PI_2)));
        assert!(is_bad(&lyapunov(&mut m, &[0.0], &[], FRAC_PI_2)));
        assert!(is_bad(&lyapunov(&mut m, &[0.0], &[1.0, 2.0], FRAC_PI_2)));
        for w in [0.0, f64::NAN, 1e-9, -1.0, f64::INFINITY] {
            assert!(is_bad(&lyapunov(&mut m, &[0.0], &a, w)), "omega = {w}");
        }
        let mut t = model("par a=1.5, tau=1\nu'=-a*delay(u,tau)*(1+u)\ndone\n");
        assert!(is_bad(&lyapunov(&mut t, &[0.0], &[1.5, -1.0], 1.0)));
        assert!(is_bad(&lyapunov(&mut t, &[0.0], &[1.5, 0.0], 1.0)));
    }

    #[test]
    fn a_zero_root_is_singular() {
        let mut m = model(
            "par a=0, b=1, tau=1.5707963267948966\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n",
        );
        let r = lyapunov(&mut m, &[0.0, 0.0], &[0.0, 1.0, FRAC_PI_2], 1.0);
        assert!(is_singular(&r, "Delta(0)"), "{r:?}");
    }

    #[test]
    fn a_one_to_two_resonance_is_singular() {
        let mut m = model("x1'=-y1+x1^3\ny1'=x1\nx2'=-2*y2\ny2'=2*x2\nz'=-2*z-delay(z,1)\ndone\n");
        let r = lyapunov(&mut m, &[0.0; 5], &[], 1.0);
        assert!(is_singular(&r, "Delta(2 i omega)"), "{r:?}");
    }
}
