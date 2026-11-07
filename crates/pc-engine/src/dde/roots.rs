//! Characteristic roots of `x'(t) = A_0 x(t) + sum_j A_j x(t - tau_j)` (SPEC section 8c).
//!
//! The eigenvalues of a pseudospectral matrix give the starts. Newton's method on the
//! characteristic equation with a normalized null vector refines them, and the result is
//! kept only when the scaled residual is small.

use super::nodes::Nodes;
use crate::error::EngineError;
use pc_linalg::{Lu, eigenvalues};
use std::ops::{Add, Mul, Neg, Sub};

/// Largest `d (n + 1)`, the order of the pseudospectral matrix.
const MAX_ORDER: usize = 4000;
/// Newton steps per start.
const NEWTON_STEPS: usize = 30;
/// A root is accepted when the scaled residual is at most this.
const RESIDUAL_BOUND: f64 = 1e-12;
/// Newton stops when `|d lambda| <= STEP_TOL (1 + |lambda|)`.
const STEP_TOL: f64 = 1e-13;
/// Roots closer than `DEDUPE_TOL (1 + |lambda|)` are one root.
const DEDUPE_TOL: f64 = 1e-11;
/// Starts with real part down to `re_min - START_SLACK (1 + |re_min|)` are refined.
const START_SLACK: f64 = 1e-6;
/// Shift of the inverse iteration, relative to `1 + |lambda|`.
pub(super) const SHIFT: f64 = 1e-8;

/// A linear delay equation `x' = A_0 x + sum_j A_j x(t - tau_j)` with real `dim`-square
/// matrices stored row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearDde {
    pub dim: usize,
    pub a0: Vec<f64>,
    /// The delays `tau_j` with their matrices `A_j`.
    pub delays: Vec<(f64, Vec<f64>)>,
}

/// Which roots to look for. There are no defaults: `n` sets the size of the pseudospectral
/// matrix and `re_min` the left edge of the search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootOptions {
    /// Degree of the interpolation, at least 1.
    pub n: usize,
    /// Roots with `Re lambda >= re_min` are returned.
    pub re_min: f64,
}

/// A refined characteristic root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharRoot {
    pub re: f64,
    pub im: f64,
    /// `||Delta(lambda) v|| / (||v|| s(lambda))` with the scale `s` of SPEC section 8c.
    pub residual: f64,
}

/// The result of [`characteristic_roots`].
#[derive(Clone, Debug, PartialEq)]
pub struct RootSet {
    pub n: usize,
    /// Sorted by real part, then imaginary part, both descending. Complex roots come with
    /// their conjugates.
    pub roots: Vec<CharRoot>,
    /// Eigenvalues of the pseudospectral matrix with `Re >= re_min` that did not refine to
    /// a root, as `(re, im)`. Complex ones come with their conjugates.
    pub unrefined: Vec<(f64, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };

    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }

    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }

    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// `exp(-lambda tau)`.
    fn exp_neg(self, tau: f64) -> C {
        let m = (-self.re * tau).exp();
        let (s, c) = (self.im * tau).sin_cos();
        C::new(m * c, -m * s)
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

impl Neg for C {
    type Output = C;
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadProblem(msg.into())
}

fn check_matrix(name: &str, a: &[f64], d: usize) -> Result<(), EngineError> {
    if a.len() != d * d {
        return Err(bad(format!(
            "{name} has {} entries, expected {}",
            a.len(),
            d * d
        )));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(bad(format!("{name} has a NaN or infinite entry")));
    }
    Ok(())
}

fn validate(p: &LinearDde, o: &RootOptions) -> Result<usize, EngineError> {
    let d = p.dim;
    if d == 0 {
        return Err(bad("dim must be at least 1"));
    }
    if d > MAX_ORDER {
        return Err(bad(format!("dim {d} is too large")));
    }
    check_matrix("a0", &p.a0, d)?;
    if p.delays.is_empty() {
        return Err(bad("at least one delay is required"));
    }
    for (j, (tau, a)) in p.delays.iter().enumerate() {
        if !tau.is_finite() || *tau <= 0.0 {
            return Err(bad(format!("delay {j} is {tau}, it must be positive")));
        }
        check_matrix(&format!("the matrix of delay {j}"), a, d)?;
    }
    if o.n == 0 {
        return Err(bad("n must be at least 1"));
    }
    let order =
        o.n.checked_add(1)
            .and_then(|m| m.checked_mul(d))
            .filter(|&m| m <= MAX_ORDER)
            .ok_or_else(|| bad(format!("dim * (n + 1) must be at most {MAX_ORDER}")))?;
    if !o.re_min.is_finite() {
        return Err(bad("re_min must be finite"));
    }
    Ok(order)
}

/// The pseudospectral matrix `A_n` of order `d (n + 1)`, column-major.
fn assemble(p: &LinearDde, nodes: &Nodes, n: usize) -> Vec<f64> {
    let d = p.dim;
    let order = d * (n + 1);
    let mut a = vec![0.0; order * order];
    // Row block 0: A_0 on the first block column plus the interpolated delay terms.
    for (r, row) in p.a0.chunks(d).enumerate() {
        for (c, v) in row.iter().enumerate() {
            a[c * order + r] += v;
        }
    }
    for (tau, aj) in &p.delays {
        let l = nodes.lagrange_at(-tau);
        for (k, lk) in l.iter().enumerate() {
            if *lk == 0.0 {
                continue;
            }
            for (r, row) in aj.chunks(d).enumerate() {
                for (c, v) in row.iter().enumerate() {
                    a[(k * d + c) * order + r] += lk * v;
                }
            }
        }
    }
    // Row blocks i >= 1: the differentiation matrix times the identity.
    for i in 1..=n {
        for k in 0..=n {
            let dik = nodes.diff(i, k);
            for r in 0..d {
                a[(k * d + r) * order + i * d + r] = dik;
            }
        }
    }
    a
}

/// `Delta(lambda)` and `Delta'(lambda)`, row-major.
fn delta(p: &LinearDde, lam: C) -> (Vec<C>, Vec<C>) {
    let d = p.dim;
    let mut dl = vec![C::ZERO; d * d];
    let mut dp = vec![C::ZERO; d * d];
    for r in 0..d {
        for c in 0..d {
            dl[r * d + c] = C::new(-p.a0[r * d + c], 0.0);
        }
        dl[r * d + r] = dl[r * d + r] + lam;
        dp[r * d + r] = C::new(1.0, 0.0);
    }
    for (tau, a) in &p.delays {
        let e = lam.exp_neg(*tau);
        for i in 0..d * d {
            dl[i] = dl[i] - e.scale(a[i]);
            dp[i] = dp[i] + e.scale(tau * a[i]);
        }
    }
    (dl, dp)
}

/// The real embedding of the complex `k`-square row-major matrix `m`, column-major of order
/// `2k`: `[[Re, -Im], [Im, Re]]`.
fn embed(m: &[C], k: usize) -> Vec<f64> {
    let n = 2 * k;
    let mut out = vec![0.0; n * n];
    for r in 0..k {
        for c in 0..k {
            let z = m[r * k + c];
            out[c * n + r] = z.re;
            out[(k + c) * n + r] = -z.im;
            out[c * n + k + r] = z.im;
            out[(k + c) * n + k + r] = z.re;
        }
    }
    out
}

fn frobenius(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The scale `s(lambda)` of the residual.
fn scale(p: &LinearDde, norms: &(f64, Vec<f64>), lam: C) -> f64 {
    let mut s = lam.abs() + norms.0;
    for ((tau, _), nj) in p.delays.iter().zip(&norms.1) {
        s += nj * (-lam.re * tau).exp();
    }
    s
}

fn norm2(v: &[C]) -> f64 {
    v.iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .sum::<f64>()
        .sqrt()
}

fn mat_vec(m: &[C], v: &[C], d: usize) -> Vec<C> {
    (0..d)
        .map(|r| (0..d).fold(C::ZERO, |acc, c| acc + m[r * d + c] * v[c]))
        .collect()
}

/// Newton's method from the start `start`. Returns the root and its scaled residual, or
/// `None` when a factorization is singular or an iterate is not finite.
fn refine(p: &LinearDde, norms: &(f64, Vec<f64>), start: C) -> Option<(C, f64)> {
    let d = p.dim;
    let real = start.im == 0.0;
    // Start vector: two steps of inverse iteration at a shifted eigenvalue.
    let shifted = C::new(start.re + SHIFT * (1.0 + start.abs()), start.im);
    let lu = Lu::new(&embed(&delta(p, shifted).0, d), 2 * d).ok()?;
    let mut v: Vec<C> = (0..d)
        .map(|i| C::new(1.0 + i as f64 / (d as f64 + 1.0), 0.0))
        .collect();
    for _ in 0..2 {
        let mut rhs: Vec<f64> = v.iter().map(|z| z.re).collect();
        rhs.extend(v.iter().map(|z| z.im));
        lu.solve(&mut rhs).ok()?;
        v = (0..d).map(|i| C::new(rhs[i], rhs[d + i])).collect();
        let nrm = norm2(&v);
        if !nrm.is_finite() || nrm == 0.0 {
            return None;
        }
        for z in &mut v {
            *z = z.scale(1.0 / nrm);
        }
    }
    // The normalization c^H v = 1 stays fixed during the iteration.
    let vhv: f64 = v.iter().map(|z| z.re * z.re + z.im * z.im).sum();
    let c: Vec<C> = v.iter().map(|z| z.scale(1.0 / vhv)).collect();
    let mut lam = start;
    let k = d + 1;
    for _ in 0..NEWTON_STEPS {
        let (dl, dp) = delta(p, lam);
        let dv = mat_vec(&dl, &v, d);
        let dpv = mat_vec(&dp, &v, d);
        let chv = (0..d).fold(C::ZERO, |acc, i| acc + c[i].conj() * v[i]);
        let mut j = vec![C::ZERO; k * k];
        for r in 0..d {
            for col in 0..d {
                j[r * k + col] = dl[r * d + col];
            }
            j[r * k + d] = dpv[r];
        }
        for col in 0..d {
            j[d * k + col] = c[col].conj();
        }
        let mut rhs = Vec::with_capacity(2 * k);
        rhs.extend(dv.iter().map(|z| -z.re));
        rhs.push(-(chv.re - 1.0));
        rhs.extend(dv.iter().map(|z| -z.im));
        rhs.push(-chv.im);
        Lu::new(&embed(&j, k), 2 * k).ok()?.solve(&mut rhs).ok()?;
        for i in 0..d {
            v[i] = v[i] + C::new(rhs[i], rhs[k + i]);
        }
        let step = C::new(rhs[d], rhs[k + d]);
        lam = lam + step;
        if real {
            lam.im = 0.0;
            for z in &mut v {
                z.im = 0.0;
            }
        }
        if !lam.is_finite() || v.iter().any(|z| !z.is_finite()) {
            return None;
        }
        if step.abs() <= STEP_TOL * (1.0 + lam.abs()) {
            break;
        }
    }
    let (dl, _) = delta(p, lam);
    let rho = norm2(&mat_vec(&dl, &v, d)) / (norm2(&v) * scale(p, norms, lam));
    rho.is_finite().then_some((lam, rho))
}

/// Finds the characteristic roots of `p` with `Re lambda >= o.re_min` (SPEC section 8c).
pub fn characteristic_roots(p: &LinearDde, o: &RootOptions) -> Result<RootSet, EngineError> {
    let order = validate(p, o)?;
    let tau_max = p.delays.iter().fold(0.0_f64, |m, (t, _)| m.max(*t));
    let nodes = Nodes::chebyshev(o.n, tau_max);
    let a = assemble(p, &nodes, o.n);
    let ev = eigenvalues(&a, order)?;
    let norms = (
        frobenius(&p.a0),
        p.delays
            .iter()
            .map(|(_, a)| frobenius(a))
            .collect::<Vec<_>>(),
    );
    let slack = START_SLACK * (1.0 + o.re_min.abs());
    let mut found: Vec<(C, f64)> = Vec::new();
    let mut unrefined: Vec<(f64, f64)> = Vec::new();
    for &(re, im) in &ev {
        if im < 0.0 || re < o.re_min - slack {
            continue;
        }
        match refine(p, &norms, C::new(re, im)) {
            Some((lam, rho)) if rho <= RESIDUAL_BOUND => {
                if lam.re >= o.re_min {
                    // The problem is real, so work on the upper half plane.
                    let lam = if lam.im < 0.0 { lam.conj() } else { lam };
                    found.push((lam, rho));
                }
            }
            _ => {
                if re >= o.re_min {
                    unrefined.push((re, im));
                    if im > 0.0 {
                        unrefined.push((re, -im));
                    }
                }
            }
        }
    }
    // Two starts can refine to the same root. Keep the one with the smaller residual.
    found.sort_by(|x, y| x.1.total_cmp(&y.1));
    let mut kept: Vec<(C, f64)> = Vec::new();
    for (lam, rho) in found {
        let same = kept
            .iter()
            .any(|(k, _)| (lam - *k).abs() <= DEDUPE_TOL * (1.0 + lam.abs()));
        if !same {
            kept.push((lam, rho));
        }
    }
    let mut roots = Vec::with_capacity(2 * kept.len());
    for (lam, rho) in kept {
        roots.push(CharRoot {
            re: lam.re,
            im: lam.im,
            residual: rho,
        });
        // A root this close to the real axis is its own conjugate within the tolerance.
        if lam.im > 0.0 && 2.0 * lam.im > DEDUPE_TOL * (1.0 + lam.abs()) {
            roots.push(CharRoot {
                re: lam.re,
                im: -lam.im,
                residual: rho,
            });
        }
    }
    roots.sort_by(|x, y| y.re.total_cmp(&x.re).then(y.im.total_cmp(&x.im)));
    unrefined.sort_by(|x, y| y.0.total_cmp(&x.0).then(y.1.total_cmp(&x.1)));
    Ok(RootSet {
        n: o.n,
        roots,
        unrefined,
    })
}
