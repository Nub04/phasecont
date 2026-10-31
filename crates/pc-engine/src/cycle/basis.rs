//! Collocation basis on one mesh interval, rescaled to [0, 1].
//!
//! The m collocation points are the Gauss-Legendre nodes, found by Newton's method on
//! the Legendre polynomial P_m. The solution on the interval is the degree-m polynomial
//! through its values at the m+1 equally spaced base points 0, 1/m, ..., 1, written in
//! the Lagrange basis of those points.

use crate::EngineError;

/// Smallest and largest number of collocation points per interval.
pub const NCOL_RANGE: std::ops::RangeInclusive<usize> = 2..=7;

/// Gauss-Legendre nodes and weights and the Lagrange basis for m collocation points.
#[derive(Clone, Debug)]
pub struct CollocationBasis {
    m: usize,
    nodes: Vec<f64>,
    weights: Vec<f64>,
    base: Vec<f64>,
    /// The factors 1 / prod_{k != j} (s_j - s_k) of the Lagrange polynomials.
    denom: Vec<f64>,
    /// L_j(z_i), row-major m by (m+1).
    l: Vec<f64>,
    /// L_j'(z_i), row-major m by (m+1).
    dl: Vec<f64>,
    /// d^m L_j / dz^m, which is constant.
    dm: Vec<f64>,
}

impl CollocationBasis {
    /// The basis for `m` collocation points, `m` in [`NCOL_RANGE`].
    pub fn new(m: usize) -> Result<Self, EngineError> {
        if !NCOL_RANGE.contains(&m) {
            return Err(EngineError::BadProblem(format!(
                "the number of collocation points must be between {} and {}, got {m}",
                NCOL_RANGE.start(),
                NCOL_RANGE.end()
            )));
        }
        let (nodes, weights) = gauss_legendre(m);
        let base: Vec<f64> = (0..=m).map(|j| j as f64 / m as f64).collect();
        let denom: Vec<f64> = (0..=m)
            .map(|j| {
                let p: f64 = (0..=m)
                    .filter(|&k| k != j)
                    .map(|k| base[j] - base[k])
                    .product();
                1.0 / p
            })
            .collect();
        let fact: f64 = (1..=m).map(|k| k as f64).product();
        let dm = denom.iter().map(|d| fact * d).collect();
        let mut b = CollocationBasis {
            m,
            nodes,
            weights,
            base,
            denom,
            l: vec![0.0; m * (m + 1)],
            dl: vec![0.0; m * (m + 1)],
            dm,
        };
        let mut row = vec![0.0; m + 1];
        for i in 0..m {
            b.values(b.nodes[i], &mut row);
            b.l[i * (m + 1)..(i + 1) * (m + 1)].copy_from_slice(&row);
            b.derivatives(b.nodes[i], &mut row);
            b.dl[i * (m + 1)..(i + 1) * (m + 1)].copy_from_slice(&row);
        }
        Ok(b)
    }

    /// Number of collocation points m.
    pub fn m(&self) -> usize {
        self.m
    }

    /// The m Gauss-Legendre nodes in (0, 1), ascending.
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// The m Gauss-Legendre weights on [0, 1], which sum to 1.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The m+1 base points j/m.
    pub fn base(&self) -> &[f64] {
        &self.base
    }

    /// L_j(z) for j = 0..=m into `out`.
    pub fn values(&self, z: f64, out: &mut [f64]) {
        let m = self.m;
        for (j, o) in out.iter_mut().enumerate().take(m + 1) {
            let p: f64 = (0..=m)
                .filter(|&k| k != j)
                .map(|k| z - self.base[k])
                .product();
            *o = p * self.denom[j];
        }
    }

    /// L_j'(z) for j = 0..=m into `out`.
    pub fn derivatives(&self, z: f64, out: &mut [f64]) {
        let m = self.m;
        for (j, o) in out.iter_mut().enumerate().take(m + 1) {
            let mut s = 0.0;
            for l in (0..=m).filter(|&l| l != j) {
                let p: f64 = (0..=m)
                    .filter(|&k| k != j && k != l)
                    .map(|k| z - self.base[k])
                    .product();
                s += p;
            }
            *o = s * self.denom[j];
        }
    }

    /// L_j at node i.
    pub fn value_at_node(&self, i: usize, j: usize) -> f64 {
        self.l[i * (self.m + 1) + j]
    }

    /// L_j' at node i.
    pub fn derivative_at_node(&self, i: usize, j: usize) -> f64 {
        self.dl[i * (self.m + 1) + j]
    }

    /// The m-th derivative of each L_j, a constant: m! / prod_{k != j} (s_j - s_k).
    pub fn mth_derivative(&self) -> &[f64] {
        &self.dm
    }
}

/// P_m(x) and P_m'(x) by the three-term recurrence.
fn legendre(m: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 1..m {
        let k = k as f64;
        let p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = p2;
    }
    let dp = m as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Nodes (ascending) and weights of the m-point Gauss-Legendre rule on [0, 1]. Newton on
/// P_m from the guess cos(pi (i + 3/4) / (m + 1/2)) gives the roots in the upper half;
/// the others follow by symmetry, so the rule is exactly symmetric about 1/2.
fn gauss_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut z = vec![0.0; m];
    let mut w = vec![0.0; m];
    for i in 0..m.div_ceil(2) {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
        if 2 * i + 1 == m {
            x = 0.0;
        } else {
            for _ in 0..100 {
                let (p, dp) = legendre(m, x);
                let dx = p / dp;
                x -= dx;
                if dx.abs() <= 1e-16 {
                    break;
                }
            }
        }
        let (_, dp) = legendre(m, x);
        let wi = 1.0 / ((1.0 - x * x) * dp * dp);
        z[i] = 0.5 * (1.0 - x);
        z[m - 1 - i] = 0.5 * (1.0 + x);
        w[i] = wi;
        w[m - 1 - i] = wi;
    }
    (z, w)
}
