//! Chebyshev nodes on the delay interval, with the differentiation matrix and the Lagrange
//! basis in barycentric form (SPEC section 8c).

/// The nodes `theta_k = tau_max/2 (cos(k pi/n) - 1)`, `k = 0..=n`, the Chebyshev extreme
/// points mapped to `[-tau_max, 0]`.
pub(crate) struct Nodes {
    /// `theta[0] = 0` and `theta[n] = -tau_max`, both exact.
    pub(crate) theta: Vec<f64>,
    /// Barycentric weights, `(-1)^k` halved at both ends.
    w: Vec<f64>,
    /// `D[i][k] = l_k'(theta_i)`, row-major, `(n+1)^2` entries.
    d: Vec<f64>,
    tau_max: f64,
}

impl Nodes {
    /// Builds the `n + 1` nodes for `n >= 1` and `tau_max > 0`.
    pub(crate) fn chebyshev(n: usize, tau_max: f64) -> Nodes {
        let m = n + 1;
        let mut theta: Vec<f64> = (0..m)
            .map(|k| 0.5 * tau_max * ((k as f64 * std::f64::consts::PI / n as f64).cos() - 1.0))
            .collect();
        theta[0] = 0.0;
        theta[n] = -tau_max;
        let w: Vec<f64> = (0..m)
            .map(|k| {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                if k == 0 || k == n { 0.5 * sign } else { sign }
            })
            .collect();
        let d = differentiation(&theta, &w);
        Nodes {
            theta,
            w,
            d,
            tau_max,
        }
    }

    /// `D[i][k]`.
    pub(crate) fn diff(&self, i: usize, k: usize) -> f64 {
        self.d[i * self.theta.len() + k]
    }

    /// The values `l_0(s), ..., l_n(s)`. When `s` is within a few ulps of a node, the
    /// result is the unit vector of that node.
    pub(crate) fn lagrange_at(&self, s: f64) -> Vec<f64> {
        let m = self.theta.len();
        let snap = 8.0 * f64::EPSILON * self.tau_max;
        let mut out = vec![0.0; m];
        if let Some(i) = (0..m).find(|&i| (s - self.theta[i]).abs() <= snap) {
            out[i] = 1.0;
            return out;
        }
        let mut sum = 0.0;
        for (o, (w, th)) in out.iter_mut().zip(self.w.iter().zip(&self.theta)) {
            *o = w / (s - th);
            sum += *o;
        }
        for v in &mut out {
            *v /= sum;
        }
        out
    }
}

/// `D[i][k] = (w_k / w_i) / (x_i - x_k)` off the diagonal and the negative row sum on it
/// (Berrut and Trefethen 2004).
fn differentiation(x: &[f64], w: &[f64]) -> Vec<f64> {
    let m = x.len();
    let mut d = vec![0.0; m * m];
    for i in 0..m {
        let mut sum = 0.0;
        for k in 0..m {
            if k != i {
                let v = (w[k] / w[i]) / (x[i] - x[k]);
                d[i * m + k] = v;
                sum += v;
            }
        }
        d[i * m + i] = -sum;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_are_exact() {
        for n in [1usize, 2, 5, 12, 40] {
            let nodes = Nodes::chebyshev(n, 1.7);
            assert_eq!(nodes.theta[0], 0.0);
            assert_eq!(nodes.theta[n], -1.7);
            assert!(nodes.theta.windows(2).all(|p| p[0] > p[1]));
        }
    }

    #[test]
    fn rows_of_d_sum_to_zero() {
        let nodes = Nodes::chebyshev(9, 2.0);
        for i in 0..10 {
            let s: f64 = (0..10).map(|k| nodes.diff(i, k)).sum();
            assert!(s.abs() <= 1e-12, "row {i} sums to {s:e}");
        }
    }

    #[test]
    fn d_is_exact_on_polynomials_up_to_degree_n() {
        for n in 1..=12usize {
            let nodes = Nodes::chebyshev(n, 2.0);
            for p in 0..=n {
                let vals: Vec<f64> = nodes.theta.iter().map(|t| t.powi(p as i32)).collect();
                let exact: Vec<f64> = nodes
                    .theta
                    .iter()
                    .map(|t| {
                        if p == 0 {
                            0.0
                        } else {
                            p as f64 * t.powi(p as i32 - 1)
                        }
                    })
                    .collect();
                let scale = exact.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
                for (i, want) in exact.iter().enumerate() {
                    let got: f64 = (0..=n).map(|k| nodes.diff(i, k) * vals[k]).sum();
                    assert!(
                        (got - want).abs() <= 1e-9 * scale,
                        "n={n} p={p} i={i}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn closed_form_weights_match_the_product_formula() {
        let n = 5;
        let nodes = Nodes::chebyshev(n, 1.3);
        let x = &nodes.theta;
        let wp: Vec<f64> = (0..=n)
            .map(|k| {
                1.0 / (0..=n)
                    .filter(|&i| i != k)
                    .map(|i| x[k] - x[i])
                    .product::<f64>()
            })
            .collect();
        let dp = differentiation(x, &wp);
        for i in 0..=n {
            for k in 0..=n {
                let a = nodes.diff(i, k);
                let b = dp[i * (n + 1) + k];
                assert!(
                    (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                    "({i},{k}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn lagrange_basis_reproduces_polynomials_at_and_off_nodes() {
        let n = 8;
        let nodes = Nodes::chebyshev(n, 2.0);
        let mut points = vec![-0.3, -1.1, -1.999, -0.0001];
        points.push(nodes.theta[3]);
        points.push(nodes.theta[n]);
        points.push(-1.0 + 1e-17); // falls on the node theta[n/2] within the snap width
        for s in points {
            let l = nodes.lagrange_at(s);
            let total: f64 = l.iter().sum();
            assert!((total - 1.0).abs() <= 1e-13, "sum at {s}: {total}");
            for p in 1..=n {
                let got: f64 = l
                    .iter()
                    .zip(&nodes.theta)
                    .map(|(lk, t)| lk * t.powi(p as i32))
                    .sum();
                let exact = s.powi(p as i32);
                assert!(
                    (got - exact).abs() <= 1e-12 * (1.0 + exact.abs()),
                    "s={s} p={p}"
                );
            }
        }
        // At a node the basis is a unit vector.
        let at = nodes.lagrange_at(nodes.theta[n]);
        assert_eq!(at[n], 1.0);
        assert_eq!(at.iter().filter(|v| **v != 0.0).count(), 1);
    }
}
