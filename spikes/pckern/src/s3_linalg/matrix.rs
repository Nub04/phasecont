//! Builds the S3 test matrix (NOTES.md, "S3 design"): Gauss-Legendre points,
//! the Lagrange tables, one block per mesh interval, the periodicity, phase
//! and PALC rows and the right-hand side. Entries are visited in row order.

use crate::ffi::{Args, Result, bad_arg, internal};
use crate::rng::SplitMix64;

/// Keys that `s3_build` accepts.
pub const BUILD_KEYS: [&str; 5] = ["n", "m", "N", "T", "seed"];

/// Matrix arguments after the range checks.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub n: usize,
    pub m: usize,
    pub big_n: usize,
    pub t: f64,
    pub seed: u64,
}

/// The matrix and right-hand side. Unknowns are u_0 .. u_{Nm} (n each), then
/// T and alpha. Block j is row-major with `rows` rows and `cols` columns.
pub struct Problem {
    pub n: usize,
    pub m: usize,
    pub big_n: usize,
    pub nu: usize,
    pub nr: usize,
    pub nnz: usize,
    pub rows: usize,
    pub cols: usize,
    pub blocks: Vec<f64>,
    pub phase: Vec<f64>,
    pub palc: Vec<f64>,
    pub b: Vec<f64>,
}

/// Reads an integer argument and checks that it lies in lo..=hi.
pub fn ranged(a: &Args, key: &str, default: usize, lo: usize, hi: usize) -> Result<usize> {
    let v = a.usize_or(key, default)?;
    if v < lo || v > hi {
        return Err(bad_arg(format!("{key}={v} is out of range {lo}..={hi}")));
    }
    Ok(v)
}

/// Reads T, which must be finite and positive. `Args` has no f64 helper.
fn period(a: &Args) -> Result<f64> {
    let text = a.str_or("T", "1");
    let t: f64 = text
        .parse()
        .map_err(|_| bad_arg(format!("T={text} is not a number")))?;
    if !t.is_finite() || t <= 0.0 {
        return Err(bad_arg(format!(
            "T={text} must be finite and greater than 0"
        )));
    }
    Ok(t)
}

impl Params {
    /// Reads n, m, N, T and seed from arguments parsed by the caller.
    pub fn from_args(a: &Args) -> Result<Params> {
        Ok(Params {
            n: ranged(a, "n", 4, 1, 32)?,
            m: ranged(a, "m", 4, 2, 10)?,
            big_n: ranged(a, "N", 40, 1, 20000)?,
            t: period(a)?,
            seed: a.u64_or("seed", 2026)?,
        })
    }
}

/// P_m(x) and P_m'(x) from the three-term recurrence.
fn legendre(m: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=m {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    (p1, m as f64 * (p0 - x * p1) / (1.0 - x * x))
}

/// Gauss-Legendre nodes and weights on [0, 1], sorted ascending. Newton's
/// method finds each root of P_m from cos(pi (i + 0.75) / (m + 0.5)).
pub fn gauss_legendre(m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(m);
    for i in 0..m {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, d) = legendre(m, x);
            let next = x - p / d;
            let done = (next - x).abs() <= 1e-15;
            x = next;
            if done {
                break;
            }
        }
        let (_, d) = legendre(m, x);
        pts.push(((1.0 + x) / 2.0, 1.0 / ((1.0 - x * x) * d * d)));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    pts.into_iter().unzip()
}

/// L[k][l] = l_l(z_k) and D[k][l] = l_l'(z_k) for the points s_l = l/m.
pub fn lagrange(m: usize, z: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let s: Vec<f64> = (0..=m).map(|l| l as f64 / m as f64).collect();
    let mut lt = vec![vec![0.0; m + 1]; z.len()];
    let mut dt = lt.clone();
    for (k, &zk) in z.iter().enumerate() {
        for l in 0..=m {
            let (mut val, mut der) = (1.0, 0.0);
            for i in (0..=m).filter(|&i| i != l) {
                let f = (zk - s[i]) / (s[l] - s[i]);
                der = der * f + val / (s[l] - s[i]);
                val *= f;
            }
            lt[k][l] = val;
            dt[k][l] = der;
        }
    }
    (lt, dt)
}

/// Stored entries: collocation blocks, periodicity, phase and PALC rows.
pub fn nnz_formula(n: usize, m: usize, big_n: usize) -> usize {
    let nu = n * (big_n * m + 1);
    big_n * n * m * (n * (m + 1) + 2) + 2 * n + nu + (nu + 2)
}

/// A zeroed vector, or an error instead of an abort when memory runs out.
pub fn zeros(len: usize, what: &str) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| internal(format!("s3: cannot allocate {len} values for {what}")))?;
    v.resize(len, 0.0);
    Ok(v)
}
impl Problem {
    /// Draws the matrix and right-hand side from SplitMix64(seed). For each
    /// interval j and Gauss point k the draws are A_jk (n x n, row-major),
    /// the T and alpha entries of each of its n rows, then udot (n values).
    /// The PALC row and the right-hand side come after all intervals.
    pub fn build(p: &Params) -> Result<Problem> {
        let (n, m, big_n) = (p.n, p.m, p.big_n);
        let h = 1.0 / big_n as f64;
        let nu = n * (big_n * m + 1);
        let nr = nu + 2;
        let (rows, cols) = (n * m, n * (m + 1) + 2);
        let (z, w) = gauss_legendre(m);
        let (lt, dt) = lagrange(m, &z);
        let mut rng = SplitMix64::new(p.seed);
        let mut blocks = zeros(big_n * rows * cols, "the blocks")?;
        let mut phase = zeros(nu, "the phase row")?;
        let mut a = vec![0.0; n * n];
        for (j, blk) in blocks.chunks_exact_mut(rows * cols).enumerate() {
            for k in 0..m {
                for (idx, v) in a.iter_mut().enumerate() {
                    let eye = if idx / n == idx % n { 1.0 } else { 0.0 };
                    *v = -eye + 0.3 * rng.range(-1.0, 1.0);
                }
                for r in 0..n {
                    let row = &mut blk[(k * n + r) * cols..(k * n + r + 1) * cols];
                    for l in 0..=m {
                        for c in 0..n {
                            let d = if r == c { dt[k][l] } else { 0.0 };
                            row[l * n + c] = d - h * p.t * lt[k][l] * a[r * n + c];
                        }
                    }
                    row[cols - 2] = h * rng.range(-1.0, 1.0);
                    row[cols - 1] = h * rng.range(-1.0, 1.0);
                }
                for c in 0..n {
                    let udot = rng.range(-1.0, 1.0);
                    for l in 0..=m {
                        phase[(j * m + l) * n + c] += h * w[k] * lt[k][l] * udot;
                    }
                }
            }
        }
        let mut palc = zeros(nr, "the PALC row")?;
        palc.iter_mut().for_each(|v| *v = rng.range(-1.0, 1.0));
        let norm = palc.iter().map(|v| v * v).sum::<f64>().sqrt();
        palc.iter_mut().for_each(|v| *v /= norm);
        let mut b = zeros(nr, "the right-hand side")?;
        b.iter_mut().for_each(|v| *v = rng.range(-1.0, 1.0));
        Ok(Problem {
            n,
            m,
            big_n,
            nu,
            nr,
            nnz: nnz_formula(n, m, big_n),
            rows,
            cols,
            blocks,
            phase,
            palc,
            b,
        })
    }

    /// Order of the reduced system: the mesh values, T and alpha.
    pub fn nred(&self) -> usize {
        self.n * (self.big_n + 1) + 2
    }

    /// Block j, row-major.
    pub fn block(&self, j: usize) -> &[f64] {
        let len = self.rows * self.cols;
        &self.blocks[j * len..(j + 1) * len]
    }

    /// Entry of border row k (0 phase, 1 PALC) in column `col`.
    pub fn border(&self, k: usize, col: usize) -> f64 {
        match k {
            0 => self.phase.get(col).copied().unwrap_or(0.0),
            _ => self.palc[col],
        }
    }

    /// Visits every stored entry as (row, column, value), rows ascending and
    /// no pair twice.
    pub fn for_each_entry(&self, mut f: impl FnMut(usize, usize, f64)) {
        let (n, m, nu) = (self.n, self.m, self.nu);
        let ncol = n * (m + 1);
        for j in 0..self.big_n {
            let base = j * m * n;
            for (rr, row) in self.block(j).chunks_exact(self.cols).enumerate() {
                for (bc, &v) in row[..ncol].iter().enumerate() {
                    f(base + rr, base + bc, v);
                }
                f(base + rr, nu, row[ncol]);
                f(base + rr, nu + 1, row[ncol + 1]);
            }
        }
        let last = self.big_n * m * n;
        for c in 0..n {
            f(last + c, c, 1.0);
            f(last + c, last + c, -1.0);
        }
        for (col, &v) in self.phase.iter().enumerate() {
            f(nu, col, v);
        }
        for (col, &v) in self.palc.iter().enumerate() {
            f(nu + 1, col, v);
        }
    }

    /// ‖Ax - b‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞), or NaN when x is not finite.
    pub fn backward_error(&self, x: &[f64]) -> f64 {
        if x.len() != self.nr || x.iter().any(|v| !v.is_finite()) {
            return f64::NAN;
        }
        let mut r: Vec<f64> = self.b.iter().map(|v| -v).collect();
        let mut row_abs = vec![0.0; self.nr];
        self.for_each_entry(|i, j, v| {
            r[i] += v * x[j];
            row_abs[i] += v.abs();
        });
        let inf = |v: &[f64]| v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        inf(&r) / (inf(&row_abs) * inf(x) + inf(&self.b))
    }

    /// ‖A‖₁, the largest column sum of absolute values.
    pub fn norm1(&self) -> f64 {
        let mut col_abs = vec![0.0; self.nr];
        self.for_each_entry(|_, j, v| col_abs[j] += v.abs());
        col_abs.iter().fold(0.0f64, |a, &x| a.max(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(n: usize, m: usize, big_n: usize) -> Params {
        Params {
            n,
            m,
            big_n,
            t: 1.0,
            seed: 2026,
        }
    }

    #[test]
    fn gauss_legendre_m4_matches_the_table() {
        let (z, w) = gauss_legendre(4);
        let x = [
            -0.861136311594053,
            -0.339981043584856,
            0.339981043584856,
            0.861136311594053,
        ];
        let wx = [
            0.347854845137454,
            0.652145154862546,
            0.652145154862546,
            0.347854845137454,
        ];
        for k in 0..4 {
            assert!((z[k] - (1.0 + x[k]) / 2.0).abs() < 1e-14, "z[{k}]={}", z[k]);
            assert!((w[k] - wx[k] / 2.0).abs() < 1e-14, "w[{k}]={}", w[k]);
        }
    }

    #[test]
    fn gauss_legendre_is_exact_to_degree_2m_minus_1() {
        for m in 2..=10 {
            let (z, w) = gauss_legendre(m);
            assert!(z.windows(2).all(|p| p[0] < p[1]), "m={m}: not sorted");
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14, "m={m}");
            for d in 0..2 * m {
                let q: f64 = z.iter().zip(&w).map(|(z, w)| w * z.powi(d as i32)).sum();
                let exact = 1.0 / (d as f64 + 1.0);
                assert!((q - exact).abs() < 1e-14, "m={m} d={d}: {q} vs {exact}");
            }
        }
    }

    #[test]
    fn lagrange_tables_satisfy_the_identities() {
        for m in 2..=10 {
            let (z, _) = gauss_legendre(m);
            let (l, d) = lagrange(m, &z);
            let s: Vec<f64> = (0..=m).map(|i| i as f64 / m as f64).collect();
            for k in 0..m {
                let scale: f64 = 1.0 + d[k].iter().map(|v| v.abs()).sum::<f64>();
                let tol = 1e-13 * scale;
                let sum_l: f64 = l[k].iter().sum();
                let sum_d: f64 = d[k].iter().sum();
                let sum_ds: f64 = (0..=m).map(|i| d[k][i] * s[i]).sum();
                let sum_dsm: f64 = (0..=m).map(|i| d[k][i] * s[i].powi(m as i32)).sum();
                let want = m as f64 * z[k].powi(m as i32 - 1);
                assert!((sum_l - 1.0).abs() < 1e-13, "m={m} k={k}: sum L = {sum_l}");
                assert!(sum_d.abs() < tol, "m={m} k={k}: sum D = {sum_d}");
                assert!(
                    (sum_ds - 1.0).abs() < tol,
                    "m={m} k={k}: sum D s = {sum_ds}"
                );
                assert!(
                    (sum_dsm - want).abs() < tol,
                    "m={m} k={k}: {sum_dsm} vs {want}"
                );
            }
        }
    }

    #[test]
    fn nnz_formula_matches_the_contract_values() {
        assert_eq!(nnz_formula(4, 4, 40), 15378);
        assert_eq!(nnz_formula(4, 4, 400), 153618);
    }

    #[test]
    fn entries_come_in_row_order_without_duplicates() {
        for (n, m, big_n) in [(4, 4, 40), (4, 4, 400), (1, 2, 1), (3, 5, 7), (2, 10, 3)] {
            let p = Problem::build(&params(n, m, big_n)).unwrap();
            assert_eq!(p.nr, n * (big_n * m + 1) + 2);
            let mut count = 0usize;
            let mut last_row = 0usize;
            let mut cols_in_row: Vec<usize> = Vec::new();
            let mut ok = true;
            p.for_each_entry(|r, c, v| {
                count += 1;
                ok &= r < p.nr && c < p.nr && v.is_finite();
                if r != last_row {
                    ok &= r > last_row;
                    last_row = r;
                    cols_in_row.clear();
                }
                ok &= !cols_in_row.contains(&c);
                cols_in_row.push(c);
            });
            assert!(
                ok,
                "n={n} m={m} N={big_n}: order, range or duplicate failure"
            );
            assert_eq!(count, nnz_formula(n, m, big_n), "n={n} m={m} N={big_n}");
            assert_eq!(count, p.nnz);
            assert_eq!(last_row, p.nr - 1);
        }
    }
}
