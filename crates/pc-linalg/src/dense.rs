//! Error type and dense LU with partial pivoting.

use faer::MatRef;
use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::mat::MatMut;

/// Failure of a dense linear algebra call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgError {
    /// A slice length does not match the stated dimensions, or k exceeds the dimension.
    BadShape,
    /// The input contains NaN or infinity, or the result does.
    NotFinite,
    /// The matrix is singular to working precision.
    Singular,
    /// An iterative factorization (eigenvalues, SVD) did not converge.
    NoConvergence,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LinalgError::BadShape => "dimension mismatch",
            LinalgError::NotFinite => "NaN or infinity in matrix",
            LinalgError::Singular => "matrix is singular to working precision",
            LinalgError::NoConvergence => "factorization did not converge",
        })
    }
}

impl std::error::Error for LinalgError {}

/// Checks `a.len() == rows * cols` (without overflow) and that every entry is finite.
pub(crate) fn check(a: &[f64], rows: usize, cols: usize) -> Result<(), LinalgError> {
    if rows.checked_mul(cols) != Some(a.len()) {
        return Err(LinalgError::BadShape);
    }
    if a.iter().any(|x| !x.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    Ok(())
}

/// LU factorization with partial (row) pivoting of a square matrix.
pub struct Lu {
    n: usize,
    lu: PartialPivLu<f64>,
    det: f64,
}

impl Lu {
    /// Factors the n-by-n column-major matrix `a`.
    ///
    /// Returns `Singular` when a pivot is 0 or min |u_ii| <= n * EPSILON * max |u_ii|.
    ///
    /// At a branch point the bordered tangent matrix of a continuation is singular, and the
    /// threshold is relative, so the result there is `Singular` and not a tiny determinant.
    /// Callers on a branch treat `Singular` from that matrix as a zero of the branch-point
    /// test function, not as a failure of the run.
    pub fn new(a: &[f64], n: usize) -> Result<Lu, LinalgError> {
        let len = n.checked_mul(n).ok_or(LinalgError::BadShape)?;
        if a.len() != len {
            return Err(LinalgError::BadShape);
        }
        check(a, n, n)?;
        if n == 0 {
            return Ok(Lu {
                n,
                lu: PartialPivLu::new(MatRef::from_column_major_slice(a, 0, 0)),
                det: 1.0,
            });
        }
        let lu = PartialPivLu::new(MatRef::from_column_major_slice(a, n, n));
        let u = lu.U();
        let mut umin = f64::INFINITY;
        let mut umax = 0.0_f64;
        let mut det = 1.0;
        for i in 0..n {
            let d = u[(i, i)];
            umin = umin.min(d.abs());
            umax = umax.max(d.abs());
            det *= d;
        }
        if umin == 0.0 || umin <= (n as f64) * f64::EPSILON * umax {
            return Err(LinalgError::Singular);
        }
        // Sign of the row permutation: (-1)^(n - number of cycles).
        let (fwd, _) = lu.P().arrays();
        let mut seen = vec![false; n];
        let mut cycles = 0usize;
        for start in 0..n {
            if !seen[start] {
                cycles += 1;
                let mut j = start;
                while !seen[j] {
                    seen[j] = true;
                    j = fwd[j];
                }
            }
        }
        if (n - cycles) % 2 == 1 {
            det = -det;
        }
        Ok(Lu { n, lu, det })
    }

    /// Overwrites `b` with A^-1 b. `BadShape` when `b.len() != n`.
    pub fn solve(&self, b: &mut [f64]) -> Result<(), LinalgError> {
        self.check_rhs(b)?;
        if self.n > 0 {
            self.lu
                .solve_in_place(MatMut::from_column_major_slice_mut(b, self.n, 1));
        }
        Ok(())
    }

    /// Overwrites `b` with A^-T b. `BadShape` when `b.len() != n`.
    pub fn solve_transpose(&self, b: &mut [f64]) -> Result<(), LinalgError> {
        self.check_rhs(b)?;
        if self.n > 0 {
            self.lu
                .solve_transpose_in_place(MatMut::from_column_major_slice_mut(b, self.n, 1));
        }
        Ok(())
    }

    /// Determinant, with the sign of the row swaps counted.
    pub fn det(&self) -> f64 {
        self.det
    }

    fn check_rhs(&self, b: &[f64]) -> Result<(), LinalgError> {
        if b.len() == self.n {
            Ok(())
        } else {
            Err(LinalgError::BadShape)
        }
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    /// SplitMix64 generator for tests.
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [-1, 1).
        pub fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::Rng;
    use super::*;

    fn matvec(a: &[f64], n: usize, x: &[f64], transpose: bool) -> Vec<f64> {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if transpose { a[j + n * i] } else { a[i + n * j] } * x[j])
                    .sum()
            })
            .collect()
    }

    #[test]
    fn residuals_random_well_conditioned() {
        let mut rng = Rng(42);
        for n in 1..=20 {
            // Diagonally dominant, so well conditioned.
            let mut a: Vec<f64> = (0..n * n).map(|_| rng.unit()).collect();
            for i in 0..n {
                a[i + n * i] += n as f64 + 1.0;
            }
            let b: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
            let bn = b.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            let lu = Lu::new(&a, n).unwrap();
            for transpose in [false, true] {
                let mut x = b.clone();
                if transpose {
                    lu.solve_transpose(&mut x).unwrap();
                } else {
                    lu.solve(&mut x).unwrap();
                }
                let ax = matvec(&a, n, &x, transpose);
                let r = ax
                    .iter()
                    .zip(&b)
                    .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
                assert!(r <= 1e-12 * bn, "n={n} transpose={transpose} r={r}");
            }
        }
    }

    #[test]
    fn det_sign_on_permutations() {
        // Transposition (one swap): det -1. 3-cycle: det +1. Identity: +1.
        let swap = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(Lu::new(&swap, 2).unwrap().det(), -1.0);
        // Column-major 3-cycle: e0 -> e1 -> e2 -> e0.
        let cyc = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq!(Lu::new(&cyc, 3).unwrap().det(), 1.0);
        // Swap of rows 0 and 2 of a 3x3 identity.
        let p = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(Lu::new(&p, 3).unwrap().det(), -1.0);
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(Lu::new(&id, 3).unwrap().det(), 1.0);
    }

    #[test]
    fn det_hand_3x3() {
        // Rows: [2 0 1; 1 3 2; 1 1 1]. det = 2(3-2) - 0 + 1(1-3) = 0 is singular, so use
        // rows [2 0 1; 1 3 2; 1 1 4]: det = 2(12-2) - 0 + 1(1-3) = 18.
        let a = [2.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0, 2.0, 4.0];
        let d = Lu::new(&a, 3).unwrap().det();
        assert!((d - 18.0).abs() < 1e-13, "{d}");
        // First pivot forces a swap: rows [0 1; 1 0] scaled, det = -6.
        let b = [0.0, 2.0, 3.0, 0.0];
        let d = Lu::new(&b, 2).unwrap().det();
        assert!((d + 6.0).abs() < 1e-13, "{d}");
    }

    #[test]
    fn singular_and_shape_errors() {
        let rank1 = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        assert_eq!(Lu::new(&rank1, 3).err(), Some(LinalgError::Singular));
        assert_eq!(Lu::new(&[0.0; 4], 2).err(), Some(LinalgError::Singular));
        assert_eq!(Lu::new(&[1.0; 3], 2).err(), Some(LinalgError::BadShape));
        assert_eq!(
            Lu::new(&[1.0, f64::NAN, 0.0, 1.0], 2).err(),
            Some(LinalgError::NotFinite)
        );
        let lu = Lu::new(&[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        assert_eq!(lu.solve(&mut [1.0; 3]), Err(LinalgError::BadShape));
        assert_eq!(lu.solve_transpose(&mut [1.0]), Err(LinalgError::BadShape));
    }
}
