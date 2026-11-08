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
    /// A workspace could not be allocated.
    OutOfMemory,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LinalgError::BadShape => "dimension mismatch",
            LinalgError::NotFinite => "NaN or infinity in matrix",
            LinalgError::Singular => "matrix is singular to working precision",
            LinalgError::NoConvergence => "factorization did not converge",
            LinalgError::OutOfMemory => "not enough memory for the linear solve",
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

/// Splits a finite nonzero `x` into `(m, e)` with 0.5 <= |m| < 1 and x = m * 2^e.
fn frexp(x: f64) -> (f64, i32) {
    // A subnormal has no implicit bit; scale it into the normal range first.
    let (x, shift) = if x.is_normal() {
        (x, 0)
    } else {
        (x * 2f64.powi(64), -64)
    };
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    // Replace the exponent field with that of 0.5 (biased 1022).
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (m, biased - 1022 + shift)
}

/// m * 2^e for 0.5 <= |m| < 1. Out of range, it saturates at +-f64::MAX or at
/// +-f64::MIN_POSITIVE, keeping the sign.
fn ldexp_saturating(m: f64, e: i64) -> f64 {
    if e > 1024 {
        f64::MAX.copysign(m)
    } else if e < -1021 {
        f64::MIN_POSITIVE.copysign(m)
    } else {
        // 2m is in [1, 2) and 2^(e-1) is a normal power of two, so the product is exact.
        let p = f64::from_bits(((e - 1 + 1023) as u64) << 52);
        (2.0 * m) * p
    }
}

/// LU factorization with partial (row) pivoting of a square matrix.
pub struct Lu {
    n: usize,
    lu: PartialPivLu<f64>,
    det: f64,
    min_pivot: f64,
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
                min_pivot: f64::INFINITY,
            });
        }
        let lu = PartialPivLu::new(MatRef::from_column_major_slice(a, n, n));
        let u = lu.U();
        let mut umin = f64::INFINITY;
        let mut umax = 0.0_f64;
        for i in 0..n {
            let d = u[(i, i)];
            umin = umin.min(d.abs());
            umax = umax.max(d.abs());
        }
        if umin == 0.0 || umin <= (n as f64) * f64::EPSILON * umax {
            return Err(LinalgError::Singular);
        }
        if (0..n).any(|i| !u[(i, i)].is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        // Product of the pivots as mantissa * 2^exp, so that no partial product
        // overflows or underflows. Scaling by a power of two is exact, so the result has
        // the same bits as a plain running product whenever that product stays normal.
        let mut mant = 1.0;
        let mut exp = 0i64;
        for i in 0..n {
            let (m, e) = frexp(u[(i, i)]);
            let (m, e2) = frexp(mant * m);
            mant = m;
            exp += i64::from(e) + i64::from(e2);
        }
        let mut det = ldexp_saturating(mant, exp);
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
        Ok(Lu {
            n,
            lu,
            det,
            min_pivot: umin,
        })
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

    /// Determinant, with the sign of the row swaps counted. A determinant outside the f64
    /// range saturates at +-f64::MAX or +-f64::MIN_POSITIVE with its sign, so it is never
    /// 0, infinite or NaN.
    pub fn det(&self) -> f64 {
        self.det
    }

    /// Smallest |u_ii| of the factorization: a bounded measure of the distance to a singular
    /// matrix, with the same zero as the determinant. `INFINITY` for n = 0.
    pub fn min_pivot(&self) -> f64 {
        self.min_pivot
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

    fn diag(d: &[f64]) -> Vec<f64> {
        let n = d.len();
        let mut a = vec![0.0; n * n];
        for (i, x) in d.iter().enumerate() {
            a[i * n + i] = *x;
        }
        a
    }

    #[test]
    fn det_out_of_range_keeps_its_sign() {
        // 60 pivots of 2^20 give 2^1200, and 60 of 2^-20 give 2^-1200. Both are outside
        // f64, so the determinant saturates at the largest finite or the smallest normal
        // magnitude and keeps its sign.
        let big = 2f64.powi(20);
        let mut d = vec![big; 60];
        assert_eq!(Lu::new(&diag(&d), 60).unwrap().det(), f64::MAX);
        d[7] = -big;
        assert_eq!(Lu::new(&diag(&d), 60).unwrap().det(), -f64::MAX);
        let mut d = vec![1.0 / big; 60];
        assert_eq!(Lu::new(&diag(&d), 60).unwrap().det(), f64::MIN_POSITIVE);
        d[0] = -1.0 / big;
        assert_eq!(Lu::new(&diag(&d), 60).unwrap().det(), -f64::MIN_POSITIVE);
    }

    #[test]
    fn min_pivot_is_the_smallest_pivot_magnitude() {
        let lu = Lu::new(&diag(&[1.0, 1.0, 1.0, 1.0]), 4).unwrap();
        assert_eq!(lu.min_pivot(), 1.0);
        let lu = Lu::new(&diag(&[2.0, 0.5, 3.0]), 3).unwrap();
        assert_eq!(lu.min_pivot(), 0.5);
        // A negative entry counts by its magnitude.
        let lu = Lu::new(&diag(&[2.0, -0.25, 3.0]), 3).unwrap();
        assert_eq!(lu.min_pivot(), 0.25);
        // No pivots, no smallest one.
        assert_eq!(Lu::new(&[], 0).unwrap().min_pivot(), f64::INFINITY);
        // The scaled diagonals keep their exact value where the determinant saturates.
        let big = Lu::new(&diag(&[1e200; 5]), 5).unwrap();
        assert_eq!(big.min_pivot(), 1e200);
        let small = Lu::new(&diag(&[1e-200; 5]), 5).unwrap();
        assert_eq!(small.min_pivot(), 1e-200);
        // A row swap does not change the magnitudes: [0 2; 3 0] has pivots 3 and 2.
        let lu = Lu::new(&[0.0, 2.0, 3.0, 0.0], 2).unwrap();
        assert_eq!(lu.min_pivot(), 2.0);
    }

    #[test]
    fn frexp_and_ldexp() {
        // faer forms the reciprocal of a pivot, which overflows for a subnormal pivot, so
        // Lu::new reports NotFinite there and the subnormal branch of frexp is tested here.
        let sub = f64::from_bits(1 << 44);
        assert!(sub > 0.0 && !sub.is_normal());
        assert_eq!(frexp(sub), (0.5, -1029));
        assert_eq!(frexp(-sub), (-0.5, -1029));
        assert_eq!(frexp(1.0), (0.5, 1));
        assert_eq!(frexp(-0.75), (-0.75, 0));
        assert_eq!(frexp(f64::MAX), (f64::MAX / 2f64.powi(1023) / 2.0, 1024));
        assert_eq!(ldexp_saturating(0.5, 1), 1.0);
        assert_eq!(ldexp_saturating(-0.75, 0), -0.75);
        let (m, e) = frexp(f64::MAX);
        assert_eq!(ldexp_saturating(m, i64::from(e)), f64::MAX);
        let (m, e) = frexp(f64::MIN_POSITIVE);
        assert_eq!(ldexp_saturating(m, i64::from(e)), f64::MIN_POSITIVE);
        assert_eq!(ldexp_saturating(0.5, 1025), f64::MAX);
        assert_eq!(ldexp_saturating(-0.5, -1022), -f64::MIN_POSITIVE);
        assert_eq!(
            Lu::new(&diag(&[sub, 2f64.powi(-1000)]), 2).err(),
            Some(LinalgError::NotFinite)
        );
    }

    #[test]
    fn det_partial_products_out_of_range() {
        // 60 pivots of 2^20 then 60 of 2^-20: a plain running product overflows to
        // infinity halfway, but the determinant is exactly 1.
        let big = 2f64.powi(20);
        let mut d = vec![big; 60];
        d.extend(std::iter::repeat_n(1.0 / big, 60));
        assert_eq!(Lu::new(&diag(&d), 120).unwrap().det(), 1.0);
        // The same in the other order underflows to 0 in a plain product.
        d.reverse();
        assert_eq!(Lu::new(&diag(&d), 120).unwrap().det(), 1.0);
    }

    #[test]
    fn det_in_range_matches_the_plain_product() {
        // 2^1000 and 2^-1000 fit in f64 and come out exactly.
        let big = 2f64.powi(20);
        let d = vec![big; 50];
        assert_eq!(Lu::new(&diag(&d), 50).unwrap().det(), 2f64.powi(1000));
        let d = vec![1.0 / big; 50];
        assert_eq!(Lu::new(&diag(&d), 50).unwrap().det(), 2f64.powi(-1000));
        // Rounded products give the same bits as a plain running product.
        let d = [1.1, 3.7, -0.3, 2.9, 1.0 / 3.0, 7.25];
        let plain = d.iter().product::<f64>();
        assert_eq!(
            Lu::new(&diag(&d), 6).unwrap().det().to_bits(),
            plain.to_bits()
        );
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
