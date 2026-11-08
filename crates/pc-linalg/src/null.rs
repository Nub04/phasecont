//! Singular values and null vectors through the SVD.

use crate::dense::{LinalgError, check};
use faer::MatRef;
use faer::linalg::solvers::Svd;

fn svd(a: &[f64], rows: usize, cols: usize) -> Result<Svd<f64>, LinalgError> {
    check(a, rows, cols)?;
    if rows == 0 || cols == 0 {
        return Err(LinalgError::BadShape);
    }
    Svd::new(MatRef::from_column_major_slice(a, rows, cols)).map_err(|_| LinalgError::NoConvergence)
}

/// Singular values of the rows-by-cols column-major matrix `a`, descending. There are
/// min(rows, cols) of them.
pub fn singular_values(a: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, LinalgError> {
    let s = svd(a, rows, cols)?;
    let d = s.S().column_vector();
    Ok((0..rows.min(cols)).map(|i| d[i]).collect())
}

/// The k right singular vectors (length `cols`, unit norm) of the k smallest singular
/// values, ordered from the smallest singular value up.
///
/// A matrix with more columns than rows has cols - rows extra zero singular values, and
/// their vectors count among the k. `BadShape` when k > cols.
pub fn null_right(
    a: &[f64],
    rows: usize,
    cols: usize,
    k: usize,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    if k > cols {
        return Err(LinalgError::BadShape);
    }
    let s = svd(a, rows, cols)?;
    // Full SVD: V is cols-by-cols with columns ordered by descending singular value.
    let v = s.V();
    Ok((0..k)
        .map(|t| (0..cols).map(|i| v[(i, cols - 1 - t)]).collect())
        .collect())
}

/// The k left singular vectors (length `rows`, unit norm) of the k smallest singular
/// values, ordered from the smallest singular value up. `BadShape` when k > rows.
pub fn null_left(
    a: &[f64],
    rows: usize,
    cols: usize,
    k: usize,
) -> Result<Vec<Vec<f64>>, LinalgError> {
    if k > rows {
        return Err(LinalgError::BadShape);
    }
    let s = svd(a, rows, cols)?;
    let u = s.U();
    Ok((0..k)
        .map(|t| (0..rows).map(|i| u[(i, rows - 1 - t)]).collect())
        .collect())
}

/// Singular values and null vectors from one SVD. See [`svd_null`].
#[derive(Debug, Clone)]
pub struct SvdNull {
    /// The min(rows, cols) singular values, descending.
    pub values: Vec<f64>,
    /// The right vectors of the smallest singular values, smallest first, as `null_right`.
    pub right: Vec<Vec<f64>>,
    /// The left vectors of the smallest singular values, smallest first, as `null_left`.
    pub left: Vec<Vec<f64>>,
}

/// The singular values, the `k_right` right null vectors and the `k_left` left null vectors
/// of `a` from a single SVD. `BadShape` when `k_right > cols` or `k_left > rows`.
///
/// A caller that needs the values and a kernel pays for the decomposition once instead of
/// two or three times. The vectors are those of `null_right` and `null_left`.
pub fn svd_null(
    a: &[f64],
    rows: usize,
    cols: usize,
    k_right: usize,
    k_left: usize,
) -> Result<SvdNull, LinalgError> {
    if k_right > cols || k_left > rows {
        return Err(LinalgError::BadShape);
    }
    let s = svd(a, rows, cols)?;
    let d = s.S().column_vector();
    let values = (0..rows.min(cols)).map(|i| d[i]).collect();
    let v = s.V();
    let right = (0..k_right)
        .map(|t| (0..cols).map(|i| v[(i, cols - 1 - t)]).collect())
        .collect();
    let u = s.U();
    let left = (0..k_left)
        .map(|t| (0..rows).map(|i| u[(i, rows - 1 - t)]).collect())
        .collect();
    Ok(SvdNull {
        values,
        right,
        left,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// max |A v| (right) or max |A^T u| (left).
    fn residual(a: &[f64], rows: usize, cols: usize, v: &[f64], left: bool) -> f64 {
        if left {
            (0..cols)
                .map(|j| (0..rows).map(|i| a[i + rows * j] * v[i]).sum::<f64>().abs())
                .fold(0.0, f64::max)
        } else {
            (0..rows)
                .map(|i| (0..cols).map(|j| a[i + rows * j] * v[j]).sum::<f64>().abs())
                .fold(0.0, f64::max)
        }
    }

    #[test]
    fn rank_one_square() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let sv = singular_values(&a, 2, 2).unwrap();
        assert!((sv[0] - 2.0).abs() < 1e-14 && sv[1].abs() < 1e-14);
        for left in [false, true] {
            let v = if left {
                null_left(&a, 2, 2, 1).unwrap()
            } else {
                null_right(&a, 2, 2, 1).unwrap()
            };
            assert_eq!(v.len(), 1);
            assert!((norm(&v[0]) - 1.0).abs() < 1e-14);
            assert!(residual(&a, 2, 2, &v[0], left) <= 1e-14);
        }
    }

    #[test]
    fn rank_one_2x3_has_two_null_vectors() {
        // Rows [1 2 3] and [2 4 6], column-major.
        let a = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
        let v = null_right(&a, 2, 3, 2).unwrap();
        assert_eq!(v.len(), 2);
        for x in &v {
            assert_eq!(x.len(), 3);
            assert!((norm(x) - 1.0).abs() < 1e-14);
            assert!(residual(&a, 2, 3, x, false) <= 1e-14);
        }
        let dot: f64 = v[0].iter().zip(&v[1]).map(|(p, q)| p * q).sum();
        assert!(dot.abs() < 1e-14);
        let u = null_left(&a, 2, 3, 1).unwrap();
        assert_eq!(u[0].len(), 2);
        assert!(residual(&a, 2, 3, &u[0], true) <= 1e-14);
        assert_eq!(singular_values(&a, 2, 3).unwrap().len(), 2);
    }

    #[test]
    fn null_right_orders_from_the_smallest_singular_value() {
        // diag(5, 3, 0.5) mixed by a rotation on the left: right vectors are the unit
        // vectors, singular values 5, 3, 0.5.
        let (c, s) = (0.6_f64, 0.8_f64);
        let d = [5.0, 3.0, 0.5];
        let a = [
            c * d[0],
            s * d[0],
            0.0,
            -s * d[1],
            c * d[1],
            0.0,
            0.0,
            0.0,
            d[2],
        ];
        let sv = singular_values(&a, 3, 3).unwrap();
        for (x, y) in sv.iter().zip(&d) {
            assert!((x - y).abs() < 1e-13, "{sv:?}");
        }
        let v = null_right(&a, 3, 3, 3).unwrap();
        for (t, k) in [2usize, 1, 0].iter().enumerate() {
            assert!((v[t][*k].abs() - 1.0).abs() < 1e-13, "{v:?}");
            let av: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|j| a[i + 3 * j] * v[t][j]).sum::<f64>())
                .collect();
            assert!((norm(&av) - d[*k]).abs() < 1e-13);
        }
        // The first vector alone is the one of the smallest value.
        let one = null_right(&a, 3, 3, 1).unwrap();
        assert!((one[0][2].abs() - 1.0).abs() < 1e-13);
    }

    #[test]
    fn rank_deficient_n_by_n_plus_one_spans_the_kernel() {
        // Rows [1 0 0 1], [0 1 0 1], [1 1 0 2] (rank 2, 3 by 4, column-major). The kernel is
        // span{(0,0,1,0), (1,1,0,-1)}.
        let a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0];
        let v = null_right(&a, 3, 4, 2).unwrap();
        assert_eq!(v.len(), 2);
        let k1 = [0.0, 0.0, 1.0, 0.0];
        let k2 = [1.0, 1.0, 0.0, -1.0].map(|x| x / 3.0_f64.sqrt());
        let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(x, y)| x * y).sum::<f64>();
        for x in &v {
            assert!(residual(&a, 3, 4, x, false) <= 1e-12);
            assert!((norm(x) - 1.0).abs() <= 1e-12);
            // Inside the kernel: the projection on its orthonormal basis keeps the norm.
            let proj = dot(x, &k1).powi(2) + dot(x, &k2).powi(2);
            assert!((proj - 1.0).abs() <= 1e-12, "{proj}");
        }
        assert!(dot(&v[0], &v[1]).abs() <= 1e-12);
    }

    /// A deterministic rank-4 5 by 6 matrix (column-major): a 5 by 4 factor times a 4 by 6
    /// factor, both from a fixed linear congruential sequence.
    fn rank4_5x6() -> Vec<f64> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let l: Vec<f64> = (0..5 * 4).map(|_| next()).collect();
        let r: Vec<f64> = (0..4 * 6).map(|_| next()).collect();
        let mut a = vec![0.0; 5 * 6];
        for i in 0..5 {
            for j in 0..6 {
                a[i + 5 * j] = (0..4).map(|k| l[i + 5 * k] * r[k + 4 * j]).sum();
            }
        }
        a
    }

    /// The projector `sum v v^T` of the vectors, as a dense `len` by `len` array.
    fn projector(vs: &[Vec<f64>]) -> Vec<f64> {
        let len = vs[0].len();
        let mut p = vec![0.0; len * len];
        for v in vs {
            for i in 0..len {
                for j in 0..len {
                    p[i + len * j] += v[i] * v[j];
                }
            }
        }
        p
    }

    #[test]
    fn one_svd_gives_the_values_and_both_kernels() {
        let a = rank4_5x6();
        let one = svd_null(&a, 5, 6, 2, 1).unwrap();
        let sv = singular_values(&a, 5, 6).unwrap();
        assert_eq!(one.values.len(), 5);
        for (x, y) in one.values.iter().zip(&sv) {
            assert!((x - y).abs() <= 1e-14, "{x} {y}");
        }
        // Rank 4: the fifth value is rounding level, the fourth is not.
        assert!(one.values[4] <= 1e-14 * one.values[0], "{:?}", one.values);
        assert!(one.values[3] > 1e-3 * one.values[0], "{:?}", one.values);
        assert_eq!(one.right.len(), 2);
        assert_eq!(one.left.len(), 1);
        let right = null_right(&a, 5, 6, 2).unwrap();
        let left = null_left(&a, 5, 6, 1).unwrap();
        // The kernels agree through their projectors, which do not depend on the basis.
        for (p, q) in projector(&one.right)
            .iter()
            .zip(&projector(&right))
            .chain(projector(&one.left).iter().zip(&projector(&left)))
        {
            assert!((p - q).abs() <= 1e-12, "{p} {q}");
        }
        for x in &one.right {
            assert!((norm(x) - 1.0).abs() <= 1e-13);
            assert!(residual(&a, 5, 6, x, false) <= 1e-13);
        }
        assert!(residual(&a, 5, 6, &one.left[0], true) <= 1e-13);
    }

    #[test]
    fn svd_null_asks_for_no_more_than_the_matrix_has() {
        let a = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(svd_null(&a, 2, 2, 3, 0).err(), Some(LinalgError::BadShape));
        assert_eq!(svd_null(&a, 2, 2, 0, 3).err(), Some(LinalgError::BadShape));
        let none = svd_null(&a, 2, 2, 0, 0).unwrap();
        assert!(none.right.is_empty() && none.left.is_empty());
        assert_eq!(none.values.len(), 2);
    }

    #[test]
    fn errors() {
        let a = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(null_right(&a, 2, 2, 3).err(), Some(LinalgError::BadShape));
        assert_eq!(null_left(&a, 2, 2, 3).err(), Some(LinalgError::BadShape));
        assert_eq!(singular_values(&a, 2, 3).err(), Some(LinalgError::BadShape));
        assert_eq!(
            singular_values(&[f64::INFINITY], 1, 1).err(),
            Some(LinalgError::NotFinite)
        );
    }
}
