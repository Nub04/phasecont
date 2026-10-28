//! Complex shifted solve through a real block system.

use crate::dense::{LinalgError, Lu, check};

/// Solves (A - sI) x = b for s = sigma + i omega, with b = b_re + i b_im, and returns
/// `(x_re, x_im)`.
///
/// Writing x = x_re + i x_im, (A - sI) x = b is the real 2n-by-2n system
///
/// ```text
/// [ A - sigma I     omega I     ] [ x_re ]   [ b_re ]
/// [ -omega I      A - sigma I   ] [ x_im ] = [ b_im ]
/// ```
///
/// which is factored with [`Lu`]. `Singular` when s is an eigenvalue to working precision.
pub fn solve_shifted(
    a: &[f64],
    n: usize,
    shift: (f64, f64),
    b_re: &[f64],
    b_im: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    check(a, n, n)?;
    if b_re.len() != n || b_im.len() != n {
        return Err(LinalgError::BadShape);
    }
    let (sigma, omega) = shift;
    if !sigma.is_finite() || !omega.is_finite() || b_re.iter().chain(b_im).any(|x| !x.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let m = 2 * n;
    let mut blk = vec![0.0; m * m];
    for j in 0..n {
        for i in 0..n {
            let v = a[i + n * j];
            blk[i + m * j] = v;
            blk[(i + n) + m * (j + n)] = v;
        }
        blk[j + m * j] -= sigma;
        blk[(j + n) + m * (j + n)] -= sigma;
        blk[j + m * (j + n)] = omega;
        blk[(j + n) + m * j] = -omega;
    }
    let lu = Lu::new(&blk, m)?;
    let mut x: Vec<f64> = b_re.iter().chain(b_im).copied().collect();
    lu.solve(&mut x)?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let x_im = x.split_off(n);
    Ok((x, x_im))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_2x2_complex() {
        // A = [[1, 2], [3, 4]], s = 1 + i, so A - sI = [[-i, 2], [3, 3 - i]].
        // x = (1 + i, 2 - i) gives b = ( -i(1+i) + 2(2-i), 3(1+i) + (3-i)(2-i) )
        //   = ( 1 - i + 4 - 2i, 3 + 3i + 6 - 3i - 2i + i^2 ) = ( 5 - 3i, 2 + ... )
        // (3 - i)(2 - i) = 6 - 3i - 2i + i^2 = 5 - 5i, so b2 = 3 + 3i + 5 - 5i = 8 - 2i.
        let a = [1.0, 3.0, 2.0, 4.0];
        let (xr, xi) = solve_shifted(&a, 2, (1.0, 1.0), &[5.0, 8.0], &[-3.0, -2.0]).unwrap();
        for (got, want) in xr.iter().chain(&xi).zip([1.0, 2.0, 1.0, -1.0]) {
            assert!((got - want).abs() < 1e-13, "{got} vs {want}");
        }
    }

    #[test]
    fn errors() {
        let a = [1.0, 0.0, 0.0, 1.0];
        // s = 1 is an eigenvalue of the identity.
        assert_eq!(
            solve_shifted(&a, 2, (1.0, 0.0), &[1.0, 1.0], &[0.0, 0.0]),
            Err(LinalgError::Singular)
        );
        assert_eq!(
            solve_shifted(&a, 2, (0.0, 1.0), &[1.0], &[0.0, 0.0]),
            Err(LinalgError::BadShape)
        );
    }
}
