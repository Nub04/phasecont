//! Eigenvalues and eigenvectors of a real square matrix.

use crate::dense::{LinalgError, check};
use faer::MatRef;
use faer::linalg::solvers::Eigen;

fn decompose(a: &[f64], n: usize) -> Result<Eigen<f64>, LinalgError> {
    check(a, n, n)?;
    Eigen::new_from_real(MatRef::from_column_major_slice(a, n, n))
        .map_err(|_| LinalgError::NoConvergence)
}

/// Eigenvalues of the n-by-n column-major matrix `a` as (re, im), sorted by re descending
/// and then im descending. Complex pairs are exact conjugates of each other.
pub fn eigenvalues(a: &[f64], n: usize) -> Result<Vec<(f64, f64)>, LinalgError> {
    if n == 0 {
        check(a, n, n)?;
        return Ok(Vec::new());
    }
    let evd = decompose(a, n)?;
    let s = evd.S().column_vector();
    let mut ev: Vec<(f64, f64)> = (0..n).map(|i| (s[i].re, s[i].im)).collect();
    if ev.iter().any(|(r, i)| !r.is_finite() || !i.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    // Make each pair exact: for every eigenvalue with im > 0, the closest unused one with
    // im < 0 becomes its conjugate.
    let mut used = vec![false; n];
    for i in 0..n {
        if ev[i].1 > 0.0 && !used[i] {
            let mut best: Option<(usize, f64)> = None;
            for j in 0..n {
                if j != i && !used[j] && ev[j].1 < 0.0 {
                    let d = (ev[j].0 - ev[i].0).hypot(ev[j].1 + ev[i].1);
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((j, d));
                    }
                }
            }
            if let Some((j, _)) = best {
                used[i] = true;
                used[j] = true;
                ev[j] = (ev[i].0, -ev[i].1);
            }
        }
    }
    ev.sort_by(|p, q| q.0.total_cmp(&p.0).then(q.1.total_cmp(&p.1)));
    Ok(ev)
}

/// Eigenvector `(re, im)` of the eigenvalue closest to `target`, scaled to complex 2-norm 1.
///
/// The phase is fixed so that the component of largest modulus is real and positive.
pub fn eigvec(
    a: &[f64],
    n: usize,
    target: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    if n == 0 || !target.0.is_finite() || !target.1.is_finite() {
        check(a, n, n)?;
        return Err(if n == 0 {
            LinalgError::BadShape
        } else {
            LinalgError::NotFinite
        });
    }
    let evd = decompose(a, n)?;
    let s = evd.S().column_vector();
    let mut k = 0;
    let mut best = f64::INFINITY;
    for i in 0..n {
        let d = (s[i].re - target.0).hypot(s[i].im - target.1);
        if d < best {
            best = d;
            k = i;
        }
    }
    let u = evd.U();
    let mut re: Vec<f64> = (0..n).map(|i| u[(i, k)].re).collect();
    let mut im: Vec<f64> = (0..n).map(|i| u[(i, k)].im).collect();
    let norm = re
        .iter()
        .zip(&im)
        .map(|(r, i)| r * r + i * i)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(LinalgError::NotFinite);
    }
    // Rotate by the conjugate phase of the largest component, then scale.
    let p = (0..n)
        .max_by(|&i, &j| re[i].hypot(im[i]).total_cmp(&re[j].hypot(im[j])))
        .unwrap_or(0);
    let m = re[p].hypot(im[p]);
    let (cr, ci) = (re[p] / m, -im[p] / m);
    for i in 0..n {
        let (r, q) = (re[i], im[i]);
        re[i] = (r * cr - q * ci) / norm;
        im[i] = (r * ci + q * cr) / norm;
    }
    // The pivot component is real up to rounding; make that exact.
    im[p] = 0.0;
    Ok((re, im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::testutil::Rng;

    /// Companion matrix (column-major) of the monic polynomial with coefficients
    /// c[0] + c[1] x + ... + x^n.
    fn companion(c: &[f64]) -> Vec<f64> {
        let n = c.len();
        let mut a = vec![0.0; n * n];
        for i in 1..n {
            a[i + n * (i - 1)] = 1.0;
        }
        for i in 0..n {
            a[i + n * (n - 1)] = -c[i];
        }
        a
    }

    #[test]
    fn companion_real_roots() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6.
        let ev = eigenvalues(&companion(&[-6.0, 11.0, -6.0]), 3).unwrap();
        for (got, want) in ev.iter().zip([3.0, 2.0, 1.0]) {
            assert!(
                (got.0 - want).abs() < 1e-12 && got.1.abs() < 1e-12,
                "{ev:?}"
            );
        }
    }

    #[test]
    fn companion_imaginary_pair() {
        let ev = eigenvalues(&companion(&[1.0, 0.0]), 2).unwrap();
        assert!(
            ev[0].0.abs() < 1e-12 && (ev[0].1 - 1.0).abs() < 1e-12,
            "{ev:?}"
        );
        assert_eq!(ev[1], (ev[0].0, -ev[0].1));
    }

    #[test]
    fn random_pairs_are_exact_conjugates() {
        let mut rng = Rng(7);
        let n = 9;
        let a: Vec<f64> = (0..n * n).map(|_| rng.unit()).collect();
        let ev = eigenvalues(&a, n).unwrap();
        for w in ev.windows(2) {
            assert!(w[0].0 > w[1].0 || (w[0].0 == w[1].0 && w[0].1 >= w[1].1));
        }
        for e in ev.iter().filter(|e| e.1 > 0.0) {
            assert!(ev.contains(&(e.0, -e.1)), "{ev:?}");
        }
    }

    #[test]
    fn eigvec_residual_and_phase() {
        let mut rng = Rng(11);
        let n = 6;
        let a: Vec<f64> = (0..n * n).map(|_| rng.unit()).collect();
        for lam in eigenvalues(&a, n).unwrap() {
            let (vr, vi) = eigvec(&a, n, lam).unwrap();
            let norm: f64 = vr.iter().zip(&vi).map(|(r, i)| r * r + i * i).sum();
            assert!((norm - 1.0).abs() < 1e-14);
            let mut res = 0.0_f64;
            for i in 0..n {
                let (mut ar, mut ai) = (0.0, 0.0);
                for j in 0..n {
                    ar += a[i + n * j] * vr[j];
                    ai += a[i + n * j] * vi[j];
                }
                let er = ar - (lam.0 * vr[i] - lam.1 * vi[i]);
                let ei = ai - (lam.0 * vi[i] + lam.1 * vr[i]);
                res = res.max(er.hypot(ei));
            }
            assert!(res <= 1e-12, "lam={lam:?} res={res}");
            let p = (0..n)
                .max_by(|&i, &j| vr[i].hypot(vi[i]).total_cmp(&vr[j].hypot(vi[j])))
                .unwrap();
            assert!(vr[p] > 0.0 && vi[p] == 0.0);
        }
    }

    #[test]
    fn errors() {
        assert_eq!(eigenvalues(&[1.0; 3], 2), Err(LinalgError::BadShape));
        assert_eq!(eigenvalues(&[f64::NAN], 1), Err(LinalgError::NotFinite));
        assert_eq!(
            eigvec(&[1.0], 1, (f64::NAN, 0.0)),
            Err(LinalgError::NotFinite)
        );
        assert_eq!(eigvec(&[], 0, (0.0, 0.0)), Err(LinalgError::BadShape));
    }
}
