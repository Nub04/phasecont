//! Test functions on equilibrium branches (SPEC section 5).
//!
//! At every accepted point the engine stores `[psi_LP, psi_BP, psi_H]`. `psi_LP` is the
//! parameter component of the unit tangent and `psi_BP` is the determinant of the bordered
//! tangent matrix; both come from the tangent computation (the caller passes them in).
//! `psi_H` is computed here from the eigenvalues of A = F_u. Hopf points are the zeros of
//! the bialternate product 2A (.) I, whose determinant is `prod_{i<j} (l_i + l_j)`
//! (Kuznetsov, Elements of Applied Bifurcation Theory, 3rd ed., Sec. 10.2). The rescaling
//! below is our own choice. With `s_ij = (l_i + l_j) / max(1, |l_i| + |l_j|)` and `s_k`
//! the factor of smallest modulus, `psi_H = Re(s_k prod_{ij != k} s_ij / |s_ij|)`, which
//! is the plain product divided by the positive number
//! `prod max(1, |l_i| + |l_j|) * prod_{ij != k} |s_ij|`. That divisor is continuous in A
//! (at a tie for the smallest modulus both choices give the same value), so `psi_H` has
//! the sign and the zeros of the plain product. Its modulus is `|s_k| <= 1`, so it
//! neither overflows nor underflows to 0 away from a zero, however many small factors
//! there are. The product over all pairs of a real matrix is real, so the real part only
//! drops rounding noise.

use crate::branch::Options;
use crate::error::EngineError;
use crate::problem::Problem;
use pc_linalg::{LinalgError, eigenvalues};

/// Largest state dimension for which Hopf detection runs.
pub(crate) const MAX_HOPF_DIM: usize = 100;

/// Position of each test function in `Point.tests`.
pub(crate) const LP: usize = 0;
pub(crate) const BP: usize = 1;
pub(crate) const H: usize = 2;

/// `psi_H` of a list of eigenvalues given as (re, im). With fewer than two eigenvalues
/// there are no pairs and the result is 1. NaN in the input gives NaN.
pub(crate) fn psi_h(eigs: &[(f64, f64)]) -> f64 {
    if eigs.iter().any(|&(r, i)| r.is_nan() || i.is_nan()) {
        return f64::NAN;
    }
    let mut s = Vec::with_capacity(eigs.len() * eigs.len().saturating_sub(1) / 2);
    for (i, &(ar, ai)) in eigs.iter().enumerate() {
        for &(br, bi) in &eigs[i + 1..] {
            let d = 1.0_f64.max(ar.hypot(ai) + br.hypot(bi));
            s.push(((ar + br) / d, (ai + bi) / d));
        }
    }
    // The smallest factor keeps its modulus; every other factor contributes its phase.
    let Some(k) =
        (0..s.len()).min_by(|&a, &b| s[a].0.hypot(s[a].1).total_cmp(&s[b].0.hypot(s[b].1)))
    else {
        return 1.0;
    };
    let (mut pr, mut pi) = s[k];
    for (j, &(sr, si)) in s.iter().enumerate() {
        if j == k {
            continue;
        }
        let m = sr.hypot(si);
        if m == 0.0 {
            return 0.0;
        }
        let (ur, ui) = (sr / m, si / m);
        (pr, pi) = (pr * ur - pi * ui, pr * ui + pi * ur);
    }
    pr
}

/// What a zero of `psi_H` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopfZero {
    /// The vanishing pair is a complex conjugate pair off the real axis.
    Hopf,
    /// The vanishing pair is real (+-lambda), or complex but not conjugate.
    NeutralSaddle,
    /// Fewer than two eigenvalues, so `psi_H` has no zero.
    NoPair,
}

/// Decides what a zero of `psi_H` at a point with these eigenvalues is. Finds the pair
/// (i, j) with the smallest `|l_i + l_j| / max(1, |l_i| + |l_j|)`. It is a Hopf point when
/// that pair is a conjugate pair with `|im| > 1e-8 max(1, |l|)`, else a neutral saddle.
/// Task 7 calls this at located zeros of `psi_H` to choose the label `H` or `NSA`.
pub fn classify_h_zero(eigs: &[(f64, f64)]) -> HopfZero {
    let mut best: Option<(f64, usize, usize)> = None;
    for i in 0..eigs.len() {
        for j in i + 1..eigs.len() {
            let (a, b) = (eigs[i], eigs[j]);
            let s = (a.0 + b.0).hypot(a.1 + b.1) / 1.0_f64.max(a.0.hypot(a.1) + b.0.hypot(b.1));
            if best.is_none_or(|(m, _, _)| s < m) {
                best = Some((s, i, j));
            }
        }
    }
    let Some((_, i, j)) = best else {
        return HopfZero::NoPair;
    };
    let (a, b) = (eigs[i], eigs[j]);
    let tol = 1e-8 * 1.0_f64.max(a.0.hypot(a.1));
    if (a.0 - b.0).abs() <= tol && (a.1 + b.1).abs() <= tol && a.1.abs() > tol {
        HopfZero::Hopf
    } else {
        HopfZero::NeutralSaddle
    }
}

fn push_once(w: &mut Vec<String>, msg: &str) {
    if !w.iter().any(|m| m == msg) {
        w.push(msg.to_string());
    }
}

/// Test values and eigenvalues at one point.
pub(crate) struct PointTests {
    pub tests: [f64; 3],
    pub eigenvalues: Vec<(f64, f64)>,
}

/// Evaluates the test functions at `x`. `tangent` is the unit tangent there and `det` the
/// determinant of the bordered tangent solve (0 when that matrix was singular).
/// A test whose `detect_*` option is off is NaN. Eigenvalues are stored whenever the
/// problem has a state block. For a state dimension above `MAX_HOPF_DIM`, `psi_H` is NaN and
/// a warning is added once. A failed eigenvalue computation gives no eigenvalues, NaN `psi_H`
/// and a warning, and the run goes on.
pub(crate) fn evaluate(
    p: &mut dyn Problem,
    x: &[f64],
    tangent: &[f64],
    det: f64,
    opts: &Options,
    warnings: &mut Vec<String>,
) -> Result<PointTests, EngineError> {
    let n = p.dim() - 1;
    let mut eig_failed = false;
    let eigs = match p.state_jacobian(x)? {
        Some(a) => {
            if opts.detect_h && n > MAX_HOPF_DIM {
                push_once(warnings, "Hopf detection is off for n > 100");
            }
            match eigenvalues(&a, n) {
                Ok(e) => e,
                // The run goes on without eigenvalues at this point.
                Err(LinalgError::NoConvergence) => {
                    eig_failed = true;
                    push_once(warnings, "eigenvalues did not converge at some points");
                    Vec::new()
                }
                Err(e) => return Err(e.into()),
            }
        }
        None => Vec::new(),
    };
    let mut tests = [f64::NAN; 3];
    if opts.detect_lp {
        tests[LP] = tangent[n];
    }
    if opts.detect_bp {
        tests[BP] = det;
    }
    if opts.detect_h && n <= MAX_HOPF_DIM && !eig_failed && !eigs.is_empty() {
        tests[H] = psi_h(&eigs);
    }
    Ok(PointTests {
        tests,
        eigenvalues: eigs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small linear congruential generator, values in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn two_by_two_hand_cases() {
        // A = [[a, -1], [1, a]] has eigenvalues a +- i, one pair with sum 2a.
        for a in [-0.5, 0.25] {
            let e = eigenvalues(&[a, 1.0, -1.0, a], 2).unwrap();
            let want = 2.0 * a / 1.0_f64.max(2.0 * (a * a + 1.0).sqrt());
            assert!((psi_h(&e) - want).abs() < 1e-14, "a = {a}");
        }
        let e = eigenvalues(&[0.0, 1.0, -1.0, 0.0], 2).unwrap();
        assert!(psi_h(&e).abs() < 1e-15);
        // Real eigenvalues 1 and -1: neutral saddle, zero.
        let e = eigenvalues(&[0.0, 1.0, 1.0, 0.0], 2).unwrap();
        assert!(psi_h(&e).abs() < 1e-15);
        // One eigenvalue: no pair, psi_H = 1.
        assert_eq!(psi_h(&[(-3.0, 0.0)]), 1.0);
        assert_eq!(psi_h(&[]), 1.0);
    }

    #[test]
    fn three_by_three_hand_cases() {
        // diag(1, 2, 3): every factor is 1.
        let d = |a: f64, b: f64, c: f64| eigenvalues(&[a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c], 3);
        assert!((psi_h(&d(1.0, 2.0, 3.0).unwrap()) - 1.0).abs() < 1e-15);
        assert!((psi_h(&d(-1.0, -2.0, -3.0).unwrap()) + 1.0).abs() < 1e-15);
        // diag(-1, 1, 2): the pair (-1, 1) vanishes.
        assert!(psi_h(&d(-1.0, 1.0, 2.0).unwrap()).abs() < 1e-15);
        // Sign flips across the zero: diag(-1, 1 +- d, 2).
        let lo = psi_h(&d(-1.0, 0.9, 2.0).unwrap());
        let hi = psi_h(&d(-1.0, 1.1, 2.0).unwrap());
        assert!(lo < 0.0 && hi > 0.0, "{lo} {hi}");
    }

    #[test]
    fn scaling_keeps_the_sign_of_the_plain_product() {
        let mut g = Lcg(12345);
        for _ in 0..20 {
            let a: Vec<f64> = (0..36).map(|_| 3.0 * g.next()).collect();
            let e = eigenvalues(&a, 6).unwrap();
            let (mut pr, mut pi, mut smin) = (1.0_f64, 0.0_f64, f64::INFINITY);
            for i in 0..6 {
                for j in i + 1..6 {
                    let (sr, si) = (e[i].0 + e[j].0, e[i].1 + e[j].1);
                    (pr, pi) = (pr * sr - pi * si, pr * si + pi * sr);
                    let d = 1.0_f64.max(e[i].0.hypot(e[i].1) + e[j].0.hypot(e[j].1));
                    smin = smin.min(sr.hypot(si) / d);
                }
            }
            let scaled = psi_h(&e);
            assert!(scaled * pr > 0.0, "{scaled} {pr}");
            // The modulus is that of the smallest scaled factor.
            assert!(
                (scaled.abs() - smin).abs() <= 1e-10 * smin,
                "{scaled} {smin}"
            );
        }
    }

    #[test]
    fn many_small_factors_do_not_underflow() {
        // A pair mu +- i and 38 slow real modes near -1e-3: 703 slow pairs whose plain
        // scaled product is far below the smallest f64.
        for mu in [-1e-3, -1e-9, 1e-9, 1e-3] {
            let mut e = vec![(mu, 1.0), (mu, -1.0)];
            e.extend((0..38).map(|k| (-1e-3 * (1.0 + k as f64 / 100.0), 0.0)));
            let v = psi_h(&e);
            // The smallest factor is 2 mu / max(1, 2 |mu + i|). The 703 negative slow
            // factors flip the sign and the cross factors pair up to a positive phase.
            let want = -mu / (mu * mu + 1.0).sqrt();
            assert!((v - want).abs() <= 1e-8 * want.abs(), "mu {mu}: {v}");
            if mu < 0.0 {
                assert!(v > 0.0, "mu {mu}: {v}");
            } else {
                assert!(v < 0.0, "mu {mu}: {v}");
            }
        }
    }

    #[test]
    fn a_tie_for_the_smallest_factor_gives_one_value() {
        // Two conjugate pairs with the same real part: two factors of equal modulus.
        let e = [(0.1, 1.0), (0.1, -1.0), (0.1, 2.0), (0.1, -2.0)];
        let v = psi_h(&e);
        let mut r = e;
        r.reverse();
        assert_eq!(v, psi_h(&r));
        assert!(v > 0.0);
    }

    #[test]
    fn nan_input_gives_nan() {
        assert!(psi_h(&[(f64::NAN, 0.0), (1.0, 0.0)]).is_nan());
    }

    #[test]
    fn zero_kinds() {
        // Hopf: +- 2i and a stable real eigenvalue.
        let e = [(-3.0, 0.0), (0.0, 2.0), (0.0, -2.0)];
        assert_eq!(classify_h_zero(&e), HopfZero::Hopf);
        // Neutral saddle: +-1.
        let e = [(1.0, 0.0), (-1.0, 0.0), (-4.0, 0.0)];
        assert_eq!(classify_h_zero(&e), HopfZero::NeutralSaddle);
        // Conjugate pair with a tiny imaginary part is not Hopf.
        let e = [(0.0, 1e-12), (0.0, -1e-12)];
        assert_eq!(classify_h_zero(&e), HopfZero::NeutralSaddle);
        // Complex +-lambda that is not a conjugate pair.
        let e = [(1.0, 1.0), (-1.0, -1.0)];
        assert_eq!(classify_h_zero(&e), HopfZero::NeutralSaddle);
        assert_eq!(classify_h_zero(&[(1.0, 0.0)]), HopfZero::NoPair);
        // The smallest scaled pair decides, not a larger one.
        let e = [(0.0, 5.0), (0.0, -5.0), (2.0, 0.0), (-2.1, 0.0)];
        assert_eq!(classify_h_zero(&e), HopfZero::Hopf);
    }
}
