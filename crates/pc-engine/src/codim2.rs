//! Normal form coefficients of codim-2 points of equilibria.
//!
//! The formulas are Kuznetsov's (Numerical normalization techniques for all codim 2
//! bifurcations of equilibria in ODE's, SIAM J. Numer. Anal. 36, 1999, and the 1998 lecture
//! notes on codim 2 bifurcations), with `B` and `C` the second and third derivatives of `F`
//! in the state at the point.

use crate::curve2::{Model2, dot, norm};
use crate::error::EngineError;
use pc_linalg::{Lu, null_left, null_right};

/// Normalization recorded with the cusp coefficient.
pub(crate) const CUSP_NORMALIZATION: &str = "|q| = 1 before scaling, <p,q> = 1, <p,h2> = 0";

/// Cubic coefficient `c` of the cusp normal form `w' = c w^3 + O(w^4)` at the state `u`
/// (parameters set in `m`), where `A = F_u` has a simple zero eigenvalue and `⟨p, B(q, q)⟩`
/// vanishes:
///
/// `c = (1/6) ⟨p, C(q,q,q) + 3 B(q, h2)⟩`, with `A q = 0`, `Aᵀ p = 0`, `⟨p, q⟩ = 1` and
/// `A h2 = ⟨p, B(q,q)⟩ q - B(q,q)`, `⟨p, h2⟩ = 0`.
///
/// The value depends on the scale of `q`; `c` scales like the square of that scale. Its sign
/// does not. `⟨p, q⟩ = 0` (a Bogdanov-Takens point) gives `EngineError::Singular`.
pub(crate) fn cusp_coefficient(m: &mut Model2, u: &[f64]) -> Result<f64, EngineError> {
    let n = m.n;
    let a = m.jac_u(u)?;
    let mut q = null_right(&a, n, n, 1)?.swap_remove(0);
    let p = null_left(&a, n, n, 1)?.swap_remove(0);
    let pq = dot(&p, &q);
    if !pq.is_finite() || pq.abs() <= 1e-12 * norm(&p) * norm(&q) {
        return Err(EngineError::Singular(
            "the cusp coefficient: p and q are orthogonal".into(),
        ));
    }
    q.iter_mut().for_each(|c| *c /= pq);
    let bqq = m.b(u, &q, &q)?;
    let pb = dot(&p, &bqq);
    // [[A, p], [pᵀ, 0]] (h2; s) = (<p,B> q - B; 0). The right side is orthogonal to p, so s = 0.
    let s = n + 1;
    let mut mat = vec![0.0; s * s];
    for j in 0..n {
        mat[s * j..s * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
        mat[n + s * j] = p[j];
    }
    mat[s * n..s * n + n].copy_from_slice(&p);
    let mut rhs = vec![0.0; s];
    for i in 0..n {
        rhs[i] = pb * q[i] - bqq[i];
    }
    let lu = Lu::new(&mat, s)?;
    lu.solve(&mut rhs)?;
    let h2 = &rhs[..n];
    let cqqq = m.c(u, &q, &q, &q)?;
    let bqh = m.b(u, &q, h2)?;
    let c = dot(&p, &cqqq) / 6.0 + dot(&p, &bqh) / 2.0;
    if !c.is_finite() {
        return Err(EngineError::NotFinite("the cusp coefficient".into()));
    }
    Ok(c)
}

/// Normalization recorded with the Bogdanov-Takens coefficients.
pub(crate) const BT_NORMALIZATION: &str =
    "|q0| = 1, sign of q0 chosen so that a > 0, <q0,p0> = <q1,p1> = 1, <q1,p0> = <q0,p1> = 0";

/// Solves `[[A, col], [rowᵀ, 0]] (x; s) = (rhs; 0)` for the n by n column-major `a` and
/// returns `x`.
fn bordered(
    a: &[f64],
    n: usize,
    col: &[f64],
    row: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, EngineError> {
    let s = n + 1;
    let mut mat = vec![0.0; s * s];
    for j in 0..n {
        mat[s * j..s * j + n].copy_from_slice(&a[n * j..n * (j + 1)]);
        mat[n + s * j] = row[j];
    }
    mat[s * n..s * n + n].copy_from_slice(col);
    let mut x = vec![0.0; s];
    x[..n].copy_from_slice(rhs);
    Lu::new(&mat, s)?.solve(&mut x)?;
    x.truncate(n);
    Ok(x)
}

/// Normal form coefficients `(a, b)` of `w0' = w1, w1' = a w0^2 + b w0 w1 + O(3)` at the
/// state `u` of a Bogdanov-Takens point, where `A = F_u` has a double zero eigenvalue with a
/// Jordan block.
///
/// With the chains `A q0 = 0, A q1 = q0, Aᵀ p1 = 0, Aᵀ p0 = p1` and
/// `⟨q0,p0⟩ = ⟨q1,p1⟩ = 1`, `⟨q1,p0⟩ = ⟨q0,p1⟩ = 0`,
///
/// `a = ½ ⟨p1, B(q0,q0)⟩`, `b = ⟨p0, B(q0,q0)⟩ + ⟨p1, B(q0,q1)⟩`
///
/// (Kuznetsov 1998 notes, section Bogdanov-Takens). The chains are fixed up to the scale of
/// `q0`, and `a` and `b` scale with it. Here `|q0| = 1` and the sign of `q0` makes `a`
/// positive; see `BT_NORMALIZATION`. The vectors come from bordered solves, so `A` needs
/// only be close to a Jordan block: `⟨p1, q0⟩` above 1e-4 gives `EngineError::Singular`.
pub(crate) fn bt_coefficients(m: &mut Model2, u: &[f64]) -> Result<(f64, f64), EngineError> {
    let n = m.n;
    let a = m.jac_u(u)?;
    let q0 = null_right(&a, n, n, 1)?.swap_remove(0);
    let p1 = null_left(&a, n, n, 1)?.swap_remove(0);
    if dot(&p1, &q0).abs() > 1e-4 {
        return Err(EngineError::Singular(
            "the BT coefficients: A has no Jordan block, the null vectors are not orthogonal"
                .into(),
        ));
    }
    let q1 = bordered(&a, n, &p1, &q0, &q0)?;
    let mut at = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            at[i + n * j] = a[j + n * i];
        }
    }
    let p0 = bordered(&at, n, &q0, &p1, &p1)?;
    let k = dot(&q0, &p0);
    if !k.is_finite() || k.abs() < 1e-12 {
        return Err(EngineError::Singular(
            "the BT coefficients: <q0,p0> vanishes".into(),
        ));
    }
    // Scale the left chain: p1 -> g p1, p0 -> g p0 + d p1 with g k = 1 and <q1,p0> = 0.
    let g = 1.0 / k;
    let d = -g * dot(&q1, &p0) / k;
    let p1s: Vec<f64> = p1.iter().map(|c| g * c).collect();
    let p0s: Vec<f64> = p0.iter().zip(&p1).map(|(c0, c1)| g * c0 + d * c1).collect();
    let bq00 = m.b(u, &q0, &q0)?;
    let bq01 = m.b(u, &q0, &q1)?;
    let a0 = 0.5 * dot(&p1s, &bq00);
    let b0 = dot(&p0s, &bq00) + dot(&p1s, &bq01);
    if !(a0.is_finite() && b0.is_finite()) {
        return Err(EngineError::NotFinite("the BT coefficients".into()));
    }
    // Flipping q0 (and with it q1, p0, p1) changes the sign of both coefficients.
    Ok(if a0 < 0.0 { (-a0, -b0) } else { (a0, b0) })
}

/// The eigenvalues without the one of smallest modulus: the zero eigenvalue of a point on an
/// LP curve.
pub(crate) fn without_zero(eigs: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let Some((k, _)) = eigs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.0.hypot(a.1.1).total_cmp(&b.1.0.hypot(b.1.1)))
    else {
        return Vec::new();
    };
    let mut out = eigs.to_vec();
    out.remove(k);
    out
}

#[cfg(test)]
mod tests {
    use super::without_zero;
    use crate::{HopfZero, classify_h_zero};

    #[test]
    fn without_zero_drops_the_smallest_eigenvalue() {
        assert!(without_zero(&[]).is_empty());
        let e = [(3.0, 0.0), (1e-12, 0.0), (-1.0, 2.0), (-1.0, -2.0)];
        assert_eq!(
            without_zero(&e),
            vec![(3.0, 0.0), (-1.0, 2.0), (-1.0, -2.0)]
        );
    }

    #[test]
    fn zero_hopf_needs_a_complex_pair_on_the_axis() {
        // A pair +-2i beside the zero eigenvalue is a ZH point.
        let zh = [(0.0, 0.0), (0.0, 2.0), (0.0, -2.0), (-3.0, 0.0)];
        assert_eq!(classify_h_zero(&without_zero(&zh)), HopfZero::Hopf);
        // Real eigenvalues +-2 beside the zero make psi_h vanish without a Hopf pair.
        let ns = [(0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (-3.0, 0.0)];
        assert_eq!(classify_h_zero(&without_zero(&ns)), HopfZero::NeutralSaddle);
    }
}
