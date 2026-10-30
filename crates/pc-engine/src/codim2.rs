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
