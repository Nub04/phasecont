//! Switching branches at a branch point (BP) with the algebraic branching equation.
//!
//! At a simple BP the n by (n + 1) matrix `F_x = [A F_alpha]` has a two-dimensional right
//! kernel and a one-dimensional left kernel. The tangents of the two branches through the
//! point lie in that kernel and solve a quadratic form in the kernel coordinates. One root
//! is the branch that was followed; the other is where the run goes on after switching.

use crate::branch::{Branch, Options};
use crate::ep::EpSetup;
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::refine::Ctx;
use pc_linalg::{null_left, null_right, singular_values};

/// Roots closer than this (up to sign) are one root: the BP is not simple.
const ROOT_TOL: f64 = 1e-6;
/// A tangent component below this counts as zero when its sign is fixed.
const SIGN_TOL: f64 = 1e-12;

/// Index into `branch.specials` of the special point named `name`: a label followed by a
/// 1-based ordinal among the specials with that label. "BP1" is the first BP, "H2" the
/// second H. The label is matched without regard to case ("bp1" is "BP1"). `None` for an unknown name, an ordinal of 0 or too large, a name without an
/// ordinal ("BP") and a name without a label ("1").
pub fn special_index(branch: &Branch, name: &str) -> Option<usize> {
    let cut = name.find(|c: char| c.is_ascii_digit())?;
    let (label, digits) = name.split_at(cut);
    if label.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let ordinal: usize = digits.parse().ok()?;
    if ordinal == 0 {
        return None;
    }
    branch
        .specials
        .iter()
        .enumerate()
        .filter(|(_, s)| s.label.eq_ignore_ascii_case(label))
        .nth(ordinal - 1)
        .map(|(i, _)| i)
}

/// Setup for the branch that crosses the equilibrium branch at its refined BP
/// `branch.specials[special]`.
///
/// With `phi1, phi2` a basis of the right kernel of `F_x`, `psi` the left null vector and
/// `c_ij = psiᵀ F_xx[phi_i, phi_j]`, a tangent `t = b1 phi1 + b2 phi2` of a branch through
/// the point satisfies `c11 b1² + 2 c12 b1 b2 + c22 b2² = 0`. Source: Allgower and Georg,
/// "Numerical Continuation Methods" (Springer 1990), section 8.3, eqs. (8.3.7) to (8.3.11)
/// (their alpha[i,j] is our c_ij, their tau_i our phi_i, their e our psi). The form has two
/// real roots up to sign when `c12² - c11 c22 > 0`; this code solves it in the angle form
/// `b = (cos th, sin th)`, which needs no division by `c11` or `c22`. Of the two root
/// directions the one with the smaller `|tᵀ v|` is taken, `v` being the tangent of the
/// followed branch at the point. Its sign is positive in alpha, or positive in the first
/// component above 1e-12 when alpha is flat.
///
/// The returned setup has `u0` at the BP, `params[free]` set to its alpha, the tangent in
/// `tangent0` and `start_is_special` set. `BadProblem` when `special` is out of range, is
/// not a BP or is not refined, and `BadProblem("BP is not simple")` when the kernel of
/// `F_x` is larger than two, the form has no pair of real roots or the roots coincide.
/// `opts` is reserved for later use and is not read now.
pub fn switch_bp(
    model: &mut dyn ModelEval,
    setup: &EpSetup,
    branch: &Branch,
    special: usize,
    _opts: &Options,
) -> Result<EpSetup, EngineError> {
    let sp = branch
        .specials
        .get(special)
        .ok_or_else(|| EngineError::BadProblem(format!("no special point {special}")))?;
    if sp.label != "BP" {
        return Err(EngineError::BadProblem(format!(
            "special point {special} is a {}, not a BP",
            sp.label
        )));
    }
    if !sp.extras.refined {
        return Err(EngineError::BadProblem(
            "the BP is not refined; switching needs the refined point".into(),
        ));
    }
    let mut ctx = Ctx::new(model, &setup.params, setup.free)?;
    let n = ctx.n;
    let (u, alpha) = (&sp.point.u, sp.point.alpha);
    if u.len() != n || sp.point.tangent.len() != n + 1 {
        return Err(EngineError::BadProblem(
            "the BP does not fit the model".into(),
        ));
    }
    ctx.set_alpha(alpha);
    let fx = ctx.f_x(u)?;
    // Kernel of F_x must be exactly two-dimensional: the smallest singular value (the
    // n-th) must be about zero and, for n >= 2, the (n-1)-th clearly nonzero. The floor
    // max(sv0, 1) turns the first check off when sv0 < 1e-6; branching_root still refuses
    // a point whose quadratic form has no pair of real roots.
    let sv = singular_values(&fx, n, n + 1)?;
    let not_simple = n >= 1 && sv[n - 1] > 1e-6 * sv[0].max(1.0);
    if not_simple || (n >= 2 && sv[n - 2] <= 1e-8 * sv[0]) {
        return Err(EngineError::BadProblem("BP is not simple".into()));
    }
    let phi = null_right(&fx, n, n + 1, 2)?;
    let psi = null_left(&fx, n, n + 1, 1)?.swap_remove(0);
    let mut c = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in i..2 {
            let f2 = ctx.f_xx_dir(u, &phi[i], &phi[j])?;
            let v: f64 = psi.iter().zip(&f2).map(|(a, b)| a * b).sum();
            c[i][j] = v;
            c[j][i] = v;
        }
    }
    let t = branching_root(&c, &phi, &sp.point.tangent, SIGN_TOL)?;
    let mut next = setup.clone();
    next.u0 = u.clone();
    next.params[setup.free] = alpha;
    next.tangent0 = Some(t);
    next.start_is_special = true;
    Ok(next)
}

/// Solves `c11 b1² + 2 c12 b1 b2 + c22 b2² = 0` on the unit circle and returns the root
/// direction `t = b1 phi1 + b2 phi2` that is least parallel to `v`. The sign is positive in
/// the last component when that is at least `sign_tol` in size, else positive in the first
/// component of at least that size ([`switch_bp`] passes 1e-12).
pub(crate) fn branching_root(
    c: &[[f64; 2]; 2],
    phi: &[Vec<f64>],
    v: &[f64],
    sign_tol: f64,
) -> Result<Vec<f64>, EngineError> {
    let not_simple = || EngineError::BadProblem("BP is not simple".into());
    // q(th) = a + r cos(2 th - d) with b = (cos th, sin th).
    let a = 0.5 * (c[0][0] + c[1][1]);
    let h = 0.5 * (c[0][0] - c[1][1]);
    let r = h.hypot(c[0][1]);
    if !r.is_finite() || !a.is_finite() || r == 0.0 || a.abs() >= r {
        return Err(not_simple());
    }
    let d = c[0][1].atan2(h);
    let w = (-a / r).acos();
    let dirs: Vec<Vec<f64>> = [d + w, d - w]
        .iter()
        .map(|two_th| {
            let th = 0.5 * two_th;
            let (b2, b1) = th.sin_cos();
            let t: Vec<f64> = phi[0]
                .iter()
                .zip(&phi[1])
                .map(|(p, q)| b1 * p + b2 * q)
                .collect();
            let nrm = t.iter().map(|x| x * x).sum::<f64>().sqrt();
            t.iter().map(|x| x / nrm).collect()
        })
        .collect();
    let dist = |s: f64| {
        dirs[0]
            .iter()
            .zip(&dirs[1])
            .map(|(p, q)| (p - s * q).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    if dist(1.0).min(dist(-1.0)) < ROOT_TOL {
        return Err(not_simple());
    }
    let par = |t: &[f64]| t.iter().zip(v).map(|(p, q)| p * q).sum::<f64>().abs();
    let mut t = if par(&dirs[0]) <= par(&dirs[1]) {
        dirs[0].clone()
    } else {
        dirs[1].clone()
    };
    let m = t.len() - 1;
    let lead = if t[m].abs() >= sign_tol {
        t[m]
    } else {
        t.iter()
            .copied()
            .find(|x| x.abs() >= sign_tol)
            .unwrap_or(1.0)
    };
    if lead < 0.0 {
        for x in t.iter_mut() {
            *x = -*x;
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis() -> Vec<Vec<f64>> {
        vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]
    }

    #[test]
    fn two_separate_roots_give_a_unit_tangent() {
        // b1^2 - b2^2 = 0 has the roots b = (1, 1) and (1, -1).
        let c = [[1.0, 0.0], [0.0, -1.0]];
        let t = branching_root(&c, &basis(), &[1.0, 0.0, 0.0], SIGN_TOL).unwrap();
        let nrm: f64 = t.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((nrm - 1.0).abs() < 1e-14);
        assert!((t[0].abs() - t[1].abs()).abs() < 1e-14 && t[2] == 0.0);
    }

    #[test]
    fn the_sign_tolerance_decides_which_component_fixes_the_sign() {
        // c = diag(1, -1) has the roots b = (1, 1) and (1, -1). With v = (1, -1, 0) / sqrt 2
        // the root b = (1, -1) is taken: t = (-1, -1, 1e-8) / sqrt 2. Its last component
        // is 7e-9. A tolerance of 1e-12 counts that as nonzero and keeps t as it is; a
        // tolerance of 1e-6 counts it as zero and makes the first component positive.
        let c = [[1.0, 0.0], [0.0, -1.0]];
        let phi = vec![vec![-1.0, 0.0, 1e-8], vec![0.0, 1.0, 0.0]];
        let s = 0.5_f64.sqrt();
        let v = [s, -s, 0.0];
        let fine = branching_root(&c, &phi, &v, 1e-12).unwrap();
        assert!(fine[0] < 0.0 && fine[1] < 0.0 && fine[2] > 0.0, "{fine:?}");
        let coarse = branching_root(&c, &phi, &v, 1e-6).unwrap();
        assert!(
            coarse[0] > 0.0 && coarse[1] > 0.0 && coarse[2] < 0.0,
            "{coarse:?}"
        );
        for (p, q) in fine.iter().zip(&coarse) {
            assert!((p + q).abs() < 1e-14);
        }
    }

    #[test]
    fn coincident_roots_are_not_simple() {
        // c11 = 1, c22 = -1e-13: the discriminant is positive but the two root directions
        // are about 6e-7 apart, below ROOT_TOL, so the coincident-root branch is taken.
        let c = [[1.0, 0.0], [0.0, -1e-13]];
        let r = branching_root(&c, &basis(), &[1.0, 0.0, 0.0], SIGN_TOL);
        assert_eq!(r, Err(EngineError::BadProblem("BP is not simple".into())));
        // Just outside the tolerance the roots are accepted.
        let c = [[1.0, 0.0], [0.0, -1e-9]];
        assert!(branching_root(&c, &basis(), &[1.0, 0.0, 0.0], SIGN_TOL).is_ok());
    }
}
