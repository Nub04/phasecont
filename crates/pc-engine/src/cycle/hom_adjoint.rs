//! The adjoint variational problem along a HOM orbit and the inclination-flip tests IFS and
//! IFU (SPEC §8b-2).
//!
//! The adjoint phi solves phi' = T (-f_x(u, p)^T phi + gamma f(u, p)) on tau in [0, 1],
//! with B_u(A)^T phi(0) = 0 and B_s(A)^T phi(1) = 0, where A = f_x(x0, p) and the columns
//! of B_u(A) and B_s(A) span its unstable and stable invariant subspaces. The orbit rows
//! of Section 8b use the subspaces of A^T instead.
//!
//! Ported from AUTO-07p, `src/homcont.f90` (BSD-3), at commit
//! 44cc1c42a888179bdbd608a40613caee9cbc4675: the adjoint right-hand side of FNHO, the
//! adjoint boundary rows of FBHO (the projections with ITRANS = 2), the tests 13 and 14 of
//! PSIHO, and the sign rule of the right eigenvectors in EIGHO. The bases come from the
//! matrix sign function in `pc-linalg`, not from HomCont's Schur form.
//!
//! This module only evaluates the adjoint equations at a given phi and tests it on a
//! given phi. It does not solve for phi and no HOM run calls it yet.

use super::basis::CollocationBasis;
use super::hom::{ascending, is_real, signed_eigvec};
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::EntryKind;
use pc_linalg::{BasisOf, BasisTracker, eigenvalues};

/// The orbit mesh, the saddle and the parameters an adjoint residual refers to.
pub struct AdjointSetup<'a> {
    /// Collocation points per interval.
    pub ncol: usize,
    /// The interval ends in [0, 1], ntst + 1 values from 0 to 1.
    pub mesh: &'a [f64],
    /// The truncation time T, AUTO's PAR(11).
    pub period: f64,
    /// The saddle x0, n values.
    pub x0: &'a [f64],
    /// The model parameters, without t.
    pub params: &'a [f64],
    /// The number of stable eigenvalues of A = f_x(x0, p).
    pub ns: usize,
}

/// The adjoint equations evaluated at a given (phi, gamma).
#[derive(Clone, Debug, PartialEq)]
pub struct AdjointResidual {
    /// The collocation rows, n for each of the ntst ncol points, in the form
    /// `sum_l L_l'(z) phi_l + h T (f_x^T phi(z) - gamma f)` of the orbit rows.
    pub collocation: Vec<f64>,
    /// The n boundary rows: n_u rows B_u^T phi(0), then n_s rows B_s^T phi(1).
    pub boundary: Vec<f64>,
}

fn bad(msg: String) -> EngineError {
    EngineError::BadProblem(msg)
}

/// The residual of the adjoint equations for the orbit `u` and the adjoint `phi`, both given
/// at the base points (n (ntst ncol + 1) values). The bases of A for the boundary rows come
/// from `tracker` (slot [`BasisOf::Matrix`]) and are rotated against its reference there.
pub fn adjoint_residual(
    model: &mut dyn ModelEval,
    s: &AdjointSetup,
    tracker: &BasisTracker,
    u: &[f64],
    phi: &[f64],
    gamma: f64,
) -> Result<AdjointResidual, EngineError> {
    let n = model.n();
    let ntst = s.mesh.len().saturating_sub(1);
    if ntst == 0 {
        return Err(bad("the mesh needs at least one interval".into()));
    }
    let basis = CollocationBasis::new(s.ncol)?;
    let m = basis.m();
    let len = n * (ntst * m + 1);
    if u.len() != len || phi.len() != len {
        return Err(bad(format!(
            "expected {len} orbit and adjoint values, got {} and {}",
            u.len(),
            phi.len()
        )));
    }
    if s.x0.len() != n || s.params.len() != model.n_p() {
        return Err(bad(format!(
            "expected {n} saddle values and {} parameters, got {} and {}",
            model.n_p(),
            s.x0.len(),
            s.params.len()
        )));
    }
    if !(1..n).contains(&s.ns) {
        return Err(bad(format!(
            "the saddle needs 1 to {} stable eigenvalues, got {}",
            n - 1,
            s.ns
        )));
    }
    let mut p = s.params.to_vec();
    p.push(0.0);

    let mut collocation = vec![0.0; ntst * m * n];
    let (mut uz, mut pz) = (vec![0.0; n], vec![0.0; n]);
    let (mut f, mut jac) = (vec![0.0; n], vec![0.0; n * n]);
    for j in 0..ntst {
        let h = s.mesh[j + 1] - s.mesh[j];
        let first = j * m * n;
        let (ub, pb) = (
            &u[first..first + n * (m + 1)],
            &phi[first..first + n * (m + 1)],
        );
        for i in 0..m {
            for c in 0..n {
                uz[c] = (0..=m)
                    .map(|l| basis.value_at_node(i, l) * ub[l * n + c])
                    .sum();
                pz[c] = (0..=m)
                    .map(|l| basis.value_at_node(i, l) * pb[l * n + c])
                    .sum();
            }
            model.eval(&EntryKind::F, &uz, &p, &[], &mut f)?;
            model.eval(&EntryKind::JacU, &uz, &p, &[], &mut jac)?;
            for c in 0..n {
                let d: f64 = (0..=m)
                    .map(|l| basis.derivative_at_node(i, l) * pb[l * n + c])
                    .sum();
                // (f_x^T phi)_c = sum_r f_x[r, c] phi_r.
                let jt: f64 = (0..n).map(|r| jac[r + n * c] * pz[r]).sum();
                collocation[(j * m + i) * n + c] = d + h * s.period * (jt - gamma * f[c]);
            }
        }
    }

    model.eval(&EntryKind::JacU, s.x0, &p, &[], &mut jac)?;
    let (bs, bu) = tracker.bases(BasisOf::Matrix, &jac, n, s.ns)?;
    let nu = n - s.ns;
    let last = n * ntst * m;
    let mut boundary = vec![0.0; n];
    for r in 0..nu {
        boundary[r] = (0..n).map(|c| bu[c + n * r] * phi[c]).sum();
    }
    for r in 0..s.ns {
        boundary[nu + r] = (0..n).map(|c| bs[c + n * r] * phi[last + c]).sum();
    }
    Ok(AdjointResidual {
        collocation,
        boundary,
    })
}

/// IFS and IFU with the eigenvectors they used.
#[derive(Clone, Debug, PartialEq)]
pub struct InclinationFlips {
    /// `<v_S, phi(0)> exp(-T lambda_S / 2)`, NaN when lambda_S is not real.
    pub ifs: f64,
    /// `<v_{S+1}, phi(1)> exp(T lambda_{S+1} / 2)`, NaN when lambda_{S+1} is not real.
    pub ifu: f64,
    /// The right eigenvectors v_S and v_{S+1} of A, or None where the eigenvalue is not
    /// real. Pass them back as `prev` at the next point to keep their signs.
    pub vectors: [Option<Vec<f64>>; 2],
}

/// The inclination-flip tests IFS and IFU of the saddle with the n by n column-major `a`
/// = f_x(x0, p), `ns` stable eigenvalues and truncation time `period`, at the adjoint ends
/// `phi0` = phi(0) and `phi1` = phi(1).
///
/// lambda_S is the stable eigenvalue and lambda_{S+1} the unstable one with the least
/// absolute real part. The signs of the eigenvectors follow `prev`, as in
/// [`signed_eigvec`].
pub fn inclination_flips(
    a: &[f64],
    n: usize,
    ns: usize,
    period: f64,
    phi0: &[f64],
    phi1: &[f64],
    prev: &[Option<Vec<f64>>; 2],
) -> Result<InclinationFlips, EngineError> {
    if a.len() != n * n || phi0.len() != n || phi1.len() != n {
        return Err(bad(format!(
            "expected a {n} by {n} matrix and two vectors of {n}, got {}, {} and {}",
            a.len(),
            phi0.len(),
            phi1.len()
        )));
    }
    if !(1..n).contains(&ns) {
        return Err(bad(format!(
            "the saddle needs 1 to {} stable eigenvalues, got {ns}",
            n - 1
        )));
    }
    let lam = ascending(&eigenvalues(a, n)?);
    let dot = |v: &[f64], w: &[f64]| v.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();
    let mut out = InclinationFlips {
        ifs: f64::NAN,
        ifu: f64::NAN,
        vectors: [None, None],
    };
    for (k, (l, phi, sign)) in [(lam[ns - 1], phi0, -1.0), (lam[ns], phi1, 1.0)]
        .into_iter()
        .enumerate()
    {
        if !is_real(l) {
            continue;
        }
        let Some(v) = signed_eigvec(a, n, l.0, prev[k].as_deref()) else {
            continue;
        };
        let value = dot(&v, phi) * (sign * period * l.0 / 2.0).exp();
        if k == 0 {
            out.ifs = value;
        } else {
            out.ifu = value;
        }
        out.vectors[k] = Some(v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Column-major from rows.
    fn cm(rows: &[&[f64]]) -> Vec<f64> {
        let n = rows.len();
        (0..n)
            .flat_map(|c| rows.iter().map(move |r| r[c]))
            .collect()
    }

    const NONE: [Option<Vec<f64>>; 2] = [None, None];

    #[test]
    fn flips_match_hand_values_on_a_triangular_matrix() {
        // Eigenvalues -2, -1, 0.5 with ns = 2, so lambda_S = -1 and lambda_{S+1} = 0.5.
        // The eigenvector of -1 is (1, 1, 0)/sqrt 2 and the one of 0.5 is (11/15, 4/3, 1)/r.
        let a = cm(&[&[-2.0, 1.0, 0.5], &[0.0, -1.0, 2.0], &[0.0, 0.0, 0.5]]);
        let t = 2.0;
        let phi0 = [0.3, 0.5, -0.2];
        let phi1 = [1.0, -1.0, 2.0];
        let r = (121.0_f64 / 225.0 + 16.0 / 9.0 + 1.0).sqrt();
        let want_s = 0.8 / 2.0_f64.sqrt() * (t / 2.0_f64).exp();
        let want_u = 1.4 / r * (0.5 * t / 2.0_f64).exp();
        let got = inclination_flips(&a, 3, 2, t, &phi0, &phi1, &NONE).unwrap();
        assert!((got.ifs - want_s).abs() < 1e-12 * want_s.abs(), "{got:?}");
        assert!((got.ifu - want_u).abs() < 1e-12 * want_u.abs(), "{got:?}");
        let vs = got.vectors[0].as_ref().unwrap();
        assert!(
            (vs[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12 && (vs[2]).abs() < 1e-12,
            "{vs:?}"
        );
    }

    #[test]
    fn flips_are_linear_in_phi_and_flip_sign_with_the_eigenvector_reference() {
        let a = cm(&[&[-2.0, 1.0, 0.5], &[0.0, -1.0, 2.0], &[0.0, 0.0, 0.5]]);
        let (p0, p1) = ([0.3, 0.5, -0.2], [1.0, -1.0, 2.0]);
        let base = inclination_flips(&a, 3, 2, 2.0, &p0, &p1, &NONE).unwrap();
        let (q0, q1) = ([0.6, 1.0, -0.4], [-3.0, 3.0, -6.0]);
        let scaled = inclination_flips(&a, 3, 2, 2.0, &q0, &q1, &NONE).unwrap();
        assert!((scaled.ifs - 2.0 * base.ifs).abs() < 1e-12 * base.ifs.abs());
        assert!((scaled.ifu + 3.0 * base.ifu).abs() < 1e-12 * base.ifu.abs());
        // A reference vector in the opposite direction flips both eigenvectors, and so both
        // tests.
        let neg = [
            base.vectors[0]
                .as_ref()
                .map(|v| v.iter().map(|x| -x).collect()),
            base.vectors[1]
                .as_ref()
                .map(|v| v.iter().map(|x| -x).collect()),
        ];
        let flipped = inclination_flips(&a, 3, 2, 2.0, &p0, &p1, &neg).unwrap();
        assert!((flipped.ifs + base.ifs).abs() < 1e-12 * base.ifs.abs());
        assert!((flipped.ifu + base.ifu).abs() < 1e-12 * base.ifu.abs());
    }

    #[test]
    fn a_complex_leading_pair_gives_nan_for_that_side_only() {
        // Stable focus -0.5 +- i and a real unstable eigenvalue 1: IFS is undefined.
        let a = cm(&[&[-0.5, 1.0, 0.0], &[-1.0, -0.5, 0.0], &[0.0, 0.0, 1.0]]);
        let got =
            inclination_flips(&a, 3, 2, 4.0, &[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], &NONE).unwrap();
        assert!(got.ifs.is_nan() && got.vectors[0].is_none());
        let want = (4.0_f64 / 2.0).exp();
        assert!((got.ifu - want).abs() < 1e-12 * want, "{got:?}");
        // The mirror case: a stable real eigenvalue and an unstable focus.
        let a = cm(&[&[-1.0, 0.0, 0.0], &[0.0, 0.5, 1.0], &[0.0, -1.0, 0.5]]);
        let got =
            inclination_flips(&a, 3, 1, 4.0, &[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], &NONE).unwrap();
        assert!(got.ifu.is_nan() && got.vectors[1].is_none());
        let want = (4.0_f64 / 2.0).exp();
        assert!((got.ifs - want).abs() < 1e-12 * want, "{got:?}");
    }

    #[test]
    fn bad_flip_input() {
        let a = cm(&[&[-1.0, 0.0], &[0.0, 1.0]]);
        let z = [0.0; 2];
        assert!(inclination_flips(&a, 2, 0, 1.0, &z, &z, &NONE).is_err());
        assert!(inclination_flips(&a, 2, 2, 1.0, &z, &z, &NONE).is_err());
        assert!(inclination_flips(&a, 2, 1, 1.0, &z[..1], &z, &NONE).is_err());
        assert!(inclination_flips(&a[..3], 2, 1, 1.0, &z, &z, &NONE).is_err());
    }
}
