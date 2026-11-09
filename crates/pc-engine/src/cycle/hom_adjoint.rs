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
//! [`adjoint_residual`] evaluates the adjoint equations at a given phi. [`solve_adjoint`]
//! solves them for phi and gamma on a given orbit with the condensation of `pc-linalg`
//! (SPEC §8b-3), and [`orient_adjoint`] fixes the sign of the solution. `HomProblem` calls
//! them when `inclination_flips` is on.

use super::basis::CollocationBasis;
use super::hom::{ascending, is_real, signed_eigvec};
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::EntryKind;
use pc_linalg::{BasisOf, BasisTracker, BlockSystem, CondenseShape, Condenser, eigenvalues};

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

/// Checks the setup against the model and returns the number of intervals and the
/// collocation basis.
fn check_setup(
    model: &dyn ModelEval,
    s: &AdjointSetup,
) -> Result<(usize, CollocationBasis), EngineError> {
    let n = model.n();
    let ntst = s.mesh.len().saturating_sub(1);
    if ntst == 0 {
        return Err(bad("the mesh needs at least one interval".into()));
    }
    let basis = CollocationBasis::new(s.ncol)?;
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
    Ok((ntst, basis))
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
    let (ntst, basis) = check_setup(model, s)?;
    let m = basis.m();
    let len = n * (ntst * m + 1);
    if u.len() != len || phi.len() != len {
        return Err(bad(format!(
            "expected {len} orbit and adjoint values, got {} and {}",
            u.len(),
            phi.len()
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

/// The bounded adjoint of an orbit and its multiplier.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjointSolution {
    /// phi at the base points, n (ntst ncol + 1) values, with 2-norm 1. The sign is free;
    /// see [`orient_adjoint`].
    pub phi: Vec<f64>,
    /// gamma, the multiplier of f in the adjoint equation.
    pub gamma: f64,
}

/// The seed of the first normalization row of [`solve_adjoint`].
const NORMAL_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// n values in [-1, 1) from an xorshift64 generator started at `seed`.
fn pseudo_random(seed: u64, len: usize) -> Vec<f64> {
    let mut x = seed.max(1);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

/// Solves the adjoint equations of the orbit `u` (n (ntst ncol + 1) base-point values) for
/// phi and gamma and scales phi to 2-norm 1 (SPEC §8b-3).
///
/// The rows are the collocation rows of [`adjoint_residual`] with gamma in a column of its
/// own, and its n boundary rows. That is one unknown more than rows, so the system has a
/// null vector. A normalization row `<r, phi> = 1` makes the system square, and the
/// condensation of `pc-linalg` solves it. The first r is pseudo-random with a fixed seed.
/// It is then replaced by the first solution and the system is solved again, so that the
/// row is not nearly orthogonal to the null vector.
pub fn solve_adjoint(
    model: &mut dyn ModelEval,
    s: &AdjointSetup,
    tracker: &BasisTracker,
    u: &[f64],
) -> Result<AdjointSolution, EngineError> {
    solve_adjoint_seeded(model, s, tracker, u, NORMAL_SEED)
}

fn solve_adjoint_seeded(
    model: &mut dyn ModelEval,
    s: &AdjointSetup,
    tracker: &BasisTracker,
    u: &[f64],
    seed: u64,
) -> Result<AdjointSolution, EngineError> {
    let n = model.n();
    let (ntst, basis) = check_setup(model, s)?;
    let m = basis.m();
    let nodes = ntst * m + 1;
    if u.len() != n * nodes {
        return Err(bad(format!(
            "expected {} orbit values, got {}",
            n * nodes,
            u.len()
        )));
    }
    let shape = CondenseShape::new(n, m, ntst, 1)?;
    let (nm, tcol) = (n * m, n * (m + 1));
    let bcols = tcol + 1;
    let mut p = s.params.to_vec();
    p.push(0.0);

    // Interval blocks, column-major: row i n + c, column l n + c2 for phi_{jm+l}[c2], and
    // the column tcol for gamma. The row is d + h T (f_x^T phi - gamma f).
    let mut blocks = vec![0.0; ntst * nm * bcols];
    let (mut uz, mut f) = (vec![0.0; n], vec![0.0; n]);
    let mut jac = vec![0.0; n * n];
    for j in 0..ntst {
        let h = s.mesh[j + 1] - s.mesh[j];
        let ht = h * s.period;
        let blk = &mut blocks[j * nm * bcols..(j + 1) * nm * bcols];
        let ub = &u[j * m * n..j * m * n + tcol];
        for i in 0..m {
            for (c, z) in uz.iter_mut().enumerate() {
                *z = (0..=m)
                    .map(|l| basis.value_at_node(i, l) * ub[l * n + c])
                    .sum();
            }
            model.eval(&EntryKind::F, &uz, &p, &[], &mut f)?;
            model.eval(&EntryKind::JacU, &uz, &p, &[], &mut jac)?;
            for c in 0..n {
                let row = i * n + c;
                for l in 0..=m {
                    let (d, v) = (basis.derivative_at_node(i, l), basis.value_at_node(i, l));
                    for c2 in 0..n {
                        let diag = if c == c2 { d } else { 0.0 };
                        // The entry of phi_l[c2] in (f_x^T phi)_c is f_x[c2, c].
                        blk[row + nm * (l * n + c2)] = diag + ht * v * jac[c2 + n * c];
                    }
                }
                blk[row + nm * tcol] = -ht * f[c];
            }
        }
    }

    // Boundary rows, n by (2n + 1): B_u^T on phi(0), then B_s^T on phi(1).
    model.eval(&EntryKind::JacU, s.x0, &p, &[], &mut jac)?;
    let (bs, bu) = tracker.bases(BasisOf::Matrix, &jac, n, s.ns)?;
    let nu = n - s.ns;
    let mut bc = vec![0.0; n * (2 * n + 1)];
    for r in 0..nu {
        for c in 0..n {
            bc[r + n * c] = bu[c + n * r];
        }
    }
    for r in 0..s.ns {
        for c in 0..n {
            bc[(nu + r) + n * (n + c)] = bs[c + n * r];
        }
    }

    let dim = shape.dim();
    let parallel = shape.parallel_pays();
    let mut condenser = Condenser::new(shape)?;
    let mut rhs = vec![0.0; dim];
    rhs[dim - 1] = 1.0;
    let mut border = pseudo_random(seed, dim);
    border[dim - 1] = 0.0;
    let mut sol = vec![0.0; dim];
    for _ in 0..2 {
        let sys = BlockSystem {
            blocks: &blocks,
            bc: &bc,
            border: &border,
        };
        condenser.solve(&sys, &rhs, &mut sol, parallel)?;
        let norm = sol[..dim - 1].iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(EngineError::Singular(
                "the adjoint solve returned a zero vector".into(),
            ));
        }
        border[..dim - 1]
            .iter_mut()
            .zip(&sol)
            .for_each(|(r, z)| *r = z / norm);
    }
    let norm = sol[..dim - 1].iter().map(|v| v * v).sum::<f64>().sqrt();
    let gamma = sol[dim - 1] / norm;
    sol.truncate(dim - 1);
    sol.iter_mut().for_each(|v| *v /= norm);
    Ok(AdjointSolution { phi: sol, gamma })
}

/// Fixes the sign of `phi` (n values per base point, ntst ncol + 1 points) and returns its
/// end values `[phi(0), phi(1)]`, 2n numbers (SPEC §8b-3).
///
/// With `prev` None, the component of largest magnitude of the ends is made positive. With
/// the whole `phi` of the previous accepted point as `prev` (on the same mesh), `phi` is
/// flipped when the dot product of the two vectors over all base points is negative. The
/// ends are not used for that: phi has unit norm over all points, so its ends are a small
/// and noisy share of it, and their dot product changes sign erratically at a long T.
/// A `prev` of another length is ignored, as if there were none.
pub fn orient_adjoint(phi: &mut [f64], n: usize, prev: Option<&[f64]>) -> Vec<f64> {
    let mut ends = phi[..n].to_vec();
    ends.extend_from_slice(&phi[phi.len() - n..]);
    let flip = match prev.filter(|q| q.len() == phi.len()) {
        Some(q) => phi.iter().zip(q).map(|(a, b)| a * b).sum::<f64>() < 0.0,
        None => {
            let big = ends
                .iter()
                .fold(0.0_f64, |m, v| if v.abs() > m.abs() { *v } else { m });
            big < 0.0
        }
    };
    if flip {
        phi.iter_mut().for_each(|v| *v = -*v);
        ends.iter_mut().for_each(|v| *v = -*v);
    }
    ends
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

    /// f(x) = A x with a quadratic term in the first row, so f_x depends on x.
    struct Quad {
        a: Vec<f64>,
    }

    impl ModelEval for Quad {
        fn n(&self) -> usize {
            3
        }
        fn n_p(&self) -> usize {
            0
        }
        fn eval(
            &mut self,
            kind: &EntryKind,
            x: &[f64],
            _p: &[f64],
            _v: &[f64],
            out: &mut [f64],
        ) -> Result<(), EngineError> {
            match kind {
                EntryKind::F => {
                    for (r, o) in out.iter_mut().enumerate() {
                        *o = (0..3).map(|c| self.a[r + 3 * c] * x[c]).sum();
                    }
                    out[0] += x[1] * x[1];
                }
                EntryKind::JacU => {
                    out.copy_from_slice(&self.a);
                    out[3] += 2.0 * x[1];
                }
                _ => unreachable!(),
            }
            Ok(())
        }
    }

    #[test]
    fn the_solution_does_not_depend_on_the_normalization_seed() {
        let a = cm(&[&[-2.0, 1.0, 0.5], &[0.3, -1.0, 2.0], &[0.0, 0.2, 0.5]]);
        let (ntst, ncol) = (6, 3);
        let mesh: Vec<f64> = (0..=ntst).map(|j| j as f64 / ntst as f64).collect();
        let u: Vec<f64> = (0..3 * (ntst * ncol + 1))
            .map(|i| ((i as f64) * 0.37).sin())
            .collect();
        let setup = AdjointSetup {
            ncol,
            mesh: &mesh,
            period: 3.0,
            x0: &[0.0; 3],
            params: &[],
            ns: 2,
        };
        let mut model = Quad { a };
        let tracker = BasisTracker::new();
        let base = solve_adjoint_seeded(&mut model, &setup, &tracker, &u, 1).unwrap();
        for seed in [2, 12345, 0xDEAD_BEEF] {
            let other = solve_adjoint_seeded(&mut model, &setup, &tracker, &u, seed).unwrap();
            let dot: f64 = base.phi.iter().zip(&other.phi).map(|(x, y)| x * y).sum();
            let sgn = dot.signum();
            let err = base
                .phi
                .iter()
                .zip(&other.phi)
                .fold(0.0_f64, |m, (x, y)| m.max((x - sgn * y).abs()));
            assert!(err < 1e-9, "seed {seed}: {err:e}");
            assert!((base.gamma - sgn * other.gamma).abs() < 1e-9, "seed {seed}");
        }
    }

    #[test]
    fn bad_solve_input() {
        let a = cm(&[&[-2.0, 0.0, 0.0], &[0.0, -1.0, 0.0], &[0.0, 0.0, 0.5]]);
        let mesh = [0.0, 1.0];
        let u: Vec<f64> = (0..12).map(|i| 0.1 + ((i as f64) * 0.7).sin()).collect();
        let mut setup = AdjointSetup {
            ncol: 3,
            mesh: &mesh,
            period: 1.0,
            x0: &[0.0; 3],
            params: &[],
            ns: 2,
        };
        let mut model = Quad { a };
        let t = BasisTracker::new();
        assert!(solve_adjoint(&mut model, &setup, &t, &u).is_ok());
        assert!(solve_adjoint(&mut model, &setup, &t, &u[..9]).is_err());
        setup.ns = 3;
        assert!(solve_adjoint(&mut model, &setup, &t, &u).is_err());
        setup.ns = 2;
        setup.mesh = &[0.0];
        assert!(solve_adjoint(&mut model, &setup, &t, &u).is_err());
    }

    /// phi with n = 2 and three base points.
    fn phi2(first: [f64; 2], mid: [f64; 2], last: [f64; 2]) -> Vec<f64> {
        [first, mid, last].concat()
    }

    #[test]
    fn the_first_point_gets_its_largest_end_component_positive() {
        let mut phi = phi2([0.2, -0.3], [9.0, 9.0], [0.1, -0.7]);
        let ends = orient_adjoint(&mut phi, 2, None);
        assert_eq!(ends, vec![-0.2, 0.3, -0.1, 0.7]);
        assert_eq!(phi, phi2([-0.2, 0.3], [-9.0, -9.0], [-0.1, 0.7]));
        // Already positive: unchanged, and the middle values play no role in the choice.
        let mut phi = phi2([0.2, -0.3], [-99.0, 0.0], [0.1, 0.7]);
        let before = phi.clone();
        orient_adjoint(&mut phi, 2, None);
        assert_eq!(phi, before);
    }

    #[test]
    fn later_points_follow_the_previous_vector() {
        let prev = phi2([1.0, 0.0], [5.0, 5.0], [0.0, 1.0]);
        let mut same = phi2([0.9, 0.1], [5.0, 4.0], [-0.1, 1.1]);
        let before = same.clone();
        orient_adjoint(&mut same, 2, Some(&prev));
        assert_eq!(same, before);
        let mut opposite = phi2([-0.9, -0.1], [-5.0, -4.0], [0.1, -1.1]);
        let ends = orient_adjoint(&mut opposite, 2, Some(&prev));
        assert_eq!(opposite, phi2([0.9, 0.1], [5.0, 4.0], [-0.1, 1.1]));
        assert_eq!(ends, vec![0.9, 0.1, -0.1, 1.1]);
        // A tie (zero dot product) keeps the sign.
        let mut tie = phi2([0.0, 1.0], [0.0, 0.0], [1.0, 0.0]);
        let tie_prev = phi2([1.0, 0.0], [0.0, 0.0], [0.0, 1.0]);
        let before = tie.clone();
        orient_adjoint(&mut tie, 2, Some(&tie_prev));
        assert_eq!(tie, before);
    }

    #[test]
    fn the_interior_decides_where_the_ends_are_tiny_and_disagree() {
        // The ends are 1e-8 of the norm and their dot product with the previous ends is
        // negative, as at a long T, but the vectors are the same up to rounding.
        let prev = phi2([2e-8, -3e-8], [0.6, 0.8], [-1e-8, 4e-8]);
        let mut phi = phi2([-5e-9, 7e-9], [0.6, 0.8], [3e-9, -2e-9]);
        let before = phi.clone();
        let ends = orient_adjoint(&mut phi, 2, Some(&prev));
        assert_eq!(phi, before);
        assert_eq!(ends, vec![-5e-9, 7e-9, 3e-9, -2e-9]);
        // And the same vector with its sign reversed is flipped back.
        let mut neg: Vec<f64> = before.iter().map(|v| -v).collect();
        orient_adjoint(&mut neg, 2, Some(&prev));
        assert_eq!(neg, before);
    }

    #[test]
    fn a_previous_vector_of_another_length_is_ignored() {
        let prev = [1.0, 0.0, 0.0, 1.0];
        let mut phi = phi2([-0.2, 0.3], [9.0, 9.0], [-0.1, -0.7]);
        orient_adjoint(&mut phi, 2, Some(&prev));
        // The first-point rule: the largest end component (-0.7) is made positive.
        assert_eq!(phi, phi2([0.2, -0.3], [-9.0, -9.0], [0.1, 0.7]));
    }

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
