//! Stable and unstable invariant subspaces of a hyperbolic matrix through the matrix sign
//! function, and the rotation that keeps their bases continuous along a curve.
//!
//! The sign function is computed by Newton's method, S <- (S + S^-1)/2 (Higham, Functions of
//! Matrices, SIAM 2008, ch. 5). The rotation is the one of Beyn (1990, eq. 4.4).
//!
//! The two rotation slots of [`BasisTracker`] follow the per-ITRANS state of PRJCTN in AUTO-07p
//! `src/homcont.f90` (BSD-3, see THIRD_PARTY_NOTICES.md). The bases themselves are our own.

use crate::dense::{LinalgError, Lu, check};
use faer::MatRef;
use faer::linalg::solvers::ColPivQr;

/// Newton steps before the sign iteration gives up.
pub const SIGN_MAX_STEPS: usize = 100;
/// Relative 1-norm change at which the sign iteration stops.
pub const SIGN_TOL: f64 = 1e-14;

fn norm1(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|j| a[j * n..(j + 1) * n].iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>, LinalgError> {
    let lu = Lu::new(a, n)?;
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let col = &mut inv[j * n..(j + 1) * n];
        col[j] = 1.0;
        lu.solve(col)?;
    }
    Ok(inv)
}

/// The sign function of the n-by-n column-major matrix `a` and the number of Newton steps
/// it took.
///
/// Stops when ||S_{k+1} - S_k||_1 <= 1e-14 ||S_{k+1}||_1. `NoConvergence` after 100 steps,
/// `Singular` when an iterate is singular, which happens when `a` has an eigenvalue on or
/// very near the imaginary axis.
pub fn sign_function(a: &[f64], n: usize) -> Result<(Vec<f64>, usize), LinalgError> {
    check(a, n, n)?;
    if n == 0 {
        return Err(LinalgError::BadShape);
    }
    let mut s = a.to_vec();
    for step in 1..=SIGN_MAX_STEPS {
        let inv = inverse(&s, n)?;
        let next: Vec<f64> = s.iter().zip(&inv).map(|(x, y)| 0.5 * (x + y)).collect();
        if next.iter().any(|x| !x.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let diff: Vec<f64> = next.iter().zip(&s).map(|(x, y)| x - y).collect();
        let done = norm1(&diff, n) <= SIGN_TOL * norm1(&next, n);
        s = next;
        if done {
            return Ok((s, step));
        }
    }
    Err(LinalgError::NoConvergence)
}

/// Orthonormal bases of the stable and unstable invariant subspaces of `a`, as n-by-ns and
/// n-by-(n - ns) column-major matrices.
///
/// The ranks are fixed by the caller: QR with column pivoting of the spectral projector
/// (I - S)/2 gives the first `ns` columns, and of (I + S)/2 the first n - ns. `BadShape`
/// when ns > n.
pub fn invariant_bases(
    a: &[f64],
    n: usize,
    ns: usize,
) -> Result<(Vec<f64>, Vec<f64>), LinalgError> {
    if ns > n {
        return Err(LinalgError::BadShape);
    }
    let (s, _) = sign_function(a, n)?;
    let basis = |sign: f64, k: usize| -> Result<Vec<f64>, LinalgError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut p = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let id = if i == j { 1.0 } else { 0.0 };
                p[i + n * j] = 0.5 * (id + sign * s[i + n * j]);
            }
        }
        let qr = ColPivQr::new(MatRef::from_column_major_slice(&p, n, n));
        let q = qr.compute_Q();
        let mut out = Vec::with_capacity(n * k);
        for j in 0..k {
            for i in 0..n {
                out.push(q[(i, j)]);
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(out)
    };
    Ok((basis(-1.0, ns)?, basis(1.0, n - ns)?))
}

/// Rotates the n-by-k basis `v` against the n-by-k reference basis `c`: solves
/// (C^T V) D = C^T C and returns L = V D, so that C^T L = C^T C.
///
/// `Singular` when C^T V is singular, which means the two subspaces are not close.
pub fn rotate_basis(v: &[f64], c: &[f64], n: usize, k: usize) -> Result<Vec<f64>, LinalgError> {
    check(v, n, k)?;
    check(c, n, k)?;
    if k == 0 {
        return Ok(Vec::new());
    }
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f64>();
    let mut ctv = vec![0.0; k * k];
    let mut ctc = vec![0.0; k * k];
    for j in 0..k {
        for i in 0..k {
            ctv[i + k * j] = dot(&c[i * n..(i + 1) * n], &v[j * n..(j + 1) * n]);
            ctc[i + k * j] = dot(&c[i * n..(i + 1) * n], &c[j * n..(j + 1) * n]);
        }
    }
    let lu = Lu::new(&ctv, k)?;
    let mut l = vec![0.0; n * k];
    for j in 0..k {
        let d = &mut ctc[j * k..(j + 1) * k];
        lu.solve(d)?;
        for (t, dt) in d.iter().enumerate() {
            for i in 0..n {
                l[i + n * j] += v[i + n * t] * dt;
            }
        }
    }
    Ok(l)
}

/// The matrix whose invariant subspaces a set of boundary rows uses (HomCont's ITRANS).
///
/// The rows of the orbit take the subspaces of A^T, the rows of the adjoint those of A.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisOf {
    /// The subspaces of A^T.
    Transpose,
    /// The subspaces of A.
    Matrix,
}

impl BasisOf {
    fn slot(self) -> usize {
        match self {
            BasisOf::Transpose => 0,
            BasisOf::Matrix => 1,
        }
    }
}

/// An orthonormal basis of the stable subspace (n by ns) and one of the unstable subspace
/// (n by n - ns), column-major.
pub type BasisPair = (Vec<f64>, Vec<f64>);

/// The reference bases of the rotation of Beyn, one pair for each [`BasisOf`].
///
/// AUTO keeps this state per (equilibrium, ITRANS). A homoclinic orbit has one equilibrium,
/// so only the kind separates the slots. A slot with no reference returns the bases as the
/// sign function gives them.
#[derive(Clone, Debug, Default)]
pub struct BasisTracker {
    refs: [Option<BasisPair>; 2],
}

fn transposed(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            t[c + n * r] = a[r + n * c];
        }
    }
    t
}

impl BasisTracker {
    /// A tracker with no references.
    pub fn new() -> Self {
        Self::default()
    }

    /// The reference of `of`, if one was set.
    pub fn reference(&self, of: BasisOf) -> Option<&BasisPair> {
        self.refs[of.slot()].as_ref()
    }

    /// Makes `pair` the reference of `of`.
    pub fn set(&mut self, of: BasisOf, pair: BasisPair) {
        self.refs[of.slot()] = Some(pair);
    }

    /// The bases of the n-by-n column-major `a` (or of its transpose, as `of` says) with `ns`
    /// stable directions, rotated against the reference of `of` when there is one.
    ///
    /// The reference is not changed. `BadShape` when the reference has other sizes.
    pub fn bases(
        &self,
        of: BasisOf,
        a: &[f64],
        n: usize,
        ns: usize,
    ) -> Result<BasisPair, LinalgError> {
        check(a, n, n)?;
        let (vs, vu) = match of {
            BasisOf::Matrix => invariant_bases(a, n, ns)?,
            BasisOf::Transpose => invariant_bases(&transposed(a, n), n, ns)?,
        };
        match &self.refs[of.slot()] {
            None => Ok((vs, vu)),
            Some((cs, cu)) => Ok((
                rotate_basis(&vs, cs, n, ns)?,
                rotate_basis(&vu, cu, n, n - ns)?,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::testutil::Rng;

    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for j in 0..n {
            for t in 0..k {
                for i in 0..m {
                    c[i + m * j] += a[i + m * t] * b[t + k * j];
                }
            }
        }
        c
    }

    fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        let mut t = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                t[j + n * i] = a[i + m * j];
            }
        }
        t
    }

    /// Distance of each column of `x` (n-by-p) from the span of the orthonormal `q` (n-by-k).
    fn out_of_span(x: &[f64], q: &[f64], n: usize, p: usize, k: usize) -> f64 {
        let qtx = matmul(&transpose(q, n, k), x, k, n, p);
        let back = matmul(q, &qtx, n, k, p);
        x.iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// Random T^-1 B T with B block upper triangular, from a seed.
    fn similar(b: &[f64], n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng(seed);
        let mut t: Vec<f64> = (0..n * n).map(|_| 0.3 * rng.unit()).collect();
        for i in 0..n {
            t[i + n * i] += 1.0;
        }
        let ti = inverse(&t, n).unwrap();
        matmul(&matmul(&ti, b, n, n, n), &t, n, n, n)
    }

    fn check_bases(a: &[f64], n: usize, ns: usize) -> usize {
        let at = transpose(a, n, n);
        let (_, steps) = sign_function(&at, n).unwrap();
        let (ls, lu) = invariant_bases(&at, n, ns).unwrap();
        let nu = n - ns;
        // Orthonormal columns.
        let gs = matmul(&transpose(&ls, n, ns), &ls, ns, n, ns);
        for j in 0..ns {
            for i in 0..ns {
                let id = if i == j { 1.0 } else { 0.0 };
                assert!((gs[i + ns * j] - id).abs() < 1e-12);
            }
        }
        // A^T L_s stays in span L_s, and A^T L_u in span L_u.
        let als = matmul(&at, &ls, n, n, ns);
        assert!(out_of_span(&als, &ls, n, ns, ns) < 1e-10);
        let alu = matmul(&at, &lu, n, n, nu);
        assert!(out_of_span(&alu, &lu, n, nu, nu) < 1e-10);
        // L_s^T is zero on the unstable subspace of A and L_u^T on its stable subspace,
        // with those subspaces computed from A itself.
        let (es, eu) = invariant_bases(a, n, ns).unwrap();
        let z = matmul(&transpose(&ls, n, ns), &eu, ns, n, nu);
        assert!(z.iter().all(|x| x.abs() < 1e-10), "{z:?}");
        let z = matmul(&transpose(&lu, n, nu), &es, nu, n, ns);
        assert!(z.iter().all(|x| x.abs() < 1e-10), "{z:?}");
        steps
    }

    #[test]
    fn diagonal_saddle() {
        let a = [-2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        let (s, _) = sign_function(&a, 3).unwrap();
        let want = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
        for (x, y) in s.iter().zip(&want) {
            assert!((x - y).abs() < 1e-15);
        }
        check_bases(&a, 3, 2);
    }

    #[test]
    fn jordan_block_in_stable_part() {
        // A double eigenvalue -0.4 with a Jordan block, plus -1 and +1, then a similarity.
        let mut b = vec![0.0; 16];
        b[0] = -0.4;
        b[4] = 1.0; // (0,1)
        b[5] = -0.4;
        b[10] = -1.0;
        b[8] = 0.5; // (0,2) coupling
        b[15] = 1.0;
        b[12] = 0.7; // (0,3)
        for seed in 1..6 {
            let a = similar(&b, 4, seed);
            let steps = check_bases(&a, 4, 3);
            assert!(steps < 30, "steps {steps}");
        }
    }

    #[test]
    fn complex_pair() {
        // Stable focus -0.3 +- 2i, stable -1, unstable 0.5 +- 1i.
        let mut b = vec![0.0; 25];
        let set = |b: &mut Vec<f64>, i: usize, j: usize, v: f64| b[i + 5 * j] = v;
        set(&mut b, 0, 0, -0.3);
        set(&mut b, 0, 1, 2.0);
        set(&mut b, 1, 0, -2.0);
        set(&mut b, 1, 1, -0.3);
        set(&mut b, 2, 2, -1.0);
        set(&mut b, 3, 3, 0.5);
        set(&mut b, 3, 4, 1.0);
        set(&mut b, 4, 3, -1.0);
        set(&mut b, 4, 4, 0.5);
        set(&mut b, 0, 3, 0.4);
        set(&mut b, 1, 4, -0.2);
        set(&mut b, 2, 3, 0.3);
        for seed in 10..14 {
            let a = similar(&b, 5, seed);
            let steps = check_bases(&a, 5, 3);
            assert!(steps < 30, "steps {steps}");
        }
    }

    #[test]
    fn stiff_spread() {
        // Eigenvalues -1e3, -1e-2 and 0.3, with off-diagonal coupling.
        let b = [-1e3, 0.0, 0.0, 5.0, -1e-2, 0.0, 2.0, 1.0, 0.3];
        for seed in 20..24 {
            let a = similar(&b, 3, seed);
            let steps = check_bases(&a, 3, 2);
            assert!(steps < 40, "steps {steps}");
        }
    }

    #[test]
    fn imaginary_axis_eigenvalue_fails() {
        // A rotation (eigenvalues +-i) never converges, and a zero eigenvalue is singular.
        let a = [0.0, -1.0, 1.0, 0.0];
        assert!(sign_function(&a, 2).is_err());
        let a = [0.0, 0.0, 0.0, -1.0];
        assert_eq!(sign_function(&a, 2), Err(LinalgError::Singular));
    }

    #[test]
    fn rotation_identity_and_subspace() {
        let mut rng = Rng(7);
        let n = 5;
        let k = 2;
        let c: Vec<f64> = (0..n * k).map(|_| rng.unit()).collect();
        // V spans a nearby subspace: C times an invertible k-by-k plus a small perturbation.
        let m = [1.3, -0.4, 0.2, 0.9];
        let mut v = matmul(&c, &m, n, k, k);
        for x in v.iter_mut() {
            *x += 1e-3 * rng.unit();
        }
        let l = rotate_basis(&v, &c, n, k).unwrap();
        let ctl = matmul(&transpose(&c, n, k), &l, k, n, k);
        let ctc = matmul(&transpose(&c, n, k), &c, k, n, k);
        for (x, y) in ctl.iter().zip(&ctc) {
            assert!((x - y).abs() < 1e-12);
        }
        // L spans the same subspace as V: orthonormalize V and check L against it.
        let qr = ColPivQr::new(MatRef::from_column_major_slice(&v, n, k));
        let q = qr.compute_thin_Q();
        let q: Vec<f64> = (0..k)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| q[(i, j)])
            .collect();
        assert!(out_of_span(&l, &q, n, k, k) < 1e-12);
        // Rotating a basis against itself returns it.
        let same = rotate_basis(&c, &c, n, k).unwrap();
        for (x, y) in same.iter().zip(&c) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn rotation_does_not_depend_on_the_basis() {
        let mut rng = Rng(9);
        let n = 4;
        let k = 2;
        let c: Vec<f64> = (0..n * k).map(|_| rng.unit()).collect();
        let v: Vec<f64> = c.iter().map(|x| x + 0.05 * rng.unit()).collect();
        let m = [0.7, 1.1, -0.6, 0.8];
        let v2 = matmul(&v, &m, n, k, k);
        let l1 = rotate_basis(&v, &c, n, k).unwrap();
        let l2 = rotate_basis(&v2, &c, n, k).unwrap();
        for (x, y) in l1.iter().zip(&l2) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    /// An orthonormal basis of the span of the n-by-k `v`.
    fn orthonormal(v: &[f64], n: usize, k: usize) -> Vec<f64> {
        let q = ColPivQr::new(MatRef::from_column_major_slice(v, n, k)).compute_thin_Q();
        (0..k)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .map(|(i, j)| q[(i, j)])
            .collect()
    }

    fn saddle4(seed: u64) -> Vec<f64> {
        let b = [
            -1.0, 0.0, 0.0, 0.0, 0.3, -2.0, 0.0, 0.0, 0.2, 0.1, 0.5, 0.0, -0.4, 0.2, 0.6, 1.5,
        ];
        similar(&b, 4, seed)
    }

    #[test]
    fn tracker_without_reference_is_the_sign_function_bases() {
        let a = saddle4(3);
        let t = BasisTracker::new();
        assert!(t.reference(BasisOf::Matrix).is_none());
        assert!(t.reference(BasisOf::Transpose).is_none());
        let got = t.bases(BasisOf::Matrix, &a, 4, 2).unwrap();
        assert_eq!(got, invariant_bases(&a, 4, 2).unwrap());
        let got = t.bases(BasisOf::Transpose, &a, 4, 2).unwrap();
        assert_eq!(got, invariant_bases(&transpose(&a, 4, 4), 4, 2).unwrap());
    }

    #[test]
    fn tracker_slots_are_independent_and_bases_keeps_the_reference() {
        let a = saddle4(4);
        let mut t = BasisTracker::new();
        let pair = t.bases(BasisOf::Transpose, &a, 4, 2).unwrap();
        t.set(BasisOf::Transpose, pair.clone());
        assert_eq!(t.reference(BasisOf::Transpose), Some(&pair));
        assert!(t.reference(BasisOf::Matrix).is_none());
        let _ = t.bases(BasisOf::Transpose, &saddle4(5), 4, 2).unwrap();
        assert_eq!(t.reference(BasisOf::Transpose), Some(&pair));
    }

    #[test]
    fn tracker_rotates_each_slot_against_its_own_reference() {
        let a0 = saddle4(6);
        let mut a1 = a0.clone();
        for (k, x) in a1.iter_mut().enumerate() {
            *x += 1e-3 * (k as f64).sin();
        }
        let mut t = BasisTracker::new();
        for of in [BasisOf::Matrix, BasisOf::Transpose] {
            let r = t.bases(of, &a0, 4, 2).unwrap();
            t.set(of, r);
        }
        for of in [BasisOf::Matrix, BasisOf::Transpose] {
            let (cs, cu) = t.reference(of).unwrap().clone();
            let (ls, lu) = t.bases(of, &a1, 4, 2).unwrap();
            for (l, c, k) in [(&ls, &cs, 2), (&lu, &cu, 2)] {
                let ctl = matmul(&transpose(c, 4, k), l, k, 4, k);
                let ctc = matmul(&transpose(c, 4, k), c, k, 4, k);
                for (x, y) in ctl.iter().zip(&ctc) {
                    assert!((x - y).abs() < 1e-12, "{of:?}");
                }
            }
            // The rotated bases still span the invariant subspaces of a1 (or its transpose).
            let m = if of == BasisOf::Matrix {
                a1.clone()
            } else {
                transpose(&a1, 4, 4)
            };
            let q = orthonormal(&ls, 4, 2);
            let am = matmul(&m, &ls, 4, 4, 2);
            assert!(out_of_span(&am, &q, 4, 2, 2) < 1e-9, "{of:?}");
        }
    }

    #[test]
    fn tracker_rejects_a_reference_of_another_size() {
        let mut t = BasisTracker::new();
        t.set(
            BasisOf::Matrix,
            (vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]),
        );
        let a = saddle4(7);
        assert_eq!(
            t.bases(BasisOf::Matrix, &a, 4, 2),
            Err(LinalgError::BadShape)
        );
    }

    #[test]
    fn bad_input() {
        assert_eq!(sign_function(&[1.0; 3], 2), Err(LinalgError::BadShape));
        assert_eq!(
            invariant_bases(&[1.0, 0.0, 0.0, -1.0], 2, 3),
            Err(LinalgError::BadShape)
        );
        assert_eq!(
            rotate_basis(&[1.0, 0.0], &[0.0, 1.0], 2, 1),
            Err(LinalgError::Singular)
        );
    }
}
