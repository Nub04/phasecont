//! Mesh adaptation for periodic orbits (SPEC §7). The N + 1 mesh points move so that a
//! monitor of the jumps of the m-th derivative of the orbit is equidistributed (Russell
//! and Christiansen 1978), and the orbit and the tangent are interpolated onto the new
//! mesh. N and m stay the same.
//!
//! Ported from AUTO-07p src/mesh.f90 (EQDF, NEWMSH and INTERP, the mesh part of ADAPT),
//! https://github.com/auto-07p/auto-07p at commit cb5dd4616be3fabd3ace9fb9f9b22fcbac9fc447,
//! BSD license, Copyright (c) 1979-2007 E. J. Doedel, California Institute of Technology,
//! and Concordia University. See THIRD_PARTY_NOTICES.md.

use super::basis::CollocationBasis;

/// AUTO's HMACH: when no m-th derivative is above it, the mesh stays as it is.
const SMALL_DERIVATIVE: f64 = 1e-7;

/// The m-th derivative of the orbit on interval `j` into `out` (n values), from its base
/// points in `u` (point-major).
fn mth_derivative(
    basis: &CollocationBasis,
    n: usize,
    mesh: &[f64],
    u: &[f64],
    j: usize,
    out: &mut [f64],
) {
    let m = basis.m();
    let sc = (mesh[j + 1] - mesh[j]).powi(-(m as i32));
    let dm = basis.mth_derivative();
    for (c, o) in out.iter_mut().enumerate() {
        *o = sc * (0..=m).map(|l| dm[l] * u[(j * m + l) * n + c]).sum::<f64>();
    }
}

/// The equidistribution function at the N + 1 mesh points and its slope on each interval
/// (EQDF, periodic case). The slope on interval j sums, over the components, the jump of
/// the m-th derivative at its left end, over the mean length of the two intervals there,
/// to the power 1 / (m + 1). Interval N - 1 takes the jump across the end of the period.
/// `None` when every m-th derivative is below [`SMALL_DERIVATIVE`].
fn equidistribution(
    basis: &CollocationBasis,
    n: usize,
    mesh: &[f64],
    u: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ntst = mesh.len() - 1;
    let pwr = 1.0 / (basis.m() as f64 + 1.0);
    let h = |j: usize| mesh[j + 1] - mesh[j];
    let mut eqf = vec![0.0; ntst + 1];
    let mut deqf = vec![0.0; ntst];
    let mut small = true;
    let (mut nd, mut hd) = (vec![0.0; n], vec![0.0; n]);
    for j in 0..ntst {
        mth_derivative(basis, n, mesh, u, j, &mut nd);
        let mut e = 0.0;
        for c in 0..n {
            if j > 0 {
                e += (2.0 * (nd[c] - hd[c]) / (h(j - 1) + h(j))).abs().powf(pwr);
            }
            if nd[c].abs() > SMALL_DERIVATIVE {
                small = false;
            }
        }
        hd.copy_from_slice(&nd);
        if j > 0 {
            eqf[j] = eqf[j - 1] + h(j - 1) * e;
            deqf[j - 1] = e;
        }
    }
    // Extend by periodicity: the jump from the last interval to the first.
    mth_derivative(basis, n, mesh, u, 0, &mut nd);
    let dtav = (h(ntst - 1) + h(0)) / 2.0;
    let mut e = 0.0;
    for c in 0..n {
        e += ((nd[c] - hd[c]) / dtav).abs().powf(pwr);
        if nd[c].abs() > SMALL_DERIVATIVE {
            small = false;
        }
    }
    eqf[ntst] = eqf[ntst - 1] + h(ntst - 1) * e;
    deqf[ntst - 1] = e;
    (!small).then_some((eqf, deqf))
}

/// The new mesh for the orbit `u` on `mesh` (NEWMSH with the same N): the points where
/// the equidistribution function takes the values i / N of its range. `None` keeps the
/// mesh: when the derivatives are small, when the monitor is zero or not finite, or when
/// the new points are not strictly increasing.
pub(crate) fn new_mesh(
    basis: &CollocationBasis,
    n: usize,
    mesh: &[f64],
    u: &[f64],
) -> Option<Vec<f64>> {
    let ntst = mesh.len() - 1;
    if ntst < 2 {
        return None;
    }
    let (eqf, deqf) = equidistribution(basis, n, mesh, u)?;
    let total = eqf[ntst];
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let dal = total / ntst as f64;
    let mut new = vec![0.0; ntst + 1];
    new[0] = mesh[0];
    let mut j = 1;
    for (j1, t) in new.iter_mut().enumerate().take(ntst).skip(1) {
        // The interval of the old mesh in which the equidistribution function reaches
        // j1 * dal. It exists because eqf ascends to total > j1 * dal.
        let uneq = j1 as f64 * dal;
        while j <= ntst && eqf[j] <= uneq {
            j += 1;
        }
        j -= 1;
        let (x1, x2) = (uneq - eqf[j], eqf[j + 1] - uneq);
        *t = if x1 <= x2 {
            mesh[j] + x1 / deqf[j]
        } else {
            // Rounds better when uneq is closer to eqf[j + 1].
            mesh[j + 1] - x2 / deqf[j]
        };
    }
    new[ntst] = mesh[ntst];
    let increasing = new.windows(2).all(|w| w[1] > w[0]);
    let finite = new.iter().all(|t| t.is_finite());
    (increasing && finite).then_some(new)
}

/// Interpolates `vals`, n values at each base point of `old` with the m of `basis`, onto
/// the base points of `new` with `new_m` points per interval (INTERP). Both meshes run
/// from 0 to 1, and as in INTERP their numbers of intervals may differ.
pub(crate) fn interpolate(
    basis: &CollocationBasis,
    n: usize,
    old: &[f64],
    new: &[f64],
    new_m: usize,
    vals: &[f64],
) -> Vec<f64> {
    let m = basis.m();
    let (old_n, new_n) = (old.len() - 1, new.len() - 1);
    let mut out = vec![0.0; n * (new_n * new_m + 1)];
    let mut w = vec![0.0; m + 1];
    let mut j = 1;
    for j1 in 0..new_n {
        let dt = new[j1 + 1] - new[j1];
        for i in 0..new_m {
            let z = new[j1] + i as f64 * dt / new_m as f64;
            while j <= old_n && old[j] <= z {
                j += 1;
            }
            j = (j - 1).min(old_n - 1);
            basis.values((z - old[j]) / (old[j + 1] - old[j]), &mut w);
            let at = (j1 * new_m + i) * n;
            for c in 0..n {
                out[at + c] = (0..=m).map(|l| w[l] * vals[(j * m + l) * n + c]).sum();
            }
        }
    }
    let (last, old_last) = (new_n * new_m * n, old_n * m * n);
    out[last..last + n].copy_from_slice(&vals[old_last..old_last + n]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The orbit at the base points of `mesh` for a function of tau.
    fn sample(basis: &CollocationBasis, mesh: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
        let m = basis.m();
        let mut u = Vec::new();
        for j in 0..mesh.len() - 1 {
            for i in 0..m {
                let h = mesh[j + 1] - mesh[j];
                u.push(f(mesh[j] + h * i as f64 / m as f64));
            }
        }
        u.push(f(mesh[mesh.len() - 1]));
        u
    }

    fn uniform(ntst: usize) -> Vec<f64> {
        (0..=ntst).map(|j| j as f64 / ntst as f64).collect()
    }

    #[test]
    fn interpolation_reproduces_polynomials_of_degree_m() {
        let basis = CollocationBasis::new(3).unwrap();
        let old = uniform(5);
        let new = [0.0, 0.1, 0.35, 0.4, 0.8, 1.0];
        let f = |t: f64| 2.0 - t + 3.0 * t * t - 4.0 * t.powi(3);
        let u = interpolate(&basis, 1, &old, &new, 3, &sample(&basis, &old, f));
        for (a, b) in u.iter().zip(sample(&basis, &new, f)) {
            assert!((a - b).abs() <= 1e-13, "{a} {b}");
        }
    }

    /// The PD switch interpolates onto a mesh with other N and m.
    #[test]
    fn interpolation_changes_the_number_of_intervals_and_points() {
        let (b3, b4) = (
            CollocationBasis::new(3).unwrap(),
            CollocationBasis::new(4).unwrap(),
        );
        let old = uniform(5);
        let new = [0.0, 0.05, 0.2, 0.35, 0.5, 0.6, 0.9, 1.0];
        let f = |t: f64| 1.0 + 2.0 * t - 5.0 * t * t + 3.0 * t.powi(3);
        let u = interpolate(&b3, 1, &old, &new, 4, &sample(&b3, &old, f));
        let want = sample(&b4, &new, f);
        assert_eq!(u.len(), want.len());
        for (a, b) in u.iter().zip(want) {
            assert!((a - b).abs() <= 1e-13, "{a} {b}");
        }
    }

    #[test]
    fn small_derivatives_keep_the_mesh() {
        let basis = CollocationBasis::new(4).unwrap();
        let mesh = uniform(8);
        let u = sample(&basis, &mesh, |t| 1.0 + 0.1 * t);
        assert_eq!(new_mesh(&basis, 1, &mesh, &u), None);
    }

    /// A steep front at tau = 0.5 pulls mesh points towards it.
    #[test]
    fn mesh_points_move_to_a_steep_front() {
        let basis = CollocationBasis::new(4).unwrap();
        let mesh = uniform(20);
        let f = |t: f64| ((t - 0.5) / 0.02).tanh();
        let u = sample(&basis, &mesh, f);
        let new = new_mesh(&basis, 1, &mesh, &u).unwrap();
        assert_eq!((new[0], new[20]), (0.0, 1.0));
        assert!(new.windows(2).all(|w| w[1] > w[0]));
        let near = |m: &[f64]| m.iter().filter(|t| (*t - 0.5).abs() < 0.1).count();
        assert!(near(&new) > near(&mesh), "{new:?}");
    }
}
