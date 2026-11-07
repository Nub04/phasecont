//! The adjoint variational problem of a HOM orbit (SPEC §8b-2) on the closed-form orbit of
//! Sandstede's model at a = 0, b = 1, c = -2, mu = mut = alpha = 0.
//!
//! There the plane z = 0 holds x = sech^2(t/2), y = -x tanh(t/2), with x' = y and
//! y' = x - 1.5 x^2. The vector field is divergence free in the plane, so the adjoint is
//! phi = (y', -x') = (x - 1.5 x^2, -y), phi_z = 0, with gamma = 0.

mod common;

use common::load;
use pc_engine::{AdjointSetup, Evaluator, ModelEval, adjoint_residual, build, inclination_flips};
use pc_linalg::{BasisOf, BasisTracker};

/// Parameter values of `sandstede.ode` at the test point, in file order.
const PARAMS: [f64; 8] = [0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// u(tau) and phi(tau) of the closed forms at the base points of a uniform mesh.
fn closed_form(period: f64, ntst: usize, ncol: usize) -> (Vec<f64>, Vec<f64>) {
    let pts = ntst * ncol + 1;
    let (mut u, mut phi) = (Vec::new(), Vec::new());
    for k in 0..pts {
        let t = period * (k as f64 / (pts - 1) as f64 - 0.5);
        let w = (t / 2.0).tanh();
        let x = 1.0 - w * w;
        let y = -w * x;
        u.extend([x, y, 0.0]);
        phi.extend([x - 1.5 * x * x, -y, 0.0]);
    }
    (u, phi)
}

fn mesh(ntst: usize) -> Vec<f64> {
    (0..=ntst).map(|j| j as f64 / ntst as f64).collect()
}

fn model(which: Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load("sandstede");
    build(&m, &info, which).unwrap()
}

fn run(
    which: Evaluator,
    period: f64,
    ntst: usize,
    ncol: usize,
    tweak: impl Fn(&mut Vec<f64>, &mut f64),
) -> (f64, f64) {
    let mut m = model(which);
    let mesh = mesh(ntst);
    let (u, mut phi) = closed_form(period, ntst, ncol);
    let mut gamma = 0.0;
    tweak(&mut phi, &mut gamma);
    let setup = AdjointSetup {
        ncol,
        mesh: &mesh,
        period,
        x0: &[0.0; 3],
        params: &PARAMS,
        ns: 2,
    };
    let r = adjoint_residual(m.as_mut(), &setup, &BasisTracker::new(), &u, &phi, gamma).unwrap();
    let max = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    (max(&r.collocation), max(&r.boundary))
}

/// The closed-form adjoint satisfies the collocation rows to the order of the scheme and
/// the boundary rows to e^{-T}, on both evaluators.
#[test]
fn closed_form_adjoint_has_a_small_residual() {
    for which in [Evaluator::Jit, Evaluator::Fallback] {
        let (c, b) = run(which.clone(), 20.0, 80, 4, |_, _| {});
        println!("{which:?}: collocation {c:.3e}, boundary {b:.3e}");
        assert!(c < 1e-4, "{which:?}: collocation residual {c:e}");
        assert!(b < 1e-6, "{which:?}: boundary residual {b:e}");
    }
}

/// Doubling the number of intervals shrinks the collocation residual, so the residual
/// measures the discretization and not a wrong equation.
#[test]
fn collocation_residual_falls_with_the_mesh() {
    let (coarse, _) = run(Evaluator::Jit, 20.0, 40, 4, |_, _| {});
    let (fine, _) = run(Evaluator::Jit, 20.0, 80, 4, |_, _| {});
    println!("coarse {coarse:.3e}, fine {fine:.3e}");
    assert!(fine < coarse / 8.0, "coarse {coarse:e}, fine {fine:e}");
}

/// Wrong adjoints are not hidden by the tolerance: a nonzero gamma, a scaled z component
/// that does not decay at the rate c, and a phi that breaks the boundary directions each
/// give a residual far above the closed form's.
#[test]
fn wrong_adjoints_are_caught() {
    let (c0, b0) = run(Evaluator::Jit, 20.0, 80, 4, |_, _| {});
    let (c, _) = run(Evaluator::Jit, 20.0, 80, 4, |_, g| *g = 0.5);
    assert!(c > 1e3 * c0, "gamma: {c:e} against {c0:e}");
    let (c, _) = run(Evaluator::Jit, 20.0, 80, 4, |p, _| {
        for v in p.chunks_mut(3) {
            v[2] = 1.0;
        }
    });
    assert!(c > 1e3 * c0, "constant phi_z: {c:e} against {c0:e}");
    let (_, b) = run(Evaluator::Jit, 20.0, 80, 4, |p, _| {
        for v in p.chunks_mut(3) {
            v[0] += 0.1;
        }
    });
    assert!(b > 1e3 * b0, "shifted phi_x: {b:e} against {b0:e}");
}

/// A reference basis taken from the same matrix leaves the boundary rows unchanged. A is
/// symmetric here, so this does not tell A from its transpose; the next test does.
#[test]
fn a_matching_reference_leaves_the_boundary_rows_unchanged() {
    let mut m = model(Evaluator::Jit);
    let mesh = mesh(20);
    let (u, phi) = closed_form(20.0, 20, 4);
    let setup = AdjointSetup {
        ncol: 4,
        mesh: &mesh,
        period: 20.0,
        x0: &[0.0; 3],
        params: &PARAMS,
        ns: 2,
    };
    // A reference in the Matrix slot is rotated against, so a reference built from a
    // different matrix changes the rows but a self-consistent one does not.
    let fresh = BasisTracker::new();
    let r0 = adjoint_residual(m.as_mut(), &setup, &fresh, &u, &phi, 0.0).unwrap();
    let mut a = vec![0.0; 9];
    let mut p = PARAMS.to_vec();
    p.push(0.0);
    m.eval(&pc_ir::EntryKind::JacU, &[0.0; 3], &p, &[], &mut a)
        .unwrap();
    let mut tracked = BasisTracker::new();
    tracked.set(
        BasisOf::Matrix,
        fresh.bases(BasisOf::Matrix, &a, 3, 2).unwrap(),
    );
    let r1 = adjoint_residual(m.as_mut(), &setup, &tracked, &u, &phi, 0.0).unwrap();
    for (x, y) in r0.boundary.iter().zip(&r1.boundary) {
        assert!(
            (x - y).abs() < 1e-12,
            "{:?} against {:?}",
            r0.boundary,
            r1.boundary
        );
    }
}

/// At mu != 0 the matrix A = f_x(0) = [[0,1,0],[1,0,0],[mu,0,-2]] is not symmetric, so its
/// invariant subspaces differ from those of A^T and the boundary rows show which one is used.
/// The eigenvalues are -2, -1, 1, so n_s = 2 and n_u = 1, and by hand
/// E^u(A) = span (1, 1, mu/3), E^s(A) = span {(1, -1, mu), (0, 0, 1)},
/// E^u(A^T) = span (1, 1, 0), E^s(A^T) = span {(1, -1, 0), (-2 mu/3, mu/3, 1)}.
/// phi(0) = (0, mu/3, -1) and phi(1) = (1, 1, 0) are orthogonal to the subspaces of A and
/// not to those of A^T, and phi(0) = (1, -1, 1) is the other way round.
#[test]
fn boundary_rows_follow_a_and_not_its_transpose() {
    let mu = 0.9;
    let mut params = PARAMS;
    params[3] = mu;
    let mut m = model(Evaluator::Jit);
    let (ntst, ncol) = (4, 4);
    let mesh = mesh(ntst);
    let (u, _) = closed_form(20.0, ntst, ncol);
    let setup = AdjointSetup {
        ncol,
        mesh: &mesh,
        period: 20.0,
        x0: &[0.0; 3],
        params: &params,
        ns: 2,
    };
    // Confirm the hand matrix before relying on the hand subspaces.
    let mut a = vec![0.0; 9];
    let mut p = params.to_vec();
    p.push(0.0);
    m.eval(&pc_ir::EntryKind::JacU, &[0.0; 3], &p, &[], &mut a)
        .unwrap();
    let hand = [0.0, 1.0, mu, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0];
    for (x, y) in a.iter().zip(hand) {
        assert!((x - y).abs() < 1e-14, "{a:?} against {hand:?}");
    }
    let mut boundary = |first: [f64; 3], last: [f64; 3]| {
        let mut phi = vec![0.0; u.len()];
        phi[..3].copy_from_slice(&first);
        let n = phi.len();
        phi[n - 3..].copy_from_slice(&last);
        adjoint_residual(m.as_mut(), &setup, &BasisTracker::new(), &u, &phi, 0.0)
            .unwrap()
            .boundary
    };
    let max = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let right = boundary([0.0, mu / 3.0, -1.0], [1.0, 1.0, 0.0]);
    assert!(max(&right) < 1e-12, "bases of A: {right:?}");
    // phi(0) = (1, -1, 1) is orthogonal to E^u(A^T) but not to E^u(A): the row is
    // (mu/3) / |(1, 1, mu/3)|, so a basis of A^T would have given zero here.
    let wrong = boundary([1.0, -1.0, 1.0], [1.0, 1.0, 0.0]);
    let expect = (mu / 3.0) / (2.0 + mu * mu / 9.0).sqrt();
    assert!(
        (wrong[0].abs() - expect).abs() < 1e-12,
        "{:?} against {expect}",
        wrong
    );
    // phi(1) = (1, -1, 0) is orthogonal to the first stable vector of A^T, not of A.
    let stable = boundary([0.0, mu / 3.0, -1.0], [1.0, -1.0, 0.0]);
    assert!(max(&stable[1..]) > 0.5, "stable rows: {stable:?}");
}

/// IFS and IFU of the closed-form orbit at the saddle at the origin, where lambda_S = -1 and
/// lambda_{S+1} = +1 with unit eigenvectors (1, -1, 0)/sqrt 2 and (1, 1, 0)/sqrt 2.
/// The orbit ends are x(+-T/2) = 4 e^{-T/2} to leading order, phi(0) is x (1, -1, 0) and
/// phi(1) is x (1, 1, 0), so |IFS| and |IFU| both tend to 4 sqrt 2 as T grows. This is a
/// generic point, not a flip, so the limit is not zero.
#[test]
fn closed_form_flips_tend_to_four_root_two() {
    let mut m = model(Evaluator::Jit);
    let mut a = vec![0.0; 9];
    let mut p = PARAMS.to_vec();
    p.push(0.0);
    m.eval(&pc_ir::EntryKind::JacU, &[0.0; 3], &p, &[], &mut a)
        .unwrap();
    let limit = 4.0 * 2.0_f64.sqrt();
    let mut previous = f64::INFINITY;
    for period in [10.0, 20.0, 30.0] {
        let (_, phi) = closed_form(period, 4, 4);
        let last = phi.len() - 3;
        let f =
            inclination_flips(&a, 3, 2, period, &phi[..3], &phi[last..], &[None, None]).unwrap();
        println!("T = {period}: IFS {:.6e}, IFU {:.6e}", f.ifs, f.ifu);
        let worst = (f.ifs.abs() - limit).abs().max((f.ifu.abs() - limit).abs());
        assert!(
            worst < previous,
            "T = {period}: {worst:e} after {previous:e}"
        );
        previous = worst;
    }
    assert!(previous < 1e-4 * limit, "{previous:e}");
}
