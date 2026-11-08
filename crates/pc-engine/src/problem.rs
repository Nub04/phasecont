//! The interface between a continuation problem and the PALC loop.

use crate::error::EngineError;
use pc_linalg::Lu;

/// A continuation problem: `dim()` unknowns x and `dim() - 1` equations F(x) = 0, whose
/// solution set is a curve.
pub trait Problem {
    /// Number of unknowns N.
    fn dim(&self) -> usize;

    /// Writes F(x), N - 1 values, to `out`.
    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError>;

    /// Writes the (N-1) by N Jacobian of F at `x` to `out`, column-major. `out` has
    /// `jacobian_len()` values.
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError>;

    /// Length of the buffer that the PALC loop passes to `jacobian` and `bordered_solve`.
    /// The default is (N-1) N, the dense Jacobian. A problem that keeps its Jacobian in its
    /// own storage returns 0: `jacobian` then gets an empty buffer, `bordered_solve` an
    /// empty `jac`, and a singular bordered matrix stops the run instead of giving a
    /// tangent from the null space.
    fn jacobian_len(&self) -> usize {
        let n = self.dim();
        n.saturating_sub(1) * n
    }

    /// A square state matrix at `x`, column-major, whose eigenvalues the engine reports
    /// with the point: the (dim-1) by (dim-1) state Jacobian A = F_u on equilibrium
    /// branches, where the eigenvalues feed the Hopf test function, and the n by n
    /// monodromy matrix on cycle branches, where they are the Floquet multipliers. The
    /// default is `None`: no matrix, no eigenvalues, no Hopf test.
    fn state_jacobian(&mut self, _x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        Ok(None)
    }

    /// Whether the equilibrium-branch tests (`psi_LP`, `psi_BP`, `psi_H`) apply to this
    /// problem. Curve problems return false, which leaves those three NaN and moves the
    /// detection to `codim2_tests`. The default is true.
    fn ep_tests(&self) -> bool {
        true
    }

    /// The label of a sign change of the branch-point test `psi_BP`, or `None` when this
    /// problem has no such test. The default is `"BP"` where `ep_tests` is true. A problem
    /// that returns a label here gets `psi_BP` from the determinant of its bordered solve, so
    /// it must return a value with the sign of the determinant from `bordered_solve`.
    fn bp_label(&self) -> Option<&'static str> {
        self.ep_tests().then_some("BP")
    }

    /// Names of the codim-2 test functions, in the order `codim2_tests` returns them. The
    /// default is none.
    fn codim2_labels(&self) -> &'static [&'static str] {
        &[]
    }

    /// Values of the codim-2 test functions at the accepted point `x` with unit tangent
    /// `tangent`. `eigs` are the eigenvalues of the state Jacobian there (empty when they
    /// were not computed). A value that is not defined at the point is NaN. The result has
    /// `codim2_labels().len()` entries, or none. The default is none.
    fn codim2_tests(
        &mut self,
        _x: &[f64],
        _tangent: &[f64],
        _eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        Ok(Vec::new())
    }

    /// Label of a zero of codim-2 test `i` at a point with these eigenvalues; `None` drops
    /// the zero. The default is `codim2_labels()[i]`.
    fn codim2_label(&self, i: usize, _eigs: &[(f64, f64)]) -> Option<&'static str> {
        self.codim2_labels().get(i).copied()
    }

    /// Whether a zero of codim-2 test `i` between the values `a` and `b` at two consecutive
    /// accepted points may get a label. Called for a bracket with a sign change or with
    /// exactly one end at zero, and never for the LP, BP and H slots. Returning false
    /// drops the bracket without a warning, so a problem can keep rounding noise around
    /// zero from becoming a label. The default accepts every bracket.
    fn crossing_ok(&self, _i: usize, _a: f64, _b: f64) -> bool {
        true
    }

    /// Called with each accepted point that the next step starts from: the first point,
    /// and each later point once the specials of the step that reached it are located.
    /// The default does nothing.
    fn accept(&mut self, _x: &[f64]) {}

    /// The mesh that the unknowns live on, stored with each point; empty for a problem
    /// without one.
    fn mesh(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Called with the point x that the next step starts from and its tangent v: after
    /// each accepted point (`failed` false) and after each failed corrector (`failed`
    /// true). A problem with a mesh may move it and rewrite x and v on the new mesh.
    /// Returns whether it did. The default does nothing.
    fn adapt(&mut self, _x: &mut [f64], _v: &mut [f64], _failed: bool) -> bool {
        false
    }

    /// Solves `[jac; borderᵀ] sol = rhs` for the N by N matrix made of `jac` (N-1 rows,
    /// column-major) and one extra row `border`. Returns the determinant of that matrix.
    /// A singular matrix gives `EngineError::Singular`.
    fn bordered_solve(
        &mut self,
        jac: &[f64],
        border: &[f64],
        rhs: &[f64],
        sol: &mut [f64],
    ) -> Result<f64, EngineError> {
        let n = self.dim();
        if sol.len() != n || rhs.len() != n {
            return Err(EngineError::BadProblem(
                "bordered_solve: wrong lengths".into(),
            ));
        }
        let lu = dense_bordered_lu(n, jac, border)?;
        sol.copy_from_slice(rhs);
        lu.solve(sol)?;
        Ok(lu.det())
    }
}

/// The LU factorization of the N by N matrix made of `jac` (N-1 rows, column-major) and
/// one extra row `border`. A singular matrix gives `EngineError::Singular`.
pub(crate) fn dense_bordered_lu(n: usize, jac: &[f64], border: &[f64]) -> Result<Lu, EngineError> {
    if n == 0 || jac.len() != (n - 1) * n || border.len() != n {
        return Err(EngineError::BadProblem(
            "bordered_solve: wrong lengths".into(),
        ));
    }
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        a[n * j..n * j + n - 1].copy_from_slice(&jac[(n - 1) * j..(n - 1) * (j + 1)]);
        a[n * j + n - 1] = border[j];
    }
    Ok(Lu::new(&a, n)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A problem with the default `bp_label` and the given `ep_tests`.
    struct Stub {
        ep: bool,
    }

    impl Problem for Stub {
        fn dim(&self) -> usize {
            2
        }
        fn residual(&mut self, _: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out[0] = 0.0;
            Ok(())
        }
        fn jacobian(&mut self, _: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
            out.fill(0.0);
            Ok(())
        }
        fn ep_tests(&self) -> bool {
            self.ep
        }
    }

    #[test]
    fn the_default_bp_label_follows_ep_tests() {
        assert_eq!(Stub { ep: true }.bp_label(), Some("BP"));
        // A curve problem has no `psi_BP`, so it has no label for one.
        assert_eq!(Stub { ep: false }.bp_label(), None);
    }

    #[test]
    fn the_bordered_lu_of_a_wrong_shape_is_an_error() {
        // n = 2: one row of jac with two entries, and a border of two.
        assert!(dense_bordered_lu(2, &[1.0, 0.0], &[0.0, 1.0]).is_ok());
        for (n, jac, border) in [
            (0usize, vec![], vec![]),
            (2, vec![1.0], vec![0.0, 1.0]),
            (2, vec![1.0, 0.0], vec![1.0]),
        ] {
            match dense_bordered_lu(n, &jac, &border) {
                Err(EngineError::BadProblem(m)) => assert!(m.starts_with("bordered_solve")),
                other => panic!("n = {n}: {:?}", other.map(|_| ())),
            }
        }
    }
}
