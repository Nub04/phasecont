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
        if n == 0 || jac.len() != (n - 1) * n || border.len() != n || rhs.len() != n {
            return Err(EngineError::BadProblem(
                "bordered_solve: wrong lengths".into(),
            ));
        }
        if sol.len() != n {
            return Err(EngineError::BadProblem(
                "bordered_solve: wrong lengths".into(),
            ));
        }
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            a[n * j..n * j + n - 1].copy_from_slice(&jac[(n - 1) * j..(n - 1) * (j + 1)]);
            a[n * j + n - 1] = border[j];
        }
        let lu = Lu::new(&a, n)?;
        sol.copy_from_slice(rhs);
        lu.solve(sol)?;
        Ok(lu.det())
    }
}
