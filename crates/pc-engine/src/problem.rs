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

    /// Writes the (N-1) by N Jacobian of F at `x` to `out`, column-major.
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError>;

    /// The (dim-1) by (dim-1) state Jacobian A = F_u at `x`, column-major, for problems on
    /// equilibrium branches. The engine takes the eigenvalues of A for the Hopf test
    /// function. The default is `None`: no state block, no eigenvalues, no Hopf test.
    fn state_jacobian(&mut self, _x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        Ok(None)
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
