//! The bordered systems N inside the defining functions of the curves of cycles (SPEC §8):
//! storage in the layout of [`BlockSystem`], one factorization per x and the solves with
//! unit right-hand sides.

use super::problem::{boundary_len, try_zeros};
use crate::error::EngineError;
use pc_linalg::{BlockSystem, CondenseShape, Condenser};

pub(super) struct Bordered {
    pub(super) blocks: Vec<f64>,
    pub(super) bc: Vec<f64>,
    pub(super) border: Vec<f64>,
    condenser: Condenser,
    rhs: Vec<f64>,
    /// The x at which N is factored and the forward solve done, when `factored`.
    at: Vec<f64>,
    factored: bool,
}

impl Bordered {
    pub(super) fn new(shape: CondenseShape) -> Result<Bordered, EngineError> {
        let len = shape.dim();
        Ok(Bordered {
            blocks: try_zeros(
                shape
                    .nint()
                    .checked_mul(shape.block_rows())
                    .and_then(|k| k.checked_mul(shape.block_cols())),
            )?,
            bc: try_zeros(boundary_len(shape.n(), shape.k()))?,
            border: try_zeros(len.checked_mul(shape.k()))?,
            condenser: Condenser::new(shape)?,
            rhs: vec![0.0; len],
            at: Vec::new(),
            factored: false,
        })
    }

    /// Number of unknowns of N.
    pub(super) fn dim(&self) -> usize {
        self.rhs.len()
    }

    /// Whether N is factored at exactly this x.
    pub(super) fn is_at(&self, x: &[f64]) -> bool {
        self.factored
            && self.at.len() == x.len()
            && self
                .at
                .iter()
                .zip(x)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Drops the factorization, after N or the matrix it was built from changed.
    pub(super) fn forget(&mut self) {
        self.factored = false;
    }

    /// Factors N as built and solves N s = e_i into `out`. N then counts as factored at
    /// x, but only when both steps succeed.
    pub(super) fn factor_solve(
        &mut self,
        x: &[f64],
        i: usize,
        out: &mut [f64],
        parallel: bool,
    ) -> Result<(), EngineError> {
        self.factored = false;
        let sys = BlockSystem {
            blocks: &self.blocks,
            bc: &self.bc,
            border: &self.border,
        };
        self.condenser.factor(&sys, parallel)?;
        self.rhs.fill(0.0);
        self.rhs[i] = 1.0;
        self.condenser.solve_factored(&self.rhs, out, parallel)?;
        self.at.clear();
        self.at.extend_from_slice(x);
        self.factored = true;
        Ok(())
    }

    /// Solves N s = e_i into `out` with the factored N.
    pub(super) fn solve(
        &mut self,
        i: usize,
        out: &mut [f64],
        parallel: bool,
    ) -> Result<(), EngineError> {
        self.rhs.fill(0.0);
        self.rhs[i] = 1.0;
        self.condenser.solve_factored(&self.rhs, out, parallel)?;
        Ok(())
    }

    /// Solves Nᵀ s = e_i into `out` with the factored N.
    pub(super) fn solve_transpose(
        &mut self,
        i: usize,
        out: &mut [f64],
        parallel: bool,
    ) -> Result<(), EngineError> {
        self.rhs.fill(0.0);
        self.rhs[i] = 1.0;
        self.condenser
            .solve_transpose_factored(&self.rhs, out, parallel)?;
        Ok(())
    }
}

/// `v` scaled to 2-norm 1, or `None` when its norm is zero or not finite.
pub(super) fn unit(v: &[f64]) -> Option<Vec<f64>> {
    let nrm = v.iter().map(|t| t * t).sum::<f64>().sqrt();
    (nrm.is_finite() && nrm > 0.0).then(|| v.iter().map(|t| t / nrm).collect())
}

/// An orthonormal basis of the span of `a` and `b` by Gram-Schmidt, or `None` when the two
/// are not independent to 1e-10 relative or not finite.
pub(super) fn orthonormal(a: &[f64], b: &[f64]) -> Option<[Vec<f64>; 2]> {
    let a = unit(a)?;
    let nb = b.iter().map(|t| t * t).sum::<f64>().sqrt();
    let d: f64 = a.iter().zip(b).map(|(s, t)| s * t).sum();
    let r: Vec<f64> = b.iter().zip(&a).map(|(t, s)| t - d * s).collect();
    let nr = r.iter().map(|t| t * t).sum::<f64>().sqrt();
    // A NaN passes here and fails in `unit`.
    if nr <= 1e-10 * nb {
        return None;
    }
    Some([a, unit(&r)?])
}
