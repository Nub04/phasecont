//! faer wrappers, bordered solvers, condensation and Floquet multipliers.
//!
//! Dense matrices are column-major `&[f64]`, entry (i, j) at `i + rows * j`. No faer type
//! appears in the public API. Functions return an error on bad input and never panic.

mod block;
mod condense;
mod dense;
mod eig;
mod invariant;
mod null;

pub use block::solve_shifted;
pub use condense::{BlockSystem, CondenseShape, Condenser, PARALLEL_WORK};
pub use dense::{LinalgError, Lu};
pub use eig::{eigenvalues, eigvec};
pub use invariant::{SIGN_MAX_STEPS, SIGN_TOL, invariant_bases, rotate_basis, sign_function};
pub use null::{null_left, null_right, singular_values};
