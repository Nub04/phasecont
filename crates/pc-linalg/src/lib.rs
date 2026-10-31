//! faer wrappers, bordered solvers, condensation and Floquet multipliers.
//!
//! Dense matrices are column-major `&[f64]`, entry (i, j) at `i + rows * j`. No faer type
//! appears in the public API. Functions return an error on bad input and never panic.

mod block;
mod condense;
mod dense;
mod eig;
mod null;

pub use block::solve_shifted;
pub use condense::{BlockSystem, CondenseShape, Condenser};
pub use dense::{LinalgError, Lu};
pub use eig::{eigenvalues, eigvec};
pub use null::{null_left, null_right, singular_values};
