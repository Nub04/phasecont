//! Delay equations, step 1 of phase E9a (SPEC section 8c): the characteristic roots of a
//! linear equation with discrete delays.

mod branch;
mod eq;
mod nodes;
mod roots;

pub use branch::{DdeBranch, DdePoint, DdeSpecial, continue_equilibrium};
pub use eq::{equilibrium, linearize, residual, stability};

pub use roots::{CharRoot, LinearDde, RootOptions, RootSet, characteristic_roots};
