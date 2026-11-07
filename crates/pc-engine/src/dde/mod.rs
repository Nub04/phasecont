//! Delay equations, step 1 of phase E9a (SPEC section 8c): the characteristic roots of a
//! linear equation with discrete delays.

mod branch;
mod eq;
mod hopf;
mod lyap;
mod lyap2;
mod nodes;
mod roots;
mod zerohopf;

pub use branch::{DdeBranch, DdePoint, DdeSpecial, continue_equilibrium};
pub use eq::{equilibrium, linearize, residual, stability};
pub use hopf::{
    DdeHopfBranch, DdeHopfPoint, DdeHopfSpecial, HopfSetup, continue_hopf,
    continue_hopf_with_lyapunov,
};
pub use lyap::{Lyapunov, lyapunov};
pub use lyap2::{Lyapunov2, lyapunov2};
pub use zerohopf::{ZeroHopf, zero_hopf};

pub use roots::{CharRoot, LinearDde, RootOptions, RootSet, characteristic_roots};
