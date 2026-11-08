//! Delay equations, step 1 of phase E9a (SPEC section 8c): the characteristic roots of a
//! linear equation with discrete delays.

mod branch;
mod cycle;
mod eq;
mod floquet;
mod hopf;
mod hopfhopf;
mod lyap;
mod lyap2;
mod nodes;
mod roots;
mod zerohopf;

pub use branch::{DdeBranch, DdePoint, DdeSpecial, continue_equilibrium};
pub use cycle::{
    DdeCycle, DdeCycleBranch, DdeCyclePoint, DdeCycleSetup, DdeCycleSpecial, DdeCycleStart,
    DdeHopfCycle, check_cycle_floquet, continue_cycle, continue_cycle_floquet, correct_cycle,
    cycle_jacobian, cycle_residual, hopf_cycle,
};
pub use eq::{equilibrium, linearize, residual, stability};
pub use floquet::{DdeFloquet, DdeFloquetOptions, floquet_multipliers};
pub use hopf::{
    DdeHopfBranch, DdeHopfPoint, DdeHopfSpecial, HopfExtras, HopfPairs, HopfSetup, continue_hopf,
    continue_hopf_with, continue_hopf_with_lyapunov, hopf_pairs,
};
pub use hopfhopf::{HopfHopf, hopf_hopf};
pub use lyap::{Lyapunov, lyapunov};
pub use lyap2::{Lyapunov2, lyapunov2};
pub use zerohopf::{ZeroHopf, zero_hopf};

pub use roots::{CharRoot, LinearDde, RootOptions, RootSet, characteristic_roots};
