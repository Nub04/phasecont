//! Limit cycles: orthogonal collocation on a moving mesh of N intervals with m Gauss
//! points each (SPEC §7).

mod basis;
mod problem;

pub use basis::{CollocationBasis, NCOL_RANGE};
pub use problem::{CycleProblem, CycleSetup, PeriodicOrbit, periodic_orbit};
