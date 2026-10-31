//! Limit cycles: orthogonal collocation on a moving mesh of N intervals with m Gauss
//! points each (SPEC §7).

mod basis;
mod continuation;
mod mesh;
mod problem;

pub use basis::{CollocationBasis, NCOL_RANGE};
pub use continuation::{CycleStart, HopfStart, PdStart, continue_cycle};
pub use problem::{CYCLE_LABELS, CycleProblem, CycleSetup, PeriodicOrbit, periodic_orbit};
