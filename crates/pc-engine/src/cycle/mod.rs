//! Limit cycles: orthogonal collocation on a moving mesh of N intervals with m Gauss
//! points each (SPEC §7).

mod basis;

pub use basis::{CollocationBasis, NCOL_RANGE};
