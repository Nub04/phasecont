//! Limit cycles: orthogonal collocation on a moving mesh of N intervals with m Gauss
//! points each (SPEC §7).

mod basis;
mod bordered;
mod codim2;
mod continuation;
mod curve;
mod lpc;
mod mesh;
mod normal;
mod ns;
mod pd;
mod problem;

pub use basis::{CollocationBasis, NCOL_RANGE};
pub use codim2::{LPC_CURVE_LABELS, NS_CURVE_LABELS, PD_CURVE_LABELS};
pub use continuation::{CycleStart, HopfStart, PdStart, continue_cycle};
pub use curve::CycleCurveStart;
pub use lpc::{LpcProblem, continue_lpc_curve};
pub use normal::{CycleBifurcation, CycleNormalForm, lc_point_normal_form};
pub use ns::{NsProblem, continue_ns_curve};
pub use pd::{PdProblem, continue_pd_curve};
pub use problem::{CYCLE_LABELS, CycleProblem, CycleSetup, PeriodicOrbit, periodic_orbit};
