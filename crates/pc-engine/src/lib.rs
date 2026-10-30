//! Continuation core and problem types (EP, LP, H, LC, LPC, PD, NS).
//!
//! The model layer comes first: [`ModelEval`] evaluates the entries of a model in f64,
//! [`build`] makes one from a loaded `.ode` model, and [`EngineError`] is the error type
//! of the whole crate. On top of it, [`continue_ep`] follows an equilibrium branch with
//! pseudo-arclength continuation. [`Run`] does the same on a worker thread with poll and
//! cancel, and [`install_hook`] makes worker panics quiet and recoverable.

mod aot;
mod branch;
mod codim2;
mod curve2;
mod ep;
mod error;
mod locate;
mod lpcurve;
mod model;
mod normal;
mod palc;
mod panic;
mod problem;
mod refine;
mod run;
mod switch;
mod tests_fn;

pub use aot::{AotEval, engine_kinds, entry_symbol};
pub use branch::{Branch, Extras, Options, Point, SpecialPoint, StopReason};
pub use curve2::CurveSetup;
pub use ep::{EpProblem, EpSetup, continue_ep, equilibrium};
pub use error::EngineError;
pub use lpcurve::{LpCurveProblem, continue_lp_curve};
pub use model::{Evaluator, ModelEval, build, build_allowing_t};
pub use normal::{
    H_NORMALIZATION, LP_NORMALIZATION, first_lyapunov, fold_coefficient, hopf_eigenvectors,
    special_coefficient,
};
pub use palc::continue_branch;
pub use panic::{install_hook, payload_message, recording, take_last_panic};
pub use problem::Problem;
pub use refine::{
    BpSystem, DefiningSystem, HopfSystem, LpSystem, MAX_REFINE_ITERS, NewtonResult, NewtonStep,
    Refined, newton, refine_special, refine_specials,
};
pub use run::{Run, RunSpec, RunState, Status};
pub use switch::{special_index, switch_bp};
pub use tests_fn::{HopfZero, classify_h_zero};
