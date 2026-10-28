//! Continuation core and problem types (EP, LP, H, LC, LPC, PD, NS).
//!
//! The model layer comes first: [`ModelEval`] evaluates the entries of a model in f64,
//! [`build`] makes one from a loaded `.ode` model, and [`EngineError`] is the error type
//! of the whole crate. On top of it, [`continue_ep`] follows an equilibrium branch with
//! pseudo-arclength continuation.

mod branch;
mod ep;
mod error;
mod locate;
mod model;
mod normal;
mod palc;
mod problem;
mod refine;
mod tests_fn;

pub use branch::{Branch, Extras, Options, Point, SpecialPoint, StopReason};
pub use ep::{EpProblem, EpSetup, continue_ep, equilibrium};
pub use error::EngineError;
pub use model::{Evaluator, ModelEval, build};
pub use normal::{
    H_NORMALIZATION, LP_NORMALIZATION, first_lyapunov, fold_coefficient, hopf_eigenvectors,
    special_coefficient,
};
pub use palc::continue_branch;
pub use problem::Problem;
pub use refine::{
    BpSystem, DefiningSystem, HopfSystem, LpSystem, MAX_REFINE_ITERS, NewtonResult, NewtonStep,
    Refined, newton, refine_special, refine_specials,
};
pub use tests_fn::{HopfZero, classify_h_zero};
