//! Continuation core and problem types (EP, LP, H, LC, LPC, PD, NS).
//!
//! The model layer comes first: [`ModelEval`] evaluates the entries of a model in f64,
//! [`build`] makes one from a loaded `.ode` model, and [`EngineError`] is the error type
//! of the whole crate. On top of it, [`continue_ep`] follows an equilibrium branch with
//! pseudo-arclength continuation.

mod branch;
mod ep;
mod error;
mod model;
mod palc;
mod problem;

pub use branch::{Branch, Extras, Options, Point, SpecialPoint, StopReason};
pub use ep::{EpProblem, EpSetup, continue_ep, equilibrium};
pub use error::EngineError;
pub use model::{Evaluator, ModelEval, build};
pub use palc::continue_branch;
pub use problem::Problem;
