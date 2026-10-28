//! Continuation core and problem types (EP, LP, H, LC, LPC, PD, NS).
//!
//! The model layer comes first: [`ModelEval`] evaluates the entries of a model in f64,
//! [`build`] makes one from a loaded `.ode` model, and [`EngineError`] is the error type
//! of the whole crate.

mod error;
mod model;

pub use error::EngineError;
pub use model::{Evaluator, ModelEval, build};
