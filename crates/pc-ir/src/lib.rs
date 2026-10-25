//! Expression DAG, common subexpression elimination and symbolic differentiation.
//!
//! `math` holds the scalar functions every evaluator shares (interpreter, JIT, AOT C)
//! and the `#[repr(C)]` shim table that compiled code calls through. `lower` turns a
//! checked `.ode` model into a DAG.

pub mod dag;
pub mod error;
pub mod eval;
pub mod lower;
pub mod math;

pub use error::{IrError, IrErrorKind};
pub use lower::{IrModel, lower};
