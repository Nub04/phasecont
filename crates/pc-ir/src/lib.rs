//! Expression DAG, common subexpression elimination and symbolic differentiation.
//!
//! `math` holds the scalar functions every evaluator shares (interpreter, JIT, AOT C)
//! and the `#[repr(C)]` shim table that compiled code calls through. `lower` turns a
//! checked `.ode` model into a DAG. `interp` walks the model directly with dual numbers
//! and serves as the independent f64 fallback.

pub mod complex;
pub mod dag;
pub mod diff;
pub mod entry;
pub mod error;
pub mod eval;
pub mod interp;
pub mod lower;
pub mod math;
pub mod multi;

pub use diff::{Wrt, d};
pub use entry::{Entry, EntryKind, Slot};
pub use error::{IrError, IrErrorKind};
pub use interp::{Dual, Fallback, Interp, Scalar};
pub use lower::{IrModel, lower};
