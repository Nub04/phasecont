//! Expression DAG, common subexpression elimination and symbolic differentiation.
//!
//! `math` holds the scalar functions every evaluator shares (interpreter, JIT, AOT C)
//! and the `#[repr(C)]` shim table that compiled code calls through.

pub mod math;
