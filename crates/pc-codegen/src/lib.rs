//! Emitters for C, CUDA, OpenCL C, Metal Shading Language, LaTeX and MATLAB reference code.

mod c;
pub mod kernel;

pub use c::{CNameError, EmittedC, c_ident, emit_c, emit_c_tagged, fnv1a64};

#[cfg(test)]
mod c_tests;
