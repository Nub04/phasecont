//! Emitters for C, CUDA, OpenCL C, Metal Shading Language, LaTeX and MATLAB reference code.

mod c;

pub use c::{CNameError, c_ident, emit_c};

#[cfg(test)]
mod c_tests;
