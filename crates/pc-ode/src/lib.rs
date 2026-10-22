//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds the shared
//! foundation types: source spans and the source map ([`span`]), and
//! diagnostics with their stable codes ([`diag`]).

pub mod diag;
pub mod span;

pub use diag::{CODES, Code, Diagnostic, FixIt, Severity};
pub use span::{FileId, SourceMap, Span};
