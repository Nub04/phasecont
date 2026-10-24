//! Parsers that turn tokens into syntax trees.
//!
//! [`expr`] parses one expression. [`stmt`] splits a file into statements
//! and uses [`expr`] for the expressions inside them.

pub mod expr;
pub mod stmt;

#[doc(hidden)]
pub use expr::parse_expr_text;
pub(crate) use expr::{ExprCursor, parse_expr, parse_index, parse_name};
pub use stmt::parse;
