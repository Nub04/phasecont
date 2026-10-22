//! Parsers that turn tokens into syntax trees.
//!
//! [`expr`] parses one expression. The statement parser builds on it.

pub mod expr;

pub use expr::parse_expr_text;
// The statement parser (Task 5) is the first user outside this module.
#[allow(unused_imports)]
pub(crate) use expr::{ExprCursor, parse_expr};
