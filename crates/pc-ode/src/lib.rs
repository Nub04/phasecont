//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds the shared
//! foundation types: source spans and the source map ([`span`]),
//! diagnostics with their stable codes ([`diag`]), the tokenizer
//! ([`lexer`]), the expression tree ([`ast`]) and the expression parser
//! ([`parser`]).

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod span;

pub use ast::{BinaryOp, DUMMY_SPAN, Expr, Ident, Index, IndexOp, UnaryOp};
pub use diag::{CODES, Code, Diagnostic, FixIt, Severity};
pub use lexer::{Token, TokenKind, lex};
#[doc(hidden)]
pub use parser::parse_expr_text;
pub use span::{FileId, SourceMap, Span};
