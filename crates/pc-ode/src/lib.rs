//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds source spans and
//! the source map ([`span`]), diagnostics with their stable codes
//! ([`diag`]), the tokenizer ([`lexer`]), the syntax tree ([`ast`]), the
//! known `@` option names ([`options`]) and the parser ([`parser`]).
//! [`parse`] turns one file into a [`Model`].

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod options;
pub mod parser;
pub mod span;

pub use ast::{
    Assign, BinaryOp, Comment, DUMMY_SPAN, Entry, Expr, Ident, Index, IndexOp, MarkovRow, Model,
    OdeForm, RawAssign, Stmt, StmtKind, UnaryOp, Unsupported, Value,
};
pub use diag::{CODES, Code, Diagnostic, FixIt, Severity};
pub use lexer::{Token, TokenKind, lex};
pub use parser::parse;
#[doc(hidden)]
pub use parser::parse_expr_text;
pub use span::{FileId, SourceMap, Span};
