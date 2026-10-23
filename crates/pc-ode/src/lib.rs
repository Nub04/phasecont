//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds source spans and
//! the source map ([`span`]), diagnostics with their stable codes
//! ([`diag`]), the tokenizer ([`lexer`]), the syntax tree ([`ast`]), the
//! known `@` option names ([`options`]), the parser ([`parser`]), the
//! builtin table ([`builtins`]), the name checks ([`check`]) and the
//! include pass ([`include`]). [`parse`] turns one file into a [`Model`],
//! [`resolve_includes`] replaces its `#include` lines with the files they
//! name, and [`check()`] resolves names and reports model errors.

pub mod ast;
pub mod builtins;
pub mod check;
pub mod diag;
pub mod include;
pub mod lexer;
pub mod options;
pub mod parser;
pub mod span;

pub use ast::{
    Assign, BinaryOp, Comment, DUMMY_SPAN, Entry, Expr, Ident, Index, IndexOp, MarkovLine,
    MarkovRow, Model, OdeForm, RawAssign, Stmt, StmtKind, UnaryOp, Unsupported, Value,
};
pub use check::{ModelInfo, ModelKind, StateKind, Symbol, SymbolKind, check};
pub use diag::{CODES, Code, Diagnostic, FixIt, Severity};
pub use include::{
    FsLoader, MAX_INCLUDE_DEPTH, MAX_INCLUDES, MapLoader, SourceLoader, resolve_includes,
};
pub use lexer::{Token, TokenKind, lex};
pub use parser::parse;
#[doc(hidden)]
pub use parser::parse_expr_text;
pub use span::{FileId, SourceMap, Span};
