//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds source spans and
//! the source map ([`span`]), diagnostics with their stable codes
//! ([`diag`]), the tokenizer ([`lexer`]), the syntax tree ([`ast`]), the
//! known `@` option names ([`options`]), the parser ([`parser`]), the
//! builtin table ([`builtins`]), the name checks ([`check`]), the include
//! pass ([`include`]) and the array pass ([`expand`]). [`parse`] turns one
//! file into a [`Model`], [`resolve_includes`] replaces its `#include` lines
//! with the files they name, [`expand()`] copies its range lines and array
//! blocks once per index, and [`check()`] resolves names and reports model
//! errors. [`load`] runs all four.

pub mod ast;
pub mod builtins;
pub mod check;
pub mod diag;
pub mod expand;
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
pub use expand::{MAX_ARRAY_COPIES, MAX_INDEX, expand};
pub use include::{
    FsLoader, MAX_INCLUDE_DEPTH, MAX_INCLUDES, MapLoader, SourceLoader, resolve_includes,
};
pub use lexer::{Token, TokenKind, lex};
pub use parser::parse;
#[doc(hidden)]
pub use parser::parse_expr_text;
pub use span::{FileId, SourceMap, Span};

/// What [`load`] gives for one model file.
#[derive(Debug, Clone, PartialEq)]
pub struct Loaded {
    /// The top file as parsed, with its `#include` lines and array
    /// templates kept, for the writer.
    pub template: Model,
    /// The model after includes are resolved and arrays are expanded; this
    /// is what [`check()`] saw.
    pub expanded: Model,
    /// What [`check()`] learned about `expanded`.
    pub info: ModelInfo,
    /// The diagnostics of every pass, sorted by file and then by start
    /// offset. The sort is stable, so diagnostics that start at the same
    /// place keep the order of the passes.
    pub diagnostics: Vec<Diagnostic>,
}

/// Loads the file `file` of `map`: parses it, resolves its includes through
/// `loader` (adding each included file to `map`), expands its arrays and
/// checks the result.
pub fn load(file: FileId, map: &mut SourceMap, loader: &mut dyn SourceLoader) -> Loaded {
    let (template, mut diagnostics) = parse(map.text(file), file);
    let (resolved, include_diags) = resolve_includes(&template, file, map, loader);
    diagnostics.extend(include_diags);
    let (expanded, expand_diags) = expand(&resolved);
    diagnostics.extend(expand_diags);
    let (info, check_diags) = check(&expanded);
    diagnostics.extend(check_diags);
    diagnostics.sort_by_key(|d| (d.span.file.0, d.span.start));
    Loaded {
        template,
        expanded,
        info,
        diagnostics,
    }
}
