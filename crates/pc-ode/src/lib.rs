//! Lexer, parser, AST, diagnostics and `.ode` writer for XPP/XPPAUT model files.
//!
//! See `docs/SPEC.md`, section 2. So far the crate holds source spans and
//! the source map ([`span`]), diagnostics with their stable codes
//! ([`diag`]), the tokenizer ([`lexer`]), the syntax tree ([`ast`]), the
//! known `@` option names ([`options`]), the parser ([`parser`]), the
//! builtin table ([`builtins`]), the name checks ([`mod@check`]), the include
//! pass ([`mod@include`]), the array pass ([`mod@expand`]) and the writer
//! ([`writer`]). [`parse`] turns one file into a [`Model`],
//! [`resolve_includes`] replaces its `#include` lines with the files they
//! name, [`expand()`] copies its range lines and array blocks once per
//! index, and [`check()`] resolves names and reports model errors. [`load`]
//! runs all four. [`write()`] turns a [`Model`] back into `.ode` text that
//! parses to the same model.

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
pub mod writer;

pub use ast::{
    Assign, BinaryOp, Comment, DUMMY_SPAN, Entry, Expr, Ident, Index, IndexOp, MarkovLine,
    MarkovRow, Model, OdeForm, RawAssign, Stmt, StmtKind, SuppressedName, UnaryOp, Unsupported,
    Value,
};
pub use check::{ModelInfo, ModelKind, StateKind, Symbol, SymbolKind, check};
pub use diag::{CODES, Code, Diagnostic, FixIt, MAX_DIAGNOSTICS, Severity, TOO_MANY_DIAGNOSTICS};
pub use expand::{MAX_ARRAY_COPIES, MAX_ARRAY_WORK, MAX_INDEX, expand};
pub use include::{
    FsLoader, MAX_INCLUDE_DEPTH, MAX_INCLUDED_BYTES, MAX_INCLUDED_STATEMENTS, MAX_INCLUDES,
    MapLoader, SourceLoader, resolve_includes,
};
pub use lexer::{Token, TokenKind, lex};
pub use parser::parse;
#[doc(hidden)]
pub use parser::parse_expr_text;
pub use span::{FileId, SourceMap, Span};
pub use writer::{format_number, write, write_expanded};

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
    /// place keep the order of the passes. There are at most
    /// [`MAX_DIAGNOSTICS`] of them, followed by one error
    /// ([`TOO_MANY_DIAGNOSTICS`]) that counts the rest when there were
    /// more.
    pub diagnostics: Vec<Diagnostic>,
}

/// Loads the file `file` of `map`: parses it, resolves its includes through
/// `loader` (adding each included file to `map`), expands its arrays and
/// checks the result.
pub fn load(file: FileId, map: &mut SourceMap, loader: &mut dyn SourceLoader) -> Loaded {
    let (template, mut diagnostics) = parse(map.text(file), file);
    let (resolved, include_diags, mut dropped) =
        include::resolve_counted(&template, file, map, loader);
    diagnostics.extend(include_diags);
    let (expanded, expand_diags, n) = expand::expand_counted(&resolved);
    diagnostics.extend(expand_diags);
    dropped += n;
    let (info, check_diags, n) = check::check_counted(&expanded);
    diagnostics.extend(check_diags);
    dropped += n;
    diagnostics.sort_by_key(|d| (d.span.file.0, d.span.start));
    if diagnostics.len() > MAX_DIAGNOSTICS {
        dropped += diagnostics.len() - MAX_DIAGNOSTICS;
        diagnostics.truncate(MAX_DIAGNOSTICS);
    }
    let diagnostics = diag::with_summary(diagnostics, dropped, file);
    Loaded {
        template,
        expanded,
        info,
        diagnostics,
    }
}
