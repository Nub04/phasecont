//! Statement parser.
//!
//! Splits the token stream into lines and turns each line into one
//! [`Stmt`], following the rules in `docs/language.md` (Lines, Statement
//! classification, Statement forms, Arrays, Includes). The first word of a
//! line decides what it is: a keyword when a blank or the end of the line
//! follows it and the next token is not `=`, otherwise a formula told apart
//! by its left side. Expressions are parsed by [`parse_expr`].
//!
//! The parser never panics. A line it cannot read gives a diagnostic and is
//! skipped, so later lines are still parsed. Statements are not recursive:
//! the only statement that holds others is an array block, and blocks do not
//! nest, so the stack stays bounded.

use crate::ast::{
    Assign, Comment, Entry, Expr, Ident, Index, MarkovLine, MarkovRow, Model, OdeForm, RawAssign,
    Stmt, StmtKind, Unsupported, Value,
};
use crate::diag::{Code, Diagnostic, FixIt};
use crate::lexer::{Token, TokenKind, lex};
use crate::options::is_known_option;
use crate::parser::expr::clip;
use crate::parser::{ExprCursor, parse_expr, parse_index, parse_name};
use crate::span::{FileId, Span};

/// The file starts with a digit, the old XPPAUT format.
pub const OLD_STYLE_FILE: Code = Code(18);
/// A line that is not any statement form.
pub const UNRECOGNIZED_STATEMENT: Code = Code(19);
/// A statement whose form is known but whose parts are wrong (a missing
/// `=`, name or brace).
pub const MALFORMED_STATEMENT: Code = Code(20);
/// Tokens left over after a complete statement.
pub const TRAILING_TEXT: Code = Code(21);
/// A `#` comment after a statement other than `par`, `number` and `@`.
pub const TRAILING_COMMENT: Code = Code(22);
/// A list value or `x(0)` value that is not a number literal.
pub const BAD_VALUE: Code = Code(23);
/// An `@` option name that XPPAUT does not document.
pub const UNKNOWN_OPTION: Code = Code(24);
/// An `@` option with blanks around `=`, which XPPAUT 8.0 ignores.
pub const SPACED_OPTION: Code = Code(25);
/// A `markov` row with the wrong number of entries, or missing rows.
pub const MARKOV_ROWS: Code = Code(26);
/// A nested, unclosed or stray array block line.
pub const BAD_BLOCK: Code = Code(27);
/// Volterra equations.
pub const UNSUPPORTED_VOLTERRA: Code = Code(28);
/// Boundary value problems (`bndry`).
pub const UNSUPPORTED_BOUNDARY: Code = Code(29);
/// Algebraic constraints (`0=`, `solv`).
pub const UNSUPPORTED_DAE: Code = Code(30);
pub const UNSUPPORTED_SPECIAL: Code = Code(31);
pub const UNSUPPORTED_EXPORT: Code = Code(32);
pub const UNSUPPORTED_ONLY: Code = Code(33);
pub const UNSUPPORTED_OPTIONS_FILE: Code = Code(34);
pub const UNSUPPORTED_HOM_BCS: Code = Code(35);
/// `table name file`.
pub const UNSUPPORTED_FILE_TABLE: Code = Code(36);
/// `sum(..)of(..)`.
pub const UNSUPPORTED_SUM_OF: Code = Code(37);
/// A line that starts with `"`.
pub const UNSUPPORTED_ACTIVE_COMMENT: Code = Code(38);

/// The code and message for an unsupported statement family.
fn unsupported_info(family: Unsupported) -> (Code, &'static str) {
    use Unsupported::*;
    match family {
        Volterra => (
            UNSUPPORTED_VOLTERRA,
            "Volterra integral equations are not supported; the line is kept as written",
        ),
        Boundary => (
            UNSUPPORTED_BOUNDARY,
            "boundary conditions are not supported; the line is kept as written",
        ),
        Dae => (
            UNSUPPORTED_DAE,
            "algebraic constraints are not supported; the line is kept as written",
        ),
        Special => (
            UNSUPPORTED_SPECIAL,
            "`special` is not supported; the line is kept as written",
        ),
        Export => (
            UNSUPPORTED_EXPORT,
            "`export` is not supported; the line is kept as written",
        ),
        Only => (
            UNSUPPORTED_ONLY,
            "`only` is not supported; the line is kept as written",
        ),
        OptionsFile => (
            UNSUPPORTED_OPTIONS_FILE,
            "`options` files are not supported; the line is kept as written",
        ),
        HomBcs => (
            UNSUPPORTED_HOM_BCS,
            "`hom_bcs` is not supported; the line is kept as written",
        ),
        FileTable => (
            UNSUPPORTED_FILE_TABLE,
            "tables read from a file are not supported; the line is kept as written",
        ),
        SumOf => (
            UNSUPPORTED_SUM_OF,
            "`sum(..)of(..)` is not supported; the line is kept as written",
        ),
        ActiveComment => (
            UNSUPPORTED_ACTIVE_COMMENT,
            "active comments are not supported; the line is kept as written",
        ),
    }
}

/// What the first word of a line declares, when it is a keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keyword {
    Markov,
    Par,
    Wiener,
    Table,
    Aux,
    Init,
    Global,
    Done,
    Number,
    Set,
    Unsupported(Unsupported),
}

/// The keyword for a lower-case first word. The words in the first group
/// need the full spelling; for the nine declarations only the first letter
/// counts (`pzz` is `par`).
fn keyword_of(key: &str) -> Option<Keyword> {
    use Unsupported as U;
    let full = match key {
        "number" => Some(Keyword::Number),
        "set" => Some(Keyword::Set),
        "options" => Some(Keyword::Unsupported(U::OptionsFile)),
        "only" => Some(Keyword::Unsupported(U::Only)),
        "special" => Some(Keyword::Unsupported(U::Special)),
        "solv" => Some(Keyword::Unsupported(U::Dae)),
        "export" => Some(Keyword::Unsupported(U::Export)),
        "volt" => Some(Keyword::Unsupported(U::Volterra)),
        "hom_bcs" => Some(Keyword::Unsupported(U::HomBcs)),
        _ => None,
    };
    if full.is_some() {
        return full;
    }
    match key.as_bytes().first()? {
        b'm' => Some(Keyword::Markov),
        b'p' => Some(Keyword::Par),
        b'w' => Some(Keyword::Wiener),
        b't' => Some(Keyword::Table),
        b'a' => Some(Keyword::Aux),
        b'i' => Some(Keyword::Init),
        b'b' => Some(Keyword::Unsupported(U::Boundary)),
        b'g' => Some(Keyword::Global),
        b'd' => Some(Keyword::Done),
        _ => None,
    }
}

/// Parses one `.ode` file into a [`Model`]. The diagnostics hold the
/// lexer's and the parser's, sorted by where they start.
///
/// Lexer diagnostics inside text kept raw (an `@` or `set` value, an active
/// comment, an unsupported statement) or after `done` are dropped, since
/// that text is not read as tokens. An old-style file gives one error and an
/// empty model.
pub fn parse(src: &str, file: FileId) -> (Model, Vec<Diagnostic>) {
    let (tokens, lex_diags) = lex(file, src);
    let mut p = StmtParser {
        file,
        text: src,
        tokens: &tokens,
        pos: 0,
        diags: Vec::new(),
        raw: Vec::new(),
    };
    let model = p.run();
    let raw = merge_ranges(std::mem::take(&mut p.raw));
    let mut diags: Vec<Diagnostic> = lex_diags
        .into_iter()
        .filter(|d| !in_ranges(&raw, d.span.start))
        .collect();
    diags.extend(p.diags);
    diags.sort_by_key(|d| d.span.start);
    (model, diags)
}

/// Sorts half-open byte ranges and merges the ones that overlap or touch.
fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// True when `offset` is inside one of the sorted, disjoint `ranges`.
fn in_ranges(ranges: &[(usize, usize)], offset: usize) -> bool {
    let after = ranges.partition_point(|&(s, _)| s <= offset);
    after > 0 && offset < ranges[after - 1].1
}

/// An array block that has been opened and not yet closed.
struct OpenBlock {
    /// The bounds and the span of `[i1..i2]`, or `None` for a block whose
    /// opener line was malformed and reported.
    range: Option<(i64, i64, Span)>,
    /// The `%` of the opener line.
    open: Span,
    /// The whole opener line, comments excluded.
    head: Span,
    body: Vec<Stmt>,
    /// Nested openers that were reported and ignored, so that their
    /// closers are ignored too.
    ignored: usize,
}

struct StmtParser<'a> {
    file: FileId,
    text: &'a str,
    tokens: &'a [Token],
    /// Index of the current token.
    pos: usize,
    diags: Vec<Diagnostic>,
    /// Byte ranges kept as raw text, where lexer diagnostics are dropped.
    raw: Vec<(usize, usize)>,
}

static EOF: TokenKind = TokenKind::Eof;

fn is_line_end(kind: &TokenKind) -> bool {
    matches!(kind, TokenKind::Newline | TokenKind::Eof)
}

/// The text after `#include` when a comment line is an include line.
fn include_rest(text: &str) -> Option<&str> {
    text.strip_prefix("#include")
        .filter(|rest| rest.is_empty() || rest.starts_with([' ', '\t']))
}

/// A token that ends the useful part of a line.
fn ends_line(kind: &TokenKind) -> bool {
    is_line_end(kind) || *kind == TokenKind::Comment
}

/// Helpers for moving over tokens and reporting.
impl StmtParser<'_> {
    fn tok(&self, k: usize) -> Option<&Token> {
        self.tokens.get(self.pos + k)
    }

    fn kind(&self, k: usize) -> &TokenKind {
        self.tok(k).map_or(&EOF, |t| &t.kind)
    }

    fn span_at(&self, k: usize) -> Span {
        match self.tok(k) {
            Some(t) => t.span,
            None => {
                let end = self.text.len();
                Span::new(self.file, end, end)
            }
        }
    }

    fn blank_at(&self, k: usize) -> bool {
        self.tok(k).is_some_and(|t| t.preceded_by_blank)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file, start, end)
    }

    /// The end of the last consumed token, or `floor` when nothing past
    /// `floor` has been consumed.
    fn prev_end(&self, floor: usize) -> usize {
        self.pos
            .checked_sub(1)
            .and_then(|i| self.tokens.get(i))
            .map_or(floor, |t| t.span.end.max(floor))
    }

    fn src(&self, span: Span) -> &str {
        self.text.get(span.start..span.end).unwrap_or("")
    }

    fn quote(&self, span: Span) -> String {
        clip(self.src(span))
    }

    fn describe(&self, k: usize) -> String {
        match self.kind(k) {
            TokenKind::Newline | TokenKind::Eof => "the end of the line".to_string(),
            TokenKind::Comment => "a comment".to_string(),
            _ => format!("`{}`", self.quote(self.span_at(k))),
        }
    }

    fn report(&mut self, code: Code, span: Span, message: String, fix_it: Option<FixIt>) {
        self.diags.push(Diagnostic {
            code,
            severity: code.severity(),
            span,
            message,
            fix_it,
            notes: Vec::new(),
        });
    }

    /// Index of the `Newline` or `Eof` token that ends the line holding
    /// token `from`.
    fn line_end(&self, from: usize) -> usize {
        let mut i = from;
        while i < self.tokens.len() && !is_line_end(&self.tokens[i].kind) {
            i += 1;
        }
        i.min(self.tokens.len().saturating_sub(1))
    }

    /// The span from the current token to the last token before the line
    /// end, comments included.
    fn rest_of_line(&self) -> Span {
        let start = self.span_at(0).start;
        let end_idx = self.line_end(self.pos);
        let end = if end_idx > self.pos {
            self.tokens[end_idx - 1].span.end
        } else {
            start
        };
        self.span(start, end)
    }

    /// Moves to the line end token.
    fn skip_line(&mut self) {
        self.pos = self.line_end(self.pos);
    }

    /// Moves past unknown tokens, which the lexer has reported.
    fn skip_unknown(&mut self) {
        while *self.kind(0) == TokenKind::Unknown {
            self.advance();
        }
    }

    /// Moves to the next comment or line end on this line.
    fn skip_to_comment(&mut self) {
        while !ends_line(self.kind(0)) {
            self.advance();
        }
    }

    /// Runs the expression parser at the current token.
    fn expr(&mut self) -> crate::ast::Expr {
        let mut c = ExprCursor {
            tokens: self.tokens,
            pos: self.pos,
            text: self.text,
            diags: &mut self.diags,
        };
        let e = parse_expr(&mut c);
        self.pos = c.pos;
        e
    }

    /// A name with an optional index form at the current token.
    fn name(&mut self) -> Option<Ident> {
        let mut c = ExprCursor {
            tokens: self.tokens,
            pos: self.pos,
            text: self.text,
            diags: &mut self.diags,
        };
        let parsed = parse_name(&mut c);
        self.pos = c.pos;
        parsed.map(|(ident, _)| ident)
    }

    /// An index form at the current `[`.
    fn index(&mut self) -> Option<Index> {
        let mut c = ExprCursor {
            tokens: self.tokens,
            pos: self.pos,
            text: self.text,
            diags: &mut self.diags,
        };
        let parsed = parse_index(&mut c);
        self.pos = c.pos;
        parsed
    }

    /// A plain name at the current token, without an index.
    fn plain_name(&mut self) -> Option<Ident> {
        let TokenKind::Name(key) = self.kind(0) else {
            return None;
        };
        let key = key.clone();
        let span = self.span_at(0);
        self.advance();
        Some(Ident {
            text: self.src(span).to_string(),
            key,
            index: None,
            span,
        })
    }

    /// The span of token `k`, or a zero-width span at its start when it is
    /// a line end, so that LF, CRLF and the file end give the same span.
    fn point_at(&self, k: usize) -> Span {
        let span = self.span_at(k);
        if is_line_end(self.kind(k)) {
            self.span(span.start, span.start)
        } else {
            span
        }
    }

    /// Reports a malformed statement at token `k` and skips to the next
    /// comment or line end. An unknown token is skipped without a report,
    /// since the lexer has reported it.
    fn malformed(&mut self, k: usize, what: &str) {
        if *self.kind(k) == TokenKind::Unknown {
            self.skip_to_comment();
            return;
        }
        let span = self.point_at(k);
        let message = format!("{what}, found {}", self.describe(k));
        self.report(MALFORMED_STATEMENT, span, message, None);
        self.skip_to_comment();
    }

    /// Consumes `=` or reports a malformed statement.
    fn expect_eq(&mut self, after: &str) -> bool {
        if *self.kind(0) == TokenKind::Eq {
            self.advance();
            true
        } else {
            self.malformed(0, &format!("expected `=` after `{after}`"));
            false
        }
    }

    /// Finishes a statement that began at byte `start`: reports leftover
    /// tokens, takes a trailing comment (an error unless `comment_ok`) and
    /// stops at the line end.
    fn end_statement(&mut self, start: usize, kind: StmtKind, comment_ok: bool) -> Stmt {
        self.skip_unknown();
        if !ends_line(self.kind(0)) {
            let first = self.span_at(0);
            let first_text = self.describe(0);
            self.skip_to_comment();
            let span = self.span(first.start, self.prev_end(first.end));
            let message = format!(
                "unexpected {first_text} after the statement; `{}` is not part of it",
                self.quote(span)
            );
            self.report(TRAILING_TEXT, span, message, None);
        }
        let mut comment = None;
        if *self.kind(0) == TokenKind::Comment {
            let span = self.span_at(0);
            if !comment_ok {
                let before = self.prev_end(start);
                let message = "XPPAUT rejects a comment after this statement; put the comment \
                               on a line of its own"
                    .to_string();
                let fix = FixIt {
                    span: self.span(before, span.end),
                    replacement: String::new(),
                };
                self.report(TRAILING_COMMENT, span, message, Some(fix));
            }
            comment = Some(Comment {
                text: self.src(span).to_string(),
                span,
            });
            self.advance();
        }
        let end = self.prev_end(start);
        self.skip_line();
        Stmt {
            kind,
            comment,
            span: self.span(start, end),
            array_index: None,
            array_template: None,
        }
    }

    /// Keeps the rest of the line, from the current token, as an
    /// unsupported statement and gives the family's warning.
    fn unsupported(&mut self, family: Unsupported) -> Stmt {
        let span = self.rest_of_line();
        let (code, message) = unsupported_info(family);
        self.report(code, span, message.to_string(), None);
        self.raw.push((span.start, span.end));
        self.skip_line();
        Stmt {
            kind: StmtKind::Unsupported {
                family,
                text: self.src(span).to_string(),
            },
            comment: None,
            span,
            array_index: None,
            array_template: None,
        }
    }
}

/// The file, lines and array blocks.
impl StmtParser<'_> {
    fn run(&mut self) -> Model {
        if self.old_style() {
            return Model::default();
        }
        let mut top: Vec<Stmt> = Vec::new();
        let mut block: Option<OpenBlock> = None;
        loop {
            while *self.kind(0) == TokenKind::Newline {
                self.advance();
            }
            if *self.kind(0) == TokenKind::Eof {
                break;
            }
            if *self.kind(0) == TokenKind::Percent {
                self.block_line(&mut block, &mut top);
                continue;
            }
            let Some(stmt) = self.statement() else {
                continue;
            };
            let done = stmt.kind == StmtKind::Done;
            if done {
                // `done` ends the file, and any open block with it.
                self.close_unclosed(&mut block, &mut top);
                top.push(stmt);
                break;
            }
            let target = match block.as_mut() {
                Some(b) => &mut b.body,
                None => &mut top,
            };
            target.push(stmt);
        }
        self.close_unclosed(&mut block, &mut top);
        Model {
            statements: top,
            suppressed: Vec::new(),
        }
    }

    /// Reports an old-style file (first line starts with a digit) and keeps
    /// every other diagnostic out.
    fn old_style(&mut self) -> bool {
        let bom = if self.text.starts_with('\u{feff}') {
            '\u{feff}'.len_utf8()
        } else {
            0
        };
        let rest = &self.text[bom..];
        let line = rest.split('\n').next().unwrap_or("");
        let line = line.strip_suffix('\r').unwrap_or(line);
        let lead = line.len() - line.trim_start_matches([' ', '\t']).len();
        if !line[lead..].starts_with(|c: char| c.is_ascii_digit()) {
            return false;
        }
        let span = self.span(bom + lead, bom + line.len());
        let message = "this file starts with a number, so it is in the old XPPAUT format, \
                       which is not read; convert it to the current format"
            .to_string();
        self.report(OLD_STYLE_FILE, span, message, None);
        self.raw.push((0, self.text.len() + 1));
        true
    }

    /// The statement for a block that ends at byte `end`.
    fn block_stmt(&self, b: OpenBlock, end: usize) -> Stmt {
        let kind = match b.range {
            Some((start, end, range_span)) => StmtKind::Block {
                start,
                end,
                range_span,
                body: b.body,
            },
            None => StmtKind::BadBlock { body: b.body },
        };
        Stmt {
            span: self.span(b.open.start, end),
            array_index: None,
            array_template: None,
            kind,
            comment: None,
        }
    }

    /// Reports a block that is still open and adds it to `top`.
    fn close_unclosed(&mut self, block: &mut Option<OpenBlock>, top: &mut Vec<Stmt>) {
        let Some(b) = block.take() else {
            return;
        };
        let message =
            "this array block is not closed; end it with `%` on a line of its own".to_string();
        let at = b.range.map_or(b.head, |r| r.2);
        self.report(BAD_BLOCK, at, message, None);
        let end = b.body.last().map_or(b.head.end, |s| s.span.end);
        top.push(self.block_stmt(b, end));
    }

    /// A line that starts with `%`: `%[i1..i2]` opens a block, `%` alone
    /// closes it. A malformed `%[` line is reported and still opens a block,
    /// kept as [`StmtKind::BadBlock`], so that its `%` closes it quietly.
    fn block_line(&mut self, block: &mut Option<OpenBlock>, top: &mut Vec<Stmt>) {
        let percent = self.span_at(0);
        self.advance();
        if *self.kind(0) != TokenKind::LBracket {
            if !ends_line(self.kind(0)) {
                self.malformed_block(percent);
                return;
            }
            self.block_tail(percent.start);
            match block {
                Some(b) if b.ignored > 0 => b.ignored -= 1,
                Some(_) => {
                    if let Some(b) = block.take() {
                        top.push(self.block_stmt(b, percent.end));
                    }
                }
                None => {
                    let message = "this `%` closes no array block".to_string();
                    self.report(BAD_BLOCK, percent, message, None);
                }
            }
            return;
        }
        let range = self.block_range().filter(|_| ends_line(self.kind(0)));
        let head = if range.is_some() {
            let head = self.span(percent.start, self.prev_end(percent.end));
            self.block_tail(percent.start);
            head
        } else {
            self.malformed_block(percent)
        };
        match block {
            Some(b) => {
                if let Some((_, _, range_span)) = range {
                    let message =
                        "array blocks cannot be nested; close the open block first".to_string();
                    self.report(BAD_BLOCK, percent.to(range_span), message, None);
                }
                b.ignored += 1;
            }
            None => {
                *block = Some(OpenBlock {
                    range,
                    open: percent,
                    head,
                    body: Vec::new(),
                    ignored: 0,
                });
            }
        }
    }

    /// `[i1..i2]` with optionally negative integers, at the current `[`.
    fn block_range(&mut self) -> Option<(i64, i64, Span)> {
        let open = self.span_at(0);
        self.advance();
        let start = self.block_bound()?;
        if *self.kind(0) != TokenKind::DotDot {
            return None;
        }
        self.advance();
        let end = self.block_bound()?;
        if *self.kind(0) != TokenKind::RBracket {
            return None;
        }
        let close = self.span_at(0);
        self.advance();
        Some((start, end, open.to(close)))
    }

    fn block_bound(&mut self) -> Option<i64> {
        let negative = *self.kind(0) == TokenKind::Minus;
        if negative {
            self.advance();
        }
        let TokenKind::Num(_) = self.kind(0) else {
            return None;
        };
        let text = self.src(self.span_at(0));
        if !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let value: i64 = text.parse().ok()?;
        self.advance();
        Some(if negative { -value } else { value })
    }

    /// Reports a malformed block line from `percent` to the line end and
    /// returns the span it reported.
    fn malformed_block(&mut self, percent: Span) -> Span {
        let span = self.span(percent.start, self.rest_of_line().end.max(percent.end));
        let message = format!(
            "expected `%[i1..i2]` or `%` alone on a line, found `{}`",
            self.quote(span)
        );
        self.report(BAD_BLOCK, span, message, None);
        self.skip_line();
        span
    }

    /// The end of a block line: a trailing comment is an error.
    fn block_tail(&mut self, start: usize) {
        if *self.kind(0) == TokenKind::Comment {
            let span = self.span_at(0);
            let before = self.prev_end(start);
            let message = "a comment cannot follow an array block line; put the comment on a \
                           line of its own"
                .to_string();
            let fix = FixIt {
                span: self.span(before, span.end),
                replacement: String::new(),
            };
            self.report(TRAILING_COMMENT, span, message, Some(fix));
        }
        self.skip_line();
    }
}

/// Statement dispatch and keyword statements.
impl StmtParser<'_> {
    /// One statement from the current line, or `None` after reporting a
    /// line that cannot be read. Leaves the position at the line end of the
    /// last line used.
    fn statement(&mut self) -> Option<Stmt> {
        match self.kind(0) {
            TokenKind::Comment => return self.comment_line(),
            TokenKind::At => return Some(self.options_line()),
            TokenKind::Unknown if self.text[self.span_at(0).start..].starts_with('"') => {
                return Some(self.unsupported(Unsupported::ActiveComment));
            }
            TokenKind::Name(key) if self.keyword_position() => {
                if let Some(keyword) = keyword_of(key) {
                    return self.keyword_statement(keyword);
                }
            }
            _ => {}
        }
        self.formula()
    }

    /// True when the first word is in keyword position: a blank or the end
    /// of the line follows it, and the next token is not `=`.
    fn keyword_position(&self) -> bool {
        let next = self.kind(1);
        (self.blank_at(1) || is_line_end(next)) && !matches!(next, TokenKind::Eq)
    }

    fn keyword_statement(&mut self, keyword: Keyword) -> Option<Stmt> {
        let start = self.span_at(0).start;
        match keyword {
            Keyword::Unsupported(family) => Some(self.unsupported(family)),
            Keyword::Done => {
                let word = self.span_at(0);
                let mut span = word;
                let mut comment = None;
                if *self.kind(1) == TokenKind::Comment {
                    // R12 m: a comment after `done` is reported like any
                    // other trailing comment; the rest of the file is not.
                    let at = self.span_at(1);
                    let message = "XPPAUT rejects a comment after this statement; put the \
                                   comment on a line of its own"
                        .to_string();
                    let fix = FixIt {
                        span: self.span(word.end, at.end),
                        replacement: String::new(),
                    };
                    self.report(TRAILING_COMMENT, at, message, Some(fix));
                    comment = Some(Comment {
                        text: self.src(at).to_string(),
                        span: at,
                    });
                    span = word.to(at);
                }
                self.raw.push((span.end, self.text.len() + 1));
                self.pos = self.tokens.len().saturating_sub(1);
                Some(Stmt {
                    kind: StmtKind::Done,
                    comment,
                    span,
                    array_index: None,
                    array_template: None,
                })
            }
            Keyword::Par | Keyword::Number | Keyword::Init => {
                let word = self.quote(self.span_at(0));
                self.advance();
                let entries = self.entry_list(&word, keyword == Keyword::Par);
                let (kind, comment_ok) = match keyword {
                    Keyword::Par => (StmtKind::Par { entries }, true),
                    Keyword::Number => (StmtKind::Number { entries }, true),
                    _ => (StmtKind::Init { entries }, false),
                };
                Some(self.end_statement(start, kind, comment_ok))
            }
            Keyword::Aux => {
                if let Some(family) = self.formula_family() {
                    return Some(self.unsupported(family));
                }
                self.advance();
                let Some(name) = self.name() else {
                    self.malformed(0, "expected a name after `aux`");
                    self.skip_line();
                    return None;
                };
                if !self.expect_eq(&name.text) {
                    self.skip_line();
                    return None;
                }
                let rhs = self.expr();
                Some(self.end_statement(start, StmtKind::Aux { name, rhs }, false))
            }
            Keyword::Wiener => {
                self.advance();
                let names = self.wiener_names();
                Some(self.end_statement(start, StmtKind::Wiener { names }, false))
            }
            Keyword::Table => self.table(start),
            Keyword::Global => self.global(start),
            Keyword::Set => self.set(start),
            Keyword::Markov => self.markov(start),
        }
    }

    /// A full-line comment, or `#include path`.
    fn comment_line(&mut self) -> Option<Stmt> {
        let span = self.span_at(0);
        let text = self.src(span).to_string();
        self.advance();
        self.skip_line();
        let Some(rest) = include_rest(&text) else {
            let kind = StmtKind::Comment(Comment { text, span });
            return Some(Stmt {
                kind,
                comment: None,
                array_index: None,
                array_template: None,
                span,
            });
        };
        let rest_start = span.start + "#include".len();
        // A blank followed by `#` starts a trailing comment.
        let hash = rest.char_indices().find_map(|(i, c)| {
            let after = &rest[i..];
            let tail = after.trim_start_matches([' ', '\t']);
            (matches!(c, ' ' | '\t') && tail.starts_with('#')).then(|| i + after.len() - tail.len())
        });
        let path_part = &rest[..hash.unwrap_or(rest.len())];
        let path = path_part.trim_matches([' ', '\t']);
        if path.is_empty() {
            let message = "`#include` needs a file name, as in `#include other.ode`".to_string();
            self.report(MALFORMED_STATEMENT, span, message, None);
            return None;
        }
        let offset =
            rest_start + (path_part.len() - path_part.trim_start_matches([' ', '\t']).len());
        let path_span = self.span(offset, offset + path.len());
        let comment = hash.map(|h| {
            let at = self.span(rest_start + h, span.end);
            let message = "XPPAUT does not load the file when a comment follows the name; put \
                           the comment on a line of its own"
                .to_string();
            let fix = FixIt {
                span: self.span(path_span.end, at.end),
                replacement: String::new(),
            };
            self.report(TRAILING_COMMENT, at, message, Some(fix));
            Comment {
                text: self.src(at).to_string(),
                span: at,
            }
        });
        Some(Stmt {
            kind: StmtKind::Include {
                path: path.to_string(),
                path_span,
            },
            comment,
            span,
            array_index: None,
            array_template: None,
        })
    }

    /// The entries of a `par`, `number` or `init` list, separated by commas
    /// or blanks. A bare name is allowed only when `bare_ok`.
    fn entry_list(&mut self, word: &str, bare_ok: bool) -> Vec<Entry> {
        let keyword_end = self.prev_end(0);
        // Set when the list was reported or holds an unknown token, so that
        // an empty list is not reported a second time.
        let mut bad = false;
        let mut entries = Vec::new();
        loop {
            if ends_line(self.kind(0)) {
                break;
            }
            if *self.kind(0) == TokenKind::Comma && !entries.is_empty() {
                self.advance();
                continue;
            }
            let Some(name) = self.name() else {
                self.malformed(0, &format!("expected a name in the `{word}` list"));
                bad = true;
                break;
            };
            let value = if *self.kind(0) == TokenKind::Eq {
                self.advance();
                Some(self.list_value(false, None))
            } else if bare_ok {
                None
            } else {
                let point = self.span(name.span.end, name.span.end);
                let message = format!(
                    "expected `=` and a value after `{}` in the `{word}` list",
                    name.text
                );
                self.report(MALFORMED_STATEMENT, name.span, message, None);
                Some(Value::Error { span: point })
            };
            let span = self.span(name.span.start, self.prev_end(name.span.end));
            entries.push(Entry { name, value, span });
            let next_is_entry = matches!(self.kind(0), TokenKind::Name(_)) && self.blank_at(0);
            if !(ends_line(self.kind(0)) || *self.kind(0) == TokenKind::Comma || next_is_entry) {
                self.malformed(0, "expected `,` or another entry");
                break;
            }
        }
        if entries.is_empty() && !bad {
            let span = self.span(keyword_end, keyword_end);
            let message = format!("`{word}` needs at least one entry");
            self.report(MALFORMED_STATEMENT, span, message, None);
        }
        entries
    }

    /// A value: an optionally negative number literal or a bare index form.
    /// In a list (`to_line_end` false) the value ends at a comma, a blank,
    /// a comment or the line end; after `x(0)=` it runs to the comment or
    /// line end. `field` names a table field in the bad-value message.
    fn list_value(&mut self, to_line_end: bool, field: Option<&str>) -> Value {
        if ends_line(self.kind(0)) || *self.kind(0) == TokenKind::Comma {
            let point = self.span(self.prev_end(0), self.prev_end(0));
            let message = format!("expected a value after `=`, found {}", self.describe(0));
            self.report(BAD_VALUE, point, message, None);
            return Value::Error { span: point };
        }
        let first = self.pos;
        let start = self.span_at(0).start;
        let mut value = None;
        if *self.kind(0) == TokenKind::LBracket {
            let before = self.diags.len();
            match self.index() {
                Some(index @ Index::Var { .. }) => value = Some(Value::Index(index)),
                Some(index) => {
                    let message = "a range cannot be a value; use `[j]` or a number".to_string();
                    self.report(BAD_VALUE, index.span(), message, None);
                    value = Some(Value::Error { span: index.span() });
                }
                None => {
                    let span = self.span(start, self.prev_end(start));
                    if self.diags.len() == before {
                        self.report(BAD_VALUE, span, "malformed index value".to_string(), None);
                    }
                    value = Some(Value::Error { span });
                }
            }
        } else {
            let negative = *self.kind(0) == TokenKind::Minus;
            let num_at = usize::from(negative);
            if let TokenKind::Num(v) = self.kind(num_at) {
                let v = *v;
                let num_span = self.span_at(num_at);
                let attached = !negative || !self.blank_at(1);
                if attached {
                    let digits = self.src(num_span);
                    let text = if negative {
                        format!("-{digits}")
                    } else {
                        digits.to_string()
                    };
                    value = Some(Value::Num {
                        value: if negative { -v } else { v },
                        text,
                        span: self.span(start, num_span.end),
                    });
                    self.pos += num_at + 1;
                } else {
                    // `- 1`: the sign and the number make one bad value.
                    self.pos += 2;
                }
            }
        }
        // Anything else up to the end of the value makes the value bad.
        let more = |p: &Self| {
            let kind = p.kind(0);
            !(ends_line(kind)
                || *kind == TokenKind::Comma
                || (!to_line_end && p.blank_at(0) && p.pos > first))
        };
        if value.is_some() && !more(self) {
            return value.unwrap_or(Value::Error {
                span: self.span(start, start),
            });
        }
        let mut silent = matches!(value, Some(Value::Error { .. }));
        while more(self) {
            silent |= *self.kind(0) == TokenKind::Unknown;
            self.advance();
        }
        let span = self.span(start, self.prev_end(start));
        if !silent {
            let what = field.map_or_else(|| "the value".to_string(), |f| format!("the {f}"));
            let message = format!(
                "`{}` is not a number literal; XPPAUT reads only a leading number here, so \
                 write {what} as a number such as `1.5` or `-2`",
                self.quote(span)
            );
            self.report(BAD_VALUE, span, message, None);
        }
        Value::Error { span }
    }

    /// The names of a `wiener` line, separated by commas or blanks.
    fn wiener_names(&mut self) -> Vec<Ident> {
        let mut bad = false;
        let keyword_end = self.prev_end(0);
        let mut names = Vec::new();
        loop {
            if ends_line(self.kind(0)) {
                break;
            }
            if *self.kind(0) == TokenKind::Comma && !names.is_empty() {
                self.advance();
                continue;
            }
            match self.name() {
                Some(name) => names.push(name),
                None => {
                    self.malformed(0, "expected a name in the `wiener` list");
                    bad = true;
                    break;
                }
            }
        }
        if names.is_empty() && !bad {
            let span = self.span(keyword_end, keyword_end);
            let message = "`wiener` needs at least one name".to_string();
            self.report(MALFORMED_STATEMENT, span, message, None);
        }
        names
    }
}

/// `table`, `global`, `set`, `markov` and `@` lines.
impl StmtParser<'_> {
    /// `table name % npts lo hi expr`, or `table name file` (unsupported).
    fn table(&mut self, start: usize) -> Option<Stmt> {
        let line = self.pos;
        self.advance();
        let Some(name) = self.name() else {
            self.malformed(0, "expected a name after `table`");
            self.skip_line();
            return None;
        };
        if ends_line(self.kind(0)) {
            self.malformed(
                0,
                &format!("expected `%` or a file name after `{}`", name.text),
            );
            self.skip_line();
            return None;
        }
        if *self.kind(0) != TokenKind::Percent {
            self.pos = line;
            return Some(self.unsupported(Unsupported::FileTable));
        }
        self.advance();
        // Stop at the first missing or bad field, so that one mistake gives
        // one diagnostic.
        let mut fields = Vec::with_capacity(3);
        for field in ["number of points", "lower bound", "upper bound"] {
            let value = self.table_field(field, &name.text);
            let bad = matches!(value, Value::Error { .. });
            fields.push(value);
            if bad {
                break;
            }
        }
        let rhs = if fields.len() == 3 && !matches!(fields[2], Value::Error { .. }) {
            if ends_line(self.kind(0)) {
                self.malformed(0, &format!("expected the formula of table `{}`", name.text));
                None
            } else {
                Some(self.expr())
            }
        } else {
            None
        };
        let point = self.point_at(0);
        if rhs.is_none() {
            self.skip_to_comment();
        }
        let mut fields = fields.into_iter();
        let mut next = || fields.next().unwrap_or(Value::Error { span: point });
        let (npts, lo, hi) = (next(), next(), next());
        let kind = StmtKind::Table {
            name,
            npts,
            lo,
            hi,
            rhs: rhs.unwrap_or(Expr::Error { span: point }),
        };
        Some(self.end_statement(start, kind, false))
    }

    /// One number field of a `table` line. A missing or bad field is
    /// reported and gives [`Value::Error`].
    fn table_field(&mut self, field: &str, table: &str) -> Value {
        if ends_line(self.kind(0)) {
            let point = self.point_at(0);
            self.malformed(0, &format!("expected the {field} of table `{table}`"));
            return Value::Error { span: point };
        }
        let value = self.list_value(false, Some(field));
        if let Value::Num { text, span, .. } = &value
            && field == "number of points"
            && !(text.bytes().all(|b| b.is_ascii_digit()) && text.bytes().any(|b| b != b'0'))
        {
            let span = *span;
            let message = format!(
                "the number of points must be a positive integer, not `{}`",
                self.quote(span)
            );
            self.report(BAD_VALUE, span, message, None);
            return Value::Error { span };
        }
        value
    }

    /// `global sign cond {name=expr; ...}`.
    fn global(&mut self, start: usize) -> Option<Stmt> {
        self.advance();
        let negative = *self.kind(0) == TokenKind::Minus;
        let at = usize::from(negative);
        let sign = match self.kind(at) {
            TokenKind::Num(_) if !(negative && self.blank_at(1)) => {
                match (negative, self.src(self.span_at(at))) {
                    (false, "0") => Some(0),
                    (false, "1") => Some(1),
                    (true, "1") => Some(-1),
                    _ => None,
                }
            }
            _ => None,
        };
        let Some(sign) = sign else {
            self.malformed(
                0,
                "expected the crossing direction -1, 0 or 1 after `global`",
            );
            self.skip_line();
            return None;
        };
        self.pos += at + 1;
        let cond = self.expr();
        let mut assigns = Vec::new();
        if *self.kind(0) != TokenKind::LBrace {
            self.malformed(0, "expected `{` and the assignments of the event");
        } else {
            let open = self.span_at(0);
            self.advance();
            self.global_assigns(open, &mut assigns);
        }
        let kind = StmtKind::Global {
            sign,
            cond,
            assigns,
        };
        Some(self.end_statement(start, kind, false))
    }

    /// The `name=expr` assignments of a `global` block, after its `{`.
    fn global_assigns(&mut self, open: Span, assigns: &mut Vec<Assign>) {
        loop {
            if *self.kind(0) == TokenKind::RBrace {
                if assigns.is_empty() {
                    // Measured: XPPAUT 8.0 rejects `{}` ("Bad global").
                    let span = open.to(self.span_at(0));
                    let message = "`global` needs at least one assignment, as in \
                                   `global 1 x {x=0}`"
                        .to_string();
                    self.report(MALFORMED_STATEMENT, span, message, None);
                }
                self.advance();
                return;
            }
            if ends_line(self.kind(0)) {
                let message = "this `{` is not closed".to_string();
                self.report(MALFORMED_STATEMENT, open, message, None);
                return;
            }
            let Some(name) = self.name() else {
                self.malformed(0, "expected a name to assign in the `global` block");
                return;
            };
            if !self.expect_eq(&name.text) {
                return;
            }
            let rhs = self.expr();
            let span = self.span(name.span.start, self.prev_end(name.span.end));
            assigns.push(Assign { name, rhs, span });
            match self.kind(0) {
                TokenKind::Semicolon => self.advance(),
                TokenKind::RBrace => {}
                kind if ends_line(kind) => {
                    let message = "this `{` is not closed".to_string();
                    self.report(MALFORMED_STATEMENT, open, message, None);
                    return;
                }
                _ => {
                    self.malformed(0, "expected `;` or `}` after the assignment");
                    return;
                }
            }
        }
    }

    /// `set name {a=1,b=2}`.
    fn set(&mut self, start: usize) -> Option<Stmt> {
        self.advance();
        let Some(name) = self.name() else {
            self.malformed(0, "expected a name after `set`");
            self.skip_line();
            return None;
        };
        let mut entries = Vec::new();
        if *self.kind(0) != TokenKind::LBrace {
            self.malformed(0, &format!("expected `{{` after `set {}`", name.text));
        } else {
            let open = self.span_at(0);
            self.advance();
            self.set_entries(open, &mut entries);
        }
        Some(self.end_statement(start, StmtKind::Set { name, entries }, false))
    }

    /// The entries of a `set` block, after its `{`.
    fn set_entries(&mut self, open: Span, entries: &mut Vec<RawAssign>) {
        loop {
            match self.kind(0) {
                TokenKind::RBrace => {
                    self.advance();
                    return;
                }
                TokenKind::Comma if !entries.is_empty() => {
                    self.advance();
                    continue;
                }
                kind if ends_line(kind) => {
                    let message = "this `{` is not closed".to_string();
                    self.report(MALFORMED_STATEMENT, open, message, None);
                    return;
                }
                _ => {}
            }
            let Some(name) = self.name() else {
                self.malformed(0, "expected a name in the `set` block");
                return;
            };
            if !self.expect_eq(&name.text) {
                return;
            }
            let Some((value, value_span)) = self.raw_value(true) else {
                self.malformed(0, &format!("expected a value after `{}=`", name.text));
                return;
            };
            let span = name.span.to(value_span);
            entries.push(RawAssign {
                name,
                value,
                value_span,
                span,
            });
        }
    }

    /// A value kept as text: the tokens from here up to a comma, a blank, a
    /// comment, the line end or (when `in_braces`) a `}`. The first token
    /// may follow a blank. Returns `None` when there is no value.
    fn raw_value(&mut self, in_braces: bool) -> Option<(String, Span)> {
        let stops = |kind: &TokenKind| {
            ends_line(kind)
                || *kind == TokenKind::Comma
                || (in_braces && *kind == TokenKind::RBrace)
        };
        if stops(self.kind(0)) {
            return None;
        }
        let start = self.span_at(0).start;
        self.advance();
        while !stops(self.kind(0)) && !self.blank_at(0) {
            self.advance();
        }
        let span = self.span(start, self.prev_end(start));
        self.raw.push((span.start, span.end));
        Some((self.src(span).to_string(), span))
    }

    /// `markov name n`, then `n` rows of `n` `{expr}` entries on the next
    /// lines. Blank lines and comment lines between the rows are allowed;
    /// the comments are kept in the statement.
    fn markov(&mut self, start: usize) -> Option<Stmt> {
        self.advance();
        let Some(name) = self.name() else {
            self.malformed(0, "expected a name after `markov`");
            self.skip_line();
            return None;
        };
        let states = match self.kind(0) {
            TokenKind::Num(_) => {
                let text = self.src(self.span_at(0));
                if text.bytes().all(|b| b.is_ascii_digit()) {
                    text.parse::<u32>().ok().filter(|&n| n > 0)
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(states) = states else {
            self.malformed(
                0,
                &format!("expected the number of states of `{}`", name.text),
            );
            self.skip_line();
            return None;
        };
        let header_end = self.span_at(0).end;
        self.advance();
        self.skip_unknown();
        if !ends_line(self.kind(0)) {
            let first = self.span_at(0).start;
            self.skip_to_comment();
            let span = self.span(first, self.prev_end(first));
            let message = format!(
                "XPPAUT ignores `{}` after the number of states; delete it",
                self.quote(span)
            );
            let fix = FixIt {
                span: self.span(header_end, span.end),
                replacement: String::new(),
            };
            self.report(MALFORMED_STATEMENT, span, message, Some(fix));
        }
        let name_span = name.span;
        let mut stmt = self.end_statement(
            start,
            StmtKind::Markov {
                name,
                states,
                lines: Vec::new(),
            },
            false,
        );
        let lines = self.markov_lines(states, name_span);
        let end = lines.iter().map(|line| match line {
            MarkovLine::Row(r) => r.comment.as_ref().map_or(r.span.end, |c| c.span.end),
            MarkovLine::Comment(c) => c.span.end,
        });
        if let Some(end) = end.max() {
            stmt.span = self.span(stmt.span.start, end.max(stmt.span.end));
        }
        if let StmtKind::Markov { lines: slot, .. } = &mut stmt.kind {
            *slot = lines;
        }
        Some(stmt)
    }

    /// The rows of a `markov` statement and the comment lines between them.
    /// A line that is neither ends the rows; when rows are missing, it is
    /// reported and comments after the last row are left for the caller.
    fn markov_lines(&mut self, states: u32, name_span: Span) -> Vec<MarkovLine> {
        let mut lines = Vec::new();
        let mut rows = 0;
        // Where the last row ends, in tokens and in `lines`.
        let mut after_row = (self.pos, 0);
        while rows < states as usize {
            while *self.kind(0) == TokenKind::Newline {
                self.advance();
            }
            match self.kind(0) {
                TokenKind::LBrace => {
                    lines.push(MarkovLine::Row(self.markov_row(states)));
                    rows += 1;
                    after_row = (self.pos, lines.len());
                }
                TokenKind::Comment if include_rest(self.src(self.span_at(0))).is_none() => {
                    let span = self.span_at(0);
                    let text = self.src(span).to_string();
                    lines.push(MarkovLine::Comment(Comment { text, span }));
                    self.advance();
                    self.skip_line();
                }
                _ => {
                    let message = format!(
                        "`{}` has {states} states, so it needs {states} rows of {states} \
                         `{{...}}` entries, but it has {rows}",
                        self.src(name_span),
                    );
                    self.report(MARKOV_ROWS, name_span, message, None);
                    (self.pos, _) = after_row;
                    lines.truncate(after_row.1);
                    break;
                }
            }
        }
        lines
    }

    /// One row of `{expr}` groups, at the current `{`.
    fn markov_row(&mut self, states: u32) -> MarkovRow {
        let start = self.span_at(0).start;
        let mut entries = Vec::new();
        while *self.kind(0) == TokenKind::LBrace {
            let open = self.span_at(0);
            self.advance();
            entries.push(self.expr());
            if *self.kind(0) == TokenKind::RBrace {
                self.advance();
            } else {
                let message = "this `{` is not closed".to_string();
                self.report(MARKOV_ROWS, open, message, None);
                self.skip_to_comment();
            }
        }
        let span = self.span(start, self.prev_end(start));
        // Ends the row line: text or a comment after the row is reported,
        // and the comment is kept on the row.
        let comment = self.end_statement(start, StmtKind::Done, false).comment;
        if entries.len() != states as usize {
            let message = format!(
                "this row has {} entries, but the chain has {states} states",
                entries.len()
            );
            self.report(MARKOV_ROWS, span, message, None);
        }
        MarkovRow {
            entries,
            comment,
            span,
        }
    }

    /// `@ name=value ...`, entries split by commas or blanks.
    fn options_line(&mut self) -> Stmt {
        let start = self.span_at(0).start;
        let at_end = self.span_at(0).end;
        let mut bad = false;
        self.advance();
        let mut entries = Vec::new();
        loop {
            match self.kind(0) {
                kind if ends_line(kind) => break,
                TokenKind::Comma => {
                    self.advance();
                    continue;
                }
                _ => {}
            }
            let Some(name) = self.plain_name() else {
                self.option_error(0, "expected an option name");
                bad = true;
                continue;
            };
            if *self.kind(0) != TokenKind::Eq {
                self.option_error(0, &format!("expected `=` after option `{}`", name.text));
                bad = true;
                continue;
            }
            let eq = self.span_at(0);
            self.advance();
            let Some((value, value_span)) = self.raw_value(false) else {
                self.option_error(0, &format!("expected a value after `{}=`", name.text));
                bad = true;
                continue;
            };
            // Only real blanks count: a backslash-newline continuation also
            // sets `preceded_by_blank`, and XPPAUT 8.0 applies `dt=\<newline>.5`
            // (measured).
            let blank = |s: usize, e: usize| self.src(self.span(s, e)).contains([' ', '\t']);
            if blank(name.span.end, eq.start) || blank(eq.end, value_span.start) {
                let gap = self.span(name.span.end, value_span.start);
                let message = format!(
                    "XPPAUT 8.0 ignores option `{}` because of the blanks around `=`; write \
                     `{}={}`",
                    clip(&name.text),
                    clip(&name.text),
                    clip(&value)
                );
                let fix = FixIt {
                    span: gap,
                    replacement: "=".to_string(),
                };
                self.report(SPACED_OPTION, gap, message, Some(fix));
            }
            if !is_known_option(&name.key) {
                let message = format!(
                    "unknown option `{}`; XPPAUT ignores it",
                    self.quote(name.span)
                );
                self.report(UNKNOWN_OPTION, name.span, message, None);
            }
            let span = name.span.to(value_span);
            entries.push(RawAssign {
                name,
                value,
                value_span,
                span,
            });
        }
        if entries.is_empty() && !bad {
            let span = self.span(at_end, at_end);
            let message = "`@` needs at least one option, as in `@ total=100`".to_string();
            self.report(MALFORMED_STATEMENT, span, message, None);
        }
        self.end_statement(start, StmtKind::Options { entries }, true)
    }

    /// Reports a malformed `@` entry at token `k` and skips to the next
    /// comma, comment or line end.
    fn option_error(&mut self, k: usize, what: &str) {
        if *self.kind(k) != TokenKind::Unknown {
            let span = self.point_at(k);
            let message = format!("{what}, found {}", self.describe(k));
            self.report(MALFORMED_STATEMENT, span, message, None);
        }
        while !ends_line(self.kind(0)) && *self.kind(0) != TokenKind::Comma {
            self.advance();
        }
    }
}

/// Formula lines, told apart by their left side.
impl StmtParser<'_> {
    /// `x'=`, `dx/dt=`, `x(t+1)=`, `x(0)=`, `f(a,b)=`, `!x=` or `x=`.
    fn formula(&mut self) -> Option<Stmt> {
        if let Some(family) = self.formula_family() {
            return Some(self.unsupported(family));
        }
        let start = self.span_at(0).start;
        match self.kind(0) {
            TokenKind::Unknown => {
                // The lexer has reported the character already.
                self.skip_line();
                return None;
            }
            TokenKind::Bang if matches!(self.kind(1), TokenKind::Name(_)) => {
                self.advance();
                let name = self.name()?;
                if !self.expect_eq(&format!("!{}", name.text)) {
                    self.skip_line();
                    return None;
                }
                let rhs = self.expr();
                return Some(self.end_statement(start, StmtKind::Derived { name, rhs }, false));
            }
            TokenKind::Name(_) => {}
            _ => return self.unrecognized(),
        }
        if let Some(name) = self.ddt_name() {
            self.pos += 4;
            let rhs = self.expr();
            let kind = StmtKind::Ode {
                name,
                form: OdeForm::Ddt,
                rhs,
            };
            return Some(self.end_statement(start, kind, false));
        }
        let name = self.name()?;
        match self.kind(0) {
            TokenKind::Quote => {
                self.advance();
                if !self.expect_eq(&format!("{}'", name.text)) {
                    self.skip_line();
                    return None;
                }
                let rhs = self.expr();
                let kind = StmtKind::Ode {
                    name,
                    form: OdeForm::Prime,
                    rhs,
                };
                Some(self.end_statement(start, kind, false))
            }
            TokenKind::LParen => self.paren_left(start, name),
            TokenKind::Eq => {
                self.advance();
                let rhs = self.expr();
                Some(self.end_statement(start, StmtKind::Fixed { name, rhs }, false))
            }
            _ => {
                self.pos = self.line_start(self.pos);
                self.unrecognized()
            }
        }
    }

    /// Index of the first token of the line holding token `at`.
    fn line_start(&self, at: usize) -> usize {
        let mut i = at.min(self.tokens.len());
        while i > 0 && !is_line_end(&self.tokens[i - 1].kind) {
            i -= 1;
        }
        i
    }

    /// Reports the line from the current token as unrecognized.
    /// A line that holds an unknown token is skipped quietly, since the
    /// lexer has reported that token.
    fn unrecognized(&mut self) -> Option<Stmt> {
        let line_end = self.line_end(self.pos);
        let tokens = self.tokens.get(self.pos..line_end).unwrap_or(&[]);
        if tokens.iter().any(|t| t.kind == TokenKind::Unknown) {
            self.skip_line();
            return None;
        }
        let span = self.rest_of_line();
        let message = format!(
            "`{}` is not a statement XPPAUT reads; expected a formula such as `x'=...` \
             or `x=...`, or a keyword such as `par`",
            self.quote(span)
        );
        self.report(UNRECOGNIZED_STATEMENT, span, message, None);
        self.skip_line();
        None
    }

    /// The name of a `dx/dt=` left side at the current token, if it is one.
    fn ddt_name(&self) -> Option<Ident> {
        let TokenKind::Name(key) = self.kind(0) else {
            return None;
        };
        let mut chars = key.chars();
        if chars.next() != Some('d') || !chars.next().is_some_and(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        let is_dt = matches!(self.kind(2), TokenKind::Name(k) if k == "dt");
        if !(*self.kind(1) == TokenKind::Slash && is_dt && *self.kind(3) == TokenKind::Eq) {
            return None;
        }
        let span = self.span_at(0);
        let name_span = self.span(span.start + 1, span.end);
        Some(Ident {
            text: self.src(name_span).to_string(),
            key: key[1..].to_string(),
            index: None,
            span: name_span,
        })
    }

    /// A left side `name(...)`, at the `(`: `x(t+1)`, `x(0)`, `x(t)`
    /// (Volterra) or `f(a,b)`.
    fn paren_left(&mut self, start: usize, name: Ident) -> Option<Stmt> {
        let src_at = |p: &Self, k: usize| p.src(p.span_at(k)).to_string();
        let is_t = |p: &Self, k: usize| matches!(p.kind(k), TokenKind::Name(n) if n == "t");
        let close = |p: &Self, k: usize| *p.kind(k) == TokenKind::RParen;
        // `x(t+1)`
        if is_t(self, 1)
            && *self.kind(2) == TokenKind::Plus
            && matches!(self.kind(3), TokenKind::Num(_))
            && src_at(self, 3) == "1"
            && close(self, 4)
        {
            self.pos += 5;
            let label = format!("{}(t+1)", name.text);
            if !self.expect_eq(&label) {
                self.skip_line();
                return None;
            }
            let rhs = self.expr();
            return Some(self.end_statement(start, StmtKind::Map { name, rhs }, false));
        }
        // `x(0)`
        if matches!(self.kind(1), TokenKind::Num(_)) && src_at(self, 1) == "0" && close(self, 2) {
            self.pos += 3;
            let label = format!("{}(0)", name.text);
            if !self.expect_eq(&label) {
                self.skip_line();
                return None;
            }
            let value = self.list_value(true, None);
            let kind = StmtKind::InitValue { name, value };
            return Some(self.end_statement(start, kind, false));
        }
        // `x(t)`: a Volterra equation.
        if is_t(self, 1) && close(self, 2) {
            self.pos = self.line_start(self.pos);
            return Some(self.unsupported(Unsupported::Volterra));
        }
        // `f(a,b)`
        let open = self.pos;
        self.advance();
        let mut params = Vec::new();
        loop {
            match self.plain_name() {
                Some(p) => params.push(p),
                None if *self.kind(0) == TokenKind::Unknown => {
                    self.skip_line();
                    return None;
                }
                None => {
                    self.pos = open;
                    self.malformed(
                        0,
                        &format!(
                            "expected `(t+1)`, `(0)` or a list of parameter names after `{}`",
                            name.text
                        ),
                    );
                    self.skip_line();
                    return None;
                }
            }
            match self.kind(0) {
                TokenKind::Comma => self.advance(),
                TokenKind::RParen => {
                    self.advance();
                    break;
                }
                _ => {
                    self.malformed(0, "expected `,` or `)` in the parameter list");
                    self.skip_line();
                    return None;
                }
            }
        }
        if let Some(index) = &name.index {
            let message = format!(
                "a function name cannot have an index; remove `{}`",
                self.quote(index.span())
            );
            self.report(MALFORMED_STATEMENT, index.span(), message, None);
        }
        if !self.expect_eq(&format!("{}(..)", name.text)) {
            self.skip_line();
            return None;
        }
        let rhs = self.expr();
        let kind = StmtKind::Function { name, params, rhs };
        Some(self.end_statement(start, kind, false))
    }

    /// An unsupported formula family on the current line: `0=` (an
    /// algebraic constraint), `int{..}` or `int[..]` (Volterra) or
    /// `sum(..)of(..)`.
    fn formula_family(&self) -> Option<Unsupported> {
        let tokens = &self.tokens[self.pos.min(self.tokens.len())..];
        let line: Vec<&TokenKind> = tokens
            .iter()
            .map(|t| &t.kind)
            .take_while(|k| !ends_line(k))
            .collect();
        if let (Some(TokenKind::Num(_)), Some(TokenKind::Eq)) = (line.first(), line.get(1))
            && self.src(self.span_at(0)) == "0"
        {
            return Some(Unsupported::Dae);
        }
        let name_is = |k: &TokenKind, word: &str| matches!(k, TokenKind::Name(n) if n == word);
        let mut sum_open = false;
        for (i, kind) in line.iter().enumerate() {
            let next = line.get(i + 1).copied();
            if name_is(kind, "int") && matches!(next, Some(TokenKind::LBrace | TokenKind::LBracket))
            {
                return Some(Unsupported::Volterra);
            }
            if name_is(kind, "sum") && matches!(next, Some(TokenKind::LParen)) {
                sum_open = true;
            }
            if sum_open && **kind == TokenKind::RParen && next.is_some_and(|n| name_is(n, "of")) {
                return Some(Unsupported::SumOf);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests;
