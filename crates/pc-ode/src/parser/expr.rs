//! Expression parser.
//!
//! A hand-written precedence climber over the token slice from the lexer. It
//! implements the measured precedence table in `docs/language.md`, loosest
//! to tightest:
//!
//! 1. binary `+ -` and `|`
//! 2. `* /`, `&` and implicit multiplication
//! 3. prefix `-` and `not`, whose operand is parsed at level 4
//! 4. `< > <= >= ==`
//! 5. `^` and `**`
//!
//! All binary levels are left-associative. The parser never panics: every
//! mistake becomes a diagnostic, the tree gets an [`Expr::Error`] node where
//! an operand is missing, and parsing goes on so later mistakes are reported
//! too. Tokens the lexer already reported ([`TokenKind::Unknown`]) are used
//! as silent placeholders. Precedence warnings are left out for any part of
//! the tree that came from error recovery, since its reading is a guess.
//!
//! No tree is deeper than `MAX_DEPTH` levels, so the parser and every
//! recursive pass over the tree stay within a small stack.
//!
//! Parentheses are tracked while parsing, so that the precedence warnings
//! leave parenthesized operands alone, but they are not kept in the tree.

use crate::ast::{BinaryOp, Expr, Ident, Index, IndexOp, UnaryOp};
use crate::diag::{Code, Diagnostic, FixIt};
use crate::lexer::{Token, TokenKind, lex};
use crate::span::{FileId, Span};

/// An operand is missing (`2*`, `*3`, `()`).
pub const EXPECTED_EXPRESSION: Code = Code(4);
/// Two operands with no operator between them (`2 (x)`, `(a)(b)`).
pub const MISSING_OPERATOR: Code = Code(5);
/// A token that has no place in an expression (`a'b`, `a % 2`).
pub const UNEXPECTED_TOKEN: Code = Code(6);
/// `!` or `!=`, which `.ode` files do not have.
pub const BANG_OPERATOR: Code = Code(7);
/// A single `=` used as a comparison inside an expression.
pub const EQUALS_IN_EXPRESSION: Code = Code(8);
/// A `(` or `[` with no matching closer on the line.
pub const UNCLOSED_DELIMITER: Code = Code(9);
/// Prefix `-` or `not` after an operator (`2*-3`).
pub const MISPLACED_PREFIX: Code = Code(10);
/// Unary `+`, which XPPAUT rejects.
pub const UNARY_PLUS: Code = Code(11);
/// A blank that XPPAUT would delete, joining its neighbors into one token
/// (`2 x`, `x * * 2`, `3< =3`).
pub const BLANK_IN_TOKEN: Code = Code(12);
/// A construct XPPAUT 8.0 rejects. The parser uses it for implicit
/// multiplication; `check` uses it for `ceil` and long names.
pub const PORTABILITY: Code = Code(13);
/// An operator mix that XPPAUT reads differently from C (`2*x>1`).
pub const PRECEDENCE: Code = Code(14);
/// An index form other than `[i1..i2]`, `[j]`, `[j+n]`, `[j-n]`, `[j*n]`,
/// or a range `[i1..i2]` that does not follow a name.
pub const UNSUPPORTED_INDEX: Code = Code(15);
/// An `if` that is not `if(c)then(a)else(b)`.
pub const MALFORMED_IF: Code = Code(16);
/// An expression whose tree would be deeper than `MAX_DEPTH`.
pub const EXPRESSION_TOO_DEEP: Code = Code(17);

/// The deepest expression tree the parser builds. A leaf has height 1, and
/// every node adds one level: each operator in a chain such as `x+x+x`,
/// each prefix operator, call and `if`. Parentheses do not add a level to
/// the tree, but nesting them counts against the same limit while parsing.
///
/// An expression that would go deeper gets one [`EXPRESSION_TOO_DEEP`]
/// error and the rest of it is skipped. The limit keeps the parser and
/// every later recursive pass over the tree (`strip_spans`, `Clone`, `Drop`,
/// `check`, the writer) within a 2 MB stack. 200 levels is far more than a
/// hand-written model needs.
const MAX_DEPTH: usize = 200;

/// The longest piece of source text a message quotes, in bytes. Longer text
/// is cut in the middle, so that the size of a message does not grow with
/// the size of the expression.
const QUOTE_MAX: usize = 60;

/// `s` as it should appear in a message: unchanged when it is at most
/// [`QUOTE_MAX`] bytes, otherwise its start and end joined by `...`.
pub(crate) fn clip(s: &str) -> String {
    if s.len() <= QUOTE_MAX {
        return s.to_string();
    }
    let keep = (QUOTE_MAX - 3) / 2;
    let mut head = keep;
    while !s.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = s.len() - keep;
    while !s.is_char_boundary(tail) {
        tail += 1;
    }
    format!("{}...{}", &s[..head], &s[tail..])
}

/// A position in a token slice, for the statement parser.
///
/// `tokens` should end with [`TokenKind::Eof`], as [`lex`] returns them; the
/// parser treats the end of the slice as the end of the text in any case.
/// `text` is the source text that the token spans point into.
pub(crate) struct ExprCursor<'a> {
    pub(crate) tokens: &'a [Token],
    pub(crate) pos: usize,
    pub(crate) text: &'a str,
    pub(crate) diags: &'a mut Vec<Diagnostic>,
}

/// Parses one expression starting at `c.pos` and leaves `c.pos` on the token
/// that ended it, which is always a stop token: `Newline`, `Eof`, `Comment`,
/// `;`, or at nesting depth 0 `Comma`, `RParen`, `RBracket`, `RBrace` or
/// `LBrace`. The stop token is not consumed. Other stray tokens are reported
/// and skipped up to the next stop token.
///
/// The first token counts as the start of an expression, so a prefix `-` or
/// `not` is allowed there. The diagnostics this call adds are sorted by
/// where they start.
pub(crate) fn parse_expr(c: &mut ExprCursor) -> Expr {
    if c.tokens.is_empty() {
        let span = Span::new(FileId(0), 0, 0);
        c.diags.push(Diagnostic {
            code: EXPECTED_EXPRESSION,
            severity: EXPECTED_EXPRESSION.severity(),
            span,
            message: "expected an expression, found end of the text".to_string(),
            fix_it: None,
            notes: Vec::new(),
        });
        return Expr::Error { span };
    }
    if c.pos >= c.tokens.len() {
        c.pos = c.tokens.len() - 1;
    }
    let before = c.diags.len();
    let expr = *Parser::new(c).full().expr;
    c.diags[before..].sort_by_key(|d| d.span.start);
    expr
}

/// Lexes `text` and parses it as one expression, for tests and tools. Line
/// ends after the expression are allowed; any other token left over before
/// the end of the text is reported as unexpected. The diagnostics come
/// sorted by where they start.
#[doc(hidden)]
pub fn parse_expr_text(file: FileId, text: &str) -> (Expr, Vec<Diagnostic>) {
    let (tokens, mut diags) = lex(file, text);
    let mut c = ExprCursor {
        tokens: &tokens,
        pos: 0,
        text,
        diags: &mut diags,
    };
    let e = parse_expr(&mut c);
    let mut p = Parser::new(&mut c);
    while *p.kind(0) == TokenKind::Newline {
        p.advance();
    }
    if *p.kind(0) != TokenKind::Eof {
        let span = p.span_at(0);
        let message = format!("unexpected {} after the expression", p.describe(0));
        p.report(UNEXPECTED_TOKEN, span, message, None);
    }
    diags.sort_by_key(|d| d.span.start);
    (e, diags)
}

/// Parses the name at `c.pos` with an optional index form after it, for
/// the left sides of statements. Returns `None`, consuming nothing, when
/// the current token is not a name. The flag is true when a bad index form
/// was reported and skipped; the name is still returned without it.
pub(crate) fn parse_name(c: &mut ExprCursor) -> Option<(Ident, bool)> {
    if !matches!(
        c.tokens.get(c.pos).map(|t| &t.kind),
        Some(TokenKind::Name(_))
    ) {
        return None;
    }
    let (ident, recovered) = Parser::new(c).name();
    Some((*ident, recovered))
}

/// Parses the index form that starts at the `[` at `c.pos`. Returns `None`
/// after reporting an unsupported or unclosed form, which is skipped.
pub(crate) fn parse_index(c: &mut ExprCursor) -> Option<Index> {
    Parser::new(c).index()
}

/// A parsed operand plus what the precedence warnings and the depth limit
/// need to know about it.
struct Parsed {
    expr: Box<Expr>,
    /// The operand was written in parentheses.
    parens: bool,
    /// The span including those parentheses.
    outer: Span,
    /// The height of `expr`: 1 for a leaf, plus 1 per level of nodes.
    height: usize,
    /// `expr`, or something inside it, came from error recovery.
    recovered: bool,
}

impl Parsed {
    fn new(expr: Expr, height: usize, recovered: bool) -> Parsed {
        let outer = expr.span();
        Parsed {
            expr: Box::new(expr),
            parens: false,
            outer,
            height,
            recovered,
        }
    }

    fn leaf(expr: Expr) -> Parsed {
        let recovered = expr.is_error();
        Parsed::new(expr, 1, recovered)
    }

    fn error(span: Span) -> Parsed {
        Parsed::new(Expr::Error { span }, 1, true)
    }
}

struct Parser<'p, 'a> {
    c: &'p mut ExprCursor<'a>,
    /// Token index where the expression began, a prefix position.
    start: usize,
    /// Current nesting of atoms, checked against [`MAX_DEPTH`] so that the
    /// parser's own recursion stays bounded.
    depth: usize,
    /// [`EXPRESSION_TOO_DEEP`] has been reported for this expression.
    too_deep_reported: bool,
}

static EOF: TokenKind = TokenKind::Eof;

/// Tokens that end an expression at nesting depth 0.
fn is_stop(kind: &TokenKind) -> bool {
    use TokenKind::*;
    ends_at_any_depth(kind) || matches!(kind, Comma | RParen | RBracket | RBrace | LBrace)
}

/// Tokens that end an expression at any depth, so that a missing `)` cannot
/// swallow the next line or the next assignment in a `global` block.
fn ends_at_any_depth(kind: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(kind, Newline | Eof | Comment | Semicolon)
}

/// Tokens that can start an operand, for the juxtaposition rules.
fn starts_operand(kind: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(kind, Num(_) | Name(_) | LParen | LBracket | Unknown)
}

/// Tokens that can come right after a complete operand: a binary operator
/// (including the ones that are reported but still parsed) or a stop token.
fn can_follow_operand(kind: &TokenKind) -> bool {
    use TokenKind::*;
    is_stop(kind)
        || matches!(
            kind,
            Plus | Minus
                | Star
                | Slash
                | Amp
                | Pipe
                | Caret
                | StarStar
                | Lt
                | Gt
                | Le
                | Ge
                | EqEq
                | Eq
                | Bang
        )
}

fn is_name(kind: &TokenKind, key: &str) -> bool {
    matches!(kind, TokenKind::Name(k) if k == key)
}

impl<'p, 'a> Parser<'p, 'a> {
    fn new(c: &'p mut ExprCursor<'a>) -> Self {
        Parser {
            start: c.pos,
            c,
            depth: 0,
            too_deep_reported: false,
        }
    }
}

impl Parser<'_, '_> {
    fn tok(&self, k: usize) -> Option<&Token> {
        self.c.tokens.get(self.c.pos + k)
    }

    fn kind(&self, k: usize) -> &TokenKind {
        self.tok(k).map_or(&EOF, |t| &t.kind)
    }

    fn prev(&self) -> Option<&Token> {
        self.c.pos.checked_sub(1).and_then(|i| self.c.tokens.get(i))
    }

    /// The span of token `k`, or an empty span at the end of the last token
    /// when the slice has run out.
    fn span_at(&self, k: usize) -> Span {
        match self.tok(k) {
            Some(t) => t.span,
            None => {
                let last = self.c.tokens[self.c.tokens.len() - 1].span;
                Span::new(last.file, last.end, last.end)
            }
        }
    }

    fn blank_at(&self, k: usize) -> bool {
        self.tok(k).is_some_and(|t| t.preceded_by_blank)
    }

    /// An empty span at the start of token `k`.
    fn point_at(&self, k: usize) -> Span {
        let s = self.span_at(k);
        Span::new(s.file, s.start, s.start)
    }

    /// The end of the last consumed token, as an empty span.
    fn prev_end(&self) -> Span {
        match self.prev() {
            Some(t) if self.c.pos > self.start => Span::new(t.span.file, t.span.end, t.span.end),
            _ => self.point_at(0),
        }
    }

    /// The blank between token `k` and the token before it (which may be the
    /// last consumed token when `k` is 0).
    fn gap_before(&self, k: usize) -> Span {
        let right = self.span_at(k);
        let left_end = (self.c.pos + k)
            .checked_sub(1)
            .and_then(|i| self.c.tokens.get(i))
            .map_or(right.start, |t| t.span.end.min(right.start));
        Span::new(right.file, left_end, right.start)
    }

    fn advance(&mut self) {
        if self.c.pos < self.c.tokens.len() {
            self.c.pos += 1;
        }
    }

    fn src(&self, span: Span) -> &str {
        self.c.text.get(span.start..span.end).unwrap_or("")
    }

    /// The text of `span`, shortened for a message with [`clip`].
    fn quote(&self, span: Span) -> String {
        clip(self.src(span))
    }

    /// The text from byte `start` to byte `end`, shortened with [`clip`].
    fn quote_range(&self, start: usize, end: usize) -> String {
        clip(self.c.text.get(start..end).unwrap_or(""))
    }

    fn describe(&self, k: usize) -> String {
        match self.kind(k) {
            TokenKind::Newline => "line break".to_string(),
            TokenKind::Eof => "end of the text".to_string(),
            TokenKind::Comment => "comment".to_string(),
            _ => format!("`{}`", self.quote(self.span_at(k))),
        }
    }

    fn report(&mut self, code: Code, span: Span, message: String, fix_it: Option<FixIt>) {
        self.c.diags.push(Diagnostic {
            code,
            severity: code.severity(),
            span,
            message,
            fix_it,
            notes: Vec::new(),
        });
    }

    /// Reports [`EXPRESSION_TOO_DEEP`] at `span`, once per expression, and
    /// skips the rest of the expression up to the next stop token.
    fn too_deep(&mut self, span: Span) {
        if !self.too_deep_reported {
            self.too_deep_reported = true;
            let message = format!(
                "this expression is more than {MAX_DEPTH} levels deep; split it into \
                 smaller parts"
            );
            self.report(EXPRESSION_TOO_DEEP, span, message, None);
        }
        self.skip_to_stop();
    }

    /// A fix-it that replaces `span` with its own text in parentheses.
    fn wrap_fix(&self, span: Span) -> FixIt {
        FixIt {
            span,
            replacement: format!("({})", self.src(span)),
        }
    }

    /// A fix-it that deletes `span`.
    fn delete_fix(span: Span) -> FixIt {
        FixIt {
            span,
            replacement: String::new(),
        }
    }

    /// True when a prefix operator at the current token is allowed: at the
    /// start of the expression, right after `(` or right after `,`.
    fn prefix_position(&self) -> bool {
        self.c.pos == self.start
            || self
                .prev()
                .is_some_and(|t| matches!(t.kind, TokenKind::LParen | TokenKind::Comma))
    }

    /// Skips tokens up to the next stop token at nesting depth 0, without
    /// consuming it. Parentheses and brackets in between are balanced.
    fn skip_to_stop(&mut self) {
        let mut depth = 0usize;
        loop {
            let kind = self.kind(0);
            if ends_at_any_depth(kind) || (depth == 0 && is_stop(kind)) {
                return;
            }
            match kind {
                TokenKind::LParen | TokenKind::LBracket => depth += 1,
                TokenKind::RParen | TokenKind::RBracket => depth -= 1,
                _ => {}
            }
            self.advance();
        }
    }
}

impl Parser<'_, '_> {
    /// One whole expression, then any stray tokens up to a stop token.
    fn full(&mut self) -> Parsed {
        let mut parsed = self.level1();
        let kind = self.kind(0);
        if !is_stop(kind) {
            let span = self.span_at(0);
            if *kind == TokenKind::Bang {
                let message = "`!` is not an operator; use `not(...)` for negation".to_string();
                self.report(BANG_OPERATOR, span, message, None);
            } else {
                let message = format!("unexpected {} in an expression", self.describe(0));
                self.report(UNEXPECTED_TOKEN, span, message, None);
            }
            self.skip_to_stop();
            parsed.recovered = true;
        }
        parsed
    }

    /// Level 1: binary `+`, `-` and `|`.
    fn level1(&mut self) -> Parsed {
        let mut lhs = self.level2();
        loop {
            let op = match self.kind(0) {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                TokenKind::Pipe => BinaryOp::Or,
                _ => return lhs,
            };
            let op_span = self.span_at(0);
            self.advance();
            let rhs = self.level2();
            lhs = self.binary(op, op_span, lhs, rhs, false, false);
        }
    }

    /// Level 2: `*`, `/`, `&` and every kind of juxtaposition.
    fn level2(&mut self) -> Parsed {
        let mut lhs = self.unary();
        // The operator just before the current operand was `/`.
        let mut after_div = false;
        loop {
            let op = match self.kind(0) {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Amp => BinaryOp::And,
                kind if starts_operand(kind) => {
                    lhs = self.juxtaposition(lhs, after_div);
                    after_div = false;
                    continue;
                }
                _ => return lhs,
            };
            let op_span = self.span_at(0);
            self.advance();
            let rhs = self.unary();
            lhs = self.binary(op, op_span, lhs, rhs, false, false);
            after_div = op == BinaryOp::Div;
        }
    }

    /// An operand directly after another operand. Valid only as implicit
    /// multiplication (a number followed directly by a name or `(`); the
    /// other cases are errors that recover as a multiplication.
    fn juxtaposition(&mut self, lhs: Parsed, after_div: bool) -> Parsed {
        // The recursive path keeps only token positions on the stack; the
        // diagnostics are worked out afterwards in `report_juxtaposition`.
        let right = self.c.pos;
        let right_point = self.point_at(0);
        let rhs = self.unary();
        let recovered = self.report_juxtaposition(right, &lhs, &rhs, after_div);
        self.binary(BinaryOp::Mul, right_point, lhs, rhs, true, recovered)
    }

    /// Reports the juxtaposition whose right operand starts at token index
    /// `right_at` and returns true unless it is valid implicit
    /// multiplication.
    fn report_juxtaposition(
        &mut self,
        right_at: usize,
        lhs: &Parsed,
        rhs: &Parsed,
        after_div: bool,
    ) -> bool {
        let tokens = self.c.tokens;
        let Some(right_tok) = tokens.get(right_at) else {
            return true;
        };
        let left_tok = right_at.checked_sub(1).and_then(|i| tokens.get(i));
        let left_kind = left_tok.map(|t| &t.kind);
        let right = &right_tok.kind;
        let right_span = right_tok.span;
        let right_point = Span::new(right_span.file, right_span.start, right_span.start);
        let blank = right_tok.preceded_by_blank;
        let gap = Span::new(
            right_span.file,
            left_tok.map_or(right_span.start, |t| t.span.end.min(right_span.start)),
            right_span.start,
        );
        let silent = lhs.expr.is_error()
            || rhs.expr.is_error()
            || matches!(left_kind, Some(TokenKind::Unknown))
            || *right == TokenKind::Unknown;
        if silent {
            return true;
        }
        let numeric_left = matches!(left_kind, Some(TokenKind::Num(_)));
        let word_left = match left_kind {
            Some(TokenKind::Num(_)) => true,
            Some(TokenKind::Name(k)) => !matches!(k.as_str(), "not" | "if" | "then" | "else"),
            _ => false,
        };
        let word_right = matches!(right, TokenKind::Num(_) | TokenKind::Name(_));
        let insert_star = FixIt {
            span: right_point,
            replacement: "*".to_string(),
        };
        let left_span = left_tok.map_or(gap, |t| t.span);
        if numeric_left && !blank && matches!(right, TokenKind::Name(_) | TokenKind::LParen) {
            let l = self.quote(left_span);
            let r = self.quote(rhs.outer);
            let mut message =
                format!("XPPAUT 8.0 rejects implicit multiplication; write `{l}*{r}`");
            if after_div {
                let quotient = self.quote(lhs.outer);
                message.push_str(&format!("; note that this reads as `({quotient})*{r}`"));
            }
            let span = left_span.to(rhs.outer);
            self.report(PORTABILITY, span, message, Some(insert_star));
            return false;
        }
        if word_left && word_right && blank {
            let message = format!(
                "XPPAUT deletes this blank and joins `{}` and `{}` into one word; \
                 delete the blank or add an operator",
                self.quote(left_span),
                self.quote(right_span)
            );
            self.report(BLANK_IN_TOKEN, gap, message, Some(Self::delete_fix(gap)));
        } else {
            let message = format!(
                "missing operator before `{}`; write `*` to multiply",
                self.quote(right_span)
            );
            self.report(MISSING_OPERATOR, right_span, message, Some(insert_star));
        }
        true
    }

    /// Level 3: prefix `-` and `not`, after skipping any unary `+`.
    fn unary(&mut self) -> Parsed {
        let allowed = self.prefix_position();
        self.skip_unary_plus();
        match self.kind(0) {
            TokenKind::Minus => self.prefix(UnaryOp::Neg, allowed),
            kind if is_name(kind, "not") => self.prefix(UnaryOp::Not, allowed),
            _ => self.level4(),
        }
    }

    fn skip_unary_plus(&mut self) {
        while *self.kind(0) == TokenKind::Plus {
            let span = self.span_at(0);
            let message = "unary `+` is not allowed; delete it".to_string();
            self.report(UNARY_PLUS, span, message, Some(Self::delete_fix(span)));
            self.advance();
        }
    }

    /// A prefix operator at the current token with its level-4 operand.
    fn prefix(&mut self, op: UnaryOp, allowed: bool) -> Parsed {
        let op_span = self.span_at(0);
        self.advance();
        let mut operand = self.level4();
        if operand.height >= MAX_DEPTH {
            self.too_deep(op_span);
            operand.recovered = true;
            return operand;
        }
        let span = op_span.to(operand.outer);
        if !allowed {
            let word = if op == UnaryOp::Neg { "-" } else { "not" };
            let message = format!(
                "prefix `{word}` is allowed only at the start of an expression or after \
                 `(` or `,`; write `({})`",
                self.quote(span)
            );
            let fix = self.wrap_fix(span);
            self.report(MISPLACED_PREFIX, span, message, Some(fix));
        }
        if op == UnaryOp::Neg && allowed && !operand.recovered {
            self.warn_precedence(span, |o| o.is_comparison(), &operand);
        }
        let height = operand.height + 1;
        let recovered = operand.recovered || !allowed;
        let expr = Expr::Unary {
            op,
            operand: operand.expr,
            span,
        };
        Parsed::new(expr, height, recovered)
    }

    /// Level 4: comparisons, with recovery for `!=`, `=` and split tokens.
    fn level4(&mut self) -> Parsed {
        let mut lhs = self.level5();
        loop {
            let span = self.span_at(0);
            let split = matches!(self.kind(1), TokenKind::Eq) && self.blank_at(1);
            let (op, count, recovered) = match self.kind(0) {
                TokenKind::Lt if split => (BinaryOp::Le, 2, true),
                TokenKind::Gt if split => (BinaryOp::Ge, 2, true),
                TokenKind::Eq if split => (BinaryOp::Eq, 2, true),
                TokenKind::Lt => (BinaryOp::Lt, 1, false),
                TokenKind::Gt => (BinaryOp::Gt, 1, false),
                TokenKind::Le => (BinaryOp::Le, 1, false),
                TokenKind::Ge => (BinaryOp::Ge, 1, false),
                TokenKind::EqEq => (BinaryOp::Eq, 1, false),
                TokenKind::Eq => {
                    let message = "`=` cannot compare values; write `==`".to_string();
                    let fix = FixIt {
                        span,
                        replacement: "==".to_string(),
                    };
                    self.report(EQUALS_IN_EXPRESSION, span, message, Some(fix));
                    (BinaryOp::Eq, 1, true)
                }
                TokenKind::Bang if *self.kind(1) == TokenKind::Eq && !self.blank_at(1) => {
                    self.report_bang_equals();
                    (BinaryOp::Eq, 2, true)
                }
                _ => return lhs,
            };
            if count == 2 && split {
                self.report_split(1);
            }
            for _ in 0..count {
                self.advance();
            }
            let rhs = self.level5();
            lhs = self.binary(op, span, lhs, rhs, false, recovered);
        }
    }

    fn report_bang_equals(&mut self) {
        let span = self.span_at(0).to(self.span_at(1));
        let message = "`!=` is not an operator; write not(a==b) instead of a!=b".to_string();
        self.report(BANG_OPERATOR, span, message, None);
    }

    /// Reports the blank between token `k - 1` and token `k`, which XPPAUT
    /// would delete to form one operator.
    fn report_split(&mut self, k: usize) {
        let gap = self.gap_before(k);
        let joined = format!(
            "{}{}",
            self.quote(self.span_at(k - 1)),
            self.quote(self.span_at(k))
        );
        let message = format!("XPPAUT deletes this blank and reads `{joined}`; delete it");
        self.report(BLANK_IN_TOKEN, gap, message, Some(Self::delete_fix(gap)));
    }

    /// Level 5: `^` and `**`, with recovery for `* *`. A chain of three or
    /// more operands gets one precedence warning for the whole chain.
    fn level5(&mut self) -> Parsed {
        let mut lhs = self.atom();
        // The outer spans of the chain's operands, once there are two.
        let mut operands: Vec<Span> = Vec::new();
        loop {
            let op_span = self.span_at(0);
            let (op, recovered) = match self.kind(0) {
                TokenKind::Caret => (BinaryOp::Pow, false),
                TokenKind::StarStar => (BinaryOp::StarStar, false),
                TokenKind::Star if *self.kind(1) == TokenKind::Star && self.blank_at(1) => {
                    self.report_split(1);
                    self.advance();
                    (BinaryOp::StarStar, true)
                }
                _ => break,
            };
            self.advance();
            let rhs = self.atom();
            if operands.is_empty() {
                operands.push(lhs.outer);
            }
            operands.push(rhs.outer);
            lhs = self.binary(op, op_span, lhs, rhs, false, recovered);
        }
        if operands.len() >= 3 && !lhs.recovered {
            self.warn_power_chain(&operands);
        }
        lhs
    }

    /// Warns that a chain of `^` and `**` groups to the left. The fix-it
    /// puts parentheses around every left part but the whole chain, so
    /// `a^b^c^d` becomes `((a^b)^c)^d`.
    fn warn_power_chain(&mut self, operands: &[Span]) {
        let n = operands.len();
        let text = |a: usize, b: usize| self.c.text.get(a..b).unwrap_or("");
        let mut inner = "(".repeat(n - 2);
        inner.push_str(self.src(operands[0]));
        for i in 1..n - 1 {
            inner.push_str(text(operands[i - 1].end, operands[i].end));
            inner.push(')');
        }
        let rewritten = format!("{inner}{}", text(operands[n - 2].end, operands[n - 1].end));
        let message = format!(
            "`^` and `**` group to the left, so this reads as `{}`; add parentheses to \
             say so",
            clip(&rewritten)
        );
        let whole = operands[0].to(operands[n - 1]);
        let fix = FixIt {
            span: operands[0].to(operands[n - 2]),
            replacement: inner,
        };
        self.report(PRECEDENCE, whole, message, Some(fix));
    }

    /// Builds a binary node and gives the precedence warnings for it. A node
    /// that would be deeper than [`MAX_DEPTH`] is not built: the depth error
    /// is reported at `op_span`, the rest of the expression is skipped and
    /// `lhs` is returned. `recovered` marks a node built by error recovery.
    fn binary(
        &mut self,
        op: BinaryOp,
        op_span: Span,
        lhs: Parsed,
        rhs: Parsed,
        implicit: bool,
        recovered: bool,
    ) -> Parsed {
        let height = 1 + lhs.height.max(rhs.height);
        if height > MAX_DEPTH {
            self.too_deep(op_span);
            let mut lhs = lhs;
            lhs.recovered = true;
            return lhs;
        }
        let span = lhs.outer.to(rhs.outer);
        let recovered = recovered || lhs.recovered || rhs.recovered;
        if !recovered {
            let child_warns = |child: BinaryOp| {
                (op.is_arithmetic() && (child.is_comparison() || child.is_logical()))
                    || (op.is_logical() && child.is_arithmetic())
            };
            self.warn_precedence(span, child_warns, &lhs);
            self.warn_precedence(span, child_warns, &rhs);
        }
        let expr = Expr::Binary {
            op,
            lhs: lhs.expr,
            rhs: rhs.expr,
            implicit,
            span,
        };
        Parsed::new(expr, height, recovered)
    }

    /// Warns when `child` is an unparenthesized binary node whose operator
    /// `applies` accepts. `whole` is the span of the parent node. The
    /// message quotes only the parent's operator and operands.
    fn warn_precedence(&mut self, whole: Span, applies: impl Fn(BinaryOp) -> bool, child: &Parsed) {
        let Expr::Binary { op, .. } = &*child.expr else {
            return;
        };
        if child.parens || child.recovered || !applies(*op) {
            return;
        }
        let inner = child.outer;
        let message = format!(
            "this reads as `{}({}){}`, unlike C; add parentheses to say so",
            self.quote_range(whole.start, inner.start),
            self.quote(inner),
            self.quote_range(inner.end, whole.end)
        );
        let fix = self.wrap_fix(inner);
        self.report(PRECEDENCE, whole, message, Some(fix));
    }
}

impl Parser<'_, '_> {
    /// Level 6: number, name, call, `( e )`, `if`, bare index form. Also
    /// recovers from prefix operators, unary `+` and `!` in this position.
    fn atom(&mut self) -> Parsed {
        if self.depth >= MAX_DEPTH {
            let span = self.span_at(0);
            let point = self.point_at(0);
            self.too_deep(span);
            return Parsed::error(point);
        }
        self.depth += 1;
        let parsed = self.atom_inner();
        self.depth -= 1;
        parsed
    }

    /// Dispatches on the current token. The functions on the recursive path
    /// (this one, `parens`, `call` and the levels) keep their frames small,
    /// with the diagnostics in separate functions, so that [`MAX_DEPTH`]
    /// levels of nesting fit in a 2 MB stack in a debug build.
    fn atom_inner(&mut self) -> Parsed {
        match self.kind(0) {
            TokenKind::Num(value) => {
                let value = *value;
                self.number(value)
            }
            TokenKind::Name(key) if key == "not" => self.prefix(UnaryOp::Not, false),
            TokenKind::Name(key) if key == "if" => self.if_form(),
            TokenKind::Name(_) => self.name_or_call(),
            TokenKind::LParen => self.parens(),
            TokenKind::LBracket => self.bracket_atom(),
            TokenKind::Minus => self.prefix(UnaryOp::Neg, false),
            TokenKind::Plus => {
                self.skip_unary_plus();
                self.atom()
            }
            TokenKind::Bang => {
                self.report_bang_atom();
                let mut parsed = self.atom();
                parsed.recovered = true;
                parsed
            }
            _ => self.no_operand(),
        }
    }

    fn number(&mut self, value: f64) -> Parsed {
        let span = self.span_at(0);
        self.advance();
        let text = self.src(span).to_string();
        Parsed::leaf(Expr::Num { value, text, span })
    }

    /// An index form in operand position. Only the `[j]` forms are values;
    /// a range must follow a name.
    fn bracket_atom(&mut self) -> Parsed {
        let span = self.span_at(0);
        match self.index() {
            Some(index @ Index::Range { .. }) => {
                let range = index.span();
                let message = format!(
                    "a range can only follow a name, as in `x{}`",
                    self.quote(range)
                );
                self.report(UNSUPPORTED_INDEX, range, message, None);
                Parsed::error(range)
            }
            Some(index) => Parsed::leaf(Expr::Index(index)),
            None => Parsed::error(span.to(self.prev_end())),
        }
    }

    /// Reports `!` or `!=` in operand position and consumes it.
    fn report_bang_atom(&mut self) {
        if *self.kind(1) == TokenKind::Eq && !self.blank_at(1) {
            self.report_bang_equals();
            self.advance();
        } else {
            let span = self.span_at(0);
            let message = "`!` is not an operator; use `not(...)` for negation".to_string();
            self.report(BANG_OPERATOR, span, message, None);
        }
        self.advance();
    }

    /// A token that cannot start an operand. A token the lexer already
    /// reported is used silently; anything else is reported.
    fn no_operand(&mut self) -> Parsed {
        let span = self.span_at(0);
        if *self.kind(0) == TokenKind::Unknown {
            self.advance();
            return Parsed::error(span);
        }
        let message = format!("expected an expression, found {}", self.describe(0));
        self.report(EXPECTED_EXPRESSION, span, message, None);
        let consumable = matches!(
            self.kind(0),
            TokenKind::Quote | TokenKind::Percent | TokenKind::DotDot | TokenKind::At
        );
        if consumable {
            self.advance();
            Parsed::error(span)
        } else {
            Parsed::error(self.point_at(0))
        }
    }

    /// A name with an optional index form, and a call when `(` follows.
    fn name_or_call(&mut self) -> Parsed {
        let (ident, recovered) = self.name();
        if *self.kind(0) != TokenKind::LParen {
            return Parsed::new(Expr::Name(*ident), 1, recovered);
        }
        self.call(ident, recovered)
    }

    /// The name at the current token with an optional index form, and
    /// whether a bad index form was skipped after it.
    fn name(&mut self) -> (Box<Ident>, bool) {
        let span = self.span_at(0);
        let key = match self.kind(0) {
            TokenKind::Name(key) => key.clone(),
            _ => String::new(),
        };
        self.advance();
        let mut ident = Box::new(Ident {
            text: self.src(span).to_string(),
            key,
            index: None,
            span,
        });
        let mut recovered = false;
        if *self.kind(0) == TokenKind::LBracket {
            match self.index() {
                Some(index) => {
                    ident.span = span.to(index.span());
                    ident.index = Some(index);
                }
                None => recovered = true,
            }
        }
        (ident, recovered)
    }

    /// The arguments of a call to `ident`, starting at `(`.
    fn call(&mut self, ident: Box<Ident>, mut recovered: bool) -> Parsed {
        let open = self.span_at(0);
        self.advance();
        let mut args = Vec::new();
        let mut height = 0;
        if *self.kind(0) == TokenKind::RParen {
            self.advance();
        } else {
            loop {
                let arg = self.full();
                height = height.max(arg.height);
                recovered |= arg.recovered;
                args.push(arg.expr);
                match self.kind(0) {
                    TokenKind::Comma => self.advance(),
                    TokenKind::RParen => {
                        self.advance();
                        break;
                    }
                    _ => {
                        self.report_unclosed(open, "(");
                        recovered = true;
                        break;
                    }
                }
            }
        }
        self.finish_call(ident, open, args, height, recovered)
    }

    /// Builds the call node, or reports that it would be too deep.
    #[expect(
        clippy::boxed_local,
        clippy::vec_box,
        reason = "the boxes keep the recursive frames small: 200 nested \
                  calls need about 800 KB of stack boxed and 1 MB unboxed"
    )]
    fn finish_call(
        &mut self,
        ident: Box<Ident>,
        open: Span,
        args: Vec<Box<Expr>>,
        height: usize,
        recovered: bool,
    ) -> Parsed {
        let span = ident.span.to(self.prev_end());
        if height >= MAX_DEPTH {
            self.too_deep(open);
            return Parsed::error(span);
        }
        let expr = Expr::Call {
            name: *ident,
            args: args.into_iter().map(|a| *a).collect(),
            span,
        };
        Parsed::new(expr, height + 1, recovered)
    }

    fn report_unclosed(&mut self, open: Span, what: &str) {
        let message = format!("this `{what}` is not closed");
        self.report(UNCLOSED_DELIMITER, open, message, None);
    }

    /// `( e )`. A comma inside is reported once and skipped with the rest.
    fn parens(&mut self) -> Parsed {
        let open = self.span_at(0);
        self.advance();
        let inner = self.full();
        let mut recovered = inner.recovered;
        let mut reported = false;
        while *self.kind(0) == TokenKind::Comma {
            if !reported {
                let span = self.span_at(0);
                let message = "unexpected `,`: parentheses hold one expression".to_string();
                self.report(UNEXPECTED_TOKEN, span, message, None);
                reported = true;
                recovered = true;
            }
            self.advance();
            self.skip_to_stop();
        }
        if *self.kind(0) == TokenKind::RParen {
            self.advance();
        } else {
            self.report_unclosed(open, "(");
            recovered = true;
        }
        Parsed {
            expr: inner.expr,
            parens: true,
            outer: open.to(self.prev_end()),
            height: inner.height,
            recovered,
        }
    }

    /// `if(c)then(a)else(b)`, with blanks allowed around the words.
    fn if_form(&mut self) -> Parsed {
        let start = self.span_at(0);
        self.advance();
        let mut parts: Vec<Parsed> = Vec::new();
        for word in ["if", "then", "else"] {
            if word != "if" {
                if is_name(self.kind(0), word) {
                    self.advance();
                } else {
                    return self.bad_if(start, parts, &format!("expected `{word}(...)`"));
                }
            }
            if *self.kind(0) != TokenKind::LParen {
                return self.bad_if(start, parts, &format!("expected `(` after `{word}`"));
            }
            parts.push(self.parens());
        }
        self.make_if(start, parts, false)
    }

    /// Builds the `if` node from the `if` keyword span and its three parts.
    fn make_if(&mut self, start: Span, parts: Vec<Parsed>, recovered: bool) -> Parsed {
        let span = start.to(self.prev_end());
        let height = parts.iter().map(|p| p.height).max().unwrap_or(0);
        let recovered = recovered || parts.iter().any(|p| p.recovered);
        if height >= MAX_DEPTH {
            self.too_deep(start);
            return Parsed::error(span);
        }
        let mut parts = parts.into_iter().map(|p| p.expr);
        let (Some(cond), Some(then), Some(else_)) = (parts.next(), parts.next(), parts.next())
        else {
            return Parsed::error(span);
        };
        let expr = Expr::If {
            cond,
            then,
            else_,
            span,
        };
        Parsed::new(expr, height + 1, recovered)
    }

    /// Reports a malformed `if` at the current token and fills the missing
    /// parts with error nodes. When the current token can follow an operand
    /// (an operator or a stop token), parsing goes on from it; otherwise the
    /// rest of the expression is skipped.
    fn bad_if(&mut self, start: Span, mut parts: Vec<Parsed>, what: &str) -> Parsed {
        let span = self.span_at(0);
        let message = format!(
            "{what}, found {}: write if(condition)then(value)else(value)",
            self.describe(0)
        );
        self.report(MALFORMED_IF, span, message, None);
        let missing = self.point_at(0);
        if !can_follow_operand(self.kind(0)) {
            self.skip_to_stop();
        }
        while parts.len() < 3 {
            parts.push(Parsed::error(missing));
        }
        self.make_if(start, parts, true)
    }

    /// An index form starting at `[`. Returns `None` after reporting an
    /// unsupported or unclosed form, with the tokens up to the matching `]`
    /// (or the next stop token) skipped.
    fn index(&mut self) -> Option<Index> {
        let open = self.span_at(0);
        if let Some((index, count)) = self.match_index(open) {
            for _ in 0..count {
                self.advance();
            }
            return Some(index);
        }
        // Look for the matching `]` on this line.
        let mut depth = 0usize;
        let mut k = 1;
        let close = loop {
            match self.kind(k) {
                kind if ends_at_any_depth(kind) => break None,
                TokenKind::LBracket => depth += 1,
                TokenKind::RBracket if depth == 0 => break Some(k),
                TokenKind::RBracket => depth -= 1,
                _ => {}
            }
            k += 1;
        };
        match close {
            Some(k) => {
                let span = open.to(self.span_at(k));
                let message = format!(
                    "unsupported index form `{}`: use [i1..i2], [j], [j+n], [j-n] or [j*n] \
                     with non-negative integers",
                    self.quote(span)
                );
                self.report(UNSUPPORTED_INDEX, span, message, None);
                for _ in 0..=k {
                    self.advance();
                }
            }
            None => {
                self.report_unclosed(open, "[");
                self.advance();
                self.skip_to_stop();
            }
        }
        None
    }

    /// Matches a supported index form at the current `[` and returns it with
    /// the number of tokens it covers.
    fn match_index(&self, open: Span) -> Option<(Index, usize)> {
        use TokenKind::*;
        let int = |k: usize| -> Option<u32> {
            match self.kind(k) {
                Num(_) => {
                    let text = self.src(self.span_at(k));
                    if text.bytes().all(|b| b.is_ascii_digit()) {
                        text.parse().ok()
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };
        let close = |k: usize| -> Option<Span> {
            (*self.kind(k) == RBracket).then(|| open.to(self.span_at(k)))
        };
        if let Some(start) = int(1) {
            if *self.kind(2) != DotDot {
                return None;
            }
            let end = int(3)?;
            let span = close(4)?;
            return Some((Index::Range { start, end, span }, 5));
        }
        if !is_name(self.kind(1), "j") {
            return None;
        }
        if let Some(span) = close(2) {
            let index = Index::Var {
                op: IndexOp::None,
                n: 0,
                span,
            };
            return Some((index, 3));
        }
        let op = match self.kind(2) {
            Plus => IndexOp::Add,
            Minus => IndexOp::Sub,
            Star => IndexOp::Mul,
            _ => return None,
        };
        let n = int(3)?;
        let span = close(4)?;
        Some((Index::Var { op, n, span }, 5))
    }
}

#[cfg(test)]
mod tests;
