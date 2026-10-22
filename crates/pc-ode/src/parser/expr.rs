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
//! as silent placeholders.
//!
//! Parentheses are tracked while parsing, so that the precedence warnings
//! leave parenthesized operands alone, but they are not kept in the tree.

use crate::ast::{BinaryOp, Expr, Ident, Index, IndexOp, UnaryOp};
use crate::diag::{CODES, Code, Diagnostic, FixIt, Severity};
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
/// An index form other than `[i1..i2]`, `[j]`, `[j+n]`, `[j-n]`, `[j*n]`.
pub const UNSUPPORTED_INDEX: Code = Code(15);
/// An `if` that is not `if(c)then(a)else(b)`.
pub const MALFORMED_IF: Code = Code(16);

/// How deeply atoms may nest (parentheses, calls, chains of prefix
/// operators) before the parser gives up on the rest of the expression.
/// This keeps pathological input from overflowing the stack.
const MAX_DEPTH: usize = 100;

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
/// or at nesting depth 0 `Comma`, `RParen`, `RBracket`, `RBrace` or `LBrace`.
/// The stop token is not consumed. Other stray tokens are reported and
/// skipped up to the next stop token.
///
/// The first token counts as the start of an expression, so a prefix `-` or
/// `not` is allowed there.
pub(crate) fn parse_expr(c: &mut ExprCursor) -> Expr {
    if c.tokens.is_empty() {
        return Expr::Error {
            span: Span::new(FileId(0), 0, 0),
        };
    }
    if c.pos >= c.tokens.len() {
        c.pos = c.tokens.len() - 1;
    }
    let mut p = Parser {
        start: c.pos,
        c,
        depth: 0,
    };
    p.full().expr
}

/// Lexes `text` and parses it as one expression. Lexer diagnostics come
/// first. A token left over after the expression, before the end of the
/// text, is reported as unexpected.
pub fn parse_expr_text(file: FileId, text: &str) -> (Expr, Vec<Diagnostic>) {
    let (tokens, mut diags) = lex(file, text);
    let mut c = ExprCursor {
        tokens: &tokens,
        pos: 0,
        text,
        diags: &mut diags,
    };
    let e = parse_expr(&mut c);
    let mut p = Parser {
        start: c.pos,
        c: &mut c,
        depth: 0,
    };
    if p.kind(0) != &TokenKind::Eof {
        let span = p.span_at(0);
        let message = format!("unexpected {} after the expression", p.describe(0));
        p.report(UNEXPECTED_TOKEN, span, message, None);
    }
    (e, diags)
}

/// A parsed operand plus what the precedence warnings need to know about it.
struct Parsed {
    expr: Expr,
    /// The operand was written in parentheses.
    parens: bool,
    /// The span including those parentheses.
    outer: Span,
}

impl Parsed {
    fn plain(expr: Expr) -> Parsed {
        let outer = expr.span();
        Parsed {
            expr,
            parens: false,
            outer,
        }
    }
}

struct Parser<'p, 'a> {
    c: &'p mut ExprCursor<'a>,
    /// Token index where the expression began, a prefix position.
    start: usize,
    /// Current nesting of atoms, checked against [`MAX_DEPTH`].
    depth: usize,
}

static EOF: TokenKind = TokenKind::Eof;

fn severity_of(code: Code) -> Severity {
    CODES
        .iter()
        .find(|(c, _, _)| *c == code)
        .map_or(Severity::Error, |e| e.1)
}

/// Tokens that end an expression at nesting depth 0.
fn is_stop(kind: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(
        kind,
        Newline | Eof | Comment | Comma | RParen | RBracket | RBrace | LBrace
    )
}

/// Tokens that end an expression at any depth.
fn is_line_end(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Newline | TokenKind::Eof | TokenKind::Comment
    )
}

/// Tokens that can start an operand, for the juxtaposition rules.
fn starts_operand(kind: &TokenKind) -> bool {
    use TokenKind::*;
    matches!(kind, Num(_) | Name(_) | LParen | LBracket | Unknown)
}

fn is_name(kind: &TokenKind, key: &str) -> bool {
    matches!(kind, TokenKind::Name(k) if k == key)
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

    fn describe(&self, k: usize) -> String {
        match self.kind(k) {
            TokenKind::Newline => "line break".to_string(),
            TokenKind::Eof => "end of the text".to_string(),
            TokenKind::Comment => "comment".to_string(),
            _ => format!("`{}`", self.src(self.span_at(k))),
        }
    }

    fn report(&mut self, code: Code, span: Span, message: String, fix_it: Option<FixIt>) {
        self.c.diags.push(Diagnostic {
            code,
            severity: severity_of(code),
            span,
            message,
            fix_it,
            notes: Vec::new(),
        });
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
            if is_line_end(kind) || (depth == 0 && is_stop(kind)) {
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
        let parsed = self.level1();
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
            self.advance();
            let rhs = self.level2();
            lhs = self.binary(op, lhs, rhs, false);
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
            self.advance();
            let rhs = self.unary();
            lhs = self.binary(op, lhs, rhs, false);
            after_div = op == BinaryOp::Div;
        }
    }

    /// An operand directly after another operand. Valid only as implicit
    /// multiplication (a number followed directly by a name or `(`); the
    /// other cases are errors that recover as a multiplication.
    fn juxtaposition(&mut self, lhs: Parsed, after_div: bool) -> Parsed {
        let left = self.prev().map(|t| (t.kind.clone(), t.span));
        let right = self.kind(0).clone();
        let right_span = self.span_at(0);
        let blank = self.blank_at(0);
        let gap = self.gap_before(0);
        let rhs = self.unary();
        let left_kind = left.as_ref().map(|(k, _)| k);
        let silent = lhs.expr.is_error()
            || rhs.expr.is_error()
            || matches!(left_kind, Some(TokenKind::Unknown))
            || right == TokenKind::Unknown;
        if !silent {
            let numeric_left = matches!(left_kind, Some(TokenKind::Num(_)));
            let word_left = match left_kind {
                Some(TokenKind::Num(_)) => true,
                Some(TokenKind::Name(k)) => !matches!(k.as_str(), "not" | "if" | "then" | "else"),
                _ => false,
            };
            let word_right = matches!(right, TokenKind::Num(_) | TokenKind::Name(_));
            let insert_star = FixIt {
                span: Span::new(right_span.file, right_span.start, right_span.start),
                replacement: "*".to_string(),
            };
            if numeric_left && !blank && matches!(right, TokenKind::Name(_) | TokenKind::LParen) {
                let l = self.src(lhs.outer).to_string();
                let r = self.src(rhs.outer).to_string();
                let mut message = format!(
                    "XPPAUT 8.0 rejects implicit multiplication; write `{l}*{r}` instead of `{l}{r}`"
                );
                if after_div {
                    message.push_str(&format!("; note that `{l}{r}` reads as `({l})*{r}`"));
                }
                let span = lhs.outer.to(rhs.outer);
                self.report(PORTABILITY, span, message, Some(insert_star));
            } else if word_left && word_right && blank {
                let left_span = left.as_ref().map_or(gap, |(_, s)| *s);
                let message = format!(
                    "XPPAUT deletes this blank and joins `{}` and `{}` into one word; \
                     delete the blank or add an operator",
                    self.src(left_span),
                    self.src(right_span)
                );
                self.report(BLANK_IN_TOKEN, gap, message, Some(Self::delete_fix(gap)));
            } else {
                let message = format!(
                    "missing operator before `{}`; write `*` to multiply",
                    self.src(right_span)
                );
                self.report(MISSING_OPERATOR, right_span, message, Some(insert_star));
            }
        }
        self.binary(BinaryOp::Mul, lhs, rhs, true)
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
        let operand = self.level4();
        let span = op_span.to(operand.outer);
        if !allowed {
            let word = if op == UnaryOp::Neg { "-" } else { "not" };
            let message = format!(
                "prefix `{word}` is allowed only at the start of an expression or after \
                 `(` or `,`; write `({})`",
                self.src(span)
            );
            let fix = self.wrap_fix(span);
            self.report(MISPLACED_PREFIX, span, message, Some(fix));
        }
        if op == UnaryOp::Neg && !operand.expr.is_error() {
            self.warn_precedence(span, |o| o.is_comparison(), &operand);
        }
        Parsed::plain(Expr::Unary {
            op,
            operand: Box::new(operand.expr),
            span,
        })
    }

    /// Level 4: comparisons, with recovery for `!=`, `=` and split tokens.
    fn level4(&mut self) -> Parsed {
        let mut lhs = self.level5();
        loop {
            let span = self.span_at(0);
            let split = matches!(self.kind(1), TokenKind::Eq) && self.blank_at(1);
            let (op, count) = match self.kind(0) {
                TokenKind::Lt if split => (BinaryOp::Le, 2),
                TokenKind::Gt if split => (BinaryOp::Ge, 2),
                TokenKind::Eq if split => (BinaryOp::Eq, 2),
                TokenKind::Lt => (BinaryOp::Lt, 1),
                TokenKind::Gt => (BinaryOp::Gt, 1),
                TokenKind::Le => (BinaryOp::Le, 1),
                TokenKind::Ge => (BinaryOp::Ge, 1),
                TokenKind::EqEq => (BinaryOp::Eq, 1),
                TokenKind::Eq => {
                    let message = "`=` cannot compare values; write `==`".to_string();
                    let fix = FixIt {
                        span,
                        replacement: "==".to_string(),
                    };
                    self.report(EQUALS_IN_EXPRESSION, span, message, Some(fix));
                    (BinaryOp::Eq, 1)
                }
                TokenKind::Bang if *self.kind(1) == TokenKind::Eq && !self.blank_at(1) => {
                    self.report_bang_equals();
                    (BinaryOp::Eq, 2)
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
            lhs = self.binary(op, lhs, rhs, false);
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
            self.src(self.span_at(k - 1)),
            self.src(self.span_at(k))
        );
        let message = format!("XPPAUT deletes this blank and reads `{joined}`; delete it");
        self.report(BLANK_IN_TOKEN, gap, message, Some(Self::delete_fix(gap)));
    }

    /// Level 5: `^` and `**`, with recovery for `* *`.
    fn level5(&mut self) -> Parsed {
        let mut lhs = self.atom();
        loop {
            let op = match self.kind(0) {
                TokenKind::Caret => BinaryOp::Pow,
                TokenKind::StarStar => BinaryOp::StarStar,
                TokenKind::Star if *self.kind(1) == TokenKind::Star && self.blank_at(1) => {
                    self.report_split(1);
                    self.advance();
                    BinaryOp::StarStar
                }
                _ => return lhs,
            };
            self.advance();
            let rhs = self.atom();
            lhs = self.binary(op, lhs, rhs, false);
        }
    }

    /// Builds a binary node and gives the precedence warnings for it.
    fn binary(&mut self, op: BinaryOp, lhs: Parsed, rhs: Parsed, implicit: bool) -> Parsed {
        let span = lhs.outer.to(rhs.outer);
        if !lhs.expr.is_error() && !rhs.expr.is_error() {
            let child_warns = |child: BinaryOp, is_lhs: bool| {
                (op.is_arithmetic() && child.is_comparison())
                    || (op.is_arithmetic() && child.is_logical())
                    || (op.is_logical() && child.is_arithmetic())
                    || (op.is_power() && child.is_power() && is_lhs)
            };
            self.warn_precedence(span, |c| child_warns(c, true), &lhs);
            self.warn_precedence(span, |c| child_warns(c, false), &rhs);
        }
        Parsed::plain(Expr::Binary {
            op,
            lhs: Box::new(lhs.expr),
            rhs: Box::new(rhs.expr),
            implicit,
            span,
        })
    }

    /// Warns when `child` is an unparenthesized binary node whose operator
    /// `applies` accepts. `whole` is the span of the parent node.
    fn warn_precedence(&mut self, whole: Span, applies: impl Fn(BinaryOp) -> bool, child: &Parsed) {
        let Expr::Binary { op, .. } = &child.expr else {
            return;
        };
        if child.parens || !applies(*op) {
            return;
        }
        let inner = child.outer;
        let before = self.c.text.get(whole.start..inner.start).unwrap_or("");
        let after = self.c.text.get(inner.end..whole.end).unwrap_or("");
        let message = format!(
            "XPPAUT reads `{}` as `{before}({}){after}`; add parentheses to say so",
            self.src(whole),
            self.src(inner)
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
            let message = "expression nested too deeply".to_string();
            self.report(UNEXPECTED_TOKEN, span, message, None);
            self.skip_to_stop();
            return Parsed::plain(Expr::Error {
                span: self.point_at(0),
            });
        }
        self.depth += 1;
        let parsed = self.atom_inner();
        self.depth -= 1;
        parsed
    }

    fn atom_inner(&mut self) -> Parsed {
        let span = self.span_at(0);
        match self.kind(0).clone() {
            TokenKind::Num(value) => {
                self.advance();
                let text = self.src(span).to_string();
                Parsed::plain(Expr::Num { value, text, span })
            }
            TokenKind::Name(key) if key == "not" => self.prefix(UnaryOp::Not, false),
            TokenKind::Name(key) if key == "if" => self.if_form(),
            TokenKind::Name(key) => self.name_or_call(key),
            TokenKind::LParen => self.parens(),
            TokenKind::LBracket => {
                let expr = match self.index() {
                    Some(index) => Expr::Index(index),
                    None => Expr::Error {
                        span: span.to(self.prev_end()),
                    },
                };
                Parsed::plain(expr)
            }
            TokenKind::Unknown => {
                self.advance();
                Parsed::plain(Expr::Error { span })
            }
            TokenKind::Minus => self.prefix(UnaryOp::Neg, false),
            TokenKind::Plus => {
                self.skip_unary_plus();
                self.atom()
            }
            TokenKind::Bang => {
                if *self.kind(1) == TokenKind::Eq && !self.blank_at(1) {
                    self.report_bang_equals();
                    self.advance();
                } else {
                    let message = "`!` is not an operator; use `not(...)` for negation".to_string();
                    self.report(BANG_OPERATOR, span, message, None);
                }
                self.advance();
                self.atom()
            }
            kind => {
                let message = format!("expected an expression, found {}", self.describe(0));
                self.report(EXPECTED_EXPRESSION, span, message, None);
                let consumable = matches!(
                    kind,
                    TokenKind::Quote | TokenKind::Percent | TokenKind::DotDot | TokenKind::At
                );
                if consumable {
                    self.advance();
                    Parsed::plain(Expr::Error { span })
                } else {
                    Parsed::plain(Expr::Error {
                        span: self.point_at(0),
                    })
                }
            }
        }
    }

    /// A name with an optional index form, and a call when `(` follows.
    fn name_or_call(&mut self, key: String) -> Parsed {
        let span = self.span_at(0);
        self.advance();
        let mut ident = Ident {
            text: self.src(span).to_string(),
            key,
            index: None,
            span,
        };
        if *self.kind(0) == TokenKind::LBracket
            && let Some(index) = self.index()
        {
            ident.span = span.to(index.span());
            ident.index = Some(index);
        }
        if *self.kind(0) != TokenKind::LParen {
            return Parsed::plain(Expr::Name(ident));
        }
        let open = self.span_at(0);
        self.advance();
        let mut args = Vec::new();
        if *self.kind(0) == TokenKind::RParen {
            self.advance();
        } else {
            loop {
                args.push(self.full().expr);
                match self.kind(0) {
                    TokenKind::Comma => self.advance(),
                    TokenKind::RParen => {
                        self.advance();
                        break;
                    }
                    _ => {
                        self.report_unclosed(open, "(");
                        break;
                    }
                }
            }
        }
        let span = ident.span.to(self.prev_end());
        Parsed::plain(Expr::Call {
            name: ident,
            args,
            span,
        })
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
        let mut reported = false;
        while *self.kind(0) == TokenKind::Comma {
            if !reported {
                let span = self.span_at(0);
                let message = "unexpected `,`: parentheses hold one expression".to_string();
                self.report(UNEXPECTED_TOKEN, span, message, None);
                reported = true;
            }
            self.advance();
            self.skip_to_stop();
        }
        if *self.kind(0) == TokenKind::RParen {
            self.advance();
        } else {
            self.report_unclosed(open, "(");
        }
        Parsed {
            expr: inner.expr,
            parens: true,
            outer: open.to(self.prev_end()),
        }
    }

    /// `if(c)then(a)else(b)`, with blanks allowed around the words.
    fn if_form(&mut self) -> Parsed {
        let start = self.span_at(0);
        self.advance();
        let mut parts: Vec<Expr> = Vec::new();
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
            parts.push(self.parens().expr);
        }
        self.make_if(start, parts)
    }

    /// Builds the `if` node from the `if` keyword span and its three parts.
    fn make_if(&self, start: Span, parts: Vec<Expr>) -> Parsed {
        let span = start.to(self.prev_end());
        let mut parts = parts.into_iter().map(Box::new);
        let (Some(cond), Some(then), Some(else_)) = (parts.next(), parts.next(), parts.next())
        else {
            return Parsed::plain(Expr::Error { span });
        };
        Parsed::plain(Expr::If {
            cond,
            then,
            else_,
            span,
        })
    }

    /// Reports a malformed `if` at the current token, skips the rest of the
    /// expression and fills the missing parts with error nodes.
    fn bad_if(&mut self, start: Span, mut parts: Vec<Expr>, what: &str) -> Parsed {
        let span = self.span_at(0);
        let message = format!(
            "{what}, found {}: write if(condition)then(value)else(value)",
            self.describe(0)
        );
        self.report(MALFORMED_IF, span, message, None);
        let missing = self.point_at(0);
        self.skip_to_stop();
        while parts.len() < 3 {
            parts.push(Expr::Error { span: missing });
        }
        self.make_if(start, parts)
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
                kind if is_line_end(kind) => break None,
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
                    self.src(span)
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
