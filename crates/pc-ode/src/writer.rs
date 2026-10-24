//! Writes a [`Model`] back as `.ode` text.
//!
//! The writer gives one canonical text per model, so that
//! `parse(write(m))` gives `m` again (spans aside) for every model the
//! parser produces without an error.
//!
//! - One statement per line, each line ending in `\n`. Keywords are written
//!   in full and in lower case (`par`, `aux`, `wiener`, `done`).
//! - Names keep their spelling (`Ident::text`; the key when the text is
//!   empty). Number literals keep the text they were written with; a
//!   literal with empty text is written by [`format_number`].
//! - Expressions get parentheses where the measured precedence needs them
//!   (`docs/language.md`, "Precedence"), where the parser would give a
//!   precedence warning (a comparison or `&`/`|` inside arithmetic,
//!   arithmetic inside `&`/`|`, `-` of a comparison), around the inner
//!   comparison or power of a chain (`(a < b) < c`, `(a^b)^c`), and around a
//!   whole operand whose text starts with a prefix `-` or `not` when it is
//!   not at the start of an expression, after `(` or after `,`
//!   (`a + (-2x)`, `a*(-b)`). Binary `+`, `-` and the comparisons have a
//!   blank on each side; the other operators have none.
//! - Implicit multiplication is always written as juxtaposition, since the
//!   parser builds it only where the left operand's text ends with a number
//!   literal. When the right operand's text cannot follow the literal
//!   directly, its first operand goes in parentheses, one level more than
//!   the source may have had: a number (`2(3)`), a bare index (`2([j])`), a
//!   name that starts with `e` or `E` and a digit, which would lex as an
//!   exponent (`2(e1)`, `2(e1)^2`), and, after a literal made only of zeros,
//!   a name that starts with `x` or `X` and a hex digit, which would lex as
//!   a malformed hex number (`0(x1)`). The parentheses that would only avoid
//!   a precedence warning are left out of the left operand, because
//!   `(a<2)x` is not implicit: `a<2x` is written `a < 2x` and keeps the
//!   warning of the source. A hand-built product whose left operand does
//!   not end with a literal is written with `*` and loses the flag.
//! - `#include` lines, range lines, `%` blocks, comments and the trailing
//!   comments of `par`, `number` and `@` lines are kept. Unsupported
//!   statements are written as their raw text.
//! - A text whose first line would start with a digit (`0=x-y`) gets a
//!   blank first line, since that digit marks an old-format file.
//! - A line that would end in `\` outside a `#` comment (raw text, an `@`
//!   value) gets a blank after it, so that it does not continue the line.
//!
//! Limitations of the round trip:
//!
//! - A `BadBlock` is written as `%[]`, its body and `%`. The malformed
//!   `%[..]` line is not stored, so the text reparses as a `BadBlock` with
//!   an error, but not as the original line.
//! - `Expr::Error` and `Value::Error` are written as `?`, which the lexer
//!   rejects. A model that holds error nodes is not guaranteed to round-trip;
//!   the text only reparses with an error.
//! - A literal with empty text and a value that is not finite is written as
//!   `1e999`, which reparses with an error.
//! - A negative `Expr::Num` value (hand-built only; the parser never makes
//!   one) is written as `-2`, or `(-2)` where a prefix needs parentheses. It
//!   reparses without an error but as `Neg` of `2`. `Value::Num` keeps its
//!   sign.
//! - A comment or `#include` path that holds a `\r` (hand-built only; the
//!   parser never makes one) reparses as a shorter comment followed by a
//!   new line, since a `\r` ends a line.
//!
//! The writer ignores `Stmt::array_index`, `Stmt::array_template` and
//! `Model::suppressed`, which the parser never sets.

use crate::ast::{
    BinaryOp, Comment, Entry, Expr, Ident, Index, IndexOp, MarkovLine, Model, OdeForm, RawAssign,
    Stmt, StmtKind, UnaryOp, Value,
};

/// Writes `model` as `.ode` text in the canonical form described in the
/// module documentation. On a template model (from `parse`) this keeps
/// `#include` lines, range lines and `%` blocks.
pub fn write(model: &Model) -> String {
    let mut out = String::new();
    for s in &model.statements {
        stmt(&mut out, s);
    }
    // A first line that starts with a digit marks the old file format.
    if out
        .trim_start_matches([' ', '\t'])
        .starts_with(|c: char| c.is_ascii_digit())
    {
        out.insert(0, '\n');
    }
    out
}

/// Writes the flat form of a model: call it on `Loaded::expanded`, which
/// has no includes, ranges or blocks left. It is the same function as
/// [`write()`]; the input decides whether the text is flat.
pub fn write_expanded(model: &Model) -> String {
    write(model)
}

/// The canonical text of a number literal: the shortest digits that read
/// back as the same `f64`, in plain notation when the decimal exponent is
/// in -5..=15 and in `e` notation otherwise (`0.00001`, `1e-6`,
/// `1000000000000000`, `1e16`). A negative value gets a leading `-`, and so
/// does `-0.0` (`-0`). A value that is not finite gives `1e999`, which the
/// lexer rejects.
pub fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return "1e999".to_string();
    }
    if value.is_sign_negative() {
        return format!("-{}", format_number(-value));
    }
    if value == 0.0 {
        return "0".to_string();
    }
    // `{:e}` gives the shortest round-trip digits as `d.ddde<exp>`.
    let sci = format!("{value:e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    if !(-5..=15).contains(&exp) {
        return sci;
    }
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if exp >= 0 {
        let int_len = exp as usize + 1;
        if digits.len() <= int_len {
            format!("{digits}{}", "0".repeat(int_len - digits.len()))
        } else {
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        }
    } else {
        format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
    }
}

fn stmt(out: &mut String, s: &Stmt) {
    match &s.kind {
        StmtKind::Ode { name, form, rhs } => {
            if *form == OdeForm::Ddt && name.index.is_none() {
                out.push('d');
                out.push_str(&ident(name));
                out.push_str("/dt=");
            } else {
                out.push_str(&ident(name));
                out.push_str("'=");
            }
            out.push_str(&expr_text(rhs));
        }
        StmtKind::Map { name, rhs } => {
            out.push_str(&format!("{}(t+1)={}", ident(name), expr_text(rhs)));
        }
        StmtKind::InitValue { name, value: v } => {
            out.push_str(&format!("{}(0)={}", ident(name), value(v)));
        }
        StmtKind::Init { entries } => entry_list(out, "init", entries),
        StmtKind::Par { entries } => entry_list(out, "par", entries),
        StmtKind::Number { entries } => entry_list(out, "number", entries),
        StmtKind::Derived { name, rhs } => {
            out.push_str(&format!("!{}={}", ident(name), expr_text(rhs)));
        }
        StmtKind::Fixed { name, rhs } => {
            out.push_str(&format!("{}={}", ident(name), expr_text(rhs)));
        }
        StmtKind::Aux { name, rhs } => {
            out.push_str(&format!("aux {}={}", ident(name), expr_text(rhs)));
        }
        StmtKind::Function { name, params, rhs } => {
            let params: Vec<String> = params.iter().map(ident).collect();
            out.push_str(&format!(
                "{}({})={}",
                ident(name),
                params.join(","),
                expr_text(rhs)
            ));
        }
        StmtKind::Table {
            name,
            npts,
            lo,
            hi,
            rhs,
        } => {
            out.push_str(&format!(
                "table {} % {} {} {} {}",
                ident(name),
                value(npts),
                value(lo),
                value(hi),
                expr_text(rhs)
            ));
        }
        StmtKind::Options { entries } => {
            out.push_str("@ ");
            out.push_str(&raw_assigns(entries, ", "));
        }
        StmtKind::Include { path, .. } => {
            out.push_str("#include ");
            out.push_str(path);
        }
        StmtKind::Set { name, entries } => {
            out.push_str(&format!(
                "set {} {{{}}}",
                ident(name),
                raw_assigns(entries, ",")
            ));
        }
        StmtKind::Global {
            sign,
            cond,
            assigns,
        } => {
            let assigns: Vec<String> = assigns
                .iter()
                .map(|a| format!("{}={}", ident(&a.name), expr_text(&a.rhs)))
                .collect();
            out.push_str(&format!(
                "global {sign} {} {{{}}}",
                expr_text(cond),
                assigns.join(";")
            ));
        }
        StmtKind::Markov {
            name,
            states,
            lines,
        } => {
            out.push_str(&format!("markov {} {states}", ident(name)));
            for line in lines {
                out.push('\n');
                match line {
                    MarkovLine::Row(row) => {
                        for e in &row.entries {
                            out.push('{');
                            out.push_str(&expr_text(e));
                            out.push('}');
                        }
                        trailing(out, &row.comment);
                    }
                    MarkovLine::Comment(c) => out.push_str(&c.text),
                }
            }
        }
        StmtKind::Wiener { names } => {
            let names: Vec<String> = names.iter().map(ident).collect();
            out.push_str("wiener ");
            out.push_str(&names.join(","));
        }
        StmtKind::Block {
            start, end, body, ..
        } => {
            out.push_str(&format!("%[{start}..{end}]\n"));
            for b in body {
                stmt(out, b);
            }
            out.push('%');
        }
        StmtKind::BadBlock { body } => {
            out.push_str("%[]\n");
            for b in body {
                stmt(out, b);
            }
            out.push('%');
        }
        StmtKind::Comment(c) => out.push_str(&c.text),
        StmtKind::Unsupported { text, .. } => out.push_str(text),
        StmtKind::Done => out.push_str("done"),
    }
    let ends_in_comment = s.comment.is_some()
        || matches!(s.kind, StmtKind::Comment(_) | StmtKind::Include { .. })
        || matches!(&s.kind, StmtKind::Markov { lines, .. }
            if matches!(lines.last(), Some(MarkovLine::Comment(_))));
    trailing(out, &s.comment);
    // A backslash right before the newline would join the next line.
    if !ends_in_comment && out.ends_with('\\') {
        out.push(' ');
    }
    out.push('\n');
}

fn trailing(out: &mut String, comment: &Option<Comment>) {
    if let Some(c) = comment {
        out.push(' ');
        out.push_str(&c.text);
    }
}

fn entry_list(out: &mut String, keyword: &str, entries: &[Entry]) {
    let entries: Vec<String> = entries
        .iter()
        .map(|e| match &e.value {
            Some(v) => format!("{}={}", ident(&e.name), value(v)),
            None => ident(&e.name),
        })
        .collect();
    out.push_str(keyword);
    out.push(' ');
    out.push_str(&entries.join(","));
}

fn raw_assigns(entries: &[RawAssign], sep: &str) -> String {
    let entries: Vec<String> = entries
        .iter()
        .map(|a| format!("{}={}", ident(&a.name), a.value))
        .collect();
    entries.join(sep)
}

fn ident(id: &Ident) -> String {
    let mut s = if id.text.is_empty() {
        id.key.clone()
    } else {
        id.text.clone()
    };
    if let Some(ix) = &id.index {
        s.push_str(&index(ix));
    }
    s
}

fn index(ix: &Index) -> String {
    match ix {
        Index::Range { start, end, .. } => format!("[{start}..{end}]"),
        Index::Var { op, n, .. } => match op {
            IndexOp::None => "[j]".to_string(),
            IndexOp::Add => format!("[j+{n}]"),
            IndexOp::Sub => format!("[j-{n}]"),
            IndexOp::Mul => format!("[j*{n}]"),
        },
    }
}

fn literal(value: f64, text: &str) -> String {
    if text.is_empty() {
        format_number(value)
    } else {
        text.to_string()
    }
}

fn value(v: &Value) -> String {
    match v {
        Value::Num { value, text, .. } => literal(*value, text),
        Value::Index(ix) => index(ix),
        Value::Error { .. } => "?".to_string(),
    }
}

/// The precedence level of a binary operator (`docs/language.md`).
fn level(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Or => 1,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::And => 2,
        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge | BinaryOp::Eq => 4,
        BinaryOp::Pow | BinaryOp::StarStar => 5,
    }
}

/// The level of an expression as an operand: 3 for a prefix operator, 6
/// for an atom.
fn expr_level(e: &Expr) -> u8 {
    match e {
        Expr::Binary { op, .. } => level(*op),
        Expr::Unary { .. } => 3,
        _ => 6,
    }
}

/// Why an operand gets parentheses.
#[derive(Clone, Copy, PartialEq)]
enum Wrap {
    No,
    /// The text would parse differently, or with an error, without them.
    Needed,
    /// Only to avoid a precedence warning, or to make the grouping of a
    /// comparison or power chain visible. The tree is the same without them.
    Warning,
}

/// The text of an operand and what its neighbors need to know about it.
#[derive(Clone)]
struct Out {
    text: String,
    /// The text ends with a number literal that is not inside parentheses.
    /// The flag tells whether that literal is made only of zeros.
    tail_literal: Option<bool>,
    /// The text starts with a prefix `-` or `not` that is not inside
    /// parentheses.
    prefix: bool,
    /// The length of the first operand at the start of the text: a leaf, a
    /// call or a group in parentheses. The whole text for a prefix operator.
    head: usize,
}

impl Out {
    fn wrapped(self) -> Out {
        let text = format!("({})", self.text);
        Out {
            head: text.len(),
            text,
            tail_literal: None,
            prefix: false,
        }
    }

    fn wrap_if(self, yes: bool) -> Out {
        if yes { self.wrapped() } else { self }
    }
}

/// An operand written two ways: `normal` with every parenthesis of
/// [`Wrap`], and `relaxed` without the [`Wrap::Warning`] ones on its right
/// edge (the right operands of binary nodes and the operands of prefix
/// operators, down from the top). The relaxed form is used only for the
/// left operand of an implicit multiplication, where it lets the text end
/// with a literal.
struct Both {
    normal: Out,
    relaxed: Out,
}

/// Why `child`, written as `out`, needs parentheses as an operand of the
/// binary operator `parent`.
fn binary_child_wrap(parent: BinaryOp, child: &Expr, out: &Out, is_rhs: bool) -> Wrap {
    // A prefix operator is allowed only at the start of an expression, and
    // its operand is parsed at level 4, so it cannot start the left operand
    // of a comparison or power either.
    if out.prefix && (is_rhs || parent.is_comparison() || parent.is_power()) {
        return Wrap::Needed;
    }
    let Expr::Binary { op, .. } = child else {
        return Wrap::No;
    };
    let (p, c) = (level(parent), level(*op));
    if c < p || (c == p && is_rhs) {
        Wrap::Needed
    } else if (parent.is_comparison() && op.is_comparison())
        || (parent.is_power() && op.is_power())
        || (parent.is_arithmetic() && (op.is_comparison() || op.is_logical()))
        || (parent.is_logical() && op.is_arithmetic())
    {
        Wrap::Warning
    } else {
        Wrap::No
    }
}

/// Why the operand of a prefix operator needs parentheses.
fn unary_operand_wrap(op: UnaryOp, operand: &Expr, out: &Out) -> Wrap {
    if expr_level(operand) < 4 || out.prefix {
        Wrap::Needed
    } else if op == UnaryOp::Neg && matches!(operand, Expr::Binary { op, .. } if op.is_comparison())
    {
        Wrap::Warning
    } else {
        Wrap::No
    }
}

fn expr_text(e: &Expr) -> String {
    expr(e).normal.text
}

fn leaf(text: String) -> Both {
    let out = Out {
        head: text.len(),
        text,
        tail_literal: None,
        prefix: false,
    };
    Both {
        normal: out.clone(),
        relaxed: out,
    }
}

/// Writes `e` in both forms of [`Both`]. Every child is written once, so
/// the time stays linear in the size of the tree.
fn expr(e: &Expr) -> Both {
    match e {
        Expr::Num { value, text, .. } => {
            let text = literal(*value, text);
            let digits = text.strip_prefix('-').unwrap_or(&text);
            let zeros = !digits.is_empty() && digits.bytes().all(|b| b == b'0');
            let out = Out {
                prefix: text.starts_with('-'),
                head: text.len(),
                text,
                tail_literal: Some(zeros),
            };
            Both {
                normal: out.clone(),
                relaxed: out,
            }
        }
        Expr::Name(id) => leaf(ident(id)),
        Expr::Index(ix) => leaf(index(ix)),
        Expr::Error { .. } => leaf("?".to_string()),
        Expr::Unary { op, operand, .. } => {
            let inner = expr(operand);
            let prefix = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "not ",
            };
            let form = |o: Out, relaxed: bool| {
                let wrap = unary_operand_wrap(*op, operand, &o);
                let o = o.wrap_if(wrap == Wrap::Needed || (wrap == Wrap::Warning && !relaxed));
                let text = format!("{prefix}{}", o.text);
                Out {
                    head: text.len(),
                    text,
                    tail_literal: o.tail_literal,
                    prefix: true,
                }
            };
            Both {
                normal: form(inner.normal, false),
                relaxed: form(inner.relaxed, true),
            }
        }
        Expr::Binary {
            op,
            lhs,
            rhs,
            implicit,
            ..
        } => {
            let l = expr(lhs);
            let r = expr(rhs);
            let left = if *op == BinaryOp::Mul && *implicit {
                implicit_left(lhs, l)
            } else {
                let wrap = binary_child_wrap(*op, lhs, &l.normal, false);
                l.normal.wrap_if(wrap != Wrap::No)
            };
            let form = |o: Out, relaxed: bool| {
                let wrap = binary_child_wrap(*op, rhs, &o, true);
                let o = o.wrap_if(wrap == Wrap::Needed || (wrap == Wrap::Warning && !relaxed));
                binary_text(*op, *implicit, &left, o)
            };
            Both {
                normal: form(r.normal, false),
                relaxed: form(r.relaxed, true),
            }
        }
        Expr::Call { name, args, .. } => {
            let args: Vec<String> = args.iter().map(expr_text).collect();
            leaf(format!("{}({})", ident(name), args.join(",")))
        }
        Expr::If {
            cond, then, else_, ..
        } => leaf(format!(
            "if({})then({})else({})",
            expr_text(cond),
            expr_text(then),
            expr_text(else_)
        )),
    }
}

/// The left operand of an implicit multiplication. It is written with all
/// its parentheses when that text ends with a literal. Otherwise, when the
/// parentheses that keep the literal off the end are there only for a
/// warning, they are left out, so that the product can still be written as
/// juxtaposition (`a < 2x`). The text then gives the same precedence
/// warning as the source.
fn implicit_left(lhs: &Expr, l: Both) -> Out {
    let wrap = binary_child_wrap(BinaryOp::Mul, lhs, &l.normal, false);
    let normal = l.normal.wrap_if(wrap != Wrap::No);
    if normal.tail_literal.is_some() || wrap == Wrap::Needed || l.relaxed.tail_literal.is_none() {
        normal
    } else {
        l.relaxed
    }
}

/// Joins two written operands with the operator `op`.
fn binary_text(op: BinaryOp, implicit: bool, left: &Out, right: Out) -> Out {
    let mut tail_literal = right.tail_literal;
    let text = match op {
        BinaryOp::Mul if implicit && left.tail_literal.is_some() => {
            let zeros = left.tail_literal == Some(true);
            if can_follow_literal(&right.text, zeros) {
                format!("{}{}", left.text, right.text)
            } else {
                // Only the first operand of the right side goes in
                // parentheses (`2(e1)^2`, `0(0) < 0`), so that a literal at
                // its end can still start another juxtaposition.
                let (head, rest) = right.text.split_at(right.head);
                if rest.is_empty() {
                    tail_literal = None;
                }
                format!("{}({head}){rest}", left.text)
            }
        }
        BinaryOp::Add | BinaryOp::Sub => format!("{} {} {}", left.text, op.symbol(), right.text),
        op if op.is_comparison() => format!("{} {} {}", left.text, op.symbol(), right.text),
        op => format!("{}{}{}", left.text, op.symbol(), right.text),
    };
    Out {
        text,
        tail_literal,
        prefix: left.prefix,
        head: left.head,
    }
}

/// True when `rhs` can follow a number literal directly and still be read
/// as implicit multiplication. The parser needs a name or `(` right after
/// the literal, and the lexer must not take any of `rhs` into the number:
/// `e` or `E` followed by a digit, or by a sign and a digit, is an
/// exponent, and after a literal made only of zeros (`zeros`), `x` or `X`
/// followed by a hex digit is a malformed hex number.
fn can_follow_literal(rhs: &str, zeros: bool) -> bool {
    let b = rhs.as_bytes();
    let digit_at = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    match b.first() {
        Some(b'(') => true,
        Some(b'e' | b'E') => {
            !(digit_at(1) || (matches!(b.get(1), Some(b'+' | b'-')) && digit_at(2)))
        }
        Some(b'x' | b'X') if zeros => !b.get(1).is_some_and(u8::is_ascii_hexdigit),
        Some(c) => c.is_ascii_alphabetic(),
        None => false,
    }
}
