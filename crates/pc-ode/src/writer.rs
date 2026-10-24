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
//! - Expressions get parentheses only where the measured precedence needs
//!   them (`docs/language.md`, "Precedence"), where the parser would give a
//!   precedence warning, and around a prefix `-` or `not` that is not at the
//!   start of an expression, after `(` or after `,`. Binary `+`, `-` and the
//!   comparisons have a blank on each side; the other operators have none.
//! - Implicit multiplication is written as juxtaposition (`2x`) when the
//!   left operand's text ends with a number literal and the right operand's
//!   text starts with a name or `(`, unless that name starts with `e` or `E`
//!   and a digit, which would lex as part of the number. Otherwise it is
//!   written with `*`. When the left operand needs parentheses (`(a<2)x`
//!   cannot be written), the `*` form loses the `implicit` flag.
//! - `#include` lines, range lines, `%` blocks, comments and the trailing
//!   comments of `par`, `number` and `@` lines are kept. Unsupported
//!   statements are written as their raw text.
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
/// `1000000000000000`, `1e16`). A negative value gets a leading `-`. A value
/// that is not finite gives `1e999`, which the lexer rejects.
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
    trailing(out, &s.comment);
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

/// True when `child`, an operand of the binary operator `parent`, needs
/// parentheses. `at_start` tells whether the operand's text would start the
/// expression (after `(`, `,` or `=`).
fn binary_child_parens(parent: BinaryOp, child: &Expr, is_rhs: bool, at_start: bool) -> bool {
    match child {
        // The operand of a prefix operator is parsed at level 4, so a prefix
        // expression cannot be the left operand of a comparison or power
        // without parentheses. Elsewhere it is wrapped unless it starts the
        // expression.
        Expr::Unary { .. } => parent.is_comparison() || parent.is_power() || is_rhs || !at_start,
        Expr::Binary { op, .. } => {
            let (p, c) = (level(parent), level(*op));
            c < p
                || (c == p && is_rhs)
                || (parent.is_comparison() && op.is_comparison())
                || (parent.is_power() && op.is_power())
                || (parent.is_arithmetic() && (op.is_comparison() || op.is_logical()))
                || (parent.is_logical() && op.is_arithmetic())
        }
        _ => false,
    }
}

fn expr_text(e: &Expr) -> String {
    expr(e, true).0
}

/// Writes `e`. Returns the text and whether it ends with a number literal
/// that is not inside parentheses.
fn expr(e: &Expr, at_start: bool) -> (String, bool) {
    match e {
        Expr::Num { value, text, .. } => (literal(*value, text), true),
        Expr::Name(id) => (ident(id), false),
        Expr::Index(ix) => (index(ix), false),
        Expr::Error { .. } => ("?".to_string(), false),
        Expr::Unary { op, operand, .. } => {
            let wrap = expr_level(operand) < 4
                || (*op == UnaryOp::Neg
                    && matches!(**operand, Expr::Binary { op, .. } if op.is_comparison()));
            let (inner, lit) = child(operand, wrap, false);
            let prefix = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Not => "not ",
            };
            (format!("{prefix}{inner}"), lit)
        }
        Expr::Binary {
            op,
            lhs,
            rhs,
            implicit,
            ..
        } => {
            let lwrap = binary_child_parens(*op, lhs, false, at_start);
            let rwrap = binary_child_parens(*op, rhs, true, false);
            let (ltext, llit) = child(lhs, lwrap, at_start);
            let (rtext, rlit) = child(rhs, rwrap, false);
            let sep = match op {
                BinaryOp::Mul if *implicit && llit && juxtaposable(&rtext) => "",
                BinaryOp::Add | BinaryOp::Sub => {
                    return (format!("{ltext} {} {rtext}", op.symbol()), rlit);
                }
                op if op.is_comparison() => {
                    return (format!("{ltext} {} {rtext}", op.symbol()), rlit);
                }
                op => op.symbol(),
            };
            (format!("{ltext}{sep}{rtext}"), rlit)
        }
        Expr::Call { name, args, .. } => {
            let args: Vec<String> = args.iter().map(expr_text).collect();
            (format!("{}({})", ident(name), args.join(",")), false)
        }
        Expr::If {
            cond, then, else_, ..
        } => (
            format!(
                "if({})then({})else({})",
                expr_text(cond),
                expr_text(then),
                expr_text(else_)
            ),
            false,
        ),
    }
}

/// Writes an operand, in parentheses when `wrap` is set.
fn child(e: &Expr, wrap: bool, at_start: bool) -> (String, bool) {
    if wrap {
        (format!("({})", expr(e, true).0), false)
    } else {
        expr(e, at_start)
    }
}

/// True when `rhs` can follow a number literal without `*`: it starts with
/// a letter or `(`, and not with `e` or `E` followed by a digit.
fn juxtaposable(rhs: &str) -> bool {
    let b = rhs.as_bytes();
    match b.first() {
        Some(b'(') => true,
        Some(c) if c.is_ascii_alphabetic() => {
            !((*c == b'e' || *c == b'E') && b.get(1).is_some_and(u8::is_ascii_digit))
        }
        _ => false,
    }
}
