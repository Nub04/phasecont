//! Pass 3: expands pseudo-arrays and array blocks.
//!
//! [`expand`] runs after the include pass and before [`check`](crate::check).
//! A statement with one range `[i1..i2]` is replaced, in place, by one copy
//! per index from `i1` to `i2` (counting down when `i1 > i2`). An array block
//! `%[i1..i2] ... %` is replaced by its body copied once per index: the
//! whole body for `j = i1`, then the whole body for the next index. In a
//! copy, a name with an index becomes the name followed by the decimal
//! index (`x[j+1]` at `j = 2` is `x3`), and a bare index form such as
//! `[j*2]` becomes the integer it stands for. Every copy keeps the spans of
//! the template and records its index in [`Stmt::array_index`], so that
//! later passes can say which copy a diagnostic comes from.
//!
//! A statement whose array form is wrong (two ranges, a range inside a
//! block, `[j]` with no range in scope, an index below 0 or above
//! [`MAX_INDEX`], `markov` in an array) is reported and makes no copies.
//! See `docs/language.md`, "Arrays".

use crate::ast::{Expr, Ident, Index, IndexOp, MarkovLine, Model, Stmt, StmtKind, Value};
use crate::diag::{Code, Diagnostic};
use crate::span::Span;

/// A statement with more than one range `[i1..i2]`.
pub const MULTIPLE_RANGES: Code = Code(65);
/// A range `[i1..i2]` inside an array block.
pub const RANGE_IN_BLOCK: Code = Code(66);
/// An index form `[j]`, `[j+n]`, `[j-n]` or `[j*n]` with no range in scope.
pub const INDEX_OUT_OF_SCOPE: Code = Code(67);
/// A negative array block bound, or an index form that goes below 0.
pub const NEGATIVE_INDEX: Code = Code(68);
/// An index larger than [`MAX_INDEX`].
pub const INDEX_TOO_LARGE: Code = Code(69);
/// `markov` inside an array block, or with a range in it.
pub const MARKOV_ARRAY: Code = Code(70);
/// Expansion that would make more than [`MAX_ARRAY_COPIES`] statements.
pub const TOO_MANY_COPIES: Code = Code(71);

/// The most statements one [`expand`] call makes as copies.
pub const MAX_ARRAY_COPIES: usize = 100_000;

/// The largest index: 2^53, so that every index is exact as a double.
pub const MAX_INDEX: i64 = 1 << 53;

/// Expands every range line and array block of `model`. Statements without
/// an array form are kept as they are; a bad array block (`BadBlock`) is
/// kept too, since the parser already reported it.
pub fn expand(model: &Model) -> (Model, Vec<Diagnostic>) {
    let mut x = Expander::default();
    for stmt in &model.statements {
        x.statement(stmt);
    }
    (Model { statements: x.out }, x.diags)
}

/// One place in a statement where an index form can appear.
enum Slot<'a> {
    /// A name, declared or used.
    Name(&'a mut Ident),
    /// An expression that may be a bare index form.
    Bare(&'a mut Expr),
    /// A list value that may be a bare index form.
    Value(&'a mut Value),
}

/// Calls `f` on every name, expression and value of `kind` that can hold an
/// index form, in source order. The name of a function is skipped; the
/// caller handles it.
fn visit(kind: &mut StmtKind, f: &mut dyn FnMut(Slot)) {
    match kind {
        StmtKind::Ode { name, rhs, .. }
        | StmtKind::Map { name, rhs }
        | StmtKind::Derived { name, rhs }
        | StmtKind::Fixed { name, rhs }
        | StmtKind::Aux { name, rhs } => {
            f(Slot::Name(name));
            visit_expr(rhs, f);
        }
        StmtKind::InitValue { name, value } => {
            f(Slot::Name(name));
            f(Slot::Value(value));
        }
        StmtKind::Init { entries } | StmtKind::Par { entries } | StmtKind::Number { entries } => {
            for e in entries {
                f(Slot::Name(&mut e.name));
                if let Some(v) = &mut e.value {
                    f(Slot::Value(v));
                }
            }
        }
        StmtKind::Function { params, rhs, .. } => {
            for p in params {
                f(Slot::Name(p));
            }
            visit_expr(rhs, f);
        }
        StmtKind::Table {
            name,
            npts,
            lo,
            hi,
            rhs,
        } => {
            f(Slot::Name(name));
            f(Slot::Value(npts));
            f(Slot::Value(lo));
            f(Slot::Value(hi));
            visit_expr(rhs, f);
        }
        StmtKind::Options { entries } => {
            for e in entries {
                f(Slot::Name(&mut e.name));
            }
        }
        StmtKind::Set { name, entries } => {
            f(Slot::Name(name));
            for e in entries {
                f(Slot::Name(&mut e.name));
            }
        }
        StmtKind::Global { cond, assigns, .. } => {
            visit_expr(cond, f);
            for a in assigns {
                f(Slot::Name(&mut a.name));
                visit_expr(&mut a.rhs, f);
            }
        }
        StmtKind::Markov { name, lines, .. } => {
            f(Slot::Name(name));
            for line in lines {
                if let MarkovLine::Row(row) = line {
                    for e in &mut row.entries {
                        visit_expr(e, f);
                    }
                }
            }
        }
        StmtKind::Wiener { names } => {
            for n in names {
                f(Slot::Name(n));
            }
        }
        StmtKind::Include { .. }
        | StmtKind::Block { .. }
        | StmtKind::BadBlock { .. }
        | StmtKind::Comment(_)
        | StmtKind::Unsupported { .. }
        | StmtKind::Done => {}
    }
}

fn visit_expr(expr: &mut Expr, f: &mut dyn FnMut(Slot)) {
    match expr {
        Expr::Num { .. } | Expr::Error { .. } => {}
        Expr::Index(_) => f(Slot::Bare(expr)),
        Expr::Name(id) => f(Slot::Name(id)),
        Expr::Unary { operand, .. } => visit_expr(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            visit_expr(lhs, f);
            visit_expr(rhs, f);
        }
        Expr::Call { name, args, .. } => {
            f(Slot::Name(name));
            for a in args {
                visit_expr(a, f);
            }
        }
        Expr::If {
            cond, then, else_, ..
        } => {
            visit_expr(cond, f);
            visit_expr(then, f);
            visit_expr(else_, f);
        }
    }
}

/// One index form found in a statement.
struct Occurrence {
    index: Index,
    /// The form as written for messages: the name and its index, or the
    /// bare index.
    text: String,
    /// Where to report a value that is out of bounds: the whole name, or
    /// the bare index.
    span: Span,
}

/// Every index form of `kind`, in source order.
fn occurrences(kind: &StmtKind) -> Vec<Occurrence> {
    let mut kind = kind.clone();
    let mut found = Vec::new();
    visit(&mut kind, &mut |slot| match slot {
        Slot::Name(id) => {
            if let Some(index) = &id.index {
                found.push(Occurrence {
                    index: index.clone(),
                    text: format!("{}{}", id.text, index_text(index)),
                    span: id.span,
                });
            }
        }
        Slot::Bare(Expr::Index(index)) | Slot::Value(Value::Index(index)) => {
            found.push(Occurrence {
                index: index.clone(),
                text: index_text(index),
                span: index.span(),
            });
        }
        Slot::Bare(_) | Slot::Value(_) => {}
    });
    found
}

/// An index form as XPPAUT writes it.
fn index_text(index: &Index) -> String {
    match index {
        Index::Range { start, end, .. } => format!("[{start}..{end}]"),
        Index::Var { op, n, .. } => match op {
            IndexOp::None => "[j]".to_string(),
            IndexOp::Add => format!("[j+{n}]"),
            IndexOp::Sub => format!("[j-{n}]"),
            IndexOp::Mul => format!("[j*{n}]"),
        },
    }
}

/// The value of an index form at `j`, or `None` when it overflows. A range
/// stands for `j` itself.
fn index_value(index: &Index, j: i64) -> Option<i64> {
    match index {
        Index::Range { .. } => Some(j),
        Index::Var { op, n, .. } => {
            let n = i64::from(*n);
            match op {
                IndexOp::None => Some(j),
                IndexOp::Add => j.checked_add(n),
                IndexOp::Sub => j.checked_sub(n),
                IndexOp::Mul => j.checked_mul(n),
            }
        }
    }
}

/// The note that points at the first of several ranges.
fn first_range_note(span: Span) -> (Span, String) {
    (span, "the first range is here".to_string())
}

#[derive(Default)]
struct Expander {
    out: Vec<Stmt>,
    diags: Vec<Diagnostic>,
    /// Copies made so far.
    produced: usize,
    /// Set once a statement would pass [`MAX_ARRAY_COPIES`]; from then on
    /// no statement makes copies.
    capped: bool,
}

impl Expander {
    fn report(&mut self, code: Code, span: Span, message: String) -> &mut Diagnostic {
        self.diags.push(Diagnostic::new(code, span, message));
        self.diags.last_mut().expect("just pushed")
    }

    fn statement(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Block {
                start,
                end,
                range_span,
                body,
            } => self.block(*start, *end, *range_span, body),
            // The parser reported it; keep it as written.
            StmtKind::BadBlock { .. } => self.out.push(stmt.clone()),
            // The parser reported the index on a function name.
            StmtKind::Function { name, .. } if name.index.is_some() => {}
            _ => self.line(stmt),
        }
    }

    /// A statement outside an array block.
    fn line(&mut self, stmt: &Stmt) {
        let found = occurrences(&stmt.kind);
        let ranges: Vec<&Occurrence> = found
            .iter()
            .filter(|o| matches!(o.index, Index::Range { .. }))
            .collect();
        if let StmtKind::Markov { .. } = stmt.kind
            && let Some(first) = ranges.first()
        {
            let message = "a `markov` statement cannot be copied with a range; write one \
                           `markov` statement per state variable"
                .to_string();
            self.report(MARKOV_ARRAY, first.index.span(), message);
            return;
        }
        match ranges.as_slice() {
            [] => {
                if let Some(var) = found.first() {
                    let message = format!(
                        "`{}` uses the index `j`, but this line has no range `[i1..i2]` and is \
                         not inside an array block",
                        var.text
                    );
                    self.report(INDEX_OUT_OF_SCOPE, var.index.span(), message);
                } else {
                    self.out.push(stmt.clone());
                }
            }
            [range] => {
                let Index::Range { start, end, span } = range.index else {
                    unreachable!("filtered to ranges");
                };
                let (start, end) = (i64::from(start), i64::from(end));
                if !self.bounds_ok(&found, start.min(end), start.max(end)) {
                    return;
                }
                let count = start.abs_diff(end) + 1;
                if !self.take(count, span) {
                    return;
                }
                for j in indices(start, end) {
                    self.out.push(copy(stmt, j));
                }
            }
            [first, second, ..] => {
                let message = format!(
                    "a statement can have only one range; `{}` is a second one",
                    second.text
                );
                let note = first_range_note(first.index.span());
                self.report(MULTIPLE_RANGES, second.index.span(), message)
                    .notes
                    .push(note);
            }
        }
    }

    fn block(&mut self, start: i64, end: i64, range_span: Span, body: &[Stmt]) {
        for bound in [start, end] {
            if bound < 0 {
                let message = format!(
                    "array block bound {bound} is negative; array indices must be 0 or more"
                );
                self.report(NEGATIVE_INDEX, range_span, message);
                return;
            }
            if bound > MAX_INDEX {
                let message = format!(
                    "array block bound {bound} is larger than the largest index, {MAX_INDEX}"
                );
                self.report(INDEX_TOO_LARGE, range_span, message);
                return;
            }
        }
        let (lo, hi) = (start.min(end), start.max(end));
        let mut valid = Vec::new();
        for stmt in body {
            match &stmt.kind {
                // The include pass reported it.
                StmtKind::Include { .. } => continue,
                // The parser reported nested and bad blocks, and an index on
                // a function name.
                StmtKind::Block { .. } | StmtKind::BadBlock { .. } => continue,
                StmtKind::Function { name, .. } if name.index.is_some() => continue,
                StmtKind::Markov { name, .. } => {
                    let message = "a `markov` statement cannot be inside an array block; write \
                                   one `markov` statement per state variable"
                        .to_string();
                    self.report(MARKOV_ARRAY, name.span, message);
                    continue;
                }
                _ => {}
            }
            let found = occurrences(&stmt.kind);
            if let Some(range) = found
                .iter()
                .find(|o| matches!(o.index, Index::Range { .. }))
            {
                let message = format!(
                    "`{}` has a range inside an array block; use the block index `j` instead",
                    range.text
                );
                self.report(RANGE_IN_BLOCK, range.index.span(), message);
                continue;
            }
            if self.bounds_ok(&found, lo, hi) {
                valid.push(stmt);
            }
        }
        if valid.is_empty() {
            return;
        }
        let count = (u128::from(start.abs_diff(end)) + 1) * valid.len() as u128;
        let count = u64::try_from(count).unwrap_or(u64::MAX);
        if !self.take(count, range_span) {
            return;
        }
        for j in indices(start, end) {
            for stmt in &valid {
                self.out.push(copy(stmt, j));
            }
        }
    }

    /// Checks that every index form in `found` stays in `0..=MAX_INDEX` for
    /// `j` from `lo` to `hi`, and reports each one that does not. Every
    /// form is monotone in `j`, so checking the two ends is enough.
    fn bounds_ok(&mut self, found: &[Occurrence], lo: i64, hi: i64) -> bool {
        let mut ok = true;
        for o in found {
            if let Index::Var {
                op: IndexOp::Sub, ..
            } = o.index
                && let Some(v) = index_value(&o.index, lo)
                && v < 0
            {
                let message = format!(
                    "`{}` is {v} at j = {lo}; array indices must be 0 or more",
                    o.text
                );
                self.report(NEGATIVE_INDEX, o.span, message);
                ok = false;
                continue;
            }
            let message = match index_value(&o.index, hi) {
                Some(v) if v > MAX_INDEX => format!(
                    "`{}` is {v} at j = {hi}, larger than the largest index, {MAX_INDEX}",
                    o.text
                ),
                None => format!(
                    "`{}` overflows at j = {hi}; the largest index is {MAX_INDEX}",
                    o.text
                ),
                Some(_) => continue,
            };
            self.report(INDEX_TOO_LARGE, o.span, message);
            ok = false;
        }
        ok
    }

    /// Counts `count` copies against [`MAX_ARRAY_COPIES`]. Returns false,
    /// after reporting once at `span`, when they do not fit or an earlier
    /// statement already passed the limit.
    fn take(&mut self, count: u64, span: Span) -> bool {
        if self.capped {
            return false;
        }
        let room = (MAX_ARRAY_COPIES - self.produced) as u64;
        if count > room {
            self.capped = true;
            let message = format!(
                "this array makes {count} statements, and with the copies before it that is \
                 more than {MAX_ARRAY_COPIES}; it and every later array are not expanded"
            );
            self.report(TOO_MANY_COPIES, span, message);
            return false;
        }
        self.produced += count as usize;
        true
    }
}

/// The indices from `start` to `end`, counting down when `start > end`.
fn indices(start: i64, end: i64) -> Box<dyn Iterator<Item = i64>> {
    if start <= end {
        Box::new(start..=end)
    } else {
        Box::new((end..=start).rev())
    }
}

/// The copy of `stmt` for index `j`. The caller checked every index form,
/// so no value overflows here.
fn copy(stmt: &Stmt, j: i64) -> Stmt {
    let mut out = stmt.clone();
    out.array_index = Some(j);
    let value = |index: &Index| index_value(index, j).expect("index checked before copying");
    visit(&mut out.kind, &mut |slot| match slot {
        Slot::Name(id) => {
            if let Some(index) = id.index.take() {
                let v = value(&index);
                id.text = format!("{}{v}", id.text);
                id.key = format!("{}{v}", id.key);
            }
        }
        Slot::Bare(expr) => {
            if let Expr::Index(index) = expr {
                let v = value(index);
                *expr = Expr::Num {
                    value: v as f64,
                    text: v.to_string(),
                    span: index.span(),
                };
            }
        }
        Slot::Value(val) => {
            if let Value::Index(index) = val {
                let v = value(index);
                *val = Value::Num {
                    value: v as f64,
                    text: v.to_string(),
                    span: index.span(),
                };
            }
        }
    });
    out
}

#[cfg(test)]
mod tests;
