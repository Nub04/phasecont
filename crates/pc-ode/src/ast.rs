//! Syntax tree for `.ode` files.
//!
//! Every node carries the span of the text it came from. Parentheses are not
//! kept as nodes: `(2*x)` and `2*x` give the same tree, and the writer adds
//! parentheses only where the precedence rules need them. The span of a node
//! never includes parentheses written around it.
//!
//! A file is a [`Model`]: a list of [`Stmt`] in source order.

use crate::span::{FileId, Span};

/// The span that [`Expr::strip_spans`] puts on every node, so that trees
/// parsed from different text can be compared.
pub const DUMMY_SPAN: Span = Span {
    file: FileId(0),
    start: 0,
    end: 0,
};

/// A prefix operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Prefix `-`.
    Neg,
    /// Prefix `not`.
    Not,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `^`
    Pow,
    /// `**`, the same operation as `^`, kept apart so the writer can write it
    /// back as written.
    StarStar,
    Lt,
    Gt,
    Le,
    Ge,
    /// `==`
    Eq,
    /// `&`
    And,
    /// `|`
    Or,
}

impl BinaryOp {
    /// `<`, `>`, `<=`, `>=` and `==`.
    pub fn is_comparison(self) -> bool {
        matches!(self, Self::Lt | Self::Gt | Self::Le | Self::Ge | Self::Eq)
    }

    /// `+`, `-`, `*` (also implicit) and `/`.
    pub fn is_arithmetic(self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }

    /// `&` and `|`.
    pub fn is_logical(self) -> bool {
        matches!(self, Self::And | Self::Or)
    }

    /// `^` and `**`.
    pub fn is_power(self) -> bool {
        matches!(self, Self::Pow | Self::StarStar)
    }

    /// The operator as written in `.ode` text.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Pow => "^",
            Self::StarStar => "**",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Ge => ">=",
            Self::Eq => "==",
            Self::And => "&",
            Self::Or => "|",
        }
    }
}

/// The operator of an index form `[j+n]`, `[j-n]` or `[j*n]`; `None` is `[j]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexOp {
    None,
    Add,
    Sub,
    Mul,
}

/// An array index form in square brackets. The span covers the brackets.
#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    /// `[start..end]`. `start` may be larger than `end`.
    Range { start: u32, end: u32, span: Span },
    /// `[j]` (with `op` `None` and `n` 0), `[j+n]`, `[j-n]` or `[j*n]`.
    Var { op: IndexOp, n: u32, span: Span },
}

impl Index {
    pub fn span(&self) -> Span {
        match self {
            Self::Range { span, .. } | Self::Var { span, .. } => *span,
        }
    }

    /// A copy with every span replaced by [`DUMMY_SPAN`].
    pub fn strip_spans(&self) -> Index {
        match self {
            Self::Range { start, end, .. } => Self::Range {
                start: *start,
                end: *end,
                span: DUMMY_SPAN,
            },
            Self::Var { op, n, .. } => Self::Var {
                op: *op,
                n: *n,
                span: DUMMY_SPAN,
            },
        }
    }
}

/// A name as written, with its lower-case lookup key and an optional index
/// form. The span covers the name and its index.
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub text: String,
    pub key: String,
    pub index: Option<Index>,
    pub span: Span,
}

impl Ident {
    /// A copy with every span replaced by [`DUMMY_SPAN`].
    pub fn strip_spans(&self) -> Ident {
        Ident {
            text: self.text.clone(),
            key: self.key.clone(),
            index: self.index.as_ref().map(Index::strip_spans),
            span: DUMMY_SPAN,
        }
    }
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A number literal with its value and its text as written.
    Num {
        value: f64,
        text: String,
        span: Span,
    },
    /// A name, possibly with an index form (`x`, `x[j+1]`).
    Name(Ident),
    /// A bare index form used as a value (`[j]`, `[j+1]`).
    Index(Index),
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// `implicit` is true only for a `Mul` written without `*` (`2x`).
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        implicit: bool,
        span: Span,
    },
    /// A call of any arity; builtin arity is checked later.
    Call {
        name: Ident,
        args: Vec<Expr>,
        span: Span,
    },
    /// `if(cond)then(then)else(else_)`.
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
        span: Span,
    },
    /// A missing or unusable operand, already reported.
    Error { span: Span },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Self::Num { span, .. }
            | Self::Unary { span, .. }
            | Self::Binary { span, .. }
            | Self::Call { span, .. }
            | Self::If { span, .. }
            | Self::Error { span } => *span,
            Self::Name(ident) => ident.span,
            Self::Index(index) => index.span(),
        }
    }

    /// True for [`Expr::Error`].
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// A copy with every span replaced by [`DUMMY_SPAN`], so that two trees
    /// compare equal when they differ only in where they came from.
    pub fn strip_spans(&self) -> Expr {
        let b = |e: &Expr| Box::new(e.strip_spans());
        match self {
            Self::Num { value, text, .. } => Self::Num {
                value: *value,
                text: text.clone(),
                span: DUMMY_SPAN,
            },
            Self::Name(ident) => Self::Name(ident.strip_spans()),
            Self::Index(index) => Self::Index(index.strip_spans()),
            Self::Unary { op, operand, .. } => Self::Unary {
                op: *op,
                operand: b(operand),
                span: DUMMY_SPAN,
            },
            Self::Binary {
                op,
                lhs,
                rhs,
                implicit,
                ..
            } => Self::Binary {
                op: *op,
                lhs: b(lhs),
                rhs: b(rhs),
                implicit: *implicit,
                span: DUMMY_SPAN,
            },
            Self::Call { name, args, .. } => Self::Call {
                name: name.strip_spans(),
                args: args.iter().map(Expr::strip_spans).collect(),
                span: DUMMY_SPAN,
            },
            Self::If {
                cond, then, else_, ..
            } => Self::If {
                cond: b(cond),
                then: b(then),
                else_: b(else_),
                span: DUMMY_SPAN,
            },
            Self::Error { .. } => Self::Error { span: DUMMY_SPAN },
        }
    }
}

/// A whole `.ode` file: its statements in source order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model {
    pub statements: Vec<Stmt>,
}

impl Model {
    /// A copy with every span replaced by [`DUMMY_SPAN`].
    pub fn strip_spans(&self) -> Model {
        Model {
            statements: self.statements.iter().map(Stmt::strip_spans).collect(),
        }
    }
}

/// One statement. `span` covers its full text: every continuation line, a
/// trailing comment and, for `markov` and array blocks, the lines that
/// belong to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    /// A `#` comment after the statement on the same line. It is allowed
    /// after `par`, `number` and `@` lines; anywhere else the parser reports
    /// it but keeps it here.
    pub comment: Option<Comment>,
    pub span: Span,
}

impl Stmt {
    /// A copy with every span replaced by [`DUMMY_SPAN`].
    pub fn strip_spans(&self) -> Stmt {
        Stmt {
            kind: self.kind.strip_spans(),
            comment: self.comment.as_ref().map(Comment::strip_spans),
            span: DUMMY_SPAN,
        }
    }
}

/// A `#` comment. `text` starts with the `#` and has no line terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct Comment {
    pub text: String,
    pub span: Span,
}

impl Comment {
    pub fn strip_spans(&self) -> Comment {
        Comment {
            text: self.text.clone(),
            span: DUMMY_SPAN,
        }
    }
}

/// A value in a `par`, `number`, `init` or `table` header, or on the right
/// of `x(0)=`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// An optionally negative number literal. `text` is the literal as
    /// written, with a leading `-` when there is one (`-1.5e3`).
    Num {
        value: f64,
        text: String,
        span: Span,
    },
    /// A bare index form (`[j]`, `[j+1]`), expanded with the arrays.
    Index(Index),
    /// A value that is not a literal, already reported.
    Error { span: Span },
}

impl Value {
    pub fn span(&self) -> Span {
        match self {
            Self::Num { span, .. } | Self::Error { span } => *span,
            Self::Index(index) => index.span(),
        }
    }

    pub fn strip_spans(&self) -> Value {
        match self {
            Self::Num { value, text, .. } => Self::Num {
                value: *value,
                text: text.clone(),
                span: DUMMY_SPAN,
            },
            Self::Index(index) => Self::Index(index.strip_spans()),
            Self::Error { .. } => Self::Error { span: DUMMY_SPAN },
        }
    }
}

/// One `name=value` entry of a `par`, `number` or `init` list. `value` is
/// `None` only for a bare name in a `par` list, which means 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub name: Ident,
    pub value: Option<Value>,
    pub span: Span,
}

impl Entry {
    pub fn strip_spans(&self) -> Entry {
        Entry {
            name: self.name.strip_spans(),
            value: self.value.as_ref().map(Value::strip_spans),
            span: DUMMY_SPAN,
        }
    }
}

/// One `name=value` entry of an `@` line or a `set` block. The value is
/// kept as source text (it may be a number, a name, a file name or a
/// button definition such as `Fit:wf`).
#[derive(Debug, Clone, PartialEq)]
pub struct RawAssign {
    pub name: Ident,
    pub value: String,
    pub value_span: Span,
    pub span: Span,
}

impl RawAssign {
    pub fn strip_spans(&self) -> RawAssign {
        RawAssign {
            name: self.name.strip_spans(),
            value: self.value.clone(),
            value_span: DUMMY_SPAN,
            span: DUMMY_SPAN,
        }
    }
}

/// One `name=expr` assignment in a `global` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub name: Ident,
    pub rhs: Expr,
    pub span: Span,
}

impl Assign {
    pub fn strip_spans(&self) -> Assign {
        Assign {
            name: self.name.strip_spans(),
            rhs: self.rhs.strip_spans(),
            span: DUMMY_SPAN,
        }
    }
}

/// One row of a `markov` statement: one `{expr}` per state.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovRow {
    pub entries: Vec<Expr>,
    pub span: Span,
}

impl MarkovRow {
    pub fn strip_spans(&self) -> MarkovRow {
        MarkovRow {
            entries: self.entries.iter().map(Expr::strip_spans).collect(),
            span: DUMMY_SPAN,
        }
    }
}

/// How an ODE left side was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeForm {
    /// `x'=`
    Prime,
    /// `dx/dt=`
    Ddt,
}

/// The family of a statement PhaseCont keeps but does not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unsupported {
    /// `volt`, `x(t)=...`, or `int{` / `int[` in a formula.
    Volterra,
    /// `bndry`, `bdry` (any first word starting with `b`).
    Boundary,
    /// `0=...` or `solv`.
    Dae,
    Special,
    Export,
    Only,
    /// `options file`.
    OptionsFile,
    HomBcs,
    /// `table name file`.
    FileTable,
    /// `sum(..)of(..)` in a formula.
    SumOf,
    /// A line that starts with `"`.
    ActiveComment,
}

/// What a statement is.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    /// `x'=rhs` or `dx/dt=rhs`.
    Ode {
        name: Ident,
        form: OdeForm,
        rhs: Expr,
    },
    /// `x(t+1)=rhs`.
    Map { name: Ident, rhs: Expr },
    /// `x(0)=value`.
    InitValue { name: Ident, value: Value },
    /// `init x=1,y=2`.
    Init { entries: Vec<Entry> },
    /// `par a=1,b=2`.
    Par { entries: Vec<Entry> },
    /// `number c=3`.
    Number { entries: Vec<Entry> },
    /// `!name=rhs`.
    Derived { name: Ident, rhs: Expr },
    /// `name=rhs`.
    Fixed { name: Ident, rhs: Expr },
    /// `aux name=rhs`.
    Aux { name: Ident, rhs: Expr },
    /// `f(u,v)=rhs`. The name and the parameters have no index.
    Function {
        name: Ident,
        params: Vec<Ident>,
        rhs: Expr,
    },
    /// `table name % npts lo hi rhs`.
    Table {
        name: Ident,
        npts: Value,
        lo: Value,
        hi: Value,
        rhs: Expr,
    },
    /// `@ name=value ...`.
    Options { entries: Vec<RawAssign> },
    /// `#include path`. `path` is the text after the keyword, trimmed.
    Include { path: String, path_span: Span },
    /// `set name {a=1,b=2}`.
    Set {
        name: Ident,
        entries: Vec<RawAssign>,
    },
    /// `global sign cond {name=expr; ...}`, with `sign` -1, 0 or 1.
    Global {
        sign: i8,
        cond: Expr,
        assigns: Vec<Assign>,
    },
    /// `markov name n`, then `n` rows of `n` entries each.
    Markov {
        name: Ident,
        states: u32,
        rows: Vec<MarkovRow>,
    },
    /// `wiener w1,w2`.
    Wiener { names: Vec<Ident> },
    /// `%[start..end]`, the statements, `%`. Blocks do not nest, so `body`
    /// holds no `Block`. Bounds are as written; Task 8 rejects negative ones.
    Block {
        start: i64,
        end: i64,
        range_span: Span,
        body: Vec<Stmt>,
    },
    /// A full-line `#` comment.
    Comment(Comment),
    /// A statement kept as raw source text (the statement's span).
    Unsupported { family: Unsupported, text: String },
    /// `done`. Nothing after it is parsed.
    Done,
}

impl StmtKind {
    /// A copy with every span replaced by [`DUMMY_SPAN`].
    pub fn strip_spans(&self) -> StmtKind {
        let entries = |e: &[Entry]| e.iter().map(Entry::strip_spans).collect();
        let raw = |e: &[RawAssign]| e.iter().map(RawAssign::strip_spans).collect();
        match self {
            Self::Ode { name, form, rhs } => Self::Ode {
                name: name.strip_spans(),
                form: *form,
                rhs: rhs.strip_spans(),
            },
            Self::Map { name, rhs } => Self::Map {
                name: name.strip_spans(),
                rhs: rhs.strip_spans(),
            },
            Self::InitValue { name, value } => Self::InitValue {
                name: name.strip_spans(),
                value: value.strip_spans(),
            },
            Self::Init { entries: e } => Self::Init {
                entries: entries(e),
            },
            Self::Par { entries: e } => Self::Par {
                entries: entries(e),
            },
            Self::Number { entries: e } => Self::Number {
                entries: entries(e),
            },
            Self::Derived { name, rhs } => Self::Derived {
                name: name.strip_spans(),
                rhs: rhs.strip_spans(),
            },
            Self::Fixed { name, rhs } => Self::Fixed {
                name: name.strip_spans(),
                rhs: rhs.strip_spans(),
            },
            Self::Aux { name, rhs } => Self::Aux {
                name: name.strip_spans(),
                rhs: rhs.strip_spans(),
            },
            Self::Function { name, params, rhs } => Self::Function {
                name: name.strip_spans(),
                params: params.iter().map(Ident::strip_spans).collect(),
                rhs: rhs.strip_spans(),
            },
            Self::Table {
                name,
                npts,
                lo,
                hi,
                rhs,
            } => Self::Table {
                name: name.strip_spans(),
                npts: npts.strip_spans(),
                lo: lo.strip_spans(),
                hi: hi.strip_spans(),
                rhs: rhs.strip_spans(),
            },
            Self::Options { entries } => Self::Options {
                entries: raw(entries),
            },
            Self::Include { path, .. } => Self::Include {
                path: path.clone(),
                path_span: DUMMY_SPAN,
            },
            Self::Set { name, entries } => Self::Set {
                name: name.strip_spans(),
                entries: raw(entries),
            },
            Self::Global {
                sign,
                cond,
                assigns,
            } => Self::Global {
                sign: *sign,
                cond: cond.strip_spans(),
                assigns: assigns.iter().map(Assign::strip_spans).collect(),
            },
            Self::Markov { name, states, rows } => Self::Markov {
                name: name.strip_spans(),
                states: *states,
                rows: rows.iter().map(MarkovRow::strip_spans).collect(),
            },
            Self::Wiener { names } => Self::Wiener {
                names: names.iter().map(Ident::strip_spans).collect(),
            },
            Self::Block {
                start, end, body, ..
            } => Self::Block {
                start: *start,
                end: *end,
                range_span: DUMMY_SPAN,
                body: body.iter().map(Stmt::strip_spans).collect(),
            },
            Self::Comment(c) => Self::Comment(c.strip_spans()),
            Self::Unsupported { family, text } => Self::Unsupported {
                family: *family,
                text: text.clone(),
            },
            Self::Done => Self::Done,
        }
    }
}
