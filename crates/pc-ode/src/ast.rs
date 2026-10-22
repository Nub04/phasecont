//! Syntax tree for `.ode` expressions.
//!
//! Every node carries the span of the text it came from. Parentheses are not
//! kept as nodes: `(2*x)` and `2*x` give the same tree, and the writer adds
//! parentheses only where the precedence rules need them. The span of a node
//! never includes parentheses written around it.
//!
//! `Model` and the statement types arrive with the statement parser.

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
