//! The expression tree every importer builds, and its `.ode` printer.
//!
//! Each node stores its depth and size, and the constructors refuse a node past the limits.
//! That keeps every later walk (printing, substitution, dropping) within a fixed stack depth
//! and keeps an inlined function from growing without bound. A stack overflow aborts the
//! process, and no panic may cross the C ABI (rule 7), so the limits are part of the contract.

use crate::error::{ImportError, Kind, Result};

/// Deepest expression the importer builds. It stays under the `.ode` parser's own limit of 200.
pub const MAX_DEPTH: u32 = 150;
/// Most nodes in one expression.
pub const MAX_NODES: u64 = 200_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    And,
    Or,
}

impl Op {
    fn text(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mul => "*",
            Op::Div => "/",
            Op::Pow => "^",
            Op::Lt => "<",
            Op::Le => "<=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Eq => "==",
            Op::And => "&",
            Op::Or => "|",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Num(f64),
    /// A name as the source wrote it. The namer maps it when the expression is printed.
    Var(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    /// A call to an `.ode` builtin, by its `.ode` name.
    Call(String, Vec<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub node: Node,
    depth: u32,
    size: u64,
}

fn too_large(what: &str) -> ImportError {
    ImportError::new(
        Kind::TooLarge,
        format!("an expression is {what}, past the importer's limit"),
    )
}

impl Expr {
    fn make(node: Node, children: &[&Expr]) -> Result<Expr> {
        let depth = 1 + children.iter().map(|c| c.depth).max().unwrap_or(0);
        let size = 1 + children.iter().map(|c| c.size).sum::<u64>();
        Expr::check(depth, size)?;
        Ok(Expr { node, depth, size })
    }

    /// A number. A value that is not finite is refused: `.ode` has no literal for it.
    pub fn num(v: f64) -> Result<Expr> {
        if !v.is_finite() {
            return Err(ImportError::new(
                Kind::Unsupported,
                "a constant is infinite or not a number, and `.ode` has no literal for it",
            ));
        }
        Expr::make(Node::Num(v), &[])
    }

    pub fn var(name: impl Into<String>) -> Expr {
        Expr {
            node: Node::Var(name.into()),
            depth: 1,
            size: 1,
        }
    }

    pub fn negate(e: Expr) -> Result<Expr> {
        let (depth, size) = (e.depth, e.size);
        Expr::check(depth + 1, size + 1)?;
        Ok(Expr {
            node: Node::Neg(Box::new(e)),
            depth: depth + 1,
            size: size + 1,
        })
    }

    pub fn logical_not(e: Expr) -> Result<Expr> {
        let (depth, size) = (e.depth, e.size);
        Expr::check(depth + 1, size + 1)?;
        Ok(Expr {
            node: Node::Not(Box::new(e)),
            depth: depth + 1,
            size: size + 1,
        })
    }

    fn check(depth: u32, size: u64) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(too_large("nested too deeply"));
        }
        if size > MAX_NODES {
            return Err(too_large("too long"));
        }
        Ok(())
    }

    pub fn bin(op: Op, a: Expr, b: Expr) -> Result<Expr> {
        let depth = 1 + a.depth.max(b.depth);
        let size = 1 + a.size + b.size;
        Expr::check(depth, size)?;
        Ok(Expr {
            node: Node::Bin(op, Box::new(a), Box::new(b)),
            depth,
            size,
        })
    }

    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Result<Expr> {
        let depth = 1 + args.iter().map(|c| c.depth).max().unwrap_or(0);
        let size = 1 + args.iter().map(|c| c.size).sum::<u64>();
        Expr::check(depth, size)?;
        Ok(Expr {
            node: Node::Call(name.into(), args),
            depth,
            size,
        })
    }

    pub fn cond(c: Expr, a: Expr, b: Expr) -> Result<Expr> {
        let depth = 1 + c.depth.max(a.depth).max(b.depth);
        let size = 1 + c.size + a.size + b.size;
        Expr::check(depth, size)?;
        Ok(Expr {
            node: Node::If(Box::new(c), Box::new(a), Box::new(b)),
            depth,
            size,
        })
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Rebuilds the tree with `f` applied to every variable. `f` returns the replacement
    /// expression, or `None` to keep the variable. Fails when the result passes a limit.
    pub fn substitute(&self, f: &mut dyn FnMut(&str) -> Option<Expr>) -> Result<Expr> {
        Ok(match &self.node {
            Node::Num(_) => self.clone(),
            Node::Var(n) => match f(n) {
                Some(e) => e,
                None => self.clone(),
            },
            Node::Neg(e) => Expr::negate(e.substitute(f)?)?,
            Node::Not(e) => Expr::logical_not(e.substitute(f)?)?,
            Node::Bin(op, a, b) => Expr::bin(*op, a.substitute(f)?, b.substitute(f)?)?,
            Node::Call(n, args) => {
                let mut out = Vec::with_capacity(args.len());
                for a in args {
                    out.push(a.substitute(f)?);
                }
                Expr::call(n.clone(), out)?
            }
            Node::If(c, a, b) => Expr::cond(c.substitute(f)?, a.substitute(f)?, b.substitute(f)?)?,
        })
    }

    /// Calls `f` with every variable name, in reading order.
    pub fn visit_vars(&self, f: &mut dyn FnMut(&str)) {
        match &self.node {
            Node::Num(_) => {}
            Node::Var(n) => f(n),
            Node::Neg(e) | Node::Not(e) => e.visit_vars(f),
            Node::Bin(_, a, b) => {
                a.visit_vars(f);
                b.visit_vars(f);
            }
            Node::Call(_, args) => args.iter().for_each(|a| a.visit_vars(f)),
            Node::If(c, a, b) => {
                c.visit_vars(f);
                a.visit_vars(f);
                b.visit_vars(f);
            }
        }
    }

    /// The `.ode` text, fully parenthesized. `name` maps a source name to its `.ode` name.
    pub fn to_ode(&self, name: &dyn Fn(&str) -> String) -> String {
        let mut out = String::new();
        self.print(&mut out, name);
        out
    }

    fn print(&self, out: &mut String, name: &dyn Fn(&str) -> String) {
        match &self.node {
            Node::Num(v) => {
                let s = pc_ode::format_number(*v);
                if s.starts_with('-') {
                    out.push('(');
                    out.push_str(&s);
                    out.push(')');
                } else {
                    out.push_str(&s);
                }
            }
            Node::Var(n) => out.push_str(&name(n)),
            Node::Neg(e) => {
                out.push_str("(-");
                e.print(out, name);
                out.push(')');
            }
            Node::Not(e) => {
                out.push_str("(not (");
                e.print(out, name);
                out.push_str("))");
            }
            Node::Bin(op, a, b) => {
                out.push('(');
                a.print(out, name);
                out.push_str(op.text());
                b.print(out, name);
                out.push(')');
            }
            Node::Call(n, args) => {
                out.push_str(n);
                out.push('(');
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    a.print(out, name);
                }
                out.push(')');
            }
            Node::If(c, a, b) => {
                out.push_str("(if(");
                c.print(out, name);
                out.push_str(")then(");
                a.print(out, name);
                out.push_str(")else(");
                b.print(out, name);
                out.push_str("))");
            }
        }
    }
}
