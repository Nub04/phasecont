//! MathML (the subset SBML uses) to an expression tree.
//!
//! Every operator is translated with a single copy of each argument, so the result is never
//! larger than a small multiple of the XML. The n-ary operators build a balanced tree, which keeps
//! a sum of thousands of terms within the depth limit. A sum is therefore added in a different
//! order than the file lists it, which can change the last bit of the result.
//!
//! What the importer does not translate is refused by name, with the position of the element:
//! `delay`, `rateOf`, `avogadro`, chained comparisons, `xor`, `factorial`, `quotient`, `rem`, the
//! inverse hyperbolic functions and `arcsec`, `arccsc`, `arccot` and friends, whose branches this
//! importer does not fix. A `piecewise` without `otherwise` is refused because its value is
//! undefined when no piece holds, and the importer does not pick a value.

use roxmltree::Node as X;

use crate::error::{ImportError, Kind, Result};
use crate::expr::{Expr, Op};
use crate::names::TIME;

/// Deepest element nesting the converter follows.
const MAX_NEST: u32 = 200;

pub(crate) fn err(n: X, kind: Kind, msg: impl Into<String>) -> ImportError {
    let pos = n.document().text_pos_at(n.range().start);
    ImportError::at(kind, msg, pos.row, pos.col)
}

/// Converts the first expression under a `<math>` element.
///
/// `rename` maps an identifier in the file to the name the expression should hold (local
/// parameters get their own names there). Warnings go into `warnings`.
pub fn convert(
    math: X,
    rename: &dyn Fn(&str) -> String,
    warnings: &mut Vec<String>,
) -> Result<Expr> {
    let mut kids = math.children().filter(X::is_element);
    let Some(first) = kids.next() else {
        return Err(err(math, Kind::Syntax, "a `math` element is empty"));
    };
    if kids.next().is_some() {
        return Err(err(
            math,
            Kind::Syntax,
            "a `math` element holds more than one expression",
        ));
    }
    Conv { rename, warnings }.expr(first, 1)
}

/// Combines `items` with `op` in a balanced tree. Returns `None` for an empty list.
pub fn balanced(op: Op, mut items: Vec<Expr>) -> Result<Option<Expr>> {
    while items.len() > 1 {
        let mut next = Vec::with_capacity(items.len().div_ceil(2));
        let mut it = items.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(Expr::bin(op, a, b)?),
                None => next.push(a),
            }
        }
        items = next;
    }
    Ok(items.pop())
}

struct Conv<'a> {
    rename: &'a dyn Fn(&str) -> String,
    warnings: &'a mut Vec<String>,
}

fn text(n: X) -> String {
    n.children()
        .filter(X::is_text)
        .filter_map(|c| c.text())
        .collect()
}

/// The arguments of an `apply`: everything after the operator except the `degree` and `logbase`
/// qualifiers of `root` and `log`.
fn plain<'a, 'i>(rest: &[X<'a, 'i>]) -> Vec<X<'a, 'i>> {
    rest.iter()
        .copied()
        .filter(|c| !matches!(c.tag_name().name(), "degree" | "logbase"))
        .collect()
}

fn one<'a, 'i>(n: X, name: &str, args: &[X<'a, 'i>]) -> Result<X<'a, 'i>> {
    match args {
        [a] => Ok(*a),
        _ => Err(err(n, Kind::Syntax, format!("`{name}` takes one argument"))),
    }
}

fn two<'a, 'i>(n: X, name: &str, args: &[X<'a, 'i>]) -> Result<(X<'a, 'i>, X<'a, 'i>)> {
    match args {
        [a, b] => Ok((*a, *b)),
        _ => Err(err(
            n,
            Kind::Syntax,
            format!("`{name}` takes two arguments"),
        )),
    }
}

impl Conv<'_> {
    fn expr(&mut self, n: X, depth: u32) -> Result<Expr> {
        if depth > MAX_NEST {
            return Err(err(n, Kind::TooLarge, "the math is nested too deeply"));
        }
        match n.tag_name().name() {
            "cn" => self.cn(n),
            "ci" => {
                let id = text(n);
                let id = id.trim();
                if id.is_empty() {
                    return Err(err(n, Kind::Syntax, "a `ci` element has no identifier"));
                }
                Ok(Expr::var((self.rename)(id)))
            }
            "csymbol" => Self::csymbol(n),
            "true" => Expr::num(1.0),
            "false" => Expr::num(0.0),
            "pi" => Expr::num(std::f64::consts::PI),
            "exponentiale" => Expr::num(std::f64::consts::E),
            "infinity" | "notanumber" => Err(err(
                n,
                Kind::Unsupported,
                format!("`{}` has no `.ode` literal", n.tag_name().name()),
            )),
            "apply" => self.apply(n, depth),
            "piecewise" => self.piecewise(n, depth),
            "semantics" => {
                let inner = n.children().find(|c| {
                    c.is_element()
                        && !matches!(c.tag_name().name(), "annotation" | "annotation-xml")
                });
                match inner {
                    Some(c) => self.expr(c, depth + 1),
                    None => Err(err(n, Kind::Syntax, "a `semantics` element is empty")),
                }
            }
            other => Err(err(
                n,
                Kind::Unsupported,
                format!("the MathML element `{other}` is not translated"),
            )),
        }
    }

    fn cn(&mut self, n: X) -> Result<Expr> {
        let kind = n.attribute("type").unwrap_or("real");
        // Text before and after each `<sep/>`.
        let mut parts: Vec<String> = vec![String::new()];
        for c in n.children() {
            if c.is_text() {
                parts
                    .last_mut()
                    .expect("parts is never empty")
                    .push_str(c.text().unwrap_or(""));
            } else if c.is_element() && c.tag_name().name() == "sep" {
                parts.push(String::new());
            } else if c.is_element() {
                return Err(err(n, Kind::Syntax, "a `cn` element holds an element"));
            }
        }
        let parse = |s: &str| -> Result<f64> {
            s.trim()
                .parse::<f64>()
                .map_err(|_| err(n, Kind::Syntax, format!("`{}` is not a number", s.trim())))
        };
        let value = match (kind, parts.as_slice()) {
            ("real" | "integer", [v]) => parse(v)?,
            ("e-notation", [m, e]) => {
                // Mantissa and exponent as the file wrote them, joined as `<m>e<e>`.
                let joined = format!("{}e{}", m.trim(), e.trim());
                parse(&joined)?
            }
            ("rational", [a, b]) => {
                let (a, b) = (parse(a)?, parse(b)?);
                if b == 0.0 {
                    return Err(err(n, Kind::Syntax, "a rational number has denominator 0"));
                }
                a / b
            }
            ("real" | "integer" | "e-notation" | "rational", _) => {
                return Err(err(
                    n,
                    Kind::Syntax,
                    format!("a `cn` of type `{kind}` has the wrong number of parts"),
                ));
            }
            (other, _) => {
                return Err(err(
                    n,
                    Kind::Unsupported,
                    format!("a `cn` of type `{other}` is not translated"),
                ));
            }
        };
        Expr::num(value).map_err(|e| err(n, e.kind, e.message))
    }

    fn csymbol(n: X) -> Result<Expr> {
        let url = n.attribute("definitionURL").unwrap_or("");
        let word = url.rsplit('/').next().unwrap_or("");
        match word {
            "time" => Ok(Expr::var(TIME)),
            "delay" | "avogadro" | "rateOf" => Err(err(
                n,
                Kind::Unsupported,
                format!("the SBML symbol `{word}` is not translated"),
            )),
            _ => Err(err(
                n,
                Kind::Unsupported,
                format!("the symbol `{}` is not translated", text(n).trim()),
            )),
        }
    }

    fn args(&mut self, args: &[X], depth: u32) -> Result<Vec<Expr>> {
        args.iter().map(|a| self.expr(*a, depth + 1)).collect()
    }

    /// Dispatches on the operator. The work for each family sits in its own small function, so
    /// the frame that stays on the stack while a nested argument converts is small.
    fn apply<'a, 'i>(&mut self, n: X<'a, 'i>, depth: u32) -> Result<Expr> {
        let mut kids = n.children().filter(X::is_element);
        let Some(op) = kids.next() else {
            return Err(err(n, Kind::Syntax, "an `apply` element has no operator"));
        };
        let rest: Vec<X<'a, 'i>> = kids.collect();
        let name = op.tag_name().name();
        match name {
            "plus" | "times" | "and" | "or" => self.nary(n, name, &rest, depth),
            "minus" => self.minus(n, &rest, depth),
            "divide" | "power" | "eq" | "gt" | "lt" | "geq" | "leq" | "neq" => {
                self.binary(n, name, &rest, depth)
            }
            "not" => {
                let a = one(n, name, &plain(&rest))?;
                Expr::logical_not(self.expr(a, depth + 1)?)
            }
            "root" | "log" => self.qualified(n, name, &rest, depth),
            "exp" | "ln" | "abs" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "arcsin"
            | "arccos" | "arctan" | "floor" | "ceiling" => self.function(n, name, &rest, depth),
            "sec" | "csc" | "cot" | "sech" | "csch" | "coth" => {
                self.reciprocal(n, name, &rest, depth)
            }
            "ci" => Err(err(
                op,
                Kind::Unsupported,
                format!(
                    "a call to the function `{}`: function definitions are not translated",
                    text(op).trim()
                ),
            )),
            "csymbol" => {
                // `delay` and the other SBML symbols that act as functions.
                Self::csymbol(op)?;
                Err(err(
                    op,
                    Kind::Unsupported,
                    "a symbol used as a function is not translated",
                ))
            }
            other => Err(err(
                n,
                Kind::Unsupported,
                format!("the MathML operator `{other}` is not translated"),
            )),
        }
    }

    #[inline(never)]
    fn nary(&mut self, n: X, name: &str, rest: &[X], depth: u32) -> Result<Expr> {
        let (op, empty) = match name {
            "plus" => (Op::Add, 0.0),
            "times" => (Op::Mul, 1.0),
            "and" => (Op::And, 1.0),
            _ => (Op::Or, 0.0),
        };
        let items = self.args(&plain(rest), depth)?;
        match balanced(op, items)? {
            Some(e) => Ok(e),
            None => Expr::num(empty).map_err(|e| err(n, e.kind, e.message)),
        }
    }

    #[inline(never)]
    fn minus(&mut self, n: X, rest: &[X], depth: u32) -> Result<Expr> {
        match plain(rest).as_slice() {
            [a] => Expr::negate(self.expr(*a, depth + 1)?),
            [a, b] => {
                let a = self.expr(*a, depth + 1)?;
                let b = self.expr(*b, depth + 1)?;
                Expr::bin(Op::Sub, a, b)
            }
            _ => Err(err(n, Kind::Syntax, "`minus` takes one or two arguments")),
        }
    }

    #[inline(never)]
    fn binary(&mut self, n: X, name: &str, rest: &[X], depth: u32) -> Result<Expr> {
        let args = plain(rest);
        if args.len() > 2 && matches!(name, "eq" | "gt" | "lt" | "geq" | "leq" | "neq") {
            return Err(err(
                n,
                Kind::Unsupported,
                format!("a chained `{name}` with more than two arguments is not translated"),
            ));
        }
        let (a, b) = two(n, name, &args)?;
        let a = self.expr(a, depth + 1)?;
        let b = self.expr(b, depth + 1)?;
        match name {
            "divide" => Expr::bin(Op::Div, a, b),
            "power" => Expr::bin(Op::Pow, a, b),
            "eq" => Expr::bin(Op::Eq, a, b),
            "gt" => Expr::bin(Op::Gt, a, b),
            "lt" => Expr::bin(Op::Lt, a, b),
            "geq" => Expr::bin(Op::Ge, a, b),
            "leq" => Expr::bin(Op::Le, a, b),
            _ => Expr::logical_not(Expr::bin(Op::Eq, a, b)?),
        }
    }

    /// `root` with an optional `degree` and `log` with an optional `logbase`.
    #[inline(never)]
    fn qualified(&mut self, n: X, name: &str, rest: &[X], depth: u32) -> Result<Expr> {
        let is_root = name == "root";
        let x = self.expr(one(n, name, &plain(rest))?, depth + 1)?;
        let tag = if is_root { "degree" } else { "logbase" };
        let Some(q) = rest.iter().copied().find(|c| c.tag_name().name() == tag) else {
            return Expr::call(if is_root { "sqrt" } else { "log10" }, vec![x]);
        };
        let inner = q
            .children()
            .find(X::is_element)
            .ok_or_else(|| err(q, Kind::Syntax, format!("a `{tag}` element is empty")))?;
        let k = self.expr(inner, depth + 1)?;
        if is_root {
            if k.node == crate::expr::Node::Num(2.0) {
                Expr::call("sqrt", vec![x])
            } else {
                Expr::bin(Op::Pow, x, Expr::bin(Op::Div, Expr::num(1.0)?, k)?)
            }
        } else if k.node == crate::expr::Node::Num(10.0) {
            Expr::call("log10", vec![x])
        } else {
            Expr::bin(
                Op::Div,
                Expr::call("ln", vec![x])?,
                Expr::call("ln", vec![k])?,
            )
        }
    }

    #[inline(never)]
    fn function(&mut self, n: X, name: &str, rest: &[X], depth: u32) -> Result<Expr> {
        let x = self.expr(one(n, name, &plain(rest))?, depth + 1)?;
        let target = match name {
            "arcsin" => "asin",
            "arccos" => "acos",
            "arctan" => "atan",
            "floor" => "flr",
            "ceiling" => {
                let w = "`ceil` is not accepted by XPPAUT 8.0 (PhaseCont accepts it)";
                if !self.warnings.iter().any(|s| s == w) {
                    self.warnings.push(w.to_string());
                }
                "ceil"
            }
            other => other,
        };
        Expr::call(target, vec![x])
    }

    #[inline(never)]
    fn reciprocal(&mut self, n: X, name: &str, rest: &[X], depth: u32) -> Result<Expr> {
        let x = self.expr(one(n, name, &plain(rest))?, depth + 1)?;
        let base = match name {
            "sec" => "cos",
            "csc" => "sin",
            "cot" => "tan",
            "sech" => "cosh",
            "csch" => "sinh",
            _ => "tanh",
        };
        Expr::bin(Op::Div, Expr::num(1.0)?, Expr::call(base, vec![x])?)
    }

    fn piecewise(&mut self, n: X, depth: u32) -> Result<Expr> {
        let mut pieces: Vec<(Expr, Expr)> = Vec::new();
        let mut otherwise: Option<Expr> = None;
        for c in n.children().filter(X::is_element) {
            match c.tag_name().name() {
                "piece" => {
                    let kids: Vec<X> = c.children().filter(X::is_element).collect();
                    let [v, cond] = kids.as_slice() else {
                        return Err(err(
                            c,
                            Kind::Syntax,
                            "a `piece` holds a value and a condition",
                        ));
                    };
                    let v = self.expr(*v, depth + 2)?;
                    let cond = self.expr(*cond, depth + 2)?;
                    pieces.push((cond, v));
                }
                "otherwise" => {
                    if otherwise.is_some() {
                        return Err(err(c, Kind::Syntax, "a `piecewise` has two `otherwise`"));
                    }
                    let Some(inner) = c.children().find(X::is_element) else {
                        return Err(err(c, Kind::Syntax, "an `otherwise` element is empty"));
                    };
                    otherwise = Some(self.expr(inner, depth + 2)?);
                }
                other => {
                    return Err(err(
                        c,
                        Kind::Syntax,
                        format!("`{other}` inside a `piecewise`"),
                    ));
                }
            }
        }
        let Some(mut acc) = otherwise else {
            return Err(err(
                n,
                Kind::Unsupported,
                "a `piecewise` without `otherwise` has no value when no piece holds; \
                 add an `otherwise` branch",
            ));
        };
        for (cond, v) in pieces.into_iter().rev() {
            acc = Expr::cond(cond, v, acc)?;
        }
        Ok(acc)
    }
}
