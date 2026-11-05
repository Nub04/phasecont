//! Translates a MATLAB function handle or a MATLAB function file into a [`crate::emit::OdeModel`].
//!
//! The handle is read as text and never called. Only the subset below is translated; anything
//! else is refused with its position, so a model never runs with a guessed meaning.
//!
//! - Handle: `@(t,x)` or `@(t,x,p)` followed by one expression or one vector `[e1; e2; ...]`.
//! - File: `function dydt = f(t,y,p)` with scalar assignments and `dydt = [...]` or
//!   `dydt(i) = ...`. No `if`, loop or call to a user function.
//! - Values: numbers, `y(i)`, `p(k)`, captured variables, scalar locals, `pi`.
//! - Operators: `+ - * / ^`, their element-wise forms, comparisons, `& && | || ~`.
//! - Functions: see [`map_function`].

mod lex;

use std::collections::{HashMap, HashSet};

use crate::emit::{OdeModel, State};
use crate::error::{ImportError, Kind, Result, line_col};
use crate::expr::{Expr, Node, Op};
use crate::names::TIME;
use lex::{T, Tok, lex};

/// Largest input text the translator reads.
pub const MAX_INPUT: usize = 1 << 20;
/// Deepest nesting of brackets, calls and prefix operators.
const MAX_NEST: u32 = 64;

/// What the caller supplies besides the text.
#[derive(Debug, Default)]
pub struct FunctionInput<'a> {
    /// The text of `func2str(h)`, or of a function file.
    pub text: &'a str,
    /// Variables the handle captured from its workspace, by name. A scalar has one value.
    pub captures: &'a [(String, Vec<f64>)],
    /// Names of the states. Without them state `i` is named after the state argument, as `x1`.
    pub state_names: Option<&'a [String]>,
    /// Parameters, in the order `p(1)`, `p(2)`, ... read them.
    pub params: &'a [(String, f64)],
    /// One initial value per state.
    pub init: &'a [f64],
    /// Written into a comment of the `.ode` text.
    pub title: Option<&'a str>,
}

/// The functions the translator maps, as MATLAB name to `.ode` name and argument count.
pub fn map_function(name: &str) -> Option<(&'static str, usize)> {
    Some(match name {
        "sin" => ("sin", 1),
        "cos" => ("cos", 1),
        "tan" => ("tan", 1),
        "asin" => ("asin", 1),
        "acos" => ("acos", 1),
        "atan" => ("atan", 1),
        "sinh" => ("sinh", 1),
        "cosh" => ("cosh", 1),
        "tanh" => ("tanh", 1),
        "exp" => ("exp", 1),
        "sqrt" => ("sqrt", 1),
        "abs" => ("abs", 1),
        "sign" => ("sign", 1),
        "erf" => ("erf", 1),
        "erfc" => ("erfc", 1),
        "log" => ("ln", 1),
        "log10" => ("log10", 1),
        "floor" => ("flr", 1),
        "ceil" => ("ceil", 1),
        "gammaln" => ("lgamma", 1),
        "atan2" => ("atan2", 2),
        "min" => ("min", 2),
        "max" => ("max", 2),
        _ => return None,
    })
}

const SUPPORTED: &str = "sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt abs \
                         sign floor ceil min max erf erfc gammaln";

const STATEMENT_WORDS: [&str; 18] = [
    "if",
    "elseif",
    "else",
    "for",
    "parfor",
    "while",
    "switch",
    "case",
    "otherwise",
    "try",
    "catch",
    "return",
    "global",
    "persistent",
    "break",
    "continue",
    "function",
    "spmd",
];

pub fn translate(input: &FunctionInput<'_>) -> Result<(OdeModel, Vec<String>)> {
    if input.text.len() > MAX_INPUT {
        return Err(ImportError::new(
            Kind::TooLarge,
            format!("the function text is longer than {MAX_INPUT} bytes"),
        ));
    }
    let toks = lex(input.text)?;
    let mut p = Parser::new(input, toks);
    let first = p.skip_newlines();
    let body = match &p.toks[first].t {
        T::At => p.handle()?,
        T::Id(w) if w == "function" => p.function_file()?,
        _ => {
            return Err(p.err_at(
                first,
                Kind::Syntax,
                "expected a function handle `@(t,x) ...` or a file that starts with `function`",
            ));
        }
    };
    p.finish(body)
}

/// What a parsed function body gives.
struct Body {
    state_arg: String,
    /// One expression per state.
    rhs: Vec<Expr>,
    fixed: Vec<(String, Expr)>,
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    pos: usize,
    nest: u32,
    input: &'a FunctionInput<'a>,
    captures: HashMap<&'a str, &'a [f64]>,
    time: Option<String>,
    state_arg: String,
    param_arg: Option<String>,
    out_var: Option<String>,
    locals: HashSet<String>,
    /// True right inside `[...]`, where a blank can separate two elements.
    in_vector: bool,
    max_state: usize,
    warnings: Vec<String>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a FunctionInput<'a>, toks: Vec<Tok>) -> Parser<'a> {
        Parser {
            src: input.text,
            toks,
            pos: 0,
            nest: 0,
            input,
            captures: input
                .captures
                .iter()
                .map(|(n, v)| (n.as_str(), v.as_slice()))
                .collect(),
            time: None,
            state_arg: String::new(),
            param_arg: None,
            out_var: None,
            locals: HashSet::new(),
            in_vector: false,
            max_state: 0,
            warnings: Vec::new(),
        }
    }

    // ---- token helpers -------------------------------------------------------------------

    fn err_at(&self, tok: usize, kind: Kind, msg: impl Into<String>) -> ImportError {
        let at = self.toks.get(tok).map_or(self.src.len(), |t| t.at);
        let (l, c) = line_col(self.src, at);
        ImportError::at(kind, msg, l, c)
    }

    fn err_here(&self, kind: Kind, msg: impl Into<String>) -> ImportError {
        self.err_at(self.pos, kind, msg)
    }

    fn peek(&self) -> &T {
        &self.toks[self.pos].t
    }

    fn bump(&mut self) -> usize {
        let i = self.pos;
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        i
    }

    fn eat(&mut self, t: &T) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &T, what: &str) -> Result<()> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(self.err_here(Kind::Syntax, format!("expected {what}")))
        }
    }

    fn skip_newlines(&mut self) -> usize {
        while matches!(self.peek(), T::Newline | T::Semi) {
            self.bump();
        }
        self.pos
    }

    fn enter(&mut self) -> Result<()> {
        self.nest += 1;
        if self.nest > MAX_NEST {
            return Err(self.err_here(
                Kind::TooLarge,
                format!("nested more than {MAX_NEST} levels deep"),
            ));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.nest -= 1;
    }

    // ---- headers -------------------------------------------------------------------------

    /// Reads `(a, b, c)` and returns the names. `~` is an unused argument and gives `None`.
    fn arg_list(&mut self) -> Result<Vec<Option<String>>> {
        self.expect(&T::LParen, "`(`")?;
        let mut args = Vec::new();
        if self.eat(&T::RParen) {
            return Ok(args);
        }
        loop {
            match self.peek().clone() {
                T::Id(n) => {
                    self.bump();
                    args.push(Some(n));
                }
                T::Not => {
                    self.bump();
                    args.push(None);
                }
                _ => return Err(self.err_here(Kind::Syntax, "expected an argument name")),
            }
            if self.eat(&T::RParen) {
                return Ok(args);
            }
            self.expect(&T::Comma, "`,` or `)`")?;
        }
    }

    fn set_args(&mut self, at: usize, args: Vec<Option<String>>) -> Result<()> {
        if !(2..=3).contains(&args.len()) {
            return Err(self.err_at(
                at,
                Kind::Unsupported,
                format!(
                    "the function has {} arguments; the translated form is (t, y) or (t, y, p)",
                    args.len()
                ),
            ));
        }
        let Some(y) = args[1].clone() else {
            return Err(self.err_at(at, Kind::Syntax, "the state argument cannot be `~`"));
        };
        self.time = args[0].clone();
        self.state_arg = y;
        self.param_arg = args.get(2).cloned().flatten();
        let mut seen = HashSet::new();
        for n in args.iter().flatten() {
            if !seen.insert(n.clone()) {
                return Err(self.err_at(
                    at,
                    Kind::Syntax,
                    format!("the argument `{n}` appears twice"),
                ));
            }
        }
        Ok(())
    }

    // ---- function handle -----------------------------------------------------------------

    fn handle(&mut self) -> Result<Body> {
        self.bump(); // @
        let at = self.pos;
        if !matches!(self.peek(), T::LParen) {
            return Err(self.err_here(
                Kind::Unsupported,
                "a handle to a named function is not translated; use the function file instead",
            ));
        }
        let args = self.arg_list()?;
        self.set_args(at, args)?;
        let rhs = self.vector_or_scalar()?;
        self.skip_newlines();
        if !matches!(self.peek(), T::Eof) {
            return Err(self.err_here(
                Kind::Unsupported,
                "text after the end of the expression (vector arithmetic is not translated)",
            ));
        }
        Ok(Body {
            state_arg: self.state_arg.clone(),
            rhs,
            fixed: Vec::new(),
        })
    }

    /// Either `[e1; e2; ...]` (optionally transposed) or a single expression.
    fn vector_or_scalar(&mut self) -> Result<Vec<Expr>> {
        if matches!(self.peek(), T::LBrack) {
            let v = self.vector()?;
            while self.eat(&T::Quote) {}
            return Ok(v);
        }
        Ok(vec![self.expr()?])
    }

    fn vector(&mut self) -> Result<Vec<Expr>> {
        self.enter()?;
        self.bump(); // [
        let saved = std::mem::replace(&mut self.in_vector, true);
        let mut out = Vec::new();
        loop {
            while matches!(self.peek(), T::Semi | T::Comma | T::Newline) {
                self.bump();
            }
            if self.eat(&T::RBrack) {
                break;
            }
            out.push(self.expr()?);
            if !matches!(self.peek(), T::Semi | T::Comma | T::Newline | T::RBrack) {
                return Err(self.err_here(
                    Kind::Syntax,
                    "expected `;`, `,` or `]` between the elements of a vector",
                ));
            }
        }
        self.in_vector = saved;
        self.leave();
        if out.is_empty() {
            return Err(self.err_here(Kind::Syntax, "an empty vector `[]`"));
        }
        Ok(out)
    }

    // ---- function file -------------------------------------------------------------------

    fn function_file(&mut self) -> Result<Body> {
        self.bump(); // function
        let out_at = self.pos;
        let T::Id(out) = self.peek().clone() else {
            return Err(self.err_here(
                Kind::Syntax,
                "expected the output name: `function dydt = name(t,y,p)`",
            ));
        };
        self.bump();
        self.expect(&T::Assign, "`=` after the output name")?;
        let T::Id(_) = self.peek().clone() else {
            return Err(self.err_here(Kind::Syntax, "expected the function name"));
        };
        self.bump();
        let at = self.pos;
        let args = self.arg_list()?;
        self.set_args(at, args)?;
        if [
            self.time.as_deref(),
            Some(&self.state_arg),
            self.param_arg.as_deref(),
        ]
        .iter()
        .flatten()
        .any(|a| *a == out)
        {
            return Err(self.err_at(
                out_at,
                Kind::Syntax,
                "the output has the name of an argument",
            ));
        }
        self.out_var = Some(out.clone());

        let mut slots: Vec<Option<Expr>> = Vec::new();
        let mut preallocated = false;
        let mut whole = false;
        let mut fixed: Vec<(String, Expr)> = Vec::new();
        loop {
            self.skip_newlines();
            let st = self.pos;
            let T::Id(word) = self.peek().clone() else {
                if matches!(self.peek(), T::Eof) {
                    break;
                }
                return Err(self.err_here(Kind::Syntax, "expected a statement"));
            };
            if word == "end" {
                self.bump();
                self.skip_newlines();
                if !matches!(self.peek(), T::Eof) {
                    return Err(self.err_here(
                        Kind::Unsupported,
                        "more than one function in the file is not translated",
                    ));
                }
                break;
            }
            if STATEMENT_WORDS.contains(&word.as_str()) {
                return Err(self.err_here(Kind::Unsupported, statement_hint(&word)));
            }
            self.bump();
            match self.peek().clone() {
                T::Assign => {
                    self.bump();
                    if word == out {
                        if whole || !slots.is_empty() || preallocated {
                            return Err(self.err_at(
                                st,
                                Kind::Unsupported,
                                format!("`{out}` is assigned twice"),
                            ));
                        }
                        if let Some(n) = self.zeros()? {
                            slots = vec![None; n];
                            preallocated = true;
                        } else {
                            slots = self.vector_or_scalar()?.into_iter().map(Some).collect();
                            whole = true;
                        }
                    } else {
                        let e = self.expr()?;
                        self.declare_local(st, &word)?;
                        fixed.push((word, e));
                    }
                }
                T::LParen if word == out => {
                    self.bump();
                    let T::Num(v) = self.peek().clone() else {
                        return Err(self.err_here(
                            Kind::Unsupported,
                            "the index of the output must be a whole number written as a literal",
                        ));
                    };
                    let i = self.literal_index(v, "the index of the output")?;
                    self.bump();
                    self.expect(&T::RParen, "`)`")?;
                    self.expect(&T::Assign, "`=`")?;
                    if whole {
                        return Err(self.err_at(
                            st,
                            Kind::Unsupported,
                            format!("`{out}` is assigned twice"),
                        ));
                    }
                    if i > slots.len() {
                        if preallocated {
                            return Err(self.err_at(
                                st,
                                Kind::Mismatch,
                                format!("`{out}({i})` is past the size given to `zeros`"),
                            ));
                        }
                        slots.resize(i, None);
                    }
                    let e = self.expr()?;
                    if slots[i - 1].is_some() {
                        return Err(self.err_at(
                            st,
                            Kind::Unsupported,
                            format!("`{out}({i})` is assigned twice"),
                        ));
                    }
                    slots[i - 1] = Some(e);
                }
                _ => {
                    return Err(self.err_at(
                        st,
                        Kind::Unsupported,
                        format!("`{word}` starts something other than an assignment of a scalar"),
                    ));
                }
            }
            if !matches!(self.peek(), T::Semi | T::Newline | T::Eof) {
                return Err(self.err_here(Kind::Syntax, "expected the end of the statement"));
            }
        }
        if slots.is_empty() {
            return Err(self.err_here(
                Kind::Mismatch,
                format!("the function never assigns its output `{out}`"),
            ));
        }
        let mut rhs = Vec::with_capacity(slots.len());
        for (i, s) in slots.into_iter().enumerate() {
            match s {
                Some(e) => rhs.push(e),
                None => {
                    return Err(ImportError::new(
                        Kind::Mismatch,
                        format!("`{out}({})` is never assigned", i + 1),
                    ));
                }
            }
        }
        Ok(Body {
            state_arg: self.state_arg.clone(),
            rhs,
            fixed,
        })
    }

    /// `zeros(n,1)`, `zeros(1,n)` or `zeros(n)`: the size of a preallocated output.
    fn zeros(&mut self) -> Result<Option<usize>> {
        if !matches!(self.peek(), T::Id(w) if w == "zeros") {
            return Ok(None);
        }
        self.bump();
        self.expect(&T::LParen, "`(` after `zeros`")?;
        let mut dims = Vec::new();
        loop {
            let T::Num(v) = self.peek().clone() else {
                return Err(self.err_here(Kind::Unsupported, "`zeros` takes whole-number literals"));
            };
            dims.push(self.literal_index(v, "a size of `zeros`")?);
            self.bump();
            if self.eat(&T::RParen) {
                break;
            }
            self.expect(&T::Comma, "`,` or `)`")?;
        }
        match dims.as_slice() {
            [n] | [n, 1] | [1, n] => Ok(Some(*n)),
            _ => Err(self.err_here(Kind::Unsupported, "the output must be a vector")),
        }
    }

    fn declare_local(&mut self, at: usize, name: &str) -> Result<()> {
        let taken = Some(name) == self.time.as_deref()
            || name == self.state_arg
            || Some(name) == self.param_arg.as_deref()
            || self.captures.contains_key(name);
        if taken {
            return Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("`{name}` is an argument and cannot be assigned"),
            ));
        }
        if !self.locals.insert(name.to_string()) {
            return Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("`{name}` is assigned twice; give the second value a new name"),
            ));
        }
        Ok(())
    }

    fn literal_index(&self, v: f64, what: &str) -> Result<usize> {
        if v.fract() != 0.0 || !(1.0..=1e6).contains(&v) {
            return Err(self.err_here(
                Kind::Unsupported,
                format!("{what} must be a whole number of at least 1, written as a literal"),
            ));
        }
        Ok(v as usize)
    }

    // ---- expressions ---------------------------------------------------------------------

    fn expr(&mut self) -> Result<Expr> {
        self.or_or()
    }

    fn or_or(&mut self) -> Result<Expr> {
        let mut l = self.and_and()?;
        while self.eat(&T::OrOr) {
            let r = self.and_and()?;
            l = Expr::bin(Op::Or, l, r)?;
        }
        Ok(l)
    }

    fn and_and(&mut self) -> Result<Expr> {
        let mut l = self.or()?;
        while self.eat(&T::AndAnd) {
            let r = self.or()?;
            l = Expr::bin(Op::And, l, r)?;
        }
        Ok(l)
    }

    fn or(&mut self) -> Result<Expr> {
        let mut l = self.and()?;
        while self.eat(&T::Or) {
            let r = self.and()?;
            l = Expr::bin(Op::Or, l, r)?;
        }
        Ok(l)
    }

    fn and(&mut self) -> Result<Expr> {
        let mut l = self.compare()?;
        while self.eat(&T::And) {
            let r = self.compare()?;
            l = Expr::bin(Op::And, l, r)?;
        }
        Ok(l)
    }

    fn compare(&mut self) -> Result<Expr> {
        let mut l = self.additive()?;
        loop {
            let (op, negate) = match self.peek() {
                T::Lt => (Op::Lt, false),
                T::Le => (Op::Le, false),
                T::Gt => (Op::Gt, false),
                T::Ge => (Op::Ge, false),
                T::EqEq => (Op::Eq, false),
                T::Ne => (Op::Eq, true),
                _ => return Ok(l),
            };
            self.bump();
            let r = self.additive()?;
            l = Expr::bin(op, l, r)?;
            if negate {
                l = Expr::logical_not(l)?;
            }
        }
    }

    /// True when the `+` or `-` at the cursor starts a new vector element: a blank before it and
    /// none after, as in `[a -b]`.
    fn splits_vector(&self) -> bool {
        self.in_vector
            && self.toks[self.pos].ws
            && !self.toks.get(self.pos + 1).is_some_and(|t| t.ws)
    }

    fn additive(&mut self) -> Result<Expr> {
        let mut l = self.term()?;
        loop {
            let op = match self.peek() {
                T::Plus => Op::Add,
                T::Minus => Op::Sub,
                _ => return Ok(l),
            };
            if self.splits_vector() {
                return Err(self.err_here(
                    Kind::Unsupported,
                    "a blank before a sign inside `[...]` can mean a new element; \
                     separate the elements with `;`",
                ));
            }
            self.bump();
            let r = self.term()?;
            l = Expr::bin(op, l, r)?;
        }
    }

    fn term(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                T::Star => Op::Mul,
                T::Slash => Op::Div,
                _ => return Ok(l),
            };
            self.bump();
            let r = self.unary()?;
            l = Expr::bin(op, l, r)?;
        }
    }

    fn unary(&mut self) -> Result<Expr> {
        match self.peek() {
            T::Minus => {
                self.bump();
                self.enter()?;
                let e = self.unary()?;
                self.leave();
                negate(e)
            }
            T::Plus => {
                self.bump();
                self.enter()?;
                let e = self.unary()?;
                self.leave();
                Ok(e)
            }
            T::Not => {
                self.bump();
                self.enter()?;
                let e = self.unary()?;
                self.leave();
                Expr::logical_not(e)
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr> {
        let mut base = self.primary()?;
        while self.eat(&T::Quote) {}
        while self.eat(&T::Caret) {
            let exp = self.power_operand()?;
            base = Expr::bin(Op::Pow, base, exp)?;
        }
        Ok(base)
    }

    /// The exponent: a sign is allowed (`2^-x`), another `^` is not part of it.
    fn power_operand(&mut self) -> Result<Expr> {
        match self.peek() {
            T::Minus => {
                self.bump();
                self.enter()?;
                let e = self.power_operand()?;
                self.leave();
                negate(e)
            }
            T::Plus => {
                self.bump();
                self.enter()?;
                let e = self.power_operand()?;
                self.leave();
                Ok(e)
            }
            _ => {
                let e = self.primary()?;
                while self.eat(&T::Quote) {}
                Ok(e)
            }
        }
    }

    fn primary(&mut self) -> Result<Expr> {
        let at = self.pos;
        match self.peek().clone() {
            T::Num(v) => {
                self.bump();
                Ok(Expr::num(v).map_err(|e| self.relocate(e, at))?)
            }
            T::LParen => {
                self.bump();
                self.enter()?;
                let saved = std::mem::replace(&mut self.in_vector, false);
                let e = self.expr()?;
                self.in_vector = saved;
                self.leave();
                self.expect(&T::RParen, "`)`")?;
                Ok(e)
            }
            T::LBrack => Err(self.err_here(
                Kind::Unsupported,
                "a vector inside an expression is not translated; only the whole result may be a vector",
            )),
            T::Id(name) => {
                self.bump();
                if matches!(self.peek(), T::LParen) {
                    self.indexed_or_call(at, &name)
                } else {
                    self.bare(at, &name)
                }
            }
            T::Eof | T::Newline | T::Semi => {
                Err(self.err_here(Kind::Syntax, "the expression ends too early"))
            }
            _ => Err(self.err_here(Kind::Syntax, "expected a value")),
        }
    }

    fn relocate(&self, e: ImportError, tok: usize) -> ImportError {
        if e.line.is_some() {
            return e;
        }
        self.err_at(tok, e.kind, e.message)
    }

    fn state_name(&mut self, at: usize, i: usize) -> Result<String> {
        if let Some(names) = self.input.state_names {
            return match names.get(i - 1) {
                Some(n) => Ok(n.clone()),
                None => Err(self.err_at(
                    at,
                    Kind::Mismatch,
                    format!(
                        "`{}({i})` is read, but only {} state names were given",
                        self.state_arg,
                        names.len()
                    ),
                )),
            };
        }
        self.max_state = self.max_state.max(i);
        Ok(format!("{}{i}", self.state_arg))
    }

    /// A name without a following `(`.
    fn bare(&mut self, at: usize, name: &str) -> Result<Expr> {
        if self.time.as_deref() == Some(name) {
            return Ok(Expr::var(TIME));
        }
        if name == self.state_arg || self.param_arg.as_deref() == Some(name) {
            return Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("`{name}` is a vector; write `{name}(1)`, `{name}(2)`, ..."),
            ));
        }
        if let Some(v) = self.captures.get(name) {
            return match v {
                [one] => Expr::num(*one).map_err(|e| self.relocate(e, at)),
                _ => Err(self.err_at(
                    at,
                    Kind::Unsupported,
                    format!(
                        "`{name}` is a vector with {} values; index it as `{name}(1)`",
                        v.len()
                    ),
                )),
            };
        }
        if self.locals.contains(name) {
            return Ok(Expr::var(name));
        }
        match name {
            "pi" => Expr::num(std::f64::consts::PI),
            "Inf" | "inf" | "NaN" | "nan" => Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("`{name}` has no `.ode` literal"),
            )),
            "eps" | "i" | "j" => {
                Err(self.err_at(at, Kind::Unsupported, format!("`{name}` is not translated")))
            }
            _ if Some(name) == self.out_var.as_deref() => Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("the output `{name}` cannot be read"),
            )),
            _ => Err(self.err_at(
                at,
                Kind::Undeclared,
                format!(
                    "`{name}` is not an argument, a captured variable or an earlier assignment"
                ),
            )),
        }
    }

    /// `name(`: an index into the states, the parameters or a captured vector, or a call.
    fn indexed_or_call(&mut self, at: usize, name: &str) -> Result<Expr> {
        let is_state = name == self.state_arg;
        let is_param = self.param_arg.as_deref() == Some(name);
        let capture = self.captures.get(name).copied();
        if is_state || is_param || capture.is_some() {
            self.bump(); // (
            let T::Num(v) = self.peek().clone() else {
                return Err(self.err_here(
                    Kind::Unsupported,
                    format!("the index of `{name}` must be a whole number written as a literal"),
                ));
            };
            let i = self.literal_index(v, &format!("the index of `{name}`"))?;
            self.bump();
            self.expect(&T::RParen, "`)`")?;
            if is_state {
                return Ok(Expr::var(self.state_name(at, i)?));
            }
            if is_param {
                return match self.input.params.get(i - 1) {
                    Some((n, _)) => Ok(Expr::var(n.clone())),
                    None => Err(self.err_at(
                        at,
                        Kind::MissingValue,
                        format!(
                            "`{name}({i})` is read, but only {} parameter values were given",
                            self.input.params.len()
                        ),
                    )),
                };
            }
            let vals = capture.unwrap_or(&[]);
            return match vals.get(i - 1) {
                Some(v) => Expr::num(*v).map_err(|e| self.relocate(e, at)),
                None => Err(self.err_at(
                    at,
                    Kind::Mismatch,
                    format!("`{name}({i})` is past the {} values captured", vals.len()),
                )),
            };
        }
        if self.locals.contains(name) || self.time.as_deref() == Some(name) {
            return Err(self.err_at(
                at,
                Kind::Syntax,
                format!("`{name}` is a scalar and cannot be indexed or called"),
            ));
        }
        match name {
            "mod" | "rem" => {
                return Err(self.err_at(
                    at,
                    Kind::Unsupported,
                    format!(
                        "`{name}` is not translated: the `.ode` `mod` differs from MATLAB's for a \
                         negative divisor"
                    ),
                ));
            }
            _ => {}
        }
        let Some((ode, arity)) = map_function(name) else {
            return Err(self.err_at(
                at,
                Kind::Unsupported,
                format!("the function `{name}` is not translated (translated: {SUPPORTED})"),
            ));
        };
        self.bump(); // (
        self.enter()?;
        let saved = std::mem::replace(&mut self.in_vector, false);
        let mut args = Vec::new();
        if !self.eat(&T::RParen) {
            loop {
                args.push(self.expr()?);
                if self.eat(&T::RParen) {
                    break;
                }
                self.expect(&T::Comma, "`,` or `)`")?;
            }
        }
        self.in_vector = saved;
        self.leave();
        if args.len() != arity {
            return Err(self.err_at(
                at,
                Kind::Mismatch,
                format!(
                    "`{name}` takes {arity} argument(s) here, not {}",
                    args.len()
                ),
            ));
        }
        if name == "ceil" {
            self.warnings
                .push("`ceil` is not accepted by XPPAUT 8.0 (PhaseCont accepts it)".to_string());
        }
        Expr::call(ode, args).map_err(|e| self.relocate(e, at))
    }

    // ---- result --------------------------------------------------------------------------

    fn finish(mut self, body: Body) -> Result<(OdeModel, Vec<String>)> {
        let n = body.rhs.len();
        let names: Vec<String> = match self.input.state_names {
            Some(s) => {
                if s.len() != n {
                    return Err(ImportError::new(
                        Kind::Mismatch,
                        format!(
                            "{} state names were given, but the function has {n} equations",
                            s.len()
                        ),
                    ));
                }
                s.to_vec()
            }
            None => {
                if self.max_state > n {
                    return Err(ImportError::new(
                        Kind::Mismatch,
                        format!(
                            "`{}({})` is read, but the function has {n} equations",
                            body.state_arg, self.max_state
                        ),
                    ));
                }
                (1..=n).map(|i| format!("{}{i}", body.state_arg)).collect()
            }
        };
        if self.input.init.is_empty() {
            return Err(ImportError::new(
                Kind::MissingValue,
                format!("initial values are required: give {n}, one per state"),
            ));
        }
        if self.input.init.len() != n {
            return Err(ImportError::new(
                Kind::Mismatch,
                format!(
                    "{} initial values were given for {n} states",
                    self.input.init.len()
                ),
            ));
        }
        // `th = y(1)` with a state called `th` only repeats the name: drop it.
        let mut fixed = body.fixed;
        fixed.retain(|(n, e)| !(matches!(&e.node, Node::Var(v) if v == n) && names.contains(n)));
        let states = names
            .into_iter()
            .zip(body.rhs)
            .zip(self.input.init)
            .map(|((name, rhs), v)| State {
                name,
                rhs,
                init: Some(*v),
            })
            .collect();
        let model = OdeModel {
            title: self
                .input
                .title
                .unwrap_or("imported from MATLAB")
                .to_string(),
            params: self.input.params.to_vec(),
            fixed,
            states,
        };
        Ok((model, std::mem::take(&mut self.warnings)))
    }
}

/// `-e`, folding a literal so `-3` prints as one number.
fn negate(e: Expr) -> Result<Expr> {
    if let Node::Num(v) = e.node {
        return Expr::num(-v);
    }
    Expr::negate(e)
}

fn statement_hint(word: &str) -> String {
    match word {
        "if" | "elseif" | "else" => "`if` is not translated; write the condition as a comparison \
                                      inside an expression, for example `(y(1) > 0) * a`"
            .to_string(),
        "for" | "parfor" | "while" => format!("`{word}` loops are not translated"),
        _ => format!("`{word}` is not translated"),
    }
}
