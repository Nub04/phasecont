//! Pass 2: name resolution and model checks.
//!
//! [`check`] builds the symbol table of a parsed model and reports the
//! errors the rules in `docs/language.md` ("Names and order", "Builtins")
//! describe: undefined, reserved and duplicate names, fixed quantities used
//! above their definition, circular definitions, function arity and
//! recursion, `aux` quantities used in formulas and files that mix ODEs and
//! maps. It also sets the model flags.
//!
//! [`expand`](crate::expand()) runs before `check`, so the model normally
//! has no array templates. If one is left, `check` skips array block
//! bodies, skips declarations whose name carries an index, and never
//! reports a name with an index as undefined. Includes must be resolved
//! too; an `Include` statement left in the model is ignored.
//!
//! A diagnostic about a statement that `expand` copied gets the note "in
//! the copy with j = N of this array line" at the template line. When
//! several copies of one template give the same diagnostic (same code,
//! severity, span and message), only the first one is kept. Name length
//! warnings from copies of one template at one span collapse to the first,
//! with a note that counts the others. Uses of names that a statement
//! dropped by `expand` would have declared ([`Model::suppressed`]) are not
//! reported as undefined.

use std::collections::{HashMap, HashSet};

use crate::ast::{Expr, Ident, MarkovLine, Model, StmtKind, UnaryOp, Unsupported};
use crate::builtins::{Status, builtin};
use crate::diag::{Code, Diagnostic, MAX_DIAGNOSTICS, Sink, with_summary};
use crate::parser::expr::PORTABILITY;
use crate::parser::stmt::UNSUPPORTED_SUM_OF;
use crate::span::{FileId, Span};

/// A name that is used but never declared.
pub const UNDEFINED_NAME: Code = Code(39);
/// A declaration of `t`, `pi`, a builtin, `arg1`..`arg9` or a keyword word.
pub const RESERVED_NAME: Code = Code(40);
/// A name declared twice, including a repeated `par` or function parameter.
pub const DUPLICATE_NAME: Code = Code(41);
/// A fixed quantity that uses one defined on a later line.
pub const FIXED_FORWARD: Code = Code(42);
/// Fixed quantities or derived parameters defined in terms of each other.
pub const CIRCULAR: Code = Code(43);
/// `init x=` or `x(0)=` where `x` is not a state.
pub const INIT_NOT_STATE: Code = Code(44);
/// Two initial values for one state.
pub const INIT_TWICE: Code = Code(45);
/// A user function with more than 9 parameters.
pub const TOO_MANY_PARAMS: Code = Code(46);
/// A user function that calls itself, directly or through others.
pub const RECURSIVE_FUNCTION: Code = Code(47);
/// A call with the wrong number of arguments.
pub const WRONG_ARITY: Code = Code(48);
/// A function used as a value, or a value called as a function.
pub const NOT_CALLABLE: Code = Code(49);
/// `argK` outside a function with at least `K` parameters.
pub const BAD_ARG: Code = Code(50);
/// An `aux` quantity used in a formula.
pub const AUX_IN_FORMULA: Code = Code(51);
/// A file with both ODEs and maps.
pub const MIXED_KIND: Code = Code(52);
/// A derived parameter that uses a state, fixed quantity, `t` or noise.
pub const DERIVED_NON_PARAMETER: Code = Code(53);
/// `delay(x,tau)`, planned for v2.
pub const UNSUPPORTED_DELAY: Code = Code(54);
/// `shift(x,n)`.
pub const UNSUPPORTED_SHIFT: Code = Code(55);
/// `del_shft(...)`.
pub const UNSUPPORTED_DEL_SHFT: Code = Code(56);
/// An undefined name in a model with an unsupported statement that may
/// declare it.
pub const MAYBE_UNDEFINED: Code = Code(57);
/// A `global` assignment to a name that is not a state or a parameter.
pub const BAD_TARGET: Code = Code(58);

/// Longest name XPPAUT 8.0 keeps whole.
const XPP_NAME_LEN: usize = 10;
/// Most parameters a user function may have.
const MAX_PARAMS: usize = 9;

/// What kind of system the model is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// No ODE and no map.
    None,
    Ode,
    Map,
}

/// How a state is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    Ode,
    Map,
    Markov,
}

/// What a declared name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    State(StateKind),
    Parameter,
    Number,
    Derived,
    Fixed,
    Aux,
    Function { params: usize },
    Table,
    Wiener,
}

/// One declared name.
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    /// The name as declared (the first declaration when there are two).
    pub name: Ident,
    pub kind: SymbolKind,
    /// Index of the declaring statement in `Model::statements`.
    pub stmt: usize,
}

/// What [`check`] learns about a model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    /// `Ode` or `Map` after the first ODE or map statement, `None` when
    /// there is neither. A mixed file is reported and keeps the first kind.
    pub kind: ModelKind,
    /// False when the model calls a non-smooth builtin, uses `if`, a
    /// comparison, `&`, `|` or `not`, or has a table.
    pub smooth: bool,
    /// True when the model calls `ran`, `normal` or `poisson`, or has a
    /// `markov`, `wiener` or `global` statement.
    pub simulation_only: bool,
    /// Declared names in declaration order.
    pub symbols: Vec<Symbol>,
    /// Lower-case name to position in `symbols`.
    index: HashMap<String, usize>,
}

impl ModelInfo {
    /// The symbol with the lower-case name `key`.
    pub fn symbol(&self, key: &str) -> Option<&Symbol> {
        self.index.get(key).map(|&s| &self.symbols[s])
    }
}

/// Why `key` cannot be declared, or `None` when it can.
fn reserved_reason(key: &str) -> Option<&'static str> {
    match key {
        "t" => Some("is the time variable"),
        "pi" => Some("is a constant"),
        "not" | "if" | "then" | "else" => Some("is a keyword"),
        _ if arg_number(key).is_some() => Some("names a function argument"),
        _ if builtin(key).is_some() => Some("is a builtin function"),
        _ => None,
    }
}

/// `K` for `arg1` to `arg9`.
fn arg_number(key: &str) -> Option<usize> {
    let rest = key.strip_prefix("arg")?;
    let b = rest.as_bytes();
    (b.len() == 1 && (b'1'..=b'9').contains(&b[0])).then(|| usize::from(b[0] - b'0'))
}

/// Unsupported families whose raw text may declare a name.
fn may_declare(family: Unsupported) -> bool {
    matches!(
        family,
        Unsupported::Volterra
            | Unsupported::Dae
            | Unsupported::Special
            | Unsupported::FileTable
            | Unsupported::SumOf
    )
}

/// A noun for a symbol kind, used in messages.
fn noun(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::State(_) => "a state",
        SymbolKind::Parameter => "a parameter",
        SymbolKind::Number => "a number",
        SymbolKind::Derived => "a derived parameter",
        SymbolKind::Fixed => "a fixed quantity",
        SymbolKind::Aux => "an aux quantity",
        SymbolKind::Function { .. } => "a function",
        SymbolKind::Table => "a table",
        SymbolKind::Wiener => "a Wiener variable",
    }
}

/// The message for `argK` used outside a function with `K` parameters.
fn bad_arg_message(id: &Ident, k: usize) -> String {
    let plural = if k == 1 { "" } else { "s" };
    format!(
        "`{}` names argument {k} only inside a function with at least {k} parameter{plural}",
        id.text
    )
}

/// Checks `model` and returns what it learned plus the diagnostics, sorted
/// by where they start.
///
/// The list holds at most [`MAX_DIAGNOSTICS`] diagnostics; if there were
/// more, an error ends it that counts the rest.
pub fn check(model: &Model) -> (ModelInfo, Vec<Diagnostic>) {
    let (info, diags, dropped) = check_counted(model);
    let file = model.statements.first().map_or(FileId(0), |s| s.span.file);
    (info, with_summary(diags, dropped, file))
}

/// [`check()`] without the summary error; also returns the number of
/// diagnostics that were not stored.
pub(crate) fn check_counted(model: &Model) -> (ModelInfo, Vec<Diagnostic>, usize) {
    let mut c = Checker::new(model);
    c.declare_all();
    c.at(None);
    c.check_inits();
    c.at(None);
    c.resolve_all();
    c.at(None);
    c.report_graph();
    c.report_long_names();
    c.add_collapse_notes();
    let (mut diags, dropped) = c.diags.finish();
    diags.sort_by_key(|d| d.span.start);
    let info = ModelInfo {
        kind: c.kind.map_or(ModelKind::None, |k| k.0),
        smooth: c.smooth,
        simulation_only: c.simulation_only,
        symbols: c.symbols,
        index: c.index,
    };
    (info, diags, dropped)
}

/// The start of the note that names the copy a diagnostic comes from.
const COPY_NOTE: &str = "in the copy with j = ";

/// Where a name is used, which decides what it may refer to.
#[derive(Clone, Copy)]
enum Owner<'a> {
    /// An ODE, map, aux, table, markov or global formula.
    Formula,
    /// The right side of a fixed quantity, with its symbol index when this
    /// statement declared it.
    Fixed(Option<usize>),
    /// The right side of a derived parameter, with its symbol index when
    /// this statement declared it.
    Derived(Option<usize>),
    /// The body of a user function: its symbol index and parameters.
    Function(Option<usize>, &'a [Ident]),
}

/// A use inside a function body that a derived parameter may not reach:
/// the name as written, what it is, and where it is used.
type BadUse = (String, &'static str, Span);

/// Template, code, severity, span and message of a kept copy diagnostic.
type SeenKey = (Option<u32>, Code, u8, (u32, usize, usize), String);
/// Template and span of a name length warning from a copy.
type LongKey = (Option<u32>, (u32, usize, usize));

struct Checker<'a> {
    model: &'a Model,
    diags: Sink,
    symbols: Vec<Symbol>,
    index: HashMap<String, usize>,
    /// The first ODE or map statement.
    kind: Option<(ModelKind, Span)>,
    mixed: bool,
    smooth: bool,
    simulation_only: bool,
    /// The first unsupported statement whose text may declare a name.
    declarer: Option<Span>,
    /// Keys that a statement tried to declare but are reserved.
    reserved_tried: HashSet<String>,
    /// Every global name that was declared without error, for the name
    /// length warning.
    declared: Vec<(Option<usize>, &'a Ident)>,
    /// The parameters of each function that were declared without error,
    /// one group per function, for the name length warning.
    params: Vec<(Option<usize>, Vec<&'a Ident>)>,
    /// Uses between symbols: (user, used, span of the use). Only uses that
    /// can close a cycle are recorded: among fixed quantities, derived
    /// parameters and functions.
    edges: Vec<(usize, usize, Span)>,
    /// For each function symbol, the first use in its body of `t`, a
    /// state, a fixed quantity or a Wiener variable.
    fn_bad: HashMap<usize, BadUse>,
    /// The statement being checked, when it is an array copy: its index in
    /// the model. [`Checker::report`] adds the copy note for it.
    current: Option<usize>,
    /// Diagnostics from copies that were already kept: template, code,
    /// severity, span and message. A repeat is dropped as it is reported.
    seen: HashSet<SeenKey>,
    /// For each (template, span) of a name length warning from a copy: the
    /// position of the kept one in `diags`, how many others it stands for and
    /// the template line.
    long: HashMap<LongKey, (usize, usize, Span)>,
    /// Keys that a statement dropped by `expand` would have declared.
    suppressed_plain: HashSet<String>,
    /// Roots of indexed names that a dropped statement would have declared.
    suppressed_roots: HashSet<String>,
}

impl<'a> Checker<'a> {
    fn new(model: &'a Model) -> Self {
        Checker {
            model,
            diags: Sink::default(),
            symbols: Vec::new(),
            index: HashMap::new(),
            kind: None,
            mixed: false,
            smooth: true,
            simulation_only: false,
            declarer: None,
            reserved_tried: HashSet::new(),
            declared: Vec::new(),
            params: Vec::new(),
            edges: Vec::new(),
            fn_bad: HashMap::new(),
            current: None,
            seen: HashSet::new(),
            long: HashMap::new(),
            suppressed_plain: model
                .suppressed
                .iter()
                .filter(|s| !s.indexed)
                .map(|s| s.key.clone())
                .collect(),
            suppressed_roots: model
                .suppressed
                .iter()
                .filter(|s| s.indexed)
                .map(|s| s.key.clone())
                .collect(),
        }
    }

    /// Reports a diagnostic. One from an array copy is collapsed as it
    /// arrives, so that the copies of one line cannot fill the list:
    ///
    /// - The name length warnings (ODE0013) from copies of one template at
    ///   one span collapse to the first, which gets a note that counts the
    ///   others (see [`Checker::add_collapse_notes`]).
    /// - Of the other diagnostics from copies of one template that agree in
    ///   code, severity, span and message, only the first is kept.
    ///
    /// Diagnostics outside copies are kept.
    fn report(&mut self, code: Code, span: Span, message: String, notes: Vec<(Span, String)>) {
        let mut d = Diagnostic::new(code, span, message);
        d.notes = notes;
        if let Some(i) = self.current {
            let stmt = &self.model.statements[i];
            if let Some(j) = stmt.array_index {
                d.notes
                    .push((stmt.span, format!("{COPY_NOTE}{j} of this array line")));
                let template = stmt.array_template;
                let line = stmt.span;
                let at = (d.span.file.0, d.span.start, d.span.end);
                if d.code == PORTABILITY {
                    if let Some((_, others, _)) = self.long.get_mut(&(template, at)) {
                        *others += 1;
                    } else {
                        if self.diags.len() < MAX_DIAGNOSTICS {
                            self.long
                                .insert((template, at), (self.diags.len(), 0, line));
                        }
                        self.diags.push(d);
                    }
                    return;
                }
                if self.diags.len() >= MAX_DIAGNOSTICS {
                    // Full: count it without remembering it.
                    self.diags.push(d);
                    return;
                }
                let key = (template, d.code, d.severity as u8, at, d.message.clone());
                if self.seen.insert(key) {
                    self.diags.push(d);
                }
                return;
            }
        }
        self.diags.push(d);
    }

    /// Adds to each kept name length warning from copies the note that
    /// counts the other copies it stands for.
    fn add_collapse_notes(&mut self) {
        for (kept, others, line) in std::mem::take(&mut self.long).into_values() {
            if others > 0 {
                let message = if others == 1 {
                    "1 more copy of this array line gives the same warning".to_string()
                } else {
                    format!("{others} more copies of this array line give the same warning")
                };
                self.diags.get_mut(kept).notes.push((line, message));
            }
        }
    }

    /// Makes statement `i` the one whose copy note [`Checker::report`]
    /// adds, or clears it with `None`.
    fn at(&mut self, i: Option<usize>) {
        self.current = i.filter(|&i| self.model.statements[i].array_index.is_some());
    }

    fn lookup(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Records a declaration. Names with an index are skipped; reserved and
    /// repeated names are reported and not added.
    fn declare(&mut self, name: &'a Ident, kind: SymbolKind, stmt: usize) {
        if name.index.is_some() {
            return;
        }
        if let Some(reason) = reserved_reason(&name.key) {
            self.reserved_tried.insert(name.key.clone());
            let message = format!("`{}` {reason} and cannot be declared", name.text);
            self.report(RESERVED_NAME, name.span, message, Vec::new());
            return;
        }
        if let Some(first) = self.lookup(&name.key) {
            let first = &self.symbols[first];
            let mut notes = vec![(
                first.name.span,
                format!("`{}` is first declared here", first.name.text),
            )];
            let message = format!(
                "`{}` is declared twice; it is already {}",
                name.text,
                noun(first.kind)
            );
            // Two copies of one template declare the name from one place.
            let first_stmt = &self.model.statements[first.stmt];
            let this = &self.model.statements[stmt];
            if first_stmt.array_template.is_some()
                && first_stmt.array_template == this.array_template
                && first_stmt.array_index != this.array_index
                && first.name.span == name.span
            {
                notes.push((
                    this.span,
                    format!(
                        "`{}` does not change with the index of this array line, so every \
                         copy declares it again",
                        name.text
                    ),
                ));
            }
            self.report(DUPLICATE_NAME, name.span, message, notes);
            return;
        }
        self.declared.push((self.current, name));
        self.index.insert(name.key.clone(), self.symbols.len());
        self.symbols.push(Symbol {
            name: name.clone(),
            kind,
            stmt,
        });
    }

    fn set_kind(&mut self, kind: ModelKind, span: Span) {
        match self.kind {
            None => self.kind = Some((kind, span)),
            Some((first_kind, first)) if first_kind != kind && !self.mixed => {
                self.mixed = true;
                let what = if first_kind == ModelKind::Ode {
                    "ODE"
                } else {
                    "map"
                };
                let message = "this file has both ODEs and maps; a model must be one or the other"
                    .to_string();
                let note = (first, format!("the first {what} is here"));
                self.report(MIXED_KIND, span, message, vec![note]);
            }
            Some(_) => {}
        }
    }

    /// Pass A: the symbol table, the model kind and the statement flags.
    fn declare_all(&mut self) {
        let model = self.model;
        for (i, stmt) in model.statements.iter().enumerate() {
            self.at(Some(i));
            match &stmt.kind {
                StmtKind::Ode { name, .. } => {
                    self.set_kind(ModelKind::Ode, stmt.span);
                    self.declare(name, SymbolKind::State(StateKind::Ode), i);
                }
                StmtKind::Map { name, .. } => {
                    self.set_kind(ModelKind::Map, stmt.span);
                    self.declare(name, SymbolKind::State(StateKind::Map), i);
                }
                StmtKind::Markov { name, .. } => {
                    self.simulation_only = true;
                    self.declare(name, SymbolKind::State(StateKind::Markov), i);
                }
                StmtKind::Par { entries } => {
                    for e in entries {
                        self.declare(&e.name, SymbolKind::Parameter, i);
                    }
                }
                StmtKind::Number { entries } => {
                    for e in entries {
                        self.declare(&e.name, SymbolKind::Number, i);
                    }
                }
                StmtKind::Derived { name, .. } => self.declare(name, SymbolKind::Derived, i),
                StmtKind::Fixed { name, .. } => self.declare(name, SymbolKind::Fixed, i),
                StmtKind::Aux { name, .. } => self.declare(name, SymbolKind::Aux, i),
                StmtKind::Function { name, params, .. } => {
                    let kind = SymbolKind::Function {
                        params: params.len(),
                    };
                    self.declare(name, kind, i);
                    self.check_params(name, params);
                }
                StmtKind::Table { name, .. } => {
                    self.smooth = false;
                    self.declare(name, SymbolKind::Table, i);
                }
                StmtKind::Wiener { names } => {
                    self.simulation_only = true;
                    for n in names {
                        self.declare(n, SymbolKind::Wiener, i);
                    }
                }
                StmtKind::Global { .. } => self.simulation_only = true,
                StmtKind::Unsupported { family, .. } => {
                    if *family == Unsupported::FileTable {
                        self.smooth = false;
                    }
                    if may_declare(*family) && self.declarer.is_none() {
                        self.declarer = Some(stmt.span);
                    }
                }
                _ => {}
            }
        }
    }

    fn check_params(&mut self, name: &Ident, params: &'a [Ident]) {
        if params.len() > MAX_PARAMS {
            let span = params[MAX_PARAMS].span.to(params[params.len() - 1].span);
            let message = format!(
                "a function takes at most {MAX_PARAMS} parameters; `{}` has {}",
                name.text,
                params.len()
            );
            self.report(TOO_MANY_PARAMS, span, message, Vec::new());
        }
        let mut group = Vec::new();
        for (k, p) in params.iter().enumerate() {
            if let Some(reason) = reserved_reason(&p.key) {
                let message = format!("`{}` {reason} and cannot be a parameter name", p.text);
                self.report(RESERVED_NAME, p.span, message, Vec::new());
            } else if let Some(first) = params[..k].iter().find(|q| q.key == p.key) {
                let message = format!("parameter `{}` is declared twice", p.text);
                let note = (first.span, "the first one is here".to_string());
                self.report(DUPLICATE_NAME, p.span, message, vec![note]);
            } else {
                group.push(p);
            }
        }
        self.params.push((self.current, group));
    }

    /// True when a statement that `expand` dropped would have declared
    /// `key`: it is a recorded plain key, or a recorded root followed by a
    /// decimal index (`0`, or digits with no leading zero).
    fn suppressed(&self, key: &str) -> bool {
        if self.suppressed_plain.contains(key) {
            return true;
        }
        let digits = key.len() - key.trim_end_matches(|c: char| c.is_ascii_digit()).len();
        (key.len() - digits..key.len()).any(|i| {
            let index = &key[i..];
            (index == "0" || !index.starts_with('0')) && self.suppressed_roots.contains(&key[..i])
        })
    }

    /// Reports an undefined name, or the softer warning when an
    /// unsupported statement may declare it. A name that a dropped array
    /// statement would have declared is not reported, since `expand`
    /// already reported why it was dropped.
    fn undefined(&mut self, name: &Ident, function: bool) {
        if self.suppressed(&name.key) {
            return;
        }
        let what = if function { "function " } else { "" };
        match self.declarer {
            Some(decl) => {
                let message = format!(
                    "{what}`{}` is not declared; an unsupported statement may declare it",
                    name.text
                );
                let note = (
                    decl,
                    "this unsupported statement may declare it".to_string(),
                );
                self.report(MAYBE_UNDEFINED, name.span, message, vec![note]);
            }
            None => {
                let message = format!("{what}`{}` is not declared", name.text);
                self.report(UNDEFINED_NAME, name.span, message, Vec::new());
            }
        }
    }

    /// `init` entries and `x(0)=` lines must name a state, once.
    fn check_inits(&mut self) {
        let model = self.model;
        let mut seen: HashMap<&str, Span> = HashMap::new();
        for (i, stmt) in model.statements.iter().enumerate() {
            self.at(Some(i));
            let names: Vec<&Ident> = match &stmt.kind {
                StmtKind::Init { entries } => entries.iter().map(|e| &e.name).collect(),
                StmtKind::InitValue { name, .. } => vec![name],
                _ => continue,
            };
            for name in names.into_iter().filter(|n| n.index.is_none()) {
                match self.lookup(&name.key).map(|s| &self.symbols[s]) {
                    Some(sym) if matches!(sym.kind, SymbolKind::State(_)) => {
                        if let Some(&first) = seen.get(name.key.as_str()) {
                            let message =
                                format!("`{}` is given an initial value more than once", name.text);
                            let note = (first, "the first initial value is here".to_string());
                            self.report(INIT_TWICE, name.span, message, vec![note]);
                        } else {
                            seen.insert(&name.key, name.span);
                        }
                    }
                    Some(sym) => {
                        let message = format!(
                            "`{}` is {}, not a state, so it cannot have an initial value",
                            name.text,
                            noun(sym.kind)
                        );
                        let note = (
                            sym.name.span,
                            format!("`{}` is declared here", sym.name.text),
                        );
                        self.report(INIT_NOT_STATE, name.span, message, vec![note]);
                    }
                    // The declaration was already reported as reserved.
                    None if self.reserved_tried.contains(&name.key) => {}
                    None => {
                        if let Some(reason) = reserved_reason(&name.key) {
                            let message = format!(
                                "`{}` {reason}, not a state, so it cannot have an initial value",
                                name.text
                            );
                            self.report(INIT_NOT_STATE, name.span, message, Vec::new());
                        } else if self.declarer.is_some() {
                            self.undefined(name, false);
                        } else {
                            let message = format!(
                                "`{}` is not a state: no ODE, map or markov statement declares it",
                                name.text
                            );
                            self.report(INIT_NOT_STATE, name.span, message, Vec::new());
                        }
                    }
                }
            }
        }
    }

    /// The symbol a statement declares, when this statement is the one that
    /// declared it (not a reported duplicate) and it has the given kind.
    fn own_symbol(&self, name: &Ident, stmt: usize, kind: SymbolKind) -> Option<usize> {
        self.lookup(&name.key)
            .filter(|&s| self.symbols[s].stmt == stmt && self.symbols[s].kind == kind)
    }

    /// Pass B: resolves every name used in a formula.
    fn resolve_all(&mut self) {
        let model = self.model;
        for (i, stmt) in model.statements.iter().enumerate() {
            self.at(Some(i));
            match &stmt.kind {
                StmtKind::Ode { name, rhs, .. }
                | StmtKind::Map { name, rhs }
                | StmtKind::Aux { name, rhs }
                | StmtKind::Table { name, rhs, .. } => {
                    if name.index.is_none() {
                        self.walk(rhs, Owner::Formula);
                    }
                }
                StmtKind::Fixed { name, rhs } if name.index.is_none() => {
                    let owner = Owner::Fixed(self.own_symbol(name, i, SymbolKind::Fixed));
                    self.walk(rhs, owner);
                }
                StmtKind::Derived { name, rhs } if name.index.is_none() => {
                    let owner = Owner::Derived(self.own_symbol(name, i, SymbolKind::Derived));
                    self.walk(rhs, owner);
                }
                StmtKind::Function { name, params, rhs } => {
                    let kind = SymbolKind::Function {
                        params: params.len(),
                    };
                    let owner = Owner::Function(self.own_symbol(name, i, kind), params);
                    self.walk(rhs, owner);
                }
                StmtKind::Markov { name, lines, .. } if name.index.is_none() => {
                    for line in lines {
                        if let MarkovLine::Row(row) = line {
                            for e in &row.entries {
                                self.walk(e, Owner::Formula);
                            }
                        }
                    }
                }
                StmtKind::Global { cond, assigns, .. } => {
                    self.walk(cond, Owner::Formula);
                    for a in assigns {
                        self.assign_target(&a.name);
                        self.walk(&a.rhs, Owner::Formula);
                    }
                }
                _ => {}
            }
        }
    }

    fn walk(&mut self, expr: &Expr, owner: Owner<'a>) {
        match expr {
            Expr::Num { .. } | Expr::Index(_) | Expr::Error { .. } => {}
            Expr::Name(id) => self.name_use(id, owner),
            Expr::Unary { op, operand, .. } => {
                if *op == UnaryOp::Not {
                    self.smooth = false;
                }
                self.walk(operand, owner);
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                if op.is_comparison() || op.is_logical() {
                    self.smooth = false;
                }
                self.walk(lhs, owner);
                self.walk(rhs, owner);
            }
            Expr::If {
                cond, then, else_, ..
            } => {
                self.smooth = false;
                self.walk(cond, owner);
                self.walk(then, owner);
                self.walk(else_, owner);
            }
            Expr::Call { name, args, span } => {
                self.call(name, args.len(), *span, owner);
                for a in args {
                    self.walk(a, owner);
                }
            }
        }
    }

    /// True when `key` is a parameter of the function `owner` is in, by
    /// name or as `argK`.
    fn is_param(key: &str, owner: Owner<'a>) -> bool {
        match owner {
            Owner::Function(_, params) => {
                params.iter().any(|p| p.key == key)
                    || arg_number(key).is_some_and(|k| k <= params.len())
            }
            _ => false,
        }
    }

    fn derived_misuse(&mut self, id: &Ident, what: &str) {
        let message = format!(
            "`{}` is {what}, not a parameter; a derived parameter may use only parameters, \
             numbers, derived parameters and functions",
            id.text
        );
        self.report(DERIVED_NON_PARAMETER, id.span, message, Vec::new());
    }

    /// Notes a use that a derived parameter may not reach through the
    /// function `owner` is in.
    fn bad_in_function(&mut self, id: &Ident, what: &'static str, owner: Owner<'a>) {
        if let Owner::Function(Some(f), _) = owner {
            self.fn_bad
                .entry(f)
                .or_insert_with(|| (id.text.clone(), what, id.span));
        }
    }

    fn declared_note(&self, s: usize) -> Vec<(Span, String)> {
        let sym = &self.symbols[s];
        vec![(
            sym.name.span,
            format!("`{}` is declared here", sym.name.text),
        )]
    }

    /// A target of a `global` assignment: XPPAUT 8.0 assigns states and
    /// parameters.
    fn assign_target(&mut self, id: &Ident) {
        if id.index.is_some() {
            return;
        }
        const ONLY: &str = "`global` can assign only states and parameters";
        if let Some(reason) = reserved_reason(&id.key) {
            let message = format!("`{}` {reason} and cannot be assigned; {ONLY}", id.text);
            self.report(BAD_TARGET, id.span, message, Vec::new());
            return;
        }
        let Some(s) = self.lookup(&id.key) else {
            if self.suppressed(&id.key) {
                return;
            }
            if self.declarer.is_some() {
                self.undefined(id, false);
            } else {
                let message = format!("`{}` is not declared and cannot be assigned", id.text);
                self.report(BAD_TARGET, id.span, message, Vec::new());
            }
            return;
        };
        let kind = self.symbols[s].kind;
        if !matches!(kind, SymbolKind::State(_) | SymbolKind::Parameter) {
            let message = format!(
                "`{}` is {} and cannot be assigned; {ONLY}",
                id.text,
                noun(kind)
            );
            let notes = self.declared_note(s);
            self.report(BAD_TARGET, id.span, message, notes);
        }
    }

    /// A name used as a value.
    fn name_use(&mut self, id: &Ident, owner: Owner<'a>) {
        if id.index.is_some() || Self::is_param(&id.key, owner) {
            return;
        }
        if let Some(k) = arg_number(&id.key) {
            let message = bad_arg_message(id, k);
            self.report(BAD_ARG, id.span, message, Vec::new());
            return;
        }
        match id.key.as_str() {
            "t" => {
                if matches!(owner, Owner::Derived(_)) {
                    self.derived_misuse(id, "the time variable");
                }
                self.bad_in_function(id, "the time variable", owner);
                return;
            }
            "pi" => return,
            _ => {}
        }
        if builtin(&id.key).is_some() {
            let message = format!("`{}` is a function; call it with arguments", id.text);
            self.report(NOT_CALLABLE, id.span, message, Vec::new());
            return;
        }
        let Some(s) = self.lookup(&id.key) else {
            self.undefined(id, false);
            return;
        };
        let kind = self.symbols[s].kind;
        match kind {
            SymbolKind::Function { .. } | SymbolKind::Table => {
                let message = format!("`{}` is {}; call it with arguments", id.text, noun(kind));
                let notes = self.declared_note(s);
                self.report(NOT_CALLABLE, id.span, message, notes);
            }
            SymbolKind::Aux => {
                let message = format!(
                    "`{}` is an aux quantity, which is output only and cannot be used in a formula",
                    id.text
                );
                let notes = self.declared_note(s);
                self.report(AUX_IN_FORMULA, id.span, message, notes);
            }
            SymbolKind::Fixed => {
                match owner {
                    Owner::Fixed(Some(u)) | Owner::Function(Some(u), _) => {
                        self.edges.push((u, s, id.span))
                    }
                    Owner::Derived(_) => self.derived_misuse(id, noun(kind)),
                    _ => {}
                }
                self.bad_in_function(id, noun(kind), owner);
            }
            SymbolKind::Derived => {
                if let Owner::Derived(Some(u)) | Owner::Function(Some(u), _) = owner {
                    self.edges.push((u, s, id.span));
                }
            }
            SymbolKind::State(_) | SymbolKind::Wiener => {
                if matches!(owner, Owner::Derived(_)) {
                    self.derived_misuse(id, noun(kind));
                }
                self.bad_in_function(id, noun(kind), owner);
            }
            SymbolKind::Parameter | SymbolKind::Number => {}
        }
    }

    fn arity(
        &mut self,
        name: &Ident,
        want: usize,
        got: usize,
        span: Span,
        notes: Vec<(Span, String)>,
    ) {
        if want == got {
            return;
        }
        let plural = if want == 1 { "" } else { "s" };
        let message = format!("`{}` takes {want} argument{plural}, not {got}", name.text);
        self.report(WRONG_ARITY, span, message, notes);
    }

    /// A call `name(...)` with `nargs` arguments.
    fn call(&mut self, name: &Ident, nargs: usize, span: Span, owner: Owner<'a>) {
        if name.index.is_some() {
            return;
        }
        if Self::is_param(&name.key, owner) {
            let message = format!(
                "`{}` is a parameter of this function, not a function",
                name.text
            );
            self.report(NOT_CALLABLE, name.span, message, Vec::new());
            return;
        }
        if let Some(k) = arg_number(&name.key) {
            let message = bad_arg_message(name, k);
            self.report(BAD_ARG, name.span, message, Vec::new());
            return;
        }
        if let Some(reason) = match name.key.as_str() {
            "t" => Some("the time variable"),
            "pi" => Some("a constant"),
            _ => None,
        } {
            let message = format!("`{}` is {reason}, not a function", name.text);
            self.report(NOT_CALLABLE, name.span, message, Vec::new());
            return;
        }
        if let Some(b) = builtin(&name.key) {
            self.smooth &= b.smooth;
            self.simulation_only |= b.simulation_only;
            let kept = format!(
                "`{}` is not supported; the call is kept as written",
                name.text
            );
            let warning = match b.status {
                Status::Full => None,
                Status::Portability => Some((
                    PORTABILITY,
                    format!(
                        "XPPAUT 8.0 rejects this: it has no `{}` function, although its manual lists one",
                        name.text
                    ),
                )),
                Status::Delay => Some((
                    UNSUPPORTED_DELAY,
                    format!(
                        "`{}` is not supported yet; delay equations are planned for v2",
                        name.text
                    ),
                )),
                Status::Shift => Some((UNSUPPORTED_SHIFT, kept)),
                Status::DelShft => Some((UNSUPPORTED_DEL_SHFT, kept)),
            };
            if let Some((code, message)) = warning {
                self.report(code, name.span, message, Vec::new());
            }
            if let Some(want) = b.arity {
                self.arity(name, want, nargs, span, Vec::new());
            }
            return;
        }
        let Some(s) = self.lookup(&name.key) else {
            if name.key == "sum" {
                let message =
                    "`sum(..)of(..)` is not supported; the call is kept as written".to_string();
                self.report(UNSUPPORTED_SUM_OF, name.span, message, Vec::new());
            } else {
                self.undefined(name, true);
            }
            return;
        };
        let kind = self.symbols[s].kind;
        let notes = self.declared_note(s);
        match kind {
            SymbolKind::Function { params } => {
                if let Owner::Function(Some(u), _)
                | Owner::Fixed(Some(u))
                | Owner::Derived(Some(u)) = owner
                {
                    self.edges.push((u, s, name.span));
                }
                self.arity(name, params, nargs, span, notes);
            }
            SymbolKind::Table => self.arity(name, 1, nargs, span, notes),
            _ => {
                let message = format!("`{}` is {}, not a function", name.text, noun(kind));
                self.report(NOT_CALLABLE, name.span, message, notes);
            }
        }
    }

    fn is_function(&self, s: usize) -> bool {
        matches!(self.symbols[s].kind, SymbolKind::Function { .. })
    }

    /// Reports cycles among fixed quantities, derived parameters and
    /// functions, fixed quantities used above their definition (directly
    /// or through functions) and derived parameters that reach a state,
    /// a fixed quantity, `t` or noise through functions.
    fn report_graph(&mut self) {
        let n = self.symbols.len();
        let mut adj = vec![Vec::new(); n];
        for &(u, w, _) in &self.edges {
            adj[u].push(w);
        }
        let comp = components(n, &adj);
        let ncomp = comp.iter().max().map_or(0, |&c| c + 1);
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); ncomp];
        for (s, &c) in comp.iter().enumerate() {
            members[c].push(s);
        }
        let cyclic: Vec<bool> = (0..n)
            .map(|s| members[comp[s]].len() > 1 || adj[s].contains(&s))
            .collect();
        let mut groups: Vec<Vec<usize>> =
            members.iter().filter(|m| cyclic[m[0]]).cloned().collect();
        for g in &mut groups {
            g.sort_by_key(|&s| (self.symbols[s].stmt, self.symbols[s].name.span.start));
        }
        groups.sort_by_key(|g| g[0]);
        for g in groups {
            self.at(Some(self.symbols[g[0]].stmt));
            let first = &self.symbols[g[0]];
            let names: Vec<&str> = g
                .iter()
                .map(|&s| self.symbols[s].name.text.as_str())
                .collect();
            let is_function = g.iter().all(|&s| self.is_function(s));
            let message = match (is_function, names.len()) {
                (true, 1) => format!("function `{}` calls itself", names[0]),
                (true, _) => format!("functions {} call each other", name_list(&names)),
                (false, 1) => format!("`{}` is defined in terms of itself", names[0]),
                (false, _) => format!("{} are defined in terms of each other", name_list(&names)),
            };
            let code = if is_function {
                RECURSIVE_FUNCTION
            } else {
                CIRCULAR
            };
            let span = first.name.span;
            let notes = g[1..]
                .iter()
                .map(|&s| {
                    let sym = &self.symbols[s];
                    let what = if is_function { "defined" } else { "declared" };
                    (sym.name.span, format!("`{}` is {what} here", sym.name.text))
                })
                .collect();
            self.report(code, span, message, notes);
        }

        // What each component of functions reaches through function calls
        // only: the fixed quantity on the latest line, and a use that a
        // derived parameter may not reach. Tarjan numbers a component after
        // every component it reaches, so increasing order sees callees
        // first.
        let mut latest: Vec<Option<usize>> = vec![None; ncomp];
        let mut bad: Vec<Option<BadUse>> = vec![None; ncomp];
        for c in 0..ncomp {
            let mut fixed: Option<usize> = None;
            let mut use_: Option<BadUse> = None;
            for &f in members[c].iter().filter(|&&f| self.is_function(f)) {
                if let Some(b) = self.fn_bad.get(&f)
                    && use_.as_ref().is_none_or(|u| b.2.start < u.2.start)
                {
                    use_ = Some(b.clone());
                }
                for &w in &adj[f] {
                    let cand = if self.symbols[w].kind == SymbolKind::Fixed {
                        Some(w)
                    } else if self.is_function(w) && comp[w] != c {
                        if use_.is_none() {
                            use_.clone_from(&bad[comp[w]]);
                        }
                        latest[comp[w]]
                    } else {
                        None
                    };
                    if let Some(x) = cand
                        && fixed.is_none_or(|y| self.symbols[x].stmt > self.symbols[y].stmt)
                    {
                        fixed = Some(x);
                    }
                }
            }
            latest[c] = fixed;
            bad[c] = use_;
        }

        let edges = std::mem::take(&mut self.edges);
        for &(u, w, span) in &edges {
            self.at(Some(self.symbols[u].stmt));
            let (user, used) = (&self.symbols[u], &self.symbols[w]);
            let same_cycle = comp[u] == comp[w] && cyclic[u];
            if same_cycle {
                continue;
            }
            match (user.kind, used.kind) {
                (SymbolKind::Fixed, SymbolKind::Fixed) if used.stmt > user.stmt => {
                    let message = format!(
                        "fixed quantity `{}` is defined on a later line; XPPAUT 8.0 evaluates \
                         fixed quantities in file order and reads it as 0 here",
                        used.name.text
                    );
                    let notes = self.declared_note(w);
                    self.report(FIXED_FORWARD, span, message, notes);
                }
                (SymbolKind::Fixed, SymbolKind::Function { .. }) => {
                    let Some(x) = latest[comp[w]] else { continue };
                    if self.symbols[x].stmt <= user.stmt {
                        continue;
                    }
                    let message = format!(
                        "fixed quantity `{}`, which `{}` uses, is defined on a later line; \
                         XPPAUT 8.0 evaluates fixed quantities in file order and reads it as 0 \
                         here",
                        self.symbols[x].name.text, used.name.text
                    );
                    let notes = self.declared_note(x);
                    self.report(FIXED_FORWARD, span, message, notes);
                }
                (SymbolKind::Derived, SymbolKind::Function { .. }) => {
                    let Some((text, what, at)) = bad[comp[w]].clone() else {
                        continue;
                    };
                    let message = format!(
                        "derived parameter `{}` calls `{}`, which uses `{text}`, {what}, not a \
                         parameter",
                        user.name.text, used.name.text
                    );
                    let note = (at, format!("`{text}` is used here"));
                    self.report(DERIVED_NON_PARAMETER, span, message, vec![note]);
                }
                _ => {}
            }
        }
        self.edges = edges;
        self.at(None);
    }

    /// Portability warning for names longer than XPPAUT 8.0 keeps, with a
    /// note at one other name that agrees with it in the kept characters.
    /// Global names are compared with global names, and function
    /// parameters with the other parameters of the same function.
    fn report_long_names(&mut self) {
        let declared = std::mem::take(&mut self.declared);
        let params = std::mem::take(&mut self.params);
        let (at, names): (Vec<_>, Vec<_>) = declared.into_iter().unzip();
        self.long_names(&names, &at);
        for (at, group) in &params {
            self.long_names(group, &vec![*at; group.len()]);
        }
        self.at(None);
    }

    /// `at[i]` is the statement of `names[i]`, for the copy note.
    fn long_names(&mut self, names: &[&Ident], at: &[Option<usize>]) {
        let prefix = |id: &Ident| id.key.chars().take(XPP_NAME_LEN).collect::<String>();
        let is_long = |id: &Ident| id.key.chars().count() > XPP_NAME_LEN;
        // The first two names of at least 10 characters for each kept
        // prefix. Names in `names` have distinct keys, so one of the two
        // differs from any name.
        let mut by_prefix: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, id) in names.iter().enumerate() {
            if id.key.chars().count() >= XPP_NAME_LEN {
                let e = by_prefix.entry(prefix(id)).or_default();
                if e.len() < 2 {
                    e.push(i);
                }
            }
        }
        for (i, id) in names.iter().enumerate() {
            if !is_long(id) {
                continue;
            }
            let notes = by_prefix[&prefix(id)]
                .iter()
                .find(|&&j| j != i)
                .map(|&j| {
                    let other = names[j];
                    let message = format!(
                        "`{}` agrees with this name in its first {XPP_NAME_LEN} characters; \
                         XPPAUT 8.0 reads them as one name",
                        other.text
                    );
                    (other.span, message)
                })
                .into_iter()
                .collect();
            let message = format!(
                "XPPAUT 8.0 rejects this: `{}` is longer than {XPP_NAME_LEN} characters, and \
                 XPPAUT 8.0 truncates names to {XPP_NAME_LEN} characters",
                id.text
            );
            self.at(at[i]);
            self.report(PORTABILITY, id.span, message, notes);
        }
    }
}

/// The strongly connected components of a graph with `n` nodes and the
/// adjacency lists `adj`, by Tarjan's algorithm without recursion. Returns
/// the component number of each node. A component is numbered after every
/// component it reaches.
fn components(n: usize, adj: &[Vec<usize>]) -> Vec<usize> {
    const UNSEEN: usize = usize::MAX;
    let mut order = vec![UNSEEN; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut comp = vec![UNSEEN; n];
    let mut stack = Vec::new();
    let mut next = 0;
    let mut ncomp = 0;
    for root in 0..n {
        if order[root] != UNSEEN {
            continue;
        }
        // Each frame is a node and the position of its next edge.
        let mut frames = vec![(root, 0)];
        order[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut k)) = frames.last_mut() {
            if let Some(&w) = adj[v].get(*k) {
                *k += 1;
                if order[w] == UNSEEN {
                    order[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    frames.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(order[w]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(u, _)) = frames.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == order[v] {
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    comp[w] = ncomp;
                    if w == v {
                        break;
                    }
                }
                ncomp += 1;
            }
        }
    }
    comp
}

/// `a`, `b` and `c` as "`a`, `b` and `c`".
fn name_list(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
    match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => quoted.concat(),
    }
}

#[cfg(test)]
mod tests;
