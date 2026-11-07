//! Lowering of checked `.ode` models to the DAG.
//!
//! `lower` takes the expanded model and the checker's [`ModelInfo`] and builds one
//! [`Dag`] that holds the right sides of the states and of the `aux` quantities.
//! Numbers, derived parameters, fixed quantities and user functions are inlined as
//! subtrees. A fixed or derived quantity is lowered the first time something reads it,
//! so a quantity that nothing reaches adds no node and does not make a model
//! non-autonomous. The input must have no check errors; surprising input still
//! produces an [`IrError`], never a panic.

use crate::dag::{Cmp, Dag, Fn1, Fn2, MAX_NODES, NodeId, Op};
use crate::delays::{self, DelaySlot, Names, SlotKey};
use crate::error::{IrError, IrErrorKind};
use crate::eval::eval;
use pc_ode::builtins::builtin;
use pc_ode::{
    BinaryOp, DUMMY_SPAN, Entry, Expr, Ident, Model, ModelInfo, ModelKind, Span, Stmt, StmtKind,
    UnaryOp, Unsupported, Value,
};
use std::collections::{HashMap, HashSet};

/// Deepest expression or function nesting the lowering follows.
const MAX_DEPTH: usize = 1000;

pub struct IrModel {
    pub dag: Dag,
    pub kind: ModelKind,
    pub state_names: Vec<String>,
    pub param_names: Vec<String>,
    pub aux_names: Vec<String>,
    /// Right sides of the states, in state order.
    pub f: Vec<NodeId>,
    /// The `aux` quantities, in declaration order.
    pub aux: Vec<NodeId>,
    pub init: Vec<f64>,
    /// One value per parameter, hidden delay slots included (NaN there).
    pub par_values: Vec<f64>,
    /// Number of parameters the model declares. The hidden delay slots of `lower_dde`
    /// follow them in `param_names`, before the `t` slot.
    pub n_user_params: usize,
    /// The hidden delay slots; slot `j` is parameter `n_user_params + j`.
    pub delays: Vec<DelaySlot>,
    /// False when a formula reachable from `f` or `aux` reads `t`.
    pub autonomous: bool,
    pub smooth: bool,
    pub(crate) entries: std::collections::HashMap<crate::entry::EntryKind, crate::entry::Entry>,
    pub(crate) diff_memo: std::collections::HashMap<(NodeId, crate::diff::Wrt), NodeId>,
}

impl IrModel {
    pub fn n(&self) -> usize {
        self.state_names.len()
    }

    /// Number of parameters, hidden delay slots included. `p` has one more entry, `t`.
    pub fn n_p(&self) -> usize {
        self.param_names.len()
    }
}

/// Lowers a checked model. `model` is `Loaded::expanded` and `info` is `Loaded::info`.
/// A `delay(...)` call is refused; `lower_dde` accepts it.
pub fn lower(model: &Model, info: &ModelInfo) -> Result<IrModel, Vec<IrError>> {
    lower_impl(model, info, MAX_NODES, false)
}

/// Lowers a model with `delay(x, tau)` calls. Each distinct pair of a state and a delay
/// time becomes one hidden parameter slot after the user parameters (see `delays`). A
/// model without a delay call is an error, so a caller that wants a delay equation does
/// not get an ODE by accident.
pub fn lower_dde(model: &Model, info: &ModelInfo) -> Result<IrModel, Vec<IrError>> {
    lower_impl(model, info, MAX_NODES, true)
}

/// Everything the lowering reads from the statements.
struct Env<'a> {
    /// Index of the `t` slot: the user parameters plus the hidden delay slots.
    n_p: u32,
    allow_delays: bool,
    /// Parameter index of each hidden delay slot.
    delay_slots: HashMap<SlotKey, u32>,
    states: HashMap<&'a str, u32>,
    params: HashMap<&'a str, u32>,
    numbers: HashMap<&'a str, f64>,
    /// Derived parameters and fixed quantities: key to formula.
    named: HashMap<&'a str, &'a Expr>,
    funcs: HashMap<&'a str, (Vec<&'a str>, &'a Expr)>,
    table_ids: HashMap<&'a str, u32>,
    par_values: Vec<f64>,
    init: Vec<f64>,
    cap: usize,
}

struct Frame {
    params: Vec<String>,
    args: Vec<NodeId>,
}

struct Lw<'e, 'a> {
    env: &'e Env<'a>,
    dag: Dag,
    /// Lowered fixed and derived quantities.
    memo: HashMap<&'a str, NodeId>,
    visiting: HashSet<&'a str>,
    errors: Vec<IrError>,
    /// Inside a table formula `t` is `Var(0)` and state `i` is `Var(i + 1)`.
    table_mode: bool,
    /// Table ids at or above this are not built yet.
    table_limit: u32,
    uses_t: bool,
    frames: Vec<Frame>,
    depth: usize,
}

fn small_int(v: f64) -> Option<i32> {
    (v.fract() == 0.0 && v.abs() <= 16.0).then_some(v as i32)
}

fn num(v: &Option<Value>) -> f64 {
    match v {
        Some(Value::Num { value, .. }) => *value,
        _ => 0.0,
    }
}

impl<'e, 'a> Lw<'e, 'a> {
    fn new(env: &'e Env<'a>, table_mode: bool, table_limit: u32) -> Self {
        Lw {
            env,
            dag: Dag::new(env.cap),
            memo: HashMap::new(),
            visiting: HashSet::new(),
            errors: Vec::new(),
            table_mode,
            table_limit,
            uses_t: false,
            frames: Vec::new(),
            depth: 0,
        }
    }

    fn fail(&mut self, kind: IrErrorKind, span: Span, msg: impl Into<String>) -> NodeId {
        self.errors.push(IrError::new(kind, span, msg));
        self.dag.konst(f64::NAN)
    }

    fn expr(&mut self, e: &'a Expr) -> NodeId {
        if self.depth >= MAX_DEPTH {
            return self.fail(
                IrErrorKind::TooLarge,
                e.span(),
                "the formula is nested too deeply",
            );
        }
        self.depth += 1;
        let r = self.expr_inner(e);
        self.depth -= 1;
        r
    }

    fn expr_inner(&mut self, e: &'a Expr) -> NodeId {
        match e {
            Expr::Num { value, .. } => self.dag.konst(*value),
            Expr::Name(id) => self.name(id),
            Expr::Index(ix) => self.fail(
                IrErrorKind::Unsupported,
                ix.span(),
                "an index form outside an array block",
            ),
            Expr::Error { span } => {
                self.fail(IrErrorKind::Unsupported, *span, "a formula with an error")
            }
            Expr::Unary { op, operand, .. } => {
                let a = self.expr(operand);
                match op {
                    UnaryOp::Neg => self.dag.neg(a),
                    UnaryOp::Not => self.dag.not(a),
                }
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let a = self.expr(lhs);
                let b = self.expr(rhs);
                self.binary(*op, a, b)
            }
            Expr::If {
                cond, then, else_, ..
            } => {
                let c = self.expr(cond);
                let a = self.expr(then);
                let b = self.expr(else_);
                self.dag.select(c, a, b)
            }
            Expr::Call { name, args, span } => self.call(name, args, *span),
        }
    }

    fn binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        let d = &mut self.dag;
        match op {
            BinaryOp::Add => d.add(a, b),
            BinaryOp::Sub => d.sub(a, b),
            BinaryOp::Mul => d.mul(a, b),
            BinaryOp::Div => d.div(a, b),
            BinaryOp::Pow | BinaryOp::StarStar => {
                if let Op::Const(bits) = *d.op(b)
                    && let Some(n) = small_int(f64::from_bits(bits))
                {
                    return d.powi(a, n);
                }
                d.call2(Fn2::Pow, a, b)
            }
            BinaryOp::Lt => d.cmp(Cmp::Lt, a, b),
            BinaryOp::Gt => d.cmp(Cmp::Gt, a, b),
            BinaryOp::Le => d.cmp(Cmp::Le, a, b),
            BinaryOp::Ge => d.cmp(Cmp::Ge, a, b),
            BinaryOp::Eq => d.cmp(Cmp::Eq, a, b),
            BinaryOp::And => d.and(a, b),
            BinaryOp::Or => d.or(a, b),
        }
    }

    fn time(&mut self) -> NodeId {
        if self.table_mode {
            self.dag.var(0)
        } else {
            self.uses_t = true;
            self.dag.param(self.env.n_p)
        }
    }

    fn name(&mut self, id: &'a Ident) -> NodeId {
        let key = id.key.as_str();
        if let Some(fr) = self.frames.last() {
            if let Some(pos) = fr.params.iter().position(|p| p == key)
                && let Some(&n) = fr.args.get(pos)
            {
                return n;
            }
            if let Some(k) = key.strip_prefix("arg")
                && let Ok(k @ 1..=9) = k.parse::<usize>()
                && let Some(&n) = fr.args.get(k - 1)
            {
                return n;
            }
        }
        let env = self.env;
        if key == "t" {
            return self.time();
        }
        if key == "pi" {
            return self.dag.konst(std::f64::consts::PI);
        }
        if let Some(&i) = env.states.get(key) {
            return self.dag.var(i + u32::from(self.table_mode));
        }
        if let Some(&j) = env.params.get(key) {
            return self.dag.param(j);
        }
        if let Some(&v) = env.numbers.get(key) {
            return self.dag.konst(v);
        }
        if let Some((&k, &rhs)) = env.named.get_key_value(key) {
            return self.named(k, rhs, id.span);
        }
        self.fail(
            IrErrorKind::Unsupported,
            id.span,
            format!("`{}` cannot be read in a formula", id.text),
        )
    }

    /// A fixed quantity or derived parameter, lowered once in the global scope.
    fn named(&mut self, key: &'a str, rhs: &'a Expr, span: Span) -> NodeId {
        if let Some(&n) = self.memo.get(key) {
            return n;
        }
        if !self.visiting.insert(key) {
            return self.fail(
                IrErrorKind::Unsupported,
                span,
                format!("`{key}` is defined in terms of itself"),
            );
        }
        let saved = std::mem::take(&mut self.frames);
        let n = self.expr(rhs);
        self.frames = saved;
        self.visiting.remove(key);
        self.memo.insert(key, n);
        n
    }

    /// A `delay` call under `lower_dde`: a read of its hidden parameter slot. The scan
    /// classified every call already, so a call that fails here sits where the scan does
    /// not look, which is a table formula reading a fixed quantity that holds a delay.
    fn delay(&mut self, args: &'a [Expr], span: Span) -> NodeId {
        let env = self.env;
        if self.table_mode {
            return self.fail(
                IrErrorKind::Delay,
                span,
                "a table cannot read a delayed value",
            );
        }
        let formals: Vec<String> = self
            .frames
            .last()
            .map(|f| f.params.clone())
            .unwrap_or_default();
        let slot = delays::classify(args, &formals, env).and_then(|(state, time)| {
            let key = DelaySlot { state, time }.key();
            env.delay_slots
                .get(&key)
                .copied()
                .ok_or_else(|| "this delay has no slot".to_string())
        });
        match slot {
            Ok(idx) => self.dag.param(idx),
            Err(msg) => self.fail(IrErrorKind::Delay, span, msg),
        }
    }

    fn call(&mut self, name: &'a Ident, args: &'a [Expr], span: Span) -> NodeId {
        let key = name.key.as_str();
        let env = self.env;
        if let Some((params, body)) = env.funcs.get(key) {
            if params.len() != args.len() {
                return self.fail(
                    IrErrorKind::Unsupported,
                    span,
                    format!("`{}` takes {} arguments", name.text, params.len()),
                );
            }
            let nodes: Vec<NodeId> = args.iter().map(|a| self.expr(a)).collect();
            if self.frames.len() >= MAX_DEPTH {
                return self.fail(
                    IrErrorKind::TooLarge,
                    span,
                    "user functions are nested too deeply",
                );
            }
            self.frames.push(Frame {
                params: params.iter().map(|p| (*p).to_string()).collect(),
                args: nodes,
            });
            let n = self.expr(body);
            self.frames.pop();
            return n;
        }
        if let Some(&id) = env.table_ids.get(key) {
            let (Some(a), 1) = (args.first(), args.len()) else {
                return self.fail(
                    IrErrorKind::Unsupported,
                    span,
                    format!("table `{}` takes one argument", name.text),
                );
            };
            let x = self.expr(a);
            if id >= self.table_limit {
                return self.fail(
                    IrErrorKind::BadTable,
                    span,
                    format!("table `{}` is used before it is built", name.text),
                );
            }
            return self.dag.table(id, x);
        }
        let Some(b) = builtin(key) else {
            return self.fail(
                IrErrorKind::Unsupported,
                span,
                format!("`{}` is not a function", name.text),
            );
        };
        if key == "delay" {
            if !env.allow_delays {
                // The pre-scan reported every delay call already.
                return self.dag.konst(f64::NAN);
            }
            return self.delay(args, span);
        }
        if b.arity != Some(args.len()) {
            return self.fail(
                IrErrorKind::Unsupported,
                span,
                format!("wrong number of arguments for `{}`", name.text),
            );
        }
        let nodes: Vec<NodeId> = args.iter().map(|a| self.expr(a)).collect();
        let (Some(&a), second) = (nodes.first(), nodes.get(1)) else {
            return self.fail(IrErrorKind::Unsupported, span, "a call without arguments");
        };
        if let Some(&c) = second {
            let f = match key {
                "atan2" => Fn2::Atan2,
                "mod" => Fn2::Mod,
                "min" => Fn2::Min,
                "max" => Fn2::Max,
                "besselj" => Fn2::BesselJ,
                "bessely" => Fn2::BesselY,
                _ => {
                    return self.fail(
                        IrErrorKind::Unsupported,
                        span,
                        format!("`{}` is not supported", name.text),
                    );
                }
            };
            return self.dag.call2(f, a, c);
        }
        let f = match key {
            "sin" => Fn1::Sin,
            "cos" => Fn1::Cos,
            "tan" => Fn1::Tan,
            "asin" => Fn1::Asin,
            "acos" => Fn1::Acos,
            "atan" => Fn1::Atan,
            "sinh" => Fn1::Sinh,
            "cosh" => Fn1::Cosh,
            "tanh" => Fn1::Tanh,
            "exp" => Fn1::Exp,
            "ln" | "log" => Fn1::Ln,
            "log10" => Fn1::Log10,
            "sqrt" => Fn1::Sqrt,
            "abs" => Fn1::Abs,
            "erf" => Fn1::Erf,
            "erfc" => Fn1::Erfc,
            "lgamma" => Fn1::Lgamma,
            "flr" => Fn1::Floor,
            "ceil" => Fn1::Ceil,
            "heav" => Fn1::Heav,
            "sign" => Fn1::Sign,
            _ => {
                return self.fail(
                    IrErrorKind::Unsupported,
                    span,
                    format!("`{}` is not supported", name.text),
                );
            }
        };
        self.dag.call1(f, a)
    }
}

/// Calls `f` on every call expression inside `e`.
fn each_call<'a>(e: &'a Expr, f: &mut dyn FnMut(&'a Ident, Span)) {
    match e {
        Expr::Unary { operand, .. } => each_call(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            each_call(lhs, f);
            each_call(rhs, f);
        }
        Expr::If {
            cond, then, else_, ..
        } => {
            each_call(cond, f);
            each_call(then, f);
            each_call(else_, f);
        }
        Expr::Call { name, args, span } => {
            f(name, *span);
            for a in args {
                each_call(a, f);
            }
        }
        Expr::Num { .. } | Expr::Name(_) | Expr::Index(_) | Expr::Error { .. } => {}
    }
}

/// Flattens array blocks that the expansion pass left in place (it leaves none for a
/// valid model; this keeps odd input from being silently skipped).
fn each_stmt<'a>(stmts: &'a [Stmt], f: &mut dyn FnMut(&'a Stmt)) {
    for s in stmts {
        match &s.kind {
            StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => each_stmt(body, f),
            _ => f(s),
        }
    }
}

pub(crate) fn formulas(kind: &StmtKind) -> Vec<&Expr> {
    match kind {
        StmtKind::Ode { rhs, .. }
        | StmtKind::Map { rhs, .. }
        | StmtKind::Derived { rhs, .. }
        | StmtKind::Fixed { rhs, .. }
        | StmtKind::Aux { rhs, .. }
        | StmtKind::Function { rhs, .. }
        | StmtKind::Table { rhs, .. } => vec![rhs],
        _ => Vec::new(),
    }
}

impl Names for Env<'_> {
    fn state(&self, key: &str) -> Option<usize> {
        self.states.get(key).map(|&i| i as usize)
    }

    fn param(&self, key: &str) -> Option<usize> {
        self.params.get(key).map(|&j| j as usize)
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.numbers.get(key).copied()
    }
}

/// Refusals that need no lowering: simulation-only, delay, unsupported statements. With
/// `allow_delays` the delay calls are left to `delays::scan`.
fn prescan(model: &Model, info: &ModelInfo, errors: &mut Vec<IrError>, allow_delays: bool) {
    let mut sim_span = None;
    each_stmt(&model.statements, &mut |s| {
        match &s.kind {
            StmtKind::Global { .. } | StmtKind::Markov { .. } | StmtKind::Wiener { .. } => {
                sim_span.get_or_insert(s.span);
            }
            StmtKind::Unsupported { family, text } if *family != Unsupported::ActiveComment => {
                let msg = format!("unsupported statement `{}`", text.trim());
                errors.push(IrError::new(IrErrorKind::Unsupported, s.span, msg));
            }
            _ => {}
        }
        for e in formulas(&s.kind) {
            each_call(e, &mut |name, span| match name.key.as_str() {
                "delay" if !allow_delays => {
                    let msg = "delay(...) is not supported";
                    errors.push(IrError::new(IrErrorKind::Delay, span, msg));
                }
                "ran" | "normal" | "poisson" => {
                    let msg = format!("`{}` is for simulation only", name.text);
                    errors.push(IrError::new(IrErrorKind::SimulationOnly, span, msg));
                }
                _ => {}
            });
        }
    });
    let reported = errors.iter().any(|e| e.kind == IrErrorKind::SimulationOnly);
    if info.simulation_only && !reported {
        let span = sim_span.unwrap_or(DUMMY_SPAN);
        let msg = "the model is for simulation only";
        errors.push(IrError::new(IrErrorKind::SimulationOnly, span, msg));
    }
}

fn table_header(
    npts: &Value,
    lo: &Value,
    hi: &Value,
    span: Span,
) -> Result<(usize, f64, f64), IrError> {
    let bad = |m: &str| IrError::new(IrErrorKind::BadTable, span, m);
    let (Value::Num { value: n, .. }, Value::Num { value: lo, .. }, Value::Num { value: hi, .. }) =
        (npts, lo, hi)
    else {
        return Err(bad("a table header field is not a number"));
    };
    if !(n.fract() == 0.0 && (2.0..=1_000_000.0).contains(n)) {
        return Err(bad("the table size must be an integer from 2 to 1000000"));
    }
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        return Err(bad("the table range must be finite with lo < hi"));
    }
    Ok((*n as usize, *lo, *hi))
}

#[cfg(test)]
pub(crate) fn lower_with_cap(
    model: &Model,
    info: &ModelInfo,
    cap: usize,
) -> Result<IrModel, Vec<IrError>> {
    lower_impl(model, info, cap, false)
}

fn lower_impl(
    model: &Model,
    info: &ModelInfo,
    cap: usize,
    allow_delays: bool,
) -> Result<IrModel, Vec<IrError>> {
    let mut errors = Vec::new();
    if info.kind == ModelKind::None {
        let msg = "the model has no ODE and no map";
        errors.push(IrError::new(IrErrorKind::NoEquations, DUMMY_SPAN, msg));
    }
    prescan(model, info, &mut errors, allow_delays);
    if !errors.is_empty() {
        return Err(errors);
    }

    // Collect the declarations.
    let mut stmts: Vec<&Stmt> = Vec::new();
    each_stmt(&model.statements, &mut |s| stmts.push(s));
    let mut env = Env {
        n_p: 0,
        allow_delays,
        delay_slots: HashMap::new(),
        states: HashMap::new(),
        params: HashMap::new(),
        numbers: HashMap::new(),
        named: HashMap::new(),
        funcs: HashMap::new(),
        table_ids: HashMap::new(),
        par_values: Vec::new(),
        init: Vec::new(),
        cap,
    };
    let mut state_names = Vec::new();
    let mut param_names = Vec::new();
    let mut rhs_of_state: Vec<&Expr> = Vec::new();
    let mut aux_decl: Vec<(&Ident, &Expr)> = Vec::new();
    let mut tables: Vec<(&Stmt, &Ident, &Expr)> = Vec::new();
    let mut inits: Vec<(&Ident, &Value, Span)> = Vec::new();
    let mut number_values: Vec<(&Entry, bool)> = Vec::new();
    for s in &stmts {
        match &s.kind {
            StmtKind::Ode { name, rhs, .. } | StmtKind::Map { name, rhs } => {
                env.states
                    .insert(name.key.as_str(), state_names.len() as u32);
                state_names.push(name.text.clone());
                rhs_of_state.push(rhs);
            }
            StmtKind::Par { entries } => {
                for e in entries {
                    env.params
                        .insert(e.name.key.as_str(), param_names.len() as u32);
                    param_names.push(e.name.text.clone());
                    env.par_values.push(num(&e.value));
                    if matches!(e.value, Some(Value::Index(_) | Value::Error { .. })) {
                        number_values.push((e, false));
                    }
                }
            }
            StmtKind::Number { entries } => {
                for e in entries {
                    env.numbers.insert(e.name.key.as_str(), num(&e.value));
                    if matches!(e.value, Some(Value::Index(_) | Value::Error { .. })) {
                        number_values.push((e, true));
                    }
                }
            }
            StmtKind::Derived { name, rhs } | StmtKind::Fixed { name, rhs } => {
                env.named.insert(name.key.as_str(), rhs);
            }
            StmtKind::Aux { name, rhs } => aux_decl.push((name, rhs)),
            StmtKind::Function { name, params, rhs } => {
                let keys = params.iter().map(|p| p.key.as_str()).collect();
                env.funcs.insert(name.key.as_str(), (keys, rhs));
            }
            StmtKind::Table { name, rhs, .. } => {
                env.table_ids.insert(name.key.as_str(), tables.len() as u32);
                tables.push((s, name, rhs));
            }
            StmtKind::InitValue { name, value } => inits.push((name, value, s.span)),
            StmtKind::Init { entries } => {
                for e in entries {
                    if let Some(v) = &e.value {
                        inits.push((&e.name, v, e.span));
                    }
                }
            }
            _ => {}
        }
    }
    let n_user_params = param_names.len();
    let mut delay_slots = Vec::new();
    if allow_delays {
        let found = delays::scan(&stmts, &env, info.kind == ModelKind::Map);
        if found.calls == 0 {
            let msg = "the model has no delay(...) call";
            errors.push(IrError::new(IrErrorKind::Delay, DUMMY_SPAN, msg));
        }
        errors.extend(found.errors);
        if !errors.is_empty() {
            return Err(errors);
        }
        for (j, slot) in found.slots.iter().enumerate() {
            let name = delays::hidden_name(slot, &state_names, &param_names[..n_user_params]);
            param_names.push(name);
            env.par_values.push(f64::NAN);
            env.delay_slots
                .insert(slot.key(), (n_user_params + j) as u32);
        }
        delay_slots = found.slots;
    }
    env.n_p = param_names.len() as u32;
    for (e, _) in &number_values {
        let msg = format!("the value of `{}` is not a number", e.name.text);
        errors.push(IrError::new(IrErrorKind::Unsupported, e.span, msg));
    }
    env.init = vec![0.0; state_names.len()];
    for (name, value, span) in inits {
        match (value, env.states.get(name.key.as_str())) {
            (Value::Num { value, .. }, Some(&i)) => {
                if let Some(slot) = env.init.get_mut(i as usize) {
                    *slot = *value;
                }
            }
            _ => {
                let msg = format!("cannot read the initial value of `{}`", name.text);
                errors.push(IrError::new(IrErrorKind::Unsupported, span, msg));
            }
        }
    }

    let mut lw = Lw::new(&env, false, u32::MAX);

    // Tables first, in declaration order, so ids match the order of the file.
    for (id, (s, _, rhs)) in tables.iter().enumerate() {
        let StmtKind::Table { npts, lo, hi, .. } = &s.kind else {
            continue;
        };
        let block = match table_header(npts, lo, hi, s.span) {
            Ok((n, lo, hi)) => tabulate(&env, &lw.dag.tables, id as u32, rhs, n, lo, hi, s.span)
                .unwrap_or_else(|mut e| {
                    errors.append(&mut e);
                    Vec::new()
                }),
            Err(e) => {
                errors.push(e);
                Vec::new()
            }
        };
        lw.dag.add_table(block);
    }

    let f: Vec<NodeId> = rhs_of_state.iter().map(|e| lw.expr(e)).collect();
    let aux: Vec<NodeId> = aux_decl.iter().map(|(_, e)| lw.expr(e)).collect();
    errors.append(&mut lw.errors);
    if lw.dag.exceeded() {
        let msg = "the model needs more nodes than the limit";
        errors.push(IrError::new(IrErrorKind::TooLarge, DUMMY_SPAN, msg));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let autonomous = !lw.uses_t;
    let dag = lw.dag;
    Ok(IrModel {
        dag,
        kind: info.kind,
        state_names,
        param_names,
        aux_names: aux_decl.iter().map(|(n, _)| n.text.clone()).collect(),
        f,
        aux,
        init: env.init.clone(),
        par_values: env.par_values.clone(),
        n_user_params,
        delays: delay_slots,
        autonomous,
        smooth: info.smooth,
        entries: HashMap::new(),
        diff_memo: HashMap::new(),
    })
}

/// Builds the block for table number `id` by evaluating its formula at each node. The
/// formula is lowered into a scratch DAG where `t` is `Var(0)` and state `i` is
/// `Var(i + 1)`; parameters and states take their declared values.
#[allow(clippy::too_many_arguments)]
fn tabulate(
    env: &Env,
    built: &[Vec<f64>],
    id: u32,
    rhs: &Expr,
    npts: usize,
    lo: f64,
    hi: f64,
    span: Span,
) -> Result<Vec<f64>, Vec<IrError>> {
    let mut lw = Lw::new(env, true, id);
    for t in built.iter().take(id as usize) {
        lw.dag.add_table(t.clone());
    }
    let root = lw.expr(rhs);
    let mut errors = std::mem::take(&mut lw.errors);
    if lw.dag.exceeded() {
        let msg = "the table formula needs more nodes than the limit";
        errors.push(IrError::new(IrErrorKind::TooLarge, span, msg));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut x = vec![0.0; env.init.len() + 1];
    if let Some(dst) = x.get_mut(1..) {
        dst.copy_from_slice(&env.init);
    }
    let mut p = env.par_values.clone();
    p.push(0.0);
    let dx = (hi - lo) / (npts - 1) as f64;
    let mut data = Vec::with_capacity(npts + 3);
    data.extend([npts as f64, lo, hi]);
    let mut out = [0.0];
    for i in 0..npts {
        if let Some(t) = x.first_mut() {
            *t = lo + i as f64 * dx;
        }
        eval(&lw.dag, &[root], &x, &p, &[], &mut out);
        data.push(out[0]);
    }
    Ok(data)
}

#[cfg(test)]
#[path = "lower_tests.rs"]
mod tests;
