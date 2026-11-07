//! Hidden-parameter lowering of `delay(x, tau)` calls (SPEC §8d).
//!
//! `lower_dde` gives every distinct pair of a state and a delay time one extra parameter
//! slot, placed after the user parameters and before the `t` slot. The call lowers to a
//! read of that slot, so the vector field is an ordinary function `f(x, lag; p)` and the
//! JIT, the AOT code and the entry kinds work unchanged. At an equilibrium every lag
//! equals the state, which is how the engine collapses the slots (`pc-engine::dde`).
//!
//! The accepted form is `delay(x, tau)` with `x` the name of a state and `tau` a
//! parameter, a `number` constant or a positive numeric literal. Anything else is refused
//! with [`IrErrorKind::Delay`] and the span of the call. A delay inside a table, a map or
//! a derived parameter is refused too.

use crate::error::{IrError, IrErrorKind};
use crate::lower::formulas;
use pc_ode::{Expr, Span, Stmt, StmtKind};
use std::collections::HashMap;

/// The delay time of a hidden slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DelayTime {
    /// The user parameter with this index.
    Param(usize),
    /// A fixed positive time, written as a literal or as a `number` constant.
    Lit(f64),
}

/// One hidden parameter slot. Slot `j` of the list is parameter `n_user_params + j`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaySlot {
    /// Index of the delayed state.
    pub state: usize,
    pub time: DelayTime,
}

/// A hashable form of [`DelayTime`]: literals compare by bit pattern, which is exact for
/// the positive finite values the classifier lets through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TauKey {
    Param(usize),
    Lit(u64),
}

pub(crate) type SlotKey = (usize, TauKey);

impl DelayTime {
    pub(crate) fn key(self) -> TauKey {
        match self {
            DelayTime::Param(j) => TauKey::Param(j),
            DelayTime::Lit(v) => TauKey::Lit(v.to_bits()),
        }
    }
}

impl DelaySlot {
    pub(crate) fn key(self) -> SlotKey {
        (self.state, self.time.key())
    }
}

/// The name tables the classifier reads. `lower` and `Interp` each supply their own, so
/// both apply one rule to a call.
pub(crate) trait Names {
    fn state(&self, key: &str) -> Option<usize>;
    fn param(&self, key: &str) -> Option<usize>;
    fn number(&self, key: &str) -> Option<f64>;
}

/// True when `key` names a formal argument of the enclosing function, by its own name or
/// as `arg1`..`arg9`. Such a name hides the global state or parameter of the same key.
fn hides(formals: &[String], key: &str) -> bool {
    formals.iter().any(|p| p == key)
        || key
            .strip_prefix("arg")
            .and_then(|k| k.parse::<usize>().ok())
            .is_some_and(|k| (1..=9).contains(&k) && k <= formals.len())
}

const FORM: &str = "delay needs the name of a state and a delay time that is a parameter, a number \
     constant or a positive number";

/// Reads `delay(args)` as a state and a delay time. `formals` are the formal argument
/// keys of the user function whose body holds the call, empty elsewhere.
pub(crate) fn classify(
    args: &[Expr],
    formals: &[String],
    names: &dyn Names,
) -> Result<(usize, DelayTime), String> {
    let [a, b] = args else {
        return Err(format!("delay takes two arguments; {FORM}"));
    };
    let Expr::Name(x) = a else {
        return Err(format!(
            "the first argument of delay is not a state name; {FORM}"
        ));
    };
    let state = if hides(formals, &x.key) {
        None
    } else {
        names.state(&x.key)
    };
    let Some(state) = state else {
        return Err(format!("`{}` is not a state here; {FORM}", x.text));
    };
    let time = match b {
        Expr::Num { value, .. } => positive(*value)?,
        Expr::Name(id) => {
            let key = id.key.as_str();
            let reserved = hides(formals, key) || key == "t" || key == "pi";
            if reserved || names.state(key).is_some() {
                return Err(format!("`{}` cannot be a delay time; {FORM}", id.text));
            } else if let Some(j) = names.param(key) {
                DelayTime::Param(j)
            } else if let Some(v) = names.number(key) {
                positive(v)?
            } else {
                return Err(format!("`{}` cannot be a delay time; {FORM}", id.text));
            }
        }
        _ => {
            return Err(format!(
                "the delay time is not a plain name or number; {FORM}"
            ));
        }
    };
    Ok((state, time))
}

fn positive(v: f64) -> Result<DelayTime, String> {
    if v.is_finite() && v > 0.0 {
        Ok(DelayTime::Lit(v))
    } else {
        Err(format!("the delay time {v} is not positive and finite"))
    }
}

/// Calls `f` on the arguments of every `delay` call inside `e`, outer calls first.
fn each_delay<'a>(e: &'a Expr, f: &mut dyn FnMut(&'a [Expr], Span)) {
    match e {
        Expr::Unary { operand, .. } => each_delay(operand, f),
        Expr::Binary { lhs, rhs, .. } => {
            each_delay(lhs, f);
            each_delay(rhs, f);
        }
        Expr::If {
            cond, then, else_, ..
        } => {
            each_delay(cond, f);
            each_delay(then, f);
            each_delay(else_, f);
        }
        Expr::Call { name, args, span } => {
            if name.key == "delay" {
                f(args, *span);
            }
            for a in args {
                each_delay(a, f);
            }
        }
        Expr::Num { .. } | Expr::Name(_) | Expr::Index(_) | Expr::Error { .. } => {}
    }
}

/// What `scan` found.
pub(crate) struct Scan {
    pub slots: Vec<DelaySlot>,
    /// Position in `slots` of each distinct pair.
    pub index: HashMap<SlotKey, usize>,
    /// Number of `delay` calls seen, accepted or refused.
    pub calls: usize,
    pub errors: Vec<IrError>,
}

/// The name of slot `slot` in `param_names`: `delay(x,tau)` or `delay(x,1.5)`.
pub(crate) fn hidden_name(slot: &DelaySlot, states: &[String], params: &[String]) -> String {
    let x = states.get(slot.state).map_or("?", String::as_str);
    match slot.time {
        DelayTime::Param(j) => format!("delay({x},{})", params.get(j).map_or("?", String::as_str)),
        DelayTime::Lit(v) => format!("delay({x},{v})"),
    }
}

/// Walks every formula in statement order and collects the slots. A call in a table, a
/// map or a derived parameter is an error, and so is a call of the wrong form. With
/// `map_model` set every call is an error, since a map has no lag in continuous time.
pub(crate) fn scan(stmts: &[&Stmt], names: &dyn Names, map_model: bool) -> Scan {
    let mut out = Scan {
        slots: Vec::new(),
        index: HashMap::new(),
        calls: 0,
        errors: Vec::new(),
    };
    for s in stmts {
        let (formals, allowed): (Vec<String>, bool) = match &s.kind {
            StmtKind::Ode { .. } | StmtKind::Aux { .. } | StmtKind::Fixed { .. } => {
                (Vec::new(), true)
            }
            StmtKind::Function { params, .. } => {
                (params.iter().map(|p| p.key.clone()).collect(), true)
            }
            StmtKind::Derived { .. } | StmtKind::Map { .. } | StmtKind::Table { .. } => {
                (Vec::new(), false)
            }
            _ => continue,
        };
        for e in formulas(&s.kind) {
            each_delay(e, &mut |args, span| {
                out.calls += 1;
                let found = if map_model {
                    Err("a delay is not allowed in a map".to_string())
                } else if allowed {
                    classify(args, &formals, names)
                } else {
                    Err("a delay is not allowed in a table, a map or a derived parameter".into())
                };
                match found {
                    Ok((state, time)) => {
                        let slot = DelaySlot { state, time };
                        if !out.index.contains_key(&slot.key()) {
                            out.index.insert(slot.key(), out.slots.len());
                            out.slots.push(slot);
                        }
                    }
                    Err(msg) => out.errors.push(IrError::new(IrErrorKind::Delay, span, msg)),
                }
            });
        }
    }
    out
}

#[cfg(test)]
#[path = "delays_tests.rs"]
mod tests;
