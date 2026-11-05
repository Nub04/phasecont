//! Turns an imported model into `.ode` text.
//!
//! The emitter checks the model first: every name an expression reads is declared, every state
//! has an initial value, and every number is finite. Nothing defaults to 0. The text lists
//! `par`, `init`, the fixed quantities, the derivatives and `done`, and every expression is fully
//! parenthesized.

use std::collections::HashSet;

use crate::error::{ImportError, Kind, Result};
use crate::expr::Expr;
use crate::names::{Namer, TIME};

/// Largest `.ode` text the emitter returns. The engine reads up to 16 MiB.
pub const MAX_OUTPUT: usize = 8 << 20;
/// Entries on one `par` or `init` line.
const PER_LINE: usize = 8;

#[derive(Debug, Clone)]
pub struct State {
    /// The name as the source wrote it.
    pub name: String,
    /// The right side of `name' = ...`.
    pub rhs: Expr,
    /// The initial value. The emitter refuses a state without one.
    pub init: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct OdeModel {
    pub title: String,
    /// Parameters with their values, in the order the source gave them.
    pub params: Vec<(String, f64)>,
    /// Fixed quantities in evaluation order. One may read parameters, states, `t` and the
    /// quantities before it.
    pub fixed: Vec<(String, Expr)>,
    pub states: Vec<State>,
}

/// The result of an import.
#[derive(Debug, Clone, PartialEq)]
pub struct Imported {
    /// The `.ode` text.
    pub ode: String,
    /// Changes the importer made that the user should know about.
    pub warnings: Vec<String>,
}

pub fn emit(m: &OdeModel) -> Result<Imported> {
    if m.states.is_empty() {
        return Err(ImportError::new(
            Kind::NoDynamics,
            "the model has no differential equation",
        ));
    }
    let mut namer = Namer::new();
    let mut seen: HashSet<&str> = HashSet::new();
    let all = m
        .params
        .iter()
        .map(|(n, _)| n.as_str())
        .chain(m.fixed.iter().map(|(n, _)| n.as_str()))
        .chain(m.states.iter().map(|s| s.name.as_str()));
    for n in all {
        if !seen.insert(n) {
            return Err(ImportError::new(
                Kind::NameCollision,
                format!(
                    "`{n}` is declared more than once (as a parameter, a fixed quantity or a state)"
                ),
            ));
        }
    }
    for (n, v) in &m.params {
        namer.declare(n)?;
        if !v.is_finite() {
            return Err(ImportError::new(
                Kind::Unsupported,
                format!("the parameter `{n}` is infinite or not a number"),
            ));
        }
    }
    for (n, _) in &m.fixed {
        namer.declare(n)?;
    }
    for s in &m.states {
        namer.declare(&s.name)?;
    }

    // Names an expression may read.
    let mut known: HashSet<&str> = HashSet::new();
    known.insert(TIME);
    known.extend(m.params.iter().map(|(n, _)| n.as_str()));
    known.extend(m.states.iter().map(|s| s.name.as_str()));
    let fixed_names: HashSet<&str> = m.fixed.iter().map(|(n, _)| n.as_str()).collect();
    let mut earlier: HashSet<&str> = HashSet::new();
    for (n, e) in &m.fixed {
        check_reads(e, n, &known, &earlier, &fixed_names)?;
        earlier.insert(n.as_str());
    }
    let all_fixed = fixed_names.clone();
    for s in &m.states {
        check_reads(&s.rhs, &s.name, &known, &all_fixed, &fixed_names)?;
    }

    let mut out = String::new();
    out.push_str("# ");
    out.push_str(&comment(&m.title));
    out.push('\n');

    for chunk in m.params.chunks(PER_LINE) {
        out.push_str("par ");
        let items: Vec<String> = chunk
            .iter()
            .map(|(n, v)| format!("{}={}", namer.print_name(n), pc_ode::format_number(*v)))
            .collect();
        out.push_str(&items.join(","));
        out.push('\n');
    }

    let mut inits = Vec::with_capacity(m.states.len());
    for s in &m.states {
        let Some(v) = s.init else {
            return Err(ImportError::new(
                Kind::MissingValue,
                format!("the state `{}` has no initial value", s.name),
            ));
        };
        if !v.is_finite() {
            return Err(ImportError::new(
                Kind::Unsupported,
                format!(
                    "the initial value of `{}` is infinite or not a number",
                    s.name
                ),
            ));
        }
        inits.push(format!(
            "{}={}",
            namer.print_name(&s.name),
            pc_ode::format_number(v)
        ));
    }
    for chunk in inits.chunks(PER_LINE) {
        out.push_str("init ");
        out.push_str(&chunk.join(","));
        out.push('\n');
    }

    let print = |n: &str| namer.print_name(n);
    for (n, e) in &m.fixed {
        out.push_str(&print(n));
        out.push('=');
        out.push_str(&e.to_ode(&print));
        out.push('\n');
        guard(&out)?;
    }
    for s in &m.states {
        out.push_str(&print(&s.name));
        out.push_str("'=");
        out.push_str(&s.rhs.to_ode(&print));
        out.push('\n');
        guard(&out)?;
    }
    out.push_str("done\n");

    Ok(Imported {
        ode: out,
        warnings: namer.take_warnings(),
    })
}

fn guard(out: &str) -> Result<()> {
    if out.len() > MAX_OUTPUT {
        return Err(ImportError::new(
            Kind::TooLarge,
            format!("the `.ode` text would be longer than {MAX_OUTPUT} bytes"),
        ));
    }
    Ok(())
}

/// Checks that `e` reads only declared names. A fixed quantity may read only the ones before it
/// (`earlier`); everything else sees all of them.
fn check_reads(
    e: &Expr,
    owner: &str,
    known: &HashSet<&str>,
    earlier: &HashSet<&str>,
    all_fixed: &HashSet<&str>,
) -> Result<()> {
    let mut bad: Option<ImportError> = None;
    e.visit_vars(&mut |v| {
        if bad.is_some() || known.contains(v) || earlier.contains(v) {
            return;
        }
        bad = Some(if all_fixed.contains(v) {
            ImportError::new(
                Kind::Undeclared,
                format!(
                    "`{owner}` reads the fixed quantity `{v}` before it is defined; \
                     `.ode` evaluates fixed quantities in file order"
                ),
            )
        } else {
            ImportError::new(
                Kind::Undeclared,
                format!("`{owner}` reads `{v}`, which nothing declares"),
            )
        });
    });
    match bad {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// A title as a one-line `.ode` comment: ASCII only, no line break.
fn comment(title: &str) -> String {
    let s: String = title
        .chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else {
                ' '
            }
        })
        .collect();
    s.trim().to_string()
}
