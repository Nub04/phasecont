//! Maps the names of the source model to names `.ode` accepts.
//!
//! `.ode` names start with a letter, hold letters, digits, `_` and `.`, ignore case, and
//! cannot be a builtin, a keyword or one of the statement words. A name that needs a change
//! gets one (a trailing `_`, or a leading `v`) and a warning. Two names that `.ode` reads as
//! one are an error: the importer does not pick which one wins.

use std::collections::HashMap;

use crate::error::{ImportError, Kind, Result};

/// The internal key for the time variable. It cannot be a source name, so a species called `t`
/// never reads as time.
pub const TIME: &str = "\u{1}time";

/// Words the `.ode` reader treats as statements when they start a line.
const STATEMENT_WORDS: [&str; 10] = [
    "number", "set", "options", "only", "special", "solv", "export", "volt", "hom_bcs", "done",
];

/// True when `.ode` reserves the lower-case `key`.
pub fn is_reserved(key: &str) -> bool {
    matches!(key, "t" | "pi" | "not" | "if" | "then" | "else")
        || is_arg(key)
        || pc_ode::builtins::builtin(key).is_some()
        || STATEMENT_WORDS.contains(&key)
}

fn is_arg(key: &str) -> bool {
    key.strip_prefix("arg")
        .is_some_and(|r| r.len() == 1 && matches!(r.as_bytes()[0], b'1'..=b'9'))
}

#[derive(Debug, Default)]
pub struct Namer {
    by_source: HashMap<String, String>,
    /// Lower-case `.ode` name to the source name that owns it.
    used: HashMap<String, String>,
    warnings: Vec<String>,
}

impl Namer {
    pub fn new() -> Namer {
        Namer::default()
    }

    /// Gives `source` its `.ode` name. Asking again for the same source name returns the same
    /// `.ode` name.
    pub fn declare(&mut self, source: &str) -> Result<String> {
        if let Some(n) = self.by_source.get(source) {
            return Ok(n.clone());
        }
        if source.is_empty() {
            return Err(ImportError::new(Kind::Syntax, "a name is empty"));
        }
        if let Some(bad) = source
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '.'))
        {
            return Err(ImportError::new(
                Kind::Unsupported,
                format!(
                    "the name `{source}` has the character `{bad}`, which `.ode` names cannot hold"
                ),
            ));
        }
        let mut name = source.to_string();
        if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
            name.insert(0, 'v');
        }
        if is_reserved(&name.to_ascii_lowercase()) {
            name.push('_');
        }
        if name != source {
            self.warnings.push(format!(
                "`{source}` is written as `{name}` in the `.ode` text"
            ));
        }
        let key = name.to_ascii_lowercase();
        if let Some(other) = self.used.get(&key) {
            return Err(ImportError::new(
                Kind::NameCollision,
                format!(
                    "`{source}` and `{other}` are the same name in `.ode`, which ignores case \
                     (and the rename of a reserved name): rename one of them"
                ),
            ));
        }
        self.used.insert(key, source.to_string());
        self.by_source.insert(source.to_string(), name.clone());
        Ok(name)
    }

    /// The `.ode` name of an already declared source name.
    pub fn lookup(&self, source: &str) -> Option<&str> {
        self.by_source.get(source).map(String::as_str)
    }

    /// The text an expression prints for the variable `source`. The time key prints as `t`.
    pub fn print_name(&self, source: &str) -> String {
        if source == TIME {
            "t".to_string()
        } else {
            self.lookup(source).unwrap_or(source).to_string()
        }
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }
}
