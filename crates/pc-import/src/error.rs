//! Import errors. Each one has a kind with a stable id, which the MATLAB layer turns into
//! `phasecont:import:<id>`, and a message that names the thing at fault.

use std::fmt;

/// What went wrong. The ids are part of the MATLAB API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The text is not valid in the source language.
    Syntax,
    /// Valid in the source language, but PhaseCont does not translate it.
    Unsupported,
    /// A name that nothing declares.
    Undeclared,
    /// A value the model needs and the input does not give (a parameter, an initial value).
    MissingValue,
    /// Two things disagree: a count, a name list and a body, two initial values.
    Mismatch,
    /// Two names that `.ode` reads as one, or a name `.ode` reserves.
    NameCollision,
    /// The input is bigger or deeper than the importer accepts.
    TooLarge,
    /// The imported model has no differential equation.
    NoDynamics,
    /// The XML is not well formed.
    Xml,
}

impl Kind {
    /// The last part of the MATLAB error id.
    pub fn id(self) -> &'static str {
        match self {
            Kind::Syntax => "syntax",
            Kind::Unsupported => "unsupported",
            Kind::Undeclared => "undeclared",
            Kind::MissingValue => "missingValue",
            Kind::Mismatch => "mismatch",
            Kind::NameCollision => "nameCollision",
            Kind::TooLarge => "tooLarge",
            Kind::NoDynamics => "noDynamics",
            Kind::Xml => "xml",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportError {
    pub kind: Kind,
    pub message: String,
    /// 1-based line of the input, when the error has a position.
    pub line: Option<u32>,
    /// 1-based column, counted in characters.
    pub col: Option<u32>,
}

impl ImportError {
    pub fn new(kind: Kind, message: impl Into<String>) -> ImportError {
        ImportError {
            kind,
            message: message.into(),
            line: None,
            col: None,
        }
    }

    pub fn at(kind: Kind, message: impl Into<String>, line: u32, col: u32) -> ImportError {
        ImportError {
            kind,
            message: message.into(),
            line: Some(line),
            col: Some(col),
        }
    }

    /// `phasecont:import:<kind>`.
    pub fn error_id(&self) -> String {
        format!("phasecont:import:{}", self.kind.id())
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.col) {
            (Some(l), Some(c)) => write!(f, "{} (line {l}, column {c})", self.message),
            _ => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for ImportError {}

pub type Result<T> = std::result::Result<T, ImportError>;

/// Line and column (both 1-based, the column in characters) of byte `offset` in `text`.
pub fn line_col(text: &str, offset: usize) -> (u32, u32) {
    let end = offset.min(text.len());
    let mut line = 1u32;
    let mut col = 1u32;
    for (i, ch) in text.char_indices() {
        if i >= end {
            break;
        }
        if ch == '\n' {
            line = line.saturating_add(1);
            col = 1;
        } else {
            col = col.saturating_add(1);
        }
    }
    (line, col)
}
