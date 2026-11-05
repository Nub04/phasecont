use std::fmt;

/// Why a render failed. [`ErrorKind::as_str`] gives the identifier that the MATLAB layer
/// appends to `phasecont:latex:`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The document has no `digits`.
    MissingDigits,
    /// `digits` is not an integer from 1 to 17.
    BadDigits,
    /// A value is `null`, NaN or infinite.
    NonFinite,
    /// A figure has no series, or a series has no points.
    EmptySeries,
    /// A table has no rows.
    EmptyTable,
    /// `x` and `y` differ in length, or a row has the wrong number of values.
    LengthMismatch,
    /// `basename` is empty or has a character outside `A-Z a-z 0-9 _ -`.
    BadName,
    /// The JSON is malformed, a key is unknown or a value has the wrong type.
    BadInput,
}

impl ErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::MissingDigits => "missingDigits",
            ErrorKind::BadDigits => "badDigits",
            ErrorKind::NonFinite => "nonFinite",
            ErrorKind::EmptySeries => "emptySeries",
            ErrorKind::EmptyTable => "emptyTable",
            ErrorKind::LengthMismatch => "lengthMismatch",
            ErrorKind::BadName => "badName",
            ErrorKind::BadInput => "badInput",
        }
    }
}

/// A render error: a kind and a message that names the offending series, point or row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatexError {
    kind: ErrorKind,
    message: String,
}

impl LatexError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        LatexError {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for LatexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LatexError {}
