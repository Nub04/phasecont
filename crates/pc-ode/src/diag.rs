//! Diagnostics and their stable codes.

use std::fmt;

use crate::span::Span;

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

/// A suggested edit: replace the text under `span` with `replacement`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixIt {
    pub span: Span,
    pub replacement: String,
}

/// A stable diagnostic code, shown as `ODE` plus four digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Code(pub u16);

impl Code {
    /// The default severity of this code in [`CODES`]. A code missing from
    /// the table is an error.
    pub fn severity(self) -> Severity {
        CODES
            .iter()
            .find(|(c, _, _)| *c == self)
            .map_or(Severity::Error, |e| e.1)
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ODE{:04}", self.0)
    }
}

/// One message about the input, with an optional fix and related locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: Code,
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    pub fix_it: Option<FixIt>,
    pub notes: Vec<(Span, String)>,
}

impl Diagnostic {
    /// A diagnostic with the default severity of `code`, no fix and no notes.
    pub fn new(code: Code, span: Span, message: String) -> Diagnostic {
        Diagnostic {
            code,
            severity: code.severity(),
            span,
            message,
            fix_it: None,
            notes: Vec::new(),
        }
    }
}

/// Every diagnostic code with its default severity and a one-line summary.
///
/// Entries are in increasing code order. New codes are appended and existing
/// codes are never renumbered.
pub const CODES: &[(Code, Severity, &str)] = &[
    // Lexer (`lexer.rs`).
    (Code(1), Severity::Error, "malformed number"),
    (
        Code(2),
        Severity::Error,
        "number too large for a double precision value",
    ),
    (Code(3), Severity::Error, "stray character"),
    // Expression parser (`parser/expr.rs`).
    (Code(4), Severity::Error, "expected an expression"),
    (
        Code(5),
        Severity::Error,
        "missing operator between two operands",
    ),
    (
        Code(6),
        Severity::Error,
        "unexpected token in an expression",
    ),
    (Code(7), Severity::Error, "`!` is not an operator"),
    (Code(8), Severity::Error, "`=` inside an expression"),
    (Code(9), Severity::Error, "unclosed `(` or `[`"),
    (
        Code(10),
        Severity::Error,
        "prefix operator in the wrong place",
    ),
    (Code(11), Severity::Error, "unary `+`"),
    (Code(12), Severity::Error, "blank inside a token"),
    (Code(13), Severity::Warning, "XPPAUT 8.0 rejects this"),
    (
        Code(14),
        Severity::Warning,
        "operator precedence differs from C",
    ),
    (Code(15), Severity::Error, "unsupported index form"),
    (Code(16), Severity::Error, "malformed if expression"),
    (Code(17), Severity::Error, "expression too deep"),
    // Statement parser (`parser/stmt.rs`).
    (Code(18), Severity::Error, "old-style XPPAUT file"),
    (Code(19), Severity::Error, "unrecognized statement"),
    (Code(20), Severity::Error, "malformed statement"),
    (
        Code(21),
        Severity::Error,
        "unexpected text after a statement",
    ),
    (
        Code(22),
        Severity::Error,
        "comment not allowed after this statement",
    ),
    (Code(23), Severity::Error, "value is not a number literal"),
    (Code(24), Severity::Warning, "unknown option name"),
    (
        Code(25),
        Severity::Warning,
        "XPPAUT 8.0 ignores an option with blanks around `=`",
    ),
    (Code(26), Severity::Error, "malformed markov rows"),
    (Code(27), Severity::Error, "malformed array block"),
    (
        Code(28),
        Severity::Warning,
        "Volterra equations are not supported",
    ),
    (
        Code(29),
        Severity::Warning,
        "boundary value problems are not supported",
    ),
    (
        Code(30),
        Severity::Warning,
        "algebraic constraints are not supported",
    ),
    (Code(31), Severity::Warning, "`special` is not supported"),
    (Code(32), Severity::Warning, "`export` is not supported"),
    (Code(33), Severity::Warning, "`only` is not supported"),
    (
        Code(34),
        Severity::Warning,
        "`options` files are not supported",
    ),
    (Code(35), Severity::Warning, "`hom_bcs` is not supported"),
    (Code(36), Severity::Warning, "file tables are not supported"),
    (
        Code(37),
        Severity::Warning,
        "`sum(..)of(..)` is not supported",
    ),
    (
        Code(38),
        Severity::Warning,
        "active comments are not supported",
    ),
    // Name resolution and model checks (`check.rs`).
    (Code(39), Severity::Error, "undefined name"),
    (Code(40), Severity::Error, "reserved name declared"),
    (Code(41), Severity::Error, "name declared twice"),
    (
        Code(42),
        Severity::Error,
        "fixed quantity used above its definition",
    ),
    (Code(43), Severity::Error, "circular definition"),
    (
        Code(44),
        Severity::Error,
        "initial value for a name that is not a state",
    ),
    (Code(45), Severity::Error, "initial value given twice"),
    (Code(46), Severity::Error, "more than 9 function parameters"),
    (Code(47), Severity::Error, "recursive function"),
    (Code(48), Severity::Error, "wrong number of arguments"),
    (
        Code(49),
        Severity::Error,
        "function used as a value, or a value called as a function",
    ),
    (
        Code(50),
        Severity::Error,
        "`argK` outside a function that has it",
    ),
    (
        Code(51),
        Severity::Error,
        "`aux` quantity used in a formula",
    ),
    (Code(52), Severity::Error, "ODEs and maps in one file"),
    (
        Code(53),
        Severity::Error,
        "derived parameter uses a name that is not a parameter",
    ),
    (Code(54), Severity::Warning, "`delay` is not supported yet"),
    (Code(55), Severity::Warning, "`shift` is not supported"),
    (Code(56), Severity::Warning, "`del_shft` is not supported"),
    (
        Code(57),
        Severity::Warning,
        "undefined name that an unsupported statement may declare",
    ),
    (
        Code(58),
        Severity::Error,
        "name cannot be assigned in `global`",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks that codes are strictly increasing, so unique and sorted.
    fn check_sorted_unique(table: &[(Code, Severity, &str)]) -> Result<(), String> {
        for pair in table.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(format!("{} is not before {}", pair[0].0, pair[1].0));
            }
        }
        Ok(())
    }

    #[test]
    fn code_table_is_unique_and_sorted() {
        check_sorted_unique(CODES).unwrap();
    }

    #[test]
    fn code_table_summaries_are_one_line() {
        for (code, _, summary) in CODES {
            assert!(!summary.is_empty(), "{code} has an empty summary");
            assert!(!summary.contains('\n'), "{code} summary spans lines");
        }
    }

    #[test]
    fn checker_rejects_bad_tables() {
        let ok = [
            (Code(1), Severity::Error, "a"),
            (Code(2), Severity::Warning, "b"),
        ];
        assert!(check_sorted_unique(&ok).is_ok());
        assert!(check_sorted_unique(&[]).is_ok());
        let dup = [
            (Code(1), Severity::Error, "a"),
            (Code(1), Severity::Error, "b"),
        ];
        assert!(check_sorted_unique(&dup).is_err());
        let unsorted = [
            (Code(2), Severity::Error, "a"),
            (Code(1), Severity::Error, "b"),
        ];
        assert!(check_sorted_unique(&unsorted).is_err());
    }

    #[test]
    fn severity_comes_from_the_table() {
        assert_eq!(Code(1).severity(), Severity::Error);
        assert_eq!(Code(13).severity(), Severity::Warning);
        assert_eq!(Code(9999).severity(), Severity::Error);
        let d = Diagnostic::new(
            Code(13),
            Span::new(crate::span::FileId(0), 1, 2),
            "m".into(),
        );
        assert_eq!(d.severity, Severity::Warning);
        assert!(d.fix_it.is_none() && d.notes.is_empty());
    }

    #[test]
    fn code_displays_with_four_digits() {
        assert_eq!(Code(1).to_string(), "ODE0001");
        assert_eq!(Code(12).to_string(), "ODE0012");
        assert_eq!(Code(9999).to_string(), "ODE9999");
        assert_eq!(Code(65535).to_string(), "ODE65535");
    }
}
