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

/// More diagnostics than [`MAX_DIAGNOSTICS`].
pub const TOO_MANY_DIAGNOSTICS: Code = Code(73);

/// The most diagnostics one pass keeps, and the most [`load`](crate::load)
/// returns before its summary. A pass that finds more counts the rest
/// without storing them.
pub const MAX_DIAGNOSTICS: usize = 10_000;

/// Collects up to [`MAX_DIAGNOSTICS`] diagnostics and counts the rest, so
/// that a hostile input cannot make a pass hold millions of them.
#[derive(Debug, Default)]
pub(crate) struct Sink {
    items: Vec<Diagnostic>,
    dropped: usize,
}

impl Sink {
    pub(crate) fn push(&mut self, d: Diagnostic) {
        if self.items.len() < MAX_DIAGNOSTICS {
            self.items.push(d);
        } else {
            self.dropped += 1;
        }
    }

    /// The number of stored diagnostics.
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn get_mut(&mut self, i: usize) -> &mut Diagnostic {
        &mut self.items[i]
    }

    /// The stored diagnostics and the number that were not stored.
    pub(crate) fn finish(self) -> (Vec<Diagnostic>, usize) {
        (self.items, self.dropped)
    }
}

/// `diags` with the "N more diagnostics not shown" error added at the end
/// when `dropped` is not 0. The error sits at the end of the last
/// diagnostic, or at the start of file `file` when there is none.
pub(crate) fn with_summary(
    mut diags: Vec<Diagnostic>,
    dropped: usize,
    file: crate::span::FileId,
) -> Vec<Diagnostic> {
    if dropped > 0 {
        let at = diags.last().map_or(0, |d| d.span.end);
        let file = diags.last().map_or(file, |d| d.span.file);
        let message = format!(
            "{dropped} more diagnostics not shown; the limit is {MAX_DIAGNOSTICS} per pass"
        );
        diags.push(Diagnostic::new(
            TOO_MANY_DIAGNOSTICS,
            Span::new(file, at, at),
            message,
        ));
    }
    diags
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
    (
        Code(54),
        Severity::Warning,
        "`delay` is read only by `pc-cli dde-eq`, `dde-branch`, `dde-hopf` and `dde-lyap`",
    ),
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
    // Includes (`include.rs`).
    (Code(59), Severity::Error, "quoted `#include` file name"),
    (Code(60), Severity::Error, "included file cannot be read"),
    (Code(61), Severity::Error, "include cycle"),
    (
        Code(62),
        Severity::Error,
        "includes nested more than 16 deep",
    ),
    (
        Code(63),
        Severity::Error,
        "`#include` inside an array block",
    ),
    (
        Code(64),
        Severity::Error,
        "more than 4096 includes in total",
    ),
    // Arrays (`expand.rs`).
    (
        Code(65),
        Severity::Error,
        "more than one range in a statement",
    ),
    (Code(66), Severity::Error, "range inside an array block"),
    (
        Code(67),
        Severity::Error,
        "index form with no range in scope",
    ),
    (Code(68), Severity::Error, "negative array index"),
    (Code(69), Severity::Error, "array index too large"),
    (
        Code(70),
        Severity::Error,
        "`markov` inside an array block or with a range",
    ),
    (
        Code(71),
        Severity::Error,
        "arrays expand past the statement or work limit",
    ),
    (
        Code(72),
        Severity::Error,
        "included files add more statements or text than the limit",
    ),
    (Code(73), Severity::Error, "more diagnostics than the limit"),
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

    const LANGUAGE: &str = include_str!("../docs/language.md");

    fn severity_word(s: Severity) -> &'static str {
        match s {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }

    /// Compares the rows of the "Diagnostic codes" table in `doc`
    /// (`| ODE0001 | error | meaning |`) with `table`: same codes, same
    /// order, same severities, none twice.
    fn check_doc_table(doc: &str, table: &[(Code, Severity, &str)]) -> Result<(), String> {
        let start = doc
            .find("\n## Diagnostic codes\n")
            .ok_or("no Diagnostic codes section")?;
        let section = &doc[start + 1..];
        let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
        let mut rows: Vec<(String, String)> = Vec::new();
        for line in section[..end].lines().map(str::trim_start) {
            if !line
                .strip_prefix('|')
                .is_some_and(|r| r.trim_start().starts_with("ODE"))
            {
                continue;
            }
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            if cells.len() != 5 || cells[3].is_empty() {
                return Err(format!("row needs a code, severity and meaning: {line}"));
            }
            rows.push((cells[1].to_string(), cells[2].to_string()));
        }
        for (i, (code, severity)) in rows.iter().enumerate() {
            if rows[..i].iter().any(|(c, _)| c == code) {
                return Err(format!("{code} has two rows"));
            }
            match table.iter().find(|(c, _, _)| c.to_string() == *code) {
                None => return Err(format!("{code} is not in CODES")),
                Some((_, s, _)) if severity_word(*s) != severity => {
                    return Err(format!(
                        "{code} is {} in CODES but {severity} in the doc",
                        severity_word(*s)
                    ));
                }
                Some(_) => {}
            }
        }
        for (i, (code, _, _)) in table.iter().enumerate() {
            match rows.get(i) {
                Some((c, _)) if *c == code.to_string() => {}
                Some((c, _)) => return Err(format!("row {} is {c}, expected {code}", i + 1)),
                None => return Err(format!("{code} has no row")),
            }
        }
        if rows.len() > table.len() {
            return Err("the doc has more rows than CODES".to_string());
        }
        Ok(())
    }

    #[test]
    fn language_md_lists_every_code() {
        check_doc_table(LANGUAGE, CODES).unwrap();
    }

    #[test]
    fn doc_table_checker_rejects_mismatches() {
        let table = [
            (Code(1), Severity::Error, "a"),
            (Code(2), Severity::Warning, "b"),
        ];
        let doc = |rows: &str| format!("x\n## Diagnostic codes\n\n{rows}\n## Next\n");
        let good = doc("| ODE0001 | error | a |\n| ODE0002 | warning | b |\n");
        assert_eq!(check_doc_table(&good, &table), Ok(()));
        let tight = doc("|ODE0001|error|a|\n  | ODE0002 |  warning  | b |\n");
        assert_eq!(check_doc_table(&tight, &table), Ok(()));
        let cases = [
            ("| ODE0001 | error | a |\n", "has no row"),
            (
                "| ODE0001 | error | a |\n| ODE0002 | warning | b |\n| ODE0003 | error | c |\n",
                "not in CODES",
            ),
            (
                "| ODE0001 | error | a |\n| ODE0002 | error | b |\n",
                "in the doc",
            ),
            (
                "| ODE0002 | warning | b |\n| ODE0001 | error | a |\n",
                "expected",
            ),
            (
                "| ODE0001 | error | a |\n| ODE0001 | error | a |\n| ODE0002 | warning | b |\n",
                "two rows",
            ),
            (
                "| ODE0001 | error | |\n| ODE0002 | warning | b |\n",
                "needs a code",
            ),
        ];
        for (rows, want) in cases {
            let err = check_doc_table(&doc(rows), &table).unwrap_err();
            assert!(err.contains(want), "{rows}: {err}");
        }
        assert!(check_doc_table("no section", &table).is_err());
    }
}
