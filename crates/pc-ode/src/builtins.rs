//! The builtin functions of the `.ode` language.
//!
//! [`BUILTINS`] holds one entry per name, in the order of the "Builtins"
//! table in `docs/language.md`. A test reads that table and checks that the
//! names, arities and flags agree. `if(c)then(a)else(b)` is an expression
//! form, not a call, so it is not listed here; `check` treats it as
//! non-smooth. `sum(..)of(..)` never reaches `check` as a call, since the
//! parser keeps such a line as an unsupported statement.

/// How far PhaseCont supports a builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Supported.
    Full,
    /// Supported, but XPPAUT 8.0 rejects it, so each use gives the
    /// portability warning (`ceil`).
    Portability,
    /// `delay`: kept and warned about (ODE0054); only `pc-cli dde-eq`, `dde-branch`, `dde-hopf`, `dde-lyap` and `dde-zh` read it.
    Delay,
    /// `shift`: kept and warned about (ODE0055).
    Shift,
    /// `del_shft`: kept and warned about (ODE0056).
    DelShft,
}

/// One builtin function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Builtin {
    /// Lower-case name.
    pub name: &'static str,
    /// Exact argument count, or `None` when `check` does not count them.
    pub arity: Option<usize>,
    /// False when the function lacks the derivatives continuation needs.
    /// True for the unsupported entries, which do not change the flag.
    pub smooth: bool,
    /// True when a model that calls it can only be simulated.
    pub simulation_only: bool,
    pub status: Status,
    /// Where the rule comes from: the start of the Source cell of its row
    /// in `docs/language.md`.
    pub source: &'static str,
}

const fn f(
    name: &'static str,
    arity: usize,
    smooth: bool,
    simulation_only: bool,
    source: &'static str,
) -> Builtin {
    Builtin {
        name,
        arity: Some(arity),
        smooth,
        simulation_only,
        status: Status::Full,
        source,
    }
}

/// Every builtin, one entry per name.
pub const BUILTINS: &[Builtin] = &[
    f("sin", 1, true, false, "manual, measured"),
    f("cos", 1, true, false, "manual, measured"),
    f("tan", 1, true, false, "manual, measured"),
    f("asin", 1, true, false, "manual, measured"),
    f("acos", 1, true, false, "manual, measured"),
    f("atan", 1, true, false, "manual, measured"),
    f("atan2", 2, true, false, "measured"),
    f("sinh", 1, true, false, "manual, measured"),
    f("cosh", 1, true, false, "manual, measured"),
    f("tanh", 1, true, false, "manual, measured"),
    f("exp", 1, true, false, "manual, measured"),
    f("ln", 1, true, false, "measured"),
    f("log", 1, true, false, "measured"),
    f("log10", 1, true, false, "measured"),
    f("sqrt", 1, true, false, "manual, measured"),
    f("abs", 1, false, false, "manual, measured"),
    f("heav", 1, false, false, "manual, measured"),
    f("sign", 1, false, false, "manual, measured"),
    f("mod", 2, false, false, "measured"),
    f("min", 2, false, false, "manual, measured"),
    f("max", 2, false, false, "manual, measured"),
    f("flr", 1, false, false, "measured"),
    Builtin {
        status: Status::Portability,
        ..f("ceil", 1, false, false, "manual; XPPAUT 8.0 rejects it")
    },
    f("erf", 1, true, false, "manual, measured"),
    f("erfc", 1, true, false, "manual, measured"),
    f("lgamma", 1, true, false, "measured"),
    f("besselj", 2, true, false, "manual, measured"),
    f("bessely", 2, true, false, "manual, measured"),
    f("ran", 1, false, true, "manual, measured"),
    f("normal", 2, false, true, "manual"),
    f("poisson", 1, false, true, "manual, measured"),
    Builtin {
        status: Status::Delay,
        ..f(
            "delay",
            2,
            true,
            false,
            "manual; warning, dde commands only",
        )
    },
    Builtin {
        arity: None,
        status: Status::Shift,
        ..f("shift", 0, true, false, "manual; unsupported, warning")
    },
    Builtin {
        arity: None,
        status: Status::DelShft,
        ..f("del_shft", 0, true, false, "manual; unsupported, warning")
    },
];

/// The builtin with the lower-case name `key`.
pub fn builtin(key: &str) -> Option<&'static Builtin> {
    BUILTINS.iter().find(|b| b.name == key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LANGUAGE: &str = include_str!("../docs/language.md");

    /// One builtin as the language.md table lists it: name, arity, smooth
    /// cell, simulation-only cell and source cell.
    type Row = (String, Option<usize>, String, String, String);

    /// The function rows of the "Builtins" table in language.md, one per
    /// name. Names that are not plain identifiers (`if(c)then(a)else(b)`,
    /// `sum(..)of(..)`) are left out.
    fn doc_rows() -> Vec<Row> {
        let start = LANGUAGE.find("\n## Builtins\n").expect("Builtins section");
        let section = &LANGUAGE[start + 1..];
        let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
        let section = &section[..end];
        let mut rows = Vec::new();
        for line in section.lines().filter(|l| l.starts_with("| `")) {
            // `\|` is an escaped bar inside a cell.
            let line = line.replace("\\|", "\u{1}");
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            assert_eq!(cells.len(), 8, "{line}");
            let arity = (!cells[2].is_empty()).then(|| {
                cells[2]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("{line}"))
            });
            for name in cells[1].split('`').skip(1).step_by(2) {
                for word in name.split_whitespace() {
                    let plain = word
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
                    if plain {
                        rows.push((
                            word.to_string(),
                            arity,
                            cells[4].to_string(),
                            cells[5].to_string(),
                            cells[6].to_string(),
                        ));
                    }
                }
            }
        }
        rows
    }

    fn flag(cell: &str, row: &str) -> Option<bool> {
        if cell.is_empty() {
            None
        } else if cell.starts_with("yes") {
            Some(true)
        } else if cell == "no" {
            Some(false)
        } else {
            panic!("{row}: bad flag {cell:?}")
        }
    }

    #[test]
    fn table_matches_language_md() {
        let rows = doc_rows();
        let names: Vec<&str> = rows.iter().map(|r| r.0.as_str()).collect();
        let ours: Vec<&str> = BUILTINS.iter().map(|b| b.name).collect();
        assert_eq!(names, ours, "names or order differ from language.md");
        for ((name, arity, smooth, sim, source), b) in rows.iter().zip(BUILTINS) {
            assert_eq!(*arity, b.arity, "{name}: arity");
            assert!(
                !b.source.is_empty() && source.starts_with(b.source),
                "{name}: source {:?} does not start the doc cell {source:?}",
                b.source
            );
            match (flag(smooth, name), flag(sim, name)) {
                (Some(smooth), Some(sim)) => {
                    assert_eq!(smooth, b.smooth, "{name}: smooth");
                    assert_eq!(sim, b.simulation_only, "{name}: simulation only");
                    assert!(
                        matches!(b.status, Status::Full | Status::Portability),
                        "{name}: status"
                    );
                }
                (None, None) => {
                    assert!(
                        matches!(b.status, Status::Delay | Status::Shift | Status::DelShft),
                        "{name}: status"
                    );
                    assert!(b.smooth && !b.simulation_only, "{name}: flags");
                }
                _ => panic!("{name}: one flag cell is empty"),
            }
        }
    }

    #[test]
    fn lookup_and_special_entries() {
        assert_eq!(builtin("atan2").unwrap().arity, Some(2));
        assert_eq!(builtin("ceil").unwrap().status, Status::Portability);
        assert_eq!(builtin("delay").unwrap().status, Status::Delay);
        assert_eq!(builtin("shift").unwrap().status, Status::Shift);
        assert_eq!(builtin("del_shft").unwrap().status, Status::DelShft);
        assert_eq!(builtin("shift").unwrap().arity, None);
        assert!(builtin("ran").unwrap().simulation_only);
        assert!(!builtin("abs").unwrap().smooth);
        assert!(builtin("lgamma").unwrap().smooth);
        assert!(builtin("e").is_none());
        assert!(builtin("int").is_none());
        assert!(builtin("gamma").is_none());
        assert!(builtin("fact").is_none());
        assert!(builtin("if").is_none());
        assert!(builtin("SIN").is_none(), "keys are lower case");
    }
}
