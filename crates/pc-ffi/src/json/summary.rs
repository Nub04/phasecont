//! The counts shared by the text and JSON reports.

use pc_ode::{Loaded, ModelKind, Severity, SymbolKind};

pub struct Summary {
    pub kind: &'static str,
    pub states: usize,
    pub parameters: usize,
    pub fixed: usize,
    pub functions: usize,
    pub aux: usize,
    pub smooth: bool,
    pub simulation_only: bool,
    pub errors: usize,
    pub warnings: usize,
}

impl Summary {
    pub fn of(loaded: &Loaded) -> Summary {
        let info = &loaded.info;
        let count =
            |pred: fn(SymbolKind) -> bool| info.symbols.iter().filter(|s| pred(s.kind)).count();
        let sev = |s: Severity| {
            loaded
                .diagnostics
                .iter()
                .filter(|d| d.severity == s)
                .count()
        };
        Summary {
            kind: match info.kind {
                ModelKind::None => "none",
                ModelKind::Ode => "ode",
                ModelKind::Map => "map",
            },
            states: count(|k| matches!(k, SymbolKind::State(_))),
            parameters: count(|k| matches!(k, SymbolKind::Parameter)),
            fixed: count(|k| matches!(k, SymbolKind::Fixed)),
            functions: count(|k| matches!(k, SymbolKind::Function { .. })),
            aux: count(|k| matches!(k, SymbolKind::Aux)),
            smooth: info.smooth,
            simulation_only: info.simulation_only,
            errors: sev(Severity::Error),
            warnings: sev(Severity::Warning),
        }
    }
}
