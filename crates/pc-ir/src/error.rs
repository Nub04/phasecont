//! Errors from lowering a checked `.ode` model to the DAG.

use pc_ode::Span;

/// Why a model cannot be lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrErrorKind {
    /// The model has neither an ODE nor a map.
    NoEquations,
    /// The model uses `ran`, `normal`, `poisson`, `global`, `markov` or `wiener`.
    SimulationOnly,
    /// The model calls `delay(...)`.
    Delay,
    /// A statement, call or name that the lowering has no rule for.
    Unsupported,
    /// A table header or a table formula is bad.
    BadTable,
    /// The DAG grew past its node limit.
    TooLarge,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrError {
    pub kind: IrErrorKind,
    pub span: Span,
    pub message: String,
}

impl IrError {
    pub fn new(kind: IrErrorKind, span: Span, message: impl Into<String>) -> IrError {
        IrError {
            kind,
            span,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:?})", self.message, self.kind)
    }
}

impl std::error::Error for IrError {}
