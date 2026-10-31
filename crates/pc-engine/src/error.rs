//! The error type of the engine.

use pc_jit::JitError;
use pc_linalg::LinalgError;

/// Why an engine call failed.
#[derive(Clone, Debug, PartialEq)]
pub enum EngineError {
    /// The model reads `t`; continuation needs an autonomous system.
    NonAutonomous,
    /// The problem or its arguments cannot be run (wrong lengths, unsupported model).
    BadProblem(String),
    /// The model failed to lower or to compile.
    Model(String),
    /// An entry produced NaN or infinity. Carries the entry kind.
    NotFinite(String),
    /// A matrix was singular where a solve needed it not to be.
    Singular(String),
    /// Newton's method did not reach the tolerance.
    NewtonFailed { iters: usize, residual: f64 },
    /// The step length fell below the minimum.
    StepTooSmall { h: f64 },
    /// The caller cancelled the run.
    Cancelled,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::NonAutonomous => f.write_str("the model depends on t"),
            EngineError::BadProblem(m) => write!(f, "bad problem: {m}"),
            EngineError::Model(m) => write!(f, "model error: {m}"),
            EngineError::NotFinite(k) => write!(f, "NaN or infinity in {k}"),
            EngineError::Singular(m) => write!(f, "singular matrix: {m}"),
            EngineError::NewtonFailed { iters, residual } => {
                write!(
                    f,
                    "Newton failed after {iters} iterations, residual {residual:e}"
                )
            }
            EngineError::StepTooSmall { h } => write!(f, "step length {h:e} is below the minimum"),
            EngineError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LinalgError> for EngineError {
    fn from(e: LinalgError) -> Self {
        match e {
            LinalgError::BadShape => EngineError::BadProblem(e.to_string()),
            LinalgError::NotFinite => EngineError::NotFinite("a linear algebra result".into()),
            LinalgError::Singular => EngineError::Singular(e.to_string()),
            LinalgError::NoConvergence => EngineError::Singular(
                "the factorization did not converge, so the matrix is treated as singular".into(),
            ),
            LinalgError::OutOfMemory => EngineError::BadProblem(
                "not enough memory for a cycle of this size; use a smaller NTST or NCOL".into(),
            ),
        }
    }
}

impl From<JitError> for EngineError {
    fn from(e: JitError) -> Self {
        EngineError::Model(e.to_string())
    }
}
