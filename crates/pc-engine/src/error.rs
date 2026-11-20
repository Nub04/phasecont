//! The error type of the engine.

use pc_ir::machine::EvalError;
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
    /// A call into a recursive function stopped the evaluation: one more call would pass the
    /// cap on active calls, or the call frames ran out of memory. The text names the function
    /// and the cap. It is a numeric error, but unlike `NotFinite` no step retries on it.
    CallLimit(String),
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
            EngineError::CallLimit(m) => write!(f, "recursion limit: {m}"),
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

impl From<EvalError> for EngineError {
    fn from(e: EvalError) -> Self {
        match e {
            EvalError::Depth { func, cap } => EngineError::CallLimit(format!(
                "a call of {func} would make more than {cap} calls of recursive functions \
                 active at once"
            )),
            EvalError::OutOfMemory => EngineError::CallLimit(
                "out of memory for the call frames of recursive functions".into(),
            ),
            EvalError::Malformed(m) => EngineError::Model(format!("malformed call DAG: {m}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_errors_become_call_limits_naming_the_function_and_the_cap() {
        let e = EngineError::from(EvalError::Depth {
            func: "sum".into(),
            cap: 1_000_000,
        });
        assert_eq!(
            e.to_string(),
            "recursion limit: a call of sum would make more than 1000000 calls of recursive \
             functions active at once"
        );
        assert!(matches!(
            EngineError::from(EvalError::OutOfMemory),
            EngineError::CallLimit(_)
        ));
        assert!(matches!(
            EngineError::from(EvalError::Malformed("x".into())),
            EngineError::Model(_)
        ));
    }
}
