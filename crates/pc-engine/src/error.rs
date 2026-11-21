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

impl EngineError {
    /// True for the recursion cap error, which no step retries on and no code may hide as a
    /// NaN, a missing value or an unrefined point (SPEC 8af-5, Errors).
    pub fn is_call_limit(&self) -> bool {
        matches!(self, EngineError::CallLimit(_))
    }
}

/// Keeps a recursion cap error as an error and maps every other error with `other`, which is
/// what the caller did with it before (SPEC 8af-5, Errors). An `Ok` passes through.
pub(crate) fn keep_call_limit<T>(
    r: Result<T, EngineError>,
    other: impl FnOnce(EngineError) -> T,
) -> Result<T, EngineError> {
    match r {
        Err(e) if !e.is_call_limit() => Ok(other(e)),
        r => r,
    }
}

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
    /// An evaluation error of a call, the cap or out of memory, stays a recursion limit error;
    /// every other JIT error is a model error.
    fn from(e: JitError) -> Self {
        match e {
            JitError::Eval(e) => EngineError::from(e),
            e => EngineError::Model(e.to_string()),
        }
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

    #[test]
    fn jit_eval_errors_keep_their_meaning_and_other_jit_errors_are_model_errors() {
        let e = EngineError::from(JitError::Eval(EvalError::Depth {
            func: "sum".into(),
            cap: 7,
        }));
        assert!(e.is_call_limit(), "{e:?}");
        assert!(e.to_string().contains("sum"), "{e}");
        assert!(EngineError::from(JitError::Eval(EvalError::OutOfMemory)).is_call_limit());
        for e in [
            JitError::Eval(EvalError::Malformed("x".into())),
            JitError::BadLength("x has 1 values, expected 2".into()),
            JitError::Cranelift("y".into()),
        ] {
            let mapped = EngineError::from(e);
            assert!(matches!(mapped, EngineError::Model(_)), "{mapped:?}");
        }
    }
}
