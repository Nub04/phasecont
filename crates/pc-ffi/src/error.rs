//! Error codes returned across the C ABI and the error type behind them.

use std::fmt;

use pc_engine::EngineError;
use pc_ensemble::EnsembleError;

pub const PC_OK: i32 = 0;
pub const PC_ERR_PANIC: i32 = 1;
pub const PC_ERR_BAD_ARG: i32 = 2;
pub const PC_ERR_BAD_HANDLE: i32 = 3;
pub const PC_ERR_INTERNAL: i32 = 4;
pub const PC_ERR_MODEL: i32 = 5;
pub const PC_ERR_BAD_PROBLEM: i32 = 6;
pub const PC_ERR_NON_AUTONOMOUS: i32 = 7;
pub const PC_ERR_NUMERIC: i32 = 8;
pub const PC_ERR_GPU_UNAVAILABLE: i32 = 9;
pub const PC_ERR_GPU_COMPILE: i32 = 10;
pub const PC_ERR_GPU_LAUNCH: i32 = 11;
pub const PC_ERR_GPU_TIMEOUT: i32 = 12;
pub const PC_ERR_UNSUPPORTED_MODEL: i32 = 13;

/// A failure that an export turns into an error code and a `last_error` text.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A panic that was caught; the text is already `panic: <msg> (at <loc>)`.
    Panic(String),
    BadArg(String),
    BadHandle(String),
    Internal(String),
    Model(String),
    BadProblem(String),
    NonAutonomous(String),
    Numeric(String),
    /// The GPU backend cannot start, or the device asked for is not there.
    GpuUnavailable(String),
    /// The device compiler rejected the kernel.
    GpuCompile(String),
    /// A copy, launch or synchronize failed on the device.
    GpuLaunch(String),
    /// The driver stopped a launch that ran too long.
    GpuTimeout(String),
    /// The model uses a construct the ensemble kernels cannot run.
    UnsupportedModel(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The code the C caller sees.
    pub fn code(&self) -> i32 {
        match self {
            Error::Panic(_) => PC_ERR_PANIC,
            Error::BadArg(_) => PC_ERR_BAD_ARG,
            Error::BadHandle(_) => PC_ERR_BAD_HANDLE,
            Error::Internal(_) => PC_ERR_INTERNAL,
            Error::Model(_) => PC_ERR_MODEL,
            Error::BadProblem(_) => PC_ERR_BAD_PROBLEM,
            Error::NonAutonomous(_) => PC_ERR_NON_AUTONOMOUS,
            Error::Numeric(_) => PC_ERR_NUMERIC,
            Error::GpuUnavailable(_) => PC_ERR_GPU_UNAVAILABLE,
            Error::GpuCompile(_) => PC_ERR_GPU_COMPILE,
            Error::GpuLaunch(_) => PC_ERR_GPU_LAUNCH,
            Error::GpuTimeout(_) => PC_ERR_GPU_TIMEOUT,
            Error::UnsupportedModel(_) => PC_ERR_UNSUPPORTED_MODEL,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Panic(m)
            | Error::BadArg(m)
            | Error::BadHandle(m)
            | Error::Internal(m)
            | Error::Model(m)
            | Error::BadProblem(m)
            | Error::NonAutonomous(m)
            | Error::Numeric(m)
            | Error::GpuUnavailable(m)
            | Error::GpuCompile(m)
            | Error::GpuLaunch(m)
            | Error::GpuTimeout(m)
            | Error::UnsupportedModel(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<EngineError> for Error {
    fn from(e: EngineError) -> Self {
        let text = e.to_string();
        match e {
            EngineError::NonAutonomous => Error::NonAutonomous(text),
            EngineError::BadProblem(_) => Error::BadProblem(text),
            EngineError::Model(_) => Error::Model(text),
            EngineError::NotFinite(_)
            | EngineError::Singular(_)
            | EngineError::NewtonFailed { .. }
            | EngineError::StepTooSmall { .. } => Error::Numeric(text),
            EngineError::Cancelled => Error::Internal(text),
        }
    }
}

impl From<EnsembleError> for Error {
    /// A bad job and a tolerance below what the precision resolves are both `BadProblem`;
    /// the text starts with `bad ensemble job` or `tolerance too tight`, which is how the
    /// MATLAB side tells them apart. A cancelled run is not an error to the caller, so it
    /// reaches here only when a path forgot to handle it.
    fn from(e: EnsembleError) -> Self {
        let text = e.to_string();
        match e {
            EnsembleError::BadJob(_) | EnsembleError::ToleranceTooTight(_) => {
                Error::BadProblem(text)
            }
            EnsembleError::Unsupported(_) => Error::UnsupportedModel(text),
            EnsembleError::GpuUnavailable(_) => Error::GpuUnavailable(text),
            EnsembleError::GpuCompile(_) => Error::GpuCompile(text),
            EnsembleError::GpuLaunch(_) => Error::GpuLaunch(text),
            EnsembleError::GpuTimeout(_) => Error::GpuTimeout(text),
            EnsembleError::Cancelled => Error::Internal(text),
        }
    }
}

pub(crate) fn bad_arg(msg: impl Into<String>) -> Error {
    Error::BadArg(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_errors_map_to_abi_errors() {
        let cases: Vec<(EngineError, i32)> = vec![
            (EngineError::NonAutonomous, PC_ERR_NON_AUTONOMOUS),
            (EngineError::BadProblem("x".into()), PC_ERR_BAD_PROBLEM),
            (EngineError::Model("x".into()), PC_ERR_MODEL),
            (EngineError::NotFinite("f".into()), PC_ERR_NUMERIC),
            (EngineError::Singular("x".into()), PC_ERR_NUMERIC),
            (
                EngineError::NewtonFailed {
                    iters: 3,
                    residual: 1.0,
                },
                PC_ERR_NUMERIC,
            ),
            (EngineError::StepTooSmall { h: 1e-12 }, PC_ERR_NUMERIC),
            (EngineError::Cancelled, PC_ERR_INTERNAL),
        ];
        for (e, code) in cases {
            let text = e.to_string();
            let mapped = Error::from(e);
            assert_eq!(mapped.code(), code, "{text}");
            assert_eq!(mapped.to_string(), text);
        }
    }

    #[test]
    fn ensemble_errors_map_to_abi_errors() {
        let cases: Vec<(EnsembleError, i32, &str)> = vec![
            (
                EnsembleError::BadJob("x".into()),
                PC_ERR_BAD_PROBLEM,
                "bad ensemble job",
            ),
            (
                EnsembleError::ToleranceTooTight("x".into()),
                PC_ERR_BAD_PROBLEM,
                "tolerance too tight",
            ),
            (
                EnsembleError::Unsupported("x".into()),
                PC_ERR_UNSUPPORTED_MODEL,
                "unsupported model",
            ),
            (
                EnsembleError::GpuUnavailable("x".into()),
                PC_ERR_GPU_UNAVAILABLE,
                "GPU unavailable",
            ),
            (
                EnsembleError::GpuCompile("x".into()),
                PC_ERR_GPU_COMPILE,
                "GPU compile failed",
            ),
            (
                EnsembleError::GpuLaunch("x".into()),
                PC_ERR_GPU_LAUNCH,
                "GPU launch failed",
            ),
            (
                EnsembleError::GpuTimeout("x".into()),
                PC_ERR_GPU_TIMEOUT,
                "GPU launch timed out",
            ),
            (
                EnsembleError::Cancelled,
                PC_ERR_INTERNAL,
                "ensemble run cancelled",
            ),
        ];
        for (e, code, prefix) in cases {
            let text = e.to_string();
            let mapped = Error::from(e);
            assert_eq!(mapped.code(), code, "{text}");
            assert_eq!(mapped.to_string(), text);
            assert!(text.starts_with(prefix), "{text}");
        }
    }

    #[test]
    fn codes_are_distinct_and_fixed() {
        let all = [
            Error::Panic(String::new()),
            Error::BadArg(String::new()),
            Error::BadHandle(String::new()),
            Error::Internal(String::new()),
            Error::Model(String::new()),
            Error::BadProblem(String::new()),
            Error::NonAutonomous(String::new()),
            Error::Numeric(String::new()),
            Error::GpuUnavailable(String::new()),
            Error::GpuCompile(String::new()),
            Error::GpuLaunch(String::new()),
            Error::GpuTimeout(String::new()),
            Error::UnsupportedModel(String::new()),
        ];
        let codes: Vec<i32> = all.iter().map(Error::code).collect();
        assert_eq!(codes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    }
}
