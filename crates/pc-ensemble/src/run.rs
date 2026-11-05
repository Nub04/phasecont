//! Runs a job on a chosen device.

use std::sync::atomic::AtomicBool;

use crate::cpu::run_cpu;
use crate::device::{Backend, DeviceInfo};
use crate::job::{EnsembleError, EnsembleResult, Job};

/// Runs `job` on `device`, in one launch on the CPU and in chunked launches on a GPU.
///
/// A backend that is not built into this binary gives [`EnsembleError::Unsupported`] naming
/// it. The result carries the label of the backend that ran.
pub fn run_on(
    device: &DeviceInfo,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    match device.backend {
        Backend::Cpu => run_cpu(job, u64::MAX, cancel),
        other => Err(EnsembleError::Unsupported(format!(
            "the {} backend is not built into this binary",
            other.name()
        ))),
    }
}
