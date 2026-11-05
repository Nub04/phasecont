//! Runs a job on a chosen device.

use std::sync::atomic::AtomicBool;

use crate::cpu::run_cpu;
use crate::cuda::run_cuda;
use crate::device::{Backend, DeviceInfo};
use crate::job::{EnsembleError, EnsembleResult, Job};
use crate::opencl::run_opencl;

/// Steps per member between two reads of the cancel flag on the CPU. The result does not
/// depend on it (`tests/determinism.rs` cuts a run at other sizes and compares bit for bit).
const CPU_STEPS_PER_LAUNCH: u64 = 1 << 16;

/// Runs `job` on `device`, in launches of [`CPU_STEPS_PER_LAUNCH`] steps per member on the
/// CPU and in chunked launches on a GPU, so `cancel` is read between launches on both.
///
/// A backend that is not built into this binary gives [`EnsembleError::Unsupported`] naming
/// it. The result carries the label of the backend that ran.
pub fn run_on(
    device: &DeviceInfo,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    match device.backend {
        Backend::Cpu => run_cpu(job, CPU_STEPS_PER_LAUNCH, cancel),
        Backend::Cuda => run_cuda(device, job, cancel),
        Backend::OpenCl => run_opencl(device, job, cancel),
        other => Err(EnsembleError::Unsupported(format!(
            "the {} backend is not built into this binary",
            other.name()
        ))),
    }
}
