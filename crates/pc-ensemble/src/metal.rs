//! The Metal backend: one thread per member, compiled at run time by the Metal driver.
//!
//! Metal exists on macOS only and has no `double`, so this backend runs single precision.
//! The host code is in `metal/host.rs` and is built on macOS only. Everything in this file
//! builds on every platform, so the rules that decide how a launch is cut and how a failure
//! is reported run in the tests on any machine. On other platforms the backend reports that
//! it is not built into the binary.
//!
//! The device comes from `MTLCopyAllDevices`, so on a Mac with two GPUs both show up. A run
//! looks its device up again by registry id and keeps every Objective-C object inside the
//! run. No such object lives in a static. A panic in Rust is caught by [`guard`]. An
//! Objective-C exception is not a panic and `guard` cannot catch it. The calls used here
//! return `nil` or an `NSError` on failure and are not expected to raise one.
//!
//! The command buffer reports a watchdog stop as `MTLCommandBufferErrorTimeout` (code 2).
//! The text check for "Impacting Interactivity" covers the error macOS gives when a launch
//! blocks the display. Both mappings are unverified on a Mac.
//!
//! [`guard`]: crate::gpu::guard

// Only the macOS host code calls the launch helpers. The tests run them on every platform.
#![cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]

use std::time::Duration;

use crate::job::EnsembleError;

/// The name of the kernel entry point.
pub(crate) const ENTRY: &str = "pc_ens";
/// Threads per threadgroup, unless the kernel allows fewer.
pub(crate) const GROUP: usize = 64;

/// `MTLCommandBufferErrorTimeout`.
const CODE_TIMEOUT: isize = 2;
/// `MTLCommandBufferErrorPageFault`.
const CODE_PAGE_FAULT: isize = 3;
/// `MTLCommandBufferErrorOutOfMemory`.
const CODE_OUT_OF_MEMORY: isize = 8;

/// Threads per threadgroup for a pipeline that allows `max_total` threads and runs its
/// threads in groups of `width`.
///
/// It aims at [`GROUP`], stays at or under `max_total` and is a whole number of execution
/// widths whenever the limit leaves room for one.
pub(crate) fn group_for(max_total: usize, width: usize) -> usize {
    let g = max_total.clamp(1, GROUP);
    let width = width.max(1);
    if g >= width { g / width * width } else { g }
}

/// The error for a command buffer that finished with an error.
///
/// `code` is the `NSError` code, `text` its localized description and `took` the time since
/// the launch started.
pub(crate) fn command_error(code: isize, text: &str, took: Duration) -> EnsembleError {
    let secs = took.as_secs_f64();
    if code == CODE_TIMEOUT || text.contains("Impacting Interactivity") {
        return EnsembleError::GpuTimeout(format!(
            "the launch was stopped after {secs:.1} s: {text} (code {code}), which looks like \
             the GPU watchdog"
        ));
    }
    let what = match code {
        CODE_PAGE_FAULT => "the kernel read or wrote outside its buffers",
        CODE_OUT_OF_MEMORY => "the device ran out of memory",
        _ => "the command buffer failed",
    };
    EnsembleError::GpuLaunch(format!("{what}: {text} (code {code})"))
}

#[cfg(target_os = "macos")]
mod host;
#[cfg(target_os = "macos")]
pub use host::{backend_state, devices, run_metal};

#[cfg(not(target_os = "macos"))]
mod stub {
    use std::sync::atomic::AtomicBool;

    use crate::device::{BackendState, DeviceInfo};
    use crate::job::{EnsembleError, EnsembleResult, Job};

    /// Metal is not built into a binary for this platform.
    pub fn backend_state() -> BackendState {
        BackendState::NotBuilt
    }

    /// No devices: the backend is not built.
    pub fn devices() -> Vec<DeviceInfo> {
        Vec::new()
    }

    /// Always an error: the backend is not built.
    pub fn run_metal(
        _device: &DeviceInfo,
        _job: &Job,
        _cancel: &AtomicBool,
    ) -> Result<EnsembleResult, EnsembleError> {
        Err(EnsembleError::Unsupported(
            "the metal backend is not built into this binary".into(),
        ))
    }
}
#[cfg(not(target_os = "macos"))]
pub use stub::{backend_state, devices, run_metal};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_threadgroup_is_whole_execution_widths_up_to_the_group_size() {
        assert_eq!(group_for(1024, 32), 64);
        assert_eq!(group_for(64, 32), 64);
        assert_eq!(group_for(96, 32), 64);
        assert_eq!(group_for(48, 32), 32);
        assert_eq!(group_for(32, 32), 32);
        // A limit below one width keeps the limit.
        assert_eq!(group_for(16, 32), 16);
        // A device that reports nothing useful still gets a group of at least one.
        assert_eq!(group_for(0, 0), 1);
        assert_eq!(group_for(1024, 0), 64);
        // A width that does not divide the group size rounds down to a multiple.
        assert_eq!(group_for(1024, 24), 48);
    }

    #[test]
    fn a_timeout_code_or_an_interactivity_error_is_a_timeout() {
        let t = Duration::from_millis(2100);
        let by_code = command_error(2, "Caused GPU Timeout Error", t);
        assert!(matches!(&by_code, EnsembleError::GpuTimeout(m)
            if m.contains("2.1 s") && m.contains("(code 2)")));
        let by_text = command_error(1, "Impacting Interactivity (0000000e)", t);
        assert!(matches!(by_text, EnsembleError::GpuTimeout(_)));
    }

    #[test]
    fn other_command_errors_are_launch_errors_that_name_the_cause() {
        let t = Duration::from_millis(30);
        let page = command_error(3, "GPU page fault", t);
        assert!(matches!(&page, EnsembleError::GpuLaunch(m)
            if m.contains("outside its buffers") && m.contains("(code 3)")));
        let mem = command_error(8, "Insufficient Memory", t);
        assert!(matches!(&mem, EnsembleError::GpuLaunch(m) if m.contains("ran out of memory")));
        let other = command_error(1, "Internal Error", t);
        assert!(matches!(&other, EnsembleError::GpuLaunch(m)
            if m.contains("command buffer failed") && m.contains("Internal Error")));
        // A slow launch with another code is still not a timeout.
        let slow = command_error(1, "Internal Error", Duration::from_secs(30));
        assert!(matches!(slow, EnsembleError::GpuLaunch(_)));
    }

    #[test]
    fn the_entry_name_is_the_function_the_msl_kernel_defines() {
        use pc_codegen::kernel::{Dialect, Scalar};

        use crate::kernel::{Integrator, kernel_source};
        use crate::tape::Tape;

        use pc_ode::{MapLoader, Severity, SourceMap, load};

        let mut map = SourceMap::new();
        let top = map.add("t.ode", "par a=1\nx'=-a*x\ndone\n");
        let l = load(top, &mut map, &mut MapLoader::new());
        assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
        let model = pc_ir::lower(&l.expanded, &l.info).expect("lower");
        let tape = Tape::build(&model).expect("tape");
        let src = kernel_source(&tape, Integrator::Rk4, Scalar::F32, Dialect::Msl)
            .expect("kernel source");
        assert!(src.contains(&format!("kernel void {ENTRY}(")), "{src}");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn without_macos_the_backend_is_not_built() {
        use crate::device::BackendState;
        assert_eq!(backend_state(), BackendState::NotBuilt);
        assert!(devices().is_empty());
    }
}
