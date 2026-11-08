//! A one-shot fault for the device tests. This module exists only with the `fault-injection`
//! feature, which no default, release or CI build turns on except the kernel cache test.
//!
//! Arming the flag makes the next launch of a GPU backend use a configuration that the driver
//! rejects, so the error comes from the real driver call. That drives the same path as a
//! launch error on a real run: the run returns `gpuLaunch`, and the device entry is removed.

use std::sync::atomic::{AtomicBool, Ordering};

static LAUNCH: AtomicBool = AtomicBool::new(false);

/// Makes the next CUDA or OpenCL launch in this process fail once.
pub fn arm_launch_fault() {
    LAUNCH.store(true, Ordering::SeqCst);
}

/// True once after [`arm_launch_fault`], then false until it is armed again.
pub(crate) fn take_launch_fault() -> bool {
    LAUNCH.swap(false, Ordering::SeqCst)
}
