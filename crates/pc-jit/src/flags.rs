//! Cranelift settings and the host ISA.

use crate::{JitError, cl};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};

/// The host ISA with the flags from spike S2. Nothing here enables a fused multiply-add.
pub(crate) fn host_isa() -> Result<OwnedTargetIsa, JitError> {
    let mut fb = settings::builder();
    let mut flags = vec![
        ("opt_level", "speed"),
        ("use_colocated_libcalls", "false"),
        // Windows grows the stack through a guard page, so frames over 4 KiB need
        // probes. Nothing provides __cranelift_probestack, so probe inline.
        ("enable_probestack", "true"),
        ("probestack_strategy", "inline"),
    ];
    // is_pic is an x86_64 setting here; other hosts keep Cranelift's default.
    if cfg!(target_arch = "x86_64") {
        flags.push(("is_pic", "true"));
    }
    for (k, v) in flags {
        cl(k, fb.set(k, v))?;
    }
    let isa = cl("host ISA", cranelift_native::builder())?;
    cl("ISA flags", isa.finish(settings::Flags::new(fb)))
}
