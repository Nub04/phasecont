//! The CUDA kernel through NVRTC, with no driver and no device.
//!
//! NVRTC comes from MATLAB, a CUDA toolkit or `CUDA_PATH`. A machine without it skips these
//! tests with a `SKIPPED` line, and `PC_REQUIRE_GPU=nvrtc` turns that skip into a failure.
//! The hosted CI job that installs NVRTC sets it.

mod common;

use common::*;
use pc_codegen::kernel::Scalar;
use pc_ensemble::cuda::{compile_ptx_text, compile_ptx_text_in};
use pc_ensemble::kernel::{Integrator, Layout};
use pc_ensemble::{EnsembleError, Tape};

const VARIANTS: [(Integrator, Scalar); 4] = [
    (Integrator::Rk4, Scalar::F32),
    (Integrator::Rk4, Scalar::F64),
    (Integrator::Dp5, Scalar::F32),
    (Integrator::Dp5, Scalar::F64),
];

/// The oldest and the newest architecture the kernel is checked for in every case.
const EDGE_ARCHS: [&str; 2] = ["compute_75", "compute_120"];

/// Compiles, or ends the test with a loud skip when NVRTC is missing.
fn ptx(src: &str, i: Integrator, s: Scalar, arch: &str) -> Option<String> {
    let tape = Tape::build(&build_model(src)).expect("tape");
    match compile_ptx_text(&tape, i, s, arch) {
        Ok(text) => Some(text),
        Err(EnsembleError::GpuUnavailable(why)) => {
            skip_gpu("nvrtc", &why);
            None
        }
        Err(e) => panic!("{arch} {i:?} {s:?}: {e}"),
    }
}

#[test]
fn every_model_and_variant_compiles_for_the_edge_architectures() {
    for src in [DECAY, VDP, OSC, MIXED] {
        for (i, s) in VARIANTS {
            for arch in EDGE_ARCHS {
                let Some(text) = ptx(src, i, s, arch) else {
                    return;
                };
                assert!(text.contains(".entry pc_ens"), "{arch} {i:?} {s:?}");
            }
        }
    }
}

#[test]
fn the_middle_architectures_compile_too() {
    for (i, s) in VARIANTS {
        for arch in ["compute_80", "compute_90"] {
            let Some(text) = ptx(VDP, i, s, arch) else {
                return;
            };
            assert!(text.contains(".entry pc_ens"), "{arch} {i:?} {s:?}");
        }
    }
}

#[test]
fn the_ptx_targets_the_requested_architecture() {
    let Some(text) = ptx(VDP, Integrator::Rk4, Scalar::F64, "compute_80") else {
        return;
    };
    assert!(
        text.contains(".target sm_80"),
        "{}",
        &text[..text.len().min(400)]
    );
}

#[test]
fn the_double_kernel_uses_double_arithmetic() {
    let Some(double) = ptx(VDP, Integrator::Rk4, Scalar::F64, "compute_80") else {
        return;
    };
    assert!(double.contains(".f64"));
    // `--fmad=false`: the compiler must not fuse a multiply and an add on its own.
    assert!(
        !double.contains("fma.rn"),
        "the compiler fused an operation the source did not ask to fuse"
    );
}

#[test]
fn the_same_source_gives_the_same_ptx() {
    let Some(a) = ptx(MIXED, Integrator::Dp5, Scalar::F32, "compute_80") else {
        return;
    };
    let b = ptx(MIXED, Integrator::Dp5, Scalar::F32, "compute_80").expect("second compile");
    assert_eq!(a, b);
}

#[test]
fn an_unknown_architecture_is_a_compile_error() {
    let tape = Tape::build(&build_model(DECAY)).expect("tape");
    match compile_ptx_text(&tape, Integrator::Rk4, Scalar::F64, "compute_1") {
        Err(EnsembleError::GpuCompile(m)) => assert!(!m.is_empty()),
        Err(EnsembleError::GpuUnavailable(why)) => skip_gpu("nvrtc", &why),
        other => panic!("expected GpuCompile, got {other:?}"),
    }
}

/// Like [`ptx`], for the kernel that keeps the step arrays in global memory.
fn global_ptx(src: &str, i: Integrator, s: Scalar, arch: &str) -> Option<String> {
    let tape = Tape::build(&build_model(src)).expect("tape");
    match compile_ptx_text_in(&tape, i, s, Layout::Global, arch) {
        Ok(text) => Some(text),
        Err(EnsembleError::GpuUnavailable(why)) => {
            skip_gpu("nvrtc", &why);
            None
        }
        Err(e) => panic!("global {arch} {i:?} {s:?}: {e}"),
    }
}

/// The size in bytes of the biggest `__local_depot` the PTX declares: the local memory of
/// one thread.
fn local_bytes(ptx: &str) -> usize {
    ptx.lines()
        .filter(|l| l.contains(".local") && l.contains("__local_depot"))
        .filter_map(|l| {
            let open = l.rfind('[')?;
            let close = l.rfind(']')?;
            l[open + 1..close].parse::<usize>().ok()
        })
        .max()
        .unwrap_or(0)
}

#[test]
fn the_global_kernel_compiles_for_every_variant_and_edge_architecture() {
    for src in [DECAY, VDP, OSC, MIXED] {
        for (i, s) in VARIANTS {
            for arch in EDGE_ARCHS {
                let Some(text) = global_ptx(src, i, s, arch) else {
                    return;
                };
                assert!(text.contains(".entry pc_ens"), "{arch} {i:?} {s:?}");
            }
        }
    }
}

#[test]
fn the_global_kernel_keeps_the_step_arrays_out_of_local_memory() {
    // 300 states of Lorenz-96 in double precision need 24 KB of private arrays under
    // Dormand-Prince. The global layout must not hold them in local memory.
    let src = lorenz96(300);
    let Some(private) = ptx(&src, Integrator::Dp5, Scalar::F64, "compute_80") else {
        return;
    };
    let global = global_ptx(&src, Integrator::Dp5, Scalar::F64, "compute_80").expect("global");
    assert!(
        local_bytes(&private) >= 24_000,
        "the private text should hold its arrays in local memory: {} bytes",
        local_bytes(&private)
    );
    assert!(
        local_bytes(&global) < 8 * 1024,
        "the global text still has {} bytes of local memory",
        local_bytes(&global)
    );
}

#[test]
fn the_global_ptx_targets_the_requested_architecture_too() {
    let Some(text) = global_ptx(VDP, Integrator::Rk4, Scalar::F64, "compute_80") else {
        return;
    };
    assert!(text.contains(".target sm_80"));
}
