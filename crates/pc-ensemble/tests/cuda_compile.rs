//! The CUDA kernel through NVRTC, with no driver and no device.
//!
//! NVRTC comes from MATLAB, a CUDA toolkit or `CUDA_PATH`. A machine without it skips these
//! tests with a `SKIPPED` line, and `PC_REQUIRE_GPU=nvrtc` turns that skip into a failure.
//! The hosted CI job that installs NVRTC sets it.

mod common;

use common::*;
use pc_codegen::kernel::Scalar;
use pc_ensemble::cuda::compile_ptx_text;
use pc_ensemble::kernel::Integrator;
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
