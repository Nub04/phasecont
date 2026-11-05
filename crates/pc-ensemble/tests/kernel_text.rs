//! The CUDA kernel text, checked without a compiler or a device.
//!
//! These tests read the text. Whether NVRTC accepts it is checked in `cuda_compile.rs`.

mod common;

use common::*;
use pc_codegen::kernel::{Dialect, KernelError, Scalar};
use pc_ensemble::Tape;
use pc_ensemble::kernel::{Integrator, kernel_source};

fn source(src: &str, integrator: Integrator, scalar: Scalar, dialect: Dialect) -> String {
    let tape = Tape::build(&build_model(src)).expect("tape");
    kernel_source(&tape, integrator, scalar, dialect).expect("kernel source")
}

const BOTH: [(Integrator, Scalar); 4] = [
    (Integrator::Rk4, Scalar::F32),
    (Integrator::Rk4, Scalar::F64),
    (Integrator::Dp5, Scalar::F32),
    (Integrator::Dp5, Scalar::F64),
];

#[test]
fn the_cuda_entry_is_one_guarded_thread_per_member() {
    for (i, s) in BOTH {
        let text = source(VDP, i, s, Dialect::Cuda);
        assert_eq!(text.matches("__global__").count(), 1, "{i:?} {s:?}");
        assert!(
            text.contains("extern \"C\" __global__ void pc_ens("),
            "{i:?} {s:?}"
        );
        assert!(text.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        let guard = text.find("if (m >= members)").expect("guard");
        let call = text.rfind("pc_member(rc").expect("call");
        assert!(guard < call, "the guard comes before the member call");
        // The C entry loops over members; the CUDA entry must not.
        assert!(!text.contains("for (m = 0u"), "{i:?} {s:?}");
    }
}

#[test]
fn the_cuda_text_has_no_host_only_constructs() {
    for (i, s) in BOTH {
        let text = source(VDP, i, s, Dialect::Cuda);
        for bad in ["#include", "pc_shim_init", "static inline", "pc_tbl"] {
            assert!(!text.contains(bad), "{i:?} {s:?}: {bad}");
        }
        assert!(text.contains("__device__"), "{i:?} {s:?}");
    }
}

#[test]
fn the_scalar_type_follows_the_precision() {
    let single = source(VDP, Integrator::Rk4, Scalar::F32, Dialect::Cuda);
    let double = source(VDP, Integrator::Rk4, Scalar::F64, Dialect::Cuda);
    assert!(single.contains("typedef float pc_real;"));
    assert!(double.contains("typedef double pc_real;"));
    assert!(!single.contains("typedef double"));
}

#[test]
fn the_model_sizes_are_macros() {
    let text = source(OSC, Integrator::Rk4, Scalar::F64, Dialect::Cuda);
    assert!(text.contains("#define PC_N 2u"));
    assert!(text.contains("#define PC_WL 23u"));
    assert!(text.contains("#define PC_CTL 8u"));
}

#[test]
fn the_dp5_tables_live_in_constant_memory() {
    let text = source(VDP, Integrator::Dp5, Scalar::F64, Dialect::Cuda);
    assert!(text.contains("#define PC_CONSTANT __constant__"));
    for table in ["pc_cn", "pc_am", "pc_em"] {
        assert!(
            text.contains(&format!("PC_CONSTANT pc_real {table}[")),
            "{table}"
        );
    }
}

#[test]
fn the_same_model_gives_the_same_text() {
    for (i, s) in BOTH {
        assert_eq!(
            source(MIXED, i, s, Dialect::Cuda),
            source(MIXED, i, s, Dialect::Cuda)
        );
    }
}

#[test]
fn a_dialect_without_an_entry_point_is_named() {
    let tape = Tape::build(&build_model(DECAY)).expect("tape");
    for d in [Dialect::OpenCl, Dialect::Msl] {
        let e = kernel_source(&tape, Integrator::Rk4, Scalar::F32, d).unwrap_err();
        let KernelError::Unsupported(m) = e else {
            panic!("expected Unsupported, got {e:?}");
        };
        assert!(m.contains(d.name()), "{m}");
    }
}
