//! The CUDA, OpenCL and Metal kernel text, checked without a compiler or a device.
//!
//! These tests read the text. Whether NVRTC accepts the CUDA text is checked in
//! `cuda_compile.rs`, and whether clang accepts the OpenCL text in `opencl_compile.rs`.

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
fn the_opencl_entry_is_one_guarded_work_item_per_member() {
    for (i, s) in BOTH {
        let text = source(VDP, i, s, Dialect::OpenCl);
        assert_eq!(text.matches("__kernel").count(), 1, "{i:?} {s:?}");
        assert!(text.contains("__kernel void pc_ens("), "{i:?} {s:?}");
        assert!(text.contains("get_global_id(0)"), "{i:?} {s:?}");
        let guard = text.find("if (m >= members)").expect("guard");
        let call = text.rfind("pc_member(rc").expect("call");
        assert!(guard < call, "the guard comes before the member call");
        assert!(!text.contains("for (m = 0u"), "{i:?} {s:?}");
    }
}

#[test]
fn the_opencl_text_rounds_every_operation() {
    for (i, s) in BOTH {
        let text = source(VDP, i, s, Dialect::OpenCl);
        assert!(
            text.contains("#pragma OPENCL FP_CONTRACT OFF"),
            "{i:?} {s:?}"
        );
        assert!(!text.contains("fast_relaxed"), "{i:?} {s:?}");
    }
}

#[test]
fn the_opencl_text_asks_for_fp64_only_in_double() {
    let single = source(VDP, Integrator::Rk4, Scalar::F32, Dialect::OpenCl);
    let double = source(VDP, Integrator::Rk4, Scalar::F64, Dialect::OpenCl);
    assert!(!single.contains("cl_khr_fp64"));
    assert!(double.contains("cl_khr_fp64"));
    assert!(single.contains("typedef float pc_real;"));
    assert!(double.contains("typedef double pc_real;"));
}

#[test]
fn the_opencl_text_has_no_host_only_constructs() {
    for (i, s) in BOTH {
        let text = source(VDP, i, s, Dialect::OpenCl);
        for bad in ["#include", "pc_shim_init", "static inline", "pc_tbl"] {
            assert!(!text.contains(bad), "{i:?} {s:?}: {bad}");
        }
    }
}

#[test]
fn the_opencl_dp5_tables_live_in_constant_memory() {
    let text = source(VDP, Integrator::Dp5, Scalar::F64, Dialect::OpenCl);
    assert!(text.contains("#define PC_CONSTANT __constant"));
    for table in ["pc_cn", "pc_am", "pc_em"] {
        assert!(
            text.contains(&format!("PC_CONSTANT pc_real {table}[")),
            "{table}"
        );
    }
}

const SINGLE: [Integrator; 2] = [Integrator::Rk4, Integrator::Dp5];

#[test]
fn the_msl_entry_is_one_guarded_thread_per_member() {
    for i in SINGLE {
        let text = source(VDP, i, Scalar::F32, Dialect::Msl);
        assert_eq!(text.matches("kernel void").count(), 1, "{i:?}");
        assert!(text.contains("kernel void pc_ens("), "{i:?}");
        assert!(text.contains("uint gid [[thread_position_in_grid]]"));
        let guard = text.find("if (m >= members)").expect("guard");
        let call = text.rfind("pc_member(rc").expect("call");
        assert!(guard < call, "the guard comes before the member call");
        assert!(!text.contains("for (m = 0u"), "{i:?}");
    }
}

#[test]
fn the_msl_buffers_sit_at_the_indices_the_host_binds() {
    let text = source(VDP, Integrator::Rk4, Scalar::F32, Dialect::Msl);
    for (k, name) in [
        "rc",
        "segm",
        "x0",
        "par",
        "work",
        "out",
        "ctl",
        "members",
        "steps",
        "max_steps",
        "ns",
        "pstride",
    ]
    .into_iter()
    .enumerate()
    {
        let want = format!("{name} [[buffer({k})]]");
        assert!(text.contains(&want), "missing {want}");
    }
    assert_eq!(text.matches("[[buffer(").count(), 12);
}

#[test]
fn the_msl_text_turns_off_fast_math_and_contraction() {
    for i in SINGLE {
        let text = source(DECAY, i, Scalar::F32, Dialect::Msl);
        let start = text.find("kernel void pc_ens(").expect("entry");
        for pragma in [
            "#pragma METAL fp math_mode(safe)",
            "#pragma METAL fp contract(off)",
        ] {
            let at = text
                .find(pragma)
                .unwrap_or_else(|| panic!("missing {pragma}"));
            assert!(at < start, "{pragma} comes before the code it governs");
        }
    }
}

#[test]
fn the_msl_text_uses_the_precise_math_functions() {
    let text = source(MIXED, Integrator::Dp5, Scalar::F32, Dialect::Msl);
    for name in ["sin", "cos", "exp", "tanh", "pow"] {
        assert!(
            text.contains(&format!("#define pc_{name} metal::precise::{name}")),
            "{name}"
        );
    }
    assert!(text.contains("#define PC_CONSTANT constant"));
    for table in ["pc_cn", "pc_am", "pc_em"] {
        assert!(
            text.contains(&format!("PC_CONSTANT pc_real {table}[")),
            "{table}"
        );
    }
}

#[test]
fn msl_has_no_double_whatever_the_integrator() {
    let tape = Tape::build(&build_model(DECAY)).expect("tape");
    for i in SINGLE {
        let e = kernel_source(&tape, i, Scalar::F64, Dialect::Msl).unwrap_err();
        assert_eq!(e, KernelError::NoDouble, "{i:?}");
    }
}
