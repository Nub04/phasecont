//! The OpenCL kernel through clang's OpenCL front end, with no OpenCL runtime and no device.
//!
//! `clang -x cl -fsyntax-only` parses and type-checks the kernel text for OpenCL C 1.2. It
//! does not run the driver compiler, so it cannot show that a given driver accepts the text,
//! but it catches a wrong type, a missing function and a bad address space on any machine
//! that has clang. A machine without clang skips these tests with a `SKIPPED` line, and
//! `PC_REQUIRE_GPU=clang-opencl` turns that skip into a failure. The hosted CI job that
//! installs clang sets it.

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::*;
use pc_codegen::kernel::{Dialect, Scalar};
use pc_ensemble::Tape;
use pc_ensemble::kernel::{Integrator, kernel_source};

const VARIANTS: [(Integrator, Scalar); 4] = [
    (Integrator::Rk4, Scalar::F32),
    (Integrator::Rk4, Scalar::F64),
    (Integrator::Dp5, Scalar::F32),
    (Integrator::Dp5, Scalar::F64),
];

/// The clang to use: `PC_CLANG` if set, else `clang` on the path.
fn clang() -> String {
    std::env::var("PC_CLANG").unwrap_or_else(|_| "clang".into())
}

/// Checks `src` with clang. `Ok(None)` means clang is missing.
fn check(src: &str, name: &str) -> Result<Option<()>, String> {
    let dir: PathBuf = std::env::temp_dir().join(format!("pc-opencl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(format!("{name}.cl"));
    std::fs::write(&file, src).map_err(|e| e.to_string())?;
    let run = Command::new(clang())
        .args([
            "-x",
            "cl",
            "-cl-std=CL1.2",
            "-Xclang",
            "-finclude-default-header",
            "-fsyntax-only",
            "-Wno-unknown-pragmas",
        ])
        .arg(&file)
        .output();
    let _ = std::fs::remove_file(&file);
    match run {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
        Ok(out) if out.status.success() => Ok(Some(())),
        Ok(out) => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
    }
}

fn kernel(model: &str, i: Integrator, s: Scalar) -> String {
    let tape = Tape::build(&build_model(model)).expect("tape");
    kernel_source(&tape, i, s, Dialect::OpenCl).expect("kernel source")
}

#[test]
fn every_model_and_variant_parses_as_opencl_c() {
    for (n, model) in [DECAY, VDP, OSC, MIXED].into_iter().enumerate() {
        for (i, s) in VARIANTS {
            let src = kernel(model, i, s);
            match check(&src, &format!("m{n}-{i:?}-{s:?}")) {
                Ok(Some(())) => {}
                Ok(None) => {
                    skip_gpu("clang-opencl", "clang is not installed (set PC_CLANG)");
                    return;
                }
                Err(log) => {
                    panic!("model {n} {i:?} {s:?} does not parse:\n{log}\n--- source ---\n{src}")
                }
            }
        }
    }
}

#[test]
fn a_broken_kernel_is_rejected_so_the_check_means_something() {
    let mut src = kernel(VDP, Integrator::Rk4, Scalar::F32);
    src.push_str("\n__kernel void broken(__global float *p) { p[0] = no_such_function(1.0f); }\n");
    match check(&src, "broken") {
        Ok(Some(())) => panic!("clang accepted a call to an undeclared function"),
        Ok(None) => skip_gpu("clang-opencl", "clang is not installed (set PC_CLANG)"),
        Err(log) => assert!(log.contains("no_such_function"), "{log}"),
    }
}
