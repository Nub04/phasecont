//! The Metal kernel through clang's C++ front end, with a stand-in for the Metal headers.
//!
//! MSL is C++14 or C++17 with address spaces and attributes. The shim below defines
//! `metal_stdlib` so that `clang++ -fsyntax-only` parses the kernel text: `device` vanishes,
//! `constant` becomes `const`, and `metal::precise` forwards to `<cmath>`. That catches a wrong
//! type, a missing function, a bad pointer conversion or a typo in the template. It does not
//! check what only Metal's compiler knows (address-space rules, the `[[buffer]]` attributes,
//! the pragmas), so it shows nothing about whether a Mac accepts the text. A machine without
//! clang++ skips these tests with a `SKIPPED` line, and `PC_REQUIRE_GPU=clang-msl` turns that
//! skip into a failure. The hosted macOS job compiles the text with Metal itself: it sets
//! `PC_DUMP_MSL` to a folder, this file writes each kernel there as a `.metal` file, and the
//! job runs Apple's `metal` tool on them.

mod common;

use std::path::PathBuf;
use std::process::Command;

use common::*;
use pc_codegen::kernel::{Dialect, Scalar};
use pc_ensemble::Tape;
use pc_ensemble::kernel::{Integrator, kernel_source};

const VARIANTS: [Integrator; 2] = [Integrator::Rk4, Integrator::Dp5];

/// What `<metal_stdlib>` provides to this kernel, in C++.
const SHIM: &str = r#"
#include <cmath>
#include <math.h>
typedef unsigned int uint;
typedef unsigned long ulong;
namespace metal {
namespace precise {
using std::sin; using std::cos; using std::tan;
using std::asin; using std::acos; using std::atan;
using std::sinh; using std::cosh; using std::tanh;
using std::exp; using std::log; using std::log10;
using std::atan2; using std::pow; using std::fmod; using std::sqrt;
}
using std::fabs; using std::floor; using std::ceil; using std::isfinite;
}
#define device
#define constant const
#define kernel
"#;

/// The clang++ to use: `PC_CLANGXX` if set, else `clang++` on the path.
fn clangxx() -> String {
    std::env::var("PC_CLANGXX").unwrap_or_else(|_| "clang++".into())
}

/// Checks `src` with clang++. `Ok(None)` means clang++ is missing.
fn check(src: &str, name: &str) -> Result<Option<()>, String> {
    let dir: PathBuf = std::env::temp_dir().join(format!("pc-msl-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("metal_stdlib"), SHIM).map_err(|e| e.to_string())?;
    let file = dir.join("kernel.cpp");
    std::fs::write(&file, src).map_err(|e| e.to_string())?;
    let run = Command::new(clangxx())
        .args([
            "-std=c++17",
            "-fsyntax-only",
            "-Wno-unknown-pragmas",
            "-Wno-unknown-attributes",
        ])
        .arg("-I")
        .arg(&dir)
        .arg(&file)
        .output();
    let _ = std::fs::remove_dir_all(&dir);
    match run {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
        Ok(out) if out.status.success() => Ok(Some(())),
        Ok(out) => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
    }
}

/// Writes `src` to `$PC_DUMP_MSL/<name>.metal` when that variable is set.
fn dump(src: &str, name: &str) {
    if let Some(dir) = std::env::var_os("PC_DUMP_MSL") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create the dump folder");
        std::fs::write(dir.join(format!("{name}.metal")), src).expect("write the kernel");
    }
}

fn kernel(model: &str, i: Integrator) -> String {
    let tape = Tape::build(&build_model(model)).expect("tape");
    kernel_source(&tape, i, Scalar::F32, Dialect::Msl).expect("kernel source")
}

#[test]
fn every_model_and_integrator_parses_as_cpp_with_the_shim() {
    for (n, model) in [DECAY, VDP, OSC, MIXED].into_iter().enumerate() {
        for i in VARIANTS {
            let src = kernel(model, i);
            dump(&src, &format!("m{n}-{i:?}"));
            match check(&src, &format!("m{n}-{i:?}")) {
                Ok(Some(())) => {}
                Ok(None) => {
                    skip_gpu("clang-msl", "clang++ is not installed (set PC_CLANGXX)");
                    return;
                }
                Err(log) => {
                    panic!("model {n} {i:?} does not parse:\n{log}\n--- source ---\n{src}")
                }
            }
        }
    }
}

#[test]
fn a_broken_kernel_is_rejected_so_the_check_means_something() {
    let mut src = kernel(VDP, Integrator::Rk4);
    src.push_str("\nkernel void broken(device float *p) { p[0] = no_such_function(1.0f); }\n");
    match check(&src, "broken") {
        Ok(Some(())) => panic!("clang++ accepted a call to an undeclared function"),
        Ok(None) => skip_gpu("clang-msl", "clang++ is not installed (set PC_CLANGXX)"),
        Err(log) => assert!(log.contains("no_such_function"), "{log}"),
    }
}
