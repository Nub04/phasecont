//! Shared by every test that compiles emitted C and loads the library: which compilers
//! count. Test crates other than pc-codegen include this file with `#[path]`.

use std::process::Command;

/// The CPU field of a target triple, spelled as `std::env::consts::ARCH` spells it. Apple
/// compilers print `arm64` where Rust writes `aarch64`.
pub fn triple_arch(triple: &str) -> &str {
    match triple.trim().split('-').next().unwrap_or("") {
        "arm64" => "aarch64",
        cpu => cpu,
    }
}

/// True when `cc` runs and builds libraries this test process can load. Clang does not
/// count on Windows, where it targets MSVC and rejects `-fPIC`. A compiler for another CPU
/// does not count either: the windows-11-arm runner has an x86_64 MinGW gcc, and an aarch64
/// process cannot load its DLLs (LoadLibraryExW error 193). That case prints a note.
pub fn usable(cc: &str) -> bool {
    if cfg!(windows) && cc == "clang" {
        return false;
    }
    let Ok(out) = Command::new(cc).arg("-dumpmachine").output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let triple = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if triple_arch(&triple) == std::env::consts::ARCH {
        return true;
    }
    println!(
        "SKIPPED: {cc} builds for {triple}, and this test process runs on {}",
        std::env::consts::ARCH
    );
    false
}
