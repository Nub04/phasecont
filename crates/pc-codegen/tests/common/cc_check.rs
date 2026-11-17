//! Shared by every test that compiles emitted C and loads the library: which compilers
//! count. Test crates other than pc-codegen include this file with `#[path]`.

use std::process::Command;

/// The CPU field of a target triple, spelled as `std::env::consts::ARCH` spells it. GNU and
/// Apple compilers name some CPUs differently from Rust: Apple prints `arm64` for `aarch64`,
/// FreeBSD `amd64` for `x86_64`, and GNU names the variant (`i686`, `armv7l`, `powerpc64le`,
/// `mipsel`) where Rust names the family.
pub fn triple_arch(triple: &str) -> &str {
    match triple.trim().split('-').next().unwrap_or("") {
        "arm64" | "aarch64_be" => "aarch64",
        "amd64" => "x86_64",
        "i386" | "i486" | "i586" | "i686" => "x86",
        "powerpc64le" => "powerpc64",
        "powerpcle" => "powerpc",
        "mipsel" => "mips",
        "mips64el" => "mips64",
        "sparcv9" => "sparc64",
        cpu if cpu.starts_with("arm") || cpu.starts_with("thumb") => "arm",
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
