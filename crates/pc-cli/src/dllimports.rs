//! `pc-cli dll-imports`: list the DLLs of a 64-bit PE file and check them
//! against an allow-list.

use std::path::Path;
use std::process::ExitCode;

use crate::pe;

/// DLL names a shipped `pckern.dll` may import, compared without regard to
/// case. This is what Rust std on `x86_64-pc-windows-gnu` imports today, plus
/// `bcrypt.dll` and `advapi32.dll`, which other std releases use, and
/// `dbghelp.dll`, a Windows system DLL that `dlopen2` (under `opencl3`)
/// imports for its symbol lookup. Names that start with `api-ms-win-` are
/// allowed too.
const ALLOWED: &[&str] = &[
    "kernel32.dll",
    "ntdll.dll",
    "userenv.dll",
    "ws2_32.dll",
    "bcryptprimitives.dll",
    "bcrypt.dll",
    "advapi32.dll",
    "msvcrt.dll",
    "dbghelp.dll",
];

/// Prefix of the Windows API set DLL names.
const API_SET_PREFIX: &str = "api-ms-win-";

/// Whether `name` is on the allow-list.
pub fn is_allowed(name: &str) -> bool {
    ALLOWED.iter().any(|ok| name.eq_ignore_ascii_case(ok))
        || name
            .get(..API_SET_PREFIX.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(API_SET_PREFIX))
}

/// Prints the imports of `file`, one DLL per line, and fails when one is not
/// on the allow-list.
pub fn dll_imports(file: &Path) -> ExitCode {
    let shown = file.display();
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("pc-cli: cannot read {shown}: {e}");
            return ExitCode::from(1);
        }
    };
    let found = match pe::imports(&bytes) {
        Ok(found) => found,
        Err(e) => {
            eprintln!("pc-cli: {shown}: {e}");
            return ExitCode::from(1);
        }
    };
    let mut text = String::new();
    for name in &found.normal {
        text.push_str(name);
        text.push('\n');
    }
    for name in &found.delayed {
        text.push_str(name);
        text.push_str(" (delay)\n");
    }
    let mut outside: Vec<&str> = Vec::new();
    for name in found.normal.iter().chain(&found.delayed) {
        if !is_allowed(name) && !outside.iter().any(|s| s.eq_ignore_ascii_case(name)) {
            outside.push(name);
        }
    }
    if outside.is_empty() {
        return crate::out(&text, 0);
    }
    eprintln!(
        "pc-cli: {shown} imports DLLs outside the allow-list: {}",
        outside.join(", ")
    );
    crate::out(&text, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_list_ignores_case_and_accepts_api_sets() {
        for ok in [
            "kernel32.dll",
            "KERNEL32.dll",
            "NTDLL.DLL",
            "msvcrt.dll",
            "bcrypt.dll",
            "ADVAPI32.dll",
            "DbgHelp.dll",
            "api-ms-win-core-synch-l1-2-0.dll",
            "API-MS-WIN-core-x.dll",
        ] {
            assert!(is_allowed(ok), "{ok}");
        }
        for bad in [
            "user32.dll",
            "libgcc_s_seh-1.dll",
            "libwinpthread-1.dll",
            "libstdc++-6.dll",
            "kernel32.dll.evil",
            "api-ms-win",
            "",
        ] {
            assert!(!is_allowed(bad), "{bad}");
        }
    }

    /// The GPU backends load their libraries at run time. A static or delay-load import of one
    /// would stop `pckern.dll` from loading on a machine without that driver.
    #[test]
    fn gpu_libraries_are_never_allowed_imports() {
        for gpu in [
            "nvcuda.dll",
            "NVCUDA.dll",
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc-builtins64_130.dll",
            "OpenCL.dll",
            "opencl.dll",
        ] {
            assert!(!is_allowed(gpu), "{gpu}");
        }
    }
}
