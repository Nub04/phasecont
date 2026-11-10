//! `pc-cli dll-imports`: list the DLLs of a 64-bit PE file and check them
//! against an allow-list.

use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use crate::pe;

/// DLL names a shipped `pckern.dll` may import, compared without regard to
/// case. This is what Rust std on `x86_64-pc-windows-gnu` imports today, plus
/// `bcrypt.dll` and `advapi32.dll`, which other std releases use, and
/// `dbghelp.dll`, a Windows system DLL that `dlopen2` (under `opencl3`)
/// imports for its symbol lookup. API set names of the form
/// `api-ms-win-<words>-l<n>-<n>-<n>.dll` are allowed too.
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

/// Largest file the command reads. `pckern.dll` is about 20 MB.
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Whether `name` has the shape of a Windows API set DLL name, such as
/// `api-ms-win-core-synch-l1-2-0.dll`: the prefix, one or more words of ASCII
/// letters and digits, then `l<n>`, two more numbers, and `.dll`, all joined by
/// hyphens. Case is ignored.
fn is_api_set(name: &str) -> bool {
    let Some(rest) = name.get(API_SET_PREFIX.len()..) else {
        return false;
    };
    if !name[..API_SET_PREFIX.len()].eq_ignore_ascii_case(API_SET_PREFIX) {
        return false;
    }
    let Some(stem) = rest
        .len()
        .checked_sub(".dll".len())
        .and_then(|cut| rest.get(..cut).zip(rest.get(cut..)))
        .filter(|(_, ext)| ext.eq_ignore_ascii_case(".dll"))
        .map(|(stem, _)| stem)
    else {
        return false;
    };
    let parts: Vec<&str> = stem.split('-').collect();
    let [words @ .., level, minor, patch] = parts.as_slice() else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let is_level =
        level.get(..1).is_some_and(|l| l.eq_ignore_ascii_case("l")) && digits(&level[1..]);
    is_level
        && digits(minor)
        && digits(patch)
        && !words.is_empty()
        && words
            .iter()
            .all(|w| !w.is_empty() && w.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Whether `name` is on the allow-list.
pub fn is_allowed(name: &str) -> bool {
    ALLOWED.iter().any(|ok| name.eq_ignore_ascii_case(ok)) || is_api_set(name)
}

/// Reads `file`, refusing one over [`MAX_FILE_BYTES`]. The length is checked
/// before the read and the read itself is capped, so a file that grows in
/// between is refused too.
fn read_limited(file: &Path) -> Result<Vec<u8>, String> {
    let too_big = || {
        format!(
            "{}: file is larger than {} MiB",
            file.display(),
            MAX_FILE_BYTES / (1024 * 1024)
        )
    };
    let read_err = |e: std::io::Error| format!("cannot read {}: {e}", file.display());
    let f = std::fs::File::open(file).map_err(read_err)?;
    if f.metadata().map_err(read_err)?.len() > MAX_FILE_BYTES {
        return Err(too_big());
    }
    let mut bytes = Vec::new();
    f.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(read_err)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(too_big());
    }
    Ok(bytes)
}

/// Prints the imports of `file`, one DLL per line, and fails when one is not
/// on the allow-list.
pub fn dll_imports(file: &Path) -> ExitCode {
    let shown = file.display();
    let bytes = match read_limited(file) {
        Ok(b) => b,
        Err(msg) => {
            eprintln!("pc-cli: {msg}");
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
            "API-MS-WIN-Core-Synch-L1-2-0.DLL",
            "api-ms-win-crt-runtime-l1-1-0.dll",
            "api-ms-win-core-l1-1-0.dll",
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
            "api-ms-win-.dll",
            "api-ms-win-core.dll",
            "API-MS-WIN-core-x.dll",
            "api-ms-win-core-synch-l1-2.dll",
            "api-ms-win-core-synch-l1-2-0",
            "api-ms-win-core-synch-l1-2-0.exe",
            "api-ms-win-core-synch-l1-2-0.dll.evil",
            "api-ms-win-core-synch-x1-2-0.dll",
            "api-ms-win-core-synch-l-2-0.dll",
            "api-ms-win-core-synch-l1-a-0.dll",
            "api-ms-win-core--synch-l1-2-0.dll",
            "api-ms-win-core\\evil-l1-2-0.dll",
            "api-ms-win-core/evil-l1-2-0.dll",
            "api-ms-win-core evil-l1-2-0.dll",
            "api-ms-win-core-sync\u{e9}-l1-2-0.dll",
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
