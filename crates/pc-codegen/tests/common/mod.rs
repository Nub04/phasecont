//! Shared by the AOT integration tests: builds a model, compiles emitted C with a
//! system compiler, loads the library and calls its entries.

#![allow(dead_code)]

#[path = "lower_text.rs"]
mod lower_text;

use libloading::{Library, Symbol};
#[allow(unused_imports)]
pub use lower_text::build_model;
use pc_ir::math::{MATH, MathTable};
use std::path::PathBuf;
use std::process::Command;

pub const FLAGS: [&str; 7] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-std=c99",
    "-Wall",
    "-Wextra",
];

pub fn cc_version(cc: &str) -> Option<String> {
    let o = Command::new(cc).arg("--version").output().ok()?;
    if !o.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&o.stdout);
    Some(s.lines().next()?.trim().to_string())
}

pub fn gcc_version() -> Option<String> {
    cc_version("gcc")
}

/// Compilers the AOT text is checked with: gcc, and clang when installed.
pub fn compilers() -> Vec<&'static str> {
    ["gcc", "clang"]
        .into_iter()
        .filter(|cc| cc_version(cc).is_some())
        .collect()
}

/// Writes references/aot/compiler.txt (first line of `gcc --version`, then the flags) when
/// the content changed. A read-only checkout only gets a note.
pub fn record_compiler(version: &str) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/aot");
    let path = dir.join("compiler.txt");
    let text = format!("{version}\nflags: gcc {} -Werror\n", FLAGS.join(" "));
    if std::fs::read_to_string(&path).is_ok_and(|old| old == text) {
        return;
    }
    let r = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, text));
    if let Err(e) = r {
        println!("note: could not write {}: {e}", path.display());
    }
}

pub struct Temp(PathBuf);

impl Temp {
    pub fn new(tag: &str) -> Temp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("pc-codegen-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        Temp(dir)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A compiled and loaded AOT library. The temp dir holding the files lives as long as the
/// library.
pub struct Aot {
    pub lib: Library,
    pub _tmp: Temp,
}

/// Compiles `c` with `cc` and loads it. `Err` carries the compiler output.
pub fn try_compile_c(cc: &str, tag: &str, c: &str) -> Result<Aot, String> {
    let tmp = Temp::new(tag);
    let src = tmp.0.join("model.c");
    let so = tmp.0.join("model.so");
    std::fs::write(&src, c).map_err(|e| format!("write c: {e}"))?;
    let out = Command::new(cc)
        .args(FLAGS)
        .arg("-Werror")
        .arg("-o")
        .arg(&so)
        .arg(&src)
        .output()
        .map_err(|e| format!("run {cc}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cc} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    // SAFETY: loads the library this test just built; its initializers are empty.
    let lib = unsafe { Library::new(&so) }.map_err(|e| format!("load so: {e}"))?;
    // SAFETY: pc_aot_init takes the table pointer and stores it.
    unsafe {
        let init: Symbol<unsafe extern "C" fn(*const MathTable)> = lib
            .get(b"pc_aot_init")
            .map_err(|e| format!("pc_aot_init: {e}"))?;
        init(&MATH);
    }
    Ok(Aot { lib, _tmp: tmp })
}

/// Like [`try_compile_c`], but panics with the compiler output and the C text on failure.
pub fn compile_c(cc: &str, tag: &str, c: &str) -> Aot {
    match try_compile_c(cc, tag, c) {
        Ok(a) => a,
        Err(e) => panic!("{e}\n{c}"),
    }
}

impl Aot {
    pub fn call(&self, name: &str, x: &[f64], p: &[f64], v: Option<&[f64]>, out: &mut [f64]) {
        let sym = name.as_bytes();
        // SAFETY: the emitted signatures are (x, p, out) and (x, p, v, out); the caller
        // sizes every slice from the entry contract.
        unsafe {
            match v {
                None => {
                    let f: Symbol<unsafe extern "C" fn(*const f64, *const f64, *mut f64)> =
                        self.lib.get(sym).expect("symbol");
                    f(x.as_ptr(), p.as_ptr(), out.as_mut_ptr());
                }
                Some(v) => {
                    let f: Symbol<
                        unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64),
                    > = self.lib.get(sym).expect("symbol");
                    f(x.as_ptr(), p.as_ptr(), v.as_ptr(), out.as_mut_ptr());
                }
            }
        }
    }
}

/// Distance in units in the last place; `u64::MAX` when exactly one value is NaN.
pub fn ulp_dist(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f64| {
        let u = x.to_bits();
        if u >> 63 == 1 { !u } else { u | (1 << 63) }
    };
    key(a).abs_diff(key(b))
}
