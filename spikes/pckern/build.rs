// Stamps the target, profile and build time into the library, so the MEX
// gateway can print which pckern build it loaded and a stale DLL shows up.
// Host builds also compile spikes/mex/fhn_aot.c, the C reference for spike S2.
// Windows builds get that code from the MEX gateway instead, built by MSVC.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // Any change under src/ reruns this script, so the stamp follows the code.
    for path in ["build.rs", "Cargo.toml", "src", "../mex/fhn_aot.c"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rustc-check-cfg=cfg(fhn_c)");
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let target = std::env::var("TARGET").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    println!("cargo:rustc-env=PCKERN_BUILD_STAMP={secs}");
    println!("cargo:rustc-env=PCKERN_TARGET={target}");
    println!("cargo:rustc-env=PCKERN_PROFILE={profile}");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        compile_c();
    }
}

fn compile_c() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    // -O2 matches the optimization mex passes to MSVC. No contraction, so
    // the C code rounds like the JIT and Rust code.
    let flags = ["-O2", "-fPIC", "-ffp-contract=off"];
    let obj = out.join("fhn_aot.o");
    run(Command::new(&cc)
        .args(flags)
        .args(["-c", "../mex/fhn_aot.c", "-o"])
        .arg(&obj));
    run(Command::new("ar")
        .arg("crs")
        .arg(out.join("libfhn_aot.a"))
        .arg(&obj));
    let version = Command::new(&cc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(str::to_owned))
        .unwrap_or_else(|| cc.clone());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fhn_aot");
    println!("cargo:rustc-cfg=fhn_c");
    println!("cargo:rustc-env=PCKERN_CC={version}, {}", flags.join(" "));
}

fn run(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("cannot run {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed with {status}");
}
