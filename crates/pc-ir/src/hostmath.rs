//! Engine math whose bits must not depend on the calling thread.
//!
//! On x86_64-pc-windows-gnu, Rust links MinGW-w64's msvcrt runtime, whose `pow`, `exp`,
//! `exp2`, `expm1`, `log1p`, `atanh`, `asinh` and `acosh` run on the x87 unit. A `powi`
//! whose exponent is not a constant can reach the x87 `pow` too: the compiler turned the one
//! in `cycle::mesh` into `pow(x, n as f64)` on this target. The x87 precision-control field
//! rounds their intermediate results, and a MinGW exe starts its main thread at 64-bit
//! precision while every other thread starts at Windows' 53-bit default. The same call then
//! gives different bits on different threads. These wrappers call `libm` on that target,
//! which uses SSE2 only, and the platform functions everywhere else, so Linux and macOS keep
//! their bits and speed.
//!
//! Engine code calls these instead of the `f64` methods. `clippy.toml` bans the methods except
//! `powi`, whose calls with a constant exponent become multiplications and are fine; a scan of
//! the built binaries for calls to the x87 functions finds the rest.

// The platform functions off windows-gnu, where they are fast and do not use the x87 unit.
#[cfg(not(all(target_os = "windows", target_env = "gnu")))]
#[allow(clippy::disallowed_methods)]
mod imp {
    pub fn powf(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
    pub fn powi(x: f64, n: i32) -> f64 {
        x.powi(n)
    }
    pub fn exp(x: f64) -> f64 {
        x.exp()
    }
    pub fn exp2(x: f64) -> f64 {
        x.exp2()
    }
    pub fn exp_m1(x: f64) -> f64 {
        x.exp_m1()
    }
    pub fn ln_1p(x: f64) -> f64 {
        x.ln_1p()
    }
    pub fn atanh(x: f64) -> f64 {
        x.atanh()
    }
    pub fn asinh(x: f64) -> f64 {
        x.asinh()
    }
    pub fn acosh(x: f64) -> f64 {
        x.acosh()
    }
}

// `libm` on windows-gnu. It is also faster there: 41 ns against 115 ns per `pow`, and 8 ns
// against 58 ns per `exp` (2026-11-25, one core).
#[cfg(all(target_os = "windows", target_env = "gnu"))]
mod imp {
    pub use libm::{acosh, asinh, atanh, exp, exp2, pow as powf};
    // The compiler turns a `powi` with a variable exponent into `pow(x, n as f64)` here
    // (seen in `cycle::mesh`), so this makes the same call through `libm`.
    pub fn powi(x: f64, n: i32) -> f64 {
        libm::pow(x, f64::from(n))
    }
    pub fn exp_m1(x: f64) -> f64 {
        libm::expm1(x)
    }
    pub fn ln_1p(x: f64) -> f64 {
        libm::log1p(x)
    }
}

pub use imp::{acosh, asinh, atanh, exp, exp_m1, exp2, ln_1p, powf, powi};
