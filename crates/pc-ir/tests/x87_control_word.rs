//! `pc_ir::hostmath` gives the same bits whatever the x87 precision control is.
//!
//! A MinGW-w64 exe starts its main thread at 64-bit x87 precision (control word 0x037F)
//! and other threads at 53-bit (0x027F). This test switches the control word inside one
//! thread, so it does not depend on which thread libtest runs it on.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::disallowed_methods)]

use std::arch::asm;

fn control_word() -> u16 {
    let mut cw: u16 = 0;
    // SAFETY: fnstcw stores two bytes to a valid, aligned u16 on this stack frame.
    unsafe { asm!("fnstcw [{0}]", in(reg) &mut cw, options(nostack)) };
    cw
}

fn set_control_word(cw: u16) {
    // SAFETY: fldcw loads two bytes from a valid u16; it changes only x87 rounding state,
    // which the guard below restores.
    unsafe { asm!("fldcw [{0}]", in(reg) &cw, options(nostack, readonly)) };
}

/// Restores the control word on drop, also when an assertion fails.
struct Restore(u16);
impl Drop for Restore {
    fn drop(&mut self) {
        set_control_word(self.0);
    }
}

const PC53: u16 = 0x027F;
const PC64: u16 = 0x037F;
const N: usize = 20_000;

fn inputs(seed: u64) -> Vec<(f64, f64)> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..N).map(|_| (next(), next())).collect()
}

fn bits_at(cw: u16, f: &dyn Fn(f64, f64) -> f64, xs: &[(f64, f64)]) -> Vec<u64> {
    set_control_word(cw);
    xs.iter()
        .map(|&(a, b)| f(std::hint::black_box(a), std::hint::black_box(b)).to_bits())
        .collect()
}

fn differing(f: &dyn Fn(f64, f64) -> f64, xs: &[(f64, f64)]) -> usize {
    let _restore = Restore(control_word());
    let lo = bits_at(PC53, f, xs);
    let hi = bits_at(PC64, f, xs);
    lo.iter().zip(&hi).filter(|(p, q)| p != q).count()
}

type Case = (&'static str, fn(f64, f64) -> f64);

const HOST: [Case; 9] = [
    ("powf", |x, y| {
        pc_ir::hostmath::powf(x * 10.0, y * 6.0 - 3.0)
    }),
    // The exponent comes from the input, so the compiler cannot turn the call into products.
    ("powi", |x, y| {
        pc_ir::hostmath::powi(x * 10.0, (y * 20.0) as i32 - 10)
    }),
    ("exp", |x, _| pc_ir::hostmath::exp(x * 40.0 - 20.0)),
    ("exp2", |x, _| pc_ir::hostmath::exp2(x * 40.0 - 20.0)),
    ("exp_m1", |x, _| pc_ir::hostmath::exp_m1(x * 4.0 - 2.0)),
    ("ln_1p", |x, _| pc_ir::hostmath::ln_1p(x * 4.0 - 0.5)),
    ("atanh", |x, _| pc_ir::hostmath::atanh(x * 2.0 - 1.0)),
    ("asinh", |x, _| pc_ir::hostmath::asinh(x * 20.0 - 10.0)),
    ("acosh", |x, _| pc_ir::hostmath::acosh(x * 20.0 + 1.0)),
];

#[test]
fn hostmath_bits_do_not_depend_on_x87_precision() {
    let xs = inputs(0x9e37_79b9_7f4a_7c15);
    // Report whether this toolchain's std functions take the x87 path at all, so a green
    // run on a runtime without it (glibc, ucrt) is not read as coverage.
    let std_pow: fn(f64, f64) -> f64 = |x, y| (x * 10.0).powf(y * 6.0 - 3.0);
    let std_powi: fn(f64, f64) -> f64 = |x, y| (x * 10.0).powi((y * 20.0) as i32 - 10);
    let std_exp: fn(f64, f64) -> f64 = |x, _| (x * 40.0 - 20.0).exp();
    eprintln!(
        "x87 path in std: powf {} of {N} differ, powi {} of {N} differ, exp {} of {N} differ",
        differing(&std_pow, &xs),
        differing(&std_powi, &xs),
        differing(&std_exp, &xs)
    );
    let bad: Vec<String> = HOST
        .iter()
        .map(|(name, f)| (name, differing(f, &xs)))
        .filter(|&(_, d)| d > 0)
        .map(|(name, d)| format!("{name}: {d} of {N}"))
        .collect();
    assert!(bad.is_empty(), "bits depend on x87 precision: {bad:?}");
}

#[test]
fn the_guard_restores_the_control_word() {
    let before = control_word();
    let _ = differing(&|x, _| pc_ir::hostmath::exp(x), &inputs(7));
    assert_eq!(control_word(), before);
}
