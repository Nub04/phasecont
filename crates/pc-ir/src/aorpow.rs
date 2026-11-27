//! `pow` ported from Arm optimized-routines `math/pow.c` (Copyright (c) 2018-2026, Arm
//! Limited) and `math/pow_common.h` (`checkint`, Copyright (c) 2026, Arm Limited), commit
//! 9a0d5027ec6c3c529642f2d81391125935d99321, used under its MIT license (see
//! THIRD_PARTY_NOTICES.md). The port follows the C code line by line, with the tables in
//! `aorpow/tables.rs`.
//!
//! glibc's x86_64 `pow` is this code too. It picks one of two builds when the program starts:
//! one where the compiler fuses 21 multiply-adds, used when the CPU has FMA and AVX2, and one
//! with no fused operations. The two give different bits on some inputs. [`pow_fused`] is the
//! first build, with each fused operation written as a call to `fma`, and [`pow_plain`] is the
//! second.
//! [`pow`] makes the same choice glibc makes. Against Fedora's glibc, the fused build matched
//! it bit for bit on an FMA CPU and the plain build matched it with the FMA path masked off, on
//! every input class of the 2026-11-27 spike (`docs/STATUS.md`).
//!
//! The engine uses this on x86_64-pc-windows-gnu only (`hostmath::powf`), where the MinGW-w64
//! `pow` runs on the x87 unit. Linux and macOS keep the platform `pow`.
//!
//! Rust never fuses `a * b + c` on its own, so only the `fma` calls round once. `fma` uses the
//! `vfmadd` instruction through `_mm_fmadd_sd`, in debug builds too, so the port never calls
//! the C `fma`.

mod tables;

use std::hint::black_box;
use std::ops::ControlFlow;
use tables as t;

const OFF: u64 = 0x3fe6955500000000;
const SIGN_BIAS: u32 = 0x800 << 7;
const ONE: u64 = 0x3ff0000000000000;
const INF: u64 = 0x7ff0000000000000;

#[inline(always)]
fn d(b: u64) -> f64 {
    f64::from_bits(b)
}

#[inline(always)]
fn top12(x: f64) -> u32 {
    (x.to_bits() >> 52) as u32
}

/// True for 2|x| = 0, inf or nan, from the bits of x.
#[inline(always)]
fn zeroinfnan(i: u64) -> bool {
    (i << 1).wrapping_sub(1) >= (INF << 1) - 1
}

/// 0 when y is not an integer, 1 when it is odd, 2 when it is even.
#[inline(always)]
fn checkint(iy: u64) -> i32 {
    let e = ((iy >> 52) & 0x7ff) as i32;
    if e < 0x3ff {
        return 0;
    }
    if e > 0x3ff + 52 {
        return 2;
    }
    if iy & ((1u64 << (0x3ff + 52 - e)) - 1) != 0 {
        return 0;
    }
    if iy & (1u64 << (0x3ff + 52 - e)) != 0 {
        return 1;
    }
    2
}

#[inline(always)]
fn issignaling(x: f64) -> bool {
    ((x.to_bits() ^ 0x0008000000000000) << 1) > (0x7ff8000000000000u64 << 1)
}

/// The C code's `__math_xflow`: a product that overflows or underflows, so the result and the
/// floating-point flags are those of the real operation.
#[inline(never)]
fn xflow(sign: u32, y: f64) -> f64 {
    black_box(if sign != 0 { -y } else { y }) * y
}

fn oflow(sign: u32) -> f64 {
    xflow(sign, d(0x7000000000000000)) // 0x1p769
}

fn uflow(sign: u32) -> f64 {
    xflow(sign, d(0x1000000000000000)) // 0x1p-767
}

// 0/0 as the C code computes it, so the NaN has the bits of the C result.
#[inline(never)]
#[allow(clippy::eq_op)]
fn invalid(x: f64) -> f64 {
    let x = black_box(x);
    (x - x) / (x - x)
}

/// The special cases of the C `pow`, which both builds share. `Break` holds the result.
/// `Continue` holds the bits of |x| (subnormal x normalized) and the sign bias of the result.
#[inline(always)]
fn prologue(x: f64, y: f64) -> ControlFlow<f64, (u64, u32)> {
    let mut sign_bias = 0u32;
    let mut ix = x.to_bits();
    let iy = y.to_bits();
    let mut topx = top12(x);
    let topy = top12(y);
    if topx.wrapping_sub(0x001) >= 0x7ff - 0x001
        || (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be
    {
        // x is subnormal, zero, inf, nan or negative, or |y| is tiny, huge, inf or nan.
        if zeroinfnan(iy) {
            if iy << 1 == 0 {
                return ControlFlow::Break(if issignaling(x) { x + y } else { 1.0 });
            }
            if ix == ONE {
                return ControlFlow::Break(if issignaling(y) { x + y } else { 1.0 });
            }
            if ix << 1 > INF << 1 || iy << 1 > INF << 1 {
                return ControlFlow::Break(x + y);
            }
            if ix << 1 == ONE << 1 {
                return ControlFlow::Break(1.0);
            }
            if (ix << 1 < ONE << 1) == (iy >> 63 == 0) {
                return ControlFlow::Break(0.0); // |x| < 1 && y == inf or |x| > 1 && y == -inf.
            }
            return ControlFlow::Break(y * y);
        }
        if zeroinfnan(ix) {
            let mut x2 = x * x;
            if ix >> 63 != 0 && checkint(iy) == 1 {
                x2 = -x2;
            }
            return ControlFlow::Break(if iy >> 63 != 0 {
                1.0 / black_box(x2)
            } else {
                x2
            });
        }
        // Here x and y are non-zero finite.
        if ix >> 63 != 0 {
            // Finite x < 0.
            let yint = checkint(iy);
            if yint == 0 {
                return ControlFlow::Break(invalid(x));
            }
            if yint == 1 {
                sign_bias = SIGN_BIAS;
            }
            ix &= 0x7fffffffffffffff;
            topx &= 0x7ff;
        }
        if (topy & 0x7ff).wrapping_sub(0x3be) >= 0x43e - 0x3be {
            // Note: sign_bias == 0 here because y is not odd.
            if ix == ONE {
                return ControlFlow::Break(1.0);
            }
            if (topy & 0x7ff) < 0x3be {
                // |y| < 2^-65, x^y ~= 1 + y*log(x).
                return ControlFlow::Break(if ix > ONE { 1.0 + y } else { 1.0 - y });
            }
            return ControlFlow::Break(if (ix > ONE) == (topy < 0x800) {
                oflow(0)
            } else {
                uflow(0)
            });
        }
        if topx == 0 {
            // Normalize subnormal x so the exponent becomes negative.
            ix = (black_box(x) * d((52u64 + 1023) << 52)).to_bits();
            ix &= 0x7fffffffffffffff;
            ix = ix.wrapping_sub(52u64 << 52);
        }
    }
    ControlFlow::Continue((ix, sign_bias))
}

/// The table index, exponent and reduced argument of the log, shared by both builds.
#[inline(always)]
fn log_split(ix: u64) -> (usize, f64, u64) {
    // x = 2^k z, where z is in [OFF, 2 OFF) and exact. The range is split into N subintervals.
    // The i-th subinterval contains z, and c is near its center.
    let tmp = ix.wrapping_sub(OFF);
    let i = ((tmp >> (52 - 7)) % 128) as usize;
    let k = ((tmp as i64) >> 52) as i32; // arithmetic shift
    let iz = ix.wrapping_sub(tmp & (0xfffu64 << 52));
    (i, f64::from(k), iz)
}

/// The range checks of exp, shared by both builds. `Err` holds the result when |x| is tiny or
/// at least 1024. `Ok(true)` means the scale may overflow or underflow, so `specialcase` must
/// finish the job.
#[inline(always)]
fn exp_range(x: f64, sign_bias: u32) -> Result<bool, f64> {
    let abstop = top12(x) & 0x7ff;
    const T54: u32 = 0x3c9; // top12(0x1p-54)
    const T512: u32 = 0x408; // top12(512.0)
    const T1024: u32 = 0x409; // top12(1024.0)
    if abstop.wrapping_sub(T54) >= T512 - T54 {
        if abstop.wrapping_sub(T54) >= 0x80000000 {
            // Tiny x: exp(x) = 1 + x, with the sign of the result.
            let one = 1.0 + x;
            return Err(if sign_bias != 0 { -one } else { one });
        }
        if abstop >= T1024 {
            // |x| >= 1024: the result overflows or underflows.
            return Err(if x.to_bits() >> 63 != 0 {
                uflow(sign_bias)
            } else {
                oflow(sign_bias)
            });
        }
        // Large |x|: the scale may overflow or underflow, so `specialcase` finishes.
        return Ok(true);
    }
    Ok(false)
}

/// The k < 0 branch of the C `specialcase`, which neither build fuses.
#[inline(always)]
#[allow(clippy::assign_op_pattern)] // kept in the order of the C code
fn specialcase_neg(tmp: f64, sbits: u64) -> f64 {
    // k < 0, need special care in the subnormal range.
    let sbits = sbits.wrapping_add(1022u64 << 52);
    let scale = d(sbits);
    let mut y = scale + scale * tmp;
    if y.abs() < 1.0 {
        // Round y to the right precision before scaling it into the subnormal range to avoid
        // a double rounding that can cause a 0.5+e ulp error.
        let one = if y < 0.0 { -1.0 } else { 1.0 };
        let mut lo = scale - y + scale * tmp;
        let hi = one + y;
        lo = one - hi + y + lo;
        y = black_box(hi + lo) - one;
        // Fix the sign of 0.
        if y == 0.0 {
            y = d(sbits & 0x8000000000000000);
        }
    }
    d(1u64 << 52) * y // 0x1p-1022
}

/// The build with no fused operations.
mod plain {
    use super::{d, t};

    #[inline(always)]
    fn log_inline(ix: u64) -> (f64, f64) {
        let (i, kd, iz) = super::log_split(ix);
        let z = d(iz);
        let invc = d(t::LOG_TAB[i][0]);
        let logc = d(t::LOG_TAB[i][1]);
        let logctail = d(t::LOG_TAB[i][2]);
        let a = |j: usize| d(t::A[j]);

        // Split z such that rhi, rlo and rhi*rhi are exact and not subnormal.
        let zhi = d(iz.wrapping_add(1u64 << 31) & (u64::MAX << 32));
        let zlo = z - zhi;
        let rhi = zhi * invc - 1.0;
        let rlo = zlo * invc;
        let r = rhi + rlo;

        // k*Ln2 + log(c) + r.
        let t1 = kd * d(t::LN2HI) + logc;
        let t2 = t1 + r;
        let lo1 = kd * d(t::LN2LO) + logctail;
        let lo2 = t1 - t2 + r;

        // Evaluation is optimized assuming superscalar pipelined execution.
        let ar = a(0) * r; // A[0] = -0.5
        let ar2 = r * ar;
        let ar3 = r * ar2;
        // k*Ln2 + log(c) + r + A[0]*r*r.
        let arhi = a(0) * rhi;
        let arhi2 = rhi * arhi;
        let hi = t2 + arhi2;
        let lo3 = rlo * (ar + arhi);
        let lo4 = t2 - hi + arhi2;
        // p = log1p(r) - r - A[0]*r*r.
        let p = ar3 * (a(1) + r * a(2) + ar2 * (a(3) + r * a(4) + ar2 * (a(5) + r * a(6))));
        let lo = lo1 + lo2 + lo3 + lo4 + p;
        let y = hi + lo;
        let tail = hi - y + lo;
        (y, tail)
    }

    #[inline(never)]
    fn specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
        if ki & 0x80000000 == 0 {
            // k > 0, the exponent of scale might have overflowed by <= 460.
            let sbits = sbits.wrapping_sub(1009u64 << 52);
            let scale = d(sbits);
            let y = scale + scale * tmp;
            return y * d((1009u64 + 1023) << 52); // 0x1p1009
        }
        super::specialcase_neg(tmp, sbits)
    }

    #[inline(always)]
    fn exp_inline(x: f64, xtail: f64, sign_bias: u32) -> f64 {
        let special = match super::exp_range(x, sign_bias) {
            Ok(s) => s,
            Err(v) => return v,
        };
        // exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N), 2^(1/2N)].
        // x = ln2/N*k + r, with int k and r in [-ln2/2N, ln2/2N].
        let z = d(t::INVLN2N) * x;
        // z - kd is in [-1, 1] in non-nearest rounding modes.
        let kd = z + d(t::SHIFT);
        let ki = kd.to_bits();
        let kd = kd - d(t::SHIFT);
        let mut r = x + kd * d(t::NEGLN2HIN) + kd * d(t::NEGLN2LON);
        // The code assumes 2^-200 < |xtail| < 2^-8/N.
        r += xtail;
        // 2^(k/N) ~= scale * (1 + tail).
        let idx = (2 * (ki % 128)) as usize;
        let top = ki.wrapping_add(u64::from(sign_bias)) << (52 - 7);
        let tail = d(t::EXP_TAB[idx]);
        // This is only a valid scale when -1023*N < k < 1024*N.
        let sbits = t::EXP_TAB[idx + 1].wrapping_add(top);
        // exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1).
        let r2 = r * r;
        let c = |j: usize| d(t::C[j]);
        // Without fma the worst case error is 0.25/N ulp larger.
        let tmp = tail + r + r2 * (c(0) + r * c(1)) + r2 * r2 * (c(2) + r * c(3));
        if special {
            return specialcase(tmp, sbits, ki);
        }
        let scale = d(sbits);
        scale + scale * tmp
    }

    pub fn pow(x: f64, y: f64) -> f64 {
        let (ix, sign_bias) = match super::prologue(x, y) {
            std::ops::ControlFlow::Break(v) => return v,
            std::ops::ControlFlow::Continue(s) => s,
        };
        let (hi, lo) = log_inline(ix);
        let iy = y.to_bits();
        let yhi = d(iy & (u64::MAX << 27));
        let ylo = y - yhi;
        let lhi = d(hi.to_bits() & (u64::MAX << 27));
        let llo = hi - lhi + lo;
        let ehi = yhi * lhi;
        let elo = ylo * lhi + y * llo; // |elo| < |y| * 0x1p-25
        exp_inline(ehi, elo, sign_bias)
    }
}

/// The build where the compiler fuses multiply-adds, with each fused operation written as
/// `fma` at the place gcc fuses it under `-mfma -mavx2 -ffp-contract=fast`.
#[cfg(target_arch = "x86_64")]
mod fused {
    use super::{d, t};

    /// a*b + c with one rounding, as the `vfmadd` instruction. `f64::mul_add` would not do:
    /// in a debug build it stays a call into a function compiled without the FMA feature,
    /// which calls MinGW-w64's `fma` on windows-gnu, and that gave other bits (2026-11-27).
    #[target_feature(enable = "fma")]
    #[inline]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        use std::arch::x86_64::{_mm_cvtsd_f64, _mm_fmadd_sd, _mm_set_sd};
        _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
    }

    #[target_feature(enable = "fma")]
    #[inline]
    fn log_inline(ix: u64) -> (f64, f64) {
        let (i, kd, iz) = super::log_split(ix);
        let z = d(iz);
        let invc = d(t::LOG_TAB[i][0]);
        let logc = d(t::LOG_TAB[i][1]);
        let logctail = d(t::LOG_TAB[i][2]);
        let a = |j: usize| d(t::A[j]);

        // With fma, r = z*invc - 1 is exact, and the C code takes this branch.
        let r = fma(z, invc, -1.0);

        // k*Ln2 + log(c) + r.
        let t1 = fma(kd, d(t::LN2HI), logc);
        let t2 = r + t1;
        let lo1 = fma(kd, d(t::LN2LO), logctail);
        let lo2 = t1 - t2 + r;

        let ar = a(0) * r; // A[0] = -0.5
        let ar2 = r * ar;
        let ar3 = r * ar2;
        // k*Ln2 + log(c) + r + A[0]*r*r.
        let hi = t2 + ar2;
        let lo3 = fma(r, ar, -ar2);
        let lo4 = t2 - hi + ar2;
        // p = log1p(r) - r - A[0]*r*r, with lo1 + lo2 + lo3 + lo4 added in the last fma.
        let q1 = fma(r, a(2), a(1));
        let q3 = fma(r, a(4), a(3));
        let q5 = fma(r, a(6), a(5));
        let q = fma(ar2, fma(ar2, q5, q3), q1);
        let lo = fma(ar3, q, lo1 + lo2 + lo3 + lo4);
        let y = hi + lo;
        let tail = hi - y + lo;
        (y, tail)
    }

    #[target_feature(enable = "fma")]
    #[inline(never)]
    fn specialcase(tmp: f64, sbits: u64, ki: u64) -> f64 {
        if ki & 0x80000000 == 0 {
            // k > 0, the exponent of scale might have overflowed by <= 460.
            let sbits = sbits.wrapping_sub(1009u64 << 52);
            let scale = d(sbits);
            let y = fma(tmp, scale, scale);
            return y * d((1009u64 + 1023) << 52); // 0x1p1009
        }
        super::specialcase_neg(tmp, sbits)
    }

    #[target_feature(enable = "fma")]
    #[inline]
    fn exp_inline(x: f64, xtail: f64, sign_bias: u32) -> f64 {
        let special = match super::exp_range(x, sign_bias) {
            Ok(s) => s,
            Err(v) => return v,
        };
        let kd = fma(x, d(t::INVLN2N), d(t::SHIFT));
        let ki = kd.to_bits();
        let kd = kd - d(t::SHIFT);
        let mut r = fma(kd, d(t::NEGLN2LON), fma(kd, d(t::NEGLN2HIN), x));
        r += xtail;
        let idx = (2 * (ki % 128)) as usize;
        let top = ki.wrapping_add(u64::from(sign_bias)) << (52 - 7);
        let tail = d(t::EXP_TAB[idx]);
        let sbits = t::EXP_TAB[idx + 1].wrapping_add(top);
        let r2 = r * r;
        let c = |j: usize| d(t::C[j]);
        let tmp = fma(
            r2 * r2,
            fma(r, c(3), c(2)),
            fma(r2, fma(r, c(1), c(0)), tail + r),
        );
        if special {
            return specialcase(tmp, sbits, ki);
        }
        let scale = d(sbits);
        fma(tmp, scale, scale)
    }

    #[target_feature(enable = "fma")]
    pub fn pow(x: f64, y: f64) -> f64 {
        let (ix, sign_bias) = match super::prologue(x, y) {
            std::ops::ControlFlow::Break(v) => return v,
            std::ops::ControlFlow::Continue(s) => s,
        };
        let (hi, lo) = log_inline(ix);
        let ehi = y * hi;
        let elo = fma(y, lo, fma(y, hi, -ehi));
        exp_inline(ehi, elo, sign_bias)
    }
}

/// True when the CPU has FMA and AVX2, the test glibc uses to pick its fused `pow`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_fma_avx2() -> bool {
    std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx2")
}

/// `x^y` from the build with no fused operations, on every CPU.
pub fn pow_plain(x: f64, y: f64) -> f64 {
    plain::pow(x, y)
}

/// `x^y` from the fused build, or `None` when the CPU lacks FMA or AVX2 or is not x86_64.
pub fn pow_fused(x: f64, y: f64) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    if has_fma_avx2() {
        // SAFETY: the CPU has the FMA feature that `fused::pow` is compiled for.
        return Some(unsafe { fused::pow(x, y) });
    }
    let _ = (x, y);
    None
}

/// `x^y` with the same choice of build that glibc makes on this CPU: fused when the CPU has
/// FMA and AVX2, plain otherwise.
pub fn pow(x: f64, y: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if has_fma_avx2() {
        // SAFETY: the CPU has the FMA feature that `fused::pow` is compiled for.
        return unsafe { fused::pow(x, y) };
    }
    plain::pow(x, y)
}

#[cfg(test)]
#[path = "aorpow_tests.rs"]
mod tests;
