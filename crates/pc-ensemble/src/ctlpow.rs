// Ported from the `libm` crate 0.2.16 (MIT, see THIRD_PARTY_NOTICES.md), file
// `src/math/pow.rs`, which carries this notice:
//
//   origin: FreeBSD /usr/src/lib/msun/src/e_pow.c
//   ====================================================
//   Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
//
//   Permission to use, copy, modify, and distribute this
//   software is freely granted, provided that this notice
//   is preserved.
//   ====================================================
//
// What changed: `pow` is split in two at the point where it has log2(x) as the pair
// (t1, t2), which depends on x alone, so that two powers of the same x share one log.
// Only the path for a positive normal x other than 1 and an exponent y with 0 < y < 0.9,
// y != 0.5 is kept. On that path none of the special-case branches of `pow` is taken, and
// |y log2(x)| < 0.9 * 1075, so the overflow, underflow and subnormal-result branches are
// never taken either. Every other input calls `pc_ir::math::pow`, which is `libm::pow`. The operations that remain are
// those of `pow`, in the same order, so the result has the same bits.

//! Both powers of the Dormand-Prince step-size controller from one logarithm
//! (SPEC section 13, `cpu`).

// The constants keep the digits of the source, so the lints about their precision and
// about LG2 being `LN_2` are off.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

const BP: [f64; 2] = [1.0, 1.5];
const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; /* 0x3fe2b803_40000000 */
const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; /* 0x3E4CFDEB, 0x43CFD006 */
const L1: f64 = 5.99999999999994648725e-01; /* 0x3fe33333_33333303 */
const L2: f64 = 4.28571428578550184252e-01; /* 0x3fdb6db6_db6fabff */
const L3: f64 = 3.33333329818377432918e-01; /* 0x3fd55555_518f264d */
const L4: f64 = 2.72728123808534006489e-01; /* 0x3fd17460_a91d4101 */
const L5: f64 = 2.30660745775561754067e-01; /* 0x3fcd864a_93c9db65 */
const L6: f64 = 2.06975017800338417784e-01; /* 0x3fca7e28_4a454eef */
const P1: f64 = 1.66666666666666019037e-01; /* 0x3fc55555_5555553e */
const P2: f64 = -2.77777777770155933842e-03; /* 0xbf66c16c_16bebd93 */
const P3: f64 = 6.61375632143793436117e-05; /* 0x3f11566a_af25de2c */
const P4: f64 = -1.65339022054652515390e-06; /* 0xbebbbd41_c5d26bf1 */
const P5: f64 = 4.13813679705723846039e-08; /* 0x3e663769_72bea4d0 */
const LG2: f64 = 6.93147180559945286227e-01; /* 0x3fe62e42_fefa39ef */
const LG2_H: f64 = 6.93147182464599609375e-01; /* 0x3fe62e43_00000000 */
const LG2_L: f64 = -1.90465429995776804525e-09; /* 0xbe205c61_0ca86c39 */
const CP: f64 = 9.61796693925975554329e-01; /* 0x3feec709_dc3a03fd =2/(3ln2) */
const CP_H: f64 = 9.61796700954437255859e-01; /* 0x3feec709_e0000000 =(float)cp */
const CP_L: f64 = -7.02846165095275826516e-09; /* 0xbe3e2fe0_145b01f5 =tail of cp_h*/

#[inline(always)]
fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline(always)]
fn with_set_high_word(x: f64, hi: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff) | ((hi as u64) << 32))
}

#[inline(always)]
fn with_set_low_word(x: f64, lo: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | lo as u64)
}

/// True for an exponent the split path handles: 0 < y < 0.9 and y != 0.5.
#[inline(always)]
fn exponent_ok(y: f64) -> bool {
    y > 0.0 && y < 0.9 && y != 0.5
}

/// log2(x) as the pair (t1, t2) of `pow`, for a positive normal x other than 1, and None
/// for every other x.
#[inline(always)]
fn log2_pair(x: f64) -> Option<(f64, f64)> {
    let b = x.to_bits();
    // Below 2^-1022 (zero, subnormal), above the largest finite (inf, NaN), negative, or 1.
    if b.wrapping_sub(0x0010_0000_0000_0000) >= 0x7fe0_0000_0000_0000 || b == 0x3ff0_0000_0000_0000
    {
        return None;
    }
    let mut ix = (b >> 32) as i32;
    let mut n: i32 = (ix >> 20) - 0x3ff;
    let j = ix & 0x000fffff;

    /* determine interval */
    let k: usize;
    ix = j | 0x3ff00000; /* normalize ix */
    if j <= 0x3988E {
        /* |x|<sqrt(3/2) */
        k = 0;
    } else if j < 0xBB67A {
        /* |x|<sqrt(3)   */
        k = 1;
    } else {
        k = 0;
        n += 1;
        ix -= 0x00100000;
    }
    let ax = with_set_high_word(x, ix as u32);

    /* compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
    let u = ax - BP[k]; /* bp[0]=1.0, bp[1]=1.5 */
    let v = 1.0 / (ax + BP[k]);
    let ss = u * v;
    let s_h = with_set_low_word(ss, 0);

    /* t_h=ax+bp[k] High */
    let t_h = with_set_high_word(
        0.0,
        ((ix as u32 >> 1) | 0x20000000) + 0x00080000 + ((k as u32) << 18),
    );
    let t_l = ax - (t_h - BP[k]);
    let s_l = v * ((u - s_h * t_h) - s_h * t_l);

    /* compute log(ax) */
    let s2 = ss * ss;
    let mut r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
    r += s_l * (s_h + ss);
    let s2 = s_h * s_h;
    let t_h = with_set_low_word(3.0 + s2 + r, 0);
    let t_l = r - ((t_h - 3.0) - s2);

    /* u+v = ss*(1+...) */
    let u = s_h * t_h;
    let v = s_l * t_h + t_l * ss;

    /* 2/(3log2)*(ss+...) */
    let p_h = with_set_low_word(u + v, 0);
    let p_l = v - (p_h - u);
    let z_h = CP_H * p_h; /* cp_h+cp_l = 2/(3*log2) */
    let z_l = CP_L * p_h + p_l * CP + DP_L[k];

    /* log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l */
    let t = n as f64;
    let t1 = with_set_low_word(((z_h + z_l) + DP_H[k]) + t, 0);
    let t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
    Some((t1, t2))
}

/// 2^(y (t1 + t2)) as `pow` forms it, for an exponent that passes `exponent_ok` and the
/// pair of a positive normal x. Then |z| < 1024 and the result is normal, so the overflow,
/// underflow and subnormal branches of `pow` are left out, and the sign is +1.
#[inline(always)]
fn exp2_part(y: f64, t1: f64, t2: f64) -> f64 {
    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    let y1 = with_set_low_word(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let j = (z.to_bits() >> 32) as i32;

    /* compute 2**(p_h+p_l) */
    let i = j & 0x7fffffff_i32;
    let mut k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;
    if i > 0x3fe00000 {
        /* if |z| > 0.5, set n = [z+0.5] */
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff; /* new k for n */
        let t = with_set_high_word(0.0, (n & !(0x000fffff >> k)) as u32);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_set_low_word(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    let jj = (high_word(z) as i32) + (n << 20);
    with_set_high_word(z, jj as u32)
}

/// `(pc_ir::math::pow(x, a), pc_ir::math::pow(x, b))` (which is `libm::pow`), bit for bit, with one log2(x) when x is positive,
/// normal and not 1 and both exponents are in (0, 0.9) and not 0.5.
///
/// Not inlined: inlined into the DP5 step loop it made Lorenz-96 with 40 states run twice
/// as long on x86_64-pc-windows-gnu (2026-11-23), while the call costs nothing measurable.
#[inline(never)]
pub fn pow_pair(x: f64, a: f64, b: f64) -> (f64, f64) {
    if exponent_ok(a)
        && exponent_ok(b)
        && let Some((t1, t2)) = log2_pair(x)
    {
        return (exp2_part(a, t1, t2), exp2_part(b, t1, t2));
    }
    (pc_ir::math::pow(x, a), pc_ir::math::pow(x, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPONENTS: [f64; 2] = [0.2 - 0.04 * 0.75, 0.04];

    fn check(x: f64) {
        for &a in &EXPONENTS {
            for &b in &EXPONENTS {
                let (pa, pb) = pow_pair(x, a, b);
                assert_eq!(
                    pa.to_bits(),
                    pc_ir::math::pow(x, a).to_bits(),
                    "x = {x:e}, y = {a}"
                );
                assert_eq!(
                    pb.to_bits(),
                    pc_ir::math::pow(x, b).to_bits(),
                    "x = {x:e}, y = {b}"
                );
            }
        }
    }

    /// A small deterministic generator, so the test needs no dependency.
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn special_values_match_pow() {
        let tiny = f64::from_bits(1);
        for x in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            tiny,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE / 2.0,
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            -2.0,
            -1e-300,
            1e-4,
            0.5,
            2.0,
            1.5f64.sqrt(),
            3f64.sqrt(),
            f64::EPSILON,
            1.0 + f64::EPSILON,
            1.0 - f64::EPSILON / 2.0,
        ] {
            check(x);
        }
    }

    #[test]
    fn other_exponents_take_pow() {
        for x in [1e-9, 0.3, 1.0, 7.0, 0.0, f64::NAN] {
            for (a, b) in [
                (0.5, 0.04),
                (0.17, 1.0),
                (0.0, 0.2),
                (-0.1, 0.04),
                (0.9, 2.0),
            ] {
                let (pa, pb) = pow_pair(x, a, b);
                assert_eq!(
                    pa.to_bits(),
                    pc_ir::math::pow(x, a).to_bits(),
                    "x = {x:e}, y = {a}"
                );
                assert_eq!(
                    pb.to_bits(),
                    pc_ir::math::pow(x, b).to_bits(),
                    "x = {x:e}, y = {b}"
                );
            }
        }
    }

    #[test]
    fn every_binade_matches_pow() {
        // Random doubles in each binade from 2^-1022 to 2^1023, and a denser sample where
        // the controller works (errors from 2^-60 to 2^8).
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for e in 1u64..=2046 {
            let per = if (1023 - 60..=1023 + 8).contains(&e) {
                4096
            } else {
                64
            };
            for _ in 0..per {
                let m = next(&mut s) & 0x000f_ffff_ffff_ffff;
                check(f64::from_bits((e << 52) | m));
            }
        }
    }

    #[test]
    fn the_interval_boundaries_match_pow() {
        // The log picks its interval from the high word at sqrt(3/2) and sqrt(3), and the
        // exp part changes form at |z| = 0.5; walk the doubles next to each edge.
        for edge in [1.5f64.sqrt(), 3f64.sqrt(), 1.0, 2.0, 0.5] {
            let b = edge.to_bits();
            for d in 0..20_000u64 {
                check(f64::from_bits(b + d));
                check(f64::from_bits(b - d));
            }
        }
        for &y in &EXPONENTS {
            // x with y log2(x) near +-0.5 and near +-1.
            for zz in [0.5f64, -0.5, 1.0, -1.0] {
                let b = (zz / y).exp2().to_bits();
                for d in 0..20_000u64 {
                    check(f64::from_bits(b + d));
                    check(f64::from_bits(b - d));
                }
            }
        }
    }
}
