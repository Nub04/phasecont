//! Scalar math builtins, the polygamma function, lookup tables and the C shim table.
//!
//! Every function is a plain `fn ... -> f64` backed by `libm`, so results are the same
//! bits on every host. None of them panics. The `pc_math_*` functions wrap them for
//! `extern "C"` callers, and [`MATH`] lists the wrappers in a frozen field order.

pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}
/// `atan2(y, x)`, C argument order.
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}
pub fn sinh(x: f64) -> f64 {
    libm::sinh(x)
}
pub fn cosh(x: f64) -> f64 {
    libm::cosh(x)
}
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}
/// Natural logarithm.
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}
/// Same as `libm::pow`: `pow(0, 0)` is 1 and `pow(-8, 1/3)` is NaN.
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}
/// C `fmod`: the result has the sign of `x`.
pub fn fmod(x: f64, y: f64) -> f64 {
    libm::fmod(x, y)
}
/// `ln|Gamma(x)|`.
pub fn lgamma(x: f64) -> f64 {
    libm::lgamma_r(x).0
}

fn order(n: f64) -> Option<i32> {
    let t = n.trunc();
    if t.is_finite() && t.abs() <= 2147483647.0 {
        Some(t as i32)
    } else {
        None
    }
}

/// `J_n(x)` with the order truncated toward zero.
pub fn besselj(n: f64, x: f64) -> f64 {
    match order(n) {
        Some(k) => libm::jn(k, x),
        None => f64::NAN,
    }
}

/// `Y_n(x)` with the order truncated toward zero.
pub fn bessely(n: f64, x: f64) -> f64 {
    match order(n) {
        Some(k) => libm::yn(k, x),
        None => f64::NAN,
    }
}

/// XPPAUT `mod`: `fmod(x, y)`, plus `y` when that is negative. `pc_mod(x, 0)` is NaN.
pub fn pc_mod(x: f64, y: f64) -> f64 {
    let r = libm::fmod(x, y);
    if r < 0.0 { r + y } else { r }
}

/// 1 when `x >= 0`, else 0 (NaN gives 0).
pub fn heav(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

/// -1, 0 or 1; NaN gives NaN.
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

// ---------------------------------------------------------------------------
// Polygamma
// ---------------------------------------------------------------------------

/// B_2k for k = 1..=8 (DLMF Table 24.2.1).
const BERNOULLI: [f64; 8] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
];

/// Below this the recurrence shifts the argument up before the asymptotic series.
const SHIFT: f64 = 20.0;

fn factorial(n: u32) -> f64 {
    (2..=n).fold(1.0, |a, k| a * f64::from(k))
}

/// Asymptotic series for x >= SHIFT (DLMF 5.11.2 for n = 0, 5.15.9 for n >= 1).
fn polygamma_asymptotic(n: u32, z: f64) -> f64 {
    if n == 0 {
        let z2 = z * z;
        let mut zp = z2;
        let mut sum = 0.0;
        for (k, b) in BERNOULLI.iter().enumerate() {
            let kk = (k + 1) as f64;
            sum += b / (2.0 * kk * zp);
            zp *= z2;
        }
        return libm::log(z) - 0.5 / z - sum;
    }
    let zn = libm::pow(z, f64::from(n));
    let mut total = factorial(n - 1) / zn + factorial(n) / (2.0 * zn * z);
    let z2 = z * z;
    let mut zp = zn * z2;
    for (k, b) in BERNOULLI.iter().enumerate() {
        let k2 = 2 * (k as u32 + 1);
        // (2k+n-1)! / (2k)! = (2k+1) * ... * (2k+n-1)
        let ratio = (k2 + 1..k2 + n).fold(1.0, |a, j| a * f64::from(j));
        total += ratio * b / zp;
        zp *= z2;
    }
    if (n - 1).is_multiple_of(2) {
        total
    } else {
        -total
    }
}

/// P_n(c) with d^n/dz^n cot(pi z) = pi^n P_n(cot(pi z)), from P_{k+1} = -(1 + c^2) P_k'.
fn cot_poly(n: u32, c: f64) -> f64 {
    let c2 = c * c;
    match n {
        0 => c,
        1 => -(1.0 + c2),
        2 => 2.0 * c * (1.0 + c2),
        3 => -(2.0 + c2 * (8.0 + 6.0 * c2)),
        _ => c * (16.0 + c2 * (40.0 + 24.0 * c2)),
    }
}

/// Polygamma `psi^(m)(x)` for `m` in 0..=4; NaN for any other `m`.
///
/// Uses the recurrence DLMF 5.15.5 to shift `x > 0` up to 20, the asymptotic series of
/// DLMF 5.11.2 and 5.15.9 there, and the reflection formula of DLMF 5.15.6 for `x < 0`.
/// `x` = 0 or a negative integer gives NaN.
pub fn polygamma(m: f64, x: f64) -> f64 {
    let n = if m == 0.0 {
        0
    } else if m == 1.0 {
        1
    } else if m == 2.0 {
        2
    } else if m == 3.0 {
        3
    } else if m == 4.0 {
        4
    } else {
        return f64::NAN;
    };
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return if n == 0 { f64::INFINITY } else { 0.0 };
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 || (x < 0.0 && x == libm::floor(x)) {
        return f64::NAN;
    }
    if x < 0.0 {
        // psi^(n)(z) = (-1)^n psi^(n)(1-z) - pi^(n+1) P_n(cot(pi z)); cot has period 1.
        let f = x - libm::round(x);
        let c = libm::cos(core::f64::consts::PI * f) / libm::sin(core::f64::consts::PI * f);
        let refl = libm::pow(core::f64::consts::PI, f64::from(n) + 1.0) * cot_poly(n, c);
        let up = polygamma_positive(n, 1.0 - x);
        let up = if n % 2 == 0 { up } else { -up };
        return up - refl;
    }
    polygamma_positive(n, x)
}

fn polygamma_positive(n: u32, x: f64) -> f64 {
    // psi^(n)(z) = psi^(n)(z+1) - (-1)^n n! z^(-n-1), summed while shifting up.
    let nf = factorial(n);
    let mut z = x;
    let mut acc = 0.0;
    while z < SHIFT {
        let mut d = z;
        for _ in 0..n {
            d *= z;
        }
        acc += nf / d;
        z += 1.0;
    }
    let corr = if n.is_multiple_of(2) { -acc } else { acc };
    polygamma_asymptotic(n, z) + corr
}

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

/// Number of points of a table block, or `None` when the header is not valid.
fn table_npts(data: &[f64]) -> Option<usize> {
    let n = *data.first()?;
    if !(n.is_finite() && (2.0..=2147483647.0).contains(&n)) {
        return None;
    }
    let npts = n as usize;
    if data.len() < 3 + npts {
        None
    } else {
        Some(npts)
    }
}

/// Segment index, node position, step and the two end values for `x`.
fn table_segment(data: &[f64], x: f64) -> Option<(f64, f64, f64, f64)> {
    let npts = table_npts(data)?;
    let lo = data[1];
    let hi = data[2];
    let dx = (hi - lo) / (npts - 1) as f64;
    if x.is_nan() {
        return None;
    }
    let raw = libm::floor((x - lo) / dx);
    let i = if raw.is_nan() {
        0
    } else {
        raw.clamp(0.0, (npts - 2) as f64) as usize
    };
    let xi = lo + i as f64 * dx;
    let y0 = *data.get(3 + i)?;
    let y1 = *data.get(4 + i)?;
    Some((xi, dx, y0, y1))
}

/// Piecewise linear lookup. The end segments extrapolate outside `[lo, hi]`.
///
/// Block layout: `data[0]` = npts (at least 2), `data[1]` = lo, `data[2]` = hi,
/// `data[3..3 + npts]` = values at `lo + i * (hi - lo) / (npts - 1)`.
/// A short or malformed block, or a NaN `x`, gives NaN.
pub fn table(data: &[f64], x: f64) -> f64 {
    match table_segment(data, x) {
        Some((xi, dx, y0, y1)) => y0 + (x - xi) * (y1 - y0) / dx,
        None => f64::NAN,
    }
}

/// Slope of the segment that [`table`] uses (the right segment at an interior node).
pub fn table_slope(data: &[f64], x: f64) -> f64 {
    match table_segment(data, x) {
        Some((_, dx, y0, y1)) => (y1 - y0) / dx,
        None => f64::NAN,
    }
}

/// Builds the slice for a table block from a raw pointer, or `None` when the header is bad.
///
/// # Safety
/// `data` must be null or point to `3 + npts` readable `f64`, where `npts` is `*data`.
unsafe fn table_block<'a>(data: *const f64) -> Option<&'a [f64]> {
    if data.is_null() {
        return None;
    }
    // SAFETY: the caller guarantees at least the header is readable.
    let n = unsafe { *data };
    if !(n.is_finite() && (2.0..=2147483647.0).contains(&n)) {
        return None;
    }
    // SAFETY: the caller guarantees 3 + npts readable values.
    Some(unsafe { core::slice::from_raw_parts(data, 3 + n as usize) })
}

// ---------------------------------------------------------------------------
// C shims
// ---------------------------------------------------------------------------

pub type F1 = unsafe extern "C" fn(f64) -> f64;
pub type F2 = unsafe extern "C" fn(f64, f64) -> f64;
pub type FT = unsafe extern "C" fn(*const f64, f64) -> f64;

macro_rules! shim1 {
    ($($shim:ident => $f:ident),* $(,)?) => {
        $(pub extern "C" fn $shim(x: f64) -> f64 { $f(x) })*
    };
}
macro_rules! shim2 {
    ($($shim:ident => $f:ident),* $(,)?) => {
        $(pub extern "C" fn $shim(x: f64, y: f64) -> f64 { $f(x, y) })*
    };
}

shim1! {
    pc_math_sin => sin, pc_math_cos => cos, pc_math_tan => tan, pc_math_asin => asin,
    pc_math_acos => acos, pc_math_atan => atan, pc_math_sinh => sinh, pc_math_cosh => cosh,
    pc_math_tanh => tanh, pc_math_exp => exp, pc_math_ln => ln, pc_math_log10 => log10,
    pc_math_sqrt => sqrt, pc_math_erf => erf, pc_math_erfc => erfc, pc_math_lgamma => lgamma,
    pc_math_floor => floor, pc_math_ceil => ceil,
}
shim2! {
    pc_math_atan2 => atan2, pc_math_pow => pow, pc_math_besselj => besselj,
    pc_math_bessely => bessely, pc_math_polygamma => polygamma, pc_math_mod => pc_mod,
}

/// Shim for [`table`].
///
/// # Safety
/// `data` must be null or point to a table block of `3 + npts` readable `f64`, where
/// `npts` is `*data`. A null pointer or an invalid header gives NaN.
pub unsafe extern "C" fn pc_math_table(data: *const f64, x: f64) -> f64 {
    // SAFETY: forwarded from the caller's contract.
    match unsafe { table_block(data) } {
        Some(d) => table(d, x),
        None => f64::NAN,
    }
}

/// Shim for [`table_slope`]; same contract as [`pc_math_table`].
///
/// # Safety
/// See [`pc_math_table`].
pub unsafe extern "C" fn pc_math_table_slope(data: *const f64, x: f64) -> f64 {
    // SAFETY: forwarded from the caller's contract.
    match unsafe { table_block(data) } {
        Some(d) => table_slope(d, x),
        None => f64::NAN,
    }
}

/// Function table that compiled code calls. The field order is frozen and Task 12 mirrors
/// it as the C struct `pc_math_t`.
#[repr(C)]
pub struct MathTable {
    pub sin: F1,
    pub cos: F1,
    pub tan: F1,
    pub asin: F1,
    pub acos: F1,
    pub atan: F1,
    pub atan2: F2,
    pub sinh: F1,
    pub cosh: F1,
    pub tanh: F1,
    pub exp: F1,
    pub ln: F1,
    pub log10: F1,
    pub sqrt: F1,
    pub pow: F2,
    pub erf: F1,
    pub erfc: F1,
    pub lgamma: F1,
    pub besselj: F2,
    pub bessely: F2,
    pub polygamma: F2,
    pub floor: F1,
    pub ceil: F1,
    pub pc_mod: F2,
    pub table: FT,
    pub table_slope: FT,
}

pub static MATH: MathTable = MathTable {
    sin: pc_math_sin,
    cos: pc_math_cos,
    tan: pc_math_tan,
    asin: pc_math_asin,
    acos: pc_math_acos,
    atan: pc_math_atan,
    atan2: pc_math_atan2,
    sinh: pc_math_sinh,
    cosh: pc_math_cosh,
    tanh: pc_math_tanh,
    exp: pc_math_exp,
    ln: pc_math_ln,
    log10: pc_math_log10,
    sqrt: pc_math_sqrt,
    pow: pc_math_pow,
    erf: pc_math_erf,
    erfc: pc_math_erfc,
    lgamma: pc_math_lgamma,
    besselj: pc_math_besselj,
    bessely: pc_math_bessely,
    polygamma: pc_math_polygamma,
    floor: pc_math_floor,
    ceil: pc_math_ceil,
    pc_mod: pc_math_mod,
    table: pc_math_table,
    table_slope: pc_math_table_slope,
};

/// Every shim under the symbol name the JIT imports, in `MathTable` field order.
pub fn symbols() -> Vec<(&'static str, *const u8)> {
    let m = &MATH;
    vec![
        ("pc_sin", m.sin as *const u8),
        ("pc_cos", m.cos as *const u8),
        ("pc_tan", m.tan as *const u8),
        ("pc_asin", m.asin as *const u8),
        ("pc_acos", m.acos as *const u8),
        ("pc_atan", m.atan as *const u8),
        ("pc_atan2", m.atan2 as *const u8),
        ("pc_sinh", m.sinh as *const u8),
        ("pc_cosh", m.cosh as *const u8),
        ("pc_tanh", m.tanh as *const u8),
        ("pc_exp", m.exp as *const u8),
        ("pc_ln", m.ln as *const u8),
        ("pc_log10", m.log10 as *const u8),
        ("pc_sqrt", m.sqrt as *const u8),
        ("pc_pow", m.pow as *const u8),
        ("pc_erf", m.erf as *const u8),
        ("pc_erfc", m.erfc as *const u8),
        ("pc_lgamma", m.lgamma as *const u8),
        ("pc_besselj", m.besselj as *const u8),
        ("pc_bessely", m.bessely as *const u8),
        ("pc_polygamma", m.polygamma as *const u8),
        ("pc_floor", m.floor as *const u8),
        ("pc_ceil", m.ceil as *const u8),
        ("pc_mod", m.pc_mod as *const u8),
        ("pc_table", m.table as *const u8),
        ("pc_table_slope", m.table_slope as *const u8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    const GAMMA: f64 = core::f64::consts::EULER_GAMMA;
    const ZETA3: f64 = 1.202_056_903_159_594_3;
    const ZETA5: f64 = 1.036_927_755_143_37;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        let e = rel(a, b);
        assert!(e <= tol, "{what}: got {a:e}, want {b:e}, rel err {e:e}");
    }

    #[test]
    fn heav_and_sign_edges() {
        assert_eq!(heav(0.0), 1.0);
        assert_eq!(heav(-0.0), 1.0);
        assert_eq!(heav(-1e-300), 0.0);
        assert_eq!(heav(f64::NEG_INFINITY), 0.0);
        assert_eq!(heav(f64::INFINITY), 1.0);
        assert_eq!(heav(f64::NAN), 0.0);
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
        assert_eq!(sign(3.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(f64::NEG_INFINITY), -1.0);
        assert!(sign(f64::NAN).is_nan());
    }

    #[test]
    fn mod_semantics() {
        assert_eq!(pc_mod(-7.0, 3.0), 2.0);
        assert_eq!(pc_mod(7.0, -3.0), 1.0);
        assert_eq!(pc_mod(-7.0, -3.0), -4.0);
        assert_eq!(pc_mod(-7.5, 2.0), 0.5);
        assert!(pc_mod(1.0, 0.0).is_nan());
        assert_eq!(pc_mod(0.75, -1.0), 0.75);
        assert_eq!(pc_mod(2.5, -1.0), 0.5);
        assert_eq!(pc_mod(-2.5, -1.0), -1.5);
        assert_eq!(pc_mod(-123456.75, 7.0), 2.25);
        assert!((pc_mod(-0.1, 1.0) - 0.9).abs() < 1e-15);
        assert!(pc_mod(f64::INFINITY, 3.0).is_nan());
    }

    #[test]
    fn bessel_orders() {
        assert_eq!(besselj(1.5, 1.0), besselj(1.0, 1.0));
        assert_eq!(besselj(-0.5, 1.0), besselj(0.0, 1.0));
        assert_eq!(bessely(1.5, 1.0), bessely(1.0, 1.0));
        assert!(besselj(f64::NAN, 1.0).is_nan());
        assert!(besselj(f64::INFINITY, 1.0).is_nan());
        assert!(bessely(3e9, 1.0).is_nan());
        assert!((besselj(0.0, 1.0) - 0.765_197_686_557_966_6).abs() < 1e-15);
        assert!((bessely(0.0, 1.0) - 0.088_256_964_215_676_96).abs() < 1e-15);
    }

    #[test]
    fn builtin_edges() {
        assert_eq!(pow(0.0, 0.0), 1.0);
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert!(pow(-8.0, 1.0 / 3.0).is_nan());
        assert!(ln(-1.0).is_nan());
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(sqrt(-1.0).is_nan());
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(log10(100.0), 2.0);
        assert!((tan(1.0) - 1.557_407_724_654_902_3).abs() < 1e-15);
        assert!((asin(1.0) - PI / 2.0).abs() < 1e-16);
        assert!((atan2(1.0, 0.0) - PI / 2.0).abs() < 1e-16);
        assert!((atan2(-1.0, -1.0) + 3.0 * PI / 4.0).abs() < 1e-15);
        assert_eq!(floor(-2.5), -3.0);
        assert_eq!(floor(-2.0), -2.0);
        assert_eq!(ceil(-2.5), -2.0);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-15);
        assert!((erfc(1.0) - 0.157_299_207_050_285_1).abs() < 1e-15);
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-14);
        assert!((lgamma(-0.5) - 1.265_512_123_484_645_4).abs() < 1e-14);
        assert!(sin(f64::INFINITY).is_nan());
        assert!(sin(f64::NAN).is_nan());
        assert!(sin(-0.0).is_sign_negative());
        assert_eq!(tanh(f64::NEG_INFINITY), -1.0);
        assert_eq!(fmod(-7.0, 3.0), -1.0);
    }

    #[test]
    fn polygamma_at_one() {
        close(polygamma(0.0, 1.0), -GAMMA, 1e-14, "psi(1)");
        close(polygamma(1.0, 1.0), PI * PI / 6.0, 1e-14, "psi'(1)");
        close(polygamma(2.0, 1.0), -2.0 * ZETA3, 1e-14, "psi''(1)");
        close(polygamma(3.0, 1.0), PI.powi(4) / 15.0, 1e-14, "psi'''(1)");
        close(polygamma(4.0, 1.0), -24.0 * ZETA5, 1e-14, "psi4(1)");
    }

    #[test]
    fn polygamma_at_half() {
        close(
            polygamma(0.0, 0.5),
            -GAMMA - 2.0 * 2.0_f64.ln(),
            1e-14,
            "psi(1/2)",
        );
        let z = [PI * PI / 6.0, ZETA3, PI.powi(4) / 90.0, ZETA5];
        let mut fact = 1.0;
        for n in 1..=4usize {
            fact *= n as f64;
            let sgn = if (n + 1) % 2 == 0 { 1.0 } else { -1.0 };
            let want = sgn * fact * (2.0_f64.powi(n as i32 + 1) - 1.0) * z[n - 1];
            close(
                polygamma(n as f64, 0.5),
                want,
                1e-14,
                &format!("psi^{n}(1/2)"),
            );
        }
    }

    /// psi^(n)(z) - psi^(n)(z+1) = -(-1)^n n! z^(-n-1)
    fn step(n: i32, z: f64) -> f64 {
        let f: f64 = (1..=n).map(f64::from).product();
        let s = if n % 2 == 0 { 1.0 } else { -1.0 };
        -s * f / z.powi(n + 1)
    }

    #[test]
    fn polygamma_recurrence_forward() {
        for n in 0..=4 {
            let a = polygamma(f64::from(n), 3.7);
            let b = polygamma(f64::from(n), 4.7);
            close(a, b + step(n, 3.7), 1e-13, &format!("recurrence n={n}"));
        }
    }

    #[test]
    fn polygamma_reflection() {
        for n in 0..=4 {
            let nf = f64::from(n);
            close(
                polygamma(nf, -0.5),
                polygamma(nf, 0.5) + step(n, -0.5),
                1e-12,
                &format!("reflect -0.5 n={n}"),
            );
            close(
                polygamma(nf, -2.3),
                polygamma(nf, -1.3) + step(n, -2.3),
                1e-12,
                &format!("reflect -2.3 n={n}"),
            );
            close(
                polygamma(nf, -1.3),
                polygamma(nf, -0.3) + step(n, -1.3),
                1e-12,
                &format!("reflect -1.3 n={n}"),
            );
            close(
                polygamma(nf, -0.3),
                polygamma(nf, 0.7) + step(n, -0.3),
                1e-12,
                &format!("reflect -0.3 n={n}"),
            );
        }
    }

    #[test]
    fn polygamma_small_and_large_arguments() {
        // psi(x) ~ -1/x - gamma near 0, so psi(1e-8) = psi(1+1e-8) - 1e8.
        close(polygamma(0.0, 1e-8), -1e8 - GAMMA, 1e-12, "psi(1e-8)");
        close(
            polygamma(1.0, 1e-4),
            1e8 + PI * PI / 6.0,
            1e-8,
            "psi'(1e-4)",
        );
        close(
            polygamma(0.0, 1e6),
            1e6_f64.ln() - 0.5e-6,
            1e-14,
            "psi(1e6)",
        );
        close(
            polygamma(1.0, 1e6),
            1e-6 + 0.5e-12 + 1e-18 / 6.0,
            1e-14,
            "psi'(1e6)",
        );
        assert_eq!(polygamma(0.0, 1e300), 1e300_f64.ln());
        assert_eq!(polygamma(2.0, 1e300), 0.0);
    }

    #[test]
    fn polygamma_cot_polynomials() {
        // Rebuild P_n by the recurrence P_{k+1} = -(1 + c^2) P_k' on coefficient vectors.
        let mut p: Vec<f64> = vec![0.0, 1.0];
        for n in 0..=4u32 {
            for &c in &[-1.7, -0.3, 0.0, 0.4, 2.2] {
                let v = p.iter().rev().fold(0.0, |a, &co| a * c + co);
                let got = cot_poly(n, c);
                assert!((v - got).abs() <= 1e-13 * (1.0 + v.abs()), "P{n}({c})");
            }
            let d: Vec<f64> = p
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, &co)| co * i as f64)
                .collect();
            let mut q = vec![0.0; d.len() + 2];
            for (i, &co) in d.iter().enumerate() {
                q[i] -= co;
                q[i + 2] -= co;
            }
            p = q;
        }
    }

    #[test]
    fn polygamma_specials() {
        for n in 0..=4 {
            let nf = f64::from(n);
            assert!(polygamma(nf, 0.0).is_nan());
            assert!(polygamma(nf, -0.0).is_nan());
            assert!(polygamma(nf, -1.0).is_nan());
            assert!(polygamma(nf, -7.0).is_nan());
            assert!(polygamma(nf, f64::NAN).is_nan());
            assert!(polygamma(nf, f64::NEG_INFINITY).is_nan());
        }
        assert_eq!(polygamma(0.0, f64::INFINITY), f64::INFINITY);
        for n in 1..=4 {
            assert_eq!(polygamma(f64::from(n), f64::INFINITY), 0.0);
        }
        for m in [-1.0, 5.0, 0.5, 1.5, f64::NAN, f64::INFINITY, 1e300] {
            assert!(polygamma(m, 2.0).is_nan(), "m={m}");
        }
    }

    const TAB: [f64; 6] = [3.0, 0.0, 2.0, 0.0, 1.0, 4.0];

    #[test]
    fn table_examples() {
        assert_eq!(table(&TAB, 1.5), 2.5);
        assert_eq!(table(&TAB, -1.0), -1.0);
        assert_eq!(table(&TAB, 10.0), 28.0);
        assert_eq!(table(&TAB, 0.0), 0.0);
        assert_eq!(table(&TAB, 1.0), 1.0);
        assert_eq!(table(&TAB, 2.0), 4.0);
        assert_eq!(table_slope(&TAB, 0.5), 1.0);
        assert_eq!(table_slope(&TAB, 1.0), 3.0);
        assert_eq!(table_slope(&TAB, 1.5), 3.0);
        assert_eq!(table_slope(&TAB, -5.0), 1.0);
        assert_eq!(table_slope(&TAB, 2.0), 3.0);
        assert_eq!(table_slope(&TAB, 99.0), 3.0);
        assert!(table(&TAB, f64::NAN).is_nan());
        assert!(table_slope(&TAB, f64::NAN).is_nan());
    }

    #[test]
    fn table_bad_blocks() {
        assert!(table(&[], 1.0).is_nan());
        assert!(table(&[3.0, 0.0, 2.0, 0.0, 1.0], 1.0).is_nan());
        assert!(table(&[1.0, 0.0, 2.0, 0.0], 1.0).is_nan());
        assert!(table(&[f64::NAN, 0.0, 2.0, 0.0, 1.0], 1.0).is_nan());
        assert!(table_slope(&[2.0, 0.0, 1.0, 0.0], 0.5).is_nan());
        // SAFETY: null is allowed and gives NaN.
        unsafe {
            assert!(pc_math_table(core::ptr::null(), 1.0).is_nan());
            assert!(pc_math_table_slope([1.0].as_ptr(), 1.0).is_nan());
        }
    }

    #[test]
    fn table_extreme_x() {
        assert_eq!(table(&TAB, f64::INFINITY), f64::INFINITY);
        assert_eq!(table(&TAB, f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(table_slope(&TAB, f64::INFINITY), 3.0);
    }

    type P1 = (F1, fn(f64) -> f64);
    type P2 = (F2, fn(f64, f64) -> f64);

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[test]
    fn shims_match_plain_functions_bitwise() {
        let xs = [
            0.0,
            -0.0,
            0.3,
            -0.5,
            1.0,
            2.5,
            -7.25,
            20.0,
            1e10,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let f1: [P1; 18] = [
            (MATH.sin, sin),
            (MATH.cos, cos),
            (MATH.tan, tan),
            (MATH.asin, asin),
            (MATH.acos, acos),
            (MATH.atan, atan),
            (MATH.sinh, sinh),
            (MATH.cosh, cosh),
            (MATH.tanh, tanh),
            (MATH.exp, exp),
            (MATH.ln, ln),
            (MATH.log10, log10),
            (MATH.sqrt, sqrt),
            (MATH.erf, erf),
            (MATH.erfc, erfc),
            (MATH.lgamma, lgamma),
            (MATH.floor, floor),
            (MATH.ceil, ceil),
        ];
        let f2: [P2; 6] = [
            (MATH.atan2, atan2),
            (MATH.pow, pow),
            (MATH.besselj, besselj),
            (MATH.bessely, bessely),
            (MATH.polygamma, polygamma),
            (MATH.pc_mod, pc_mod),
        ];
        for &x in &xs {
            for (s, f) in &f1 {
                // SAFETY: the shims are safe functions behind an unsafe pointer type.
                assert!(same(unsafe { s(x) }, f(x)), "x={x}");
            }
            for &y in &xs {
                for (s, f) in &f2 {
                    // SAFETY: as above.
                    assert!(same(unsafe { s(x, y) }, f(x, y)), "x={x} y={y}");
                }
            }
            // SAFETY: TAB is a valid block.
            unsafe {
                assert!(same((MATH.table)(TAB.as_ptr(), x), table(&TAB, x)));
                assert!(same(
                    (MATH.table_slope)(TAB.as_ptr(), x),
                    table_slope(&TAB, x)
                ));
            }
        }
        for m in 0..=4 {
            // SAFETY: as above.
            assert!(same(
                unsafe { (MATH.polygamma)(f64::from(m), 0.37) },
                polygamma(f64::from(m), 0.37)
            ));
        }
    }

    #[test]
    fn symbols_follow_field_order() {
        let names: Vec<&str> = symbols().iter().map(|s| s.0).collect();
        assert_eq!(
            names,
            [
                "pc_sin",
                "pc_cos",
                "pc_tan",
                "pc_asin",
                "pc_acos",
                "pc_atan",
                "pc_atan2",
                "pc_sinh",
                "pc_cosh",
                "pc_tanh",
                "pc_exp",
                "pc_ln",
                "pc_log10",
                "pc_sqrt",
                "pc_pow",
                "pc_erf",
                "pc_erfc",
                "pc_lgamma",
                "pc_besselj",
                "pc_bessely",
                "pc_polygamma",
                "pc_floor",
                "pc_ceil",
                "pc_mod",
                "pc_table",
                "pc_table_slope",
            ]
        );
        let sym = symbols();
        assert!(sym.iter().all(|s| !s.1.is_null()));
        // Pointers follow the struct layout: read the table as an array of fn pointers.
        assert_eq!(
            core::mem::size_of::<MathTable>(),
            26 * core::mem::size_of::<usize>()
        );
        let base = &MATH as *const MathTable as *const usize;
        for (i, s) in sym.iter().enumerate() {
            // SAFETY: MathTable is repr(C) with 26 pointer-sized fields.
            assert_eq!(unsafe { *base.add(i) }, s.1 as usize, "{}", s.0);
        }
    }
}
