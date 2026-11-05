//! The scalar type of an ensemble run: `f32` or `f64`.
//!
//! Every transcendental goes through the pure Rust `libm` crate (`libm::sinf` for `f32`, the
//! `pc_ir::math` wrappers for `f64`), so a CPU result does not depend on the platform's C
//! library. `sqrt`, `abs`, `floor` and `ceil` are exact in IEEE arithmetic and use the
//! standard library.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait Real:
    Copy
    + Send
    + Sync
    + Debug
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const NAN: Self;
    /// Unit roundoff for the step-too-small test of the adaptive method: 2.3e-16 for `f64`
    /// as in `dopri5`, and 2^-23 for `f32`.
    const UROUND_F64: f64;
    /// Name used in labels and file names: `"single"` or `"double"`.
    const NAME: &'static str;

    /// Rounds to nearest (ties to even).
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_u32(n: u32) -> Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn powf(self, y: Self) -> Self;
    fn fmod(self, y: Self) -> Self;

    /// Smaller of the two; a NaN operand is ignored, as Rust's `min` does. The controller
    /// uses this. Model `min` and `max` are written out in the tape instead, because they
    /// follow `x < y ? x : y`.
    fn fmin(self, y: Self) -> Self;
    /// Larger of the two, ignoring a NaN operand.
    fn fmax(self, y: Self) -> Self;

    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
}

macro_rules! impl_real {
    (
        $t:ident, $name:literal, $uround:expr,
        sin $sin:path, cos $cos:path, tan $tan:path, asin $asin:path, acos $acos:path,
        atan $atan:path, sinh $sinh:path, cosh $cosh:path, tanh $tanh:path, exp $exp:path,
        ln $ln:path, log10 $log10:path, atan2 $atan2:path, pow $pow:path, fmod $fmod:path
    ) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NAN: Self = <$t>::NAN;
            const UROUND_F64: f64 = $uround;
            const NAME: &'static str = $name;

            fn from_f64(x: f64) -> Self {
                x as $t
            }
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            fn from_u32(n: u32) -> Self {
                n as $t
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            fn sin(self) -> Self {
                $sin(self)
            }
            fn cos(self) -> Self {
                $cos(self)
            }
            fn tan(self) -> Self {
                $tan(self)
            }
            fn asin(self) -> Self {
                $asin(self)
            }
            fn acos(self) -> Self {
                $acos(self)
            }
            fn atan(self) -> Self {
                $atan(self)
            }
            fn sinh(self) -> Self {
                $sinh(self)
            }
            fn cosh(self) -> Self {
                $cosh(self)
            }
            fn tanh(self) -> Self {
                $tanh(self)
            }
            fn exp(self) -> Self {
                $exp(self)
            }
            fn ln(self) -> Self {
                $ln(self)
            }
            fn log10(self) -> Self {
                $log10(self)
            }
            fn atan2(self, x: Self) -> Self {
                $atan2(self, x)
            }
            fn powf(self, y: Self) -> Self {
                $pow(self, y)
            }
            fn fmod(self, y: Self) -> Self {
                $fmod(self, y)
            }
            fn fmin(self, y: Self) -> Self {
                <$t>::min(self, y)
            }
            fn fmax(self, y: Self) -> Self {
                <$t>::max(self, y)
            }
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }
        }
    };
}

impl_real!(
    f32, "single", 1.192_092_9e-7,
    sin libm::sinf, cos libm::cosf, tan libm::tanf, asin libm::asinf, acos libm::acosf,
    atan libm::atanf, sinh libm::sinhf, cosh libm::coshf, tanh libm::tanhf, exp libm::expf,
    ln libm::logf, log10 libm::log10f, atan2 libm::atan2f, pow libm::powf, fmod libm::fmodf
);

impl_real!(
    f64, "double", 2.3e-16,
    sin pc_ir::math::sin, cos pc_ir::math::cos, tan pc_ir::math::tan, asin pc_ir::math::asin,
    acos pc_ir::math::acos, atan pc_ir::math::atan, sinh pc_ir::math::sinh,
    cosh pc_ir::math::cosh, tanh pc_ir::math::tanh, exp pc_ir::math::exp,
    ln pc_ir::math::ln, log10 pc_ir::math::log10, atan2 pc_ir::math::atan2,
    pow pc_ir::math::pow, fmod pc_ir::math::fmod
);
