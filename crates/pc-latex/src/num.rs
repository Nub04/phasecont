/// Formats `x` with `digits` significant digits, in the style of C's `%.{digits}g`:
/// fixed notation for exponents from -4 up to `digits - 1`, scientific notation outside
/// that range, and no trailing zeros. `-0.0` prints as `0`. The exponent has no `+` and
/// no leading zeros, so `1e-7` and `1.5e20`. `digits` must be 1 to 17 and `x` finite;
/// the callers check both.
pub fn format_significant(x: f64, digits: usize) -> String {
    debug_assert!((1..=17).contains(&digits) && x.is_finite());
    if x == 0.0 {
        return "0".to_string();
    }
    // `{:e}` rounds correctly and gives `d.ddde<exp>`.
    let sci = format!("{:.*e}", digits - 1, x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    let digit_str: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let digits_i = digits as i32;
    if exp < -4 || exp >= digits_i {
        let (head, tail) = digit_str.split_at(1);
        let tail = tail.trim_end_matches('0');
        let dot = if tail.is_empty() { "" } else { "." };
        format!("{sign}{head}{dot}{tail}e{exp}")
    } else if exp >= 0 {
        let (int, frac) = digit_str.split_at(exp as usize + 1);
        let frac = frac.trim_end_matches('0');
        let dot = if frac.is_empty() { "" } else { "." };
        format!("{sign}{int}{dot}{frac}")
    } else {
        let zeros = "0".repeat((-exp - 1) as usize);
        let frac = format!("{zeros}{digit_str}");
        let frac = frac.trim_end_matches('0');
        format!("{sign}0.{frac}")
    }
}

/// `format_significant` for a table cell: a TeX math expression. Scientific notation
/// becomes `m\times10^{e}`.
pub(crate) fn format_tex_math(x: f64, digits: usize) -> String {
    let s = format_significant(x, digits);
    match s.split_once('e') {
        None => format!("${s}$"),
        Some((m, e)) => format!("${m}\\times10^{{{e}}}$"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn fixed_notation() {
        assert_eq!(format_significant(1.25, 6), "1.25");
        assert_eq!(format_significant(-0.5, 6), "-0.5");
        assert_eq!(format_significant(100.0, 6), "100");
        assert_eq!(format_significant(123456.0, 6), "123456");
        assert_eq!(format_significant(0.0001, 6), "0.0001");
        assert_eq!(format_significant(0.000123456789, 3), "0.000123");
        assert_eq!(format_significant(5.67891, 4), "5.679");
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(format_significant(123456789.0, 6), "1.23457e8");
        assert_eq!(format_significant(1e-7, 6), "1e-7");
        assert_eq!(format_significant(0.00001, 6), "1e-5");
        assert_eq!(format_significant(1.5e20, 3), "1.5e20");
        assert_eq!(format_significant(-2.5e-10, 3), "-2.5e-10");
        assert_eq!(format_significant(100000.0, 5), "1e5");
    }

    #[test]
    fn rounding_carries_into_the_exponent() {
        assert_eq!(format_significant(9.9999, 2), "10");
        assert_eq!(format_significant(999999.5, 6), "1e6");
        assert_eq!(format_significant(0.00009999999, 3), "0.0001");
    }

    #[test]
    fn zero_has_no_sign() {
        assert_eq!(format_significant(0.0, 4), "0");
        assert_eq!(format_significant(-0.0, 4), "0");
    }

    #[test]
    fn one_digit() {
        assert_eq!(format_significant(0.26, 1), "0.3");
        assert_eq!(format_significant(26.0, 1), "3e1");
        assert_eq!(format_significant(7.0, 1), "7");
    }

    #[test]
    fn tex_math() {
        assert_eq!(format_tex_math(-0.5, 5), "$-0.5$");
        assert_eq!(format_tex_math(1.23456e-5, 5), "$1.2346\\times10^{-5}$");
        assert_eq!(format_tex_math(1e5, 5), "$1\\times10^{5}$");
    }

    proptest! {
        /// With 17 digits the text parses back to the same double.
        #[test]
        fn seventeen_digits_round_trip(bits in any::<u64>()) {
            let x = f64::from_bits(bits);
            prop_assume!(x.is_finite());
            let s = format_significant(x, 17);
            let back: f64 = s.parse().unwrap();
            prop_assert_eq!(back, x, "{}", s);
        }

        /// Fewer digits stay within half a unit in the last place of the request.
        #[test]
        fn rounding_error_is_bounded(x in -1e12f64..1e12, digits in 1usize..=12) {
            prop_assume!(x != 0.0);
            let s = format_significant(x, digits);
            let back: f64 = s.parse().unwrap();
            let rel = ((back - x) / x).abs();
            prop_assert!(rel <= 0.5 * 10f64.powi(1 - digits as i32) * (1.0 + 1e-12), "{} {} {}", x, s, rel);
        }
    }
}
