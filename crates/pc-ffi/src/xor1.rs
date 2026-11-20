//! The `xor1` block codec of session files (SPEC section 8af-4).
//!
//! A block is an r-by-c matrix of f64. Each column is coded on its own, top to bottom. Value i
//! of a column is predicted from the two values above it, and the coded residual is the XOR
//! of the bits of the value and of the prediction, so decoding gives the same bits back:
//! NaN payloads, the sign of zero and infinities survive.
//!
//! The values cross this module column-major, the order MATLAB stores a matrix in. `pc-cli`
//! keeps blocks row-major and transposes before and after.

use std::fmt;

/// The header byte of a zero residual: eight leading zero bytes and no middle bytes.
const ZERO: u8 = 0x80;

/// Why a byte string is not the `xor1` coding of an r-by-c block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// r * c overflows.
    Shape,
    /// Fewer bytes than values (each value takes at least one byte), or more than nine per
    /// value.
    Length { values: usize, bytes: usize },
    /// A header byte with L + T > 8, or L + T = 8 other than 0x80, at this offset.
    Header { pos: usize },
    /// A header that is not the shortest coding of its residual: the first or the last
    /// middle byte is zero.
    NotCanonical { pos: usize },
    /// The bytes end inside the value whose header is at this offset.
    Truncated { pos: usize },
    /// Bytes are left after the last value, from this offset.
    Trailing { pos: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "the shape has more values than memory can hold"),
            Self::Length { values, bytes } => {
                write!(f, "{bytes} bytes cannot code {values} values")
            }
            Self::Header { pos } => write!(f, "invalid header byte at offset {pos}"),
            Self::NotCanonical { pos } => {
                write!(f, "the value at offset {pos} is not in its shortest form")
            }
            Self::Truncated { pos } => write!(f, "the bytes end inside the value at offset {pos}"),
            Self::Trailing { pos } => {
                write!(f, "extra bytes after the last value, at offset {pos}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// The bits of the prediction of value i of a column, from the values `a` = v(i-1) and
/// `b` = v(i-2) above it. For i >= 2 the prediction is 2 a - b in f64, without `mul_add`.
/// A NaN result is replaced by the bits of `a`, because the sign and payload of a NaN that
/// arithmetic makes differ between processors and the file must decode the same everywhere.
#[inline]
fn predict(i: usize, a: f64, b: f64) -> u64 {
    match i {
        0 => 0,
        1 => a.to_bits(),
        _ => {
            let p = 2.0 * a - b;
            if p.is_nan() { a.to_bits() } else { p.to_bits() }
        }
    }
}

/// Appends the coding of one residual.
#[inline]
fn put(r: u64, out: &mut Vec<u8>) {
    if r == 0 {
        out.push(ZERO);
        return;
    }
    let lead = (r.leading_zeros() / 8) as usize;
    let trail = (r.trailing_zeros() / 8) as usize;
    out.push((16 * lead + trail) as u8);
    out.extend_from_slice(&r.to_be_bytes()[lead..8 - trail]);
}

/// The most bytes `encode` can write for `values` values: nine each.
pub fn max_encoded_len(values: usize) -> Option<usize> {
    values.checked_mul(9)
}

/// The `xor1` bytes of the r-by-c block `values`, given column-major.
///
/// # Panics
/// If `values.len()` is not r * c.
pub fn encode(rows: usize, cols: usize, values: &[f64]) -> Vec<u8> {
    assert_eq!(
        Some(values.len()),
        rows.checked_mul(cols),
        "xor1::encode: shape"
    );
    let mut out = Vec::with_capacity(values.len() + values.len() / 2);
    for col in values.chunks_exact(rows.max(1)).take(cols) {
        let (mut a, mut b) = (0.0f64, 0.0f64);
        for (i, &v) in col.iter().enumerate() {
            put(v.to_bits() ^ predict(i, a, b), &mut out);
            b = a;
            a = v;
        }
    }
    out
}

/// Decodes `bytes` into `out`, the r-by-c block column-major. `out` must hold r * c values.
pub fn decode_into(
    rows: usize,
    cols: usize,
    bytes: &[u8],
    out: &mut [f64],
) -> Result<(), DecodeError> {
    let n = rows.checked_mul(cols).ok_or(DecodeError::Shape)?;
    if out.len() != n {
        return Err(DecodeError::Length {
            values: n,
            bytes: bytes.len(),
        });
    }
    if bytes.len() < n || max_encoded_len(n).is_some_and(|m| bytes.len() > m) {
        return Err(DecodeError::Length {
            values: n,
            bytes: bytes.len(),
        });
    }
    let mut pos = 0usize;
    for col in out.chunks_exact_mut(rows.max(1)).take(cols) {
        let (mut a, mut b) = (0.0f64, 0.0f64);
        for (i, slot) in col.iter_mut().enumerate() {
            let at = pos;
            let h = *bytes.get(pos).ok_or(DecodeError::Truncated { pos: at })?;
            pos += 1;
            let r = if h == ZERO {
                0
            } else {
                let (lead, trail) = ((h >> 4) as usize, (h & 15) as usize);
                if lead + trail >= 8 {
                    return Err(DecodeError::Header { pos: at });
                }
                let len = 8 - lead - trail;
                let mid = bytes
                    .get(pos..pos + len)
                    .ok_or(DecodeError::Truncated { pos: at })?;
                if mid[0] == 0 || mid[len - 1] == 0 {
                    return Err(DecodeError::NotCanonical { pos: at });
                }
                pos += len;
                let mut be = [0u8; 8];
                be[lead..8 - trail].copy_from_slice(mid);
                u64::from_be_bytes(be)
            };
            let v = f64::from_bits(r ^ predict(i, a, b));
            *slot = v;
            b = a;
            a = v;
        }
    }
    if pos != bytes.len() {
        return Err(DecodeError::Trailing { pos });
    }
    Ok(())
}

/// The r-by-c block, column-major, that `bytes` codes.
pub fn decode(rows: usize, cols: usize, bytes: &[u8]) -> Result<Vec<f64>, DecodeError> {
    let n = rows.checked_mul(cols).ok_or(DecodeError::Shape)?;
    // Each value takes at least one byte, so a short string is refused before the
    // allocation a huge shape would ask for.
    if bytes.len() < n {
        return Err(DecodeError::Length {
            values: n,
            bytes: bytes.len(),
        });
    }
    let mut out = vec![0.0; n];
    decode_into(rows, cols, bytes, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, for repeatable random bits without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn round_trip(rows: usize, cols: usize, v: &[f64]) -> Vec<u8> {
        let b = encode(rows, cols, v);
        let back = decode(rows, cols, &b).expect("decode");
        assert_eq!(bits(&back), bits(v), "round trip {rows}x{cols}");
        b
    }

    #[test]
    fn golden_linear_column() {
        // 1, 2, 3: p = 0, then 1, then 2*2-1 = 3 exactly, so the third residual is zero.
        let b = encode(3, 1, &[1.0, 2.0, 3.0]);
        // 1.0 = 0x3FF0_0000_0000_0000 against 0: L = 0, T = 6, middle 3F F0.
        // 2.0 = 0x4000_0000_0000_0000 against 1.0: r = 0x7FF0_0000_0000_0000, L = 0, T = 6.
        assert_eq!(b, vec![0x06, 0x3f, 0xf0, 0x06, 0x7f, 0xf0, 0x80]);
    }

    #[test]
    fn golden_infinities_and_nan_prediction() {
        // Inf, Inf, 1: p(1) = Inf, so r = 0; p(2) = 2 Inf - Inf = NaN, replaced by bits(Inf).
        let inf = f64::INFINITY;
        let b = encode(3, 1, &[inf, inf, 1.0]);
        // Inf = 0x7FF0..; 1.0 XOR Inf = 0x3FF0.. ^ 0x7FF0.. = 0x4000_0000_0000_0000: L 0, T 7.
        assert_eq!(b, vec![0x06, 0x7f, 0xf0, 0x80, 0x07, 0x40]);
    }

    #[test]
    fn golden_nan_payload_column() {
        // A NaN with a payload, then the same NaN: p(1) = v(0) bit for bit, r = 0.
        let nan = f64::from_bits(0x7ff8_0000_dead_beef);
        let b = encode(2, 1, &[nan, nan]);
        assert_eq!(
            b,
            vec![0x00, 0x7f, 0xf8, 0x00, 0x00, 0xde, 0xad, 0xbe, 0xef, 0x80]
        );
        // A third NaN: 2 NaN - NaN is NaN, so p = bits(v(1)) and r = 0 again.
        let b = encode(3, 1, &[nan, nan, nan]);
        assert_eq!(&b[9..], &[0x80, 0x80]);
    }

    #[test]
    fn golden_signed_zero_and_columns_apart() {
        // Two columns of a 2-by-2 block: [0 -0; 0 -0] column-major is 0, 0, -0, -0.
        let b = encode(2, 2, &[0.0, 0.0, -0.0, -0.0]);
        // Column 2 starts again from p = 0: -0 = 0x8000.. is L 0, T 7, middle 80.
        assert_eq!(b, vec![0x80, 0x80, 0x07, 0x80, 0x80]);
    }

    #[test]
    fn special_values_round_trip() {
        let specials = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff4_0000_0000_1234),
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            -f64::from_bits(3),
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            1.0,
            -1.0,
        ];
        // Every ordered triple, so every prediction case meets every pair above it.
        let mut v = Vec::new();
        for &a in &specials {
            for &b in &specials {
                for &c in &specials {
                    v.extend_from_slice(&[a, b, c]);
                }
            }
        }
        let cols = v.len() / 3;
        round_trip(3, cols, &v);
        round_trip(v.len(), 1, &v);
    }

    #[test]
    fn random_bits_round_trip() {
        let mut rng = Rng(42);
        for (rows, cols) in [(1, 1), (1, 7), (7, 1), (13, 5), (1000, 3)] {
            let v: Vec<f64> = (0..rows * cols)
                .map(|_| f64::from_bits(rng.next()))
                .collect();
            let b = round_trip(rows, cols, &v);
            assert!(b.len() <= max_encoded_len(v.len()).unwrap());
        }
    }

    #[test]
    fn constant_linear_and_smooth_columns_shrink() {
        let n = 1000;
        let constant = vec![std::f64::consts::PI; n];
        let linear: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * i as f64).collect();
        let smooth: Vec<f64> = (0..n).map(|i| (i as f64 * 1e-3).sin()).collect();
        for v in [&constant, &linear, &smooth] {
            let b = round_trip(n, 1, v);
            assert!(b.len() < 8 * n, "{} bytes for {n} values", b.len());
        }
        // pi takes nine bytes, then every prediction is exact.
        assert_eq!(round_trip(n, 1, &constant).len(), 9 + (n - 1));
        assert_eq!(
            round_trip(n, 1, &linear)
                .iter()
                .filter(|&&x| x == ZERO)
                .count(),
            n - 2
        );
    }

    #[test]
    fn empty_blocks() {
        for (r, c) in [(0, 0), (0, 3), (3, 0)] {
            assert!(encode(r, c, &[]).is_empty());
            assert!(decode(r, c, &[]).unwrap().is_empty());
        }
        assert_eq!(
            decode(0, 3, &[0x80]),
            Err(DecodeError::Length {
                values: 0,
                bytes: 1
            })
        );
    }

    #[test]
    fn malformed_input_is_refused() {
        // Too few bytes for the values, before any allocation.
        assert_eq!(
            decode(usize::MAX / 2, 1, &[0x80]),
            Err(DecodeError::Length {
                values: usize::MAX / 2,
                bytes: 1
            })
        );
        assert_eq!(decode(usize::MAX, 2, &[]), Err(DecodeError::Shape));
        // L + T > 8, and L + T = 8 other than 0x80.
        assert_eq!(decode(1, 1, &[0x90]), Err(DecodeError::Header { pos: 0 }));
        assert_eq!(decode(1, 1, &[0x08]), Err(DecodeError::Header { pos: 0 }));
        assert_eq!(decode(1, 1, &[0x44]), Err(DecodeError::Header { pos: 0 }));
        assert_eq!(decode(1, 1, &[0x0f]), Err(DecodeError::Header { pos: 0 }));
        // The bytes end inside a value.
        assert_eq!(
            decode(2, 1, &[0x80, 0x06, 0x3f]),
            Err(DecodeError::Truncated { pos: 1 })
        );
        // Bytes left over.
        assert_eq!(
            decode(1, 1, &[0x80, 0x80]),
            Err(DecodeError::Trailing { pos: 1 })
        );
        // A middle byte of zero at either end is not the shortest form.
        assert_eq!(
            decode(1, 1, &[0x06, 0x00, 0xf0]),
            Err(DecodeError::NotCanonical { pos: 0 })
        );
        assert_eq!(
            decode(1, 1, &[0x06, 0x3f, 0x00]),
            Err(DecodeError::NotCanonical { pos: 0 })
        );
        // More than nine bytes per value.
        assert!(matches!(
            decode(1, 1, &[0u8; 10]),
            Err(DecodeError::Length { .. })
        ));
    }

    #[test]
    fn every_header_decodes_or_is_refused_without_panic() {
        let mut rng = Rng(7);
        for h in 0..=255u8 {
            for _ in 0..8 {
                let mut b = vec![h];
                b.extend((0..8).map(|_| rng.next() as u8));
                for len in 1..=b.len() {
                    if let Ok(v) = decode(1, 1, &b[..len]) {
                        assert_eq!(encode(1, 1, &v), &b[..len], "decode then encode");
                    }
                }
            }
        }
    }
}
