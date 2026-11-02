//! Base64 as RFC 4648 section 4 defines it, with padding. The session file stores arrays of
//! f64 as base64 text, and the MATLAB reader and writer use the same alphabet.
//!
//! The decoder is strict. It accepts only the standard alphabet, requires the padding, and
//! rejects non-zero bits in the unused tail of the last character, so one byte string has
//! exactly one text.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Why a text is not base64.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The length is not a multiple of 4.
    Length(usize),
    /// A byte outside the alphabet, at this position.
    Symbol { pos: usize },
    /// `=` anywhere except the last one or two places.
    Padding { pos: usize },
    /// Non-zero bits in the unused tail of the last character.
    TrailingBits,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(n) => write!(f, "length {n} is not a multiple of 4"),
            Self::Symbol { pos } => write!(f, "invalid character at position {pos}"),
            Self::Padding { pos } => write!(f, "misplaced padding at position {pos}"),
            Self::TrailingBits => write!(f, "non-zero bits after the last byte"),
        }
    }
}

/// The base64 text of `data`.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        let sym = |shift: u32| char::from(ALPHABET[(n >> shift & 63) as usize]);
        out.push(sym(18));
        out.push(sym(12));
        out.push(if chunk.len() > 1 { sym(6) } else { '=' });
        out.push(if chunk.len() > 2 { sym(0) } else { '=' });
    }
    out
}

/// The 6-bit value of one alphabet character.
fn value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// The bytes of a base64 text.
pub fn decode(text: &str) -> Result<Vec<u8>, DecodeError> {
    let t = text.as_bytes();
    if !t.len().is_multiple_of(4) {
        return Err(DecodeError::Length(t.len()));
    }
    let mut out = Vec::with_capacity(t.len() / 4 * 3);
    let (all, _) = t.as_chunks::<4>();
    let quads = all.len();
    for (q, quad) in all.iter().enumerate() {
        let last = q + 1 == quads;
        let pad = quad.iter().rev().take_while(|&&c| c == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            let pos = q * 4 + quad.iter().position(|&c| c == b'=').unwrap_or(0);
            return Err(DecodeError::Padding { pos });
        }
        let mut n = 0u32;
        for (i, &c) in quad.iter().enumerate() {
            let v = if i >= 4 - pad {
                0
            } else if c == b'=' {
                return Err(DecodeError::Padding { pos: q * 4 + i });
            } else {
                value(c).ok_or(DecodeError::Symbol { pos: q * 4 + i })?
            };
            n = n << 6 | v;
        }
        // The bits below the last byte must be zero.
        let spare = match pad {
            0 => 0,
            1 => n & 0xff,
            _ => n & 0xffff,
        };
        if spare != 0 {
            return Err(DecodeError::TrailingBits);
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 4648 section 10.
    const VECTORS: [(&str, &str); 7] = [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ];

    #[test]
    fn rfc_4648_vectors() {
        for (plain, text) in VECTORS {
            assert_eq!(encode(plain.as_bytes()), text, "encode {plain:?}");
            assert_eq!(decode(text).unwrap(), plain.as_bytes(), "decode {text:?}");
        }
    }

    #[test]
    fn every_byte_value_and_length_round_trips() {
        // A fixed sequence that visits all 256 values, up to 300 bytes long.
        let data: Vec<u8> = (0..300u32).map(|i| (i * 167 % 256) as u8).collect();
        for len in 0..=data.len() {
            let text = encode(&data[..len]);
            assert_eq!(text.len(), len.div_ceil(3) * 4);
            assert_eq!(decode(&text).unwrap(), &data[..len], "length {len}");
        }
    }

    #[test]
    fn the_two_special_symbols_are_plus_and_slash() {
        assert_eq!(encode(&[0xfb, 0xff, 0xbf]), "+/+/");
        assert_eq!(decode("+/+/").unwrap(), [0xfb, 0xff, 0xbf]);
    }

    #[test]
    fn bad_texts_are_errors() {
        assert_eq!(decode("Zg="), Err(DecodeError::Length(3)));
        assert_eq!(decode("Zm9"), Err(DecodeError::Length(3)));
        assert_eq!(decode("Zm9v\n"), Err(DecodeError::Length(5)));
        assert_eq!(decode("Zm 9"), Err(DecodeError::Symbol { pos: 2 }));
        assert_eq!(decode("Zm9-"), Err(DecodeError::Symbol { pos: 3 }));
        assert_eq!(decode("Z==="), Err(DecodeError::Padding { pos: 1 }));
        assert_eq!(decode("Zg==Zm9v"), Err(DecodeError::Padding { pos: 2 }));
        assert_eq!(decode("Zm=v"), Err(DecodeError::Padding { pos: 2 }));
        // "Zh==" has a set bit below the one byte it carries: "Zg==" is the only text.
        assert_eq!(decode("Zh=="), Err(DecodeError::TrailingBits));
        assert_eq!(decode("Zm9="), Err(DecodeError::TrailingBits));
    }
}
