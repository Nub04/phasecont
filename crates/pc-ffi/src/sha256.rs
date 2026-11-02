//! SHA-256 as FIPS 180-4 section 6.2 defines it. The session file records the digest of a
//! model's source text, so a replay can tell that it runs the same model. The code is
//! checked against the NIST example messages in the tests below.

/// Round constants, FIPS 180-4 section 4.2.2: the first 32 bits of the fractional parts
/// of the cube roots of the first 64 primes.
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash value, FIPS 180-4 section 5.3.3: the first 32 bits of the fractional parts
/// of the square roots of the first 8 primes.
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// One 64-byte block into the state, FIPS 180-4 section 6.2.2.
fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (t, word) in block.as_chunks::<4>().0.iter().enumerate() {
        w[t] = u32::from_be_bytes(*word);
    }
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = s1
            .wrapping_add(w[t - 7])
            .wrapping_add(s0)
            .wrapping_add(w[t - 16]);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for t in 0..64 {
        let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh
            .wrapping_add(big_s1)
            .wrapping_add(ch)
            .wrapping_add(K[t])
            .wrapping_add(w[t]);
        let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = big_s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (state, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
        *state = state.wrapping_add(v);
    }
}

/// The SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = H0;
    let (blocks, rest) = data.as_chunks::<64>();
    for block in blocks {
        compress(&mut h, block);
    }
    // Padding, FIPS 180-4 section 5.1.1: a 1 bit, zeros up to 56 mod 64, then the message
    // length in bits as a big-endian 64-bit number.
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let len = if rest.len() < 56 { 64 } else { 128 };
    let bits = (data.len() as u64).wrapping_mul(8);
    tail[len - 8..len].copy_from_slice(&bits.to_be_bytes());
    for block in tail[..len].as_chunks::<64>().0 {
        compress(&mut h, block);
    }
    let mut out = [0u8; 32];
    for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        *bytes = word.to_be_bytes();
    }
    out
}

/// The digest as 64 lower-case hex digits.
pub fn sha256_hex(data: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in sha256(data) {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    // The example messages of the NIST "SHA-256 Examples" page and FIPS 180-4 appendix B.
    #[test]
    fn nist_example_messages() {
        let cases: [(&[u8], &str); 4] = [
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            // 448 bits: the padding needs a second block.
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            // 896 bits.
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
            ),
        ];
        for (msg, want) in cases {
            assert_eq!(sha256_hex(msg), want, "{}", String::from_utf8_lossy(msg));
        }
    }

    #[test]
    fn one_million_a() {
        let msg = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&msg),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    // Lengths around the padding boundaries: 55 bytes fit one block with the length, 56
    // and 63 need two, 64 fills a block and needs a second for the padding.
    #[test]
    fn padding_boundaries_agree_with_sha256sum() {
        let want = [
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                63,
                "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
        ];
        for (len, digest) in want {
            let msg = vec![b'a'; len];
            assert_eq!(sha256_hex(&msg), digest, "{len} bytes");
        }
    }
}
