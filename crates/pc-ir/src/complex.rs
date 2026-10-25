//! Complex directions for the multilinear forms.
//!
//! A form `D^k f[v1..vk]` is linear in each direction, so for `v_s = a_s + i b_s` it is
//! the sum over choices of real or imaginary part per slot of `i^m D^k f[c_1..c_k]`, with
//! `m` the number of imaginary choices and `i^2 = -1` fixing the sign.

/// Evaluates the complex form through a real evaluator. `eval(v, out)` receives the packed
/// real directions (slot after slot) and writes `out.len()` values. `dirs_re[s]` and
/// `dirs_im[s]` are the real and imaginary parts of slot `s`; an empty imaginary part
/// means zero. A choice that includes an all-zero part contributes nothing and costs no
/// call, so a real form costs one call and the most is 2^k. Mismatched lengths, or more
/// than 16 slots, fill the outputs with NaN.
pub fn dir_complex(
    mut eval: impl FnMut(&[f64], &mut [f64]),
    dirs_re: &[&[f64]],
    dirs_im: &[&[f64]],
    out_re: &mut [f64],
    out_im: &mut [f64],
) {
    let k = dirs_re.len();
    let bad = k > 16
        || dirs_im.len() != k
        || out_re.len() != out_im.len()
        || dirs_re
            .iter()
            .zip(dirs_im)
            .any(|(r, i)| !i.is_empty() && i.len() != r.len());
    out_re.fill(if bad { f64::NAN } else { 0.0 });
    out_im.fill(if bad { f64::NAN } else { 0.0 });
    if bad {
        return;
    }
    let is_zero = |d: &[f64]| d.iter().all(|&x| x == 0.0);
    let mut packed: Vec<f64> = Vec::new();
    let mut tmp = vec![0.0; out_re.len()];
    for mask in 0u32..(1u32 << k) {
        packed.clear();
        let mut skip = false;
        for (s, (re, im)) in dirs_re.iter().zip(dirs_im).enumerate() {
            let part: &[f64] = if mask >> s & 1 == 1 { im } else { re };
            if part.is_empty() || is_zero(part) {
                skip = true;
                break;
            }
            packed.extend_from_slice(part);
        }
        if skip {
            continue;
        }
        eval(&packed, &mut tmp);
        // i^m: m mod 4 = 0 gives +1, 1 gives +i, 2 gives -1, 3 gives -i.
        let (target, sign) = match mask.count_ones() % 4 {
            0 => (&mut *out_re, 1.0),
            1 => (&mut *out_im, 1.0),
            2 => (&mut *out_re, -1.0),
            _ => (&mut *out_im, -1.0),
        };
        for (o, t) in target.iter_mut().zip(&tmp) {
            *o += sign * t;
        }
    }
}
