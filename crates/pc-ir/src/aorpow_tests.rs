//! Tests for the `pow` port: the C99 special values, a bit-pinned sweep of each build, and,
//! on Linux x86_64 with glibc, a direct comparison with the platform `pow`.
//!
//! The pinned hashes came from glibc, not from the port. On 2026-11-27,
//! `print_platform_pow_hashes` printed the hash of the `pow` in glibc 2.44 (Fedora 45) over
//! the sweep on an FMA CPU (`FUSED_HASH`), and again with
//! `GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2`, which makes glibc pick its unfused build
//! (`PLAIN_HASH`).

use super::{pow, pow_fused, pow_plain};

const PLAIN_HASH: u64 = 0x0c89_4b4a_3dcd_faec;
const FUSED_HASH: u64 = 0x663a_d78e_70aa_f6c7;

/// Inputs per sweep class in the tests that run by default.
const N: usize = 1 << 13;

// ---- sweep ----

struct Rng(u64);

impl Rng {
    // splitmix64
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Any positive finite double: random biased exponent 0..2046 and random mantissa.
fn x_full(r: &mut Rng) -> f64 {
    let e = r.next() % 2047;
    let m = r.next() & ((1u64 << 52) - 1);
    f64::from_bits((e << 52) | m)
}

/// A step error or mesh ratio: log-uniform in [2^-54, 2^21).
fn x_err(r: &mut Rng) -> f64 {
    let e = 1023 - 54 + r.next() % 75;
    let m = r.next() & ((1u64 << 52) - 1);
    f64::from_bits((e << 52) | m)
}

/// Any bit pattern, so zeros, subnormals, infinities and NaNs of both signs occur.
fn x_bits(r: &mut Rng) -> f64 {
    f64::from_bits(r.next())
}

/// Just above 2, so `x^y` is close to `2^y` and `y` sets the binade of the result.
fn x_near2(r: &mut Rng) -> f64 {
    f64::from_bits(0x4000_0000_0000_0000 | (r.next() & ((1u64 << 42) - 1)))
}

type Gen = Box<dyn Fn(&mut Rng) -> (f64, f64)>;

struct Class {
    name: String,
    count: usize,
    inputs: Gen,
}

fn class(name: impl Into<String>, count: usize, inputs: Gen) -> Class {
    Class {
        name: name.into(),
        count,
        inputs,
    }
}

fn classes(n: usize) -> Vec<Class> {
    // The exponents the engine uses: step-size control in `ivp` and mesh grading in `mesh`.
    let mut ys = vec![("1/5".to_string(), 0.2), ("BETA".to_string(), 0.04)];
    ys.push(("expo1".to_string(), 0.2 - 0.04 * 0.75));
    for m in 1..=8 {
        ys.push((format!("1/(m+1) m={m}"), 1.0 / (f64::from(m) + 1.0)));
    }
    let mut out = Vec::new();
    for (name, y) in ys {
        out.push(class(
            format!("{name} full"),
            n,
            Box::new(move |r| (x_full(r), y)),
        ));
        out.push(class(
            format!("{name} err"),
            n,
            Box::new(move |r| (x_err(r), y)),
        ));
    }
    out.push(class(
        "generic",
        4 * n,
        Box::new(|r| (x_full(r), -40.0 + 80.0 * r.unit())),
    ));
    out.push(class(
        "near 1",
        4 * n,
        Box::new(|r| (1.0 + (r.unit() - 0.5) * 1e-3, (r.unit() - 0.5) * 2e5)),
    ));
    out.push(class(
        "integer y, x<0",
        n,
        Box::new(|r| (-x_err(r), (r.next() % 41) as f64 - 20.0)),
    ));
    out.push(class(
        "subnormal result",
        n,
        Box::new(|r| (x_near2(r), -1090.0 + 100.0 * r.unit())),
    ));
    out.push(class(
        "overflow edge",
        n,
        Box::new(|r| (x_near2(r), 990.0 + 40.0 * r.unit())),
    ));
    out.push(class(
        "odd y, x<0, edges",
        n,
        Box::new(|r| (-x_near2(r), (2 * (r.next() % 1050)) as f64 - 1075.0)),
    ));
    out.push(class(
        "raw bits",
        4 * n,
        Box::new(|r| (x_bits(r), x_bits(r))),
    ));
    out.push(class(
        "raw x, small y",
        n,
        Box::new(|r| (x_bits(r), (r.unit() - 0.5) * 8.0)),
    ));
    out
}

/// Calls `f` on every input of every class, in a fixed order.
fn each_input(n: usize, mut f: impl FnMut(usize, f64, f64)) {
    for (i, c) in classes(n).iter().enumerate() {
        let mut r = Rng(0x1234_5678 ^ i as u64);
        for _ in 0..c.count {
            let (x, y) = (c.inputs)(&mut r);
            f(i, x, y);
        }
    }
}

fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(17)
}

/// Per-class hashes of the result bits of `f` over the sweep, then their combined hash.
/// NaN payloads are not compared.
fn hashes(n: usize, f: fn(f64, f64) -> f64) -> (Vec<(String, u64)>, u64) {
    let names: Vec<String> = classes(n).into_iter().map(|c| c.name).collect();
    let mut h = vec![0xcbf2_9ce4_8422_2325u64; names.len()];
    each_input(n, |i, x, y| {
        let v = f(x, y);
        h[i] = mix(
            h[i],
            if v.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                v.to_bits()
            },
        );
    });
    let total = h.iter().fold(0, |t, &v| mix(t, v));
    (names.into_iter().zip(h).collect(), total)
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn fused(x: f64, y: f64) -> f64 {
    pow_fused(x, y).expect("the CPU has FMA and AVX2")
}

fn has_fused() -> bool {
    pow_fused(2.0, 0.5).is_some()
}

fn assert_hash(name: &str, f: fn(f64, f64) -> f64, pinned: u64) {
    let (per_class, total) = hashes(N, f);
    if total != pinned {
        for (class, h) in &per_class {
            eprintln!("{name} {class:<18} {h:016x}");
        }
        panic!("{name}: sweep hash {total:016x}, pinned {pinned:016x}");
    }
}

// ---- special values ----

const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

/// `(x, y, x^y)` from C99 Annex F.9.4.4, plus exact powers.
fn special_cases() -> Vec<(f64, f64, f64)> {
    vec![
        // Zero base.
        (0.0, -3.0, INF),
        (-0.0, -3.0, -INF),
        (-0.0, -2.0, INF),
        (-0.0, -0.5, INF),
        (0.0, -INF, INF),
        (-0.0, -INF, INF),
        (0.0, 3.0, 0.0),
        (-0.0, 3.0, -0.0),
        (-0.0, 2.0, 0.0),
        (-0.0, 0.5, 0.0),
        (-0.0, INF, 0.0),
        // Base 1, base -1 and exponent 0.
        (-1.0, INF, 1.0),
        (-1.0, -INF, 1.0),
        (1.0, NAN, 1.0),
        (1.0, INF, 1.0),
        (1.0, -3.7, 1.0),
        (NAN, 0.0, 1.0),
        (NAN, -0.0, 1.0),
        (-INF, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        // Negative finite base with a non-integer exponent.
        (-2.0, 0.5, NAN),
        (-0.5, 1e-300, NAN),
        // Infinite exponent.
        (0.5, -INF, INF),
        (-0.5, -INF, INF),
        (2.0, -INF, 0.0),
        (-2.0, -INF, 0.0),
        (0.5, INF, 0.0),
        (-0.5, INF, 0.0),
        (2.0, INF, INF),
        (-2.0, INF, INF),
        // Infinite base.
        (-INF, -3.0, -0.0),
        (-INF, -2.0, 0.0),
        (-INF, -0.5, 0.0),
        (-INF, 3.0, -INF),
        (-INF, 2.0, INF),
        (-INF, 0.5, INF),
        (INF, -1.0, 0.0),
        (INF, 1.0, INF),
        (INF, 0.5, INF),
        // NaN in, NaN out, except the cases above.
        (NAN, 1.0, NAN),
        (2.0, NAN, NAN),
        (NAN, NAN, NAN),
        (-1.0, NAN, NAN),
        // Exact results.
        (-3.0, 3.0, -27.0),
        (2.0, 10.0, 1024.0),
        (2.0, -2.0, 0.25),
        (4.0, 0.5, 2.0),
        (-2.0, -3.0, -0.125),
        (10.0, 2.0, 100.0),
        (2.0, 1023.0, f64::from_bits(0x7fe0_0000_0000_0000)),
        (-2.0, 1023.0, -f64::from_bits(0x7fe0_0000_0000_0000)),
        (2.0, -1022.0, f64::MIN_POSITIVE),
        (2.0, -1074.0, f64::from_bits(1)),
        (0.5, 1074.0, f64::from_bits(1)),
        (
            f64::from_bits(1),
            0.5,
            f64::from_bits(0x1e60_0000_0000_0000),
        ),
        (f64::from_bits(1), 1.0, f64::from_bits(1)),
        // Overflow, underflow and exponents too large or too small to matter.
        (2.0, 1024.0, INF),
        (-2.0, 1025.0, -INF),
        (f64::from_bits(1), -1.0, INF),
        (2.0, 1e300, INF),
        (2.0, -1e300, 0.0),
        (0.5, 1e300, 0.0),
        (2.0, 1e-300, 1.0),
    ]
}

fn check_special(name: &str, f: fn(f64, f64) -> f64) {
    let bad: Vec<String> = special_cases()
        .into_iter()
        .filter(|&(x, y, want)| !same(f(x, y), want))
        .map(|(x, y, want)| format!("pow({x:e}, {y:e}) = {:e}, want {want:e}", f(x, y)))
        .collect();
    assert!(bad.is_empty(), "{name}: {bad:#?}");
}

#[test]
fn special_values_follow_c99() {
    check_special("pow_plain", pow_plain);
    check_special("pow", pow);
    if has_fused() {
        check_special("pow_fused", fused);
    }
}

/// The expected values above are the platform's too, so they are not just this port's.
#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
#[test]
fn special_values_match_glibc() {
    check_special("glibc pow", f64::powf);
}

// ---- pinned sweeps ----

#[test]
fn plain_build_gives_the_pinned_bits() {
    assert_hash("pow_plain", pow_plain, PLAIN_HASH);
}

#[test]
fn fused_build_gives_the_pinned_bits() {
    if !has_fused() {
        eprintln!("skipped: this CPU lacks FMA or AVX2");
        return;
    }
    assert_hash("pow_fused", fused, FUSED_HASH);
}

#[test]
fn pow_picks_the_build_for_this_cpu() {
    let pick: fn(f64, f64) -> f64 = if has_fused() { fused } else { pow_plain };
    let mut bad = 0usize;
    each_input(N / 8, |_, x, y| {
        bad += usize::from(!same(pow(x, y), pick(x, y)))
    });
    assert_eq!(bad, 0);
}

// ---- glibc ----

/// Mismatches against glibc's `pow`, per class.
#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
fn glibc_mismatches(n: usize, port: fn(f64, f64) -> f64) {
    let names: Vec<String> = classes(n).into_iter().map(|c| c.name).collect();
    let mut bad = vec![0usize; names.len()];
    let mut shown = Vec::new();
    each_input(n, |i, x, y| {
        let (a, b) = (x.powf(y), port(x, y));
        if !same(a, b) {
            bad[i] += 1;
            if shown.len() < 10 {
                shown.push(format!(
                    "{}: x={x:e} ({:016x}) y={y:e} ({:016x}) glibc {:016x} port {:016x}",
                    names[i],
                    x.to_bits(),
                    y.to_bits(),
                    a.to_bits(),
                    b.to_bits()
                ));
            }
        }
    });
    let counts: Vec<String> = names
        .iter()
        .zip(&bad)
        .filter(|&(_, &k)| k > 0)
        .map(|(name, k)| format!("{name}: {k}"))
        .collect();
    assert!(
        counts.is_empty(),
        "mismatches against glibc {counts:?}, first ones {shown:#?}"
    );
}

/// Linux keeps glibc's `pow`; this checks that the port gives the same bits, so windows-gnu
/// and Linux agree on the same CPU. It fails if a glibc release changes its `pow`.
#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
#[test]
fn matches_glibc_pow() {
    glibc_mismatches(N, pow);
}

#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
#[test]
#[ignore = "40 million calls; run with --release --ignored"]
fn matches_glibc_pow_large() {
    glibc_mismatches(1 << 20, pow);
}

/// The plain build against glibc's unfused build over the large sweep. Run it as
/// `GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2 cargo test --release -p pc-ir --lib
/// plain_build_matches_unfused_glibc -- --ignored`.
#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
#[test]
#[ignore = "needs glibc's FMA path masked off; see the comment"]
fn plain_build_matches_unfused_glibc() {
    let masked = std::env::var("GLIBC_TUNABLES").is_ok_and(|v| v.contains("-FMA"));
    assert!(masked, "set GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2");
    glibc_mismatches(1 << 20, pow_plain);
}

/// Prints the sweep hashes of the platform `pow`, which is how the pins above were made.
#[test]
#[ignore = "prints the pins; run with --ignored --nocapture"]
fn print_platform_pow_hashes() {
    let (per_class, total) = hashes(N, f64::powf);
    for (class, h) in &per_class {
        println!("platform {class:<18} {h:016x}");
    }
    println!("platform TOTAL {total:016x}");
}
