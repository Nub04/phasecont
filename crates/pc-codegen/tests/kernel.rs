//! The kernel text: float literals round trip through a C compiler bit for bit, the prelude
//! compiles in every scalar width, and the right-hand side refuses what no kernel can run.

use std::path::PathBuf;
use std::process::Command;

use libloading::{Library, Symbol};
use pc_codegen::kernel::{Dialect, KernelError, Scalar, emit_rhs, float_literal, prelude};
use pc_ir::dag::{Fn1, Fn2, NodeId, Op};

const FLAGS: [&str; 8] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-std=c99",
    "-Wall",
    "-Wextra",
    "-Werror",
];

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let p =
            std::env::temp_dir().join(format!("pc-kernel-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compiler() -> Option<&'static str> {
    ["gcc", "clang", "cc"].into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// Compiles `c` to a shared library. `None` after a loud skip when there is no compiler.
fn build(tag: &str, c: &str) -> Option<(Dir, Library)> {
    let Some(cc) = compiler() else {
        assert!(
            std::env::var_os("PC_REQUIRE_CC").is_none(),
            "no C compiler found and PC_REQUIRE_CC is set"
        );
        println!("SKIPPED: no C compiler");
        return None;
    };
    let dir = Dir::new(tag);
    let src = dir.0.join("k.c");
    let lib = dir.0.join("k.so");
    std::fs::write(&src, c).unwrap();
    let out = Command::new(cc)
        .args(FLAGS)
        .arg(&src)
        .arg("-o")
        .arg(&lib)
        .arg("-lm")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{cc} failed:\n{}\n--- source ---\n{c}",
        String::from_utf8_lossy(&out.stderr)
    );
    // SAFETY: the library is the one just built from text this test wrote.
    let l = unsafe { Library::new(&lib) }.unwrap();
    Some((dir, l))
}

const F64_CASES: [f64; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    -0.1,
    1.0 / 3.0,
    std::f64::consts::PI,
    2.0,
    0.5,
    1e300,
    -1e300,
    1e-300,
    f64::MAX,
    f64::MIN_POSITIVE,
    -f64::MIN_POSITIVE,
    5e-324,
    1e-310,
    f64::EPSILON,
    1.0 + f64::EPSILON,
    123_456_789.123_456_79,
    0.7,
    1.0 / 5.0,
    -9007199254740993.0,
];

const F32_CASES: [f64; 28] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    -0.1,
    1.0 / 3.0,
    std::f64::consts::PI,
    2.0,
    0.5,
    3.4028235e38,
    -3.4028235e38,
    1e-30,
    f32::MIN_POSITIVE as f64,
    1e-40,
    1.4e-45,
    -1e-40,
    f32::EPSILON as f64,
    1.0 + f32::EPSILON as f64,
    16777217.0,
    0.7,
    1.0 / 5.0,
    // Rounds to infinity, to zero and to a negative zero in single precision.
    1e300,
    -1e300,
    1e-300,
    -1e-300,
    // Halfway between two f32 values: ties to even.
    1.0 + f32::EPSILON as f64 / 2.0,
    1.0 + 1.5 * f32::EPSILON as f64,
];

#[test]
fn double_literals_round_trip_bit_for_bit() {
    let mut c = prelude(Dialect::C, Scalar::F64).unwrap();
    for (i, &v) in F64_CASES.iter().enumerate() {
        c.push_str(&format!(
            "double lit{i}(void) {{ return {}; }}\n",
            float_literal(v, Scalar::F64)
        ));
    }
    let Some((_dir, lib)) = build("f64", &c) else {
        return;
    };
    for (i, &v) in F64_CASES.iter().enumerate() {
        // SAFETY: `lit{i}` is `double lit{i}(void)` in the text built above.
        let f: Symbol<unsafe extern "C" fn() -> f64> =
            unsafe { lib.get(format!("lit{i}").as_bytes()) }.unwrap();
        let got = unsafe { f() };
        assert_eq!(
            got.to_bits(),
            v.to_bits(),
            "{v:e} came back as {got:e} from `{}`",
            float_literal(v, Scalar::F64)
        );
    }
}

#[test]
fn single_literals_round_trip_bit_for_bit() {
    let mut c = prelude(Dialect::C, Scalar::F32).unwrap();
    for (i, &v) in F32_CASES.iter().enumerate() {
        c.push_str(&format!(
            "float lit{i}(void) {{ return {}; }}\n",
            float_literal(v, Scalar::F32)
        ));
    }
    let Some((_dir, lib)) = build("f32", &c) else {
        return;
    };
    for (i, &v) in F32_CASES.iter().enumerate() {
        // SAFETY: `lit{i}` is `float lit{i}(void)` in the text built above.
        let f: Symbol<unsafe extern "C" fn() -> f32> =
            unsafe { lib.get(format!("lit{i}").as_bytes()) }.unwrap();
        let got = unsafe { f() };
        assert_eq!(
            got.to_bits(),
            (v as f32).to_bits(),
            "{v:e} came back as {got:e} from `{}`",
            float_literal(v, Scalar::F32)
        );
    }
}

#[test]
fn infinities_and_nan_use_the_macros() {
    for s in [Scalar::F32, Scalar::F64] {
        assert_eq!(float_literal(f64::INFINITY, s), "PC_INF");
        assert_eq!(float_literal(f64::NEG_INFINITY, s), "(-PC_INF)");
        assert_eq!(float_literal(f64::NAN, s), "PC_NAN");
    }
    let mut c = prelude(Dialect::C, Scalar::F64).unwrap();
    c.push_str("int check(void) { return isinf(PC_INF) && isnan(PC_NAN) && PC_INF > 0; }\n");
    if let Some((_d, lib)) = build("inf", &c) {
        // SAFETY: `check` is `int check(void)` in the text above.
        let f: Symbol<unsafe extern "C" fn() -> i32> = unsafe { lib.get(b"check") }.unwrap();
        assert_eq!(unsafe { f() }, 1);
    }
}

#[test]
fn an_f64_value_that_overflows_f32_becomes_infinity() {
    assert_eq!(float_literal(1e300, Scalar::F32), "PC_INF");
    assert_eq!(float_literal(-1e300, Scalar::F32), "(-PC_INF)");
}

#[test]
fn hex_literals_have_the_expected_spelling() {
    assert_eq!(float_literal(1.0, Scalar::F64), "0x1p+0");
    assert_eq!(float_literal(0.5, Scalar::F64), "0x1p-1");
    assert_eq!(float_literal(1.5, Scalar::F64), "0x1.8p+0");
    assert_eq!(float_literal(-2.0, Scalar::F64), "(-0x1p+1)");
    assert_eq!(float_literal(0.0, Scalar::F64), "0x0p+0");
    assert_eq!(float_literal(1.0, Scalar::F32), "0x1p+0f");
    assert_eq!(float_literal(1.5, Scalar::F32), "0x1.8p+0f");
    assert_eq!(float_literal(0.0, Scalar::F32), "0x0p+0f");
    assert_eq!(
        float_literal(5e-324, Scalar::F64),
        "0x0.0000000000001p-1022"
    );
}

#[test]
fn every_prelude_compiles_as_plain_c_in_both_widths() {
    for s in [Scalar::F32, Scalar::F64] {
        let mut c = prelude(Dialect::C, s).unwrap();
        c.push_str("pc_real use_all(pc_real x) { return pc_sin(x) + pc_sqrt(x) + pc_fabs(x); }\n");
        let _ = build(&format!("prelude-{s:?}"), &c);
    }
}

#[test]
fn metal_has_no_double() {
    assert_eq!(
        prelude(Dialect::Msl, Scalar::F64).unwrap_err(),
        KernelError::NoDouble
    );
    assert!(prelude(Dialect::Msl, Scalar::F32).is_ok());
}

#[test]
fn the_other_dialects_name_their_address_spaces() {
    let cl = prelude(Dialect::OpenCl, Scalar::F64).unwrap();
    assert!(cl.contains("cl_khr_fp64"));
    assert!(cl.contains("__global"));
    assert!(
        !prelude(Dialect::OpenCl, Scalar::F32)
            .unwrap()
            .contains("cl_khr_fp64")
    );
    let cu = prelude(Dialect::Cuda, Scalar::F32).unwrap();
    assert!(cu.contains("__device__") && cu.contains("sinf"));
    let ms = prelude(Dialect::Msl, Scalar::F32).unwrap();
    assert!(ms.contains("metal_stdlib") && ms.contains("device"));
}

fn n(i: u32) -> NodeId {
    NodeId(i)
}

#[test]
fn a_construct_no_kernel_runs_is_named_once() {
    let ops = [
        Op::Var(0),
        Op::Call1(Fn1::Erf, n(0)),
        Op::Call1(Fn1::Erf, n(1)),
        Op::Call2(Fn2::BesselJ, n(0), n(0)),
        Op::Table(0, n(0)),
    ];
    let e = emit_rhs(&ops, &[4], 0, Scalar::F64, "rhs").unwrap_err();
    assert_eq!(
        e,
        KernelError::Unsupported("erf, besselj, table".to_string())
    );
}

#[test]
fn powi_follows_the_documented_multiplication_order() {
    // x^5: m=5 -> acc = x; base = x^2; m=2 -> base = x^4; m=1 -> acc = x * x^4.
    let ops = [Op::Var(0), Op::PowI(n(0), 5)];
    let t = emit_rhs(&ops, &[1], 0, Scalar::F64, "rhs").unwrap();
    assert!(t.contains("v1_0 = v0 * v0;"), "{t}");
    assert!(t.contains("v1_1 = v1_0 * v1_0;"), "{t}");
    assert!(t.contains("v1_2 = v0 * v1_1;"), "{t}");
    assert!(t.contains("v1 = v1_2;"), "{t}");
    // x^-3: m=3 -> acc = x; base = x^2; m=1 -> acc = x * x^2; result 1 / acc.
    let ops = [Op::Var(0), Op::PowI(n(0), -3)];
    let t = emit_rhs(&ops, &[1], 0, Scalar::F64, "rhs").unwrap();
    assert!(t.contains("PC_ONE / v1_1;"), "{t}");
}

#[test]
fn the_time_input_is_the_function_argument() {
    // With n_p = 2, parameter slot 2 is t.
    let ops = [Op::Param(2), Op::Param(1)];
    let t = emit_rhs(&ops, &[0, 1], 2, Scalar::F32, "rhs").unwrap();
    assert!(t.contains("const pc_real v0 = t;"), "{t}");
    assert!(t.contains("const pc_real v1 = p[1];"), "{t}");
    assert!(
        t.contains("out[0] = v0;") && t.contains("out[1] = v1;"),
        "{t}"
    );
}
