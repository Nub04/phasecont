//! Unit tests for the emitter's private tables and names. The tests that compile the C
//! text live in `tests/aot.rs`.

use crate::c::MATH_FIELDS;
use crate::{CNameError, c_ident, emit_c, emit_c_tagged, fnv1a64};
use pc_ir::EntryKind;
use pc_ir::math::MathTable;

#[path = "../tests/common/lower_text.rs"]
mod lower_text;
use lower_text::build_model;

#[test]
fn math_fields_follow_the_math_table_order() {
    assert_eq!(
        std::mem::size_of::<MathTable>(),
        MATH_FIELDS.len() * std::mem::size_of::<usize>()
    );
    macro_rules! check {
        ($($i:expr => $f:ident),* $(,)?) => {
            $(
                assert_eq!(MATH_FIELDS[$i], stringify!($f));
                assert_eq!(std::mem::offset_of!(MathTable, $f), $i * std::mem::size_of::<usize>());
            )*
        };
    }
    check!(0 => sin, 1 => cos, 2 => tan, 3 => asin, 4 => acos, 5 => atan, 6 => atan2,
        7 => sinh, 8 => cosh, 9 => tanh, 10 => exp, 11 => ln, 12 => log10, 13 => sqrt,
        14 => pow, 15 => erf, 16 => erfc, 17 => lgamma, 18 => besselj, 19 => bessely,
        20 => polygamma, 21 => floor, 22 => ceil, 23 => pc_mod, 24 => table,
        25 => table_slope);
    let ir = build_model("x'=-x\ndone\n");
    let c = emit_c(&ir, &[]).expect("emit");
    let start = c.find("typedef struct pc_math_t").expect("struct");
    let end = c[start..].find("} pc_math_t;").expect("struct end") + start;
    let body = &c[start..end];
    let mut last = 0;
    for f in MATH_FIELDS {
        let pat = format!("(*{f})(");
        let at = body
            .find(&pat)
            .unwrap_or_else(|| panic!("field {f} missing"));
        assert!(at >= last, "field {f} out of order");
        last = at;
    }
    assert_eq!(body.matches("(*").count(), 26);
}

#[test]
fn entry_names_are_sanitized() {
    let mut ir = build_model("x'=-x\ndone\n");
    let e = ir.entry(&EntryKind::F).expect("entry");
    let c = emit_c(&ir, &[("my-f".into(), &e), ("g.2".into(), &e)]).expect("emit");
    assert!(c.contains("void my_f(const double *x"), "{c}");
    assert!(c.contains("void g_2(const double *x"), "{c}");
}

#[test]
fn names_that_collide_after_sanitizing_are_rejected() {
    let mut ir = build_model("x'=-x\ndone\n");
    let e = ir.entry(&EntryKind::F).expect("entry");
    for (a, b) in [("a-b", "a_b"), ("f", "f"), ("a.b", "a b")] {
        let err = emit_c(&ir, &[(a.into(), &e), (b.into(), &e)]).expect_err(a);
        assert_eq!(
            err,
            CNameError::Duplicate {
                first: a.into(),
                second: b.into(),
                ident: c_ident(a),
            }
        );
        assert!(err.to_string().contains(&c_ident(a)), "{err}");
    }
}

#[test]
fn reserved_names_are_rejected() {
    let mut ir = build_model("x'=-x\ndone\n");
    let e = ir.entry(&EntryKind::F).expect("entry");
    for name in [
        "int",
        "double",
        "return",
        "inline",
        "restrict",
        "bool",
        "true",
        "nullptr",
        "typeof",
        "constexpr",
        "_Bool",
        "1st",
        "",
        "_f",
        "__f",
        "pc_f",
        "pc_math",
        "pc_aot_init",
        "pc_aot_tag",
        "PC_EXPORT",
        "uint8_t",
        "int64_t",
        "uintptr_t",
        "INT8_MAX",
        "UINT64_C",
        "SIZE_MAX",
        "PTRDIFF_MIN",
        "WCHAR_MAX",
        "main",
        "linux",
        "unix",
    ] {
        let err = emit_c(&ir, &[(name.into(), &e)]).expect_err(name);
        assert_eq!(
            err,
            CNameError::Reserved {
                name: name.into(),
                ident: c_ident(name),
            }
        );
    }
    // Close to reserved forms but not reserved.
    let ok = [
        "f", "aux", "jac_u", "jac_p", "d_u_u", "integer", "x_t", "uint", "Int8",
    ];
    let named: Vec<(String, &pc_ir::Entry)> = ok.iter().map(|s| (s.to_string(), &e)).collect();
    emit_c(&ir, &named).expect("plain names");
}

/// The text up to the tag function, which is what the tag hashes.
fn before_tag(c: &str) -> &str {
    let at = c
        .find("PC_EXPORT unsigned long long pc_aot_tag(void)")
        .expect("pc_aot_tag is exported");
    &c[..at]
}

#[test]
fn the_tag_hashes_the_text_before_the_tag_function() {
    let mut ir = build_model("par p=2\nx'=-x*p\ndone\n");
    let e = ir.entry(&EntryKind::F).expect("entry");
    let named = [("f".to_string(), &e)];
    let t = emit_c_tagged(&ir, &named).expect("emit");
    assert_eq!(t.tag, fnv1a64(before_tag(&t.text).as_bytes()));
    // The function is the last thing in the file and returns that value.
    let tail = &t.text[t
        .text
        .find("PC_EXPORT unsigned long long pc_aot_tag(void)")
        .unwrap()..];
    assert!(
        tail.contains(&format!("return 0x{:016x}ULL;", t.tag)),
        "{tail}"
    );
    assert_eq!(tail.matches("PC_EXPORT").count(), 1, "{tail}");
    // `emit_c` gives the same text.
    assert_eq!(emit_c(&ir, &named).expect("emit"), t.text);
}

#[test]
fn the_same_model_gives_the_same_tag_and_another_model_another() {
    let tag = |src: &str| {
        let mut ir = build_model(src);
        let e = ir.entry(&EntryKind::F).expect("entry");
        emit_c_tagged(&ir, &[("f".to_string(), &e)])
            .expect("emit")
            .tag
    };
    let a = "par p=2\nx'=-x*p\ndone\n";
    assert_eq!(tag(a), tag(a));
    assert_ne!(tag(a), tag("par p=2\nx'=-x*p+1\ndone\n"));
}

#[test]
fn fnv1a64_matches_the_published_vectors() {
    // Fowler, Noll and Vo, test vectors for FNV-1a 64.
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}
