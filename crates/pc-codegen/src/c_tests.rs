//! Unit tests for the emitter's private tables and names. The tests that compile the C
//! text live in `tests/aot.rs`.

use crate::c::MATH_FIELDS;
use crate::emit_c;
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
    let c = emit_c(&ir, &[]);
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
    let c = emit_c(
        &ir,
        &[("my-f".into(), &e), ("1st".into(), &e), ("".into(), &e)],
    );
    assert!(c.contains("void my_f(const double *x"), "{c}");
    assert!(c.contains("void _1st(const double *x"), "{c}");
    assert!(c.contains("void _(const double *x"), "{c}");
}
