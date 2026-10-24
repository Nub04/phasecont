//! Tests for the `.ode` writer: unit tests for the canonical text and
//! property tests for the round trip `parse(write(m)) == m`, for surface
//! noise, and for parser robustness on arbitrary input.
//!
//! Every property test name contains `round_trip`, so
//! `PROPTEST_CASES=2000 cargo test -p pc-ode --release round_trip` runs
//! them with more cases.

use pc_ode::{
    Assign, BinaryOp, Code, Comment, DUMMY_SPAN, Entry, Expr, FileId, Ident, Index, IndexOp,
    MapLoader, MarkovLine, MarkovRow, Model, OdeForm, RawAssign, Severity, SourceMap, Stmt,
    StmtKind, TokenKind, UnaryOp, Value, format_number, lex, load, parse, write, write_expanded,
};
use proptest::prelude::*;

const FILE: FileId = FileId(0);

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// Parses `src` and fails the test on any error-severity diagnostic.
fn parse_ok(src: &str) -> Model {
    let (model, diags) = parse(src, FILE);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "errors in\n{src}\n{errors:#?}");
    model
}

/// The canonical text of the right side of `x=<src>`.
fn canon(src: &str) -> String {
    let text = write(&parse_ok(&format!("x={src}\n")));
    text.strip_prefix("x=")
        .and_then(|s| s.strip_suffix('\n'))
        .unwrap_or_else(|| panic!("unexpected writer output {text:?}"))
        .to_string()
}

fn ident(text: &str) -> Ident {
    Ident {
        text: text.to_string(),
        key: text.to_ascii_lowercase(),
        index: None,
        span: DUMMY_SPAN,
    }
}

fn indexed(text: &str, index: Index) -> Ident {
    Ident {
        index: Some(index),
        ..ident(text)
    }
}

fn var(op: IndexOp, n: u32) -> Index {
    Index::Var {
        op,
        n,
        span: DUMMY_SPAN,
    }
}

fn range(start: u32, end: u32) -> Index {
    Index::Range {
        start,
        end,
        span: DUMMY_SPAN,
    }
}

fn num(value: f64) -> Expr {
    Expr::Num {
        value,
        text: format_number(value),
        span: DUMMY_SPAN,
    }
}

fn name(text: &str) -> Expr {
    Expr::Name(ident(text))
}

fn bin(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        implicit: false,
        span: DUMMY_SPAN,
    }
}

fn implicit(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        implicit: true,
        span: DUMMY_SPAN,
    }
}

fn unary(op: UnaryOp, operand: Expr) -> Expr {
    Expr::Unary {
        op,
        operand: Box::new(operand),
        span: DUMMY_SPAN,
    }
}

fn stmt(kind: StmtKind) -> Stmt {
    Stmt {
        kind,
        comment: None,
        span: DUMMY_SPAN,
        array_index: None,
        array_template: None,
    }
}

fn model(statements: Vec<Stmt>) -> Model {
    Model {
        statements,
        suppressed: Vec::new(),
    }
}

/// The text of `x=<e>` as the writer gives it, without `x=` and the line end.
fn expr_text(e: Expr) -> String {
    let m = model(vec![stmt(StmtKind::Fixed {
        name: ident("x"),
        rhs: e,
    })]);
    let text = write(&m);
    text.strip_prefix("x=")
        .and_then(|s| s.strip_suffix('\n'))
        .unwrap_or_else(|| panic!("unexpected writer output {text:?}"))
        .to_string()
}

/// Asserts that `m` survives `write` then `parse` (spans stripped) and that
/// the written text parses without errors.
fn assert_round_trip(m: &Model) {
    let text = write(m);
    let back = parse_ok(&text);
    assert_eq!(back.strip_spans(), m.strip_spans(), "text:\n{text}");
}

/// Asserts that every span of every diagnostic, note and fix-it lies in
/// `text` and on char boundaries.
fn assert_spans_in(text: &str, diags: &[pc_ode::Diagnostic]) {
    let ok = |s: pc_ode::Span| {
        s.file == FILE
            && s.start <= s.end
            && s.end <= text.len()
            && text.is_char_boundary(s.start)
            && text.is_char_boundary(s.end)
    };
    for d in diags {
        assert!(
            ok(d.span),
            "diagnostic span {:?} out of bounds: {d:?}",
            d.span
        );
        for (span, _) in &d.notes {
            assert!(ok(*span), "note span {span:?} out of bounds: {d:?}");
        }
        if let Some(fix) = &d.fix_it {
            assert!(
                ok(fix.span),
                "fix-it span {:?} out of bounds: {d:?}",
                fix.span
            );
        }
    }
}

// ---------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------

#[test]
fn format_number_switches_to_e_form_outside_the_plain_range() {
    assert_eq!(format_number(0.0), "0");
    assert_eq!(format_number(1.0), "1");
    assert_eq!(format_number(0.5), "0.5");
    assert_eq!(format_number(2.5), "2.5");
    assert_eq!(format_number(100.0), "100");
    assert_eq!(format_number(1e-5), "0.00001");
    assert_eq!(format_number(1.5e-5), "0.000015");
    assert_eq!(format_number(1e-6), "1e-6");
    assert_eq!(format_number(1.25e-6), "1.25e-6");
    assert_eq!(format_number(1e15), "1000000000000000");
    assert_eq!(format_number(1.5e15), "1500000000000000");
    assert_eq!(format_number(1e16), "1e16");
    assert_eq!(format_number(1.5e16), "1.5e16");
    assert_eq!(format_number(0.1 + 0.2), "0.30000000000000004");
    assert_eq!(format_number(-2.5), "-2.5");
}

#[test]
fn format_number_never_prints_hundreds_of_digits() {
    for v in [
        f64::MAX,
        f64::MIN_POSITIVE,
        5e-324,
        1e300,
        1e-300,
        123_456_789_012_345_680.0,
        0.000_001_234_567_890_123_456_7,
    ] {
        let text = format_number(v);
        assert!(text.len() <= 24, "{v:e} gives {text}");
        assert_eq!(
            text.parse::<f64>().unwrap().to_bits(),
            v.to_bits(),
            "{text}"
        );
    }
    assert_eq!(format_number(f64::MAX), "1.7976931348623157e308");
    assert_eq!(format_number(5e-324), "5e-324");
}

#[test]
fn literal_text_is_kept_as_written() {
    assert_eq!(canon("1.50E+2"), "1.50E+2");
    assert_eq!(canon(".5"), ".5");
    assert_eq!(write(&parse_ok("par a=-1.0e3\n")), "par a=-1.0e3\n");
}

#[test]
fn empty_literal_text_falls_back_to_the_canonical_form() {
    let e = Expr::Num {
        value: 1e20,
        text: String::new(),
        span: DUMMY_SPAN,
    };
    assert_eq!(expr_text(e), "1e20");
    let m = model(vec![stmt(StmtKind::Par {
        entries: vec![Entry {
            name: ident("a"),
            value: Some(Value::Num {
                value: -0.25,
                text: String::new(),
                span: DUMMY_SPAN,
            }),
            span: DUMMY_SPAN,
        }],
    })]);
    assert_eq!(write(&m), "par a=-0.25\n");
}

// ---------------------------------------------------------------------
// Precedence and parentheses (ruling j)
// ---------------------------------------------------------------------

#[test]
fn prefix_minus_and_power() {
    assert_eq!(canon("-x^2"), "-x^2");
    assert_eq!(canon("(-x)^2"), "(-x)^2");
    assert_eq!(canon("2^(-1)"), "2^(-1)");
    // `2^-1` is a misplaced prefix in the parser; built by hand it is
    // written with parentheses.
    let e = bin(BinaryOp::Pow, num(2.0), unary(UnaryOp::Neg, num(1.0)));
    assert_eq!(expr_text(e), "2^(-1)");
}

#[test]
fn power_chains_keep_their_parentheses() {
    // `a^b^c` reads as `(a^b)^c`; the writer adds the parentheses that
    // silence the chain warning.
    assert_eq!(canon("a^b^c"), "(a^b)^c");
    assert_eq!(canon("(a^b)^c"), "(a^b)^c");
    assert_eq!(canon("a^(b^c)"), "a^(b^c)");
    assert_eq!(canon("a**b"), "a**b");
    assert_eq!(canon("a**b^c"), "(a**b)^c");
    let back = parse_ok("x=a**b\n").strip_spans();
    let StmtKind::Fixed { rhs, .. } = &back.statements[0].kind else {
        panic!()
    };
    assert!(matches!(
        rhs,
        Expr::Binary {
            op: BinaryOp::StarStar,
            ..
        }
    ));
}

#[test]
fn left_associative_operators() {
    assert_eq!(canon("a-(b-c)"), "a - (b - c)");
    assert_eq!(canon("(a-b)-c"), "a - b - c");
    assert_eq!(canon("a/(b*c)"), "a/(b*c)");
    assert_eq!(canon("a/b*c"), "a/b*c");
    assert_eq!(canon("a+b*c"), "a + b*c");
    assert_eq!(canon("(a+b)*c"), "(a + b)*c");
}

#[test]
fn implicit_multiplication_after_division_keeps_its_parentheses() {
    assert_eq!(canon("a/(2v)"), "a/(2v)");
    assert_eq!(canon("a/2v"), "a/2v");
}

#[test]
fn comparisons_are_not_associative() {
    assert_eq!(canon("(a<b)<c"), "(a < b) < c");
    assert_eq!(canon("a<(b<c)"), "a < (b < c)");
    assert_eq!(canon("a<=b"), "a <= b");
    assert_eq!(canon("a==b"), "a == b");
    // A comparison inside arithmetic gets the parentheses XPPAUT's reading
    // implies, so the writer adds no precedence warning.
    assert_eq!(canon("2*(x>1)"), "2*(x > 1)");
    assert_eq!(canon("(3>2)+5"), "(3 > 2) + 5");
    assert_eq!(canon("(a*b)<c"), "(a*b) < c");
    assert_eq!(canon("a^b<c"), "a^b < c");
    assert_eq!(canon("a|b<c"), "a|b < c");
}

#[test]
fn logic_and_arithmetic_mixes_get_parentheses() {
    assert_eq!(canon("(a|b)+c"), "(a|b) + c");
    assert_eq!(canon("a|(b+c)"), "a|(b + c)");
    assert_eq!(canon("(a&b)*c"), "(a&b)*c");
    assert_eq!(canon("a&b|c"), "a&b|c");
    assert_eq!(canon("a&(b|c)"), "a&(b|c)");
}

#[test]
fn not_and_minus_inside_binary_operands() {
    assert_eq!(canon("not x"), "not x");
    assert_eq!(canon("not x>1"), "not x > 1");
    assert_eq!(canon("not (a+b)"), "not (a + b)");
    assert_eq!(canon("-(x>2)"), "-(x > 2)");
    assert_eq!(canon("-x*2"), "-x*2");
    assert_eq!(canon("(not 0)+1"), "not 0 + 1");
    assert_eq!(canon("a&(not b)"), "a&(not b)");
    assert_eq!(canon("2*(-3)"), "2*(-3)");
    assert_eq!(canon("x-(-1)"), "x - (-1)");
    assert_eq!(canon("(-x)<2"), "(-x) < 2");
    assert_eq!(canon("sin(-x)+max(-a,not b)"), "sin(-x) + max(-a,not b)");
    assert_eq!(
        canon("if(-x)then(-1)else(not y)"),
        "if(-x)then(-1)else(not y)"
    );
    // Built by hand: a prefix in a later operand is wrapped.
    let e = bin(BinaryOp::Add, name("a"), unary(UnaryOp::Not, name("b")));
    assert_eq!(expr_text(e), "a + (not b)");
    let e = unary(UnaryOp::Neg, unary(UnaryOp::Neg, name("b")));
    assert_eq!(expr_text(e), "-(-b)");
}

#[test]
fn implicit_multiplication_is_written_as_juxtaposition() {
    assert_eq!(canon("2x"), "2x");
    assert_eq!(canon("2(x+1)"), "2(x + 1)");
    assert_eq!(canon("2v^2(1-w)"), "2v^2(1 - w)");
    assert_eq!(canon("2ex"), "2ex");
    // `2e+1` is the literal 20; `2e + 1` is `2*e + 1`.
    assert_eq!(canon("2e+1"), "2e+1");
    assert_eq!(canon("2e + 1"), "2e + 1");
    let e = bin(BinaryOp::Add, implicit(num(2.0), name("e")), num(1.0));
    assert_eq!(expr_text(e.clone()), "2e + 1");
    assert_round_trip(&model(vec![stmt(StmtKind::Fixed {
        name: ident("x"),
        rhs: e,
    })]));
    assert_eq!(canon("2e+x"), "2e + x");
    assert_eq!(canon("2*e1"), "2*e1");
    assert_eq!(canon("2exp(x)"), "2exp(x)");
    assert_eq!(canon("2e-3x"), "2e-3x");
}

#[test]
fn implicit_multiplication_puts_a_merging_right_side_in_parentheses() {
    // `2` then `e1` would re-lex as the number `2e1`; `2(e1)` is still
    // implicit (R24 A1, replacing the brief's `2*e1`).
    let e = implicit(num(2.0), name("e1"));
    assert_eq!(expr_text(e), "2(e1)");
    let e = implicit(num(2.0), name("E5x"));
    assert_eq!(expr_text(e), "2(E5x)");
    // The right side does not start with a name or `(`.
    let e = implicit(num(2.0), num(3.0));
    assert_eq!(expr_text(e), "2(3)");
    let e = implicit(num(2.0), num(0.5));
    assert_eq!(expr_text(e), "2(0.5)");
    let e = implicit(num(2.0), Expr::Index(var(IndexOp::None, 0)));
    assert_eq!(expr_text(e), "2([j])");
    // A literal of zeros then `x` and a hex digit lexes as a malformed
    // hex number.
    let e = implicit(num(0.0), name("x1"));
    assert_eq!(expr_text(e), "0(x1)");
    let zeros = Expr::Num {
        value: 0.0,
        text: "00".to_string(),
        span: DUMMY_SPAN,
    };
    let e = implicit(zeros, name("XF"));
    assert_eq!(expr_text(e), "00(XF)");
    // Not the hex trap: a dot, a non-zero digit or no hex digit after `x`.
    assert_eq!(canon("0.x1"), "0.x1");
    assert_eq!(canon("10x1"), "10x1");
    assert_eq!(canon("0xg"), "0xg");
}

#[test]
fn implicit_multiplication_falls_back_to_star() {
    // The left side does not end with a literal (hand-built only).
    let e = implicit(name("a"), name("b"));
    assert_eq!(expr_text(e), "a*b");
    let e = implicit(name("x1"), name("b"));
    assert_eq!(expr_text(e), "x1*b");
    let e = implicit(bin(BinaryOp::Add, name("a"), num(2.0)), name("b"));
    assert_eq!(expr_text(e), "(a + 2)*b");
}

/// Asserts that `src` (the right side of `x=`) parses without errors, is
/// written as `want`, and that `want` parses back to the same tree.
fn assert_canon_round_trip(src: &str, want: &str) {
    let m = parse_ok(&format!("x={src}\n"));
    assert_eq!(canon(src), want, "{src}");
    assert_round_trip(&m);
}

#[test]
fn implicit_flag_survives_every_parser_input() {
    // Review 1, finding 1 (hex trap).
    for (src, want) in [
        ("0(x1)", "0(x1)"),
        ("0(xb2)", "0(xb2)"),
        ("00(XF)", "00(XF)"),
        ("p+0(xff)", "p + 0(xff)"),
    ] {
        assert_canon_round_trip(src, want);
    }
    // Review 1, finding 3 (right side that would merge or is no name).
    for (src, want) in [("2(3)", "2(3)"), ("2([j])", "2([j])"), ("2(e1)", "2(e1)")] {
        assert_canon_round_trip(src, want);
    }
    // Only the first operand of such a right side gets the parentheses, so
    // a literal at its end can start the next juxtaposition (found by the
    // widened generator).
    for (src, want) in [
        ("0(0)<0(e1)", "0(0) < 0(e1)"),
        ("2(3)<4x", "2(3) < 4x"),
        ("2(e1)^2", "2(e1)^2"),
        ("2(e1^2)", "2(e1)^2"),
    ] {
        assert_canon_round_trip(src, want);
    }
    // Review 1, finding 4, and review 0, finding 1: a comparison, a logical
    // operator or a prefix on the left keeps the juxtaposition.
    for (src, want) in [
        ("a<2x", "a < 2x"),
        ("-a<2x", "-a < 2x"),
        ("a&b<2x", "a&b < 2x"),
        ("a>0.5x", "a > 0.5x"),
        ("a&1x", "a&1x"),
        ("a/b<2x", "a/b < 2x"),
        ("a+(-2x)", "a + (-2x)"),
        ("a-(-2x)", "a - (-2x)"),
        ("a+(not 2x)", "a + (not 2x)"),
        ("a*(-2x)", "a*(-2x)"),
        ("2b<2c<2d", "2b < 2c < 2d"),
    ] {
        assert_canon_round_trip(src, want);
    }
    // Review 0, finding 4: a prefix on the left of a later operand.
    for (src, want) in [
        ("a+(-b)*c", "a + (-b*c)"),
        ("a-(not b)*c", "a - (not b*c)"),
        ("(-b)*c", "-b*c"),
    ] {
        assert_canon_round_trip(src, want);
    }
}

#[test]
fn implicit_flag_survives_a_long_chain_quickly() {
    // Each link has a comparison on the right, so a writer that tried both
    // parenthesizations at every level would take exponential time.
    let src = vec!["2b<2"; 90].join("c<");
    let m = parse_ok(&format!("x={src}\n"));
    assert_round_trip(&m);
}

#[test]
fn expanded_implicit_multiplication_round_trips() {
    // Review 0, finding 2, and review 1, findings 1 and 3.
    for (template, want) in [
        ("x[0..2]'=0x[j]\n", "x0'=0(x0)\nx1'=0(x1)\nx2'=0(x2)\n"),
        (
            "x[10..12]'=0x[j]\n",
            "x10'=0(x10)\nx11'=0(x11)\nx12'=0(x12)\n",
        ),
        ("x[0..2]'=2([j])\n", "x0'=2(0)\nx1'=2(1)\nx2'=2(2)\n"),
        ("e[0..2]'=2e[j]\n", "e0'=2(e0)\ne1'=2(e1)\ne2'=2(e2)\n"),
    ] {
        let mut map = SourceMap::new();
        let top = map.add("/m/top.ode", template);
        let mut loader = MapLoader::new();
        let loaded = load(top, &mut map, &mut loader);
        let errors: Vec<_> = loaded
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "{template}: {errors:#?}");
        let mut expanded = loaded.expanded.clone();
        expanded.suppressed.clear();
        for s in &mut expanded.statements {
            s.array_index = None;
            s.array_template = None;
        }
        let text = write_expanded(&expanded);
        assert_eq!(text, want, "{template}");
        let back = parse_ok(&text);
        assert_eq!(back.strip_spans(), expanded.strip_spans(), "{template}");
    }
}

#[test]
fn a_leading_digit_gets_a_blank_line() {
    // Review 1, finding 2: `0=x-y` on the first line would mark an
    // old-format file.
    for src in ["\n0=x-y\n", "   \n0=x-y\n"] {
        let m = parse_ok(src);
        let text = write(&m);
        assert_eq!(text, "\n0=x-y\n", "{src:?}");
        assert_eq!(write_expanded(&m), text);
        assert_round_trip(&m);
    }
}

#[test]
fn a_trailing_backslash_gets_a_blank() {
    // Review 1, finding 5: raw text that ends in `\` must not continue the
    // line.
    for (src, want) in [
        ("@ total=10\\ \n", "@ total=10\\ \n"),
        ("@ total=x\\ \nx'=1\n", "@ total=x\\ \nx'=1\n"),
        ("bndry x\\ \ny'=1\n", "bndry x\\ \ny'=1\n"),
        ("bndry x\\ ", "bndry x\\ \n"),
        ("\"hello\\ \nx'=1\n", "\"hello\\ \nx'=1\n"),
    ] {
        let (m, diags) = parse(src, FILE);
        assert!(
            diags.iter().all(|d| d.severity != Severity::Error),
            "{src:?}: {diags:#?}"
        );
        let text = write(&m);
        assert_eq!(text, want, "{src:?}");
        let (back, _) = parse(&text, FILE);
        assert_eq!(back.strip_spans(), m.strip_spans(), "{src:?}");
    }
    // A `#` comment never continues, so it keeps its text as it is. An
    // `#include` line is a comment token too.
    let src = "# ends in \\\npar a=1 # also \\\n#include foo\\\nx'=1\n";
    let m = parse_ok(src);
    assert_eq!(m.statements.len(), 4);
    assert_eq!(write(&m), src);
    assert_round_trip(&m);
}

#[test]
fn negative_literals_never_give_a_parse_error() {
    // Hand-built only: the parser never gives a negative `Expr::Num`.
    let neg = |v: f64| Expr::Num {
        value: v,
        text: String::new(),
        span: DUMMY_SPAN,
    };
    let cases = [
        (neg(-5.0), "-5"),
        (bin(BinaryOp::Pow, num(2.0), neg(-5.0)), "2^(-5)"),
        (bin(BinaryOp::Pow, neg(-5.0), num(2.0)), "(-5)^2"),
        (bin(BinaryOp::Add, name("a"), neg(-5.0)), "a + (-5)"),
        (bin(BinaryOp::Lt, neg(-5.0), name("a")), "(-5) < a"),
        (unary(UnaryOp::Neg, neg(-5.0)), "-(-5)"),
        (implicit(num(2.0), neg(-5.0)), "2(-5)"),
        (
            bin(BinaryOp::Add, name("a"), implicit(neg(-0.0), name("x1"))),
            "a + (-0(x1))",
        ),
    ];
    for (e, want) in cases {
        let text = expr_text(e);
        assert_eq!(text, want);
        parse_ok(&format!("x={text}\n"));
    }
}

#[test]
fn names_and_index_forms() {
    assert_eq!(
        canon("x[j]+y[j+1]-z[j-2]+w[j*3]"),
        "x[j] + y[j+1] - z[j-2] + w[j*3]"
    );
    assert_eq!(canon("[j]+[j+1]"), "[j] + [j+1]");
    assert_eq!(canon("Foo+BAR"), "Foo + BAR");
    let e = Expr::Name(Ident {
        text: String::new(),
        key: "k".to_string(),
        index: None,
        span: DUMMY_SPAN,
    });
    assert_eq!(expr_text(e), "k");
}

#[test]
fn error_nodes_write_text_that_reparses_with_an_error() {
    let e = bin(BinaryOp::Add, name("a"), Expr::Error { span: DUMMY_SPAN });
    let m = model(vec![stmt(StmtKind::Fixed {
        name: ident("x"),
        rhs: e,
    })]);
    let text = write(&m);
    let (_, diags) = parse(&text, FILE);
    assert!(
        diags.iter().any(|d| d.severity == Severity::Error),
        "{text}"
    );

    let m = model(vec![stmt(StmtKind::InitValue {
        name: ident("x"),
        value: Value::Error { span: DUMMY_SPAN },
    })]);
    let text = write(&m);
    let (_, diags) = parse(&text, FILE);
    assert!(
        diags.iter().any(|d| d.severity == Severity::Error),
        "{text}"
    );

    let text = format!("x={}\n", format_number(f64::INFINITY));
    let (_, diags) = parse(&text, FILE);
    assert!(
        diags.iter().any(|d| d.severity == Severity::Error),
        "{text}"
    );
}

// ---------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------

/// A model text that is already canonical: writing its parse gives it back.
const CANONICAL: &str = "\
# a model with every statement form
x'=-x + a*y
dy/dt=x - y^2
p'=1
a=1
d=3
P.E.=x + y
!dd=a*2
aux en=x^2 + y^2
f(u,v)=u*v + 2u
x(0)=1
init y=0.5,p=2
par a2=1,b=-2.5,c # trailing note
par q
number k=3
wiener w1,w2
table h % 11 0 6.25 sin(t)
@ total=100, dt=.05 # options note
#include other.ode
set low {a2=1,b=2}
set none {}
global 1 x - 1 {x=0;y=2}
global -1 y {p=1}
markov z 2
{0}{a}
# between rows
{b}{0}
volt u=exp(-t)
bndry x-1
\"x=1 an active comment
x[0..3]'=-x[j] + x[j+1]
%[1..4]
c[j]'=c[j-1]*[j]
# in the block
%
done
";

#[test]
fn canonical_text_is_written_back_unchanged() {
    let (m, _) = parse(CANONICAL, FILE);
    assert_eq!(write(&m), CANONICAL);
    let back = parse(&write(&m), FILE).0;
    assert_eq!(back.strip_spans(), m.strip_spans());
}

#[test]
fn spelling_and_abbreviations_are_written_canonically() {
    let src = "PZZ A = 1 , b=2\nAUX V=X\nWin W\ndX/DT=X\nDONE\n";
    let (m, _) = parse(src, FILE);
    assert_eq!(write(&m), "par A=1,b=2\naux V=X\nwiener W\ndX/dt=X\ndone\n");
}

#[test]
fn keyword_letter_names_round_trip() {
    let src = "p'=1\na=1\nd=3\np = 5\nw=2\ni=1\ng=1\nb=1\nm=1\npar=1\nnumber=2\n";
    let m = parse_ok(src);
    let text = write(&m);
    assert_eq!(
        text,
        "p'=1\na=1\nd=3\np=5\nw=2\ni=1\ng=1\nb=1\nm=1\npar=1\nnumber=2\n"
    );
    assert_round_trip(&m);
}

#[test]
fn map_statements_round_trip() {
    let m = parse_ok("x(t+1)=r*x*(1 - x)\nx(0)=0.25\npar r=3.5\n");
    assert_eq!(write(&m), "x(t+1)=r*x*(1 - x)\nx(0)=0.25\npar r=3.5\n");
    assert_round_trip(&m);
}

#[test]
fn unsupported_statements_are_written_as_raw_text() {
    let src = "special k=mmult(4,4,w,y)\nexport {x} {y}\nonly x,y\nsolv z=x\n\
               0=x - y\nhom_bcs 0 x\noptions my.opt\ntable h data.tab\n\
               w(t)=1 + int{exp(-t)}\nz=sum(0,2)of(i')\nbdry x - 1\n";
    let (m, _) = parse(src, FILE);
    let n = m
        .statements
        .iter()
        .filter(|s| matches!(s.kind, StmtKind::Unsupported { .. }))
        .count();
    assert_eq!(n, 11, "{m:#?}");
    assert_eq!(write(&m), src);
    let (back, _) = parse(&write(&m), FILE);
    assert_eq!(back.strip_spans(), m.strip_spans());
}

#[test]
fn bad_blocks_reparse_as_bad_blocks() {
    let src = "%[a..b]\nx[j]'=1\n# kept\n%\ny'=2\n";
    let (m, diags) = parse(src, FILE);
    assert!(diags.iter().any(|d| d.severity == Severity::Error));
    assert!(matches!(m.statements[0].kind, StmtKind::BadBlock { .. }));
    let text = write(&m);
    assert_eq!(text, "%[]\nx[j]'=1\n# kept\n%\ny'=2\n");
    let (back, diags) = parse(&text, FILE);
    assert!(diags.iter().any(|d| d.severity == Severity::Error));
    assert_eq!(back.strip_spans(), m.strip_spans());
}

#[test]
fn trailing_comments_are_kept() {
    let src = "par a=1 # one\nnumber n=2 # two\n@ dt=.1 # three\n";
    let m = parse_ok(src);
    assert_eq!(write(&m), src);
    // A comment the parser rejects is still kept and written back.
    let src = "x'=1 # four\n";
    let (m, _) = parse(src, FILE);
    assert_eq!(write(&m), src);
}

#[test]
fn hand_built_statements_round_trip() {
    let row = |entries: Vec<Expr>| {
        MarkovLine::Row(MarkovRow {
            entries,
            comment: None,
            span: DUMMY_SPAN,
        })
    };
    let m = model(vec![
        stmt(StmtKind::Ode {
            name: indexed("x", range(0, 2)),
            form: OdeForm::Prime,
            rhs: bin(
                BinaryOp::Sub,
                Expr::Name(indexed("x", var(IndexOp::Add, 1))),
                Expr::Index(var(IndexOp::Mul, 2)),
            ),
        }),
        stmt(StmtKind::Global {
            sign: 0,
            cond: unary(UnaryOp::Neg, name("y")),
            assigns: vec![Assign {
                name: ident("y"),
                rhs: num(0.0),
                span: DUMMY_SPAN,
            }],
        }),
        stmt(StmtKind::Markov {
            name: ident("s"),
            states: 1,
            lines: vec![
                MarkovLine::Comment(Comment {
                    text: "# only row".to_string(),
                    span: DUMMY_SPAN,
                }),
                row(vec![num(1.0)]),
            ],
        }),
        stmt(StmtKind::Options {
            entries: vec![RawAssign {
                name: ident("meth"),
                value: "stiff".to_string(),
                value_span: DUMMY_SPAN,
                span: DUMMY_SPAN,
            }],
        }),
        stmt(StmtKind::Block {
            start: 2,
            end: 0,
            range_span: DUMMY_SPAN,
            body: vec![stmt(StmtKind::Aux {
                name: indexed("r", var(IndexOp::None, 0)),
                rhs: Expr::Index(var(IndexOp::Sub, 0)),
            })],
        }),
    ]);
    assert_eq!(
        write(&m),
        "x[0..2]'=x[j+1] - [j*2]\nglobal 0 -y {y=0}\nmarkov s 1\n# only row\n{1}\n\
         @ meth=stiff\n%[2..0]\naux r[j]=[j-0]\n%\n"
    );
    assert_round_trip(&m);
}

#[test]
fn expanded_models_round_trip() {
    let mut map = SourceMap::new();
    let top = map.add(
        "/m/top.ode",
        "#include inc.ode\npar a[1..3]=0.5\n%[0..1]\nx[j]'=-a[j+1]*x[j] + [j]\n%\ndone\n",
    );
    let mut loader = MapLoader::new();
    loader.insert("inc.ode", "number k=2\n");
    let loaded = load(top, &mut map, &mut loader);
    let mut expanded = loaded.expanded.clone();
    expanded.suppressed.clear();
    for s in &mut expanded.statements {
        s.array_index = None;
        s.array_template = None;
    }
    let text = write_expanded(&expanded);
    assert_eq!(
        text,
        "number k=2\npar a1=0.5\npar a2=0.5\npar a3=0.5\nx0'=-a1*x0 + 0\n\
         x1'=-a2*x1 + 1\ndone\n"
    );
    let back = parse_ok(&text);
    assert_eq!(back.strip_spans(), expanded.strip_spans());
    // The template keeps its include, range and block.
    let template = write(&loaded.template);
    assert_eq!(
        template,
        "#include inc.ode\npar a[1..3]=0.5\n%[0..1]\nx[j]'=-a[j+1]*x[j] + [j]\n%\ndone\n"
    );
}

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

/// Names for generated models. They include the keyword letters
/// `p a i d w m g b` and longer names that start with them (ruling g).
const NAMES: &[&str] = &[
    "p",
    "a",
    "i",
    "d",
    "w",
    "m",
    "g",
    "b",
    "x",
    "y",
    "z",
    "u",
    "v",
    "c",
    "k",
    "e",
    "q",
    "r",
    "h",
    "pa",
    "amp",
    "ival",
    "dd",
    "dv",
    "wv",
    "mu",
    "gk",
    "beta",
    "bx",
    "pval",
    "aux1",
    "par2",
    "init0",
    "done1",
    "mx",
    "gna",
    "w.2",
    "p.e",
    "ex",
    "ek2",
    "x_1",
    "vlong_name_1",
    "Vm",
    "X",
    "Pa",
];

/// Only the keyword letters and names that start with them.
const KEYWORD_NAMES: &[&str] = &[
    "p", "a", "i", "d", "w", "m", "g", "b", "pa", "amp", "ival", "dd", "wv", "mu", "gk", "beta",
    "bx", "par", "aux", "init", "done", "wiener", "markov", "global", "table",
];

/// Builtins used as call names.
const CALLS: &[&str] = &["sin", "exp", "max", "abs", "atan2"];

/// The reserved names of `check.rs` (`reserved_reason` is private, so the
/// list is copied here); builtins come from `pc_ode::builtins::builtin`.
/// `int`, `sum` and `of` are left out of the pools too, since they mark
/// unsupported formula families.
fn is_reserved(key: &str) -> bool {
    matches!(
        key,
        "t" | "pi" | "not" | "if" | "then" | "else" | "int" | "sum" | "of"
    ) || (key.len() == 4 && key.starts_with("arg") && key.as_bytes()[3].is_ascii_digit())
        || pc_ode::builtins::builtin(key).is_some()
}

/// True when `name` starts with `e` or `E` followed by a digit.
fn e_digit(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() >= 2 && (b[0] == b'e' || b[0] == b'E') && b[1].is_ascii_digit()
}

#[test]
fn name_pools_hold_no_reserved_names() {
    for n in NAMES.iter().chain(KEYWORD_NAMES) {
        assert!(!is_reserved(n) && !e_digit(n), "{n}");
    }
    for n in CALLS {
        assert!(pc_ode::builtins::builtin(n).is_some(), "{n}");
    }
}

/// What index forms an expression may use.
#[derive(Debug, Clone, Copy)]
struct Ctx {
    pool: &'static [&'static str],
    /// Index forms are allowed (inside a range line or a block).
    index: bool,
    /// The largest `n` for `[j-n]`, so the index stays non-negative.
    max_sub: u32,
}

impl Ctx {
    fn plain(pool: &'static [&'static str]) -> Ctx {
        Ctx {
            pool,
            index: false,
            max_sub: 0,
        }
    }
}

fn index_var(ctx: Ctx) -> BoxedStrategy<Index> {
    let max_sub = ctx.max_sub;
    prop_oneof![
        Just(var(IndexOp::None, 0)),
        (1u32..4).prop_map(|n| var(IndexOp::Add, n)),
        (0..=max_sub).prop_map(|n| var(IndexOp::Sub, n)),
        (0u32..4).prop_map(|n| var(IndexOp::Mul, n)),
    ]
    .boxed()
}

fn pool_name(ctx: Ctx) -> BoxedStrategy<Ident> {
    prop::sample::select(ctx.pool).prop_map(ident).boxed()
}

/// A name, with an index form when the context allows one.
fn use_name(ctx: Ctx) -> BoxedStrategy<Ident> {
    if ctx.index {
        prop_oneof![
            pool_name(ctx),
            (pool_name(ctx), index_var(ctx)).prop_map(|(mut id, ix)| {
                id.index = Some(ix);
                id
            }),
        ]
        .boxed()
    } else {
        pool_name(ctx)
    }
}

fn literal_value() -> BoxedStrategy<f64> {
    prop_oneof![
        (0u32..100).prop_map(f64::from),
        (0u32..1000).prop_map(|n| f64::from(n) / 8.0),
        prop::sample::select(vec![1e-7, 2.5e-6, 1e-5, 1e15, 1e16, 3e20, 0.1, 1e300]),
        any::<f64>().prop_filter_map("finite", |v| v.is_finite().then_some(v.abs())),
    ]
    .boxed()
}

fn num_expr() -> BoxedStrategy<Expr> {
    literal_value().prop_map(num).boxed()
}

fn leaf(ctx: Ctx) -> BoxedStrategy<Expr> {
    if ctx.index {
        prop_oneof![
            3 => num_expr(),
            3 => use_name(ctx).prop_map(Expr::Name),
            1 => index_var(ctx).prop_map(Expr::Index),
        ]
        .boxed()
    } else {
        prop_oneof![num_expr(), use_name(ctx).prop_map(Expr::Name)].boxed()
    }
}

const BINARY_OPS: &[BinaryOp] = &[
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Pow,
    BinaryOp::StarStar,
    BinaryOp::Lt,
    BinaryOp::Gt,
    BinaryOp::Le,
    BinaryOp::Ge,
    BinaryOp::Eq,
    BinaryOp::And,
    BinaryOp::Or,
];

fn call_name(ctx: Ctx) -> BoxedStrategy<Ident> {
    prop_oneof![pool_name(ctx), prop::sample::select(CALLS).prop_map(ident)].boxed()
}

/// A literal, sometimes one made only of zeros (`0`, `00`), which turns a
/// following `x1` into a malformed hex number.
fn lit_expr() -> BoxedStrategy<Expr> {
    prop_oneof![
        4 => num_expr(),
        1 => prop::sample::select(vec!["0", "00"]).prop_map(|text| Expr::Num {
            value: 0.0,
            text: text.to_string(),
            span: DUMMY_SPAN,
        }),
    ]
    .boxed()
}

/// Names that merge with a literal written before them: an exponent
/// (`e1`, `E2x`) or, after a zero literal, a hex prefix (`x1`, `xa`, `Xf`).
const MERGING_NAMES: &[&str] = &["e1", "E2x", "e0b", "x1", "xa", "Xf", "x", "X", "xg"];

/// A left operand of implicit multiplication as the parser builds it: an
/// expression whose text ends with a literal, through any binary or prefix
/// operator above level 1 (`docs/language.md`, "Precedence"). Each right
/// operand binds tighter than its parent, so the writer never has to put
/// it in parentheses for precedence; the parentheses it would add only to
/// avoid a warning are dropped in this position.
fn lit_tail(inner: BoxedStrategy<Expr>) -> BoxedStrategy<Expr> {
    let power = prop::sample::select(vec![BinaryOp::Pow, BinaryOp::StarStar]);
    let comparison = prop::sample::select(vec![
        BinaryOp::Lt,
        BinaryOp::Gt,
        BinaryOp::Le,
        BinaryOp::Ge,
        BinaryOp::Eq,
    ]);
    let term = prop::sample::select(vec![BinaryOp::Mul, BinaryOp::Div, BinaryOp::And]);
    // Level 5 and up.
    let t5 = prop_oneof![
        2 => lit_expr(),
        1 => (inner.clone(), lit_expr(), power).prop_map(|(a, n, op)| bin(op, a, n)),
    ]
    .boxed();
    // Level 4 and up.
    let t4 = prop_oneof![
        2 => t5.clone(),
        1 => (inner.clone(), t5.clone(), comparison.clone()).prop_map(|(a, b, op)| bin(op, a, b)),
    ]
    .boxed();
    prop_oneof![
        3 => t4.clone(),
        1 => (inner.clone(), t4.clone(), term).prop_map(|(a, b, op)| bin(op, a, b)),
        1 => (t4.clone(), any::<bool>())
            .prop_map(|(a, neg)| unary(if neg { UnaryOp::Neg } else { UnaryOp::Not }, a)),
        // `2b<2c`: a product whose right side is a comparison that ends
        // with a literal.
        1 => (lit_expr(), inner, t5, comparison)
            .prop_map(|(n, a, b, op)| implicit(n, bin(op, a, b))),
    ]
    .boxed()
}

/// An expression. Implicit multiplication appears only where the parser
/// can build it: the left operand's text ends with a literal (`lit_tail`).
fn expr(ctx: Ctx) -> BoxedStrategy<Expr> {
    leaf(ctx)
        .prop_recursive(4, 40, 3, move |inner| {
            let lit_end = lit_tail(inner.clone());
            let index_rhs = if ctx.index {
                index_var(ctx).prop_map(Expr::Index).boxed()
            } else {
                lit_expr()
            };
            let name_start = prop_oneof![
                prop::sample::select(MERGING_NAMES).prop_map(|n| Expr::Name(ident(n))),
                lit_expr(),
                index_rhs,
                use_name(ctx).prop_map(Expr::Name),
                (call_name(ctx), prop::collection::vec(inner.clone(), 1..3)).prop_map(
                    |(name, args)| Expr::Call {
                        name,
                        args,
                        span: DUMMY_SPAN,
                    }
                ),
                (
                    use_name(ctx),
                    inner.clone(),
                    prop::sample::select(vec![BinaryOp::Pow, BinaryOp::Lt, BinaryOp::Ge])
                )
                    .prop_map(|(n, b, op)| bin(op, Expr::Name(n), b)),
                (
                    inner.clone(),
                    inner.clone(),
                    prop::sample::select(vec![
                        BinaryOp::Add,
                        BinaryOp::Sub,
                        BinaryOp::Mul,
                        BinaryOp::Div,
                        BinaryOp::And,
                        BinaryOp::Or
                    ])
                )
                    .prop_map(|(a, b, op)| bin(op, a, b)),
                (inner.clone(), any::<bool>()).prop_map(|(a, neg)| {
                    unary(if neg { UnaryOp::Neg } else { UnaryOp::Not }, a)
                }),
            ];
            prop_oneof![
                4 => (inner.clone(), inner.clone(), prop::sample::select(BINARY_OPS))
                    .prop_map(|(a, b, op)| bin(op, a, b)),
                2 => (lit_end, name_start).prop_map(|(a, b)| implicit(a, b)),
                2 => (inner.clone(), any::<bool>()).prop_map(|(a, neg)| {
                    unary(if neg { UnaryOp::Neg } else { UnaryOp::Not }, a)
                }),
                1 => (call_name(ctx), prop::collection::vec(inner.clone(), 1..4)).prop_map(
                    |(name, args)| Expr::Call {
                        name,
                        args,
                        span: DUMMY_SPAN,
                    }
                ),
                1 => (inner.clone(), inner.clone(), inner).prop_map(|(c, a, b)| Expr::If {
                    cond: Box::new(c),
                    then: Box::new(a),
                    else_: Box::new(b),
                    span: DUMMY_SPAN,
                }),
            ]
        })
        .boxed()
}

fn value_num() -> BoxedStrategy<Value> {
    (literal_value(), any::<bool>())
        .prop_map(|(v, neg)| {
            let v = if neg && v != 0.0 { -v } else { v };
            Value::Num {
                value: v,
                text: format_number(v),
                span: DUMMY_SPAN,
            }
        })
        .boxed()
}

fn value(ctx: Ctx) -> BoxedStrategy<Value> {
    if ctx.index {
        prop_oneof![3 => value_num(), 1 => index_var(ctx).prop_map(Value::Index)].boxed()
    } else {
        value_num()
    }
}

fn trailing_comment() -> BoxedStrategy<Option<Comment>> {
    prop::option::weighted(
        0.3,
        prop::sample::select(vec!["# note", "#x=1", "# two words, here", "#"]),
    )
    .prop_map(|c| {
        c.map(|text| Comment {
            text: text.to_string(),
            span: DUMMY_SPAN,
        })
    })
    .boxed()
}

fn raw_assign(
    names: &'static [&'static str],
    values: &'static [&'static str],
) -> BoxedStrategy<RawAssign> {
    (prop::sample::select(names), prop::sample::select(values))
        .prop_map(|(n, v)| RawAssign {
            name: ident(n),
            value: v.to_string(),
            value_span: DUMMY_SPAN,
            span: DUMMY_SPAN,
        })
        .boxed()
}

const OPTION_NAMES: &[&str] = &[
    "meth", "dt", "total", "xlo", "xhi", "maxstor", "bounds", "Ntst",
];
const RAW_VALUES: &[&str] = &["stiff", "0.01", "1e-3", "200", "-5", "x1", "Qrk", "10\\"];

/// Lines the parser keeps as `Unsupported` statements.
const UNSUPPORTED_LINES: &[&str] = &[
    "special k=mmult(4,4,w,y)",
    "export {x} {y}",
    "only x,y",
    "solv z=x",
    "0=x - y",
    "hom_bcs 0 x",
    "options my.opt",
    "table h data.tab",
    "w(t)=1 + int{exp(-t)}",
    "z=sum(0,2)of(i')",
    "bdry x - 1",
    // Raw text that ends in a backslash; the parser trims the blank.
    "bndry x\\ ",
    "\"hello\\ ",
];

fn unsupported_stmt() -> BoxedStrategy<Stmt> {
    prop::sample::select(UNSUPPORTED_LINES)
        .prop_map(|line| {
            // A first line that starts with a digit marks an old-format file,
            // so the line is parsed after a comment.
            let (m, _) = parse(&format!("#\n{line}\n"), FILE);
            let s = m
                .statements
                .into_iter()
                .nth(1)
                .unwrap_or_else(|| panic!("{line:?}"));

            assert!(matches!(s.kind, StmtKind::Unsupported { .. }), "{line}");
            s.strip_spans()
        })
        .boxed()
}

fn comment_stmt() -> BoxedStrategy<Stmt> {
    prop::sample::select(vec!["# a comment", "#", "# par a=1", "#x'=y", "## two"])
        .prop_map(|text| {
            stmt(StmtKind::Comment(Comment {
                text: text.to_string(),
                span: DUMMY_SPAN,
            }))
        })
        .boxed()
}

fn entries(ctx: Ctx, bare_ok: bool) -> BoxedStrategy<Vec<Entry>> {
    let entry = (
        use_name(ctx),
        if bare_ok {
            prop::option::weighted(0.8, value(ctx)).boxed()
        } else {
            value(ctx).prop_map(Some).boxed()
        },
    )
        .prop_map(|(name, value)| Entry {
            name,
            value,
            span: DUMMY_SPAN,
        });
    prop::collection::vec(entry, 1..4).boxed()
}

fn with_comment(kind: BoxedStrategy<StmtKind>) -> BoxedStrategy<Stmt> {
    (kind, trailing_comment())
        .prop_map(|(kind, comment)| Stmt {
            comment,
            ..stmt(kind)
        })
        .boxed()
}

/// A statement that may sit inside a `%` block or on a range line. `name`
/// is the declared name (it carries the index form, if any).
fn declaring_stmt(ctx: Ctx, name: BoxedStrategy<Ident>) -> BoxedStrategy<Stmt> {
    let e = expr(ctx);
    prop_oneof![
        3 => (name.clone(), e.clone(), any::<bool>()).prop_map(|(name, rhs, ddt)| {
            let form = if ddt && name.index.is_none() {
                OdeForm::Ddt
            } else {
                OdeForm::Prime
            };
            stmt(StmtKind::Ode { name, form, rhs })
        }),
        1 => (name.clone(), e.clone()).prop_map(|(name, rhs)| stmt(StmtKind::Map { name, rhs })),
        1 => (name.clone(), value(ctx))
            .prop_map(|(name, value)| stmt(StmtKind::InitValue { name, value })),
        1 => (name.clone(), e.clone()).prop_map(|(name, rhs)| stmt(StmtKind::Derived { name, rhs })),
        2 => (name.clone(), e.clone()).prop_map(|(name, rhs)| stmt(StmtKind::Fixed { name, rhs })),
        1 => (name.clone(), e).prop_map(|(name, rhs)| stmt(StmtKind::Aux { name, rhs })),
        1 => with_comment(
            (name.clone(), prop::option::of(value(ctx)))
                .prop_map(|(name, value)| StmtKind::Par {
                    entries: vec![Entry {
                        name,
                        value,
                        span: DUMMY_SPAN,
                    }],
                })
                .boxed()
        ),
        1 => (name.clone(), value(ctx)).prop_map(|(name, value)| stmt(StmtKind::Init {
            entries: vec![Entry {
                name,
                value: Some(value),
                span: DUMMY_SPAN,
            }],
        })),
        1 => name.prop_map(|n| stmt(StmtKind::Wiener { names: vec![n] })),
    ]
    .boxed()
}

/// A range line: one statement whose declared name has `[s..e]`.
fn range_stmt(pool: &'static [&'static str]) -> BoxedStrategy<Stmt> {
    (0u32..5, 0u32..5)
        .prop_flat_map(move |(s, e)| {
            let ctx = Ctx {
                pool,
                index: true,
                max_sub: s.min(e),
            };
            let name = prop::sample::select(pool)
                .prop_map(move |n| indexed(n, range(s, e)))
                .boxed();
            declaring_stmt(ctx, name)
        })
        .boxed()
}

/// A `%[s..e]` block with `[j]` forms in its body.
fn block_stmt(pool: &'static [&'static str]) -> BoxedStrategy<Stmt> {
    (0i64..5, 0i64..5)
        .prop_flat_map(move |(s, e)| {
            let ctx = Ctx {
                pool,
                index: true,
                max_sub: s.min(e) as u32,
            };
            let body_stmt = prop_oneof![
                4 => declaring_stmt(ctx, use_name(ctx)),
                1 => comment_stmt(),
            ];
            prop::collection::vec(body_stmt, 0..4).prop_map(move |body| {
                stmt(StmtKind::Block {
                    start: s,
                    end: e,
                    range_span: DUMMY_SPAN,
                    body,
                })
            })
        })
        .boxed()
}

fn markov_stmt(ctx: Ctx) -> BoxedStrategy<Stmt> {
    (pool_name(ctx), 1u32..3)
        .prop_flat_map(move |(name, states)| {
            let n = states as usize;
            let row = prop::collection::vec(expr(ctx), n).prop_map(|entries| {
                MarkovLine::Row(MarkovRow {
                    entries,
                    comment: None,
                    span: DUMMY_SPAN,
                })
            });
            let line = (prop::option::weighted(0.2, Just("# row")), row).prop_map(|(c, row)| {
                let mut lines = Vec::new();
                if let Some(text) = c {
                    lines.push(MarkovLine::Comment(Comment {
                        text: text.to_string(),
                        span: DUMMY_SPAN,
                    }));
                }
                lines.push(row);
                lines
            });
            prop::collection::vec(line, n).prop_map(move |lines| {
                stmt(StmtKind::Markov {
                    name: name.clone(),
                    states,
                    lines: lines.into_iter().flatten().collect(),
                })
            })
        })
        .boxed()
}

/// A top-level statement of any form except `done`.
fn top_stmt(pool: &'static [&'static str]) -> BoxedStrategy<Stmt> {
    let ctx = Ctx::plain(pool);
    let e = expr(ctx);
    prop_oneof![
        6 => declaring_stmt(ctx, pool_name(ctx)),
        1 => with_comment(entries(ctx, true).prop_map(|entries| StmtKind::Par { entries }).boxed()),
        1 => with_comment(
            entries(ctx, false).prop_map(|entries| StmtKind::Number { entries }).boxed()
        ),
        1 => entries(ctx, false).prop_map(|entries| stmt(StmtKind::Init { entries })),
        1 => (pool_name(ctx), prop::collection::vec(pool_name(ctx), 1..4), e.clone())
            .prop_map(|(name, params, rhs)| stmt(StmtKind::Function { name, params, rhs })),
        1 => (pool_name(ctx), 1u32..200, value_num(), value_num(), e.clone()).prop_map(
            |(name, n, lo, hi, rhs)| {
                let n = f64::from(n);
                stmt(StmtKind::Table {
                    name,
                    npts: Value::Num {
                        value: n,
                        text: format_number(n),
                        span: DUMMY_SPAN,
                    },
                    lo,
                    hi,
                    rhs,
                })
            }
        ),
        1 => with_comment(
            prop::collection::vec(raw_assign(OPTION_NAMES, RAW_VALUES), 1..4)
                .prop_map(|entries| StmtKind::Options { entries })
                .boxed()
        ),
        1 => prop::sample::select(vec!["inc.ode", "sub/m2.ode", "../a b.ode"]).prop_map(|p| {
            stmt(StmtKind::Include {
                path: p.to_string(),
                path_span: DUMMY_SPAN,
            })
        }),
        1 => (pool_name(ctx), prop::collection::vec(raw_assign(pool, RAW_VALUES), 0..3))
            .prop_map(|(name, entries)| stmt(StmtKind::Set { name, entries })),
        1 => (
            prop::sample::select(vec![-1i8, 0, 1]),
            e.clone(),
            prop::collection::vec(
                (pool_name(ctx), e).prop_map(|(name, rhs)| Assign {
                    name,
                    rhs,
                    span: DUMMY_SPAN,
                }),
                1..3
            )
        )
            .prop_map(|(sign, cond, assigns)| stmt(StmtKind::Global { sign, cond, assigns })),
        1 => markov_stmt(ctx),
        1 => prop::collection::vec(pool_name(ctx), 1..4)
            .prop_map(|names| stmt(StmtKind::Wiener { names })),
        1 => comment_stmt(),
        1 => unsupported_stmt(),
        1 => range_stmt(pool),
        1 => block_stmt(pool),
    ]
    .boxed()
}

fn gen_model(pool: &'static [&'static str]) -> BoxedStrategy<Model> {
    (prop::collection::vec(top_stmt(pool), 0..8), any::<bool>())
        .prop_map(|(mut statements, done)| {
            // A model may start with `0=x - y`; the writer puts a blank line
            // before it so that the text does not look like an old-format file.
            if done {
                statements.push(stmt(StmtKind::Done));
            }
            model(statements)
        })
        .boxed()
}

// ---------------------------------------------------------------------
// Surface noise
// ---------------------------------------------------------------------

/// A small xorshift generator, seeded by proptest, for the noise choices.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// Text inserted where blanks are allowed: blanks, tabs and backslash
/// line continuations.
const GAPS: &[&str] = &[" ", "  ", "\t", " \\\n", "\\\n ", "\\\n"];

/// The comment text inserted as noise; the normalizer drops it again.
const NOISE_COMMENT: &str = "#~noise";

#[derive(Clone, Copy, PartialEq)]
enum LineClass {
    /// Written back without any change.
    Raw,
    /// A keyword line: the keyword may be abbreviated.
    Keyword { abbreviate: bool },
    /// A formula line.
    Formula,
}

fn random_case(rng: &mut Rng, word: &str) -> String {
    word.chars()
        .map(|c| {
            if rng.chance(50) {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// Adds noise to one statement line (no line end).
fn noise_line(rng: &mut Rng, line: &str, class: LineClass) -> String {
    if class == LineClass::Raw {
        return line.to_string();
    }
    let (tokens, _) = lex(FILE, line);
    let tokens: Vec<_> = tokens
        .into_iter()
        .filter(|t| !matches!(t.kind, TokenKind::Newline | TokenKind::Eof))
        .collect();
    let first_eq = tokens.iter().position(|t| t.kind == TokenKind::Eq);
    let mut out = String::new();
    if rng.chance(20) {
        out.push_str(rng.pick(&[" ", "  ", "\t"]));
    }
    let mut depth = 0i32;
    for (i, t) in tokens.iter().enumerate() {
        let text = &line[t.span.start..t.span.end];
        if i > 0 {
            let prev = &tokens[i - 1];
            let gap = &line[prev.span.end..t.span.start];
            out.push_str(gap);
            let blank_ok =
                t.preceded_by_blank || Some(i) == first_eq || first_eq.is_some_and(|q| i > q);
            let excluded = t.kind == TokenKind::Comment
                || depth > 0
                || (matches!(prev.kind, TokenKind::Num(_))
                    && matches!(t.kind, TokenKind::Name(_) | TokenKind::LParen)
                    && !t.preceded_by_blank)
                || (matches!(prev.kind, TokenKind::Name(_))
                    && matches!(t.kind, TokenKind::LParen | TokenKind::LBracket))
                || (matches!(prev.kind, TokenKind::Minus | TokenKind::Bang)
                    && !t.preceded_by_blank);
            if blank_ok && !excluded && rng.chance(30) {
                out.push_str(rng.pick(GAPS));
            }
        }
        match &t.kind {
            TokenKind::LBracket => depth += 1,
            TokenKind::RBracket => depth -= 1,
            _ => {}
        }
        let new_text = match &t.kind {
            TokenKind::Name(_)
                if i == 0 && matches!(class, LineClass::Keyword { abbreviate: true }) =>
            {
                let mut word = text[..1].to_string();
                for _ in 0..rng.below(4) {
                    word.push_str(rng.pick(&["x", "Z", "q", "k"]));
                }
                random_case(rng, &word)
            }
            TokenKind::Name(_) if depth == 0 => random_case(rng, text),
            TokenKind::Num(_) if rng.chance(50) => text.replace('e', "E"),
            _ => text.to_string(),
        };
        out.push_str(&new_text);
    }
    // Keep whatever followed the last token (nothing, for writer output).
    if let Some(last) = tokens.last() {
        out.push_str(&line[last.span.end..]);
    }
    out
}

fn line_class(kind: &StmtKind) -> LineClass {
    match kind {
        StmtKind::Comment(_)
        | StmtKind::Unsupported { .. }
        | StmtKind::Options { .. }
        | StmtKind::Set { .. }
        | StmtKind::Include { .. }
        | StmtKind::Block { .. }
        | StmtKind::BadBlock { .. } => LineClass::Raw,
        StmtKind::Number { .. } => LineClass::Keyword { abbreviate: false },
        StmtKind::Par { .. }
        | StmtKind::Init { .. }
        | StmtKind::Aux { .. }
        | StmtKind::Wiener { .. }
        | StmtKind::Table { .. }
        | StmtKind::Global { .. }
        | StmtKind::Markov { .. }
        | StmtKind::Done => LineClass::Keyword { abbreviate: true },
        _ => LineClass::Formula,
    }
}

fn noise_comment(rng: &mut Rng, out: &mut String) {
    if rng.chance(15) {
        out.push_str(NOISE_COMMENT);
        out.push('\n');
    }
}

/// Writes one statement with noise.
fn noise_stmt(rng: &mut Rng, s: &Stmt, out: &mut String) {
    noise_comment(rng, out);
    match &s.kind {
        StmtKind::Block {
            start, end, body, ..
        } => {
            out.push_str(&format!("%[{start}..{end}]\n"));
            for b in body {
                noise_stmt(rng, b, out);
            }
            noise_comment(rng, out);
            out.push_str("%\n");
        }
        StmtKind::Markov { .. } => {
            // Only the `markov` line gets noise; the rows stay as written.
            let text = write(&model(vec![s.clone()]));
            let mut lines = text.lines();
            let head = lines.next().unwrap_or_default();
            out.push_str(&noise_line(rng, head, line_class(&s.kind)));
            out.push('\n');
            for l in lines {
                out.push_str(l);
                out.push('\n');
            }
        }
        kind => {
            let text = write(&model(vec![s.clone()]));
            let line = text.strip_suffix('\n').unwrap_or(&text);
            out.push_str(&noise_line(rng, line, line_class(kind)));
            out.push('\n');
        }
    }
}

fn noisy_text(seed: u64, m: &Model) -> String {
    let mut rng = Rng::new(seed);
    let mut out = String::new();
    for s in &m.statements {
        noise_stmt(&mut rng, s, &mut out);
    }
    out
}

// The normalizer of ruling f. It folds only the fields that the noise
// changes by design: `Ident.text` becomes `Ident.key`, literal text
// (`Expr::Num`, `Value::Num`) is lower-cased, and the inserted `#~noise`
// comment statements are dropped. No keyword spelling is stored in the AST.

fn norm_ident(id: &mut Ident) {
    id.text = id.key.clone();
}

fn norm_value(v: &mut Value) {
    if let Value::Num { text, .. } = v {
        *text = text.to_ascii_lowercase();
    }
}

fn norm_expr(e: &mut Expr) {
    match e {
        Expr::Num { text, .. } => *text = text.to_ascii_lowercase(),
        Expr::Name(id) => norm_ident(id),
        Expr::Index(_) | Expr::Error { .. } => {}
        Expr::Unary { operand, .. } => norm_expr(operand),
        Expr::Binary { lhs, rhs, .. } => {
            norm_expr(lhs);
            norm_expr(rhs);
        }
        Expr::Call { name, args, .. } => {
            norm_ident(name);
            args.iter_mut().for_each(norm_expr);
        }
        Expr::If {
            cond, then, else_, ..
        } => {
            norm_expr(cond);
            norm_expr(then);
            norm_expr(else_);
        }
    }
}

fn norm_entries(entries: &mut [Entry]) {
    for en in entries {
        norm_ident(&mut en.name);
        if let Some(v) = &mut en.value {
            norm_value(v);
        }
    }
}

fn norm_stmts(stmts: &mut Vec<Stmt>) {
    stmts.retain(|s| !matches!(&s.kind, StmtKind::Comment(c) if c.text == NOISE_COMMENT));
    for s in stmts {
        match &mut s.kind {
            StmtKind::Ode { name, rhs, .. }
            | StmtKind::Map { name, rhs }
            | StmtKind::Derived { name, rhs }
            | StmtKind::Fixed { name, rhs }
            | StmtKind::Aux { name, rhs } => {
                norm_ident(name);
                norm_expr(rhs);
            }
            StmtKind::InitValue { name, value } => {
                norm_ident(name);
                norm_value(value);
            }
            StmtKind::Init { entries }
            | StmtKind::Par { entries }
            | StmtKind::Number { entries } => norm_entries(entries),
            StmtKind::Function { name, params, rhs } => {
                norm_ident(name);
                params.iter_mut().for_each(norm_ident);
                norm_expr(rhs);
            }
            StmtKind::Table {
                name,
                npts,
                lo,
                hi,
                rhs,
            } => {
                norm_ident(name);
                norm_value(npts);
                norm_value(lo);
                norm_value(hi);
                norm_expr(rhs);
            }
            StmtKind::Global { cond, assigns, .. } => {
                norm_expr(cond);
                for a in assigns {
                    norm_ident(&mut a.name);
                    norm_expr(&mut a.rhs);
                }
            }
            StmtKind::Markov { name, lines, .. } => {
                norm_ident(name);
                for l in lines {
                    if let MarkovLine::Row(r) = l {
                        r.entries.iter_mut().for_each(norm_expr);
                    }
                }
            }
            StmtKind::Wiener { names } => names.iter_mut().for_each(norm_ident),
            StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => norm_stmts(body),
            // Raw classes: the noise leaves them alone.
            StmtKind::Options { .. }
            | StmtKind::Include { .. }
            | StmtKind::Set { .. }
            | StmtKind::Comment(_)
            | StmtKind::Unsupported { .. }
            | StmtKind::Done => {}
        }
    }
}

fn normalize(m: &Model) -> Model {
    let mut m = m.strip_spans();
    norm_stmts(&mut m.statements);
    m
}

// ---------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------

/// Collects every expression of `stmts`, at any depth.
fn stmt_exprs<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a Expr>) {
    fn sub<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
        out.push(e);
        match e {
            Expr::Unary { operand, .. } => sub(operand, out),
            Expr::Binary { lhs, rhs, .. } => {
                sub(lhs, out);
                sub(rhs, out);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| sub(a, out)),
            Expr::If {
                cond, then, else_, ..
            } => {
                sub(cond, out);
                sub(then, out);
                sub(else_, out);
            }
            _ => {}
        }
    }
    for s in stmts {
        match &s.kind {
            StmtKind::Ode { rhs, .. }
            | StmtKind::Map { rhs, .. }
            | StmtKind::Derived { rhs, .. }
            | StmtKind::Fixed { rhs, .. }
            | StmtKind::Aux { rhs, .. }
            | StmtKind::Function { rhs, .. }
            | StmtKind::Table { rhs, .. } => sub(rhs, out),
            StmtKind::Global { cond, assigns, .. } => {
                sub(cond, out);
                assigns.iter().for_each(|a| sub(&a.rhs, out));
            }
            StmtKind::Markov { lines, .. } => {
                for l in lines {
                    if let MarkovLine::Row(r) = l {
                        r.entries.iter().for_each(|e| sub(e, out));
                    }
                }
            }
            StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => stmt_exprs(body, out),
            _ => {}
        }
    }
}

/// True for an implicit product whose left operand, on its right edge
/// (`Binary.rhs` and `Unary.operand`, down from the top), holds an operator
/// pair that the parser warns about: a comparison or `&`/`|` under
/// arithmetic (`a < 2x`, `a&1x`, `a/b < 2x`), arithmetic under `&`/`|`, or a
/// comparison under prefix `-` (`-a < 2x`). The writer leaves out the
/// parentheses that would avoid the warning there, since `(a < 2)x` is not
/// implicit.
fn warning_implicit(e: &Expr) -> bool {
    #[derive(Clone, Copy)]
    enum Parent {
        Bin(BinaryOp),
        Un(UnaryOp),
    }
    fn warns(parent: Parent, child: &Expr) -> bool {
        let Expr::Binary { op, .. } = child else {
            return false;
        };
        match parent {
            Parent::Bin(p) => {
                (p.is_arithmetic() && (op.is_comparison() || op.is_logical()))
                    || (p.is_logical() && op.is_arithmetic())
            }
            Parent::Un(u) => u == UnaryOp::Neg && op.is_comparison(),
        }
    }
    fn edge(parent: Parent, e: &Expr) -> bool {
        warns(parent, e)
            || match e {
                Expr::Binary { op, rhs, .. } => edge(Parent::Bin(*op), rhs),
                Expr::Unary { op, operand, .. } => edge(Parent::Un(*op), operand),
                _ => false,
            }
    }
    matches!(e, Expr::Binary { op: BinaryOp::Mul, implicit: true, lhs, .. }
        if edge(Parent::Bin(BinaryOp::Mul), lhs))
}

/// Property 1 for one generated model.
fn check_round_trip(m: &Model) -> Result<(), TestCaseError> {
    let text = write(m);
    let (back, diags) = parse(&text, FILE);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    prop_assert!(errors.is_empty(), "errors in\n{}\n{:#?}", text, errors);
    prop_assert_eq!(back.strip_spans(), m.strip_spans(), "text:\n{}", text);
    // The writer adds the parentheses that avoid a precedence warning,
    // except where they would turn an implicit product into `*`. Each
    // warning must fall in a statement that holds such a product.
    for d in diags.iter().filter(|d| d.code == Code(14)) {
        let i = back
            .statements
            .iter()
            .position(|s| s.span.start <= d.span.start && d.span.start < s.span.end);
        let exempt = i.is_some_and(|i| {
            let mut exprs = Vec::new();
            stmt_exprs(std::slice::from_ref(&m.statements[i]), &mut exprs);
            exprs.iter().any(|e| warning_implicit(e))
        });
        prop_assert!(exempt, "warning in\n{}\n{:#?}", text, d);
    }
    // parse(write(parse(s))) == parse(s) on the generated text.
    let again = write(&back);
    prop_assert_eq!(&again, &text);
    let (back2, _) = parse(&again, FILE);
    prop_assert_eq!(back2.strip_spans(), back.strip_spans());
    Ok(())
}

/// Property 2 for one generated model.
fn check_noise(m: &Model, seed: u64) -> Result<(), TestCaseError> {
    let text = noisy_text(seed, m);
    let (back, diags) = parse(&text, FILE);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    prop_assert!(errors.is_empty(), "errors in\n{}\n{:#?}", text, errors);
    prop_assert_eq!(normalize(&back), normalize(m), "text:\n{}", text);
    Ok(())
}

/// Property 4 for one input: no pass panics and every span is in bounds.
fn check_robust(text: &str) {
    let (m, diags) = parse(text, FILE);
    assert_spans_in(text, &diags);
    let (expanded, diags) = pc_ode::expand(&m);
    assert_spans_in(text, &diags);
    let (_, diags) = pc_ode::check(&expanded);
    assert_spans_in(text, &diags);
    let written = write(&m);
    let _ = parse(&written, FILE);
    let _ = write_expanded(&expanded);
}

/// Fragments of `.ode` text for the token soup.
const SOUP: &[&str] = &[
    "'",
    "(0)=",
    "(t+1)=",
    "[",
    "]",
    "..",
    "[j]",
    "[j+1]",
    "[0..3]",
    "%",
    "%[",
    "@",
    "#include ",
    "#",
    "\\\n",
    "\n",
    " ",
    "\t",
    "=",
    ",",
    "{",
    "}",
    ";",
    "(",
    ")",
    "!",
    "par",
    "p",
    "aux",
    "init",
    "number",
    "wiener",
    "markov",
    "global",
    "table",
    "set",
    "done",
    "d",
    "x",
    "y",
    "dt",
    "/",
    "1",
    "2.5",
    "1e5",
    "1e-3",
    "3E+2",
    "1e",
    ".5",
    "e1",
    "+",
    "-",
    "*",
    "**",
    "^",
    "<",
    "<=",
    "==",
    "&",
    "|",
    "not",
    "if",
    "then",
    "else",
    "sin",
    "int{",
    "sum(",
    "of",
    "\"",
    "~",
    "é",
    "\u{0}",
    "\r\n",
    "0=",
    "special",
    "export",
    "only",
];

proptest! {
    // Failures are reported, not stored: an integration test has no
    // `lib.rs` next to it for the default persistence file. Copy the
    // shrunk input of a failure into a unit test by hand, so that it is
    // checked on every run.
    #![proptest_config(ProptestConfig {
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn round_trip_format_number(bits in any::<u64>()) {
        let v = f64::from_bits(bits).abs();
        prop_assume!(v.is_finite());
        let text = format_number(v);
        prop_assert!(text.len() <= 24, "{}", text);
        let (tokens, diags) = lex(FILE, &text);
        prop_assert!(diags.is_empty(), "{} {:?}", text, diags);
        prop_assert_eq!(tokens.len(), 2, "{} {:?}", text, tokens);
        match tokens[0].kind {
            TokenKind::Num(back) => prop_assert_eq!(back.to_bits(), v.to_bits(), "{}", text),
            ref other => prop_assert!(false, "{} lexed as {:?}", text, other),
        }
    }

    #[test]
    fn round_trip_generated_models(m in gen_model(NAMES)) {
        check_round_trip(&m)?;
    }

    #[test]
    fn round_trip_survives_surface_noise(m in gen_model(NAMES), seed in any::<u64>()) {
        check_noise(&m, seed)?;
    }

    #[test]
    fn round_trip_keyword_letter_names(m in gen_model(KEYWORD_NAMES), seed in any::<u64>()) {
        check_round_trip(&m)?;
        check_noise(&m, seed)?;
    }

    #[test]
    fn round_trip_parser_never_panics_on_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..200),
    ) {
        check_robust(&String::from_utf8_lossy(&bytes));
    }

    #[test]
    fn round_trip_parser_never_panics_on_token_soup(
        parts in prop::collection::vec(prop::sample::select(SOUP), 0..60),
    ) {
        check_robust(&parts.concat());
    }
}
