use super::*;
use crate::ast::{BinaryOp as B, DUMMY_SPAN, Ident, Index, IndexOp, UnaryOp};
use crate::diag::FixIt;
use crate::diag::{CODES, Severity};
use crate::lexer::TokenKind;

const F: FileId = FileId(0);

fn parse(text: &str) -> (Expr, Vec<Diagnostic>) {
    parse_expr_text(F, text)
}

fn tree(text: &str) -> Expr {
    parse(text).0.strip_spans()
}

fn codes(text: &str) -> Vec<Code> {
    parse(text).1.iter().map(|d| d.code).collect()
}

/// The tree of `text`, which must parse with no diagnostic at all.
fn clean(text: &str) -> Expr {
    let (e, diags) = parse(text);
    assert!(diags.is_empty(), "{text:?}: {diags:#?}");
    e.strip_spans()
}

fn sp(start: usize, end: usize) -> Span {
    Span::new(F, start, end)
}

fn n(text: &str) -> Expr {
    Expr::Num {
        value: text.parse().unwrap(),
        text: text.to_string(),
        span: DUMMY_SPAN,
    }
}

fn ident(text: &str, index: Option<Index>) -> Ident {
    Ident {
        text: text.to_string(),
        key: text.to_ascii_lowercase(),
        index,
        span: DUMMY_SPAN,
    }
}

fn v(text: &str) -> Expr {
    Expr::Name(ident(text, None))
}

fn vi(text: &str, index: Index) -> Expr {
    Expr::Name(ident(text, Some(index)))
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

fn bin(op: B, lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        implicit: false,
        span: DUMMY_SPAN,
    }
}

fn imp(lhs: Expr, rhs: Expr) -> Expr {
    Expr::Binary {
        op: B::Mul,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        implicit: true,
        span: DUMMY_SPAN,
    }
}

fn un(op: UnaryOp, operand: Expr) -> Expr {
    Expr::Unary {
        op,
        operand: Box::new(operand),
        span: DUMMY_SPAN,
    }
}

fn neg(e: Expr) -> Expr {
    un(UnaryOp::Neg, e)
}

fn not(e: Expr) -> Expr {
    un(UnaryOp::Not, e)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        name: ident(name, None),
        args,
        span: DUMMY_SPAN,
    }
}

fn if_(cond: Expr, then: Expr, else_: Expr) -> Expr {
    Expr::If {
        cond: Box::new(cond),
        then: Box::new(then),
        else_: Box::new(else_),
        span: DUMMY_SPAN,
    }
}

fn err() -> Expr {
    Expr::Error { span: DUMMY_SPAN }
}

/// The one diagnostic of `text`.
fn one(text: &str) -> Diagnostic {
    let (_, diags) = parse(text);
    assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
    diags.into_iter().next().unwrap()
}

/// Checks code, severity, span and fix-it of the one diagnostic of `text`.
fn check_one(text: &str, code: Code, span: (usize, usize), fix: Option<((usize, usize), &str)>) {
    let d = one(text);
    assert_eq!(d.code, code, "{text:?}: {d:#?}");
    let severity = CODES.iter().find(|(c, _, _)| *c == code).unwrap().1;
    assert_eq!(d.severity, severity, "{text:?}");
    assert_eq!(d.span, sp(span.0, span.1), "{text:?}: {d:#?}");
    let fix = fix.map(|((s, e), r)| FixIt {
        span: sp(s, e),
        replacement: r.to_string(),
    });
    assert_eq!(d.fix_it, fix, "{text:?}: {d:#?}");
}

// ---------------------------------------------------------------------------
// The measured precedence table in language.md.

/// Each row: text, tree, and whether one precedence warning fires.
fn table() -> Vec<(&'static str, Expr, bool)> {
    let x = || v("x");
    vec![
        (
            "1|0+1",
            bin(B::Add, bin(B::Or, n("1"), n("0")), n("1")),
            true,
        ),
        (
            "2+1|0",
            bin(B::Or, bin(B::Add, n("2"), n("1")), n("0")),
            true,
        ),
        (
            "2*1&1",
            bin(B::And, bin(B::Mul, n("2"), n("1")), n("1")),
            true,
        ),
        (
            "1&1*2",
            bin(B::Mul, bin(B::And, n("1"), n("1")), n("2")),
            true,
        ),
        (
            "3>2>1",
            bin(B::Gt, bin(B::Gt, n("3"), n("2")), n("1")),
            false,
        ),
        (
            "2==2>1",
            bin(B::Gt, bin(B::Eq, n("2"), n("2")), n("1")),
            false,
        ),
        (
            "2^3^2",
            bin(B::Pow, bin(B::Pow, n("2"), n("3")), n("2")),
            true,
        ),
        (
            "2**3**2",
            bin(B::StarStar, bin(B::StarStar, n("2"), n("3")), n("2")),
            true,
        ),
        ("-x^2", neg(bin(B::Pow, x(), n("2"))), false),
        ("-x>2", neg(bin(B::Gt, x(), n("2"))), true),
        ("-x*2", bin(B::Mul, neg(x()), n("2")), false),
        ("-2&1", bin(B::And, neg(n("2")), n("1")), false),
        ("not 0+1", bin(B::Add, not(n("0")), n("1")), false),
        ("not 1>2", not(bin(B::Gt, n("1"), n("2"))), false),
        (
            "2*3>5",
            bin(B::Mul, n("2"), bin(B::Gt, n("3"), n("5"))),
            true,
        ),
        (
            "3>2+5",
            bin(B::Add, bin(B::Gt, n("3"), n("2")), n("5")),
            true,
        ),
        (
            "2^3>9",
            bin(B::Gt, bin(B::Pow, n("2"), n("3")), n("9")),
            false,
        ),
    ]
}

#[test]
fn measured_precedence_table() {
    for (text, expected, warns) in table() {
        let (e, diags) = parse(text);
        assert_eq!(e.strip_spans(), expected, "{text:?}");
        let want = if warns { vec![PRECEDENCE] } else { vec![] };
        let got: Vec<Code> = diags.iter().map(|d| d.code).collect();
        assert_eq!(got, want, "{text:?}: {diags:#?}");
    }
}

#[test]
fn parenthesized_forms_do_not_warn_and_give_the_same_tree() {
    let pairs = [
        ("(1|0)+1", "1|0+1"),
        ("(2+1)|0", "2+1|0"),
        ("(2*1)&1", "2*1&1"),
        ("(1&1)*2", "1&1*2"),
        ("(2^3)^2", "2^3^2"),
        ("(2**3)**2", "2**3**2"),
        ("-(x>2)", "-x>2"),
        ("2*(3>5)", "2*3>5"),
        ("(3>2)+5", "3>2+5"),
    ];
    for (with, without) in pairs {
        assert_eq!(clean(with), tree(without), "{with:?}");
    }
    assert_eq!(clean("(2*x)"), clean("2*x"));
    clean("(2*x)>1");
    clean("2*(x>1)");
    clean("(a&b)+c");
    clean("a&(b+c)");
    clean("2^(3^2)");
}

#[test]
fn comparisons_under_and_or_do_not_warn() {
    assert_eq!(
        clean("x>1&y<2|z==3"),
        bin(
            B::Or,
            bin(
                B::And,
                bin(B::Gt, v("x"), n("1")),
                bin(B::Lt, v("y"), n("2"))
            ),
            bin(B::Eq, v("z"), n("3"))
        )
    );
    clean("a&b|c");
}

#[test]
fn levels_are_left_associative() {
    assert_eq!(
        clean("1-2+3"),
        bin(B::Add, bin(B::Sub, n("1"), n("2")), n("3"))
    );
    assert_eq!(
        clean("8/4*2"),
        bin(B::Mul, bin(B::Div, n("8"), n("4")), n("2"))
    );
    assert_eq!(
        clean("1<=2>=3"),
        bin(B::Ge, bin(B::Le, n("1"), n("2")), n("3"))
    );
    assert_eq!(
        clean("1+2*3"),
        bin(B::Add, n("1"), bin(B::Mul, n("2"), n("3")))
    );
}

#[test]
fn precedence_warning_spans_and_fix_its() {
    // The fix-it wraps the operand that XPPAUT reads as a unit.
    check_one("2*x>1", PRECEDENCE, (0, 5), Some(((2, 5), "(x>1)")));
    check_one("-x>2", PRECEDENCE, (0, 4), Some(((1, 4), "(x>2)")));
    check_one("a|b+c", PRECEDENCE, (0, 5), Some(((0, 3), "(a|b)")));
    check_one("a+b|c", PRECEDENCE, (0, 5), Some(((0, 3), "(a+b)")));
    check_one("a&b+c", PRECEDENCE, (0, 5), Some(((0, 3), "(a&b)")));
    check_one("a+b&c", PRECEDENCE, (0, 5), Some(((2, 5), "(b&c)")));
    check_one("a*b&c", PRECEDENCE, (0, 5), Some(((0, 3), "(a*b)")));
    check_one("a^b^c", PRECEDENCE, (0, 5), Some(((0, 3), "(a^b)")));
    let d = one("2*x>1");
    assert!(d.message.contains("2*(x>1)"), "{}", d.message);
    let d = one("a^b^c");
    assert!(d.message.contains("(a^b)^c"), "{}", d.message);
}

#[test]
fn implicit_multiplication_counts_as_arithmetic_next_to_and() {
    let (e, diags) = parse("2x&1");
    assert_eq!(e.strip_spans(), bin(B::And, imp(n("2"), v("x")), n("1")));
    let got: Vec<Code> = diags.iter().map(|d| d.code).collect();
    assert_eq!(got, vec![PORTABILITY, PRECEDENCE]);
}

// ---------------------------------------------------------------------------
// Prefix operators.

#[test]
fn prefix_at_a_start_is_accepted() {
    assert_eq!(clean("-x"), neg(v("x")));
    assert_eq!(clean("(-3)"), neg(n("3")));
    assert_eq!(clean("f(-1,-2)"), call("f", vec![neg(n("1")), neg(n("2"))]));
    assert_eq!(
        clean("if(-x)then(-1)else(-2)"),
        if_(neg(v("x")), neg(n("1")), neg(n("2")))
    );
    assert_eq!(clean("not(x)"), not(v("x")));
    assert_eq!(clean("not 0"), not(n("0")));
    assert_eq!(clean("(not 0)"), not(n("0")));
}

#[test]
fn prefix_elsewhere_is_an_error_with_a_fix_it() {
    check_one("2*-3", MISPLACED_PREFIX, (2, 4), Some(((2, 4), "(-3)")));
    assert_eq!(tree("2*-3"), bin(B::Mul, n("2"), neg(n("3"))));
    check_one("2+-1", MISPLACED_PREFIX, (2, 4), Some(((2, 4), "(-1)")));
    assert_eq!(tree("2+-1"), bin(B::Add, n("2"), neg(n("1"))));
    check_one("2^-1", MISPLACED_PREFIX, (2, 4), Some(((2, 4), "(-1)")));
    assert_eq!(tree("2^-1"), bin(B::Pow, n("2"), neg(n("1"))));
    check_one("--2", MISPLACED_PREFIX, (1, 3), Some(((1, 3), "(-2)")));
    assert_eq!(tree("--2"), neg(neg(n("2"))));
    check_one("x - -1", MISPLACED_PREFIX, (4, 6), Some(((4, 6), "(-1)")));
    assert_eq!(tree("x - -1"), bin(B::Sub, v("x"), neg(n("1"))));
    check_one(
        "0&not 0",
        MISPLACED_PREFIX,
        (2, 7),
        Some(((2, 7), "(not 0)")),
    );
    assert_eq!(tree("0&not 0"), bin(B::And, n("0"), not(n("0"))));
    check_one(
        "5*not 0",
        MISPLACED_PREFIX,
        (2, 7),
        Some(((2, 7), "(not 0)")),
    );
    check_one(
        "2^not 0",
        MISPLACED_PREFIX,
        (2, 7),
        Some(((2, 7), "(not 0)")),
    );
    assert_eq!(tree("2^not 0"), bin(B::Pow, n("2"), not(n("0"))));
    check_one(
        "-not 0",
        MISPLACED_PREFIX,
        (1, 6),
        Some(((1, 6), "(not 0)")),
    );
    assert_eq!(tree("-not 0"), neg(not(n("0"))));
    check_one(
        "not not 0",
        MISPLACED_PREFIX,
        (4, 9),
        Some(((4, 9), "(not 0)")),
    );
    assert_eq!(tree("not not 0"), not(not(n("0"))));
}

#[test]
fn misplaced_prefix_takes_a_level_four_operand() {
    // The fix-it wraps the prefix operator plus its level-4 operand.
    let (e, diags) = parse("2*-x^2+1");
    assert_eq!(
        e.strip_spans(),
        bin(
            B::Add,
            bin(B::Mul, n("2"), neg(bin(B::Pow, v("x"), n("2")))),
            n("1")
        )
    );
    assert_eq!(diags.len(), 1);
    assert_eq!(
        diags[0].fix_it.as_ref().unwrap().replacement,
        "(-x^2)".to_string()
    );
}

#[test]
fn unary_plus_is_an_error_and_is_ignored() {
    check_one("+x", UNARY_PLUS, (0, 1), Some(((0, 1), "")));
    assert_eq!(tree("+x"), v("x"));
    check_one("2*+3", UNARY_PLUS, (2, 3), Some(((2, 3), "")));
    assert_eq!(tree("2*+3"), bin(B::Mul, n("2"), n("3")));
    // A `-` after an ignored leading `+` is still at the start.
    check_one("+-x", UNARY_PLUS, (0, 1), Some(((0, 1), "")));
    assert_eq!(tree("+-x"), neg(v("x")));
    check_one("2^+3", UNARY_PLUS, (2, 3), Some(((2, 3), "")));
}

// ---------------------------------------------------------------------------
// Blanks.

#[test]
fn blanks_that_split_a_token_are_errors() {
    check_one("2 x", BLANK_IN_TOKEN, (1, 2), Some(((1, 2), "")));
    assert_eq!(tree("2 x"), imp(n("2"), v("x")));
    check_one("a x", BLANK_IN_TOKEN, (1, 2), Some(((1, 2), "")));
    assert_eq!(tree("a x"), imp(v("a"), v("x")));
    check_one("1 2", BLANK_IN_TOKEN, (1, 2), Some(((1, 2), "")));
    check_one("1. 5", BLANK_IN_TOKEN, (2, 3), Some(((2, 3), "")));
    check_one("si n(0)", BLANK_IN_TOKEN, (2, 3), Some(((2, 3), "")));
    assert_eq!(tree("si n(0)"), imp(v("si"), call("n", vec![n("0")])));
    check_one("x * * 2", BLANK_IN_TOKEN, (3, 4), Some(((3, 4), "")));
    assert_eq!(tree("x * * 2"), bin(B::StarStar, v("x"), n("2")));
    check_one("3< =3", BLANK_IN_TOKEN, (2, 3), Some(((2, 3), "")));
    assert_eq!(tree("3< =3"), bin(B::Le, n("3"), n("3")));
    check_one("3>  =3", BLANK_IN_TOKEN, (2, 4), Some(((2, 4), "")));
    assert_eq!(tree("3> =3"), bin(B::Ge, n("3"), n("3")));
    check_one("3= =3", BLANK_IN_TOKEN, (2, 3), Some(((2, 3), "")));
    assert_eq!(tree("3= =3"), bin(B::Eq, n("3"), n("3")));
}

#[test]
fn blank_split_pairs_do_not_also_warn_about_portability() {
    // `2 x` is reported once, as a blank error, not as implicit multiplication.
    assert_eq!(codes("2 x"), vec![BLANK_IN_TOKEN]);
}

#[test]
fn blanks_between_ordinary_tokens_are_fine() {
    assert_eq!(clean(" 1 + 2 * x "), clean("1+2*x"));
    assert_eq!(clean("not 0"), not(n("0")));
    assert_eq!(
        clean("if (x) then (1) else (2)"),
        if_(v("x"), n("1"), n("2"))
    );
    assert_eq!(clean("sin (x)"), call("sin", vec![v("x")]));
    assert_eq!(clean("sin (x)"), clean("sin(x)"));
    assert_eq!(clean("x ** 2"), bin(B::StarStar, v("x"), n("2")));
    assert_eq!(clean("3 <= 3"), bin(B::Le, n("3"), n("3")));
}

// ---------------------------------------------------------------------------
// `!`, `!=` and `=`.

#[test]
fn bang_equals_is_an_error_read_as_equals() {
    check_one("a!=b", BANG_OPERATOR, (1, 3), None);
    assert!(one("a!=b").message.contains("not(a==b)"));
    assert_eq!(tree("a!=b"), bin(B::Eq, v("a"), v("b")));
}

#[test]
fn lone_bang_is_an_error() {
    check_one("!x", BANG_OPERATOR, (0, 1), None);
    assert_eq!(tree("!x"), v("x"));
    check_one("a ! b", BANG_OPERATOR, (2, 3), None);
}

#[test]
fn equals_in_an_expression_is_an_error_read_as_double_equals() {
    check_one("a=b", EQUALS_IN_EXPRESSION, (1, 2), Some(((1, 2), "==")));
    assert_eq!(tree("a=b"), bin(B::Eq, v("a"), v("b")));
}

// ---------------------------------------------------------------------------
// Implicit multiplication.

#[test]
fn implicit_multiplication_binds_at_level_two() {
    let (e, diags) = parse("2v^2(1-w)");
    assert_eq!(
        e.strip_spans(),
        imp(
            imp(n("2"), bin(B::Pow, v("v"), n("2"))),
            bin(B::Sub, n("1"), v("w"))
        )
    );
    let got: Vec<Code> = diags.iter().map(|d| d.code).collect();
    assert_eq!(got, vec![PORTABILITY, PORTABILITY]);
    assert_eq!(tree("2P.E."), imp(n("2"), v("P.E.")));
    assert_eq!(tree("2exp(x)"), imp(n("2"), call("exp", vec![v("x")])));
    assert_eq!(tree("2e-3x"), imp(n("2e-3"), v("x")));
    assert_eq!(tree("3x+1"), bin(B::Add, imp(n("3"), v("x")), n("1")));
    assert_eq!(tree("-2x"), imp(neg(n("2")), v("x")));
}

#[test]
fn implicit_multiplication_warning() {
    check_one("2x", PORTABILITY, (0, 2), Some(((1, 1), "*")));
    let d = one("2x");
    assert!(d.message.contains("XPPAUT 8.0 rejects"), "{}", d.message);
    assert!(
        d.message.contains("implicit multiplication"),
        "{}",
        d.message
    );
    check_one("2(x)", PORTABILITY, (0, 4), Some(((1, 1), "*")));
    check_one("2e-3x", PORTABILITY, (0, 5), Some(((4, 4), "*")));
}

#[test]
fn implicit_multiplication_after_a_division_says_how_it_reads() {
    let d = one("a/2v");
    assert_eq!(d.code, PORTABILITY);
    assert_eq!(d.fix_it.as_ref().unwrap().span, sp(3, 3));
    assert!(d.message.contains("(a/2)*v"), "{}", d.message);
    assert_eq!(tree("a/2v"), imp(bin(B::Div, v("a"), n("2")), v("v")));
    let d = one("a/2(b+c)");
    assert!(d.message.contains("(a/2)*(b+c)"), "{}", d.message);
    assert!(!one("a*2v").message.contains("reads as"));
}

#[test]
fn only_a_number_multiplies_implicitly() {
    assert_eq!(clean("x(1)"), call("x", vec![n("1")]));
    check_one("2 (x)", MISSING_OPERATOR, (2, 3), Some(((2, 2), "*")));
    check_one("(a)(b)", MISSING_OPERATOR, (3, 4), Some(((3, 3), "*")));
    assert_eq!(tree("(a)(b)"), imp(v("a"), v("b")));
    check_one("x[j] y", MISSING_OPERATOR, (5, 6), Some(((5, 5), "*")));
}

// ---------------------------------------------------------------------------
// Calls and if.

#[test]
fn calls_take_any_number_of_arguments() {
    assert_eq!(clean("f()"), call("f", vec![]));
    assert_eq!(clean("Max(1,2,3)"), {
        let mut e = call("Max", vec![n("1"), n("2"), n("3")]);
        if let Expr::Call { name, .. } = &mut e {
            assert_eq!(name.key, "max");
        }
        e
    });
    assert_eq!(
        clean("f(g(x),y+1)"),
        call(
            "f",
            vec![call("g", vec![v("x")]), bin(B::Add, v("y"), n("1"))]
        )
    );
}

#[test]
fn if_is_an_atom() {
    assert_eq!(
        clean("1+if(x>0)then(x)else(-x)*2"),
        bin(
            B::Add,
            n("1"),
            bin(
                B::Mul,
                if_(bin(B::Gt, v("x"), n("0")), v("x"), neg(v("x"))),
                n("2")
            )
        )
    );
    assert_eq!(clean("IF(x)THEN(1)ELSE(2)"), clean("if(x)then(1)else(2)"));
}

#[test]
fn malformed_if_is_one_error() {
    check_one("if(x)then(1)", MALFORMED_IF, (12, 12), None);
    check_one("if x", MALFORMED_IF, (3, 4), None);
    check_one("if(x)1 else(2)", MALFORMED_IF, (5, 6), None);
    check_one("if(x)then(1)els(2)", MALFORMED_IF, (12, 15), None);
    assert_eq!(tree("if(x)then(1)"), if_(v("x"), n("1"), err()));
}

// ---------------------------------------------------------------------------
// Index forms.

#[test]
fn index_forms_after_a_name() {
    assert_eq!(clean("x[j]"), vi("x", var(IndexOp::None, 0)));
    assert_eq!(clean("x[j+1]"), vi("x", var(IndexOp::Add, 1)));
    assert_eq!(clean("x[j-1]"), vi("x", var(IndexOp::Sub, 1)));
    assert_eq!(clean("x[J*2]"), vi("x", var(IndexOp::Mul, 2)));
    assert_eq!(clean("x[ j + 1 ]"), vi("x", var(IndexOp::Add, 1)));
    assert_eq!(clean("x[1..10]"), vi("x", range(1, 10)));
    assert_eq!(clean("x[10..1]"), vi("x", range(10, 1)));
    assert_eq!(clean("x[1 .. 3]"), vi("x", range(1, 3)));
    assert_eq!(
        clean("x[j]-x[j-1]"),
        bin(
            B::Sub,
            vi("x", var(IndexOp::None, 0)),
            vi("x", var(IndexOp::Sub, 1))
        )
    );
    assert_eq!(
        clean("f[j](x)"),
        Expr::Call {
            name: ident("f", Some(var(IndexOp::None, 0))),
            args: vec![v("x")],
            span: DUMMY_SPAN,
        }
    );
}

#[test]
fn bare_index_forms_are_operands() {
    assert_eq!(clean("[j]"), Expr::Index(var(IndexOp::None, 0)));
    assert_eq!(
        clean("2*[j+1]"),
        bin(B::Mul, n("2"), Expr::Index(var(IndexOp::Add, 1)))
    );
}

#[test]
fn index_spans() {
    let (e, _) = parse("ab[j+1]");
    let Expr::Name(id) = e else { panic!() };
    assert_eq!(id.span, sp(0, 7));
    assert_eq!(id.index.unwrap().span(), sp(2, 7));
}

#[test]
fn unsupported_index_forms() {
    for (text, end) in [
        ("x[-1]", 5),
        ("x[k]", 4),
        ("x[j/2]", 6),
        ("x[2+j]", 6),
        ("x[1.5]", 6),
        ("x[1.0]", 6),
        ("x[]", 3),
        ("x[j+k]", 6),
        ("x[j+1+2]", 8),
        ("x[1..]", 6),
        ("x[1e2]", 6),
    ] {
        check_one(text, UNSUPPORTED_INDEX, (1, end), None);
        assert_eq!(tree(text), v("x"), "{text:?}");
    }
    // Recovery skips to the matching `]` and goes on.
    assert_eq!(
        codes("x[k]+y[k]"),
        vec![UNSUPPORTED_INDEX, UNSUPPORTED_INDEX]
    );
    check_one("[k]", UNSUPPORTED_INDEX, (0, 3), None);
    assert_eq!(tree("[k]"), err());
}

#[test]
fn unclosed_brackets() {
    check_one("x[j", UNCLOSED_DELIMITER, (1, 2), None);
    check_one("(1+2", UNCLOSED_DELIMITER, (0, 1), None);
    check_one("f(1,2", UNCLOSED_DELIMITER, (1, 2), None);
    assert_eq!(tree("(1+2"), bin(B::Add, n("1"), n("2")));
    assert_eq!(tree("f(1,2"), call("f", vec![n("1"), n("2")]));
}

// ---------------------------------------------------------------------------
// Missing operands, stray tokens and recovery.

#[test]
fn missing_operands() {
    check_one("2*", EXPECTED_EXPRESSION, (2, 2), None);
    assert_eq!(tree("2*"), bin(B::Mul, n("2"), err()));
    check_one("*3", EXPECTED_EXPRESSION, (0, 1), None);
    assert_eq!(tree("*3"), bin(B::Mul, err(), n("3")));
    check_one("()", EXPECTED_EXPRESSION, (1, 2), None);
    assert_eq!(tree("()"), err());
    check_one("", EXPECTED_EXPRESSION, (0, 0), None);
    check_one("-", EXPECTED_EXPRESSION, (1, 1), None);
    check_one("f(1,)", EXPECTED_EXPRESSION, (4, 5), None);
    assert_eq!(tree("f(1,)"), call("f", vec![n("1"), err()]));
}

#[test]
fn unknown_tokens_are_not_reported_again() {
    // The lexer reports the stray `$` and the malformed number; the parser
    // adds nothing.
    let (e, diags) = parse("2*$");
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, crate::lexer::STRAY_CHARACTER);
    assert_eq!(e.strip_spans(), bin(B::Mul, n("2"), err()));
    let (_, diags) = parse("2 $ 3");
    assert_eq!(diags.len(), 1);
    let (_, diags) = parse("1_000+x");
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, crate::lexer::MALFORMED_NUMBER);
}

#[test]
fn unexpected_tokens() {
    check_one("a'b", UNEXPECTED_TOKEN, (1, 2), None);
    assert_eq!(tree("a'b"), v("a"));
    check_one("a % 2", UNEXPECTED_TOKEN, (2, 3), None);
    check_one("(a'b)+1", UNEXPECTED_TOKEN, (2, 3), None);
    assert_eq!(tree("(a'b)+1"), bin(B::Add, v("a"), n("1")));
    check_one("(a,b)", UNEXPECTED_TOKEN, (2, 3), None);
    // A leftover stop token is reported by parse_expr_text.
    check_one("1+2)", UNEXPECTED_TOKEN, (3, 4), None);
    check_one("1+2, 3", UNEXPECTED_TOKEN, (3, 4), None);
}

#[test]
fn one_expression_reports_every_independent_mistake() {
    let text = "2*-3 + (a!=b) + +c + 2 x";
    assert_eq!(
        codes(text),
        vec![MISPLACED_PREFIX, BANG_OPERATOR, UNARY_PLUS, BLANK_IN_TOKEN]
    );
    let text = "x * * 2 + 5*not 0 + y[k] + (a=b) + 2v";
    assert_eq!(
        codes(text),
        vec![
            BLANK_IN_TOKEN,
            MISPLACED_PREFIX,
            UNSUPPORTED_INDEX,
            EQUALS_IN_EXPRESSION,
            PORTABILITY
        ]
    );
    let text = "f(2*, -x^-1, (1+2)";
    assert_eq!(
        codes(text),
        vec![EXPECTED_EXPRESSION, MISPLACED_PREFIX, UNCLOSED_DELIMITER]
    );
}

/// Parses with the slice parser from `pos` and returns the tree, the
/// position where it stopped and the diagnostics.
fn slice(text: &str, pos: usize) -> (Expr, usize, Vec<Diagnostic>) {
    let (tokens, _) = lex(F, text);
    let mut diags = Vec::new();
    let mut c = ExprCursor {
        tokens: &tokens,
        pos,
        text,
        diags: &mut diags,
    };
    let e = parse_expr(&mut c);
    let stop = c.pos;
    (e.strip_spans(), stop, diags)
}

#[test]
fn slice_parser_stops_before_a_stop_token() {
    let cases: [(&str, usize, TokenKind); 8] = [
        ("1+2, 3", 3, TokenKind::Comma),
        ("1+2) 3", 3, TokenKind::RParen),
        ("x] 3", 1, TokenKind::RBracket),
        ("x} 3", 1, TokenKind::RBrace),
        ("x-1 {x=0}", 3, TokenKind::LBrace),
        ("x-1\ny", 3, TokenKind::Newline),
        ("x-1 # c", 3, TokenKind::Comment),
        ("x-1", 3, TokenKind::Eof),
    ];
    for (text, stop, kind) in cases {
        let (tokens, _) = lex(F, text);
        let (_, at, diags) = slice(text, 0);
        assert_eq!(at, stop, "{text:?}");
        assert_eq!(tokens[at].kind, kind, "{text:?}");
        assert!(diags.is_empty(), "{text:?}: {diags:#?}");
    }
    // Inside parentheses a comma or brace does not end the expression early.
    let (e, at, diags) = slice("f(1,2),3", 0);
    assert_eq!(e, call("f", vec![n("1"), n("2")]));
    assert_eq!(at, 6);
    assert!(diags.is_empty());
}

#[test]
fn slice_parser_start_counts_as_a_prefix_position() {
    // `x=-1` from a statement: the parser starts after `=`.
    let (e, at, diags) = slice("x=-1", 2);
    assert_eq!(e, neg(n("1")));
    assert_eq!(at, 4);
    assert!(diags.is_empty(), "{diags:#?}");
    let (_, _, diags) = slice("{x=-1}", 3);
    assert!(diags.is_empty(), "{diags:#?}");
}

#[test]
fn slice_parser_skips_junk_to_the_stop_token() {
    let (e, at, diags) = slice("a'b(c,d), 3", 0);
    assert_eq!(e, v("a"));
    assert_eq!(at, 8);
    assert_eq!(diags.len(), 1);
}

#[test]
fn slice_parser_survives_an_empty_or_short_token_list() {
    let mut diags = Vec::new();
    let mut c = ExprCursor {
        tokens: &[],
        pos: 0,
        text: "",
        diags: &mut diags,
    };
    assert!(parse_expr(&mut c).is_error());
    let (tokens, _) = lex(F, "1");
    let mut c = ExprCursor {
        tokens: &tokens,
        pos: 7,
        text: "1",
        diags: &mut diags,
    };
    assert!(parse_expr(&mut c).is_error());
}

#[test]
fn codes_are_registered_in_order() {
    let ours = [
        (
            EXPECTED_EXPRESSION,
            4,
            Severity::Error,
            "expected an expression",
        ),
        (
            MISSING_OPERATOR,
            5,
            Severity::Error,
            "missing operator between two operands",
        ),
        (
            UNEXPECTED_TOKEN,
            6,
            Severity::Error,
            "unexpected token in an expression",
        ),
        (BANG_OPERATOR, 7, Severity::Error, "`!` is not an operator"),
        (
            EQUALS_IN_EXPRESSION,
            8,
            Severity::Error,
            "`=` inside an expression",
        ),
        (
            UNCLOSED_DELIMITER,
            9,
            Severity::Error,
            "unclosed `(` or `[`",
        ),
        (
            MISPLACED_PREFIX,
            10,
            Severity::Error,
            "prefix operator in the wrong place",
        ),
        (UNARY_PLUS, 11, Severity::Error, "unary `+`"),
        (BLANK_IN_TOKEN, 12, Severity::Error, "blank inside a token"),
        (
            PORTABILITY,
            13,
            Severity::Warning,
            "XPPAUT 8.0 rejects this",
        ),
        (
            PRECEDENCE,
            14,
            Severity::Warning,
            "operator precedence differs from C",
        ),
        (
            UNSUPPORTED_INDEX,
            15,
            Severity::Error,
            "unsupported index form",
        ),
        (MALFORMED_IF, 16, Severity::Error, "malformed if expression"),
    ];
    for (code, number, severity, summary) in ours {
        assert_eq!(code, Code(number));
        let entry = CODES.iter().find(|(c, _, _)| *c == code).unwrap();
        assert_eq!(entry.1, severity, "{code}");
        assert_eq!(entry.2, summary, "{code}");
    }
}

/// Every diagnostic and fix-it span lies inside the text, every node span
/// too, and parsing never panics.
fn check_spans(text: &str) {
    let (e, diags) = parse(text);
    let inside = |s: Span| {
        s.file == F
            && s.start <= s.end
            && s.end <= text.len()
            && text.is_char_boundary(s.start)
            && text.is_char_boundary(s.end)
    };
    for d in &diags {
        assert!(inside(d.span), "{text:?}: {d:#?}");
        if let Some(fix) = &d.fix_it {
            assert!(inside(fix.span), "{text:?}: {d:#?}");
        }
    }
    fn walk(e: &Expr, f: &dyn Fn(Span) -> bool, text: &str) {
        assert!(f(e.span()), "{text:?}: {e:#?}");
        match e {
            Expr::Unary { operand, .. } => walk(operand, f, text),
            Expr::Binary { lhs, rhs, .. } => {
                walk(lhs, f, text);
                walk(rhs, f, text);
            }
            Expr::Call { args, .. } => args.iter().for_each(|a| walk(a, f, text)),
            Expr::If {
                cond, then, else_, ..
            } => {
                walk(cond, f, text);
                walk(then, f, text);
                walk(else_, f, text);
            }
            _ => {}
        }
    }
    walk(&e, &inside, text);
}

#[test]
fn parser_keeps_spans_inside_the_text_on_every_short_string() {
    let alphabet = [
        "x", "2", "-", "+", "*", "(", ")", "[", "]", "j", "..", ",", " ", "not ", "if", "=", "!",
        "<", "^", "'", "$", "é",
    ];
    let mut frontier = vec![String::new()];
    check_spans("");
    for _ in 0..4 {
        let mut next = Vec::new();
        for s in &frontier {
            for a in alphabet {
                let mut t = s.clone();
                t.push_str(a);
                check_spans(&t);
                next.push(t);
            }
        }
        frontier = next;
    }
}

#[test]
fn deep_nesting_is_an_error_not_a_crash() {
    let text = format!("{}x{}", "(".repeat(500), ")".repeat(500));
    let (_, diags) = parse(&text);
    assert!(diags.iter().any(|d| d.code == UNEXPECTED_TOKEN));
    let text = format!("2^{}1", "-".repeat(5000));
    let (_, diags) = parse(&text);
    assert!(!diags.is_empty());
    clean(&format!("{}x{}", "(".repeat(50), ")".repeat(50)));
}

#[test]
fn applying_a_fix_it_removes_the_diagnostic() {
    for text in [
        "2*-3", "2 x", "x * * 2", "3< =3", "+x", "a=b", "2x", "2*x>1", "a|b+c", "a^b^c", "2 (x)",
    ] {
        let d = one(text);
        let fix = d.fix_it.unwrap();
        let fixed = format!(
            "{}{}{}",
            &text[..fix.span.start],
            fix.replacement,
            &text[fix.span.end..]
        );
        let (_, after) = parse(&fixed);
        assert!(
            after.iter().all(|a| a.code != d.code),
            "{text:?} -> {fixed:?}: {after:#?}"
        );
    }
}
