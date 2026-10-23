use super::*;
use crate::ast::{BinaryOp as B, DUMMY_SPAN, Expr, IndexOp};
use crate::diag::CODES;
use crate::lexer::STRAY_CHARACTER;

const F: FileId = FileId(0);

fn parse_all(text: &str) -> (Model, Vec<Diagnostic>) {
    parse(text, F)
}

fn codes(text: &str) -> Vec<Code> {
    parse_all(text).1.iter().map(|d| d.code).collect()
}

/// The statements of `text`, without spans.
fn stmts(text: &str) -> Vec<Stmt> {
    parse_all(text).0.strip_spans().statements
}

/// The statement kinds of `text`, which must parse with no diagnostic.
fn clean(text: &str) -> Vec<StmtKind> {
    let (model, diags) = parse_all(text);
    assert!(diags.is_empty(), "{text:?}: {diags:#?}");
    model
        .strip_spans()
        .statements
        .into_iter()
        .map(|s| s.kind)
        .collect()
}

/// The one statement kind of `text`, which must parse cleanly.
fn clean_one(text: &str) -> StmtKind {
    let kinds = clean(text);
    assert_eq!(kinds.len(), 1, "{text:?}: {kinds:#?}");
    kinds.into_iter().next().unwrap()
}

fn one(text: &str) -> Diagnostic {
    let (_, diags) = parse_all(text);
    assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
    diags.into_iter().next().unwrap()
}

fn sp(start: usize, end: usize) -> Span {
    Span::new(F, start, end)
}

/// The byte range of the first `needle` in `text`.
fn at(text: &str, needle: &str) -> (usize, usize) {
    let start = text.find(needle).unwrap_or_else(|| panic!("{needle:?}"));
    (start, start + needle.len())
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
    assert!(d.message.len() <= 340, "{text:?}: {}", d.message);
}

fn ident(text: &str) -> Ident {
    Ident {
        text: text.to_string(),
        key: text.to_ascii_lowercase(),
        index: None,
        span: DUMMY_SPAN,
    }
}

fn ident_j(text: &str) -> Ident {
    Ident {
        index: Some(Index::Var {
            op: IndexOp::None,
            n: 0,
            span: DUMMY_SPAN,
        }),
        ..ident(text)
    }
}

fn n(text: &str) -> Expr {
    Expr::Num {
        value: text.parse().unwrap(),
        text: text.to_string(),
        span: DUMMY_SPAN,
    }
}

fn v(text: &str) -> Expr {
    Expr::Name(ident(text))
}

fn neg(e: Expr) -> Expr {
    Expr::Unary {
        op: crate::ast::UnaryOp::Neg,
        operand: Box::new(e),
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

fn num(text: &str) -> Value {
    Value::Num {
        value: text.parse().unwrap(),
        text: text.to_string(),
        span: DUMMY_SPAN,
    }
}

fn entry(name: &str, value: Option<Value>) -> Entry {
    Entry {
        name: ident(name),
        value,
        span: DUMMY_SPAN,
    }
}

fn raw(name: &str, value: &str) -> RawAssign {
    RawAssign {
        name: ident(name),
        value: value.to_string(),
        value_span: DUMMY_SPAN,
        span: DUMMY_SPAN,
    }
}

fn comment(text: &str) -> Comment {
    Comment {
        text: text.to_string(),
        span: DUMMY_SPAN,
    }
}

fn row(entries: Vec<Expr>) -> MarkovRow {
    MarkovRow {
        entries,
        comment: None,
        span: DUMMY_SPAN,
    }
}

fn unsupported(family: Unsupported, text: &str) -> StmtKind {
    StmtKind::Unsupported {
        family,
        text: text.to_string(),
    }
}

/// The kinds of `text`, ignoring diagnostics.
fn kinds(text: &str) -> Vec<StmtKind> {
    stmts(text).into_iter().map(|s| s.kind).collect()
}

// ---------------------------------------------------------------------------
// Statement forms.

#[test]
fn formulas() {
    assert_eq!(
        clean_one("x'=-x"),
        StmtKind::Ode {
            name: ident("x"),
            form: OdeForm::Prime,
            rhs: neg(v("x")),
        }
    );
    assert_eq!(
        clean_one("dx/dt = -x"),
        StmtKind::Ode {
            name: ident("x"),
            form: OdeForm::Ddt,
            rhs: neg(v("x")),
        }
    );
    assert_eq!(
        clean_one("x(t+1)=a*x"),
        StmtKind::Map {
            name: ident("x"),
            rhs: bin(B::Mul, v("a"), v("x")),
        }
    );
    assert_eq!(
        clean_one("x(0)=0.5"),
        StmtKind::InitValue {
            name: ident("x"),
            value: num("0.5"),
        }
    );
    assert_eq!(
        clean_one("!c=a*2"),
        StmtKind::Derived {
            name: ident("c"),
            rhs: bin(B::Mul, v("a"), n("2")),
        }
    );
    assert_eq!(
        clean_one("w=a+1"),
        StmtKind::Fixed {
            name: ident("w"),
            rhs: bin(B::Add, v("a"), n("1")),
        }
    );
    assert_eq!(
        clean_one("f(u,v)=u+v"),
        StmtKind::Function {
            name: ident("f"),
            params: vec![ident("u"), ident("v")],
            rhs: bin(B::Add, v("u"), v("v")),
        }
    );
}

#[test]
fn ddt_needs_a_letter_after_d_and_no_index() {
    // `d2/dt=1` is not a d/dt form: `d2` is a name and `/` is not a left side.
    assert_eq!(codes("d2/dt=1"), vec![UNRECOGNIZED_STATEMENT]);
    // `dx[j]/dt` is not a d/dt form either.
    assert_eq!(codes("dx[j]/dt=1"), vec![UNRECOGNIZED_STATEMENT]);
}

#[test]
fn keyword_letter_names_are_formulas() {
    let fixed = |name: &str, value: &str| StmtKind::Fixed {
        name: ident(name),
        rhs: n(value),
    };
    assert_eq!(
        clean_one("p'=1"),
        StmtKind::Ode {
            name: ident("p"),
            form: OdeForm::Prime,
            rhs: n("1"),
        }
    );
    assert_eq!(clean_one("a=1"), fixed("a", "1"));
    assert_eq!(clean_one("d=3"), fixed("d", "3"));
    assert_eq!(clean_one("p=5"), fixed("p", "5"));
    assert_eq!(clean_one("p = 5"), fixed("p", "5"));
    assert_eq!(clean_one("w = 1"), fixed("w", "1"));
    assert_eq!(clean_one("par = 1"), fixed("par", "1"));
    assert_eq!(
        clean_one("t(0)=1"),
        StmtKind::InitValue {
            name: ident("t"),
            value: num("1"),
        }
    );
}

#[test]
fn every_keyword_abbreviation() {
    let par = StmtKind::Par {
        entries: vec![entry("a", Some(num("1")))],
    };
    for word in ["p", "par", "param", "pzz", "PAR"] {
        assert_eq!(clean_one(&format!("{word} a=1")), par, "{word}");
    }
    let init = StmtKind::Init {
        entries: vec![entry("x", Some(num("1")))],
    };
    for word in ["i", "init", "ic"] {
        assert_eq!(clean_one(&format!("{word} x=1")), init, "{word}");
    }
    let aux = StmtKind::Aux {
        name: ident("e"),
        rhs: v("x"),
    };
    for word in ["a", "aux", "ax"] {
        assert_eq!(clean_one(&format!("{word} e=x")), aux, "{word}");
    }
    let wiener = StmtKind::Wiener {
        names: vec![ident("w1")],
    };
    for word in ["w", "wiener"] {
        assert_eq!(clean_one(&format!("{word} w1")), wiener, "{word}");
    }
    for word in ["t", "table", "tab"] {
        assert!(matches!(
            clean_one(&format!("{word} f % 2 0 1 t")),
            StmtKind::Table { .. }
        ));
    }
    for word in ["g", "global"] {
        assert!(matches!(
            clean_one(&format!("{word} 1 x {{x=0}}")),
            StmtKind::Global { .. }
        ));
    }
    for word in ["m", "markov"] {
        assert!(matches!(
            clean_one(&format!("{word} z 1\n{{0}}")),
            StmtKind::Markov { .. }
        ));
    }
    for word in ["d", "done"] {
        assert_eq!(
            clean(&format!("x'=1\n{word}")).last(),
            Some(&StmtKind::Done)
        );
    }
    for word in ["b", "bndry"] {
        assert_eq!(
            kinds(&format!("{word} x-1")),
            vec![unsupported(Unsupported::Boundary, &format!("{word} x-1"))]
        );
    }
    // The full-spelling words do not abbreviate.
    assert_eq!(codes("num c=3"), vec![UNRECOGNIZED_STATEMENT]);
    assert_eq!(codes("se f {a=1}"), vec![UNRECOGNIZED_STATEMENT]);
}

#[test]
fn entry_lists() {
    assert_eq!(
        clean_one("par a=1, b=-2  c=1e-3,d=[j]"),
        StmtKind::Par {
            entries: vec![
                entry("a", Some(num("1"))),
                entry("b", Some(num("-2"))),
                entry("c", Some(num("1e-3"))),
                entry(
                    "d",
                    Some(Value::Index(Index::Var {
                        op: IndexOp::None,
                        n: 0,
                        span: DUMMY_SPAN
                    }))
                ),
            ],
        }
    );
    // `par` allows a bare name (a parameter with value 0).
    assert_eq!(
        clean_one("par a, b=2"),
        StmtKind::Par {
            entries: vec![entry("a", None), entry("b", Some(num("2")))],
        }
    );
    assert_eq!(
        clean_one("number c=3,e=2.5"),
        StmtKind::Number {
            entries: vec![entry("c", Some(num("3"))), entry("e", Some(num("2.5")))],
        }
    );
    assert_eq!(
        clean_one("init x=1 y=2"),
        StmtKind::Init {
            entries: vec![entry("x", Some(num("1"))), entry("y", Some(num("2")))],
        }
    );
    assert_eq!(
        clean_one("par x[1..2]=1"),
        StmtKind::Par {
            entries: vec![Entry {
                name: Ident {
                    index: Some(Index::Range {
                        start: 1,
                        end: 2,
                        span: DUMMY_SPAN
                    }),
                    ..ident("x")
                },
                value: Some(num("1")),
                span: DUMMY_SPAN,
            }],
        }
    );
}

#[test]
fn literal_values_keep_their_text() {
    let StmtKind::Par { entries } = clean_one("par a=1.50E+2") else {
        panic!();
    };
    assert_eq!(entries[0].value, Some(num("1.50E+2")));
}

#[test]
fn trailing_comments_after_par_number_and_options_are_kept() {
    let text = "par a=1 # rate";
    let s = &parse_all(text).0.statements[0];
    assert!(parse_all(text).1.is_empty());
    assert_eq!(s.comment.as_ref().unwrap().text, "# rate");
    assert_eq!(s.comment.as_ref().unwrap().span, sp(8, 14));
    assert_eq!(s.span, sp(0, 14));
    for text in ["number c=3 # c", "@ dt=0.1 # step"] {
        let (model, diags) = parse_all(text);
        assert!(diags.is_empty(), "{text:?}: {diags:#?}");
        assert!(model.statements[0].comment.is_some(), "{text:?}");
    }
}

#[test]
fn options_and_set() {
    assert_eq!(
        clean_one("@ total=100, dt=.1 maxstor=5000,meth=stiff"),
        StmtKind::Options {
            entries: vec![
                raw("total", "100"),
                raw("dt", ".1"),
                raw("maxstor", "5000"),
                raw("meth", "stiff"),
            ],
        }
    );
    assert_eq!(
        clean_one("set fast {a=10,b=2}"),
        StmtKind::Set {
            name: ident("fast"),
            entries: vec![raw("a", "10"), raw("b", "2")],
        }
    );
    let text = "@ xp=x";
    let (model, _) = parse_all(text);
    let StmtKind::Options { entries } = &model.statements[0].kind else {
        panic!();
    };
    assert_eq!(entries[0].value_span, sp(5, 6));
    assert_eq!(entries[0].span, sp(2, 6));
}

#[test]
fn global_markov_wiener_include_comment() {
    assert_eq!(
        clean_one("global -1 x-1 {x=0; y=y+1}"),
        StmtKind::Global {
            sign: -1,
            cond: bin(B::Sub, v("x"), n("1")),
            assigns: vec![
                Assign {
                    name: ident("x"),
                    rhs: n("0"),
                    span: DUMMY_SPAN,
                },
                Assign {
                    name: ident("y"),
                    rhs: bin(B::Add, v("y"), n("1")),
                    span: DUMMY_SPAN,
                },
            ],
        }
    );
    for (text, sign) in [("global 0 x {x=0}", 0), ("global 1 x {x=0}", 1)] {
        assert!(matches!(clean_one(text), StmtKind::Global { sign: s, .. } if s == sign));
    }
    let text = "markov z 2\n{0} {a}\n# note\n{b} {0}\nx'=z";
    let kinds = clean(text);
    assert_eq!(
        kinds[0],
        StmtKind::Markov {
            name: ident("z"),
            states: 2,
            lines: vec![
                MarkovLine::Row(row(vec![n("0"), v("a")])),
                MarkovLine::Comment(comment("# note")),
                MarkovLine::Row(row(vec![v("b"), n("0")])),
            ],
        }
    );
    // The comment between the rows stays inside the statement.
    assert_eq!(kinds.len(), 2);
    assert!(matches!(kinds[1], StmtKind::Ode { .. }));
    let model = parse_all(text).0;
    assert_eq!(model.statements[0].span, sp(0, at(text, "{b} {0}").1));
    assert_eq!(
        clean_one("wiener w1, w2 w3"),
        StmtKind::Wiener {
            names: vec![ident("w1"), ident("w2"), ident("w3")],
        }
    );
    let text = "#include  lib/other.ode ";
    assert_eq!(
        clean_one(text),
        StmtKind::Include {
            path: "lib/other.ode".to_string(),
            path_span: DUMMY_SPAN,
        }
    );
    let StmtKind::Include { path_span, .. } = &parse_all(text).0.statements[0].kind else {
        panic!();
    };
    assert_eq!(*path_span, sp(10, 23));
    assert_eq!(
        clean_one("# just a note"),
        StmtKind::Comment(comment("# just a note"))
    );
    // `#includes` is a comment.
    assert!(matches!(clean_one("#includes"), StmtKind::Comment(_)));
}

#[test]
fn table_forms() {
    assert_eq!(
        clean_one("table f % 51 -1 1 t^2"),
        StmtKind::Table {
            name: ident("f"),
            npts: num("51"),
            lo: num("-1"),
            hi: num("1"),
            rhs: bin(B::Pow, v("t"), n("2")),
        }
    );
    assert_eq!(
        kinds("table f data.tab"),
        vec![unsupported(Unsupported::FileTable, "table f data.tab")]
    );
}

#[test]
fn blocks() {
    let text = "%[1..3]\nx[j]'=-x[j]\n%\ny'=1";
    let kinds = clean(text);
    assert_eq!(
        kinds[0],
        StmtKind::Block {
            start: 1,
            end: 3,
            range_span: DUMMY_SPAN,
            body: vec![Stmt {
                kind: StmtKind::Ode {
                    name: ident_j("x"),
                    form: OdeForm::Prime,
                    rhs: neg(Expr::Name(ident_j("x"))),
                },
                comment: None,
                span: DUMMY_SPAN,
                array_index: None,
            }],
        }
    );
    assert!(matches!(kinds[1], StmtKind::Ode { .. }));
    let model = parse_all(text).0;
    assert_eq!(model.statements[0].span, sp(0, 21));
    let StmtKind::Block { range_span, .. } = &model.statements[0].kind else {
        panic!();
    };
    assert_eq!(*range_span, sp(1, 7));
    // Negative bounds are kept for Task 8 to reject.
    assert!(matches!(
        clean_one("%[-1..2]\n%"),
        StmtKind::Block {
            start: -1,
            end: 2,
            ..
        }
    ));
}

#[test]
fn done_ends_the_file() {
    let text = "x'=1\ndone\n$$ not ode @ junk";
    assert_eq!(
        clean(text),
        vec![
            StmtKind::Ode {
                name: ident("x"),
                form: OdeForm::Prime,
                rhs: n("1"),
            },
            StmtKind::Done,
        ]
    );
    assert_eq!(parse_all(text).0.statements[1].span, sp(5, 9));
}

#[test]
fn every_statement_span_covers_its_text() {
    let text = "x'=1\n  par a=1  \n@ dt=1\n!b=2\n";
    let model = parse_all(text).0;
    let spans: Vec<Span> = model.statements.iter().map(|s| s.span).collect();
    assert_eq!(spans, vec![sp(0, 4), sp(7, 14), sp(17, 23), sp(24, 28)]);
}

#[test]
fn crlf_lines() {
    assert_eq!(clean("x'=1\r\npar a=1\r\n").len(), 2);
}

// ---------------------------------------------------------------------------
// Unsupported constructs.

#[test]
fn unsupported_families() {
    let cases = [
        ("volt x=1", Unsupported::Volterra, UNSUPPORTED_VOLTERRA),
        (
            "x(t)=1+int{exp(-t)#x}",
            Unsupported::Volterra,
            UNSUPPORTED_VOLTERRA,
        ),
        (
            "u=int[1]{exp(-t)#v}",
            Unsupported::Volterra,
            UNSUPPORTED_VOLTERRA,
        ),
        (
            "aux q=int{exp(-t)#v}",
            Unsupported::Volterra,
            UNSUPPORTED_VOLTERRA,
        ),
        ("bndry x-1", Unsupported::Boundary, UNSUPPORTED_BOUNDARY),
        ("solv z=1", Unsupported::Dae, UNSUPPORTED_DAE),
        (
            "special k=conv(x,1,w,u)",
            Unsupported::Special,
            UNSUPPORTED_SPECIAL,
        ),
        ("export {x} {y}", Unsupported::Export, UNSUPPORTED_EXPORT),
        ("only x y", Unsupported::Only, UNSUPPORTED_ONLY),
        (
            "options my.opt",
            Unsupported::OptionsFile,
            UNSUPPORTED_OPTIONS_FILE,
        ),
        ("hom_bcs", Unsupported::HomBcs, UNSUPPORTED_HOM_BCS),
        (
            "table f data.tab",
            Unsupported::FileTable,
            UNSUPPORTED_FILE_TABLE,
        ),
        (
            "z=sum(0,4)of(x[i'])",
            Unsupported::SumOf,
            UNSUPPORTED_SUM_OF,
        ),
        (
            "\" {x=1} set x",
            Unsupported::ActiveComment,
            UNSUPPORTED_ACTIVE_COMMENT,
        ),
    ];
    for (text, family, code) in cases {
        assert_eq!(kinds(text), vec![unsupported(family, text)], "{text:?}");
        check_one(text, code, (0, text.len()), None);
        assert!(one(text).message.contains("not supported"), "{text:?}");
    }
    // `0=` cannot start a file, which would make it an old-style file.
    let text = "x'=1\n0=x+y-1";
    assert_eq!(kinds(text)[1], unsupported(Unsupported::Dae, "0=x+y-1"));
    check_one(text, UNSUPPORTED_DAE, (5, 12), None);
}

#[test]
fn delay_shift_and_del_shft_stay_calls() {
    assert!(matches!(
        clean_one("x'=delay(x,1)+shift(y,1)+del_shft(y,1,2)"),
        StmtKind::Ode { .. }
    ));
}

// ---------------------------------------------------------------------------
// Diagnostics, one per code.

#[test]
fn old_style_file() {
    let text = "2\nx'=1 $\n";
    check_one(text, OLD_STYLE_FILE, (0, 1), None);
    assert!(parse_all(text).0.statements.is_empty());
    let text = "\u{feff}  3 1 0\r\n";
    check_one(text, OLD_STYLE_FILE, (5, 10), None);
}

#[test]
fn unrecognized_statement() {
    check_one("+x=1", UNRECOGNIZED_STATEMENT, (0, 4), None);
    check_one("x y", UNRECOGNIZED_STATEMENT, (0, 3), None);
    check_one("x'=1\n  x y z # c\n", UNRECOGNIZED_STATEMENT, (7, 16), None);
}

#[test]
fn malformed_statements() {
    check_one("par", MALFORMED_STATEMENT, (3, 3), None);
    check_one("init x", MALFORMED_STATEMENT, (5, 6), None);
    check_one("aux 3=1", MALFORMED_STATEMENT, (4, 5), None);
    check_one("x' 1", MALFORMED_STATEMENT, (3, 4), None);
    check_one("x(0) 1", MALFORMED_STATEMENT, (5, 6), None);
    check_one("f(1)=2", MALFORMED_STATEMENT, (1, 2), None);
    // A function needs at least one parameter (PhaseCont choice).
    check_one("g()=1", MALFORMED_STATEMENT, (1, 2), None);
    check_one("f[j](u)=u", MALFORMED_STATEMENT, (1, 4), None);
    check_one("global 2 x {x=0}", MALFORMED_STATEMENT, (7, 8), None);
    check_one("global 1 x {x=0", MALFORMED_STATEMENT, (11, 12), None);
    check_one("set s a=1", MALFORMED_STATEMENT, (6, 7), None);
    check_one("set s {a=1", MALFORMED_STATEMENT, (6, 7), None);
    check_one("@", MALFORMED_STATEMENT, (1, 1), None);
    check_one("@ dt", MALFORMED_STATEMENT, (4, 4), None);
    check_one("#include", MALFORMED_STATEMENT, (0, 8), None);
    check_one("wiener", MALFORMED_STATEMENT, (6, 6), None);
    check_one("table f", MALFORMED_STATEMENT, (7, 7), None);
    check_one("markov z", MALFORMED_STATEMENT, (8, 8), None);
    check_one("markov z 0", MALFORMED_STATEMENT, (9, 10), None);
    let text = "markov z 1 extra words\n{0}";
    check_one(text, MALFORMED_STATEMENT, (11, 22), Some(((10, 22), "")));
    assert!(matches!(kinds(text)[0], StmtKind::Markov { states: 1, .. }));
}

#[test]
fn trailing_text() {
    check_one("x'=1)", TRAILING_TEXT, (4, 5), None);
    check_one("x(0)=1 2", BAD_VALUE, (5, 8), None);
    check_one("global 1 x {x=0} y", TRAILING_TEXT, (17, 18), None);
}

#[test]
fn trailing_comment_is_an_error_but_the_statement_stays() {
    let text = "x'=1 # c";
    check_one(text, TRAILING_COMMENT, (5, 8), Some(((4, 8), "")));
    let s = &parse_all(text).0.statements[0];
    assert!(matches!(s.kind, StmtKind::Ode { .. }));
    assert_eq!(s.comment.as_ref().unwrap().text, "# c");
    for text in [
        "init x=1 # c",
        "aux e=x # c",
        "f(u)=u # c",
        "x(0)=1 # c",
        "wiener w # c",
        "set s {a=1} # c",
        "global 1 x {x=0} # c",
    ] {
        assert_eq!(codes(text), vec![TRAILING_COMMENT], "{text:?}");
    }
    check_one(
        "%[1..2] # c\n%",
        TRAILING_COMMENT,
        (8, 11),
        Some(((7, 11), "")),
    );
}

#[test]
fn bad_values() {
    check_one("par a=x", BAD_VALUE, (6, 7), None);
    check_one("par a=2*3, b=1", BAD_VALUE, (6, 9), None);
    check_one("par a=", BAD_VALUE, (6, 6), None);
    check_one("x(0)=1+y", BAD_VALUE, (5, 8), None);
    check_one("par a=[1..2]", BAD_VALUE, (6, 12), None);
    check_one("par a=-x", BAD_VALUE, (6, 8), None);
    check_one("table f % 2.5 0 1 t", BAD_VALUE, (10, 13), None);
    check_one("table f % 0 0 1 t", BAD_VALUE, (10, 11), None);
    let StmtKind::Par { entries } = &kinds("par a=x, b=1")[0] else {
        panic!();
    };
    assert_eq!(entries[0].value, Some(Value::Error { span: DUMMY_SPAN }));
    assert_eq!(entries[1].value, Some(num("1")));
}

#[test]
fn unknown_option() {
    check_one("@ dt=1, nosuch=3", UNKNOWN_OPTION, (8, 14), None);
    assert!(codes("@ XP=x, xp2=y, s1=a, epsl=1e-5, NEWT_TOL=1e-6").is_empty());
}

#[test]
fn spaced_option() {
    check_one(
        "@ total = 100",
        SPACED_OPTION,
        (7, 10),
        Some(((7, 10), "=")),
    );
    check_one("@ total =100", SPACED_OPTION, (7, 9), Some(((7, 9), "=")));
    check_one("@ total= 100", SPACED_OPTION, (7, 9), Some(((7, 9), "=")));
    let StmtKind::Options { entries } = &kinds("@ total = 100")[0] else {
        panic!();
    };
    assert_eq!(entries[0].value, "100");
}

#[test]
fn markov_rows() {
    let text = "markov z 2\n{0} {1}\nx'=1";
    check_one(text, MARKOV_ROWS, (7, 8), None);
    // The line after the short chain is still parsed.
    assert!(matches!(kinds(text)[1], StmtKind::Ode { .. }));
    let text = "markov z 2\n{0}\n{1} {0}";
    check_one(text, MARKOV_ROWS, (11, 14), None);
    let text = "markov z 1\n{0";
    check_one(text, MARKOV_ROWS, (11, 12), None);
}

#[test]
fn bad_blocks() {
    let text = "%\nx'=1";
    check_one(text, BAD_BLOCK, (0, 1), None);
    let text = "%[1..2]\n%[3..4]\nx[j]'=1\n%\n%";
    check_one(text, BAD_BLOCK, (8, 15), None);
    let kinds = kinds(text);
    assert_eq!(kinds.len(), 1);
    let StmtKind::Block { body, .. } = &kinds[0] else {
        panic!();
    };
    assert_eq!(body.len(), 1);
    let text = "%[1..2]\nx[j]'=1";
    check_one(text, BAD_BLOCK, (1, 7), None);
    assert!(matches!(self::kinds(text)[0], StmtKind::Block { .. }));
    check_one("%[1..2] x\n%", BAD_BLOCK, (0, 9), None);
    check_one("%[a..2]\n%", BAD_BLOCK, (0, 7), None);
}

// ---------------------------------------------------------------------------
// R12 details.

#[test]
fn lexer_diagnostics_inside_raw_text_are_dropped() {
    for text in [
        "@ dt=1$",
        "set s {a=1$}",
        "\" $ {x=1}",
        "special k=$",
        "done\n$ $ \"",
        "1\n$",
    ] {
        assert!(
            !codes(text).contains(&STRAY_CHARACTER),
            "{text:?}: {:?}",
            codes(text)
        );
    }
    // Outside raw text they are kept.
    assert_eq!(codes("x'=1$"), vec![STRAY_CHARACTER]);
    assert!(!codes("x'=1 2").is_empty());
}

#[test]
fn sorted_diagnostics_from_lexer_and_parser() {
    let text = "par a=x\nx'=1$\n+";
    let (_, diags) = parse_all(text);
    let starts: Vec<usize> = diags.iter().map(|d| d.span.start).collect();
    let mut sorted = starts.clone();
    sorted.sort();
    assert_eq!(starts, sorted);
    assert_eq!(
        diags.iter().map(|d| d.code).collect::<Vec<_>>(),
        vec![BAD_VALUE, STRAY_CHARACTER, UNRECOGNIZED_STATEMENT]
    );
}

#[test]
fn recovery_reports_every_bad_line() {
    let text = "\
par a=1
x' 1
+y=2
y'=a*y
global 5 y {y=0}
par b=q
%
markov m 2
{0} {1}
z'=1
init q=1 # c
w=2
";
    assert_eq!(
        codes(text),
        vec![
            MALFORMED_STATEMENT,
            UNRECOGNIZED_STATEMENT,
            MALFORMED_STATEMENT,
            BAD_VALUE,
            BAD_BLOCK,
            MARKOV_ROWS,
            TRAILING_COMMENT,
        ]
    );
    let kinds = kinds(text);
    let good = kinds
        .iter()
        .filter(|k| {
            matches!(
                k,
                StmtKind::Par { .. }
                    | StmtKind::Ode { .. }
                    | StmtKind::Markov { .. }
                    | StmtKind::Init { .. }
                    | StmtKind::Fixed { .. }
            )
        })
        .count();
    // par a, y', par b, markov, z', init, w.
    assert_eq!(good, 7, "{kinds:#?}");
}

#[test]
fn messages_stay_short() {
    let long = "x".repeat(2000);
    for text in [
        format!("{long} y"),
        format!("par a={long}+1"),
        format!("@ {long}=1"),
        format!("x'=1 {long}"),
    ] {
        for d in parse_all(&text).1 {
            assert!(d.message.len() <= 340, "{}", d.message);
        }
    }
}

#[test]
fn deep_nesting_is_bounded() {
    let text = format!("x'={}1{}", "(".repeat(5000), ")".repeat(5000));
    let (_, diags) = parse_all(&text);
    assert!(!diags.is_empty());
    let text = "%[1..2]\n".repeat(5000);
    let (model, _) = parse_all(&text);
    assert_eq!(model.statements.len(), 1);
}

#[test]
fn strip_spans_is_idempotent() {
    let text = "par a=1 # c\n%[1..2]\nx[j]'=1\n%\n@ dt=1";
    let model = parse_all(text).0;
    let once = model.strip_spans();
    assert_eq!(once.strip_spans(), once);
    assert_ne!(once, model);
}

#[test]
fn one_error_for_an_empty_list_with_a_bad_entry() {
    check_one("par 3", MALFORMED_STATEMENT, (4, 5), None);
    check_one("wiener 3", MALFORMED_STATEMENT, (7, 8), None);
}

#[test]
fn every_prefix_of_a_model_parses_without_panic() {
    let text = "\
# model
par a=1, b=-2 # rates
number c=3
init x=1 y=[j]
x'=-a*x + c
dy/dt = b*y
z(t+1)=z/2
z(0)=1
!d=a*b
f(u,v)=u*v
aux e=f(x,y)
table h % 3 0 1 t^2
global 1 x-1 {x=0; y=1}
set fast {a=10}
markov m 2
{0} {1}
{1} {0}
wiener w1
%[1..2]
q[j]'=-q[j]
%
\" active $
@ dt=.1 total = 5 nosuch=1
x(t)=int{exp(-t)#x}
done
junk é
";
    for (i, _) in text.char_indices() {
        let _ = parse_all(&text[..i]);
        let _ = parse_all(&text[i..]);
    }
    let (model, diags) = parse_all(text);
    assert_eq!(
        diags.iter().map(|d| d.code).collect::<Vec<_>>(),
        vec![
            UNSUPPORTED_ACTIVE_COMMENT,
            SPACED_OPTION,
            UNKNOWN_OPTION,
            UNSUPPORTED_VOLTERRA,
        ]
    );
    assert_eq!(model.statements.len(), 21);
}

#[test]
fn include_spelling() {
    assert!(matches!(clean_one("#INCLUDE a.ode"), StmtKind::Comment(_)));
    assert!(matches!(clean_one("#done"), StmtKind::Comment(_)));
    assert!(matches!(
        clean_one("  #include\ta.ode"),
        StmtKind::Include { .. }
    ));
    // `done` inside a block ends the file and the block with it.
    let text = "%[1..2]\nx[j]'=1\ndone";
    check_one(text, BAD_BLOCK, (1, 7), None);
    assert_eq!(kinds(text).last(), Some(&StmtKind::Done));
}

// ---------------------------------------------------------------------------
// Fix round 1.

/// The parser diagnostics with `code` in `text`.
fn count(text: &str, code: Code) -> usize {
    codes(text).iter().filter(|&&c| c == code).count()
}

#[test]
fn continued_option_is_not_spaced() {
    // XPPAUT 8.0 applies `dt=\<newline>.5` (measured).
    let text = "@ total=1, dt=\\\n.5";
    assert_eq!(
        clean_one(text),
        StmtKind::Options {
            entries: vec![raw("total", "1"), raw("dt", ".5")],
        }
    );
    assert!(clean("@ dt\\\n=.5").len() == 1);
    // A real blank is still reported, and the message has no line break.
    let text = "@ dt= .5";
    check_one(text, SPACED_OPTION, (4, 6), Some(((4, 6), "=")));
    let d = one(text);
    assert!(d.message.contains("`dt=.5`"), "{}", d.message);
    let d = one("@ dt=\\\n .5");
    assert_eq!(d.code, SPACED_OPTION);
    assert!(!d.message.contains('\n'), "{}", d.message);
}

#[test]
fn include_with_a_trailing_comment() {
    let text = "#include a.ode # c";
    check_one(text, TRAILING_COMMENT, (15, 18), Some(((14, 18), "")));
    let s = &parse_all(text).0.statements[0];
    let StmtKind::Include { path, path_span } = &s.kind else {
        panic!("{s:#?}");
    };
    assert_eq!(path, "a.ode");
    assert_eq!(*path_span, sp(9, 14));
    assert_eq!(s.comment.as_ref().unwrap().text, "# c");
    // A `#` inside the path is part of it; quotes stay as written.
    let path_of = |text: &str| match clean_one(text) {
        StmtKind::Include { path, .. } => path,
        other => panic!("{other:#?}"),
    };
    assert_eq!(path_of("#include a#b.ode"), "a#b.ode");
    assert_eq!(path_of("#include \"a.ode\""), "\"a.ode\"");
    check_one("#include # c", MALFORMED_STATEMENT, (0, 12), None);
}

#[test]
fn malformed_block_openers_poison_their_block() {
    for opener in [
        "%[1.5..3]",
        "%[1..2",
        "%[a..3]",
        "%[1..3] x",
        "%[1e2..2]",
        "%[1..99999999999999999999]",
    ] {
        let text = format!("{opener}\nx[j]'=1\n%\ny'=1");
        assert_eq!(count(&text, BAD_BLOCK), 1, "{text:?}: {:?}", codes(&text));
        let kinds = kinds(&text);
        assert_eq!(kinds.len(), 2, "{text:?}: {kinds:#?}");
        let StmtKind::BadBlock { body } = &kinds[0] else {
            panic!("{text:?}: {kinds:#?}");
        };
        assert_eq!(body.len(), 1);
        assert!(matches!(kinds[1], StmtKind::Ode { .. }));
    }
    let text = "%[a..3]\nx'=1\n%";
    check_one(text, BAD_BLOCK, (0, 7), None);
    assert_eq!(parse_all(text).0.statements[0].span, sp(0, 14));
    // An unclosed bad block is reported too.
    assert_eq!(codes("%[a..3]\nx'=1"), vec![BAD_BLOCK, BAD_BLOCK]);
    // A bad opener inside a block: its `%` is ignored.
    let text = "%[1..2]\n%[a..3]\nx[j]'=1\n%\n%";
    check_one(text, BAD_BLOCK, (8, 15), None);
    let kinds = kinds(text);
    assert_eq!(kinds.len(), 1);
    assert!(matches!(&kinds[0], StmtKind::Block { body, .. } if body.len() == 1));
    // A malformed closer changes nothing.
    let text = "%[1..2]\n% x\nx[j]'=1\n%";
    check_one(text, BAD_BLOCK, (8, 11), None);
    assert!(matches!(&self::kinds(text)[0], StmtKind::Block { body, .. } if body.len() == 1));
}

#[test]
fn raw_ranges_filter_lexer_diagnostics() {
    let text = "@ dt=$ total=$\nx'=1$\nspecial k=$\ndone $\ny'=1$";
    assert_eq!(codes(text), vec![STRAY_CHARACTER, UNSUPPORTED_SPECIAL]);
    let text = "special k=$\nspecial j=$\nx'=1$";
    assert_eq!(
        codes(text),
        vec![UNSUPPORTED_SPECIAL, UNSUPPORTED_SPECIAL, STRAY_CHARACTER]
    );
}

#[test]
fn unknown_tokens_are_reported_once() {
    for text in [
        "par é=1",
        "par a=1 é",
        "wiener é",
        "@ é=1",
        "@ dt=1 é",
        "global 1 x {é=1}",
        "global é x {x=0}",
        "set é {a=1}",
        "set s {é=1}",
        "table é % 3 0 1 t",
        "markov é 2",
        "markov z é",
        "par a=1, $",
        "f(é)=1",
        "xé'=1",
        "!é=1",
        "!é",
        "aux é=1",
        "x' é",
        "global 1 x {x=0} é",
        "markov z 1\n{1} é",
        "markov z 1 é\n{1}",
        "x'=1 é",
    ] {
        assert_eq!(codes(text), vec![STRAY_CHARACTER], "{text:?}");
    }
    // Text after the unknown token is still reported.
    assert_eq!(
        codes("global 1 x {x=0} é y"),
        vec![STRAY_CHARACTER, TRAILING_TEXT]
    );
}

#[test]
fn table_fields_are_named() {
    for (text, field, at) in [
        ("table h %", "number of points", 9),
        ("table h % 11", "lower bound", 12),
        ("table h % 11 0", "upper bound", 14),
        ("table h % 11 0 1", "formula", 16),
    ] {
        check_one(text, MALFORMED_STATEMENT, (at, at), None);
        let d = one(text);
        assert!(d.message.contains(field), "{text:?}: {}", d.message);
        assert!(matches!(kinds(text)[0], StmtKind::Table { .. }), "{text:?}");
    }
    let text = "table h % 3 - 1 0 t";
    check_one(text, BAD_VALUE, (12, 15), None);
    assert!(one(text).message.contains("lower bound"));
    assert!(matches!(kinds(text)[0], StmtKind::Table { .. }));
    let text = "table h % x 0 1 t";
    check_one(text, BAD_VALUE, (10, 11), None);
    assert!(one(text).message.contains("number of points"));
    check_one("table h % 3 0 y t", BAD_VALUE, (14, 15), None);
}

#[test]
fn spaced_sign_is_one_bad_value() {
    check_one("init x=- 1", BAD_VALUE, (7, 10), None);
    check_one("par a=- 1, b=2", BAD_VALUE, (6, 9), None);
}

#[test]
fn missing_at_line_end_is_zero_width() {
    for text in ["markov z", "markov z\n", "markov z\r\n", "markov z  \n"] {
        let d = one(text);
        assert_eq!(d.code, MALFORMED_STATEMENT, "{text:?}");
        assert_eq!(d.span.start, d.span.end, "{text:?}: {d:#?}");
    }
    check_one("markov z\r\n", MALFORMED_STATEMENT, (8, 8), None);
    check_one("aux x\n", MALFORMED_STATEMENT, (5, 5), None);
    check_one("@ dt\r\n", MALFORMED_STATEMENT, (4, 4), None);
}

#[test]
fn trailing_comments_on_done_and_markov() {
    let text = "done # c";
    check_one(text, TRAILING_COMMENT, (5, 8), Some(((4, 8), "")));
    let s = &parse_all(text).0.statements[0];
    assert_eq!(s.kind, StmtKind::Done);
    assert_eq!(s.comment.as_ref().unwrap().text, "# c");
    // Text after `done` is still ignored.
    assert!(codes("done x # c").is_empty());
    let text = "markov z 1\n{1} # c";
    check_one(text, TRAILING_COMMENT, (15, 18), Some(((14, 18), "")));
    let StmtKind::Markov { lines, .. } = &kinds(text)[0] else {
        panic!();
    };
    let MarkovLine::Row(r) = &lines[0] else {
        panic!();
    };
    assert_eq!(r.comment.as_ref().unwrap().text, "# c");
    assert_eq!(parse_all(text).0.statements[0].span, sp(0, 18));
    let text = "markov z 1 # c\n{1}";
    check_one(text, TRAILING_COMMENT, (11, 14), Some(((10, 14), "")));
    let s = &parse_all(text).0.statements[0];
    assert_eq!(s.comment.as_ref().unwrap().text, "# c");
    assert_eq!(s.span, sp(0, 18));
}

#[test]
fn markov_lines_keep_source_order() {
    let text = "# a\nmarkov z 2\n{0} {a}\n\n# note\n{b} {0}\nx'=z";
    let model = parse_all(text).0;
    assert_eq!(model.statements.len(), 3);
    let spans: Vec<Span> = model.statements.iter().map(|s| s.span).collect();
    for pair in spans.windows(2) {
        assert!(pair[0].end <= pair[1].start, "{spans:?}");
    }
    // An `#include` line ends the rows and is parsed as a statement.
    let text = "markov z 2\n{0} {1}\n#include a.ode\n{1} {0}";
    assert_eq!(codes(text), vec![MARKOV_ROWS, UNRECOGNIZED_STATEMENT]);
    assert!(matches!(kinds(text)[1], StmtKind::Include { .. }));
    // Comments after a short chain are not taken into it.
    let text = "markov z 2\n{0} {1}\n# c\nx'=1";
    check_one(text, MARKOV_ROWS, (7, 8), None);
    let kinds = kinds(text);
    assert_eq!(kinds.len(), 3);
    assert!(matches!(&kinds[0], StmtKind::Markov { lines, .. } if lines.len() == 1));
    assert_eq!(kinds[1], StmtKind::Comment(comment("# c")));
}

#[test]
fn set_and_global_brace_lists() {
    // Measured with XPPAUT 8.0: blanks separate set entries, and an empty
    // set list loads.
    assert_eq!(
        clean_one("set s {a=2 b=3}"),
        StmtKind::Set {
            name: ident("s"),
            entries: vec![raw("a", "2"), raw("b", "3")],
        }
    );
    assert_eq!(
        clean_one("set s {}"),
        StmtKind::Set {
            name: ident("s"),
            entries: vec![],
        }
    );
    // Measured: XPPAUT rejects an empty global list ("Bad global").
    let text = "global 1 x-5 {}";
    check_one(text, MALFORMED_STATEMENT, (13, 15), None);
    assert!(one(text).message.contains("at least one assignment"));
}

#[test]
fn coverage_gaps() {
    // `0=` on line 1 is an old-style file, not an algebraic constraint.
    check_one("0=x+y", OLD_STYLE_FILE, (0, 5), None);
    // An unsupported statement inside a block stays in the block.
    let text = "%[1..2]\nbndry x\n%";
    assert_eq!(codes(text), vec![UNSUPPORTED_BOUNDARY]);
    let StmtKind::Block { body, .. } = &kinds(text)[0] else {
        panic!();
    };
    assert_eq!(body[0].kind, unsupported(Unsupported::Boundary, "bndry x"));
    // Backslash continuation inside statements.
    let kinds = clean("par a=1,\\\n  b=2\nx'=a+\\\nb\nglobal 1 x \\\n{x=0}");
    assert_eq!(kinds.len(), 3);
    assert!(matches!(&kinds[0], StmtKind::Par { entries } if entries.len() == 2));
    assert_eq!(
        kinds[1],
        StmtKind::Ode {
            name: ident("x"),
            form: OdeForm::Prime,
            rhs: bin(B::Add, v("a"), v("b")),
        }
    );
}

#[test]
fn message_text_per_code() {
    let cases: [(&str, Code, &str); 21] = [
        ("2\n", OLD_STYLE_FILE, "old XPPAUT format"),
        (
            "+x=1",
            UNRECOGNIZED_STATEMENT,
            "is not a statement XPPAUT reads",
        ),
        ("par", MALFORMED_STATEMENT, "needs at least one entry"),
        ("x'=1)", TRAILING_TEXT, "after the statement"),
        ("x'=1 # c", TRAILING_COMMENT, "on a line of its own"),
        ("par a=x", BAD_VALUE, "is not a number literal"),
        ("@ nosuch=1", UNKNOWN_OPTION, "unknown option `nosuch`"),
        ("@ dt =1", SPACED_OPTION, "blanks around `=`"),
        ("markov z 2\n{0} {1}", MARKOV_ROWS, "needs 2 rows"),
        ("%\nx'=1", BAD_BLOCK, "closes no array block"),
        (
            "volt x=1",
            UNSUPPORTED_VOLTERRA,
            "Volterra integral equations are not supported",
        ),
        (
            "bndry x",
            UNSUPPORTED_BOUNDARY,
            "boundary conditions are not supported",
        ),
        (
            "solv x",
            UNSUPPORTED_DAE,
            "algebraic constraints are not supported",
        ),
        (
            "special k=1",
            UNSUPPORTED_SPECIAL,
            "`special` is not supported",
        ),
        (
            "export {x} {y}",
            UNSUPPORTED_EXPORT,
            "`export` is not supported",
        ),
        ("only x", UNSUPPORTED_ONLY, "`only` is not supported"),
        (
            "options a.opt",
            UNSUPPORTED_OPTIONS_FILE,
            "`options` files are not supported",
        ),
        (
            "hom_bcs 1",
            UNSUPPORTED_HOM_BCS,
            "`hom_bcs` is not supported",
        ),
        (
            "table f a.tab",
            UNSUPPORTED_FILE_TABLE,
            "tables read from a file are not supported",
        ),
        (
            "x'=sum(1,2)of(i')",
            UNSUPPORTED_SUM_OF,
            "`sum(..)of(..)` is not supported",
        ),
        (
            "\" note",
            UNSUPPORTED_ACTIVE_COMMENT,
            "active comments are not supported",
        ),
    ];
    for (text, code, needle) in cases {
        let d = one(text);
        assert_eq!(d.code, code, "{text:?}: {d:#?}");
        assert!(d.message.contains(needle), "{text:?}: {}", d.message);
    }
}
