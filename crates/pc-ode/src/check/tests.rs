use super::*;
use crate::diag::CODES;
use crate::parser::expr::PORTABILITY;
use crate::parser::parse;
use crate::parser::stmt::UNSUPPORTED_SUM_OF;
use crate::span::{FileId, Span};

const F: FileId = FileId(0);

/// Parses and checks `text`. The parse must give no error; its warnings
/// (unsupported statements, portability) are dropped. Only the check
/// diagnostics are returned.
fn run(text: &str) -> (ModelInfo, Vec<Diagnostic>) {
    let (model, parse_diags) = parse(text, F);
    assert!(
        parse_diags.iter().all(|d| d.severity != Severity::Error),
        "{text:?}: {parse_diags:#?}"
    );
    let (info, diags) = check(&model);
    for d in &diags {
        assert!(d.span.end <= text.len(), "{text:?}: {d:#?}");
        let severity = CODES.iter().find(|(c, _, _)| *c == d.code).unwrap().1;
        assert_eq!(d.severity, severity, "{text:?}: {d:#?}");
        assert!(d.message.len() <= 340, "{text:?}: {}", d.message);
    }
    let starts: Vec<usize> = diags.iter().map(|d| d.span.start).collect();
    let mut sorted = starts.clone();
    sorted.sort();
    assert_eq!(starts, sorted, "{text:?}: not sorted");
    (info, diags)
}

fn codes(text: &str) -> Vec<Code> {
    run(text).1.iter().map(|d| d.code).collect()
}

fn info(text: &str) -> ModelInfo {
    let (info, diags) = run(text);
    assert!(diags.is_empty(), "{text:?}: {diags:#?}");
    info
}

fn clean(text: &str) {
    info(text);
}

fn one(text: &str) -> Diagnostic {
    let (_, diags) = run(text);
    assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
    diags.into_iter().next().unwrap()
}

fn sp(start: usize, end: usize) -> Span {
    Span::new(F, start, end)
}

/// The byte range of the `n`th (from 0) occurrence of `needle` in `text`
/// that is not part of a longer name or number.
fn nth(text: &str, needle: &str, n: usize) -> Span {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    let start = text
        .match_indices(needle)
        .map(|(i, _)| i)
        .filter(|&i| {
            let before = text[..i].chars().next_back();
            let after = text[i + needle.len()..].chars().next();
            let first = needle.chars().next().unwrap();
            let last = needle.chars().next_back().unwrap();
            !(is_word(first) && before.is_some_and(is_word))
                && !(is_word(last) && after.is_some_and(is_word))
        })
        .nth(n)
        .unwrap_or_else(|| panic!("{needle:?} #{n} in {text:?}"));
    sp(start, start + needle.len())
}

/// Checks the code and span of the one diagnostic of `text`, and the spans
/// of its notes.
fn check_one(text: &str, code: Code, span: Span, notes: &[Span]) -> Diagnostic {
    let d = one(text);
    assert_eq!(d.code, code, "{text:?}: {d:#?}");
    assert_eq!(d.span, span, "{text:?}: {d:#?}");
    let note_spans: Vec<Span> = d.notes.iter().map(|n| n.0).collect();
    assert_eq!(note_spans, notes, "{text:?}: {d:#?}");
    d
}

// ---------------------------------------------------------------------------
// Clean models and the symbol table.

#[test]
fn clean_model_and_symbols() {
    let text = "par a=1, b=2\nnumber c=3\n!d=a*c\nx'=-a*x+w+f(y)\ndy/dt=x*t\n\
                x(0)=1\ninit y=2\naux e=x^2\nw=x+pi\nf(u)=u*b\n";
    let info = info(text);
    assert_eq!(info.kind, ModelKind::Ode);
    assert!(info.smooth);
    assert!(!info.simulation_only);
    let kinds: Vec<(&str, SymbolKind)> = info
        .symbols
        .iter()
        .map(|s| (s.name.key.as_str(), s.kind))
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("a", SymbolKind::Parameter),
            ("b", SymbolKind::Parameter),
            ("c", SymbolKind::Number),
            ("d", SymbolKind::Derived),
            ("x", SymbolKind::State(StateKind::Ode)),
            ("y", SymbolKind::State(StateKind::Ode)),
            ("e", SymbolKind::Aux),
            ("w", SymbolKind::Fixed),
            ("f", SymbolKind::Function { params: 1 }),
        ]
    );
    assert_eq!(info.symbol("x").unwrap().stmt, 3);
    assert!(info.symbol("q").is_none());
}

#[test]
fn case_is_ignored() {
    clean("PAR A=2\nX'=a*x\nAUX V=A");
    assert_eq!(codes("par a=1\npar A=2"), vec![DUPLICATE_NAME]);
}

#[test]
fn model_kind() {
    assert_eq!(info("x'=1").kind, ModelKind::Ode);
    assert_eq!(info("x(t+1)=x/2").kind, ModelKind::Map);
    assert_eq!(info("par a=1").kind, ModelKind::None);
    assert_eq!(info("").kind, ModelKind::None);
    let text = "x'=1\ny(t+1)=y\nz(t+1)=z";
    let d = check_one(
        text,
        MIXED_KIND,
        nth(text, "y(t+1)=y", 0),
        &[nth(text, "x'=1", 0)],
    );
    assert!(d.message.contains("ODEs and maps"), "{}", d.message);
    assert_eq!(run(text).0.kind, ModelKind::Ode);
    let text = "y(t+1)=y\nx'=1";
    check_one(
        text,
        MIXED_KIND,
        nth(text, "x'=1", 0),
        &[nth(text, "y(t+1)=y", 0)],
    );
    assert_eq!(run(text).0.kind, ModelKind::Map);
}

#[test]
fn markov_states_do_not_set_the_kind() {
    let info = info("markov z 2\n{0} {1}\n{1} {0}\nx'=z");
    assert_eq!(info.kind, ModelKind::Ode);
    assert_eq!(
        info.symbol("z").unwrap().kind,
        SymbolKind::State(StateKind::Markov)
    );
}

// ---------------------------------------------------------------------------
// Undefined and reserved names.

#[test]
fn undefined_names() {
    let text = "x'=-q*x";
    let d = check_one(text, UNDEFINED_NAME, nth(text, "q", 0), &[]);
    assert!(d.message.contains("`q`"), "{}", d.message);
    let text = "x'=foo(x)";
    let d = check_one(text, UNDEFINED_NAME, nth(text, "foo", 0), &[]);
    assert!(d.message.contains("function `foo`"), "{}", d.message);
    // Every use is reported.
    assert_eq!(codes("x'=q+q"), vec![UNDEFINED_NAME, UNDEFINED_NAME]);
    // Uses before the declaration are fine.
    clean("x'=a\npar a=1");
    // `e` is an ordinary name, `t` and `pi` are always known.
    assert_eq!(codes("x'=e"), vec![UNDEFINED_NAME]);
    clean("x'=t*pi");
    // `arg10` is an ordinary name.
    assert_eq!(codes("f(u)=arg10"), vec![UNDEFINED_NAME]);
}

#[test]
fn undefined_names_in_every_statement_form() {
    for text in [
        "x(t+1)=q",
        "aux v=q",
        "w=q",
        "!d=q",
        "f(u)=u+q",
        "table h % 11 0 1 q*t",
        "x'=1\nglobal 1 x-q {x=0}",
        "x'=1\nglobal 1 x {x=q}",
        "x'=1\nglobal 1 x {q=0}",
        "markov z 2\n{0} {q}\n{1} {0}",
    ] {
        assert_eq!(codes(text), vec![UNDEFINED_NAME], "{text:?}");
    }
}

#[test]
fn reserved_names() {
    for (text, name) in [
        ("t'=0", "t"),
        ("par sin=2", "sin"),
        ("pi=3", "pi"),
        ("aux arg1=2", "arg1"),
        ("number arg9=2", "arg9"),
        ("x'=1\nx(0)=0\nmarkov delay 2\n{0} {1}\n{1} {0}", "delay"),
        ("ceil(u)=u", "ceil"),
        ("table ran % 11 0 1 t", "ran"),
        ("x'=1\nwiener poisson", "poisson"),
        ("!shift=1", "shift"),
        ("f(u,sin)=u", "sin"),
        ("f(arg1)=1", "arg1"),
    ] {
        let (_, diags) = run(text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, RESERVED_NAME, "{text:?}: {d:#?}");
        assert_eq!(d.span, nth(text, name, 0), "{text:?}");
        assert!(d.message.contains(&format!("`{name}`")), "{}", d.message);
    }
    clean("par e=2\nx'=e");
    // A reserved name keeps its meaning: only the declaration is reported.
    assert_eq!(codes("par sin=2\nx'=sin(x)"), vec![RESERVED_NAME]);
}

#[test]
fn keyword_words_are_reserved() {
    let mut checked = 0;
    for word in ["not", "if", "then", "else"] {
        let text = format!("par {word}=1");
        let (model, parse_diags) = parse(&text, F);
        if parse_diags.iter().any(|d| d.severity == Severity::Error) {
            continue;
        }
        checked += 1;
        let (_, diags) = check(&model);
        let got: Vec<Code> = diags.iter().map(|d| d.code).collect();
        assert_eq!(got, vec![RESERVED_NAME], "{text:?}");
    }
    assert!(checked > 0, "no keyword word reached check");
}

// ---------------------------------------------------------------------------
// Duplicates and initial values.

#[test]
fn duplicates() {
    let text = "par a=1\npar a=2";
    let d = check_one(
        text,
        DUPLICATE_NAME,
        nth(text, "a", 1),
        &[nth(text, "a", 0)],
    );
    assert!(d.message.contains("`a`"), "{}", d.message);
    let text = "par a=1, a=2";
    check_one(
        text,
        DUPLICATE_NAME,
        nth(text, "a", 1),
        &[nth(text, "a", 0)],
    );
    for text in [
        "x'=1\nx=2",
        "x'=1\ndx/dt=2",
        "x'=1\nmarkov x 2\n{0} {1}\n{1} {0}",
        "x'=1\naux x=1",
        "f(u)=u\nf=1",
        "table h % 11 0 1 t\npar h=1",
        "x'=w\nwiener w, w",
        "x(t+1)=1\nnumber x=1",
        "!d=1\npar d=1",
    ] {
        assert_eq!(codes(text), vec![DUPLICATE_NAME], "{text:?}");
    }
    // The first declaration wins.
    assert_eq!(
        run("par a=1\nx'=a\naux a=1").0.symbol("a").unwrap().kind,
        SymbolKind::Parameter
    );
}

#[test]
fn initial_values() {
    clean("x'=1\ny'=1\nx(0)=1\ninit y=2");
    clean("markov z 2\n{0} {1}\n{1} {0}\ninit z=1");
    clean("init x=1\nx(t+1)=x");
    let text = "x'=1\ninit y=1";
    let d = check_one(text, INIT_NOT_STATE, nth(text, "y", 0), &[]);
    assert!(d.message.contains("`y`"), "{}", d.message);
    let text = "par a=1\nx'=a\na(0)=1";
    check_one(
        text,
        INIT_NOT_STATE,
        nth(text, "a", 2),
        &[nth(text, "a", 0)],
    );
    let text = "x'=1\nx(0)=1\ninit x=2";
    let d = check_one(text, INIT_TWICE, nth(text, "x", 2), &[nth(text, "x", 1)]);
    assert!(d.message.contains("`x`"), "{}", d.message);
    assert_eq!(codes("x'=1\ninit x=1, x=2"), vec![INIT_TWICE]);
}

// ---------------------------------------------------------------------------
// Order of fixed quantities, cycles, derived parameters.

#[test]
fn fixed_quantity_forward_reference() {
    let text = "b=c+1\nc=2\nx'=b";
    let d = check_one(text, FIXED_FORWARD, nth(text, "c", 0), &[nth(text, "c", 1)]);
    assert!(d.message.contains("`c`"), "{}", d.message);
    // ODEs, aux lines and functions see every fixed quantity.
    clean("x'=b\naux v=b\nf(u)=u*b\nb=2");
    clean("c=2\nb=c+1\nx'=b");
}

#[test]
fn circular_definitions() {
    let text = "w=w+1";
    check_one(text, CIRCULAR, nth(text, "w", 0), &[]);
    // One cycle error, no forward reference error on top.
    let text = "a=b\nb=a";
    let d = check_one(text, CIRCULAR, nth(text, "a", 0), &[nth(text, "b", 1)]);
    assert!(
        d.message.contains("`a`") && d.message.contains("`b`"),
        "{}",
        d.message
    );
    // A forward reference into a cycle is still reported.
    assert_eq!(codes("c=a\na=b\nb=a"), vec![FIXED_FORWARD, CIRCULAR]);
    // Derived parameters may refer forward but not in a circle.
    clean("!d=e*2\n!e=a\npar a=1");
    assert_eq!(codes("!d=d+1"), vec![CIRCULAR]);
    assert_eq!(codes("!d=e\n!e=d"), vec![CIRCULAR]);
    // A cycle through a user function is still a cycle.
    let text = "a=g(2)\ng(u)=u*a";
    let d = check_one(text, CIRCULAR, nth(text, "a", 0), &[nth(text, "g", 1)]);
    assert!(d.message.contains("`g`"), "{}", d.message);
    assert_eq!(codes("a=g(1)\ng(u)=u+b\nb=a"), vec![CIRCULAR]);
    assert_eq!(codes("!d=g(1)\ng(u)=u*d"), vec![CIRCULAR]);
    // Functions see final values, so a call into a lower line is fine.
    clean("a=g(2)\ng(u)=u*b\nb=1");
}

#[test]
fn derived_parameters() {
    clean("par a=1\nnumber c=2\n!d=a*c+pi+sin(a)+f(a)\nf(u)=2*u");
    for (text, name) in [
        ("x'=1\n!d=x", "x"),
        ("!d=t", "t"),
        ("w=1\n!d=w", "w"),
        ("x'=v\nwiener v\n!d=v", "v"),
    ] {
        let (_, diags) = run(text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        assert_eq!(diags[0].code, DERIVED_NON_PARAMETER, "{text:?}");
        assert_eq!(diags[0].span.end, text.len(), "{text:?}");
        assert!(diags[0].message.contains(&format!("`{name}`")), "{text:?}");
    }
}

// ---------------------------------------------------------------------------
// Functions and calls.

#[test]
fn function_parameters() {
    clean("f(a,b,c,d,e,g,h,i,j)=a+b+c+d+e+g+h+i+j");
    let text = "f(a,b,c,d,e,g,h,i,j,k,l)=a";
    let d = check_one(text, TOO_MANY_PARAMS, nth(text, "k,l", 0), &[]);
    assert!(d.message.contains("11"), "{}", d.message);
    let text = "f(u,u)=u";
    check_one(
        text,
        DUPLICATE_NAME,
        nth(text, "u", 1),
        &[nth(text, "u", 0)],
    );
    // Parameters shadow global names.
    clean("par u=1\nf(u)=u*2\nx'=f(x)");
    // A body may use states, fixed quantities, `t` and other functions.
    clean("x'=f(1)\nw=2\nf(u)=u*x*w*t+g(u)\ng(v)=v");
}

#[test]
fn recursive_functions() {
    let text = "f(u)=f(u)";
    let d = check_one(text, RECURSIVE_FUNCTION, nth(text, "f", 0), &[]);
    assert!(d.message.contains("`f`"), "{}", d.message);
    let text = "f(u)=g(u)\ng(u)=h(u)\nh(u)=f(u)";
    check_one(
        text,
        RECURSIVE_FUNCTION,
        nth(text, "f", 0),
        &[nth(text, "g", 1), nth(text, "h", 1)],
    );
    clean("f(u)=g(u)+g(u)\ng(u)=u");
}

#[test]
fn argument_counts() {
    let text = "f(u,v)=u+v\nx'=f(1)";
    let d = check_one(
        text,
        WRONG_ARITY,
        nth(text, "f(1)", 0),
        &[nth(text, "f", 0)],
    );
    assert!(
        d.message.contains("2") && d.message.contains("1"),
        "{}",
        d.message
    );
    assert_eq!(codes("table h % 11 0 1 2*t\nx'=h(1,2)"), vec![WRONG_ARITY]);
    clean("table h % 11 0 1 2*t\nx'=h(x)");
    for text in ["x'=max(1,2,3)", "x'=atan2(1)", "x'=sin()", "x'=delay(x)"] {
        let c = codes(text);
        assert!(c.contains(&WRONG_ARITY), "{text:?}: {c:?}");
    }
    clean("x'=atan2(1,x)+min(x,1)+besselj(1,x)+normal(0,1)*0");
}

#[test]
fn values_and_functions_used_the_wrong_way() {
    for (text, name) in [
        ("x'=sin", "sin"),
        ("par a=1\nx'=a(1)", "a"),
        ("f(u)=u\nx'=f", "f"),
        ("f(g)=g(1)", "g"),
        ("x'=t(1)", "t"),
        ("x'=pi(1)", "pi"),
        ("table h % 11 0 1 t\nx'=h", "h"),
    ] {
        let (_, diags) = run(text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        assert_eq!(diags[0].code, NOT_CALLABLE, "{text:?}");
        assert!(diags[0].message.contains(&format!("`{name}`")), "{text:?}");
    }
}

#[test]
fn arg_names() {
    clean("f(u,v)=arg2+arg1*u");
    let text = "f(u)=arg2";
    let d = check_one(text, BAD_ARG, nth(text, "arg2", 0), &[]);
    assert!(d.message.contains("`arg2`"), "{}", d.message);
    assert_eq!(codes("x'=arg1"), vec![BAD_ARG]);
}

#[test]
fn aux_in_formulas() {
    let text = "aux e=x\nx'=e";
    let d = check_one(
        text,
        AUX_IN_FORMULA,
        nth(text, "e", 1),
        &[nth(text, "e", 0)],
    );
    assert!(d.message.contains("`e`"), "{}", d.message);
    for text in [
        "aux e=1\naux g=e",
        "aux e=1\nf(u)=u*e",
        "aux e=1\nw=e",
        "aux e=1\nx'=1\nglobal 1 e {x=0}",
    ] {
        assert_eq!(codes(text), vec![AUX_IN_FORMULA], "{text:?}");
    }
    // A function parameter with the same name hides the aux quantity.
    clean("aux e=1\nf(e)=e");
}

// ---------------------------------------------------------------------------
// Delay forms, flags and portability.

#[test]
fn delay_shift_and_del_shft() {
    let text = "x'=delay(x,1)";
    let d = check_one(text, UNSUPPORTED_DELAY, nth(text, "delay", 0), &[]);
    assert!(d.message.contains("`delay`"), "{}", d.message);
    let text = "x'=shift(x,1)";
    check_one(text, UNSUPPORTED_SHIFT, nth(text, "shift", 0), &[]);
    let text = "x'=del_shft(x,1,2)";
    check_one(text, UNSUPPORTED_DEL_SHFT, nth(text, "del_shft", 0), &[]);
    // Their arguments are still resolved.
    assert_eq!(
        codes("x'=delay(q,1)"),
        vec![UNSUPPORTED_DELAY, UNDEFINED_NAME]
    );
    // `sum(..)` without `of(..)` stays a call and gets the sum warning.
    assert_eq!(codes("x'=sum(1,2)"), vec![UNSUPPORTED_SUM_OF]);
}

#[test]
fn smooth_flag() {
    assert!(info("x'=sin(x)+exp(x)+lgamma(x)").smooth);
    for text in [
        "x'=abs(x)",
        "x'=heav(x)",
        "x'=sign(x)",
        "x'=mod(x,1)",
        "x'=min(x,1)",
        "x'=max(x,1)",
        "x'=flr(x)",
        "x'=if(x>0)then(1)else(0)",
        "table h % 11 0 1 t\nx'=h(x)",
        "aux v=abs(x)\nx'=1",
        "f(u)=abs(u)\nx'=1",
    ] {
        assert!(!info(text).smooth, "{text:?}");
    }
    let (info, _) = run("x'=ceil(x)");
    assert!(!info.smooth);
    let (info, _) = run("table h data.tab\nx'=1");
    assert!(!info.smooth, "a file table is a table");
}

#[test]
fn simulation_only_flag() {
    assert!(!info("x'=x").simulation_only);
    for text in [
        "x'=ran(1)",
        "x'=normal(0,1)",
        "x'=poisson(1)",
        "markov z 2\n{0} {1}\n{1} {0}",
        "x'=w\nwiener w",
        "x'=1\nglobal 1 x {x=0}",
    ] {
        assert!(info(text).simulation_only, "{text:?}");
    }
}

#[test]
fn ceil_portability() {
    let text = "x'=ceil(x)";
    let d = check_one(text, PORTABILITY, nth(text, "ceil", 0), &[]);
    assert!(d.message.contains("XPPAUT 8.0 rejects"), "{}", d.message);
    assert!(d.message.contains("`ceil`"), "{}", d.message);
}

#[test]
fn long_names() {
    clean("par abcdefghij=1\nx'=abcdefghij");
    let text = "par abcdefghijk=1\nx'=abcdefghijk";
    let d = check_one(text, PORTABILITY, nth(text, "abcdefghijk", 0), &[]);
    assert!(d.message.contains("`abcdefghijk`"), "{}", d.message);
    assert!(d.message.contains("10 characters"), "{}", d.message);
    // Two names that agree in their first 10 characters, ignoring case.
    let text = "par abcdefghijk=1, ABCDEFGHIJl=2\nx'=abcdefghijk+abcdefghijl";
    let (_, diags) = run(text);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert_eq!(diags[0].span, nth(text, "abcdefghijk", 0));
    assert_eq!(diags[0].notes.len(), 1);
    assert_eq!(diags[0].notes[0].0, nth(text, "ABCDEFGHIJl", 0));
    assert!(diags[0].notes[0].1.contains("first 10"), "{diags:#?}");
    assert_eq!(diags[1].notes[0].0, nth(text, "abcdefghijk", 0));
    // A long name next to its own 10-character prefix.
    let text = "par abcdefghij=1, abcdefghijk=2";
    let d = check_one(
        text,
        PORTABILITY,
        nth(text, "abcdefghijk", 0),
        &[nth(text, "abcdefghij", 0)],
    );
    assert!(d.code == PORTABILITY);
    // Function parameters get the warning too.
    assert_eq!(codes("f(abcdefghijk)=abcdefghijk"), vec![PORTABILITY]);
}

// ---------------------------------------------------------------------------
// Unsupported statements, arrays, includes.

#[test]
fn unsupported_statements_that_may_declare_names() {
    for (text, decl) in [
        ("x'=v\nv(t)=1+x", "v(t)=1+x"),
        ("table h data.tab\nx'=h(x)", "table h data.tab"),
        ("x'=y\n0=x+y-1", "0=x+y-1"),
        (
            "x'=k\nspecial k=conv(even,101,2,w,x)",
            "special k=conv(even,101,2,w,x)",
        ),
        ("x'=z\nz=sum(0,4)of(x)", "z=sum(0,4)of(x)"),
    ] {
        let (_, diags) = run(text);
        let ours: Vec<&Diagnostic> = diags.iter().filter(|d| d.code != DUPLICATE_NAME).collect();
        assert_eq!(ours.len(), 1, "{text:?}: {diags:#?}");
        let d = ours[0];
        assert_eq!(d.code, MAYBE_UNDEFINED, "{text:?}: {d:#?}");
        assert_eq!(d.notes.len(), 1, "{text:?}");
        assert_eq!(d.notes[0].0, nth(text, decl, 0), "{text:?}");
        assert!(d.message.contains("unsupported"), "{}", d.message);
    }
    // Active comments and other unsupported lines declare nothing.
    for text in [
        "\" {x=1}\nx'=q",
        "x'=q\nbndry x-1",
        "x'=q\nonly x",
        "x'=q\nexport {x} {y}",
        "x'=q\nhom_bcs 1",
        "x'=q\noptions a.opt",
    ] {
        assert_eq!(codes(text), vec![UNDEFINED_NAME], "{text:?}");
    }
    // Initial values for such names are downgraded too.
    assert_eq!(codes("v(t)=1\ninit v=1"), vec![MAYBE_UNDEFINED]);
}

#[test]
fn array_templates_are_skipped() {
    clean("x[1..3]'=-x[j]+q");
    clean("%[1..2]\nx[j]'=q\n%");
    clean("par a[1..2]=1\nx'=y[j]");
    clean("aux v[1..2]=q");
    // Plain entries next to indexed ones are still checked.
    assert_eq!(codes("par a[1..2]=1, b=2, b=3"), vec![DUPLICATE_NAME]);
}

#[test]
fn includes_options_and_sets_are_ignored() {
    clean("#include other.ode\n@ total=10, xp=q\nset s {q=1}\nx'=1\ndone\ny'=q");
}

#[test]
fn notes_point_at_declarations() {
    let text = "par a=1\nx'=a(2)";
    let d = check_one(text, NOT_CALLABLE, nth(text, "a", 1), &[nth(text, "a", 0)]);
    assert!(d.notes[0].1.contains("declared"), "{d:#?}");
}

#[test]
fn message_text_per_code() {
    let cases: [(&str, Code, &str); 19] = [
        ("x'=q", UNDEFINED_NAME, "`q` is not declared"),
        ("par sin=1", RESERVED_NAME, "builtin function"),
        ("par a=1\npar a=2", DUPLICATE_NAME, "declared twice"),
        ("b=c\nc=1", FIXED_FORWARD, "below"),
        ("w=w", CIRCULAR, "in terms of itself"),
        ("x'=1\ninit y=1", INIT_NOT_STATE, "not a state"),
        ("x'=1\ninit x=1, x=2", INIT_TWICE, "more than once"),
        ("f(a,b,c,d,e,g,h,i,j,k)=a", TOO_MANY_PARAMS, "at most 9"),
        ("f(u)=f(u)", RECURSIVE_FUNCTION, "calls itself"),
        ("x'=sin(1,2)", WRONG_ARITY, "takes 1 argument"),
        ("x'=sin", NOT_CALLABLE, "is a function"),
        ("x'=arg1", BAD_ARG, "only inside a function"),
        ("aux v=1\nx'=v", AUX_IN_FORMULA, "output only"),
        ("x'=1\ny(t+1)=1", MIXED_KIND, "ODEs and maps"),
        ("x'=1\n!d=x", DERIVED_NON_PARAMETER, "not a parameter"),
        ("x'=delay(x,1)", UNSUPPORTED_DELAY, "not supported yet"),
        (
            "x'=shift(x,1)",
            UNSUPPORTED_SHIFT,
            "`shift` is not supported",
        ),
        (
            "x'=del_shft(x,1,1)",
            UNSUPPORTED_DEL_SHFT,
            "`del_shft` is not supported",
        ),
        ("x'=v\nv(t)=1", MAYBE_UNDEFINED, "may declare it"),
    ];
    for (text, code, needle) in cases {
        let d = one(text);
        assert_eq!(d.code, code, "{text:?}: {d:#?}");
        assert!(d.message.contains(needle), "{text:?}: {}", d.message);
    }
}
