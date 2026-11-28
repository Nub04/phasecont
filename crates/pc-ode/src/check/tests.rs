use super::*;
use crate::diag::{CODES, Severity};
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
        ("f(u,t)=u", "t"),
        ("f(u,pi)=u", "pi"),
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
    // A second aux of one parameter name is a duplicate, and so is a second
    // parameter or a fixed quantity next to an aux that shares the name.
    for text in [
        "x'=-x\npar p=1\naux p=1\naux p=2",
        "x'=-x\naux p=1\npar p=1\naux p=2",
        "x'=-x\npar p=1\naux p=1\np=3",
        "x'=-x\naux p=1\npar p=1\npar p=2",
        "x'=-x\npar p=1\naux p=1\npar p=2",
        "x'=-x\naux p=1\npar p=1\np=3",
    ] {
        assert_eq!(codes(text), vec![DUPLICATE_NAME], "{text:?}");
    }
}

/// XPPAUT 8.2.2 accepts `aux f=...` for a name that is also a fixed
/// quantity, in either order. Every reader of `f` gets the fixed quantity
/// and the aux column keeps its own formula (`aux-fixed-*` probes).
#[test]
fn aux_may_repeat_a_fixed_name() {
    for text in [
        "x'=-x+0*f\nf=2*x+1\naux f=7*x",
        "x'=-x+0*f\naux f=7*x\nf=2*x+1",
        "x'=-x\nf=2*x+1\naux f=f+1",
        "x'=-x\naux f=f+1\nf=2*x+1",
        "x'=-x\nf=2*x+1\naux F=7*x",
    ] {
        let info = info(text);
        // The name resolves to the fixed quantity, so a read of `f` is not
        // an aux misuse (AUX_IN_FORMULA).
        assert_eq!(
            info.symbol("f").unwrap().kind,
            SymbolKind::Fixed,
            "{text:?}"
        );
        let count = |want: fn(SymbolKind) -> bool| {
            info.symbols
                .iter()
                .filter(|s| s.name.key == "f" && want(s.kind))
                .count()
        };
        assert_eq!(count(|k| k == SymbolKind::Fixed), 1, "{text:?}");
        assert_eq!(count(|k| k == SymbolKind::Aux), 1, "{text:?}");
    }
    // Other formulas read the fixed quantity, whatever the order.
    clean("x'=-x\ny'=f\naux f=7*x\nf=2*x+1\naux g=f");
    // Only one aux per fixed name; the repeat is reported as before.
    for text in [
        "x'=-x\nf=2*x+1\naux f=1\naux f=2",
        "x'=-x\naux f=1\nf=2*x+1\naux f=2",
        "x'=-x\naux f=1\naux f=2\nf=2*x+1",
    ] {
        let (_, diags) = run(text);
        let dups = diags.iter().filter(|d| d.code == DUPLICATE_NAME).count();
        assert_eq!(dups, 1, "{text:?}: {diags:#?}");
    }
    // A second fixed quantity is still a duplicate.
    for text in ["x'=-x\nf=1\naux f=2\nf=3", "x'=-x\naux f=2\nf=1\nf=3"] {
        assert_eq!(codes(text), vec![DUPLICATE_NAME], "{text:?}");
    }
    // An aux repeating a state, a function or a derived parameter stays an
    // error.
    for text in [
        "x'=-x\naux x=2",
        "x'=-x\ng(u)=u\naux g=2",
        "x'=-x\n!d=2\naux d=2",
    ] {
        assert_eq!(codes(text), vec![DUPLICATE_NAME], "{text:?}");
    }
}

/// XPPAUT 8.2.2 accepts `aux p=...` for a name that is also a parameter, in
/// either order. Every reader of `p` gets the parameter and the aux column
/// keeps its own formula (`aux-param-*` probes).
#[test]
fn aux_may_repeat_a_parameter_name() {
    for text in [
        "x'=-x*p\npar p=2\naux p=7*x",
        "x'=-x*p\naux p=7*x\npar p=2",
        "x'=-x\npar p=2\naux p=7*p",
        "x'=-x\naux p=7*p\npar p=2",
        "x'=-x\npar p=2\naux P=7*x",
        "x'=-x\npar p=2\nq=p\naux p=q",
        "x'=-x\npar p=2\naux p=7*x\naux g=p",
    ] {
        let info = info(text);
        // The name resolves to the parameter, so a read of `p` is not an aux
        // misuse (AUX_IN_FORMULA).
        assert_eq!(
            info.symbol("p").unwrap().kind,
            SymbolKind::Parameter,
            "{text:?}"
        );
        let count = |want: SymbolKind| {
            info.symbols
                .iter()
                .filter(|s| s.name.key == "p" && s.kind == want)
                .count()
        };
        assert_eq!(count(SymbolKind::Parameter), 1, "{text:?}");
        assert_eq!(count(SymbolKind::Aux), 1, "{text:?}");
    }
    // A reader of the name gets the parameter, whatever the order.
    clean("x'=-x\ny'=p\naux p=7*x\npar p=2\naux g=p");
    // A global target of that name is the parameter.
    clean("x'=-x\npar p=2\naux p=7*x\nglobal 1 t-0.5 {p=3}");
    // A parameter and a fixed quantity of one name stay a duplicate.
    assert_eq!(codes("x'=-x\npar p=2\np=1"), vec![DUPLICATE_NAME]);
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
    // A call into a lower line is a forward reference too.
    assert_eq!(codes("a=g(2)\ng(u)=u*b\nb=1"), vec![FIXED_FORWARD]);
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
    // F-4: a warning that names the depth cap, not an error.
    assert_eq!(d.severity, Severity::Warning);
    assert!(d.message.contains("1,000,000"), "{}", d.message);
    let text = "f(u)=g(u)\ng(u)=h(u)\nh(u)=f(u)";
    let d = check_one(
        text,
        RECURSIVE_FUNCTION,
        nth(text, "f", 0),
        &[nth(text, "g", 1), nth(text, "h", 1)],
    );
    assert_eq!(d.severity, Severity::Warning);
    assert!(d.message.contains("1,000,000"), "{}", d.message);
    clean("f(u)=g(u)+g(u)\ng(u)=u");
}

#[test]
fn recursive_set_holds_the_cyclic_components_only() {
    let keys = |text: &str| -> Vec<String> { run(text).0.recursive.into_iter().collect() };
    assert_eq!(keys("F(u)=f(u)"), ["f"]);
    assert_eq!(
        keys("f(u)=g(u)\ng(u)=h(u)\nh(u)=f(u)\nk(u)=f(u)"),
        ["f", "g", "h"]
    );
    assert_eq!(
        keys("ev(n)=od(n-1)\nod(n)=ev(n-1)\nw(n)=w(n)"),
        ["ev", "od", "w"]
    );
    assert!(keys("f(u)=g(u)+g(u)\ng(u)=u").is_empty());
    // A cycle through a fixed quantity is ODE0043, not recursion.
    assert!(keys("a=f(1)\nf(u)=a*u\nx'=a").is_empty());
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
    assert_eq!(
        codes("f(u,v)=arg2+arg1*u"),
        vec![POSITIONAL_ARG, POSITIONAL_ARG]
    );
    let text = "f(u)=arg2";
    let d = check_one(text, BAD_ARG, nth(text, "arg2", 0), &[]);
    assert!(d.message.contains("`arg2`"), "{}", d.message);
    assert_eq!(codes("x'=arg1"), vec![BAD_ARG]);
}

/// XPPAUT binds a declared formal named `argK` by name, so `f(u,arg1)` has
/// `arg1` as its second argument. The declaration is allowed and wins over the
/// positional meaning inside that function.
#[test]
fn a_declared_arg_formal_is_accepted() {
    clean("f(u,arg1)=u+arg1");
    clean("f(arg2,arg1)=arg1-arg2");
    clean("f(arg1)=1");
    clean("f(arg9)=arg9*2");
    clean("f(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9)=arg9-arg1");
    // The formal is not a global symbol.
    let i = info("f(u,arg1)=u+arg1");
    assert!(i.symbol("arg1").is_none());
    // A call of the function checks the count as usual.
    clean("x'=f(x,2)\nf(u,arg1)=u+arg1");
    assert_eq!(codes("x'=f(x)\nf(u,arg1)=u+arg1"), vec![WRONG_ARITY]);
}

#[test]
fn a_declared_arg_formal_beyond_the_arity_is_not_an_arity_error() {
    // `arg3` is declared, so it is the second parameter and not argument 3.
    clean("f(u,arg3)=u+arg3");
    clean("f(arg5)=arg5");
    // An `argK` that is not declared and exceeds the arity is still ODE0050.
    let text = "f(u,v)=arg3";
    let d = check_one(text, BAD_ARG, nth(text, "arg3", 0), &[]);
    assert!(d.message.contains("`arg3`"), "{}", d.message);
    // A declared `arg3` does not make the other `argK` valid.
    let text = "f(u,arg3)=arg4";
    check_one(text, BAD_ARG, nth(text, "arg4", 0), &[]);
    // The declaration is local to its function.
    let text = "f(u,arg3)=u+arg3\ng(u,v)=arg3";
    check_one(text, BAD_ARG, nth(text, "arg3", 2), &[]);
    // Outside every function it stays an error, and so does a call.
    assert_eq!(
        codes("f(u,arg1)=u+arg1\nx'=arg1"),
        vec![BAD_ARG],
        "global use of arg1"
    );
}

#[test]
fn undeclared_arg_stays_positional_next_to_a_declared_one() {
    // `arg2` is declared; `arg1` and `arg3` read arguments 1 and 3, with
    // ODE0078 at each.
    let text = "g(arg2,v,w)=arg1*100+arg2*10+arg3";
    let spans: Vec<Span> = run(text).1.iter().map(|d| d.span).collect();
    assert_eq!(spans, vec![nth(text, "arg1", 0), nth(text, "arg3", 0)]);
    assert_eq!(codes(text), vec![POSITIONAL_ARG, POSITIONAL_ARG]);
    // `arg4` is above the arity of 3.
    assert_eq!(codes("g(arg2,v,w)=arg4"), vec![BAD_ARG]);
    // A declared formal is not a function.
    assert_eq!(codes("f(u,arg3)=arg3(1)"), vec![NOT_CALLABLE]);
    // A repeated formal is still a duplicate.
    let text = "f(arg1,arg1)=1";
    let d = check_one(
        text,
        DUPLICATE_NAME,
        nth(text, "arg1", 1),
        &[nth(text, "arg1", 0)],
    );
    assert!(d.message.contains("`arg1`"), "{}", d.message);
}

/// XPPAUT 8.2.2 stops on an `argK` that the function does not declare.
/// PhaseCont reads it as argument `K` and warns.
#[test]
fn an_undeclared_arg_within_the_count_warns() {
    let text = "f(u)=arg1*2";
    let d = check_one(text, POSITIONAL_ARG, nth(text, "arg1", 0), &[]);
    assert_eq!(d.severity, Severity::Warning);
    assert!(d.message.contains("`arg1`"), "{}", d.message);
    assert!(d.message.contains("argument 1"), "{}", d.message);
    assert!(d.message.contains("XPPAUT 8.2.2"), "{}", d.message);
    let text = "x'=f(x,2)\nf(u,v)=u+arg2";
    check_one(text, POSITIONAL_ARG, nth(text, "arg2", 0), &[]);
    // Each use is reported.
    assert_eq!(
        codes("f(u)=arg1*arg1"),
        vec![POSITIONAL_ARG, POSITIONAL_ARG]
    );
    // A declared formal named `argK` reads by name and does not warn.
    clean("f(u,arg1)=u+arg1");
    // Above the count it is still the error ODE0050, not the warning.
    assert_eq!(codes("f(u)=arg2"), vec![BAD_ARG]);
}

#[test]
fn every_other_declaration_of_an_arg_name_is_still_reserved() {
    for (text, name) in [
        ("par arg1=1\nx'=-x", "arg1"),
        ("par a=1, arg9=2\nx'=-x", "arg9"),
        ("aux arg1=2\nx'=-x", "arg1"),
        ("number arg3=2\nx'=-x", "arg3"),
        ("arg2=3\nx'=-x", "arg2"),
        ("!arg4=3\nx'=-x", "arg4"),
        ("arg1'=1", "arg1"),
        ("arg1(u)=u", "arg1"),
    ] {
        let (_, diags) = run(text);
        let reserved: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == RESERVED_NAME).collect();
        assert_eq!(reserved.len(), 1, "{text:?}: {diags:#?}");
        assert!(
            reserved[0].message.contains(&format!("`{name}`")),
            "{}",
            reserved[0].message
        );
        assert!(
            reserved[0].message.contains("names a function argument"),
            "{}",
            reserved[0].message
        );
    }
}

#[test]
fn the_other_reserved_names_are_still_refused_as_formals() {
    for (text, name) in [("f(u,t)=u", "t"), ("f(pi)=1", "pi"), ("f(u,sin)=u", "sin")] {
        let (_, diags) = run(text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        assert_eq!(diags[0].code, RESERVED_NAME, "{text:?}");
        assert!(diags[0].message.contains(&format!("`{name}`")), "{text:?}");
    }
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
    // Names that agree in their first 10 characters get a warning each and
    // no note: XPPAUT 8.0 does not collide them (probe `name-11-collide`).
    let text = "par abcdefghijk=1, ABCDEFGHIJl=2\nx'=abcdefghijk+abcdefghijl";
    let (_, diags) = run(text);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert_eq!(diags[0].span, nth(text, "abcdefghijk", 0));
    assert!(diags.iter().all(|d| d.notes.is_empty()), "{diags:#?}");
    assert!(!diags[0].message.contains("agrees"), "{diags:#?}");
    // A long name next to its own 10-character prefix.
    let text = "par abcdefghij=1, abcdefghijk=2";
    let d = check_one(text, PORTABILITY, nth(text, "abcdefghijk", 0), &[]);
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
    let cases: [(&str, Code, &str); 20] = [
        ("x'=q", UNDEFINED_NAME, "`q` is not declared"),
        ("par sin=1", RESERVED_NAME, "builtin function"),
        ("par a=1\npar a=2", DUPLICATE_NAME, "declared twice"),
        ("b=c\nc=1", FIXED_FORWARD, "later line"),
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
        (
            "x'=delay(x,1)",
            UNSUPPORTED_DELAY,
            "is read by `phasecont.continuation`",
        ),
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
        ("x'=1\nglobal 1 x {t=0}", BAD_TARGET, "cannot be assigned"),
    ];
    for (text, code, needle) in cases {
        let d = one(text);
        assert_eq!(d.code, code, "{text:?}: {d:#?}");
        assert!(d.message.contains(needle), "{text:?}: {}", d.message);
    }
}

// ---------------------------------------------------------------------------
// Review fix round 1.

#[test]
fn long_names_scale_linearly() {
    let n = 5000;
    let mut text = String::new();
    for i in 0..n {
        text.push_str(&format!("par longparameter{i}=1\n"));
    }
    text.push_str("x'=1\n");
    let start = std::time::Instant::now();
    let (_, diags) = run(&text);
    let elapsed = start.elapsed();
    assert_eq!(diags.len(), n);
    assert!(diags.iter().all(|d| d.code == PORTABILITY));
    assert!(diags.iter().all(|d| d.notes.is_empty()), "no notes");
    assert!(
        elapsed < std::time::Duration::from_secs(1),
        "{n} colliding long names took {elapsed:?}"
    );
}

#[test]
fn long_name_notes() {
    // Names with the same first 10 characters get no note.
    let text = "par abcdefghijk1=1, abcdefghijk2=2, abcdefghijk3=3";
    let (_, diags) = run(text);
    assert_eq!(diags.len(), 3);
    assert!(diags.iter().all(|d| d.notes.is_empty()), "{diags:#?}");
    // A name declared twice gets one warning, plus the duplicate error.
    let text = "par abcdefghijklm=1\npar abcdefghijklm=2\nx'=1";
    let (_, diags) = run(text);
    let got: Vec<(Code, Span)> = diags.iter().map(|d| (d.code, d.span)).collect();
    assert_eq!(
        got,
        vec![
            (PORTABILITY, nth(text, "abcdefghijklm", 0)),
            (DUPLICATE_NAME, nth(text, "abcdefghijklm", 1)),
        ]
    );
    assert!(diags[0].notes.is_empty(), "{diags:#?}");
    let text = "f(abcdefghijk1,abcdefghijk2)=1";
    let (_, diags) = run(text);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert!(diags.iter().all(|d| d.notes.is_empty()), "{diags:#?}");
    // Not with global names, which a parameter hides.
    let (_, diags) = run("par abcdefghijkz=1\nf(abcdefghijkq)=1\ng(abcdefghijkr)=1");
    assert_eq!(diags.len(), 3, "{diags:#?}");
    assert!(diags.iter().all(|d| d.notes.is_empty()), "{diags:#?}");
}

#[test]
fn fixed_forward_through_functions() {
    let text = "a2=g(1)\ng(u)=u*b2\nb2=3\nx'=0\naux v=a2";
    let d = check_one(
        text,
        FIXED_FORWARD,
        nth(text, "g", 0),
        &[nth(text, "b2", 1)],
    );
    assert!(
        d.message.contains("`b2`") && d.message.contains("`g`"),
        "{}",
        d.message
    );
    // Through two functions, defined in any order.
    let text = "h(u)=u*b\na=g(1)\ng(u)=h(u)\nb=3";
    check_one(text, FIXED_FORWARD, nth(text, "g", 0), &[nth(text, "b", 1)]);
    // A fixed quantity defined above is fine, wherever the function is.
    clean("b=3\na=g(1)\ng(u)=u*b");
    // A function defined below is fine on its own.
    clean("a=g(1)\ng(u)=2*u");
    // A cycle through a function is one error, without forward errors.
    assert_eq!(codes("a=g(1)\ng(u)=u+b\nb=a"), vec![CIRCULAR]);
    // Uses from ODEs, aux and functions are never forward references.
    clean("x'=g(1)\ng(u)=u*b\nb=3");
}

#[test]
fn derived_through_functions() {
    for (text, used) in [
        ("!d4=h(1)\nh(u)=u+x\npar a=1\nx'=d4", "x"),
        ("!d=g(1)\ng(u)=u*t", "t"),
        ("w=1\n!d=g(1)\ng(u)=u*w", "w"),
        ("x'=v\nwiener v\n!d=g(1)\ng(u)=u*v", "v"),
        ("!d=g(1)\ng(u)=h(u)\nh(u)=u*x\nx'=1", "x"),
    ] {
        let (_, diags) = run(text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, DERIVED_NON_PARAMETER, "{text:?}");
        let bang = text.find('!').unwrap();
        let eq = bang + text[bang..].find('=').unwrap();
        let callee = &text[eq + 1..eq + 2];
        assert_eq!(d.span, Span::new(F, eq + 1, eq + 2), "{text:?}");
        assert!(d.message.contains(&format!("`{callee}`")), "{}", d.message);
        assert!(d.message.contains(&format!("`{used}`")), "{}", d.message);
        assert!(d.message.contains("not a parameter"), "{}", d.message);
        let last_use = text
            .rfind(&format!("*{used}"))
            .or_else(|| text.rfind(&format!("+{used}")));
        let at = last_use.unwrap() + 1;
        assert_eq!(d.notes.len(), 1, "{d:#?}");
        assert_eq!(d.notes[0].0, Span::new(F, at, at + used.len()), "{text:?}");
    }
    clean("par a=1\n!d=g(1)\ng(u)=u*a");
    clean("!e=1\n!d=g(1)\ng(u)=u*e*pi+sin(u)");
    // Recursion does not hide the derived error.
    let c = codes("!d=g(1)\ng(u)=h(u)+x\nh(u)=g(u)\nx'=1");
    assert!(
        c.contains(&DERIVED_NON_PARAMETER) && c.contains(&RECURSIVE_FUNCTION),
        "{c:?}"
    );
}

#[test]
fn derived_cycle_through_a_function_still_reports_the_state() {
    // `d` and `g` use each other, so ODE0043 reports the cycle; `g` also
    // reads the state `x`, which ODE0053 reports on the call.
    let text = "!d=g(1)\ng(u)=u*d+x\nx'=1";
    let c = codes(text);
    assert!(c.contains(&CIRCULAR), "{c:?}");
    assert!(c.contains(&DERIVED_NON_PARAMETER), "{c:?}");
    // Without the state there is only the cycle.
    assert_eq!(codes("!d=g(1)\ng(u)=u*d"), vec![CIRCULAR]);
}

#[test]
fn comparisons_and_logic_are_not_smooth() {
    for text in [
        "x'=-x+(x>1)",
        "x'=x<1",
        "x'=x<=1",
        "x'=x>=1",
        "x'=x==1",
        "x'=x&1",
        "x'=x|1",
        "x'=not(x)",
        "f(u)=u>1\nx'=1",
    ] {
        assert!(!info(text).smooth, "{text:?}");
    }
    assert!(info("x'=-x+x^2-x/2").smooth);
}

#[test]
fn if_in_an_uncalled_function_is_not_smooth() {
    // The flags look at every formula, called or not.
    assert!(!info("f(u)=if(u>0)then(1)else(0)\nx'=1").smooth);
}

#[test]
fn init_of_a_reserved_name() {
    // One fault, one diagnostic.
    assert_eq!(codes("t'=1\nt(0)=2"), vec![RESERVED_NAME]);
    assert_eq!(codes("par sin=1\ninit sin=2\nx'=1"), vec![RESERVED_NAME]);
    // A reserved name nobody tried to declare is still not a state.
    let text = "x'=1\ninit pi=1";
    let d = check_one(text, INIT_NOT_STATE, nth(text, "pi", 0), &[]);
    assert!(d.message.contains("`pi` is a constant"), "{}", d.message);
}

#[test]
fn global_targets() {
    // States and parameters can be assigned (measured in XPPAUT 8.0).
    clean("par a=1\nx'=a\nglobal 1 x-1 {a=5}");
    clean("x'=1\nglobal 1 x-1 {x=0}");
    clean("markov z 2\n{0} {1}\n{1} {0}\nx'=z\nglobal 1 x-1 {z=0}");
    for (pre, target) in [
        ("", "t"),
        ("", "sin"),
        ("", "pi"),
        ("", "arg1"),
        ("", "q"),
        ("aux v=1\n", "v"),
        ("w=1\n", "w"),
        ("!d=1\n", "d"),
        ("f(u)=u\n", "f"),
        ("number c=1\n", "c"),
        ("table h % 11 0 1 t\n", "h"),
        ("wiener n\n", "n"),
    ] {
        let text = format!("{pre}x'=1\nglobal 1 x-1 {{{target}=0}}");
        let (_, diags) = run(&text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, BAD_TARGET, "{text:?}");
        let at = text.rfind(&format!("{{{target}=")).unwrap() + 1;
        assert_eq!(d.span, Span::new(F, at, at + target.len()), "{text:?}");
        assert!(d.message.contains(&format!("`{target}`")), "{}", d.message);
        assert!(d.message.contains("cannot be assigned"), "{}", d.message);
        if !pre.is_empty() {
            assert_eq!(d.notes.len(), 1, "{d:#?}");
            assert!(d.notes[0].1.contains("declared"), "{d:#?}");
        }
    }
    // An unsupported statement that may declare the target softens it.
    assert_eq!(
        codes("v(t)=1\nx'=1\nglobal 1 x-1 {q=0}"),
        vec![MAYBE_UNDEFINED]
    );
    // The right side is still read.
    assert_eq!(codes("x'=1\nglobal 1 x-1 {x=q}"), vec![UNDEFINED_NAME]);
}

#[test]
fn global_arret_and_out_put_are_event_actions() {
    // XPPAUT 8.2.2 reads `arret=` and `out_put=` in a `global` action list
    // as stop-the-run and record-a-row (probes global-arret, global-out-put).
    // The model is a simulation-only model, so the IR refuses it by name.
    for text in [
        "x'=1\nglobal 1 x-0.5 {arret=1}",
        "x'=1\nglobal 1 x-0.5 {out_put=1}",
        "x'=1\nglobal 1 x-0.5 {ARRET=(x>0.8)}",
        "x'=1\nglobal 1 x-0.5 {Out_Put=1; x=0}",
    ] {
        let (info, diags) = run(text);
        assert!(diags.is_empty(), "{text:?}: {diags:#?}");
        assert!(info.simulation_only, "{text:?}");
    }
    // A declared parameter or state of that name is an ordinary target
    // (probes global-arret-par, global-arret-state).
    clean("par arret=0\nx'=1\nglobal 1 x-0.5 {arret=1}");
    clean("x'=1\narret'=0\nglobal 1 x-0.5 {arret=1}");
    // Neither word is a value: XPPAUT refuses a model that reads `arret`.
    assert_eq!(
        codes("x'=1\nglobal 1 x-0.5 {arret=1}\naux a=arret"),
        vec![UNDEFINED_NAME]
    );
    // A shorter spelling is still an undeclared target.
    assert_eq!(codes("x'=1\nglobal 1 x-0.5 {arre=1}"), vec![BAD_TARGET]);
    assert_eq!(codes("x'=1\nglobal 1 x-0.5 {out_pu=1}"), vec![BAD_TARGET]);
    // A fixed quantity of that name stays refused.
    assert_eq!(
        codes("arret=2\nx'=1\nglobal 1 x-0.5 {arret=1}"),
        vec![BAD_TARGET]
    );
}

#[test]
fn arg_calls() {
    assert_eq!(codes("f(u)=arg2(1)"), vec![BAD_ARG]);
    assert_eq!(codes("x'=arg1(2)"), vec![BAD_ARG]);
    assert_eq!(codes("f(u,v)=arg2(1)"), vec![NOT_CALLABLE]);
}

#[test]
fn message_grammar() {
    let d = one("x'=arg1");
    assert!(d.message.contains("at least 1 parameter"), "{}", d.message);
    assert!(!d.message.contains("1 parameters"), "{}", d.message);
    let d = one("f(u)=arg2");
    assert!(d.message.contains("at least 2 parameters"), "{}", d.message);
    let d = one("b=c\nc=1");
    assert!(
        !(d.message.contains("above") && d.message.contains("below")),
        "{}",
        d.message
    );
}

#[test]
fn delay_with_one_argument() {
    let text = "x'=delay(x)";
    let (_, diags) = run(text);
    assert_eq!(
        diags,
        vec![
            Diagnostic {
                code: UNSUPPORTED_DELAY,
                severity: Severity::Warning,
                span: nth(text, "delay", 0),
                message: "`delay` is read by `phasecont.continuation` (equilibria, Hopf curves and periodic orbits from a Hopf point) and by `pc-cli dde-eq`, `dde-branch`, `dde-hopf`, `dde-lyap`, `dde-zh`, `dde-hh` and `dde-cycle`; simulation and the ODE-only entries do not support it"
                    .to_string(),
                fix_it: None,
                fixes: Vec::new(),
                notes: Vec::new(),
            },
            Diagnostic {
                code: WRONG_ARITY,
                severity: Severity::Error,
                span: nth(text, "delay(x)", 0),
                message: "`delay` takes 2 arguments, not 1".to_string(),
                fix_it: None,
                fixes: Vec::new(),
                notes: Vec::new(),
            },
        ]
    );
}

#[test]
fn unsupported_statement_and_undefined_function() {
    let text = "v(t)=1\nx'=foo(x)";
    let d = check_one(text, MAYBE_UNDEFINED, nth(text, "foo", 0), &[sp(0, 6)]);
    assert_eq!(
        d.message,
        "function `foo` is not declared; an unsupported statement may declare it"
    );
}

#[test]
fn symbol_lookup() {
    let info = info("par a=1\nx'=a");
    assert_eq!(info.symbol("a").unwrap().kind, SymbolKind::Parameter);
    assert_eq!(info.symbol("x").unwrap().stmt, 1);
    assert!(info.symbol("q").is_none());
}

/// The model of the `meth` tests, with `entries` on its `@` lines.
fn logistic_with(lines: &[&str]) -> String {
    let mut text = "par r=2\nx'=r*x*(1-x)\n".to_string();
    for l in lines {
        text.push_str("@ ");
        text.push_str(l);
        text.push('\n');
    }
    text
}

#[test]
fn meth_with_d_makes_an_ode_file_a_map() {
    for entry in [
        "meth=discrete",
        "meth=d",
        "METH=Discrete",
        "method=discrete",
        "methx=dx",
    ] {
        let text = logistic_with(&[entry]);
        let (info, diags) = run(&text);
        assert_eq!(info.kind, ModelKind::Map, "{entry}");
        assert_eq!(diags.len(), 1, "{entry}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, METH_MAP, "{entry}");
        assert_eq!(d.severity, Severity::Warning, "{entry}");
        assert_eq!(d.span, nth(&text, entry, 0), "{entry}");
        assert_eq!(
            d.message,
            format!(
                "`{entry}` makes XPPAUT 8.0 iterate this file as a map, so PhaseCont reads it \
                 as a map"
            )
        );
    }
}

#[test]
fn meth_with_another_letter_keeps_an_ode_and_warns() {
    for entry in ["meth=ode45", "meth=xyz", "meth=1", "meth=Fe", "meth=/"] {
        let text = logistic_with(&[entry]);
        let (info, diags) = run(&text);
        assert_eq!(info.kind, ModelKind::Ode, "{entry}");
        assert_eq!(diags.len(), 1, "{entry}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, METH_MAP, "{entry}");
        assert_eq!(d.span, nth(&text, entry, 0), "{entry}");
        assert_eq!(
            d.message,
            format!(
                "XPPAUT 8.0 iterates this file as a map for `{entry}`, but PhaseCont integrates \
                 it as an ODE; write `x(t+1)=...` for a map"
            )
        );
    }
}

#[test]
fn meth_naming_an_integrator_changes_nothing() {
    for entry in [
        "meth=adams",
        "meth=backeul",
        "meth=cvode",
        "meth=euler",
        "meth=gear",
        "meth=modeuler",
        "meth=qualrk",
        "meth=rungekutta",
        "meth=RungeKutta",
        "meth=stiff",
        "meth=ymp",
        "meth=2rb",
        "meth=5dp",
        "meth=83",
        "meth=volterra",
    ] {
        let text = logistic_with(&[entry]);
        assert_eq!(info(&text).kind, ModelKind::Ode, "{entry}");
    }
}

#[test]
fn the_last_meth_entry_decides() {
    let text = logistic_with(&["meth=discrete", "meth=euler"]);
    assert_eq!(info(&text).kind, ModelKind::Ode);
    let text = logistic_with(&["meth=euler, method=d"]);
    let d = one(&text);
    assert_eq!((d.code, d.span), (METH_MAP, nth(&text, "method=d", 0)));
    assert_eq!(run(&text).0.kind, ModelKind::Map);
    // A spaced entry is ignored, as XPPAUT 8.0 ignores it, so the entry
    // before it still decides.
    let text = logistic_with(&["meth=d", "meth = euler"]);
    let d = one(&text);
    assert_eq!((d.code, d.span), (METH_MAP, nth(&text, "meth=d", 0)));
    assert_eq!(run(&text).0.kind, ModelKind::Map);
}

#[test]
fn a_spaced_meth_entry_is_ignored() {
    let text = logistic_with(&["meth = discrete"]);
    let (model, parse_diags) = parse(&text, F);
    let parse_codes: Vec<Code> = parse_diags.iter().map(|d| d.code).collect();
    assert_eq!(parse_codes, vec![crate::parser::stmt::SPACED_OPTION]);
    let (info, diags) = check(&model);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(info.kind, ModelKind::Ode);
}

#[test]
fn meth_does_not_touch_a_map_a_mixed_file_or_a_file_without_equations() {
    for entry in ["meth=euler", "meth=ode45", "meth=discrete"] {
        let text = format!("x(t+1)=x/2\n@ {entry}\n");
        assert_eq!(info(&text).kind, ModelKind::Map, "{entry}");
        let text = format!("x'=1\ny(t+1)=y\n@ {entry}\n");
        assert_eq!(codes(&text), vec![MIXED_KIND], "{entry}");
        assert_eq!(run(&text).0.kind, ModelKind::Ode, "{entry}");
        let text = format!("par a=1\n@ {entry}\n");
        assert_eq!(info(&text).kind, ModelKind::None, "{entry}");
    }
}

// ---------------------------------------------------------------------------
// `only` lists (round E).

/// The kept `only` list of a model that checks without a diagnostic.
fn only_list(text: &str) -> Vec<String> {
    info(text).only
}

#[test]
fn only_keeps_states_aux_and_t_in_order() {
    assert_eq!(
        only_list("x'=-x\ny'=x\naux a=x+y\nonly a,t,y"),
        vec!["a", "t", "y"]
    );
    // Blanks and commas both separate; two lines add to one list in order.
    assert_eq!(
        only_list("x'=-x\ny'=x\naux a=x+y\nonly y a\nonly , x ,t"),
        vec!["y", "a", "x", "t"]
    );
    // A map state is a state too.
    assert_eq!(only_list("x(t+1)=x/2\nonly x,t"), vec!["x", "t"]);
}

#[test]
fn only_matches_names_without_regard_to_case_and_keeps_the_declared_spelling() {
    assert_eq!(
        only_list("Xy'=-Xy\naux Qq=Xy\nonly qQ,XY,T"),
        vec!["Qq", "Xy", "t"]
    );
}

#[test]
fn no_only_line_and_a_bare_only_give_no_list() {
    assert!(only_list("x'=-x").is_empty());
    assert!(only_list("x'=-x\nonly").is_empty());
    assert!(only_list("x'=-x\nonly , ,\nonly").is_empty());
}

#[test]
fn only_drops_a_name_that_is_not_a_state_an_aux_or_t() {
    let cases = [
        ("par k=1", "k", "a parameter"),
        ("number k=1", "k", "a number"),
        ("!k=2*3", "k", "a derived parameter"),
        ("k=2*x", "k", "a fixed quantity"),
        ("k(u)=u", "k", "a function"),
        ("table k % 3 0 1 t", "k", "a table"),
        ("wiener k", "k", "a Wiener variable"),
    ];
    for (decl, name, noun) in cases {
        let text = format!("x'=-x\n{decl}\nonly x,{name}");
        let (info, diags) = run(&text);
        assert_eq!(info.only, vec!["x"], "{text:?}");
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, ONLY_DROPPED, "{text:?}");
        assert_eq!(d.span, nth(&text, name, 1), "{text:?}: {d:#?}");
        assert!(d.message.contains(noun), "{text:?}: {}", d.message);
        assert!(d.message.contains("only"), "{text:?}: {}", d.message);
    }
}

#[test]
fn only_drops_an_unknown_name() {
    let text = "x'=-x\nonly x,zz";
    let (info, diags) = run(text);
    assert_eq!(info.only, vec!["x"]);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, ONLY_DROPPED);
    assert_eq!(diags[0].span, nth(text, "zz", 0));
    assert!(
        diags[0].message.contains("not declared"),
        "{}",
        diags[0].message
    );
}

#[test]
fn only_drops_a_repeat_and_points_at_the_first_use() {
    for (text, want) in [
        ("x'=-x\nonly x,x", vec!["x"]),
        ("x'=-x\nonly x,X", vec!["x"]),
        ("x'=-x\nonly x\nonly x", vec!["x"]),
        ("x'=-x\ny'=x\nonly y,x,y", vec!["y", "x"]),
        ("x'=-x\nonly t,T", vec!["t"]),
    ] {
        let (info, diags) = run(text);
        assert_eq!(info.only, want, "{text:?}");
        assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, ONLY_DROPPED, "{text:?}");
        assert!(d.message.contains("already"), "{text:?}: {}", d.message);
        assert_eq!(d.notes.len(), 1, "{text:?}: {d:#?}");
        assert!(d.notes[0].0.start < d.span.start, "{text:?}: {d:#?}");
    }
}

#[test]
fn a_list_with_no_name_left_is_no_list() {
    for text in [
        "x'=-x\npar k=1\nonly k",
        "x'=-x\nonly nothing,k",
        "x'=-x\npar k=1\nonly k\nonly zz",
    ] {
        let (info, diags) = run(text);
        assert!(info.only.is_empty(), "{text:?}");
        assert!(!diags.is_empty(), "{text:?}");
        assert!(diags.iter().all(|d| d.code == ONLY_DROPPED), "{diags:#?}");
    }
}

#[test]
fn only_keeps_an_aux_that_shares_its_name_with_a_parameter_or_fixed_quantity() {
    // XPPAUT 8.2.2 writes the aux column (measured), and the name index of
    // the checker keeps pointing at the parameter, so the list looks at
    // every symbol of the name.
    for text in [
        "x'=-x\npar a=1\naux a=2*x\nonly a",
        "x'=-x\naux a=2*x\npar a=1\nonly a",
        "x'=-x\nf=x\naux f=2*x\nonly f",
    ] {
        assert_eq!(only_list(text).len(), 1, "{text:?}");
    }
}

#[test]
fn only_reads_the_text_after_a_hash_as_names() {
    // XPPAUT 8.2.2 reads it as names (measured). `#` is never a name and
    // `c` is dropped as unknown, each with the hint; a state named `c` is
    // kept.
    let text = "x'=-x\nonly x # c";
    let (info, diags) = run(text);
    assert_eq!(info.only, vec!["x"]);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert_eq!(diags[0].span, nth(text, "#", 0));
    assert_eq!(diags[1].span, nth(text, "c", 0));
    for d in &diags {
        assert_eq!(d.code, ONLY_DROPPED);
        assert!(
            d.message.contains("line of its own"),
            "the hint is missing: {}",
            d.message
        );
    }
    let text = "x'=-x\nc'=x\nonly x # c";
    let (info, diags) = run(text);
    assert_eq!(info.only, vec!["x", "c"]);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].span, nth(text, "#", 0));
}

#[test]
fn only_is_not_read_as_a_declaration_or_a_use() {
    // Naming a state does not count as a use, and naming an undefined name
    // is ODE0077, not ODE0039.
    assert_eq!(codes("x'=-x\nonly q"), vec![ONLY_DROPPED]);
    assert!(
        !info("x'=-x\nonly x")
            .symbols
            .iter()
            .any(|s| s.name.key == "q")
    );
}

#[test]
fn only_keeps_the_list_of_a_model_that_has_other_errors() {
    // The list follows the declarations that exist, whatever else is wrong.
    let (info, diags) = run("x'=-x+q\nonly x");
    assert_eq!(info.only, vec!["x"]);
    assert_eq!(
        diags.iter().map(|d| d.code).collect::<Vec<_>>(),
        vec![UNDEFINED_NAME]
    );
}
