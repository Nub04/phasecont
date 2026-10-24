use std::time::{Duration, Instant};

use super::*;
use crate::ast::{Expr, Stmt, StmtKind};
use crate::check::{AUX_IN_FORMULA, DUPLICATE_NAME, UNDEFINED_NAME};
use crate::diag::{CODES, Severity};
use crate::include::{INCLUDE_IN_BLOCK, MapLoader};
use crate::parser::parse;
use crate::parser::stmt::{BAD_BLOCK, MALFORMED_STATEMENT};
use crate::span::{FileId, SourceMap, Span};
use crate::{Loaded, load};

const F: FileId = FileId(0);

/// Checks what every diagnostic list must satisfy.
fn sane(diags: &[Diagnostic], map: &SourceMap) {
    for d in diags {
        let severity = CODES.iter().find(|(c, _, _)| *c == d.code).unwrap().1;
        assert_eq!(d.severity, severity, "{d:#?}");
        assert!(d.span.end <= map.text(d.span.file).len(), "{d:#?}");
        assert!(!d.message.chars().any(char::is_control), "{d:#?}");
        assert!(d.message.len() <= 340, "{}", d.message);
        for (span, _) in &d.notes {
            assert!(span.end <= map.text(span.file).len(), "{d:#?}");
        }
    }
}

/// Parses `text` and expands it. The parse must give no error.
fn run(text: &str) -> (Model, Vec<Diagnostic>) {
    let (model, parse_diags) = parse(text, F);
    assert!(
        parse_diags.iter().all(|d| d.severity != Severity::Error),
        "{text:?}: {parse_diags:#?}"
    );
    let (expanded, diags) = expand(&model);
    let mut map = SourceMap::new();
    map.add("main.ode", text);
    sane(&diags, &map);
    (expanded, diags)
}

/// Expands `text`, which must give no diagnostic.
fn clean(text: &str) -> Model {
    let (model, diags) = run(text);
    assert!(diags.is_empty(), "{text:?}: {diags:#?}");
    model
}

/// The statements without spans, array indices or template ids.
fn plain(model: &Model) -> Vec<Stmt> {
    model
        .strip_spans()
        .statements
        .into_iter()
        .map(|mut s| {
            s.array_index = None;
            s.array_template = None;
            s
        })
        .collect()
}

/// The statements of `text` as the parser gives them, without spans.
fn flat(text: &str) -> Vec<Stmt> {
    let (model, diags) = parse(text, F);
    assert!(
        diags.iter().all(|d| d.severity != Severity::Error),
        "{diags:#?}"
    );
    model.strip_spans().statements
}

fn indices(model: &Model) -> Vec<Option<i64>> {
    model.statements.iter().map(|s| s.array_index).collect()
}

fn codes(diags: &[Diagnostic]) -> Vec<Code> {
    diags.iter().map(|d| d.code).collect()
}

/// The span of the `n`-th (0-based) occurrence of `needle` in `text`.
fn nth(text: &str, needle: &str, n: usize) -> Span {
    let start = text
        .match_indices(needle)
        .nth(n)
        .unwrap_or_else(|| panic!("{needle:?} not in {text:?}"))
        .0;
    Span::new(F, start, start + needle.len())
}

/// Expands `text`, which must give exactly one diagnostic, with `code` at
/// the `n`-th `needle`.
fn one(text: &str, code: Code, needle: &str, n: usize) -> (Model, Diagnostic) {
    let (model, diags) = run(text);
    assert_eq!(codes(&diags), vec![code], "{text:?}: {diags:#?}");
    let d = diags.into_iter().next().unwrap();
    assert_eq!(d.span, nth(text, needle, n), "{text:?}: {d:#?}");
    (model, d)
}

/// Loads `top` as `main.ode` with the other files in `files`.
fn load_text(top: &str, files: &[(&str, &str)]) -> (Loaded, SourceMap) {
    let mut map = SourceMap::new();
    let id = map.add("main.ode", top);
    let mut loader = MapLoader::new();
    for (name, text) in files {
        loader.insert(*name, *text);
    }
    let loaded = load(id, &mut map, &mut loader);
    sane(&loaded.diagnostics, &map);
    let keys: Vec<(u32, usize)> = loaded
        .diagnostics
        .iter()
        .map(|d| (d.span.file.0, d.span.start))
        .collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "not sorted: {:#?}", loaded.diagnostics);
    (loaded, map)
}

/// True when `d` has a note saying it comes from the copy with `j`.
fn has_copy_note(d: &Diagnostic, j: i64) -> bool {
    d.notes
        .iter()
        .any(|(_, m)| m.contains(&format!("j = {j} ")))
}

// ---------------------------------------------------------------- copies

#[test]
fn plain_statements_pass_through() {
    let text = "par a=1\nx'=-a*x\n# c\n@ dt=.1";
    let model = clean(text);
    assert_eq!(model.strip_spans().statements, flat(text));
    assert_eq!(indices(&model), vec![None; 4]);
}

#[test]
fn every_index_form() {
    let model = clean("x[1..3]'=x[j]+x[j+1]-x[j-1]*y[j*2]");
    assert_eq!(
        plain(&model),
        flat("x1'=x1+x2-x0*y2\nx2'=x2+x3-x1*y4\nx3'=x3+x4-x2*y6")
    );
    assert_eq!(indices(&model), vec![Some(1), Some(2), Some(3)]);
}

#[test]
fn bare_index_forms_become_integers() {
    let model = clean("x[2..3]'=[j]+[j+1]-[j-2]*[j*10]");
    assert_eq!(plain(&model), flat("x2'=2+3-0*20\nx3'=3+4-1*30"));
    // `[j*0]` is 0 at every index.
    let model = clean("aux v[4..5]=[j*0]");
    assert_eq!(plain(&model), flat("aux v4=0\naux v5=0"));
}

#[test]
fn ranges_count_down() {
    let model = clean("x[3..1]'=-x[j]");
    assert_eq!(plain(&model), flat("x3'=-x3\nx2'=-x2\nx1'=-x1"));
    assert_eq!(indices(&model), vec![Some(3), Some(2), Some(1)]);
    // A one-element range.
    assert_eq!(plain(&clean("x[7..7]'=1")), flat("x7'=1"));
}

#[test]
fn range_on_the_declared_name() {
    let cases = [
        ("x[1..2]'=-x[j]", "x1'=-x1\nx2'=-x2"),
        ("x[1..2](t+1)=x[j]", "x1(t+1)=x1\nx2(t+1)=x2"),
        ("aux q[0..1]=[j]", "aux q0=0\naux q1=1"),
        ("c[1..2]=[j]*2", "c1=1*2\nc2=2*2"),
        ("!d[1..2]=[j]", "!d1=1\n!d2=2"),
        ("par a[1..2]=.5", "par a1=.5\npar a2=.5"),
        ("number z[1..2]=3", "number z1=3\nnumber z2=3"),
        ("wiener w[1..2]", "wiener w1\nwiener w2"),
        ("init x[1..2]=[j]", "init x1=1\ninit x2=2"),
        ("x[1..3](0)=[j]", "x1(0)=1\nx2(0)=2\nx3(0)=3"),
        (
            "table tb[1..2] % 3 0 [j] t",
            "table tb1 % 3 0 1 t\ntable tb2 % 3 0 2 t",
        ),
    ];
    for (template, expected) in cases {
        assert_eq!(plain(&clean(template)), flat(expected), "{template:?}");
    }
}

#[test]
fn range_inside_an_expression() {
    let model = clean("global 1 x[1..2]-1 {x[j]=0;y[j+1]=[j]}");
    assert_eq!(
        plain(&model),
        flat("global 1 x1-1 {x1=0;y2=1}\nglobal 1 x2-1 {x2=0;y3=2}")
    );
    let model = clean("aux s=p[1..2]*[j]");
    assert_eq!(plain(&model), flat("aux s=p1*1\naux s=p2*2"));
}

#[test]
fn shared_range_line_with_index_forms() {
    let model = clean("par a[1..2]=.25,b[j]=.3,e[j+1]=[j]");
    assert_eq!(
        plain(&model),
        flat("par a1=.25,b1=.3,e2=1\npar a2=.25,b2=.3,e3=2")
    );
}

#[test]
fn copies_keep_the_template_spans() {
    let text = "par k=1\nx[1..2]'=k*Y[j-1]";
    let (model, _) = run(text);
    for stmt in &model.statements[1..] {
        assert_eq!(stmt.span, nth(text, "x[1..2]'=k*Y[j-1]", 0));
        let StmtKind::Ode { name, rhs, .. } = &stmt.kind else {
            panic!("{stmt:#?}");
        };
        assert_eq!(name.span, nth(text, "x[1..2]", 0));
        assert!(name.index.is_none());
        let Expr::Binary { rhs: y, .. } = rhs else {
            panic!("{rhs:#?}");
        };
        let Expr::Name(y) = &**y else {
            panic!("{y:#?}")
        };
        assert_eq!(y.span, nth(text, "Y[j-1]", 0));
        // The text keeps the written case; the key is lower case.
        let j = stmt.array_index.unwrap();
        assert_eq!(y.text, format!("Y{}", j - 1));
        assert_eq!(y.key, format!("y{}", j - 1));
    }
    // A bare index form takes the span of the brackets.
    let text = "aux v[1..2]=[j+1]";
    let (model, _) = run(text);
    let StmtKind::Aux { rhs, .. } = &model.statements[1].kind else {
        panic!();
    };
    assert_eq!(rhs.span(), nth(text, "[j+1]", 0));
    assert!(matches!(rhs, Expr::Num { value, text, .. } if *value == 3.0 && text == "3"));
}

#[test]
fn blocks_copy_their_body_per_index() {
    let text = "%[1..2]\nx[j]'=-x[j]\n# note\naux v[j]=x[j]+[j]\n%\ny'=1";
    let model = clean(text);
    assert_eq!(
        plain(&model),
        flat("x1'=-x1\n# note\naux v1=x1+1\nx2'=-x2\n# note\naux v2=x2+2\ny'=1")
    );
    assert_eq!(
        indices(&model),
        vec![Some(1), Some(1), Some(1), Some(2), Some(2), Some(2), None]
    );
    // Statements without an index form are copied too.
    assert_eq!(
        plain(&clean("%[1..2]\npar a=1\n%")),
        flat("par a=1\npar a=1")
    );
}

#[test]
fn blocks_count_down() {
    let model = clean("%[2..0]\nx[j]'=[j]\n%");
    assert_eq!(plain(&model), flat("x2'=2\nx1'=1\nx0'=0"));
    assert_eq!(indices(&model), vec![Some(2), Some(1), Some(0)]);
}

#[test]
fn set_names_are_expanded() {
    let model = clean("%[1..2]\nset s[j] {a[j]=1}\n%");
    assert_eq!(plain(&model), flat("set s1 {a1=1}\nset s2 {a2=1}"));
}

#[test]
fn bad_blocks_are_kept_as_written() {
    let text = "%[a..2]\nx[j]'=1\n%";
    let (model, parse_diags) = parse(text, F);
    assert_eq!(codes(&parse_diags), vec![BAD_BLOCK]);
    let (expanded, diags) = expand(&model);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(expanded, model);
}

#[test]
fn indexed_function_names_are_dropped_quietly() {
    // The parser already reports the index.
    let text = "f[j](u)=u\nx'=1";
    let (model, parse_diags) = parse(text, F);
    assert_eq!(codes(&parse_diags), vec![MALFORMED_STATEMENT]);
    let (expanded, diags) = expand(&model);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(plain(&expanded), flat("x'=1"));
}

#[test]
fn the_input_model_is_unchanged() {
    let text = "x[1..2]'=1\n%[1..2]\ny[j]'=1\n%";
    let (model, _) = parse(text, F);
    let before = model.clone();
    let _ = expand(&model);
    assert_eq!(model, before);
}

#[test]
fn chain_as_range_lines_and_as_a_block() {
    let lines = "par d=.1\nx0'=-x0\nx[1..20]'=d*(x[j-1]-x[j])\ny[1..20]'=x[j]-y[j]\n";
    let block = "par d=.1\nx0'=-x0\n%[1..20]\nx[j]'=d*(x[j-1]-x[j])\ny[j]'=x[j]-y[j]\n%\n";
    let a = plain(&clean(lines));
    let b = plain(&clean(block));
    assert_eq!(a.len(), 42);
    assert_ne!(a, b, "the orders should differ");
    let sorted = |v: &[Stmt]| {
        let mut s: Vec<String> = v.iter().map(|s| format!("{s:?}")).collect();
        s.sort();
        s
    };
    assert_eq!(sorted(&a), sorted(&b));
    // Both load without a diagnostic.
    for text in [lines, block] {
        let (loaded, _) = load_text(text, &[]);
        assert!(loaded.diagnostics.is_empty(), "{:#?}", loaded.diagnostics);
        assert_eq!(loaded.info.symbols.len(), 42);
    }
}

// ---------------------------------------------------------------- errors

#[test]
fn more_than_one_range() {
    let text = "x[1..2]'=y[1..2]\nz'=1";
    let (model, d) = one(text, MULTIPLE_RANGES, "[1..2]", 1);
    assert_eq!(d.notes[0].0, nth(text, "[1..2]", 0));
    // The statement is dropped.
    assert_eq!(plain(&model), flat("z'=1"));
    // Ranges are counted across names, values and expressions.
    one("par a[1..2]=1,b[3..4]=2", MULTIPLE_RANGES, "[3..4]", 0);
    one("global 1 x[1..2] {y[1..2]=0}", MULTIPLE_RANGES, "[1..2]", 1);
    // Three ranges give one error, at the second.
    one(
        "aux v=a[1..2]+b[1..2]+c[1..2]",
        MULTIPLE_RANGES,
        "[1..2]",
        1,
    );
}

#[test]
fn range_inside_a_block() {
    let text = "%[1..2]\nx[1..3]'=1\ny[j]'=1\n%";
    let (model, _) = one(text, RANGE_IN_BLOCK, "[1..3]", 0);
    assert_eq!(plain(&model), flat("y1'=1\ny2'=1"));
}

#[test]
fn index_form_with_no_range() {
    let (model, _) = one("x'=y[j]\nz'=1", INDEX_OUT_OF_SCOPE, "[j]", 0);
    assert_eq!(plain(&model), flat("z'=1"));
    one("x'=[j+1]", INDEX_OUT_OF_SCOPE, "[j+1]", 0);
    one("par a=[j]", INDEX_OUT_OF_SCOPE, "[j]", 0);
    one("x[j]'=1", INDEX_OUT_OF_SCOPE, "[j]", 0);
    // One error per statement, at the first form.
    one("x'=y[j]+z[j*2]", INDEX_OUT_OF_SCOPE, "[j]", 0);
    // A block does not reach past its `%`.
    let text = "%[1..2]\nx[j]'=1\n%\ny'=x[j]";
    one(text, INDEX_OUT_OF_SCOPE, "[j]", 1);
}

#[test]
fn negative_block_bound() {
    let text = "%[-1..2]\nx[j]'=1\n%\ny'=1";
    let (model, d) = one(text, NEGATIVE_INDEX, "[-1..2]", 0);
    assert!(d.message.contains("-1"), "{d:#?}");
    assert_eq!(plain(&model), flat("y'=1"));
    one("%[2..-3]\n%", NEGATIVE_INDEX, "[2..-3]", 0);
}

#[test]
fn index_form_below_zero() {
    // `x[j-1]` at j = 0 is negative; the whole line is dropped.
    let text = "x[0..2]'=x[j-1]\nz'=1";
    let (model, d) = one(text, NEGATIVE_INDEX, "x[j-1]", 0);
    assert!(d.message.contains("j = 0"), "{d:#?}");
    assert_eq!(plain(&model), flat("z'=1"));
    // Counting down finds the same smallest index.
    one("x[2..0]'=[j-1]", NEGATIVE_INDEX, "[j-1]", 0);
    // In a block only the statement with the error is dropped.
    let text = "%[1..3]\nx[j]'=x[j-2]\ny[j]'=1\n%";
    let (model, d) = one(text, NEGATIVE_INDEX, "x[j-2]", 0);
    assert!(d.message.contains("j = 1"), "{d:#?}");
    assert_eq!(plain(&model), flat("y1'=1\ny2'=1\ny3'=1"));
    // Each form that goes below 0 is reported.
    let (_, diags) = run("x[0..1]'=a[j-1]+b[j-2]");
    assert_eq!(codes(&diags), vec![NEGATIVE_INDEX, NEGATIVE_INDEX]);
}

#[test]
fn index_too_large() {
    let big = MAX_INDEX + 1;
    let text = format!("%[{big}..{big}]\nx[j]'=1\n%\ny'=1");
    let (model, d) = one(&text, INDEX_TOO_LARGE, &format!("[{big}..{big}]"), 0);
    assert!(d.message.contains(&big.to_string()), "{d:#?}");
    assert_eq!(plain(&model), flat("y'=1"));
    // An index form past the limit at the largest index.
    let top = MAX_INDEX / 2 + 1;
    let text = format!("%[{top}..{}]\nx[j*2]'=1\ny[j]'=1\n%", top - 3);
    let (_, diags) = run(&text);
    assert_eq!(codes(&diags), vec![INDEX_TOO_LARGE], "{diags:#?}");
    assert_eq!(diags[0].span, nth(&text, "x[j*2]", 0));
    // Arithmetic that overflows a 64-bit integer.
    let text = format!("%[{MAX_INDEX}..{MAX_INDEX}]\naux v[j]=[j*4294967295]\n%");
    let (_, diags) = run(&text);
    assert_eq!(codes(&diags), vec![INDEX_TOO_LARGE], "{diags:#?}");
    // The limit itself is allowed.
    let text = format!("%[{MAX_INDEX}..{MAX_INDEX}]\naux v[j]=[j]\n%");
    let model = clean(&text);
    assert_eq!(
        plain(&model),
        flat(&format!("aux v{MAX_INDEX}={MAX_INDEX}"))
    );
}

#[test]
fn markov_in_arrays() {
    let text = "%[1..2]\nmarkov m[j] 2\n{0} {1}\n{1} {0}\ny[j]'=1\n%";
    let (model, _) = one(text, MARKOV_ARRAY, "m[j]", 0);
    assert_eq!(plain(&model), flat("y1'=1\ny2'=1"));
    let text = "markov m[1..2] 2\n{0} {1}\n{1} {0}\ny'=1";
    let (model, _) = one(text, MARKOV_ARRAY, "[1..2]", 0);
    assert_eq!(plain(&model), flat("y'=1"));
    // A range in a row has the same problem.
    let text = "markov m 2\n{0} {a[1..2]}\n{1} {0}";
    one(text, MARKOV_ARRAY, "[1..2]", 0);
}

#[test]
fn the_copy_cap() {
    // A huge block returns at once, with one error and no copies.
    let start = Instant::now();
    let text = "%[0..4000000000]\nx[j]'=1\n%\ny[1..3]'=1\nz'=1\n%[1..2]\nw[j]'=1\n%";
    let (model, d) = one(text, TOO_MANY_COPIES, "[0..4000000000]", 0);
    // Later arrays make no copies; plain statements pass through.
    assert_eq!(plain(&model), flat("z'=1"));
    assert!(d.message.contains("more than 100000 statements"), "{d:#?}");
    // A block with no body makes no statements, however large.
    let model = clean("%[0..4000000000]\n%\nz'=1");
    assert_eq!(plain(&model), flat("z'=1"));
    assert!(start.elapsed() < Duration::from_secs(5));

    // The statement that passes the cap gets the error; the ones before
    // it are expanded.
    let text = "a[1..60000]'=1\nb[1..60000]'=1\nc[1..2]'=1";
    let (model, _) = one(text, TOO_MANY_COPIES, "[1..60000]", 1);
    assert_eq!(model.statements.len(), 60000);
    // Exactly at the cap is fine.
    let model = clean("a[1..50000]'=1\nb[1..50000]'=1");
    assert_eq!(model.statements.len(), MAX_ARRAY_COPIES);
    // A range line with a template error does not count toward the cap.
    let (model, diags) = run("a[1..99999]'=1\nb[0..10]'=[j-1]\nc[1..1]'=1");
    assert_eq!(codes(&diags), vec![NEGATIVE_INDEX]);
    assert_eq!(model.statements.len(), MAX_ARRAY_COPIES);
    // Block copies count body statements times indices.
    let text = "%[1..50001]\nx[j]'=1\ny[j]'=1\n%";
    one(text, TOO_MANY_COPIES, "[1..50001]", 0);
}

// ------------------------------------------------------- through `load`

#[test]
fn load_runs_every_pass() {
    let top = "#include lib.ode\nx[1..2]'=-k*x[j]\n";
    let (loaded, _) = load_text(top, &[("lib.ode", "par k=1")]);
    assert!(loaded.diagnostics.is_empty(), "{:#?}", loaded.diagnostics);
    assert!(matches!(
        loaded.template.statements[0].kind,
        StmtKind::Include { .. }
    ));
    assert_eq!(
        plain(&loaded.expanded),
        flat("par k=1\nx1'=-k*x1\nx2'=-k*x2")
    );
    assert!(loaded.info.symbol("x2").is_some());
    // The template keeps the array form.
    let StmtKind::Ode { name, .. } = &loaded.template.statements[1].kind else {
        panic!();
    };
    assert!(name.index.is_some());
}

#[test]
fn load_sorts_diagnostics_by_file_then_start() {
    let top = "x'=q\n#include lib.ode\n";
    let (loaded, _) = load_text(top, &[("lib.ode", "aux v=r\npar a=[j]")]);
    let got: Vec<(u32, Code)> = loaded
        .diagnostics
        .iter()
        .map(|d| (d.span.file.0, d.code))
        .collect();
    assert_eq!(
        got,
        vec![
            (0, UNDEFINED_NAME),
            (1, UNDEFINED_NAME),
            (1, INDEX_OUT_OF_SCOPE)
        ]
    );
}

#[test]
fn includes_in_blocks_are_dropped_quietly() {
    let top = "%[1..2]\n#include lib.ode\nx[j]'=1\n%";
    let (loaded, _) = load_text(top, &[("lib.ode", "par k=1")]);
    assert_eq!(codes(&loaded.diagnostics), vec![INCLUDE_IN_BLOCK]);
    assert_eq!(plain(&loaded.expanded), flat("x1'=1\nx2'=1"));
}

#[test]
fn undefined_name_in_a_copy_names_the_index() {
    let top = "x[1..3]'=x[j-1]";
    let (loaded, _) = load_text(top, &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![UNDEFINED_NAME]);
    let d = &loaded.diagnostics[0];
    assert_eq!(d.span, nth(top, "x[j-1]", 0));
    assert!(d.message.contains("`x0`"), "{d:#?}");
    assert!(has_copy_note(d, 1), "{d:#?}");
    // The note points at the template line.
    let note = d.notes.iter().find(|(_, m)| m.contains("j = 1 ")).unwrap();
    assert_eq!(note.0, nth(top, top, 0));
    // In a block.
    let top = "%[1..2]\nx[j]'=x[j-1]\n%";
    let (loaded, _) = load_text(top, &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![UNDEFINED_NAME]);
    assert!(has_copy_note(&loaded.diagnostics[0], 1));
    // A diagnostic outside a copy has no such note.
    let (loaded, _) = load_text("x'=q", &[]);
    assert!(loaded.diagnostics[0].notes.is_empty());
}

#[test]
fn identical_diagnostics_from_copies_collapse() {
    let top = "aux v=1\nx[1..20]'=v";
    let (loaded, _) = load_text(top, &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![AUX_IN_FORMULA]);
    // The first copy is the one kept.
    assert!(has_copy_note(&loaded.diagnostics[0], 1));
    // Names that differ by index stay separate.
    let (loaded, _) = load_text("x[1..3]'=y[j]", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![UNDEFINED_NAME; 3]);
    let names: Vec<bool> = (1..=3)
        .map(|k| {
            loaded.diagnostics[k - 1]
                .message
                .contains(&format!("`y{k}`"))
        })
        .collect();
    assert_eq!(names, vec![true; 3]);
    // Two errors in each copy of a block collapse to two in all.
    let (loaded, _) = load_text("aux v=1\n%[1..5]\nx[j]'=v+q\n%", &[]);
    assert_eq!(
        codes(&loaded.diagnostics),
        vec![AUX_IN_FORMULA, UNDEFINED_NAME]
    );
}

#[test]
fn duplicate_from_a_shared_range_line() {
    let top = "par a[1..3]=.5, c=.1";
    let (loaded, _) = load_text(top, &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME]);
    let d = &loaded.diagnostics[0];
    assert_eq!(d.span, nth(top, "c", 0));
    assert!(has_copy_note(d, 2), "{d:#?}");
    assert!(
        d.notes
            .iter()
            .any(|(_, m)| m.contains("does not change with the index")),
        "{d:#?}"
    );
    // A duplicate between two different lines gets no such note.
    let (loaded, _) = load_text("par c=1\npar a[1..2]=.5, c=.1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME]);
    let d = &loaded.diagnostics[0];
    assert!(
        !d.notes
            .iter()
            .any(|(_, m)| m.contains("does not change with the index")),
        "{d:#?}"
    );
    assert!(has_copy_note(d, 1), "{d:#?}");
}

#[test]
fn graph_and_name_length_diagnostics_name_the_copy() {
    use crate::check::CIRCULAR;
    use crate::parser::expr::PORTABILITY;
    // Each copy defines a fixed quantity in terms of itself.
    let (loaded, _) = load_text("q[1..2]=q[j]+1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![CIRCULAR, CIRCULAR]);
    assert!(has_copy_note(&loaded.diagnostics[0], 1));
    assert!(has_copy_note(&loaded.diagnostics[1], 2));
    // Names longer than XPPAUT keeps: the first copy warns, with a count.
    let (loaded, _) = load_text("par conductance[1..2]=1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![PORTABILITY]);
    assert!(has_copy_note(&loaded.diagnostics[0], 1));
}

// ------------------------------------------------------ fix round 1 (R22)

#[test]
fn work_budget_stops_a_wide_par_line() {
    // 400 indexed entries on one range line of 100000 copies.
    let mut text = String::from("par p[1..100000]=0");
    for k in 0..400 {
        text.push_str(&format!(",q{k}[j]=0"));
    }
    text.push_str("\nx'=1");
    let start = Instant::now();
    let (model, d) = one(&text, TOO_MANY_COPIES, "[1..100000]", 0);
    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(
        d.message
            .contains(&format!("more than {MAX_ARRAY_WORK} units of work")),
        "{d:#?}"
    );
    assert_eq!(plain(&model), flat("x'=1"));
}

#[test]
fn work_budget_stops_a_deep_expression() {
    let mut text = String::from("par c=1\nx[1..100000]'=1");
    for _ in 0..150 {
        text.push_str("+c");
    }
    let start = Instant::now();
    let (model, d) = one(&text, TOO_MANY_COPIES, "[1..100000]", 0);
    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(d.message.contains("units of work"), "{d:#?}");
    assert_eq!(plain(&model), flat("par c=1"));
    // The cap holds for later arrays too.
    let (_, diags) = run(&format!("{text}\ny[1..2]'=1"));
    assert_eq!(codes(&diags), vec![TOO_MANY_COPIES]);
}

#[test]
fn work_budget_counts_names_values_and_nodes() {
    // A `par` line with `n` entries weighs 1 + 2n per copy, plus one unit
    // per started 64 bytes of the line.
    let line = |n: usize, copies: usize| {
        let mut text = format!("par p0[1..{copies}]=0");
        for k in 1..n {
            text.push_str(&format!(",p{k}[j]=0"));
        }
        text
    };
    // 999 entries: 9888 bytes, so 1999 + 155 = 2154 units per copy.
    assert_eq!(line(999, 928).len(), 9888);
    // 928 copies are 1998912 units: just under the budget.
    let model = clean(&line(999, 928));
    assert_eq!(model.statements.len(), 928);
    // 929 copies are 2001066 units: just over it.
    one(&line(999, 929), TOO_MANY_COPIES, "[1..929]", 0);
    // Expression nodes count: `x'=c+c` weighs 1 + 1 + 3, plus its length.
    let rhs = vec!["c"; 100].join("+");
    // 1 + 1 + 199 = 201 nodes and 211 bytes (4 units) = 205 per copy;
    // 9756 copies fit, 9757 do not.
    clean(&format!("x[1..9756]'={rhs}"));
    one(
        &format!("x[1..9757]'={rhs}"),
        TOO_MANY_COPIES,
        "[1..9757]",
        0,
    );
    // A block weighs the sum of its body statements per index.
    let text = "%[1..400000]\nx[j]'=1\n%";
    let (_, d) = one(text, TOO_MANY_COPIES, "[1..400000]", 0);
    assert!(d.message.contains("statements"), "{d:#?}");
    let text = "%[1..60000]\nx[j]'=1\ny[j]'=-y[j]\n%";
    // 60000 * (4 + 5) = 540000 units, 120000 statements.
    let (_, d) = one(text, TOO_MANY_COPIES, "[1..60000]", 0);
    assert!(d.message.contains("more than 100000 statements"), "{d:#?}");
}

#[test]
fn work_budget_charges_template_length() {
    // A comment line in a block weighs 1 plus one unit per started 64
    // bytes: "# " and 4000 letters are 4002 bytes, so 1 + 63 = 64 units.
    let long = "a".repeat(4000);
    // 31250 copies are exactly 2000000 units.
    let model = clean(&format!("%[1..31250]\n# {long}\n%"));
    assert_eq!(model.statements.len(), 31250);
    let text = format!("%[1..31251]\n# {long}\n%");
    let (model, d) = one(&text, TOO_MANY_COPIES, "[1..31251]", 0);
    assert!(d.message.contains("units of work"), "{d:#?}");
    assert!(model.statements.is_empty());
    // Each body statement is weighed by its own span: two 64-unit comments
    // and `x[j]'=1` (3 nodes, 1 length unit) are 132 units per index.
    let text = format!("%[1..15151]\n# {long}\nx[j]'=1\n# {long}\n%");
    assert_eq!(clean(&text).statements.len(), 3 * 15151);
    let text = format!("%[1..15152]\n# {long}\nx[j]'=1\n# {long}\n%");
    one(&text, TOO_MANY_COPIES, "[1..15152]", 0);
    // A trailing comment counts as part of its statement: this `par` line
    // is 4020 bytes, so it weighs 3 + 63 = 66 units; 30303 copies fit.
    let text = format!("par p[1..30303]=1 # {long}");
    assert_eq!(text.len(), 4020);
    assert_eq!(clean(&text).statements.len(), 30303);
    let text = format!("par p[1..30304]=1 # {long}");
    one(&text, TOO_MANY_COPIES, "[1..30304]", 0);
}

#[test]
fn a_normal_chain_still_expands() {
    // A 1000-cell chain is far under the budget once length is charged.
    let text = "par k=1, d=0.1\nx0'=0\nx[1..1000]'=k*(x[j-1]-x[j])-d*x[j]\n\
                %[1..1000]\naux e[j]=x[j]^2\n# energy of cell [j]\n%";
    let model = clean(text);
    assert_eq!(model.statements.len(), 2 + 1000 + 2000);
}

#[test]
fn cap_message_counts() {
    // The count is the exact product of the bound and the body length.
    let mut text = format!("%[0..{MAX_INDEX}]\n");
    for k in 0..2049 {
        text.push_str(&format!("v{k}[j]'=1\n"));
    }
    text.push('%');
    let (_, d) = one(&text, TOO_MANY_COPIES, &format!("[0..{MAX_INDEX}]"), 0);
    let count = (MAX_INDEX as u128 + 1) * 2049;
    assert!(
        d.message.contains(&format!("makes {count} statements")),
        "{d:#?}"
    );
    // One statement is singular.
    let (_, d) = one("a[1..100000]'=1\nb[1..1]'=1", TOO_MANY_COPIES, "[1..1]", 0);
    assert!(d.message.contains("makes 1 statement,"), "{d:#?}");
}

#[test]
fn copies_carry_a_template_id() {
    let model = clean("x[1..2]'=1\n%[1..2]\na[j]'=1\nb[j]'=1\n%\nz'=1");
    let ids: Vec<Option<u32>> = model.statements.iter().map(|s| s.array_template).collect();
    let (x, a, b) = (ids[0], ids[2], ids[3]);
    assert!(x.is_some() && a.is_some() && b.is_some());
    assert_eq!(ids, vec![x, x, a, b, a, b, None]);
    assert!(x != a && a != b && x != b);
    // The parser sets none.
    let (parsed, _) = parse("x[1..2]'=1", F);
    assert_eq!(parsed.statements[0].array_template, None);
}

#[test]
fn dropped_lines_do_not_cascade() {
    let cases = [
        ("x[0..2]'=x[j-1]\ny'=x1+x2", NEGATIVE_INDEX),
        (
            "a[1..100000]=1\nb[1..3]'=-b[j]\nz'=b1+b2+b3",
            TOO_MANY_COPIES,
        ),
        ("markov m[1..2] 2\n{0} {1}\n{1} {0}\nx'=m1", MARKOV_ARRAY),
        // A plain name on a dropped line.
        ("par k[0..1]=[j-1],c=2\nx'=k1+c", NEGATIVE_INDEX),
        // A dropped block drops every body statement.
        ("%[-1..2]\nx[j]'=1\npar k=1\n%\ny'=x1+k", NEGATIVE_INDEX),
        // Two ranges, and `[j]` with no range.
        ("x[1..2]'=y[1..2]\nz'=x1", MULTIPLE_RANGES),
        ("aux w[j]=1\nz'=w3", INDEX_OUT_OF_SCOPE),
        // A `global` target.
        ("x[0..1]'=[j-1]\nglobal 1 t-1 {x1=0}", NEGATIVE_INDEX),
    ];
    for (top, code) in cases {
        let (loaded, _) = load_text(top, &[]);
        assert_eq!(
            codes(&loaded.diagnostics),
            vec![code],
            "{top:?}: {:#?}",
            loaded.diagnostics
        );
    }
    // Only the root followed by a plain index is suppressed.
    let top = "x[0..2]'=x[j-1]\ny'=x0+x10+x01+xq+x";
    let (loaded, _) = load_text(top, &[]);
    let undefined: Vec<&str> = loaded
        .diagnostics
        .iter()
        .filter(|d| d.code == UNDEFINED_NAME)
        .map(|d| d.message.as_str())
        .collect();
    assert_eq!(undefined.len(), 3, "{:#?}", loaded.diagnostics);
    for name in ["`x01`", "`xq`", "`x`"] {
        assert!(
            undefined.iter().any(|m| m.contains(name)),
            "{name}: {undefined:#?}"
        );
    }
    // The record is data on the expanded model, so `check` alone sees it.
    let (model, _) = parse("x[0..2]'=x[j-1]\ny'=x1", F);
    let (expanded, _) = expand(&model);
    assert_eq!(
        expanded.suppressed,
        vec![SuppressedName {
            key: "x".to_string(),
            indexed: true
        }]
    );
    let (_, diags) = crate::check(&expanded);
    assert!(diags.is_empty(), "{diags:#?}");
}

#[test]
fn shared_line_note_needs_one_template_and_one_name() {
    let note = "does not change with the index of this array line";
    let has_note = |d: &Diagnostic| d.notes.iter().any(|(_, m)| m.contains(note));
    // (a) Both names have an index and come from one copy.
    let (loaded, _) = load_text("par a[1..2]=.5, a[j]=.1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME; 2]);
    assert!(
        !loaded.diagnostics.iter().any(has_note),
        "{:#?}",
        loaded.diagnostics
    );
    // (b) One file included twice: separate templates.
    let top = "#include inc1.ode\n#include inc1.ode\n";
    let (loaded, _) = load_text(top, &[("inc1.ode", "par a[1..2]=.5")]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME; 2]);
    assert!(
        !loaded.diagnostics.iter().any(has_note),
        "{:#?}",
        loaded.diagnostics
    );
    // (c) An index that does not change with `j`.
    let top = "par a[0..1]=1, b[j*0]=2";
    let (loaded, _) = load_text(top, &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME]);
    let d = &loaded.diagnostics[0];
    assert_eq!(d.span, nth(top, "b[j*0]", 0));
    let text = &d.notes.iter().find(|(_, m)| m.contains(note)).unwrap().1;
    assert_eq!(
        text,
        "`b0` does not change with the index of this array line, so every copy declares it \
         again"
    );
    // Spec input A: the duplicate comes from two different names.
    let (loaded, _) = load_text("par a[1..3]=1, a[j+1]=2", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![DUPLICATE_NAME; 2]);
    assert!(
        !loaded.diagnostics.iter().any(has_note),
        "{:#?}",
        loaded.diagnostics
    );
}

#[test]
fn separate_inclusions_do_not_collapse() {
    // (d) One error per inclusion.
    let top = "aux v=1\n#include inc2.ode\n#include inc2.ode\n";
    let (loaded, _) = load_text(top, &[("inc2.ode", "y[1..3]'=v")]);
    let aux = loaded
        .diagnostics
        .iter()
        .filter(|d| d.code == AUX_IN_FORMULA)
        .count();
    assert_eq!(aux, 2, "{:#?}", loaded.diagnostics);
}

#[test]
fn long_name_warnings_collapse_per_template() {
    use crate::parser::expr::PORTABILITY;
    let (loaded, _) = load_text("averylongname[1..100000]'=1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![PORTABILITY]);
    let d = &loaded.diagnostics[0];
    assert!(has_copy_note(d, 1), "{d:#?}");
    assert!(
        d.notes
            .iter()
            .any(|(_, m)| m == "99999 more copies of this array line give the same warning"),
        "{d:#?}"
    );
    let (loaded, _) = load_text("par conductance[1..2]=1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![PORTABILITY]);
    assert!(
        loaded.diagnostics[0]
            .notes
            .iter()
            .any(|(_, m)| m == "1 more copy of this array line gives the same warning"),
        "{:#?}",
        loaded.diagnostics
    );
    // Two templates warn separately.
    let (loaded, _) = load_text("par conductance[1..2]=1\naux resistance[1..2]=1", &[]);
    assert_eq!(codes(&loaded.diagnostics), vec![PORTABILITY; 2]);
}

#[test]
fn load_keeps_the_pass_order_at_one_start() {
    use crate::parser::expr::PRECEDENCE;
    // The parser warns about the power chain and `check` about `q`, both at
    // the start of `q`.
    let top = "x'=q^2^3";
    let (loaded, _) = load_text(top, &[]);
    let got: Vec<(usize, Code)> = loaded
        .diagnostics
        .iter()
        .map(|d| (d.span.start, d.code))
        .collect();
    let at = nth(top, "q", 0).start;
    assert_eq!(got, vec![(at, PRECEDENCE), (at, UNDEFINED_NAME)]);
}

#[test]
fn load_keeps_the_pass_order_at_many_equal_starts() {
    use crate::parser::expr::PRECEDENCE;
    // 200 lines, each with a parser warning and a `check` error at the same
    // start. A sort that is not stable would swap some of these pairs: short
    // inputs (under 20 items) are sorted by insertion, which is stable by
    // accident, so the input has to be long.
    let top: String = (0..200).map(|i| format!("y{i}'=u{i}^2^3\n")).collect();
    let (loaded, _) = load_text(&top, &[]);
    let got: Vec<(usize, Code)> = loaded
        .diagnostics
        .iter()
        .map(|d| (d.span.start, d.code))
        .collect();
    assert_eq!(got.len(), 400);
    for (i, pair) in got.chunks(2).enumerate() {
        let at = nth(&top, &format!("u{i}^"), 0).start;
        assert_eq!(pair, [(at, PRECEDENCE), (at, UNDEFINED_NAME)], "line {i}");
    }
}
