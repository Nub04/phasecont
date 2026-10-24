//! Limits on the work and the output of one pipeline run: the statement and
//! size caps of the include pass and the diagnostic caps of every pass.

use pc_ode::{
    Diagnostic, MAX_DIAGNOSTICS, MAX_INCLUDED_BYTES, MAX_INCLUDED_STATEMENTS, MapLoader, SourceMap,
    TOO_MANY_DIAGNOSTICS, check, expand, load, parse, resolve_includes,
};

fn load_text(top: &str, files: &[(String, String)]) -> pc_ode::Loaded {
    let mut map = SourceMap::new();
    let id = map.add("main.ode", top);
    let mut loader = MapLoader::new();
    for (name, text) in files {
        loader.insert(name.as_str(), text.as_str());
    }
    load(id, &mut map, &mut loader)
}

/// `n` ODE lines, each using a different undefined name.
fn undefined_lines(n: usize) -> String {
    (0..n).map(|i| format!("y{i}'=u{i}\n")).collect()
}

fn count(diags: &[Diagnostic], code: pc_ode::Code) -> usize {
    diags.iter().filter(|d| d.code == code).count()
}

/// The number in "N more diagnostics not shown".
fn dropped_in(d: &Diagnostic) -> usize {
    assert_eq!(d.code, TOO_MANY_DIAGNOSTICS);
    d.message
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no count in {:?}", d.message))
}

#[test]
fn small_file_included_many_times_stops_at_the_statement_cap() {
    // 50 comment lines per copy: 2000 copies are exactly the cap.
    let leaf = "# c\n".repeat(50);
    let files = [("leaf.ode".to_string(), leaf)];
    let at_cap = "#include leaf.ode\n".repeat(MAX_INCLUDED_STATEMENTS / 50);
    let loaded = load_text(&at_cap, &files);
    assert_eq!(loaded.expanded.statements.len(), MAX_INCLUDED_STATEMENTS);
    assert!(loaded.diagnostics.is_empty(), "{:#?}", loaded.diagnostics);

    let over = "#include leaf.ode\n".repeat(MAX_INCLUDED_STATEMENTS / 50 + 5);
    let loaded = load_text(&over, &files);
    let too_much: Vec<_> = loaded
        .diagnostics
        .iter()
        .filter(|d| d.code == pc_ode::include::INCLUDE_TOO_MUCH)
        .collect();
    assert_eq!(too_much.len(), 1, "{:#?}", loaded.diagnostics);
    assert_eq!(loaded.diagnostics.len(), 1, "{:#?}", loaded.diagnostics);
    assert_eq!(loaded.expanded.statements.len(), MAX_INCLUDED_STATEMENTS);
    // It points at the first include past the cap.
    let line = "#include leaf.ode\n".len() * (MAX_INCLUDED_STATEMENTS / 50);
    assert_eq!(too_much[0].span.start, line);
    assert!(too_much[0].message.contains("100000"));
}

#[test]
fn text_cap_stops_a_few_long_statements() {
    // One 200000-byte comment per copy.
    let leaf = format!("# {}\n", "x".repeat(199_998));
    let files = [("leaf.ode".to_string(), leaf)];
    let fits = MAX_INCLUDED_BYTES / 200_000;
    let loaded = load_text(&"#include leaf.ode\n".repeat(fits), &files);
    assert!(loaded.diagnostics.is_empty(), "{:#?}", loaded.diagnostics);
    assert_eq!(loaded.expanded.statements.len(), fits);

    let loaded = load_text(&"#include leaf.ode\n".repeat(fits + 3), &files);
    assert_eq!(loaded.diagnostics.len(), 1, "{:#?}", loaded.diagnostics);
    assert_eq!(
        loaded.diagnostics[0].code,
        pc_ode::include::INCLUDE_TOO_MUCH
    );
    assert!(loaded.diagnostics[0].message.contains("bytes"));
    assert_eq!(loaded.expanded.statements.len(), fits);
}

#[test]
fn load_keeps_at_most_the_limit_and_says_how_many_more() {
    // One undefined name per line, all different: nothing collapses.
    let n = MAX_DIAGNOSTICS + 5_000;
    let text = undefined_lines(n);
    let loaded = load_text(&text, &[]);
    let d = &loaded.diagnostics;
    assert_eq!(d.len(), MAX_DIAGNOSTICS + 1);
    assert_eq!(dropped_in(&d[MAX_DIAGNOSTICS]), 5_000);
    assert_eq!(count(d, pc_ode::check::UNDEFINED_NAME), MAX_DIAGNOSTICS);
}

#[test]
fn check_alone_keeps_at_most_the_limit() {
    let text = undefined_lines(MAX_DIAGNOSTICS + 7);
    let (model, _) = parse(&text, pc_ode::FileId(0));
    let (_, d) = check(&model);
    assert_eq!(d.len(), MAX_DIAGNOSTICS + 1);
    assert_eq!(dropped_in(d.last().unwrap()), 7);
    // Under the limit there is no summary.
    let (model, _) = parse("x'=0+a+b", pc_ode::FileId(0));
    let (_, d) = check(&model);
    assert_eq!(d.len(), 2);
}

#[test]
fn expand_alone_keeps_at_most_the_limit() {
    let n = MAX_DIAGNOSTICS + 11;
    let text = "x[j]=1\n".repeat(n);
    let (model, parse_diags) = parse(&text, pc_ode::FileId(0));
    assert!(parse_diags.is_empty());
    let (_, d) = expand(&model);
    assert_eq!(d.len(), MAX_DIAGNOSTICS + 1);
    assert_eq!(dropped_in(d.last().unwrap()), 11);
}

#[test]
fn include_pass_alone_keeps_at_most_the_limit() {
    // 4000 files with 3 parse errors each, read once each.
    let one = "par a=(\npar b=(\npar c=(\n";
    let per_file = parse(one, pc_ode::FileId(0)).1.len();
    assert!(per_file >= 1);
    let files = 4000;
    let mut map = SourceMap::new();
    let mut loader = MapLoader::new();
    let mut top = String::new();
    for i in 0..files {
        loader.insert(format!("f{i}.ode"), one);
        top.push_str(&format!("#include f{i}.ode\n"));
    }
    let id = map.add("main.ode", top.as_str());
    let (model, _) = parse(&top, id);
    let (_, d) = resolve_includes(&model, id, &mut map, &mut loader);
    let total = files * per_file;
    assert!(total > MAX_DIAGNOSTICS, "make the test bigger");
    assert_eq!(d.len(), MAX_DIAGNOSTICS + 1);
    assert_eq!(dropped_in(d.last().unwrap()), total - MAX_DIAGNOSTICS);
}

#[test]
fn copies_of_one_line_do_not_fill_the_list() {
    // 20000 copies of a call with 14 undefined names: the copies give the
    // same diagnostics, which collapse as they arrive.
    let text = "x[1..20000]'=f(a,b,c,d,e,g,h,i,j,k,l,m,n,o)\n";
    let loaded = load_text(text, &[]);
    let d = &loaded.diagnostics;
    assert!(d.len() < 40, "{} diagnostics", d.len());
    assert_eq!(count(d, TOO_MANY_DIAGNOSTICS), 0);
    // Each of the 14 names is reported once, at its copy of j = 1.
    assert_eq!(count(d, pc_ode::check::UNDEFINED_NAME), 15);
}

#[test]
fn a_full_list_still_collapses_nothing_wrongly() {
    // Fill the list, then add copies: they are counted, not stored, and the
    // result still ends with exactly one summary.
    let mut text = undefined_lines(MAX_DIAGNOSTICS);
    text.push_str("\nx[1..1000]'=f(b,c,d)\n");
    let loaded = load_text(&text, &[]);
    let d = &loaded.diagnostics;
    assert_eq!(d.len(), MAX_DIAGNOSTICS + 1);
    assert_eq!(count(d, TOO_MANY_DIAGNOSTICS), 1);
    assert!(dropped_in(d.last().unwrap()) >= 4);
}
