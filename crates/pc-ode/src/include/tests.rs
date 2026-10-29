use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::ast::Stmt;
use crate::check::{DUPLICATE_NAME, check};
use crate::diag::{CODES, Severity};
use crate::parser::stmt::{TRAILING_COMMENT, UNRECOGNIZED_STATEMENT};

/// Parses `top` as `main.ode` and resolves its includes from `files`.
fn resolve(top: &str, files: &[(&str, &str)]) -> (Model, Model, Vec<Diagnostic>, SourceMap) {
    let mut map = SourceMap::new();
    let id = map.add("main.ode", top);
    let (model, _) = parse(top, id);
    let mut loader = MapLoader::new();
    for (name, text) in files {
        loader.insert(*name, *text);
    }
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    for d in &diags {
        let severity = CODES.iter().find(|(c, _, _)| *c == d.code).unwrap().1;
        assert_eq!(d.severity, severity, "{d:#?}");
        assert!(d.span.end <= map.text(d.span.file).len(), "{d:#?}");
        assert!(!d.message.chars().any(char::is_control), "{d:#?}");
    }
    (model, resolved, diags, map)
}

/// The statements of `text`, without spans.
fn flat(text: &str) -> Vec<Stmt> {
    parse(text, FileId(0)).0.strip_spans().statements
}

fn codes(diags: &[Diagnostic]) -> Vec<Code> {
    diags.iter().map(|d| d.code).collect()
}

/// The span of the `n`-th (0-based) occurrence of `needle` in `text`.
fn nth(file: FileId, text: &str, needle: &str, n: usize) -> Span {
    let start = text
        .match_indices(needle)
        .nth(n)
        .unwrap_or_else(|| panic!("{needle:?} not in {text:?}"))
        .0;
    Span::new(file, start, start + needle.len())
}

#[test]
fn one_include() {
    let (_, resolved, diags, map) = resolve(
        "x'=-x\n#include lib.ode\naux v=b",
        &[("lib.ode", "# lib\npar b=1")],
    );
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("x'=-x\n# lib\npar b=1\naux v=b")
    );
    // The included statements carry the included file's id and spans.
    let lib = FileId(1);
    assert_eq!(map.name(lib), "lib.ode");
    assert_eq!(
        resolved.statements[2].span,
        nth(lib, "# lib\npar b=1", "par b=1", 0)
    );
    assert_eq!(resolved.statements[0].span.file, FileId(0));
    assert_eq!(resolved.statements[3].span.file, FileId(0));
    let (_, check_diags) = check(&resolved);
    assert!(check_diags.is_empty(), "{check_diags:#?}");
}

#[test]
fn template_keeps_the_include() {
    let (template, resolved, _, _) = resolve("#include lib.ode # c", &[("lib.ode", "par b=1")]);
    assert!(matches!(
        &template.statements[0].kind,
        StmtKind::Include { path, .. } if path == "lib.ode"
    ));
    assert!(template.statements[0].comment.is_some());
    assert_eq!(resolved.strip_spans().statements, flat("par b=1"));
}

#[test]
fn trailing_comment_still_resolves() {
    // R17 d: the parser reports ODE0022 once; resolving the file anyway keeps
    // undefined names from cascading.
    let top = "#include lib.ode # c\nx'=b";
    let (_, parse_diags) = parse(top, FileId(0));
    assert_eq!(codes(&parse_diags), [TRAILING_COMMENT]);
    let (_, resolved, diags, _) = resolve(top, &[("lib.ode", "par b=1")]);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(resolved.strip_spans().statements, flat("par b=1\nx'=b"));
    assert!(check(&resolved).1.is_empty());
}

#[test]
fn nested_includes() {
    let (_, resolved, diags, map) = resolve(
        "#include a.ode\nx'=1",
        &[
            ("a.ode", "par a=1\n#include b.ode\npar a2=2"),
            ("b.ode", "par b=3\n#include c.ode"),
            ("c.ode", "par c=4"),
        ],
    );
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par a=1\npar b=3\npar c=4\npar a2=2\nx'=1")
    );
    let files: Vec<FileId> = resolved.statements.iter().map(|s| s.span.file).collect();
    assert_eq!(
        files,
        [FileId(1), FileId(2), FileId(3), FileId(1), FileId(0)]
    );
    assert_eq!(map.name(FileId(3)), "c.ode");
}

#[test]
fn diamond_is_allowed() {
    // R17 g: the same file twice without a cycle is spliced twice, and
    // `check` reports the duplicate names. R19 B: it is read and parsed once
    // and keeps one FileId, so its parse diagnostics appear once.
    let (_, resolved, diags, map) = resolve(
        "#include a.ode\n#include b.ode\nx'=1",
        &[
            ("a.ode", "#include c.ode"),
            ("b.ode", "#include c.ode"),
            ("c.ode", "par c=1\npar q=("),
        ],
    );
    assert_eq!(
        codes(&diags),
        [crate::parser::stmt::BAD_VALUE],
        "{diags:#?}"
    );
    assert_eq!(map.name(diags[0].span.file), "c.ode");
    assert_eq!(map.file_count(), 4);
    assert_eq!(map.name(FileId(2)), "c.ode");
    assert_eq!(map.name(FileId(3)), "b.ode");
    let files: Vec<FileId> = resolved.statements.iter().map(|s| s.span.file).collect();
    assert_eq!(
        files,
        [FileId(2), FileId(2), FileId(2), FileId(2), FileId(0)]
    );
    assert_eq!(resolved.statements[0], resolved.statements[2]);
    let (_, check_diags) = check(&resolved);
    assert!(
        codes(&check_diags).contains(&DUPLICATE_NAME),
        "{check_diags:#?}"
    );
}

#[test]
fn self_include_is_a_cycle() {
    let top = "#include main.ode\nx'=1";
    let (_, resolved, diags, _) = resolve(top, &[("main.ode", top)]);
    assert_eq!(codes(&diags), [INCLUDE_CYCLE]);
    let d = &diags[0];
    assert_eq!(d.span, nth(FileId(0), top, "#include main.ode", 0));
    assert!(d.message.contains("main.ode -> main.ode"), "{}", d.message);
    // The include is dropped and the rest of the file stays.
    assert_eq!(resolved.strip_spans().statements, flat("x'=1"));
}

#[test]
fn two_file_cycle() {
    let top = "#include b.ode\nx'=1";
    let b = "par b=1\n#include a.ode";
    let mut map = SourceMap::new();
    let id = map.add("a.ode", top);
    let (model, _) = parse(top, id);
    let mut loader = MapLoader::new();
    loader.insert("b.ode", b);
    loader.insert("a.ode", top);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert_eq!(codes(&diags), [INCLUDE_CYCLE]);
    let d = &diags[0];
    // The error points at the line in `b.ode` that closes the cycle.
    assert_eq!(d.span, nth(FileId(1), b, "#include a.ode", 0));
    assert!(
        d.message.contains("a.ode -> b.ode -> a.ode"),
        "{}",
        d.message
    );
    // A note points at the include line in `a.ode`.
    assert_eq!(d.notes.len(), 1, "{d:#?}");
    assert_eq!(d.notes[0].0, nth(FileId(0), top, "#include b.ode", 0));
    assert_eq!(resolved.strip_spans().statements, flat("par b=1\nx'=1"));
}

#[test]
fn map_loader_top_spelled_differently() {
    // With `MapLoader` the top file is not known under its include
    // spelling, so a self include loads one extra copy before the copy's
    // own self include closes the cycle.
    let top = "#include model.ode\npar a=1";
    let mut map = SourceMap::new();
    let id = map.add("/m/model.ode", top);
    let (model, _) = parse(top, id);
    let mut loader = MapLoader::new();
    loader.insert("model.ode", top);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert_eq!(codes(&diags), [INCLUDE_CYCLE]);
    assert_eq!(diags[0].span.file, FileId(1));
    assert!(
        diags[0]
            .message
            .contains("/m/model.ode -> model.ode -> model.ode"),
        "{}",
        diags[0].message
    );
    assert_eq!(resolved.strip_spans().statements, flat("par a=1\npar a=1"));
}

#[test]
fn missing_file() {
    let top = "x'=1\n#include nope.ode\npar a=1";
    let (_, resolved, diags, map) = resolve(top, &[]);
    assert_eq!(codes(&diags), [INCLUDE_UNREADABLE]);
    let d = &diags[0];
    assert_eq!(d.span, nth(FileId(0), top, "#include nope.ode", 0));
    assert!(d.message.contains("`nope.ode`"), "{}", d.message);
    assert!(d.message.contains("main.ode -> nope.ode"), "{}", d.message);
    assert!(d.notes.is_empty());
    assert_eq!(resolved.strip_spans().statements, flat("x'=1\npar a=1"));
    // Nothing was added to the source map.
    assert_eq!(map.file_count(), 1);
}

#[test]
fn missing_file_in_a_nested_include() {
    let a = "par a=1\n#include gone.ode";
    let top = "#include a.ode";
    let (_, _, diags, _) = resolve(top, &[("a.ode", a)]);
    assert_eq!(codes(&diags), [INCLUDE_UNREADABLE]);
    let d = &diags[0];
    assert_eq!(d.span, nth(FileId(1), a, "#include gone.ode", 0));
    assert!(
        d.message.contains("main.ode -> a.ode -> gone.ode"),
        "{}",
        d.message
    );
    assert_eq!(
        d.notes,
        [(
            nth(FileId(0), top, "#include a.ode", 0),
            "included here".to_string()
        )]
    );
}

#[test]
fn done_ends_the_included_file_only() {
    // P-I3: `done` in an included file ends that file; the including file
    // goes on.
    let (_, resolved, diags, _) = resolve(
        "#include d.ode\nx'=0\naux v=b\npar c=5\ndone\npar z=1",
        &[("d.ode", "par b=3\ndone\npar b=9")],
    );
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par b=3\nx'=0\naux v=b\npar c=5\ndone")
    );
    // Only the top file's `done` is left.
    let dones = resolved
        .statements
        .iter()
        .filter(|s| s.kind == StmtKind::Done)
        .count();
    assert_eq!(dones, 1);
    assert_eq!(resolved.statements.last().unwrap().span.file, FileId(0));
}

#[test]
fn done_in_a_nested_include() {
    let (_, resolved, diags, _) = resolve(
        "#include a.ode\nx'=1",
        &[
            ("a.ode", "#include b.ode\npar a=1\ndone\npar a=2"),
            ("b.ode", "par b=1\ndone"),
        ],
    );
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par b=1\npar a=1\nx'=1")
    );
}

#[test]
fn diagnostics_in_an_included_file_carry_its_file_id() {
    let lib = "par b=1\n+q=2";
    let (_, resolved, diags, map) = resolve("#include lib.ode\nx'=b", &[("lib.ode", lib)]);
    assert_eq!(codes(&diags), [UNRECOGNIZED_STATEMENT]);
    let d = &diags[0];
    assert_eq!(map.name(d.span.file), "lib.ode");
    assert_eq!(d.span.file, FileId(1));
    assert_eq!(d.span, nth(FileId(1), lib, "+q=2", 0));
    assert_eq!(resolved.strip_spans().statements, flat("par b=1\nx'=b"));
}

#[test]
fn check_diagnostics_point_into_the_included_file() {
    let lib = "par b=1\naux w=nope";
    let (_, resolved, diags, _) = resolve("#include lib.ode\nx'=b", &[("lib.ode", lib)]);
    assert!(diags.is_empty());
    let (_, check_diags) = check(&resolved);
    assert_eq!(check_diags.len(), 1, "{check_diags:#?}");
    assert_eq!(check_diags[0].span, nth(FileId(1), lib, "nope", 0));
}

/// A chain `main.ode -> f1.ode -> ... -> f<n>.ode`, where the last file
/// declares `par last=1`.
fn chain(n: usize) -> Vec<(String, String)> {
    (1..=n)
        .map(|i| {
            let text = if i == n {
                "par last=1".to_string()
            } else {
                format!("#include f{}.ode", i + 1)
            };
            (format!("f{i}.ode"), text)
        })
        .collect()
}

#[test]
fn depth_limit() {
    // Sixteen nested files load: the deepest is at depth 16.
    let files = chain(MAX_INCLUDE_DEPTH);
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let (_, resolved, diags, _) = resolve("#include f1.ode", &refs);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(resolved.strip_spans().statements, flat("par last=1"));

    // Seventeen do not: the include in f16.ode is at depth 17.
    let files = chain(MAX_INCLUDE_DEPTH + 1);
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let (_, resolved, diags, map) = resolve("#include f1.ode", &refs);
    assert_eq!(codes(&diags), [INCLUDE_TOO_DEEP]);
    let d = &diags[0];
    assert_eq!(map.name(d.span.file), "f16.ode");
    assert_eq!(
        d.span,
        nth(d.span.file, "#include f17.ode", "#include f17.ode", 0)
    );
    assert!(d.message.contains("16"), "{}", d.message);
    assert!(
        d.message.contains("main.ode -> f1.ode -> f2.ode")
            && d.message.contains("f16.ode -> f17.ode"),
        "{}",
        d.message
    );
    assert_eq!(d.notes.len(), 16, "{d:#?}");
    // f17.ode was never read.
    assert_eq!(map.file_count(), 17);
    assert!(resolved.statements.is_empty());
}

#[test]
fn quoted_path() {
    // R17 b: XPPAUT does not load a quoted name (P-I2). The error has a
    // fix-it that removes the quotes, and the unquoted name still loads.
    for (top, inner) in [
        ("#include \"lib.ode\"\nx'=b", "\"lib.ode\""),
        ("#include 'lib.ode'\nx'=b", "'lib.ode'"),
    ] {
        let (_, resolved, diags, _) = resolve(top, &[("lib.ode", "par b=1")]);
        assert_eq!(codes(&diags), [INCLUDE_QUOTED], "{top:?}");
        let d = &diags[0];
        let span = nth(FileId(0), top, inner, 0);
        assert_eq!(d.span, span);
        let fix = d.fix_it.as_ref().unwrap();
        assert_eq!(fix.span, span);
        assert_eq!(fix.replacement, "lib.ode");
        assert_eq!(resolved.strip_spans().statements, flat("par b=1\nx'=b"));
    }
}

#[test]
fn unbalanced_quotes_are_a_plain_name() {
    for top in [
        "#include \"lib.ode",
        "#include lib.ode'",
        "#include \"lib.ode'",
        "#include \"",
    ] {
        let (_, _, diags, _) = resolve(top, &[("lib.ode", "par b=1")]);
        assert_eq!(codes(&diags), [INCLUDE_UNREADABLE], "{top:?}");
    }
}

#[test]
fn include_inside_an_array_block() {
    // R17 f: the parser keeps the include in the block body; this pass
    // reports it and leaves it unresolved.
    let top = "%[1..2]\n#include lib.ode\nx[j]'=1\n%";
    let (_, parse_diags) = parse(top, FileId(0));
    assert!(parse_diags.is_empty(), "{parse_diags:#?}");
    let (template, resolved, diags, map) = resolve(top, &[("lib.ode", "par b=1")]);
    assert_eq!(codes(&diags), [INCLUDE_IN_BLOCK]);
    assert_eq!(diags[0].span, nth(FileId(0), top, "#include lib.ode", 0));
    assert_eq!(resolved, template);
    assert_eq!(map.file_count(), 1);

    // A malformed block reports it too.
    let top = "%[1..x]\n#include lib.ode\n%";
    let (template, resolved, diags, _) = resolve(top, &[("lib.ode", "par b=1")]);
    assert_eq!(codes(&diags), [INCLUDE_IN_BLOCK]);
    assert_eq!(resolved, template);
}

/// Top file text, other files, expected code, message needle.
type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], Code, &'a str);

#[test]
fn message_text_per_code() {
    let cases: [Case; 5] = [
        (
            "#include \"a.ode\"",
            &[("a.ode", "")],
            INCLUDE_QUOTED,
            "remove the quotes",
        ),
        ("#include a.ode", &[], INCLUDE_UNREADABLE, "cannot read"),
        (
            "#include main.ode",
            &[("main.ode", "#include main.ode")],
            INCLUDE_CYCLE,
            "includes itself",
        ),
        (
            "#include f1.ode",
            &[],
            INCLUDE_TOO_DEEP,
            "more than 16 deep",
        ),
        (
            "%[1..2]\n#include a.ode\n%",
            &[],
            INCLUDE_IN_BLOCK,
            "inside an array block",
        ),
    ];
    let deep = chain(MAX_INCLUDE_DEPTH + 1);
    for (top, files, code, needle) in cases {
        let mut files = files.to_vec();
        if code == INCLUDE_TOO_DEEP {
            files = deep.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        }
        let (_, _, diags, _) = resolve(top, &files);
        assert_eq!(diags.len(), 1, "{top:?}: {diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, code, "{top:?}: {d:#?}");
        assert_eq!(d.severity, Severity::Error);
        assert!(d.message.contains(needle), "{top:?}: {}", d.message);
    }
}

#[test]
fn map_loader_ignores_the_including_file() {
    let mut loader = MapLoader::new();
    loader.insert("a.ode", "par a=1");
    assert_eq!(
        loader.load(FileId(7), "x/y.ode", "a.ode"),
        Ok(("a.ode".to_string(), "par a=1".to_string()))
    );
    assert!(loader.load(FileId(0), "main.ode", "b.ode").is_err());
    assert_eq!(loader.identity("a.ode"), "a.ode");
}

#[test]
fn normalize_paths() {
    let cases = [
        ("a.ode", "a.ode"),
        ("./a.ode", "a.ode"),
        ("sub/./a.ode", "sub/a.ode"),
        ("sub/../a.ode", "a.ode"),
        ("sub/x/../../a.ode", "a.ode"),
        ("../a.ode", "../a.ode"),
        ("../../a.ode", "../../a.ode"),
        ("sub/../../a.ode", "../a.ode"),
        ("/r/sub/../a.ode", "/r/a.ode"),
        ("/../a.ode", "/a.ode"),
    ];
    for (input, want) in cases {
        assert_eq!(
            normalize(Path::new(input)),
            PathBuf::from(want),
            "{input:?}"
        );
    }
}

/// A fresh directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "pc-ode-include-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn write(&self, rel: &str, text: &str) {
        let path = self.0.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn fs_loader_resolves_against_the_including_file() {
    // `sub/a.ode` includes `b.ode`, which exists only as `sub/b.ode`, and
    // `sub/b.ode` includes `../c.ode`. A decoy `b.ode` next to `main.ode`
    // shows the path is not taken from the top file's directory.
    let tmp = TempDir::new("rel");
    tmp.write("main.ode", "#include sub/a.ode\nx'=a+b+c");
    tmp.write("sub/a.ode", "par a=1\n#include b.ode");
    tmp.write("sub/b.ode", "par b=2\n#include ../c.ode");
    tmp.write("b.ode", "par decoy=9");
    tmp.write("c.ode", "par c=3");

    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let top = tmp.0.join("main.ode");
    let id = loader.load_top(&mut map, &top).unwrap();
    assert_eq!(Path::new(map.name(id)), top.as_path());
    let (model, parse_diags) = parse(map.text(id), id);
    assert!(parse_diags.is_empty());
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par a=1\npar b=2\npar c=3\nx'=a+b+c")
    );
    assert_eq!(Path::new(map.name(FileId(1))), tmp.0.join("sub/a.ode"));
    assert_eq!(Path::new(map.name(FileId(2))), tmp.0.join("sub/b.ode"));
    // R20 1: `..` is kept in the display name, so it names the opened file.
    assert_eq!(
        map.name(FileId(3)),
        tmp.0
            .join("sub")
            .join("..")
            .join("c.ode")
            .display()
            .to_string()
    );
}

#[test]
fn fs_loader_cycle_through_dot_dot() {
    let tmp = TempDir::new("cyc");
    tmp.write("main.ode", "#include sub/a.ode\nx'=1");
    tmp.write("sub/a.ode", "#include ../main.ode");
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let id = loader.load_top(&mut map, tmp.0.join("main.ode")).unwrap();
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert_eq!(codes(&diags), [INCLUDE_CYCLE]);
    assert_eq!(map.name(diags[0].span.file), map.name(FileId(1)));
    assert_eq!(resolved.strip_spans().statements, flat("x'=1"));
}

#[test]
fn fs_loader_absolute_and_missing_paths() {
    let tmp = TempDir::new("abs");
    tmp.write("lib/l.ode", "par l=1");
    let abs = tmp.0.join("lib/l.ode");
    let top = format!("#include {}\n#include nope.ode\nx'=l", abs.display());
    tmp.write("main.ode", &top);
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let id = loader.load_top(&mut map, tmp.0.join("main.ode")).unwrap();
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert_eq!(codes(&diags), [INCLUDE_UNREADABLE]);
    assert!(
        diags[0].message.contains("`nope.ode`"),
        "{}",
        diags[0].message
    );
    assert_eq!(resolved.strip_spans().statements, flat("par l=1\nx'=l"));
    assert_eq!(Path::new(map.name(FileId(1))), abs.as_path());
}

#[test]
fn fs_loader_errors() {
    let tmp = TempDir::new("err");
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    assert!(loader.load_top(&mut map, tmp.0.join("none.ode")).is_err());
    assert_eq!(map.file_count(), 0);
    // The error names the full path that was tried.
    let err = FsLoader::new()
        .load(
            FileId(0),
            &tmp.0.join("m.ode").display().to_string(),
            "a.ode",
        )
        .unwrap_err();
    assert!(
        err.starts_with(&tmp.0.join("a.ode").display().to_string()),
        "{err}"
    );
}

#[test]
fn depth_summary_matches_the_limit() {
    let summary = CODES
        .iter()
        .find(|(c, _, _)| *c == INCLUDE_TOO_DEEP)
        .unwrap()
        .2;
    assert!(
        summary.contains(&MAX_INCLUDE_DEPTH.to_string()),
        "{summary}"
    );
}

#[test]
fn temp_dir_is_removed() {
    let tmp = TempDir::new("drop");
    tmp.write("sub/a.ode", "par a=1");
    let dir = tmp.0.clone();
    assert!(dir.join("sub/a.ode").is_file());
    drop(tmp);
    assert!(!dir.exists());
}

#[test]
fn fs_loader_top_added_by_hand() {
    // R19 A3: the loader is stateless, so a top file the caller holds in
    // memory resolves whenever its name is a path.
    let tmp = TempDir::new("hand");
    tmp.write("d/b.ode", "par b=1");
    let mut map = SourceMap::new();
    let name = tmp.0.join("d/hand.ode").display().to_string();
    let id = map.add(name, "#include b.ode\nx'=b");
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut FsLoader::new());
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(resolved.strip_spans().statements, flat("par b=1\nx'=b"));
}

#[cfg(unix)]
#[test]
fn fs_loader_symlink_self_include_is_a_cycle() {
    let tmp = TempDir::new("sym");
    tmp.write("d/s.ode", "#include loop/s.ode\npar s=1");
    std::os::unix::fs::symlink(".", tmp.0.join("d/loop")).unwrap();
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let id = loader.load_top(&mut map, tmp.0.join("d/s.ode")).unwrap();
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut FsLoader::new());
    assert_eq!(codes(&diags), [INCLUDE_CYCLE], "{diags:#?}");
    assert_eq!(diags[0].span.file, id);
    assert_eq!(map.file_count(), 1);
    assert_eq!(resolved.strip_spans().statements, flat("par s=1"));
}

/// `abs` as a path relative to the working directory, going up to the root
/// first, so the result does not depend on where the tests run.
#[cfg(unix)]
fn relative_to_cwd(abs: &Path) -> PathBuf {
    let cwd = std::env::current_dir().unwrap();
    let mut rel = PathBuf::new();
    for part in cwd.components() {
        if matches!(part, std::path::Component::Normal(_)) {
            rel.push("..");
        }
    }
    for part in abs.components() {
        if let std::path::Component::Normal(p) = part {
            rel.push(p);
        }
    }
    rel
}

#[cfg(unix)]
#[test]
fn fs_loader_relative_top_including_itself_by_absolute_path() {
    let tmp = TempDir::new("self");
    let abs = tmp.0.join("d/abs.ode");
    let text = format!("#include {}\npar a=1", abs.display());
    tmp.write("d/abs.ode", &text);
    let rel = relative_to_cwd(&abs);
    assert!(rel.is_relative());
    let mut map = SourceMap::new();
    let id = map.add(rel.display().to_string(), text.as_str());
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut FsLoader::new());
    assert_eq!(codes(&diags), [INCLUDE_CYCLE], "{diags:#?}");
    assert_eq!(diags[0].span.file, id);
    assert_eq!(map.file_count(), 1);
    assert_eq!(resolved.strip_spans().statements, flat("par a=1"));
}

#[test]
fn include_count_limit() {
    // Exactly MAX_INCLUDES includes resolve.
    let top = "#include a.ode\n".repeat(MAX_INCLUDES);
    let (_, resolved, diags, map) = resolve(&top, &[("a.ode", "par a=1")]);
    assert!(diags.is_empty());
    assert_eq!(resolved.statements.len(), MAX_INCLUDES);
    assert_eq!(map.file_count(), 2);

    // One more gives one error, at the first include past the limit.
    let top = "#include a.ode\n".repeat(MAX_INCLUDES + 3);
    let (_, resolved, diags, _) = resolve(&top, &[("a.ode", "par a=1")]);
    assert_eq!(codes(&diags), [INCLUDE_TOO_MANY]);
    let line = "#include a.ode\n".len() * MAX_INCLUDES;
    assert_eq!(diags[0].span, Span::new(FileId(0), line, line + 14));
    assert!(diags[0].message.contains("4096"), "{}", diags[0].message);
    assert_eq!(resolved.statements.len(), MAX_INCLUDES);
}

#[test]
fn include_count_limit_in_a_lattice() {
    // Each level includes the next one twice: 2 + 4 + ... + 2^13 includes.
    let mut files: Vec<(String, String)> = (0..13)
        .map(|i| {
            (
                format!("l{i}.ode"),
                format!("#include l{0}.ode\n#include l{0}.ode", i + 1),
            )
        })
        .collect();
    files.push(("l13.ode".to_string(), "par q=(".to_string()));
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let (_, _, diags, map) = resolve("#include l0.ode\n#include l0.ode", &refs);
    let mut want = vec![crate::parser::stmt::BAD_VALUE, INCLUDE_TOO_MANY];
    want.sort();
    let mut got = codes(&diags);
    got.sort();
    assert_eq!(got, want, "{diags:#?}");
    // Each file is read and parsed once.
    assert_eq!(map.file_count(), 15);
}

#[test]
fn empty_quoted_name() {
    for top in ["#include \"\"", "#include ''"] {
        let (_, resolved, diags, map) = resolve(top, &[("", "par a=1")]);
        assert_eq!(
            codes(&diags),
            [crate::parser::stmt::MALFORMED_STATEMENT],
            "{top:?}"
        );
        assert!(diags[0].fix_it.is_none());
        assert!(
            diags[0].message.contains("names no file"),
            "{}",
            diags[0].message
        );
        assert_eq!(map.file_count(), 1);
        assert!(resolved.statements.is_empty());
    }
}

#[test]
fn control_characters_are_escaped_in_messages() {
    let (_, _, diags, _) = resolve("#include a.ode\u{c}", &[]);
    assert_eq!(codes(&diags), [INCLUDE_UNREADABLE]);
    assert!(
        diags[0].message.contains("a.ode\\u{c}"),
        "{}",
        diags[0].message
    );
    assert_eq!(show("a\\b \"c\"\t\r"), "a\\b \"c\"\\t\\r");
}

#[test]
fn nested_file_diagnostics_get_notes() {
    let top = "#include a.ode";
    let via = nth(FileId(0), top, "#include a.ode", 0);

    // A quoted name in a nested file.
    let a = "#include \"b.ode\"";
    let (_, resolved, diags, _) = resolve(top, &[("a.ode", a), ("b.ode", "par b=1")]);
    assert_eq!(codes(&diags), [INCLUDE_QUOTED]);
    assert_eq!(diags[0].span, nth(FileId(1), a, "\"b.ode\"", 0));
    assert_eq!(diags[0].notes, [(via, "included here".to_string())]);
    assert_eq!(resolved.strip_spans().statements, flat("par b=1"));

    // A trailing comment in a nested file: the parser reports it in the
    // nested file and the include still resolves.
    let a = "#include b.ode # c";
    let (_, resolved, diags, _) = resolve(top, &[("a.ode", a), ("b.ode", "par b=1")]);
    assert_eq!(codes(&diags), [TRAILING_COMMENT]);
    assert_eq!(diags[0].span.file, FileId(1));
    assert_eq!(resolved.strip_spans().statements, flat("par b=1"));

    // An include inside a block in a nested file.
    let a = "%[1..2]\n#include b.ode\n%";
    let (_, _, diags, map) = resolve(top, &[("a.ode", a), ("b.ode", "par b=1")]);
    assert_eq!(codes(&diags), [INCLUDE_IN_BLOCK]);
    assert_eq!(diags[0].span, nth(FileId(1), a, "#include b.ode", 0));
    assert_eq!(diags[0].notes, [(via, "included here".to_string())]);
    assert_eq!(map.file_count(), 2);
}

#[test]
fn self_include_at_the_depth_limit_is_too_deep() {
    // R19 C1: the depth check runs before the file is read, so a cycle that
    // closes exactly at the limit reports ODE0062.
    let mut files = chain(MAX_INCLUDE_DEPTH);
    files.last_mut().unwrap().1 = format!("#include f{MAX_INCLUDE_DEPTH}.ode");
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let (_, _, diags, _) = resolve("#include f1.ode", &refs);
    assert_eq!(codes(&diags), [INCLUDE_TOO_DEEP]);
}

#[test]
fn drive_relative_paths() {
    // Only Windows paths have a prefix; on other systems this is a plain
    // relative name.
    assert!(drive_relative(Path::new("a.ode")).is_none());
    assert!(drive_relative(Path::new("/a.ode")).is_none());
}

#[test]
fn display_names_drop_only_dot() {
    let cases = [
        ("a.ode", "a.ode"),
        ("./a.ode", "a.ode"),
        ("sub/./a.ode", "sub/a.ode"),
        ("sub/../a.ode", "sub/../a.ode"),
        ("../a.ode", "../a.ode"),
        ("/r/sub/../a.ode", "/r/sub/../a.ode"),
    ];
    for (input, want) in cases {
        assert_eq!(
            display_name(Path::new(input)),
            PathBuf::from(want),
            "{input:?}"
        );
    }
}

/// Resolves the top file `top` with `FsLoader`.
#[cfg(unix)]
fn resolve_fs(top: &Path) -> (Model, Vec<Diagnostic>, SourceMap) {
    let mut map = SourceMap::new();
    let id = FsLoader::new().load_top(&mut map, top).unwrap();
    let (model, _) = parse(map.text(id), id);
    let (resolved, diags) = resolve_includes(&model, id, &mut map, &mut FsLoader::new());
    (resolved, diags, map)
}

/// The tree of re-review probe fs/p1: `d/sub/lnk` links to `other/deep`, so
/// `sub/lnk/../t.ode` opens `other/t.ode`, while folding the `..` as text
/// would give `d/sub/t.ode`.
#[cfg(unix)]
fn symlink_dot_dot_tree(tag: &str, with_decoy: bool) -> TempDir {
    let tmp = TempDir::new(tag);
    tmp.write("other/n.ode", "par n=1");
    tmp.write("other/t.ode", "#include n.ode\npar t=1");
    std::fs::create_dir_all(tmp.0.join("other/deep")).unwrap();
    tmp.write("d/sub/n.ode", "par n=99");
    if with_decoy {
        tmp.write("d/sub/t.ode", "par wrong=1");
        tmp.write(
            "d/main.ode",
            "#include sub/lnk/../t.ode\n#include sub/t.ode",
        );
    } else {
        tmp.write("d/main.ode", "#include sub/lnk/../t.ode");
    }
    std::os::unix::fs::symlink("../../other/deep", tmp.0.join("d/sub/lnk")).unwrap();
    tmp
}

#[cfg(unix)]
#[test]
fn fs_loader_symlink_then_dot_dot_opens_the_linked_file() {
    // Probe fs/p1: no file is put in place of another.
    let tmp = symlink_dot_dot_tree("p1", true);
    let (resolved, diags, map) = resolve_fs(&tmp.0.join("d/main.ode"));
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par n=1\npar t=1\npar wrong=1")
    );
    assert_eq!(
        map.name(FileId(1)),
        tmp.0.join("d/sub/lnk/../t.ode").display().to_string()
    );

    // Probe fs/p1b: without the decoy, folding would make the file missing.
    let tmp = symlink_dot_dot_tree("p1b", false);
    let (resolved, diags, _) = resolve_fs(&tmp.0.join("d/main.ode"));
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(resolved.strip_spans().statements, flat("par n=1\npar t=1"));
}

#[cfg(unix)]
#[test]
fn fs_loader_one_file_under_two_names_resolves_per_site() {
    // Probe fs/p2: `link.ode` links to `sub/real.ode`, which includes
    // `x.ode`. Each site resolves `x.ode` next to its own name, in either
    // order, although both share one parse.
    let tmp = TempDir::new("p2");
    tmp.write("d/x.ode", "par x=1");
    tmp.write("d/sub/x.ode", "par x=2");
    tmp.write("d/sub/real.ode", "#include x.ode\npar r=1");
    tmp.write("d/main1.ode", "#include link.ode\n#include sub/real.ode");
    tmp.write("d/main2.ode", "#include sub/real.ode\n#include link.ode");
    std::os::unix::fs::symlink("sub/real.ode", tmp.0.join("d/link.ode")).unwrap();

    let (resolved, diags, _) = resolve_fs(&tmp.0.join("d/main1.ode"));
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par x=1\npar r=1\npar x=2\npar r=1")
    );

    let (resolved, diags, _) = resolve_fs(&tmp.0.join("d/main2.ode"));
    assert!(diags.is_empty(), "{diags:#?}");
    assert_eq!(
        resolved.strip_spans().statements,
        flat("par x=2\npar r=1\npar x=1\npar r=1")
    );
}

#[test]
fn include_count_includes_unresolved_ones() {
    // R20 3: every include reached counts, resolved or not.
    let mut top = "#include missing.ode\n".repeat(MAX_INCLUDES);
    top.push_str("#include a.ode\n#include missing.ode\n");
    let (_, resolved, diags, _) = resolve(&top, &[("a.ode", "par a=1")]);
    let mut want = vec![INCLUDE_UNREADABLE; MAX_INCLUDES];
    want.push(INCLUDE_TOO_MANY);
    assert_eq!(codes(&diags), want);
    assert!(resolved.statements.is_empty());

    // The last include within the limit may be quoted.
    let mut top = "#include a.ode\n".repeat(MAX_INCLUDES - 1);
    top.push_str("#include \"a.ode\"\n");
    let (_, resolved, diags, _) = resolve(&top, &[("a.ode", "par a=1")]);
    assert_eq!(codes(&diags), [INCLUDE_QUOTED]);
    assert_eq!(resolved.statements.len(), MAX_INCLUDES);
}

#[test]
fn nothing_is_reported_past_the_include_limit() {
    // Probe `cap`: no quoted-name error, empty-name error or array-block
    // error after ODE0064.
    let mut top = "#include a.ode\n".repeat(MAX_INCLUDES + 1);
    top.push_str("#include \"a.ode\"\n#include \"a.ode\"\n#include \"\"\n");
    top.push_str("%[1..2]\n#include a.ode\nx[j]'=1\n%\n");
    let (_, _, diags, _) = resolve(&top, &[("a.ode", "par a=1")]);
    assert_eq!(codes(&diags), [INCLUDE_TOO_MANY]);
}

/// Counts calls and answers every path with an error.
struct Missing(usize);

impl SourceLoader for Missing {
    fn load(&mut self, _: FileId, _: &str, _: &str) -> Result<(String, String), String> {
        self.0 += 1;
        Err("there is no such file".to_string())
    }
}

#[test]
fn missing_files_in_a_lattice_are_bounded() {
    // Probe `blow`, scaled down: each level includes the next one twice and
    // the leaf names a missing file 100 times, which without the limit would
    // give 2^12 * 100 errors.
    let mut loader = MapLoader::new();
    for i in 0..11 {
        loader.insert(
            format!("l{i}.ode"),
            format!("#include l{0}.ode\n#include l{0}.ode", i + 1),
        );
    }
    loader.insert("l11.ode", "#include missing.ode\n".repeat(100));
    let top = "#include l0.ode\n#include l0.ode";
    let mut map = SourceMap::new();
    let id = map.add("main.ode", top);
    let (model, _) = parse(top, id);
    let start = std::time::Instant::now();
    let (_, diags) = resolve_includes(&model, id, &mut map, &mut loader);
    assert!(start.elapsed() < std::time::Duration::from_secs(10));
    assert!(diags.len() <= MAX_INCLUDES, "{}", diags.len());
    let too_many = diags.iter().filter(|d| d.code == INCLUDE_TOO_MANY).count();
    assert_eq!(too_many, 1);
    assert!(
        diags
            .iter()
            .all(|d| d.code == INCLUDE_TOO_MANY || d.code == INCLUDE_UNREADABLE)
    );

    // A missing path is asked for once per including file name.
    let mut missing = Missing(0);
    let top = "#include m.ode\n".repeat(10);
    let mut map = SourceMap::new();
    let id = map.add("main.ode", top.as_str());
    let (model, _) = parse(&top, id);
    let (_, diags) = resolve_includes(&model, id, &mut map, &mut missing);
    assert_eq!(codes(&diags), vec![INCLUDE_UNREADABLE; 10]);
    assert_eq!(missing.0, 1);
}

#[test]
fn blank_quoted_name() {
    // R20 4: a quoted name that is blank after trimming names no file.
    for top in ["#include \" \"", "#include '  '"] {
        let (_, resolved, diags, map) = resolve(top, &[(" ", "par a=1"), ("  ", "par a=1")]);
        assert_eq!(
            codes(&diags),
            [crate::parser::stmt::MALFORMED_STATEMENT],
            "{top:?}"
        );
        assert!(diags[0].fix_it.is_none());
        assert_eq!(map.file_count(), 1);
        assert!(resolved.statements.is_empty());
    }
}
