//! `pc-ode` diagnostics against the LSP diagnostics the server publishes.
//!
//! Every file in `pc-ode/tests/invalid/` is loaded through the server's
//! loader. Each diagnostic must come out with the same code, message and
//! severity, and with a range that agrees with a position computed another
//! way: `SourceMap::line_col` for the line and a count of UTF-16 units over
//! the line text for the column.

mod common;

use std::path::{Path, PathBuf};

use lsp_types::{DiagnosticSeverity, NumberOrString, Uri};
use pc_lsp::analysis::{Analysis, analyze};
use pc_lsp::diagnostics::to_lsp;
use pc_lsp::overlay::{Overlay, normalize};
use pc_lsp::uri::uri_from_path;
use pc_ode::{CODES, Severity, Span};

fn invalid_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/invalid")
}

fn name_to_uri(name: &str) -> Option<Uri> {
    uri_from_path(name)
}

fn analysis_of(path: &Path, text: String) -> Analysis {
    let dir = std::fs::canonicalize(path.parent().unwrap()).unwrap();
    let dir = normalize(&dir.display().to_string());
    let name = format!("{dir}/{}", path.file_name().unwrap().to_str().unwrap());
    let mut overlay = Overlay::new();
    overlay.set_roots([dir.clone()]);
    let mut loader = overlay.loader(Some(&dir));
    analyze(&name, text, &mut loader)
}

/// The 0-based line and UTF-16 column of `offset`, from the source map.
fn expected(a: &Analysis, file: pc_ode::FileId, offset: usize) -> (u32, u32) {
    let (line, col) = a.map.line_col(Span::new(file, offset, offset));
    let text = a.map.line_text(file, line).unwrap_or("");
    let units: usize = text.chars().take(col - 1).map(char::len_utf16).sum();
    ((line - 1) as u32, units as u32)
}

fn severity(s: Severity) -> DiagnosticSeverity {
    match s {
        Severity::Error => DiagnosticSeverity::ERROR,
        Severity::Warning => DiagnosticSeverity::WARNING,
        Severity::Note => DiagnosticSeverity::INFORMATION,
    }
}

fn check_file(path: &Path, text: String) -> usize {
    let a = analysis_of(path, text);
    let lsp = to_lsp(&a, &mut |n| name_to_uri(n));
    assert_eq!(
        lsp.len(),
        a.loaded.diagnostics.len(),
        "{}: one LSP diagnostic per pc-ode diagnostic",
        path.display()
    );
    let count = lsp.len();
    for (i, (d, (uri, l))) in a.loaded.diagnostics.iter().zip(&lsp).enumerate() {
        let what = format!("{} {}", path.display(), d.code);
        assert_eq!(
            l.code,
            Some(NumberOrString::String(d.code.to_string())),
            "{what}"
        );
        assert_eq!(l.severity, Some(severity(d.severity)), "{what}");
        assert_eq!(l.message, d.message, "{what}");
        assert_eq!(l.source.as_deref(), Some("phasecont"), "{what}");
        let file_name = a.map.name(d.span.file);
        assert_eq!(
            uri,
            &name_to_uri(file_name).unwrap(),
            "{what}: the file of the diagnostic"
        );
        // The oracle below walks the line text for every diagnostic, which is
        // quadratic on a line with thousands of them (ODE0073). The first
        // 200, every 97th after that and the last one are enough.
        if i < 200 || i % 97 == 0 || i + 1 == count {
            let (sl, sc) = expected(&a, d.span.file, d.span.start);
            let (el, ec) = expected(&a, d.span.file, d.span.end);
            assert_eq!(
                (l.range.start.line, l.range.start.character),
                (sl, sc),
                "{what}: start"
            );
            assert_eq!(
                (l.range.end.line, l.range.end.character),
                (el, ec),
                "{what}: end"
            );
        }
        assert_eq!(
            l.related_information.as_ref().map_or(0, Vec::len),
            d.notes.len(),
            "{what}: notes become related information"
        );
        assert_eq!(
            l.data.is_some(),
            d.fix_it
                .as_ref()
                .is_some_and(|f| f.span.file == d.span.file),
            "{what}: fix-it data"
        );
    }
    lsp.len()
}

#[test]
fn every_invalid_file_maps_code_message_severity_and_range() {
    let mut files: Vec<PathBuf> = std::fs::read_dir(invalid_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .collect();
    files.sort();
    assert!(files.len() > 50, "found only {} files", files.len());
    let mut total = 0;
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        total += check_file(path, text);
    }
    assert!(total > 50, "only {total} diagnostics were checked");
}

#[test]
fn codes_print_as_ode_and_four_digits() {
    for (code, _, _) in CODES {
        assert_eq!(code.to_string(), format!("ODE{:04}", code.0));
    }
}

#[test]
fn columns_count_utf16_units_after_astral_characters() {
    // Two emoji, each one scalar and two UTF-16 units, before the error.
    let text = "# \u{1f600}\u{1f600}\nx'=\"\u{1f600}\u{1f600}\" 1.2.3\n".to_string();
    let path = invalid_dir().join("synthetic.ode");
    let a = analysis_of(&path, text.clone());
    let first_on_line_2 = a
        .loaded
        .diagnostics
        .iter()
        .filter(|d| a.map.line_col(d.span).0 == 2)
        .max_by_key(|d| d.span.start)
        .expect("a diagnostic on line 2");
    // The cross-check below fails if the conversion counted scalars or bytes.
    check_file(&path, text);
    let (_, units) = expected(&a, first_on_line_2.span.file, first_on_line_2.span.start);
    let (_, scalars) = a.map.line_col(first_on_line_2.span);
    assert!(
        units as usize > scalars - 1,
        "the test text must put a diagnostic after an astral character"
    );
}

#[test]
fn a_diagnostic_in_an_included_file_goes_to_that_file() {
    let tmp = common::TempDir::new("included");
    let main = tmp.write("main.ode", "#include bad.ode\nx'=-x\n");
    tmp.write("bad.ode", "par a=1.2.3\n");
    let a = analysis_of(Path::new(&main), "#include bad.ode\nx'=-x\n".to_string());
    let lsp = to_lsp(&a, &mut |n| name_to_uri(n));
    let bad_uri = name_to_uri(&tmp.name("bad.ode")).unwrap();
    assert!(
        lsp.iter().any(|(uri, _)| *uri == bad_uri),
        "no diagnostic for bad.ode in {lsp:?}"
    );
}

#[test]
fn a_fix_it_in_the_same_file_travels_with_the_diagnostic() {
    let found = std::fs::read_dir(invalid_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .find_map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            let a = analysis_of(&p, text);
            a.loaded
                .diagnostics
                .iter()
                .any(|d| d.fix_it.is_some())
                .then_some(p)
        })
        .expect("some invalid file offers a fix-it");
    let text = std::fs::read_to_string(&found).unwrap();
    let a = analysis_of(&found, text);
    let lsp = to_lsp(&a, &mut |n| name_to_uri(n));
    let with_fix: Vec<_> = lsp.iter().filter(|(_, d)| d.data.is_some()).collect();
    assert!(!with_fix.is_empty());
    let data = with_fix[0].1.data.as_ref().unwrap();
    assert!(data["fix"]["range"]["start"]["line"].is_number());
    assert!(data["fix"]["newText"].is_string());
}
