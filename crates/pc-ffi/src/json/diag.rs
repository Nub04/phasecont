//! JSON rendering of a loaded model. Keys come out in sorted order
//! (`serde_json` maps are ordered), so the output is stable.

use pc_ode::{Diagnostic, Loaded, Severity, SourceMap, Span};
use serde_json::{Value, json};

use super::summary::Summary;

/// The name of a severity in the JSON and text reports.
pub fn severity_name(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    }
}

/// `line`, `col`, `end_line`, `end_col` of `span`, 1-based, end exclusive.
fn range(map: &SourceMap, span: Span) -> (usize, usize, usize, usize) {
    let (line, col) = map.line_col(span);
    let (end_line, end_col) = map.line_col(Span::new(span.file, span.end, span.end));
    (line, col, end_line, end_col)
}

fn diagnostic(map: &SourceMap, d: &Diagnostic) -> Value {
    let (line, col, end_line, end_col) = range(map, d.span);
    let fix_it = d.fix_it.as_ref().map(|f| {
        let (line, col, end_line, end_col) = range(map, f.span);
        json!({
            "line": line, "col": col, "end_line": end_line, "end_col": end_col,
            "replacement": f.replacement,
        })
    });
    let notes: Vec<Value> = d
        .notes
        .iter()
        .map(|(span, message)| {
            let (line, col, end_line, end_col) = range(map, *span);
            json!({
                "file": map.name(span.file),
                "line": line, "col": col, "end_line": end_line, "end_col": end_col,
                "message": message,
            })
        })
        .collect();
    json!({
        "code": d.code.to_string(),
        "severity": severity_name(d.severity),
        "file": map.name(d.span.file),
        "line": line, "col": col, "end_line": end_line, "end_col": end_col,
        "message": d.message,
        "fix_it": fix_it,
        "notes": notes,
    })
}

pub fn value(file: &str, map: &SourceMap, loaded: &Loaded) -> Value {
    let diagnostics: Vec<Value> = loaded
        .diagnostics
        .iter()
        .map(|d| diagnostic(map, d))
        .collect();
    let s = Summary::of(loaded);
    json!({
        "file": file,
        "diagnostics": diagnostics,
        "summary": {
            "kind": s.kind, "states": s.states, "parameters": s.parameters,
            "fixed": s.fixed, "functions": s.functions, "aux": s.aux,
            "smooth": s.smooth, "simulation_only": s.simulation_only,
            "errors": s.errors, "warnings": s.warnings,
        },
    })
}

/// Pretty-printed JSON with a final newline.
pub fn render(file: &str, map: &SourceMap, loaded: &Loaded) -> String {
    let mut s = serde_json::to_string_pretty(&value(file, map, loaded))
        .expect("a JSON value always serializes");
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ode::{Code, FixIt, MapLoader, load};

    #[test]
    fn shape_of_an_error_report() {
        let mut map = SourceMap::new();
        let top = map.add("a.ode", "x'=-x\nfoo(\ndone\n");
        let loaded = load(top, &mut map, &mut MapLoader::new());
        let v = value("a.ode", &map, &loaded);
        assert_eq!(v["file"], "a.ode");
        assert_eq!(v["diagnostics"][0]["severity"], "error");
        assert_eq!(v["diagnostics"][0]["line"], 2);
        assert_eq!(v["summary"]["errors"], 1);
        assert_eq!(v["summary"]["kind"], "ode");
    }

    #[test]
    fn fix_it_and_notes_have_every_key_with_exact_values() {
        let mut map = SourceMap::new();
        let f = map.add("a.ode", "\tx'=\u{e9}@\ny=1\n");
        let g = map.add("inc/b.ode", "zz\n");
        let mut d = Diagnostic::new(Code(12), Span::new(f, 3, 7), "msg".to_string());
        d.fix_it = Some(FixIt {
            span: Span::new(f, 6, 7),
            replacement: "a\"b".to_string(),
        });
        d.notes.push((Span::new(f, 8, 9), "here".to_string()));
        d.notes.push((Span::new(g, 0, 2), "there".to_string()));
        let want = json!({
            "code": "ODE0012",
            "severity": "error",
            "file": "a.ode",
            "line": 1, "col": 4, "end_line": 1, "end_col": 7,
            "message": "msg",
            "fix_it": {"line": 1, "col": 6, "end_line": 1, "end_col": 7, "replacement": "a\"b"},
            "notes": [
                {"file": "a.ode", "line": 2, "col": 1, "end_line": 2, "end_col": 2, "message": "here"},
                {"file": "inc/b.ode", "line": 1, "col": 1, "end_line": 1, "end_col": 3, "message": "there"},
            ],
        });
        assert_eq!(diagnostic(&map, &d), want);
    }
}
