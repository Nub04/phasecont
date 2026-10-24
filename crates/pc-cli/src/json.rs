//! JSON rendering of a loaded model. Keys come out in sorted order
//! (`serde_json` maps are ordered), so the output is stable.

use pc_ode::{Loaded, SourceMap, Span};
use serde_json::{Value, json};

use crate::summary::Summary;
use crate::text::severity_name;

/// `line`, `col`, `end_line`, `end_col` of `span`, 1-based, end exclusive.
fn range(map: &SourceMap, span: Span) -> (usize, usize, usize, usize) {
    let (line, col) = map.line_col(span);
    let (end_line, end_col) = map.line_col(Span::new(span.file, span.end, span.end));
    (line, col, end_line, end_col)
}

pub fn value(file: &str, map: &SourceMap, loaded: &Loaded) -> Value {
    let diagnostics: Vec<Value> = loaded
        .diagnostics
        .iter()
        .map(|d| {
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
        })
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
    use pc_ode::{MapLoader, load};

    #[test]
    fn shape_of_an_error_report() {
        let mut map = SourceMap::new();
        let top = map.add("a.ode", "x'=-x\nfoo(\ndone\n");
        let loaded = load(top, &mut map, &mut MapLoader::new());
        let v = value("a.ode", &map, &loaded);
        assert_eq!(v["file"], "a.ode");
        let d = &v["diagnostics"][0];
        assert_eq!(d["severity"], "error");
        assert_eq!(d["line"], 2);
        assert!(d["code"].as_str().unwrap().starts_with("ODE"));
        assert!(d["fix_it"].is_null() || d["fix_it"]["replacement"].is_string());
        assert!(d["notes"].is_array());
        assert_eq!(v["summary"]["errors"], 1);
        assert_eq!(v["summary"]["kind"], "ode");
    }
}
