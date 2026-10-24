//! Text rendering of a loaded model.

use std::fmt::Write as _;

use pc_ode::{Diagnostic, Loaded, Severity, SourceMap, Span};

use crate::summary::Summary;

pub fn severity_name(s: Severity) -> &'static str {
    match s {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Note => "note",
    }
}

/// `file:line:col` of the start of `span`.
fn at(map: &SourceMap, span: Span) -> String {
    let (line, col) = map.line_col(span);
    format!("{}:{line}:{col}", map.name(span.file))
}

/// The source line of `span` and a caret line under it.
fn snippet(map: &SourceMap, span: Span) -> Option<String> {
    let (line, col) = map.line_col(span);
    let text = map.line_text(span.file, line)?;
    let text = text.strip_suffix('\r').unwrap_or(text);
    let (end_line, end_col) = map.line_col(Span::new(span.file, span.end, span.end));
    let width = text.chars().count();
    let end_col = if end_line == line {
        end_col.min(width + 1)
    } else {
        width + 1
    };
    // One caret at the start, then `~` up to the exclusive end.
    let tildes = end_col.saturating_sub(col + 1);
    let mut caret = String::new();
    for c in text.chars().take(col - 1) {
        caret.push(if c == '\t' { '\t' } else { ' ' });
    }
    // A span past the end of the line (a missing token at the line end)
    // still gets its caret, in the column after the last character.
    for _ in text.chars().count()..col - 1 {
        caret.push(' ');
    }
    caret.push('^');
    caret.extend(std::iter::repeat_n('~', tildes));
    Some(format!("{text}\n{caret}\n"))
}

fn diagnostic(out: &mut String, map: &SourceMap, d: &Diagnostic) {
    let _ = writeln!(
        out,
        "{}: {}[{}]: {}",
        at(map, d.span),
        severity_name(d.severity),
        d.code,
        d.message
    );
    if let Some(s) = snippet(map, d.span) {
        out.push_str(&s);
    }
    if let Some(fix) = &d.fix_it {
        let (l1, c1) = map.line_col(fix.span);
        let (l2, c2) = map.line_col(Span::new(fix.span.file, fix.span.end, fix.span.end));
        let _ = writeln!(
            out,
            "  fix-it: replace {l1}:{c1}-{l2}:{c2} with {:?}",
            fix.replacement
        );
    }
    for (span, message) in &d.notes {
        let _ = writeln!(out, "  note: {}: {message}", at(map, *span));
    }
}

/// The whole text report: diagnostics, a blank line, the summary.
pub fn render(map: &SourceMap, loaded: &Loaded) -> String {
    let mut out = String::new();
    for d in &loaded.diagnostics {
        diagnostic(&mut out, map, d);
    }
    if !loaded.diagnostics.is_empty() {
        out.push('\n');
    }
    let s = Summary::of(loaded);
    let _ = writeln!(out, "kind: {}", s.kind);
    let _ = writeln!(out, "states: {}", s.states);
    let _ = writeln!(out, "parameters: {}", s.parameters);
    let _ = writeln!(out, "fixed: {}", s.fixed);
    let _ = writeln!(out, "functions: {}", s.functions);
    let _ = writeln!(out, "aux: {}", s.aux);
    let _ = writeln!(out, "smooth: {}", s.smooth);
    let _ = writeln!(out, "simulation_only: {}", s.simulation_only);
    let _ = writeln!(out, "{} errors, {} warnings", s.errors, s.warnings);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ode::{MapLoader, load};

    fn run(name: &str, text: &str) -> String {
        let mut map = SourceMap::new();
        let top = map.add(name, text);
        let loaded = load(top, &mut map, &mut MapLoader::new());
        render(&map, &loaded)
    }

    #[test]
    fn clean_model_has_only_a_summary() {
        let out = run("a.ode", "x'=-x\ndone\n");
        assert!(out.starts_with("kind: ode\nstates: 1\n"), "{out}");
        assert!(out.ends_with("0 errors, 0 warnings\n"), "{out}");
    }

    #[test]
    fn error_has_location_source_and_caret() {
        let out = run("a.ode", "x'=-x\nfoo(\ndone\n");
        let first = out.lines().next().unwrap();
        assert!(first.starts_with("a.ode:2:"), "{out}");
        assert!(first.contains(": error[ODE"), "{out}");
        assert!(out.contains("\nfoo(\n"), "{out}");
        assert!(out.contains('^'), "{out}");
        assert!(out.contains("\n\nkind: "), "{out}");
    }

    #[test]
    fn caret_copies_tabs_and_ignores_cr() {
        let out = run("a.ode", "x'=-x\n\tx'=@\r\ndone\n");
        assert!(!out.contains('\r'), "{out:?}");
        let caret = out.lines().find(|l| l.contains('^')).unwrap();
        assert!(caret.starts_with('\t'), "{caret:?}");
    }
}
