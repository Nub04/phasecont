//! Text rendering of a loaded model.
//!
//! Columns count characters, not display cells. The summary follows a blank
//! line only when there are diagnostics, so a clean model starts with
//! `kind:`.

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

/// The most characters of a source line that the text report echoes.
const WINDOW: usize = 120;
/// How many characters before the span the window starts, when the line is
/// cut on the left.
const LEFT_CONTEXT: usize = 40;

/// `s` with every control character other than tab replaced by U+FFFD, one
/// for one, so the terminal sees no escape codes and no line breaks. JSON
/// output keeps the true strings, since it escapes them.
fn plain(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c != '\t' && c.is_control() {
                '\u{FFFD}'
            } else {
                c
            }
        })
        .collect()
}

/// `file:line:col` of the start of `span`.
fn at(map: &SourceMap, span: Span) -> String {
    let (line, col) = map.line_col(span);
    plain(&format!("{}:{line}:{col}", map.name(span.file)))
}

/// The source line of `span` and a caret line under it. A line longer than
/// `WINDOW` characters is cut to a window around the span, and each cut end
/// is marked with `...`.
fn snippet(map: &SourceMap, span: Span) -> Option<String> {
    let (line, col) = map.line_col(span);
    let text = map.line_text(span.file, line)?;
    let text = text.strip_suffix('\r').unwrap_or(text);
    let (end_line, end_col) = map.line_col(Span::new(span.file, span.end, span.end));
    let chars: Vec<char> = text.chars().collect();
    let width = chars.len();
    let end_col = if end_line == line {
        end_col.min(width + 1)
    } else {
        width + 1
    };
    // One caret at the start, then `~` up to the exclusive end.
    let tildes = end_col.saturating_sub(col + 1);
    // The window: `WINDOW` characters, starting `LEFT_CONTEXT` before the
    // span, and moved left when the line ends sooner.
    let start = col - 1;
    let (lo, hi) = if width <= WINDOW {
        (0, width)
    } else {
        let hi = (start.saturating_sub(LEFT_CONTEXT) + WINDOW).min(width);
        (hi - WINDOW, hi)
    };
    let (cut_left, cut_right) = (lo > 0, hi < width);
    let mut shown = String::new();
    let mut caret = String::new();
    if cut_left {
        shown.push_str("...");
        caret.push_str("   ");
    }
    shown.push_str(&plain(&chars[lo..hi].iter().collect::<String>()));
    if cut_right {
        shown.push_str("...");
    }
    // Tabs before the span are copied so the caret lines up.
    for c in chars.iter().take(start.min(hi)).skip(lo) {
        caret.push(if *c == '\t' { '\t' } else { ' ' });
    }
    // A span past the end of the line (a missing token at the line end)
    // still gets its caret, in the column after the last character.
    for _ in width..start {
        caret.push(' ');
    }
    caret.push('^');
    // The tildes stop at the end of the window.
    caret.extend(std::iter::repeat_n(
        '~',
        tildes.min(hi.saturating_sub(start + 1)),
    ));
    Some(format!("{shown}\n{caret}\n"))
}

fn diagnostic(out: &mut String, map: &SourceMap, d: &Diagnostic) {
    let _ = writeln!(
        out,
        "{}: {}[{}]: {}",
        at(map, d.span),
        severity_name(d.severity),
        d.code,
        plain(&d.message)
    );
    if let Some(s) = snippet(map, d.span) {
        out.push_str(&s);
    }
    if let Some(fix) = &d.fix_it {
        let (l1, c1) = map.line_col(fix.span);
        let (l2, c2) = map.line_col(Span::new(fix.span.file, fix.span.end, fix.span.end));
        let _ = writeln!(
            out,
            "  fix-it: replace {l1}:{c1}-{l2}:{c2} with {}",
            serde_json::to_string(&fix.replacement).expect("a string serializes")
        );
    }
    for (span, message) in &d.notes {
        let _ = writeln!(out, "  note: {}: {}", at(map, *span), plain(message));
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
    use pc_ode::{Code, FixIt, MapLoader, load};

    fn run(name: &str, text: &str) -> String {
        let mut map = SourceMap::new();
        let top = map.add(name, text);
        let loaded = load(top, &mut map, &mut MapLoader::new());
        render(&map, &loaded)
    }

    /// The text of one diagnostic with span `start..end` in file `text`.
    fn one(text: &str, start: usize, end: usize) -> String {
        let mut map = SourceMap::new();
        let f = map.add("a.ode", text);
        let d = Diagnostic::new(Code(12), Span::new(f, start, end), "msg".to_string());
        let mut out = String::new();
        diagnostic(&mut out, &map, &d);
        out
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
        assert!(out.contains("\n\nkind: "), "{out}");
    }

    #[test]
    fn exact_block_with_tab_prefix_and_wide_char() {
        // Span `=\u{e9}@` after a tab, `x` and `'`: caret under `=`,
        // tab copied, tildes up to the exclusive end.
        let got = one("\tx'=\u{e9}@\ny=1\n", 3, 7);
        assert_eq!(
            got,
            "a.ode:1:4: error[ODE0012]: msg\n\tx'=\u{e9}@\n\t  ^~~\n"
        );
        // A non-ASCII char before the span counts as one column.
        let got = one("\tx'=\u{e9}@\ny=1\n", 6, 7);
        assert_eq!(
            got,
            "a.ode:1:6: error[ODE0012]: msg\n\tx'=\u{e9}@\n\t    ^\n"
        );
    }

    #[test]
    fn multi_line_span_is_clamped_to_the_first_line() {
        let got = one("abcd\nef\n", 1, 6);
        assert_eq!(got, "a.ode:1:2: error[ODE0012]: msg\nabcd\n ^~~\n");
    }

    #[test]
    fn crlf_and_span_at_end_of_line() {
        let got = one("a\r\nbc@\r\n", 5, 5);
        assert_eq!(got, "a.ode:2:3: error[ODE0012]: msg\nbc@\n  ^\n");
        let got = one("ab", 2, 2);
        assert_eq!(got, "a.ode:1:3: error[ODE0012]: msg\nab\n  ^\n");
    }

    #[test]
    fn control_characters_in_the_source_line_are_replaced() {
        // ESC before the span and BEL after it.
        let got = one("x'=\u{1b}@\u{7}a\n", 4, 5);
        assert_eq!(
            got,
            "a.ode:1:5: error[ODE0012]: msg\nx'=\u{fffd}@\u{fffd}a\n    ^\n"
        );
    }

    #[test]
    fn long_line_is_cut_on_both_sides_around_a_span_in_the_middle() {
        // 149 `a`, the span `@`, 150 `b`: the window holds 40 `a`, `@` and 79
        // `b`, and both ends are cut.
        let text = format!("{}@{}\n", "a".repeat(149), "b".repeat(150));
        let got = one(&text, 149, 150);
        let echoed = format!("...{}@{}...", "a".repeat(40), "b".repeat(79));
        let caret = format!("{}^", " ".repeat(3 + 40));
        assert_eq!(
            got,
            format!("a.ode:1:150: error[ODE0012]: msg\n{echoed}\n{caret}\n")
        );
    }

    #[test]
    fn long_line_is_cut_on_the_left_only_when_the_span_is_at_the_end() {
        let text = format!("{}@\n", "a".repeat(299));
        let got = one(&text, 299, 300);
        let echoed = format!("...{}@", "a".repeat(119));
        let caret = format!("{}^", " ".repeat(3 + 119));
        assert_eq!(
            got,
            format!("a.ode:1:300: error[ODE0012]: msg\n{echoed}\n{caret}\n")
        );
        // A span at the start is cut on the right only.
        let text = format!("@{}\n", "b".repeat(299));
        let got = one(&text, 0, 3);
        let echoed = format!("@{}...", "b".repeat(119));
        assert_eq!(
            got,
            format!("a.ode:1:1: error[ODE0012]: msg\n{echoed}\n^~~\n")
        );
    }

    #[test]
    fn a_line_of_exactly_the_window_is_not_cut() {
        let text = format!("{}\n", "a".repeat(WINDOW));
        let got = one(&text, 5, 6);
        assert_eq!(
            got,
            format!(
                "a.ode:1:6: error[ODE0012]: msg\n{}\n{}^\n",
                "a".repeat(WINDOW),
                " ".repeat(5)
            )
        );
    }

    #[test]
    fn control_characters_in_names_and_messages_are_replaced() {
        let mut map = SourceMap::new();
        let f = map.add("a\u{1b}[2J.ode", "x\n");
        let d = Diagnostic::new(Code(12), Span::new(f, 0, 1), "m\nn\u{7}".to_string());
        let mut out = String::new();
        diagnostic(&mut out, &map, &d);
        assert_eq!(
            out,
            "a\u{fffd}[2J.ode:1:1: error[ODE0012]: m\u{fffd}n\u{fffd}\nx\n^\n"
        );
    }

    #[test]
    fn fix_it_and_note_lines_are_exact() {
        let mut map = SourceMap::new();
        let f = map.add("a.ode", "\tx'=\u{e9}@\ny=1\n");
        let g = map.add("inc/b.ode", "zz\n");
        let mut d = Diagnostic::new(Code(12), Span::new(f, 6, 7), "msg".to_string());
        d.fix_it = Some(FixIt {
            span: Span::new(f, 6, 7),
            replacement: "a\"b\u{e9}".to_string(),
        });
        d.notes.push((Span::new(f, 8, 9), "here".to_string()));
        d.notes.push((Span::new(g, 0, 2), "there".to_string()));
        let mut out = String::new();
        diagnostic(&mut out, &map, &d);
        assert_eq!(
            out,
            "a.ode:1:6: error[ODE0012]: msg\n\tx'=\u{e9}@\n\t    ^\n\
             \x20 fix-it: replace 1:6-1:7 with \"a\\\"b\u{e9}\"\n\
             \x20 note: a.ode:2:1: here\n\
             \x20 note: inc/b.ode:1:1: there\n"
        );
    }
}
