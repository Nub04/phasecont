use lsp_types::{
    CodeActionKind, CodeActionOrCommand, CompletionItemKind, Diagnostic, HoverContents, Position,
    Range, SymbolKind as LspKind,
};

use super::*;
use crate::analysis::{Analysis, analyze};
use crate::convert::LineIndex;
use crate::overlay::Overlay;
use crate::uri::uri_from_path;

const MODEL: &str = "\
# test model
par a=1, b=0.5
x'=a*x-b*x*y
y'=-y+sin(x)
w=x+y
aux total=w
f(u,v)=u*v
init x=1, y=2
@ total=100
done
";

const NAME: &str = "/work/m.ode";

fn analysis(text: &str) -> Analysis {
    let overlay = Overlay::new();
    let mut loader = overlay.loader(None);
    analyze(NAME, text.to_string(), &mut loader)
}

/// The position of the `nth` (0-based) occurrence of `needle`, moved
/// `inside` characters into it.
fn pos_of(text: &str, needle: &str, nth: usize, inside: usize) -> Position {
    let at = text
        .match_indices(needle)
        .nth(nth)
        .expect("needle not found")
        .0;
    LineIndex::new(text).position(at + inside)
}

fn hover_text(a: &Analysis, pos: Position) -> Option<String> {
    match hover(a, a.top, pos)?.contents {
        HoverContents::Markup(m) => Some(m.value),
        other => panic!("unexpected hover contents {other:?}"),
    }
}

#[test]
fn the_test_model_has_no_diagnostics() {
    let a = analysis(MODEL);
    assert!(
        a.loaded.diagnostics.is_empty(),
        "{:?}",
        a.loaded.diagnostics
    );
}

#[test]
fn word_at_finds_the_identifier_around_an_offset() {
    let t = "x'=a*x";
    assert_eq!(word_at(t, 3), Some((3, 4)));
    // Just after the word still counts, as in an editor.
    assert_eq!(word_at(t, 4), Some((3, 4)));
    assert_eq!(word_at(t, 5), Some((5, 6)));
    assert_eq!(word_at(t, 0), Some((0, 1)));
    assert_eq!(word_at(t, 2), None);
    assert_eq!(word_at("a  b", 2), None);
    assert_eq!(word_at("", 0), None);
    assert_eq!(word_at("ab", 99), Some((0, 2)));
    // Identifiers are ASCII; a character inside a multi-byte one is no word.
    assert_eq!(word_at("\u{e9}ab", 3), Some((2, 4)));
    assert_eq!(word_at("\u{e9}ab", 1), None);
}

#[test]
fn hover_names_the_kind_of_a_declared_name() {
    let a = analysis(MODEL);
    let text = hover_text(&a, pos_of(MODEL, "a*x", 0, 0)).unwrap();
    assert!(text.contains("`a`") && text.contains("parameter"), "{text}");
    let text = hover_text(&a, pos_of(MODEL, "w=x+y", 0, 0)).unwrap();
    assert!(text.contains("fixed quantity"), "{text}");
    let text = hover_text(&a, pos_of(MODEL, "f(u,v)", 0, 0)).unwrap();
    assert!(
        text.contains("function") && text.contains("2 arguments"),
        "{text}"
    );
}

#[test]
fn hover_gives_the_signature_of_a_builtin() {
    let a = analysis(MODEL);
    let text = hover_text(&a, pos_of(MODEL, "sin(", 0, 1)).unwrap();
    assert!(text.contains("`sin`") && text.contains("builtin"), "{text}");
    assert!(text.contains("1 argument"), "{text}");
}

#[test]
fn hover_on_nothing_gives_none() {
    let a = analysis(MODEL);
    // The `#` that starts a comment, and a word that is not declared.
    assert!(hover_text(&a, Position::new(0, 0)).is_none());
    assert!(hover_text(&a, pos_of(MODEL, "test", 0, 1)).is_none());
    assert!(hover(&a, a.top, Position::new(500, 0)).is_none());
}

#[test]
fn hover_range_covers_the_word() {
    let a = analysis(MODEL);
    let pos = pos_of(MODEL, "a*x", 0, 0);
    let range = hover(&a, a.top, pos).unwrap().range.unwrap();
    assert_eq!(
        range,
        Range::new(pos, Position::new(pos.line, pos.character + 1))
    );
}

fn labels(items: &[lsp_types::CompletionItem]) -> Vec<&str> {
    items.iter().map(|i| i.label.as_str()).collect()
}

#[test]
fn completion_offers_builtins_and_declared_names() {
    let a = analysis(MODEL);
    let items = completion(&a, a.top, pos_of(MODEL, "sin(", 0, 2));
    let names = labels(&items);
    for want in ["sin", "cos", "a", "b", "x", "w", "total", "f"] {
        assert!(names.contains(&want), "missing {want}");
    }
    assert!(!names.contains(&"xp2"), "option names only after an @");
    let sin = items.iter().find(|i| i.label == "sin").unwrap();
    assert_eq!(sin.kind, Some(CompletionItemKind::FUNCTION));
    let a_item = items.iter().find(|i| i.label == "a").unwrap();
    assert_eq!(a_item.kind, Some(CompletionItemKind::CONSTANT));
}

#[test]
fn completion_after_an_at_sign_offers_option_names() {
    let a = analysis(MODEL);
    let items = completion(&a, a.top, pos_of(MODEL, "total=100", 0, 2));
    let names = labels(&items);
    for want in ["total", "xp2", "yp8", "s3", "maxstor"] {
        assert!(names.contains(&want), "missing {want}");
    }
    assert!(!names.contains(&"sin"), "no builtins inside an @ line");
    assert!(
        items
            .iter()
            .all(|i| i.kind == Some(CompletionItemKind::PROPERTY))
    );
}

#[test]
fn document_symbols_list_the_declarations_of_the_file() {
    let a = analysis(MODEL);
    let symbols = document_symbols(&a, a.top);
    let by_name = |n: &str| {
        symbols
            .iter()
            .find(|s| s.name == n)
            .unwrap_or_else(|| panic!("no {n}"))
    };
    for n in ["a", "b", "x", "y", "w", "total", "f"] {
        by_name(n);
    }
    assert_eq!(by_name("x").kind, LspKind::VARIABLE);
    assert_eq!(by_name("f").kind, LspKind::FUNCTION);
    let a_sym = by_name("a");
    let at = pos_of(MODEL, "par a=1", 0, 4);
    assert_eq!(
        a_sym.selection_range,
        Range::new(at, Position::new(at.line, at.character + 1))
    );
    assert_eq!(a_sym.range, a_sym.selection_range);
}

#[test]
fn definition_points_at_the_declaring_name() {
    let a = analysis(MODEL);
    let mut uri_of = |n: &str| uri_from_path(n);
    let loc = definition(&a, a.top, pos_of(MODEL, "a*x", 0, 0), &mut uri_of).unwrap();
    assert_eq!(loc.uri, uri_from_path(NAME).unwrap());
    let at = pos_of(MODEL, "par a=1", 0, 4);
    assert_eq!(
        loc.range,
        Range::new(at, Position::new(at.line, at.character + 1))
    );
    // A builtin and a blank have no definition in the file.
    assert!(definition(&a, a.top, pos_of(MODEL, "sin(", 0, 1), &mut uri_of).is_none());
    assert!(definition(&a, a.top, Position::new(900, 0), &mut uri_of).is_none());
}

fn diagnostic_with(data: Option<serde_json::Value>, source: &str) -> Diagnostic {
    Diagnostic {
        range: Range::new(Position::new(1, 0), Position::new(1, 3)),
        message: "m".to_string(),
        source: Some(source.to_string()),
        data,
        ..Diagnostic::default()
    }
}

fn fix(new_text: &str) -> serde_json::Value {
    serde_json::json!({ "fix": {
        "range": Range::new(Position::new(1, 0), Position::new(1, 3)),
        "newText": new_text,
    }})
}

#[test]
fn a_fix_it_becomes_a_quick_fix() {
    let uri = uri_from_path(NAME).unwrap();
    let d = diagnostic_with(Some(fix("par")), "phasecont");
    let actions = code_actions(&uri, std::slice::from_ref(&d));
    assert_eq!(actions.len(), 1);
    let CodeActionOrCommand::CodeAction(action) = &actions[0] else {
        panic!("not a code action");
    };
    assert_eq!(action.kind, Some(CodeActionKind::QUICKFIX));
    assert_eq!(action.title, "Replace with `par`");
    assert_eq!(
        action.diagnostics.as_deref(),
        Some(std::slice::from_ref(&d))
    );
    let edits = action.edit.as_ref().unwrap().changes.as_ref().unwrap();
    let list = &edits[&uri];
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].new_text, "par");
    assert_eq!(list[0].range, d.range);
}

#[test]
fn an_empty_replacement_is_a_removal() {
    let uri = uri_from_path(NAME).unwrap();
    let d = diagnostic_with(Some(fix("")), "phasecont");
    let actions = code_actions(&uri, &[d]);
    let CodeActionOrCommand::CodeAction(action) = &actions[0] else {
        panic!("not a code action");
    };
    assert_eq!(action.title, "Remove this text");
}

#[test]
fn diagnostics_without_a_fix_or_from_another_source_give_no_action() {
    let uri = uri_from_path(NAME).unwrap();
    assert!(code_actions(&uri, &[diagnostic_with(None, "phasecont")]).is_empty());
    assert!(code_actions(&uri, &[diagnostic_with(Some(fix("x")), "rustc")]).is_empty());
    let bad = serde_json::json!({ "fix": { "range": 3 } });
    assert!(code_actions(&uri, &[diagnostic_with(Some(bad), "phasecont")]).is_empty());
}
