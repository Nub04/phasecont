//! What the server answers besides diagnostics: hover, completion, document
//! symbols, go to definition and code actions.
//!
//! All of them work on one [`Analysis`] and on the file of the request.
//! A name is looked up in the declared symbols of the model first and then
//! in the builtin table of `pc-ode`. A name that the model builds from an
//! array template (`x1` from `x[1..3]`) is found, and its definition is the
//! template line.

use std::collections::HashMap;

use lsp_types::{
    CodeAction, CodeActionKind, CodeActionOrCommand, CompletionItem, CompletionItemKind,
    Diagnostic, DocumentSymbol, Hover, HoverContents, Location, MarkupContent, MarkupKind,
    Position, Range, SymbolKind as LspKind, TextEdit, Uri, WorkspaceEdit,
};
use pc_ode::builtins::{BUILTINS, Builtin, builtin};
use pc_ode::options::{NAMES, NUMBERED};
use pc_ode::{FileId, Span, StateKind, Symbol, SymbolKind};

use crate::analysis::Analysis;
use crate::convert::LineIndex;
use crate::diagnostics::SOURCE;

/// The byte range of the identifier that touches `offset`: the one the
/// offset is inside, or the one that ends at it. Identifiers are ASCII
/// letters, digits and `_`. An offset past the end or inside a character is
/// moved back to a character start first.
pub fn word_at(text: &str, offset: usize) -> Option<(usize, usize)> {
    let mut offset = offset.min(text.len());
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let bytes = text.as_bytes();
    let mut start = offset;
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset;
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }
    (start < end).then_some((start, end))
}

/// The text of `file` and the word at `pos` in it.
fn word_in(a: &Analysis, file: FileId, pos: Position) -> Option<(&str, usize, usize)> {
    if file.0 as usize >= a.map.file_count() {
        return None;
    }
    let text = a.map.text(file);
    let offset = LineIndex::new(text).offset(pos);
    let (start, end) = word_at(text, offset)?;
    Some((text, start, end))
}

fn symbol_of<'a>(a: &'a Analysis, word: &str) -> Option<&'a Symbol> {
    a.loaded.info.symbol(&word.to_ascii_lowercase())
}

fn plural(n: usize) -> String {
    if n == 1 {
        "1 argument".to_string()
    } else {
        format!("{n} arguments")
    }
}

fn describe(kind: SymbolKind) -> String {
    match kind {
        SymbolKind::State(StateKind::Ode) => "ODE variable".to_string(),
        SymbolKind::State(StateKind::Map) => "map variable".to_string(),
        SymbolKind::State(StateKind::Markov) => "Markov variable".to_string(),
        SymbolKind::Parameter => "parameter".to_string(),
        SymbolKind::Number => "number".to_string(),
        SymbolKind::Derived => "derived parameter".to_string(),
        SymbolKind::Fixed => "fixed quantity".to_string(),
        SymbolKind::Aux => "aux quantity".to_string(),
        SymbolKind::Function { params } => format!("function of {}", plural(params)),
        SymbolKind::Table => "table".to_string(),
        SymbolKind::Wiener => "Wiener variable".to_string(),
    }
}

fn describe_builtin(b: &Builtin) -> String {
    let mut text = match b.arity {
        Some(n) => format!("builtin function of {}", plural(n)),
        None => "builtin function".to_string(),
    };
    if !b.smooth {
        text.push_str(". Continuation cannot use it, because it has no derivative");
    }
    if b.simulation_only {
        text.push_str(". A model that calls it can only be simulated");
    }
    text
}

fn symbol_kind(kind: SymbolKind) -> LspKind {
    match kind {
        SymbolKind::State(_) => LspKind::VARIABLE,
        SymbolKind::Parameter | SymbolKind::Derived => LspKind::CONSTANT,
        SymbolKind::Number => LspKind::NUMBER,
        SymbolKind::Fixed | SymbolKind::Aux => LspKind::FIELD,
        SymbolKind::Function { .. } => LspKind::FUNCTION,
        SymbolKind::Table => LspKind::ARRAY,
        SymbolKind::Wiener => LspKind::EVENT,
    }
}

fn completion_kind(kind: SymbolKind) -> CompletionItemKind {
    match kind {
        SymbolKind::State(_) | SymbolKind::Fixed | SymbolKind::Aux | SymbolKind::Wiener => {
            CompletionItemKind::VARIABLE
        }
        SymbolKind::Parameter | SymbolKind::Number | SymbolKind::Derived => {
            CompletionItemKind::CONSTANT
        }
        SymbolKind::Function { .. } | SymbolKind::Table => CompletionItemKind::FUNCTION,
    }
}

/// The range of a span in its own file, or `None` for a span whose file the
/// source map does not have (a placeholder span).
fn span_range(a: &Analysis, span: Span) -> Option<Range> {
    if span.file.0 as usize >= a.map.file_count() {
        return None;
    }
    let text = a.map.text(span.file);
    Some(LineIndex::new(text).range(span.start, span.end))
}

/// Hover text for the name at `pos`: what a declared name is, or the
/// signature of a builtin.
pub fn hover(a: &Analysis, file: FileId, pos: Position) -> Option<Hover> {
    let (text, start, end) = word_in(a, file, pos)?;
    let word = &text[start..end];
    let value = if let Some(symbol) = symbol_of(a, word) {
        format!("`{}`: {}", symbol.name.text, describe(symbol.kind))
    } else {
        let b = builtin(&word.to_ascii_lowercase())?;
        format!("`{}`: {}", b.name, describe_builtin(b))
    };
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value,
        }),
        range: Some(LineIndex::new(text).range(start, end)),
    })
}

/// Completion items at `pos`. On a line that starts with `@` they are the
/// option names. Elsewhere they are the builtins and the declared names. The
/// client filters them by the prefix it has.
pub fn completion(a: &Analysis, file: FileId, pos: Position) -> Vec<CompletionItem> {
    if file.0 as usize >= a.map.file_count() {
        return Vec::new();
    }
    let text = a.map.text(file);
    let offset = LineIndex::new(text).offset(pos);
    let word_start = word_at(text, offset).map_or(offset, |(s, _)| s);
    let line_start = text[..word_start].rfind(['\n', '\r']).map_or(0, |i| i + 1);
    if text[line_start..word_start].trim_start().starts_with('@') {
        return option_items();
    }
    let mut items: Vec<CompletionItem> = BUILTINS
        .iter()
        .map(|b| CompletionItem {
            label: b.name.to_string(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: Some(describe_builtin(b)),
            ..CompletionItem::default()
        })
        .collect();
    items.extend(a.loaded.info.symbols.iter().map(|s| CompletionItem {
        label: s.name.text.clone(),
        kind: Some(completion_kind(s.kind)),
        detail: Some(describe(s.kind)),
        ..CompletionItem::default()
    }));
    items
}

fn option_items() -> Vec<CompletionItem> {
    let numbered = NUMBERED
        .iter()
        .flat_map(|&(root, lo, hi)| (lo..=hi).map(move |n| format!("{root}{n}")));
    NAMES
        .iter()
        .map(|n| n.to_string())
        .chain(numbered)
        .map(|label| CompletionItem {
            label,
            kind: Some(CompletionItemKind::PROPERTY),
            detail: Some("option".to_string()),
            ..CompletionItem::default()
        })
        .collect()
}

/// The names declared in `file`, in declaration order, as a flat list.
#[allow(deprecated)] // `DocumentSymbol::deprecated` has no default in lsp-types.
pub fn document_symbols(a: &Analysis, file: FileId) -> Vec<DocumentSymbol> {
    a.loaded
        .info
        .symbols
        .iter()
        .filter(|s| s.name.span.file == file)
        .filter_map(|s| {
            let range = span_range(a, s.name.span)?;
            Some(DocumentSymbol {
                name: s.name.text.clone(),
                detail: Some(describe(s.kind)),
                kind: symbol_kind(s.kind),
                tags: None,
                deprecated: None,
                range,
                selection_range: range,
                children: None,
            })
        })
        .collect()
}

/// Where the name at `pos` is declared. A builtin, an undeclared name and a
/// declaration whose file has no URI give `None`.
pub fn definition(
    a: &Analysis,
    file: FileId,
    pos: Position,
    uri_of: &mut dyn FnMut(&str) -> Option<Uri>,
) -> Option<Location> {
    let (text, start, end) = word_in(a, file, pos)?;
    let symbol = symbol_of(a, &text[start..end])?;
    let range = span_range(a, symbol.name.span)?;
    let uri = uri_of(a.map.name(symbol.name.span.file))?;
    Some(Location::new(uri, range))
}

/// The fix that the server put into a diagnostic's `data`, if it reads as one.
fn fix_of(d: &Diagnostic) -> Option<(Range, String)> {
    let fix = d.data.as_ref()?.get("fix")?;
    let range = serde_json::from_value(fix.get("range")?.clone()).ok()?;
    let new_text = fix.get("newText")?.as_str()?.to_string();
    Some((range, new_text))
}

/// Quick fixes for the diagnostics of `uri` that a code action request
/// echoes back. A diagnostic from another source, one without a fix, and one
/// whose data does not read as a fix give nothing.
pub fn code_actions(uri: &Uri, diagnostics: &[Diagnostic]) -> Vec<CodeActionOrCommand> {
    diagnostics
        .iter()
        .filter(|d| d.source.as_deref() == Some(SOURCE))
        .filter_map(|d| {
            let (range, new_text) = fix_of(d)?;
            let title = if new_text.is_empty() {
                "Remove this text".to_string()
            } else {
                format!("Replace with `{new_text}`")
            };
            let edit = TextEdit::new(range, new_text);
            Some(CodeActionOrCommand::CodeAction(CodeAction {
                title,
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![d.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(HashMap::from([(uri.clone(), vec![edit])])),
                    ..WorkspaceEdit::default()
                }),
                is_preferred: Some(true),
                ..CodeAction::default()
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests;
