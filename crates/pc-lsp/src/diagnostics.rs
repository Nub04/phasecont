//! `pc-ode` diagnostics as LSP diagnostics.
//!
//! The severity maps one to one (a note becomes information). The code is
//! the `ODE0001` form and the source is `phasecont`. Notes become related
//! information. A fix-it in the same file as its diagnostic travels in the
//! diagnostic's `data` as `{"fix": {"range": ..., "newText": ...}}`, so a code
//! action request, which echoes the diagnostics back, can offer it without
//! another analysis.

use lsp_types::{
    Diagnostic, DiagnosticRelatedInformation, DiagnosticSeverity, Location, NumberOrString, Uri,
};
use pc_ode::{FileId, Severity, Span};
use serde_json::json;

use crate::analysis::Analysis;
use crate::convert::LineIndex;

/// The source name shown next to each diagnostic.
pub const SOURCE: &str = "phasecont";

/// Every diagnostic of `analysis`, each with the URI of the file it is in.
/// `uri_of` turns a file display name into a URI; a diagnostic whose file
/// has none is left out. The order is the order of `pc-ode`.
pub fn to_lsp(
    analysis: &Analysis,
    uri_of: &mut dyn FnMut(&str) -> Option<Uri>,
) -> Vec<(Uri, Diagnostic)> {
    let files = analysis.map.file_count();
    let indexes: Vec<LineIndex<'_>> = (0..files)
        .map(|i| LineIndex::new(analysis.map.text(FileId(i as u32))))
        .collect();
    let mut uris: Vec<Option<Uri>> = (0..files)
        .map(|i| uri_of(analysis.map.name(FileId(i as u32))))
        .collect();
    let range = |span: Span| {
        let index = indexes.get(span.file.0 as usize)?;
        Some(index.range(span.start, span.end))
    };
    let mut out = Vec::with_capacity(analysis.loaded.diagnostics.len());
    for d in &analysis.loaded.diagnostics {
        let slot = d.span.file.0 as usize;
        let (Some(uri), Some(range_of_d)) = (uris.get(slot).cloned().flatten(), range(d.span))
        else {
            continue;
        };
        let related: Vec<DiagnosticRelatedInformation> = d
            .notes
            .iter()
            .filter_map(|(span, message)| {
                let uri = uris.get(span.file.0 as usize)?.clone()?;
                Some(DiagnosticRelatedInformation {
                    location: Location::new(uri, range(*span)?),
                    message: message.clone(),
                })
            })
            .collect();
        let data = d.fix_it.as_ref().and_then(|fix| {
            if fix.span.file != d.span.file {
                return None;
            }
            let r = range(fix.span)?;
            Some(json!({ "fix": { "range": r, "newText": fix.replacement } }))
        });
        out.push((
            uri,
            Diagnostic {
                range: range_of_d,
                severity: Some(severity(d.severity)),
                code: Some(NumberOrString::String(d.code.to_string())),
                code_description: None,
                source: Some(SOURCE.to_string()),
                message: d.message.clone(),
                related_information: (!related.is_empty()).then_some(related),
                tags: None,
                data,
            },
        ));
    }
    uris.clear();
    out
}

fn severity(s: Severity) -> DiagnosticSeverity {
    match s {
        Severity::Error => DiagnosticSeverity::ERROR,
        Severity::Warning => DiagnosticSeverity::WARNING,
        Severity::Note => DiagnosticSeverity::INFORMATION,
    }
}
