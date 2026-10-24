//! Helpers shared by the integration tests of `pc-ode`.

#![allow(dead_code)]

use pc_ode::{FileId, Model, Severity, parse, write, write_expanded};

const FILE: FileId = FileId(0);

/// Parses `src` and fails the test on any error-severity diagnostic.
pub fn parse_ok(src: &str) -> Model {
    let (model, diags) = parse(src, FILE);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "errors in\n{src}\n{errors:#?}");
    model
}

/// Asserts that `m` survives `write` then `parse` (spans stripped) and that
/// the written text parses without errors.
pub fn assert_round_trip(m: &Model) {
    let text = write(m);
    let back = parse_ok(&text);
    assert_eq!(back.strip_spans(), m.strip_spans(), "text:\n{text}");
}

/// A copy of an expanded model without the array marks (`array_index`,
/// `array_template` and `suppressed`), which the flat text cannot carry.
pub fn without_array_marks(expanded: &Model) -> Model {
    let mut flat = expanded.clone();
    flat.suppressed.clear();
    for s in &mut flat.statements {
        s.array_index = None;
        s.array_template = None;
    }
    flat
}

/// Asserts that an expanded model survives `write_expanded` then `parse`
/// once the array marks are cleared.
pub fn assert_expanded_round_trip(expanded: &Model) {
    let flat = without_array_marks(expanded);
    let text = write_expanded(&flat);
    let back = parse_ok(&text);
    assert_eq!(back.strip_spans(), flat.strip_spans(), "text:\n{text}");
}
