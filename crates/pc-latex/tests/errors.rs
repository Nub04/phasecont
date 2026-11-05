//! Bad input is an error that names what is wrong. Nothing is dropped or defaulted.

use pc_latex::{ErrorKind, Kind, render};

fn fig(extra: &str) -> String {
    format!(
        r#"{{"xlabel":"a","ylabel":"x","digits":4,"series":[{{"class":"stable","x":[0,1],"y":[0,1]}}]{extra}}}"#
    )
}

fn err(kind: Kind, json: &str) -> (ErrorKind, String) {
    let e = render(kind, json).expect_err(json);
    (e.kind(), e.to_string())
}

#[test]
fn missing_digits() {
    let json = r#"{"xlabel":"a","ylabel":"x","series":[{"class":"stable","x":[0],"y":[0]}]}"#;
    let (k, msg) = err(Kind::Pgfplots, json);
    assert_eq!(k, ErrorKind::MissingDigits);
    assert!(msg.contains("digits"), "{msg}");
    let t = r#"{"columns":["a"],"rows":[{"label":"p","type":"t","values":[1]}]}"#;
    assert_eq!(err(Kind::Table, t).0, ErrorKind::MissingDigits);
}

#[test]
fn bad_digits() {
    for d in ["0", "18", "-1", "2.5", "\"4\"", "null"] {
        let json = fig("").replace("\"digits\":4", &format!("\"digits\":{d}"));
        assert_eq!(
            err(Kind::Pgfplots, &json).0,
            ErrorKind::BadDigits,
            "digits {d}"
        );
    }
}

#[test]
fn null_is_how_matlab_writes_nan_and_inf() {
    let json = r#"{"xlabel":"a","ylabel":"x","digits":4,
        "series":[{"class":"stable","x":[0,1],"y":[0,1]},
                  {"class":"stable","x":[0,1,2],"y":[0,null,2]}]}"#;
    let (k, msg) = err(Kind::Pgfplots, json);
    assert_eq!(k, ErrorKind::NonFinite);
    assert!(msg.contains("series 2") && msg.contains("y(2)"), "{msg}");
}

#[test]
fn non_finite_mark_and_table_value() {
    let json = fig(r#","marks":[{"label":"HB","x":1,"y":null}]"#);
    let (k, msg) = err(Kind::Pgfplots, &json);
    assert_eq!(k, ErrorKind::NonFinite);
    assert!(msg.contains("mark 1") && msg.contains("y"), "{msg}");
    let t = r#"{"columns":["a","b"],"digits":3,"rows":[
        {"label":"p","type":"t","values":[1,2]},
        {"label":"q","type":"t","values":[3,null]}]}"#;
    let (k, msg) = err(Kind::Table, t);
    assert_eq!(k, ErrorKind::NonFinite);
    assert!(msg.contains("row 2") && msg.contains("values(2)"), "{msg}");
}

#[test]
fn a_number_too_big_for_f64_is_not_read_as_infinity() {
    let json = fig("").replace("[0,1],\"y\":[0,1]", "[0,1],\"y\":[0,1e999]");
    assert!(render(Kind::Pgfplots, &json).is_err());
}

#[test]
fn empty_series_and_no_series() {
    let json =
        r#"{"xlabel":"a","ylabel":"x","digits":4,"series":[{"class":"stable","x":[],"y":[]}]}"#;
    let (k, msg) = err(Kind::Pgfplots, json);
    assert_eq!(k, ErrorKind::EmptySeries);
    assert!(msg.contains("series 1"), "{msg}");
    let json = r#"{"xlabel":"a","ylabel":"x","digits":4,"series":[]}"#;
    assert_eq!(err(Kind::Pgfplots, json).0, ErrorKind::EmptySeries);
}

#[test]
fn empty_table() {
    let json = r#"{"columns":["a"],"digits":3,"rows":[]}"#;
    assert_eq!(err(Kind::Table, json).0, ErrorKind::EmptyTable);
}

#[test]
fn length_mismatch() {
    let json = fig("").replace("\"y\":[0,1]", "\"y\":[0,1,2]");
    let (k, msg) = err(Kind::Pgfplots, &json);
    assert_eq!(k, ErrorKind::LengthMismatch);
    assert!(msg.contains("series 1"), "{msg}");
    let t = r#"{"columns":["a","b"],"digits":3,"rows":[{"label":"p","type":"t","values":[1]}]}"#;
    assert_eq!(err(Kind::Table, t).0, ErrorKind::LengthMismatch);
}

#[test]
fn external_data_needs_a_safe_basename() {
    let ext = |b: &str| fig(&format!(r#","data":"external","basename":"{b}""#));
    assert_eq!(
        err(Kind::Pgfplots, &fig(r#","data":"external""#)).0,
        ErrorKind::BadName
    );
    for bad in ["", "a b", "../x", "a/b", "x.dat", "a;b", "é"] {
        assert_eq!(
            err(Kind::Pgfplots, &ext(bad)).0,
            ErrorKind::BadName,
            "{bad:?}"
        );
    }
    assert!(render(Kind::Pgfplots, &ext("fold_1-b")).is_ok());
    // A basename with inline data is a mistake, not something to ignore.
    let json = fig(r#","basename":"x""#);
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
}

#[test]
fn bad_input_shapes() {
    for (kind, json) in [
        (Kind::Pgfplots, "not json"),
        (Kind::Pgfplots, "[]"),
        (Kind::Pgfplots, r#"{"digits":4}"#),
        (Kind::Table, r#"{"digits":4}"#),
        (
            Kind::Pgfplots,
            r#"{"xlabel":1,"ylabel":"x","digits":4,"series":[]}"#,
        ),
    ] {
        assert_eq!(err(kind, json).0, ErrorKind::BadInput, "{json}");
    }
    // A scalar where an array is required (MATLAB's jsonencode writes one for a
    // single element).
    let json = fig("").replace("\"x\":[0,1]", "\"x\":5");
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
    let json = fig("").replace("\"stable\"", "\"fast\"");
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
    let json = fig("").replace("\"class\":\"stable\",", "");
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
    let json = fig(r#","data":"remote""#);
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
    // A misspelled `series` leaves the required key missing.
    let json = fig("").replace("\"series\"", "\"serie\"");
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
}

#[test]
fn unknown_keys_are_errors() {
    let (k, msg) = err(Kind::Pgfplots, &fig(r#","standalone_":true"#));
    assert_eq!(k, ErrorKind::BadInput);
    assert!(msg.contains("standalone_"), "{msg}");
    let json = fig("").replace(
        "\"class\":\"stable\"",
        "\"class\":\"stable\",\"colour\":\"#000000\"",
    );
    assert!(err(Kind::Pgfplots, &json).1.contains("colour"));
    let json = fig("").replace(
        "\"class\":\"stable\"",
        "\"class\":\"stable\",\"color\":\"red\"",
    );
    assert_eq!(err(Kind::Pgfplots, &json).0, ErrorKind::BadInput);
}

#[test]
fn kinds_have_the_names_the_matlab_layer_uses() {
    let names: Vec<&str> = [
        ErrorKind::MissingDigits,
        ErrorKind::BadDigits,
        ErrorKind::NonFinite,
        ErrorKind::EmptySeries,
        ErrorKind::EmptyTable,
        ErrorKind::LengthMismatch,
        ErrorKind::BadName,
        ErrorKind::BadInput,
    ]
    .iter()
    .map(|k| k.as_str())
    .collect();
    assert_eq!(
        names,
        [
            "missingDigits",
            "badDigits",
            "nonFinite",
            "emptySeries",
            "emptyTable",
            "lengthMismatch",
            "badName",
            "badInput"
        ]
    );
}
