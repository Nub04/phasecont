//! `latex_render` through the table from `pc_api_v1()`: a figure or table document in, TeX
//! and data files out. A rejected document is `PC_ERR_BAD_PROBLEM` with the error object in
//! `out_json`. The documents are our own.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const BAD_ARG: i32 = 2;
const BAD_PROBLEM: i32 = 6;

const PGFPLOTS: u32 = 1;
const TABLE: u32 = 2;

fn api() -> &'static PcApiV1 {
    let p = pc_api_v1();
    assert!(!p.is_null());
    unsafe { &*p }
}

fn last_error() -> String {
    let p = (api().last_error)();
    assert!(!p.is_null());
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn take_string(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(p) };
    Some(s)
}

/// Returns (code, parsed JSON result or None when the slot stayed NULL).
fn render(kind: u32, doc: &str) -> (i32, Option<Value>) {
    let c = CString::new(doc).unwrap();
    // A pre-set pointer proves the slot is cleared.
    let mut out = 0x10 as *mut c_char;
    let rc = unsafe { (api().latex_render)(kind, c.as_ptr(), &mut out) };
    let v = take_string(out).map(|s| serde_json::from_str(&s).unwrap());
    (rc, v)
}

const FIGURE: &str = r#"{"xlabel":"a","ylabel":"x","digits":4,
    "series":[{"class":"stable","x":[0,1,2],"y":[0,1,4]},
              {"class":"unstable","name":"fold","x":[2,3],"y":[4,9]}],
    "marks":[{"label":"LP","x":2,"y":4}]}"#;

const TABLE_DOC: &str = r#"{"columns":["a","x"],"digits":3,
    "rows":[{"label":"LP1","type":"LP","values":[1.5,2.25]}]}"#;

#[test]
fn a_figure_renders_to_the_same_text_as_the_crate() {
    let (rc, v) = render(PGFPLOTS, FIGURE);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    assert_eq!(v["ok"], true);
    let direct = pc_latex::render(pc_latex::Kind::Pgfplots, FIGURE).unwrap();
    assert_eq!(v["tex"], direct.tex);
    assert!(v["tex"].as_str().unwrap().contains("\\begin{axis}"));
    assert_eq!(v["data_files"].as_array().unwrap().len(), 0);
    assert_eq!(last_error(), "");
}

#[test]
fn a_table_renders() {
    let (rc, v) = render(TABLE, TABLE_DOC);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    assert_eq!(v["ok"], true);
    assert!(v["tex"].as_str().unwrap().contains("\\toprule"));
}

#[test]
fn external_data_comes_back_as_named_files() {
    let doc = FIGURE.replace(
        "\"digits\":4,",
        "\"digits\":4,\"data\":\"external\",\"basename\":\"fold-curve\",",
    );
    let (rc, v) = render(PGFPLOTS, &doc);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    let files = v["data_files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0]["name"], "fold-curve-s1.dat");
    assert_eq!(files[1]["name"], "fold-curve-s2.dat");
    assert!(files[0]["content"].as_str().unwrap().contains('4'));
}

#[test]
fn a_rejected_document_is_bad_problem_with_the_error_object() {
    let doc = FIGURE.replace("\"digits\":4,", "");
    let (rc, v) = render(PGFPLOTS, &doc);
    assert_eq!(rc, BAD_PROBLEM);
    let v = v.expect("out_json carries the error object");
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["id"], "phasecont:latex:missingDigits");
    assert_eq!(v["error"]["kind"], "missingDigits");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(message.contains("digits"), "{message}");
    assert_eq!(last_error(), message);
}

#[test]
fn every_error_kind_has_an_id() {
    let cases = [
        (
            PGFPLOTS,
            FIGURE.replace("\"digits\":4", "\"digits\":0"),
            "badDigits",
        ),
        (
            PGFPLOTS,
            FIGURE.replace("[0,1,4]", "[0,null,4]"),
            "nonFinite",
        ),
        (
            PGFPLOTS,
            FIGURE.replace("[0,1,4]", "[0,1]"),
            "lengthMismatch",
        ),
        (PGFPLOTS, "not json".to_string(), "badInput"),
        (
            TABLE,
            TABLE_DOC.replace("[1.5,2.25]", "[1.5]"),
            "lengthMismatch",
        ),
    ];
    for (kind, doc, expect) in cases {
        let (rc, v) = render(kind, &doc);
        assert_eq!(rc, BAD_PROBLEM, "{expect}: {doc}");
        let v = v.unwrap();
        assert_eq!(v["error"]["kind"], expect);
        assert_eq!(v["error"]["id"], format!("phasecont:latex:{expect}"));
    }
}

#[test]
fn an_unknown_kind_is_a_bad_argument_and_leaves_the_slot_null() {
    for kind in [0, 3, u32::MAX] {
        let (rc, v) = render(kind, FIGURE);
        assert_eq!(rc, BAD_ARG, "kind {kind}");
        assert!(v.is_none());
        assert!(last_error().contains("kind"), "{}", last_error());
    }
}

#[test]
fn null_and_invalid_arguments_are_bad_arguments() {
    let mut out = 0x10 as *mut c_char;
    let rc = unsafe { (api().latex_render)(PGFPLOTS, ptr::null(), &mut out) };
    assert_eq!(rc, BAD_ARG);
    assert!(out.is_null());

    let c = CString::new(FIGURE).unwrap();
    let rc = unsafe { (api().latex_render)(PGFPLOTS, c.as_ptr(), ptr::null_mut()) };
    assert_eq!(rc, BAD_ARG);

    // Invalid UTF-8 never reaches the renderer.
    let bad = CString::new(vec![b'{', 0xff, b'}']).unwrap();
    let mut out = 0x10 as *mut c_char;
    let rc = unsafe { (api().latex_render)(PGFPLOTS, bad.as_ptr(), &mut out) };
    assert_eq!(rc, BAD_ARG);
    assert!(out.is_null());
}
