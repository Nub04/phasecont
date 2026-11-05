//! `import` through the table from `pc_api_v1()`: SBML and MATLAB text become `.ode` text that
//! `model_from_ode` accepts. The models are our own.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};
use serde_json::{Value, json};

const BAD_ARG: i32 = 2;
const MODEL: i32 = 5;

const SBML: u32 = 1;
const MATLAB: u32 = 2;

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
fn import(format: u32, bytes: &[u8], opts: Option<&str>) -> (i32, Option<Value>) {
    let c = opts.map(|s| CString::new(s).unwrap());
    let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    // A pre-set pointer proves the slot is cleared.
    let mut out = 0x10 as *mut c_char;
    let rc = unsafe { (api().import)(format, bytes.as_ptr().cast(), bytes.len(), p, &mut out) };
    let v = take_string(out).map(|s| serde_json::from_str(&s).unwrap());
    (rc, v)
}

fn matlab(text: &str, opts: Value) -> (i32, Option<Value>) {
    import(MATLAB, text.as_bytes(), Some(&opts.to_string()))
}

/// Feeds `.ode` text to `model_from_ode`; returns the code.
fn parses(ode: &str) -> i32 {
    let mut id = 0u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().model_from_ode)(ode.as_ptr().cast(), ode.len(), &mut id, &mut diag) };
    take_string(diag);
    if rc == 0 {
        assert_eq!((api().free_handle)(id), 0);
    }
    rc
}

const DECAY: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
    <sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\
    <model id=\"m\"><listOfCompartments>\
    <compartment id=\"C\" size=\"2\" constant=\"true\"/></listOfCompartments>\
    <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"4\" \
    hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/></listOfSpecies>\
    <listOfParameters><parameter id=\"k\" value=\"0.5\" constant=\"true\"/></listOfParameters>\
    <listOfReactions><reaction id=\"R\" reversible=\"false\" fast=\"false\">\
    <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\" constant=\"true\"/>\
    </listOfReactants><kineticLaw>\
    <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><times/><ci>k</ci><ci>S</ci></apply></math>\
    </kineticLaw></reaction></listOfReactions></model></sbml>";

#[test]
fn matlab_handle_gives_ode_text_the_parser_accepts() {
    let opts = json!({
        "params": [{"name": "w", "value": 2.0}, {"name": "c", "value": 0.1}],
        "state_names": ["u", "v"],
        "init": [1.0, 0.0],
        "title": "oscillator"
    });
    let (rc, v) = matlab("@(t,x,p) [x(2); -p(1)^2*x(1) - p(2)*x(2)]", opts);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    assert_eq!(v["ok"], true);
    let ode = v["ode"].as_str().unwrap();
    assert!(ode.contains("u'"), "{ode}");
    assert!(v["warnings"].is_array());
    assert_eq!(parses(ode), 0, "{ode}");
}

#[test]
fn matlab_captures_and_single_element_lists_follow_jsonencode() {
    // jsonencode writes a one-element array as the bare item.
    let opts = json!({
        "captures": {"name": "a", "value": 1.5},
        "state_names": "x",
        "init": 2.0
    });
    let (rc, v) = matlab("@(t,x) -a*x(1)", opts);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    assert_eq!(parses(v["ode"].as_str().unwrap()), 0);
}

#[test]
fn sbml_decay_gives_ode_text_the_parser_accepts() {
    let (rc, v) = import(SBML, DECAY.as_bytes(), None);
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    assert_eq!(v["ok"], true);
    let ode = v["ode"].as_str().unwrap();
    assert_eq!(parses(ode), 0, "{ode}");
    // Empty options are the same as none.
    let (rc, v2) = import(SBML, DECAY.as_bytes(), Some("{}"));
    assert_eq!(rc, 0);
    assert_eq!(v2.unwrap()["ode"], v["ode"]);
}

#[test]
fn refused_model_returns_model_code_and_the_error_object() {
    let (rc, v) = matlab(
        "@(t,x) -k*x(1)",
        json!({"state_names": ["x"], "init": [1.0]}),
    );
    assert_eq!(rc, MODEL);
    let v = v.expect("the failure JSON is returned");
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["id"], "phasecont:import:undeclared");
    assert_eq!(v["error"]["kind"], "undeclared");
    assert!(v["error"]["message"].as_str().unwrap().contains('k'));
    assert!(last_error().starts_with(v["error"]["message"].as_str().unwrap()));
}

#[test]
fn missing_initial_value_is_not_defaulted() {
    let (rc, v) = matlab("@(t,x) -x(1)", json!({"state_names": ["x"]}));
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["error"]["kind"], "missingValue");
}

#[test]
fn refusal_with_a_position_carries_line_and_col() {
    let text = "function dydt = f(t, y)\n\
                dydt = zeros(1, 1);\n\
                for i = 1:3\n\
                dydt(1) = dydt(1) - y(1);\n\
                end\n\
                end\n";
    let (rc, v) = matlab(text, json!({"state_names": ["a"], "init": [1.0]}));
    assert_eq!(rc, MODEL);
    let e = &v.unwrap()["error"];
    assert_eq!(e["kind"], "unsupported");
    assert!(e["line"].as_u64().is_some(), "{e}");
    assert!(e["col"].as_u64().is_some(), "{e}");
}

#[test]
fn sbml_event_is_refused_by_name() {
    let events = DECAY.replace(
        "</listOfReactions>",
        "</listOfReactions><listOfEvents><event id=\"e\" useValuesFromTriggerTime=\"true\">\
         <trigger initialValue=\"true\" persistent=\"true\">\
         <math xmlns=\"http://www.w3.org/1998/Math/MathML\"><true/></math></trigger>\
         </event></listOfEvents>",
    );
    let (rc, v) = import(SBML, events.as_bytes(), None);
    assert_eq!(rc, MODEL);
    let e = &v.unwrap()["error"];
    assert_eq!(e["id"], "phasecont:import:unsupported");
    assert!(e["message"].as_str().unwrap().contains("event"), "{e}");
}

#[test]
fn malformed_xml_is_an_xml_error() {
    let (rc, v) = import(SBML, b"<sbml><model>", None);
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["error"]["id"], "phasecont:import:xml");
}

#[test]
fn bad_arguments_are_bad_arg_and_leave_the_slot_null() {
    // Unknown format.
    let (rc, v) = import(9, b"x", None);
    assert_eq!(rc, BAD_ARG);
    assert!(v.is_none());
    assert!(last_error().contains("format"), "{}", last_error());

    // Unknown option key.
    let (rc, v) = matlab("@(t,x) -x(1)", json!({"nope": 1}));
    assert_eq!(rc, BAD_ARG);
    assert!(v.is_none());
    assert!(last_error().contains("nope"), "{}", last_error());

    // SBML takes no options.
    let (rc, v) = import(SBML, DECAY.as_bytes(), Some("{\"init\":[1]}"));
    assert_eq!(rc, BAD_ARG);
    assert!(v.is_none());

    // Options that are not JSON, or not an object.
    assert_eq!(matlab_raw("@(t,x) -x(1)", "not json").0, BAD_ARG);
    assert_eq!(matlab_raw("@(t,x) -x(1)", "[1,2]").0, BAD_ARG);

    // A value that is not a finite number (jsonencode writes NaN as null).
    let (rc, _) = matlab(
        "@(t,x) -x(1)",
        json!({"state_names": ["x"], "init": [null]}),
    );
    assert_eq!(rc, BAD_ARG);

    // MATLAB text that is not UTF-8.
    let (rc, v) = import(MATLAB, &[0x40, 0x28, 0xff, 0xfe], None);
    assert_eq!(rc, BAD_ARG);
    assert!(v.is_none());
}

fn matlab_raw(text: &str, opts: &str) -> (i32, Option<Value>) {
    import(MATLAB, text.as_bytes(), Some(opts))
}

#[test]
fn null_pointers_are_rejected() {
    let t = api();
    // NULL result slot.
    let rc = unsafe {
        (t.import)(
            MATLAB,
            b"x".as_ptr().cast(),
            1,
            ptr::null(),
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, BAD_ARG);
    // NULL bytes with a length.
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { (t.import)(MATLAB, ptr::null(), 4, ptr::null(), &mut out) };
    assert_eq!(rc, BAD_ARG);
    assert!(out.is_null());
}

#[test]
fn empty_input_is_an_import_error_not_a_crash() {
    let (rc, v) = import(SBML, b"", None);
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["ok"], false);
    let (rc, v) = import(MATLAB, b"", None);
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["ok"], false);
}

#[test]
fn oversize_input_is_bad_arg() {
    let big = vec![b' '; 17 * 1024 * 1024 + 1];
    let (rc, v) = import(SBML, &big, None);
    assert_eq!(rc, BAD_ARG);
    assert!(v.is_none());
}

#[test]
fn deep_nesting_is_an_error_and_no_panic_crosses() {
    let deep = format!("@(t,x) {}x(1){}", "(".repeat(20_000), ")".repeat(20_000));
    let (rc, v) = matlab(&deep, json!({"state_names": ["x"], "init": [1.0]}));
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["ok"], false);

    let xml = format!("{}{}", "<a>".repeat(50_000), "</a>".repeat(50_000));
    let (rc, v) = import(SBML, xml.as_bytes(), None);
    assert_eq!(rc, MODEL);
    assert_eq!(v.unwrap()["ok"], false);
}

#[test]
fn every_failure_kind_has_an_import_id() {
    let cases: [(&str, Value, &str); 3] = [
        (
            "@(t,x) mod(x(1), 2)",
            json!({"state_names": ["x"], "init": [1.0]}),
            "unsupported",
        ),
        (
            "@(t,x) -(x(1)",
            json!({"state_names": ["a"], "init": [1.0]}),
            "syntax",
        ),
        (
            "@(t,x) [x(2); -x(1)]",
            json!({"state_names": ["a"], "init": [1.0, 0.0]}),
            "mismatch",
        ),
    ];
    for (text, opts, kind) in cases {
        let (rc, v) = matlab(text, opts);
        assert_eq!(rc, MODEL, "{text}");
        let e = &v.unwrap()["error"];
        assert_eq!(e["kind"], kind, "{text}: {e}");
        assert_eq!(e["id"], format!("phasecont:import:{kind}"));
    }
}
