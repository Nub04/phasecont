//! Delay models through the table from `pc_api_v1()` (SPEC section 8s).

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const HUTCH: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";
const ODE: &str = "par a=2\ninit x=3\nx'=-a*x\ndone\n";

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

/// Frees the handle when the test ends.
struct Id(u64);

impl Drop for Id {
    fn drop(&mut self) {
        if self.0 != 0 {
            assert_eq!((api().free_handle)(self.0), 0);
        }
    }
}

fn take_string(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(p) };
    Some(s)
}

/// Returns (code, model id, diagnostics JSON).
fn try_parse(text: &str) -> (i32, Id, Option<Value>) {
    let mut id = 0u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc =
        unsafe { (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, &mut diag) };
    let v = take_string(diag).map(|s| serde_json::from_str(&s).unwrap());
    (rc, Id(if rc == 0 { id } else { 0 }), v)
}

fn parse(text: &str) -> Id {
    let (rc, id, _) = try_parse(text);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(id.0, 0);
    id
}

fn compile(m: &Id, opts: &str) -> i32 {
    let c = CString::new(opts).unwrap();
    unsafe { (api().model_compile)(m.0, c.as_ptr()) }
}

fn info(m: &Id) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { (api().model_info)(m.0, &mut s) }, 0);
    serde_json::from_str(&take_string(s).unwrap()).unwrap()
}

#[test]
fn a_delay_model_loads_compiles_and_reports_its_user_parameters() {
    let m = parse(HUTCH);
    assert_eq!(compile(&m, r#"{"evaluator":"jit"}"#), 0, "{}", last_error());
    let v = info(&m);
    assert_eq!(v["delay"], true, "{v}");
    assert_eq!(v["n"], 1);
    assert_eq!(v["n_p"], 2);
    assert_eq!(v["variables"], serde_json::json!(["x"]));
    assert_eq!(v["parameters"], serde_json::json!(["r", "tau"]));
    assert_eq!(v["values"], serde_json::json!([1.3, 1.0]));
    assert_eq!(v["evaluator"], "jit");
}

#[test]
fn an_ode_model_reports_delay_false() {
    let m = parse(ODE);
    let v = info(&m);
    assert_eq!(v["delay"], false, "{v}");
    assert_eq!(v["parameters"], serde_json::json!(["a"]));
    assert_eq!(v["values"], serde_json::json!([2.0]));
    assert_eq!(v["n_p"], 1);
}

#[test]
fn a_delay_call_of_the_wrong_form_is_an_error_that_names_the_call() {
    let (rc, id, _) = try_parse("par tau=1\nx'=-delay(x+1,tau)\ndone\n");
    assert_ne!(rc, 0);
    assert_eq!(id.0, 0);
    let msg = last_error();
    assert!(msg.contains("delay"), "{msg}");
}

#[test]
fn a_syntax_error_in_a_delay_model_is_reported_as_the_syntax_error() {
    let (rc, _, diag) = try_parse("par tau=1\nx'=-delay(x,tau\ndone\n");
    assert_ne!(rc, 0);
    let msg = last_error();
    assert!(!msg.contains("delay terms"), "{msg}");
    let d = diag.unwrap();
    assert!(d["summary"]["errors"].as_u64().unwrap() > 0, "{d}");
}

#[test]
fn aot_on_a_delay_model_is_an_error() {
    let m = parse(HUTCH);
    let rc = compile(&m, r#"{"evaluator":"aot","aot_lib":"no-such-library.so"}"#);
    assert_ne!(rc, 0);
    assert!(last_error().contains("no AOT path"), "{}", last_error());
}
