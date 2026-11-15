//! Map models through the table from `pc_api_v1()` (SPEC section 8t, tests 4 and 5).

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

/// The refusal text of SPEC 8t.
const MAP_MODEL: &str = "the model is a map (`x(t+1)=...` or `@ meth=discrete`), and PhaseCont \
     does not iterate maps or find their fixed points yet; evaluating the right-hand side works";
const UNSUPPORTED_MODEL: i32 = 13;

/// The logistic map, as in `crates/pc-ode/tests/corpus/logistic_map.ode`.
const LOGISTIC_MAP: &str = include_str!("../../pc-ode/tests/corpus/logistic_map.ode");
/// The same right-hand side as an ODE file, with and without `@ meth=discrete`.
const LOGISTIC_ODE: &str = "par r=2\ninit x=0.2\nx'=r*x*(1-x)\ndone\n";
const LOGISTIC_DISCRETE: &str = "par r=2\ninit x=0.2\nx'=r*x*(1-x)\n@ meth=discrete\ndone\n";
/// A map that reads `t`, for `model_rhs_t`.
const TIMED_ODE: &str = "par r=2\nx'=r*x*(1-x)+t\ndone\n";
const TIMED_DISCRETE: &str = "par r=2\nx'=r*x*(1-x)+t\n@ meth=discrete\ndone\n";
const HUTCH: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

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

/// A scratch folder that is removed at the end.
struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Temp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Temp(std::env::temp_dir().join(format!("pc-ffi-map-{tag}-{}-{nanos}", std::process::id())))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

fn parse(text: &str) -> Id {
    let mut id = 0u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc =
        unsafe { (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, &mut diag) };
    let _ = take_string(diag);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(id, 0);
    Id(id)
}

fn compile(m: &Id, opts: &str) -> i32 {
    let c = CString::new(opts).unwrap();
    unsafe { (api().model_compile)(m.0, c.as_ptr()) }
}

fn compiled(text: &str) -> Id {
    let m = parse(text);
    assert_eq!(compile(&m, r#"{"evaluator":"jit"}"#), 0, "{}", last_error());
    m
}

fn info(m: &Id) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { (api().model_info)(m.0, &mut s) }, 0);
    serde_json::from_str(&take_string(s).unwrap()).unwrap()
}

/// Returns (code, run id). The run id is 0 unless the call succeeds.
fn start_with(m: &Id, json: &str, u0: &[f64], p: &[f64]) -> (i32, u64) {
    let c = CString::new(json).unwrap();
    let mut run = 0u64;
    let rc = unsafe {
        (api().run_start)(
            m.0,
            c.as_ptr(),
            u0.as_ptr(),
            u0.len(),
            p.as_ptr(),
            p.len(),
            &mut run,
        )
    };
    (rc, run)
}

fn equilibrium(m: &Id, x: &[f64], p: &[f64]) -> i32 {
    let opts = CString::new("{}").unwrap();
    let mut u = vec![0.0; x.len()];
    let mut eig = vec![0.0; 2 * x.len()];
    unsafe {
        (api().model_equilibrium)(
            m.0,
            x.as_ptr(),
            x.len(),
            p.as_ptr(),
            p.len(),
            opts.as_ptr(),
            u.as_mut_ptr(),
            u.len(),
            eig.as_mut_ptr(),
            eig.len(),
        )
    }
}

fn rhs(m: &Id, x: f64, p: f64) -> f64 {
    let mut out = [f64::NAN];
    let rc = unsafe { (api().model_rhs)(m.0, &x, 1, &p, 1, out.as_mut_ptr(), 1) };
    assert_eq!(rc, 0, "{}", last_error());
    out[0]
}

fn rhs_t(m: &Id, t: f64, x: f64, p: f64) -> f64 {
    let mut out = [f64::NAN];
    let rc = unsafe { (api().model_rhs_t)(m.0, t, &x, 1, &p, 1, out.as_mut_ptr(), 1) };
    assert_eq!(rc, 0, "{}", last_error());
    out[0]
}

/// Compiles with `emit_c` into `dir` and returns the text of the C file.
fn emitted(m: &Id, dir: &Path) -> String {
    let opts = serde_json::json!({ "evaluator": "jit", "emit_c": dir.to_str().unwrap() });
    assert_eq!(compile(m, &opts.to_string()), 0, "{}", last_error());
    let path = info(m)["c_file"].as_str().expect("c_file").to_owned();
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn model_info_reports_map_for_both_kinds_of_map() {
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let v = info(&parse(text));
        assert_eq!(v["map"], true, "{text}: {v}");
        assert_eq!(v["delay"], false, "{text}: {v}");
    }
    for text in [LOGISTIC_ODE, HUTCH] {
        let v = info(&parse(text));
        assert_eq!(v["map"], false, "{text}: {v}");
    }
}

#[test]
fn every_run_type_is_refused_on_a_map_handle() {
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = compiled(text);
        let n_p = info(&m)["n_p"].as_u64().unwrap() as usize;
        let p = vec![2.0; n_p];
        for ty in ["EP", "LP", "H", "LC", "LPC", "PD", "NS", "HOM", "IVP"] {
            let json = format!(r#"{{"type":"{ty}","free":"r"}}"#);
            let (rc, run) = start_with(&m, &json, &[0.5], &p);
            assert_eq!(rc, UNSUPPORTED_MODEL, "{ty}: {}", last_error());
            assert_eq!(run, 0, "{ty}");
            assert_eq!(last_error(), MAP_MODEL, "{ty}");
        }
        // A problem that is not JSON gets the same error, since the map is refused first.
        let (rc, run) = start_with(&m, "not json", &[0.5], &p);
        assert_eq!(rc, UNSUPPORTED_MODEL);
        assert_eq!(run, 0);
        assert_eq!(last_error(), MAP_MODEL);
    }
}

#[test]
fn a_map_handle_that_was_not_compiled_gets_the_map_error() {
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = parse(text);
        let (rc, run) = start_with(&m, r#"{"type":"EP","free":"r"}"#, &[0.5], &[2.0]);
        assert_eq!(rc, UNSUPPORTED_MODEL, "{}", last_error());
        assert_eq!(run, 0);
        assert_eq!(last_error(), MAP_MODEL);
    }
}

#[test]
fn model_equilibrium_is_refused_on_a_map_handle() {
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = compiled(text);
        assert_eq!(equilibrium(&m, &[0.5], &[2.0]), UNSUPPORTED_MODEL);
        assert_eq!(last_error(), MAP_MODEL);
    }
}

#[test]
fn an_ode_handle_still_finds_its_equilibrium() {
    let m = compiled(LOGISTIC_ODE);
    assert_eq!(equilibrium(&m, &[0.9], &[2.0]), 0, "{}", last_error());
}

#[test]
fn meth_discrete_does_not_change_the_right_hand_side() {
    let (ode, map) = (compiled(LOGISTIC_ODE), compiled(LOGISTIC_DISCRETE));
    for &(x, r) in &[(0.2, 2.0), (0.7, 3.7), (-1.5, 0.25), (1e-300, 1e300)] {
        let (a, b) = (rhs(&ode, x, r), rhs(&map, x, r));
        assert_eq!(a.to_bits(), b.to_bits(), "x={x} r={r}: {a} against {b}");
    }
    let (ode, map) = (compiled(TIMED_ODE), compiled(TIMED_DISCRETE));
    for &(t, x, r) in &[(0.0, 0.2, 2.0), (3.0, 0.7, 3.7), (-2.5, 1.5, 0.25)] {
        let (a, b) = (rhs_t(&ode, t, x, r), rhs_t(&map, t, x, r));
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "t={t} x={x} r={r}: {a} against {b}"
        );
    }
}

#[test]
fn meth_discrete_does_not_change_the_emitted_c() {
    let (a, b) = (Temp::new("ode"), Temp::new("discrete"));
    let ode = emitted(&parse(LOGISTIC_ODE), &a.0);
    let map = emitted(&parse(LOGISTIC_DISCRETE), &b.0);
    assert!(!ode.is_empty());
    assert_eq!(ode, map);
}
