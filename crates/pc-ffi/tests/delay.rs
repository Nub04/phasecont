//! Delay models through the table from `pc_api_v1()` (SPEC section 8s).

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

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

// ---- Entries that refuse a delay model, and the run (SPEC section 8s, tests 2 to 4) ----

const DELAY_MODEL: &str = "the model has delay terms, and this entry is for ODE models";
const BAD_ARG: i32 = 2;
const BAD_PROBLEM: i32 = 6;

const OPTIONS: &str = r#""options":{"hmax":0.05,"bounds":[1.2,2.0],"direction":1}"#;
const ROOTS: &str = r#""roots":{"n":30,"re_min":-0.5}"#;

fn compiled(text: &str) -> Id {
    let m = parse(text);
    assert_eq!(compile(&m, r#"{"evaluator":"jit"}"#), 0, "{}", last_error());
    m
}

/// A good problem with `extra` spliced in after the options.
fn problem(extra: &str) -> String {
    format!(r#"{{"type":"EP","free":"r",{ROOTS},{OPTIONS}{extra}}}"#)
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

fn start(m: &Id, json: &str) -> Id {
    let (rc, run) = start_with(m, json, &[1.0], &[1.3, 1.0]);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(run, 0);
    Id(run)
}

fn poll_raw(run: u64) -> (i32, Option<Value>) {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().run_poll)(run, &mut s) };
    (
        rc,
        take_string(s).map(|t| serde_json::from_str(&t).unwrap()),
    )
}

/// Polls every 5 ms until the state is not "running".
fn wait(run: &Id) -> Value {
    let t0 = Instant::now();
    loop {
        let (rc, v) = poll_raw(run.0);
        assert_eq!(rc, 0, "{}", last_error());
        let v = v.unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "run did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Returns (code, rows, values).
fn copy(run: u64, branch: u32, what: u32, count: usize, stride: usize) -> (i32, usize, Vec<f64>) {
    let mut buf = vec![f64::NAN; count.max(1) * stride];
    let mut rows = 99usize;
    let rc = unsafe {
        (api().run_copy)(
            run,
            branch,
            what,
            0,
            count,
            buf.as_mut_ptr(),
            buf.len(),
            &mut rows,
        )
    };
    buf.truncate(rows.min(count) * stride);
    (rc, rows, buf)
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn entries_for_ode_models_refuse_a_delay_handle() {
    let m = compiled(HUTCH);
    let x = [1.0];
    let p = [1.3, 1.0];
    let mut out = [0.0];
    let rc = unsafe { (api().model_rhs)(m.0, x.as_ptr(), 1, p.as_ptr(), 2, out.as_mut_ptr(), 1) };
    assert_eq!(rc, BAD_ARG);
    assert_eq!(last_error(), DELAY_MODEL);

    let rc =
        unsafe { (api().model_rhs_t)(m.0, 0.0, x.as_ptr(), 1, p.as_ptr(), 2, out.as_mut_ptr(), 1) };
    assert_eq!(rc, BAD_ARG);
    assert_eq!(last_error(), DELAY_MODEL);

    let opts = CString::new("{}").unwrap();
    let mut u = [0.0];
    let mut eig = [0.0; 2];
    let rc = unsafe {
        (api().model_equilibrium)(
            m.0,
            x.as_ptr(),
            1,
            p.as_ptr(),
            2,
            opts.as_ptr(),
            u.as_mut_ptr(),
            1,
            eig.as_mut_ptr(),
            2,
        )
    };
    assert_eq!(rc, BAD_ARG);
    assert_eq!(last_error(), DELAY_MODEL);

    let job = CString::new(
        r#"{"backend":"cpu","precision":"double","t0":0,"t1":1,"method":{"type":"rk4","dt":0.01},"output":"final","bounds":null,"max_steps":1000,"params":"shared"}"#,
    )
    .unwrap();
    let mut ens = 0u64;
    let rc =
        unsafe { (api().ens_start)(m.0, job.as_ptr(), x.as_ptr(), 1, p.as_ptr(), 2, &mut ens) };
    assert_eq!(rc, BAD_ARG);
    assert_eq!(ens, 0);
    assert_eq!(last_error(), DELAY_MODEL);
}

#[test]
fn emit_c_on_a_delay_handle_is_an_error() {
    let m = parse(HUTCH);
    let dir = std::env::temp_dir().join(format!("pc-ffi-delay-emit-{}", std::process::id()));
    let opts = format!(
        r#"{{"evaluator":"jit","emit_c":{:?}}}"#,
        dir.to_str().unwrap()
    );
    let rc = compile(&m, &opts);
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(rc, 0);
    assert!(last_error().contains(DELAY_MODEL), "{}", last_error());
}

#[test]
fn every_run_type_except_ep_is_refused_on_a_delay_handle() {
    let m = compiled(HUTCH);
    for ty in ["LP", "H", "LC", "LPC", "PD", "NS", "HOM", "IVP"] {
        let json = format!(r#"{{"type":"{ty}","free":"r"}}"#);
        let (rc, run) = start_with(&m, &json, &[1.0], &[1.3, 1.0]);
        assert_eq!(rc, BAD_ARG, "{ty}");
        assert_eq!(run, 0, "{ty}");
        assert_eq!(last_error(), DELAY_MODEL, "{ty}");
    }
}

#[test]
fn an_ep_run_with_roots_on_an_ode_model_is_an_error() {
    let m = compiled(ODE);
    let json = r#"{"type":"EP","free":"a","roots":{"n":30,"re_min":-0.5}}"#;
    let (rc, run) = start_with(&m, json, &[3.0], &[2.0]);
    assert_eq!(rc, BAD_PROBLEM);
    assert_eq!(run, 0);
    assert!(last_error().contains("roots"), "{}", last_error());
}

#[test]
fn a_hutchinson_run_equals_the_engine_through_the_run_entries() {
    use pc_engine::{
        EpSetup, Evaluator, Options, RootOptions, StopReason, build_dde, dde_continue_equilibrium,
    };
    use pc_ode::{MapLoader, SourceMap, load};
    use std::sync::atomic::AtomicBool;

    let mut map = SourceMap::new();
    let top = map.add("t.ode", HUTCH);
    let l = load(top, &mut map, &mut MapLoader::new());
    let mut dm = build_dde(&l.expanded, &l.info, Evaluator::Jit).unwrap();
    let setup = EpSetup::new(0, vec![1.3, 1.0], vec![1.0]);
    let ro = RootOptions {
        n: 30,
        re_min: -0.5,
    };
    let o = Options {
        hmax: 0.05,
        bounds: (1.2, 2.0),
        direction: 1.0,
        ..Options::default()
    };
    let want = dde_continue_equilibrium(&mut dm, &setup, &ro, &o, &AtomicBool::new(false))
        .expect("the engine run");

    let m = compiled(HUTCH);
    let run = start(&m, &problem(""));
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    assert_eq!(st["kind"], "EP");
    assert_eq!(st["n"], 1);
    assert_eq!(st["stride"], 3);
    let branches = st["branches"].as_array().unwrap();
    assert_eq!(branches.len(), 1, "{st}");
    let jb = &branches[0];
    assert_eq!(jb["points"], want.points.len());
    assert_eq!(jb["stop"], "Bounds");
    assert!(matches!(want.stop, StopReason::Bounds));
    assert_eq!(jb["warnings"], serde_json::json!([]));

    let (rc, rows, vals) = copy(run.0, 0, 0, want.points.len(), 3);
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(rows, want.points.len());
    for (r, p) in want.points.iter().enumerate() {
        let row = [p.alpha, p.x[0], p.unstable as f64];
        assert!(same_bits(&vals[r * 3..(r + 1) * 3], &row), "row {r}");
    }

    // One H at r = pi/2 with omega = pi/2, as in section 8e test 1.
    let js = jb["specials"].as_array().unwrap();
    assert_eq!(js.len(), 1, "{jb}");
    assert_eq!(want.specials.len(), 1);
    let (j, s) = (&js[0], &want.specials[0]);
    assert_eq!(j["label"], "H");
    assert_eq!(j["index"], s.between[1]);
    assert_eq!(j["refined"], false);
    assert!((j["omega"].as_f64().unwrap() - std::f64::consts::FRAC_PI_2).abs() <= 1e-9);
    assert_eq!(
        j["omega"].as_f64().unwrap().to_bits(),
        s.omega.unwrap().to_bits()
    );
    let vals_of = |key: &str| j["coefficients"][key].as_f64();
    assert_eq!(vals_of("unstable_before"), Some(0.0), "{j}");
    assert_eq!(vals_of("unstable_after"), Some(2.0), "{j}");
    assert_eq!((s.unstable_before, s.unstable_after), (0, 2));
    let (rc, rows, vals) = copy(run.0, 0, 1, 1, 3);
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(rows, 1);
    assert!(same_bits(
        &vals,
        &[s.alpha, s.x[0], s.unstable_after as f64]
    ));
}

#[test]
fn bad_delay_runs_name_the_key() {
    let m = compiled(HUTCH);
    let run_with = |json: &str, u0: &[f64], p: &[f64]| {
        let (rc, run) = start_with(&m, json, u0, p);
        assert_eq!(run, 0, "{json}");
        (rc, last_error())
    };
    let ok_u = [1.0];
    let ok_p = [1.3, 1.0];
    let cases: [(String, &str); 9] = [
        (format!(r#"{{"type":"EP","free":"r",{OPTIONS}}}"#), "roots"),
        (
            format!(r#"{{"type":"EP","free":"r","roots":{{"n":0,"re_min":-0.5}},{OPTIONS}}}"#),
            "roots.n",
        ),
        (
            format!(r#"{{"type":"EP","free":"r","roots":{{"n":30,"re_min":0}},{OPTIONS}}}"#),
            "roots.re_min",
        ),
        (
            format!(r#"{{"type":"EP","free":"r","roots":{{"n":30,"re_min":0.5}},{OPTIONS}}}"#),
            "roots.re_min",
        ),
        (
            format!(
                r#"{{"type":"EP","free":"r","roots":{{"n":30,"re_min":-0.5,"m":3}},{OPTIONS}}}"#
            ),
            "roots",
        ),
        (
            format!(
                r#"{{"type":"EP","free":"r",{ROOTS},"options":{{"hmax":0.05,"bounds":[1.2,2.0],"detect":{{"h":true}}}}}}"#
            ),
            "detect",
        ),
        (
            format!(
                r#"{{"type":"EP","free":"r",{ROOTS},"options":{{"hmax":0.05,"bounds":[1.2,2.0],"mesh_adapt":2}}}}"#
            ),
            "mesh_adapt",
        ),
        (problem(r#","switch":"BP1""#), "switch"),
        (problem(r#","start_is_special":true"#), "start_is_special"),
    ];
    for (json, key) in &cases {
        let (rc, msg) = run_with(json, &ok_u, &ok_p);
        assert_eq!(rc, BAD_PROBLEM, "{json}: {msg}");
        assert!(msg.contains(key), "{json}: {msg}");
    }
    // A buffer of the wrong length.
    let (rc, msg) = run_with(&problem(""), &ok_u, &[1.3]);
    assert_eq!(rc, BAD_ARG, "{msg}");
    assert!(msg.contains("params"), "{msg}");
    let (rc, msg) = run_with(&problem(""), &ok_u, &[1.3, 1.0, 0.0]);
    assert_eq!(rc, BAD_ARG, "{msg}");
    assert!(msg.contains("params"), "{msg}");
    let (rc, msg) = run_with(&problem(""), &[1.0, 2.0], &ok_p);
    assert_eq!(rc, BAD_ARG, "{msg}");
    assert!(msg.contains("u0"), "{msg}");
    // A hidden delay slot is not a parameter the caller can free.
    let hidden = problem("").replace(r#""free":"r""#, r#""free":"delay(x,tau)""#);
    let (rc, msg) = run_with(&hidden, &ok_u, &ok_p);
    assert_eq!(rc, BAD_PROBLEM, "{msg}");
    assert!(msg.contains("free"), "{msg}");
}

#[test]
fn a_run_cancelled_right_after_the_start_keeps_no_points() {
    let m = compiled(HUTCH);
    let json = format!(
        r#"{{"type":"EP","free":"r",{ROOTS},"options":{{"h0":0.0001,"hmax":0.0001,"max_steps":100000,"bounds":[0.5,1000]}}}}"#
    );
    let run = start(&m, &json);
    assert_eq!((api().run_cancel)(run.0), 0, "{}", last_error());
    let st = wait(&run);
    assert_eq!(st["state"], "cancelled", "{st}");
    let total: u64 = st["branches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["points"].as_u64().unwrap())
        .sum();
    assert_eq!(total, 0, "{st}");
}
