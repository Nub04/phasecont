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
    for ty in ["LP", "LC", "LPC", "PD", "NS", "HOM", "IVP"] {
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

// ---- Slice 2: the Hopf curve of a delay equilibrium (SPEC section 8s, slice 2) ----

const WRIGHT3: &str = "par a=1.5, b=0\nu'=-a*delay(u,1)*(1+u)+b*u^3\ndone\n";
const COUPLED: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)\n\
    y'=-d*x-b*delay(y,1.5)\n\
    done\n";
/// The numbers of `dde_hopf_lyap.rs`: l1 at b = 0 and the slope in b of the Wright cubic.
const L1_AT_ZERO: f64 = -0.214_130_922_441_773_92;
const B_STAR: f64 = 1.184_419_463_185_973_2;
const A_STAR: f64 = 1.454_526_952_022_688;
const OMEGA1_STAR: f64 = 1.516_027_941_313_437_7;
const OMEGA2_STAR: f64 = 1.130_821_068_412_625_6;
const GH_NORMALIZATION: &str = "|q|=1, p Delta'(i omega) q=1";

fn wright_slope() -> f64 {
    6.0 / (std::f64::consts::PI * (1.0 + std::f64::consts::PI * std::f64::consts::PI / 4.0))
}

/// The options of the engine Hopf tests, as JSON.
fn hopf_options(bounds: &str, direction: i32) -> String {
    format!(
        r#""options":{{"hmax":0.05,"bounds":{bounds},"direction":{direction},"tol_x":1e-12,"tol_f":1e-12,"max_newton":20}}"#
    )
}

/// A delay "H" problem. `extra` is spliced in after the options.
fn hopf_problem(free: &str, omega: f64, bounds: &str, extra: &str) -> String {
    format!(
        r#"{{"type":"H","free":{free},"omega":{omega:?},{}{extra}}}"#,
        hopf_options(bounds, 1)
    )
}

fn engine_options(bounds: (f64, f64)) -> pc_engine::Options {
    pc_engine::Options {
        hmax: 0.05,
        bounds,
        direction: 1.0,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        ..pc_engine::Options::default()
    }
}

fn engine_model(text: &str) -> pc_engine::DdeModel {
    use pc_ode::{MapLoader, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", text);
    let l = load(top, &mut map, &mut MapLoader::new());
    pc_engine::build_dde(&l.expanded, &l.info, pc_engine::Evaluator::Jit).unwrap()
}

/// The engine run the run entries must reproduce. `extras` None calls `continue_hopf`.
fn engine_hopf(
    text: &str,
    params: &[f64],
    x0: &[f64],
    omega: f64,
    bounds: (f64, f64),
    extras: Option<pc_engine::DdeHopfExtras>,
) -> pc_engine::DdeHopfBranch {
    use std::sync::atomic::AtomicBool;
    let mut dm = engine_model(text);
    let setup = pc_engine::DdeHopfSetup {
        free: [0, 1],
        params: params.to_vec(),
        x0: x0.to_vec(),
        omega,
    };
    let o = engine_options(bounds);
    let cancel = AtomicBool::new(false);
    match extras {
        None => pc_engine::dde_continue_hopf(&mut dm, &setup, &o, &cancel),
        Some(e) => pc_engine::dde_continue_hopf_with(&mut dm, &setup, &o, &cancel, &e),
    }
    .expect("the engine run")
}

/// The row of a point: `[alpha2, x, alpha1, kappa, l1, hh]`.
fn hopf_row(p: &pc_engine::DdeHopfPoint) -> Vec<f64> {
    let mut r = vec![p.alpha[1]];
    r.extend_from_slice(&p.x);
    r.extend([
        p.alpha[0],
        p.kappa,
        p.l1.unwrap_or(f64::NAN),
        p.hh.unwrap_or(f64::NAN),
    ]);
    r
}

fn hopf_special_row(s: &pc_engine::DdeHopfSpecial) -> Vec<f64> {
    let mut r = vec![s.alpha[1]];
    r.extend_from_slice(&s.x);
    r.extend([s.alpha[0], s.kappa, f64::NAN, f64::NAN]);
    r
}

/// Starts the run, waits for it, and checks it against the engine branch. Returns the poll
/// state, the rows and the special rows.
fn check_hopf_run(
    m: &Id,
    json: &str,
    u0: &[f64],
    p: &[f64],
    want: &pc_engine::DdeHopfBranch,
) -> (Value, Vec<Vec<f64>>) {
    let n = u0.len();
    let stride = n + 5;
    let (rc, run) = start_with(m, json, u0, p);
    assert_eq!(rc, 0, "{json}: {}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    assert_eq!(st["kind"], "H");
    assert_eq!(st["n"], n);
    assert_eq!(st["stride"], stride);
    assert_eq!(st["codim2_labels"], serde_json::json!(["GH", "HH"]));
    let jb = &st["branches"].as_array().unwrap()[0];
    assert_eq!(jb["points"], want.points.len(), "{st}");
    assert_eq!(jb["stop"], "Bounds");
    assert!(matches!(want.stop, pc_engine::StopReason::Bounds));
    assert_eq!(jb["warnings"], serde_json::json!(want.warnings));
    let (rc, rows, vals) = copy(run.0, 0, 0, want.points.len(), stride);
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(rows, want.points.len());
    for (r, p) in want.points.iter().enumerate() {
        assert!(
            same_bits(&vals[r * stride..(r + 1) * stride], &hopf_row(p)),
            "row {r}: {:?} vs {:?}",
            &vals[r * stride..(r + 1) * stride],
            hopf_row(p)
        );
    }
    // The specials agree with the engine's in label, index and parameters, and their rows
    // carry NaN in the two test columns.
    let js = jb["specials"].as_array().unwrap();
    assert_eq!(js.len(), want.specials.len(), "{jb}");
    let mut special_rows = Vec::new();
    for (k, s) in want.specials.iter().enumerate() {
        assert_eq!(js[k]["label"], s.label, "{jb}");
        assert_eq!(js[k]["index"], s.index, "{jb}");
        assert_eq!(js[k]["refined"], false);
        let (rc, rows, v) = copy(run.0, 0, 1 + k as u32, 1, stride);
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(rows, 1);
        assert!(same_bits(&v, &hopf_special_row(s)), "special {k}");
        special_rows.push(v);
    }
    (st, special_rows)
}

#[test]
fn a_hutchinson_hopf_run_equals_the_engine_through_the_run_entries() {
    let want = engine_hopf(HUTCH, &[1.57, 1.0], &[1.001], 1.57, (0.5, 3.0), None);
    assert!(want.specials.is_empty(), "{:?}", want.specials);
    let m = compiled(HUTCH);
    let json = hopf_problem(r#"["r","tau"]"#, 1.57, "[0.5,3.0]", "");
    let (_, _) = check_hopf_run(&m, &json, &[1.001], &[1.57, 1.0], &want);
    // l1 and hh are NaN in every row, and r tau = pi/2 on the whole curve.
    for p in &want.points {
        assert!(p.l1.is_none() && p.hh.is_none());
        let rtau = p.alpha[0] * p.alpha[1];
        assert!((rtau - std::f64::consts::FRAC_PI_2).abs() <= 1e-8, "{rtau}");
    }
}

#[test]
fn a_wright_hopf_run_labels_gh_with_the_lyapunov_option() {
    use pc_engine::DdeHopfExtras;
    let extras = DdeHopfExtras {
        lyapunov: true,
        hh: None,
        coefficients: false,
    };
    let want = engine_hopf(
        WRIGHT3,
        &[1.57, 0.0],
        &[0.001],
        1.57,
        (-1.0, 1.0),
        Some(extras),
    );
    let m = compiled(WRIGHT3);
    let json = hopf_problem(r#"["a","b"]"#, 1.57, "[-1.0,1.0]", r#","lyapunov":true"#);
    let (st, special_rows) = check_hopf_run(&m, &json, &[0.001], &[1.57, 0.0], &want);
    // l1 follows the closed form in every row.
    let (_, rows, vals) = copy_all(&st, &m, &json, &[0.001], &[1.57, 0.0], 6);
    assert_eq!(rows, want.points.len());
    for r in 0..rows {
        let (b, l1) = (vals[r * 6], vals[r * 6 + 4]);
        let target = L1_AT_ZERO + wright_slope() * b;
        assert!((l1 - target).abs() <= 1e-7, "b = {b}: l1 {l1} vs {target}");
    }
    // Exactly one special, GH, at the closed form b*.
    let b_star = -L1_AT_ZERO / wright_slope();
    let js = st["branches"][0]["specials"].as_array().unwrap();
    assert_eq!(js.len(), 1, "{st}");
    assert_eq!(js[0]["label"], "GH");
    assert!(
        (special_rows[0][0] - b_star).abs() <= 1e-6,
        "{:?}",
        special_rows[0]
    );
    let l2 = want.specials[0].l2.expect("l2 of the engine GH");
    assert_eq!(
        js[0]["coefficient"].as_f64().unwrap().to_bits(),
        l2.to_bits()
    );
    assert_eq!(js[0]["normalization"], GH_NORMALIZATION);
    // The Hopf locus of this equation is a = pi/2 at every b, with omega = pi/2 (the start omega
    // 1.57 is only a guess that the corrector moves).
    let omega = js[0]["omega"].as_f64().unwrap();
    assert!(
        (omega - std::f64::consts::FRAC_PI_2).abs() <= 1e-6,
        "{omega}"
    );
}

/// Runs the problem again and returns all rows of branch 0 (a second run of the same JSON).
fn copy_all(
    _st: &Value,
    m: &Id,
    json: &str,
    u0: &[f64],
    p: &[f64],
    stride: usize,
) -> (i32, usize, Vec<f64>) {
    let (rc, run) = start_with(m, json, u0, p);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    let count = st["branches"][0]["points"].as_u64().unwrap() as usize;
    copy(run.0, 0, 0, count, stride)
}

#[test]
fn a_wright_hopf_run_without_the_option_has_no_gh_and_nan_l1() {
    let want = engine_hopf(WRIGHT3, &[1.57, 0.0], &[0.001], 1.57, (-1.0, 1.0), None);
    assert!(want.specials.iter().all(|s| s.label != "GH"));
    let m = compiled(WRIGHT3);
    let off = hopf_problem(r#"["a","b"]"#, 1.57, "[-1.0,1.0]", r#","lyapunov":false"#);
    let left_out = hopf_problem(r#"["a","b"]"#, 1.57, "[-1.0,1.0]", "");
    for json in [&off, &left_out] {
        let (st, _) = check_hopf_run(&m, json, &[0.001], &[1.57, 0.0], &want);
        let js = st["branches"][0]["specials"].as_array().unwrap();
        assert!(js.iter().all(|j| j["label"] != "GH"), "{st}");
    }
}

#[test]
fn a_coupled_hopf_run_labels_hh_with_roots() {
    use pc_engine::{DdeHopfExtras, RootOptions};
    let ro = RootOptions {
        n: 30,
        re_min: -3.0,
    };
    let extras = DdeHopfExtras {
        lyapunov: false,
        hh: Some(ro),
        coefficients: false,
    };
    let params = [1.46, 0.8, 1.0, 0.1];
    let want = engine_hopf(
        COUPLED,
        &params,
        &[0.001, 0.001],
        1.54,
        (0.5, 1.4),
        Some(extras),
    );
    let m = compiled(COUPLED);
    let roots = r#","roots":{"n":30,"re_min":-3.0}"#;
    let json = hopf_problem(r#"["a","b"]"#, 1.54, "[0.5,1.4]", roots);
    let (st, special_rows) = check_hopf_run(&m, &json, &[0.001, 0.001], &params, &want);
    let js = st["branches"][0]["specials"].as_array().unwrap();
    assert_eq!(js.len(), 1, "{st}");
    let j = &js[0];
    assert_eq!(j["label"], "HH");
    // Row: [b, x1, x2, a, kappa, l1, hh].
    let row = &special_rows[0];
    assert!((row[0] - B_STAR).abs() <= 1e-6, "b = {}", row[0]);
    assert!((row[3] - A_STAR).abs() <= 1e-6, "a = {}", row[3]);
    let omega2 = j["coefficients"]["omega2"].as_f64().expect("omega2");
    assert!((omega2 - OMEGA2_STAR).abs() <= 1e-6, "{omega2}");
    let omega = j["omega"].as_f64().unwrap();
    assert!((omega - OMEGA1_STAR).abs() <= 1e-6, "{omega}");
    assert_eq!(j["normalization"], Value::Null);
    // Without roots there is no HH.
    let plain = engine_hopf(COUPLED, &params, &[0.001, 0.001], 1.54, (0.5, 1.4), None);
    let no_roots = hopf_problem(r#"["a","b"]"#, 1.54, "[0.5,1.4]", "");
    let (st, _) = check_hopf_run(&m, &no_roots, &[0.001, 0.001], &params, &plain);
    assert!(
        st["branches"][0]["specials"].as_array().unwrap().is_empty(),
        "{st}"
    );
}

#[test]
fn bad_delay_hopf_runs_name_the_key() {
    let m = compiled(HUTCH);
    let ok_u = [1.001];
    let ok_p = [1.57, 1.0];
    let free = r#"["r","tau"]"#;
    let bounds = "[0.5,3.0]";
    let opts = hopf_options(bounds, 1);
    let with = |frag: &str| format!(r#"{{"type":"H",{frag},{opts}}}"#);
    let cases: Vec<(String, &str)> = vec![
        // omega
        (with(&format!(r#""free":{free}"#)), "omega"),
        (with(&format!(r#""free":{free},"omega":"x""#)), "omega"),
        (with(&format!(r#""free":{free},"omega":0"#)), "omega"),
        (with(&format!(r#""free":{free},"omega":null"#)), "omega"),
        // lyapunov
        (
            with(&format!(r#""free":{free},"omega":1.57,"lyapunov":1"#)),
            "lyapunov",
        ),
        (
            with(&format!(r#""free":{free},"omega":1.57,"lyapunov":"yes""#)),
            "lyapunov",
        ),
        (
            with(&format!(r#""free":{free},"omega":1.57,"lyapunov":null"#)),
            "lyapunov",
        ),
        (
            with(&format!(r#""free":{free},"omega":1.57,"lyapunov":[true]"#)),
            "lyapunov",
        ),
        // free
        (with(r#""free":"r","omega":1.57"#), "free"),
        (with(r#""free":["r"],"omega":1.57"#), "free"),
        (with(r#""free":["r","r"],"omega":1.57"#), "free"),
        (with(r#""free":["r","nope"],"omega":1.57"#), "free"),
        (with(r#""free":["r","delay(x,tau)"],"omega":1.57"#), "free"),
        // roots
        (
            with(&format!(
                r#""free":{free},"omega":1.57,"roots":{{"n":0,"re_min":-1}}"#
            )),
            "roots.n",
        ),
        (
            with(&format!(
                r#""free":{free},"omega":1.57,"roots":{{"n":30,"re_min":0}}"#
            )),
            "roots.re_min",
        ),
        (
            with(&format!(r#""free":{free},"omega":1.57,"roots":{{"n":30}}"#)),
            "roots",
        ),
        (
            with(&format!(
                r#""free":{free},"omega":1.57,"roots":{{"n":30,"re_min":-1,"m":3}}"#
            )),
            "roots",
        ),
        // keys a delay Hopf curve refuses
        (
            with(&format!(r#""free":{free},"omega":1.57,"switch":"BP1""#)),
            "switch",
        ),
        (
            with(&format!(
                r#""free":{free},"omega":1.57,"start_is_special":true"#
            )),
            "start_is_special",
        ),
        (
            format!(
                r#"{{"type":"H","free":{free},"omega":1.57,"options":{{"hmax":0.05,"bounds":{bounds},"detect":{{"h":true}}}}}}"#
            ),
            "detect",
        ),
        (
            format!(
                r#"{{"type":"H","free":{free},"omega":1.57,"options":{{"hmax":0.05,"bounds":{bounds},"mesh_adapt":2}}}}"#
            ),
            "mesh_adapt",
        ),
    ];
    for (json, key) in &cases {
        let (rc, run) = start_with(&m, json, &ok_u, &ok_p);
        assert_eq!(rc, BAD_PROBLEM, "{json}: {}", last_error());
        assert_eq!(run, 0, "{json}");
        let msg = last_error();
        assert!(msg.contains(key), "{json}: {msg}");
    }
    // Buffers of the wrong length.
    let good = hopf_problem(free, 1.57, bounds, "");
    for (u, p, key) in [
        (&[1.001, 2.0][..], &[1.57, 1.0][..], "u0"),
        (&[1.001][..], &[1.57][..], "params"),
        (&[1.001][..], &[1.57, 1.0, 0.0][..], "params"),
    ] {
        let (rc, run) = start_with(&m, &good, u, p);
        assert_eq!(rc, BAD_ARG, "{key}: {}", last_error());
        assert_eq!(run, 0);
        assert!(last_error().contains(key), "{}", last_error());
    }
    // A type of another run is still refused on a delay model.
    let (rc, run) = start_with(&m, r#"{"type":"LP","free":"r"}"#, &ok_u, &ok_p);
    assert_eq!(rc, BAD_ARG);
    assert_eq!(run, 0);
    assert_eq!(last_error(), DELAY_MODEL);
}

#[test]
fn the_new_hopf_keys_are_errors_on_an_ode_hopf_run() {
    let m = compiled("par a=1, b=2\nx'=-a*x+b*y\ny'=-b*x-a*y\ndone\n");
    for extra in [
        r#","omega":1.0"#,
        r#","lyapunov":true"#,
        r#","roots":{"n":30,"re_min":-1}"#,
    ] {
        let json = format!(r#"{{"type":"H","free":["a","b"]{extra}}}"#);
        let (rc, run) = start_with(&m, &json, &[0.0, 0.0], &[1.0, 2.0]);
        assert_eq!(rc, BAD_PROBLEM, "{json}: {}", last_error());
        assert_eq!(run, 0);
        let key = extra.split('"').nth(1).unwrap();
        assert!(last_error().contains(key), "{json}: {}", last_error());
    }
}

const ZH: &str = "par a=0, tau=1, b=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";
// The coupled model with quadratic and cubic terms (SPEC 8s slice 3 test 2).
const COUPLED_NL: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)+x*x+x*y-x*x*x\n\
    y'=-d*x-b*delay(y,1.5)+y*y*y+x*y*y\n\
    done\n";
const ZH_PARAMS: [f64; 3] = [0.07, 1.407_136_057_451_117_4, 1.0];
const ZH_X0: [f64; 2] = [0.001, -0.249];
const NL_PARAMS: [f64; 4] = [1.46, 0.8, 1.0, 0.1];

fn coefficient_extras(hh: bool) -> pc_engine::DdeHopfExtras {
    pc_engine::DdeHopfExtras {
        lyapunov: false,
        hh: hh.then_some(pc_engine::RootOptions {
            n: 30,
            re_min: -3.0,
        }),
        coefficients: true,
    }
}

/// The `coefficients` object of a special as (name, value) pairs, sorted by name.
fn coefficient_pairs(j: &Value) -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = j["coefficients"]
        .as_object()
        .expect("coefficients object")
        .iter()
        .map(|(k, x)| (k.clone(), x.as_f64().expect("a number")))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn engine_pairs(s: &pc_engine::DdeHopfSpecial) -> Vec<(String, f64)> {
    let mut v: Vec<(String, f64)> = s
        .coefficients
        .iter()
        .map(|(n, x)| (n.to_string(), *x))
        .collect();
    if let Some(w) = s.omega2 {
        v.push(("omega2".to_string(), w));
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn same_pairs(a: &[(String, f64)], b: &[(String, f64)]) {
    assert_eq!(
        a.iter().map(|p| &p.0).collect::<Vec<_>>(),
        b.iter().map(|p| &p.0).collect::<Vec<_>>()
    );
    for ((n, x), (_, y)) in a.iter().zip(b) {
        assert_eq!(x.to_bits(), y.to_bits(), "{n}: {x} against {y}");
    }
}

#[test]
fn a_zh_run_with_coefficients_carries_the_engine_values() {
    let want = engine_hopf(
        ZH,
        &ZH_PARAMS,
        &ZH_X0,
        0.98,
        (1.3, 2.0),
        Some(coefficient_extras(false)),
    );
    assert_eq!(want.specials.len(), 1);
    assert_eq!(want.specials[0].label, "ZH");
    let m = compiled(ZH);
    let json = hopf_problem(
        r#"["a","tau"]"#,
        0.98,
        "[1.3,2.0]",
        r#","coefficients":true"#,
    );
    let (st, _) = check_hopf_run(&m, &json, &ZH_X0, &ZH_PARAMS, &want);
    let js = st["branches"][0]["specials"].as_array().unwrap();
    assert_eq!(js.len(), 1, "{st}");
    assert_eq!(js[0]["label"], "ZH");
    let got = coefficient_pairs(&js[0]);
    same_pairs(&got, &engine_pairs(&want.specials[0]));
    let get = |name: &str| got.iter().find(|p| p.0 == name).map(|p| p.1).unwrap();
    // The closed form of SPEC 8i test 7.
    assert!((get("g200_re") + 1.0).abs() <= 1e-6);
    assert!((get("g110_re") + 0.288_400_439_142_001).abs() <= 1e-6);
    assert!(get("s_product").abs() <= 1e-6);
    // No warning for a special that has values.
    assert_eq!(st["branches"][0]["warnings"], serde_json::json!([]));
}

#[test]
fn an_hh_run_with_coefficients_carries_the_engine_values_and_omega2() {
    let want = engine_hopf(
        COUPLED_NL,
        &NL_PARAMS,
        &[0.001, 0.001],
        1.54,
        (0.5, 1.4),
        Some(coefficient_extras(true)),
    );
    assert_eq!(want.specials.len(), 1, "{:?}", want.specials.len());
    assert_eq!(want.specials[0].label, "HH");
    let m = compiled(COUPLED_NL);
    let json = hopf_problem(
        r#"["a","b"]"#,
        1.54,
        "[0.5,1.4]",
        r#","roots":{"n":30,"re_min":-3.0},"coefficients":true"#,
    );
    let (st, _) = check_hopf_run(&m, &json, &[0.001, 0.001], &NL_PARAMS, &want);
    let js = st["branches"][0]["specials"].as_array().unwrap();
    assert_eq!(js.len(), 1, "{st}");
    assert_eq!(js[0]["label"], "HH");
    let got = coefficient_pairs(&js[0]);
    same_pairs(&got, &engine_pairs(&want.specials[0]));
    let get = |name: &str| got.iter().find(|p| p.0 == name).map(|p| p.1).unwrap();
    assert!((get("omega_large") - OMEGA1_STAR).abs() <= 1e-6);
    assert!((get("omega_small") - OMEGA2_STAR).abs() <= 1e-6);
    // omega2 is still there next to the new values.
    assert!((get("omega2") - OMEGA2_STAR).abs() <= 1e-6);
}

#[test]
fn rows_do_not_change_when_coefficients_is_false_or_absent() {
    let off = engine_hopf(ZH, &ZH_PARAMS, &ZH_X0, 0.98, (1.3, 2.0), None);
    let m = compiled(ZH);
    for extra in [r#","coefficients":false"#, ""] {
        let json = hopf_problem(r#"["a","tau"]"#, 0.98, "[1.3,2.0]", extra);
        let (st, _) = check_hopf_run(&m, &json, &ZH_X0, &ZH_PARAMS, &off);
        let js = st["branches"][0]["specials"].as_array().unwrap();
        assert_eq!(js.len(), 1, "{st}");
        assert_eq!(js[0]["label"], "ZH");
        assert!(
            js[0]["coefficients"].as_object().unwrap().is_empty(),
            "{extra}: {st}"
        );
    }
    // The same curve with the option on has the same rows.
    let on = engine_hopf(
        ZH,
        &ZH_PARAMS,
        &ZH_X0,
        0.98,
        (1.3, 2.0),
        Some(coefficient_extras(false)),
    );
    assert_eq!(on.points.len(), off.points.len());
    for (p, q) in on.points.iter().zip(&off.points) {
        assert!(same_bits(&hopf_row(p), &hopf_row(q)));
    }
}

#[test]
fn bad_coefficients_keys_are_errors_that_name_the_key() {
    let m = compiled(HUTCH);
    let free = r#"["r","tau"]"#;
    let opts = hopf_options("[0.5,3.0]", 1);
    for bad in ["1", r#""yes""#, "null", "[true]", r#"{"a":1}"#, "0.5"] {
        let json =
            format!(r#"{{"type":"H","free":{free},"omega":1.57,"coefficients":{bad},{opts}}}"#);
        let (rc, run) = start_with(&m, &json, &[1.001], &[1.57, 1.0]);
        assert_eq!(rc, BAD_PROBLEM, "{bad}: {}", last_error());
        assert_eq!(run, 0, "{bad}");
        assert!(
            last_error().contains("coefficients"),
            "{bad}: {}",
            last_error()
        );
    }
}

#[test]
fn the_coefficients_key_is_an_error_on_every_other_run_type() {
    // A delay model with an equilibrium run.
    let dm = compiled(HUTCH);
    let json = problem(r#","coefficients":true"#);
    let (rc, run) = start_with(&dm, &json, &[1.0], &[1.3, 1.0]);
    assert_eq!(rc, BAD_PROBLEM, "{}", last_error());
    assert_eq!(run, 0);
    assert!(last_error().contains("coefficients"), "{}", last_error());
    // ODE models: H, EP, LP and the cycle types.
    let om = compiled("par a=1, b=2\nx'=-a*x+b*y\ny'=-b*x-a*y\ndone\n");
    let cases = [
        r#"{"type":"H","free":["a","b"],"coefficients":true}"#,
        r#"{"type":"EP","free":"a","coefficients":true}"#,
        r#"{"type":"LP","free":["a","b"],"coefficients":true}"#,
        r#"{"type":"LC","free":"a","ntst":10,"ncol":4,"coefficients":true}"#,
        r#"{"type":"LPC","free":["a","b"],"ntst":10,"ncol":4,"coefficients":true}"#,
        r#"{"type":"HOM","free":["a","b"],"ntst":10,"ncol":4,"coefficients":true}"#,
    ];
    for json in cases {
        let (rc, run) = start_with(&om, json, &[0.0, 0.0], &[1.0, 2.0]);
        assert_ne!(rc, 0, "{json}");
        assert_eq!(run, 0, "{json}");
        assert!(
            last_error().contains("coefficients"),
            "{json}: {}",
            last_error()
        );
    }
}

#[test]
fn a_hopf_run_cancelled_right_after_the_start_keeps_no_points() {
    let m = compiled(HUTCH);
    let json = r#"{"type":"H","free":["r","tau"],"omega":1.57,"options":{"h0":0.0001,"hmax":0.0001,"max_steps":100000,"bounds":[0.5,1000]}}"#;
    let (rc, run) = start_with(&m, json, &[1.001], &[1.57, 1.0]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
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
