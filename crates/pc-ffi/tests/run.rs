//! Runs through the table from `pc_api_v1()`.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

use pc_engine::{
    CurveSetup, CycleStart, EpSetup, Evaluator, HopfStart, Options, Point, Run, RunKind, RunSpec,
    RunState, StopReason, build,
};
use pc_ode::{MapLoader, SourceMap, load};
use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const LORENZ: &str = "par s=10, r=28, b=2.6666666666666665\n\
init x=1,y=1,z=1\n\
x'=s*(y-x)\n\
y'=x*(r-z)-y\n\
z'=x*y-b*z\n\
done\n";

const NON_AUTONOMOUS: &str = "par a=1\nx'=-a*x+sin(t)\ndone\n";
const P: [f64; 3] = [10.0, 28.0, 2.6666666666666665];
const ORIGIN: [f64; 3] = [0.0; 3];
const NOT_COMPILED: &str = "model is not compiled; call model_compile first";

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

/// Frees the handle when the test ends. A handle freed by the test has id 0.
struct Id(u64);

impl Drop for Id {
    fn drop(&mut self) {
        if self.0 != 0 {
            assert_eq!((api().free_handle)(self.0), 0);
        }
    }
}

fn parse(text: &str) -> Id {
    let mut id = 0u64;
    let rc = unsafe {
        (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, ptr::null_mut())
    };
    assert_eq!(rc, 0, "{}", last_error());
    Id(id)
}

fn compile(m: &Id, evaluator: &str) {
    let opts = CString::new(format!("{{\"evaluator\":\"{evaluator}\"}}")).unwrap();
    let rc = unsafe { (api().model_compile)(m.0, opts.as_ptr()) };
    assert_eq!(rc, 0, "{}", last_error());
}

fn lorenz(evaluator: &str) -> Id {
    let m = parse(LORENZ);
    compile(&m, evaluator);
    m
}

const OPTIONS: &str =
    r#""options":{"h0":0.05,"hmax":0.5,"max_steps":300,"direction":-1,"bounds":[-5,60]}"#;

fn problem(extra: &str) -> String {
    format!(r#"{{"type":"EP","free":"r",{OPTIONS}{extra}}}"#)
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
    let (rc, run) = start_with(m, json, &ORIGIN, &P);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(run, 0);
    Id(run)
}

/// A run that cannot finish within a test.
fn long_run(m: &Id) -> Id {
    start(
        m,
        r#"{"type":"EP","free":"r","options":{"h0":0.001,"hmax":0.001,"max_steps":1000000,"bounds":[-1000,1000]}}"#,
    )
}

fn poll_raw(run: u64) -> (i32, Option<Value>) {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().run_poll)(run, &mut s) };
    if s.is_null() {
        return (rc, None);
    }
    let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(s) };
    (rc, Some(serde_json::from_str(&text).unwrap()))
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
fn copy(
    run: u64,
    branch: u32,
    what: u32,
    start: usize,
    count: usize,
    stride: usize,
) -> (i32, usize, Vec<f64>) {
    let mut buf = vec![f64::NAN; count.min(1 << 20) * stride];
    let mut rows = 99usize;
    let rc = unsafe {
        (api().run_copy)(
            run,
            branch,
            what,
            start,
            count,
            buf.as_mut_ptr(),
            buf.len(),
            &mut rows,
        )
    };
    buf.truncate(rows * stride);
    (rc, rows, buf)
}

/// The packing the ABI documents, written out again.
fn expected_row(p: &Point, n: usize) -> Vec<f64> {
    let mut row = vec![p.alpha];
    let get = |i: usize| p.u.get(i).copied().unwrap_or(f64::NAN);
    row.extend((0..n).map(get));
    let eig = |i: usize| {
        p.eigenvalues
            .get(i)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN))
    };
    row.extend((0..n).map(|i| eig(i).0));
    row.extend((0..n).map(|i| eig(i).1));
    row.extend(p.tests);
    row.push(p.h);
    row.push(p.newton_iters as f64);
    row
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

fn engine_spec(evaluator: Evaluator, switch: Option<&str>) -> RunSpec {
    let mut map = SourceMap::new();
    let f = map.add("model.ode", LORENZ);
    let l = load(f, &mut map, &mut MapLoader::new());
    let setup = EpSetup::new(1, P.to_vec(), ORIGIN.to_vec());
    let opts = Options {
        h0: 0.05,
        hmax: 0.5,
        max_steps: 300,
        direction: -1.0,
        bounds: (-5.0, 60.0),
        ..Options::default()
    };
    let mut spec = RunSpec::new(l.expanded, l.info, evaluator, setup, opts);
    spec.switch_at = switch.map(str::to_owned);
    spec
}

fn stop_name(s: &StopReason) -> &'static str {
    match s {
        StopReason::MaxSteps => "MaxSteps",
        StopReason::Bounds => "Bounds",
        StopReason::StepTooSmall => "StepTooSmall",
        StopReason::Singular => "Singular",
        _ => "Cancelled",
    }
}

#[test]
fn running_without_compiling_gives_2() {
    let m = parse(LORENZ);
    let (rc, run) = start_with(&m, &problem(""), &ORIGIN, &P);
    assert_eq!(rc, 2);
    assert_eq!(run, 0);
    assert_eq!(last_error(), NOT_COMPILED);
}

fn check_against_engine(evaluator: &str, which: Evaluator, switch: Option<&str>) {
    let m = lorenz(evaluator);
    let extra = switch.map_or(String::new(), |s| format!(r#","switch":"{s}""#));
    let run = start(&m, &problem(&extra));
    let st = wait(&run);
    assert_eq!(st["state"], "done");
    assert_eq!(st["n"], 3);
    assert_eq!(st["stride"], 15);

    // Reading the engine's model directly checks the C ABI against `build` too.
    let engine = Run::start(engine_spec(which, switch));
    let engine_st = engine.poll();
    let t0 = Instant::now();
    while engine.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(5));
    }
    let engine_st = if engine_st.state == RunState::Running {
        engine.poll()
    } else {
        engine_st
    };
    assert_eq!(engine_st.state, RunState::Done);
    let branches = st["branches"].as_array().unwrap();
    assert_eq!(branches.len(), engine_st.points.len());
    for (b, jb) in branches.iter().enumerate() {
        assert_eq!(jb["points"], engine_st.points[b], "branch {b}");
        assert_eq!(
            jb["stop"],
            stop_name(&engine.stop(b).unwrap()),
            "branch {b}"
        );
        let (rc, rows, vals) = copy(run.0, b as u32, 0, 0, engine_st.points[b], 15);
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(rows, engine_st.points[b]);
        let pts = engine.points(b, 0, usize::MAX);
        for (r, p) in pts.iter().enumerate() {
            assert!(
                same_bits(&vals[r * 15..(r + 1) * 15], &expected_row(p, 3)),
                "branch {b} row {r}"
            );
        }
        // Specials and their rows.
        let specials = engine.specials(b);
        let js = jb["specials"].as_array().unwrap();
        assert_eq!(js.len(), specials.len(), "branch {b}");
        for (j, s) in js.iter().zip(&specials) {
            assert_eq!(j["label"], s.label);
            assert_eq!(j["index"], s.index);
            assert_eq!(j["refined"], s.extras.refined);
            for (key, want) in [
                ("coefficient", s.extras.coefficient),
                ("omega", s.extras.omega),
            ] {
                match want {
                    Some(x) => assert_eq!(j[key].as_f64().unwrap().to_bits(), x.to_bits()),
                    None => assert!(j[key].is_null(), "{key}"),
                }
            }
            match &s.extras.normalization {
                Some(x) => assert_eq!(j["normalization"], x.as_str()),
                None => assert!(j["normalization"].is_null()),
            }
        }
        let (rc, rows, vals) = copy(run.0, b as u32, 1, 0, specials.len(), 15);
        assert_eq!(rc, 0, "{}", last_error());
        assert_eq!(rows, specials.len());
        for (r, s) in specials.iter().enumerate() {
            assert!(same_bits(
                &vals[r * 15..(r + 1) * 15],
                &expected_row(&s.point, 3)
            ));
        }
        let warnings: Vec<String> = jb["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(warnings, engine.warnings(b));
    }
    // Unused for the comparison, but proves the model builds in the test process.
    let _ = build;
}

#[test]
fn rows_equal_the_engine_on_the_jit() {
    check_against_engine("jit", Evaluator::Jit, None);
}

#[test]
fn rows_equal_the_engine_on_the_fallback() {
    check_against_engine("fallback", Evaluator::Fallback, None);
}

#[test]
fn specials_and_stops_match_with_a_switch() {
    check_against_engine("jit", Evaluator::Jit, Some("BP1"));
}

#[test]
fn a_switch_at_bp1_gives_two_branches() {
    let m = lorenz("jit");
    let run = start(&m, &problem(r#","switch":"BP1""#));
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    let b = st["branches"].as_array().unwrap();
    assert_eq!(b.len(), 2);
    assert!(b[1]["points"].as_u64().unwrap() > 0);
    assert!(
        b[0]["specials"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["label"] == "BP")
    );
}

#[test]
fn a_switch_to_a_missing_special_gives_6_from_run_poll() {
    // `XY1` has a valid shape, and the labels exist only once the first branch is done,
    // so the worker rejects it and `run_poll` reports the code.
    let m = lorenz("jit");
    let run = start(&m, &problem(r#","switch":"XY1""#));
    let t0 = Instant::now();
    loop {
        let (rc, v) = poll_raw(run.0);
        if rc != 0 {
            assert_eq!(rc, 6, "{}", last_error());
            assert!(v.is_none());
            assert!(last_error().contains("XY1"), "{}", last_error());
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_malformed_switch_gives_6_at_run_start() {
    let m = lorenz("jit");
    let (rc, run) = start_with(&m, &problem(r#","switch":"BP""#), &ORIGIN, &P);
    assert_eq!(rc, 6);
    assert_eq!(run, 0);
}

#[test]
fn u0_and_params_keys_give_6_with_the_parser_message() {
    let m = lorenz("jit");
    let (rc, run) = start_with(&m, &problem(r#","u0":[0,0,0]"#), &ORIGIN, &P);
    assert_eq!((rc, run), (6, 0));
    assert_eq!(
        last_error(),
        "u0: not allowed here; pass the state as a buffer"
    );
    let (rc, run) = start_with(&m, &problem(r#","params":{"s":1}"#), &ORIGIN, &P);
    assert_eq!((rc, run), (6, 0));
    assert_eq!(
        last_error(),
        "params: not allowed here; pass the parameters as a buffer"
    );
    let (rc, _) = start_with(&m, "{not json", &ORIGIN, &P);
    assert_eq!(rc, 6);
    assert!(
        last_error().starts_with("problem_json: "),
        "{}",
        last_error()
    );
}

#[test]
fn buffer_lengths_are_checked() {
    let m = lorenz("jit");
    let (rc, run) = start_with(&m, &problem(""), &ORIGIN[..2], &P);
    assert_eq!((rc, run), (2, 0));
    assert_eq!(last_error(), "u0: expected 3 values, got 2");
    let (rc, run) = start_with(&m, &problem(""), &ORIGIN, &P[..2]);
    assert_eq!((rc, run), (2, 0));
    assert_eq!(last_error(), "params: expected 3 values, got 2");
    let c = CString::new(problem("")).unwrap();
    let mut run = 0u64;
    let rc = unsafe { (api().run_start)(m.0, c.as_ptr(), ptr::null(), 3, P.as_ptr(), 3, &mut run) };
    assert_eq!((rc, run), (2, 0));
}

#[test]
fn a_non_autonomous_model_compiles_and_run_start_gives_7() {
    // `model_compile` accepts a model that uses t; runs need an autonomous one.
    let m = parse(NON_AUTONOMOUS);
    let opts = c"{\"evaluator\":\"jit\"}";
    assert_eq!(unsafe { (api().model_compile)(m.0, opts.as_ptr()) }, 0);
    let json = r#"{"type":"EP","free":"a"}"#;
    let (rc, run) = start_with(&m, json, &[0.0], &[1.0]);
    assert_eq!((rc, run), (7, 0));
    assert_eq!(last_error(), "the model depends on t");
}

#[test]
fn cancel_ends_in_cancelled() {
    let m = lorenz("jit");
    let run = long_run(&m);
    assert_eq!((api().run_cancel)(run.0), 0);
    let st = wait(&run);
    assert_eq!(st["state"], "cancelled");
}

#[test]
fn free_while_running_joins_the_worker() {
    let m = lorenz("jit");
    let mut run = long_run(&m);
    let id = run.0;
    run.0 = 0;
    assert_eq!((api().free_handle)(id), 0);
    let (rc, v) = poll_raw(id);
    assert_eq!(rc, 3);
    assert!(v.is_none());
}

#[test]
fn kinds_are_not_interchangeable() {
    let m = lorenz("jit");
    let run = start(&m, &problem(""));
    let mut s: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { (api().model_info)(run.0, &mut s) }, 3);
    assert!(
        last_error().contains("is a run, not a model"),
        "{}",
        last_error()
    );
    assert_eq!(unsafe { (api().run_poll)(m.0, &mut s) }, 3);
    assert!(s.is_null());
    assert!(
        last_error().contains("is a model, not a run"),
        "{}",
        last_error()
    );
}

#[test]
fn a_worker_panic_reaches_run_poll() {
    let mut id = 0u64;
    assert_eq!(unsafe { (api().debug_panic)(1, &mut id) }, 0);
    assert_ne!(id, 0);
    let run = Id(id);
    let t0 = Instant::now();
    loop {
        let (rc, v) = poll_raw(run.0);
        if rc != 0 {
            assert_eq!(rc, 1);
            assert!(v.is_none());
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(5));
    }
    let e = last_error();
    assert!(e.starts_with("panic: start_panicking (at "), "{e}");
    assert!(e.replace('\\', "/").contains("run.rs:"), "{e}");
    drop(run);
}

#[test]
fn run_copy_checks_its_arguments() {
    let m = lorenz("jit");
    let run = start(&m, &problem(""));
    let st = wait(&run);
    let points = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert!(points > 3);
    assert_eq!(copy(run.0, 0, 2, 0, 1, 15).0, 2);
    assert!(last_error().starts_with("what: expected 0 (points) or 1 (specials), got 2"));
    assert_eq!(copy(run.0, 5, 0, 0, 1, 15).0, 2);
    assert_eq!(last_error(), "branch: the run has 1 branches, got 5");
    let mut buf = [0.0f64; 14];
    let mut rows = 99usize;
    let rc = unsafe { (api().run_copy)(run.0, 0, 0, 0, 1, buf.as_mut_ptr(), buf.len(), &mut rows) };
    assert_eq!((rc, rows), (2, 0));
    assert_eq!(last_error(), "buf: expected at least 15 values, got 14");
    let (rc, rows, _) = copy(run.0, 0, 0, points + 10, 4, 15);
    assert_eq!((rc, rows), (0, 0));
    let (rc, rows, _) = copy(run.0, 0, 0, points - 2, 10, 15);
    assert_eq!((rc, rows), (0, 2));
    let mut rows = 99usize;
    let rc = unsafe { (api().run_copy)(run.0, 0, 0, 0, 0, ptr::null_mut(), 0, &mut rows) };
    assert_eq!((rc, rows), (0, 0));
    let mut rows = 99usize;
    let rc = unsafe {
        (api().run_copy)(
            run.0,
            0,
            0,
            0,
            usize::MAX,
            buf.as_mut_ptr(),
            buf.len(),
            &mut rows,
        )
    };
    assert_eq!((rc, rows), (2, 0));
}

#[test]
fn the_model_can_be_freed_mid_run() {
    let mut m = lorenz("jit");
    let run = start(&m, &problem(""));
    let id = m.0;
    m.0 = 0;
    assert_eq!((api().free_handle)(id), 0);
    let st = wait(&run);
    assert_eq!(st["state"], "done");
    let n = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert!(n > 0);
    let (rc, rows, _) = copy(run.0, 0, 0, 0, n, 15);
    assert_eq!((rc, rows), (0, n));
}

#[test]
fn null_out_pointers_give_2() {
    let m = lorenz("jit");
    let c = CString::new(problem("")).unwrap();
    let rc = unsafe {
        (api().run_start)(
            m.0,
            c.as_ptr(),
            ORIGIN.as_ptr(),
            3,
            P.as_ptr(),
            3,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 2);
    let run = start(&m, &problem(""));
    assert_eq!(unsafe { (api().run_poll)(run.0, ptr::null_mut()) }, 2);
    let rc = unsafe { (api().run_copy)(run.0, 0, 0, 0, 0, ptr::null_mut(), 0, ptr::null_mut()) };
    assert_eq!(rc, 2);
}

const FOLD: &str = "par a=0\nx'=a+x^2\ninit x=0\ndone\n";

#[test]
fn a_run_from_an_lp_start_is_done() {
    let m = parse(FOLD);
    compile(&m, "jit");
    let json = r#"{"type":"EP","free":"a","start_is_special":true,
        "options":{"h0":0.05,"hmax":0.05,"max_steps":30,"direction":1,"bounds":[-5,5]}}"#;
    let (rc, run) = start_with(&m, json, &[0.0], &[0.0]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    assert!(st["branches"][0]["points"].as_u64().unwrap() > 5, "{st}");
}

#[test]
fn a_run_from_a_bp_start_fails_with_6_and_names_switch() {
    let m = lorenz("jit");
    let json = problem(r#","start_is_special":true"#);
    let p = [10.0, 1.0, 2.6666666666666665];
    let (rc, run) = start_with(&m, &json, &ORIGIN, &p);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let t0 = Instant::now();
    loop {
        let (rc, v) = poll_raw(run.0);
        if rc != 0 {
            assert_eq!(rc, 6, "{}", last_error());
            assert!(v.is_none());
            assert!(last_error().contains("Switch"), "{}", last_error());
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(5));
    }
}

// ---- LP and Hopf curves ----

const CUSP: &str = "par b1=0,b2=1\nx'=b1+b2*x-x^3\ninit x=-1\ndone\n";
const BT: &str = "par b1=0,b2=0,s=1\nx'=y\ny'=b1+b2*y+x^2+s*x*y\ninit x=0,y=0\ndone\n";
const BRUSS: &str = "par a=1,b=1.5\nx'=a-(b+1)*x+x^2*y\ny'=b*x-x^2*y\ninit x=1,y=1\ndone\n";

fn compiled(text: &str) -> Id {
    let m = parse(text);
    compile(&m, "jit");
    m
}

/// Runs an equilibrium branch to the end and returns the first special point called
/// `label` as (value of the free parameter, state).
fn ep_special(
    m: &Id,
    free: &str,
    u0: &[f64],
    p: &[f64],
    bounds: &str,
    label: &str,
) -> (f64, Vec<f64>) {
    let json = format!(r#"{{"type":"EP","free":"{free}","options":{{"bounds":{bounds}}}}}"#);
    let (rc, run) = start_with(m, &json, u0, p);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    let n = st["n"].as_u64().unwrap() as usize;
    let stride = st["stride"].as_u64().unwrap() as usize;
    let k = st["branches"][0]["specials"]
        .as_array()
        .unwrap()
        .iter()
        .position(|s| s["label"] == label)
        .unwrap_or_else(|| panic!("no {label}: {st}"));
    let (rc, rows, row) = copy(run.0, 0, 1, k, 1, stride);
    assert_eq!((rc, rows), (0, 1));
    (row[0], row[1..=n].to_vec())
}

/// The curve row the ABI documents, written out again: `[a2, x(n), a1, extra,
/// eig_re(n), eig_im(n), codim2(k), h, newton_iters]`, where `extra` is kappa on a Hopf curve.
fn expected_curve_row(p: &Point, n: usize, extra: usize, k: usize) -> Vec<f64> {
    let mut row = vec![p.alpha];
    let get = |i: usize| p.u.get(i).copied().unwrap_or(f64::NAN);
    row.extend((0..n + 1 + extra).map(get));
    let eig = |i: usize| {
        p.eigenvalues
            .get(i)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN))
    };
    row.extend((0..n).map(|i| eig(i).0));
    row.extend((0..n).map(|i| eig(i).1));
    row.extend((0..k).map(|i| p.codim2.get(i).copied().unwrap_or(f64::NAN)));
    row.push(p.h);
    row.push(p.newton_iters as f64);
    row
}

/// Starts the curve run `json` and compares every row with the engine's own run of `kind`.
fn check_curve(
    text: &str,
    json: &str,
    kind: RunKind,
    u0: &[f64],
    p: &[f64],
    opts: Options,
    (n, extra, k): (usize, usize, usize),
) -> Value {
    let m = compiled(text);
    let (rc, run) = start_with(&m, json, u0, p);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    let stride = st["stride"].as_u64().unwrap() as usize;
    assert_eq!(stride, 1 + n + 1 + extra + 2 * n + k + 2, "{st}");
    assert_eq!(st["n"].as_u64().unwrap() as usize, n);

    let mut map = SourceMap::new();
    let f = map.add("model.ode", text);
    let l = load(f, &mut map, &mut MapLoader::new());
    let spec = RunSpec::curve(l.expanded, l.info, Evaluator::Jit, kind, opts);
    let engine = Run::start(spec);
    let t0 = Instant::now();
    while engine.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(2));
    }
    let want = engine.points(0, 0, usize::MAX);
    let count = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert_eq!(count, want.len());
    assert!(count > 5, "{st}");
    let (rc, rows, got) = copy(run.0, 0, 0, 0, count, stride);
    assert_eq!((rc, rows), (0, count), "{}", last_error());
    for (i, point) in want.iter().enumerate() {
        let w = expected_curve_row(point, n, extra, k);
        assert!(same_bits(&got[i * stride..(i + 1) * stride], &w), "row {i}");
    }
    st
}

#[test]
fn an_lp_curve_run_has_curve_rows_and_finds_the_cusp() {
    let m = compiled(CUSP);
    let (b1, u) = ep_special(&m, "b1", &[-1.0], &[0.0, 1.0], "[-1,1]", "LP");
    let setup = CurveSetup::new([0, 1], vec![b1, 1.0], u.clone());
    let opts = Options {
        bounds: (-0.5, 1.5),
        ..Options::default()
    };
    let json = r#"{"type":"LP","free":["b1","b2"],"options":{"bounds":[-0.5,1.5]}}"#;
    let st = check_curve(
        CUSP,
        json,
        RunKind::LpCurve(setup),
        &u,
        &[b1, 1.0],
        opts,
        (1, 0, 3),
    );
    assert_eq!(st["kind"], "LP");
    assert_eq!(st["codim2_labels"], serde_json::json!(["CP", "BT", "ZH"]));
    let cp = st["branches"][0]["specials"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["label"] == "CP")
        .unwrap_or_else(|| panic!("no CP: {st}"));
    assert!(
        (cp["coefficient"].as_f64().unwrap() + 1.0).abs() < 1e-8,
        "{cp}"
    );
    assert_eq!(cp["refined"], true);
}

#[test]
fn a_hopf_curve_run_has_curve_rows_with_kappa() {
    let m = compiled(BRUSS);
    let (b, u) = ep_special(&m, "b", &[1.0, 1.2], &[1.0, 1.2], "[0.5,4]", "H");
    let setup = CurveSetup::new([0, 1], vec![1.0, b], u.clone());
    let opts = Options {
        bounds: (1.5, 5.0),
        ..Options::default()
    };
    let json = r#"{"type":"H","free":["a","b"],"options":{"bounds":[1.5,5]}}"#;
    let st = check_curve(
        BRUSS,
        json,
        RunKind::HopfCurve(setup),
        &u,
        &[1.0, b],
        opts,
        (2, 1, 4),
    );
    assert_eq!(st["kind"], "H");
    assert_eq!(
        st["codim2_labels"],
        serde_json::json!(["GH", "BT", "ZH", "HH"])
    );
}

#[test]
fn a_bt_point_reports_both_coefficients() {
    let m = compiled(BT);
    let (b1, u) = ep_special(&m, "b1", &[-1.0, 0.0], &[-1.0, 0.5, 1.0], "[-1,1]", "LP");
    let json = r#"{"type":"LP","free":["b1","b2"],"options":{"direction":-1,"bounds":[-0.5,1]}}"#;
    let (rc, run) = start_with(&m, json, &u, &[b1, 0.5, 1.0]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    let bt = st["branches"][0]["specials"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["label"] == "BT")
        .unwrap_or_else(|| panic!("no BT: {st}"));
    assert!(
        (bt["coefficient"].as_f64().unwrap() - 1.0).abs() < 1e-8,
        "{bt}"
    );
    assert!(
        (bt["coefficient2"].as_f64().unwrap() - 1.0).abs() < 1e-8,
        "{bt}"
    );
    assert!(bt["coefficients"].is_object(), "{bt}");
}

#[test]
fn equilibrium_status_names_its_kind_and_has_no_codim2_labels() {
    let m = lorenz("jit");
    let run = start(&m, &problem(""));
    let st = wait(&run);
    assert_eq!(st["kind"], "EP");
    assert_eq!(st["codim2_labels"], serde_json::json!([]));
    assert_eq!(st["stride"].as_u64().unwrap(), 15);
    for s in st["branches"][0]["specials"].as_array().unwrap() {
        assert!(s["coefficient2"].is_null(), "{s}");
        assert!(s["coefficients"].as_object().unwrap().is_empty(), "{s}");
    }
}

#[test]
fn bad_curve_problems_give_6_at_run_start() {
    let m = compiled(CUSP);
    let cases = [
        (
            r#"{"type":"LP","free":"b1"}"#,
            "free: expected two parameter names",
        ),
        (
            r#"{"type":"LP","free":["b1"]}"#,
            "free: expected two parameter names",
        ),
        (
            r#"{"type":"LP","free":["b1","b1"]}"#,
            "free: the two parameters must differ",
        ),
        (
            r#"{"type":"H","free":["b1","zz"]}"#,
            "free[1]: unknown parameter",
        ),
        (
            r#"{"type":"EP","free":["b1","b2"]}"#,
            "free: expected a string",
        ),
        (
            r#"{"type":"LP","free":["b1","b2"],"switch":"BP1"}"#,
            "switch: only an EP run",
        ),
        (
            r#"{"type":"LP","free":["b1","b2"],"start_is_special":true}"#,
            "start_is_special: only an EP run",
        ),
        (
            r#"{"type":"XX","free":"b1"}"#,
            r#"type: expected "EP", "LP", "H" or "LC""#,
        ),
    ];
    for (json, msg) in cases {
        let (rc, run) = start_with(&m, json, &[0.0], &[0.0, 1.0]);
        assert_eq!((rc, run), (6, 0), "{json}");
        assert!(last_error().contains(msg), "{json}: {}", last_error());
    }
}

/// The cycle row the ABI documents, written out again: `[alpha, u(n (N m + 1)), T,
/// mesh(N + 1), mult_re(n), mult_im(n), psi_LPC, psi_PD, psi_NS, h, newton_iters]`.
fn expected_cycle_row(p: &Point, n: usize, ntst: usize, ncol: usize) -> Vec<f64> {
    let mut row = vec![p.alpha];
    let get = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(f64::NAN);
    row.extend((0..n * (ntst * ncol + 1) + 1).map(|i| get(&p.u, i)));
    row.extend((0..=ntst).map(|i| get(&p.mesh, i)));
    let mult = |i: usize| {
        p.eigenvalues
            .get(i)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN))
    };
    row.extend((0..n).map(|i| mult(i).0));
    row.extend((0..n).map(|i| mult(i).1));
    row.extend((0..3).map(|i| get(&p.codim2, i)));
    row.push(p.h);
    row.push(p.newton_iters as f64);
    row
}

const LC_OPTIONS: &str = r#""options":{"hmax":0.05,"max_steps":15,"bounds":[1.5,2.6]}"#;

/// The Brusselator Hopf point in b at a = 1, as (b, equilibrium).
fn bruss_hopf(m: &Id) -> (f64, Vec<f64>) {
    let (b, u) = ep_special(m, "b", &[1.0, 1.2], &[1.0, 1.2], "[0.5,4]", "H");
    // The Hopf point is at b = 1 + a^2 = 2, where the equilibrium is (a, b / a) = (1, 2).
    assert!((b - 2.0).abs() < 1e-8, "b = {b}");
    (b, u)
}

#[test]
fn a_cycle_run_from_a_hopf_point_has_cycle_rows() {
    let m = compiled(BRUSS);
    let (b, u) = bruss_hopf(&m);
    let (n, ntst, ncol) = (2, 10, 3);
    let json = format!(
        r#"{{"type":"LC","free":"b","ntst":{ntst},"ncol":{ncol},"start":"hopf",{LC_OPTIONS}}}"#
    );
    let (rc, run) = start_with(&m, &json, &u, &[1.0, b]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    assert_eq!(st["kind"], "LC");
    assert_eq!(st["codim2_labels"], serde_json::json!(["LPC", "PD", "NS"]));
    assert_eq!(st["ntst"], ntst);
    assert_eq!(st["ncol"], ncol);
    let stride = st["stride"].as_u64().unwrap() as usize;
    assert_eq!(
        stride,
        1 + n * (ntst * ncol + 1) + 1 + (ntst + 1) + 2 * n + 3 + 2
    );

    let start = HopfStart {
        free: 1,
        params: vec![1.0, b],
        u0: u.clone(),
        ntst,
        ncol,
    };
    let opts = Options {
        hmax: 0.05,
        max_steps: 15,
        bounds: (1.5, 2.6),
        ..Options::default()
    };
    let mut map = SourceMap::new();
    let f = map.add("model.ode", BRUSS);
    let l = load(f, &mut map, &mut MapLoader::new());
    let kind = RunKind::Cycle(CycleStart::Hopf(start));
    let engine = Run::start(RunSpec::curve(
        l.expanded,
        l.info,
        Evaluator::Jit,
        kind,
        opts,
    ));
    let t0 = Instant::now();
    while engine.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(2));
    }
    let want = engine.points(0, 0, usize::MAX);
    let count = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert_eq!(count, want.len());
    assert!(count > 5, "{st}");
    let (rc, rows, got) = copy(run.0, 0, 0, 0, count, stride);
    assert_eq!((rc, rows), (0, count), "{}", last_error());
    for (i, point) in want.iter().enumerate() {
        let w = expected_cycle_row(point, n, ntst, ncol);
        assert!(same_bits(&got[i * stride..(i + 1) * stride], &w), "row {i}");
    }
    // The period starts near 2 pi / omega = 2 pi, and the mesh runs from 0 to 1.
    let t = got[1 + n * (ntst * ncol + 1)];
    assert!((t - 2.0 * std::f64::consts::PI).abs() < 0.1, "T = {t}");
    let mesh = &got[2 + n * (ntst * ncol + 1)..][..=ntst];
    assert_eq!((mesh[0], mesh[ntst]), (0.0, 1.0));
}

#[test]
fn a_cycle_run_from_an_orbit_continues_it() {
    let m = compiled(BRUSS);
    let (b, u) = bruss_hopf(&m);
    let (n, ntst, ncol) = (2, 6, 3);
    // Without mesh adaptation the orbit of each row is on the uniform mesh.
    let json = format!(
        r#"{{"type":"LC","free":"b","ntst":{ntst},"ncol":{ncol},"start":"hopf","options":{{"hmax":0.05,"max_steps":6,"bounds":[1.5,2.6],"mesh_adapt":0}}}}"#
    );
    let (rc, run) = start_with(&m, &json, &u, &[1.0, b]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    let stride = st["stride"].as_u64().unwrap() as usize;
    let count = st["branches"][0]["points"].as_u64().unwrap() as usize;
    let (rc, rows, last) = copy(run.0, 0, 0, count - 1, 1, stride);
    assert_eq!((rc, rows), (0, 1));
    let state = n * (ntst * ncol + 1) + 1;
    let orbit = &last[1..=state];

    let json = format!(
        r#"{{"type":"LC","free":"b","ntst":{ntst},"ncol":{ncol},"start":"orbit","options":{{"hmax":0.05,"max_steps":5,"bounds":[1.5,2.6]}}}}"#
    );
    let (rc, run) = start_with(&m, &json, orbit, &[1.0, last[0]]);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    assert!(st["branches"][0]["points"].as_u64().unwrap() > 2, "{st}");
    // The first point is the corrected start, which is already a solution.
    let (rc, _, first) = copy(run.0, 0, 0, 0, 1, stride);
    assert_eq!(rc, 0);
    assert!((first[0] - last[0]).abs() < 1e-8);
    assert!((first[state] - last[state]).abs() < 1e-6, "T");
}

#[test]
fn bad_cycle_problems_give_6_and_bad_lengths_give_2() {
    let m = compiled(BRUSS);
    let cases = [
        (
            r#"{"type":"LC","free":"b","ncol":3,"start":"hopf"}"#,
            "ntst: missing",
        ),
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":9,"start":"hopf"}"#,
            "ncol: expected an integer from 2 to 7",
        ),
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":3,"start":"x"}"#,
            "start: expected",
        ),
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":3,"start":"pd"}"#,
            "mesh: missing",
        ),
        (
            r#"{"type":"EP","free":"b","ntst":4}"#,
            "ntst: only an LC run takes this key",
        ),
        (
            r#"{"type":"EP","free":"b","options":{"mesh_adapt":2}}"#,
            "options.mesh_adapt: only an LC run",
        ),
    ];
    for (json, msg) in cases {
        let (rc, run) = start_with(&m, json, &[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!((rc, run), (6, 0), "{json}");
        assert!(last_error().contains(msg), "{json}: {}", last_error());
    }
    let lengths = [
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":3,"start":"orbit"}"#,
            "u0: expected 27 values, got 2",
        ),
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":3,"start":"pd","mesh":[0,0.5,1]}"#,
            "u0: expected n (N m + 1) + 1 values with n = 2, N = 2 from mesh",
        ),
        (
            r#"{"type":"LC","free":"b","ntst":4,"ncol":3,"start":"hopf"}"#,
            "u0: expected 2 values, got 3",
        ),
    ];
    for (i, (json, msg)) in lengths.into_iter().enumerate() {
        let u0: &[f64] = if i == 2 {
            &[1.0, 2.0, 3.0]
        } else {
            &[1.0, 2.0]
        };
        let (rc, run) = start_with(&m, json, u0, &[1.0, 2.0]);
        assert_eq!((rc, run), (2, 0), "{json}");
        assert!(last_error().contains(msg), "{json}: {}", last_error());
    }
}
