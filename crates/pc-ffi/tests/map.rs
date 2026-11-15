//! Map models through the table from `pc_api_v1()` (SPEC section 8t, tests 4 and 5, section
//! 8u, test 3, and section 8v, test 9).

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Duration, Instant};

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

/// The refusal text of SPEC 8v.
const MAP_MODEL: &str = "the model is a map (`x(t+1)=...` or `@ meth=discrete`); \
     PhaseCont iterates a map and continues its fixed points in one parameter (types ITER and \
     EP) but does not follow curves or cycles of a map yet";
const UNSUPPORTED_MODEL: i32 = 13;
const BAD_PROBLEM: i32 = 6;
const NON_AUTONOMOUS: i32 = 7;
const NOT_COMPILED: &str = "model is not compiled; call model_compile first";

/// The logistic map, as in `crates/pc-ode/tests/corpus/logistic_map.ode`.
const LOGISTIC_MAP: &str = include_str!("../../pc-ode/tests/corpus/logistic_map.ode");
/// The same right-hand side as an ODE file, with and without `@ meth=discrete`.
const LOGISTIC_ODE: &str = "par r=2\ninit x=0.2\nx'=r*x*(1-x)\ndone\n";
const LOGISTIC_DISCRETE: &str = "par r=2\ninit x=0.2\nx'=r*x*(1-x)\n@ meth=discrete\ndone\n";
/// A map that reads `t`, for `model_rhs_t`.
const TIMED_ODE: &str = "par r=2\nx'=r*x*(1-x)+t\ndone\n";
const TIMED_DISCRETE: &str = "par r=2\nx'=r*x*(1-x)+t\n@ meth=discrete\ndone\n";
const HUTCH: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";
/// A map that reads `t`: from x(0) = 0 it gives 0, 0, 1, 3, 6.
const SHIFT: &str = "init x=0\nx(t+1)=x+t\ndone\n";
/// The same map with a parameter, so that an EP problem parses and reaches the check for t.
const SHIFT_P: &str = "par a=1\ninit x=0\nx(t+1)=a*x+t\ndone\n";
/// The Henon map, as in `crates/pc-ode/tests/corpus/henon_map.ode`.
const HENON: &str = include_str!("../../pc-ode/tests/corpus/henon_map.ode");
/// A map that overflows at its ninth iterate from 10.
const SQUARE: &str = "init x=10\nx(t+1)=x*x\ndone\n";

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

fn poll(run: u64) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().run_poll)(run, &mut s) };
    assert_eq!(rc, 0, "{}", last_error());
    serde_json::from_str(&take_string(s).unwrap()).unwrap()
}

/// Polls every 2 ms until the state is not "running".
fn wait(run: &Id) -> Value {
    let t0 = Instant::now();
    loop {
        let v = poll(run.0);
        if v["state"] != "running" {
            return v;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "run did not finish");
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Every row of branch 0 of a finished run, as `stride` values a row.
fn rows(run: &Id, st: &Value) -> Vec<f64> {
    branch_rows(run, st, 0)
}

/// Every row of branch `b` of a finished run, as `stride` values a row.
fn branch_rows(run: &Id, st: &Value, b: usize) -> Vec<f64> {
    copy_rows(run, st, b, 0)
}

/// The rows of branch `b` (`what` 0) or of its special points (`what` 1).
fn copy_rows(run: &Id, st: &Value, b: usize, what: u32) -> Vec<f64> {
    let stride = st["stride"].as_u64().unwrap() as usize;
    let jb = &st["branches"][b];
    let count = if what == 0 {
        jb["points"].as_u64().unwrap() as usize
    } else {
        jb["specials"].as_array().unwrap().len()
    };
    let mut buf = vec![f64::NAN; count * stride];
    let mut got = 0usize;
    let rc = unsafe {
        (api().run_copy)(
            run.0,
            b as u32,
            what,
            0,
            count,
            buf.as_mut_ptr(),
            buf.len(),
            &mut got,
        )
    };
    assert_eq!((rc, got), (0, count), "{}", last_error());
    buf
}

/// Starts ITER from `u0` over `n` and returns the finished status and the rows.
fn iterate(m: &Id, n: [i64; 2], u0: &[f64], p: &[f64]) -> (Value, Vec<f64>) {
    let json = format!(r#"{{"type":"ITER","tspan":[{},{}]}}"#, n[0], n[1]);
    let (rc, run) = start_with(m, &json, u0, p);
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(&run);
    let v = rows(&run, &st);
    (st, v)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn equilibrium(m: &Id, x: &[f64], p: &[f64]) -> i32 {
    solve(m, x, p).0
}

/// Calls `model_equilibrium` and returns the code, the point and the eigenvalues as
/// (re, im) pairs.
fn solve(m: &Id, x: &[f64], p: &[f64]) -> (i32, Vec<f64>, Vec<(f64, f64)>) {
    let opts = CString::new("{}").unwrap();
    let n = x.len();
    let mut u = vec![f64::NAN; n];
    let mut eig = vec![f64::NAN; 2 * n];
    let rc = unsafe {
        (api().model_equilibrium)(
            m.0,
            x.as_ptr(),
            n,
            p.as_ptr(),
            p.len(),
            opts.as_ptr(),
            u.as_mut_ptr(),
            u.len(),
            eig.as_mut_ptr(),
            eig.len(),
        )
    };
    let pairs = (0..n).map(|i| (eig[i], eig[n + i])).collect();
    (rc, u, pairs)
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}"
    );
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
        for ty in ["LP", "H", "LC", "LPC", "PD", "NS", "HOM", "IVP"] {
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
        let (rc, run) = start_with(&m, r#"{"type":"LP","free":"r"}"#, &[0.5], &[2.0]);
        assert_eq!(rc, UNSUPPORTED_MODEL, "{}", last_error());
        assert_eq!(run, 0);
        assert_eq!(last_error(), MAP_MODEL);
    }
}

#[test]
fn iter_on_a_map_handle_matches_a_hand_loop_bit_for_bit() {
    // The IR keeps the association of r*x*(1-x) as (r*x)*(1-x).
    let r = 2.5;
    let mut x = 0.1_f64;
    let mut want = vec![0.0, x];
    for k in 1..=50 {
        x = (r * x) * (1.0 - x);
        want.extend([f64::from(k), x]);
    }
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = compiled(text);
        let (st, v) = iterate(&m, [0, 50], &[0.1], &[r]);
        assert_eq!(st["state"], "done", "{st}");
        assert_eq!(st["kind"], "ITER");
        assert_eq!(st["n"], 1);
        assert_eq!(st["stride"], 2);
        let b = &st["branches"][0];
        assert_eq!(b["stop"], "End");
        assert_eq!(b["points"], 51);
        assert_eq!(b["specials"].as_array().unwrap().len(), 0);
        assert_eq!(bits(&v), bits(&want), "{text}");
    }
}

#[test]
fn iter_skips_the_check_for_t_and_t_is_the_iterate_number() {
    let m = compiled(SHIFT);
    let (_, v) = iterate(&m, [0, 4], &[0.0], &[]);
    assert_eq!(v, [0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 3.0, 4.0, 6.0]);
    let (_, v) = iterate(&m, [5, 9], &[0.0], &[]);
    assert_eq!(v, [5.0, 0.0, 6.0, 5.0, 7.0, 11.0, 8.0, 18.0, 9.0, 26.0]);
}

#[test]
fn an_overflow_ends_an_iter_run_with_error_and_keeps_the_rows() {
    let m = compiled(SQUARE);
    let (st, v) = iterate(&m, [0, 20], &[10.0], &[]);
    assert_eq!(st["state"], "done", "{st}");
    assert_eq!(st["branches"][0]["stop"], "Error");
    assert_eq!(st["branches"][0]["points"], 9);
    let x8 = (0..8).fold(10.0_f64, |x, _| x * x);
    assert_eq!(&v[16..], [8.0, x8]);
}

#[test]
fn iter_is_refused_on_an_ode_and_on_a_delay_model() {
    let m = compiled(LOGISTIC_ODE);
    let (rc, run) = start_with(&m, r#"{"type":"ITER","tspan":[0,4]}"#, &[0.2], &[2.0]);
    assert_eq!((rc, run), (BAD_PROBLEM, 0));
    assert_eq!(
        last_error(),
        "type ITER: the model is not a map; integrate an ODE with type IVP"
    );
    // ITER joins the run types that a delay model refuses, so the delay text comes first.
    let m = compiled(HUTCH);
    let (rc, run) = start_with(&m, r#"{"type":"ITER","tspan":[0,4]}"#, &[0.5], &[1.3, 1.0]);
    assert_eq!((rc, run), (2, 0), "{}", last_error());
    assert_eq!(
        last_error(),
        "the model has delay terms, and this entry is for ODE models"
    );
}

#[test]
fn iter_on_a_map_handle_that_was_not_compiled_gets_the_uncompiled_error() {
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = parse(text);
        let (rc, run) = start_with(&m, r#"{"type":"ITER","tspan":[0,4]}"#, &[0.5], &[2.0]);
        assert_eq!((rc, run), (2, 0), "{}", last_error());
        assert_eq!(last_error(), NOT_COMPILED);
        // EP is taken on a map too (SPEC 8v), so it gets the same error.
        let (rc, run) = start_with(&m, r#"{"type":"EP","free":"r"}"#, &[0.5], &[2.0]);
        assert_eq!((rc, run), (2, 0), "{}", last_error());
        assert_eq!(last_error(), NOT_COMPILED);
    }
}

#[test]
fn model_equilibrium_finds_the_fixed_point_of_the_logistic_map() {
    // x -> r x (1 - x) at r = 2 has the fixed point x = 1/2 with multiplier r (1 - 2x) = 0.
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = compiled(text);
        let (rc, u, eig) = solve(&m, &[0.9], &[2.0]);
        assert_eq!(rc, 0, "{}", last_error());
        close(u[0], 0.5, 1e-12, "x");
        close(eig[0].0, 0.0, 1e-10, "multiplier re");
        assert_eq!(eig[0].1, 0.0, "multiplier im");
    }
}

#[test]
fn model_equilibrium_finds_the_fixed_point_of_the_henon_map() {
    // (x, y) -> (1 - a x^2 + y, b x) has the fixed point a x^2 + (1 - b) x - 1 = 0, y = b x.
    // The Jacobian [[-2 a x, 1], [b, 0]] has the multipliers -a x +- sqrt(a^2 x^2 + b).
    let (a, b) = (1.4_f64, 0.3_f64);
    let x = (-(1.0 - b) + ((1.0 - b).powi(2) + 4.0 * a).sqrt()) / (2.0 * a);
    let y = b * x;
    let root = (a * a * x * x + b).sqrt();
    let mut want = [-a * x - root, -a * x + root];
    want.sort_by(f64::total_cmp);
    let m = compiled(HENON);
    let (rc, u, eig) = solve(&m, &[0.6, 0.2], &[a, b]);
    assert_eq!(rc, 0, "{}", last_error());
    close(u[0], x, 1e-12, "x");
    close(u[1], y, 1e-12, "y");
    // The order of the eigenvalues is not part of the contract.
    let mut got: Vec<f64> = eig.iter().map(|e| e.0).collect();
    got.sort_by(f64::total_cmp);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        close(g, w, 1e-10, &format!("multiplier {i}"));
    }
    for e in &eig {
        assert_eq!(e.1, 0.0, "multiplier im: {eig:?}");
    }
}

#[test]
fn model_equilibrium_refuses_a_map_that_reads_t() {
    let m = compiled(SHIFT);
    assert_eq!(
        equilibrium(&m, &[0.0], &[]),
        NON_AUTONOMOUS,
        "{}",
        last_error()
    );
}

#[test]
fn an_ep_run_on_the_logistic_map_gives_rows_with_four_tests() {
    // Branch 0 is x = 0 with multiplier r, so psi_PD = det(F_x + I) = r + 1. The BP at r = 1
    // switches to x = 1 - 1/r with multiplier 2 - r, which has a PD at r = 3, x = 2/3 with
    // b = 9 (SPEC 8v, test 2).
    for text in [LOGISTIC_MAP, LOGISTIC_DISCRETE] {
        let m = compiled(text);
        let json = r#"{"type":"EP","free":"r","options":{"bounds":[0,4]},"switch":"BP1"}"#;
        let (rc, run) = start_with(&m, json, &[0.0], &[0.5]);
        assert_eq!(rc, 0, "{}", last_error());
        let run = Id(run);
        let st = wait(&run);
        assert_eq!(st["state"], "done", "{st}");
        assert_eq!(st["kind"], "EP");
        assert_eq!(st["n"], 1);
        assert_eq!(st["stride"], 10);
        assert_eq!(st["codim2_labels"], serde_json::json!([]));
        let branches = st["branches"].as_array().unwrap();
        assert_eq!(branches.len(), 2, "{st}");
        // [alpha, x, mu_re, mu_im, psi_LP, psi_BP, psi_PD, psi_NS, h, newton_iters]
        let v = rows(&run, &st);
        assert!(v.len() / 10 > 5, "{st}");
        for row in v.chunks(10) {
            let r = row[0];
            close(row[1], 0.0, 1e-10, "x = 0 branch");
            close(row[2], r, 1e-10, "multiplier");
            assert_eq!(row[3], 0.0, "multiplier im");
            close(row[6], r + 1.0, 1e-10, "psi_PD");
            assert!(row[7].is_nan(), "psi_NS of a 1-D map: {row:?}");
        }
        // The refined row of each special point comes from `run_copy` with `what` 1.
        let labels = |b: usize| -> Vec<String> {
            branches[b]["specials"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["label"].as_str().unwrap().to_owned())
                .collect()
        };
        let l0 = labels(0);
        let k = l0.iter().position(|l| l == "BP").expect("BP on branch 0");
        assert_eq!(l0.iter().filter(|l| *l == "BP").count(), 1, "{st}");
        let s0 = copy_rows(&run, &st, 0, 1);
        close(s0[k * 10], 1.0, 1e-8, "BP alpha");
        close(s0[k * 10 + 1], 0.0, 1e-10, "BP x");
        let v1 = branch_rows(&run, &st, 1);
        for row in v1.chunks(10) {
            close(row[1], 1.0 - 1.0 / row[0], 1e-9, "x = 1 - 1/r branch");
            close(row[2], 2.0 - row[0], 1e-8, "multiplier 2 - r");
        }
        let l1 = labels(1);
        let k = l1.iter().position(|l| l == "PD").expect("PD on branch 1");
        assert_eq!(l1.iter().filter(|l| *l == "PD").count(), 1, "{st}");
        let s1 = copy_rows(&run, &st, 1, 1);
        close(s1[k * 10], 3.0, 1e-8, "PD alpha");
        close(s1[k * 10 + 1], 2.0 / 3.0, 1e-8, "PD x");
        let pd = &branches[1]["specials"][k];
        let coef = pd["coefficient"].as_f64().unwrap();
        close(coef, 9.0, 9e-7, "PD b");
        assert_eq!(pd["normalization"], "|q|=1, p'q=1");
    }
}

#[test]
fn detect_h_on_a_map_and_detect_pd_on_an_ode_are_bad_problems() {
    let m = compiled(LOGISTIC_MAP);
    let json = r#"{"type":"EP","free":"r","options":{"detect":{"h":true}}}"#;
    let (rc, run) = start_with(&m, json, &[0.0], &[0.5]);
    assert_eq!((rc, run), (BAD_PROBLEM, 0), "{}", last_error());
    assert_eq!(
        last_error(),
        "options.detect.h: a map has no Hopf points; detect pd and ns"
    );
    let m = compiled(LOGISTIC_ODE);
    for key in ["pd", "ns"] {
        let json = format!(r#"{{"type":"EP","free":"r","options":{{"detect":{{"{key}":true}}}}}}"#);
        let (rc, run) = start_with(&m, &json, &[0.0], &[0.5]);
        assert_eq!((rc, run), (BAD_PROBLEM, 0), "{key}: {}", last_error());
        assert_eq!(
            last_error(),
            format!("options.detect.{key}: only an EP run of a map detects PD and NS")
        );
    }
}

#[test]
fn an_ep_run_on_a_map_that_reads_t_is_refused() {
    let m = compiled(SHIFT_P);
    let (rc, run) = start_with(&m, r#"{"type":"EP","free":"a"}"#, &[0.0], &[0.5]);
    assert_eq!((rc, run), (NON_AUTONOMOUS, 0), "{}", last_error());
    assert_eq!(last_error(), "the model depends on t");
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
