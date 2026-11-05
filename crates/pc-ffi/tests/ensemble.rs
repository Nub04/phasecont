//! Ensemble runs through the table from `pc_api_v1()`. Every test runs on the CPU backend,
//! so no GPU or driver is needed.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

use pckern::ensemble::{
    PC_ENS_ACCEPTED, PC_ENS_REJECTED, PC_ENS_STATES, PC_ENS_STATUS, PC_ENS_T_REACHED, PC_ENS_TIMES,
};
use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const PC_ERR_PANIC: i32 = 1;
const PC_ERR_BAD_ARG: i32 = 2;
const PC_ERR_BAD_HANDLE: i32 = 3;
const PC_ERR_BAD_PROBLEM: i32 = 6;
const PC_ERR_GPU_UNAVAILABLE: i32 = 9;

const DECAY: &str = "par k=1\nx'=-k*x\ndone\n";
const LORENZ: &str = "par s=10, r=28, b=2.6666666666666665\n\
init x=1,y=1,z=1\n\
x'=s*(y-x)\n\
y'=x*(r-z)-y\n\
z'=x*y-b*z\n\
done\n";
const LORENZ_P: [f64; 3] = [10.0, 28.0, 2.6666666666666665];

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

/// The job JSON of a final-state RK4 run on the CPU.
fn job(extra: &str) -> String {
    format!(
        r#"{{"backend":"cpu","precision":"double","t0":0,"t1":1,"method":{{"type":"rk4","dt":0.01}},"output":"final","bounds":null,"max_steps":100000,"params":"shared"{extra}}}"#
    )
}

/// Returns (code, ensemble id). The id is 0 unless the call succeeds.
fn start_with(m: &Id, json: &str, x0: &[f64], p: &[f64]) -> (i32, u64) {
    start_raw(m.0, json, x0, p)
}

/// Like `start_with`, for a raw id that must not be freed afterwards.
fn start_raw(model: u64, json: &str, x0: &[f64], p: &[f64]) -> (i32, u64) {
    let c = CString::new(json).unwrap();
    let mut ens = 0u64;
    let rc = unsafe {
        (api().ens_start)(
            model,
            c.as_ptr(),
            x0.as_ptr(),
            x0.len(),
            p.as_ptr(),
            p.len(),
            &mut ens,
        )
    };
    (rc, ens)
}

fn start(m: &Id, json: &str, x0: &[f64], p: &[f64]) -> Id {
    let (rc, ens) = start_with(m, json, x0, p);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(ens, 0);
    Id(ens)
}

/// A run that cannot finish within a test: three Lorenz members over a huge span.
fn long_run(m: &Id) -> Id {
    let json = r#"{"backend":"cpu","precision":"double","t0":0,"t1":1e5,"method":{"type":"rk4","dt":0.0001},"output":"final","bounds":null,"max_steps":4000000000,"params":"shared"}"#;
    start(
        m,
        json,
        &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0],
        &LORENZ_P,
    )
}

fn poll_raw(ens: u64) -> (i32, Option<Value>) {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().ens_poll)(ens, &mut s) };
    if s.is_null() {
        return (rc, None);
    }
    let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(s) };
    (rc, Some(serde_json::from_str(&text).unwrap()))
}

/// Polls every 5 ms until the state is not "running".
fn wait(ens: &Id) -> Value {
    let t0 = Instant::now();
    loop {
        let (rc, v) = poll_raw(ens.0);
        assert_eq!(rc, 0, "{}", last_error());
        let v = v.unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "run did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Returns (code, values) for a buffer of `len` values.
fn copy(ens: u64, what: u32, len: usize) -> (i32, Vec<f64>) {
    let mut buf = vec![f64::NAN; len];
    let rc = unsafe { (api().ens_copy)(ens, what, buf.as_mut_ptr(), buf.len()) };
    (rc, buf)
}

fn copied(ens: u64, what: u32, len: usize) -> Vec<f64> {
    let (rc, buf) = copy(ens, what, len);
    assert_eq!(rc, 0, "{}", last_error());
    buf
}

fn devices_raw() -> (i32, Option<Value>) {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().ens_devices)(&mut s) };
    if s.is_null() {
        return (rc, None);
    }
    let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(s) };
    (rc, Some(serde_json::from_str(&text).unwrap()))
}

#[test]
fn devices_lists_the_cpu_backend() {
    let (rc, v) = devices_raw();
    assert_eq!(rc, 0, "{}", last_error());
    let v = v.unwrap();
    let backends = v["backends"].as_array().unwrap();
    let cpu = backends.iter().find(|b| b["backend"] == "cpu").unwrap();
    assert_eq!(cpu["state"], "available");
    let devices = v["devices"].as_array().unwrap();
    let cpu = devices.iter().find(|d| d["backend"] == "cpu").unwrap();
    assert_eq!(cpu["index"], 0);
    assert!(cpu["fp64"].as_bool().unwrap());
    for key in ["name", "driver", "fp32", "memory_bytes"] {
        assert!(cpu.get(key).is_some(), "missing {key}");
    }
}

#[test]
fn devices_with_a_null_out_pointer_gives_2() {
    let rc = unsafe { (api().ens_devices)(ptr::null_mut()) };
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(last_error().contains("NULL"), "{}", last_error());
}

#[test]
fn a_cpu_run_finishes_and_copies_every_array() {
    let m = parse(DECAY);
    let ens = start(&m, &job(""), &[1.0, 2.0, 3.0], &[1.0]);
    let v = wait(&ens);
    assert_eq!(v["state"], "done");
    assert_eq!(v["members"], 3);
    assert_eq!(v["n"], 1);
    assert_eq!(v["samples"], 1);
    assert_eq!(v["counts"]["ok"], 3);
    assert_eq!(v["counts"]["bounds"], 0);
    let label = &v["label"];
    assert_eq!(label["backend"], "cpu");
    assert_eq!(label["precision"], "double");
    assert_eq!(label["single_precision"], false);
    assert!(!label["device"].as_str().unwrap().is_empty());
    assert!(!label["driver"].as_str().unwrap().is_empty());

    let states = copied(ens.0, PC_ENS_STATES, 3);
    for (x, x0) in states.iter().zip([1.0, 2.0, 3.0]) {
        let want = x0 * (-1.0f64).exp();
        assert!((x - want).abs() < 1e-9, "{x} vs {want}");
    }
    assert_eq!(copied(ens.0, PC_ENS_TIMES, 1), [1.0]);
    assert_eq!(copied(ens.0, PC_ENS_STATUS, 3), [0.0, 0.0, 0.0]);
    assert_eq!(copied(ens.0, PC_ENS_T_REACHED, 3), [1.0, 1.0, 1.0]);
    // RK4 takes 100 steps of 0.01 and rejects none.
    assert_eq!(copied(ens.0, PC_ENS_ACCEPTED, 3), [100.0, 100.0, 100.0]);
    assert_eq!(copied(ens.0, PC_ENS_REJECTED, 3), [0.0, 0.0, 0.0]);
    // A second copy of the same array gives the same numbers.
    assert_eq!(copied(ens.0, PC_ENS_STATES, 3), states);
}

#[test]
fn single_precision_is_labeled() {
    let m = parse(DECAY);
    let json = job("").replace(r#""precision":"double""#, r#""precision":"single""#);
    let ens = start(&m, &json, &[1.0], &[1.0]);
    let v = wait(&ens);
    assert_eq!(v["label"]["precision"], "single");
    assert_eq!(v["label"]["single_precision"], true);
}

#[test]
fn samples_are_stored_member_major_then_sample_then_state() {
    let m = parse(LORENZ);
    let json = job("")
        .replace(r#""output":"final""#, r#""output":{"samples":[0.5,1.0]}"#)
        .replace(r#""dt":0.01"#, r#""dt":0.001"#);
    let x0 = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
    let ens = start(&m, &json, &x0, &LORENZ_P);
    let v = wait(&ens);
    assert_eq!(v["members"], 2);
    assert_eq!(v["n"], 3);
    assert_eq!(v["samples"], 2);
    assert_eq!(copied(ens.0, PC_ENS_TIMES, 2), [0.5, 1.0]);
    let states = copied(ens.0, PC_ENS_STATES, 2 * 2 * 3);
    // The last sample of a member is the state of a run to t1 alone.
    let one = start(&m, &job(""), &x0, &LORENZ_P);
    let _ = wait(&one);
    let at_t1 = copied(one.0, PC_ENS_STATES, 6);
    for member in 0..2 {
        let last = &states[(member * 2 + 1) * 3..(member * 2 + 2) * 3];
        for (a, b) in last.iter().zip(&at_t1[member * 3..member * 3 + 3]) {
            // Different step size (0.001 against 0.01): the two agree to RK4 accuracy only.
            assert!((a - b).abs() < 1e-2 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}

#[test]
fn per_member_parameters_are_read_member_major() {
    let m = parse(DECAY);
    let json = job("").replace(r#""params":"shared""#, r#""params":"per_member""#);
    let ens = start(&m, &json, &[1.0, 1.0], &[1.0, 2.0]);
    let _ = wait(&ens);
    let s = copied(ens.0, PC_ENS_STATES, 2);
    assert!((s[0] - (-1.0f64).exp()).abs() < 1e-9);
    assert!((s[1] - (-2.0f64).exp()).abs() < 1e-9);
}

#[test]
fn a_copy_before_the_run_finishes_gives_2() {
    let m = parse(LORENZ);
    let ens = long_run(&m);
    let (rc, _) = copy(ens.0, PC_ENS_STATES, 3);
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(last_error().contains("not finished"), "{}", last_error());
}

#[test]
fn cancel_ends_a_running_run_as_cancelled() {
    let m = parse(LORENZ);
    let ens = long_run(&m);
    assert_eq!(poll_raw(ens.0).1.unwrap()["state"], "running");
    assert_eq!((api().ens_cancel)(ens.0), 0);
    let v = wait(&ens);
    assert_eq!(v["state"], "cancelled");
    let (rc, _) = copy(ens.0, PC_ENS_STATES, 9);
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(last_error().contains("cancelled"), "{}", last_error());
    // Cancelling again is allowed.
    assert_eq!((api().ens_cancel)(ens.0), 0);
}

#[test]
fn freeing_a_running_ensemble_cancels_and_joins_the_worker() {
    let m = parse(LORENZ);
    let mut ens = long_run(&m);
    let t0 = Instant::now();
    assert_eq!((api().free_handle)(ens.0), 0);
    assert!(
        t0.elapsed() < Duration::from_secs(30),
        "free_handle waited for the whole run"
    );
    let gone = ens.0;
    ens.0 = 0;
    assert_eq!((api().free_handle)(gone), PC_ERR_BAD_HANDLE);
    assert_eq!(poll_raw(gone).0, PC_ERR_BAD_HANDLE);
}

#[test]
fn a_worker_panic_reaches_poll_and_copy() {
    let ens = Id(pckern::debug_ensemble_panic());
    let t0 = Instant::now();
    let rc = loop {
        let (rc, v) = poll_raw(ens.0);
        if rc != 0 {
            assert!(v.is_none());
            break rc;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "the panic never came"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(rc, PC_ERR_PANIC);
    let text = last_error();
    assert!(text.starts_with("panic: "), "{text}");
    // The error stays: every later poll and copy returns it.
    assert_eq!(poll_raw(ens.0).0, PC_ERR_PANIC);
    assert_eq!(copy(ens.0, PC_ENS_STATES, 1).0, PC_ERR_PANIC);
    assert_eq!((api().ens_cancel)(ens.0), 0);
}

#[test]
fn bad_and_wrong_kind_handles_give_3() {
    let (rc, _) = copy(0, PC_ENS_STATES, 0);
    assert_eq!(rc, PC_ERR_BAD_HANDLE);
    assert_eq!(poll_raw(0x1234).0, PC_ERR_BAD_HANDLE);
    assert_eq!((api().ens_cancel)(0x1234), PC_ERR_BAD_HANDLE);

    let m = parse(DECAY);
    assert_eq!(poll_raw(m.0).0, PC_ERR_BAD_HANDLE);
    assert!(last_error().contains("not a ensemble"), "{}", last_error());
    assert_eq!((api().ens_cancel)(m.0), PC_ERR_BAD_HANDLE);
    let dbg = pckern::debug_handle();
    assert_eq!(poll_raw(dbg).0, PC_ERR_BAD_HANDLE);
    assert_eq!((api().free_handle)(dbg), 0);

    // An ensemble id is not a model.
    let ens = start(&m, &job(""), &[1.0], &[1.0]);
    let (rc, _) = start_raw(ens.0, &job(""), &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_BAD_HANDLE);
}

#[test]
fn a_buffer_of_the_wrong_length_gives_2() {
    let m = parse(DECAY);
    let ens = start(&m, &job(""), &[1.0, 2.0, 3.0], &[1.0]);
    let _ = wait(&ens);
    for (what, want) in [
        (PC_ENS_STATES, 3),
        (PC_ENS_TIMES, 1),
        (PC_ENS_STATUS, 3),
        (PC_ENS_T_REACHED, 3),
        (PC_ENS_ACCEPTED, 3),
        (PC_ENS_REJECTED, 3),
    ] {
        for len in [want - 1, want + 1] {
            let (rc, _) = copy(ens.0, what, len);
            assert_eq!(rc, PC_ERR_BAD_ARG, "selector {what}, length {len}");
            assert!(
                last_error().contains(&format!("expected {want}")),
                "{}",
                last_error()
            );
        }
    }
    let (rc, _) = copy(ens.0, 6, 3);
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(last_error().contains("selector"), "{}", last_error());
}

#[test]
fn a_tolerance_below_single_precision_is_a_bad_problem_with_its_own_text() {
    let m = parse(DECAY);
    let json = job("")
        .replace(r#""precision":"double""#, r#""precision":"single""#)
        .replace(
            r#""method":{"type":"rk4","dt":0.01}"#,
            r#""method":{"type":"dp5","rtol":1e-12,"atol":1e-12,"hmax":null}"#,
        );
    let (rc, ens) = start_with(&m, &json, &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
    assert_eq!(ens, 0);
    assert!(
        last_error().starts_with("tolerance too tight"),
        "{}",
        last_error()
    );
}

#[test]
fn a_bad_job_is_a_bad_problem_with_its_own_text() {
    let m = parse(DECAY);
    // Sizes the model rejects: two parameters for a model with one.
    let (rc, _) = start_with(&m, &job(""), &[1.0], &[1.0, 2.0]);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
    assert!(
        last_error().starts_with("bad ensemble job"),
        "{}",
        last_error()
    );
    // x0 that is not a whole number of members.
    let lorenz = parse(LORENZ);
    let (rc, _) = start_with(&lorenz, &job(""), &[1.0, 2.0, 3.0, 4.0], &LORENZ_P);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
    assert!(last_error().contains("multiple of 3"), "{}", last_error());
    // No members at all.
    let (rc, _) = start_with(&m, &job(""), &[], &[1.0]);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
    assert!(last_error().contains("at least one"), "{}", last_error());
}

#[test]
fn job_keys_are_checked_by_name() {
    let m = parse(DECAY);
    let cases = [
        (job(r#","colour":1"#), "unknown key colour"),
        (job("").replace(r#""t1":1,"#, ""), "missing key t1"),
        (
            job("").replace(r#""precision":"double","#, ""),
            "missing key precision",
        ),
        (
            job("").replace(r#""max_steps":100000,"#, ""),
            "missing key max_steps",
        ),
        (
            job("").replace(r#""params":"shared""#, r#""params":"some""#),
            "params must be",
        ),
        (
            job("").replace(r#","params":"shared""#, ""),
            "missing key params",
        ),
        (job("").replace("\"cpu\"", "\"tpu\""), "backend must be"),
    ];
    for (json, want) in cases {
        let (rc, ens) = start_with(&m, &json, &[1.0], &[1.0]);
        assert_eq!(rc, PC_ERR_BAD_PROBLEM, "{json}");
        assert_eq!(ens, 0);
        assert!(last_error().contains(want), "{json}: {}", last_error());
    }
    let (rc, _) = start_with(&m, "[1]", &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
    let (rc, _) = start_with(&m, "{", &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_BAD_PROBLEM);
}

#[test]
fn null_pointers_give_2() {
    let m = parse(DECAY);
    let c = CString::new(job("")).unwrap();
    let x0 = [1.0];
    let p = [1.0];
    let mut ens = 0u64;
    let call = |json: *const c_char, x: *const f64, par: *const f64, out: *mut u64| unsafe {
        (api().ens_start)(m.0, json, x, 1, par, 1, out)
    };
    assert_eq!(
        call(ptr::null(), x0.as_ptr(), p.as_ptr(), &mut ens),
        PC_ERR_BAD_ARG
    );
    assert_eq!(
        call(c.as_ptr(), ptr::null(), p.as_ptr(), &mut ens),
        PC_ERR_BAD_ARG
    );
    assert_eq!(
        call(c.as_ptr(), x0.as_ptr(), ptr::null(), &mut ens),
        PC_ERR_BAD_ARG
    );
    assert_eq!(
        call(c.as_ptr(), x0.as_ptr(), p.as_ptr(), ptr::null_mut()),
        PC_ERR_BAD_ARG
    );
    assert_eq!(ens, 0);
    assert_eq!(
        unsafe { (api().ens_poll)(1, ptr::null_mut()) },
        PC_ERR_BAD_ARG
    );
    assert_eq!(
        unsafe { (api().ens_copy)(1, PC_ENS_STATES, ptr::null_mut(), 1) },
        PC_ERR_BAD_ARG
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn a_backend_that_is_not_built_gives_gpu_unavailable() {
    let m = parse(DECAY);
    let json = job("").replace("\"cpu\"", "\"metal\"");
    let (rc, ens) = start_with(&m, &json, &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_GPU_UNAVAILABLE);
    assert_eq!(ens, 0);
    assert!(last_error().contains("metal"), "{}", last_error());
}

#[test]
fn a_device_past_the_end_gives_gpu_unavailable() {
    let m = parse(DECAY);
    let json = job(r#","device":99"#);
    let (rc, ens) = start_with(&m, &json, &[1.0], &[1.0]);
    assert_eq!(rc, PC_ERR_GPU_UNAVAILABLE);
    assert_eq!(ens, 0);
    assert!(last_error().contains("device 99"), "{}", last_error());
}
