//! The `thin` key of an IVP run through the C ABI (SPEC section 8af-2): kept rows equal the
//! unthinned rows bit for bit, the ends are kept, a running thinned run shows no rows, and
//! the key is refused on other runs and with bad values.

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const LORENZ: &str = "par s=10, r=28, b=2.6666666666666665\ninit x=1,y=1,z=1\n\
x'=s*(y-x)\ny'=x*(r-z)-y\nz'=x*y-b*z\naux zr=z-r\ndone\n";

const MAP: &str = "par r=3.7\ninit x=0.1\nx(t+1)=r*x*(1-x)\n\
aux d=x-0.5\naux kx=t*x\ndone\n";

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
        assert_eq!((api().free_handle)(self.0), 0);
    }
}

fn model(text: &str) -> Id {
    let mut id = 0u64;
    let rc = unsafe {
        (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, ptr::null_mut())
    };
    assert_eq!(rc, 0, "{}", last_error());
    let m = Id(id);
    let opts = CString::new(r#"{"evaluator":"jit"}"#).unwrap();
    let rc = unsafe { (api().model_compile)(m.0, opts.as_ptr()) };
    assert_eq!(rc, 0, "{}", last_error());
    m
}

fn start(m: &Id, json: &str, u0: &[f64], p: &[f64]) -> Id {
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
    assert_eq!(rc, 0, "{}", last_error());
    Id(run)
}

fn poll(run: &Id) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().run_poll)(run.0, &mut s) };
    assert_eq!(rc, 0, "{}", last_error());
    let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(s) };
    serde_json::from_str(&text).unwrap()
}

fn wait(run: &Id) -> Value {
    let t0 = Instant::now();
    loop {
        let v = poll(run);
        if v["state"] != "running" {
            return v;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "run did not finish"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Rows `start..start + count` of branch 0 as `stride` values a row.
fn copy(run: &Id, stride: usize, start: usize, count: usize) -> Vec<f64> {
    let mut buf = vec![f64::NAN; count * stride];
    let mut got = usize::MAX;
    let rc = unsafe {
        (api().run_copy)(
            run.0,
            0,
            0,
            start,
            count,
            buf.as_mut_ptr(),
            buf.len(),
            &mut got,
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
    buf.truncate(got * stride);
    buf
}

/// Runs `json` to the end and returns (stride, all values).
fn run_all(m: &Id, json: &str, u0: &[f64], p: &[f64]) -> (usize, Vec<f64>) {
    let run = start(m, json, u0, p);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    let stride = st["stride"].as_u64().unwrap() as usize;
    let rows = st["branches"][0]["points"].as_u64().unwrap() as usize;
    let v = copy(&run, stride, 0, rows);
    assert_eq!(v.len(), rows * stride);
    (stride, v)
}

/// The start error text of `json` on model `m`.
fn refusal(m: &Id, json: &str, u0: &[f64], p: &[f64]) -> String {
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
    assert_ne!(rc, 0, "{json} was accepted");
    last_error()
}

/// Asserts that every row of `thin` is a row of `full`, bit for bit and in order, and that
/// the first and last rows are kept. Returns the number of kept rows.
fn assert_subset(full: &[f64], thin: &[f64], w: usize) -> usize {
    let bits = |r: &[f64]| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let full: Vec<Vec<u64>> = full.chunks_exact(w).map(bits).collect();
    let thin: Vec<Vec<u64>> = thin.chunks_exact(w).map(bits).collect();
    assert_eq!(thin.first(), full.first());
    assert_eq!(thin.last(), full.last());
    let mut i = 0;
    for r in &thin {
        while i < full.len() && &full[i] != r {
            i += 1;
        }
        assert!(
            i < full.len(),
            "a kept row is not an unthinned row in order"
        );
        i += 1;
    }
    thin.len()
}

const LORENZ_RUN: &str = r#""type":"IVP","tspan":[0,20],"options":{"rtol":1e-9,"atol":1e-12}"#;
const LORENZ_U0: &[f64] = &[1.0, 1.0, 1.0];
const LORENZ_P: &[f64] = &[10.0, 28.0, 2.6666666666666665];

#[test]
fn thinned_rows_are_unthinned_rows_with_and_without_aux() {
    let m = model(LORENZ);
    for aux in ["", r#","aux":true"#] {
        let (w, full) = run_all(&m, &format!("{{{LORENZ_RUN}{aux}}}"), LORENZ_U0, LORENZ_P);
        let mut last = 0;
        for tol in ["1e-2", "1e-3", "1e-6"] {
            let json = format!(r#"{{{LORENZ_RUN}{aux},"thin":{tol}}}"#);
            let (w2, thin) = run_all(&m, &json, LORENZ_U0, LORENZ_P);
            assert_eq!(w2, w);
            let kept = assert_subset(&full, &thin, w);
            // At 1e-6 the curvature error of one step can keep every row.
            assert!(
                tol == "1e-6" || kept < full.len() / w,
                "tol {tol} keeps every row"
            );
            // The rule keeps a superset of rows at a smaller tol.
            assert!(kept >= last, "tol {tol} keeps fewer rows than a larger tol");
            last = kept;
            // A second run gives the same rows.
            let (_, again) = run_all(&m, &json, LORENZ_U0, LORENZ_P);
            assert!(
                again
                    .iter()
                    .zip(&thin)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}

#[test]
fn a_running_thinned_run_shows_no_rows() {
    let m = model(LORENZ);
    let long =
        r#"{"type":"IVP","tspan":[0,1e6],"thin":1e-3,"options":{"hmax":1e-4,"max_steps":3000000}}"#;
    let run = start(&m, long, LORENZ_U0, LORENZ_P);
    let t0 = Instant::now();
    let mut seen_running = false;
    loop {
        let st = poll(&run);
        if st["state"] != "running" {
            break;
        }
        seen_running = true;
        // Before the worker makes its branch there is nothing to copy.
        if st["branches"].as_array().is_some_and(|b| !b.is_empty()) {
            assert_eq!(st["branches"][0]["points"].as_u64().unwrap_or(0), 0, "{st}");
            assert_eq!(copy(&run, 4, 0, 10).len(), 0);
        }
        if t0.elapsed() > Duration::from_millis(300) {
            assert_eq!((api().run_cancel)(run.0), 0);
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "run did not finish"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(seen_running);
    // A cancelled run is thinned too: its first row is the start, and it has rows.
    let st = wait(&run);
    assert_eq!(st["state"], "cancelled", "{st}");
    let rows = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert!(rows >= 2, "{st}");
    let v = copy(&run, 4, 0, rows);
    assert_eq!(&v[..4], &[0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn the_thin_key_is_refused_on_other_runs_and_with_bad_values() {
    let m = model(LORENZ);
    for bad in [
        "0", "1", "-1", "1.5", "1e-400", "\"x\"", "[1e-3]", "null", "true",
    ] {
        let json = format!(r#"{{{LORENZ_RUN},"thin":{bad}}}"#);
        let e = refusal(&m, &json, LORENZ_U0, LORENZ_P);
        assert!(e.contains("thin"), "{bad}: {e}");
    }
    let e = refusal(&m, r#"{"type":"EP","thin":1e-3}"#, LORENZ_U0, LORENZ_P);
    assert!(e.contains("thin"), "{e}");
    let map = model(MAP);
    let e = refusal(
        &map,
        r#"{"type":"ITER","tspan":[0,10],"thin":1e-3}"#,
        &[0.1],
        &[3.7],
    );
    assert!(e.contains("thin"), "{e}");
}
