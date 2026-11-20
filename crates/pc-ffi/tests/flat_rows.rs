//! The rows of IVP and ITER runs through the C ABI, pinned by hashes of their bits as the
//! base commit of SPEC section 8af-1 gave them, with and without `"aux": true`, and the rows
//! of a cancelled run (SPEC section 8af-1, acceptance tests).

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::time::{Duration, Instant};

use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const VDP: &str = "par mu=1\ninit x=2,y=0\nx'=y\ny'=mu*(1-x*x)*y-x\n\
aux e=0.5*(x*x+y*y)\naux tx=t*x\ndone\n";

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

/// FNV-1a over the bits of each value, in row order.
fn hash(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Runs `json` to the end and returns (stride, rows, all values).
fn run_all(m: &Id, json: &str, u0: &[f64], p: &[f64]) -> (usize, usize, Vec<f64>) {
    let run = start(m, json, u0, p);
    let st = wait(&run);
    assert_eq!(st["state"], "done", "{st}");
    let stride = st["stride"].as_u64().unwrap() as usize;
    let rows = st["branches"][0]["points"].as_u64().unwrap() as usize;
    let v = copy(&run, stride, 0, rows);
    assert_eq!(v.len(), rows * stride);
    // A copy from the middle gives the same values as the full copy.
    let mid = rows / 3;
    let part = copy(&run, stride, mid, rows - mid);
    assert!(
        part.iter()
            .zip(&v[mid * stride..])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    (stride, rows, v)
}

/// (model, problem without the aux key, u0, params, n, naux, rows, hash plain, hash aux,
/// pinned on this platform).
type Case = (
    &'static str,
    &'static str,
    &'static [f64],
    &'static [f64],
    usize,
    usize,
    usize,
    u64,
    u64,
    bool,
);

const IVP_PINNED: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

const CASES: [Case; 3] = [
    (
        VDP,
        r#""type":"IVP","tspan":[0,50]"#,
        &[2.0, 0.0],
        &[1.0],
        2,
        2,
        ROWS_VDP,
        HASH_VDP,
        HASH_VDP_AUX,
        IVP_PINNED,
    ),
    (
        LORENZ,
        r#""type":"IVP","tspan":[0,20],"options":{"rtol":1e-9,"atol":1e-12}"#,
        &[1.0, 1.0, 1.0],
        &[10.0, 28.0, 2.6666666666666665],
        3,
        1,
        ROWS_LORENZ,
        HASH_LORENZ,
        HASH_LORENZ_AUX,
        IVP_PINNED,
    ),
    (
        MAP,
        r#""type":"ITER","tspan":[0,2000]"#,
        &[0.1],
        &[3.7],
        1,
        2,
        ROWS_MAP,
        HASH_MAP,
        HASH_MAP_AUX,
        true,
    ),
];

// Pinned on e58b9d4, before the flat store: every row of each case, hashed by its bits.
// The IVP step control calls `powf`, which each platform's libm rounds its own way, so the
// IVP values are pinned only on x86_64 Linux. The map rows use + - * / only and hold on
// every platform.
const ROWS_VDP: usize = 1098;
const HASH_VDP: u64 = 0x0ce8_ddb8_6f88_840d;
const HASH_VDP_AUX: u64 = 0x12a0_7722_9ad1_eab1;
const ROWS_LORENZ: usize = 3140;
const HASH_LORENZ: u64 = 0x4572_7d8d_5084_b7dc;
const HASH_LORENZ_AUX: u64 = 0xa0f8_ebe1_3484_eebf;
const ROWS_MAP: usize = 2001;
const HASH_MAP: u64 = 0xac6b_118f_341e_19ff;
const HASH_MAP_AUX: u64 = 0x238e_ffae_a70b_d313;

#[test]
fn rows_equal_the_base_commit_with_and_without_aux() {
    let mut report = String::new();
    let mut ok = true;
    for (text, head, u0, p, n, naux, rows, plain, with_aux, pinned) in CASES {
        let m = model(text);
        let (s0, r0, v0) = run_all(&m, &format!("{{{head}}}"), u0, p);
        let (s1, r1, v1) = run_all(&m, &format!(r#"{{{head},"aux":true}}"#), u0, p);
        assert_eq!((s0, s1), (n + 1, n + 1 + naux));
        assert_eq!(r0, r1);
        // The aux run carries the plain row in its first n + 1 entries.
        for (a, b) in v1.chunks_exact(s1).zip(v0.chunks_exact(s0)) {
            assert!(
                a[..s0]
                    .iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
        let (h0, h1) = (hash(&v0), hash(&v1));
        report += &format!("{head}: rows {r0}, hash {h0:#018x}, aux {h1:#018x}\n");
        ok &= !pinned || (r0 == rows && h0 == plain && h1 == with_aux);
    }
    assert!(ok, "rows differ from the base commit:\n{report}");
}

/// A cancelled run keeps a prefix of the rows of the full run, and its poll count equals the
/// rows that `run_copy` gives.
#[test]
fn a_cancelled_run_keeps_a_prefix_of_the_full_run() {
    let m = model(LORENZ);
    let (u0, p) = (&[1.0, 1.0, 1.0], &[10.0, 28.0, 2.6666666666666665]);
    let long = r#"{"type":"IVP","tspan":[0,1e6],"aux":true,"options":{"hmax":1e-4,"max_steps":100000000}}"#;
    let run = start(&m, long, u0, p);
    let t0 = Instant::now();
    while poll(&run)["branches"][0]["points"].as_u64().unwrap_or(0) < 2000 {
        assert!(t0.elapsed() < Duration::from_secs(60), "run did not start");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!((api().run_cancel)(run.0), 0);
    let st = wait(&run);
    assert_eq!(st["state"], "cancelled", "{st}");
    let stride = st["stride"].as_u64().unwrap() as usize;
    let rows = st["branches"][0]["points"].as_u64().unwrap() as usize;
    assert!(rows >= 2000);
    let cut = copy(&run, stride, 0, rows + 10);
    assert_eq!(
        cut.len(),
        rows * stride,
        "the poll count is the copied count"
    );
    let full = format!(
        r#"{{"type":"IVP","tspan":[0,1e6],"aux":true,"options":{{"hmax":1e-4,"max_steps":{rows}}}}}"#
    );
    let ref_run = start(&m, &full, u0, p);
    let rst = wait(&ref_run);
    let ref_rows = rst["branches"][0]["points"].as_u64().unwrap() as usize;
    assert!(ref_rows >= rows, "{rst}");
    let want = copy(&ref_run, stride, 0, rows);
    assert!(
        cut.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the cancelled rows are a prefix of the full run"
    );
}
