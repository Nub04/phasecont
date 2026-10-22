//! S3 checks: the run report, argument rejection, and the C ABI round trip
//! through the same function table that the MEX gateway uses.

use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char};
use std::ptr::{dangling_mut, null, null_mut};

use pckern::ffi::{Error, PC_ERR_BAD_ARG, PC_ERR_BAD_HANDLE, PC_OK};
use pckern::{PcApiV1, pc_api_v1};

fn api() -> &'static PcApiV1 {
    unsafe { &*pc_api_v1() }
}

fn last_error() -> String {
    let p = (api().last_error)();
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Every number that follows `"key":` in a JSON text.
fn values(json: &str, key: &str) -> Vec<f64> {
    let pat = format!("\"{key}\":");
    json.match_indices(&pat)
        .map(|(i, _)| {
            let rest = &json[i + pat.len()..];
            let end = rest.find([',', '}']).unwrap();
            rest[..end].parse().unwrap()
        })
        .collect()
}

fn build(args: &str) -> (u64, usize, usize) {
    let args = CString::new(args).unwrap();
    let (mut h, mut n, mut nnz) = (0u64, 0usize, 0usize);
    let rc = unsafe { (api().s3_build)(args.as_ptr(), &mut h, &mut n, &mut nnz) };
    assert_eq!(rc, PC_OK, "{}", last_error());
    (h, n, nnz)
}

struct Copied {
    i: Vec<f64>,
    j: Vec<f64>,
    v: Vec<f64>,
    b: Vec<f64>,
}

fn copy(h: u64, n: usize, nnz: usize) -> Copied {
    let mut c = Copied {
        i: vec![0.0; nnz],
        j: vec![0.0; nnz],
        v: vec![0.0; nnz],
        b: vec![0.0; n],
    };
    let rc = unsafe {
        (api().s3_copy)(
            h,
            n,
            nnz,
            c.i.as_mut_ptr(),
            c.j.as_mut_ptr(),
            c.v.as_mut_ptr(),
            c.b.as_mut_ptr(),
        )
    };
    assert_eq!(rc, PC_OK, "{}", last_error());
    c
}

/// Calls `s3_solve` and returns the code, x and the JSON text.
fn solve(h: u64, n: usize, args: &str) -> (i32, Vec<f64>, Option<String>) {
    let args = CString::new(args).unwrap();
    let mut x = vec![f64::NAN; n];
    let mut out: *mut c_char = dangling_mut();
    let rc = unsafe { (api().s3_solve)(h, n, args.as_ptr(), x.as_mut_ptr(), &mut out) };
    let json = if out.is_null() {
        None
    } else {
        let s = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_owned();
        unsafe { (api().free_string)(out) };
        Some(s)
    };
    (rc, x, json)
}

/// ‖Ax - b‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) from the copied triplets.
fn backward_error(c: &Copied, x: &[f64]) -> f64 {
    let n = c.b.len();
    let mut r: Vec<f64> = c.b.iter().map(|b| -b).collect();
    let mut row_abs = vec![0.0f64; n];
    for k in 0..c.v.len() {
        let (i, j) = (c.i[k] as usize - 1, c.j[k] as usize - 1);
        r[i] += c.v[k] * x[j];
        row_abs[i] += c.v[k].abs();
    }
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    inf(&r) / (inf(&row_abs) * inf(x) + inf(&c.b))
}

const METHODS: [&str; 3] = ["sparse_lu", "condensed_sparse", "condensed_dense"];

#[test]
fn run_reports_pass_on_a_small_problem() {
    let json = pckern::s3_linalg::run("N=3 threads=2 reps=1").unwrap();
    assert!(json.starts_with("{\"spike\":\"s3\",\"target\":"), "{json}");
    assert!(json.contains("\"nr\":54,"), "{json}");
    assert!(json.contains("\"nred\":18,"), "{json}");
    assert!(json.contains("\"pass\":true}"), "{json}");
    assert!(
        json.contains("\"sparse_threads_bitwise_equal\":true"),
        "{json}"
    );
    let errs = values(&json, "backward_error");
    assert_eq!(errs.len(), 5, "{json}");
    assert!(errs.iter().all(|e| *e <= 1e-13), "{errs:?}");
    assert!(values(&json, "cond1_est")[0] >= 1.0, "{json}");
    assert!(values(&json, "max_pairwise_rel_diff")[0] <= 1e-10, "{json}");
    assert_eq!(json.matches("{\"name\":\"first\"").count(), 5, "{json}");
    assert_eq!(json.matches("{\"name\":\"backsub\"").count(), 4, "{json}");
}

#[test]
fn run_with_one_thread_leaves_out_the_threaded_entries() {
    let json = pckern::s3_linalg::run("n=2 m=3 N=4 threads=1 reps=1").unwrap();
    assert_eq!(values(&json, "backward_error").len(), 3, "{json}");
    assert!(json.contains("\"pass\":true}"), "{json}");
}

#[test]
fn run_leaves_out_condensed_dense_above_order_5000() {
    // nred = 1 * 5001 + 2 = 5003.
    let json = pckern::s3_linalg::run("n=1 m=2 N=5000 threads=2 reps=1").unwrap();
    assert!(json.contains("\"nred\":5003,"), "{json}");
    assert_eq!(values(&json, "backward_error").len(), 3, "{json}");
    assert!(!json.contains("condensed_dense"), "{json}");
    assert!(json.contains("\"pass\":true}"), "{json}");
}

#[test]
fn run_rejects_bad_arguments() {
    let bad = [
        "n=0",
        "n=33",
        "m=1",
        "m=11",
        "N=0",
        "N=20001",
        "T=0",
        "T=-1",
        "T=abc",
        "T=inf",
        "T=NaN",
        "seed=-1",
        "foo=1",
        "threads=0",
        "threads=257",
        "reps=0",
        "reps=10001",
        "method=sparse_lu",
    ];
    for args in bad {
        match pckern::s3_linalg::run(args) {
            Err(Error::BadArg(_)) => {}
            other => panic!("{args}: expected BadArg, got {other:?}"),
        }
    }
}

#[test]
fn ffi_round_trip_for_every_method() {
    let (h, n, nnz) = build("n=3 m=4 N=5 T=2.5 seed=7");
    assert_eq!(n, 3 * (5 * 4 + 1) + 2);
    let c = copy(h, n, nnz);
    let mut pairs = HashSet::new();
    let mut last_row = 0.0;
    for k in 0..nnz {
        let (i, j) = (c.i[k], c.j[k]);
        assert!(i.fract() == 0.0 && j.fract() == 0.0, "{i} {j}");
        assert!((1.0..=n as f64).contains(&i) && (1.0..=n as f64).contains(&j));
        assert!(i >= last_row, "rows must ascend");
        last_row = i;
        assert!(pairs.insert((i as u64, j as u64)), "duplicate ({i}, {j})");
    }
    assert!(c.b.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    for method in METHODS {
        for threads in [1, 3] {
            let args = format!("method={method} threads={threads} reps=2");
            let (rc, x, json) = solve(h, n, &args);
            assert_eq!(rc, PC_OK, "{args}: {}", last_error());
            let json = json.unwrap();
            let head = format!("{{\"spike\":\"s3\",\"method\":\"{method}\",\"threads\":{threads},");
            assert!(json.starts_with(&head), "{json}");
            assert!(
                json.contains(&format!("\"nnz\":{nnz},\"nred\":20,")),
                "{json}"
            );
            assert!(x.iter().all(|v| v.is_finite()), "{args}");
            let be = backward_error(&c, &x);
            assert!(be <= 1e-13, "{args}: backward error {be}");
            assert!(values(&json, "backward_error")[0] <= 1e-13, "{json}");
        }
    }
    assert_eq!((api().free_handle)(h), PC_OK);
    assert_eq!((api().free_handle)(h), PC_ERR_BAD_HANDLE);
}

#[test]
fn ffi_rejects_bad_sizes_arguments_and_null_pointers() {
    let (h, n, nnz) = build("N=2");
    let mut buf = vec![0.0; nnz.max(n) + 1];
    let p = buf.as_mut_ptr();
    let copy_rc = |n, nnz, i: *mut f64| unsafe { (api().s3_copy)(h, n, nnz, i, p, p, p) };
    assert_eq!(copy_rc(n + 1, nnz, p), PC_ERR_BAD_ARG);
    assert!(last_error().contains(&n.to_string()), "{}", last_error());
    assert_eq!(copy_rc(n, nnz - 1, p), PC_ERR_BAD_ARG);
    assert_eq!(copy_rc(n, nnz, null_mut()), PC_ERR_BAD_ARG);
    for args in [
        "method=foo",
        "threads=0",
        "threads=257",
        "reps=0",
        "n=4",
        "x",
    ] {
        let (rc, _, json) = solve(h, n, args);
        assert_eq!(rc, PC_ERR_BAD_ARG, "{args}");
        assert!(json.is_none(), "{args}: out_json must stay NULL");
    }
    assert!(last_error().contains("x"), "{}", last_error());
    let (rc, _, json) = solve(h, n + 1, "");
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(json.is_none());
    let args = c"method=condensed_sparse";
    let mut out: *mut c_char = dangling_mut();
    let rc = unsafe { (api().s3_solve)(h, n, args.as_ptr(), null_mut(), &mut out) };
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(out.is_null());
    let rc = unsafe { (api().s3_solve)(h, n, args.as_ptr(), p, null_mut()) };
    assert_eq!(rc, PC_ERR_BAD_ARG);
    let (mut hh, mut nn, mut zz) = (0u64, 0usize, 0usize);
    let a = c"N=2";
    unsafe {
        assert_eq!(
            (api().s3_build)(a.as_ptr(), null_mut(), &mut nn, &mut zz),
            PC_ERR_BAD_ARG
        );
        assert_eq!(
            (api().s3_build)(a.as_ptr(), &mut hh, null_mut(), &mut zz),
            PC_ERR_BAD_ARG
        );
        assert_eq!(
            (api().s3_build)(a.as_ptr(), &mut hh, &mut nn, null_mut()),
            PC_ERR_BAD_ARG
        );
        assert_eq!(
            (api().s3_build)(null(), &mut hh, &mut nn, &mut zz),
            PC_ERR_BAD_ARG
        );
    }
    for args in [
        "n=0",
        "m=1",
        "m=11",
        "N=0",
        "T=0",
        "T=abc",
        "foo=1",
        "threads=2",
    ] {
        let a = CString::new(args).unwrap();
        let rc = unsafe { (api().s3_build)(a.as_ptr(), &mut hh, &mut nn, &mut zz) };
        assert_eq!(rc, PC_ERR_BAD_ARG, "{args}");
    }
    assert_eq!((api().free_handle)(h), PC_OK);
}

#[test]
fn ffi_rejects_unknown_and_foreign_handles() {
    let mut p = [0.0; 4];
    let q = p.as_mut_ptr();
    let rc = unsafe { (api().s3_copy)(0x5eed, 1, 1, q, q, q, q) };
    assert_eq!(rc, PC_ERR_BAD_HANDLE);
    let mut other = 0u64;
    assert_eq!(unsafe { (api().test_handle)(&mut other) }, PC_OK);
    let rc = unsafe { (api().s3_copy)(other, 1, 1, q, q, q, q) };
    assert_eq!(rc, PC_ERR_BAD_HANDLE);
    let (rc, _, json) = solve(other, 1, "");
    assert_eq!(rc, PC_ERR_BAD_HANDLE);
    assert!(json.is_none());
    assert_eq!((api().free_handle)(other), PC_OK);
}

#[test]
fn condensed_dense_rejects_large_reduced_systems() {
    // nred = 32 * 158 + 2 = 5058 > 5000.
    let (h, n, _) = build("n=32 m=2 N=157");
    let (rc, _, json) = solve(h, n, "method=condensed_dense reps=1");
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(json.is_none());
    assert!(last_error().contains("5058"), "{}", last_error());
    assert_eq!((api().free_handle)(h), PC_OK);
}
