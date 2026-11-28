//! `model_equilibrium_many` through the table from `pc_api_v1()` (SPEC section 8ai-2,
//! test 4).

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};

const OK: i32 = 0;
const BAD_ARG: i32 = 2;
const NUMERIC: i32 = 8;

const BISTABLE: &str = "x'=x-x^3\ny'=-y\ndone\n";
const SINGULAR: &str = "x'=x*x-1\ny'=-y\ndone\n";
const HENON: &str = include_str!("../../pc-ode/tests/corpus/henon_map.ode");
const HUTCH: &str = "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";
const TIMED_ODE: &str = "par r=2\nx'=r*x*(1-x)+t\ndone\n";

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

fn parsed(text: &str) -> Id {
    let mut id = 0u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc =
        unsafe { (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, &mut diag) };
    if !diag.is_null() {
        unsafe { (api().free_string)(diag) };
    }
    assert_eq!(rc, 0, "{}", last_error());
    Id(id)
}

fn compiled(text: &str) -> Id {
    let id = parsed(text);
    let opts = CString::new(r#"{"evaluator":"jit"}"#).unwrap();
    assert_eq!(
        unsafe { (api().model_compile)(id.0, opts.as_ptr()) },
        0,
        "{}",
        last_error()
    );
    id
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn cstr(opts: Option<&str>) -> Option<CString> {
    opts.map(|o| CString::new(o).unwrap())
}

/// One `model_equilibrium` call: the code, the message, u and eig.
fn eq_one(
    m: u64,
    u0: &[f64],
    p: &[f64],
    opts: Option<&str>,
    nu_out: usize,
    neig_out: usize,
) -> (i32, String, Vec<f64>, Vec<f64>) {
    let o = cstr(opts);
    let mut u = vec![f64::NAN; nu_out];
    let mut eig = vec![f64::NAN; neig_out];
    let rc = unsafe {
        (api().model_equilibrium)(
            m,
            u0.as_ptr(),
            u0.len(),
            p.as_ptr(),
            p.len(),
            o.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            u.as_mut_ptr(),
            nu_out,
            eig.as_mut_ptr(),
            neig_out,
        )
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    (rc, msg, u, eig)
}

/// The outputs of one `model_equilibrium_many` call.
struct Many {
    rc: i32,
    msg: String,
    u: Vec<f64>,
    eig: Vec<f64>,
    res: Vec<f64>,
    status: Vec<i32>,
}

/// One `model_equilibrium_many` call with buffers sized for `k` columns (capped, so an
/// overflowing `k` does not allocate). Outputs start as NaN and status as -1, so a test
/// sees what the member wrote.
#[allow(clippy::too_many_arguments)]
fn eq_many(
    m: u64,
    u0: &[f64],
    nu: usize,
    k: usize,
    p: &[f64],
    opts: Option<&str>,
    nu_out: usize,
    neig_out: usize,
    nres: usize,
    nstatus: usize,
) -> Many {
    let o = cstr(opts);
    let cap = |len: usize| len.saturating_mul(k).clamp(1, 1 << 16);
    let mut u = vec![f64::NAN; cap(nu_out)];
    let mut eig = vec![f64::NAN; cap(neig_out)];
    let mut res = vec![f64::NAN; cap(nres)];
    let mut status = vec![-1i32; cap(nstatus)];
    let rc = unsafe {
        (api().model_equilibrium_many)(
            m,
            u0.as_ptr(),
            nu,
            k,
            p.as_ptr(),
            p.len(),
            o.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            u.as_mut_ptr(),
            nu_out,
            eig.as_mut_ptr(),
            neig_out,
            res.as_mut_ptr(),
            nres,
            status.as_mut_ptr(),
            nstatus,
        )
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    Many {
        rc,
        msg,
        u,
        eig,
        res,
        status,
    }
}

/// (handle, one start, parameters, opts, nu_out, neig_out)
type Case<'a> = (u64, &'a [f64], Vec<f64>, Option<&'a str>, usize, usize);

#[test]
fn pre_checks_have_the_codes_and_messages_of_model_equilibrium() {
    let bi = compiled(BISTABLE);
    let delay = compiled(HUTCH);
    let timed = compiled(TIMED_ODE);
    let henon = compiled(HENON);
    let u2 = [0.3, -0.2, 0.1, 0.4, 0.5, -0.6];
    let cases: Vec<Case> = vec![
        (0xdead_beef, &u2[..2], vec![], None, 2, 4),
        (delay.0, &u2[..1], vec![1.3, 1.0], None, 1, 2),
        (timed.0, &u2[..1], vec![2.0], None, 1, 2),
        (bi.0, &u2[..3], vec![], None, 2, 4),
        (bi.0, &u2[..2], vec![1.0], None, 2, 4),
        (bi.0, &u2[..2], vec![], None, 3, 4),
        (bi.0, &u2[..2], vec![], None, 2, 3),
        (bi.0, &u2[..2], vec![], Some(r#"{"iterate":2}"#), 2, 4),
        (bi.0, &u2[..2], vec![], Some(r#"{"bogus":1}"#), 2, 4),
        (bi.0, &u2[..2], vec![], Some("[1]"), 2, 4),
        (
            henon.0,
            &u2[..2],
            vec![1.4, 0.3],
            Some(r#"{"iterate":0}"#),
            2,
            4,
        ),
    ];
    for (i, (m, u0, p, opts, nu_out, neig_out)) in cases.into_iter().enumerate() {
        let (rc1, msg1, _, _) = eq_one(m, u0, &p, opts, nu_out, neig_out);
        assert_ne!(rc1, OK, "case {i}");
        // Two starts, each the column the single call took.
        let both: Vec<f64> = u0.iter().chain(u0).copied().collect();
        let r = eq_many(m, &both, u0.len(), 2, &p, opts, nu_out, neig_out, 1, 1);
        assert_eq!((r.rc, r.msg.as_str()), (rc1, msg1.as_str()), "case {i}");
        assert!(r.status.iter().all(|&s| s == -1), "case {i}: no writes");
        assert!(r.u.iter().all(|v| v.is_nan()), "case {i}: no writes");
    }
}

#[test]
fn res_and_status_take_one_value_per_start() {
    let bi = compiled(BISTABLE);
    let u0 = [0.9, 0.1, -0.8, 0.2];
    for (nres, nstatus, what) in [(2, 1, "res_out"), (1, 2, "status_out"), (0, 1, "res_out")] {
        let r = eq_many(bi.0, &u0, 2, 2, &[], None, 2, 4, nres, nstatus);
        assert_eq!(r.rc, BAD_ARG, "{what}");
        assert!(r.msg.contains(what), "{}", r.msg);
    }
}

#[test]
fn an_overflowing_total_is_bad_arg() {
    let bi = compiled(BISTABLE);
    let u0 = [0.9, 0.1];
    let r = eq_many(bi.0, &u0, 2, usize::MAX / 2 + 1, &[], None, 2, 4, 1, 1);
    assert_eq!(r.rc, BAD_ARG);
    assert!(r.msg.contains("overflow"), "{}", r.msg);
    // 2n values per start overflow at half that k.
    let r = eq_many(bi.0, &u0, 2, usize::MAX / 4 + 1, &[], None, 2, 4, 1, 1);
    assert_eq!(r.rc, BAD_ARG);
    assert!(r.msg.contains("overflow"), "{}", r.msg);
}

#[test]
fn overlapping_outputs_are_bad_arg() {
    let bi = compiled(BISTABLE);
    let u0 = [0.9, 0.1, -0.8, 0.2];
    let mut a = vec![0.0f64; 64];
    let mut status = [0i32; 2];
    let base = a.as_mut_ptr();
    let call = |u: *mut f64, e: *mut f64, r: *mut f64, s: *mut i32| unsafe {
        (api().model_equilibrium_many)(
            bi.0,
            u0.as_ptr(),
            2,
            2,
            ptr::null(),
            0,
            ptr::null(),
            u,
            2,
            e,
            4,
            r,
            1,
            s,
            1,
        )
    };
    let st = status.as_mut_ptr();
    let cases = unsafe {
        [
            // eig_out starts inside u_out's 4 values.
            (
                base,
                base.add(3),
                base.add(20),
                st,
                "u_out and eig_out overlap",
            ),
            (
                base,
                base.add(4),
                base.add(11),
                st,
                "eig_out and res_out overlap",
            ),
            (
                base,
                base.add(4),
                base.add(12),
                base.add(13).cast::<i32>(),
                "res_out and status_out overlap",
            ),
            (
                base,
                base.add(4),
                base.add(12),
                base.add(1).cast::<i32>(),
                "u_out and status_out overlap",
            ),
        ]
    };
    for (u, e, r, s, want) in cases {
        let rc = call(u, e, r, s);
        assert_eq!(rc, BAD_ARG, "{want}");
        assert_eq!(last_error(), want);
    }
    // Buffers that only touch are fine.
    let rc = call(base, unsafe { base.add(4) }, unsafe { base.add(12) }, st);
    assert_eq!(rc, OK, "{}", last_error());
}

#[test]
fn zero_starts_run_the_checks_and_write_nothing() {
    let bi = compiled(BISTABLE);
    let r = eq_many(bi.0, &[], 2, 0, &[], None, 2, 4, 1, 1);
    assert_eq!(r.rc, OK, "{}", r.msg);
    assert!(r.u.iter().chain(&r.eig).chain(&r.res).all(|v| v.is_nan()));
    assert!(r.status.iter().all(|&s| s == -1));
    let r = eq_many(bi.0, &[], 3, 0, &[], None, 2, 4, 1, 1);
    assert_eq!(r.rc, BAD_ARG);
    let r = eq_many(bi.0, &[], 2, 0, &[], Some(r#"{"bogus":1}"#), 2, 4, 1, 1);
    assert_ne!(r.rc, OK);
}

#[test]
fn a_model_with_no_evaluator_is_refused() {
    let m = parsed(BISTABLE);
    let (rc1, msg1, _, _) = eq_one(m.0, &[0.9, 0.1], &[], None, 2, 4);
    assert_eq!(rc1, BAD_ARG);
    let r = eq_many(m.0, &[0.9, 0.1, -0.8, 0.2], 2, 2, &[], None, 2, 4, 1, 1);
    assert_eq!((r.rc, r.msg.as_str()), (rc1, msg1.as_str()));
    assert!(r.msg.contains("model_compile"), "{}", r.msg);
    let r = eq_many(m.0, &[], 2, 0, &[], None, 2, 4, 1, 1);
    assert_eq!((r.rc, r.msg.as_str()), (rc1, msg1.as_str()));
}

/// A model source, its parameters, its options and its starts.
type ModelCase<'a> = (&'a str, Vec<f64>, Option<&'a str>, Vec<f64>);

/// Every column has the bits of the single call from that start, and failed starts get
/// NUMERIC, NaN columns and a residual, beside converged starts.
#[test]
fn columns_match_single_calls_through_the_table() {
    let cases: [ModelCase; 3] = [
        (
            BISTABLE,
            vec![],
            None,
            vec![1.3, 0.4, -0.2, 0.1, -1.4, -0.3, 0.6, 0.0, f64::NAN, 0.0],
        ),
        (
            HENON,
            vec![1.4, 0.3],
            None,
            vec![0.6, 0.2, -1.2, -0.4, 0.0, 0.0, f64::INFINITY, 1.0],
        ),
        (
            HENON,
            vec![1.4, 0.3],
            Some(r#"{"iterate":2,"max_newton":30}"#),
            vec![0.6, 0.2, 1.0, -0.1, -0.5, 0.3, 0.0, 0.0],
        ),
    ];
    for (text, p, opts, u0) in cases {
        let m = compiled(text);
        let k = u0.len() / 2;
        let r = eq_many(m.0, &u0, 2, k, &p, opts, 2, 4, 1, 1);
        assert_eq!(r.rc, OK, "{}", r.msg);
        let mut converged = 0;
        for j in 0..k {
            let (rc1, _, u1, e1) = eq_one(m.0, &u0[2 * j..2 * j + 2], &p, opts, 2, 4);
            let ucol = &r.u[2 * j..2 * j + 2];
            let ecol = &r.eig[4 * j..4 * j + 4];
            if rc1 == OK {
                converged += 1;
                assert_eq!(r.status[j], OK, "start {j}");
                assert_eq!(bits(ucol), bits(&u1), "start {j}");
                assert_eq!(bits(ecol), bits(&e1), "start {j}");
                assert!(
                    r.res[j].is_finite() && r.res[j] < 1e-8,
                    "start {j}: {}",
                    r.res[j]
                );
            } else {
                assert_eq!(rc1, NUMERIC, "start {j}");
                assert_eq!(r.status[j], NUMERIC, "start {j}");
                assert!(ucol.iter().chain(ecol).all(|v| v.is_nan()), "start {j}");
            }
        }
        assert!(converged >= 2, "{text}: {converged} converged");
    }
}

#[test]
fn a_singular_start_and_a_nan_start_have_a_nan_residual() {
    let m = compiled(SINGULAR);
    let u0 = [0.0, 0.5, 2.0, 0.1, f64::NAN, 0.0];
    let r = eq_many(m.0, &u0, 2, 3, &[], None, 2, 4, 1, 1);
    assert_eq!(r.rc, OK, "{}", r.msg);
    assert_eq!(r.status, vec![NUMERIC, OK, NUMERIC]);
    assert!(r.res[0].is_nan() && r.res[2].is_nan());
    let (rc1, _, u1, _) = eq_one(m.0, &u0[2..4], &[], None, 2, 4);
    assert_eq!(rc1, OK);
    assert_eq!(bits(&r.u[2..4]), bits(&u1));
    // Newton that runs out of steps keeps its last residual.
    let far = [100.0, 0.0];
    let r = eq_many(
        m.0,
        &far,
        2,
        1,
        &[],
        Some(r#"{"max_newton":2}"#),
        2,
        4,
        1,
        1,
    );
    assert_eq!(r.status, vec![NUMERIC]);
    assert!(r.res[0].is_finite() && r.res[0] > 1.0, "{}", r.res[0]);
    let r = eq_many(
        m.0,
        &far,
        2,
        1,
        &[],
        Some(r#"{"max_newton":0}"#),
        2,
        4,
        1,
        1,
    );
    // max_newton 0 is refused by the options parser, as for the single call; the
    // infinity rule is covered by the unit tests in model.rs.
    let (rc1, msg1, _, _) = eq_one(m.0, &far, &[], Some(r#"{"max_newton":0}"#), 2, 4);
    assert_ne!(rc1, OK);
    assert_eq!((r.rc, r.msg.as_str()), (rc1, msg1.as_str()));
}
