//! `model_rhs_many` and `model_jacobian_many` through the table from
//! `pc_api_v1()` (SPEC section 8ai-1b, test 3).

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};

const OK: i32 = 0;
const BAD_ARG: i32 = 2;

const VDP: &str = "par mu=1.5\nx'=y\ny'=mu*(1-x*x)*y-x\ndone\n";
const SQRT: &str = "x'=sqrt(x+1)\ny'=-y\ndone\n";
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

fn compiled(text: &str) -> Id {
    let mut id = 0u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc =
        unsafe { (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, &mut diag) };
    if !diag.is_null() {
        unsafe { (api().free_string)(diag) };
    }
    assert_eq!(rc, 0, "{}", last_error());
    let opts = CString::new(r#"{"evaluator":"jit"}"#).unwrap();
    assert_eq!(
        unsafe { (api().model_compile)(id, opts.as_ptr()) },
        0,
        "{}",
        last_error()
    );
    Id(id)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The single call's code and message: `model_rhs` when `t` is None, else
/// `model_rhs_t`. `nout` is the output length.
fn rhs_one(m: u64, t: Option<f64>, x: &[f64], p: &[f64], nout: usize) -> (i32, String, Vec<f64>) {
    let mut out = vec![f64::NAN; nout];
    let rc = unsafe {
        match t {
            None => (api().model_rhs)(
                m,
                x.as_ptr(),
                x.len(),
                p.as_ptr(),
                p.len(),
                out.as_mut_ptr(),
                nout,
            ),
            Some(t) => (api().model_rhs_t)(
                m,
                t,
                x.as_ptr(),
                x.len(),
                p.as_ptr(),
                p.len(),
                out.as_mut_ptr(),
                nout,
            ),
        }
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    (rc, msg, out)
}

/// One `model_rhs_many` call; `x` holds `k` columns of `nx` values. `done`
/// starts at `usize::MAX` so a test sees whether the member set it.
fn rhs_many(
    m: u64,
    t: Option<f64>,
    x: &[f64],
    nx: usize,
    k: usize,
    p: &[f64],
    nout: usize,
) -> (i32, String, Vec<f64>, usize) {
    let mut out = vec![f64::NAN; nout.saturating_mul(k).min(1 << 20)];
    let mut done = usize::MAX;
    let tp = t.as_ref().map_or(ptr::null(), |t| t as *const f64);
    let rc = unsafe {
        (api().model_rhs_many)(
            m,
            tp,
            x.as_ptr(),
            nx,
            k,
            p.as_ptr(),
            p.len(),
            out.as_mut_ptr(),
            nout,
            &mut done,
        )
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    (rc, msg, out, done)
}

fn jac_one(
    m: u64,
    u: &[f64],
    p: &[f64],
    opts: Option<&str>,
    njac: usize,
) -> (i32, String, Vec<f64>) {
    let c = opts.map(|s| CString::new(s).unwrap());
    let o = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    let mut jac = vec![f64::NAN; njac];
    let rc = unsafe {
        (api().model_jacobian)(
            m,
            u.as_ptr(),
            u.len(),
            p.as_ptr(),
            p.len(),
            o,
            jac.as_mut_ptr(),
            njac,
        )
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    (rc, msg, jac)
}

#[allow(clippy::too_many_arguments)]
fn jac_many(
    m: u64,
    u: &[f64],
    nu: usize,
    k: usize,
    p: &[f64],
    opts: Option<&str>,
    njac: usize,
) -> (i32, String, Vec<f64>, usize) {
    let c = opts.map(|s| CString::new(s).unwrap());
    let o = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    let mut jac = vec![f64::NAN; njac.saturating_mul(k).min(1 << 20)];
    let mut done = usize::MAX;
    let rc = unsafe {
        (api().model_jacobian_many)(
            m,
            u.as_ptr(),
            nu,
            k,
            p.as_ptr(),
            p.len(),
            o,
            jac.as_mut_ptr(),
            njac,
            &mut done,
        )
    };
    let msg = if rc == OK {
        String::new()
    } else {
        last_error()
    };
    (rc, msg, jac, done)
}

/// (handle, t, columns, column length, parameters, output length)
type RhsCase<'a> = (u64, Option<f64>, &'a [f64], usize, Vec<f64>, usize);

/// (handle, columns, column length, parameters, opts, output length)
type JacCase<'a> = (u64, &'a [f64], usize, Vec<f64>, Option<&'a str>, usize);

#[test]
fn rhs_pre_checks_match_the_single_calls_and_leave_done_at_zero() {
    let vdp = compiled(VDP);
    let delay = compiled(HUTCH);
    let timed = compiled(TIMED_ODE);
    let x2 = [0.3, -0.2, 0.1, 0.4, 0.5, -0.6, 0.7, 0.8, 0.9];
    let cases: Vec<RhsCase> = vec![
        (0xdead_beef, None, &x2, 2, vec![1.5], 2),
        (delay.0, None, &[0.5, 0.6], 1, vec![1.3, 1.0], 1),
        (timed.0, None, &[0.5, 0.6], 1, vec![2.0], 1),
        (timed.0, Some(f64::NAN), &[0.5, 0.6], 1, vec![2.0], 1),
        (timed.0, Some(f64::INFINITY), &[0.5, 0.6], 1, vec![2.0], 1),
        (vdp.0, None, &x2, 3, vec![1.5], 2),
        (vdp.0, None, &x2, 2, vec![], 2),
        (vdp.0, None, &x2, 2, vec![1.5], 3),
    ];
    for (i, (m, t, x, nx, p, nout)) in cases.into_iter().enumerate() {
        let (rc1, msg1, _) = rhs_one(m, t, &x[..nx.min(x.len())], &p, nout);
        assert_ne!(rc1, OK, "case {i}");
        let (rc, msg, _, done) = rhs_many(m, t, x, nx, 2, &p, nout);
        assert_eq!((rc, msg.as_str()), (rc1, msg1.as_str()), "case {i}");
        assert_eq!(done, 0, "case {i}");
    }
}

#[test]
fn jacobian_pre_checks_match_the_single_calls_and_leave_done_at_zero() {
    let vdp = compiled(VDP);
    let delay = compiled(HUTCH);
    let timed = compiled(TIMED_ODE);
    let henon = compiled(HENON);
    let x2 = [0.3, -0.2, 0.1, 0.4, 0.5, -0.6, 0.7, 0.8, 0.9];
    let cases: Vec<JacCase> = vec![
        (0xdead_beef, &x2, 2, vec![1.5], None, 4),
        (delay.0, &[0.5, 0.6], 1, vec![1.3, 1.0], None, 1),
        (timed.0, &[0.5, 0.6], 1, vec![2.0], None, 1),
        (vdp.0, &x2, 2, vec![1.5], Some(r#"{"iterate":2}"#), 4),
        (henon.0, &x2, 2, vec![1.4, 0.3], Some(r#"{"iterate":0}"#), 4),
        (vdp.0, &x2, 3, vec![1.5], None, 9),
        (vdp.0, &x2, 2, vec![], None, 4),
        (vdp.0, &x2, 2, vec![1.5], None, 3),
    ];
    for (i, (m, u, nu, p, opts, njac)) in cases.into_iter().enumerate() {
        let (rc1, msg1, _) = jac_one(m, &u[..nu.min(u.len())], &p, opts, njac);
        assert_ne!(rc1, OK, "case {i}");
        let (rc, msg, _, done) = jac_many(m, u, nu, 2, &p, opts, njac);
        assert_eq!((rc, msg.as_str()), (rc1, msg1.as_str()), "case {i}");
        assert_eq!(done, 0, "case {i}");
    }
}

#[test]
fn an_overflowing_total_is_bad_arg() {
    let vdp = compiled(VDP);
    let x = [0.3, -0.2];
    let p = [1.5];
    let mut out = [0.0; 4];
    let mut done = usize::MAX;
    let rc = unsafe {
        (api().model_rhs_many)(
            vdp.0,
            ptr::null(),
            x.as_ptr(),
            2,
            usize::MAX / 2 + 1,
            p.as_ptr(),
            1,
            out.as_mut_ptr(),
            2,
            &mut done,
        )
    };
    assert_eq!(rc, BAD_ARG);
    assert!(last_error().contains("overflow"), "{}", last_error());
    assert_eq!(done, 0);
    let mut done = usize::MAX;
    let rc = unsafe {
        (api().model_jacobian_many)(
            vdp.0,
            x.as_ptr(),
            2,
            usize::MAX / 4 + 1,
            p.as_ptr(),
            1,
            ptr::null(),
            out.as_mut_ptr(),
            4,
            &mut done,
        )
    };
    assert_eq!(rc, BAD_ARG);
    assert!(last_error().contains("overflow"), "{}", last_error());
    assert_eq!(done, 0);
}

#[test]
fn zero_columns_and_a_null_done_are_accepted() {
    let vdp = compiled(VDP);
    let p = [1.5];
    let (rc, msg, _, done) = rhs_many(vdp.0, None, &[], 2, 0, &p, 2);
    assert_eq!((rc, done), (OK, 0), "{msg}");
    let (rc, msg, _, done) = jac_many(vdp.0, &[], 2, 0, &p, None, 4);
    assert_eq!((rc, done), (OK, 0), "{msg}");
    // k = 0 still runs the checks.
    let (rc, _, _, done) = rhs_many(vdp.0, None, &[], 3, 0, &p, 2);
    assert_eq!((rc, done), (BAD_ARG, 0));

    let x = [0.3, -0.2, 0.1, 0.4];
    let mut out = [f64::NAN; 4];
    let rc = unsafe {
        (api().model_rhs_many)(
            vdp.0,
            ptr::null(),
            x.as_ptr(),
            2,
            2,
            p.as_ptr(),
            1,
            out.as_mut_ptr(),
            2,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, OK, "{}", last_error());
    let mut want = Vec::new();
    for c in x.chunks(2) {
        let (rc, msg, o) = rhs_one(vdp.0, None, c, &p, 2);
        assert_eq!(rc, OK, "{msg}");
        want.extend(o);
    }
    assert_eq!(bits(&out), bits(&want));

    let mut jac = [f64::NAN; 8];
    let rc = unsafe {
        (api().model_jacobian_many)(
            vdp.0,
            x.as_ptr(),
            2,
            2,
            p.as_ptr(),
            1,
            ptr::null(),
            jac.as_mut_ptr(),
            4,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, OK, "{}", last_error());
    let mut want = Vec::new();
    for c in x.chunks(2) {
        let (rc, msg, j) = jac_one(vdp.0, c, &p, None, 4);
        assert_eq!(rc, OK, "{msg}");
        want.extend(j);
    }
    assert_eq!(bits(&jac), bits(&want));
}

#[test]
fn a_refused_column_sets_done_and_fills_the_columns_before_it() {
    let m = compiled(SQRT);
    // Column 2 is refused: x = -2 for F, x = -1 for the Jacobian.
    let xf = [0.5, 1.0, 0.0, 2.0, -2.0, 0.5, 0.25, 0.75];
    let (rc, msg, out, done) = rhs_many(m.0, None, &xf, 2, 4, &[], 2);
    let (rc1, msg1, _) = rhs_one(m.0, None, &xf[4..6], &[], 2);
    assert_ne!(rc1, OK);
    assert_eq!((rc, msg.as_str(), done), (rc1, msg1.as_str(), 2));
    let mut want = Vec::new();
    for c in xf[..4].chunks(2) {
        want.extend(rhs_one(m.0, None, c, &[], 2).2);
    }
    assert_eq!(bits(&out[..4]), bits(&want));
    assert!(
        out[4..].iter().all(|v| v.is_nan()),
        "later columns are not written"
    );

    let xj = [0.5, 1.0, 0.0, 2.0, -1.0, 0.5, 0.25, 0.75];
    let (rc, msg, jac, done) = jac_many(m.0, &xj, 2, 4, &[], None, 4);
    let (rc1, msg1, _) = jac_one(m.0, &xj[4..6], &[], None, 4);
    assert_ne!(rc1, OK);
    assert_eq!((rc, msg.as_str(), done), (rc1, msg1.as_str(), 2));
    let mut want = Vec::new();
    for c in xj[..4].chunks(2) {
        want.extend(jac_one(m.0, c, &[], None, 4).2);
    }
    assert_eq!(bits(&jac[..8]), bits(&want));
    assert!(
        jac[8..].iter().all(|v| v.is_nan()),
        "later columns are not written"
    );
}
