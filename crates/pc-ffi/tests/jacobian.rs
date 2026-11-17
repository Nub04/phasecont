//! `model_jacobian` through the table from `pc_api_v1()` (SPEC section 8ac, test 1).

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pckern::{PcApiV1, pc_api_v1};

const BAD_ARG: i32 = 2;
const BAD_PROBLEM: i32 = 6;
const NON_AUTONOMOUS: i32 = 7;
const NOT_COMPILED: &str = "model is not compiled; call model_compile first";
const DELAY_MODEL: &str = "the model has delay terms, and this entry is for ODE models";

/// A linear ODE u' = A u with A = [a b; c d].
const LINEAR: &str = "par a=1.5,b=-2,c=0.25,d=-3\nx'=a*x+b*y\ny'=c*x+d*y\ndone\n";
const LINEAR_P: [f64; 4] = [1.5, -2.0, 0.25, -3.0];
/// A damped pendulum, with an equilibrium at (pi, 0) for the eigenvalue check.
const PENDULUM: &str = "par g=1,k=0.1\nx'=y\ny'=-g*sin(x)-k*y\ndone\n";
/// The Henon map, as in `crates/pc-ode/tests/corpus/henon_map.ode`.
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

fn parse(text: &str) -> Id {
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
    let m = parse(text);
    let opts = CString::new(r#"{"evaluator":"jit"}"#).unwrap();
    assert_eq!(
        unsafe { (api().model_compile)(m.0, opts.as_ptr()) },
        0,
        "{}",
        last_error()
    );
    m
}

/// Calls `model_jacobian` with an n-by-n output; `opts` None passes NULL.
fn jacobian(m: &Id, u: &[f64], p: &[f64], opts: Option<&str>) -> (i32, Vec<f64>) {
    jacobian_n(m, u, p, opts, u.len() * u.len())
}

fn jacobian_n(m: &Id, u: &[f64], p: &[f64], opts: Option<&str>, njac: usize) -> (i32, Vec<f64>) {
    let c = opts.map(|s| CString::new(s).unwrap());
    let o = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    let mut jac = vec![f64::NAN; njac];
    let rc = unsafe {
        (api().model_jacobian)(
            m.0,
            u.as_ptr(),
            u.len(),
            p.as_ptr(),
            p.len(),
            o,
            jac.as_mut_ptr(),
            jac.len(),
        )
    };
    (rc, jac)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The Henon Jacobian [-2 a x, 1; b, 0], column-major.
fn henon_jac(u: [f64; 2], a: f64, b: f64) -> [f64; 4] {
    [-2.0 * a * u[0], b, 1.0, 0.0]
}

fn henon_step(u: [f64; 2], a: f64, b: f64) -> [f64; 2] {
    [1.0 - a * u[0] * u[0] + u[1], b * u[0]]
}

/// The column-major product of two 2-by-2 column-major matrices.
fn mul2(l: [f64; 4], r: [f64; 4]) -> [f64; 4] {
    [
        l[0] * r[0] + l[2] * r[1],
        l[1] * r[0] + l[3] * r[1],
        l[0] * r[2] + l[2] * r[3],
        l[1] * r[2] + l[3] * r[3],
    ]
}

#[test]
fn a_linear_ode_returns_its_matrix_bit_for_bit() {
    let m = compiled(LINEAR);
    let (rc, jac) = jacobian(&m, &[0.7, -1.1], &LINEAR_P, None);
    assert_eq!(rc, 0, "{}", last_error());
    // Column-major: [a c b d].
    assert_eq!(bits(&jac), bits(&[1.5, 0.25, -2.0, -3.0]));
    // An empty object is the same as NULL.
    let (rc, again) = jacobian(&m, &[0.7, -1.1], &LINEAR_P, Some("{}"));
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(bits(&again), bits(&jac));
}

#[test]
fn the_henon_iterate_follows_the_chain_rule() {
    let m = compiled(HENON);
    let (a, b) = (1.4, 0.3);
    let u = [0.3, 0.1];
    let (rc, j1) = jacobian(&m, &u, &[a, b], None);
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(bits(&j1), bits(&henon_jac(u, a, b)));
    let (rc, j2) = jacobian(&m, &u, &[a, b], Some(r#"{"iterate":2}"#));
    assert_eq!(rc, 0, "{}", last_error());
    let want = mul2(henon_jac(henon_step(u, a, b), a, b), henon_jac(u, a, b));
    for (g, w) in j2.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-14 * w.abs().max(1.0),
            "got {j2:?}, want {want:?}"
        );
    }
}

/// Solves for an equilibrium from `x` and returns (u, eig as re then im).
fn solve(m: &Id, x: &[f64], p: &[f64], opts: &str) -> (Vec<f64>, Vec<f64>) {
    let opts = CString::new(opts).unwrap();
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
    assert_eq!(rc, 0, "{}", last_error());
    (u, eig)
}

#[test]
fn the_eigenvalues_match_model_equilibrium_bit_for_bit() {
    for (text, x, p, opts) in [
        (PENDULUM, vec![3.0, 0.1], vec![1.0, 0.1], "{}"),
        (HENON, vec![0.6, 0.2], vec![1.4, 0.3], "{}"),
        (HENON, vec![0.9, -0.1], vec![1.4, 0.3], r#"{"iterate":2}"#),
    ] {
        let m = compiled(text);
        let (u, eig) = solve(&m, &x, &p, opts);
        let n = u.len();
        let (rc, jac) = jacobian(&m, &u, &p, Some(opts));
        assert_eq!(rc, 0, "{}", last_error());
        let got = pc_linalg::eigenvalues(&jac, n).unwrap();
        let mut flat = vec![0.0; 2 * n];
        for (i, (re, im)) in got.iter().enumerate() {
            flat[i] = *re;
            flat[n + i] = *im;
        }
        assert_eq!(bits(&flat), bits(&eig), "{text} {opts}");
    }
}

#[test]
fn bad_models_options_and_lengths_are_refused() {
    let delay = compiled(HUTCH);
    let (rc, _) = jacobian(&delay, &[0.5], &[1.3, 1.0], None);
    assert_eq!(rc, BAD_ARG);
    assert_eq!(last_error(), DELAY_MODEL);

    let timed = compiled(TIMED_ODE);
    let (rc, _) = jacobian(&timed, &[0.5], &[2.0], None);
    assert_eq!(rc, NON_AUTONOMOUS, "{}", last_error());

    let raw = parse(LINEAR);
    let (rc, _) = jacobian(&raw, &[0.0, 0.0], &LINEAR_P, None);
    assert_eq!(rc, BAD_ARG);
    assert_eq!(last_error(), NOT_COMPILED);

    let m = compiled(LINEAR);
    let u = [0.0, 0.0];
    for opts in [
        r#"{"iterate":2}"#,
        r#"{"tol":1e-9}"#,
        "not json",
        r#"{"iterate":0}"#,
    ] {
        let (rc, _) = jacobian(&m, &u, &LINEAR_P, Some(opts));
        assert_eq!(rc, BAD_PROBLEM, "{opts}: {}", last_error());
    }
    let henon = compiled(HENON);
    let (rc, _) = jacobian(&henon, &u, &[1.4, 0.3], Some(r#"{"iterate":1,"x":1}"#));
    assert_eq!(rc, BAD_PROBLEM, "{}", last_error());

    let (rc, _) = jacobian(&m, &[0.0], &LINEAR_P, None);
    assert_eq!(rc, BAD_ARG, "nu");
    let (rc, _) = jacobian(&m, &u, &LINEAR_P[..3], None);
    assert_eq!(rc, BAD_ARG, "np");
    let (rc, _) = jacobian_n(&m, &u, &LINEAR_P, None, 3);
    assert_eq!(rc, BAD_ARG, "njac");

    let mut jac = [0.0; 4];
    let rc = unsafe {
        (api().model_jacobian)(
            m.0,
            ptr::null(),
            2,
            LINEAR_P.as_ptr(),
            4,
            ptr::null(),
            jac.as_mut_ptr(),
            4,
        )
    };
    assert_eq!(rc, BAD_ARG, "NULL u");
    let rc = unsafe {
        (api().model_jacobian)(
            m.0,
            u.as_ptr(),
            2,
            LINEAR_P.as_ptr(),
            4,
            ptr::null(),
            ptr::null_mut(),
            4,
        )
    };
    assert_eq!(rc, BAD_ARG, "NULL jac_out");
    let rc = unsafe {
        (api().model_jacobian)(
            0xdead_beef,
            u.as_ptr(),
            2,
            LINEAR_P.as_ptr(),
            4,
            ptr::null(),
            jac.as_mut_ptr(),
            4,
        )
    };
    assert_ne!(rc, 0, "unknown handle");
}
