//! Model entries through the table from `pc_api_v1()`.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use pc_engine::{EpSetup, Evaluator, Options, build, equilibrium};
use pc_ir::EntryKind;
use pc_ode::{MapLoader, Model, ModelInfo, SourceMap, load};
use pckern::{PcApiV1, debug_handle, pc_api_v1};
use serde_json::Value;

const LORENZ: &str = "par s=10, r=28, b=2.6666666666666665\n\
init x=1,y=1,z=1\n\
x'=s*(y-x)\n\
y'=x*(r-z)-y\n\
z'=x*y-b*z\n\
done\n";

const NON_AUTONOMOUS: &str = "x'=-x+sin(t)\ndone\n";
const NO_PARAMS: &str = "x'=-x\ndone\n";

const P1: [f64; 3] = [10.0, 28.0, 2.6666666666666665];
const P2: [f64; 3] = [9.0, 25.0, 3.0];

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

/// Frees the model when the test ends.
struct Mod(u64);

impl Drop for Mod {
    fn drop(&mut self) {
        assert_eq!((api().free_handle)(self.0), 0);
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
fn parse(text: &str) -> (i32, u64, Option<Value>) {
    let mut id = 99u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc =
        unsafe { (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, &mut diag) };
    let v = take_string(diag).map(|s| serde_json::from_str(&s).unwrap());
    (rc, id, v)
}

fn model(text: &str) -> Mod {
    let (rc, id, _) = parse(text);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(id, 0);
    Mod(id)
}

fn compile(m: &Mod, opts: Option<&str>) -> i32 {
    let c = opts.map(|s| CString::new(s).unwrap());
    let p = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    unsafe { (api().model_compile)(m.0, p) }
}

fn rhs(m: &Mod, x: &[f64], p: &[f64], out: &mut [f64]) -> i32 {
    unsafe {
        (api().model_rhs)(
            m.0,
            x.as_ptr(),
            x.len(),
            p.as_ptr(),
            p.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    }
}

fn info(m: &Mod) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { (api().model_info)(m.0, &mut s) }, 0);
    serde_json::from_str(&take_string(s).unwrap()).unwrap()
}

fn loaded(text: &str) -> (Model, ModelInfo) {
    let mut map = SourceMap::new();
    let f = map.add("model.ode", text);
    let l = load(f, &mut map, &mut MapLoader::new());
    (l.expanded, l.info)
}

#[allow(clippy::too_many_arguments)]
fn equil(
    m: &Mod,
    u0: &[f64],
    p: &[f64],
    opts: Option<&str>,
    u_out: &mut [f64],
    eig: &mut [f64],
) -> i32 {
    let c = opts.map(|s| CString::new(s).unwrap());
    let op = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
    unsafe {
        (api().model_equilibrium)(
            m.0,
            u0.as_ptr(),
            u0.len(),
            p.as_ptr(),
            p.len(),
            op,
            u_out.as_mut_ptr(),
            u_out.len(),
            eig.as_mut_ptr(),
            eig.len(),
        )
    }
}

#[test]
fn a_parse_error_gives_5_and_a_located_diagnostic() {
    let (rc, id, diag) = parse("x'=-x\nfoo(\ndone\n");
    assert_eq!(rc, 5);
    assert_eq!(id, 0);
    let d = diag.expect("diagnostics come back on a model error");
    let first = &d["diagnostics"][0];
    assert_eq!(first["severity"], "error");
    let (line, col) = (
        first["line"].as_u64().unwrap(),
        first["col"].as_u64().unwrap(),
    );
    assert!(line >= 1 && col >= 1);
    let msg = first["message"].as_str().unwrap();
    assert_eq!(last_error(), format!("{line}:{col}: {msg}"));
    assert_eq!(d["omitted"], 0);
    assert!(d["summary"]["errors"].as_u64().unwrap() >= 1);
}

#[test]
fn a_good_model_gives_a_handle_and_omitted_zero() {
    let (rc, id, diag) = parse(LORENZ);
    assert_eq!(rc, 0, "{}", last_error());
    assert_ne!(id, 0);
    assert_eq!(diag.unwrap()["omitted"], 0);
    assert_eq!((api().free_handle)(id), 0);
}

#[test]
fn argument_checks_on_model_from_ode() {
    let t = api();
    let big = vec![b' '; 16 * 1024 * 1024 + 1];
    let mut id = 99u64;
    let mut diag: *mut c_char = ptr::null_mut();
    let rc = unsafe { (t.model_from_ode)(big.as_ptr().cast(), big.len(), &mut id, &mut diag) };
    assert_eq!(rc, 2);
    assert_eq!(last_error(), "model text is over 16 MiB");
    assert_eq!(id, 0);
    assert!(diag.is_null());
    let rc = unsafe { (t.model_from_ode)(ptr::null(), 5, &mut id, &mut diag) };
    assert_eq!(rc, 2);
    let rc = unsafe { (t.model_from_ode)(c"x".as_ptr(), 1, ptr::null_mut(), &mut diag) };
    assert_eq!(rc, 2);
    let bad = [0xffu8, 0xfe];
    let rc = unsafe { (t.model_from_ode)(bad.as_ptr().cast(), 2, &mut id, &mut diag) };
    assert_eq!(rc, 2);
    assert_eq!(last_error(), "model text is not UTF-8");
    // No diagnostics wanted.
    let rc = unsafe {
        (t.model_from_ode)(
            LORENZ.as_ptr().cast(),
            LORENZ.len(),
            &mut id,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!((t.free_handle)(id), 0);
}

#[test]
fn many_diagnostics_are_capped_at_1000() {
    let text = "foo(\n".repeat(1500);
    let (rc, id, diag) = parse(&text);
    assert_eq!(rc, 5);
    assert_eq!(id, 0);
    let d = diag.unwrap();
    assert_eq!(d["diagnostics"].as_array().unwrap().len(), 1000);
    // Each bad line gives exactly one error.
    assert_eq!(d["omitted"], 500);
    assert_eq!(d["summary"]["errors"], 1500);
}

#[test]
fn an_include_cannot_resolve() {
    let (rc, id, diag) = parse("#include \"other.ode\"\nx'=-x\ndone\n");
    assert_eq!(rc, 5);
    assert_eq!(id, 0);
    assert!(diag.unwrap()["summary"]["errors"].as_u64().unwrap() >= 1);
}

#[test]
fn info_before_and_after_compile() {
    let m = model(LORENZ);
    let v = info(&m);
    assert_eq!(v["n"], 3);
    assert_eq!(v["n_p"], 3);
    assert_eq!(v["variables"], serde_json::json!(["x", "y", "z"]));
    assert_eq!(v["parameters"], serde_json::json!(["s", "r", "b"]));
    assert_eq!(v["values"], serde_json::json!(P1));
    assert_eq!(v["init"], serde_json::json!([1.0, 1.0, 1.0]));
    assert_eq!(v["aux"], serde_json::json!([]));
    assert!(v["evaluator"].is_null());
    assert_eq!(v["jit_max_frame"], 0);
    assert_eq!(compile(&m, Some(r#"{"evaluator":"jit"}"#)), 0);
    let v = info(&m);
    assert_eq!(v["evaluator"], "jit");
    // A small model may need no stack frame, so only the type is checked.
    assert!(v["jit_max_frame"].is_u64());
    assert_eq!(compile(&m, Some(r#"{"evaluator":"fallback"}"#)), 0);
    assert_eq!(info(&m)["evaluator"], "fallback");
}

#[test]
fn compile_options() {
    let m = model(LORENZ);
    assert_eq!(compile(&m, Some(r#"{"evaluator":"jit"}"#)), 0);
    assert_eq!(compile(&m, Some(r#"{"evaluator":"fallback"}"#)), 0);
    assert_eq!(compile(&m, None), 0);
    assert_eq!(info(&m)["evaluator"], "jit");
    assert_eq!(compile(&m, Some(r#"{"evaluator":"gpu"}"#)), 2);
    assert_eq!(last_error(), "evaluator: expected \"jit\" or \"fallback\"");
    assert_eq!(compile(&m, Some(r#"{"evaluator":"jit","x":1}"#)), 2);
    assert!(last_error().contains('x'), "{}", last_error());
    assert_eq!(compile(&m, Some("{nope")), 2);
    assert_eq!(compile(&m, Some("[]")), 2);
    // A failed compile leaves the earlier evaluator in place.
    assert_eq!(info(&m)["evaluator"], "jit");
    let na = model(NON_AUTONOMOUS);
    assert_eq!(compile(&na, None), 0, "{}", last_error());
}

#[test]
fn rhs_matches_the_engine_on_both_evaluators() {
    let m = model(LORENZ);
    let (md, inf) = loaded(LORENZ);
    let mut out = [0.0; 3];
    assert_eq!(rhs(&m, &[1.0, 2.0, 3.0], &P1, &mut out), 2);
    assert_eq!(
        last_error(),
        "model is not compiled; call model_compile first"
    );
    for (which, name) in [(Evaluator::Jit, "jit"), (Evaluator::Fallback, "fallback")] {
        assert_eq!(
            compile(&m, Some(&format!(r#"{{"evaluator":"{name}"}}"#))),
            0
        );
        let mut ev = build(&md, &inf, which).unwrap();
        for (x, p) in [([1.0, 2.0, 3.0], P1), ([-4.0, 0.5, 7.0], P2)] {
            let mut want = [0.0; 3];
            let mut pt = p.to_vec();
            pt.push(0.0);
            ev.eval(&EntryKind::F, &x, &pt, &[], &mut want).unwrap();
            assert_eq!(rhs(&m, &x, &p, &mut out), 0, "{}", last_error());
            assert_eq!(out, want, "{name}");
        }
    }
    assert_eq!(rhs(&m, &[1.0, 2.0], &P1, &mut out), 2);
    assert_eq!(last_error(), "x: expected 3 values, got 2");
    assert_eq!(rhs(&m, &[1.0, 2.0, 3.0], &P1[..2], &mut out), 2);
    assert_eq!(last_error(), "p: expected 3 values, got 2");
    assert_eq!(rhs(&m, &[1.0, 2.0, 3.0], &P1, &mut out[..2]), 2);
    assert_eq!(last_error(), "out: expected 3 values, got 2");
    let rc = unsafe { (api().model_rhs)(m.0, ptr::null(), 3, P1.as_ptr(), 3, out.as_mut_ptr(), 3) };
    assert_eq!(rc, 2);
}

#[test]
fn rhs_reports_a_non_finite_output() {
    let m = model("x'=1/x\ndone\n");
    assert_eq!(compile(&m, None), 0);
    let mut out = [0.0];
    assert_eq!(rhs(&m, &[0.0], &[], &mut out), 8);
}

#[test]
fn equilibrium_matches_the_engine() {
    let m = model(LORENZ);
    assert_eq!(compile(&m, None), 0);
    let (md, inf) = loaded(LORENZ);
    let mut ev = build(&md, &inf, Evaluator::Jit).unwrap();
    let u0 = [8.0, 8.0, 26.0];
    let want = equilibrium(
        &mut *ev,
        &EpSetup::new(0, P1.to_vec(), u0.to_vec()),
        &Options::default(),
    )
    .unwrap();
    let mut u = [0.0; 3];
    let mut eig = [0.0; 6];
    assert_eq!(
        equil(&m, &u0, &P1, None, &mut u, &mut eig),
        0,
        "{}",
        last_error()
    );
    assert_eq!(u.to_vec(), want);
    let mut pt = P1.to_vec();
    pt.push(0.0);
    let mut jac = vec![0.0; 9];
    ev.eval(&EntryKind::JacU, &want, &pt, &[], &mut jac)
        .unwrap();
    let ew = pc_linalg::eigenvalues(&jac, 3).unwrap();
    for i in 0..3 {
        assert_eq!(eig[i], ew[i].0);
        assert_eq!(eig[3 + i], ew[i].1);
    }
    // The C point is a saddle-focus of Lorenz with a complex pair.
    assert!(eig[3..].iter().any(|v| *v != 0.0));
    // Options go through the problem parser.
    assert_eq!(
        equil(
            &m,
            &u0,
            &P1,
            Some(r#"{"max_newton":50,"tol_x":1e-12}"#),
            &mut u,
            &mut eig
        ),
        0
    );
}

#[test]
fn equilibrium_argument_and_failure_codes() {
    let m = model(LORENZ);
    let mut u = [0.0; 3];
    let mut eig = [0.0; 6];
    let u0 = [8.0, 8.0, 26.0];
    assert_eq!(equil(&m, &u0, &P1, None, &mut u, &mut eig), 2);
    assert_eq!(compile(&m, None), 0);
    assert_eq!(equil(&m, &u0[..2], &P1, None, &mut u, &mut eig), 2);
    assert_eq!(equil(&m, &u0, &P1[..2], None, &mut u, &mut eig), 2);
    assert_eq!(equil(&m, &u0, &P1, None, &mut u[..2], &mut eig), 2);
    assert_eq!(equil(&m, &u0, &P1, None, &mut u, &mut eig[..5]), 2);
    assert_eq!(
        equil(
            &m,
            &u0,
            &P1,
            Some(r#"{"max_newton":"x"}"#),
            &mut u,
            &mut eig
        ),
        6
    );
    assert_eq!(equil(&m, &u0, &P1, Some("{nope"), &mut u, &mut eig), 6);
    assert_eq!(
        equil(&m, &u0, &P1, Some(r#"{"max_newton":1}"#), &mut u, &mut eig),
        8
    );
    // Overlapping outputs.
    let mut buf = [0.0f64; 9];
    let rc = unsafe {
        (api().model_equilibrium)(
            m.0,
            u0.as_ptr(),
            3,
            P1.as_ptr(),
            3,
            ptr::null(),
            buf.as_mut_ptr(),
            3,
            buf.as_mut_ptr().add(2),
            6,
        )
    };
    assert_eq!(rc, 2);
    assert_eq!(last_error(), "u_out and eig_out overlap");
}

#[test]
fn a_model_without_parameters_cannot_run_equilibrium() {
    let m = model(NO_PARAMS);
    assert_eq!(compile(&m, None), 0);
    let mut u = [0.0];
    let mut eig = [0.0; 2];
    assert_eq!(equil(&m, &[1.0], &[], None, &mut u, &mut eig), 6);
}

#[test]
fn a_panic_on_the_model_thread_is_reported_and_survived() {
    let m = model(LORENZ);
    assert_eq!(compile(&m, None), 0);
    assert_eq!(pckern::debug_model_panic(m.0), 1);
    let e = last_error().replace('\\', "/");
    assert!(e.starts_with("panic: debug model panic (at "), "{e}");
    assert!(e.contains("model.rs:"), "{e}");
    let mut out = [0.0; 3];
    assert_eq!(
        rhs(&m, &[1.0, 2.0, 3.0], &P1, &mut out),
        0,
        "{}",
        last_error()
    );
}

#[test]
fn a_debug_handle_is_not_a_model() {
    let id = debug_handle();
    assert_eq!(unsafe { (api().model_compile)(id, ptr::null()) }, 3);
    assert!(
        last_error().contains("is a debug, not a model"),
        "{}",
        last_error()
    );
    assert_eq!((api().free_handle)(id), 0);
}

#[test]
fn a_freed_model_is_a_bad_handle() {
    let (rc, id, _) = parse(LORENZ);
    assert_eq!(rc, 0);
    assert_eq!((api().free_handle)(id), 0);
    let mut out = [0.0; 3];
    let rc = unsafe { (api().model_rhs)(id, P1.as_ptr(), 3, P1.as_ptr(), 3, out.as_mut_ptr(), 3) };
    assert_eq!(rc, 3);
    assert_eq!((api().free_handle)(id), 3);
}

#[test]
fn two_threads_can_share_one_model() {
    let m = model(LORENZ);
    assert_eq!(compile(&m, None), 0);
    let m = &m;
    std::thread::scope(|s| {
        for k in 0..2 {
            s.spawn(move || {
                for i in 0..50 {
                    let x = [1.0 + k as f64, i as f64, 3.0];
                    let mut out = [0.0; 3];
                    assert_eq!(rhs(m, &x, &P1, &mut out), 0);
                    assert_eq!(out[0], 10.0 * (x[1] - x[0]));
                }
            });
        }
    });
}

/// A model whose generated code needs a stack frame over 4096 bytes: 1500
/// products x*c_i stay live until two sums over them are subtracted.
fn big_frame_model() -> (String, Vec<f64>) {
    const N: usize = 1500;
    const CHUNK: usize = 100;
    let mut t = String::new();
    let vals: Vec<f64> = (1..=N).map(|i| 0.001 * i as f64).collect();
    for (i, v) in vals.iter().enumerate() {
        t += &format!("par c{}={}\n", i + 1, v);
    }
    t += "init x=0.5\n";
    for i in 1..=N {
        t += &format!("p{i}=x*c{i}\n");
    }
    let sum = |idx: Vec<usize>, tag: &str, t: &mut String| -> Vec<String> {
        idx.chunks(CHUNK)
            .enumerate()
            .map(|(k, ch)| {
                let terms: Vec<String> = ch.iter().map(|i| format!("p{i}")).collect();
                t.push_str(&format!("{tag}{k}={}\n", terms.join("+")));
                format!("{tag}{k}")
            })
            .collect()
    };
    let fwd = sum((1..=N).collect(), "s", &mut t);
    let rev = sum((1..=N).rev().collect(), "r", &mut t);
    t += &format!("x'=x+({})-({})\ndone\n", fwd.join("+"), rev.join("+"));
    (t, vals)
}

#[test]
fn jit_frame_size_is_read_from_the_model_thread() {
    let (text, vals) = big_frame_model();
    let m = model(&text);
    assert_eq!(compile(&m, Some(r#"{"evaluator":"jit"}"#)), 0);
    let frame = info(&m)["jit_max_frame"].as_u64().unwrap();
    assert!(frame > 4096, "{frame}");
    let mut jit = [f64::NAN];
    assert_eq!(rhs(&m, &[0.5], &vals, &mut jit), 0);
    assert!(jit[0].is_finite());

    let f = model(&text);
    assert_eq!(compile(&f, Some(r#"{"evaluator":"fallback"}"#)), 0);
    assert_eq!(info(&f)["jit_max_frame"], 0);
    let mut fb = [f64::NAN];
    assert_eq!(rhs(&f, &[0.5], &vals, &mut fb), 0);
    assert!(
        (jit[0] - fb[0]).abs() <= 1e-12 * fb[0].abs().max(1.0),
        "{} {}",
        jit[0],
        fb[0]
    );
}

#[test]
fn concurrent_compiles_leave_info_matching_the_thread() {
    let m = model(LORENZ);
    std::thread::scope(|s| {
        for i in 0..8 {
            let m = &m;
            s.spawn(move || {
                let o = if i % 2 == 0 { "jit" } else { "fallback" };
                assert_eq!(compile(m, Some(&format!(r#"{{"evaluator":"{o}"}}"#))), 0);
            });
        }
    });
    // The last compile to run on the thread decides; info must name an evaluator
    // that rhs can use, and a final compile must always be reported.
    assert_eq!(compile(&m, Some(r#"{"evaluator":"jit"}"#)), 0);
    assert_eq!(info(&m)["evaluator"], "jit");
}

#[test]
fn a_null_out_model_still_clears_out_diag() {
    let mut diag: *mut c_char = 8usize as *mut c_char;
    let rc = unsafe {
        (api().model_from_ode)(b"x'=-x\n".as_ptr().cast(), 6, ptr::null_mut(), &mut diag)
    };
    assert_eq!(rc, 2);
    assert!(diag.is_null());
}

#[test]
fn adjacent_outputs_are_not_an_overlap() {
    let m = model(LORENZ);
    assert_eq!(compile(&m, None), 0);
    let mut buf = [0.0f64; 9];
    let u0 = [8.0, 8.0, 26.0];
    let rc = unsafe {
        (api().model_equilibrium)(
            m.0,
            u0.as_ptr(),
            3,
            P1.as_ptr(),
            3,
            ptr::null(),
            buf.as_mut_ptr(),
            3,
            buf.as_mut_ptr().add(3),
            6,
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
}

fn rhs_t(m: &Mod, t: f64, x: &[f64], p: &[f64], out: &mut [f64]) -> i32 {
    unsafe {
        (api().model_rhs_t)(
            m.0,
            t,
            x.as_ptr(),
            x.len(),
            p.as_ptr(),
            p.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    }
}

#[test]
fn a_model_that_uses_t_evaluates_at_a_time_on_both_evaluators() {
    let m = model(NON_AUTONOMOUS);
    let mut out = [0.0];
    for name in ["jit", "fallback"] {
        let opts = format!(r#"{{"evaluator":"{name}"}}"#);
        assert_eq!(compile(&m, Some(&opts)), 0, "{name}: {}", last_error());
        assert_eq!(info(&m)["autonomous"], false);
        assert_eq!(rhs_t(&m, 1.3, &[0.5], &[], &mut out), 0, "{}", last_error());
        assert_eq!(out[0], -0.5 + 1.3f64.sin(), "{name}");
        // Another time gives another value.
        assert_eq!(rhs_t(&m, 0.0, &[0.5], &[], &mut out), 0);
        assert_eq!(out[0], -0.5, "{name}");
        // Without t the model refuses.
        assert_eq!(rhs(&m, &[0.5], &[], &mut out), 7, "{name}");
        assert_eq!(last_error(), "the model depends on t; call model_rhs_t");
    }
}

#[test]
fn model_rhs_t_checks_its_arguments() {
    let m = model(NON_AUTONOMOUS);
    assert_eq!(compile(&m, None), 0);
    let mut out = [0.0];
    for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(rhs_t(&m, t, &[0.5], &[], &mut out), 2, "t = {t}");
        assert!(last_error().starts_with("t:"), "{}", last_error());
    }
    assert_eq!(rhs_t(&m, 1.0, &[0.5, 1.0], &[], &mut out), 2);
    assert_eq!(rhs_t(&m, 1.0, &[0.5], &[1.0], &mut out), 2);
    let mut two = [0.0; 2];
    assert_eq!(rhs_t(&m, 1.0, &[0.5], &[], &mut two), 2);
    unsafe {
        let f = api().model_rhs_t;
        let x = [0.5];
        assert_eq!(
            f(
                m.0,
                1.0,
                ptr::null(),
                1,
                ptr::null(),
                0,
                out.as_mut_ptr(),
                1
            ),
            2
        );
        assert_eq!(
            f(m.0, 1.0, x.as_ptr(), 1, ptr::null(), 0, ptr::null_mut(), 1),
            2
        );
        // A zero-length p may be NULL, as for model_rhs.
        assert_eq!(
            f(m.0, 1.0, x.as_ptr(), 1, ptr::null(), 0, out.as_mut_ptr(), 1),
            0
        );
        assert_eq!(
            f(0, 1.0, x.as_ptr(), 1, ptr::null(), 0, out.as_mut_ptr(), 1),
            3
        );
    }
}

#[test]
fn model_rhs_t_on_an_autonomous_model_ignores_t() {
    let m = model(LORENZ);
    let x = [1.0, 2.0, 3.0];
    let mut want = [0.0; 3];
    let mut got = [0.0; 3];
    for name in ["jit", "fallback"] {
        let opts = format!(r#"{{"evaluator":"{name}"}}"#);
        assert_eq!(compile(&m, Some(&opts)), 0);
        assert_eq!(rhs(&m, &x, &P1, &mut want), 0);
        for t in [0.0, 7.0, -3.5, 1e300] {
            assert_eq!(rhs_t(&m, t, &x, &P1, &mut got), 0, "{}", last_error());
            assert_eq!(got, want, "{name} t = {t}");
        }
    }
}

#[test]
fn model_rhs_t_needs_a_compiled_model() {
    let m = model(NON_AUTONOMOUS);
    let mut out = [0.0];
    assert_eq!(rhs_t(&m, 1.0, &[0.5], &[], &mut out), 2);
    assert_eq!(
        last_error(),
        "model is not compiled; call model_compile first"
    );
}

#[test]
fn equilibrium_on_a_model_that_uses_t_gives_7() {
    let m = model(NON_AUTONOMOUS);
    assert_eq!(compile(&m, None), 0);
    let mut u = [0.0];
    let mut eig = [0.0; 2];
    assert_eq!(equil(&m, &[0.0], &[], None, &mut u, &mut eig), 7);
    assert_eq!(last_error(), "the model depends on t");
}
