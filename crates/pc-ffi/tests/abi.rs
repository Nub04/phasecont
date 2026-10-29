//! ABI tests: every call goes through the table from `pc_api_v1()`.

use std::ffi::{CStr, c_char};
use std::ptr;

use pckern::{PcApiV1, debug_handle, pc_api_v1};

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

#[test]
fn table_header_is_filled() {
    let t = api();
    assert_eq!(t.abi_version, 1);
    assert_eq!(t.struct_size as usize, size_of::<PcApiV1>());
}

#[test]
fn version_names_the_library_and_abi() {
    let p = (api().version)();
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
    assert!(s.starts_with("pckern "), "{s}");
    assert!(s.contains("abi 1"), "{s}");
}

#[test]
fn free_handle_rejects_zero_unknown_and_double_free() {
    let t = api();
    assert_eq!((t.free_handle)(0), 3);
    assert!(last_error().contains("unknown handle"));
    let unknown = 0x5043_0000_0000_0000u64 | (0x7fff_ffffu64 << 1) | 1;
    assert_eq!((t.free_handle)(unknown), 3);
    let id = debug_handle();
    assert_eq!((t.free_handle)(id), 0);
    assert_eq!(last_error(), "");
    assert_eq!((t.free_handle)(id), 3);
}

#[test]
fn an_id_that_went_through_a_double_is_rejected() {
    let id = debug_handle();
    let mangled = (id as f64) as u64;
    assert_ne!(mangled, id);
    assert_eq!((api().free_handle)(mangled), 3);
    assert_eq!((api().free_handle)(id), 0);
}

#[test]
fn free_string_accepts_null() {
    unsafe { (api().free_string)(ptr::null_mut()) };
}

#[test]
fn debug_panic_reports_the_location() {
    let rc = unsafe { (api().debug_panic)(0, ptr::null_mut()) };
    assert_eq!(rc, 1);
    let e = last_error().replace('\\', "/");
    assert!(e.starts_with("panic: debug panic (at "), "{e}");
    assert!(e.ends_with(')'), "{e}");
    let loc = e.trim_end_matches(')').rsplit("(at ").next().unwrap();
    assert!(loc.contains("api.rs:"), "{e}");
    let mut parts = loc.rsplitn(3, ':');
    assert!(parts.next().unwrap().parse::<u32>().is_ok(), "{e}");
    assert!(parts.next().unwrap().parse::<u32>().is_ok(), "{e}");
}

#[test]
fn the_next_call_clears_a_panic_text() {
    assert_eq!(unsafe { (api().debug_panic)(0, ptr::null_mut()) }, 1);
    assert_eq!((api().free_handle)(0), 3);
    let e = last_error();
    assert!(!e.contains("panic"), "{e}");
    let id = debug_handle();
    assert_eq!((api().free_handle)(id), 0);
    assert_eq!(last_error(), "");
}

#[test]
fn last_error_is_per_thread() {
    assert_eq!((api().free_handle)(0), 3);
    let main_text = last_error();
    std::thread::spawn(|| {
        assert_eq!(last_error(), "");
        assert_eq!(unsafe { (api().debug_panic)(7, ptr::null_mut()) }, 2);
        assert!(last_error().contains("debug_panic"));
    })
    .join()
    .unwrap();
    assert_eq!(last_error(), main_text);
}

#[test]
fn consecutive_panics_report_their_own_text() {
    let t = api();
    assert_eq!(unsafe { (t.debug_panic)(0, ptr::null_mut()) }, 1);
    let first = last_error();
    assert_eq!(unsafe { (t.debug_panic)(0, ptr::null_mut()) }, 1);
    let second = last_error();
    assert!(first.starts_with("panic: debug panic (at "));
    assert!(second.starts_with("panic: debug panic (at "));
    // Same source line, and the message is not doubled.
    assert_eq!(first, second);
    assert_eq!(second.matches("debug panic").count(), 1);
}

#[test]
fn debug_panic_checks_its_arguments() {
    let t = api();
    assert_eq!(unsafe { (t.debug_panic)(7, ptr::null_mut()) }, 2);
    assert_eq!(unsafe { (t.debug_panic)(1, ptr::null_mut()) }, 2);
    let mut run = 0u64;
    assert_eq!(unsafe { (t.debug_panic)(1, &mut run) }, 4);
}

#[test]
fn type_confusion_is_a_bad_handle() {
    // A debug handle is not a model.
    let id = debug_handle();
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().model_info)(id, &mut out) };
    assert_ne!(rc, 0);
    assert_eq!((api().free_handle)(id), 0);
}

#[test]
fn model_and_run_entries_are_stubs() {
    let t = api();
    let mut h = 0u64;
    let mut s: *mut c_char = ptr::null_mut();
    let x = [0.0f64; 2];
    let mut o = [0.0f64; 2];
    let mut n = 0usize;
    let mut run_id = 0u64;
    unsafe {
        let rcs = [
            (
                "model_from_ode",
                (t.model_from_ode)(c"x".as_ptr(), 1, &mut h, &mut s),
            ),
            ("model_compile", (t.model_compile)(1, c"jit".as_ptr())),
            (
                "model_rhs",
                (t.model_rhs)(1, x.as_ptr(), 2, x.as_ptr(), 0, o.as_mut_ptr(), 2),
            ),
            (
                "run_start",
                (t.run_start)(1, c"{}".as_ptr(), x.as_ptr(), 2, x.as_ptr(), 0, &mut run_id),
            ),
            ("run_poll", (t.run_poll)(1, &mut s)),
            (
                "run_copy",
                (t.run_copy)(1, 0, 0, 0, 1, o.as_mut_ptr(), 2, &mut n),
            ),
            ("run_cancel", (t.run_cancel)(1)),
            ("model_info", (t.model_info)(1, &mut s)),
            (
                "model_equilibrium",
                (t.model_equilibrium)(
                    1,
                    x.as_ptr(),
                    2,
                    x.as_ptr(),
                    0,
                    c"{}".as_ptr(),
                    o.as_mut_ptr(),
                    2,
                    o.as_mut_ptr(),
                    2,
                ),
            ),
        ];
        for (name, rc) in rcs {
            assert_eq!(rc, 4, "{name}");
        }
    }
}

#[test]
fn stubs_name_themselves() {
    let t = api();
    assert_eq!(unsafe { (t.model_compile)(1, c"jit".as_ptr()) }, 4);
    assert_eq!(last_error(), "model_compile is not implemented yet");
    assert_eq!((t.run_cancel)(1), 4);
    assert_eq!(last_error(), "run_cancel is not implemented yet");
}
