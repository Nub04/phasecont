//! S1 checks on the native build. They call through the same function table
//! that the MEX gateway gets from `pc_api_v1()`.

use std::ffi::{CStr, c_char};

use pckern::ffi::{PC_ERR_BAD_ARG, PC_ERR_BAD_HANDLE, PC_ERR_PANIC, PC_OK};
use pckern::{ABI_VERSION, PcApiV1, pc_api_v1};

fn api() -> &'static PcApiV1 {
    unsafe { &*pc_api_v1() }
}

fn last_error() -> String {
    let p = (api().last_error)();
    assert!(!p.is_null());
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[test]
fn table_header() {
    assert_eq!(api().abi_version, ABI_VERSION);
    assert_eq!(api().struct_size as usize, size_of::<PcApiV1>());
    let info = unsafe { CStr::from_ptr((api().info)()) }.to_str().unwrap();
    assert!(info.starts_with("pckern spike abi=1 target="), "{info}");
}

#[test]
fn table_and_info_are_not_null() {
    let table = pc_api_v1();
    assert!(!table.is_null());
    let p = (unsafe { &*table }.info)();
    assert!(!p.is_null());
    let info = unsafe { CStr::from_ptr(p) }.to_string_lossy();
    assert!(info.starts_with("pckern spike abi=1"), "{info}");
}

#[test]
fn scale_round_trip() {
    let x = [1.0, -2.5, f64::NAN, f64::INFINITY, 0.0];
    let mut out = [0.0; 5];
    let rc = unsafe { (api().scale)(x.as_ptr(), x.len(), 2.0, out.as_mut_ptr()) };
    assert_eq!(rc, PC_OK);
    assert_eq!(out[0], 2.0);
    assert_eq!(out[1], -5.0);
    assert!(out[2].is_nan());
    assert_eq!(out[3], f64::INFINITY);
    assert_eq!(out[4], 0.0);
    assert_eq!(last_error(), "");
}

#[test]
fn scale_empty_and_null() {
    let null = std::ptr::null_mut();
    assert_eq!(unsafe { (api().scale)(null, 0, 2.0, null) }, PC_OK);
    assert_eq!(unsafe { (api().scale)(null, 3, 2.0, null) }, PC_ERR_BAD_ARG);
    assert!(last_error().contains("NULL"));
}

#[test]
fn scale_in_place() {
    let mut x = [1.0, 2.0, 3.0];
    let p = x.as_mut_ptr();
    assert_eq!(unsafe { (api().scale)(p, 3, 2.0, p) }, PC_OK);
    assert_eq!(x, [2.0, 4.0, 6.0]);
}

#[test]
fn panic_is_caught_with_message_and_location() {
    for _ in 0..2 {
        assert_eq!((api().panic)(), PC_ERR_PANIC);
        let msg = last_error();
        assert!(msg.contains("spike panic: 100% on purpose"), "{msg}");
        assert!(msg.contains("s1.rs"), "{msg}");
    }
    let x = [3.0];
    let mut out = [0.0];
    assert_eq!(
        unsafe { (api().scale)(x.as_ptr(), 1, 2.0, out.as_mut_ptr()) },
        PC_OK
    );
    assert_eq!(out[0], 6.0);
}

#[test]
fn worker_panic_reaches_the_caller() {
    for _ in 0..2 {
        assert_eq!((api().panic_in_worker)(), PC_ERR_PANIC);
        let msg = last_error();
        assert!(msg.contains("spike worker panic at item 37"), "{msg}");
        assert!(msg.contains("another thread"), "{msg}");
    }
}

#[test]
fn handles_free_once() {
    let mut id = 0u64;
    assert_eq!(unsafe { (api().test_handle)(&mut id) }, PC_OK);
    assert_ne!(id, 0);
    assert_eq!((api().free_handle)(id), PC_OK);
    assert_eq!((api().free_handle)(id), PC_ERR_BAD_HANDLE);
    assert!(last_error().contains("unknown handle"));
    assert_eq!((api().free_handle)(0), PC_ERR_BAD_HANDLE);
    assert_eq!((api().free_handle)(1), PC_ERR_BAD_HANDLE);
    assert_eq!(
        unsafe { (api().test_handle)(std::ptr::null_mut()) },
        PC_ERR_BAD_ARG
    );
}

#[test]
fn run_json_rejects_unknown_spike() {
    let mut out: *mut c_char = std::ptr::dangling_mut();
    let rc = unsafe { (api().run_json)(c"nope".as_ptr(), c"".as_ptr(), &mut out) };
    assert_eq!(rc, PC_ERR_BAD_ARG);
    assert!(out.is_null());
    assert!(last_error().contains("unknown spike"));
    // free_string ignores NULL.
    unsafe { (api().free_string)(std::ptr::null_mut()) };
}
