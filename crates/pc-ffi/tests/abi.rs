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
    assert_eq!(unsafe { (t.debug_panic)(1, &mut run) }, 0);
    assert_ne!(run, 0);
    assert_eq!((t.free_handle)(run), 0);
}

#[test]
fn a_debug_handle_passed_as_a_model_is_a_bad_handle() {
    let id = debug_handle();
    let mut out: *mut c_char = ptr::null_mut();
    let rc = unsafe { (api().model_info)(id, &mut out) };
    assert_eq!(rc, 3);
    assert!(out.is_null());
    assert!(
        last_error().contains("is a debug, not a model"),
        "{}",
        last_error()
    );
    assert_eq!((api().free_handle)(id), 0);
}

fn encode(rows: usize, cols: usize, values: &[f64]) -> Vec<u8> {
    let mut v = values.to_vec();
    let mut b = vec![0u8; 9 * v.len()];
    let mut used = usize::MAX;
    let rc = unsafe {
        (api().codec)(
            pckern::CODEC_ENCODE,
            rows,
            cols,
            v.as_mut_ptr(),
            v.len(),
            b.as_mut_ptr(),
            b.len(),
            &mut used,
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
    b.truncate(used);
    b
}

fn decode(rows: usize, cols: usize, bytes: &[u8]) -> (i32, Vec<f64>) {
    let mut b = bytes.to_vec();
    let mut v = vec![f64::NAN; rows * cols];
    let rc = unsafe {
        (api().codec)(
            pckern::CODEC_DECODE,
            rows,
            cols,
            v.as_mut_ptr(),
            v.len(),
            b.as_mut_ptr(),
            b.len(),
            ptr::null_mut(),
        )
    };
    (rc, v)
}

#[test]
fn codec_round_trips_bits_through_the_table() {
    // A 3 by 2 block, column by column, with values the predictor cannot guess.
    let vals = [
        1.0,
        2.0,
        3.0,
        -0.0,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_dead_beef),
    ];
    let b = encode(3, 2, &vals);
    assert_eq!(&b[..7], &[0x06, 0x3f, 0xf0, 0x06, 0x7f, 0xf0, 0x80]);
    let (rc, back) = decode(3, 2, &b);
    assert_eq!(rc, 0, "{}", last_error());
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&back), bits(&vals));
    assert_eq!(b, pckern::xor1::encode(3, 2, &vals));
}

#[test]
fn codec_refuses_bad_bytes_shapes_room_and_ops() {
    let t = api();
    // Trailing byte after a 1 by 1 block.
    let (rc, _) = decode(1, 1, &[0x80, 0x80]);
    assert_eq!(rc, 2);
    assert!(last_error().starts_with("xor1:"), "{}", last_error());
    // n_values that is not rows*cols.
    let mut v = [0.0f64; 3];
    let mut b = [0u8; 27];
    let mut used = 7usize;
    let rc = unsafe { (t.codec)(0, 2, 2, v.as_mut_ptr(), 3, b.as_mut_ptr(), 27, &mut used) };
    assert_eq!(rc, 2);
    // Too little room writes nothing and reports 0.
    let mut v = [1.0f64, 7.5, -3.25];
    let mut b = [0xaau8; 2];
    let rc = unsafe { (t.codec)(0, 3, 1, v.as_mut_ptr(), 3, b.as_mut_ptr(), 2, &mut used) };
    assert_eq!(rc, 2);
    assert_eq!(used, 0);
    assert_eq!(b, [0xaa, 0xaa]);
    assert!(last_error().contains("needs"), "{}", last_error());
    // Unknown op, overflowing shape and a NULL `used` on encode.
    let rc = unsafe { (t.codec)(9, 0, 0, ptr::null_mut(), 0, ptr::null_mut(), 0, &mut used) };
    assert_eq!(rc, 2);
    let rc = unsafe {
        (t.codec)(
            0,
            usize::MAX,
            2,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            &mut used,
        )
    };
    assert_eq!(rc, 2);
    let rc = unsafe {
        (t.codec)(
            0,
            0,
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 2);
    // An empty block encodes to no bytes.
    let rc = unsafe { (t.codec)(0, 0, 4, ptr::null_mut(), 0, ptr::null_mut(), 0, &mut used) };
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(used, 0);
}

#[test]
fn codec_refuses_overlapping_buffers() {
    let mut buf = [0u64; 4];
    let p = buf.as_mut_ptr();
    let mut used = 0usize;
    let rc = unsafe {
        (api().codec)(
            0,
            1,
            1,
            p.cast::<f64>(),
            1,
            p.cast::<u8>().add(4),
            9,
            &mut used,
        )
    };
    assert_eq!(rc, 2);
    assert!(last_error().contains("overlap"), "{}", last_error());
}
