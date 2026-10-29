//! Panic barrier, the thread-local last error, and the argument helpers
//! that every export uses.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use pc_engine::{install_hook, payload_message, recording, take_last_panic};

use crate::error::{Error, PC_OK, Result, bad_arg};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    let text = CString::new(msg.replace('\0', "\\0")).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = text;
        }
    });
}

/// Runs `f` and returns a panic inside it as `Error::Panic` with the text
/// `panic: <msg> (at file:line:col)`. The model thread uses this around every
/// request, and `barrier` uses it on the calling thread.
pub(crate) fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    install_hook();
    recording(|| match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(payload) => {
            // `recording` restores the record on exit, so take it here.
            let (msg, loc) = take_last_panic()
                .unwrap_or_else(|| (payload_message(&*payload), "unknown".to_string()));
            // Dropping the payload runs its Drop code, which could panic again.
            if let Err(again) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
                std::mem::forget(again);
            }
            Err(Error::Panic(format!("panic: {msg} (at {loc})")))
        }
    })
}

/// Runs `f` and turns its result into an error code. Every exported function
/// goes through here, so no panic reaches the C ABI. On failure the message
/// is kept for `last_error`, which reads it on the same thread.
pub fn barrier(f: impl FnOnce() -> Result<()>) -> i32 {
    set_last_error("");
    match guarded(f) {
        Ok(()) => PC_OK,
        Err(e) => {
            set_last_error(&e.to_string());
            e.code()
        }
    }
}

/// The barrier for exports that return a pointer. A panic gives NULL and the
/// same text for `last_error` as `barrier`; an `Err` gives NULL and its text.
pub fn ptr_barrier<T>(f: impl FnOnce() -> Result<*const T>) -> *const T {
    set_last_error("");
    match guarded(f) {
        Ok(p) => p,
        Err(e) => {
            set_last_error(&e.to_string());
            std::ptr::null()
        }
    }
}

/// The message of the last failed call on this thread, or "" after a
/// successful one. The pointer stays valid until the next call on this
/// thread, so the gateway copies the text at once.
pub extern "C" fn last_error() -> *const c_char {
    catch_unwind(|| {
        LAST_ERROR
            .try_with(|slot| slot.try_borrow().map_or(std::ptr::null(), |s| s.as_ptr()))
            .unwrap_or(std::ptr::null())
    })
    .unwrap_or(std::ptr::null())
}

/// Frees a string returned through an `out` pointer. NULL is ignored.
///
/// # Safety
/// `s` must be NULL or a pointer that this library returned and that has not
/// been freed yet.
pub unsafe extern "C" fn free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { CString::from_raw(s) })));
}

/// Hands a Rust string to C. The caller frees it with `free_string`.
pub(crate) fn into_c_string(s: &str) -> *mut c_char {
    CString::new(s.replace('\0', "\\0"))
        .unwrap_or_default()
        .into_raw()
}

/// Borrows a NUL-terminated UTF-8 argument.
///
/// # Safety
/// `p` must be NULL or point to a NUL-terminated string that outlives `'a`.
pub(crate) unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| bad_arg(format!("{what} is not valid UTF-8")))
}

/// Borrows `len` bytes. NULL with `len > 0` is an error; `len == 0` gives `&[]`.
///
/// # Safety
/// A non-NULL `p` must be valid for reads of `len` bytes for `'a`.
pub(crate) unsafe fn bytes_arg<'a>(p: *const c_char, len: usize, what: &str) -> Result<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    if len > isize::MAX as usize {
        return Err(bad_arg(format!("{what}: length {len} is too large")));
    }
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL but its length is {len}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) })
}

/// Checks a pointer and length before a slice is built from them: the length
/// must fit in an allocation, the pointer must not be NULL, and it must be
/// aligned for `T`. `len == 0` needs no pointer.
fn check_slice<T>(p: *const T, len: usize, what: &str) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if len > isize::MAX as usize / size_of::<T>() {
        return Err(bad_arg(format!("{what}: length {len} is too large")));
    }
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL but its length is {len}")));
    }
    if !(p as usize).is_multiple_of(align_of::<T>()) {
        return Err(bad_arg(format!("{what}: pointer is not aligned")));
    }
    Ok(())
}

/// Borrows `len` doubles. NULL with `len > 0` is an error; `len == 0` gives `&[]`.
/// A pointer that is not 8-byte aligned or a length over `isize::MAX / 8` is an error.
///
/// # Safety
/// A non-NULL `p` must be valid for reads of `len` doubles for `'a`.
pub(crate) unsafe fn f64_arg<'a>(p: *const f64, len: usize, what: &str) -> Result<&'a [f64]> {
    check_slice(p, len, what)?;
    if len == 0 {
        return Ok(&[]);
    }
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

/// Borrows `len` doubles for writing, with the same rules as `f64_arg`.
///
/// # Safety
/// A non-NULL `p` must be valid for writes of `len` doubles for `'a`, with no other access.
pub(crate) unsafe fn f64_out<'a>(p: *mut f64, len: usize, what: &str) -> Result<&'a mut [f64]> {
    check_slice(p, len, what)?;
    if len == 0 {
        return Ok(&mut []);
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(p, len) })
}

/// Checks an output pointer and returns it as a reference. NULL and a pointer
/// that is not aligned for `T` are errors.
///
/// # Safety
/// `p` must be NULL or valid for writes of one `T`.
pub(crate) unsafe fn out_ptr<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL")));
    }
    if !(p as usize).is_multiple_of(align_of::<T>()) {
        return Err(bad_arg(format!("{what}: pointer is not aligned")));
    }
    Ok(unsafe { &mut *p })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    fn text() -> String {
        unsafe { CStr::from_ptr(last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn barrier_maps_ok_err_and_panic() {
        assert_eq!(barrier(|| Ok(())), 0);
        assert_eq!(text(), "");
        assert_eq!(barrier(|| Err(Error::Model("bad".into()))), 5);
        assert_eq!(text(), "bad");
        assert_eq!(barrier(|| panic!("boom {}", 1)), 1);
        assert!(text().starts_with("panic: boom 1 (at "), "{}", text());
    }

    #[test]
    fn ptr_barrier_gives_null_on_panic_and_error() {
        let p = ptr_barrier::<u8>(|| panic!("ptr"));
        assert!(p.is_null());
        assert!(text().starts_with("panic: ptr (at "));
        let p = ptr_barrier::<u8>(|| Err(Error::Internal("no".into())));
        assert!(p.is_null());
        assert_eq!(text(), "no");
        let x = 1u8;
        assert_eq!(ptr_barrier(|| Ok(&x as *const u8)), &x as *const u8);
        assert_eq!(text(), "");
    }

    #[test]
    fn interior_nul_is_escaped() {
        assert_eq!(barrier(|| Err(Error::Internal("a\0b".into()))), 4);
        assert_eq!(text(), "a\\0b");
    }

    #[test]
    fn arg_helpers_check_null_and_length() {
        unsafe {
            assert!(str_arg(std::ptr::null(), "s").is_err());
            assert_eq!(str_arg(c"ab".as_ptr(), "s").unwrap(), "ab");
            assert!(str_arg([0xffu8, 0].as_ptr().cast(), "s").is_err());
            assert!(bytes_arg(std::ptr::null(), 1, "b").is_err());
            assert_eq!(bytes_arg(std::ptr::null(), 0, "b").unwrap(), &[] as &[u8]);
            assert!(f64_arg(std::ptr::null(), 2, "x").is_err());
            assert!(f64_arg(std::ptr::null(), 0, "x").unwrap().is_empty());
            assert!(f64_out(std::ptr::null_mut(), 2, "x").is_err());
            assert!(out_ptr::<u64>(std::ptr::null_mut(), "o").is_err());
            let mut v = 0u64;
            assert!(out_ptr(&mut v, "o").is_ok());
        }
    }

    #[test]
    fn misaligned_pointers_and_huge_lengths_are_bad_args() {
        let words = [0u64; 4];
        let odd = unsafe { words.as_ptr().cast::<u8>().add(1) };
        unsafe {
            let e = f64_arg(odd.cast(), 2, "x").unwrap_err();
            assert_eq!(e.to_string(), "x: pointer is not aligned");
            let e = f64_out(odd.cast_mut().cast(), 2, "y").unwrap_err();
            assert_eq!(e.to_string(), "y: pointer is not aligned");
            let e = out_ptr(odd.cast_mut().cast::<u64>(), "z").unwrap_err();
            assert_eq!(e.to_string(), "z: pointer is not aligned");
            // A zero length never looks at the pointer.
            assert!(f64_arg(odd.cast(), 0, "x").unwrap().is_empty());
            let max = isize::MAX as usize / size_of::<f64>();
            let e = f64_arg(words.as_ptr().cast(), max + 1, "x").unwrap_err();
            assert_eq!(e.to_string(), format!("x: length {} is too large", max + 1));
            assert!(f64_out(words.as_ptr().cast_mut().cast(), usize::MAX, "y").is_err());
            let e = bytes_arg(words.as_ptr().cast(), usize::MAX, "t").unwrap_err();
            assert_eq!(
                e.to_string(),
                format!("t: length {} is too large", usize::MAX)
            );
            // Aligned and in range is fine.
            assert_eq!(f64_arg(words.as_ptr().cast(), 4, "x").unwrap().len(), 4);
        }
    }

    #[test]
    fn guarded_returns_a_panic_as_an_error() {
        assert_eq!(guarded(|| Ok(3)), Ok(3));
        let e = guarded::<()>(|| panic!("inner {}", 2)).unwrap_err();
        assert!(matches!(e, Error::Panic(_)));
        assert!(e.to_string().starts_with("panic: inner 2 (at "), "{e}");
        assert_eq!(e.code(), 1);
        assert_eq!(
            guarded::<()>(|| Err(Error::Model("m".into()))),
            Err(Error::Model("m".into()))
        );
    }
}
