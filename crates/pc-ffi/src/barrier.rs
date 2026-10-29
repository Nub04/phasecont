//! Panic barrier, the thread-local last error, and the argument helpers
//! that every export uses.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use pc_engine::{install_hook, payload_message, recording, take_last_panic};

use crate::error::{PC_ERR_PANIC, PC_OK, Result, bad_arg};

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

/// Runs `f` and returns a panic inside it as `panic: <msg> (at file:line:col)`.
fn run_caught<T>(f: impl FnOnce() -> T) -> std::result::Result<T, String> {
    install_hook();
    set_last_error("");
    recording(|| match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => {
            // `recording` restores the record on exit, so take it here.
            let (msg, loc) = take_last_panic()
                .unwrap_or_else(|| (payload_message(&*payload), "unknown".to_string()));
            // Dropping the payload runs its Drop code, which could panic again.
            if let Err(again) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
                std::mem::forget(again);
            }
            Err(format!("panic: {msg} (at {loc})"))
        }
    })
}

/// Runs `f` and turns its result into an error code. Every exported function
/// goes through here, so no panic reaches the C ABI. On failure the message
/// is kept for `last_error`, which reads it on the same thread.
pub fn barrier(f: impl FnOnce() -> Result<()>) -> i32 {
    match run_caught(f) {
        Ok(Ok(())) => PC_OK,
        Ok(Err(e)) => {
            set_last_error(&e.to_string());
            e.code()
        }
        Err(text) => {
            set_last_error(&text);
            PC_ERR_PANIC
        }
    }
}

/// The barrier for exports that return a pointer. A panic gives NULL and the
/// same text for `last_error` as `barrier`; an `Err` gives NULL and its text.
pub fn ptr_barrier<T>(f: impl FnOnce() -> Result<*const T>) -> *const T {
    match run_caught(f) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_last_error(&e.to_string());
            std::ptr::null()
        }
        Err(text) => {
            set_last_error(&text);
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
#[allow(dead_code)] // used by the model and run entries
pub(crate) fn into_c_string(s: &str) -> *mut c_char {
    CString::new(s.replace('\0', "\\0"))
        .unwrap_or_default()
        .into_raw()
}

/// Borrows a NUL-terminated UTF-8 argument.
///
/// # Safety
/// `p` must be NULL or point to a NUL-terminated string that outlives `'a`.
#[allow(dead_code)] // used by the model and run entries
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
#[allow(dead_code)] // used by the model and run entries
pub(crate) unsafe fn bytes_arg<'a>(p: *const c_char, len: usize, what: &str) -> Result<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL but its length is {len}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) })
}

/// Borrows `len` doubles. NULL with `len > 0` is an error; `len == 0` gives `&[]`.
///
/// # Safety
/// A non-NULL `p` must be valid for reads of `len` doubles for `'a`.
#[allow(dead_code)] // used by the model and run entries
pub(crate) unsafe fn f64_arg<'a>(p: *const f64, len: usize, what: &str) -> Result<&'a [f64]> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL but its length is {len}")));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

/// Borrows `len` doubles for writing, with the same NULL rule as `f64_arg`.
///
/// # Safety
/// A non-NULL `p` must be valid for writes of `len` doubles for `'a`, with no other access.
#[allow(dead_code)] // used by the model and run entries
pub(crate) unsafe fn f64_out<'a>(p: *mut f64, len: usize, what: &str) -> Result<&'a mut [f64]> {
    if len == 0 {
        return Ok(&mut []);
    }
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL but its length is {len}")));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(p, len) })
}

/// Checks an output pointer and returns it as a reference.
///
/// # Safety
/// `p` must be NULL or valid for writes of one `T`.
pub(crate) unsafe fn out_ptr<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    unsafe { p.as_mut() }.ok_or_else(|| bad_arg(format!("{what} is NULL")))
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
}
