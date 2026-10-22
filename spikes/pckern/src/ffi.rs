//! Panic barrier, error codes, the thread-local last error, and the small
//! text formats that cross the C ABI: "key=value" arguments in, JSON out.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::ffi::{CStr, CString, c_char};
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

pub const PC_OK: i32 = 0;
pub const PC_ERR_PANIC: i32 = 1;
pub const PC_ERR_BAD_ARG: i32 = 2;
pub const PC_ERR_BAD_HANDLE: i32 = 3;
pub const PC_ERR_INTERNAL: i32 = 4;

#[derive(Debug)]
pub enum Error {
    BadArg(String),
    BadHandle(String),
    Internal(String),
}

impl Error {
    pub fn code(&self) -> i32 {
        match self {
            Error::BadArg(_) => PC_ERR_BAD_ARG,
            Error::BadHandle(_) => PC_ERR_BAD_HANDLE,
            Error::Internal(_) => PC_ERR_INTERNAL,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadArg(m) | Error::BadHandle(m) | Error::Internal(m) => f.write_str(m),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn bad_arg(msg: impl Into<String>) -> Error {
    Error::BadArg(msg.into())
}

pub fn internal(msg: impl Into<String>) -> Error {
    Error::Internal(msg.into())
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
    /// How many `catch_panic` calls are running on this thread.
    static CATCH_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_owned()
    }
}

/// MATLAB shows no stderr, so inside `catch_panic` the hook keeps the message
/// and location instead of printing them. The hook runs on the panicking
/// thread, so a panic on a worker thread leaves no record here, and the
/// barrier falls back to the payload that rayon carries over. Panics outside
/// `catch_panic`, such as worker panics and failed test assertions, still go
/// to the previous hook, which prints them.
fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !CATCH_DEPTH.try_with(|d| d.get() > 0).unwrap_or(false) {
                previous(info);
                return;
            }
            let msg = payload_message(info.payload());
            let text = match info.location() {
                Some(loc) => format!("{msg} (at {}:{})", loc.file(), loc.line()),
                None => msg,
            };
            // try_with and try_borrow_mut: a panic inside the hook aborts.
            let _ = LAST_PANIC.try_with(|slot| {
                if let Ok(mut slot) = slot.try_borrow_mut() {
                    *slot = Some(text);
                }
            });
        }));
    });
}

fn take_last_panic() -> Option<String> {
    LAST_PANIC
        .try_with(|slot| slot.try_borrow_mut().ok().and_then(|mut s| s.take()))
        .ok()
        .flatten()
}

fn set_last_error(msg: &str) {
    let text = CString::new(msg.replace('\0', "\\0")).unwrap_or_default();
    let _ = LAST_ERROR.try_with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = text;
        }
    });
}

/// Runs `f` and returns a panic inside it as its message. The barrier uses
/// it, and so does code that keeps going after one part fails, such as S4
/// trying several CUDA paths in one call. The hook's record is taken here,
/// so a later panic in the same call is never reported with this message.
pub fn catch_panic<T>(f: impl FnOnce() -> T) -> std::result::Result<T, String> {
    install_panic_hook();
    take_last_panic();
    let _ = CATCH_DEPTH.try_with(|d| d.set(d.get() + 1));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    let _ = CATCH_DEPTH.try_with(|d| d.set(d.get() - 1));
    result.map_err(|payload| {
        let msg = take_last_panic().unwrap_or_else(|| {
            format!("{} (raised on another thread)", payload_message(&*payload))
        });
        // Dropping the payload runs its Drop code, which could panic again.
        if let Err(again) = panic::catch_unwind(AssertUnwindSafe(|| drop(payload))) {
            std::mem::forget(again);
        }
        msg
    })
}

/// Runs `f` and turns its result into an error code. Every exported function
/// goes through here, so no panic reaches the C ABI. On failure the message
/// is kept for `last_error`, which reads it on the same thread.
pub fn barrier(f: impl FnOnce() -> Result<()>) -> i32 {
    set_last_error("");
    match catch_panic(f) {
        Ok(Ok(())) => PC_OK,
        Ok(Err(e)) => {
            set_last_error(&e.to_string());
            e.code()
        }
        Err(msg) => {
            set_last_error(&format!("panic: {msg}"));
            PC_ERR_PANIC
        }
    }
}

/// The barrier for exported functions that return a pointer instead of an
/// error code. A panic returns NULL and keeps the same "panic: ..." text as
/// `barrier` for `last_error`; any other result clears that text.
pub(crate) fn ptr_barrier<T>(f: impl FnOnce() -> *const T) -> *const T {
    set_last_error("");
    match catch_panic(f) {
        Ok(p) => p,
        Err(msg) => {
            set_last_error(&format!("panic: {msg}"));
            std::ptr::null()
        }
    }
}

/// The message of the last failed call on this thread, or "" after a
/// successful one. The pointer stays valid until the next call on this
/// thread, so the gateway copies the text at once.
pub extern "C" fn last_error() -> *const c_char {
    panic::catch_unwind(|| LAST_ERROR.with(|slot| slot.borrow().as_ptr()))
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
    let _ = panic::catch_unwind(AssertUnwindSafe(|| drop(unsafe { CString::from_raw(s) })));
}

/// Hands a Rust string to C. The caller frees it with `free_string`.
pub fn into_c_string(s: &str) -> *mut c_char {
    CString::new(s.replace('\0', "\\0"))
        .unwrap_or_default()
        .into_raw()
}

/// Borrows a NUL-terminated UTF-8 argument.
///
/// # Safety
/// `p` must be NULL or point to a NUL-terminated string that outlives `'a`.
pub unsafe fn str_arg<'a>(p: *const c_char, what: &str) -> Result<&'a str> {
    if p.is_null() {
        return Err(bad_arg(format!("{what} is NULL")));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| bad_arg(format!("{what} is not valid UTF-8")))
}

/// Checks an output pointer and returns it as a reference.
///
/// # Safety
/// `p` must be NULL or valid for writes of one `T`.
pub unsafe fn out_arg<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T> {
    unsafe { p.as_mut() }.ok_or_else(|| bad_arg(format!("{what} is NULL")))
}

/// Arguments as "key=value" pairs separated by spaces, commas or semicolons.
/// Unknown and repeated keys are errors, so a typo cannot pass silently.
pub struct Args {
    pairs: Vec<(String, String)>,
}

impl Args {
    pub fn parse(text: &str, allowed: &[&str]) -> Result<Args> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        let tokens = text
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|t| !t.is_empty());
        for tok in tokens {
            let (k, v) = tok
                .split_once('=')
                .ok_or_else(|| bad_arg(format!("expected key=value, got {tok:?}")))?;
            if !allowed.contains(&k) {
                return Err(bad_arg(format!(
                    "unknown key {k:?}; allowed keys: {}",
                    allowed.join(", ")
                )));
            }
            if pairs.iter().any(|(pk, _)| pk == k) {
                return Err(bad_arg(format!("key {k:?} given twice")));
            }
            pairs.push((k.to_owned(), v.to_owned()));
        }
        Ok(Args { pairs })
    }

    fn raw(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        match self.raw(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| bad_arg(format!("{key}={v} is not a non-negative integer"))),
        }
    }

    pub fn u64_or(&self, key: &str, default: u64) -> Result<u64> {
        match self.raw(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| bad_arg(format!("{key}={v} is not a non-negative integer"))),
        }
    }

    pub fn str_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.raw(key).unwrap_or(default)
    }
}

/// Builds one JSON object. Non-finite numbers become null, since JSON has no
/// NaN or Inf.
pub struct Json {
    buf: String,
    empty: bool,
}

impl Default for Json {
    fn default() -> Self {
        Json {
            buf: String::from("{"),
            empty: true,
        }
    }
}

impl Json {
    fn key(&mut self, k: &str) {
        if !self.empty {
            self.buf.push(',');
        }
        self.empty = false;
        push_str(&mut self.buf, k);
        self.buf.push(':');
    }

    pub fn num(&mut self, k: &str, v: f64) -> &mut Self {
        self.key(k);
        push_num(&mut self.buf, v);
        self
    }

    pub fn int(&mut self, k: &str, v: i64) -> &mut Self {
        self.key(k);
        self.buf.push_str(&v.to_string());
        self
    }

    pub fn boolean(&mut self, k: &str, v: bool) -> &mut Self {
        self.key(k);
        self.buf.push_str(if v { "true" } else { "false" });
        self
    }

    pub fn str(&mut self, k: &str, v: &str) -> &mut Self {
        self.key(k);
        push_str(&mut self.buf, v);
        self
    }

    pub fn nums(&mut self, k: &str, v: &[f64]) -> &mut Self {
        self.key(k);
        self.buf.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.buf.push(',');
            }
            push_num(&mut self.buf, *x);
        }
        self.buf.push(']');
        self
    }

    pub fn obj(&mut self, k: &str, v: Json) -> &mut Self {
        self.key(k);
        self.buf.push_str(&v.finish());
        self
    }

    pub fn objs(&mut self, k: &str, v: Vec<Json>) -> &mut Self {
        self.key(k);
        self.buf.push('[');
        for (i, o) in v.into_iter().enumerate() {
            if i > 0 {
                self.buf.push(',');
            }
            self.buf.push_str(&o.finish());
        }
        self.buf.push(']');
        self
    }

    pub fn finish(mut self) -> String {
        self.buf.push('}');
        self.buf
    }
}

fn push_num(buf: &mut String, v: f64) {
    if v.is_finite() {
        // Shortest representation that reads back to the same f64.
        buf.push_str(&format!("{v:e}"));
    } else {
        buf.push_str("null");
    }
}

fn push_str(buf: &mut String, s: &str) {
    buf.push('"');
    for c in s.chars() {
        match c {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if (c as u32) < 0x20 => buf.push_str(&format!("\\u{:04x}", c as u32)),
            c => buf.push(c),
        }
    }
    buf.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_reject_unknown_and_repeated_keys() {
        assert!(Args::parse("n=3 reps=2", &["n", "reps"]).is_ok());
        assert!(matches!(Args::parse("nn=3", &["n"]), Err(Error::BadArg(_))));
        assert!(matches!(
            Args::parse("n=3,n=4", &["n"]),
            Err(Error::BadArg(_))
        ));
        assert!(matches!(Args::parse("n", &["n"]), Err(Error::BadArg(_))));
        let a = Args::parse(" n=3; ", &["n", "m"]).unwrap();
        assert_eq!(a.usize_or("n", 0).unwrap(), 3);
        assert_eq!(a.usize_or("m", 7).unwrap(), 7);
        assert!(
            Args::parse("n=-1", &["n"])
                .unwrap()
                .usize_or("n", 0)
                .is_err()
        );
    }

    #[test]
    fn catch_panic_keeps_the_message_and_clears_the_record() {
        let err = catch_panic::<i32>(|| panic!("inner {}", 7)).unwrap_err();
        assert!(err.starts_with("inner 7 (at "), "{err}");
        assert!(err.contains("ffi.rs"), "{err}");
        assert_eq!(take_last_panic(), None);
        assert_eq!(catch_panic(|| 5), Ok(5));
    }

    #[test]
    fn a_handled_panic_is_not_reported_for_a_later_one_from_another_thread() {
        let rc = barrier(|| {
            assert!(catch_panic::<()>(|| panic!("first, handled")).is_err());
            let payload = std::thread::spawn::<_, ()>(|| panic!("second, on a worker"))
                .join()
                .unwrap_err();
            panic::resume_unwind(payload)
        });
        assert_eq!(rc, PC_ERR_PANIC);
        let msg = LAST_ERROR.with(|s| s.borrow().to_string_lossy().into_owned());
        assert_eq!(msg, "panic: second, on a worker (raised on another thread)");
    }

    #[test]
    fn ptr_barrier_returns_null_on_a_panic_and_keeps_the_message() {
        let p = ptr_barrier::<c_char>(|| panic!("pointer {}", 9));
        assert!(p.is_null());
        let msg = LAST_ERROR.with(|s| s.borrow().to_string_lossy().into_owned());
        assert!(msg.starts_with("panic: pointer 9 (at "), "{msg}");
        assert!(msg.contains("ffi.rs"), "{msg}");
    }

    #[test]
    fn ptr_barrier_passes_the_pointer_through_and_clears_the_message() {
        set_last_error("stale message");
        let text = c"text";
        let p = ptr_barrier(|| text.as_ptr());
        assert_eq!(p, text.as_ptr());
        let msg = LAST_ERROR.with(|s| s.borrow().to_string_lossy().into_owned());
        assert_eq!(msg, "");
    }

    #[test]
    fn json_escapes_and_maps_non_finite_to_null() {
        let mut j = Json::default();
        j.str("s", "a\"b\\c\n").num("x", 0.08).num("nan", f64::NAN);
        j.nums("v", &[1.0, f64::INFINITY]);
        assert_eq!(
            j.finish(),
            r#"{"s":"a\"b\\c\n","x":8e-2,"nan":null,"v":[1e0,null]}"#
        );
    }
}
