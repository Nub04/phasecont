//! A panic hook that records where an engine worker panicked.
//!
//! The hook stores the message and `file:line:col` in a thread local on engine worker
//! threads and prints nothing there, because the worker turns the panic into a
//! `Panicked` state. On every other thread it calls the hook that was installed before,
//! so the test harness and the host process still report their panics.

use std::cell::{Cell, RefCell};
use std::panic::PanicHookInfo;
use std::sync::Once;

thread_local! {
    static IS_WORKER: Cell<bool> = const { Cell::new(false) };
    static LAST: RefCell<Option<(String, String)>> = const { RefCell::new(None) };
}

static INSTALL: Once = Once::new();

/// Text of a panic payload: `&str` and `String` payloads give their text, anything else
/// a fixed note.
pub fn payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}

fn record(info: &PanicHookInfo<'_>) {
    let message = payload_message(info.payload());
    let location = match info.location() {
        Some(l) => format!("{}:{}:{}", l.file(), l.line(), l.column()),
        None => "unknown".to_string(),
    };
    // try_with: a panic during thread-local teardown must not panic inside the hook.
    let _ = LAST.try_with(|c| *c.borrow_mut() = Some((message, location)));
}

/// Installs the hook. The first call replaces the process-wide panic hook and chains to
/// the previous one on every thread that is not an engine worker, so a host process
/// keeps its own reporting. Panics are recorded only on worker threads. Later calls do
/// nothing.
pub fn install_hook() {
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if IS_WORKER.try_with(Cell::get).unwrap_or(false) {
                record(info);
            } else {
                previous(info);
            }
        }));
    });
}

/// Takes the message and `file:line:col` of the last panic recorded on the calling
/// thread. Only engine worker threads record.
pub fn take_last_panic() -> Option<(String, String)> {
    LAST.try_with(|c| c.borrow_mut().take()).ok().flatten()
}

/// Forgets any recorded panic on the calling thread.
pub(crate) fn clear_last_panic() {
    let _ = LAST.try_with(|c| *c.borrow_mut() = None);
}

/// Marks the calling thread as an engine worker, so the hook records and stays silent.
pub(crate) fn mark_worker() {
    IS_WORKER.with(|c| c.set(true));
}

/// Whether the calling thread is marked as an engine worker.
#[cfg(test)]
pub(crate) fn is_worker() -> bool {
    IS_WORKER.try_with(Cell::get).unwrap_or(false)
}

/// Restores the saved mark and recorded panic on drop.
struct Restore {
    worker: bool,
    last: Option<(String, String)>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = IS_WORKER.try_with(|c| c.set(self.worker));
        let last = self.last.take();
        let _ = LAST.try_with(|c| *c.borrow_mut() = last);
    }
}

/// Runs `f` with the calling thread marked as an engine worker, so the hook records a
/// panic inside `f` (message and file:line:col) and prints nothing. The thread's
/// previous mark and recorded panic are saved before `f` and restored after it, also
/// when `f` unwinds, so nested and repeated calls do not leak state.
///
/// Inside `f` the recorded panic starts cleared, so [`take_last_panic`] after a caught
/// panic returns that panic. This function does not call [`install_hook`]; callers do.
/// The C ABI wraps each export like this:
///
/// ```ignore
/// install_hook();
/// recording(|| match catch_unwind(AssertUnwindSafe(body)) {
///     Ok(v) => Ok(v),
///     Err(p) => Err(take_last_panic()
///         .unwrap_or_else(|| (payload_message(&*p), "unknown".into()))),
/// })
/// ```
pub fn recording<R>(f: impl FnOnce() -> R) -> R {
    let _restore = Restore {
        worker: IS_WORKER.try_with(Cell::get).unwrap_or(false),
        last: LAST.try_with(|c| c.borrow_mut().take()).ok().flatten(),
    };
    mark_worker();
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn recording_gives_the_panic_inside_the_closure() {
        install_hook();
        recording(|| {
            let _ = catch_unwind(|| panic!("boom"));
            let (msg, loc) = take_last_panic().expect("recorded");
            assert_eq!(msg, "boom");
            let loc = loc.replace('\\', "/");
            assert!(loc.contains("panic.rs:"), "{loc}");
            let mut parts = loc.rsplitn(3, ':');
            assert!(parts.next().unwrap().parse::<u32>().is_ok());
            assert!(parts.next().unwrap().parse::<u32>().is_ok());
        });
    }

    #[test]
    fn recording_leaves_no_mark_or_panic_behind() {
        install_hook();
        assert!(!is_worker());
        recording(|| {
            assert!(is_worker());
            let _ = catch_unwind(|| panic!("inside"));
        });
        assert!(!is_worker());
        assert_eq!(take_last_panic(), None);
    }

    #[test]
    fn nested_recording_restores_the_outer_state() {
        install_hook();
        mark_worker();
        clear_last_panic();
        let _ = catch_unwind(|| panic!("outer"));
        recording(|| {
            assert_eq!(take_last_panic(), None);
            let _ = catch_unwind(|| panic!("inner"));
        });
        assert!(is_worker());
        assert_eq!(take_last_panic().map(|p| p.0), Some("outer".to_string()));
        IS_WORKER.with(|c| c.set(false));
    }

    #[test]
    fn recording_restores_state_when_the_closure_panics() {
        install_hook();
        let r = catch_unwind(AssertUnwindSafe(|| {
            recording(|| {
                let _ = catch_unwind(|| panic!("first"));
                panic!("second");
            })
        }));
        assert!(r.is_err());
        assert!(!is_worker());
        assert_eq!(take_last_panic(), None);
    }
}
