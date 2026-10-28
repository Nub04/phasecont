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
pub(crate) fn payload_message(payload: &(dyn std::any::Any + Send)) -> String {
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
