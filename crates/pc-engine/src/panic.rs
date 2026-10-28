//! A panic hook that records where an engine worker panicked.
//!
//! The hook stores the message and `file:line:col` in a thread local. On a thread marked
//! as an engine worker it prints nothing, because the worker turns the panic into a
//! `Panicked` state. On every other thread it calls the hook that was installed before,
//! so the test harness and other code still report their panics.

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
    LAST.with(|c| *c.borrow_mut() = Some((message, location)));
}

/// Installs the hook. Safe to call many times; only the first call has an effect.
pub fn install_hook() {
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            record(info);
            if !IS_WORKER.with(Cell::get) {
                previous(info);
            }
        }));
    });
}

/// Takes the message and `file:line:col` of the last panic on the calling thread.
pub fn take_last_panic() -> Option<(String, String)> {
    LAST.with(|c| c.borrow_mut().take())
}

/// Marks the calling thread as an engine worker, so the hook stays silent on it.
pub(crate) fn mark_worker() {
    IS_WORKER.with(|c| c.set(true));
}
