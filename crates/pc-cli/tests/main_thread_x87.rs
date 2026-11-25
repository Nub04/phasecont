//! The `_fpreset` override of `pc-cli` sets the x87 control word of the main thread.
//!
//! The msvcrt flavour of MinGW-w64 calls `_fpreset` from the exe startup code and its
//! own version leaves the main thread at 64-bit x87 precision (control word 0x037F),
//! while threads that Windows starts run at 53-bit precision (0x027F). `pc-cli` replaces
//! `_fpreset`, and this test links the same source file into its own exe. libtest runs
//! every test on a spawned thread, so this file has `harness = false` and checks the
//! main thread from `main`.
//!
//! On every target other than x86_64 windows-gnu the test only prints that it skipped.

// The override that `pc-cli` links, compiled into this exe as well.
#[path = "../src/fpreset.rs"]
mod fpreset;

#[cfg(all(target_os = "windows", target_env = "gnu", target_arch = "x86_64"))]
fn control_word() -> u16 {
    let mut cw: u16 = 0;
    // SAFETY: fnstcw stores two bytes to a valid, aligned u16 on this stack frame.
    unsafe { std::arch::asm!("fnstcw [{0}]", in(reg) &mut cw, options(nostack)) };
    cw
}

#[cfg(all(target_os = "windows", target_env = "gnu", target_arch = "x86_64"))]
fn main() {
    const PC53: u16 = 0x027F;
    let main_cw = control_word();
    let worker_cw = std::thread::spawn(control_word)
        .join()
        .expect("the worker thread panicked");
    println!("x87 control word: main thread {main_cw:#06x}, worker thread {worker_cw:#06x}");
    assert_eq!(main_cw, PC53, "main thread control word");
    assert_eq!(worker_cw, PC53, "worker thread control word");
    println!("main_thread_x87: ok");
}

#[cfg(not(all(target_os = "windows", target_env = "gnu", target_arch = "x86_64")))]
fn main() {
    println!("main_thread_x87: skipped, the override exists only on x86_64 windows-gnu");
}
