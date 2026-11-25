//! Replaces the MinGW-w64 CRT `_fpreset` in the `pc-cli` executable.
//!
//! The msvcrt flavour of MinGW-w64 calls `_fpreset` from the exe startup
//! code, and its own version sets the x87 control word to 0x037F (64-bit
//! precision). Threads that Windows starts keep the default 0x027F (53-bit
//! precision). The x87 code in that CRT's `pow`, `exp` and related
//! functions then rounds differently on the main thread and on worker
//! threads. The engine already routes those functions through
//! `pc_ir::hostmath`; this override makes the main thread match the worker
//! threads for any call that still reaches the CRT. The DLL that MATLAB
//! loads has no exe startup code, so it does not need this.
//!
//! The linker takes this definition over the one in the CRT archive, since
//! a symbol defined in an object file wins over an archive member.
//! `tests/main_thread_x87.rs` links this file into its own exe and checks
//! the main thread control word.

#[cfg(all(target_os = "windows", target_env = "gnu", target_arch = "x86_64"))]
#[unsafe(no_mangle)]
pub extern "C" fn _fpreset() {
    let cw: u16 = 0x027F;
    // SAFETY: `fninit` resets the x87 unit and `fldcw` loads a control word
    // read from a live local. Neither touches memory other than `cw` or
    // the stack. The CRT calls `_fpreset` before `main`, when no x87 state
    // is live.
    unsafe {
        std::arch::asm!("fninit", "fldcw [{}]", in(reg) &cw, options(nostack, readonly));
    }
}
