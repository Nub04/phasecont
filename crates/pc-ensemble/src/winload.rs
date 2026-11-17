//! Loading the GPU libraries on Windows without searching the current folder.
//!
//! cudarc and cl3 open their libraries by bare name: cudarc tries `cuda.dll` and `nvrtc.dll`
//! before `nvcuda.dll` and `nvrtc64_*.dll`, and cl3 opens `OpenCL.dll`. A bare-name load in
//! the standard search order also looks in the process's current folder, which inside
//! MATLAB is the user's current folder. A file with one of those names in a downloaded
//! project folder would then run inside MATLAB.
//!
//! So the backends do their first loads inside a [`NoCwdSearch`] window, which takes the
//! current folder out of the search with `SetDllDirectoryW(L"")` and puts the previous
//! setting back when it ends. cudarc and cl3 keep the libraries they opened, so later calls
//! never search again. The driver and the OpenCL loader are also loaded first from the
//! system folder by [`system_library`], and a backend whose library is not there reports
//! itself unavailable without any bare-name load.
//!
//! What stays: a bare-name load inside the window still searches the folder of the host
//! program (MATLAB's `bin\win64`), the system folders and the folders on `PATH`. Changing
//! `PATH` is not safe while other threads run, so a folder on `PATH` that someone else can
//! write to remains a risk, as it is for every program on that machine.
//!
//! On other systems the window and the system-folder load do nothing, and `pinned` is a plain
//! load by path: `dlopen` does not search the current folder unless `LD_LIBRARY_PATH` names
//! it.

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, Library,
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
        fn GetDllDirectoryW(len: u32, buf: *mut u16) -> u32;
        fn GetModuleHandleExW(flags: u32, addr: *const c_void, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, buf: *mut u16, len: u32) -> u32;
    }

    const FROM_ADDRESS: u32 = 0x4;
    const UNCHANGED_REFCOUNT: u32 = 0x2;
    /// The longest path `GetDllDirectoryW` and `GetModuleFileNameW` can return.
    const MAX_PATH_LONG: usize = 32_768;

    /// One window at a time, so two windows on two threads cannot restore out of order.
    static WINDOW: Mutex<()> = Mutex::new(());

    /// While alive, bare-name library loads in this process skip the current folder.
    pub struct NoCwdSearch {
        /// The `SetDllDirectoryW` value before the window, NUL-terminated, or `None` when
        /// none was set. The API returns an empty string both when nothing was set and
        /// when it was set to `L""`, so `None` covers both and the window restores the
        /// default order (which MATLAB R2026b uses: the proof of concept in
        /// `docs/STATUS.md` saw its current folder searched).
        prev: Option<Vec<u16>>,
        set: bool,
        _lock: MutexGuard<'static, ()>,
    }

    impl NoCwdSearch {
        pub fn new() -> NoCwdSearch {
            let lock = WINDOW.lock().unwrap_or_else(|p| p.into_inner());
            let mut buf = vec![0u16; MAX_PATH_LONG];
            // SAFETY: the buffer holds `buf.len()` u16 values.
            let n = unsafe { GetDllDirectoryW(buf.len() as u32, buf.as_mut_ptr()) } as usize;
            let prev = (n > 0 && n < buf.len()).then(|| {
                buf.truncate(n);
                buf.push(0);
                buf
            });
            let empty = [0u16];
            // SAFETY: `empty` is a NUL-terminated wide string.
            let set = unsafe { SetDllDirectoryW(empty.as_ptr()) } != 0;
            NoCwdSearch {
                prev,
                set,
                _lock: lock,
            }
        }
    }

    impl Drop for NoCwdSearch {
        fn drop(&mut self) {
            if self.set {
                let p = self.prev.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
                // SAFETY: `p` is NULL or a NUL-terminated wide string that outlives the call.
                unsafe { SetDllDirectoryW(p) };
            }
        }
    }

    /// Loads `name` from the Windows system folder only and keeps it loaded, so a later
    /// bare-name load of the same name gets this copy. False when it is not there.
    pub fn system_library(name: &str) -> bool {
        // SAFETY: loading a library runs its initializers, as any dynamic load does. The
        // flag limits the search to the system folder.
        match unsafe { Library::load_with_flags(name, LOAD_LIBRARY_SEARCH_SYSTEM32) } {
            Ok(lib) => {
                std::mem::forget(lib);
                true
            }
            Err(_) => false,
        }
    }

    /// Loads the library at the absolute path `path` and keeps it loaded. Its own
    /// dependencies come from its folder and the system folder.
    pub fn pinned(path: &Path) -> bool {
        if !path.is_absolute() {
            return false;
        }
        // SAFETY: as in `system_library`.
        match unsafe {
            Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        } {
            Ok(lib) => {
                std::mem::forget(lib);
                true
            }
            Err(_) => false,
        }
    }

    /// The folder of the loaded module that contains `addr`.
    pub fn module_dir(addr: *const c_void) -> Option<PathBuf> {
        let mut module = std::ptr::null_mut();
        // SAFETY: the flags make the call look up the module by address without changing
        // its reference count, and `module` is valid for one write.
        if unsafe { GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, addr, &mut module) } == 0
        {
            return None;
        }
        let mut buf = vec![0u16; MAX_PATH_LONG];
        // SAFETY: the buffer holds `buf.len()` u16 values.
        let n = unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        use std::os::windows::ffi::OsStringExt;
        let file = PathBuf::from(std::ffi::OsString::from_wide(&buf[..n]));
        file.parent().map(Path::to_path_buf)
    }
}

#[cfg(not(windows))]
mod imp {
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};

    pub struct NoCwdSearch;

    impl NoCwdSearch {
        pub fn new() -> NoCwdSearch {
            NoCwdSearch
        }
    }

    pub fn system_library(_name: &str) -> bool {
        true
    }

    /// Loads the library at the absolute path `path` and keeps it loaded.
    pub fn pinned(path: &Path) -> bool {
        if !path.is_absolute() {
            return false;
        }
        // SAFETY: loading a library runs its initializers, as any dynamic load does.
        match unsafe { libloading::Library::new(path) } {
            Ok(lib) => {
                std::mem::forget(lib);
                true
            }
            Err(_) => false,
        }
    }

    pub fn module_dir(_addr: *const c_void) -> Option<PathBuf> {
        None
    }
}

pub(crate) use imp::{NoCwdSearch, module_dir, pinned, system_library};

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn the_window_takes_the_current_folder_out_and_puts_the_setting_back() {
        unsafe extern "system" {
            fn GetDllDirectoryW(len: u32, buf: *mut u16) -> u32;
        }
        let read = || {
            let mut buf = vec![0u16; 1024];
            // SAFETY: the buffer holds 1024 u16 values.
            let n = unsafe { GetDllDirectoryW(1024, buf.as_mut_ptr()) };
            String::from_utf16_lossy(&buf[..n as usize])
        };
        let before = read();
        {
            let _w = NoCwdSearch::new();
            assert_eq!(read(), "");
        }
        assert_eq!(read(), before);
    }

    #[test]
    fn the_system_folder_has_kernel32_and_not_a_made_up_name() {
        assert!(system_library("kernel32.dll"));
        assert!(!system_library("pc-no-such-library-4c1f.dll"));
    }

    #[test]
    fn a_relative_path_is_never_pinned() {
        assert!(!pinned(std::path::Path::new("nvrtc64_130_0.dll")));
    }
}
