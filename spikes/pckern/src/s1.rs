//! S1: FFI round trip. `scale` is the smallest real data path, the two panic
//! functions check the barrier, and `test_handle` checks the registry.

use rayon::prelude::*;

use crate::ffi::{Result, bad_arg, barrier, internal, out_arg};
use crate::registry;

/// Writes `a * x[i]` to `out[i]`.
///
/// # Safety
/// For `n > 0`, `x` must be valid for `n` reads and `out` for `n` writes.
pub unsafe extern "C" fn scale(x: *const f64, n: usize, a: f64, out: *mut f64) -> i32 {
    barrier(|| {
        // mxGetDoubles returns NULL for an empty array, so check n first.
        if n == 0 {
            return Ok(());
        }
        if x.is_null() || out.is_null() {
            return Err(bad_arg("scale: NULL data pointer"));
        }
        for i in 0..n {
            // Raw reads and writes instead of slices, so out may alias x.
            unsafe { *out.add(i) = a * *x.add(i) };
        }
        Ok(())
    })
}

/// Panics on the calling thread. The "%" checks that the gateway passes the
/// message to mexErrMsgIdAndTxt as an argument, not as the format string.
pub extern "C" fn panic_now() -> i32 {
    barrier(|| panic!("spike panic: 100% on purpose"))
}

/// Panics inside a rayon worker. Rayon carries the panic back to the thread
/// that called `install`, where the barrier catches it. The pool is local, so
/// its threads exit when it is dropped instead of living on inside MATLAB.
pub extern "C" fn panic_in_worker() -> i32 {
    barrier(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("pckern-s1-{i}"))
            .build()
            .map_err(|e| internal(format!("cannot start the thread pool: {e}")))?;
        pool.install(|| {
            (0..64u32).into_par_iter().for_each(|i| {
                if i == 37 {
                    panic!("spike worker panic at item {i}");
                }
            })
        });
        Ok(())
    })
}

struct TestObject;

/// Registers a small object and writes its id.
///
/// # Safety
/// `out` must be NULL or valid for one write.
pub unsafe extern "C" fn test_handle(out: *mut u64) -> i32 {
    barrier(|| {
        let out = unsafe { out_arg(out, "test_handle: output") }?;
        *out = registry::insert(TestObject);
        Ok(())
    })
}

pub extern "C" fn free_handle(id: u64) -> i32 {
    barrier(|| -> Result<()> { registry::remove(id) })
}
