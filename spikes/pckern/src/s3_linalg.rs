//! S3: linear algebra for the collocation matrices of SPEC section 7.
//!
//! `matrix` builds the periodic orbit collocation matrix, `sparse` solves it
//! with a sparse LU of the whole matrix, `condense` solves it by
//! condensation of the collocation blocks, and `run` times both and writes
//! the JSON reports.

use std::ffi::c_char;

use crate::ffi::{self, Args, bad_arg, barrier, into_c_string, out_arg, str_arg};
use crate::registry;

pub mod condense;
pub mod lu_probe;
pub mod matrix;
pub mod run;
pub mod sparse;

use matrix::{BUILD_KEYS, Params, Problem};
use run::SolveArgs;

/// Builds the problem once and runs every method (contract section 4).
pub fn run(args: &str) -> ffi::Result<String> {
    run::run(args)
}

/// Builds the matrix and right-hand side and stores them under a handle.
///
/// # Safety
/// `args` must be NULL or a NUL-terminated string, and each output pointer
/// must be NULL or valid for one write.
pub unsafe extern "C" fn s3_build(
    args: *const c_char,
    out_handle: *mut u64,
    out_n: *mut usize,
    out_nnz: *mut usize,
) -> i32 {
    barrier(|| {
        let out_handle = unsafe { out_arg(out_handle, "s3_build: handle output") }?;
        let out_n = unsafe { out_arg(out_n, "s3_build: n output") }?;
        let out_nnz = unsafe { out_arg(out_nnz, "s3_build: nnz output") }?;
        let args = unsafe { str_arg(args, "s3_build: args") }?;
        let params = Params::from_args(&Args::parse(args, &BUILD_KEYS)?)?;
        let p = Problem::build(&params)?;
        let (nr, nnz) = (p.nr, p.nnz);
        *out_handle = registry::insert(p);
        *out_n = nr;
        *out_nnz = nnz;
        Ok(())
    })
}

/// Copies the triplets (1-based indices as f64) and the right-hand side.
///
/// # Safety
/// `i`, `j` and `v` must be NULL or valid for `nnz` writes, and `b` must be
/// NULL or valid for `n` writes.
pub unsafe extern "C" fn s3_copy(
    h: u64,
    n: usize,
    nnz: usize,
    i: *mut f64,
    j: *mut f64,
    v: *mut f64,
    b: *mut f64,
) -> i32 {
    barrier(|| {
        let p = registry::get::<Problem>(h)?;
        if n != p.nr {
            return Err(bad_arg(format!(
                "s3_copy: n={n}, but the problem has nr={}",
                p.nr
            )));
        }
        if nnz != p.nnz {
            return Err(bad_arg(format!(
                "s3_copy: nnz={nnz}, but the problem has nnz={}",
                p.nnz
            )));
        }
        if i.is_null() || j.is_null() || v.is_null() || b.is_null() {
            return Err(bad_arg("s3_copy: NULL data pointer"));
        }
        // The sizes match the stored problem and n, nnz >= 1, so the slices
        // cover exactly the memory the caller allocated.
        let (ii, jj, vv, bb) = unsafe {
            (
                std::slice::from_raw_parts_mut(i, nnz),
                std::slice::from_raw_parts_mut(j, nnz),
                std::slice::from_raw_parts_mut(v, nnz),
                std::slice::from_raw_parts_mut(b, n),
            )
        };
        let mut k = 0;
        p.for_each_entry(|row, col, val| {
            ii[k] = (row + 1) as f64;
            jj[k] = (col + 1) as f64;
            vv[k] = val;
            k += 1;
        });
        if k != nnz {
            return Err(ffi::internal(format!(
                "s3_copy: wrote {k} entries, expected {nnz}"
            )));
        }
        bb.copy_from_slice(&p.b);
        Ok(())
    })
}

/// Solves the stored system with one method and reports the timings.
///
/// # Safety
/// `args` must be NULL or a NUL-terminated string, `out_x` must be NULL or
/// valid for `n` writes, and `out_json` must be NULL or valid for one write.
pub unsafe extern "C" fn s3_solve(
    h: u64,
    n: usize,
    args: *const c_char,
    out_x: *mut f64,
    out_json: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let out_json = unsafe { out_arg(out_json, "s3_solve: json output") }?;
        *out_json = std::ptr::null_mut();
        let args = unsafe { str_arg(args, "s3_solve: args") }?;
        let s = SolveArgs::parse(args)?;
        let p = registry::get::<Problem>(h)?;
        if n != p.nr {
            return Err(bad_arg(format!(
                "s3_solve: n={n}, but the problem has nr={}",
                p.nr
            )));
        }
        if out_x.is_null() {
            return Err(bad_arg("s3_solve: x output is NULL"));
        }
        // solve rejects condensed_dense above DENSE_MAX before any work.
        let out = run::solve(&p, s.method, s.threads, s.reps)?;
        // n equals nr, and the caller allocated n values for x.
        unsafe { std::slice::from_raw_parts_mut(out_x, n) }.copy_from_slice(&out.x);
        *out_json = into_c_string(&run::solve_json(&p, &s, &out));
        Ok(())
    })
}
