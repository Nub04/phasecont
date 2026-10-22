# API notes for the spikes

Working notes, copied from a research pass over the cached crate sources on 2026-10-20. When they were written, nothing had been compiled. The Cranelift part has since been built in S2, and the faer calls that S3 needs were checked against the 0.24.4 sources (see `NOTES.md`).

Corrections found while building S2:

- Cranelift 0.137 takes `MemFlagsData::trusted()` in `load` and `store`, not `MemFlags::trusted()`. See `pckern/src/s2_jit.rs`.
- `FunctionBuilder::finalize` takes a `TargetFrontendConfig`.
- The Fedora package `rust-std-static-x86_64-pc-windows-gnu` exists (see `docs/ENVIRONMENT.md`), but it is not installed yet.

The recipes below are taken from the cached crate sources. Nothing was built or run, so the code is checked against signatures but not compiled. Functions return `Result<_, Box<dyn std::error::Error>>`.

## 1. Cranelift 0.137 JIT

Add `cranelift-{codegen,frontend,module,jit,native} = "0.137"` (edition 2024, MSRV 1.97, Apache-2.0 WITH LLVM-exception).

```rust
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlagsData, UserFuncName};
use cranelift_codegen::settings::{self, Configurable}; // Configurable is required for .set()
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

extern "C" fn pc_exp(x: f64) -> f64 { x.exp() } // must never panic

let mut fb = settings::builder();
fb.set("opt_level", "speed")?;
fb.set("use_colocated_libcalls", "false")?;
fb.set("is_pic", "true")?;                  // what JITBuilder::with_flags sets on x86_64
fb.set("enable_probestack", "true")?;
fb.set("probestack_strategy", "inline")?;   // "outline" calls __cranelift_probestack, which nothing provides
let isa = cranelift_native::builder()?      // Err is &'static str; does not panic
    .finish(settings::Flags::new(fb))?;
let mut jb = JITBuilder::with_isa(isa, default_libcall_names()); // returns Self, not Result
jb.symbol("pc_exp", pc_exp as *const u8);
let mut module = JITModule::new(jb);

let ptr = module.target_config().pointer_type();
let mut sig = module.make_signature();      // call_conv = ISA default
for _ in 0..3 { sig.params.push(AbiParam::new(ptr)); }
let fid = module.declare_function("rhs", Linkage::Export, &sig)?;
let mut es = module.make_signature();
es.params.push(AbiParam::new(types::F64)); es.returns.push(AbiParam::new(types::F64));
let eid = module.declare_function("pc_exp", Linkage::Import, &es)?;

let mut ctx = module.make_context();
ctx.func.signature = sig;
ctx.func.name = UserFuncName::user(0, fid.as_u32());
let exp_ref = module.declare_func_in_func(eid, &mut ctx.func); // do this before the builder borrows func
let mut fbc = FunctionBuilderContext::new();
{
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbc);
    let blk = b.create_block();
    b.append_block_params_for_function_params(blk);
    b.switch_to_block(blk); b.seal_block(blk);
    let (x, p, out) = { let v = b.block_params(blk); (v[0], v[1], v[2]) };
    let mf = MemFlagsData::trusted();
    let x0 = b.ins().load(types::F64, mf, x, 0);
    let p0 = b.ins().load(types::F64, mf, p, 0);
    let p1 = b.ins().load(types::F64, mf, p, 8);
    let t = b.ins().fmul(p0, x0); let t = b.ins().fsub(t, p1);
    let c = b.ins().call(exp_ref, &[t]); let e = b.inst_results(c)[0];
    let r = b.ins().fadd(e, x0);
    b.ins().store(mf, r, out, 0);
    b.ins().return_(&[]);
    b.finalize(module.target_config());     // this argument is new
}
module.define_function(fid, &mut ctx)?;
module.clear_context(&mut ctx);
module.finalize_definitions()?;
let f: extern "C" fn(*const f64, *const f64, *mut f64) =
    unsafe { std::mem::transmute(module.get_finalized_function(fid)) };
```

Gotchas:
- **Calling convention:** the default comes from `triple_default(Triple::host())`. target-lexicon sets the host triple from cargo's `TARGET` at build time, so the windows-gnu build gets **WindowsFastcall**, which matches Rust `extern "C"` on Win64. Linux gets SystemV. Don't hardcode either.
- **Assertion in `JITModule::new`:** it asserts `!is_pic || x86_64`.
- **Builder panics:** `JITBuilder::new` and `with_flags` panic on an unsupported host. Use `with_isa`.
- **Unresolved symbols:** `finalize_definitions` panics with "can't resolve symbol {name}". Wrap it in `catch_unwind`.
- **Windows symbol lookup:** it only checks `GetProcAddress(NULL)` (MATLAB.exe) and `ucrtbase.dll`, never our DLL. Every shim must be registered with `jb.symbol`.
- **Memory leak on drop:** `Memory::drop` calls `mem::forget`, so dropping a JITModule leaks its code pages. Keep it in an `Option<JITModule>` and call `unsafe { m.free_memory() }` in your wrapper's Drop.
- **Changes since the old demo:**
  - `finalize` now takes a `TargetFrontendConfig`.
  - `declare_var(ty)` now returns the `Variable`.
  - `JITBuilder::new` returns a `Result`.
  - `ExtFuncData` has a new `patchable` field.
  - The bundled jit-minimal example still sets `is_pic=false` with an outdated FIXME.

## 2. faer 0.24.4

`default-features = false, features = ["std","rayon","sparse-linalg"]` works. The `sparse` and `linalg` modules are not feature-gated; the features only gate prelude exports, and `rand`/`npy` only gate io and stats.

```rust
use faer::{Col, ColMut, ColRef, Mat, Par};
use faer::linalg::solvers::Solve;  // blanket impl for every SolveCore type
use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};

faer::set_global_parallelism(Par::Seq);
let trips: Vec<Triplet<usize, usize, f64>> = vec![Triplet::new(0, 0, 4.0) /* ... */];
let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trips)?; // duplicates are summed
let t0 = Instant::now();
let sym = SymbolicLu::try_new(a.symbolic())?;                 // FaerError; Arc-backed, cheap to clone
let t1 = Instant::now();
let lu = Lu::try_new_with_symbolic(sym.clone(), a.as_ref())?; // LuError
let t2 = Instant::now();
let mut x = b.clone();                                         // b: Vec<f64>
lu.solve_in_place(ColMut::from_slice_mut(&mut x));             // or: let x: Col<f64> = lu.solve(&bcol);
// one-shot alternative: a.sp_lu()?

let m = Mat::<f64>::from_fn(n, n, |i, j| /* ... */);           // or Mat::zeros(n, n)
let xd: Col<f64> = m.partial_piv_lu().solve(&bcol);
let xs: &[f64] = xd.try_as_col_major().unwrap().as_slice();
let c: ColRef<'_, f64> = m.col(j); let s: &[f64] = m.col_as_slice(j);
let bcol = Col::<f64>::from_fn(n, |i| b[i]); let bv = ColRef::from_slice(&b);
```

Gotchas:
- **Symbolic vs numeric timing:** for small or very sparse matrices the symbolic factor is `SymbolicLuRaw::Simplicial`, which stores only the dimensions and the COLAMD permutation. The symbolic time is therefore just the ordering; pivoting and fill happen in the numeric phase.
- **Same pattern, new values:** use `SymbolicSparseColMat::try_new_from_indices` followed by `SparseColMat::new_from_argsort`.
- **Error behaviour:**
  - Sparse LU reports `LuError::SymbolicSingular{index}` for a structurally singular matrix.
  - Dense `partial_piv_lu` never returns an error, so check the result for non-finite values.
  - `solve` panics (`#[track_caller]` asserts) on a dimension mismatch.
  - `MemBuffer::new` panics on out-of-memory.
- **Rayon threads:** the global pool's threads never exit. Unloading the DLL while they are alive can crash MATLAB, which is one more reason to use `Par::Seq`.
- **Licenses:** every crate in the dependency tree is on the allow-list:
  - MIT: gemm, pulp, dyn-stack, equator, nano-gemm, spindle, raw-cpuid, qd, private-gemm-x86 0.1.22 (0.1.20 has been yanked).
  - MIT/Apache: the rayon family, num-*, half and others.
  - atomic-wait: BSD-2-Clause.
  - zerocopy: BSD-2/Apache/MIT.
  - bytemuck: Zlib/Apache/MIT.
  - unicode-ident: MIT/Apache plus Unicode-3.0.

## 3. cudarc 0.19.10

No CUDA SDK is needed at build time. The `CUDARC_CUDA_VERSION` env var overrides the version feature. cudarc is MIT OR Apache-2.0, libloading 0.9.0 is ISC, and windows-link 0.2.1 is MIT/Apache.

```rust
use cudarc::driver::{sys as dsys, CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{self, compile_ptx_with_opts, sys as nsys, CompileOptions, Ptx};

// (e) Non-panicking probe. Run everything after it inside catch_unwind(AssertUnwindSafe(..)).
if !unsafe { dsys::is_culib_present() } || !unsafe { nsys::is_culib_present() } { /* no GPU */ }
// (d) There are no safe wrappers; use the raw calls.
let mut dv = 0; unsafe { dsys::cuDriverGetVersion(&mut dv) }.result()?; // 13040 means 13.4; works before cuInit
let (mut nj, mut nn) = (0, 0); unsafe { nsys::nvrtcVersion(&mut nj, &mut nn) }.result()?;
// (a)
let ctx = CudaContext::new(0)?;              // Arc<CudaContext>: cuInit + primary context
let stream = ctx.default_stream();           // or ctx.new_stream()?
let (name, (maj, min)) = (ctx.name()?, ctx.compute_capability()?); // (12, 0) here
// (b)
const SRC: &str = r#"extern "C" __global__ void saxpy(float a,const float*x,float*y,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=a*x[i]+y[i];}"#;
let ptx = compile_ptx_with_opts(SRC, CompileOptions { arch: Some("sm_120"), ..Default::default() })?;
// (c)
let module = ctx.load_module(ptx)?;
let f = module.load_function("saxpy")?;
let x_d = stream.clone_htod(&xh)?; let mut y_d = stream.clone_htod(&yh)?;
let (a, n) = (2.0f32, xh.len() as i32);
let mut l = stream.launch_builder(&f);
l.arg(&a).arg(&x_d).arg(&mut y_d).arg(&n);
unsafe { l.launch(LaunchConfig::for_num_elems(n as u32)) }?;
let y: Vec<f32> = stream.clone_dtoh(&y_d)?;  // memcpy_dtoh(&y_d, &mut buf) to fill an existing buffer
```

**Cubin path.** `Program` is `pub(crate)` and the safe API only calls `get_ptx`, so getting a cubin means driving `result` and `sys` by hand:

```rust
let src = std::ffi::CString::new(SRC)?;
let prog = nvrtc::result::create_program(&src, None)?;
unsafe { nvrtc::result::compile_program(prog, &[format!("--gpu-architecture=sm_{maj}{min}")]) }?;
let mut sz = 0usize; unsafe { nsys::nvrtcGetCUBINSize(prog, &mut sz) }.result()?;
let mut cubin = vec![0u8; sz];
unsafe { nsys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()) }.result()?;
unsafe { nvrtc::result::destroy_program(prog) }?; // call this on every error path too
let module = ctx.load_module(Ptx::from_binary(cubin))?;
// If compiling fails, unsafe { nvrtc::result::get_program_log(prog) } returns the log as Vec<c_char>.
```

**sm_120 vs compute_120.** `arch` becomes `--gpu-architecture=<arch>`.
- **`compute_120`:** produces PTX only.
- **`sm_120`:** also runs ptxas. ptxas errors show up at compile time, and a cubin is available.
- **PTX availability:** the safe API returns PTX in both cases. Whether `nvrtcGetPTX` still succeeds after an `sm_` compile depends on NVRTC itself, which the cudarc source can't show. Check it on the first run.

**Which path to use.** PTX from NVRTC 13.4 only loads on a driver at version 13.4 or newer. On an older driver it fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. An sm_120 cubin should load on any 13.x driver (NVIDIA's minor-version compatibility).
- If `dv >= nj*1000 + nn*10`, use PTX.
- Otherwise, if `dv >= 13000`, use the cubin.
- Otherwise, report that the driver is too old.

On this machine `nvidia-smi` shows an RTX 5060 Laptop GPU with driver 617.42 and compute capability 12.0.

**Gotchas seen in the cudarc source:**
- `use_fast_math: Some(true)` only emits `--fmad=true`. For real fast math, push `"--use_fast_math"` into `options`.
- S4's `max_ulp_unfused` is not an accuracy figure. Where `a*x + y` nearly cancels, rounding `a*x` before the add moves the result by up to half an ulp of `a*x`, which can be millions of ulps of the small result, so the value only shows that NVRTC fused the multiply-add.
- `arch` is a `&'static str`. To pick the arch at runtime, push `format!("--gpu-architecture=sm_{maj}{min}")` into `options` instead.
- `memcpy_stod` and `memcpy_dtov` are deprecated. Use `clone_htod` and `clone_dtoh`.
- You must import `PushKernelArg`. `launch` is `unsafe` and returns `Option<(CudaEvent, CudaEvent)>`, which is `None` unless timing flags are set.

**Panics to wrap in catch_unwind:**
- `culib()` panics with "Unable to dynamically load the \"nvrtc\" shared library - searched for library names: ...". The OnceLock stays empty after the panic, so a later retry can still succeed.
- Each `sys` function loads its symbol on first use and panics with "Missing symbol {name}: {e}" if it's absent.
- Some driver symbols are first used inside `Drop` impls. `CudaEvent` (driver/safe/core.rs:540) calls `cuEventDestroy_v2`, `CudaStream` (:664) calls `cuStreamDestroy_v2`, `CudaSlice` (:855) calls `cuMemFreeAsync` or `cuMemFree_v2`, and `CudaModule` (:2253) calls `cuModuleUnload`. If the driver lacks one of them, `load` (driver/sys/mod.rs:12) panics inside `drop`. Inside `catch_unwind` that panic is caught like any other, but if the `drop` runs while another panic is unwinding, Rust aborts the process, and MATLAB with it. `is_culib_present` only checks that the library loads. In the spike this needs a driver that loads but lacks two of these symbols, so it is theoretical. The engine keeps these objects longer and across more code, so its GPU host code has to check these symbols before it creates the first object.
- `Program::create` panics if the source contains a NUL. The compile-error path unwraps `get_program_log`, and `Program::drop` unwraps `destroy_program`.
- `load_module`, `load_function` and `compile_program` unwrap `CString::new`. `memcpy_htod` and `memcpy_dtoh` assert that `dst.len() >= src.len()`.
- Keep `panic = "unwind"` in the release profile, otherwise catch_unwind does nothing. MATLAB doesn't show stderr, so install `std::panic::set_hook` to capture the message.

**(f) Library names and search path:**
- **Driver library:**
  - Windows: `nvcuda.dll` in System32, installed with the display driver. It is present on this machine.
  - Linux: `libcuda.so.1`, found through the loader cache. On WSL2 it is `/usr/lib/wsl/lib/libcuda.so.1`.
  - cudarc tries the "cuda" names first, then "nvcuda".
- **NVRTC on Windows (cuda-13040 build):** the candidate list includes `nvrtc64_130_0.dll`, the real CUDA 13 name. It does not include `nvrtc64_120_0.dll`, so a CUDA 12 NVRTC is never found on Windows.
- **NVRTC on Linux:** cudarc tries `libnvrtc.so`, then `libnvrtc.so.13`, `.so.12` and so on. A 12.x library would load, and then any 13.x-only symbol panics when it is first called.
- **Windows search order:** libloading's `Library::new` calls `LoadLibraryExW(name, 0, 0)`. That searches MATLAB.exe's folder, the system folders, the current directory and PATH, but not our DLL's folder. cudarc has no environment variable or path override.
- **Where the toolkit puts the DLLs:** CUDA 13.4 on this machine has them in a new `x64` subfolder:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64\nvrtc64_130_0.dll`
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64\nvrtc-builtins64_134.dll`
- **Fix:** before cudarc's first call, load both DLLs by absolute path with libloading 0.9 (ISC) and keep the handles alive. Windows reuses an already-loaded module with the same base name. Alternatively, prepend that folder to PATH with MATLAB `setenv` before the first call.
- **Licensing:** shipping the NVRTC DLLs is governed by the NVIDIA CUDA EULA, not by the crate license allow-list.

**Cross-compile toolchain on this machine:** the Rust sysroot only has `x86_64-unknown-linux-gnu`, and neither `x86_64-w64-mingw32-gcc` nor `x86_64-w64-mingw32-dlltool` is installed.
- **Windows std:** the windows-gnu build needs the Windows standard library. I believe the Fedora package is `rust-std-static-x86_64-pc-windows-gnu`, but I haven't confirmed the name.
- **Linker:** it also needs `mingw64-gcc` as the linker.
- **dlltool:** libloading 0.9 and windows-sys 0.61 both go through windows-link 0.2.1, which uses `kind = "raw-dylib"`. On the gnu target that may also require mingw's `dlltool`.

