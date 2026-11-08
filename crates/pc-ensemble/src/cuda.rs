//! The CUDA backend: one thread per member, compiled at run time with NVRTC.
//!
//! Nothing here links a GPU library. cudarc opens the driver (`nvcuda.dll`, `libcuda.so`)
//! and NVRTC when the first call needs them, and it panics when one is missing. So the
//! backend probes for both libraries first and reports a missing one as
//! [`BackendState::Unavailable`], and every call into cudarc runs inside [`guard`], which
//! turns a panic into an error (rule 7).
//!
//! NVRTC is found in this order: whatever the process already resolves by its bare name
//! (inside MATLAB that is MATLAB's own copy), then a CUDA toolkit named by `CUDA_PATH` or
//! `CUDA_HOME` or installed under `/usr/local/cuda`. The toolkit copy is loaded by absolute
//! path, because a bare-name load from inside MATLAB would pick MATLAB's copy again.
//!
//! The kernel is compiled without fused multiply-add (`--fmad=false`), so a run reproduces
//! the CPU backend operation for operation. The compile route follows the spike S4 result:
//! PTX when the driver is at least as new as NVRTC, a cubin for the exact architecture
//! when the driver is a 13.x release that is older.
//!
//! A run is cut into launches that each advance every member by at most K step attempts.
//! K starts small and follows the measured time of the last launch so that one launch takes
//! about 100 ms, well under the 2 s Windows display watchdog.

use std::ffi::{CString, c_char};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaFunction, DeviceRepr, DriverError, LaunchConfig, PushKernelArg, result,
    sys as dsys,
};
use cudarc::nvrtc::{self, CompileError, CompileOptions, Ptx, compile_ptx_with_opts, sys as nsys};
use pc_codegen::kernel::{Dialect, Scalar};

use crate::abi::{Packed, integrator_of, pack};
use crate::device::{Backend, BackendState, DeviceInfo};
use crate::gpu::{drive, guard, kernel_err};
use crate::job::{EnsembleError, EnsembleResult, Job, Label, Precision, validate};
use crate::kept::{Kept, Lru};
use crate::kernel::{Integrator, kernel_source};
use crate::real::Real;
use crate::tape::Tape;

/// The name of the kernel entry point.
const ENTRY: &str = "pc_ens";
/// Threads per block. Small enough for a register-heavy double precision kernel.
const BLOCK: u32 = 64;
/// `CUDA_ERROR_LAUNCH_TIMEOUT`: the driver stopped a launch that ran too long.
const LAUNCH_TIMEOUT: u32 = 702;

/// Formats a driver error as "step: CUDA_ERROR_NAME".
fn drv_text(step: &str, e: &DriverError) -> String {
    format!("{step}: {:?}", e.0)
}

/// A driver error of a copy, a launch or a synchronize. A watchdog stop is its own error.
fn drv(step: &'static str) -> impl Fn(DriverError) -> EnsembleError {
    move |e| {
        let text = drv_text(step, &e);
        if e.0 as u32 == LAUNCH_TIMEOUT {
            EnsembleError::GpuTimeout(text)
        } else {
            EnsembleError::GpuLaunch(text)
        }
    }
}

// ---------------------------------------------------------------------------------------
// Finding the libraries

/// True when both libraries load by their bare names.
fn libraries() -> (bool, bool) {
    // SAFETY: these only try to open the libraries and close them again.
    guard(|| unsafe { (dsys::is_culib_present(), nsys::is_culib_present()) })
        .unwrap_or((false, false))
}

fn is_nvrtc_file(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("nvrtc64_") && name.ends_with(".dll") && !name.contains(".alt")
    } else {
        name.starts_with("libnvrtc.so.")
    }
}

fn is_builtins_file(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("nvrtc-builtins64_") && name.ends_with(".dll")
    } else {
        name.starts_with("libnvrtc-builtins.so.")
    }
}

/// Folders a CUDA toolkit keeps NVRTC in.
fn toolkit_dirs() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = ["CUDA_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .collect();
    if !cfg!(windows) {
        roots.push(PathBuf::from("/usr/local/cuda"));
    }
    let mut dirs = Vec::new();
    for r in roots {
        for sub in [
            "bin/x64",
            "bin",
            "lib64",
            "lib/x64",
            "targets/x86_64-linux/lib",
        ] {
            dirs.push(r.join(sub));
        }
    }
    dirs
}

/// The files in `dir` that pass `want`, the highest name first.
fn files_in(dir: &Path, want: fn(&str) -> bool) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = read
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| want(n))
        .collect();
    names.sort();
    names.reverse();
    names.into_iter().map(|n| dir.join(n)).collect()
}

/// Loads a toolkit NVRTC by absolute path and keeps it loaded. Returns where it came from.
fn preload_toolkit_nvrtc() -> Option<PathBuf> {
    for dir in toolkit_dirs() {
        let Some(nvrtc) = files_in(&dir, is_nvrtc_file).into_iter().next() else {
            continue;
        };
        // The builtins library goes first, so NVRTC finds the copy of its own release.
        for b in files_in(&dir, is_builtins_file).into_iter().take(1) {
            // SAFETY: loading a library runs its initializers, as any dynamic load does.
            if let Ok(lib) = unsafe { libloading::Library::new(&b) } {
                std::mem::forget(lib);
            }
        }
        // SAFETY: as above.
        if let Ok(lib) = unsafe { libloading::Library::new(&nvrtc) } {
            std::mem::forget(lib);
            return Some(nvrtc);
        }
    }
    None
}

/// Makes sure NVRTC can be called, by its bare name or else from a toolkit. The answer is
/// kept, so the toolkit is searched once.
fn ensure_nvrtc() -> Result<(), String> {
    static NVRTC: OnceLock<Result<(), String>> = OnceLock::new();
    NVRTC
        .get_or_init(|| {
            if libraries().1 || (preload_toolkit_nvrtc().is_some() && libraries().1) {
                Ok(())
            } else {
                Err(
                    "NVRTC was not found: it comes with MATLAB or with a CUDA toolkit \
                     (set CUDA_PATH to the toolkit folder)"
                        .to_owned(),
                )
            }
        })
        .clone()
}

// ---------------------------------------------------------------------------------------
// The probe

struct Probe {
    state: BackendState,
    devices: Vec<DeviceInfo>,
    /// `cuDriverGetVersion`, for example 13040.
    driver: i32,
    nvrtc: (i32, i32),
}

impl Probe {
    fn unavailable(why: impl Into<String>) -> Probe {
        Probe {
            state: BackendState::Unavailable(why.into()),
            devices: Vec::new(),
            driver: 0,
            nvrtc: (0, 0),
        }
    }
}

fn probe() -> &'static Probe {
    static PROBE: OnceLock<Probe> = OnceLock::new();
    PROBE.get_or_init(|| {
        guard(probe_now)
            .unwrap_or_else(|msg| Probe::unavailable(format!("the probe panicked: {msg}")))
    })
}

fn version_text(driver: i32, nvrtc: (i32, i32)) -> String {
    format!(
        "CUDA driver {}.{}, NVRTC {}.{}",
        driver / 1000,
        driver % 1000 / 10,
        nvrtc.0,
        nvrtc.1
    )
}

fn probe_now() -> Probe {
    if !libraries().0 {
        return Probe::unavailable(
            "the CUDA driver library was not found (nvcuda.dll on Windows, libcuda.so on Linux)",
        );
    }
    if let Err(why) = ensure_nvrtc() {
        return Probe::unavailable(why);
    }
    let mut driver = 0;
    // SAFETY: the pointer is valid for one write. This works before cuInit.
    if let Err(e) = unsafe { dsys::cuDriverGetVersion(&mut driver) }.result() {
        return Probe::unavailable(drv_text("cuDriverGetVersion", &e));
    }
    let (mut major, mut minor) = (0, 0);
    // SAFETY: both pointers are valid for one write.
    if let Err(e) = unsafe { nsys::nvrtcVersion(&mut major, &mut minor) }.result() {
        return Probe::unavailable(format!("nvrtcVersion: {:?}", e.0));
    }
    if let Err(e) = result::init() {
        return Probe::unavailable(drv_text("cuInit", &e));
    }
    let count = match result::device::get_count() {
        Ok(c) => c.max(0),
        Err(e) => return Probe::unavailable(drv_text("cuDeviceGetCount", &e)),
    };
    let text = version_text(driver, (major, minor));
    let mut devices = Vec::new();
    for index in 0..count {
        let listed = result::device::get(index).and_then(|dev| {
            let name = result::device::get_name(dev)?;
            // SAFETY: `dev` came from `get` just now.
            let memory = unsafe { result::device::total_mem(dev) }?;
            Ok((name, memory))
        });
        match listed {
            Ok((name, memory)) => devices.push(DeviceInfo {
                backend: Backend::Cuda,
                index: index as usize,
                name,
                driver: text.clone(),
                fp32: true,
                // Every device NVRTC 13 compiles for runs double precision.
                fp64: true,
                memory_bytes: Some(memory as u64),
            }),
            Err(e) => return Probe::unavailable(drv_text("device query", &e)),
        }
    }
    if devices.is_empty() {
        return Probe::unavailable("the CUDA driver lists no device");
    }
    Probe {
        state: BackendState::Available,
        devices,
        driver,
        nvrtc: (major, minor),
    }
}

/// Whether the CUDA backend can run in this process.
pub fn backend_state() -> BackendState {
    probe().state.clone()
}

/// The CUDA devices, empty when the backend is not available.
pub fn devices() -> Vec<DeviceInfo> {
    probe().devices.clone()
}

// ---------------------------------------------------------------------------------------
// Compiling

/// How the kernel reaches the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// PTX for the virtual architecture. It loads only on a driver at least as new as NVRTC.
    Ptx,
    /// A cubin for the exact architecture. It loads on any 13.x driver.
    Cubin,
}

/// The route for a driver and an NVRTC release, or `None` when neither loads.
pub fn select(driver: i32, nvrtc_major: i32, nvrtc_minor: i32) -> Option<Route> {
    if driver >= nvrtc_major * 1000 + nvrtc_minor * 10 {
        Some(Route::Ptx)
    } else if driver >= 13000 {
        Some(Route::Cubin)
    } else {
        None
    }
}

fn compile_error(e: CompileError) -> String {
    match e {
        CompileError::CompileError { nvrtc, log, .. } => {
            format!("{:?}: {}", nvrtc.0, log.to_string_lossy().trim())
        }
        CompileError::CreationError(e) => format!("create_program: {:?}", e.0),
        CompileError::GetLogError(e) => format!("get_program_log: {:?}", e.0),
        CompileError::GetPtxError(e) => format!("get_ptx: {:?}", e.0),
        CompileError::DestroyError(e) => format!("destroy_program: {:?}", e.0),
    }
}

/// Text from a NUL-terminated buffer that NVRTC filled.
fn c_text(raw: &[c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn options(arch: &str) -> Vec<String> {
    vec![
        format!("--gpu-architecture={arch}"),
        "--fmad=false".to_owned(),
        "--std=c++17".to_owned(),
    ]
}

/// Compiles `src` to a cubin with `nvrtc::result`, since cudarc's safe API only returns PTX.
/// The program is destroyed on every way out, including a panic inside cudarc.
fn compile_cubin(src: &str, arch: &str) -> Result<Vec<u8>, String> {
    let src = CString::new(src).map_err(|e| format!("source: {e}"))?;
    let prog = nvrtc::result::create_program(&src, None)
        .map_err(|e| format!("create_program: {:?}", e.0))?;
    // SAFETY: `prog` stays live until `destroy_program` below, and `src` outlives it.
    let built = guard(|| unsafe { cubin_from(prog, arch) });
    // SAFETY: `prog` came from `create_program` and is destroyed only here.
    let destroyed = unsafe { nvrtc::result::destroy_program(prog) };
    let cubin = built.map_err(|msg| format!("panic: {msg}"))??;
    destroyed.map_err(|e| format!("destroy_program: {:?}", e.0))?;
    Ok(cubin)
}

/// # Safety
/// `prog` must come from `create_program` and not be destroyed yet.
unsafe fn cubin_from(prog: nsys::nvrtcProgram, arch: &str) -> Result<Vec<u8>, String> {
    let opts = options(arch);
    // SAFETY (all blocks): `prog` is live, as the caller promises.
    if let Err(e) = unsafe { nvrtc::result::compile_program(prog, &opts) } {
        let log = match unsafe { nvrtc::result::get_program_log(prog) } {
            Ok(raw) => c_text(&raw).trim().to_owned(),
            Err(l) => format!("no log ({:?})", l.0),
        };
        return Err(format!("{:?}: {log}", e.0));
    }
    let mut size = 0usize;
    unsafe { nsys::nvrtcGetCUBINSize(prog, &mut size) }
        .result()
        .map_err(|e| format!("nvrtcGetCUBINSize: {:?}", e.0))?;
    let mut cubin = vec![0u8; size];
    unsafe { nsys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| format!("nvrtcGetCUBIN: {:?}", e.0))?;
    Ok(cubin)
}

/// The kernel image for a device of compute capability `cc`.
fn compile_image(src: &str, p: &Probe, cc: (i32, i32)) -> Result<Ptx, EnsembleError> {
    let Some(route) = select(p.driver, p.nvrtc.0, p.nvrtc.1) else {
        return Err(EnsembleError::GpuUnavailable(format!(
            "{} is too old: NVRTC output needs a driver of release 13.0 or newer",
            version_text(p.driver, p.nvrtc)
        )));
    };
    let compiled = match route {
        Route::Ptx => {
            let opts = CompileOptions {
                options: options(&format!("compute_{}{}", cc.0, cc.1)),
                ..Default::default()
            };
            compile_ptx_with_opts(src, opts).map_err(compile_error)
        }
        Route::Cubin => compile_cubin(src, &format!("sm_{}{}", cc.0, cc.1)).map(Ptx::from_binary),
    };
    compiled.map_err(EnsembleError::GpuCompile)
}

/// Compiles the kernel for the virtual architecture `arch`, for example `compute_80`, and
/// returns the PTX text.
///
/// It needs NVRTC but no driver and no device, so a machine with only a CUDA toolkit can
/// check that the kernel compiles. A missing NVRTC is [`EnsembleError::GpuUnavailable`] and
/// a rejected kernel is [`EnsembleError::GpuCompile`] with the compiler log.
pub fn compile_ptx_text(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    arch: &str,
) -> Result<String, EnsembleError> {
    ensure_nvrtc().map_err(EnsembleError::GpuUnavailable)?;
    let src = kernel_source(tape, integrator, scalar, Dialect::Cuda).map_err(kernel_err)?;
    let opts = CompileOptions {
        options: options(arch),
        ..Default::default()
    };
    let ptx = guard(|| compile_ptx_with_opts(&src, opts))
        .map_err(|m| EnsembleError::GpuCompile(format!("panic in NVRTC: {m}")))?
        .map_err(|e| EnsembleError::GpuCompile(compile_error(e)))?;
    guard(|| ptx.to_src()).map_err(|m| EnsembleError::GpuCompile(format!("panic in NVRTC: {m}")))
}

// ---------------------------------------------------------------------------------------
// Running

/// Runs `job` on the CUDA device `device`.
///
/// A missing library, a missing device, a kernel NVRTC rejects, a failed copy or launch and a
/// watchdog stop each come back as their own [`EnsembleError`]. A panic inside cudarc comes
/// back as [`EnsembleError::GpuLaunch`]. The cancel flag is read between launches.
pub fn run_cuda(
    device: &DeviceInfo,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let checked = validate(job)?;
    let p = probe();
    if let BackendState::Unavailable(why) = &p.state {
        return Err(EnsembleError::GpuUnavailable(why.clone()));
    }
    if device.backend != Backend::Cuda || device.index >= p.devices.len() {
        return Err(EnsembleError::GpuUnavailable(format!(
            "no CUDA device {}: the driver lists {}",
            device.index,
            p.devices.len()
        )));
    }
    let tape = Tape::build(job.model)?;
    let scalar = match job.precision {
        Precision::Single => Scalar::F32,
        Precision::Double => Scalar::F64,
    };
    let src = kernel_source(&tape, integrator_of(job.method), scalar, Dialect::Cuda)
        .map_err(kernel_err)?;
    let ran = guard(|| {
        // The context comes from the table inside the guard, because `CudaContext::new`
        // panics when a library is missing and the last drop of a context can too.
        let entry = CONTEXTS.get_or_make(device.index, || {
            let ctx = CudaContext::new(device.index).map_err(drv("create context"))?;
            Ok(Arc::new(Entry {
                kernels: Lru::new(KERNELS_KEPT),
                ctx,
            }))
        })?;
        session(&entry, device, p, job, &checked, &tape, &src, cancel)
    })
    .unwrap_or_else(|msg| Err(EnsembleError::GpuLaunch(format!("panic in cudarc: {msg}"))));
    if matches!(
        ran,
        Err(EnsembleError::GpuLaunch(_) | EnsembleError::GpuTimeout(_))
    ) {
        // A driver error can stay in the context and fail every later call, so the next run
        // on this device starts from a new one.
        let gone = CONTEXTS.forget(device.index);
        let _ = guard(|| drop(gone));
    }
    ran
}

/// What the runs on one device share: its context and the kernels compiled for it.
struct Entry {
    // The kernels drop before the context. Each loaded function holds its module, and a module
    // holds the context.
    kernels: Lru<KernelKey, CudaFunction>,
    ctx: Arc<CudaContext>,
}

/// The kernel text, the compute capability and the precision of a compiled kernel.
type KernelKey = (String, (i32, i32), Precision);

/// How many compiled kernels a device keeps.
const KERNELS_KEPT: usize = 8;

/// The entry of each device, kept between runs because creating a context takes tens of
/// milliseconds on a laptop GPU and compiling a kernel about 11 ms.
static CONTEXTS: Kept<Entry> = Kept::new();

/// How many kernels have been compiled in this process, for the cache test.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// How many kernels the kept entry of device `index` holds, for the cache test.
#[doc(hidden)]
pub fn kept_kernels(index: usize) -> usize {
    CONTEXTS.get(index).map_or(0, |e| e.kernels.len())
}

/// How many kernels this process has compiled, for the cache test.
#[doc(hidden)]
pub fn kernel_compiles() -> usize {
    COMPILES.load(Ordering::SeqCst)
}

#[allow(clippy::too_many_arguments)]
fn session(
    entry: &Entry,
    device: &DeviceInfo,
    p: &Probe,
    job: &Job,
    checked: &crate::job::Checked,
    tape: &Tape,
    src: &str,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let ctx = &entry.ctx;
    let cc = ctx
        .compute_capability()
        .map_err(drv("compute capability"))?;
    let function = entry
        .kernels
        .get_or_make((src.to_owned(), cc, job.precision), || {
            COMPILES.fetch_add(1, Ordering::SeqCst);
            let image = compile_image(src, p, cc)?;
            let load = |step: &'static str| {
                move |e: DriverError| EnsembleError::GpuCompile(drv_text(step, &e))
            };
            let module = ctx.load_module(image).map_err(load("load_module"))?;
            module.load_function(ENTRY).map_err(load("load_function"))
        })?;
    let label = Label {
        backend: Backend::Cuda.name().into(),
        device: device.name.clone(),
        driver: device.driver.clone(),
        precision: job.precision,
    };
    let ran = match job.precision {
        Precision::Single => launch_all(ctx, &function, pack::<f32>(job, checked, tape), cancel)
            .and_then(|packed| packed.finish(label)),
        Precision::Double => launch_all(ctx, &function, pack::<f64>(job, checked, tape), cancel)
            .and_then(|packed| packed.finish(label)),
    };
    // cudarc's Drop code stores its errors in the context. Take them here and charge them to
    // this run, or the next run on the device would see them.
    let leftover = ctx.check_err();
    match (ran, leftover) {
        (Ok(r), Ok(())) => Ok(r),
        (Ok(_), Err(e)) => Err(EnsembleError::GpuLaunch(drv_text("cleanup", &e))),
        (Err(e), _) => Err(e),
    }
}

/// Uploads the buffers, launches until no member is running and downloads the result.
fn launch_all<T: Real + DeviceRepr>(
    ctx: &Arc<CudaContext>,
    function: &CudaFunction,
    mut packed: Packed<T>,
    cancel: &AtomicBool,
) -> Result<Packed<T>, EnsembleError> {
    let stream = ctx.default_stream();
    let rc = stream
        .clone_htod(packed.rc.as_slice())
        .map_err(drv("copy rc to device"))?;
    let segm = stream
        .clone_htod(packed.segm.as_slice())
        .map_err(drv("copy segm to device"))?;
    let x0 = stream
        .clone_htod(packed.x0.as_slice())
        .map_err(drv("copy x0 to device"))?;
    let par = stream
        .clone_htod(packed.par.as_slice())
        .map_err(drv("copy par to device"))?;
    let mut work = stream
        .clone_htod(packed.work.as_slice())
        .map_err(drv("copy work to device"))?;
    let mut out = stream
        .clone_htod(packed.out.as_slice())
        .map_err(drv("copy out to device"))?;
    let mut ctl_d = stream
        .clone_htod(packed.ctl.as_slice())
        .map_err(drv("copy ctl to device"))?;

    let blocks = u32::try_from(packed.members.div_ceil(BLOCK as usize)).map_err(|_| {
        EnsembleError::BadJob(format!(
            "{} members need more blocks than a launch can hold",
            packed.members
        ))
    })?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let members = packed.members as u64;
    let ns = packed.ns as u32;
    let pstride = packed.pstride as u64;
    let max_steps = packed.max_steps;

    drive(&mut packed.ctl, members as usize, cancel, |steps| {
        let started = Instant::now();
        {
            let mut b = stream.launch_builder(function);
            b.arg(&rc)
                .arg(&segm)
                .arg(&x0)
                .arg(&par)
                .arg(&mut work)
                .arg(&mut out)
                .arg(&mut ctl_d)
                .arg(&members)
                .arg(&steps)
                .arg(&max_steps)
                .arg(&ns)
                .arg(&pstride);
            // SAFETY: the arguments follow the signature of `pc_ens` in order and type, the
            // buffers have the sizes `crate::abi` documents, and the kernel returns early
            // for a thread past the last member.
            unsafe { b.launch(cfg) }.map_err(drv("launch"))?;
        }
        stream.synchronize().map_err(drv("synchronize"))?;
        let took = started.elapsed();
        let ctl = stream.clone_dtoh(&ctl_d).map_err(drv("copy ctl to host"))?;
        stream
            .synchronize()
            .map_err(drv("synchronize after copy"))?;
        Ok((took, ctl))
    })?;
    packed.out = stream.clone_dtoh(&out).map_err(drv("copy out to host"))?;
    packed.work = stream.clone_dtoh(&work).map_err(drv("copy work to host"))?;
    stream
        .synchronize()
        .map_err(drv("synchronize after copy"))?;
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_codegen::kernel::KernelError;

    #[test]
    fn the_route_follows_the_driver_and_nvrtc_release() {
        assert_eq!(select(13040, 13, 4), Some(Route::Ptx));
        assert_eq!(select(13020, 13, 4), Some(Route::Cubin));
        assert_eq!(select(13000, 13, 1), Some(Route::Cubin));
        assert_eq!(select(12080, 13, 4), None);
        assert_eq!(select(13010, 13, 1), Some(Route::Ptx));
    }

    #[test]
    fn library_files_are_told_apart() {
        if cfg!(windows) {
            assert!(is_nvrtc_file("nvrtc64_130_0.dll"));
            assert!(!is_nvrtc_file("nvrtc64_130_0.alt.dll"));
            assert!(!is_nvrtc_file("nvrtc-builtins64_134.dll"));
            assert!(is_builtins_file("nvrtc-builtins64_134.dll"));
        } else {
            assert!(is_nvrtc_file("libnvrtc.so.13"));
            assert!(is_nvrtc_file("libnvrtc.so.13.4.0"));
            assert!(!is_nvrtc_file("libnvrtc.so"));
            assert!(!is_nvrtc_file("libnvrtc-builtins.so.13.4"));
            assert!(is_builtins_file("libnvrtc-builtins.so.13.4"));
        }
    }

    #[test]
    fn a_text_buffer_ends_at_the_nul() {
        let raw: Vec<c_char> = b"error: x\0garbage".iter().map(|&b| b as c_char).collect();
        assert_eq!(c_text(&raw), "error: x");
        assert_eq!(c_text(&[]), "");
    }

    #[test]
    fn a_kernel_error_is_an_unsupported_model() {
        let e = kernel_err(KernelError::Unsupported("erf".into()));
        assert_eq!(e, EnsembleError::Unsupported("erf".into()));
        assert!(matches!(
            kernel_err(KernelError::NoDouble),
            EnsembleError::Unsupported(_)
        ));
    }

    #[test]
    fn the_probe_never_panics_and_agrees_with_itself() {
        // This runs on machines with and without a driver. It checks the shape of the answer.
        let p = probe();
        match &p.state {
            BackendState::Available => assert!(!p.devices.is_empty()),
            BackendState::Unavailable(why) => {
                assert!(!why.is_empty());
                assert!(p.devices.is_empty());
            }
            BackendState::NotBuilt => panic!("the CUDA backend is built into this binary"),
        }
    }
}
