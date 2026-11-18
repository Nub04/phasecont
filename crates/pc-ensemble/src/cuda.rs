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
//! On Windows the first loads of the driver and NVRTC skip the current folder (see
//! [`crate::winload`]): the driver must be `nvcuda.dll` in the system folder, and the NVRTC
//! builtins library is loaded by absolute path from the folder of the NVRTC in use, so that
//! NVRTC's own load of it at the first compile gets that copy.
//!
//! In the default exact tier the kernel is compiled without fused multiply-add
//! (`--fmad=false`), so a run reproduces the CPU backend operation for operation. The opt-in
//! fma tier ([`crate::Math::Fma`], double precision only) drops that option, and its results
//! are not bit-reproducible; the tier is part of the kernel cache key. The compile route follows the spike S4 result:
//! PTX when the driver is at least as new as NVRTC, a cubin for the exact architecture
//! when the driver is a 13.x release that is older.
//!
//! A run is cut into launches that each advance every member by at most K step attempts.
//! K starts small and follows the measured time of the last launch so that one launch takes
//! about 100 ms, well under the 2 s Windows display watchdog.

use std::ffi::{CString, c_char};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaFunction, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
    ValidAsZeroBits, result, sys as dsys,
};
use cudarc::nvrtc::{self, CompileError, CompileOptions, Ptx, compile_ptx_with_opts, sys as nsys};
use pc_codegen::kernel::{Dialect, Scalar};

use crate::abi::{Packed, integrator_of, pack_in};
use crate::batch::{members_per_batch, run_batched};
use crate::device::{Backend, BackendState, DeviceInfo};
use crate::gpu::{GLOBAL_SCRATCH_FROM, drive, guard, kernel_err};
use crate::job::{EnsembleError, EnsembleResult, Job, Label, Math, Precision, validate};
use crate::kept::{Kept, Lru};
use crate::kernel::{Integrator, Layout, kernel_source_in, private_reals};
use crate::real::Real;
use crate::tape::Tape;
use crate::winload::{self, NoCwdSearch};

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

/// The driver errors that NVIDIA documents as leaving the process unable to use CUDA: an
/// illegal address (700), a launch timeout (702), a device assert (710), a hardware stack
/// error, an illegal instruction, a misaligned address, an invalid address space or an
/// invalid program counter (714 to 718) and a launch failure (719). On the RTX 5060 test
/// machine, a new context made after error 700 failed with 700 as well
/// (`tests/sticky.rs`), so dropping the context does not clear these errors.
const STICKY: [u32; 9] = [700, 702, 710, 714, 715, 716, 717, 718, 719];

/// Added to the text of a sticky error.
const STICKY_NOTE: &str = "; the driver can keep this error until the process ends, so if later \
GPU runs fail the same way, restart the process (in MATLAB, restart MATLAB)";

/// A driver error of a copy, a launch or a synchronize. A watchdog stop is its own error.
fn drv(step: &'static str) -> impl Fn(DriverError) -> EnsembleError {
    move |e| {
        let code = e.0 as u32;
        let mut text = drv_text(step, &e);
        if STICKY.contains(&code) {
            text.push_str(STICKY_NOTE);
        }
        if code == LAUNCH_TIMEOUT {
            EnsembleError::GpuTimeout(text)
        } else {
            EnsembleError::GpuLaunch(text)
        }
    }
}

/// The private reals of one Dormand-Prince thread from which the global layout is faster, on
/// both precisions. Measured on the RTX 5060 with 2048 members of Lorenz-96 (`cuda_layout_time`,
/// SPEC 13): the private layout wins up to 3200 reals (1.1x with doubles, 1.2x with singles) and
/// the global layout wins from 3350 (1.75x and 1.3x), and 3300 sits between them. RK4 has no
/// such point: its global layout is slower until the memory rule below applies.
const DP5_GLOBAL_FROM_REALS: usize = 3300;

/// The share of the free memory that the driver's local-memory reserve of the private layout
/// may take: 3/4. The driver reserves local memory for every resident thread, whatever the
/// member count, and a reserve that nears the free memory makes the private layout 2 to 3 times
/// slower than the global one. The change came at a reserve of 0.85 to 1.0 times the free
/// memory with about 7.5, 5.0 and 2.8 GB free, and the private layout was still 1.4 times
/// faster at 0.75. A reserve above the point costs far more than one below it, so the share
/// keeps a margin under the measured point.
const PRIVATE_RESERVE_SHARE: (u128, u128) = (3, 4);

/// The part of the free device memory that the buffers of one batch may take: 60%. The rest
/// is for the driver (local memory of the launch, the context) and for other processes that
/// allocate while this one runs. The fraction is a margin, not a tuned value.
fn budget_of(free: u64) -> u64 {
    free / 10 * 6
}

/// What a device gives the layout rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capacity {
    /// The threads the device keeps resident: multiprocessors times the most threads of one.
    pub resident_threads: u64,
    /// The free device memory of a context that holds no local memory yet: what is free now
    /// plus `held_bytes`. It does not change when a private run reserves its local memory, so
    /// the layout of one job does not flip between calls.
    pub free_bytes: u64,
    /// The local-memory reserve that this context already holds from earlier private runs.
    /// The driver keeps it until the context is destroyed.
    pub held_bytes: u64,
}

impl Capacity {
    /// The capacity from the free memory now and the reserve the context holds.
    pub fn new(resident_threads: u64, free_now: u64, held_bytes: u64) -> Capacity {
        Capacity {
            resident_threads,
            free_bytes: free_now.saturating_add(held_bytes),
            held_bytes,
        }
    }
}

/// The local-memory reserve of a private run: `bytes` of private arrays for every resident
/// thread.
fn reserve_of(bytes: usize, capacity: &Capacity) -> u64 {
    (bytes as u64).saturating_mul(capacity.resident_threads)
}

/// The scratch layout for a model of `n` states. The private layout needs all of these:
/// - below [`GLOBAL_SCRATCH_FROM`] bytes of private arrays per thread,
/// - below [`DP5_GLOBAL_FROM_REALS`] reals per thread with Dormand-Prince,
/// - a local-memory reserve within [`PRIVATE_RESERVE_SHARE`] of the free memory.
///
/// Otherwise the scratch lives in global memory.
pub fn layout_for(
    n: usize,
    integrator: Integrator,
    real_bytes: usize,
    capacity: Capacity,
) -> Layout {
    pick(None, n, integrator, real_bytes, capacity)
}

/// [`layout_for`], or `forced` when it is set.
fn pick(
    forced: Option<Layout>,
    n: usize,
    integrator: Integrator,
    real_bytes: usize,
    capacity: Capacity,
) -> Layout {
    if let Some(layout) = forced {
        return layout;
    }
    let reals = private_reals(n, integrator);
    let bytes = reals.saturating_mul(real_bytes);
    let (num, den) = PRIVATE_RESERVE_SHARE;
    let fits =
        u128::from(reserve_of(bytes, &capacity)) * den <= u128::from(capacity.free_bytes) * num;
    let dp5_large = integrator == Integrator::Dp5 && reals >= DP5_GLOBAL_FROM_REALS;
    if bytes >= GLOBAL_SCRATCH_FROM || dp5_large || !fits {
        Layout::Global
    } else {
        Layout::Private
    }
}

/// The device memory left for the buffers of the batches under `layout`: the free memory less
/// the local-memory reserve. A context that already holds a reserve at least as large has that
/// part out of the free memory already.
fn buffer_room(layout: Layout, private_bytes: usize, capacity: &Capacity) -> u64 {
    let reserve = match layout {
        Layout::Private => reserve_of(private_bytes, capacity),
        Layout::Global => 0,
    };
    capacity
        .free_bytes
        .saturating_sub(reserve.max(capacity.held_bytes))
}

/// The layout a test or a bench forces for every run in this process: 0 for none, 1 for
/// private, 2 for global.
static FORCED: AtomicU8 = AtomicU8::new(0);

/// Forces the scratch layout of every CUDA run in this process, or lifts the force with
/// `None`. The A/B bench and the device test use it to run both layouts on one model. It is
/// not part of the API.
#[doc(hidden)]
pub fn force_layout(layout: Option<Layout>) {
    let code = match layout {
        None => 0,
        Some(Layout::Private) => 1,
        Some(Layout::Global) => 2,
    };
    FORCED.store(code, Ordering::SeqCst);
}

fn forced_layout() -> Option<Layout> {
    match FORCED.load(Ordering::SeqCst) {
        1 => Some(Layout::Private),
        2 => Some(Layout::Global),
        _ => None,
    }
}

/// What to add to an allocation error when the work buffer is large because the scratch of
/// the steps lives in it: `members` slices of `work_len` values. `members` is the size of
/// the batch that failed, since the run already splits the job to fit the free memory.
fn global_note(
    layout: Layout,
    members: usize,
    work_len: usize,
    real_bytes: usize,
) -> Option<String> {
    if layout == Layout::Private {
        return None;
    }
    let bytes = members.saturating_mul(work_len).saturating_mul(real_bytes);
    Some(format!(
        "the step arrays of this kernel live in global memory, which takes {} KB for {members} \
members of this batch; use the cpu backend or a smaller model",
        bytes.div_ceil(1024)
    ))
}

/// The mapper for a driver call: [`drv`], plus `note` after the text of an error that the
/// size of the buffers can explain. A sticky error comes from an earlier fault and a timeout
/// from the watchdog, so neither gets it.
fn step_err(step: &'static str, note: Option<String>) -> impl Fn(DriverError) -> EnsembleError {
    let base = drv(step);
    move |e| {
        let sticky = STICKY.contains(&(e.0 as u32));
        match (base(e), &note) {
            (EnsembleError::GpuLaunch(m), Some(n)) if !sticky => {
                EnsembleError::GpuLaunch(format!("{m}; {n}"))
            }
            (other, _) => other,
        }
    }
}

// ---------------------------------------------------------------------------------------
// Finding the libraries

/// The `cuDriverGetVersion` of the driver, for example 13040. The first call loads the
/// driver library inside a [`NoCwdSearch`] window and keeps it; later calls reuse it. On
/// Windows the driver must be `nvcuda.dll` in the system folder, which is loaded first so
/// that cudarc's bare-name search finds that copy.
fn driver_version() -> Result<i32, String> {
    static DRIVER: OnceLock<Result<i32, String>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            let missing = || {
                "the CUDA driver library was not found (nvcuda.dll in the Windows system \
                 folder, libcuda.so on Linux)"
                    .to_owned()
            };
            let _window = NoCwdSearch::new();
            if cfg!(windows) && !winload::system_library("nvcuda.dll") {
                return Err(missing());
            }
            // SAFETY: `is_culib_present` only opens and closes the library. The version
            // call loads cudarc's copy for good, inside the window, and works before cuInit.
            guard(|| unsafe {
                if !dsys::is_culib_present() {
                    return Err(missing());
                }
                let mut driver = 0;
                dsys::cuDriverGetVersion(&mut driver)
                    .result()
                    .map(|()| driver)
                    .map_err(|e| drv_text("cuDriverGetVersion", &e))
            })
            .unwrap_or_else(|m| Err(format!("loading the CUDA driver panicked: {m}")))
        })
        .clone()
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
    dirs_under(roots)
}

/// The folders under each toolkit root that can hold NVRTC.
fn dirs_under(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for r in roots {
        for sub in [
            "bin/x64",
            "bin",
            "lib64",
            "lib/x64",
            "targets/x86_64-linux/lib",
            "targets/sbsa-linux/lib",
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
        if let Some(b) = files_in(&dir, is_builtins_file).into_iter().next() {
            winload::pinned(&b);
        }
        if winload::pinned(&nvrtc) {
            return Some(nvrtc);
        }
    }
    None
}

/// Loads the NVRTC builtins library from the folder of the NVRTC that cudarc loaded, so that
/// NVRTC's own bare-name load at its first compile, which happens outside any
/// [`NoCwdSearch`] window, gets this copy. The one named for the NVRTC release goes first.
fn preload_builtins_next_to_nvrtc(major: i32, minor: i32) {
    if !cfg!(windows) {
        return;
    }
    // SAFETY: the library is cudarc's, loaded for good by the version call before this.
    let addr = guard(|| unsafe {
        nsys::culib()
            .get::<unsafe extern "C" fn()>(b"nvrtcVersion\0")
            .map(|f| *f as *const std::ffi::c_void)
            .ok()
    })
    .ok()
    .flatten();
    let Some(dir) = addr.and_then(winload::module_dir) else {
        return;
    };
    let exact = dir.join(format!("nvrtc-builtins64_{major}{minor}.dll"));
    if exact.is_file() && winload::pinned(&exact) {
        return;
    }
    if let Some(b) = files_in(&dir, is_builtins_file).into_iter().next() {
        winload::pinned(&b);
    }
}

/// The `nvrtcVersion` of the NVRTC in use. The first call loads it inside a
/// [`NoCwdSearch`] window, by its bare name or else from a toolkit, and keeps it; the toolkit
/// is searched once.
fn nvrtc_version() -> Result<(i32, i32), String> {
    static NVRTC: OnceLock<Result<(i32, i32), String>> = OnceLock::new();
    NVRTC
        .get_or_init(|| {
            let _window = NoCwdSearch::new();
            // SAFETY: these only open the library and close it again.
            let present = || guard(|| unsafe { nsys::is_culib_present() }).unwrap_or(false);
            if !present() && (preload_toolkit_nvrtc().is_none() || !present()) {
                return Err(
                    "NVRTC was not found: it comes with MATLAB or with a CUDA toolkit \
                     (set CUDA_PATH to the toolkit folder)"
                        .to_owned(),
                );
            }
            // SAFETY: both pointers are valid for one write. The call loads cudarc's copy
            // for good, inside the window.
            let version = guard(|| unsafe {
                let (mut major, mut minor) = (0, 0);
                nsys::nvrtcVersion(&mut major, &mut minor)
                    .result()
                    .map(|()| (major, minor))
                    .map_err(|e| format!("nvrtcVersion: {:?}", e.0))
            })
            .unwrap_or_else(|m| Err(format!("loading NVRTC panicked: {m}")))?;
            preload_builtins_next_to_nvrtc(version.0, version.1);
            Ok(version)
        })
        .clone()
}

/// Makes sure NVRTC can be called.
fn ensure_nvrtc() -> Result<(), String> {
    nvrtc_version().map(|_| ())
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
    let driver = match driver_version() {
        Ok(d) => d,
        Err(why) => return Probe::unavailable(why),
    };
    let (major, minor) = match nvrtc_version() {
        Ok(v) => v,
        Err(why) => return Probe::unavailable(why),
    };
    // cuInit loads the driver's own helper libraries, so it runs inside a window too.
    let _window = NoCwdSearch::new();
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
        .map(|&c| u8::from_ne_bytes(c.to_ne_bytes()))
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The NVRTC options for `arch`. The exact tier turns off fused multiply-add, so a run gives
/// the same bits as the CPU; the fma tier leaves NVRTC free to fuse.
fn options(arch: &str, math: Math) -> Vec<String> {
    let fmad = match math {
        Math::Exact => "--fmad=false",
        Math::Fma => "--fmad=true",
    };
    vec![
        format!("--gpu-architecture={arch}"),
        fmad.to_owned(),
        "--std=c++17".to_owned(),
    ]
}

/// Compiles `src` to a cubin with `nvrtc::result`, since cudarc's safe API only returns PTX.
/// The program is destroyed on every way out, including a panic inside cudarc.
fn compile_cubin(src: &str, arch: &str, math: Math) -> Result<Vec<u8>, String> {
    let src = CString::new(src).map_err(|e| format!("source: {e}"))?;
    let prog = nvrtc::result::create_program(&src, None)
        .map_err(|e| format!("create_program: {:?}", e.0))?;
    // SAFETY: `prog` stays live until `destroy_program` below, and `src` outlives it.
    let built = guard(|| unsafe { cubin_from(prog, arch, math) });
    // SAFETY: `prog` came from `create_program` and is destroyed only here.
    let destroyed = unsafe { nvrtc::result::destroy_program(prog) };
    let cubin = built.map_err(|msg| format!("panic: {msg}"))??;
    destroyed.map_err(|e| format!("destroy_program: {:?}", e.0))?;
    Ok(cubin)
}

/// # Safety
/// `prog` must come from `create_program` and not be destroyed yet.
unsafe fn cubin_from(prog: nsys::nvrtcProgram, arch: &str, math: Math) -> Result<Vec<u8>, String> {
    let opts = options(arch, math);
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

/// The kernel image for a device of compute capability `cc`, in the math tier `math`.
fn compile_image(src: &str, p: &Probe, cc: (i32, i32), math: Math) -> Result<Ptx, EnsembleError> {
    let Some(route) = select(p.driver, p.nvrtc.0, p.nvrtc.1) else {
        return Err(EnsembleError::GpuUnavailable(format!(
            "{} is too old: NVRTC output needs a driver of release 13.0 or newer",
            version_text(p.driver, p.nvrtc)
        )));
    };
    let compiled = match route {
        Route::Ptx => {
            let opts = CompileOptions {
                options: options(&format!("compute_{}{}", cc.0, cc.1), math),
                ..Default::default()
            };
            compile_ptx_with_opts(src, opts).map_err(compile_error)
        }
        Route::Cubin => {
            compile_cubin(src, &format!("sm_{}{}", cc.0, cc.1), math).map(Ptx::from_binary)
        }
    };
    compiled.map_err(EnsembleError::GpuCompile)
}

/// Compiles the kernel for the virtual architecture `arch`, for example `compute_80`, and
/// returns the PTX text. The scratch of a step is private to the thread.
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
    compile_ptx_text_in(tape, integrator, scalar, Layout::Private, arch, Math::Exact)
}

/// [`compile_ptx_text`] with the scratch of a step in the given layout and the math tier `math`.
pub fn compile_ptx_text_in(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    layout: Layout,
    arch: &str,
    math: Math,
) -> Result<String, EnsembleError> {
    ensure_nvrtc().map_err(EnsembleError::GpuUnavailable)?;
    let src =
        kernel_source_in(tape, integrator, scalar, Dialect::Cuda, layout).map_err(kernel_err)?;
    let opts = CompileOptions {
        options: options(arch, math),
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
    let integrator = integrator_of(job.method);
    let real_bytes = match job.precision {
        Precision::Single => size_of::<f32>(),
        Precision::Double => size_of::<f64>(),
    };
    // The entry this run used, so that a failure removes that one and not a newer one.
    let mut used: Option<Arc<Entry>> = None;
    let ran = guard(|| {
        // The context comes from the table inside the guard, because `CudaContext::new`
        // panics when a library is missing and the last drop of a context can too.
        let entry = CONTEXTS.get_or_make(device.index, || {
            let ctx = CudaContext::new(device.index).map_err(drv("create context"))?;
            Ok(Arc::new(Entry {
                kernels: Lru::new(KERNELS_KEPT),
                ctx,
                held: AtomicU64::new(0),
            }))
        })?;
        let entry = used.insert(entry);
        // The layout depends on the device (its resident threads and its free memory), so it
        // is picked here, where the context exists, and it decides the kernel text.
        let capacity = capacity_of(entry)?;
        let layout = pick(forced_layout(), tape.n, integrator, real_bytes, capacity);
        let src = kernel_source_in(&tape, integrator, scalar, Dialect::Cuda, layout)
            .map_err(kernel_err)?;
        let run = Run {
            layout,
            capacity,
            src: &src,
        };
        session(entry, device, p, job, &checked, &tape, &run, cancel)
    })
    .unwrap_or_else(|msg| Err(EnsembleError::GpuLaunch(format!("panic in cudarc: {msg}"))));
    if matches!(
        ran,
        Err(EnsembleError::GpuLaunch(_) | EnsembleError::GpuTimeout(_))
    ) && let Some(seen) = &used
    {
        // A driver error can stay in the context and fail every later call, so the next run
        // on this device starts from a new one. Only the entry this run used is removed.
        let gone = CONTEXTS.forget_if(device.index, seen);
        let _ = guard(|| drop(gone));
    }
    // The last drop of an entry can panic, so this one is dropped where a panic is caught.
    let _ = guard(|| drop(used));
    ran
}

/// What the runs on one device share: its context and the kernels compiled for it.
struct Entry {
    // The kernels drop before the context. Each loaded function holds its module, and a module
    // holds the context.
    kernels: Lru<KernelKey, CudaFunction>,
    ctx: Arc<CudaContext>,
    /// The largest local-memory reserve that a private run asked of this context. The driver
    /// keeps the memory until the context is destroyed, so the free memory read later is lower
    /// by this much and the layout rule adds it back.
    held: AtomicU64,
}

/// The layout of one run, the capacity it was picked from and the kernel text it gives.
struct Run<'a> {
    layout: Layout,
    capacity: Capacity,
    src: &'a str,
}

/// The resident threads and the free memory of the device behind `entry`.
fn capacity_of(entry: &Entry) -> Result<Capacity, EnsembleError> {
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let ctx = &entry.ctx;
    let attribute = |a: Attr| {
        ctx.attribute(a)
            .map(|v| u64::try_from(v).unwrap_or(0))
            .map_err(drv("query the device"))
    };
    let multiprocessors = attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let per_multiprocessor = attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?;
    let (free, _) = ctx.mem_get_info().map_err(drv("query the free memory"))?;
    Ok(Capacity::new(
        multiprocessors.saturating_mul(per_multiprocessor),
        free as u64,
        entry.held.load(Ordering::SeqCst),
    ))
}

/// The kernel text, the compute capability, the precision and the math tier of a compiled
/// kernel. The math tier changes the NVRTC options, not the text.
type KernelKey = (String, (i32, i32), Precision, Math);

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
    run: &Run,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let Run {
        layout,
        capacity,
        src,
    } = *run;
    let ctx = &entry.ctx;
    let cc = ctx
        .compute_capability()
        .map_err(drv("compute capability"))?;
    let function =
        entry
            .kernels
            .get_or_make((src.to_owned(), cc, job.precision, job.math), || {
                COMPILES.fetch_add(1, Ordering::SeqCst);
                let image = compile_image(src, p, cc, job.math)?;
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
        math: job.math,
    };
    // The free memory bounds the members of one batch. The private layout takes its local
    // memory first: the driver reserves it for every resident thread, whatever the batch size.
    let real_bytes = match job.precision {
        Precision::Single => size_of::<f32>(),
        Precision::Double => size_of::<f64>(),
    };
    let integrator = integrator_of(job.method);
    let private_bytes = private_reals(checked.n, integrator).saturating_mul(real_bytes);
    let work_len = layout.work_len(checked.n, integrator);
    let size = members_per_batch(
        job,
        checked.times.len(),
        work_len,
        real_bytes,
        Some(budget_of(buffer_room(layout, private_bytes, &capacity))),
        0,
    )?;
    if layout == Layout::Private {
        entry
            .held
            .fetch_max(reserve_of(private_bytes, &capacity), Ordering::SeqCst);
    }
    let ran = run_batched(job, size, cancel, |sub| match job.precision {
        Precision::Single => launch_all(
            ctx,
            &function,
            pack_in::<f32>(sub, checked, tape, layout),
            layout,
            cancel,
        )
        .and_then(|packed| packed.finish(label.clone())),
        Precision::Double => launch_all(
            ctx,
            &function,
            pack_in::<f64>(sub, checked, tape, layout),
            layout,
            cancel,
        )
        .and_then(|packed| packed.finish(label.clone())),
    });
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
fn launch_all<T: Real + DeviceRepr + ValidAsZeroBits>(
    ctx: &Arc<CudaContext>,
    function: &CudaFunction,
    mut packed: Packed<T>,
    layout: Layout,
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
    // `work` starts as zeros: clear it on the device, which skips the upload of a buffer that
    // is the largest of the run (`work_len(n, integrator)` values per member).
    let work_note = global_note(layout, packed.members, packed.work_len, size_of::<T>());
    let mut work = stream
        .alloc_zeros::<T>(packed.work.len())
        .map_err(step_err("allocate work on the device", work_note))?;
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
    // With the test feature, a launch of 0 blocks is invalid and the driver reports it.
    #[cfg(feature = "fault-injection")]
    let blocks = if crate::fault::take_launch_fault() {
        0
    } else {
        blocks
    };
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let members = packed.members as u64;
    let ns = packed.ns as u32;
    let pstride = packed.pstride as u64;
    let max_steps = packed.max_steps;

    let launch_error = drv("launch");
    drive(&mut packed.ctl, members as usize, cancel, |steps, ctl| {
        let started = Instant::now();
        #[cfg(feature = "fault-injection")]
        if crate::fault::take_panic_fault() {
            panic!("injected panic in launch");
        }
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
            unsafe { b.launch(cfg) }.map_err(&launch_error)?;
        }
        stream.synchronize().map_err(drv("synchronize"))?;
        let took = started.elapsed();
        stream
            .memcpy_dtoh(&ctl_d, ctl)
            .map_err(drv("copy ctl to host"))?;
        stream
            .synchronize()
            .map_err(drv("synchronize after copy"))?;
        Ok(took)
    })?;
    packed.out = stream.clone_dtoh(&out).map_err(drv("copy out to host"))?;
    // `finish` reads `work` only for a member that did not end `Ok`.
    if packed.needs_work() {
        packed.work = stream.clone_dtoh(&work).map_err(drv("copy work to host"))?;
    }
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
    fn the_math_tier_sets_the_fmad_option() {
        let exact = options("sm_120", Math::Exact);
        let fma = options("sm_120", Math::Fma);
        assert_eq!(
            exact,
            ["--gpu-architecture=sm_120", "--fmad=false", "--std=c++17"]
        );
        assert_eq!(
            fma,
            ["--gpu-architecture=sm_120", "--fmad=true", "--std=c++17"]
        );
    }

    #[test]
    fn the_cubin_route_builds_each_tier_with_its_own_option() {
        if let Err(why) = ensure_nvrtc() {
            let required = std::env::var("PC_REQUIRE_GPU")
                .map(|v| v.split(',').any(|b| b.trim() == "cuda"))
                .unwrap_or(false);
            assert!(
                !required,
                "PC_REQUIRE_GPU names cuda, but NVRTC is missing: {why}"
            );
            eprintln!("SKIPPED: no NVRTC ({why})");
            return;
        }
        let src = "extern \"C\" __global__ void k(double *a) { a[0] = a[1] * a[2] + a[3]; }";
        // The PTX route: the fma tier fuses the multiply and the add, the exact tier does not.
        let ptx = |math| {
            let opts = CompileOptions {
                options: options("compute_120", math),
                ..Default::default()
            };
            compile_ptx_with_opts(src, opts).unwrap().to_src()
        };
        assert!(!ptx(Math::Exact).contains("fma.rn.f64"));
        assert!(ptx(Math::Fma).contains("fma.rn.f64"));
        // The cubin route: a cubin also records its options, so different bytes alone do not
        // show fusion. They do show that the tier reaches NVRTC, since the same tier compiles
        // to the same bytes.
        let exact = compile_cubin(src, "sm_120", Math::Exact).unwrap();
        let fma = compile_cubin(src, "sm_120", Math::Fma).unwrap();
        assert!(!exact.is_empty() && !fma.is_empty());
        assert_ne!(exact, fma);
        assert_eq!(exact, compile_cubin(src, "sm_120", Math::Exact).unwrap());
    }

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
    #[cfg(not(windows))]
    fn an_arm64_server_toolkit_is_searched() {
        // CUDA for Linux arm64 servers (sbsa, such as DGX Spark) keeps NVRTC in
        // targets/sbsa-linux/lib.
        let root = std::env::temp_dir().join(format!("pc-sbsa-{}", std::process::id()));
        let lib = root.join("targets/sbsa-linux/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("libnvrtc.so.13"), b"").unwrap();
        let found: Vec<PathBuf> = dirs_under(vec![root.clone()])
            .iter()
            .flat_map(|d| files_in(d, is_nvrtc_file))
            .collect();
        std::fs::remove_dir_all(&root).unwrap();
        assert_eq!(found, vec![lib.join("libnvrtc.so.13")]);
    }

    #[test]
    fn a_text_buffer_ends_at_the_nul() {
        let raw: Vec<c_char> = b"error: x\0garbage"
            .iter()
            .map(|&b| c_char::from_ne_bytes([b]))
            .collect();
        assert_eq!(c_text(&raw), "error: x");
        assert_eq!(c_text(&[]), "");
    }

    #[test]
    fn a_launch_timeout_is_a_timeout_and_other_driver_errors_are_launch_errors() {
        use dsys::CUresult;
        let timeout = drv("synchronize")(DriverError(CUresult::CUDA_ERROR_LAUNCH_TIMEOUT));
        assert!(
            matches!(&timeout, EnsembleError::GpuTimeout(m) if m.contains("LAUNCH_TIMEOUT")),
            "{timeout:?}"
        );
        // The launch failure, an illegal address and an out of memory stay launch errors.
        for code in [
            CUresult::CUDA_ERROR_LAUNCH_FAILED,
            CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
            CUresult::CUDA_ERROR_OUT_OF_MEMORY,
        ] {
            let other = drv("launch")(DriverError(code));
            assert!(
                matches!(&other, EnsembleError::GpuLaunch(m) if m.starts_with("launch: ")),
                "{other:?}"
            );
        }
        assert_eq!(LAUNCH_TIMEOUT, CUresult::CUDA_ERROR_LAUNCH_TIMEOUT as u32);
    }

    #[test]
    fn a_sticky_error_says_to_restart_and_the_others_do_not() {
        use dsys::CUresult::*;
        let codes = [
            CUDA_ERROR_ILLEGAL_ADDRESS,
            CUDA_ERROR_LAUNCH_TIMEOUT,
            CUDA_ERROR_ASSERT,
            CUDA_ERROR_HARDWARE_STACK_ERROR,
            CUDA_ERROR_ILLEGAL_INSTRUCTION,
            CUDA_ERROR_MISALIGNED_ADDRESS,
            CUDA_ERROR_INVALID_ADDRESS_SPACE,
            CUDA_ERROR_INVALID_PC,
            CUDA_ERROR_LAUNCH_FAILED,
        ];
        let listed: Vec<u32> = codes.iter().map(|&c| c as u32).collect();
        assert_eq!(listed, STICKY, "the list matches the driver's numbers");
        for code in codes {
            let e = drv("synchronize")(DriverError(code));
            let (EnsembleError::GpuLaunch(m) | EnsembleError::GpuTimeout(m)) = e else {
                panic!("a device error is a launch or a timeout");
            };
            assert!(m.contains("restart"), "{m}");
        }
        for code in [
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_OUT_OF_MEMORY,
            CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
        ] {
            let EnsembleError::GpuLaunch(m) = drv("launch")(DriverError(code)) else {
                panic!("a launch error");
            };
            assert!(!m.contains("restart"), "{m}");
        }
    }

    /// A device with room for anything: the size rules alone decide.
    const ROOMY: Capacity = Capacity {
        resident_threads: 1,
        free_bytes: u64::MAX,
        held_bytes: 0,
    };

    #[test]
    fn the_layout_is_global_from_256_kb_of_private_arrays() {
        // RK4 holds 7 n reals per thread.
        let at = |n: usize, size: usize| layout_for(n, Integrator::Rk4, size, ROOMY);
        assert_eq!(at(4681, 8), Layout::Private, "262136 bytes");
        assert_eq!(at(4682, 8), Layout::Global, "262192 bytes");
        assert_eq!(at(9362, 4), Layout::Private, "262136 bytes");
        assert_eq!(at(9363, 4), Layout::Global, "262164 bytes");
        assert_eq!(at(0, 8), Layout::Private);
        // A huge n must not overflow the size.
        assert_eq!(at(usize::MAX / 2, 8), Layout::Global);
        let dp5 = layout_for(usize::MAX / 2, Integrator::Dp5, 8, ROOMY);
        assert_eq!(dp5, Layout::Global);
    }

    #[test]
    fn dormand_prince_is_global_from_3300_private_reals_at_any_precision() {
        // 10 n reals per thread, so 3300 reals are 330 states.
        for size in [4, 8] {
            let at = |n: usize| layout_for(n, Integrator::Dp5, size, ROOMY);
            assert_eq!(at(40), Layout::Private);
            assert_eq!(at(329), Layout::Private, "3290 reals");
            assert_eq!(at(330), Layout::Global, "3300 reals");
            assert_eq!(at(400), Layout::Global);
        }
        // RK4 with as many reals (470 states) stays private.
        assert_eq!(layout_for(470, Integrator::Rk4, 8, ROOMY), Layout::Private);
    }

    /// Resident threads and free memory that make the numbers small: an RK4 model of 10 states
    /// with doubles keeps 70 reals, 560 bytes, per thread, so 1000 threads reserve 560000.
    fn small(free: u64) -> Capacity {
        Capacity::new(1000, free, 0)
    }

    #[test]
    fn the_layout_is_private_while_the_reserve_is_within_three_quarters_of_the_free_memory() {
        let at = |free: u64| layout_for(10, Integrator::Rk4, 8, small(free));
        // 560000 * 4 <= free * 3 from free = 746667.
        assert_eq!(at(746_666), Layout::Global);
        assert_eq!(at(746_667), Layout::Private);
        assert_eq!(at(u64::MAX), Layout::Private);
        assert_eq!(at(0), Layout::Global);
        // No model needs a reserve when it has no states, and nothing overflows.
        assert_eq!(layout_for(0, Integrator::Rk4, 8, small(0)), Layout::Private);
        let wide = Capacity::new(u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(layout_for(10, Integrator::Rk4, 8, wide), Layout::Global);
    }

    #[test]
    fn the_layout_does_not_change_after_the_context_holds_its_reserve() {
        // A private run takes 560000 bytes from the free memory and the context keeps them.
        let before = Capacity::new(1000, 800_000, 0);
        let after = Capacity::new(1000, 800_000 - 560_000, 560_000);
        assert_eq!(before.free_bytes, after.free_bytes);
        let at = |c: Capacity| layout_for(10, Integrator::Rk4, 8, c);
        assert_eq!(at(before), Layout::Private);
        assert_eq!(at(after), Layout::Private);
        // A held reserve is added to the free memory only once.
        assert_eq!(Capacity::new(1, u64::MAX, 5).free_bytes, u64::MAX);
    }

    #[test]
    fn the_buffers_get_the_free_memory_less_the_reserve_of_the_private_layout() {
        let c = small(1_000_000);
        assert_eq!(buffer_room(Layout::Global, 560, &c), 1_000_000);
        assert_eq!(buffer_room(Layout::Private, 560, &c), 440_000);
        // A reserve larger than the free memory leaves nothing.
        assert_eq!(buffer_room(Layout::Private, 5600, &c), 0);
        // The context already holds 560000: the free memory is net of it, so a run that needs
        // no more takes nothing more, and a run that needs 1000 more bytes per thread takes
        // the difference.
        let held = Capacity::new(1000, 440_000, 560_000);
        assert_eq!(buffer_room(Layout::Private, 560, &held), 440_000);
        assert_eq!(buffer_room(Layout::Global, 560, &held), 440_000);
        assert_eq!(buffer_room(Layout::Private, 700, &held), 300_000);
    }

    #[test]
    fn a_forced_layout_wins_over_the_size_and_the_memory() {
        let (n, i, size) = (40, Integrator::Dp5, 8);
        assert_eq!(pick(None, n, i, size, ROOMY), Layout::Private);
        assert_eq!(pick(None, n, i, size, small(0)), Layout::Global);
        assert_eq!(
            pick(Some(Layout::Global), n, i, size, ROOMY),
            Layout::Global
        );
        assert_eq!(
            pick(Some(Layout::Private), 6000, i, size, small(0)),
            Layout::Private
        );
        assert_eq!(pick(None, 6000, i, size, ROOMY), Layout::Global);
    }

    #[test]
    fn the_forced_layout_round_trips() {
        // Only this test touches the setting, and it puts it back.
        for want in [Some(Layout::Global), Some(Layout::Private), None] {
            force_layout(want);
            assert_eq!(forced_layout(), want);
        }
    }

    #[test]
    fn the_budget_is_sixty_percent_of_the_free_memory() {
        assert_eq!(budget_of(0), 0);
        assert_eq!(budget_of(100), 60);
        assert_eq!(budget_of(8 << 30), (8u64 << 30) / 10 * 6);
        assert!(budget_of(u64::MAX) < u64::MAX);
    }

    #[test]
    fn the_global_scratch_note_names_the_size_and_the_way_out() {
        // The private layout needs no note: its work buffer is small.
        assert_eq!(global_note(Layout::Private, 16, 321, 8), None);
        // 4 members of 12 n + 3 = 483 doubles each.
        let note = global_note(Layout::Global, 4, 483, 8).expect("global");
        for want in ["4 members", "16 KB", "this batch", "smaller model", "cpu"] {
            assert!(note.contains(want), "{want} in {note}");
        }
        assert!(!note.contains("restart"), "{note}");
        // A size past the address range must not overflow.
        assert!(global_note(Layout::Global, usize::MAX, usize::MAX, 8).is_some());
    }

    #[test]
    fn a_driver_error_carries_the_note_only_when_it_can_be_the_cause() {
        use dsys::CUresult::*;
        let note = Some("NOTE".to_string());
        let plain = |code| match step_err("allocate", None)(DriverError(code)) {
            EnsembleError::GpuLaunch(m) => m,
            other => panic!("{other:?}"),
        };
        let noted = |code| step_err("allocate", note.clone())(DriverError(code));
        // An allocation that does not fit gets the note after the driver text.
        let EnsembleError::GpuLaunch(m) = noted(CUDA_ERROR_OUT_OF_MEMORY) else {
            panic!("a launch error");
        };
        assert_eq!(m, format!("{}; NOTE", plain(CUDA_ERROR_OUT_OF_MEMORY)));
        assert!(!m.contains("restart"), "{m}");
        // A sticky error came from an earlier fault and keeps its own text.
        for code in [CUDA_ERROR_ILLEGAL_ADDRESS, CUDA_ERROR_LAUNCH_FAILED] {
            let EnsembleError::GpuLaunch(m) = noted(code) else {
                panic!("a launch error");
            };
            assert_eq!(m, plain(code));
            assert!(!m.contains("NOTE"), "{m}");
        }
        // A watchdog stop stays a timeout without the note.
        let timeout = noted(CUDA_ERROR_LAUNCH_TIMEOUT);
        assert!(
            matches!(&timeout, EnsembleError::GpuTimeout(m) if !m.contains("NOTE")),
            "{timeout:?}"
        );
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
