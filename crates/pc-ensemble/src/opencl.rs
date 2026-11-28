//! The OpenCL backend: one work-item per member, compiled at run time by the driver.
//!
//! This is the route for AMD, Intel and Apple (through the OpenCL framework) and the fallback
//! for NVIDIA when no NVRTC is around. Nothing here links the OpenCL library. opencl3 is built
//! with its `dynamic` feature, so the first call opens `OpenCL.dll` or `libOpenCL.so.1`, and a
//! missing library comes back as an error code instead of a link failure. Several opencl3
//! types still panic in their `Drop` code when the driver reports an error, so every call
//! runs inside [`guard`], which turns a panic into an error (rule 7). Every object is held in a
//! [`Quiet`], which guards its own drop. A guard around the whole scope is not enough: a
//! second failing release while the first one unwinds aborts the process.
//!
//! The kernel is compiled for OpenCL C 1.2. Single precision divide and square root are
//! correctly rounded when the device can do that, and the kernel text switches off floating
//! point contraction, so a run reproduces the CPU backend operation for operation wherever the
//! driver honours both. A driver that does not is the reason the tiers of the test plan have
//! a stated tolerance.
//!
//! A run is cut into launches by [`drive`], as on CUDA. OpenCL has no error code for the
//! Windows display watchdog, so a launch that dies with `CL_OUT_OF_RESOURCES` after a long
//! wait is reported as a timeout. That guess is unverified on real drivers.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device};
use opencl3::error_codes::{
    CL_DEVICE_NOT_FOUND, CL_OUT_OF_RESOURCES, CL_PLATFORM_NOT_FOUND_KHR, ClError,
    DLOPEN_RUNTIME_LOAD_FAILED,
};
use opencl3::kernel::Kernel;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE, ClMem};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id, cl_mem, cl_uint, cl_ulong};
use pc_codegen::kernel::{Dialect, Scalar};

use crate::abi::{Packed, integrator_of, pack_in};
use crate::batch::{half_budget, members_per_batch, run_batched};
use crate::device::{Backend, BackendState, DeviceInfo};
use crate::gpu::{
    ForcedLayout, Quiet, check_alloc, drive, global_size, guard, kernel_err, layout_by_size,
};
use crate::job::{
    Checked, EnsembleError, EnsembleResult, Job, Label, Precision, StackLayout, StackRan, validate,
};
use crate::kept::{Kept, Lru};
use crate::kernel::{Layout, Stack, kernel_source_with, stack_of};
use crate::real::Real;
use crate::tape::Tape;
use crate::winload::{self, NoCwdSearch};

/// The name of the kernel entry point.
const ENTRY: &str = "pc_ens";
/// Work-items per work-group, unless the kernel allows fewer.
const GROUP: usize = 64;
/// A launch that fails with `CL_OUT_OF_RESOURCES` after at least this long is reported as a
/// timeout. The Windows watchdog fires after 2 s by default.
const WATCHDOG_GUESS: Duration = Duration::from_millis(1500);

/// Formats an OpenCL error as "step: CL_NAME (code)".
fn cl_text(step: &str, e: &ClError) -> String {
    format!("{step}: {e} ({})", e.0)
}

/// An OpenCL error of a copy or a setup call.
fn cl(step: &'static str) -> impl Fn(ClError) -> EnsembleError {
    move |e| EnsembleError::GpuLaunch(cl_text(step, &e))
}

/// An error of a launch, a synchronize or a read after one. `took` is the time since the
/// launch started.
fn launch_error(step: &str, e: &ClError, took: Duration) -> EnsembleError {
    let text = cl_text(step, e);
    if e.0 == CL_OUT_OF_RESOURCES && took >= WATCHDOG_GUESS {
        EnsembleError::GpuTimeout(format!(
            "{text} after {:.1} s, which looks like the display watchdog",
            took.as_secs_f64()
        ))
    } else {
        EnsembleError::GpuLaunch(text)
    }
}

// ---------------------------------------------------------------------------------------
// The probe

/// What a run needs to know about a device besides what [`DeviceInfo`] holds.
#[derive(Clone, Copy, Debug)]
struct Handle {
    /// The `cl_device_id`, kept as a number so the probe can live in a static.
    id: usize,
    /// Largest single buffer, in bytes.
    max_alloc: u64,
    /// Single precision divide and square root can be correctly rounded.
    correct_div: bool,
}

struct Probe {
    state: BackendState,
    devices: Vec<DeviceInfo>,
    /// One entry per device, in the same order.
    handles: Vec<Handle>,
}

impl Probe {
    fn unavailable(why: impl Into<String>) -> Probe {
        Probe {
            state: BackendState::Unavailable(why.into()),
            devices: Vec::new(),
            handles: Vec::new(),
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

/// Bit of `CL_DEVICE_SINGLE_FP_CONFIG`: divide and square root are correctly rounded.
const FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: cl_ulong = 1 << 7;

fn describe(dev: &Device, platform: &str, index: usize) -> Result<(DeviceInfo, Handle), String> {
    let q = |what: &str, e: ClError| format!("{what}: {e}");
    if !dev.available().map_err(|e| q("available", e))? {
        return Err("the driver marks the device unavailable".into());
    }
    if !dev
        .compiler_available()
        .map_err(|e| q("compiler_available", e))?
    {
        return Err("the device has no compiler".into());
    }
    let name = dev.name().map_err(|e| q("name", e))?;
    let version = dev.version().unwrap_or_default();
    let driver = dev.driver_version().unwrap_or_default();
    let extensions = dev.extensions().unwrap_or_default();
    let fp64 = dev.double_fp_config().unwrap_or(0) != 0
        && extensions.split_whitespace().any(|e| e == "cl_khr_fp64");
    let correct_div = dev.single_fp_config().unwrap_or(0) & FP_CORRECTLY_ROUNDED_DIVIDE_SQRT != 0;
    let memory = dev.global_mem_size().ok();
    let max_alloc = dev.max_mem_alloc_size().unwrap_or(0);
    let info = DeviceInfo {
        backend: Backend::OpenCl,
        index,
        name: name.trim().to_owned(),
        driver: format!(
            "{}, {}, driver {}",
            platform.trim(),
            version.trim(),
            driver.trim()
        ),
        fp32: true,
        fp64,
        memory_bytes: memory,
    };
    let handle = Handle {
        id: dev.id() as usize,
        max_alloc,
        correct_div,
    };
    Ok((info, handle))
}

fn probe_now() -> Probe {
    // The ICD loader and the drivers it opens load inside the window, so none of them comes
    // from the current folder. On Windows the loader itself must be in the system folder,
    // unless OPENCL_DYLIB_PATH names one.
    let _window = NoCwdSearch::new();
    if cfg!(windows)
        && std::env::var_os("OPENCL_DYLIB_PATH").is_none()
        && !winload::system_library("OpenCL.dll")
    {
        return Probe::unavailable(
            "OpenCL.dll was not found in the Windows system folder (OPENCL_DYLIB_PATH names \
             another one)",
        );
    }
    let platforms = match get_platforms() {
        Ok(p) => p,
        Err(e) if e.0 == DLOPEN_RUNTIME_LOAD_FAILED => {
            return Probe::unavailable(
                "the OpenCL library was not found (OpenCL.dll on Windows, libOpenCL.so.1 on \
                 Linux, OpenCL.framework on macOS; OPENCL_DYLIB_PATH names another one)",
            );
        }
        Err(e) if e.0 == CL_PLATFORM_NOT_FOUND_KHR => {
            return Probe::unavailable(
                "the OpenCL library loaded but no driver is registered with it (no ICD)",
            );
        }
        Err(e) => return Probe::unavailable(cl_text("clGetPlatformIDs", &e)),
    };
    let mut devices = Vec::new();
    let mut handles = Vec::new();
    let mut notes = Vec::new();
    for platform in platforms {
        let pname = platform
            .name()
            .unwrap_or_else(|_| "unknown platform".into());
        let ids = match platform.get_devices(CL_DEVICE_TYPE_ALL) {
            Ok(ids) => ids,
            Err(e) if e.0 == CL_DEVICE_NOT_FOUND => continue,
            Err(e) => {
                notes.push(format!("{pname}: {}", cl_text("clGetDeviceIDs", &e)));
                continue;
            }
        };
        for id in ids {
            let dev = Device::new(id);
            match describe(&dev, &pname, devices.len()) {
                Ok((info, handle)) => {
                    devices.push(info);
                    handles.push(handle);
                }
                Err(why) => notes.push(format!("{pname}: skipped a device: {why}")),
            }
        }
    }
    if devices.is_empty() {
        let mut why = String::from("the OpenCL library lists no usable device");
        if !notes.is_empty() {
            why.push_str(" (");
            why.push_str(&notes.join("; "));
            why.push(')');
        }
        return Probe::unavailable(why);
    }
    Probe {
        state: BackendState::Available,
        devices,
        handles,
    }
}

/// Whether the OpenCL backend can run in this process.
pub fn backend_state() -> BackendState {
    probe().state.clone()
}

/// The OpenCL devices, empty when the backend is not available.
pub fn devices() -> Vec<DeviceInfo> {
    probe().devices.clone()
}

// ---------------------------------------------------------------------------------------
// Building

/// The build options of the kernel.
///
/// There is no `-cl-fast-relaxed-math`. Correctly rounded divide and square root are asked
/// for only on a device that reports them, because a compiler may refuse the option.
fn build_options(correct_div: bool) -> String {
    let mut o = String::from("-cl-std=CL1.2");
    if correct_div {
        o.push_str(" -cl-fp32-correctly-rounded-divide-sqrt");
    }
    o
}

// ---------------------------------------------------------------------------------------
// Running

/// Runs `job` on the OpenCL device `device`.
///
/// A missing library, a missing device, a kernel the driver rejects, a failed copy or launch
/// and a probable watchdog stop each come back as their own [`EnsembleError`]. A panic inside
/// opencl3 comes back as [`EnsembleError::GpuLaunch`]. The cancel flag is read between
/// launches.
pub fn run_opencl(
    device: &DeviceInfo,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let checked = validate(job)?;
    let p = probe();
    if let BackendState::Unavailable(why) = &p.state {
        return Err(EnsembleError::GpuUnavailable(why.clone()));
    }
    if device.backend != Backend::OpenCl || device.index >= p.devices.len() {
        return Err(EnsembleError::GpuUnavailable(format!(
            "no OpenCL device {}: the driver lists {}",
            device.index,
            p.devices.len()
        )));
    }
    // The probe is the truth about the device, not the copy the caller holds.
    let info = &p.devices[device.index];
    let handle = p.handles[device.index];
    if job.precision == Precision::Double && !info.fp64 {
        return Err(EnsembleError::GpuUnavailable(format!(
            "{} has no double precision (cl_khr_fp64): run it in single precision",
            info.name
        )));
    }
    // OpenCL and Metal do not choose a stack layout, so `auto` runs member (SPEC 8af-5).
    let asked = job.stack;
    let resolved = Job {
        stack: job.stack.or_member(),
        ..job.clone()
    };
    let job = &resolved;
    // A tape with calls runs on a per-member stack in global memory (SPEC section 8af-5).
    let tape = Tape::build(job.model)?;
    let scalar = match job.precision {
        Precision::Single => Scalar::F32,
        Precision::Double => Scalar::F64,
    };
    // OpenCL has no query for the free memory or the resident work-items, so the layout
    // follows the size of the model only (SPEC 13).
    let integrator = integrator_of(job.method);
    let real_bytes = match job.precision {
        Precision::Single => size_of::<f32>(),
        Precision::Double => size_of::<f64>(),
    };
    let layout = layout_by_size(forced_layout(), checked.n, integrator, real_bytes);
    let src = kernel_source_with(
        &tape,
        integrator,
        scalar,
        Dialect::OpenCl,
        layout,
        job.stack,
    )
    .map_err(kernel_err)?;
    if src.contains('\0') {
        return Err(EnsembleError::Unsupported(
            "the generated kernel contains a NUL character".into(),
        ));
    }
    let label = Label {
        backend: Backend::OpenCl.name().into(),
        device: info.name.clone(),
        driver: info.driver.clone(),
        precision: job.precision,
        math: job.math,
        stack: asked,
        stack_ran: StackRan::of(job.stack),
        sorted: false,
    };
    // The entry this run used, so that a failure removes that one and not a newer one.
    let mut used: Option<Arc<Shared>> = None;
    let ran = guard(|| {
        let shared = SHARED.get_or_make(device.index, || make_shared(handle))?;
        let shared = used.insert(shared);
        session(
            shared,
            handle,
            info,
            label,
            job,
            (&checked, &tape, &src, layout),
            cancel,
        )
    })
    .unwrap_or_else(|msg| Err(EnsembleError::GpuLaunch(format!("panic in opencl3: {msg}"))));
    if matches!(
        ran,
        Err(EnsembleError::GpuLaunch(_) | EnsembleError::GpuTimeout(_))
    ) && let Some(seen) = &used
    {
        // A lost device or a stuck queue would fail every later run, so the next run on this
        // device starts from a new context. Only the entry this run used is removed. The drops
        // of opencl3 panic on a failed release, so removed values are dropped inside a guard.
        let gone = SHARED.forget_if(device.index, seen);
        let _ = guard(|| drop(gone));
    }
    let _ = guard(|| drop(used));
    ran
}

/// What the runs on one device share: the context, an in-order queue on it and the programs
/// built for it.
struct Shared {
    // The programs drop first, then the queue, then the context.
    programs: Lru<ProgramKey, Arc<Quiet<Program>>>,
    queue: Quiet<CommandQueue>,
    ctx: Quiet<Context>,
}

/// The layout that a test or a bench forces for every OpenCL run in this process.
static FORCED: ForcedLayout = ForcedLayout::new();

/// Forces the scratch layout of every OpenCL run in this process, or lifts the force with
/// `None`. The device test and the A/B bench use it to run both layouts on one model. It is
/// not part of the API.
#[doc(hidden)]
pub fn force_layout(layout: Option<Layout>) {
    FORCED.set(layout);
}

fn forced_layout() -> Option<Layout> {
    FORCED.get()
}

/// The kernel text, the build options and the precision of a built program.
type ProgramKey = (String, String, Precision);

/// How many built programs a device keeps.
const PROGRAMS_KEPT: usize = 8;

/// The context, queue and programs of each device, kept between runs because creating a
/// context takes tens of milliseconds on a laptop GPU and building a program a few
/// milliseconds on some drivers.
static SHARED: Kept<Shared> = Kept::new();

/// How many programs have been built in this process, for the cache test.
static BUILDS: AtomicUsize = AtomicUsize::new(0);

/// How many programs the kept entry of device `index` holds, for the cache test.
#[doc(hidden)]
pub fn kept_kernels(index: usize) -> usize {
    SHARED.get(index).map_or(0, |s| s.programs.len())
}

/// How many programs this process has built, for the cache test.
#[doc(hidden)]
pub fn kernel_compiles() -> usize {
    BUILDS.load(Ordering::SeqCst)
}

fn make_shared(handle: Handle) -> Result<Arc<Shared>, EnsembleError> {
    let dev = Device::new(handle.id as cl_device_id);
    let ctx = Context::from_device(&dev).map_err(cl("create context"))?;
    // `clCreateCommandQueue` exists in every OpenCL version. The 2.0 entry point is missing
    // from a 1.2-only driver, and the kernel text is OpenCL C 1.2 anyway.
    let queue = CommandQueue::create_default(&ctx, 0).map_err(cl("create command queue"))?;
    Ok(Arc::new(Shared {
        programs: Lru::new(PROGRAMS_KEPT),
        queue: Quiet::new(queue),
        ctx: Quiet::new(ctx),
    }))
}

fn session(
    shared: &Shared,
    handle: Handle,
    info: &DeviceInfo,
    label: Label,
    job: &Job,
    (checked, tape, src, layout): (&Checked, &Tape, &str, Layout),
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let id = handle.id as cl_device_id;
    let (ctx, queue) = (&shared.ctx, &shared.queue);
    let options = build_options(handle.correct_div);
    let program =
        shared
            .programs
            .get_or_make((src.to_owned(), options.clone(), job.precision), || {
                BUILDS.fetch_add(1, Ordering::SeqCst);
                Program::create_and_build_from_source(ctx, src, &options)
                    .map(|program| Arc::new(Quiet::new(program)))
                    .map_err(|log| EnsembleError::GpuCompile(log.trim().to_owned()))
            })?;
    let kernel = Kernel::create(&program, ENTRY)
        .map(Quiet::new)
        .map_err(|e| EnsembleError::GpuCompile(cl_text("create kernel", &e)))?;
    let group = kernel
        .get_work_group_size(id)
        .map_err(cl("kernel work group size"))?
        .clamp(1, GROUP);
    let memory = info.memory_bytes;
    // OpenCL has no free-memory query, so the budget is a fraction of the device memory.
    // `max_alloc` limits each buffer of a batch.
    let (real_bytes, work_len) = (
        match job.precision {
            Precision::Single => size_of::<f32>(),
            Precision::Double => size_of::<f64>(),
        },
        layout.work_len(checked.n, integrator_of(job.method)),
    );
    // A kernel with calls also takes its stack, a member buffer outside the scratch rule.
    let stack = stack_of(tape);
    let stack_len = stack.as_ref().map_or(0, Stack::reals);
    let size = members_per_batch(
        job,
        checked.times.len(),
        work_len,
        stack.as_ref(),
        real_bytes,
        memory.map(half_budget),
        handle.max_alloc,
    )?;
    run_batched(job, size, cancel, |sub| match job.precision {
        Precision::Single => launch_all(
            ctx,
            queue,
            &kernel,
            (group, handle.max_alloc, memory, stack_len, job.stack),
            pack_in::<f32>(sub, checked, tape, layout),
            cancel,
        )
        .and_then(|packed| packed.finish(label.clone())),
        Precision::Double => launch_all(
            ctx,
            queue,
            &kernel,
            (group, handle.max_alloc, memory, stack_len, job.stack),
            pack_in::<f64>(sub, checked, tape, layout),
            cancel,
        )
        .and_then(|packed| packed.finish(label.clone())),
    })
}

/// Creates a device buffer for `data` and copies `data` into it.
fn upload<T>(
    ctx: &Context,
    queue: &CommandQueue,
    data: &[T],
    step: &'static str,
) -> Result<Quiet<Buffer<T>>, EnsembleError> {
    if data.is_empty() {
        return Err(EnsembleError::BadJob(format!("{step}: nothing to copy")));
    }
    // SAFETY: no host pointer and no host-pointer flag, so the driver allocates `data.len()`
    // elements on the device.
    let mut buf =
        unsafe { Buffer::<T>::create(ctx, CL_MEM_READ_WRITE, data.len(), ptr::null_mut()) }
            .map(Quiet::new)
            .map_err(cl(step))?;
    // SAFETY: the buffer holds `data.len()` elements and the write blocks, so `data` is read
    // before the call returns.
    unsafe { queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[]) }.map_err(cl(step))?;
    Ok(buf)
}

/// Creates a device buffer of `len` zeros.
///
/// The device clears the buffer with `clEnqueueFillBuffer` (OpenCL 1.2), so the zeros do not
/// cross the bus. A driver that rejects the fill gets the same zeros as an upload. The queue
/// is in order, so every later command sees the cleared buffer.
fn zeroed<T: Real>(
    ctx: &Context,
    queue: &CommandQueue,
    len: usize,
    step: &'static str,
) -> Result<Quiet<Buffer<T>>, EnsembleError> {
    if len == 0 {
        return Err(EnsembleError::BadJob(format!("{step}: nothing to clear")));
    }
    // SAFETY: no host pointer and no host-pointer flag, so the driver allocates `len`
    // elements on the device.
    let mut buf = unsafe { Buffer::<T>::create(ctx, CL_MEM_READ_WRITE, len, ptr::null_mut()) }
        .map(Quiet::new)
        .map_err(cl(step))?;
    let bytes = len * std::mem::size_of::<T>();
    // SAFETY: the pattern is one element and `bytes` is a whole number of elements that the
    // buffer holds.
    match unsafe { queue.enqueue_fill_buffer(&mut buf, &[T::ZERO], 0, bytes, &[]) } {
        Ok(event) => drop(Quiet::new(event)),
        Err(_) => {
            let zeros = vec![T::ZERO; len];
            // SAFETY: the buffer holds `len` elements and the write blocks, so `zeros` is read
            // before the call returns.
            unsafe { queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &zeros, &[]) }
                .map_err(cl(step))?;
        }
    }
    Ok(buf)
}

/// Reads all of `buf` into `out` and waits for the copy.
fn download<T>(
    queue: &CommandQueue,
    buf: &Buffer<T>,
    out: &mut [T],
    step: &str,
    since: Instant,
) -> Result<(), EnsembleError> {
    // SAFETY: the buffer was created with `out.len()` elements and the read blocks, so `out`
    // is filled before the call returns.
    unsafe { queue.enqueue_read_buffer(buf, CL_BLOCKING, 0, out, &[]) }
        .map_err(|e| launch_error(step, &e, since.elapsed()))?;
    Ok(())
}

/// Sets argument `index` of the kernel.
fn set<T>(
    kernel: &Kernel,
    index: cl_uint,
    value: &T,
    step: &'static str,
) -> Result<(), EnsembleError> {
    // SAFETY: the callers pass the type the kernel declares at `index`.
    unsafe { kernel.set_arg(index, value) }.map_err(cl(step))
}

/// Uploads the buffers, launches until no member is running and downloads the result.
/// `stack_len` is the reals of the stack of one member, 0 for a kernel without calls, and
/// `stack` its layout, which pads the buffer to whole groups of 32 members.
fn launch_all<T: Real>(
    ctx: &Context,
    queue: &CommandQueue,
    kernel: &Kernel,
    (group, max_alloc, memory, stack_len, stack): (usize, u64, Option<u64>, usize, StackLayout),
    mut packed: Packed<T>,
    cancel: &AtomicBool,
) -> Result<Packed<T>, EnsembleError> {
    let size = |len: usize, each: usize| (len * each) as u64;
    let t = std::mem::size_of::<T>();
    check_alloc(
        &[
            ("rc", size(packed.rc.len(), t)),
            ("segm", size(packed.segm.len(), 8)),
            ("x0", size(packed.x0.len(), t)),
            ("par", size(packed.par.len(), t)),
            ("work", size(packed.work.len(), t)),
            ("out", size(packed.out.len(), t)),
            ("ctl", size(packed.ctl.len(), 4)),
            (
                "stk",
                size(stack.buffer_reals(packed.members, stack_len), t),
            ),
        ],
        max_alloc,
        memory,
    )?;
    let rc = upload(ctx, queue, &packed.rc, "copy rc to device")?;
    let segm = upload(ctx, queue, &packed.segm, "copy segm to device")?;
    let x0 = upload(ctx, queue, &packed.x0, "copy x0 to device")?;
    let par = upload(ctx, queue, &packed.par, "copy par to device")?;
    let work = zeroed::<T>(ctx, queue, packed.work.len(), "clear work on device")?;
    let out = upload(ctx, queue, &packed.out, "copy out to device")?;
    let ctl_d = upload(ctx, queue, &packed.ctl, "copy ctl to device")?;
    // The kernel writes a frame before it reads it; the clear runs on the device. A kernel
    // without calls takes no stack argument.
    let stk = match stack_len {
        0 => None,
        len => Some(zeroed::<T>(
            ctx,
            queue,
            stack.buffer_reals(packed.members, len),
            "allocate the call stack on the device",
        )?),
    };

    let members = packed.members as cl_ulong;
    let max_steps: cl_uint = packed.max_steps;
    let ns = packed.ns as cl_uint;
    let pstride = packed.pstride as cl_ulong;
    let buffers: [cl_mem; 7] = [
        rc.get(),
        segm.get(),
        x0.get(),
        par.get(),
        work.get(),
        out.get(),
        ctl_d.get(),
    ];
    for (i, b) in buffers.iter().enumerate() {
        set(kernel, i as cl_uint, b, "set buffer argument")?;
    }
    set(kernel, 7, &members, "set members")?;
    set(kernel, 9, &max_steps, "set max_steps")?;
    set(kernel, 10, &ns, "set ns")?;
    set(kernel, 11, &pstride, "set pstride")?;
    if let Some(stk) = &stk {
        set(kernel, 12, &stk.get(), "set the call stack")?;
    }

    let global = [global_size(packed.members, group)];
    let local = [group];
    // With the test feature, a work group far above the device limit is invalid and the
    // driver reports it.
    #[cfg(feature = "fault-injection")]
    let (global, local) = if crate::fault::take_launch_fault() {
        ([group << 24], [group << 24])
    } else {
        (global, local)
    };
    drive(&mut packed.ctl, packed.members, cancel, |steps, ctl| {
        let started = Instant::now();
        #[cfg(feature = "fault-injection")]
        if crate::fault::take_panic_fault() {
            panic!("injected panic in launch");
        }
        let steps: cl_ulong = steps;
        set(kernel, 8, &steps, "set steps")?;
        // SAFETY: the arguments follow the signature of `pc_ens` in order and type, the
        // buffers have the sizes `crate::abi` documents, the stack has `PC_STK` reals per
        // member, `global` is a multiple of `local`,
        // and the kernel returns early for a work-item past the last member.
        unsafe {
            queue.enqueue_nd_range_kernel(
                kernel.get(),
                1,
                ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                &[],
            )
        }
        .map_err(|e| launch_error("launch", &e, started.elapsed()))?;
        queue
            .finish()
            .map_err(|e| launch_error("synchronize", &e, started.elapsed()))?;
        let took = started.elapsed();
        download(queue, &ctl_d, ctl, "copy ctl to host", started)?;
        Ok(took)
    })?;
    let done = Instant::now();
    download(queue, &out, &mut packed.out, "copy out to host", done)?;
    // `finish` reads `work` only for a member that did not end `Ok`.
    if packed.needs_work() {
        download(queue, &work, &mut packed.work, "copy work to host", done)?;
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_options_have_no_fast_math() {
        let plain = build_options(false);
        assert_eq!(plain, "-cl-std=CL1.2");
        let exact = build_options(true);
        assert!(exact.starts_with("-cl-std=CL1.2"));
        assert!(exact.contains("-cl-fp32-correctly-rounded-divide-sqrt"));
        for o in [plain, exact] {
            assert!(!o.contains("fast-relaxed-math"), "{o}");
            assert!(!o.contains("unsafe-math"), "{o}");
            assert!(!o.contains("mad-enable"), "{o}");
        }
    }

    #[test]
    fn a_slow_out_of_resources_is_a_timeout_and_a_fast_one_is_not() {
        let e = ClError(CL_OUT_OF_RESOURCES);
        let slow = launch_error("synchronize", &e, Duration::from_secs(2));
        assert!(matches!(&slow, EnsembleError::GpuTimeout(m) if m.contains("CL_OUT_OF_RESOURCES")));
        let fast = launch_error("synchronize", &e, Duration::from_millis(20));
        assert!(matches!(&fast, EnsembleError::GpuLaunch(m) if m.contains("synchronize")));
        // Another error code stays a launch error however long it took.
        let other = launch_error("launch", &ClError(-30), Duration::from_secs(10));
        assert!(matches!(other, EnsembleError::GpuLaunch(_)));
    }

    #[test]
    fn the_timeout_threshold_is_one_and_a_half_seconds() {
        let e = ClError(CL_OUT_OF_RESOURCES);
        let below = launch_error("synchronize", &e, Duration::from_millis(1400));
        assert!(matches!(below, EnsembleError::GpuLaunch(_)), "{below:?}");
        let at = launch_error("synchronize", &e, Duration::from_millis(1500));
        assert!(matches!(at, EnsembleError::GpuTimeout(_)), "{at:?}");
    }

    #[test]
    fn an_error_text_names_the_step_and_the_code() {
        let text = cl_text("copy rc to device", &ClError(CL_OUT_OF_RESOURCES));
        assert!(text.starts_with("copy rc to device: "));
        assert!(text.contains(&format!("({CL_OUT_OF_RESOURCES})")));
    }

    #[test]
    fn the_probe_never_panics_and_agrees_with_itself() {
        // This runs on machines with and without an OpenCL driver. It checks the shape.
        let p = probe();
        match &p.state {
            BackendState::Available => {
                assert!(!p.devices.is_empty());
                assert_eq!(p.devices.len(), p.handles.len());
                for (i, d) in p.devices.iter().enumerate() {
                    assert_eq!(d.index, i);
                    assert_eq!(d.backend, Backend::OpenCl);
                    assert!(d.fp32 && !d.name.is_empty());
                }
            }
            BackendState::Unavailable(why) => {
                assert!(!why.is_empty());
                assert!(p.devices.is_empty() && p.handles.is_empty());
            }
            BackendState::NotBuilt => panic!("the OpenCL backend is built into this binary"),
        }
    }
}
