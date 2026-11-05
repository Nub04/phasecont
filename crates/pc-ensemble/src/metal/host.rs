//! The Metal host code, built on macOS only. See the parent module for the design.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSProcessInfo, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use pc_codegen::kernel::{Dialect, Scalar};

use super::{ENTRY, command_error, group_for};
use crate::abi::{Packed, integrator_of, pack};
use crate::device::{Backend, BackendState, DeviceInfo};
use crate::gpu::{check_alloc, drive, global_size, guard, kernel_err};
use crate::job::{Checked, EnsembleError, EnsembleResult, Job, Label, Precision, validate};
use crate::kernel::kernel_source;
use crate::tape::Tape;

type Dev = ProtocolObject<dyn MTLDevice>;
type Buf = ProtocolObject<dyn MTLBuffer>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

// ---------------------------------------------------------------------------------------
// The probe

/// What a run needs to know about a device besides what [`DeviceInfo`] holds.
#[derive(Clone, Copy, Debug)]
struct Handle {
    /// The registry id, which names the device across calls to `MTLCopyAllDevices`.
    registry_id: u64,
    /// Largest single buffer, in bytes.
    max_alloc: u64,
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

fn probe_now() -> Probe {
    autoreleasepool(|_| {
        let os = NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string();
        let mut devices = Vec::new();
        let mut handles = Vec::new();
        for dev in MTLCopyAllDevices().iter() {
            let memory = dev.recommendedMaxWorkingSetSize();
            devices.push(DeviceInfo {
                backend: Backend::Metal,
                index: devices.len(),
                name: dev.name().to_string(),
                driver: format!("Metal, macOS {os}"),
                fp32: true,
                // MSL has no `double`.
                fp64: false,
                memory_bytes: (memory > 0).then_some(memory),
            });
            handles.push(Handle {
                registry_id: dev.registryID(),
                max_alloc: dev.maxBufferLength() as u64,
            });
        }
        if devices.is_empty() {
            return Probe::unavailable("Metal lists no device (a virtual machine can have none)");
        }
        Probe {
            state: BackendState::Available,
            devices,
            handles,
        }
    })
}

/// Whether the Metal backend can run in this process.
pub fn backend_state() -> BackendState {
    probe().state.clone()
}

/// The Metal devices, empty when the backend is not available.
pub fn devices() -> Vec<DeviceInfo> {
    probe().devices.clone()
}

// ---------------------------------------------------------------------------------------
// Running

/// Runs `job` on the Metal device `device`, in single precision.
///
/// A missing device, a kernel the driver rejects, a failed allocation or launch and a
/// watchdog stop each come back as their own [`EnsembleError`]. A request for double
/// precision is [`EnsembleError::GpuUnavailable`], because Metal has no `double`. A panic in
/// Rust comes back as [`EnsembleError::GpuLaunch`]. The cancel flag is read between launches.
pub fn run_metal(
    device: &DeviceInfo,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    let checked = validate(job)?;
    let p = probe();
    if let BackendState::Unavailable(why) = &p.state {
        return Err(EnsembleError::GpuUnavailable(why.clone()));
    }
    if device.backend != Backend::Metal || device.index >= p.devices.len() {
        return Err(EnsembleError::GpuUnavailable(format!(
            "no Metal device {}: the system lists {}",
            device.index,
            p.devices.len()
        )));
    }
    // The probe is the truth about the device, not the copy the caller holds.
    let info = &p.devices[device.index];
    let handle = p.handles[device.index];
    if job.precision == Precision::Double {
        return Err(EnsembleError::GpuUnavailable(format!(
            "{} cannot run double precision: Metal has no double type, run it in single \
             precision",
            info.name
        )));
    }
    let members = job.members;
    if members > u32::MAX as usize {
        return Err(EnsembleError::BadJob(format!(
            "{members} members is more than a Metal kernel can index (4294967295)"
        )));
    }
    let tape = Tape::build(job.model)?;
    let src = kernel_source(&tape, integrator_of(job.method), Scalar::F32, Dialect::Msl)
        .map_err(kernel_err)?;
    let label = Label {
        backend: Backend::Metal.name().into(),
        device: info.name.clone(),
        driver: info.driver.clone(),
        precision: job.precision,
    };
    guard(|| session(handle, info, label, job, &checked, &tape, &src, cancel)).unwrap_or_else(
        |msg| {
            Err(EnsembleError::GpuLaunch(format!(
                "panic in Metal host code: {msg}"
            )))
        },
    )
}

/// The device with registry id `id`, if the system still lists it.
fn device_by_id(id: u64) -> Option<Retained<Dev>> {
    MTLCopyAllDevices().iter().find(|d| d.registryID() == id)
}

/// The text of an `NSError`.
fn error_text(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

/// Compiles `src` and returns the pipeline of the entry point.
fn compile(dev: &Dev, src: &str) -> Result<Retained<Pipeline>, EnsembleError> {
    let options = MTLCompileOptions::new();
    // Fast math is the default before macOS 15. Off means safe math on every version, and
    // the pragmas in the kernel text say the same where the compiler reads them.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = dev
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&options))
        .map_err(|e| EnsembleError::GpuCompile(error_text(&e)))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(ENTRY))
        .ok_or_else(|| {
            EnsembleError::GpuCompile(format!("the compiled library has no function {ENTRY}"))
        })?;
    dev.newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| EnsembleError::GpuCompile(error_text(&e)))
}

#[allow(clippy::too_many_arguments)]
fn session(
    handle: Handle,
    info: &DeviceInfo,
    label: Label,
    job: &Job,
    checked: &Checked,
    tape: &Tape,
    src: &str,
    cancel: &AtomicBool,
) -> Result<EnsembleResult, EnsembleError> {
    autoreleasepool(|_| {
        let dev = device_by_id(handle.registry_id).ok_or_else(|| {
            EnsembleError::GpuUnavailable(format!("{} is no longer listed by Metal", info.name))
        })?;
        let queue = dev.newCommandQueue().ok_or_else(|| {
            EnsembleError::GpuLaunch("create command queue: Metal gave nil".into())
        })?;
        let pipeline = compile(&dev, src)?;
        let group = group_for(
            pipeline.maxTotalThreadsPerThreadgroup(),
            pipeline.threadExecutionWidth(),
        );
        launch_all(
            &dev,
            &queue,
            &pipeline,
            (group, handle.max_alloc, info.memory_bytes),
            pack::<f32>(job, checked, tape),
            cancel,
        )
        .and_then(|packed| packed.finish(label))
    })
}

/// Creates a shared-memory buffer holding a copy of `data`.
fn upload<T>(dev: &Dev, data: &[T], step: &str) -> Result<Retained<Buf>, EnsembleError> {
    if data.is_empty() {
        return Err(EnsembleError::BadJob(format!("{step}: nothing to copy")));
    }
    let bytes = std::mem::size_of_val(data);
    // SAFETY: the pointer addresses `bytes` readable bytes, and Metal only reads them.
    unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::from(data).cast::<c_void>(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| {
        EnsembleError::GpuLaunch(format!("{step}: Metal could not allocate {bytes} bytes"))
    })
}

/// Copies the contents of a shared buffer into `out`. The command that wrote it has finished.
fn download<T: Copy>(buf: &Buf, out: &mut [T], step: &str) -> Result<(), EnsembleError> {
    let bytes = std::mem::size_of_val(out);
    if buf.length() < bytes {
        return Err(EnsembleError::GpuLaunch(format!(
            "{step}: the buffer holds {} bytes and {bytes} were asked for",
            buf.length()
        )));
    }
    // SAFETY: shared storage is addressable by the CPU, the buffer holds at least `bytes`
    // bytes, `T` is a plain number type, and the buffer is not in use by the GPU.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr().cast::<T>(),
            out.as_mut_ptr(),
            out.len(),
        );
    }
    Ok(())
}

/// Passes a small value to the kernel as a `constant` argument at `index`.
fn set_value<T>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, value: &T, index: usize) {
    // SAFETY: the pointer addresses `size_of::<T>()` bytes, which Metal copies before the call
    // returns, and the caller passes the type the kernel declares at `index`.
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::from(value).cast::<c_void>(),
            std::mem::size_of::<T>(),
            index,
        );
    }
}

/// Uploads the buffers, launches until no member is running and downloads the result.
fn launch_all(
    dev: &Dev,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &Pipeline,
    (group, max_alloc, memory): (usize, u64, Option<u64>),
    mut packed: Packed<f32>,
    cancel: &AtomicBool,
) -> Result<Packed<f32>, EnsembleError> {
    let size = |len: usize, each: usize| (len * each) as u64;
    check_alloc(
        &[
            ("rc", size(packed.rc.len(), 4)),
            ("segm", size(packed.segm.len(), 8)),
            ("x0", size(packed.x0.len(), 4)),
            ("par", size(packed.par.len(), 4)),
            ("work", size(packed.work.len(), 4)),
            ("out", size(packed.out.len(), 4)),
            ("ctl", size(packed.ctl.len(), 4)),
        ],
        max_alloc,
        memory,
    )?;
    let rc = upload(dev, &packed.rc, "copy rc to device")?;
    let segm = upload(dev, &packed.segm, "copy segm to device")?;
    let x0 = upload(dev, &packed.x0, "copy x0 to device")?;
    let par = upload(dev, &packed.par, "copy par to device")?;
    let work = upload(dev, &packed.work, "copy work to device")?;
    let out = upload(dev, &packed.out, "copy out to device")?;
    let ctl_d = upload(dev, &packed.ctl, "copy ctl to device")?;
    let buffers: [&Buf; 7] = [&rc, &segm, &x0, &par, &work, &out, &ctl_d];

    let members = packed.members as u64;
    let max_steps: u32 = packed.max_steps;
    let ns = packed.ns as u32;
    let pstride = packed.pstride as u64;
    let groups = MTLSize {
        width: global_size(packed.members, group) / group,
        height: 1,
        depth: 1,
    };
    let per_group = MTLSize {
        width: group,
        height: 1,
        depth: 1,
    };
    let ctl_len = packed.ctl.len();
    drive(&mut packed.ctl, packed.members, cancel, |steps: u64| {
        // One pool per launch, so the command buffer and encoder of a long run are freed.
        autoreleasepool(|_| {
            let started = Instant::now();
            let cb = queue.commandBuffer().ok_or_else(|| {
                EnsembleError::GpuLaunch("create command buffer: Metal gave nil".into())
            })?;
            let enc = cb.computeCommandEncoder().ok_or_else(|| {
                EnsembleError::GpuLaunch("create compute encoder: Metal gave nil".into())
            })?;
            enc.setComputePipelineState(pipeline);
            for (i, b) in buffers.iter().enumerate() {
                // SAFETY: the buffers have the sizes `crate::abi` documents and the kernel
                // declares them at indices 0 to 6 in this order.
                unsafe { enc.setBuffer_offset_atIndex(Some(*b), 0, i) };
            }
            set_value(&enc, &members, 7);
            set_value(&enc, &steps, 8);
            set_value(&enc, &max_steps, 9);
            set_value(&enc, &ns, 10);
            set_value(&enc, &pstride, 11);
            enc.dispatchThreadgroups_threadsPerThreadgroup(groups, per_group);
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            let took = started.elapsed();
            if cb.status() == MTLCommandBufferStatus::Error {
                return Err(match cb.error() {
                    Some(e) => command_error(e.code(), &error_text(&e), took),
                    None => EnsembleError::GpuLaunch(
                        "the command buffer finished with an error and no description".into(),
                    ),
                });
            }
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(EnsembleError::GpuLaunch(format!(
                    "the command buffer ended with status {}",
                    cb.status().0
                )));
            }
            let mut ctl = vec![0u32; ctl_len];
            download(&ctl_d, &mut ctl, "copy ctl to host")?;
            Ok((took, ctl))
        })
    })?;
    download(&out, &mut packed.out, "copy out to host")?;
    download(&work, &mut packed.work, "copy work to host")?;
    Ok(packed)
}
