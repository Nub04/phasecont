//! Times the private and the global scratch layout of the CUDA kernel on one model, to tune
//! the rule in `layout_for`: the 256 KiB cap, the Dormand-Prince size and the share of the free
//! memory that the private layout's local-memory reserve may take.
//!
//! It is a measurement, not a check, so it is ignored in a normal run:
//!
//! ```text
//! PC_REQUIRE_GPU=cuda PC_LT_N=1600,3277,4900 \
//!     cargo test --release -p pc-ensemble --test cuda_layout_time -- --ignored --nocapture
//! ```
//!
//! Settings, all optional: `PC_LT_N` (comma list of Lorenz-96 sizes), `PC_LT_METHOD`
//! (`rk4`, `dp5`), `PC_LT_PRECISION` (`single`, `double`), `PC_LT_MEMBERS` (default 2048),
//! `PC_LT_STEPS` (RK4 steps, default 40), `PC_LT_REPEATS` (default 3), `PC_LT_LAYOUTS`
//! (`private,global`, in the order to run; the driver keeps the local memory that a private
//! run needed for the rest of the process, so the peak memory of `global` is only clean in a
//! process of its own), `PC_LT_BALLAST_MIB` (device memory held during the run, to imitate a
//! card with less free memory). Every case prints one
//! `LT` line per layout with the fastest run, the mean accepted steps per member, the peak
//! device memory that `nvidia-smi` showed during the case, and whether the two layouts
//! returned the same bits. The first run of a case pays for the driver's compile of the
//! kernel, so it is not timed.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use common::*;
use cudarc::driver::{CudaContext, CudaSlice};
use pc_ensemble::cuda::force_layout;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::kernel::Layout;
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleResult, Method, Precision};

/// Lifts the forced layout, also when the test panics.
struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        force_layout(None);
    }
}

fn setting(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn number(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Takes `PC_LT_BALLAST_MIB` of device memory and keeps it until dropped, to time the layouts
/// on a card with less free memory.
fn ballast(index: usize) -> Option<(Arc<CudaContext>, CudaSlice<u8>)> {
    let mib = number("PC_LT_BALLAST_MIB", 0);
    if mib == 0 {
        return None;
    }
    let ctx = CudaContext::new(index).expect("a context for the ballast");
    let held = ctx
        .default_stream()
        .alloc_zeros::<u8>(mib << 20)
        .expect("the ballast fits");
    let (free, total) = ctx.mem_get_info().expect("memory info");
    eprintln!(
        "LT ballast {mib} MiB, free {} of {} MiB",
        free >> 20,
        total >> 20
    );
    Some((ctx, held))
}

/// The device memory in use now, in MiB, from `nvidia-smi`.
fn used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Polls `nvidia-smi` until dropped and keeps the largest value.
struct Peak {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Peak {
    fn start() -> Peak {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let (s, p) = (stop.clone(), peak.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(mib) = used_mib() {
                    p.fetch_max(mib, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
        Peak {
            stop,
            peak,
            thread: Some(thread),
        }
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

fn run_once(device: &DeviceInfo, job: &pc_ensemble::Job) -> Result<EnsembleResult, String> {
    run_on(device, job, &AtomicBool::new(false)).map_err(|e| e.to_string())
}

#[test]
#[ignore = "a timing measurement that needs a CUDA device and takes minutes"]
fn time_the_private_and_the_global_layout() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    eprintln!(
        "LT device {} ({}), memory {:?} bytes",
        device.name, device.driver, device.memory_bytes
    );
    let _reset = Reset;
    let members = number("PC_LT_MEMBERS", 2048);
    let steps = number("PC_LT_STEPS", 40);
    let repeats = number("PC_LT_REPEATS", 3);
    let _ballast = ballast(device.index);
    let probe = CudaContext::new(device.index).ok();
    if let Some(ctx) = &probe {
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        let sms = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let per_sm = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        let free = ctx.mem_get_info().map(|(f, t)| (f >> 20, t >> 20));
        eprintln!(
            "LT multiprocessors {sms:?}, threads per multiprocessor {per_sm:?}, free and total MiB {free:?}"
        );
    }
    let baseline = used_mib();
    eprintln!("LT idle device memory {baseline:?} MiB, {members} members");

    for precision_name in setting("PC_LT_PRECISION", "single,double") {
        let precision = match precision_name.as_str() {
            "single" => Precision::Single,
            "double" => Precision::Double,
            other => panic!("PC_LT_PRECISION: {other}"),
        };
        if precision == Precision::Double && !device.fp64 {
            continue;
        }
        for method_name in setting("PC_LT_METHOD", "rk4,dp5") {
            let (method, t1) = match (method_name.as_str(), precision) {
                ("rk4", _) => (Method::Rk4 { dt: 0.001 }, 0.001 * steps as f64),
                ("dp5", Precision::Single) => (DP5_SINGLE, 0.05),
                ("dp5", Precision::Double) => (DP5_STD, 0.05),
                (other, _) => panic!("PC_LT_METHOD: {other}"),
            };
            for n_text in setting("PC_LT_N", "1600,3277,4900") {
                let n: usize = n_text.parse().expect("PC_LT_N is a list of integers");
                let model = build_model(&lorenz96(n));
                let x0: Vec<f64> = (0..members * n)
                    .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
                    .collect();
                let j = job(&model, members, &x0, &[8.0], t1, method, precision);
                let mut first_bits: Option<Vec<u64>> = None;
                let mut same = true;
                for layout_name in setting("PC_LT_LAYOUTS", "private,global") {
                    let layout = match layout_name.as_str() {
                        "private" => Layout::Private,
                        "global" => Layout::Global,
                        other => panic!("PC_LT_LAYOUTS: {other}"),
                    };
                    force_layout(Some(layout));
                    let peak = Peak::start();
                    let mut best = f64::INFINITY;
                    let mut mean_steps = 0.0;
                    let mut failure = None;
                    let mut last = None;
                    for round in 0..=repeats {
                        let t = Instant::now();
                        match run_once(&device, &j) {
                            Ok(r) => {
                                let secs = t.elapsed().as_secs_f64();
                                if round > 0 {
                                    best = best.min(secs);
                                } else {
                                    eprintln!("LT   first run {secs:.1} s ({layout:?} n = {n})");
                                }
                                mean_steps = r.accepted.iter().map(|a| *a as f64).sum::<f64>()
                                    / r.accepted.len() as f64;
                                last = Some(r);
                            }
                            Err(e) => {
                                failure = Some(e);
                                break;
                            }
                        }
                    }
                    let peak_mib = peak.finish();
                    let free_after = probe
                        .as_ref()
                        .and_then(|c| c.mem_get_info().ok())
                        .map(|(f, _)| f >> 20);
                    eprintln!("LT   free MiB after the {layout:?} runs: {free_after:?}");
                    match (failure, last) {
                        (Some(e), _) => eprintln!(
                            "LT n={n} {method_name} {precision_name} {layout:?} members={members} FAILED: {e}"
                        ),
                        (None, Some(r)) => {
                            let b = bits(&r.states);
                            match &first_bits {
                                None => first_bits = Some(b),
                                Some(f) => same = *f == b,
                            }
                            let kib = pc_ensemble::kernel::private_reals(
                                n,
                                if method_name == "rk4" {
                                    pc_ensemble::kernel::Integrator::Rk4
                                } else {
                                    pc_ensemble::kernel::Integrator::Dp5
                                },
                            ) * if precision == Precision::Double { 8 } else { 4 }
                                / 1024;
                            eprintln!(
                                "LT n={n} {method_name} {precision_name} {layout:?} members={members} private_kib={kib} best_ms={:.2} steps={mean_steps:.1} peak_mib={peak_mib}",
                                best * 1e3
                            );
                        }
                        (None, None) => unreachable!("a run either fails or returns"),
                    }
                }
                force_layout(None);
                if first_bits.is_some() {
                    eprintln!(
                        "LT n={n} {method_name} {precision_name} same_bits_private_global={same}"
                    );
                }
            }
        }
    }
}
