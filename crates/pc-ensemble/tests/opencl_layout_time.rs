//! Times the private and the global scratch layout of the OpenCL kernel on every OpenCL device,
//! and records the model sizes that each layout compiles at, to tune the size rule of
//! `layout_by_size`.
//!
//! It is a measurement, not a check, so it is ignored in a normal run. On Windows, with the
//! test executable built for `x86_64-pc-windows-gnu`:
//!
//! ```text
//! PC_REQUIRE_GPU=opencl PC_LT_N=1000,2000,4000 ./opencl_layout_time.exe --ignored --nocapture
//! ```
//!
//! Settings, all optional: `PC_LT_N` (comma list of model sizes), `PC_LT_MODEL` (`ring`, the
//! default, or `l96`), `PC_LT_METHOD` (`rk4`, `dp5`), `PC_LT_PRECISION` (`single`, `double`),
//! `PC_LT_MEMBERS` (default 2048), `PC_LT_STEPS` (RK4 steps, default 40), `PC_LT_REPEATS`
//! (default 3), `PC_LT_LAYOUTS` (default `private,global`) and `PC_OPENCL_DEVICE` (one device
//! index). Every case prints one `LT` line per device and layout with the fastest run, or the
//! first line of the error when the run fails (a compile that exceeds the device's scratch
//! limit, for example), and whether the two layouts returned the same bits. The first run of a
//! case pays for the driver's compile of the kernel, so it is not timed.

mod common;

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::*;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, list_devices};
use pc_ensemble::kernel::{Integrator, Layout, private_reals};
use pc_ensemble::opencl::force_layout;
use pc_ensemble::run::run_on;
use pc_ensemble::{Method, Precision};

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

#[test]
#[ignore = "a timing measurement that needs an OpenCL device and takes minutes"]
fn time_the_private_and_the_global_layout() {
    let list = list_devices();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::OpenCl)
        .map(|(_, s)| s.clone())
        .expect("opencl is listed");
    match state {
        BackendState::Available => {}
        BackendState::Unavailable(why) => return skip_gpu("opencl", &why),
        BackendState::NotBuilt => return skip_gpu("opencl", "not built into this binary"),
    }
    let only = std::env::var("PC_OPENCL_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let devices: Vec<DeviceInfo> = list
        .devices
        .into_iter()
        .filter(|d| d.backend == Backend::OpenCl && only.is_none_or(|i| d.index == i))
        .collect();
    let _reset = Reset;
    let members = number("PC_LT_MEMBERS", 2048);
    let steps = number("PC_LT_STEPS", 40);
    let repeats = number("PC_LT_REPEATS", 3);
    let model_name = setting("PC_LT_MODEL", "ring").remove(0);

    for device in &devices {
        eprintln!(
            "LT device {} {} ({}), memory {:?} bytes, fp64 {}",
            device.index, device.name, device.driver, device.memory_bytes, device.fp64
        );
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
                let (method, integrator, t1) = match (method_name.as_str(), precision) {
                    ("rk4", _) => (
                        Method::Rk4 { dt: 0.001 },
                        Integrator::Rk4,
                        0.001 * steps as f64,
                    ),
                    ("dp5", Precision::Single) => (DP5_SINGLE, Integrator::Dp5, 0.05),
                    ("dp5", Precision::Double) => (DP5_STD, Integrator::Dp5, 0.05),
                    (other, _) => panic!("PC_LT_METHOD: {other}"),
                };
                for n_text in setting("PC_LT_N", "1000,2000,4000") {
                    let n: usize = n_text.parse().expect("PC_LT_N is a list of integers");
                    let (text, params): (String, &[f64]) = match model_name.as_str() {
                        "ring" => (ring(n), &[]),
                        "l96" => (lorenz96(n), &[8.0]),
                        other => panic!("PC_LT_MODEL: {other}"),
                    };
                    let model = build_model(&text);
                    let x0: Vec<f64> = (0..members * n)
                        .map(|k| 8.0 + 0.01 * ((k % n) as f64) + 0.1 * (k / n) as f64)
                        .collect();
                    let j = job(&model, members, &x0, params, t1, method, precision);
                    let real = if precision == Precision::Double { 8 } else { 4 };
                    let kib = private_reals(n, integrator) * real / 1024;
                    let mut first_bits: Option<Vec<u64>> = None;
                    let mut same = true;
                    for layout_name in setting("PC_LT_LAYOUTS", "private,global") {
                        let layout = match layout_name.as_str() {
                            "private" => Layout::Private,
                            "global" => Layout::Global,
                            other => panic!("PC_LT_LAYOUTS: {other}"),
                        };
                        force_layout(Some(layout));
                        let mut best = f64::INFINITY;
                        let mut last = None;
                        let mut failure = None;
                        for round in 0..=repeats {
                            let t = Instant::now();
                            match run_on(device, &j, &AtomicBool::new(false)) {
                                Ok(r) => {
                                    if round > 0 {
                                        best = best.min(t.elapsed().as_secs_f64());
                                    }
                                    last = Some(r);
                                }
                                Err(e) => {
                                    failure = Some(e.to_string());
                                    break;
                                }
                            }
                        }
                        let head = format!(
                            "LT dev={} n={n} {model_name} {method_name} {precision_name} {layout:?} members={members} private_kib={kib}",
                            device.index
                        );
                        match (failure, last) {
                            (Some(e), _) => {
                                let line = e.lines().find(|l| l.contains("error")).unwrap_or(&e);
                                eprintln!("{head} FAILED: {}", line.trim());
                            }
                            (None, Some(r)) => {
                                let b = bits(&r.states);
                                match &first_bits {
                                    None => first_bits = Some(b),
                                    Some(f) => same = *f == b,
                                }
                                eprintln!("{head} best_ms={:.2}", best * 1e3);
                            }
                            (None, None) => unreachable!("a run either fails or returns"),
                        }
                    }
                    force_layout(None);
                    if first_bits.is_some() {
                        eprintln!(
                            "LT dev={} n={n} {model_name} {method_name} {precision_name} same_bits_private_global={same}",
                            device.index
                        );
                    }
                }
            }
        }
    }
}
