//! The `gpu list` and `gpu selftest` commands.
//!
//! Both print one JSON object to stdout. A missing backend or device prints a message on
//! stderr and nothing on stdout, so a report can never be mistaken for a pass.

use std::process::ExitCode;

use pc_ensemble::Precision;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, list_devices};
use pc_ensemble::selftest::{Report, Selected, SelfTestError, run_selftest};
use serde_json::{Value, json};

use crate::out;

/// Exit code when the backend is not built in or has no such device.
pub const NO_DEVICE: u8 = 3;

pub fn device_json(d: &DeviceInfo) -> Value {
    json!({
        "backend": d.backend.name(),
        "index": d.index,
        "name": d.name,
        "driver": d.driver,
        "fp32": d.fp32,
        "fp64": d.fp64,
        "memory_bytes": d.memory_bytes,
    })
}

fn to_text(v: &Value) -> String {
    let mut text =
        serde_json::to_string_pretty(v).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
    text.push('\n');
    text
}

/// `gpu list`: the state of every backend and the devices that can run a job now.
pub fn list() -> ExitCode {
    let list = list_devices();
    let backends: Vec<Value> = list
        .backends
        .iter()
        .map(|(b, state)| match state {
            BackendState::Available => json!({"backend": b.name(), "state": "available"}),
            BackendState::NotBuilt => json!({"backend": b.name(), "state": "not_built"}),
            BackendState::Unavailable(why) => {
                json!({"backend": b.name(), "state": "unavailable", "why": why})
            }
        })
        .collect();
    let devices: Vec<Value> = list.devices.iter().map(device_json).collect();
    out(
        &to_text(&json!({"backends": backends, "devices": devices})),
        0,
    )
}

/// The report as JSON. `commit` is the git commit the binary was built from when the
/// build says so, and `null` otherwise.
pub fn report_json(r: &Report) -> Value {
    let tiers: Vec<Value> = r
        .tiers
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "max_error": t.max_error,
                "limit": t.limit,
                "pass": t.pass(),
            })
        })
        .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": option_env!("PC_GIT_COMMIT"),
        "backend": r.label.backend,
        "device": {"index": r.device.index, "name": r.device.name},
        "driver": r.label.driver,
        "precision": r.label.precision.name(),
        "single_precision": r.label.single_precision(),
        "features": {"fp32": r.device.fp32, "fp64": r.device.fp64},
        "tiers": tiers,
        "bit_exact": r.bit_exact,
        "pass": r.pass,
    })
}

/// `gpu selftest`: exit 0 when every check passes, 1 when one fails or the device refuses
/// a job, 3 when there is no such device.
pub fn selftest(backend: Backend, device: usize, precision: Precision) -> ExitCode {
    match run_selftest(Selected {
        backend,
        device,
        precision,
    }) {
        Ok(report) => out(&to_text(&report_json(&report)), u8::from(!report.pass)),
        Err(SelfTestError::NoDevice(why)) => {
            eprintln!("pc-cli: {why}");
            ExitCode::from(NO_DEVICE)
        }
        Err(e) => {
            eprintln!("pc-cli: {e}");
            ExitCode::from(1)
        }
    }
}
