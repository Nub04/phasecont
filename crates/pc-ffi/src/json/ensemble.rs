//! The ensemble job as JSON, shared by `pc-cli ensemble` and the C ABI.
//!
//! The job is read by hand from a `serde_json::Value` so that every error can name the key
//! that is wrong. Nothing is filled in for a missing key, except that `device` is the first
//! device of the backend when it is left out, and the result names the device that ran.

use pc_ensemble::device::{Backend, BackendState, DeviceInfo, DeviceList};
use pc_ensemble::{Method, Output, Precision};
use serde_json::{Map, Value, json};

pub type Parsed<T> = Result<T, String>;

/// Everything in a job except the states and the parameters, which travel as arrays.
#[derive(Clone, Debug)]
pub struct Head {
    pub backend: Backend,
    pub device: usize,
    pub precision: Precision,
    pub t0: f64,
    pub t1: f64,
    pub method: Method,
    pub output: Output,
    pub bounds: f64,
    pub max_steps: u32,
}

/// The keys `parse_head` reads.
pub const HEAD_KEYS: [&str; 9] = [
    "backend",
    "device",
    "precision",
    "t0",
    "t1",
    "method",
    "output",
    "bounds",
    "max_steps",
];

pub fn object<'a>(v: &'a Value, key: &str) -> Parsed<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| format!("{key} must be an object"))
}

pub fn required<'a>(m: &'a Map<String, Value>, key: &str, path: &str) -> Parsed<&'a Value> {
    m.get(key).ok_or_else(|| format!("{path}missing key {key}"))
}

pub fn only_keys(m: &Map<String, Value>, allowed: &[&str], path: &str) -> Parsed<()> {
    match m.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!("{path}unknown key {k}")),
        None => Ok(()),
    }
}

pub fn number(v: &Value, key: &str) -> Parsed<f64> {
    v.as_f64()
        .ok_or_else(|| format!("{key} must be a number, got {v}"))
}

pub fn numbers(v: &Value, key: &str) -> Parsed<Vec<f64>> {
    let items = v
        .as_array()
        .ok_or_else(|| format!("{key} must be an array of numbers"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, x)| number(x, &format!("{key}[{i}]")))
        .collect()
}

pub fn rows(v: &Value, key: &str) -> Parsed<Vec<Vec<f64>>> {
    let items = v
        .as_array()
        .ok_or_else(|| format!("{key} must be an array of arrays of numbers"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, r)| numbers(r, &format!("{key}[{i}]")))
        .collect()
}

pub fn string<'a>(v: &'a Value, key: &str) -> Parsed<&'a str> {
    v.as_str()
        .ok_or_else(|| format!("{key} must be a string, got {v}"))
}

/// A positive number, or `null` for no limit.
pub fn limit(v: &Value, key: &str) -> Parsed<f64> {
    if v.is_null() {
        return Ok(f64::INFINITY);
    }
    let x = number(v, key)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(format!("{key} must be positive or null, got {x}"))
    }
}

/// `single` or `double`.
pub fn precision_from_name(name: &str) -> Option<Precision> {
    [Precision::Single, Precision::Double]
        .into_iter()
        .find(|p| p.name() == name)
}

pub fn method(v: &Value) -> Parsed<Method> {
    let m = object(v, "method")?;
    let kind = string(required(m, "type", "method: ")?, "method.type")?;
    let field = |key: &str| required(m, key, "method: ");
    match kind {
        "rk4" => {
            only_keys(m, &["type", "dt"], "method: ")?;
            let dt = number(field("dt")?, "method.dt")?;
            if dt > 0.0 && dt.is_finite() {
                Ok(Method::Rk4 { dt })
            } else {
                Err(format!("method.dt must be positive and finite, got {dt}"))
            }
        }
        "dp5" => {
            only_keys(m, &["type", "rtol", "atol", "hmax"], "method: ")?;
            let rtol = number(field("rtol")?, "method.rtol")?;
            let atol = number(field("atol")?, "method.atol")?;
            let hmax = limit(field("hmax")?, "method.hmax")?;
            for (name, x) in [("rtol", rtol), ("atol", atol)] {
                if !(x > 0.0 && x.is_finite()) {
                    return Err(format!(
                        "method.{name} must be positive and finite, got {x}"
                    ));
                }
            }
            Ok(Method::Dp5 { rtol, atol, hmax })
        }
        other => Err(format!(
            "method.type must be rk4 or dp5, got {other}; method is {{\"type\": \"rk4\", \"dt\": X}} or {{\"type\": \"dp5\", \"rtol\": X, \"atol\": X, \"hmax\": X or null}}"
        )),
    }
}

pub fn output(v: &Value) -> Parsed<Output> {
    match v {
        Value::String(s) if s == "final" => Ok(Output::Final),
        Value::Object(m) => {
            only_keys(m, &["samples"], "output: ")?;
            let times = numbers(required(m, "samples", "output: ")?, "output.samples")?;
            Ok(Output::Samples(times))
        }
        _ => Err(format!(
            "output must be \"final\" or {{\"samples\": [t, ..]}}, got {v}"
        )),
    }
}

fn index(v: &Value, key: &str) -> Parsed<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("{key} must be a non-negative integer, got {v}"))
}

/// Reads the keys of [`HEAD_KEYS`] from `m`. The caller has already checked that `m` has no
/// key it does not know.
pub fn parse_head(m: &Map<String, Value>) -> Parsed<Head> {
    let get = |key: &str| required(m, key, "");
    let backend_name = string(get("backend")?, "backend")?;
    let backend = Backend::from_name(backend_name)
        .ok_or_else(|| format!("backend must be cpu, cuda, opencl or metal, got {backend_name}"))?;
    let precision_name = string(get("precision")?, "precision")?;
    let precision = precision_from_name(precision_name)
        .ok_or_else(|| format!("precision must be single or double, got {precision_name}"))?;
    let max_steps = get("max_steps")?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| {
            format!(
                "max_steps must be an integer from 1 to {}, got {}",
                u32::MAX,
                m["max_steps"]
            )
        })?;
    Ok(Head {
        backend,
        device: m.get("device").map_or(Ok(0), |v| index(v, "device"))?,
        precision,
        t0: number(get("t0")?, "t0")?,
        t1: number(get("t1")?, "t1")?,
        method: method(get("method")?)?,
        output: output(get("output")?)?,
        bounds: limit(get("bounds")?, "bounds")?,
        max_steps,
    })
}

/// One device as `gpu list` and `pc_ens_devices` print it.
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

/// The state of every backend and the devices that can run a job now.
pub fn devices_json(list: &DeviceList) -> Value {
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
    json!({"backends": backends, "devices": devices})
}
