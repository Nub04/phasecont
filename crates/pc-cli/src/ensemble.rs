//! The `ensemble` command: run many trajectories of one model from a job file, or time the
//! engine with `--bench`.
//!
//! The job file is read by hand from a `serde_json::Value` so that every error can name
//! the key that is wrong. Nothing is filled in for a missing key, except that `device`
//! is the first device of the backend when it is left out and the result names the device
//! that ran. Unknown keys are errors.

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use pc_ensemble::device::{Backend, DeviceInfo, find_device, list_devices};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleError, EnsembleResult, Job, Method, Output, Params, Precision};
use pc_ir::IrModel;
use serde_json::{Map, Value, json};

use crate::args::precision_from_name;
use crate::gpu::NO_DEVICE;
use crate::ircmd::{Prepared, prepare};
use crate::out;

/// Exit code for a bad job file.
const BAD_JOB: u8 = 2;

const KEYS: [&str; 11] = [
    "backend",
    "device",
    "precision",
    "t0",
    "t1",
    "method",
    "x0",
    "params",
    "output",
    "bounds",
    "max_steps",
];

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

fn to_text(v: &Value) -> String {
    let mut text =
        serde_json::to_string_pretty(v).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
    text.push('\n');
    text
}

// ---------------------------------------------------------------------------------------
// The job file.
// ---------------------------------------------------------------------------------------

enum ParamsFile {
    Shared(Vec<f64>),
    PerMember(Vec<Vec<f64>>),
}

/// A job file after the keys and types are checked. The sizes are checked against the
/// model later.
struct JobFile {
    backend: Backend,
    device: usize,
    precision: Precision,
    t0: f64,
    t1: f64,
    method: Method,
    x0: Vec<Vec<f64>>,
    params: ParamsFile,
    output: Output,
    bounds: f64,
    max_steps: u32,
}

type Parsed<T> = Result<T, String>;

fn object<'a>(v: &'a Value, key: &str) -> Parsed<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| format!("{key} must be an object"))
}

fn required<'a>(m: &'a Map<String, Value>, key: &str, path: &str) -> Parsed<&'a Value> {
    m.get(key).ok_or_else(|| format!("{path}missing key {key}"))
}

fn only_keys(m: &Map<String, Value>, allowed: &[&str], path: &str) -> Parsed<()> {
    match m.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(format!("{path}unknown key {k}")),
        None => Ok(()),
    }
}

fn number(v: &Value, key: &str) -> Parsed<f64> {
    v.as_f64()
        .ok_or_else(|| format!("{key} must be a number, got {v}"))
}

fn numbers(v: &Value, key: &str) -> Parsed<Vec<f64>> {
    let items = v
        .as_array()
        .ok_or_else(|| format!("{key} must be an array of numbers"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, x)| number(x, &format!("{key}[{i}]")))
        .collect()
}

fn rows(v: &Value, key: &str) -> Parsed<Vec<Vec<f64>>> {
    let items = v
        .as_array()
        .ok_or_else(|| format!("{key} must be an array of arrays of numbers"))?;
    items
        .iter()
        .enumerate()
        .map(|(i, r)| numbers(r, &format!("{key}[{i}]")))
        .collect()
}

fn string<'a>(v: &'a Value, key: &str) -> Parsed<&'a str> {
    v.as_str()
        .ok_or_else(|| format!("{key} must be a string, got {v}"))
}

/// A positive number, or `null` for no limit.
fn limit(v: &Value, key: &str) -> Parsed<f64> {
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

fn method(v: &Value) -> Parsed<Method> {
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

fn params(v: &Value) -> Parsed<ParamsFile> {
    let m = object(v, "params")?;
    only_keys(m, &["shared", "per_member"], "params: ")?;
    match (m.get("shared"), m.get("per_member")) {
        (Some(s), None) => Ok(ParamsFile::Shared(numbers(s, "params.shared")?)),
        (None, Some(p)) => Ok(ParamsFile::PerMember(rows(p, "params.per_member")?)),
        (Some(_), Some(_)) => Err("params has both shared and per_member, give one".to_string()),
        (None, None) => Err("params needs shared or per_member".to_string()),
    }
}

fn output(v: &Value) -> Parsed<Output> {
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

fn parse_job(value: &Value) -> Parsed<JobFile> {
    let m = object(value, "the job file")?;
    only_keys(m, &KEYS, "")?;
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
    Ok(JobFile {
        backend,
        device: m.get("device").map_or(Ok(0), |v| index(v, "device"))?,
        precision,
        t0: number(get("t0")?, "t0")?,
        t1: number(get("t1")?, "t1")?,
        method: method(get("method")?)?,
        x0: rows(get("x0")?, "x0")?,
        params: params(get("params")?)?,
        output: output(get("output")?)?,
        bounds: limit(get("bounds")?, "bounds")?,
        max_steps,
    })
}

/// Checks the sizes of `x0` and `params` against the model and spells them out in the
/// member-major layout the engine takes.
fn flatten(file: &JobFile, ir: &IrModel) -> Parsed<(Vec<f64>, Vec<f64>)> {
    let (n, n_p) = (ir.n(), ir.n_p());
    if file.x0.is_empty() {
        return Err("x0 has no members, it needs at least one array of states".to_string());
    }
    for (i, row) in file.x0.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "x0[{i}] has {} values, the model has {n} states",
                row.len()
            ));
        }
    }
    let params = match &file.params {
        ParamsFile::Shared(p) => {
            if p.len() != n_p {
                return Err(format!(
                    "params.shared has {} values, the model has {n_p} parameters",
                    p.len()
                ));
            }
            p.clone()
        }
        ParamsFile::PerMember(rows) => {
            if rows.len() != file.x0.len() {
                return Err(format!(
                    "params.per_member has {} rows, x0 has {} members",
                    rows.len(),
                    file.x0.len()
                ));
            }
            for (i, row) in rows.iter().enumerate() {
                if row.len() != n_p {
                    return Err(format!(
                        "params.per_member[{i}] has {} values, the model has {n_p} parameters",
                        row.len()
                    ));
                }
            }
            rows.concat()
        }
    };
    Ok((file.x0.concat(), params))
}

// ---------------------------------------------------------------------------------------
// Running a job.
// ---------------------------------------------------------------------------------------

fn bad_job(path: &Path, why: &str) -> ExitCode {
    to_stderr(&format!("pc-cli: job file {}: {why}\n", path.display()));
    ExitCode::from(BAD_JOB)
}

/// `ensemble`: runs the job and prints the result. See `ENSEMBLE_HELP` for the exit codes.
pub fn ensemble(file: &Path, job_path: &Path) -> ExitCode {
    let text = match std::fs::read_to_string(job_path) {
        Ok(t) => t,
        Err(e) => return bad_job(job_path, &format!("cannot read it: {e}")),
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return bad_job(job_path, &format!("invalid JSON: {e}")),
    };
    let spec = match parse_job(&value) {
        Ok(s) => s,
        Err(why) => return bad_job(job_path, &why),
    };
    let Prepared { ir, .. } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let (x0, params) = match flatten(&spec, &ir) {
        Ok(v) => v,
        Err(why) => return bad_job(job_path, &why),
    };
    let device = match find_device(spec.backend, spec.device) {
        Ok(d) => d,
        Err(why) => {
            to_stderr(&format!("pc-cli: {why}\n"));
            return ExitCode::from(NO_DEVICE);
        }
    };
    let members = spec.x0.len();
    let job = Job {
        model: &ir,
        members,
        x0: &x0,
        params: match spec.params {
            ParamsFile::Shared(_) => Params::Shared(&params),
            ParamsFile::PerMember(_) => Params::PerMember(&params),
        },
        t0: spec.t0,
        t1: spec.t1,
        output: spec.output,
        method: spec.method,
        bounds: spec.bounds,
        max_steps: spec.max_steps,
        precision: spec.precision,
    };
    match run_on(&device, &job, &AtomicBool::new(false)) {
        Ok(result) => out(&to_text(&result_json(&result)), 0),
        Err(EnsembleError::BadJob(why)) => bad_job(job_path, &why),
        Err(e) => {
            to_stderr(&format!("pc-cli: {e}\n"));
            ExitCode::from(1)
        }
    }
}

fn result_json(r: &EnsembleResult) -> Value {
    // A value that is not finite prints as null: serde_json has no spelling for it.
    let states: Vec<Value> = (0..r.members)
        .map(|m| Value::Array((0..r.times.len()).map(|s| json!(r.state(m, s))).collect()))
        .collect();
    let status: Vec<&str> = r.status.iter().map(|s| s.name()).collect();
    json!({
        "members": r.members,
        "n": r.n,
        "times": r.times,
        "status": status,
        "states": states,
        "t_reached": r.t_reached,
        "accepted": r.accepted,
        "rejected": r.rejected,
        "label": {
            "backend": r.label.backend,
            "device": r.label.device,
            "driver": r.label.driver,
            "precision": r.label.precision.name(),
            "single_precision": r.label.single_precision(),
        },
    })
}

// ---------------------------------------------------------------------------------------
// The benchmark.
// ---------------------------------------------------------------------------------------

const BENCH_MODEL: &str = "par mu=1.5\nx'=y\ny'=mu*(1-x*x)*y-x\ndone\n";
const BENCH_TITLE: &str = "van der Pol (mu = 1.5), RK4 with dt 0.01 from t = 0 to t = 10";
const BENCH_DT: f64 = 0.01;
const BENCH_T1: f64 = 10.0;
const REPEATS: usize = 3;

fn bench_model() -> IrModel {
    use pc_ode::{MapLoader, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("bench.ode", BENCH_MODEL);
    let loaded = load(top, &mut map, &mut MapLoader::new());
    pc_ir::lower(&loaded.expanded, &loaded.info).expect("the benchmark model lowers")
}

/// `members` starting points on a fixed pattern, so every run times the same work.
fn bench_x0(members: usize) -> Vec<f64> {
    let mut x0 = Vec::with_capacity(2 * members);
    for k in 0..members {
        let u = k as f64 / members as f64;
        let v = ((k * 7) % members) as f64 / members as f64;
        x0.push(-2.0 + 4.0 * u);
        x0.push(-1.0 + 2.0 * v);
    }
    x0
}

fn bench_row(ir: &IrModel, device: &DeviceInfo, precision: Precision, members: usize) -> Value {
    let head = |extra: Value| {
        let mut row = json!({
            "backend": device.backend.name(),
            "device": device.name,
            "device_index": device.index,
            "precision": precision.name(),
            "members": members,
        });
        if let (Some(r), Value::Object(e)) = (row.as_object_mut(), extra) {
            r.extend(e);
        }
        row
    };
    let x0 = bench_x0(members);
    let params = [1.5];
    let job = Job {
        model: ir,
        members,
        x0: &x0,
        params: Params::Shared(&params),
        t0: 0.0,
        t1: BENCH_T1,
        output: Output::Final,
        method: Method::Rk4 { dt: BENCH_DT },
        bounds: f64::INFINITY,
        max_steps: 1_000_000,
        precision,
    };
    let cancel = AtomicBool::new(false);
    let mut best = f64::INFINITY;
    let mut steps = 0u64;
    for _ in 0..REPEATS {
        let start = Instant::now();
        let result = run_on(device, &job, &cancel);
        let seconds = start.elapsed().as_secs_f64();
        match result {
            Ok(r) => {
                best = best.min(seconds);
                steps = r.accepted.iter().map(|&a| u64::from(a)).sum();
            }
            Err(e) => return head(json!({"error": e.to_string()})),
        }
    }
    head(json!({
        "seconds": best,
        "member_steps": steps,
        "member_steps_per_second": steps as f64 / best,
    }))
}

/// `ensemble --bench`: times the engine on every device, in both precisions, at each size.
pub fn bench(sizes: &[usize]) -> ExitCode {
    let ir = bench_model();
    let mut rows = Vec::new();
    for device in &list_devices().devices {
        for precision in [Precision::Single, Precision::Double] {
            if precision == Precision::Double && !device.fp64 {
                rows.push(json!({
                    "backend": device.backend.name(),
                    "device": device.name,
                    "device_index": device.index,
                    "precision": precision.name(),
                    "skipped": "the device has no double precision",
                }));
                continue;
            }
            for &members in sizes {
                rows.push(bench_row(&ir, device, precision, members));
            }
        }
    }
    let report = json!({
        "model": BENCH_TITLE,
        "repeats": REPEATS,
        "seconds": "the fastest of the repeats, including the host work around the launches",
        "rows": rows,
    });
    out(&to_text(&report), 0)
}
