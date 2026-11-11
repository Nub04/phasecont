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

use pc_ensemble::device::{DeviceInfo, find_device, list_devices};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleError, EnsembleResult, Job, Method, Output, Params, Precision};
use pc_ir::IrModel;
use pckern::json::ensemble::{
    HEAD_KEYS, Head, Parsed, numbers, object, only_keys, parse_head, required, rows,
};
use serde_json::{Value, json};

use crate::args::BENCH_CASES;
use crate::gpu::NO_DEVICE;
use crate::ircmd::{Prepared, prepare};
use crate::out;

/// Exit code for a bad job file.
const BAD_JOB: u8 = 2;

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
    head: Head,
    x0: Vec<Vec<f64>>,
    params: ParamsFile,
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

/// The keys of a job file: the shared head plus the two arrays.
fn file_keys() -> Vec<&'static str> {
    let mut keys = HEAD_KEYS.to_vec();
    keys.extend(["x0", "params"]);
    keys
}

fn parse_job(value: &Value) -> Parsed<JobFile> {
    let m = object(value, "the job file")?;
    only_keys(m, &file_keys(), "")?;
    Ok(JobFile {
        head: parse_head(m)?,
        x0: rows(required(m, "x0", "")?, "x0")?,
        params: params(required(m, "params", "")?)?,
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
    let device = match find_device(spec.head.backend, spec.head.device) {
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
        t0: spec.head.t0,
        t1: spec.head.t1,
        output: spec.head.output,
        method: spec.head.method,
        bounds: spec.head.bounds,
        max_steps: spec.head.max_steps,
        precision: spec.head.precision,
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

const VDP_MODEL: &str = "par mu=1.5\nx'=y\ny'=mu*(1-x*x)*y-x\ndone\n";
const REPEATS: usize = 3;

/// One case of `ensemble --bench`. Every run of a case times the same work, since the
/// starting points and parameters follow a fixed pattern.
struct BenchCase {
    name: &'static str,
    title: String,
    ir: IrModel,
    kind: CaseKind,
}

#[derive(Clone, Copy)]
enum CaseKind {
    /// van der Pol, mu = 1.5, RK4 with dt 0.01 from t = 0 to t = 10.
    VdpRk4,
    /// van der Pol with mu from 0.1 to 10 in a shuffled order, so members need different
    /// step counts. DP5 with rtol 1e-6 and atol 1e-9 from t = 0 to t = 20.
    VdpDp5,
    /// Lorenz-96 with `n` states and F = 8, RK4 with dt 0.01 from t = 0 to t = 10. The
    /// larger state shows register pressure in the GPU kernels.
    L96(usize),
}

fn lower_text(file: &str, text: &str) -> IrModel {
    use pc_ode::{MapLoader, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add(file, text);
    let loaded = load(top, &mut map, &mut MapLoader::new());
    pc_ir::lower(&loaded.expanded, &loaded.info).expect("a benchmark model lowers")
}

/// Our own Lorenz-96 model: x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, indexes mod n.
fn l96_text(n: usize) -> String {
    let mut text = String::from("par force=8\n");
    for i in 0..n {
        let (next, back2, back1) = ((i + 1) % n, (i + n - 2) % n, (i + n - 1) % n);
        text.push_str(&format!("x{i}'=(x{next}-x{back2})*x{back1}-x{i}+force\n"));
    }
    text.push_str("done\n");
    text
}

fn bench_case(name: &'static str) -> BenchCase {
    let (title, ir, kind) = match name {
        "vdp-rk4" => (
            "van der Pol (mu = 1.5), RK4 with dt 0.01 from t = 0 to t = 10".to_string(),
            lower_text("vdp.ode", VDP_MODEL),
            CaseKind::VdpRk4,
        ),
        "vdp-dp5" => (
            "van der Pol (mu from 0.1 to 10 per member), DP5 with rtol 1e-6 and atol 1e-9 \
             from t = 0 to t = 20"
                .to_string(),
            lower_text("vdp.ode", VDP_MODEL),
            CaseKind::VdpDp5,
        ),
        "l96-8" | "l96-40" => {
            let n = if name == "l96-8" { 8 } else { 40 };
            (
                format!("Lorenz-96 ({n} states, F = 8), RK4 with dt 0.01 from t = 0 to t = 10"),
                lower_text("l96.ode", &l96_text(n)),
                CaseKind::L96(n),
            )
        }
        _ => unreachable!("the argument parser only passes names from BENCH_CASES"),
    };
    BenchCase {
        name,
        title,
        ir,
        kind,
    }
}

/// van der Pol starting points: x from -2 to 2, y from -1 to 1 in a shuffled order.
fn vdp_x0(members: usize) -> Vec<f64> {
    let mut x0 = Vec::with_capacity(2 * members);
    for k in 0..members {
        let u = k as f64 / members as f64;
        let v = ((k * 7) % members) as f64 / members as f64;
        x0.push(-2.0 + 4.0 * u);
        x0.push(-1.0 + 2.0 * v);
    }
    x0
}

/// mu from 0.1 to 10 in a shuffled order, one value per member.
fn vdp_mu(members: usize) -> Vec<f64> {
    (0..members)
        .map(|k| 0.1 + 9.9 * ((k as u64 * 7919) % members as u64) as f64 / members as f64)
        .collect()
}

/// Lorenz-96 starting points within 0.25 of the equilibrium x_i = 8.
fn l96_x0(n: usize, members: usize) -> Vec<f64> {
    let mut x0 = Vec::with_capacity(n * members);
    for k in 0..members {
        for i in 0..n {
            let u = ((7 * k + 31 * i) % 97) as f64 / 97.0;
            x0.push(8.0 + 0.5 * (u - 0.5));
        }
    }
    x0
}

fn bench_row(case: &BenchCase, device: &DeviceInfo, precision: Precision, members: usize) -> Value {
    let head = |extra: Value| {
        let mut row = json!({
            "case": case.name,
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
    let rk4 = Method::Rk4 { dt: 0.01 };
    let (x0, params, per_member, t1, method) = match case.kind {
        CaseKind::VdpRk4 => (vdp_x0(members), vec![1.5], false, 10.0, rk4),
        CaseKind::VdpDp5 => (
            vdp_x0(members),
            vdp_mu(members),
            true,
            20.0,
            Method::Dp5 {
                rtol: 1e-6,
                atol: 1e-9,
                hmax: f64::INFINITY,
            },
        ),
        CaseKind::L96(n) => (l96_x0(n, members), vec![8.0], false, 10.0, rk4),
    };
    let job = Job {
        model: &case.ir,
        members,
        x0: &x0,
        params: if per_member {
            Params::PerMember(&params)
        } else {
            Params::Shared(&params)
        },
        t0: 0.0,
        t1,
        output: Output::Final,
        method,
        bounds: f64::INFINITY,
        max_steps: 1_000_000,
        precision,
    };
    let cancel = AtomicBool::new(false);
    let mut best = f64::INFINITY;
    let mut first = f64::NAN;
    let mut steps = 0u64;
    let mut bits = String::new();
    for repeat in 0..REPEATS {
        let start = Instant::now();
        let result = run_on(device, &job, &cancel);
        let seconds = start.elapsed().as_secs_f64();
        match result {
            Ok(r) => {
                let h = result_bits(&r);
                if repeat == 0 {
                    first = seconds;
                    bits = h;
                } else if h != bits {
                    return head(json!({"error": "two repeats gave different bits"}));
                }
                best = best.min(seconds);
                steps = r.accepted.iter().map(|&a| u64::from(a)).sum();
            }
            Err(e) => return head(json!({"error": e.to_string()})),
        }
    }
    head(json!({
        "seconds": best,
        "first_seconds": first,
        "member_steps": steps,
        "member_steps_per_second": steps as f64 / best,
        "bits": bits,
    }))
}

/// FNV-1a over the bits of every state, status and step count of a result, as 16 hex
/// digits. Two builds that give the same hash for a row gave the same result.
fn result_bits(r: &EnsembleResult) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for v in &r.states {
        eat(&v.to_bits().to_le_bytes());
    }
    for m in 0..r.members {
        eat(&[r.status[m] as u8]);
        eat(&r.accepted[m].to_le_bytes());
        eat(&r.rejected[m].to_le_bytes());
        eat(&r.t_reached[m].to_bits().to_le_bytes());
    }
    format!("{h:016x}")
}

/// `ensemble --bench`: times each case on every device, in both precisions, at each size.
pub fn bench(sizes: &[usize], names: &[String]) -> ExitCode {
    let cases: Vec<BenchCase> = names
        .iter()
        .map(|name| {
            let known = BENCH_CASES
                .into_iter()
                .find(|c| c == name)
                .expect("the argument parser only passes names from BENCH_CASES");
            bench_case(known)
        })
        .collect();
    let devices = list_devices().devices;
    let mut rows = Vec::new();
    for case in &cases {
        for device in &devices {
            for precision in [Precision::Single, Precision::Double] {
                let skipped = if precision == Precision::Double && !device.fp64 {
                    Some("the device has no double precision")
                } else if precision == Precision::Single && matches!(case.kind, CaseKind::VdpDp5) {
                    // The engine refuses rtol 1e-6 in single precision, and a looser
                    // tolerance there would time a different job.
                    Some(
                        "the case runs in double precision only: rtol 1e-6 is below the floor of single precision",
                    )
                } else {
                    None
                };
                if let Some(why) = skipped {
                    rows.push(json!({
                        "case": case.name,
                        "backend": device.backend.name(),
                        "device": device.name,
                        "device_index": device.index,
                        "precision": precision.name(),
                        "skipped": why,
                    }));
                    continue;
                }
                for &members in sizes {
                    rows.push(bench_row(case, device, precision, members));
                }
            }
        }
    }
    let listed: Vec<Value> = cases
        .iter()
        .map(|c| json!({"name": c.name, "title": c.title}))
        .collect();
    let report = json!({
        "cases": listed,
        "repeats": REPEATS,
        "seconds": "the fastest of the repeats, including the host work around the launches",
        "first_seconds": "the first repeat, which also pays for the GPU context and the kernel compile when no earlier run in this process made them",
        "bits": "FNV-1a hash of the bits of every state, status, step count and end time; every repeat gives the same hash or the row is an error",
        "rows": rows,
    });
    out(&to_text(&report), 0)
}
