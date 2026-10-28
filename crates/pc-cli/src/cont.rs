//! The `cont` command: follow an equilibrium branch described by a problem file.
//!
//! The problem file is read by hand from a `serde_json::Value`, so that every error can
//! name the key that is wrong. Unknown keys are errors. The run goes through
//! `pc_engine::Run`; the result goes to stdout as one JSON object and the special points
//! go to stderr, one line each.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_engine::{
    EngineError, EpSetup, Evaluator, Options, Run, RunSpec, RunState, SpecialPoint, StopReason,
};
use pc_ir::IrModel;
use serde_json::{Map, Value, json};

use crate::ircmd::{Prepared, prepare};
use crate::out;

/// A problem file after validation.
#[derive(Debug)]
struct Problem {
    setup: EpSetup,
    opts: Options,
    switch: Option<String>,
}

/// Exit code for a bad problem file.
const BAD_PROBLEM: u8 = 2;

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

/// `cont`: runs the problem and prints the result. See `CONT_HELP` for the exit codes.
pub fn cont(file: &Path, problem_file: &Path, fallback: bool) -> ExitCode {
    let text = match std::fs::read_to_string(problem_file) {
        Ok(t) => t,
        Err(e) => {
            to_stderr(&format!(
                "pc-cli: cannot read problem file {}: {e}\n",
                problem_file.display()
            ));
            return ExitCode::from(BAD_PROBLEM);
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            to_stderr(&format!(
                "pc-cli: problem file {}: invalid JSON: {e}\n",
                problem_file.display()
            ));
            return ExitCode::from(BAD_PROBLEM);
        }
    };
    let Prepared { loaded, ir } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let problem = match parse_problem(&value, &ir) {
        Ok(p) => p,
        Err(msg) => {
            to_stderr(&format!(
                "pc-cli: problem file {}: {msg}\n",
                problem_file.display()
            ));
            return ExitCode::from(BAD_PROBLEM);
        }
    };
    let evaluator = if fallback {
        Evaluator::Fallback
    } else {
        Evaluator::Jit
    };
    let mut spec = RunSpec::new(
        loaded.expanded,
        loaded.info,
        evaluator,
        problem.setup,
        problem.opts,
    );
    spec.switch_at = problem.switch;
    let run = Run::start(spec);
    // `join` would consume the run and with it the branches, so wait by polling. The worker
    // sets the final state last; dropping `run` after the report joins the thread.
    let state = loop {
        let st = run.poll().state;
        if st != RunState::Running {
            break st;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    report(&run, &state)
}

/// Prints the branches and the special points and picks the exit code.
fn report(run: &Run, state: &RunState) -> ExitCode {
    let counts = run.poll().points;
    let mut err = String::new();
    let mut branches = Vec::new();
    for (b, &n) in counts.iter().enumerate() {
        let points: Vec<Value> = run
            .points(b, 0, n)
            .iter()
            .map(|p| {
                json!({
                    "u": p.u, "alpha": p.alpha, "tests": p.tests, "h": p.h,
                    "newton_iters": p.newton_iters,
                })
            })
            .collect();
        let specials = run.specials(b);
        let stop = run.stop(b);
        let _ = writeln!(err, "branch {b}");
        for s in &specials {
            let _ = writeln!(err, "{}", special_line(s));
        }
        if let Some(StopReason::Error(e)) = &stop
            && *e != EngineError::Cancelled
        {
            let _ = writeln!(err, "branch {b} stopped: {e}");
        }
        // A panic leaves the placeholder stop on the branch it cut short.
        let cut_short =
            b + 1 == counts.len() && matches!(state, RunState::Panicked { .. } | RunState::Running);
        let stop_text = if cut_short {
            "error"
        } else {
            stop.as_ref().map_or("none", stop_name)
        };
        branches.push(json!({
            "points": points,
            "specials": specials.iter().map(special_json).collect::<Vec<_>>(),
            "stop": stop_text,
        }));
    }
    let code = match state {
        RunState::Done => 0,
        RunState::Failed(e) => {
            let _ = writeln!(err, "pc-cli: run failed: {e}");
            1
        }
        RunState::Cancelled => {
            let _ = writeln!(err, "pc-cli: run cancelled");
            1
        }
        RunState::Panicked { message, location } => {
            let _ = writeln!(err, "pc-cli: engine panic at {location}: {message}");
            1
        }
        RunState::Running => {
            let _ = writeln!(err, "pc-cli: run did not finish");
            1
        }
    };
    to_stderr(&err);
    let mut text = json!({ "branches": branches }).to_string();
    text.push('\n');
    out(&text, code)
}

/// A stable name for each stop reason.
fn stop_name(s: &StopReason) -> &'static str {
    match s {
        StopReason::MaxSteps => "max_steps",
        StopReason::Bounds => "bounds",
        StopReason::StepTooSmall => "step_too_small",
        StopReason::Singular => "singular",
        StopReason::Cancelled | StopReason::Error(EngineError::Cancelled) => "cancelled",
        StopReason::Error(_) => "error",
    }
}

fn special_json(s: &SpecialPoint) -> Value {
    json!({
        "label": s.label,
        "index": s.index,
        "u": s.point.u,
        "alpha": s.point.alpha,
        "coefficient": s.extras.coefficient,
        "omega": s.extras.omega,
        "normalization": s.extras.normalization,
        "refined": s.extras.refined,
    })
}

/// `%e` as C prints it: six decimals and an exponent with a sign and two digits.
fn fmt_e(x: f64) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    let s = format!("{x:.6e}");
    match s.split_once('e') {
        Some((mant, exp)) => {
            let exp: i32 = exp.parse().unwrap_or(0);
            let sign = if exp < 0 { '-' } else { '+' };
            format!("{mant}e{sign}{:02}", exp.abs())
        }
        None => s,
    }
}

/// One stderr line, for example `LP  index 57  a=  1.000000e+00  free=0.000000e+00`.
/// Fields that the point does not have are left out.
fn special_line(s: &SpecialPoint) -> String {
    let mut line = format!("{:<2}  index {}", s.label, s.index);
    if let Some(c) = s.extras.coefficient {
        let name = match s.label.as_str() {
            "LP" => "a",
            "H" => "l1",
            _ => "coef",
        };
        let _ = write!(line, "  {name}={:>14}", fmt_e(c));
    }
    if let Some(w) = s.extras.omega {
        let _ = write!(line, "  omega= {}", fmt_e(w));
    }
    let _ = write!(line, "  free={}", fmt_e(s.point.alpha));
    line
}

// ---- problem file ----

/// A number that is finite.
fn number(v: &Value, path: &str) -> Result<f64, String> {
    match v.as_f64() {
        Some(x) if x.is_finite() => Ok(x),
        _ => Err(format!("{path}: expected a number")),
    }
}

fn positive(v: &Value, path: &str) -> Result<f64, String> {
    let x = number(v, path)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(format!("{path}: must be greater than 0"))
    }
}

fn count(v: &Value, path: &str) -> Result<usize, String> {
    match v.as_u64().and_then(|n| usize::try_from(n).ok()) {
        Some(n) if n >= 1 => Ok(n),
        _ => Err(format!("{path}: expected an integer of at least 1")),
    }
}

fn object<'a>(v: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    v.as_object()
        .ok_or_else(|| format!("{path}: expected an object"))
}

/// Errors on the first key of `map` that is not in `allowed`.
fn only_keys(map: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<(), String> {
    for k in map.keys() {
        if !allowed.contains(&k.as_str()) {
            let at = if path.is_empty() {
                k.clone()
            } else {
                format!("{path}.{k}")
            };
            return Err(format!("{at}: unknown key"));
        }
    }
    Ok(())
}

/// Position of the parameter `name` (names match without regard to case).
fn param_index(ir: &IrModel, name: &str) -> Option<usize> {
    let key = name.to_lowercase();
    ir.param_names.iter().position(|p| p.to_lowercase() == key)
}

/// A label followed by a 1-based ordinal, for example `BP1`: the form that
/// `pc_engine::special_index` reads.
fn check_switch(name: &str) -> bool {
    match name.find(|c: char| c.is_ascii_digit()) {
        Some(cut) if cut > 0 => {
            let digits = &name[cut..];
            digits.bytes().all(|b| b.is_ascii_digit())
                && digits.parse::<usize>().is_ok_and(|n| n >= 1)
        }
        _ => false,
    }
}

fn parse_problem(v: &Value, ir: &IrModel) -> Result<Problem, String> {
    let top = v
        .as_object()
        .ok_or_else(|| "$: expected an object".to_string())?;
    only_keys(
        top,
        &["type", "free", "u0", "params", "options", "switch"],
        "",
    )?;
    match top.get("type") {
        None => return Err("type: missing".into()),
        Some(Value::String(s)) if s == "EP" => {}
        Some(_) => return Err("type: expected \"EP\"".into()),
    }
    let free_name = match top.get("free") {
        None => return Err("free: missing".into()),
        Some(Value::String(s)) => s,
        Some(_) => return Err("free: expected a string".into()),
    };
    let free = param_index(ir, free_name)
        .ok_or_else(|| format!("free: unknown parameter \"{free_name}\""))?;

    let mut params = ir.par_values.clone();
    if let Some(p) = top.get("params") {
        for (name, val) in object(p, "params")? {
            let path = format!("params.{name}");
            let i = param_index(ir, name)
                .ok_or_else(|| format!("{path}: unknown parameter \"{name}\""))?;
            params[i] = number(val, &path)?;
        }
    }

    let u0 = match top.get("u0") {
        None => ir.init.clone(),
        Some(Value::Array(a)) => {
            if a.len() != ir.n() {
                return Err(format!("u0: expected {} values, got {}", ir.n(), a.len()));
            }
            a.iter()
                .enumerate()
                .map(|(i, x)| number(x, &format!("u0[{i}]")))
                .collect::<Result<Vec<f64>, String>>()?
        }
        Some(_) => return Err("u0: expected an array of numbers".into()),
    };

    let opts = match top.get("options") {
        None => Options::default(),
        Some(o) => parse_options(o)?,
    };

    let switch = match top.get("switch") {
        None => None,
        Some(Value::String(s)) if check_switch(s) => Some(s.clone()),
        Some(Value::String(s)) => {
            return Err(format!(
                "switch: \"{s}\" is not a label followed by a number of at least 1, like \"BP1\""
            ));
        }
        Some(_) => return Err("switch: expected a string".into()),
    };

    Ok(Problem {
        setup: EpSetup::new(free, params, u0),
        opts,
        switch,
    })
}

fn parse_options(v: &Value) -> Result<Options, String> {
    let map = object(v, "options")?;
    only_keys(
        map,
        &[
            "h0",
            "hmin",
            "hmax",
            "max_steps",
            "tol_x",
            "tol_f",
            "max_newton",
            "direction",
            "bounds",
            "detect",
        ],
        "options",
    )?;
    let mut o = Options::default();
    for (key, val) in map {
        let path = format!("options.{key}");
        match key.as_str() {
            "h0" => o.h0 = positive(val, &path)?,
            "hmin" => o.hmin = positive(val, &path)?,
            "hmax" => o.hmax = positive(val, &path)?,
            "tol_x" => o.tol_x = positive(val, &path)?,
            "tol_f" => o.tol_f = positive(val, &path)?,
            "max_steps" => o.max_steps = count(val, &path)?,
            "max_newton" => o.max_newton = count(val, &path)?,
            "direction" => {
                let d = number(val, &path)?;
                if d == 0.0 {
                    return Err(format!("{path}: must not be 0"));
                }
                o.direction = d.signum();
            }
            "bounds" => {
                let a = val
                    .as_array()
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| format!("{path}: expected [lo, hi]"))?;
                let lo = number(&a[0], &format!("{path}[0]"))?;
                let hi = number(&a[1], &format!("{path}[1]"))?;
                if lo >= hi {
                    return Err(format!("{path}: lo must be less than hi"));
                }
                o.bounds = (lo, hi);
            }
            "detect" => {
                let d = object(val, &path)?;
                only_keys(d, &["lp", "bp", "h"], &path)?;
                for (k, flag) in d {
                    let p = format!("{path}.{k}");
                    let b = flag
                        .as_bool()
                        .ok_or_else(|| format!("{p}: expected true or false"))?;
                    match k.as_str() {
                        "lp" => o.detect_lp = b,
                        "bp" => o.detect_bp = b,
                        _ => o.detect_h = b,
                    }
                }
            }
            // `only_keys` has rejected every other key.
            _ => {}
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponent_format_follows_c() {
        assert_eq!(fmt_e(1.0), "1.000000e+00");
        assert_eq!(fmt_e(-2.5e-3), "-2.500000e-03");
        assert_eq!(fmt_e(0.0), "0.000000e+00");
        assert_eq!(fmt_e(1.5e120), "1.500000e+120");
        assert_eq!(fmt_e(f64::NAN), "NaN");
    }

    #[test]
    fn switch_names() {
        for ok in ["BP1", "H12", "LP3"] {
            assert!(check_switch(ok), "{ok}");
        }
        for bad in ["BP", "1", "BP0", "BP1x", "", "B1P2"] {
            assert!(!check_switch(bad), "{bad}");
        }
    }

    #[test]
    fn stderr_line_layout() {
        use pc_engine::{Extras, Point};
        let point = Point {
            u: vec![0.0],
            alpha: 0.0,
            tangent: vec![],
            eigenvalues: vec![],
            tests: [f64::NAN; 3],
            h: 0.0,
            newton_iters: 0,
        };
        let lp = SpecialPoint {
            label: "LP".into(),
            index: 57,
            point: point.clone(),
            extras: Extras {
                coefficient: Some(1.0),
                ..Extras::default()
            },
        };
        assert_eq!(
            special_line(&lp),
            "LP  index 57  a=  1.000000e+00  free=0.000000e+00"
        );
        let h = SpecialPoint {
            label: "H".into(),
            index: 3,
            point: point.clone(),
            extras: Extras {
                coefficient: Some(-2.0),
                omega: Some(1.0),
                ..Extras::default()
            },
        };
        assert_eq!(
            special_line(&h),
            "H   index 3  l1= -2.000000e+00  omega= 1.000000e+00  free=0.000000e+00"
        );
        let bp = SpecialPoint {
            label: "BP".into(),
            index: 9,
            point,
            extras: Extras::default(),
        };
        assert_eq!(special_line(&bp), "BP  index 9  free=0.000000e+00");
    }
}
