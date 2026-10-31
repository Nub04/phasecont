//! The `cont` command: follow the branch or curve that a problem file describes.
//!
//! The problem file is read by hand from a `serde_json::Value`, so that every error can
//! name the key that is wrong. Unknown keys are errors. The run goes through
//! `pc_engine::Run`; the result goes to stdout as one JSON object and the special points
//! go to stderr, one line each.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_engine::{EngineError, Evaluator, Run, RunState, SpecialPoint, StopReason};
use pckern::json::problem::{Mode, parse_problem};
use serde_json::{Value, json};

use crate::ircmd::{Prepared, prepare};
use crate::out;

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
    let problem = match parse_problem(&value, &ir, Mode::Cli) {
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
    let run = Run::start(problem.into_spec(loaded.expanded, loaded.info, evaluator));
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
                    "u": p.u, "alpha": p.alpha, "tests": p.tests, "codim2": p.codim2,
                    "eigenvalues": p.eigenvalues, "mesh": p.mesh,
                    "h": p.h, "newton_iters": p.newton_iters,
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
        "eigenvalues": s.point.eigenvalues,
        "mesh": s.point.mesh,
        "coefficient": s.extras.coefficient,
        "coefficient2": s.extras.coefficient2,
        "coefficients": s.extras.values.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
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
    fn stderr_line_layout() {
        use pc_engine::{Extras, Point};
        let point = Point {
            u: vec![0.0],
            alpha: 0.0,
            tangent: vec![],
            eigenvalues: vec![],
            tests: [f64::NAN; 3],
            codim2: Vec::new(),
            h: 0.0,
            newton_iters: 0,
            mesh: Vec::new(),
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
