//! The `dde-roots` command: characteristic roots of a linear delay equation.
//!
//! The problem file is read by hand from a `serde_json::Value`, so that every error can
//! name the key that is wrong. Unknown keys are errors, and `n` and `re_min` have no
//! default. The roots go to stdout as one JSON object.

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_engine::{EngineError, LinearDde, RootOptions, characteristic_roots};
use serde_json::{Map, Value, json};

use crate::out;

/// Exit code for an unreadable or invalid problem file, and for a problem the engine
/// refuses as `BadProblem`.
const BAD_PROBLEM: u8 = 2;

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

/// `dde-roots`: reads the problem, computes the roots and prints them. See
/// `DDE_ROOTS_HELP` for the exit codes.
pub fn dde_roots(problem_file: &Path) -> ExitCode {
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
    let (problem, options) = match read_problem(&value) {
        Ok(p) => p,
        Err(msg) => {
            to_stderr(&format!(
                "pc-cli: problem file {}: {msg}\n",
                problem_file.display()
            ));
            return ExitCode::from(BAD_PROBLEM);
        }
    };
    match characteristic_roots(&problem, &options) {
        Ok(set) => {
            let roots: Vec<Value> = set
                .roots
                .iter()
                .map(|r| json!({"re": r.re, "im": r.im, "residual": r.residual}))
                .collect();
            let unrefined: Vec<Value> = set
                .unrefined
                .iter()
                .map(|&(re, im)| json!({"re": re, "im": im}))
                .collect();
            let mut text = json!({"n": set.n, "roots": roots, "unrefined": unrefined}).to_string();
            text.push('\n');
            out(&text, 0)
        }
        Err(EngineError::BadProblem(msg)) => {
            to_stderr(&format!(
                "pc-cli: problem file {}: {msg}\n",
                problem_file.display()
            ));
            ExitCode::from(BAD_PROBLEM)
        }
        Err(e) => {
            to_stderr(&format!("pc-cli: {e}\n"));
            ExitCode::from(1)
        }
    }
}

/// Reports the first key of `obj` that is not in `allowed`.
fn check_keys(obj: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<(), String> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{what}: unknown key \"{key}\""));
        }
    }
    Ok(())
}

fn required<'a>(obj: &'a Map<String, Value>, key: &str, what: &str) -> Result<&'a Value, String> {
    obj.get(key)
        .ok_or_else(|| format!("{what}: missing key \"{key}\""))
}

fn count(v: &Value, key: &str) -> Result<usize, String> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("\"{key}\" must be a non-negative integer"))
}

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("\"{key}\" must be a number"))
}

fn matrix(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    let items = v
        .as_array()
        .ok_or_else(|| format!("\"{key}\" must be an array of numbers"))?;
    items
        .iter()
        .map(|x| number(x, key))
        .collect::<Result<Vec<f64>, String>>()
}

/// Reads the problem object. Lengths and ranges are left to `characteristic_roots`, which
/// refuses them as `BadProblem`.
fn read_problem(value: &Value) -> Result<(LinearDde, RootOptions), String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(obj, &["dim", "a0", "delays", "n", "re_min"], "problem")?;
    let dim = count(required(obj, "dim", "problem")?, "dim")?;
    let a0 = matrix(required(obj, "a0", "problem")?, "a0")?;
    let n = count(required(obj, "n", "problem")?, "n")?;
    let re_min = number(required(obj, "re_min", "problem")?, "re_min")?;
    let items = required(obj, "delays", "problem")?
        .as_array()
        .ok_or_else(|| "\"delays\" must be an array".to_string())?;
    let mut delays = Vec::with_capacity(items.len());
    for (j, item) in items.iter().enumerate() {
        let what = format!("delays[{j}]");
        let d = item
            .as_object()
            .ok_or_else(|| format!("{what} must be an object"))?;
        check_keys(d, &["tau", "a"], &what)?;
        let tau = number(required(d, "tau", &what)?, "tau").map_err(|m| format!("{what}: {m}"))?;
        let a = matrix(required(d, "a", &what)?, "a").map_err(|m| format!("{what}: {m}"))?;
        delays.push((tau, a));
    }
    Ok((LinearDde { dim, a0, delays }, RootOptions { n, re_min }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<(LinearDde, RootOptions), String> {
        read_problem(&serde_json::from_str(text).unwrap())
    }

    const GOOD: &str =
        r#"{"dim":1,"a0":[0.0],"delays":[{"tau":1.0,"a":[-1.5]}],"n":10,"re_min":-2.0}"#;

    #[test]
    fn a_good_problem_reads() {
        let (p, o) = read(GOOD).unwrap();
        assert_eq!((p.dim, o.n, o.re_min), (1, 10, -2.0));
        assert_eq!(p.delays, vec![(1.0, vec![-1.5])]);
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        let no_n = r#"{"dim":1,"a0":[0.0],"delays":[{"tau":1.0,"a":[-1.5]}],"re_min":-2.0}"#;
        assert!(read(no_n).unwrap_err().contains("missing key \"n\""));
        let no_r = GOOD.replace(",\"re_min\":-2.0", "");
        assert!(read(&no_r).unwrap_err().contains("missing key \"re_min\""));
        let extra = GOOD.replace("\"dim\":1,", "\"dim\":1,\"nn\":3,");
        assert!(read(&extra).unwrap_err().contains("unknown key \"nn\""));
        let inner = GOOD.replace("\"tau\":1.0,", "\"tau\":1.0,\"lag\":2,");
        let msg = read(&inner).unwrap_err();
        assert!(
            msg.contains("delays[0]") && msg.contains("unknown key \"lag\""),
            "{msg}"
        );
    }

    #[test]
    fn wrong_types_name_the_key() {
        let neg = GOOD.replace("\"n\":10", "\"n\":-1");
        assert!(read(&neg).unwrap_err().contains("\"n\""));
        let frac = GOOD.replace("\"n\":10", "\"n\":2.5");
        assert!(read(&frac).unwrap_err().contains("\"n\""));
        let text = GOOD.replace("\"re_min\":-2.0", "\"re_min\":\"low\"");
        assert!(read(&text).unwrap_err().contains("\"re_min\""));
        assert!(read("[1]").unwrap_err().contains("JSON object"));
    }
}
