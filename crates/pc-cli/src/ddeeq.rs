//! The `dde-eq` command: an equilibrium of a model with `delay(x,tau)` calls and the
//! characteristic roots of the linear equation there.
//!
//! The problem file is read by hand from a `serde_json::Value`, so that every error can
//! name the key that is wrong. Unknown keys are errors, and every key is required. The
//! model is loaded the way `ir` loads it, except that it goes through `lower_dde`, which
//! accepts the delay calls that `lower` refuses. The result goes to stdout as one JSON
//! object.

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_engine::{
    EngineError, EpSetup, Evaluator, Options, RootOptions, build_dde, characteristic_roots,
    dde_equilibrium, dde_linearize, dde_residual,
};
use pc_ir::lower_dde;
use pc_ode::Severity;
use serde_json::{Map, Value, json};

use crate::{load_file, out, text};

/// Exit code for an unreadable or invalid problem file, and for a problem the engine
/// refuses as `BadProblem`.
const BAD_PROBLEM: u8 = 2;

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

/// What the problem file says.
#[derive(Debug)]
struct Problem {
    /// The model file, as written in the problem.
    model: String,
    /// Declared parameters to set, by name, in file order.
    params: Vec<(String, f64)>,
    x0: Vec<f64>,
    roots: RootOptions,
}

/// `dde-eq`: reads the problem, solves for the equilibrium and prints it with the roots.
/// See `DDE_EQ_HELP` for the exit codes.
pub fn dde_eq(problem_file: &Path) -> ExitCode {
    let shown = problem_file.display();
    let bad = |msg: String| {
        to_stderr(&format!("pc-cli: problem file {shown}: {msg}\n"));
        ExitCode::from(BAD_PROBLEM)
    };
    let text = match std::fs::read_to_string(problem_file) {
        Ok(t) => t,
        Err(e) => {
            to_stderr(&format!("pc-cli: cannot read problem file {shown}: {e}\n"));
            return ExitCode::from(BAD_PROBLEM);
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return bad(format!("invalid JSON: {e}")),
    };
    let problem = match read_problem(&value) {
        Ok(p) => p,
        Err(msg) => return bad(msg),
    };
    // A relative model path is read from the directory of the problem file.
    let model_path = match problem_file.parent() {
        Some(dir) => dir.join(&problem.model),
        None => Path::new(&problem.model).to_path_buf(),
    };
    let (map, loaded) = match load_file(&model_path) {
        Ok(v) => v,
        Err(code) => return code,
    };
    to_stderr(&text::diagnostics(&map, &loaded));
    if loaded
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        return ExitCode::from(1);
    }
    let ir = match lower_dde(&loaded.expanded, &loaded.info) {
        Ok(ir) => ir,
        Err(errors) => {
            to_stderr(&text::ir_errors(
                &map,
                &model_path.to_string_lossy(),
                &errors,
            ));
            return ExitCode::from(1);
        }
    };
    // The declared parameters come first in `param_names`; the hidden slots follow.
    let n_user = ir.n_user_params;
    let mut params = ir.par_values[..n_user].to_vec();
    for (name, value) in &problem.params {
        let key = name.to_lowercase();
        match ir.param_names[..n_user]
            .iter()
            .position(|p| p.to_lowercase() == key)
        {
            Some(i) => params[i] = *value,
            None => return bad(format!("params: the model has no parameter \"{name}\"")),
        }
    }
    let n = ir.n();
    if problem.x0.len() != n {
        let s = if n == 1 { "" } else { "s" };
        return bad(format!(
            "x0: the model has {n} state{s}, x0 has {} values",
            problem.x0.len()
        ));
    }
    let mut model = match build_dde(&loaded.expanded, &loaded.info, Evaluator::Jit) {
        Ok(m) => m,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let setup = EpSetup::new(0, params.clone(), problem.x0.clone());
    let x = match dde_equilibrium(&mut model, &setup, &Options::default()) {
        Ok(x) => x,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let result = dde_residual(&mut model, &x, &params).and_then(|residual| {
        let lin = dde_linearize(&mut model, &x, &params)?;
        let set = characteristic_roots(&lin, &problem.roots)?;
        Ok((residual, lin, set))
    });
    let (residual, lin, set) = match result {
        Ok(r) => r,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let delays: Vec<f64> = lin.delays.iter().map(|(tau, _)| *tau).collect();
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
    let mut text = json!({
        "x": x,
        "residual": residual,
        "delays": delays,
        "roots": roots,
        "unrefined": unrefined,
    })
    .to_string();
    text.push('\n');
    out(&text, 0)
}

/// Prints an engine error: a refused problem is exit 2 and names the file, anything else is
/// a failed run, exit 1.
fn engine_error(e: &EngineError, shown: &str) -> ExitCode {
    match e {
        EngineError::BadProblem(msg) => {
            to_stderr(&format!("pc-cli: problem file {shown}: {msg}\n"));
            ExitCode::from(BAD_PROBLEM)
        }
        e => {
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

/// Reads the problem object. The parameter names and the length of `x0` are checked
/// against the model later, and `n` and `re_min` by `characteristic_roots`.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(obj, &["model", "params", "x0", "n", "re_min"], "problem")?;
    let model = required(obj, "model", "problem")?
        .as_str()
        .ok_or_else(|| "\"model\" must be a string".to_string())?
        .to_string();
    let given = required(obj, "params", "problem")?
        .as_object()
        .ok_or_else(|| "\"params\" must be an object of numbers by parameter name".to_string())?;
    let mut params = Vec::with_capacity(given.len());
    for (name, v) in given {
        let x = v
            .as_f64()
            .ok_or_else(|| format!("params: \"{name}\" must be a number"))?;
        params.push((name.clone(), x));
    }
    let items = required(obj, "x0", "problem")?
        .as_array()
        .ok_or_else(|| "\"x0\" must be an array of numbers".to_string())?;
    let x0 = items
        .iter()
        .map(|v| number(v, "x0"))
        .collect::<Result<Vec<f64>, String>>()?;
    let n = count(required(obj, "n", "problem")?, "n")?;
    let re_min = number(required(obj, "re_min", "problem")?, "re_min")?;
    Ok(Problem {
        model,
        params,
        x0,
        roots: RootOptions { n, re_min },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Problem, String> {
        read_problem(&serde_json::from_str(text).unwrap())
    }

    const GOOD: &str =
        r#"{"model":"m.ode","params":{"r":1.5,"tau":2},"x0":[0.8],"n":10,"re_min":-2.0}"#;

    #[test]
    fn a_good_problem_reads() {
        let p = read(GOOD).unwrap();
        assert_eq!(p.model, "m.ode");
        assert_eq!(p.params, [("r".to_string(), 1.5), ("tau".to_string(), 2.0)]);
        assert_eq!(p.x0, [0.8]);
        assert_eq!((p.roots.n, p.roots.re_min), (10, -2.0));
        let none = GOOD.replace(r#"{"r":1.5,"tau":2}"#, "{}");
        assert!(read(&none).unwrap().params.is_empty());
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        for key in ["model", "params", "x0", "n", "re_min"] {
            let mut v: Value = serde_json::from_str(GOOD).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        let extra = GOOD.replace("\"n\":10,", "\"n\":10,\"nn\":3,");
        assert!(read(&extra).unwrap_err().contains("unknown key \"nn\""));
    }

    #[test]
    fn wrong_types_name_the_key() {
        let cases = [
            (GOOD.replace("\"n\":10", "\"n\":-1"), "\"n\""),
            (GOOD.replace("\"n\":10", "\"n\":2.5"), "\"n\""),
            (GOOD.replace("-2.0", "\"low\""), "\"re_min\""),
            (GOOD.replace("\"m.ode\"", "3"), "\"model\""),
            (GOOD.replace("[0.8]", "0.8"), "\"x0\""),
            (GOOD.replace("[0.8]", "[\"a\"]"), "\"x0\""),
            (GOOD.replace("{\"r\":1.5,\"tau\":2}", "[1]"), "\"params\""),
            (GOOD.replace("\"r\":1.5", "\"r\":\"x\""), "\"r\""),
        ];
        for (text, key) in cases {
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(key), "{text}: {msg}");
        }
        assert!(read("[1]").unwrap_err().contains("JSON object"));
    }
}
