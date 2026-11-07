//! The `dde-lyap` command: the first Lyapunov coefficient at a Hopf point of a model with
//! `delay(x,tau)` calls.
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-hopf`, and the
//! model is loaded through the same `load_dde`. The point is used as given: the command
//! does not polish `x0` or `omega`. The optional key `l2` asks for the second Lyapunov
//! coefficient too (SPEC section 8h).

use std::path::Path;
use std::process::ExitCode;

use pc_engine::{DdeLyapunov, DdeLyapunov2, Evaluator, build_dde, dde_lyapunov, dde_lyapunov2};
use serde_json::{Value, json};

use crate::ddeeq::{BAD_PROBLEM, check_keys, engine_error, load_dde, number, required, to_stderr};
use crate::ddehopf::named_numbers;
use crate::out;

/// What the problem file says.
#[derive(Debug)]
struct Problem {
    /// The model file, as written in the problem.
    model: String,
    /// Declared parameters to set, by name, in file order.
    params: Vec<(String, f64)>,
    x0: Vec<f64>,
    omega: f64,
    /// Whether to compute the second coefficient as well; false unless the problem says so.
    l2: bool,
}

/// `dde-lyap`: reads the problem and prints the coefficient. See `DDE_LYAP_HELP` for the
/// exit codes.
pub fn dde_lyap(problem_file: &Path) -> ExitCode {
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
    let (loaded, ir) = match load_dde(problem_file, &problem.model) {
        Ok(v) => v,
        Err(code) => return code,
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
    let result = if problem.l2 {
        dde_lyapunov2(&mut model, &problem.x0, &params, problem.omega).map(|r| result2_json(&r))
    } else {
        dde_lyapunov(&mut model, &problem.x0, &params, problem.omega).map(|r| result_json(&r))
    };
    let value = match result {
        Ok(v) => v,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let mut text = value.to_string();
    text.push('\n');
    out(&text, 0)
}

/// The output object of SPEC section 8g.
fn result_json(r: &DdeLyapunov) -> Value {
    json!({
        "omega": r.omega,
        "l1": r.l1,
        "c1": r.c1,
        "null_residual": r.null_residual,
        "equilibrium_residual": r.equilibrium_residual,
    })
}

/// The output object of SPEC section 8h, for a problem with `"l2": true`.
fn result2_json(r: &DdeLyapunov2) -> Value {
    json!({
        "omega": r.omega,
        "l1": r.l1,
        "l2": r.l2,
        "c1": r.c1,
        "c2": r.c2,
        "null_residual": r.null_residual,
        "equilibrium_residual": r.equilibrium_residual,
    })
}

/// Reads the problem object. The parameter names and the length of `x0` are checked
/// against the model later, and `omega` by the engine.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(obj, &["model", "params", "x0", "omega", "l2"], "problem")?;
    let model = required(obj, "model", "problem")?
        .as_str()
        .ok_or_else(|| "\"model\" must be a string".to_string())?
        .to_string();
    let given = required(obj, "params", "problem")?
        .as_object()
        .ok_or_else(|| "\"params\" must be an object of numbers by parameter name".to_string())?;
    let params = named_numbers(given)?;
    let items = required(obj, "x0", "problem")?
        .as_array()
        .ok_or_else(|| "\"x0\" must be an array of numbers".to_string())?;
    let x0 = items
        .iter()
        .map(|v| number(v, "x0"))
        .collect::<Result<Vec<f64>, String>>()?;
    let omega = number(required(obj, "omega", "problem")?, "omega")?;
    let l2 = match obj.get("l2") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "\"l2\" must be true or false".to_string())?,
    };
    Ok(Problem {
        model,
        params,
        x0,
        omega,
        l2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Problem, String> {
        read_problem(&serde_json::from_str(text).unwrap())
    }

    const GOOD: &str = r#"{"model":"m.ode","params":{"a":1.5},"x0":[0.0],"omega":1.5}"#;

    #[test]
    fn a_good_problem_reads() {
        let p = read(GOOD).unwrap();
        assert_eq!(p.model, "m.ode");
        assert_eq!(p.params, [("a".to_string(), 1.5)]);
        assert_eq!(p.x0, [0.0]);
        assert_eq!(p.omega, 1.5);
        assert!(!p.l2, "l2 is off unless the problem asks for it");
    }

    #[test]
    fn l2_is_a_boolean() {
        let on = GOOD.replace("\"omega\":1.5", "\"omega\":1.5,\"l2\":true");
        assert!(read(&on).unwrap().l2);
        let off = GOOD.replace("\"omega\":1.5", "\"omega\":1.5,\"l2\":false");
        assert!(!read(&off).unwrap().l2);
        for bad in ["1", "\"yes\"", "null", "[true]"] {
            let text = GOOD.replace("\"omega\":1.5", &format!("\"omega\":1.5,\"l2\":{bad}"));
            let msg = read(&text).unwrap_err();
            assert!(msg.contains("\"l2\""), "{bad}: {msg}");
        }
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        for key in ["model", "params", "x0", "omega"] {
            let mut v: Value = serde_json::from_str(GOOD).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        for key in ["free", "options", "n"] {
            let extra = GOOD.replace("\"omega\":1.5", &format!("\"omega\":1.5,\"{key}\":3"));
            let msg = read(&extra).unwrap_err();
            assert!(msg.contains(&format!("unknown key \"{key}\"")), "{msg}");
        }
    }

    #[test]
    fn wrong_types_name_the_key() {
        let cases = [
            (
                GOOD.replace("\"omega\":1.5", "\"omega\":\"fast\""),
                "\"omega\"",
            ),
            (GOOD.replace("[0.0]", "0.0"), "\"x0\""),
            (GOOD.replace("[0.0]", "[\"a\"]"), "x0"),
            (GOOD.replace("{\"a\":1.5}", "[1]"), "\"params\""),
            (GOOD.replace("\"a\":1.5", "\"a\":\"x\""), "\"a\""),
            (GOOD.replace("\"m.ode\"", "3"), "\"model\""),
        ];
        for (text, key) in cases {
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(key), "{text}: {msg}");
        }
        assert!(read("[1]").unwrap_err().contains("JSON object"));
    }
}
