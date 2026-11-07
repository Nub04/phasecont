//! The `dde-zh` command: the critical normal-form coefficients at a zero-Hopf point of a
//! model with `delay(x,tau)` calls (SPEC section 8i).
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-lyap`, and the
//! model is loaded through the same `load_dde`. The point is used as given: the command
//! does not polish `x0` or `omega`, and it does not classify the unfolding.

use std::path::Path;
use std::process::ExitCode;

use pc_engine::{DdeZeroHopf, Evaluator, build_dde, dde_zero_hopf};
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
}

/// `dde-zh`: reads the problem and prints the coefficients. See `DDE_ZH_HELP` for the exit
/// codes.
pub fn dde_zh(problem_file: &Path) -> ExitCode {
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
    let value = match dde_zero_hopf(&mut model, &problem.x0, &params, problem.omega) {
        Ok(r) => result_json(&r),
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let mut text = value.to_string();
    text.push('\n');
    out(&text, 0)
}

/// The output object of SPEC section 8i. A `None` prints as `null`.
fn result_json(r: &DdeZeroHopf) -> Value {
    json!({
        "omega": r.omega,
        "g200": r.g200,
        "g110": r.g110,
        "g011": r.g011,
        "g300": r.g300,
        "g111": r.g111,
        "g210": r.g210,
        "g021": r.g021,
        "s_product": r.s_product,
        "s": r.s,
        "theta": r.theta,
        "e": r.e,
        "null_residual_zero": r.null_residual_zero,
        "null_residual_hopf": r.null_residual_hopf,
        "equilibrium_residual": r.equilibrium_residual,
    })
}

/// Reads the problem object. The parameter names and the length of `x0` are checked
/// against the model later, and `omega` by the engine.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(obj, &["model", "params", "x0", "omega"], "problem")?;
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
    Ok(Problem {
        model,
        params,
        x0,
        omega,
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
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        for key in ["model", "params", "x0", "omega"] {
            let mut v: Value = serde_json::from_str(GOOD).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        for key in ["l2", "free", "options"] {
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
