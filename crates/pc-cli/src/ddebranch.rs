//! The `dde-branch` command: continue an equilibrium of a model with `delay(x,tau)` calls
//! in one parameter and report the Hopf points, folds and branch points on the way.
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-eq`, and the
//! model is loaded through the same `load_dde`. The `options` object goes through the
//! reader of the `cont` problem files, except that `detect.h` and `mesh_adapt` are refused,
//! since Hopf points come from the root count and a delay equilibrium has no mesh.

use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use pc_engine::{
    DdeBranch, EngineError, EpSetup, Evaluator, Options, RootOptions, StopReason, build_dde,
    dde_continue_equilibrium,
};
use pckern::json::problem::parse_options;
use serde_json::{Map, Value, json};

use crate::cont::stop_name;
use crate::ddeeq::{
    BAD_PROBLEM, check_keys, count, engine_error, load_dde, number, required, to_stderr,
};
use crate::out;

/// What the problem file says.
#[derive(Debug)]
struct Problem {
    /// The model file, as written in the problem.
    model: String,
    /// Declared parameters to set, by name, in file order.
    params: Vec<(String, f64)>,
    /// The declared parameter that the branch runs in.
    free: String,
    x0: Vec<f64>,
    roots: RootOptions,
    opts: Options,
}

/// `dde-branch`: reads the problem, continues the equilibrium and prints the branch. See
/// `DDE_BRANCH_HELP` for the exit codes.
pub fn dde_branch(problem_file: &Path) -> ExitCode {
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
    let find = |name: &str| {
        let key = name.to_lowercase();
        ir.param_names[..n_user]
            .iter()
            .position(|p| p.to_lowercase() == key)
    };
    let mut params = ir.par_values[..n_user].to_vec();
    for (name, value) in &problem.params {
        match find(name) {
            Some(i) => params[i] = *value,
            None => return bad(format!("params: the model has no parameter \"{name}\"")),
        }
    }
    let Some(free) = find(&problem.free) else {
        return bad(format!(
            "free: the model has no parameter \"{}\"",
            problem.free
        ));
    };
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
    let setup = EpSetup::new(free, params, problem.x0.clone());
    let cancel = AtomicBool::new(false);
    let branch = match dde_continue_equilibrium(
        &mut model,
        &setup,
        &problem.roots,
        &problem.opts,
        &cancel,
    ) {
        Ok(b) => b,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    // As `cont` does, a run that stops on an error still prints what it found, and the
    // message goes to stderr.
    if let StopReason::Error(e) = &branch.stop
        && !matches!(e, EngineError::Cancelled)
    {
        to_stderr(&format!("pc-cli: {e}\n"));
    }
    let mut text = branch_json(&branch).to_string();
    text.push('\n');
    out(&text, 0)
}

/// The output object of SPEC section 8e.
fn branch_json(branch: &DdeBranch) -> Value {
    let points: Vec<Value> = branch
        .points
        .iter()
        .map(|p| json!({"alpha": p.alpha, "x": p.x, "unstable": p.unstable}))
        .collect();
    let specials: Vec<Value> = branch
        .specials
        .iter()
        .map(|s| {
            json!({
                "label": s.label,
                "alpha": s.alpha,
                "x": s.x,
                "omega": s.omega,
                "between": s.between,
                "unstable_before": s.unstable_before,
                "unstable_after": s.unstable_after,
            })
        })
        .collect();
    json!({
        "stop": stop_name(&branch.stop),
        "points": points,
        "specials": specials,
        "warnings": branch.warnings,
    })
}

/// Refuses the keys of `options` that mean nothing for a delay equilibrium.
fn check_options(options: &Value) -> Result<(), String> {
    let Some(map) = options.as_object() else {
        // `parse_options` names the type error.
        return Ok(());
    };
    if map.contains_key("mesh_adapt") {
        return Err("options.mesh_adapt: a delay equilibrium has no mesh".into());
    }
    let detect_h = map
        .get("detect")
        .and_then(Value::as_object)
        .is_some_and(|d| d.contains_key("h"));
    if detect_h {
        return Err(
            "options.detect.h: Hopf points come from the root count, not from detect".into(),
        );
    }
    Ok(())
}

/// Reads the problem object. The parameter names and the length of `x0` are checked
/// against the model later, and `n`, `re_min` and the bounds by the engine.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(
        obj,
        &["model", "params", "free", "x0", "n", "re_min", "options"],
        "problem",
    )?;
    let model = required(obj, "model", "problem")?
        .as_str()
        .ok_or_else(|| "\"model\" must be a string".to_string())?
        .to_string();
    let given = required(obj, "params", "problem")?
        .as_object()
        .ok_or_else(|| "\"params\" must be an object of numbers by parameter name".to_string())?;
    let params = named_numbers(given)?;
    let free = required(obj, "free", "problem")?
        .as_str()
        .ok_or_else(|| "\"free\" must be a parameter name".to_string())?
        .to_string();
    let items = required(obj, "x0", "problem")?
        .as_array()
        .ok_or_else(|| "\"x0\" must be an array of numbers".to_string())?;
    let x0 = items
        .iter()
        .map(|v| number(v, "x0"))
        .collect::<Result<Vec<f64>, String>>()?;
    let n = count(required(obj, "n", "problem")?, "n")?;
    let re_min = number(required(obj, "re_min", "problem")?, "re_min")?;
    let opts = match obj.get("options") {
        None => Options::default(),
        Some(v) => {
            check_options(v)?;
            parse_options(v)?
        }
    };
    Ok(Problem {
        model,
        params,
        free,
        x0,
        roots: RootOptions { n, re_min },
        opts,
    })
}

fn named_numbers(given: &Map<String, Value>) -> Result<Vec<(String, f64)>, String> {
    let mut params = Vec::with_capacity(given.len());
    for (name, v) in given {
        let x = v
            .as_f64()
            .ok_or_else(|| format!("params: \"{name}\" must be a number"))?;
        params.push((name.clone(), x));
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Problem, String> {
        read_problem(&serde_json::from_str(text).unwrap())
    }

    const GOOD: &str = r#"{"model":"m.ode","params":{"tau":1},"free":"r","x0":[1.0],
        "n":30,"re_min":-0.5,"options":{"hmax":0.05,"bounds":[1.2,2.0],"detect":{"lp":false}}}"#;

    #[test]
    fn a_good_problem_reads() {
        let p = read(GOOD).unwrap();
        assert_eq!(p.model, "m.ode");
        assert_eq!(p.params, [("tau".to_string(), 1.0)]);
        assert_eq!(p.free, "r");
        assert_eq!(p.x0, [1.0]);
        assert_eq!((p.roots.n, p.roots.re_min), (30, -0.5));
        assert_eq!(p.opts.hmax, 0.05);
        assert_eq!(p.opts.bounds, (1.2, 2.0));
        assert!(!p.opts.detect_lp);
    }

    #[test]
    fn options_may_be_left_out() {
        let mut v: Value = serde_json::from_str(GOOD).unwrap();
        v.as_object_mut().unwrap().remove("options");
        let p = read_problem(&v).unwrap();
        assert_eq!(p.opts.hmax, Options::default().hmax);
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        for key in ["model", "params", "free", "x0", "n", "re_min"] {
            let mut v: Value = serde_json::from_str(GOOD).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        let extra = GOOD.replace("\"n\":30,", "\"n\":30,\"nn\":3,");
        assert!(read(&extra).unwrap_err().contains("unknown key \"nn\""));
    }

    #[test]
    fn wrong_types_name_the_key() {
        let cases = [
            (GOOD.replace("\"n\":30", "\"n\":-1"), "\"n\""),
            (GOOD.replace("-0.5", "\"low\""), "\"re_min\""),
            (GOOD.replace("\"r\"", "3"), "\"free\""),
            (GOOD.replace("[1.0]", "1.0"), "\"x0\""),
            (GOOD.replace("{\"tau\":1}", "[1]"), "\"params\""),
            (GOOD.replace("\"tau\":1", "\"tau\":\"x\""), "\"tau\""),
            (GOOD.replace("0.05", "-1"), "options.hmax"),
        ];
        for (text, key) in cases {
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(key), "{text}: {msg}");
        }
        assert!(read("[1]").unwrap_err().contains("JSON object"));
    }

    #[test]
    fn detect_h_and_mesh_adapt_are_refused() {
        let h = GOOD.replace("\"lp\":false", "\"h\":true");
        assert!(read(&h).unwrap_err().contains("options.detect.h"));
        let h_off = GOOD.replace("\"lp\":false", "\"h\":false");
        assert!(read(&h_off).unwrap_err().contains("options.detect.h"));
        let mesh = GOOD.replace("\"hmax\":0.05", "\"mesh_adapt\":2");
        assert!(read(&mesh).unwrap_err().contains("options.mesh_adapt"));
    }
}
