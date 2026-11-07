//! The `dde-hopf` command: continue a Hopf point of a model with `delay(x,tau)` calls in
//! two declared parameters and report the Bogdanov-Takens and zero-Hopf points on the way.
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-branch`, and
//! the model is loaded through the same `load_dde`. The `options` object goes through the
//! reader of the `cont` problem files, except that `detect` and `mesh_adapt` are refused,
//! since the codimension-2 points of this curve do not use them.

use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use pc_engine::{
    DdeHopfBranch, DdeHopfSetup, EngineError, Evaluator, Options, StopReason, build_dde,
    dde_continue_hopf, dde_continue_hopf_with_lyapunov,
};
use pckern::json::problem::parse_options;
use serde_json::{Map, Value, json};

use crate::cont::stop_name;
use crate::ddeeq::{BAD_PROBLEM, check_keys, engine_error, load_dde, number, required, to_stderr};
use crate::out;

/// What the problem file says.
#[derive(Debug)]
struct Problem {
    /// The model file, as written in the problem.
    model: String,
    /// Declared parameters to set, by name, in file order.
    params: Vec<(String, f64)>,
    /// The two declared parameters that the curve runs in. `options.bounds` bounds the second.
    free: [String; 2],
    x0: Vec<f64>,
    omega: f64,
    /// Compute the first Lyapunov coefficient at every point and label GH (off by default).
    lyapunov: bool,
    opts: Options,
}

/// `dde-hopf`: reads the problem, continues the Hopf point and prints the curve. See
/// `DDE_HOPF_HELP` for the exit codes.
pub fn dde_hopf(problem_file: &Path) -> ExitCode {
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
    let mut free = [0usize; 2];
    for (slot, name) in free.iter_mut().zip(&problem.free) {
        match find(name) {
            Some(i) => *slot = i,
            None => return bad(format!("free: the model has no parameter \"{name}\"")),
        }
    }
    if free[0] == free[1] {
        // Two spellings of one name, such as "R" and "r", land here.
        return bad(format!(
            "free: \"{}\" and \"{}\" name the same parameter",
            problem.free[0], problem.free[1]
        ));
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
    let setup = DdeHopfSetup {
        free,
        params,
        x0: problem.x0.clone(),
        omega: problem.omega,
    };
    let cancel = AtomicBool::new(false);
    let run = if problem.lyapunov {
        dde_continue_hopf_with_lyapunov
    } else {
        dde_continue_hopf
    };
    let branch = match run(&mut model, &setup, &problem.opts, &cancel) {
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
    let mut text = branch_json(&branch, problem.lyapunov).to_string();
    text.push('\n');
    out(&text, 0)
}

/// The output object of SPEC sections 8f, 8g-2 and 8h. A point carries `l1` and a `GH`
/// special carries `l2` only when the problem asked for `lyapunov`, so a run without it
/// prints what it printed before.
fn branch_json(branch: &DdeHopfBranch, lyapunov: bool) -> Value {
    let points: Vec<Value> = branch
        .points
        .iter()
        .map(|p| {
            let mut point = json!({"alpha": p.alpha, "x": p.x, "kappa": p.kappa, "omega": p.omega});
            if lyapunov {
                point["l1"] = json!(p.l1);
            }
            point
        })
        .collect();
    let specials: Vec<Value> = branch
        .specials
        .iter()
        .map(|s| {
            let mut special = json!({
                "label": s.label,
                "alpha": s.alpha,
                "x": s.x,
                "kappa": s.kappa,
                "index": s.index,
            });
            // Only a GH special of a run with `lyapunov` has the key, and it is null where
            // the second coefficient cannot be computed.
            if lyapunov && s.label == "GH" {
                special["l2"] = json!(s.l2);
            }
            special
        })
        .collect();
    json!({
        "stop": stop_name(&branch.stop),
        "points": points,
        "specials": specials,
        "warnings": branch.warnings,
    })
}

/// Refuses the keys of `options` that mean nothing for this curve.
fn check_options(options: &Value) -> Result<(), String> {
    let Some(map) = options.as_object() else {
        // `parse_options` names the type error.
        return Ok(());
    };
    if map.contains_key("mesh_adapt") {
        return Err("options.mesh_adapt: a delay equilibrium has no mesh".into());
    }
    if map.contains_key("detect") {
        return Err(
            "options.detect: the codimension-2 points of this curve do not use detect".into(),
        );
    }
    Ok(())
}

/// Reads the two names of `free`.
fn read_free(value: &Value) -> Result<[String; 2], String> {
    let names = value
        .as_array()
        .filter(|items| items.len() == 2)
        .and_then(|items| {
            items
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<Vec<String>>>()
        })
        .ok_or_else(|| "\"free\" must be an array of two parameter names".to_string())?;
    let [a, b] = <[String; 2]>::try_from(names).map_err(|_| "\"free\" needs two names")?;
    if a.to_lowercase() == b.to_lowercase() {
        return Err(format!("free: \"{a}\" is named twice"));
    }
    Ok([a, b])
}

/// Reads the problem object. The parameter names and the length of `x0` are checked
/// against the model later, and `omega` and the bounds by the engine.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(
        obj,
        &[
            "model", "params", "free", "x0", "omega", "lyapunov", "options",
        ],
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
    let free = read_free(required(obj, "free", "problem")?)?;
    let items = required(obj, "x0", "problem")?
        .as_array()
        .ok_or_else(|| "\"x0\" must be an array of numbers".to_string())?;
    let x0 = items
        .iter()
        .map(|v| number(v, "x0"))
        .collect::<Result<Vec<f64>, String>>()?;
    let omega = number(required(obj, "omega", "problem")?, "omega")?;
    let lyapunov = match obj.get("lyapunov") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "\"lyapunov\" must be true or false".to_string())?,
    };
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
        omega,
        lyapunov,
        opts,
    })
}

pub(crate) fn named_numbers(given: &Map<String, Value>) -> Result<Vec<(String, f64)>, String> {
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

    const GOOD: &str = r#"{"model":"m.ode","params":{"tau":1},"free":["r","tau"],
        "x0":[1.0],"omega":1.5,
        "options":{"hmax":0.05,"bounds":[0.5,3.0]}}"#;

    #[test]
    fn a_good_problem_reads() {
        let p = read(GOOD).unwrap();
        assert_eq!(p.model, "m.ode");
        assert_eq!(p.params, [("tau".to_string(), 1.0)]);
        assert_eq!(p.free, ["r".to_string(), "tau".to_string()]);
        assert_eq!(p.x0, [1.0]);
        assert_eq!(p.omega, 1.5);
        assert!(
            !p.lyapunov,
            "lyapunov is off unless the problem asks for it"
        );
        assert_eq!(p.opts.hmax, 0.05);
        assert_eq!(p.opts.bounds, (0.5, 3.0));
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
        for key in ["model", "params", "free", "x0", "omega"] {
            let mut v: Value = serde_json::from_str(GOOD).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        for key in ["n", "re_min", "nn"] {
            let extra = GOOD.replace("\"omega\":1.5,", &format!("\"omega\":1.5,\"{key}\":3,"));
            let msg = read(&extra).unwrap_err();
            assert!(msg.contains(&format!("unknown key \"{key}\"")), "{msg}");
        }
    }

    #[test]
    fn wrong_types_name_the_key() {
        let cases = [
            (GOOD.replace("1.5", "\"fast\""), "\"omega\""),
            (GOOD.replace("[\"r\",\"tau\"]", "\"r\""), "\"free\""),
            (GOOD.replace("[\"r\",\"tau\"]", "[\"r\"]"), "\"free\""),
            (
                GOOD.replace("[\"r\",\"tau\"]", "[\"r\",\"tau\",\"s\"]"),
                "\"free\"",
            ),
            (GOOD.replace("[\"r\",\"tau\"]", "[\"r\",2]"), "\"free\""),
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
    fn lyapunov_is_a_boolean() {
        let on = GOOD.replace("\"omega\":1.5,", "\"omega\":1.5,\"lyapunov\":true,");
        assert!(read(&on).unwrap().lyapunov);
        let off = GOOD.replace("\"omega\":1.5,", "\"omega\":1.5,\"lyapunov\":false,");
        assert!(!read(&off).unwrap().lyapunov);
        for bad in ["1", "\"yes\"", "null", "[true]"] {
            let text = GOOD.replace(
                "\"omega\":1.5,",
                &format!("\"omega\":1.5,\"lyapunov\":{bad},"),
            );
            let msg = read(&text).unwrap_err();
            assert!(msg.contains("\"lyapunov\""), "{bad}: {msg}");
        }
    }

    #[test]
    fn a_name_twice_in_free_is_refused_whatever_its_case() {
        for names in ["[\"r\",\"r\"]", "[\"r\",\"R\"]"] {
            let text = GOOD.replace("[\"r\",\"tau\"]", names);
            assert!(read(&text).unwrap_err().contains("twice"), "{names}");
        }
    }

    #[test]
    fn detect_and_mesh_adapt_are_refused() {
        let detect = GOOD.replace("\"hmax\":0.05", "\"detect\":{\"lp\":false}");
        assert!(read(&detect).unwrap_err().contains("options.detect"));
        let mesh = GOOD.replace("\"hmax\":0.05", "\"mesh_adapt\":2");
        assert!(read(&mesh).unwrap_err().contains("options.mesh_adapt"));
    }
}
