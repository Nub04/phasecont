//! The `dde-cycle` command: correct a periodic orbit of a model with `delay(x,tau)` calls
//! by collocation, from a given orbit or from a Hopf point (SPEC section 8l).
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-hopf`, and the
//! model is loaded through the same `load_dde`. The corrector reads only `max_newton`,
//! `tol_x` and `tol_f` from `options`, so the other keys of the `cont` reader are refused
//! by name instead of being accepted and ignored.

use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use pc_engine::{
    DdeCycle, DdeCycleSetup, DdeCycleStart, DdeHopfCycle, Evaluator, Options, build_dde,
    dde_correct_cycle, dde_hopf_cycle,
};
use pckern::json::problem::parse_options;
use serde_json::{Value, json};

use crate::ddeeq::{
    BAD_PROBLEM, check_keys, count, engine_error, load_dde, number, required, to_stderr,
};
use crate::ddehopf::named_numbers;
use crate::out;

/// Where the corrector starts.
#[derive(Debug, PartialEq)]
enum Start {
    /// An orbit: `u` at the base points and the period.
    Orbit { profile: Vec<f64>, period: f64 },
    /// A Hopf point of an equilibrium.
    Hopf { x0: Vec<f64>, omega: f64, eps: f64 },
}

/// What the problem file says.
#[derive(Debug)]
struct Problem {
    /// The model file, as written in the problem.
    model: String,
    /// Declared parameters to set, by name, in file order.
    params: Vec<(String, f64)>,
    /// The declared parameter that is free.
    free: String,
    ntst: usize,
    ncol: usize,
    start: Start,
    opts: Options,
}

/// `dde-cycle`: reads the problem, runs the corrector and prints the orbit. See
/// `DDE_CYCLE_HELP` for the exit codes.
pub fn dde_cycle(problem_file: &Path) -> ExitCode {
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
    let mut model = match build_dde(&loaded.expanded, &loaded.info, Evaluator::Jit) {
        Ok(m) => m,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let setup = DdeCycleSetup {
        params,
        free,
        ntst: problem.ntst,
        ncol: problem.ncol,
    };
    let cancel = AtomicBool::new(false);
    let result = match &problem.start {
        Start::Orbit { profile, period } => {
            let start = DdeCycleStart {
                profile: profile.clone(),
                period: *period,
            };
            dde_correct_cycle(&mut model, &setup, &start, &problem.opts, &cancel)
        }
        Start::Hopf { x0, omega, eps } => {
            let hopf = DdeHopfCycle {
                x: x0.clone(),
                omega: *omega,
                eps: *eps,
            };
            dde_hopf_cycle(&mut model, &setup, &hopf, &problem.opts, &cancel)
        }
    };
    match result {
        Ok(cycle) => {
            let mut text = cycle_json(&cycle, problem.ntst, problem.ncol).to_string();
            text.push('\n');
            out(&text, 0)
        }
        Err(e) => engine_error(&e, &shown.to_string()),
    }
}

/// The output object of SPEC section 8l.
fn cycle_json(cycle: &DdeCycle, ntst: usize, ncol: usize) -> Value {
    json!({
        "alpha": cycle.alpha,
        "period": cycle.period,
        "iters": cycle.iters,
        "residual": cycle.residual,
        "ntst": ntst,
        "ncol": ncol,
        "profile": cycle.profile,
    })
}

/// The keys of `options` that the corrector reads.
const OPTION_KEYS: [&str; 3] = ["max_newton", "tol_x", "tol_f"];

/// Refuses every key of `options` that the corrector does not read, so that a key of the
/// `cont` reader is not accepted and then ignored.
fn check_options(options: &Value) -> Result<(), String> {
    let Some(map) = options.as_object() else {
        // `parse_options` names the type error.
        return Ok(());
    };
    match map.keys().find(|k| !OPTION_KEYS.contains(&k.as_str())) {
        Some(key) => Err(format!(
            "options.{key}: dde-cycle reads only max_newton, tol_x and tol_f"
        )),
        None => Ok(()),
    }
}

/// Reads an array of numbers.
fn numbers(value: &Value, key: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("\"{key}\" must be an array of numbers"))?
        .iter()
        .map(|v| number(v, key))
        .collect()
}

/// Reads the `orbit` object: `profile` and `period`, both required.
fn read_orbit(value: &Value) -> Result<Start, String> {
    let obj = value.as_object().ok_or_else(|| {
        "\"orbit\" must be an object with the keys \"profile\" and \"period\"".to_string()
    })?;
    check_keys(obj, &["profile", "period"], "\"orbit\"")?;
    let named = |e: String| format!("\"orbit\": {e}");
    let profile = numbers(required(obj, "profile", "\"orbit\"")?, "profile").map_err(named)?;
    let period = number(required(obj, "period", "\"orbit\"")?, "period").map_err(named)?;
    Ok(Start::Orbit { profile, period })
}

/// Reads the `hopf` object: `x0`, `omega` and `eps`, all required. `eps` has no default.
fn read_hopf(value: &Value) -> Result<Start, String> {
    let obj = value.as_object().ok_or_else(|| {
        "\"hopf\" must be an object with the keys \"x0\", \"omega\" and \"eps\"".to_string()
    })?;
    check_keys(obj, &["x0", "omega", "eps"], "\"hopf\"")?;
    let named = |e: String| format!("\"hopf\": {e}");
    let x0 = numbers(required(obj, "x0", "\"hopf\"")?, "x0").map_err(named)?;
    let omega = number(required(obj, "omega", "\"hopf\"")?, "omega").map_err(named)?;
    let eps = number(required(obj, "eps", "\"hopf\"")?, "eps").map_err(named)?;
    Ok(Start::Hopf { x0, omega, eps })
}

/// Reads the problem object. The parameter names, the length of the profile and `x0` and the
/// ranges of `ntst`, `ncol`, `omega`, `eps` and `period` are checked against the model by the
/// engine.
fn read_problem(value: &Value) -> Result<Problem, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "the problem must be a JSON object".to_string())?;
    check_keys(
        obj,
        &[
            "model", "params", "free", "ntst", "ncol", "orbit", "hopf", "options",
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
    let free = required(obj, "free", "problem")?
        .as_str()
        .ok_or_else(|| "\"free\" must be the name of one declared parameter".to_string())?
        .to_string();
    let ntst = count(required(obj, "ntst", "problem")?, "ntst")?;
    let ncol = count(required(obj, "ncol", "problem")?, "ncol")?;
    let start = match (obj.get("orbit"), obj.get("hopf")) {
        (Some(v), None) => read_orbit(v)?,
        (None, Some(v)) => read_hopf(v)?,
        (Some(_), Some(_)) => {
            return Err("problem: give one of \"orbit\" and \"hopf\", not both".to_string());
        }
        (None, None) => {
            return Err("problem: give one of \"orbit\" and \"hopf\"".to_string());
        }
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
        ntst,
        ncol,
        start,
        opts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Result<Problem, String> {
        read_problem(&serde_json::from_str(text).unwrap())
    }

    const ORBIT: &str = r#"{"model":"m.ode","params":{"tau":1.5},"free":"tau","ntst":4,
        "ncol":3,"orbit":{"profile":[0.0,1.0],"period":6.0},
        "options":{"max_newton":30}}"#;

    const HOPF: &str = r#"{"model":"m.ode","params":{"tau":1.3},"free":"tau","ntst":4,
        "ncol":3,"hopf":{"x0":[0.0],"omega":0.95,"eps":0.01}}"#;

    #[test]
    fn a_good_orbit_problem_reads() {
        let p = read(ORBIT).unwrap();
        assert_eq!(p.model, "m.ode");
        assert_eq!(p.params, [("tau".to_string(), 1.5)]);
        assert_eq!(p.free, "tau");
        assert_eq!((p.ntst, p.ncol), (4, 3));
        assert_eq!(
            p.start,
            Start::Orbit {
                profile: vec![0.0, 1.0],
                period: 6.0
            }
        );
        assert_eq!(p.opts.max_newton, 30);
    }

    #[test]
    fn a_good_hopf_problem_reads() {
        let p = read(HOPF).unwrap();
        assert_eq!(
            p.start,
            Start::Hopf {
                x0: vec![0.0],
                omega: 0.95,
                eps: 0.01
            }
        );
        assert_eq!(p.opts.max_newton, Options::default().max_newton);
    }

    #[test]
    fn exactly_one_start_is_needed() {
        let both = ORBIT.replace(
            "\"options\"",
            "\"hopf\":{\"x0\":[0.0],\"omega\":1,\"eps\":0.1},\"options\"",
        );
        let msg = read(&both).unwrap_err();
        assert!(msg.contains("not both"), "{msg}");
        let neither = HOPF.replace(
            "\"hopf\":{\"x0\":[0.0],\"omega\":0.95,\"eps\":0.01}",
            "\"options\":{}",
        );
        let msg = read(&neither).unwrap_err();
        assert!(msg.contains("one of"), "{msg}");
    }

    #[test]
    fn missing_and_unknown_keys_name_the_key() {
        for key in ["model", "params", "free", "ntst", "ncol"] {
            let mut v: Value = serde_json::from_str(ORBIT).unwrap();
            v.as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        for key in ["profile", "period"] {
            let mut v: Value = serde_json::from_str(ORBIT).unwrap();
            v["orbit"].as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        for key in ["x0", "omega", "eps"] {
            let mut v: Value = serde_json::from_str(HOPF).unwrap();
            v["hopf"].as_object_mut().unwrap().remove(key);
            let msg = read_problem(&v).unwrap_err();
            assert!(msg.contains(&format!("missing key \"{key}\"")), "{msg}");
        }
        let extra = ORBIT.replace("\"ncol\":3,", "\"ncol\":3,\"nn\":3,");
        assert!(read(&extra).unwrap_err().contains("unknown key \"nn\""));
        let inner = ORBIT.replace("\"period\":6.0", "\"period\":6.0,\"eps\":1");
        assert!(read(&inner).unwrap_err().contains("unknown key \"eps\""));
    }

    #[test]
    fn wrong_types_name_the_key() {
        let cases = [
            (
                ORBIT.replace("\"free\":\"tau\"", "\"free\":[\"tau\"]"),
                "\"free\"",
            ),
            (ORBIT.replace("\"ntst\":4", "\"ntst\":-1"), "\"ntst\""),
            (ORBIT.replace("\"ncol\":3", "\"ncol\":2.5"), "\"ncol\""),
            (ORBIT.replace("[0.0,1.0]", "1.0"), "\"profile\""),
            (ORBIT.replace("[0.0,1.0]", "[\"a\"]"), "\"profile\""),
            (ORBIT.replace("6.0", "\"long\""), "\"period\""),
            (
                ORBIT.replace("{\"profile\":[0.0,1.0],\"period\":6.0}", "[1]"),
                "\"orbit\"",
            ),
            (HOPF.replace("0.01", "\"small\""), "\"eps\""),
            (HOPF.replace("[0.0]", "0.0"), "\"x0\""),
            (HOPF.replace("0.95", "null"), "\"omega\""),
            (
                HOPF.replace("{\"x0\":[0.0],\"omega\":0.95,\"eps\":0.01}", "3"),
                "\"hopf\"",
            ),
            (ORBIT.replace("{\"tau\":1.5}", "[1]"), "\"params\""),
        ];
        for (text, key) in cases {
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(key), "{text}: {msg}");
        }
        assert!(read("[1]").unwrap_err().contains("JSON object"));
    }

    #[test]
    fn options_the_corrector_does_not_read_are_refused() {
        for key in [
            "hmax",
            "h0",
            "hmin",
            "max_steps",
            "direction",
            "bounds",
            "detect",
            "mesh_adapt",
        ] {
            let text = ORBIT.replace("\"max_newton\":30", &format!("\"{key}\":1"));
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(&format!("options.{key}")), "{key}: {msg}");
        }
        for key in OPTION_KEYS {
            let text = ORBIT.replace("\"max_newton\":30", &format!("\"{key}\":5"));
            assert!(read(&text).is_ok(), "{key}");
        }
        let unknown = ORBIT.replace("\"max_newton\":30", "\"nope\":1");
        assert!(read(&unknown).unwrap_err().contains("nope"));
    }
}
