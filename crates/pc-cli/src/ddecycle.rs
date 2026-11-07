//! The `dde-cycle` command: correct a periodic orbit of a model with `delay(x,tau)` calls
//! by collocation, from a given orbit or from a Hopf point (SPEC section 8l), and with the
//! optional `floquet` key print the leading Floquet multipliers of the result (section 8m).
//! With `"branch": true` the corrected orbit is instead continued in the free parameter and
//! the command prints the points of the branch (section 8n).
//!
//! The problem file is read by hand from a `serde_json::Value`, as for `dde-hopf`, and the
//! model is loaded through the same `load_dde`. Without `branch` the corrector reads only
//! `max_newton`, `tol_x` and `tol_f` from `options`, so the other keys of the `cont` reader
//! are refused by name instead of being accepted and ignored. With `branch` the options go
//! through the `cont` reader, except `detect` and `mesh_adapt`, which the continuation does
//! not use.

use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use pc_engine::{
    DdeCycle, DdeCycleBranch, DdeCycleSetup, DdeCycleStart, DdeFloquet, DdeFloquetOptions,
    DdeHopfCycle, DdeModel, EngineError, Evaluator, Options, StopReason, build_dde,
    dde_continue_cycle, dde_correct_cycle, dde_floquet_multipliers, dde_hopf_cycle,
};
use pckern::json::problem::parse_options;
use serde_json::{Value, json};

use crate::cont::stop_name;
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
    /// The multipliers to print, when the problem asks for them.
    floquet: Option<DdeFloquetOptions>,
    /// Whether to continue the orbit in the free parameter instead of printing it.
    branch: bool,
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
    if problem.branch {
        return run_branch(&mut model, setup, &problem, &cancel, &shown.to_string());
    }
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
    let cycle = match result {
        Ok(c) => c,
        Err(e) => return engine_error(&e, &shown.to_string()),
    };
    let mut value = cycle_json(&cycle, problem.ntst, problem.ncol);
    if let Some(options) = &problem.floquet {
        // The multipliers belong to the orbit that the corrector returned, at its parameter.
        let mut at = setup;
        at.params[free] = cycle.alpha;
        let orbit = DdeCycleStart {
            profile: cycle.profile.clone(),
            period: cycle.period,
        };
        match dde_floquet_multipliers(&mut model, &at, &orbit, options, &cancel) {
            Ok(f) => add_floquet(&mut value, &f),
            // The orbit passed the checks of the corrector, so what is left is the options.
            Err(EngineError::BadProblem(msg)) => {
                return engine_error(
                    &EngineError::BadProblem(format!("floquet: {msg}")),
                    &shown.to_string(),
                );
            }
            Err(e) => return engine_error(&e, &shown.to_string()),
        }
    }
    let mut text = value.to_string();
    text.push('\n');
    out(&text, 0)
}

/// `branch`: continues the orbit in the free parameter and prints the points. An orbit start
/// is corrected by the continuation at the value of `free` that `params` gives. A Hopf start
/// first goes through the Hopf corrector, which finds that value.
fn run_branch(
    model: &mut DdeModel,
    mut setup: DdeCycleSetup,
    problem: &Problem,
    cancel: &AtomicBool,
    shown: &str,
) -> ExitCode {
    let orbit = match &problem.start {
        Start::Orbit { profile, period } => DdeCycleStart {
            profile: profile.clone(),
            period: *period,
        },
        Start::Hopf { x0, omega, eps } => {
            let hopf = DdeHopfCycle {
                x: x0.clone(),
                omega: *omega,
                eps: *eps,
            };
            match dde_hopf_cycle(model, &setup, &hopf, &problem.opts, cancel) {
                Ok(c) => {
                    setup.params[setup.free] = c.alpha;
                    DdeCycleStart {
                        profile: c.profile,
                        period: c.period,
                    }
                }
                Err(e) => return engine_error(&e, shown),
            }
        }
    };
    let branch = match dde_continue_cycle(model, &setup, &orbit, &problem.opts, cancel) {
        Ok(b) => b,
        Err(e) => return engine_error(&e, shown),
    };
    // As `cont` does, a run that stops on an error still prints what it found, and the
    // message goes to stderr.
    if let StopReason::Error(e) = &branch.stop
        && !matches!(e, EngineError::Cancelled)
    {
        to_stderr(&format!("pc-cli: {e}\n"));
    }
    let mut text = branch_json(&branch, problem.ntst, problem.ncol).to_string();
    text.push('\n');
    out(&text, 0)
}

/// Adds `multipliers` (`[re, im]` pairs) and `trivial_error` to the output object.
fn add_floquet(value: &mut Value, f: &DdeFloquet) {
    let pairs: Vec<Value> = f
        .multipliers
        .iter()
        .map(|&(re, im)| json!([re, im]))
        .collect();
    value["multipliers"] = Value::Array(pairs);
    value["trivial_error"] = json!(f.trivial_error);
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

/// The output object of SPEC section 8n.
fn branch_json(branch: &DdeCycleBranch, ntst: usize, ncol: usize) -> Value {
    let points: Vec<Value> = branch
        .points
        .iter()
        .map(|p| {
            json!({
                "alpha": p.alpha,
                "period": p.period,
                "profile": p.profile,
                "h": p.h,
                "iters": p.newton_iters,
            })
        })
        .collect();
    json!({
        "ntst": ntst,
        "ncol": ncol,
        "stop": stop_name(&branch.stop),
        "points": points,
        "warnings": branch.warnings,
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

/// Refuses the keys of `options` that the continuation of a cycle does not use, so that they
/// are not accepted and then ignored. The rest goes to the `cont` reader.
fn check_branch_options(options: &Value) -> Result<(), String> {
    let Some(map) = options.as_object() else {
        // `parse_options` names the type error.
        return Ok(());
    };
    for key in ["detect", "mesh_adapt"] {
        if map.contains_key(key) {
            return Err(format!(
                "options.{key}: the continuation of a cycle does not use it"
            ));
        }
    }
    Ok(())
}

/// Reads `branch`: a JSON boolean, false when the key is absent.
fn read_branch(obj: &serde_json::Map<String, Value>) -> Result<bool, String> {
    match obj.get("branch") {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err("\"branch\" must be true or false".to_string()),
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

/// Reads the `floquet` object: `nhist` and `count`, both required integers of at least 1.
/// Their upper limits depend on the model and are checked by the engine.
fn read_floquet(value: &Value) -> Result<DdeFloquetOptions, String> {
    let obj = value.as_object().ok_or_else(|| {
        "\"floquet\" must be an object with the keys \"nhist\" and \"count\"".to_string()
    })?;
    check_keys(obj, &["nhist", "count"], "\"floquet\"")?;
    let named = |e: String| format!("\"floquet\": {e}");
    let at_least_one = |key: &str| -> Result<usize, String> {
        let n = count(required(obj, key, "\"floquet\"")?, key).map_err(named)?;
        if n < 1 {
            return Err(named(format!("\"{key}\" must be at least 1")));
        }
        Ok(n)
    };
    Ok(DdeFloquetOptions {
        nhist: at_least_one("nhist")?,
        count: at_least_one("count")?,
    })
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
            "model", "params", "free", "ntst", "ncol", "orbit", "hopf", "options", "floquet",
            "branch",
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
    let branch = read_branch(obj)?;
    if branch && obj.contains_key("floquet") {
        return Err("\"floquet\": not available with \"branch\" (SPEC section 8n)".to_string());
    }
    let opts = match obj.get("options") {
        None => Options::default(),
        Some(v) => {
            if branch {
                check_branch_options(v)?;
            } else {
                check_options(v)?;
            }
            parse_options(v)?
        }
    };
    let floquet = obj.get("floquet").map(read_floquet).transpose()?;
    Ok(Problem {
        model,
        params,
        free,
        ntst,
        ncol,
        start,
        opts,
        floquet,
        branch,
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
    fn the_floquet_key_is_optional_and_reads_both_members() {
        assert_eq!(read(ORBIT).unwrap().floquet, None);
        let with = ORBIT.replace(
            "\"options\"",
            "\"floquet\":{\"nhist\":16,\"count\":4},\"options\"",
        );
        assert_eq!(
            read(&with).unwrap().floquet,
            Some(DdeFloquetOptions {
                nhist: 16,
                count: 4
            })
        );
        // It also works next to a Hopf start.
        let hopf = HOPF.replace("\"hopf\"", "\"floquet\":{\"nhist\":8,\"count\":2},\"hopf\"");
        assert_eq!(read(&hopf).unwrap().floquet.map(|f| f.count), Some(2));
    }

    #[test]
    fn a_malformed_floquet_key_names_floquet() {
        let at =
            |body: &str| ORBIT.replace("\"options\"", &format!("\"floquet\":{body},\"options\""));
        let bad = [
            "true",
            "null",
            "[16,4]",
            "\"nhist\"",
            "{}",
            "{\"nhist\":16}",
            "{\"count\":4}",
            "{\"nhist\":16,\"count\":4,\"tol\":1}",
            "{\"nhist\":16.5,\"count\":4}",
            "{\"nhist\":16,\"count\":2.0e0}",
            "{\"nhist\":0,\"count\":4}",
            "{\"nhist\":16,\"count\":0}",
            "{\"nhist\":-1,\"count\":4}",
            "{\"nhist\":16,\"count\":\"4\"}",
        ];
        for body in bad {
            let msg = read(&at(body)).unwrap_err();
            assert!(msg.contains("\"floquet\""), "{body}: {msg}");
        }
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

    #[test]
    fn the_branch_key_is_optional_and_a_boolean() {
        assert!(!read(ORBIT).unwrap().branch);
        let on = ORBIT.replace("\"options\"", "\"branch\":true,\"options\"");
        assert!(read(&on).unwrap().branch);
        let off = ORBIT.replace("\"options\"", "\"branch\":false,\"options\"");
        assert!(!read(&off).unwrap().branch);
        for body in ["null", "\"yes\"", "1", "[true]", "{}"] {
            let text = ORBIT.replace("\"options\"", &format!("\"branch\":{body},\"options\""));
            let msg = read(&text).unwrap_err();
            assert!(msg.contains("\"branch\""), "{body}: {msg}");
        }
        // It also works next to a Hopf start.
        let hopf = HOPF.replace("\"hopf\"", "\"branch\":true,\"hopf\"");
        assert!(read(&hopf).unwrap().branch);
    }

    #[test]
    fn a_branch_run_reads_the_continuation_options() {
        let text = ORBIT.replace(
            "\"max_newton\":30",
            "\"max_newton\":30,\"hmax\":0.2,\"max_steps\":50,\"bounds\":[0.5,2.0],\"direction\":-1",
        );
        // Without `branch` the extra keys are refused.
        assert!(read(&text).unwrap_err().contains("options."));
        let on = text.replace("\"options\"", "\"branch\":true,\"options\"");
        let p = read(&on).unwrap();
        assert_eq!(p.opts.hmax, 0.2);
        assert_eq!(p.opts.max_steps, 50);
        assert_eq!(p.opts.bounds, (0.5, 2.0));
        assert_eq!(p.opts.direction, -1.0);
        assert_eq!(p.opts.max_newton, 30);
    }

    #[test]
    fn a_branch_run_refuses_what_the_continuation_does_not_use() {
        let on = ORBIT.replace("\"options\"", "\"branch\":true,\"options\"");
        for (body, key) in [
            ("\"detect\":{\"lp\":true}", "options.detect"),
            ("\"detect\":{}", "options.detect"),
            ("\"mesh_adapt\":3", "options.mesh_adapt"),
        ] {
            let text = on.replace("\"max_newton\":30", body);
            let msg = read(&text).unwrap_err();
            assert!(msg.contains(key), "{body}: {msg}");
        }
        let with_floquet = on.replace(
            "\"options\"",
            "\"floquet\":{\"nhist\":8,\"count\":2},\"options\"",
        );
        assert!(read(&with_floquet).unwrap_err().contains("\"floquet\""));
        // The cont reader still checks the values.
        let bad = on.replace("\"max_newton\":30", "\"hmax\":-1");
        assert!(read(&bad).unwrap_err().contains("options.hmax"));
    }
}
