//! The problem file and its options, read by hand from a `serde_json::Value` so that
//! every error can name the key that is wrong. Unknown keys are errors.
//!
//! `pc-cli` reads a whole problem file with [`Mode::Cli`]. The C ABI uses [`Mode::Abi`],
//! where the state and the parameters come as buffers and the keys `u0` and `params`
//! are errors.

use pc_engine::{
    CurveSetup, EpSetup, Evaluator, HOPF_CURVE_LABELS, LP_CURVE_LABELS, Options, RunKind, RunSpec,
};
use pc_ir::IrModel;
use pc_ode::{Model, ModelInfo};
use serde_json::{Map, Value};

/// What a run follows: an equilibrium branch, or the curve of fold (LP) or Hopf points of two
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ep,
    Lp,
    Hopf,
}

impl Kind {
    /// The `type` string of the problem JSON.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Ep => "EP",
            Kind::Lp => "LP",
            Kind::Hopf => "H",
        }
    }

    /// Codim-2 labels in the order of `Point::codim2`; empty for an equilibrium branch.
    pub fn labels(self) -> &'static [&'static str] {
        match self {
            Kind::Ep => &[],
            Kind::Lp => &LP_CURVE_LABELS,
            Kind::Hopf => &HOPF_CURVE_LABELS,
        }
    }
}

/// A problem file after validation.
#[derive(Debug)]
pub struct Problem {
    /// The start of an `Ep` run. For a curve, `setup.u0` and `setup.params` are the start of
    /// the curve and `setup.free` is the first of its two parameters.
    pub setup: EpSetup,
    pub opts: Options,
    pub switch: Option<String>,
    pub kind: Kind,
    /// The two parameters of a curve run, `[a1, a2]`, as positions among the parameters.
    pub curve_free: Option<[usize; 2]>,
}

impl Problem {
    /// The engine's run kind, with the start taken from `setup` as it is now.
    pub fn run_kind(&self) -> RunKind {
        let curve = |free: [usize; 2]| {
            CurveSetup::new(free, self.setup.params.clone(), self.setup.u0.clone())
        };
        match (self.kind, self.curve_free) {
            (Kind::Lp, Some(free)) => RunKind::LpCurve(curve(free)),
            (Kind::Hopf, Some(free)) => RunKind::HopfCurve(curve(free)),
            _ => RunKind::Ep,
        }
    }

    /// The run to start, with the start taken from `setup` as it is now.
    pub fn into_spec(self, model: Model, info: ModelInfo, evaluator: Evaluator) -> RunSpec {
        let kind = self.run_kind();
        match kind {
            RunKind::Ep => {
                let mut spec = RunSpec::new(model, info, evaluator, self.setup, self.opts);
                spec.switch_at = self.switch;
                spec
            }
            curve => RunSpec::curve(model, info, evaluator, curve, self.opts),
        }
    }
}

/// Who is reading the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The command line: `u0` and `params` may appear in the JSON.
    Cli,
    /// The C ABI: `u0` and `params` are passed as buffers, so the JSON must not have them.
    /// `setup.u0` and `setup.params` hold the model defaults until the caller overwrites them.
    Abi,
}

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
pub fn check_switch(name: &str) -> bool {
    match name.find(|c: char| c.is_ascii_digit()) {
        Some(cut) if cut > 0 => {
            let digits = &name[cut..];
            digits.bytes().all(|b| b.is_ascii_digit())
                && digits.parse::<usize>().is_ok_and(|n| n >= 1)
        }
        _ => false,
    }
}

/// Reads and checks a problem object against the model.
pub fn parse_problem(v: &Value, ir: &IrModel, mode: Mode) -> Result<Problem, String> {
    let top = v
        .as_object()
        .ok_or_else(|| "$: expected an object".to_string())?;
    only_keys(
        top,
        &[
            "type",
            "free",
            "u0",
            "params",
            "options",
            "switch",
            "start_is_special",
        ],
        "",
    )?;
    let kind = match top.get("type") {
        None => return Err("type: missing".into()),
        Some(Value::String(s)) if s == "EP" => Kind::Ep,
        Some(Value::String(s)) if s == "LP" => Kind::Lp,
        Some(Value::String(s)) if s == "H" => Kind::Hopf,
        Some(_) => return Err("type: expected \"EP\", \"LP\" or \"H\"".into()),
    };
    let (free, curve_free) = match (kind, top.get("free")) {
        (_, None) => return Err("free: missing".into()),
        (Kind::Ep, Some(Value::String(name))) => (
            param_index(ir, name).ok_or_else(|| format!("free: unknown parameter \"{name}\""))?,
            None,
        ),
        (Kind::Ep, Some(_)) => return Err("free: expected a string".into()),
        (_, Some(Value::Array(names))) if names.len() == 2 => {
            let mut at = [0; 2];
            for (i, name) in names.iter().enumerate() {
                let name = name
                    .as_str()
                    .ok_or_else(|| format!("free[{i}]: expected a string"))?;
                at[i] = param_index(ir, name)
                    .ok_or_else(|| format!("free[{i}]: unknown parameter \"{name}\""))?;
            }
            if at[0] == at[1] {
                return Err("free: the two parameters must differ".into());
            }
            (at[0], Some(at))
        }
        (_, Some(_)) => return Err("free: expected two parameter names".into()),
    };

    if mode == Mode::Abi {
        if top.contains_key("u0") {
            return Err("u0: not allowed here; pass the state as a buffer".into());
        }
        if top.contains_key("params") {
            return Err("params: not allowed here; pass the parameters as a buffer".into());
        }
    }

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

    let start_is_special = match top.get("start_is_special") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("start_is_special: expected true or false".into()),
    };

    if kind != Kind::Ep {
        if switch.is_some() {
            return Err("switch: only an EP run can switch branches".into());
        }
        if top.contains_key("start_is_special") {
            return Err("start_is_special: only an EP run starts at a special point".into());
        }
    }

    let mut setup = EpSetup::new(free, params, u0);
    setup.start_is_special = start_is_special;
    Ok(Problem {
        setup,
        opts,
        switch,
        kind,
        curve_free,
    })
}

/// Reads the `options` object.
pub fn parse_options(v: &Value) -> Result<Options, String> {
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
    use pc_ode::{MapLoader, SourceMap, load};
    use serde_json::json;

    /// Two states, one parameter, nonzero initial values.
    fn model() -> IrModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", "x'=a-x\ny'=x-y\npar a=2\ninit x=0.5,y=1.5\ndone\n");
        let loaded = load(top, &mut map, &mut MapLoader::new());
        pc_ir::lower(&loaded.expanded, &loaded.info).expect("the test model lowers")
    }

    fn base() -> Value {
        json!({"type": "EP", "free": "a"})
    }

    fn with(key: &str, v: Value) -> Value {
        let mut b = base();
        b[key] = v;
        b
    }

    /// Two parameters, for curves.
    fn model2() -> IrModel {
        let mut map = SourceMap::new();
        let top = map.add(
            "t.ode",
            "x'=a-x\ny'=b*x-y\npar a=2,b=3\ninit x=0.5,y=1.5\ndone\n",
        );
        let loaded = load(top, &mut map, &mut MapLoader::new());
        pc_ir::lower(&loaded.expanded, &loaded.info).expect("the test model lowers")
    }

    #[test]
    fn curve_types_take_two_names_and_build_a_curve_run() {
        let ir = model2();
        for (ty, kind) in [("LP", Kind::Lp), ("H", Kind::Hopf)] {
            let v = json!({"type": ty, "free": ["B", "a"]});
            let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
            assert_eq!(p.kind, kind);
            assert_eq!(p.curve_free, Some([1, 0]));
            let run = p.run_kind();
            let (RunKind::LpCurve(cs) | RunKind::HopfCurve(cs)) = run else {
                panic!("expected a curve kind");
            };
            assert_eq!(cs.free, [1, 0]);
            assert_eq!(cs.u0, ir.init);
            assert_eq!(cs.params, ir.par_values);
        }
        let p = parse_problem(&json!({"type": "EP", "free": "a"}), &ir, Mode::Abi).unwrap();
        assert_eq!(p.kind, Kind::Ep);
        assert_eq!(p.curve_free, None);
        assert!(matches!(p.run_kind(), RunKind::Ep));
    }

    #[test]
    fn curve_types_check_free_switch_and_start_is_special() {
        let ir = model2();
        let cases = [
            (
                json!({"type": "LP", "free": "a"}),
                "free: expected two parameter names",
            ),
            (
                json!({"type": "LP", "free": ["a"]}),
                "free: expected two parameter names",
            ),
            (
                json!({"type": "LP", "free": ["a", 1]}),
                "free[1]: expected a string",
            ),
            (
                json!({"type": "LP", "free": ["a", "A"]}),
                "free: the two parameters must differ",
            ),
            (
                json!({"type": "H", "free": ["a", "zz"]}),
                "free[1]: unknown parameter \"zz\"",
            ),
            (
                json!({"type": "EP", "free": ["a", "b"]}),
                "free: expected a string",
            ),
            (
                json!({"type": "LP", "free": ["a", "b"], "switch": "BP1"}),
                "switch: only an EP run can switch branches",
            ),
            (
                json!({"type": "H", "free": ["a", "b"], "start_is_special": true}),
                "start_is_special: only an EP run starts at a special point",
            ),
            (
                json!({"type": "BP", "free": "a"}),
                "type: expected \"EP\", \"LP\" or \"H\"",
            ),
        ];
        for (v, msg) in cases {
            assert_eq!(parse_problem(&v, &ir, Mode::Abi).unwrap_err(), msg, "{v}");
        }
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
    fn abi_rejects_u0_and_params() {
        let ir = model();
        let e = parse_problem(&with("u0", json!([1.0, 2.0])), &ir, Mode::Abi).unwrap_err();
        assert_eq!(e, "u0: not allowed here; pass the state as a buffer");
        let e = parse_problem(&with("params", json!({"a": 1.0})), &ir, Mode::Abi).unwrap_err();
        assert_eq!(
            e,
            "params: not allowed here; pass the parameters as a buffer"
        );
    }

    #[test]
    fn abi_defaults_match_cli_defaults() {
        let ir = model();
        let cli = parse_problem(&base(), &ir, Mode::Cli).unwrap();
        let abi = parse_problem(&base(), &ir, Mode::Abi).unwrap();
        assert_eq!(abi.setup.u0, cli.setup.u0);
        assert_eq!(abi.setup.params, cli.setup.params);
        assert_eq!(abi.setup.u0, ir.init);
        assert_eq!(abi.setup.params, ir.par_values);
    }

    #[test]
    fn cli_accepts_u0_and_params() {
        let ir = model();
        let mut v = with("u0", json!([3.0, 4.0]));
        v["params"] = json!({"a": 7.0});
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        assert_eq!(p.setup.u0, vec![3.0, 4.0]);
        assert_eq!(p.setup.params, vec![7.0]);
    }

    #[test]
    fn start_is_special_is_a_bool() {
        let ir = model();
        for mode in [Mode::Cli, Mode::Abi] {
            let p = parse_problem(&base(), &ir, mode).unwrap();
            assert!(!p.setup.start_is_special);
            let p = parse_problem(&with("start_is_special", json!(true)), &ir, mode).unwrap();
            assert!(p.setup.start_is_special);
            let p = parse_problem(&with("start_is_special", json!(false)), &ir, mode).unwrap();
            assert!(!p.setup.start_is_special);
            let e = parse_problem(&with("start_is_special", json!("yes")), &ir, mode).unwrap_err();
            assert_eq!(e, "start_is_special: expected true or false");
        }
    }

    /// Doubles that MATLAB writes with `%.17g` must read back bit for bit.
    #[test]
    fn options_numbers_round_trip_bitwise() {
        let mut values: Vec<f64> = vec![
            1e-10,
            8.0 / 3.0,
            470.0 / 19.0,
            0.1,
            1.0 / 3.0,
            5e-324,
            f64::MAX,
            2.6666666666666665,
        ];
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        while values.len() < 1008 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits(state);
            if x.is_finite() {
                values.push(x);
            }
        }
        for &v in &values {
            let text = format!("{v:.17e}");
            // bounds [v, MAX] or [-MAX, v]
            let b = if v < f64::MAX {
                format!("[{text}, {:.17e}]", f64::MAX)
            } else {
                format!("[{:.17e}, {text}]", -f64::MAX)
            };
            let json = format!("{{\"bounds\": {b}}}");
            let o = parse_options(&serde_json::from_str(&json).unwrap()).unwrap();
            let got = if v < f64::MAX { o.bounds.0 } else { o.bounds.1 };
            assert_eq!(got.to_bits(), v.to_bits(), "bounds {text}");
            if v != 5e-324 && v != 0.0 {
                let a = v.abs();
                let json = format!("{{\"h0\": {a:.17e}}}");
                let o = parse_options(&serde_json::from_str(&json).unwrap()).unwrap();
                assert_eq!(o.h0.to_bits(), a.to_bits(), "h0 {a:.17e}");
            }
        }
    }
}
