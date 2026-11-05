//! `import`: turns SBML text or a MATLAB function into `.ode` text.
//!
//! The importers read text and never run it. The work happens on a thread with its own stack,
//! like the parser, because the expression readers recurse. The result is a JSON string:
//!
//! ```text
//! {"ok":true,"ode":"...","warnings":["..."]}
//! {"ok":false,"error":{"id":"phasecont:import:syntax","kind":"syntax","message":"...",
//!                      "line":3,"col":9}}          (line and col only when known)
//! ```
//!
//! A failed import returns `PC_ERR_MODEL` with the second form, so MATLAB can raise the error
//! with the importer's own id. Bad arguments and panics use the usual codes and set no string.

use pc_import::{FunctionInput, ImportError, Imported};
use serde_json::{Map, Value, json};

use crate::barrier::guarded;
use crate::error::{Error, Result, bad_arg};

/// `bytes` is SBML (XML, UTF-8). `opts_json` must be NULL or `{}`.
pub const FORMAT_SBML: u32 = 1;
/// `bytes` is the text of `func2str(h)` or of a function file. `opts_json` is described in
/// `pckern.h`.
pub const FORMAT_MATLAB: u32 = 2;

/// The importers recurse on nested expressions and nested XML.
const IMPORT_STACK: usize = 8 * 1024 * 1024;

/// The largest input the ABI accepts, before the importer applies its own limit.
pub const MAX_INPUT: usize = 17 * 1024 * 1024;

/// What an import produced: the JSON the caller gets and whether it is an import failure.
pub(crate) struct Outcome {
    pub json: String,
    /// The message for `last_error` when the import failed.
    pub failure: Option<String>,
}

/// Runs the import for `format` on a thread named `pckern-import` and waits for it.
pub(crate) fn run(format: u32, bytes: &[u8], opts: Option<&str>) -> Result<Outcome> {
    if bytes.len() > MAX_INPUT {
        return Err(bad_arg("import input is over 17 MiB"));
    }
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .name("pckern-import".into())
            .stack_size(IMPORT_STACK)
            .spawn_scoped(s, || guarded(|| import(format, bytes, opts)))
            .map_err(|e| Error::Internal(format!("cannot start the import thread: {e}")))?;
        handle
            .join()
            .map_err(|_| Error::Internal("the import thread ended without a result".into()))?
    })
}

fn import(format: u32, bytes: &[u8], opts: Option<&str>) -> Result<Outcome> {
    let result = match format {
        FORMAT_SBML => {
            no_options(opts)?;
            pc_import::from_sbml(bytes)
        }
        FORMAT_MATLAB => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| bad_arg("the MATLAB text is not valid UTF-8"))?;
            let req = Request::parse(opts)?;
            pc_import::from_function(&req.input(text))
        }
        other => {
            return Err(bad_arg(format!(
                "import: format must be 1 (SBML) or 2 (MATLAB), got {other}"
            )));
        }
    };
    Ok(match result {
        Ok(Imported { ode, warnings }) => Outcome {
            json: json!({ "ok": true, "ode": ode, "warnings": warnings }).to_string(),
            failure: None,
        },
        Err(e) => Outcome {
            json: failure_json(&e),
            failure: Some(e.to_string()),
        },
    })
}

fn failure_json(e: &ImportError) -> String {
    let mut err = Map::new();
    err.insert("id".into(), json!(e.error_id()));
    err.insert("kind".into(), json!(e.kind.id()));
    err.insert("message".into(), json!(e.message));
    if let (Some(line), Some(col)) = (e.line, e.col) {
        err.insert("line".into(), json!(line));
        err.insert("col".into(), json!(col));
    }
    json!({ "ok": false, "error": Value::Object(err) }).to_string()
}

fn no_options(opts: Option<&str>) -> Result<()> {
    let Some(text) = opts else { return Ok(()) };
    let v: Value = serde_json::from_str(text)
        .map_err(|e| bad_arg(format!("options: not valid JSON ({e})")))?;
    match v.as_object() {
        Some(m) if m.is_empty() => Ok(()),
        Some(m) => {
            let key = m.keys().next().map_or("", String::as_str);
            Err(bad_arg(format!("{key}: SBML import takes no options")))
        }
        None => Err(bad_arg("options: expected a JSON object")),
    }
}

/// The options of a MATLAB import. Every key is optional here; what the model needs and does not
/// get (initial values, parameter values) is an import error, never a default.
#[derive(Default)]
struct Request {
    captures: Vec<(String, Vec<f64>)>,
    state_names: Option<Vec<String>>,
    params: Vec<(String, f64)>,
    init: Vec<f64>,
    title: Option<String>,
}

impl Request {
    fn parse(opts: Option<&str>) -> Result<Request> {
        let mut out = Request::default();
        let Some(text) = opts else { return Ok(out) };
        let v: Value = serde_json::from_str(text)
            .map_err(|e| bad_arg(format!("options: not valid JSON ({e})")))?;
        let map = v
            .as_object()
            .ok_or_else(|| bad_arg("options: expected a JSON object"))?;
        for (key, val) in map {
            match key.as_str() {
                "captures" => {
                    for item in list(val) {
                        let (name, v) = named(item, "captures")?;
                        out.captures.push((name, numbers(v, "captures")?));
                    }
                }
                "params" => {
                    for item in list(val) {
                        let (name, v) = named(item, "params")?;
                        let x = number(v)
                            .ok_or_else(|| bad_arg(format!("params: `{name}` is not a number")))?;
                        out.params.push((name, x));
                    }
                }
                "state_names" => {
                    let mut names = Vec::new();
                    for item in list(val) {
                        let s = item
                            .as_str()
                            .ok_or_else(|| bad_arg("state_names: expected strings"))?;
                        names.push(s.to_string());
                    }
                    out.state_names = Some(names);
                }
                "init" => out.init = numbers(val, "init")?,
                "title" => {
                    out.title = Some(
                        val.as_str()
                            .ok_or_else(|| bad_arg("title: expected a string"))?
                            .to_string(),
                    );
                }
                other => {
                    return Err(bad_arg(format!(
                        "{other}: unknown key; the keys are \"captures\", \"params\", \
                         \"state_names\", \"init\" and \"title\""
                    )));
                }
            }
        }
        Ok(out)
    }

    fn input<'a>(&'a self, text: &'a str) -> FunctionInput<'a> {
        FunctionInput {
            text,
            captures: &self.captures,
            state_names: self.state_names.as_deref(),
            params: &self.params,
            init: &self.init,
            title: self.title.as_deref(),
        }
    }
}

/// An array as its items. `jsonencode` writes a one-element list as the bare item, so a lone
/// value counts as a list of one. `null` counts as an empty list.
fn list(v: &Value) -> Vec<&Value> {
    match v {
        Value::Array(a) => a.iter().collect(),
        Value::Null => Vec::new(),
        other => vec![other],
    }
}

/// An object `{"name": ..., "value": ...}`.
fn named<'a>(item: &'a Value, what: &str) -> Result<(String, &'a Value)> {
    let obj = item.as_object().ok_or_else(|| {
        bad_arg(format!(
            "{what}: expected objects with \"name\" and \"value\""
        ))
    })?;
    if let Some(k) = obj.keys().find(|k| *k != "name" && *k != "value") {
        return Err(bad_arg(format!("{what}: unknown key `{k}`")));
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_arg(format!("{what}: an entry has no string \"name\"")))?;
    let value = obj
        .get("value")
        .ok_or_else(|| bad_arg(format!("{what}: `{name}` has no \"value\"")))?;
    Ok((name.to_string(), value))
}

/// A finite number. `jsonencode` writes NaN and Inf as `null`, which is not a number here.
fn number(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

fn numbers(v: &Value, what: &str) -> Result<Vec<f64>> {
    list(v)
        .into_iter()
        .map(|x| number(x).ok_or_else(|| bad_arg(format!("{what}: expected finite numbers"))))
        .collect()
}
