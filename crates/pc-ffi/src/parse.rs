//! Parsing a model on its own thread and shaping the diagnostics for the C ABI.

use pc_ode::{MapLoader, SourceMap, load};
use serde_json::{Value, json};

use crate::barrier::guarded;
use crate::error::{Error, Result};
use crate::json::diag;
use crate::model::Names;
use crate::sha256::sha256_hex;

/// Diagnostics returned across the ABI. The rest are counted in `omitted`.
pub const MAX_ABI_DIAGNOSTICS: usize = 1000;

/// The largest model text the ABI accepts.
pub const MAX_MODEL_TEXT: usize = 16 * 1024 * 1024;

/// The parser and the lowering recurse on nested expressions.
const PARSE_STACK: usize = 8 * 1024 * 1024;

/// The name the model text goes by in diagnostics. MATLAB swaps in the real path.
const FILE_NAME: &str = "model.ode";

pub(crate) struct Parsed {
    pub model: pc_ode::Model,
    pub info: pc_ode::ModelInfo,
    pub names: Names,
    pub ir: pc_ir::IrModel,
}

/// The diagnostics come back whether or not the model loaded.
pub(crate) struct Outcome {
    pub diag: Value,
    pub result: Result<Parsed>,
}

/// Parses `text` on a thread named `pckern-parse` and waits for it.
pub(crate) fn parse_model(text: String) -> Result<Outcome> {
    let t = std::thread::Builder::new()
        .name("pckern-parse".into())
        .stack_size(PARSE_STACK)
        .spawn(move || guarded(|| Ok(parse_inline(&text))))
        .map_err(|e| Error::Internal(format!("cannot start the parse thread: {e}")))?;
    t.join()
        .map_err(|_| Error::Internal("the parse thread ended without a result".into()))?
}

/// Keeps the first `MAX_ABI_DIAGNOSTICS` entries and records how many were dropped.
fn cap(v: &mut Value) {
    let mut omitted = 0;
    if let Some(list) = v.get_mut("diagnostics").and_then(Value::as_array_mut) {
        omitted = list.len().saturating_sub(MAX_ABI_DIAGNOSTICS);
        list.truncate(MAX_ABI_DIAGNOSTICS);
    }
    v["omitted"] = json!(omitted);
}

/// The first error as `line:col: message`.
fn first_error(v: &Value) -> String {
    v["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|d| d["severity"] == "error")
        .map(|d| {
            format!(
                "{}:{}: {}",
                d["line"],
                d["col"],
                d["message"].as_str().unwrap_or("error")
            )
        })
        .unwrap_or_else(|| "the model has errors".to_string())
}

/// The `@` options as (lower-case name, value) pairs in the order the names first appear, with the
/// last value winning. XPPAUT 8.0 ignores an option that has a blank around `=`, so the parser
/// warns about it and this list leaves it out. Unknown names stay. The ABI parses one text, so
/// every span points into `text`.
fn option_pairs(model: &pc_ode::Model, text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for stmt in &model.statements {
        let pc_ode::StmtKind::Options { entries } = &stmt.kind else {
            continue;
        };
        for e in entries {
            let gap = text.get(e.name.span.end..e.value_span.start).unwrap_or("");
            if gap.contains([' ', '\t']) {
                continue;
            }
            match out.iter_mut().find(|(k, _)| *k == e.name.key) {
                Some(slot) => slot.1 = e.value.clone(),
                None => out.push((e.name.key.clone(), e.value.clone())),
            }
        }
    }
    out
}

fn parse_inline(text: &str) -> Outcome {
    let mut map = SourceMap::new();
    let file = map.add(FILE_NAME, text);
    // The ABI gets text only, so an `#include` cannot be read and is reported as an error.
    let loaded = load(file, &mut map, &mut MapLoader::new());
    let mut diag = diag::value(FILE_NAME, &map, &loaded);
    let errors = diag["summary"]["errors"].as_u64().unwrap_or(0);
    let first = (errors > 0).then(|| first_error(&diag));
    cap(&mut diag);
    if let Some(msg) = first {
        return Outcome {
            diag,
            result: Err(Error::Model(msg)),
        };
    }
    let result = match pc_ir::lower(&loaded.expanded, &loaded.info) {
        Ok(ir) => Ok(Parsed {
            names: Names {
                states: ir.state_names.clone(),
                parameters: ir.param_names.clone(),
                aux: ir.aux_names.clone(),
                init: ir.init.clone(),
                par_values: ir.par_values.clone(),
                autonomous: ir.autonomous,
                source_sha256: sha256_hex(text.as_bytes()),
                options: option_pairs(&loaded.expanded, text),
            },
            ir,
            model: loaded.expanded,
            info: loaded.info,
        }),
        Err(errs) => {
            let msgs: Vec<&str> = errs.iter().map(|e| e.message.as_str()).collect();
            Err(Error::Model(msgs.join("; ")))
        }
    };
    Outcome { diag, result }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_keeps_the_first_entries_and_counts_the_rest() {
        let mut v =
            json!({"diagnostics": (0..1200).collect::<Vec<_>>(), "summary": {"errors": 1200}});
        cap(&mut v);
        assert_eq!(v["diagnostics"].as_array().unwrap().len(), 1000);
        assert_eq!(v["diagnostics"][999], 999);
        assert_eq!(v["omitted"], 200);
        assert_eq!(v["summary"]["errors"], 1200);
        let mut v = json!({"diagnostics": [1, 2]});
        cap(&mut v);
        assert_eq!(v["omitted"], 0);
    }

    #[test]
    fn a_good_model_keeps_its_names() {
        let o = parse_model("par a=2\ninit x=3\nx'=-a*x\ndone\n".into()).unwrap();
        let p = o.result.unwrap();
        assert_eq!(p.names.states, ["x"]);
        assert_eq!(p.names.parameters, ["a"]);
        assert_eq!(p.names.par_values, [2.0]);
        assert_eq!(p.names.init, [3.0]);
        assert!(p.names.autonomous);
    }

    #[test]
    fn an_error_names_its_position() {
        let o = parse_model("x'=-x\nfoo(\ndone\n".into()).unwrap();
        let msg = o.result.err().unwrap().to_string();
        let first = &o.diag["diagnostics"][0];
        assert!(
            msg.starts_with(&format!("{}:{}: ", first["line"], first["col"])),
            "{msg}"
        );
    }
}
