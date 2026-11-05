//! `latex_render`: a figure or table document in, TeX and data files out.
//!
//! The renderer is `pc-latex`. A rejected document is not a failure of the call, so the
//! result string carries the error object and the call returns `PC_ERR_BAD_PROBLEM`, the
//! way a bad ensemble job does. The MATLAB layer turns the object into
//! `phasecont:latex:<kind>`.

use pc_latex::{Kind, render};
use serde_json::{Value, json};

use crate::error::{Error, Result};

pub const PGFPLOTS: u32 = 1;
pub const TABLE: u32 = 2;

/// The result JSON and, for a rejected document, the message for `last_error`.
pub struct Outcome {
    pub json: String,
    pub failure: Option<String>,
}

fn kind_of(kind: u32) -> Result<Kind> {
    match kind {
        PGFPLOTS => Ok(Kind::Pgfplots),
        TABLE => Ok(Kind::Table),
        other => Err(Error::BadArg(format!(
            "latex kind {other} is not 1 (pgfplots) or 2 (table)"
        ))),
    }
}

pub fn run(kind: u32, doc: &str) -> Result<Outcome> {
    let kind = kind_of(kind)?;
    Ok(match render(kind, doc) {
        Ok(rendered) => {
            let mut v: Value = serde_json::from_str(&rendered.to_json())
                .map_err(|e| Error::Internal(format!("latex result: {e}")))?;
            v["ok"] = Value::Bool(true);
            Outcome {
                json: v.to_string(),
                failure: None,
            }
        }
        Err(e) => {
            let name = e.kind().as_str();
            let message = e.message().to_string();
            Outcome {
                json: json!({
                    "ok": false,
                    "error": {
                        "id": format!("phasecont:latex:{name}"),
                        "kind": name,
                        "message": message,
                    },
                })
                .to_string(),
                failure: Some(message),
            }
        }
    })
}
