//! `pc-cli latex`: render a figure or a table from a JSON document.

use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_latex::{Kind, render_value};
use serde_json::Value;

use crate::out;

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

fn fail(message: &str) -> ExitCode {
    to_stderr(&format!("pc-cli: {message}\n"));
    ExitCode::from(1)
}

/// Renders `file` as `kind`. `standalone` sets the document's `standalone` key. Nothing
/// is written unless the whole render succeeds.
pub fn latex(kind: Kind, file: &Path, target: Option<&Path>, standalone: bool) -> ExitCode {
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => return fail(&format!("cannot read {}: {e}", file.display())),
    };
    let mut doc: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return fail(&format!(
                "badInput: invalid JSON in {}: {e}",
                file.display()
            ));
        }
    };
    if standalone {
        match doc.as_object_mut() {
            Some(obj) => {
                obj.insert("standalone".to_string(), Value::Bool(true));
            }
            None => return fail("badInput: the document must be a JSON object"),
        }
    }
    let rendered = match render_value(kind, &doc) {
        Ok(r) => r,
        Err(e) => return fail(&format!("{}: {e}", e.kind().as_str())),
    };
    let Some(target) = target else {
        if !rendered.data_files.is_empty() {
            return fail("external data needs --out, so the data files have a folder to go to");
        }
        return out(&rendered.tex, 0);
    };
    // The renderer allows only `A-Za-z0-9_-` in the base name, so a data file name
    // cannot leave the folder of the output file.
    let dir = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    for (name, content) in &rendered.data_files {
        let path = dir.join(name);
        if let Err(e) = std::fs::write(&path, content) {
            return fail(&format!("cannot write {}: {e}", path.display()));
        }
    }
    match std::fs::write(target, &rendered.tex) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(&format!("cannot write {}: {e}", target.display())),
    }
}
