//! LaTeX output for PhaseCont: pgfplots figures and booktabs tables of special points.
//!
//! Everything here is a pure function from a JSON document to text. The crate reads no
//! files and writes none. A figure that keeps its data in separate files returns those
//! files in [`Rendered::data_files`], and the caller decides where to write them.
//!
//! Number formatting has no default: every document names its `digits`, the count of
//! significant digits. A value that is not a finite number is an error that names the
//! series and the point, and it is never dropped.

mod error;
mod escape;
mod figure;
mod input;
mod num;
mod table;

pub use error::{ErrorKind, LatexError};
pub use escape::escape;
pub use num::format_significant;

use input::Object;
use serde_json::{Value, json};

/// What a render produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    /// The LaTeX text: a fragment for `\input`, or a whole `standalone` document.
    pub tex: String,
    /// Data files the text refers to, as (file name, content). Empty for inline data.
    pub data_files: Vec<(String, String)>,
}

impl Rendered {
    /// `{"tex": "...", "data_files": [{"name": "...", "content": "..."}]}`.
    pub fn to_json(&self) -> String {
        let files: Vec<Value> = self
            .data_files
            .iter()
            .map(|(name, content)| json!({ "name": name, "content": content }))
            .collect();
        json!({ "tex": self.tex, "data_files": files }).to_string()
    }
}

/// Which document a JSON input describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A pgfplots `axis`.
    Pgfplots,
    /// A booktabs `tabular` of special points.
    Table,
}

impl Kind {
    /// The name used on the command line and in the C ABI.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Pgfplots => "pgfplots",
            Kind::Table => "table",
        }
    }

    /// Looks a name up. Returns `None` for anything but `pgfplots` and `table`.
    pub fn from_name(name: &str) -> Option<Kind> {
        match name {
            "pgfplots" => Some(Kind::Pgfplots),
            "table" => Some(Kind::Table),
            _ => None,
        }
    }
}

/// Renders the JSON text `json` as a document of `kind`.
pub fn render(kind: Kind, json: &str) -> Result<Rendered, LatexError> {
    let value: Value = serde_json::from_str(json)
        .map_err(|e| LatexError::new(ErrorKind::BadInput, format!("invalid JSON: {e}")))?;
    render_value(kind, &value)
}

/// Renders an already parsed document.
pub fn render_value(kind: Kind, value: &Value) -> Result<Rendered, LatexError> {
    let obj = value.as_object().ok_or_else(|| {
        LatexError::new(ErrorKind::BadInput, "the document must be a JSON object")
    })?;
    match kind {
        Kind::Pgfplots => figure::render(obj.clone()),
        Kind::Table => table::render(obj.clone()),
    }
}

/// Reads and removes the optional `standalone` flag. The renderers call it before they
/// reject unknown keys.
pub(crate) fn take_standalone(obj: &mut Object) -> Result<bool, LatexError> {
    match obj.remove("standalone") {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(b),
        Some(_) => Err(LatexError::new(
            ErrorKind::BadInput,
            "`standalone` must be true or false",
        )),
    }
}

/// Wraps `body` in a `standalone` document. `preamble` is the package lines, each
/// ending in a newline.
pub(crate) fn standalone_document(preamble: &str, body: &str) -> String {
    format!(
        "\\documentclass[border=5pt]{{standalone}}\n{preamble}\\begin{{document}}\n{body}\\end{{document}}\n"
    )
}

/// The pgfplots compatibility level the output is written for.
pub(crate) const PGFPLOTS_COMPAT: &str = "1.18";
