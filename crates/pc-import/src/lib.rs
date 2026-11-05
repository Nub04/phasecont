//! Reads models written in other forms and returns `.ode` text.
//!
//! Every route ends as `.ode` text, so the session hash, the JIT, the continuation and the
//! ensemble engine work the same for all of them. The importers translate text and never run
//! it. What they do not translate is refused by name, with its position when it has one. A
//! missing value is an error and never becomes 0.

pub mod emit;
pub mod error;
pub mod expr;
pub mod mathml;
pub mod matlab;
pub mod names;
pub mod sbml;

pub use emit::{Imported, OdeModel, State, emit};
pub use error::{ImportError, Kind, Result};
pub use matlab::FunctionInput;
pub use sbml::from_sbml;

/// Translates a MATLAB function handle (the text of `func2str`) or a function file.
pub fn from_function(input: &FunctionInput<'_>) -> Result<Imported> {
    let (model, warnings) = matlab::translate(input)?;
    let mut out = emit(&model)?;
    let mut all = warnings;
    all.append(&mut out.warnings);
    out.warnings = all;
    Ok(out)
}
