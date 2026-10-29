//! C ABI for the MEX gateway: `pc_api_v1()`, the handle registry and the panic barrier.
//!
//! The `json` module holds the problem parser and the diagnostics writer that `pc-cli`
//! and the C ABI share. The ABI itself arrives in later P7a tasks (see `docs/SPEC.md`,
//! section 10).

pub mod json;
