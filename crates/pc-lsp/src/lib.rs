//! Language server for `.ode` files.
//!
//! The server calls `pc_ode::load` for every check and turns its diagnostics
//! into LSP messages. It never parses on its own.
#![deny(clippy::print_stdout)]
// `lsp_types::Uri` holds a `Cell` inside, but its `Hash` and `Eq` read only
// the text, so it is safe as a map key.
#![allow(clippy::mutable_key_type)]

pub mod analysis;
pub mod convert;
pub mod diagnostics;
pub mod features;
pub mod guard;
pub mod overlay;
pub mod server;
pub mod transport;
pub mod uri;
