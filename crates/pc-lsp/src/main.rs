//! `pc-lsp`: the language server for `.ode` files, on stdin and stdout.
#![deny(clippy::print_stdout)]

#[allow(clippy::print_stdout)] // `--version` is not the protocol.
fn main() {
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        println!("pc-lsp {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    std::process::exit(pc_lsp::transport::serve_stdio());
}
