# PhaseCont for VS Code

Syntax highlighting and a language server for XPP/XPPAUT `.ode` files, as PhaseCont reads them. The server is `pc-lsp`, written in Rust. It runs the same parser as the PhaseCont engine, so the squiggles in the editor are the errors that PhaseCont reports when it loads the model.

## Install

1. Build the server from a PhaseCont checkout: `cargo build --release -p pc-lsp`. The binary is `target/release/pc-lsp` (`pc-lsp.exe` on Windows).
2. Build the extension: `cd editors/vscode && npm ci --ignore-scripts && npm run bundle`.
3. Put the server where the extension looks for it: copy the binary to `editors/vscode/server/<platform>-<arch>/pc-lsp`, for example `server/linux-x64/pc-lsp`. The platforms are `linux-x64`, `win32-x64`, `darwin-arm64` and `darwin-x64`.
4. Run the extension from the checkout: open `editors/vscode` in VS Code and press F5.

To keep the server somewhere else, set `phasecont.server.path` to the absolute path of the binary. A non-empty setting never falls back to the bundled server, so a wrong path is reported instead of ignored. The setting has machine scope, so a workspace cannot choose the executable that VS Code starts.

## What it does

- Highlighting for the nine declaration words (first letter only, as XPPAUT reads them), the full-word keywords, `@` options, array blocks, `#include`, builtins, `pi`, `t` and `arg1` to `arg9`.
- Diagnostics from the PhaseCont parser, with the code (`ODE0001` and so on), related notes, and fix-its as code actions.
- Hover for builtins and for declared names.
- Completion for builtins, declared names and `@` option names.
- Go to definition and the outline view.
- `#include` is read from the workspace folders and the folder of the open file. Anything outside them is reported as an unresolved include.

Only files on disk are served (`file:` URIs), so an untitled buffer gets highlighting but no diagnostics until it is saved.

## Commands and settings

- `PhaseCont: Restart Language Server` (`phasecont.restartServer`).
- `phasecont.server.path`: absolute path to `pc-lsp`. Empty uses the bundled server.
- `phasecont.trace.server`: `off`, `messages` or `verbose`. The output goes to the "PhaseCont" output channel.

## Tests

`npm test` compiles the TypeScript and runs `node --test`. The tests tokenize sample models with the grammar, check the grammar against every builtin in `crates/pc-ode/src/builtins.rs`, and cover the server lookup. VS Code itself is not started by the tests.

## License

Apache-2.0. The npm packages bundled into `dist/extension.js` are listed in `THIRD_PARTY_LICENSES.txt`.
