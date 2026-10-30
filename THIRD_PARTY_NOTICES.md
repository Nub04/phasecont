# Third-party notices

PhaseCont is licensed under the Apache License 2.0 (see `LICENSE`). This file records code that was ported or copied from other projects, together with each project's license notice, as rule 3 of `CLAUDE.md` requires. Porting is allowed from AUTO-07p (BSD), DDE-BIFTOOL (BSD-2-Clause) and BifurcationKit.jl (MIT, after checking its LICENSE file).

Rust crates that the engine links against are not listed here. `cargo deny check licenses` checks them against the allowlist in `deny.toml`. Their license notices still have to ship with the binaries, since MIT and BSD licenses require that for binary distributions. `pc-cli licenses` writes `THIRD_PARTY_LICENSES.txt` from the crates that `pc-ffi` links. It reads each crate's license files from the crate's own source folder. A crate that ships no license file takes the files in `licenses/overrides/<crate name>/`, and the `SOURCE` file there gives the upstream URL and tag or commit they were copied from. The override texts match the crate versions in `Cargo.lock`. The overrides are keyed by crate name, so a version bump needs a new check of its text. The command fails when `--overrides` is not a folder, when a crate has no license file and no override folder, and when an override's `SOURCE` file is missing or empty. The `package` buildtool task runs the command and ships the file in the `.mltbx`. The `pckern-win` CI job runs it too and uploads the file as an artifact.

## Entry format

Each ported source gets one section in this form:

```
### <project> (<license>)

- Source: <URL and version or commit>
- Ported into: <paths in this repo>
- What was ported: <routines or algorithms>
- Notice: <the project's copyright notice and license text, verbatim>
```

## Entries

None yet.
