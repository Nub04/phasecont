# Third-party notices

PhaseCont is licensed under the Apache License 2.0 (see `LICENSE`). This file records code that was ported or copied from other projects, together with each project's license notice, as rule 3 of `CLAUDE.md` requires. Porting is allowed from AUTO-07p (BSD), DDE-BIFTOOL (BSD-2-Clause) and BifurcationKit.jl (MIT, after checking its LICENSE file).

Rust crates that the engine links against are not listed here. `cargo deny check licenses` checks them against the allowlist in `deny.toml`. Their license notices still have to ship with the binaries, since MIT and BSD licenses require that for binary distributions, so the packaging step collects them in P8a.

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
