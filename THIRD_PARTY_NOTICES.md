# Third-party notices

PhaseCont is licensed under the Apache License 2.0 (see `LICENSE`). This file records code that was ported or copied from other projects, together with each project's license notice. Porting is allowed from AUTO-07p (BSD), DDE-BIFTOOL (BSD-2-Clause) and BifurcationKit.jl (MIT, after checking its LICENSE file). Ported code carries a header comment naming its source and gets an entry here. No code is copied from COCO (GPLv3), XPPAUT (GPL-2), xppy (LGPL-3) or VFGEN (GPL-2), and XPPAUT's example `.ode` files stay out of the repo because they are GPL.

Rust crates that the engine links against are not listed here. `cargo deny check licenses` checks them against the allowlist in `deny.toml`. Their license notices still have to ship with the binaries, since MIT and BSD licenses require that for binary distributions. `pc-cli licenses` writes `THIRD_PARTY_LICENSES.txt` from the crates that `pc-ffi` links. It reads each crate's license files from the crate's own source folder. A crate that ships no license file takes the files in `licenses/overrides/<crate name>/`, and the `SOURCE` file there gives the upstream URL and tag or commit they were copied from. The override texts match the crate versions in `Cargo.lock`. The overrides are keyed by crate name, so a version bump needs a new check of its text. The command fails when `--overrides` is not a folder, when a crate has no license file and no override folder, and when an override's `SOURCE` file is missing or empty. Workspace crates add their own `NOTICE*` files, which carry the notices of the code listed under "Entries" below. The `package` buildtool task runs the command and ships the file in the `.mltbx`. The `pckern-win` CI job runs it too and uploads the file as an artifact.

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

### AUTO-07p (BSD license, three clauses)

- Source: https://github.com/auto-07p/auto-07p, `src/mesh.f90` at commit cb5dd4616be3fabd3ace9fb9f9b22fcbac9fc447. The license text below is taken from `doc/auto.tex` at commit 44cc1c42a888179bdbd608a40613caee9cbc4675.
- Ported into: `crates/pc-engine/src/cycle/mesh.rs`
- What was ported: EQDF, NEWMSH and INTERP, the mesh adaptation part of ADAPT, for the periodic case with a fixed number of mesh intervals.
- Notice (also in `crates/pc-engine/NOTICE`, which `pc-cli licenses` copies into `THIRD_PARTY_LICENSES.txt`):

```
Copyright (c) 1979-2007, E. J. Doedel, California Institute of
Technology, and Concordia University.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

* Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
