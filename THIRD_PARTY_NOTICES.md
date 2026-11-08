# Third-party notices

PhaseCont is licensed under the Apache License 2.0 (see `LICENSE`). This file records code that was ported or copied from other projects, together with each project's license notice. Porting is allowed from AUTO-07p (BSD), DDE-BIFTOOL (BSD-2-Clause) and BifurcationKit.jl (MIT, after checking its LICENSE file). Ported code carries a header comment naming its source and gets an entry here. No code is copied from COCO (GPLv3), XPPAUT (GPL-2), xppy (LGPL-3) or VFGEN (GPL-2), and XPPAUT's example `.ode` files stay out of the repo because they are GPL.

No code has been ported or copied from DDE-BIFTOOL. The modules under `crates/pc-engine/src/dde/` follow formulations from its manual. Before any port, note that the repository (DDE-BIFTOOL/DDE-BIFTOOL, branch `master`, read on 2026-11-07) has no license file at its root. `tools/license.txt` is BSD-2-Clause, and `tools/ku-license.txt` is a non-commercial, internal-research license that looks like the notice of the old KU Leuven releases. The source files carry no license header. The BSD-2-Clause reading is the best-supported one, but it was not confirmed from a release note, so check the license again at the revision you port from and record the file name and revision in the header and in an entry below.

Rust crates that the engine links against are not listed here. `cargo deny check licenses` checks them against the allowlist in `deny.toml`. Their license notices still have to ship with the binaries, since MIT and BSD licenses require that for binary distributions. `pc-cli licenses` writes `THIRD_PARTY_LICENSES.txt` from the crates that `pc-ffi` links. It reads each crate's license files from the crate's own source folder. A crate that ships no license file takes the files in `licenses/overrides/<crate name>/`, and the `SOURCE` file there gives the upstream URL and tag or commit they were copied from. The override texts match the crate versions in `Cargo.lock`. The overrides are keyed by crate name, so a version bump needs a new check of its text. The command fails when `--overrides` is not a folder, when a crate has no license file and no override folder, and when an override's `SOURCE` file is missing or empty. Workspace crates add their own `NOTICE*` files, which carry the notices of the code listed under "Entries" below. The `package` buildtool task runs the command and ships the file in the `.mltbx`. The `pckern-win` CI job runs it too and uploads the file as an artifact.

The VS Code extension in `editors/vscode/` bundles eight npm packages into `dist/extension.js`: `vscode-languageclient` 9.0.1, `vscode-jsonrpc` 8.2.0, `vscode-languageserver-protocol` 3.17.5 and `vscode-languageserver-types` 3.17.5 (MIT), `minimatch` 5.1.9 and `semver` 7.8.6 (ISC), and `brace-expansion` 2.1.7 and `balanced-match` 1.0.2 (MIT). Their license texts are in `editors/vscode/THIRD_PARTY_LICENSES.txt`, which the extension ships. A test in `editors/vscode/test/manifest.test.ts` fails when `package-lock.json` has a production package that the file does not list. The extension's dev packages are not shipped.

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

- Source: https://github.com/auto-07p/auto-07p, `src/mesh.f90` at commit cb5dd4616be3fabd3ace9fb9f9b22fcbac9fc447 and `src/homcont.f90` at commit 44cc1c42a888179bdbd608a40613caee9cbc4675. The license text below is taken from `doc/auto.tex` at commit 44cc1c42a888179bdbd608a40613caee9cbc4675.
- Ported into: `crates/pc-engine/src/cycle/mesh.rs`, `crates/pc-engine/src/cycle/hom.rs`, `crates/pc-engine/src/cycle/hom_adjoint.rs` and `crates/pc-linalg/src/invariant.rs`
- What was ported: EQDF, NEWMSH and INTERP, the mesh adaptation part of ADAPT, with a fixed number of mesh intervals. The periodic case came first. The non-periodic end rule of EQDF (IPER false), used for homoclinic orbits, was read at commit 44cc1c42a888179bdbd608a40613caee9cbc4675, where `src/mesh.f90` is unchanged from cb5dd461. From `src/homcont.f90`: test functions 1 to 12 of PSIHO (with test 1 split from test 4 by the type of the leading eigenvalues, see SPEC Section 8b), the sign rule of the left eigenvectors in EIGHO, and the rotation of the projection bases in PRJCTN. The bases come from our own matrix sign function, not from HomCont's Schur form. For the adjoint problem (SPEC Section 8b-2): the adjoint right-hand side of FNHO (lines 175-178), the adjoint boundary rows of FBHO (503-523, the projections with ITRANS = 2), test functions 13 and 14 of PSIHO (1941-1955), the sign rule of the right eigenvectors in EIGHO (2104-2126), and the per-ITRANS rotation state of PRJCTN, which `BasisTracker` keeps as two slots.
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
