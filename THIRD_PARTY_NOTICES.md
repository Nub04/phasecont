# Third-party notices

PhaseCont is licensed under the Apache License 2.0 (see `LICENSE`). This file records code that was ported or copied from other projects, together with each project's license notice. Porting is allowed from AUTO-07p (BSD), DDE-BIFTOOL (BSD-2-Clause) and BifurcationKit.jl (MIT, after checking its LICENSE file). Ported code carries a header comment naming its source and gets an entry here. No code is copied from COCO (GPLv3), XPPAUT (GPL-2), xppy (LGPL-3) or VFGEN (GPL-2), and XPPAUT's example `.ode` files stay out of the repo because they are GPL.

No code has been ported or copied from DDE-BIFTOOL. Three modules under `crates/pc-engine/src/dde/` cite its manual: `hopf.rs` and `cycle.rs` follow its formulations, and `lyap.rs` uses its `|q| = 1` scale. Before any port, note that the repository (DDE-BIFTOOL/DDE-BIFTOOL, branch `master`, read on 2026-11-07) has no license file at its root. `tools/license.txt` is BSD-2-Clause, and `tools/ku-license.txt` is a non-commercial, internal-research license that looks like the notice of the old KU Leuven releases. The source files carry no license header. The BSD-2-Clause reading is the best-supported one, but it was not confirmed from a release note, so check the license again at the revision you port from and record the file name and revision in the header and in an entry below.

Rust crates that the engine links against are not listed here. `cargo deny check licenses` checks them against the allowlist in `deny.toml`. Their license notices still have to ship with the binaries, since MIT and BSD licenses require that for binary distributions. `pc-cli licenses` writes `THIRD_PARTY_LICENSES.txt` from the crates that `pc-ffi` links. It reads each crate's license files from the crate's own source folder. A crate that ships no license file takes the files in `licenses/overrides/<crate name>/`, and the `SOURCE` file there gives the upstream URL and tag or commit they were copied from. The `VERSION` file there lists the crate versions the text was taken from, one per line, and the command uses the override only for those versions. A `Cargo.lock` bump of an overridden crate stops the command until someone checks the new release's text and adds its version to `VERSION`. The command fails when `--overrides` is not a folder, when a crate has no license file and no override folder, when an override's `SOURCE` or `VERSION` file is missing or empty, and when `VERSION` does not list the locked version of the crate. Workspace crates add their own `NOTICE*` files, which carry the notices of the code listed under "Entries" below. The `package` buildtool task runs the command and ships the file in the `.mltbx`. The `pckern-win` CI job runs it too and uploads the file as an artifact.

The VS Code extension in `editors/vscode/` bundles nine npm packages into `dist/extension.js`: `vscode-languageclient` 11.0.0, `vscode-jsonrpc` 10.0.0, `vscode-languageserver-protocol` 3.18.5, `vscode-languageserver-types` 3.18.5 and `vscode-languageserver-textdocument` 1.0.16 (MIT), `minimatch` 10.2.6 (BlueOak-1.0.0), `semver` 7.8.6 (ISC), and `brace-expansion` 5.0.13 and `balanced-match` 4.0.4 (MIT). Their license texts are in `editors/vscode/THIRD_PARTY_LICENSES.txt`, which the extension ships. A test in `editors/vscode/test/manifest.test.ts` fails when `package-lock.json` has a production package that the file does not list. The extension's dev packages are not shipped.

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

### libm (MIT)

- Source: the `libm` crate 0.2.16 from crates.io (https://crates.io/crates/libm/0.2.16), `src/math/pow.rs`, which carries the FreeBSD `e_pow.c` notice below. The crate's license text is its `LICENSE.txt`. `libm` is outside the list of projects named for porting in the project rules; the owner approved this port on 2026-11-23.
- Ported into: `crates/pc-ir/src/ctlpow.rs`
- What was ported: the positive-normal-x path of `pow`, split at log2(x) into `log2_pair` (the pair t1, t2) and `exp2_part` (y·log2 x and 2^z), so that the two powers of the Dormand-Prince controller share one logarithm, in the CPU ensemble and, on `x86_64-pc-windows-gnu`, in the solo integrator (`crates/pc-engine/src/ivp.rs`). The special-case, overflow, underflow and subnormal-result branches are not ported; those inputs call `libm::pow` (SPEC Section 13, `cpu`).
- Notice (also in `crates/pc-ir/NOTICE`, which `pc-cli licenses` copies into `THIRD_PARTY_LICENSES.txt`):

```
rust-lang/libm as a whole is available for use under the MIT license:

------------------------------------------------------------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------

This Rust library contains the following copyrights:

    Copyright (c) 2018 Jorge Aparicio

Portions of this software are derived from third-party works licensed under
terms compatible with the above MIT license:

* musl libc https://www.musl-libc.org/. This library contains the following
  copyright:

      Copyright © 2005-2020 Rich Felker, et al.

* The CORE-MATH project https://core-math.gitlabpages.inria.fr/. CORE-MATH
  routines are available under the MIT license on a per-file basis.

The musl libc COPYRIGHT file also includes the following notice relevant to
math portions of the library:

------------------------------------------------------------------------------
Much of the math library code (src/math/* and src/complex/*) is
Copyright © 1993,2004 Sun Microsystems or
Copyright © 2003-2011 David Schultz or
Copyright © 2003-2009 Steven G. Kargl or
Copyright © 2003-2009 Bruce D. Evans or
Copyright © 2008 Stephen L. Moshier or
Copyright © 2017-2018 Arm Limited
and labelled as such in comments in the individual source files. All
have been licensed under extremely permissive terms.
------------------------------------------------------------------------------

Copyright notices are retained in src/* files where relevant.

The source file src/math/pow.rs carries this notice:

    origin: FreeBSD /usr/src/lib/msun/src/e_pow.c
    ====================================================
    Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.

    Permission to use, copy, modify, and distribute this
    software is freely granted, provided that this notice
    is preserved.
    ====================================================
```
