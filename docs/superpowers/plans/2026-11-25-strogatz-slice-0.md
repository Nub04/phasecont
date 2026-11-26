# Strogatz figure scripts, slice 0: implementation plan

> Work through the tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the twelve slice 0 rows of the Strogatz figure scripts (one row per API family), each with our own `.ode` model, a figure script, a manifest entry and a test against an independent oracle, then report the time per row so the owner can sign a time budget.

**Architecture:** `books/strogatz/` at the repo root (not shipped) holds `models/` (our `.ode` files), `figures/` (one function per figure plus the `strogatzModel` loader), `manifest.json` (done and blocked rows) and `gaps.json` (gap and n/a rows). `tests/StrogatzFigureTest.m` is one parameterized class over the done rows plus schema tests for both JSON files. A throwaway probe runs every row's PhaseCont calls first, so open choices (directions, step counts, bracket) are settled by measurement before any committed code.

**Tech Stack:** MATLAB R2026b through `scripts/matlab-win`, `matlab.unittest`, base MATLAB `jsondecode`, `fzero`, `eig` and `ode45` as oracles, the existing PhaseCont API (`simulate`, `equilibrium`, `continuation`, `cobweb`, `ensemble`, `Branch.classify`, `Branch.tdplot`, `phasecont.app.PhasePlane`), `jq` for the one-time `gaps.json` build.

**Spec:** `docs/superpowers/specs/2026-11-25-strogatz-figure-scripts-design.md` (decisions D1 to D5 ruled, commit 9419bf0). Read it before Task 1.

## Global Constraints

- Layout: `books/strogatz/{models,figures,manifest.json,gaps.json}` and `tests/StrogatzFigureTest.m` (D1). `buildfile.m` and the `.mltbx` stay unchanged.
- Script signature: `out = fig_<ch>_<sec>_<n>(Plot = true)`. `Plot = false` draws nothing and opens no app window; an app row builds its app with `Visible = "off"`.
- Manifest fields, exactly: `figure`, `edition` (3), `model`, `script`, `api`, `values`, `source` (`"stated"` or `"chosen"` per value), `check`, `state` (`"done"` or `"blocked: <item>"`). `gaps.json` fields, exactly: `figure`, `state` (`"gap"` or `"n/a"`), `reason`.
- Glue rule: a script may plot and do array arithmetic on what PhaseCont returns. It never iterates a map, integrates or finds roots in MATLAB. Tests may use `fzero`, `eig` and `ode45` as oracles.
- Models: one `.ode` file per model family, written by us from the equations. No XPPAUT example files, no files from Ermentrout's book, no GPL code.
- No book text: no captions, no book sentences, no printed data tables in scripts, manifest, gaps, comments or commit messages. Never commit, quote or upload the PDF, and send no book content to any outside tool or search.
- Rule 9: every closed form a check uses is derived in that check's comment from the model equations, or the check uses a numeric oracle. A value that cannot be confirmed stops the row and goes to the owner.
- Done rule: a row is done when its script runs from a clean session and its check passes. A row that needs a missing feature moves to `gaps.json` with a reason. A PhaseCont defect makes the row `"blocked: <item>"` and goes to the owner. No workarounds.
- No public API change, no C ABI change, no new dependency, no install.
- Writing: no em dashes, plain verbs, nub-humanizer in embedded mode for comments and commit messages.
- Commits: local only, on `strogatz-figures`, `git add` by path (never `CLAUDE.md`), identity `git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit`, imperative subject of 72 characters or fewer, no AI attribution of any kind. No push.
- Before every commit, in the worktree: `export CARGO_TARGET_DIR=$PWD/target CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES=""` then `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo deny check` (baseline 2981 passed, 0 failed, 23 ignored; no Rust changes are expected).
- Subagents, if used: set `model` to `sonnet` or `haiku` on every one. Give them no book content.

## Review Focus

1. **A `Plot = false` run that draws anyway.** The spec says it draws nothing. `figureMatchesItsOracle` checks that no new figure appeared, or for an app row that every new figure is the hidden app window (Task 2).
2. **A script that errors after it opened a figure or an app.** The figure must not leak into the next row. `runScript` closes new figures before it rethrows, and `aFailingScriptLeavesNoFigure` pins it with a probe script that opens a figure and errors (Task 2).
3. **A manifest with no done rows.** A parameterized class with an empty parameter list errors at load. `rowsFrom` returns a placeholder row that the test marks incomplete, and `noDoneRowsGivesAPlaceholder` pins it (Task 2).
4. **Book text creeping into the JSON files.** The schema tests whitelist the fields and cap `check` at 200 characters and `reason` at 120, so a pasted caption fails (Task 2).
5. **Recorded values that differ from the values used.** Every script builds one struct `v` and passes only its fields to PhaseCont (`Params = struct(...)` from `v`). The row test compares `out.Values` with the manifest after a JSON round trip, so a manifest edit without a script edit fails. Reviewers check that no PhaseCont call in a script holds a numeric literal (Tasks 2 to 13).

---

## Local paths

The commands use these placeholders. Each stands for a folder on the developer's machine and none is in the repo.

| Placeholder | Folder |
|---|---|
| `<worktree>` | the checkout of branch `strogatz-figures` |
| `<scratch>` | a throwaway folder outside the repo |
| `<wip>` | a local notes folder outside the repo, never committed; it holds the triage files in `strogatz-triage-2026-11-25/` |
| `<stage>` | the Windows folder (under `/mnt/c`) that holds the staged win64 binaries |
| `<texbin>` | the Windows folder with the TeX programs the LaTeX tests need |

## File structure

| File | Responsibility |
|---|---|
| `books/strogatz/models/*.ode` | one model family each, our own equations |
| `books/strogatz/figures/strogatzModel.m` | loads a model from `models/` with the JIT evaluator |
| `books/strogatz/figures/fig_*.m` | one figure each: values, PhaseCont calls, optional drawing |
| `books/strogatz/manifest.json` | one entry per covered row built so far |
| `books/strogatz/gaps.json` | gap and n/a rows with a short reason |
| `tests/StrogatzFigureTest.m` | row tests, schema tests, one protected `check_<script>` per done row |
| `docs/STATUS.md` | slice 0 report (Task 14) |

Single-row test run, used by every row task (replace the parameter name):

```bash
cd <worktree> && \
scripts/matlab-win "import matlab.unittest.selectors.*; import matlab.unittest.constraints.ContainsSubstring; s = matlab.unittest.TestSuite.fromFile('tests/StrogatzFigureTest.m', HasParameter(Name = 'f2_2_6') | HasName(ContainsSubstring('Schema'))); disp(table(run(s)))"
```

The selector keeps the two schema tests in the run, which `ParameterName` alone would drop. Before a row's script exists, expect its row tests to fail with `Undefined function` (an app row skips `plotTrueDraws`) and `manifestRowsFollowTheSchema` to fail on the missing script file. Once the script exists, every selected test passes.

Whole-class run:

```bash
cd <worktree> && \
scripts/matlab-win "results = runtests('tests/StrogatzFigureTest.m'); disp(table(results))"
```

---

### Task 1: Probe every row (throwaway)

**Files:**
- Create (kept, committed by the row tasks): the eleven model files under `books/strogatz/models/`
- Create (throwaway): `<scratch>/strogatz-probe/probe.m`
- Create (local only, never committed): `<wip>/strogatz-slice0-probe-2026-11-26.md`

**Interfaces:**
- Produces: the model file names used by every row task; the probe log with the settled values for 3.4.2 (variant), 3.6.2 (direction), 8.3.4 (direction), 9.5.1 (direction, MaxSteps), 8.4.3 (LC and HOM options, oracle bracket) and 5.2.8 (whether the degenerate H at c = 0 is a problem).

- [ ] **Step 1: Check the staged binaries**

Run: `ls -l <stage>/`
Expected: `pckern.dll` and `pckern_gateway.mexw64` (or the current MEX name) dated after the last engine change on this branch (`git log -1 --format=%ci -- crates mex`). If older, restage:

```bash
cd <worktree> && \
scripts/build-pckern win64 <stage> && \
scripts/matlab-win "buildtool mex"
```

- [ ] **Step 2: Write the eleven model files**

`books/strogatz/models/cosine_flow.ode`:
```
# x' = x - cos(x)
init x=0.7
x'=x-cos(x)
done
```

`books/strogatz/models/lorenz.ode`:
```
# Lorenz equations
par sigma=10, r=28, b=2.6666666666666665
init x=0, y=1, z=0
x'=sigma*(y-x)
y'=r*x-y-x*z
z'=x*y-b*z
done
```

`books/strogatz/models/rabbits_sheep.ode`:
```
# Competition of two species, x rabbits and y sheep
init x=0.1, y=0.1
x'=x*(3-x-2*y)
y'=y*(2-x-y)
done
```

`books/strogatz/models/pitchfork.ode`:
```
# Supercritical pitchfork normal form
par r=-1
init x=0
x'=r*x-x^3
done
```

`books/strogatz/models/imperfect.ode`:
```
# Imperfect pitchfork
par h=0, r=1
init x=-1
x'=h+r*x-x^3
done
```

`books/strogatz/models/chemical_oscillator.ode`:
```
# Two-variable chemical oscillator
par a=10, b=2
init x=2, y=5
x'=a-x-4*x*y/(1+x^2)
y'=b*x*(1-y/(1+x^2))
done
```

`books/strogatz/models/homoclinic.ode`:
```
# Planar system with a homoclinic bifurcation in mu
par mu=-1.5, c=1
init x=1, y=0
x'=y
y'=mu*y+x-x^2+c*x*y
done
```

`books/strogatz/models/cosine_map.ode`:
```
# x(n+1) = cos(x(n))
init x=1
x(t+1)=cos(x)
done
```

`books/strogatz/models/logistic_map.ode`:
```
# Logistic map
par r=0.5
init x=0
x(t+1)=r*x*(1-x)
done
```

`books/strogatz/models/henon_map.ode`:
```
# Henon map
par a=1.4, b=0.3
init x=0, y=0
x(t+1)=1-a*x^2+y
y(t+1)=b*x
done
```

`books/strogatz/models/damped_oscillator.ode`:
```
# Linear oscillator with damping c
par k=1, c=-3
init x=0, y=0
x'=y
y'=-k*x-c*y
done
```

- [ ] **Step 3: Write the probe script**

`<scratch>/strogatz-probe/probe.m`:

```matlab
root = '\\wsl.localhost\FedoraLinux-44\home\nub04\dev\phasecont\.claude\worktrees\wf_382b34d0-0bc-1';
addpath(fullfile(root, 'toolbox'));
addpath(fullfile(getenv('USERPROFILE'), 'phasecont-stage', 'bin', 'win64'));
M = @(name) phasecont.Model(fullfile(root, 'books', 'strogatz', 'models', name), Evaluator = "jit");
step = @(row, fn) runStep(row, fn);

step("2.2.6", @() probe226(M));
step("9.3.2", @() probe932(M));
step("6.4.7", @() probe647(M));
step("3.4.2", @() probe342(M));
step("3.6.2", @() probe362(M));
step("8.3.4", @() probe834(M));
step("9.5.1", @() probe951(M));
step("8.4.3", @() probe843(M));
step("10.1.3", @() probe1013(M));
step("10.3.4", @() probe1034(M));
step("12.2.2", @() probe1222(M));
step("5.2.8", @() probe528(M));

function runStep(row, fn)
lastwarn("");
clock = tic;
try
    fn();
    fprintf("PROBE %s OK %.3f s\n", row, toc(clock));
catch err
    fprintf("PROBE %s FAIL %.3f s [%s] %s\n", row, toc(clock), err.identifier, err.message);
end
[msg, id] = lastwarn;
if msg ~= ""
    fprintf("PROBE %s lastwarn [%s] %s\n", row, id, msg);
end
end

function probe226(M)
m = M("cosine_flow.ode");
pt = phasecont.equilibrium(m, 0.7);
fprintf("  U %.15g eig %.15g stable %d\n", pt.U, pt.Eigenvalues, pt.Stable);
end

function probe932(M)
m = M("lorenz.ode");
tr = phasecont.simulate(m, [0 1 0], [0 50], ...
    Params = struct("sigma", 10, "r", 28, "b", 8/3), RelTol = 1e-10, AbsTol = 1e-12);
k = find(tr.T >= 1, 1);
f = @(t, u) [10*(u(2)-u(1)); 28*u(1)-u(2)-u(1)*u(3); u(1)*u(2)-8/3*u(3)];
[~, U] = ode45(f, [0 tr.T(k)], [0; 1; 0], odeset(RelTol = 1e-12, AbsTol = 1e-12));
fprintf("  rows %d T(end) %.15g err %.3g\n", numel(tr.T), tr.T(end), ...
    norm(tr.X(k,:) - U(end,:)) / norm(U(end,:)));
fprintf("  jsonencode(8/3) = %s\n", jsonencode(8/3));
end

function probe647(M)
m = M("rabbits_sheep.ode");
app = phasecont.app.PhasePlane(m, Visible = "off");
cleanup = onCleanup(@() delete(app));
app.setWindow([0 3.5 0 2.5]);
app.setT(20);
app.showNullclines(true);
P = [0.1 0.1; 0.1 1.9; 2.9 0.1; 0.9 1.1];
for k = 1:4
    pt = app.findEquilibrium(P(k,1), P(k,2));
    tr = app.addTrajectory(P(k,1), P(k,2));
    fprintf("  pt %s stable %d eig %s | tr end %s, T(1) %.3g T(end) %.3g\n", ...
        mat2str(pt.U', 10), pt.Stable, mat2str(pt.Eigenvalues', 6), ...
        mat2str(tr.X(end,:), 6), tr.T(1), tr.T(end));
end
tags = ["pc-equilibrium-stable" "pc-equilibrium-unstable" "pc-nullcline" "pc-trajectory"];
for t = tags
    fprintf("  %s: %d\n", t, numel(findall(app.Axes, "Tag", t)));
end
figs = findall(groot, "Type", "figure");
fprintf("  figures open %d, visible %s\n", numel(figs), strjoin(string(arrayfun(@(f) char(f.Visible), figs, UniformOutput = false)), ","));
end

function probe342(M)
m = M("pitchfork.ode");
brs = phasecont.continuation(m, 0, Params = struct("r", -1), Free = "r", ...
    Bounds = [-1 1], Hmax = 0.05, Switch = "BP1");
bp = brs(1).points("BP");
fprintf("  BP count %d, Index %g, r %.3g\n", numel(bp), bp(1).Index, brs(1).Alpha(bp(1).Index));
fprintf("  brs(2) U range [%.4g %.4g], r range [%.4g %.4g], N %d\n", ...
    min(brs(2).U), max(brs(2).U), min(brs(2).Alpha), max(brs(2).Alpha), numel(brs(2).Alpha));
alt = phasecont.continuation(m, -1, Params = struct("r", 1), Free = "r", ...
    Bounds = [-1 1.5], Hmax = 0.05, Direction = -1);
fprintf("  fallback U range [%.4g %.4g], labels %s\n", min(alt.U), max(alt.U), ...
    strjoin([alt.points().Label], ","));
end

function probe362(M)
m = M("imperfect.ode");
b = phasecont.continuation(m, -1, Params = struct("h", 0, "r", 1), Free = "h", Bounds = [-1 1]);
lp = b.points("LP");
fprintf("  LP count %d, h %s\n", numel(lp), mat2str(b.Alpha([lp.Index])', 10));
for d = [1 -1]
    c = phasecont.continuation(m, lp(1), Type = "LP", Free = ["h" "r"], ...
        Bounds = [-0.5 1.5], Direction = d);
    cp = c.points("CP");
    fprintf("  Direction %d: x range [%.4g %.4g], r range [%.4g %.4g], CP %d, stop %s\n", ...
        d, min(c.U), max(c.U), min(c.Alpha(:,2)), max(c.Alpha(:,2)), numel(cp), c.Stop);
end
end

function probe834(M)
m = M("chemical_oscillator.ode");
b = phasecont.continuation(m, [2 5], Params = struct("a", 10, "b", 2), Free = "b", Bounds = [2 5]);
hp = b.points("H");
fprintf("  H count %d, b %.12g, omega %.12g\n", numel(hp), b.Alpha(hp(1).Index), hp(1).Data.omega);
for d = [1 -1]
    c = phasecont.continuation(m, hp(1), Type = "H", Free = ["a" "b"], ...
        Bounds = [0.5 10], Direction = d);
    fprintf("  Direction %d: a range [%.4g %.4g], b range [%.4g %.4g], N %d, stop %s\n", ...
        d, min(c.Alpha(:,1)), max(c.Alpha(:,1)), min(c.Alpha(:,2)), max(c.Alpha(:,2)), ...
        size(c.Alpha, 1), c.Stop);
end
end

function probe951(M)
m = M("lorenz.ode");
p = struct("sigma", 10, "r", 20, "b", 8/3);
pt = phasecont.equilibrium(m, [7 7 19], Params = p);
b = phasecont.continuation(m, pt, Free = "r", Bounds = [20 30]);
hp = b.points("H");
fprintf("  H r %.12g (470/19 = %.12g), omega %.12g (sqrt(1760/19) = %.12g)\n", ...
    b.Alpha(hp(1).Index), 470/19, hp(1).Data.omega, sqrt(1760/19));
for d = [1 -1]
    clock = tic;
    lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "r", NTST = 40, NCOL = 4, ...
        Bounds = [13 25], Direction = d, MaxSteps = 400);
    fprintf("  Direction %d: %.2f s, N %d, r range [%.4g %.4g], period [%.4g %.4g], stop %s\n", ...
        d, toc(clock), numel(lc.Period), min(lc.Alpha), max(lc.Alpha), ...
        min(lc.Period), max(lc.Period), lc.Stop);
    [tc, xc] = lc.cycle(1);
    fprintf("  cycle(1): t %s, x %s\n", mat2str(size(tc)), mat2str(size(xc)));
end
end

function probe843(M)
m = M("homoclinic.ode");
b = phasecont.continuation(m, [1 0], Params = struct("mu", -1.5, "c", 1), Free = "mu", ...
    Bounds = [-1.5 -0.5]);
hp = b.points("H");
fprintf("  H mu %.12g omega %.12g\n", b.Alpha(hp(1).Index), hp(1).Data.omega);
clock = tic;
lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "mu", NTST = 80, NCOL = 4, ...
    Hmax = 0.05, MaxNewton = 15, MaxSteps = 1700);
fprintf("  LC %.2f s, N %d, mu range [%.6g %.6g], max period %.4g, stop %s\n", toc(clock), ...
    numel(lc.Period), min(lc.Alpha), max(lc.Alpha), max(lc.Period), lc.Stop);
i = find(lc.Period > 50, 1);
fprintf("  first index with period > 50: %d (mu %.8g)\n", i, lc.Alpha(i));
clock = tic;
hom = phasecont.continuation(m, lc, Type = "HOM", Index = i, Free = ["mu" "c"], ...
    Bounds = [0.9 1.1], Hmax = 0.01, MaxNewton = 15);
fprintf("  HOM %.2f s, N %d, first [%.10g %.10g], last [%.10g %.10g], stop %s\n", toc(clock), ...
    size(hom.Alpha, 1), hom.Alpha(1,:), hom.Alpha(end,:), hom.Stop);
g = @(mu) splitDistance(mu, 1);
fprintf("  split at -0.95: %.4g, at -0.8: %.4g\n", g(-0.95), g(-0.8));
fprintf("  oracle mu_c(c=1) %.10g\n", fzero(g, [-0.95 -0.8]));
end

function d = splitDistance(mu, c)
f = @(t, u) [u(2); mu*u(2) + u(1) - u(1)^2 + c*u(1)*u(2)];
lu = (mu + sqrt(mu^2 + 4)) / 2;
ls = (mu - sqrt(mu^2 + 4)) / 2;
e = 1e-6;
down = odeset(RelTol = 1e-12, AbsTol = 1e-14, Events = @(t, u) crossing(u, -1));
up = odeset(RelTol = 1e-12, AbsTol = 1e-14, Events = @(t, u) crossing(u, 1));
[~, ~, ~, uu] = ode45(f, [0 200], e*[1; lu], down);
[~, ~, ~, us] = ode45(@(t, u) -f(t, u), [0 200], e*[1; ls], up);
d = uu(1,1) - us(1,1);
end

function [value, isterminal, direction] = crossing(u, dir)
value = u(2);
isterminal = 1;
direction = dir;
end

function probe1013(M)
m = M("cosine_map.ode");
tr = phasecont.simulate(m, 1, [0 20]);
x = tr.X(:, 1);
fprintf("  rows %d, max rel step err %.3g\n", numel(x), max(abs(x(2:end) - cos(x(1:end-1))) ./ abs(cos(x(1:end-1)))));
pt = phasecont.equilibrium(m, 0.7);
fprintf("  fixed point %.15g, multiplier %.15g\n", pt.U, pt.Eigenvalues);
end

function probe1034(M)
m = M("logistic_map.ode");
brs = phasecont.continuation(m, 0, Params = struct("r", 0.5), Free = "r", Bounds = [0.5 2], Switch = "BP1");
bp = brs(1).points("BP");
fprintf("  BP r %.12g, switched r range [%.4g %.4g]\n", brs(1).Alpha(bp(1).Index), ...
    min(brs(2).Alpha), max(brs(2).Alpha));
brs = phasecont.continuation(m, 0.6, Params = struct("r", 2.5), Free = "r", ...
    Bounds = [2.5 3.6], Hmax = 0.05, Switch = "PD1");
pd = brs(1).points("PD");
pd2 = brs(2).points("PD");
fprintf("  PD r %.12g x %.12g; brs(2) Iterate %d, PD r %.12g (1+sqrt(6) = %.12g)\n", ...
    brs(1).Alpha(pd(1).Index), pd(1).U, brs(2).Iterate, brs(2).Alpha(pd2(1).Index), 1 + sqrt(6));
end

function probe1222(M)
m = M("henon_map.ode");
[X, Y] = meshgrid(linspace(-1.5, 1.5, 100), linspace(-0.4, 0.4, 100));
X0 = [X(:)'; Y(:)'];
r = phasecont.ensemble(m, X0, [1.4; 0.3], [0 1], Precision = "double", Backend = "cpu");
F = [1 - 1.4*X0(1,:).^2 + X0(2,:); 0.3*X0(1,:)];
fprintf("  size %s, max abs err %.3g\n", mat2str(size(r.Final)), max(abs(r.Final - F), [], "all"));
end

function probe528(M)
m = M("damped_oscillator.ode");
br = phasecont.continuation(m, [0 0], Params = struct("k", 1, "c", -3), Free = "c", ...
    Bounds = [-3 3], Direction = 1);
tbl = br.classify();
fprintf("  N %d, labels %s, stop %s\n", numel(br.Alpha), strjoin([br.points().Label], ","), br.Stop);
fprintf("  types %s\n", strjoin(unique(string(tbl.Type))', " | "));
fprintf("  max |T + c| %.3g, max |D - 1| %.3g\n", max(abs(tbl.T + tbl.Alpha)), max(abs(tbl.D - 1)));
end
```

- [ ] **Step 4: Run the probe**

Run:
```bash
cd <worktree> && \
scripts/matlab-win --run <scratch>/strogatz-probe/probe.m 2>&1 | tee <wip>/strogatz-slice0-probe-raw-2026-11-26.txt
```
Expected: twelve `PROBE <row>` lines. Keep the full log.

- [ ] **Step 5: Settle each open choice from the log**

Write `<wip>/strogatz-slice0-probe-2026-11-26.md` with one line per row: OK or FAIL, time, and the settled value. The rules:

- **3.4.2:** if `brs(2) U range` covers below -0.9 and above 0.9, use the switch variant (Task 5, variant A). Otherwise use the fallback run (Task 5, variant B) and check that its U range covers both.
- **3.6.2:** take the `Direction` whose run reports CP 1 and an x range with both signs. If neither does, the row is blocked (owner item).
- **8.3.4:** take the direction (or both, concatenated in the script as two runs) whose a range reaches about 6.885 and 18.87. If one run covers both ends, use it alone.
- **9.5.1:** take the direction whose r range goes below r_H - 1 = 23.737. Set `MaxSteps` to the step count it needed plus 20 percent, rounded up to 10. Record the time.
- **8.4.3:** the bracket [-0.95, -0.8] must give split values of opposite sign. If not, widen it within [-0.99, -0.7] and record the bracket used. If the LC run never passes period 50, raise `MaxSteps` once to 3000; if it still does not, the row is blocked. Record the LC and HOM times.
- **5.2.8:** if the run reports an error or a warning at c = 0, the row is blocked with that message (owner item). If it reports an H label and carries on, keep the row.
- **Any `FAIL`:** read the error. A missing feature moves the row to `gaps.json` in its task. A PhaseCont defect makes the row `"blocked: <short item>"` and goes into the Task 14 report. No workaround in the script.
- **Point.Index:** confirm from the 3.4.2 and 3.6.2 lines that `Index` indexes `Alpha` (BP at r near 0, LP at h near plus or minus 0.385). If not, stop and recheck how special points map to rows before writing checks.
- **jsonencode(8/3):** if it prints fewer than 17 significant digits, keep `RelTol = 1e-14` in the values comparison (Task 2 already uses it).

No commit in this task. The model files stay untracked until their row tasks add them.

---

### Task 2: Scaffold, schema tests and row 2.2.6

**Files:**
- Create: `books/strogatz/figures/strogatzModel.m`
- Create: `books/strogatz/figures/fig_2_2_6.m`
- Create: `books/strogatz/manifest.json`
- Create: `books/strogatz/gaps.json`
- Create: `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/cosine_flow.ode` (from Task 1)

**Interfaces:**
- Consumes: `cosine_flow.ode` from Task 1.
- Produces: `strogatzModel(name) -> phasecont.Model`; the script contract `out = fig_X(Plot = true)` with `out.Values` (struct) and `out.Model`; the test class with `runScript(name, plot) -> [out, seconds]` (static), `rowsFrom(list) -> struct of rows` (static), the protected method slot `check_<script>(testCase, out)`, and the local helper `cdJacobian(m, u, varargin) -> n-by-n double`. Later tasks add one `check_` method and one manifest entry each.

- [ ] **Step 1: Write the test class**

`tests/StrogatzFigureTest.m`:

```matlab
classdef StrogatzFigureTest < matlab.unittest.TestCase
    % Runs each done row of books/strogatz/manifest.json: the figure script
    % with Plot = false, then the row's check against an independent oracle.
    % Also checks the schema of manifest.json and gaps.json. The row tests
    % skip when pckern is not built; the schema tests always run.

    properties (TestParameter)
        Row
    end

    properties
        HasEngine (1,1) logical = false
    end

    methods (TestParameterDefinition, Static)
        function Row = initializeRow()
            Row = StrogatzFigureTest.rowsFrom(StrogatzFigureTest.readRows("manifest.json"));
        end
    end

    methods (Static)
        function rows = readRows(name)
            % The rows of books/strogatz/NAME as a cell array of structs.
            root = fileparts(fileparts(mfilename("fullpath")));
            rows = jsondecode(fileread(fullfile(root, "books", "strogatz", name)));
            if isstruct(rows)
                rows = num2cell(rows);
            end
        end

        function rows = rowsFrom(list)
            % The done rows of LIST as a struct with one field per row,
            % "f" plus the figure number with dots as underscores. With no
            % done row it returns one placeholder row, so the class still
            % loads and the row test reports it as incomplete.
            rows = struct();
            for k = 1:numel(list)
                r = list{k};
                if string(r.state) == "done"
                    rows.("f" + strrep(string(r.figure), ".", "_")) = r;
                end
            end
            if isempty(fieldnames(rows))
                rows.none = struct("figure", "none", "script", "", "state", "none");
            end
        end

        function [out, seconds] = runScript(name, plot)
            % Runs the figure script NAME. When the script errors, the
            % figures it opened are closed before the error goes on.
            before = findall(groot, "Type", "figure");
            clock = tic;
            try
                out = feval(char(name), "Plot", plot);
            catch err
                delete(StrogatzFigureTest.newFigures(before));
                rethrow(err);
            end
            seconds = toc(clock);
        end

        function f = newFigures(before)
            % The figures that are open now and were not open in BEFORE,
            % found by comparing the handles one at a time with ==.
            figs = findall(groot, "Type", "figure");
            keep = true(size(figs));
            for k = 1:numel(figs)
                keep(k) = ~any(before == figs(k));
            end
            f = figs(keep);
        end
    end

    methods (TestClassSetup)
        function setup(testCase)
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.HasEngine = isfile(fullfile(f.Bin, "pckern." + mexext));
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(root, "books", "strogatz", "figures")));
        end
    end

    methods (Test)
        function figureMatchesItsOracle(testCase, Row)
            testCase.assumeNotEqual(string(Row.state), "none", "The manifest has no done row");
            testCase.assumeTrue(testCase.HasEngine, "pckern is not built");
            before = findall(groot, "Type", "figure");
            testCase.addTeardown(@() delete(StrogatzFigureTest.newFigures(before)));
            [out, seconds] = StrogatzFigureTest.runScript(Row.script, false);
            fprintf("strogatz-time %s %.3f\n", Row.figure, seconds);
            if isfield(out, "App")
                testCase.addTeardown(@() delete(out.App));
            end
            opened = StrogatzFigureTest.newFigures(before);
            if isfield(out, "App")
                testCase.verifyTrue(all(arrayfun(@(f) f.Visible == "off", opened)), ...
                    "Plot = false must leave the app window hidden");
            else
                testCase.verifyEmpty(opened, "Plot = false must draw nothing");
            end
            testCase.verifyEqual(orderfields(jsondecode(jsonencode(out.Values))), ...
                orderfields(Row.values), RelTol = 1e-14, ...
                "out.Values must equal the values in the manifest");
            testCase.("check_" + string(Row.script))(out);
        end

        function plotTrueDraws(testCase, Row)
            testCase.assumeNotEqual(string(Row.state), "none", "The manifest has no done row");
            testCase.assumeTrue(testCase.HasEngine, "pckern is not built");
            testCase.assumeFalse(any(contains(string(Row.api), "PhasePlane")), ...
                "An app row with Plot = true opens a visible window");
            old = get(groot, "DefaultFigureVisible");
            testCase.addTeardown(@() set(groot, "DefaultFigureVisible", old));
            set(groot, "DefaultFigureVisible", "off");
            before = findall(groot, "Type", "figure");
            testCase.addTeardown(@() delete(StrogatzFigureTest.newFigures(before)));
            [~, seconds] = StrogatzFigureTest.runScript(Row.script, true);
            fprintf("strogatz-plot-time %s %.3f\n", Row.figure, seconds);
            opened = StrogatzFigureTest.newFigures(before);
            testCase.verifyNotEmpty(opened, "Plot = true must draw a figure");
            ax = findall(opened, "Type", "axes");
            testCase.verifyTrue(any(arrayfun(@(a) ~isempty(a.Children), ax)), ...
                "The figure must hold a non-empty axes");
        end

        function manifestRowsFollowTheSchema(testCase)
            rows = StrogatzFigureTest.readRows("manifest.json");
            testCase.verifyNotEmpty(rows);
            fields = ["figure" "edition" "model" "script" "api" "values" "source" "check" "state"];
            names = string({metaclass(testCase).MethodList.Name});
            root = fileparts(fileparts(mfilename("fullpath")));
            figures = strings(1, numel(rows));
            for k = 1:numel(rows)
                r = rows{k};
                id = "manifest row " + k;
                testCase.verifyEqual(sort(string(fieldnames(r)))', sort(fields), id);
                figures(k) = string(r.figure);
                testCase.verifyMatches(figures(k), "^\d+\.\d+\.\d+$", id);
                testCase.verifyEqual(r.edition, 3, id);
                testCase.verifyEqual(string(r.script), "fig_" + strrep(figures(k), ".", "_"), id);
                testCase.verifyTrue(iscellstr(r.api) || ischar(r.api), id);
                testCase.verifyNotEmpty(r.api, id);
                testCase.verifyClass(r.values, "struct", id);
                testCase.verifyClass(r.source, "struct", id);
                testCase.verifyEqual(sort(string(fieldnames(r.source))), ...
                    sort(string(fieldnames(r.values))), id);
                kinds = string(struct2cell(r.source));
                testCase.verifyTrue(all(ismember(kinds, ["stated" "chosen"])), id);
                testCase.verifyNotEmpty(r.check, id);
                testCase.verifyLessThanOrEqual(strlength(string(r.check)), 200, id);
                state = string(r.state);
                testCase.verifyTrue(state == "done" || startsWith(state, "blocked: "), id);
                if state == "done"
                    testCase.verifyTrue(isfile(fullfile(root, "books", "strogatz", "models", r.model)), id);
                    testCase.verifyTrue(isfile(fullfile(root, "books", "strogatz", "figures", ...
                        string(r.script) + ".m")), id);
                    testCase.verifyTrue(ismember("check_" + string(r.script), names), id);
                end
            end
            testCase.verifyEqual(numel(unique(figures)), numel(figures), "Figures must be unique");
        end

        function gapRowsFollowTheSchema(testCase)
            rows = StrogatzFigureTest.readRows("gaps.json");
            figures = strings(1, numel(rows));
            states = strings(1, numel(rows));
            for k = 1:numel(rows)
                r = rows{k};
                id = "gaps row " + k;
                testCase.verifyEqual(sort(string(fieldnames(r)))', ["figure" "reason" "state"], id);
                figures(k) = string(r.figure);
                states(k) = string(r.state);
                testCase.verifyMatches(figures(k), "^\d+\.\d+\.\d+$", id);
                testCase.verifyTrue(ismember(states(k), ["gap" "n/a"]), id);
                testCase.verifyNotEmpty(r.reason, id);
                testCase.verifyLessThanOrEqual(strlength(string(r.reason)), 120, id);
            end
            na = ["3.5.7" "4.2.1" "4.5.2" "6.7.6" "9.2.3" "9.6.2" "10.7.2" "10.7.3" ...
                "11.5.4" "12.2.1" "13.2.5" "13.5.3"];   % decision D3
            testCase.verifyEqual(sort(figures(states == "n/a")), sort(na));
            testCase.verifyGreaterThanOrEqual(nnz(states == "gap"), 64);
            testCase.verifyEqual(numel(unique(figures)), numel(figures), "Figures must be unique");
            manifest = string(cellfun(@(r) r.figure, ...
                StrogatzFigureTest.readRows("manifest.json"), UniformOutput = false));
            testCase.verifyEmpty(intersect(figures, manifest), ...
                "A figure is either covered or a gap, not both");
        end

        function noDoneRowsGivesAPlaceholder(testCase)
            rows = StrogatzFigureTest.rowsFrom({});
            testCase.verifyEqual(string(fieldnames(rows)), "none");
            rows = StrogatzFigureTest.rowsFrom({struct("figure", "1.1.1", "state", "blocked: x")});
            testCase.verifyEqual(string(fieldnames(rows)), "none");
        end

        function aFailingScriptLeavesNoFigure(testCase)
            folder = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            fid = fopen(fullfile(folder, "fig_fail_probe.m"), "w");
            fprintf(fid, "function out = fig_fail_probe(options)\n");
            fprintf(fid, "arguments\n    options.Plot (1,1) logical = true\nend\n");
            fprintf(fid, "figure(Visible = ""off"");\n");
            fprintf(fid, "error(""strogatz:probe:fail"", ""probe"");\nend\n");
            fclose(fid);
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(folder));
            before = findall(groot, "Type", "figure");
            testCase.verifyError(@() StrogatzFigureTest.runScript("fig_fail_probe", false), ...
                "strogatz:probe:fail");
            testCase.verifyEmpty(StrogatzFigureTest.newFigures(before));
        end
    end

    methods (Access = protected)
        function check_fig_2_2_6(testCase, out)
            % f(x) = x - cos x has f'(x) = 1 + sin x >= 0, zero only at
            % isolated points, so f is strictly increasing and has one root.
            % Oracle: fzero. The eigenvalue is f'(x*) = 1 + sin x* > 0, so
            % the point is unstable.
            pt = out.Point;
            xs = fzero(@(x) x - cos(x), out.Values.x0);
            testCase.verifyEqual(pt.U, xs, AbsTol = 1e-9);
            testCase.verifyEqual(pt.Eigenvalues, 1 + sin(xs), AbsTol = 1e-8);
            testCase.verifyFalse(pt.Stable);
        end
    end
end

function J = cdJacobian(m, u, varargin)
% The Jacobian of m.rhs at U by central differences with step 1e-6. VARARGIN
% is the parameter argument of m.rhs, left out for a model without
% parameters. It does not use the engine's symbolic Jacobian, so it is an
% independent oracle for eigenvalues.
u = u(:);
n = numel(u);
J = zeros(n);
h = 1e-6;
for k = 1:n
    e = zeros(n, 1);
    e(k) = h;
    J(:, k) = (m.rhs(u + e, varargin{:}) - m.rhs(u - e, varargin{:})) / (2*h);
end
end
```

- [ ] **Step 2: Write the manifest with the first row, and gaps.json**

`books/strogatz/manifest.json`:
```json
[
  {
    "figure": "2.2.6",
    "edition": 3,
    "model": "cosine_flow.ode",
    "script": "fig_2_2_6",
    "api": ["equilibrium"],
    "values": {"x0": 0.7, "window": [-2, 2]},
    "source": {"x0": "chosen", "window": "chosen"},
    "check": "fzero root of x - cos x; eigenvalue 1 + sin x*, unstable",
    "state": "done"
  }
]
```

Build `gaps.json` from the local triage file (fields figure, state and reason only; never `notes` or `pdf_page`):

```bash
jq '[ .[] | select(.phasecont=="gap" or (.phasecont|startswith("n/a"))) | {figure, state:(if .phasecont=="gap" then "gap" else "n/a" end), reason:(if .phasecont=="gap" then .gap else "nothing to compute (decision D3)" end)} ] + [{figure:"13.2.3",state:"gap",reason:"the .ode reader has no sum(..)of(..) (decision D4)"},{figure:"13.2.4",state:"gap",reason:"the .ode reader has no sum(..)of(..) (decision D4)"}] | sort_by(.figure|split(".")|map(tonumber? // 0))' <wip>/strogatz-triage-2026-11-25/strogatz-modelbacked.json > books/strogatz/gaps.json
jq 'length, ([.[]|select(.state=="gap")]|length), ([.[]|select(.state=="n/a")]|length)' books/strogatz/gaps.json
jq -r '.[].reason' books/strogatz/gaps.json | sort | uniq -c
```
Expected: 76, 64, 12. Read every distinct reason: each must be our short description of the missing feature (for example "1-D phase line plot"), never book text. Shorten any reason over 120 characters by hand.

- [ ] **Step 3: Run the class to see it fail**

Run the whole-class command.
Expected: the schema tests pass except `manifestRowsFollowTheSchema` (the 2.2.6 script is missing); `figureMatchesItsOracle(Row=f2_2_6)` and `plotTrueDraws(Row=f2_2_6)` fail with `Undefined function 'fig_2_2_6'`.

- [ ] **Step 4: Write the loader and the script**

`books/strogatz/figures/strogatzModel.m`:
```matlab
function m = strogatzModel(name)
%STROGATZMODEL Load a model of books/strogatz/models with the JIT evaluator.
%   M = strogatzModel(NAME) returns phasecont.Model for the file NAME in
%   books/strogatz/models.
arguments
    name (1,1) string
end
m = phasecont.Model(fullfile(fileparts(mfilename("fullpath")), "..", "models", name), ...
    Evaluator = "jit");
end
```

`books/strogatz/figures/fig_2_2_6.m`:
```matlab
function out = fig_2_2_6(options)
%FIG_2_2_6 Strogatz, 3rd edition, figure 2.2.6, from cosine_flow.ode.
%   OUT = FIG_2_2_6() draws the figure; FIG_2_2_6(Plot = false) draws
%   nothing. OUT.Values holds every value the PhaseCont calls use, and
%   OUT.Point the equilibrium.
arguments
    options.Plot (1,1) logical = true
end
v = struct("x0", 0.7, "window", [-2 2]);
m = strogatzModel("cosine_flow.ode");
pt = phasecont.equilibrium(m, v.x0);
out = struct("Values", v, "Model", m, "Point", pt);
if ~options.Plot
    return
end
xs = linspace(v.window(1), v.window(2), 400);
ax = axes(figure);
plot(ax, xs, xs, xs, xs - m.rhs(xs));   % x - f(x) = cos x
hold(ax, "on");
plot(ax, pt.U, pt.U, "ko", MarkerFaceColor = "w");
xlabel(ax, "x");
legend(ax, "y = x", "y = cos x", "x*", Location = "best");
end
```

- [ ] **Step 5: Run the class to see it pass**

Run the whole-class command.
Expected: every test passes (2.2.6 row, plot row, both schema tests, the placeholder test and the failing-script test). Note the `strogatz-time 2.2.6` line.

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add tests/StrogatzFigureTest.m books/strogatz/manifest.json books/strogatz/gaps.json \
  books/strogatz/figures/strogatzModel.m books/strogatz/figures/fig_2_2_6.m \
  books/strogatz/models/cosine_flow.ode
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add the Strogatz figure test class and figure 2.2.6"
```

---

### Task 3: Row 9.3.2 (simulate, Lorenz)

**Files:**
- Create: `books/strogatz/figures/fig_9_3_2.m`
- Modify: `books/strogatz/manifest.json` (append one entry)
- Modify: `tests/StrogatzFigureTest.m` (add `check_fig_9_3_2`)
- Add: `books/strogatz/models/lorenz.ode`

**Interfaces:**
- Consumes: `strogatzModel`, the test class from Task 2.
- Produces: `fig_9_3_2` with `out.Trajectory` (phasecont.Trajectory). `lorenz.ode` is reused by Task 8.

- [ ] **Step 1: Add the check method** (in `methods (Access = protected)`)

```matlab
function check_fig_9_3_2(testCase, out)
% Oracle: ode45 on the same equations with RelTol = AbsTol = 1e-12, up to
% the first output time at or after t = 1. The horizon is short because
% the flow is chaotic: nearby orbits separate, so a long comparison would
% measure the separation, not the solver.
v = out.Values;
tr = out.Trajectory;
testCase.verifyEqual(tr.T(end), v.tspan(2), AbsTol = 1e-12);
k = find(tr.T >= 1, 1);
f = @(t, u) [v.sigma*(u(2) - u(1)); v.r*u(1) - u(2) - u(1)*u(3); u(1)*u(2) - v.b*u(3)];
[~, U] = ode45(f, [0 tr.T(k)], v.x0(:), odeset(RelTol = 1e-12, AbsTol = 1e-12));
err = norm(tr.X(k, :) - U(end, :)) / norm(U(end, :));
fprintf("strogatz-err 9.3.2 %.3g\n", err);
testCase.verifyLessThan(err, 1e-6);
end
```

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "9.3.2",
    "edition": 3,
    "model": "lorenz.ode",
    "script": "fig_9_3_2",
    "api": ["simulate"],
    "values": {"sigma": 10, "r": 28, "b": 2.6666666666666665, "x0": [0, 1, 0],
               "tspan": [0, 50], "relTol": 1e-10, "absTol": 1e-12},
    "source": {"sigma": "stated", "r": "stated", "b": "stated", "x0": "stated",
               "tspan": "chosen", "relTol": "chosen", "absTol": "chosen"},
    "check": "ode45 at RelTol and AbsTol 1e-12 to t = 1, relative error below 1e-6",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f9_3_2 to see it fail**

Expected: `figureMatchesItsOracle(Row=f9_3_2)` and `plotTrueDraws(Row=f9_3_2)` fail with `Undefined function 'fig_9_3_2'`, and `manifestRowsFollowTheSchema` fails on the missing script file.

- [ ] **Step 4: Write the script**

```matlab
function out = fig_9_3_2(options)
%FIG_9_3_2 Strogatz, 3rd edition, figure 9.3.2, from lorenz.ode.
%   OUT = FIG_9_3_2() draws the figure; FIG_9_3_2(Plot = false) draws
%   nothing. OUT.Values holds every value the PhaseCont calls use, and
%   OUT.Trajectory the orbit.
arguments
    options.Plot (1,1) logical = true
end
v = struct("sigma", 10, "r", 28, "b", 8/3, "x0", [0 1 0], "tspan", [0 50], ...
    "relTol", 1e-10, "absTol", 1e-12);
m = strogatzModel("lorenz.ode");
tr = phasecont.simulate(m, v.x0, v.tspan, ...
    Params = struct("sigma", v.sigma, "r", v.r, "b", v.b), ...
    RelTol = v.relTol, AbsTol = v.absTol);
out = struct("Values", v, "Model", m, "Trajectory", tr);
if ~options.Plot
    return
end
ax = axes(figure);
tr.plot(Var = ["x" "z"], Parent = ax);
xlabel(ax, "x");
ylabel(ax, "z");
end
```

- [ ] **Step 5: Run row f9_3_2 to see it pass**

Expected: both row tests and both schema tests pass. Note `strogatz-time` and `strogatz-err`.

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_9_3_2.m books/strogatz/models/lorenz.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 9.3.2 (Lorenz orbit)"
```

---

### Task 4: Row 6.4.7 (PhasePlane app)

**Files:**
- Create: `books/strogatz/figures/fig_6_4_7.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/rabbits_sheep.ode`

**Interfaces:**
- Consumes: `strogatzModel`, `cdJacobian`, the test class.
- Produces: `fig_6_4_7` with `out.App` (phasecont.app.PhasePlane), `out.Points` (1-by-4 cell of Point), `out.Trajectories` (1-by-4 cell of Trajectory). The row test deletes `out.App` in its teardown.

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_6_4_7(testCase, out)
% x' = x(3 - x - 2y), y' = y(2 - x - y). Equilibria: x = 0 or x = 3 - 2y,
% and y = 0 or y = 2 - x, which gives (0,0), (0,2), (3,0) and (1,1).
% J = [3 - 2x - 2y, -2x; -y, 2 - x - 2y]:
%   (0,0): diag(3, 2), eigenvalues 3 and 2, unstable node;
%   (0,2): [-1 0; -2 -2], eigenvalues -1 and -2, stable node;
%   (3,0): [-3 -6; 0 -1], eigenvalues -3 and -1, stable node;
%   (1,1): [-1 -2; -1 -1], eigenvalues -1 +- sqrt(2), a saddle.
% The guesses in Values.points are listed in that order. The eigenvalues
% are also checked against eig of a central-difference Jacobian.
expected = [0 0; 0 2; 3 0; 1 1];
stable = [false true true false];
m = out.Model;
for k = 1:4
    pt = out.Points{k};
    testCase.verifyEqual(pt.U', expected(k, :), AbsTol = 1e-9);
    testCase.verifyEqual(sort(pt.Eigenvalues), sort(eig(cdJacobian(m, pt.U))), AbsTol = 1e-6);
    testCase.verifyEqual(pt.Stable, stable(k));
end
ax = out.App.Axes;
testCase.verifyNumElements(findall(ax, "Tag", "pc-equilibrium-stable"), 2);
testCase.verifyNumElements(findall(ax, "Tag", "pc-equilibrium-unstable"), 2);
testCase.verifyNumElements(findall(ax, "Tag", "pc-nullcline"), 2);
testCase.verifyNumElements(findall(ax, "Tag", "pc-trajectory"), 4);
% Every start leaves the saddle's stable manifold, so each orbit ends at
% one of the two stable nodes. The slowest rate there is 1 and T = 20.
for k = 1:4
    e = out.Trajectories{k}.X(end, :);
    d = min(vecnorm(expected(2:3, :) - e, 2, 2));
    testCase.verifyLessThan(d, 1e-3);
end
end
```

If the probe showed different marker or nullcline counts (for example one marker object per class), stop and record what the app draws before changing the expected counts; the counts above follow from four points and two variables.

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "6.4.7",
    "edition": 3,
    "model": "rabbits_sheep.ode",
    "script": "fig_6_4_7",
    "api": ["PhasePlane.findEquilibrium", "PhasePlane.addTrajectory", "PhasePlane.showNullclines"],
    "values": {"points": [[0.1, 0.1], [0.1, 1.9], [2.9, 0.1], [0.9, 1.1]],
               "window": [0, 3.5, 0, 2.5], "T": 20},
    "source": {"points": "chosen", "window": "chosen", "T": "chosen"},
    "check": "four equilibria, eigenvalues against eig of a central-difference Jacobian, marker counts",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f6_4_7 to see it fail** (undefined `fig_6_4_7`)

- [ ] **Step 4: Write the script**

```matlab
function out = fig_6_4_7(options)
%FIG_6_4_7 Strogatz, 3rd edition, figure 6.4.7, from rabbits_sheep.ode.
%   OUT = FIG_6_4_7() opens the PhasePlane app; FIG_6_4_7(Plot = false)
%   builds it with Visible = "off". OUT.App is the app, OUT.Points the four
%   equilibria and OUT.Trajectories the four orbits, in the order of
%   OUT.Values.points.
arguments
    options.Plot (1,1) logical = true
end
v = struct("points", [0.1 0.1; 0.1 1.9; 2.9 0.1; 0.9 1.1], ...
    "window", [0 3.5 0 2.5], "T", 20);
m = strogatzModel("rabbits_sheep.ode");
visible = "off";
if options.Plot
    visible = "on";
end
app = phasecont.app.PhasePlane(m, Visible = visible);
app.setWindow(v.window);
app.setT(v.T);
app.showNullclines(true);
n = size(v.points, 1);
points = cell(1, n);
trajectories = cell(1, n);
for k = 1:n
    points{k} = app.findEquilibrium(v.points(k, 1), v.points(k, 2));
    trajectories{k} = app.addTrajectory(v.points(k, 1), v.points(k, 2));
end
out = struct("Values", v, "Model", m, "App", app);
out.Points = points;
out.Trajectories = trajectories;
end
```

- [ ] **Step 5: Run row f6_4_7 to see it pass** (the plot test skips for this row by design)

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_6_4_7.m books/strogatz/models/rabbits_sheep.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 6.4.7 (PhasePlane app)"
```

---

### Task 5: Row 3.4.2 (branch point and switch)

**Files:**
- Create: `books/strogatz/figures/fig_3_4_2.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/pitchfork.ode`

**Interfaces:**
- Produces: `fig_3_4_2` with `out.Branches`, a 1-by-2 Branch array (variant A) or 1-by-3 (variant B: the two switch branches, then the fallback run). The check handles both.

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_3_4_2(testCase, out)
% f(x, r) = r x - x^3, f_x = r - 3x^2, f_r = x. On x = 0, f_x = r and
% f_r = 0, so both vanish at (0, 0): a branch point at r = 0. The other
% branch through it is r = x^2, where f_x = r - 3r = -2r.
brs = out.Branches;
bp = brs(1).points("BP");
testCase.verifyNumElements(bp, 1);
testCase.verifyEqual(brs(1).Alpha(bp.Index), 0, AbsTol = 1e-6);
testCase.verifyLessThan(max(abs(brs(1).U)), 1e-10);
testCase.verifyEqual(brs(1).Eigenvalues, brs(1).Alpha, AbsTol = 1e-8);
x = vertcat(brs(2:end).U);
r = vertcat(brs(2:end).Alpha);
lambda = vertcat(brs(2:end).Eigenvalues);
testCase.verifyEqual(x.^2, r, AbsTol = 1e-8);
testCase.verifyEqual(lambda, -2*r, AbsTol = 1e-8);
testCase.verifyTrue(any(x < -0.9) && any(x > 0.9), "Both halves of r = x^2");
end
```

- [ ] **Step 2: Append the manifest entry** (variant A; for variant B add `"x1": -1, "r1": 1, "bounds1": [-1, 1.5]`, all `"chosen"`)

```json
  {
    "figure": "3.4.2",
    "edition": 3,
    "model": "pitchfork.ode",
    "script": "fig_3_4_2",
    "api": ["continuation", "Switch BP1", "Branch.plot"],
    "values": {"r0": -1, "x0": 0, "bounds": [-1, 1], "hmax": 0.05},
    "source": {"r0": "chosen", "x0": "chosen", "bounds": "chosen", "hmax": "chosen"},
    "check": "branch point at r = 0; x = 0 with eigenvalue r; x^2 = r with eigenvalue -2r",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f3_4_2 to see it fail**

- [ ] **Step 4: Write the script** (variant A; the probe log says which)

```matlab
function out = fig_3_4_2(options)
%FIG_3_4_2 Strogatz, 3rd edition, figure 3.4.2, from pitchfork.ode.
%   OUT = FIG_3_4_2() draws the figure; FIG_3_4_2(Plot = false) draws
%   nothing. OUT.Branches holds the branch x = 0 and the branch that
%   leaves its branch point.
arguments
    options.Plot (1,1) logical = true
end
v = struct("r0", -1, "x0", 0, "bounds", [-1 1], "hmax", 0.05);
m = strogatzModel("pitchfork.ode");
brs = phasecont.continuation(m, v.x0, Params = struct("r", v.r0), Free = "r", ...
    Bounds = v.bounds, Hmax = v.hmax, Switch = "BP1");
out = struct("Values", v, "Model", m, "Branches", brs);
if ~options.Plot
    return
end
ax = axes(figure);
brs.plot(Parent = ax);
xlabel(ax, "r");
ylabel(ax, "x");
end
```

Variant B (only if the probe showed one half missing): `v` gains `"x1", -1, "r1", 1, "bounds1", [-1 1.5]`, and after the switch run:
```matlab
more = phasecont.continuation(m, v.x1, Params = struct("r", v.r1), Free = "r", ...
    Bounds = v.bounds1, Hmax = v.hmax, Direction = -1);
brs = [brs more];
```

- [ ] **Step 5: Run row f3_4_2 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_3_4_2.m books/strogatz/models/pitchfork.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 3.4.2 (pitchfork branch switch)"
```

---

### Task 6: Row 3.6.2 (fold curve and cusp)

**Files:**
- Create: `books/strogatz/figures/fig_3_6_2.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/imperfect.ode`

**Interfaces:**
- Produces: `fig_3_6_2` with `out.Branch` (EP branch in h) and `out.Curve` (LP curve in [h r]).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_3_6_2(testCase, out)
% f(x) = h + r x - x^3. A fold has f = 0 and f_x = r - 3x^2 = 0, so
% r = 3x^2 and h = -r x + x^3 = -2x^3. Then 27 h^2 = 108 x^6 = 4 r^3.
% At r = 1 the folds of the h branch sit at x = -+1/sqrt(3), so
% h = +-2/(3 sqrt(3)). The cusp has f_xx = -6x = 0 as well: x = 0, so
% (h, r) = (0, 0).
b = out.Branch;
lp = b.points("LP");
testCase.verifyNumElements(lp, 2);
testCase.verifyEqual(sort(b.Alpha([lp.Index])), [-1; 1] * 2/(3*sqrt(3)), AbsTol = 1e-6);
c = out.Curve;
x = c.U;
h = c.Alpha(:, 1);
r = c.Alpha(:, 2);
testCase.verifyEqual(r, 3*x.^2, AbsTol = 1e-8);
testCase.verifyEqual(h, -2*x.^3, AbsTol = 1e-8);
testCase.verifyEqual(27*h.^2, 4*r.^3, AbsTol = 1e-8);
cp = c.points("CP");
testCase.verifyNumElements(cp, 1);
testCase.verifyEqual(c.Alpha(cp.Index, :), [0 0], AbsTol = 1e-6);
testCase.verifyTrue(any(x < 0) && any(x > 0), "The curve passes the cusp");
end
```

- [ ] **Step 2: Append the manifest entry** (`direction` from the probe)

```json
  {
    "figure": "3.6.2",
    "edition": 3,
    "model": "imperfect.ode",
    "script": "fig_3_6_2",
    "api": ["continuation", "Type LP", "Branch.plot"],
    "values": {"h0": 0, "r0": 1, "x0": -1, "hBounds": [-1, 1], "rBounds": [-0.5, 1.5], "direction": 1},
    "source": {"h0": "chosen", "r0": "chosen", "x0": "chosen", "hBounds": "chosen",
               "rBounds": "chosen", "direction": "chosen"},
    "check": "fold curve r = 3x^2, h = -2x^3, 27h^2 = 4r^3; cusp at the origin",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f3_6_2 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_3_6_2(options)
%FIG_3_6_2 Strogatz, 3rd edition, figure 3.6.2, from imperfect.ode.
%   OUT = FIG_3_6_2() draws the fold curve in the (h, r) plane;
%   FIG_3_6_2(Plot = false) draws nothing. OUT.Branch is the equilibrium
%   branch in h at r = r0 and OUT.Curve the fold curve from its first fold.
arguments
    options.Plot (1,1) logical = true
end
v = struct("h0", 0, "r0", 1, "x0", -1, "hBounds", [-1 1], "rBounds", [-0.5 1.5], ...
    "direction", 1);
m = strogatzModel("imperfect.ode");
b = phasecont.continuation(m, v.x0, Params = struct("h", v.h0, "r", v.r0), ...
    Free = "h", Bounds = v.hBounds);
lp = b.points("LP");
c = phasecont.continuation(m, lp(1), Type = "LP", Free = ["h" "r"], ...
    Bounds = v.rBounds, Direction = v.direction);
out = struct("Values", v, "Model", m, "Branch", b, "Curve", c);
if ~options.Plot
    return
end
ax = axes(figure);
c.plot(Parent = ax);
xlabel(ax, "h");
ylabel(ax, "r");
end
```

- [ ] **Step 5: Run row f3_6_2 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_3_6_2.m books/strogatz/models/imperfect.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 3.6.2 (fold curve and cusp)"
```

---

### Task 7: Row 8.3.4 (Hopf curve)

**Files:**
- Create: `books/strogatz/figures/fig_8_3_4.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/chemical_oscillator.ode`

**Interfaces:**
- Produces: `fig_8_3_4` with `out.Branch` (EP in b) and `out.Curves` (1-by-1 or 1-by-2 H curve array, per the probe).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_8_3_4(testCase, out)
% f = a - x - 4xy/u, g = b x (1 - y/u), u = 1 + x^2. g = 0 with x > 0
% gives y = u; then f = a - 5x = 0, so x* = a/5 and y* = 1 + a^2/25.
% At the equilibrium:
%   f_x = -1 - 4y(1 - x^2)/u^2 = (3x^2 - 5)/u,   f_y = -4x/u,
%   g_x = b - b y (1 - x^2)/u^2 = 2b x^2/u,      g_y = -b x/u,
% so tau = (3x^2 - 5 - b x)/u and Delta = 5 b x/u > 0. Hopf has tau = 0:
% b = 3x - 5/x = 3a/5 - 25/a. At a = 10: x = 2, u = 5, b = 3.5 and
% omega^2 = Delta = 7.
b = out.Branch;
hp = b.points("H");
testCase.verifyNumElements(hp, 1);
testCase.verifyEqual(b.Alpha(hp.Index), 3.5, AbsTol = 1e-6);
testCase.verifyEqual(hp.Data.omega, sqrt(7), AbsTol = 1e-6);
a = vertcat(out.Curves.Alpha);
U = vertcat(out.Curves.U);
testCase.verifyEqual(a(:, 2), 3*a(:, 1)/5 - 25./a(:, 1), AbsTol = 1e-8);
testCase.verifyEqual(U(:, 1), a(:, 1)/5, AbsTol = 1e-8);
testCase.verifyEqual(U(:, 2), 1 + a(:, 1).^2/25, AbsTol = 1e-8);
% b = 0.5 and b = 10 bound the curve: 3a^2/5 - b a - 25 = 0 gives
% a = (5b + sqrt(25b^2 + 1500))/6, so a runs from 6.885 to 18.87.
ends = (5*[0.5 10] + sqrt(25*[0.5 10].^2 + 1500))/6;
testCase.verifyEqual([min(a(:, 1)) max(a(:, 1))], ends, AbsTol = 1e-3);
end
```

- [ ] **Step 2: Append the manifest entry** (`directions` from the probe: `[1]`, `[-1]` or `[1, -1]`)

```json
  {
    "figure": "8.3.4",
    "edition": 3,
    "model": "chemical_oscillator.ode",
    "script": "fig_8_3_4",
    "api": ["continuation", "Type H", "Branch.plot"],
    "values": {"a0": 10, "b0": 2, "x0": [2, 5], "bBounds": [2, 5], "curveBounds": [0.5, 10],
               "directions": [1, -1]},
    "source": {"a0": "chosen", "b0": "chosen", "x0": "chosen", "bBounds": "chosen",
               "curveBounds": "chosen", "directions": "chosen"},
    "check": "Hopf at b = 3.5, omega sqrt(7); Hopf curve b = 3a/5 - 25/a, x* = a/5",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f8_3_4 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_8_3_4(options)
%FIG_8_3_4 Strogatz, 3rd edition, figure 8.3.4, from chemical_oscillator.ode.
%   OUT = FIG_8_3_4() draws the Hopf curve in the (a, b) plane;
%   FIG_8_3_4(Plot = false) draws nothing. OUT.Branch is the equilibrium
%   branch in b and OUT.Curves the Hopf curve runs from its Hopf point.
arguments
    options.Plot (1,1) logical = true
end
v = struct("a0", 10, "b0", 2, "x0", [2 5], "bBounds", [2 5], "curveBounds", [0.5 10], ...
    "directions", [1 -1]);
m = strogatzModel("chemical_oscillator.ode");
b = phasecont.continuation(m, v.x0, Params = struct("a", v.a0, "b", v.b0), ...
    Free = "b", Bounds = v.bBounds);
hp = b.points("H");
curves = phasecont.Branch.empty(1, 0);
for d = v.directions
    curves(end + 1) = phasecont.continuation(m, hp(1), Type = "H", Free = ["a" "b"], ...
        Bounds = v.curveBounds, Direction = d); %#ok<AGROW>
end
out = struct("Values", v, "Model", m, "Branch", b, "Curves", curves);
if ~options.Plot
    return
end
ax = axes(figure);
curves.plot(Parent = ax);
xlabel(ax, "a");
ylabel(ax, "b");
end
```

If `phasecont.Branch.empty(1, 0)` is refused by the class, build `curves` with `[curves c]` from a first run instead; the probe already ran the same calls.

- [ ] **Step 5: Run row f8_3_4 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_8_3_4.m books/strogatz/models/chemical_oscillator.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 8.3.4 (Hopf curve)"
```

---

### Task 8: Row 9.5.1 (cycles from a Hopf point, Lorenz)

**Files:**
- Create: `books/strogatz/figures/fig_9_5_1.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Uses: `books/strogatz/models/lorenz.ode` (committed in Task 3)

**Interfaces:**
- Produces: `fig_9_5_1` with `out.Branch` (EP in r) and `out.Cycles` (LC branch).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_9_5_1(testCase, out)
% At C+- = (+-sqrt(b(r-1)), +-sqrt(b(r-1)), r-1) the Jacobian of the
% Lorenz equations has the characteristic polynomial
%   lambda^3 + (sigma + b + 1) lambda^2 + b(sigma + r) lambda + 2 b sigma (r - 1).
% A cubic l^3 + p l^2 + q l + s has roots +-i omega when p q = s, with
% omega^2 = q. Here p q = s reads (sigma + b + 1)(sigma + r) = 2 sigma (r - 1),
% so r_H = sigma (sigma + b + 3)/(sigma - b - 1). With sigma = 10 and
% b = 8/3: r_H = 470/19, omega^2 = b (sigma + r_H) = 1760/19 and the third
% root is -p = -41/3. The first cycles have period near 2 pi/omega.
% Cycle oracle: ode45 from the first mesh point over one period must close
% the orbit and pass the mesh midpoint.
v = out.Values;
rH = v.sigma*(v.sigma + v.b + 3)/(v.sigma - v.b - 1);
omega = sqrt(v.b*(v.sigma + rH));
b = out.Branch;
hp = b.points("H");
testCase.verifyNumElements(hp, 1);
testCase.verifyEqual(b.Alpha(hp.Index), rH, AbsTol = 1e-6);
testCase.verifyEqual(hp.Data.omega, omega, AbsTol = 1e-6);
lc = out.Cycles;
testCase.verifyEqual(lc.Period(1), 2*pi/omega, RelTol = 1e-3);
i = find(lc.Alpha < rH - 1, 1);
testCase.verifyNotEmpty(i, "The cycles reach r < r_H - 1");
[t, x] = lc.cycle(i);
r = lc.Alpha(i);
f = @(s, u) [v.sigma*(u(2) - u(1)); r*u(1) - u(2) - u(1)*u(3); u(1)*u(2) - v.b*u(3)];
mid = ceil(numel(t)/2);
[~, U] = ode45(f, [0 t(mid) t(end)], x(1, :)', odeset(RelTol = 1e-12, AbsTol = 1e-12));
scale = norm(x(1, :));
testCase.verifyLessThan(norm(U(end, :) - x(1, :))/scale, 1e-5, "Orbit closes");
testCase.verifyLessThan(norm(U(2, :) - x(mid, :))/scale, 1e-5, "Orbit passes the midpoint");
fprintf("strogatz-info 9.5.1 lowest r %.6g, largest period %.6g\n", min(lc.Alpha), max(lc.Period));
end
```

- [ ] **Step 2: Append the manifest entry** (`direction` and `maxSteps` from the probe)

```json
  {
    "figure": "9.5.1",
    "edition": 3,
    "model": "lorenz.ode",
    "script": "fig_9_5_1",
    "api": ["equilibrium", "continuation", "Type LC", "Branch.plot"],
    "values": {"sigma": 10, "b": 2.6666666666666665, "r0": 20, "guess": [7, 7, 19],
               "epBounds": [20, 30], "ntst": 40, "ncol": 4, "lcBounds": [13, 25],
               "direction": -1, "maxSteps": 400},
    "source": {"sigma": "stated", "b": "stated", "r0": "chosen", "guess": "chosen",
               "epBounds": "chosen", "ntst": "chosen", "ncol": "chosen", "lcBounds": "chosen",
               "direction": "chosen", "maxSteps": "chosen"},
    "check": "Hopf at r = 470/19 with omega^2 = 1760/19; a cycle checked by ode45 closure",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f9_5_1 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_9_5_1(options)
%FIG_9_5_1 Strogatz, 3rd edition, figure 9.5.1, from lorenz.ode.
%   OUT = FIG_9_5_1() draws the equilibrium branch and the cycles that leave
%   its Hopf point; FIG_9_5_1(Plot = false) draws nothing. OUT.Branch is
%   the branch of C+ in r and OUT.Cycles the LC branch.
arguments
    options.Plot (1,1) logical = true
end
v = struct("sigma", 10, "b", 8/3, "r0", 20, "guess", [7 7 19], "epBounds", [20 30], ...
    "ntst", 40, "ncol", 4, "lcBounds", [13 25], "direction", -1, "maxSteps", 400);
m = strogatzModel("lorenz.ode");
pt = phasecont.equilibrium(m, v.guess, Params = struct("sigma", v.sigma, "r", v.r0, "b", v.b));
b = phasecont.continuation(m, pt, Free = "r", Bounds = v.epBounds);
hp = b.points("H");
lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "r", NTST = v.ntst, ...
    NCOL = v.ncol, Bounds = v.lcBounds, Direction = v.direction, MaxSteps = v.maxSteps);
out = struct("Values", v, "Model", m, "Branch", b, "Cycles", lc);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
b.plot(Var = "z", Parent = ax);
lc.plot(Var = "z", Parent = ax);
xlabel(ax, "r");
ylabel(ax, "z");
end
```

- [ ] **Step 5: Run row f9_5_1 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_9_5_1.m books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 9.5.1 (Lorenz cycles from Hopf)"
```

---

### Task 9: Row 8.4.3 (homoclinic curve)

**Files:**
- Create: `books/strogatz/figures/fig_8_4_3.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/homoclinic.ode`

**Interfaces:**
- Produces: `fig_8_4_3` with `out.Branch` (EP in mu), `out.Cycles` (LC), `out.Index` (the LC point the HOM run starts from) and `out.Homoclinic` (HOM curve in [mu c]). The test adds a local function `splitDistance(mu, c)`.

- [ ] **Step 1: Add the oracle and the check method**

Local function, after `cdJacobian` at the end of the test file:

```matlab
function d = splitDistance(mu, c)
% Signed distance on y = 0, x > 1, between the unstable and the stable
% manifold of the saddle at the origin of x' = y, y' = mu y + x - x^2 + c x y.
% The Jacobian there is [0 1; 1 mu], with eigenvalues (mu +- sqrt(mu^2 + 4))/2
% and eigenvectors (1, lambda). The unstable branch leaves into x > 0, y > 0,
% passes above the focus at (1, 0) (the flow turns clockwise there, since
% x' = y) and first meets y = 0 going down. The stable branch, followed
% backward in time from (1, lambda_s) eps, meets y = 0 going up. The two
% crossings coincide at a homoclinic orbit, so d changes sign at mu_c.
f = @(t, u) [u(2); mu*u(2) + u(1) - u(1)^2 + c*u(1)*u(2)];
lu = (mu + sqrt(mu^2 + 4))/2;
ls = (mu - sqrt(mu^2 + 4))/2;
e = 1e-6;
down = odeset(RelTol = 1e-12, AbsTol = 1e-14, Events = @(t, u) yZero(u, -1));
up = odeset(RelTol = 1e-12, AbsTol = 1e-14, Events = @(t, u) yZero(u, 1));
[~, ~, ~, uu] = ode45(f, [0 200], e*[1; lu], down);
[~, ~, ~, us] = ode45(@(t, u) -f(t, u), [0 200], e*[1; ls], up);
d = uu(1, 1) - us(1, 1);
end

function [value, isterminal, direction] = yZero(u, dir)
value = u(2);
isterminal = 1;
direction = dir;
end
```

Check method:

```matlab
function check_fig_8_4_3(testCase, out)
% Equilibria: y = 0 and x - x^2 = 0, so (0, 0) and (1, 0). At (1, 0) the
% Jacobian is [0 1; -1 mu + c]; with c = 1, tau = mu + 1 and Delta = 1, so
% the Hopf point is at mu = -1 with omega = 1. The cycles grow from there
% until they meet the saddle at the origin at mu_c. Oracle for mu_c: fzero
% of splitDistance (see that function) over the bracket in Values.
v = out.Values;
b = out.Branch;
hp = b.points("H");
testCase.verifyNumElements(hp, 1);
testCase.verifyEqual(b.Alpha(hp.Index), -1, AbsTol = 1e-6);
testCase.verifyEqual(hp.Data.omega, 1, AbsTol = 1e-6);
hom = out.Homoclinic;
testCase.verifyEqual(hom.Alpha(1, 2), v.c, AbsTol = 1e-8);
for k = [1 size(hom.Alpha, 1)]
    c = hom.Alpha(k, 2);
    muc = fzero(@(mu) splitDistance(mu, c), v.bracket);
    fprintf("strogatz-err 8.4.3 c %.6g mu %.10g oracle %.10g\n", c, hom.Alpha(k, 1), muc);
    testCase.verifyEqual(hom.Alpha(k, 1), muc, AbsTol = 1e-4);
end
end
```

If the last HOM point has c outside the range where the bracket still holds a sign change, `fzero` errors; then check only the first point and note it in the report.

- [ ] **Step 2: Append the manifest entry** (values from the probe)

```json
  {
    "figure": "8.4.3",
    "edition": 3,
    "model": "homoclinic.ode",
    "script": "fig_8_4_3",
    "api": ["continuation", "Type LC", "Type HOM", "simulate", "Trajectory.plot"],
    "values": {"mu0": -1.5, "c": 1, "x0": [1, 0], "epBounds": [-1.5, -0.5], "ntst": 80,
               "ncol": 4, "hmax": 0.05, "maxNewton": 15, "maxSteps": 1700, "periodT": 50,
               "homBounds": [0.9, 1.1], "homHmax": 0.01, "bracket": [-0.95, -0.8],
               "panels": [-0.92, -0.8], "starts": [[1.1, 0], [0.01, 0.01]], "T": 30},
    "source": {"mu0": "chosen", "c": "stated", "x0": "chosen", "epBounds": "chosen",
               "ntst": "chosen", "ncol": "chosen", "hmax": "chosen", "maxNewton": "chosen",
               "maxSteps": "chosen", "periodT": "chosen", "homBounds": "chosen",
               "homHmax": "chosen", "bracket": "chosen", "panels": "chosen",
               "starts": "chosen", "T": "chosen"},
    "check": "Hopf at mu = -1, omega 1; homoclinic mu against an ode45 manifold-splitting oracle",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f8_4_3 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_8_4_3(options)
%FIG_8_4_3 Strogatz, 3rd edition, figure 8.4.3, from homoclinic.ode.
%   OUT = FIG_8_4_3() draws phase portraits below, at and above the
%   homoclinic value of mu; FIG_8_4_3(Plot = false) draws nothing.
%   OUT.Homoclinic is the HOM curve in (mu, c) and OUT.Homoclinic.Alpha(1, 1)
%   the homoclinic mu at c = Values.c.
arguments
    options.Plot (1,1) logical = true
end
v = struct("mu0", -1.5, "c", 1, "x0", [1 0], "epBounds", [-1.5 -0.5], "ntst", 80, ...
    "ncol", 4, "hmax", 0.05, "maxNewton", 15, "maxSteps", 1700, "periodT", 50, ...
    "homBounds", [0.9 1.1], "homHmax", 0.01, "bracket", [-0.95 -0.8], ...
    "panels", [-0.92 -0.8], "starts", [1.1 0; 0.01 0.01], "T", 30);
m = strogatzModel("homoclinic.ode");
b = phasecont.continuation(m, v.x0, Params = struct("mu", v.mu0, "c", v.c), ...
    Free = "mu", Bounds = v.epBounds);
hp = b.points("H");
lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "mu", NTST = v.ntst, ...
    NCOL = v.ncol, Hmax = v.hmax, MaxNewton = v.maxNewton, MaxSteps = v.maxSteps);
i = find(lc.Period > v.periodT, 1);
hom = phasecont.continuation(m, lc, Type = "HOM", Index = i, Free = ["mu" "c"], ...
    Bounds = v.homBounds, Hmax = v.homHmax, MaxNewton = v.maxNewton);
out = struct("Values", v, "Model", m, "Branch", b, "Cycles", lc, "Index", i, ...
    "Homoclinic", hom);
if ~options.Plot
    return
end
mus = [v.panels(1) hom.Alpha(1, 1) v.panels(2)];
tl = tiledlayout(figure, 1, 3);
for k = 1:3
    ax = nexttile(tl);
    hold(ax, "on");
    for s = 1:size(v.starts, 1)
        tr = phasecont.simulate(m, v.starts(s, :), [0 v.T], ...
            Params = struct("mu", mus(k), "c", v.c));
        tr.plot(Parent = ax);
    end
    title(ax, sprintf("mu = %.4f", mus(k)));
end
end
```

- [ ] **Step 5: Run row f8_4_3 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_8_4_3.m books/strogatz/models/homoclinic.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 8.4.3 (homoclinic curve)"
```

---

### Task 10: Row 10.1.3 (cobweb)

**Files:**
- Create: `books/strogatz/figures/fig_10_1_3.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/cosine_map.ode`

**Interfaces:**
- Produces: `fig_10_1_3` with `out.Trajectory` (the iterates, from `simulate` when `Plot = false`, from `cobweb` when `Plot = true`) and `out.Point` (the fixed point).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_10_1_3(testCase, out)
% The map is x(n+1) = cos x(n). Each returned iterate must be the cosine of
% the one before. A fixed point solves x = cos x (oracle: fzero), and its
% multiplier is the derivative -sin x*. cobweb must return the same
% iterates as simulate over [0 N] and draw its four tagged parts.
v = out.Values;
x = out.Trajectory.X(:, 1);
testCase.verifyNumElements(x, v.N + 1);
testCase.verifyEqual(x(2:end), cos(x(1:end-1)), RelTol = 1e-15);
xs = fzero(@(x) x - cos(x), v.guess);
testCase.verifyEqual(out.Point.U, xs, AbsTol = 1e-12);
testCase.verifyEqual(out.Point.Eigenvalues, -sin(xs), AbsTol = 1e-12);
f = figure(Visible = "off");
testCase.addTeardown(@() delete(f));
ax = axes(f);
[~, tr] = phasecont.cobweb(out.Model, v.x0, v.N, Parent = ax);
testCase.verifyEqual(tr.X, out.Trajectory.X);
for tag = ["pc-cobweb-graph" "pc-cobweb-diagonal" "pc-cobweb" "pc-start"]
    testCase.verifyNotEmpty(findall(ax, "Tag", tag), tag);
end
end
```

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "10.1.3",
    "edition": 3,
    "model": "cosine_map.ode",
    "script": "fig_10_1_3",
    "api": ["cobweb", "simulate", "equilibrium"],
    "values": {"x0": 1, "N": 20, "guess": 0.7},
    "source": {"x0": "chosen", "N": "chosen", "guess": "chosen"},
    "check": "each iterate is cos of the one before; fixed point by fzero, multiplier -sin x*",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f10_1_3 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_10_1_3(options)
%FIG_10_1_3 Strogatz, 3rd edition, figure 10.1.3, from cosine_map.ode.
%   OUT = FIG_10_1_3() draws the cobweb; FIG_10_1_3(Plot = false) draws
%   nothing and gets the same iterates from simulate. OUT.Trajectory holds
%   the iterates and OUT.Point the fixed point.
arguments
    options.Plot (1,1) logical = true
end
v = struct("x0", 1, "N", 20, "guess", 0.7);
m = strogatzModel("cosine_map.ode");
if options.Plot
    ax = axes(figure);
    [~, tr] = phasecont.cobweb(m, v.x0, v.N, Parent = ax);
    xlabel(ax, "x_n");
    ylabel(ax, "x_{n+1}");
else
    tr = phasecont.simulate(m, v.x0, [0 v.N]);
end
pt = phasecont.equilibrium(m, v.guess);
out = struct("Values", v, "Model", m, "Trajectory", tr, "Point", pt);
end
```

- [ ] **Step 5: Run row f10_1_3 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_10_1_3.m books/strogatz/models/cosine_map.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 10.1.3 (cobweb of the cosine map)"
```

---

### Task 11: Row 10.3.4 (map continuation)

**Files:**
- Create: `books/strogatz/figures/fig_10_3_4.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/logistic_map.ode`

**Interfaces:**
- Produces: `fig_10_3_4` with `out.Transcritical` (1-by-2 Branch: x = 0, then the switched branch) and `out.Doubling` (1-by-2 Branch: the fixed points, then the 2-cycle at Iterate 2).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_10_3_4(testCase, out)
% F(x) = r x (1 - x), F'(x) = r (1 - 2x). Fixed points: x = 0 with
% multiplier r, and x* = 1 - 1/r with multiplier r (1 - 2 + 2/r) = 2 - r.
% So the branch point (multiplier 1 on both) is at r = 1, and x* has a
% multiplier of -1 at r = 3, where x* = 2/3.
% A 2-cycle {p, q} solves F(F(x)) = x without F(x) = x; dividing out the
% fixed points leaves r^2 x^2 - r (r + 1) x + (r + 1) = 0, so
% p + q = (r + 1)/r and p q = (r + 1)/r^2. Its multiplier is
% F'(p) F'(q) = r^2 (1 - 2(p + q) + 4 p q) = -r^2 + 2r + 4, which is -1 at
% r^2 - 2r - 5 = 0, r = 1 + sqrt(6).
t = out.Transcritical;
bp = t(1).points("BP");
testCase.verifyNumElements(bp, 1);
testCase.verifyEqual(t(1).Alpha(bp.Index), 1, AbsTol = 1e-6);
r = t(2).Alpha;
testCase.verifyEqual(t(2).U, 1 - 1./r, AbsTol = 1e-9);
testCase.verifyEqual(t(2).Eigenvalues, 2 - r, AbsTol = 1e-8);
testCase.verifyGreaterThan(max(r), 1.9);
d = out.Doubling;
pd = d(1).points("PD");
testCase.verifyNumElements(pd, 1);
testCase.verifyEqual(d(1).Alpha(pd.Index), 3, AbsTol = 1e-6);
testCase.verifyEqual(pd.U, 2/3, AbsTol = 1e-6);
testCase.verifyEqual(d(1).U, 1 - 1./d(1).Alpha, AbsTol = 1e-9);
testCase.verifyEqual(d(2).Iterate, 2);
r2 = d(2).Alpha;
x0 = d(2).U;
x1 = r2.*x0.*(1 - x0);
testCase.verifyEqual(r2.*x1.*(1 - x1), x0, AbsTol = 1e-9);
pd2 = d(2).points("PD");
testCase.verifyNumElements(pd2, 1);
testCase.verifyEqual(d(2).Alpha(pd2.Index), 1 + sqrt(6), AbsTol = 1e-6);
end
```

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "10.3.4",
    "edition": 3,
    "model": "logistic_map.ode",
    "script": "fig_10_3_4",
    "api": ["continuation", "Switch BP1", "Switch PD1", "Branch.plot"],
    "values": {"r0": 0.5, "x0": 0, "bounds": [0.5, 2], "r1": 2.5, "x1": 0.6,
               "bounds1": [2.5, 3.6], "hmax": 0.05},
    "source": {"r0": "chosen", "x0": "chosen", "bounds": "chosen", "r1": "chosen",
               "x1": "chosen", "bounds1": "chosen", "hmax": "chosen"},
    "check": "branch point at r = 1; period doubling at r = 3 and of the 2-cycle at 1 + sqrt(6)",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f10_3_4 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_10_3_4(options)
%FIG_10_3_4 Strogatz, 3rd edition, figure 10.3.4, from logistic_map.ode.
%   OUT = FIG_10_3_4() draws the fixed points and the 2-cycle against r;
%   FIG_10_3_4(Plot = false) draws nothing. OUT.Transcritical holds x = 0
%   and the branch through its branch point, OUT.Doubling the fixed points
%   from r1 and the 2-cycle that leaves their period doubling.
arguments
    options.Plot (1,1) logical = true
end
v = struct("r0", 0.5, "x0", 0, "bounds", [0.5 2], "r1", 2.5, "x1", 0.6, ...
    "bounds1", [2.5 3.6], "hmax", 0.05);
m = strogatzModel("logistic_map.ode");
t = phasecont.continuation(m, v.x0, Params = struct("r", v.r0), Free = "r", ...
    Bounds = v.bounds, Switch = "BP1");
d = phasecont.continuation(m, v.x1, Params = struct("r", v.r1), Free = "r", ...
    Bounds = v.bounds1, Hmax = v.hmax, Switch = "PD1");
out = struct("Values", v, "Model", m, "Transcritical", t, "Doubling", d);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
t.plot(Parent = ax);
d.plot(Parent = ax);
xlabel(ax, "r");
ylabel(ax, "x");
end
```

- [ ] **Step 5: Run row f10_3_4 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_10_3_4.m books/strogatz/models/logistic_map.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 10.3.4 (logistic map branches)"
```

---

### Task 12: Row 12.2.2 (map ensemble, Henon)

**Files:**
- Create: `books/strogatz/figures/fig_12_2_2.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/henon_map.ode`

**Interfaces:**
- Produces: `fig_12_2_2` with `out.Start` (2-by-M start states) and `out.Result` (phasecont.EnsembleResult, `Final` 2-by-M).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_12_2_2(testCase, out)
% One step of the Henon map x' = 1 - a x^2 + y, y' = b x, applied to every
% start point as array arithmetic, is the oracle for the ensemble image.
v = out.Values;
X0 = out.Start;
F = [1 - v.a*X0(1, :).^2 + X0(2, :); v.b*X0(1, :)];
testCase.verifySize(out.Result.Final, size(X0));
testCase.verifyEqual(out.Result.Final, F, AbsTol = 1e-14);
end
```

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "12.2.2",
    "edition": 3,
    "model": "henon_map.ode",
    "script": "fig_12_2_2",
    "api": ["ensemble"],
    "values": {"a": 1.4, "b": 0.3, "box": [-1.5, 1.5, -0.4, 0.4], "n": 100, "steps": 1},
    "source": {"a": "stated", "b": "stated", "box": "chosen", "n": "chosen", "steps": "chosen"},
    "check": "image of the start grid against the map formula, AbsTol 1e-14",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f12_2_2 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_12_2_2(options)
%FIG_12_2_2 Strogatz, 3rd edition, figure 12.2.2, from henon_map.ode.
%   OUT = FIG_12_2_2() draws a grid of start points and its image after
%   Values.steps iterates; FIG_12_2_2(Plot = false) draws nothing.
%   OUT.Start holds the start points and OUT.Result the ensemble result.
arguments
    options.Plot (1,1) logical = true
end
v = struct("a", 1.4, "b", 0.3, "box", [-1.5 1.5 -0.4 0.4], "n", 100, "steps", 1);
m = strogatzModel("henon_map.ode");
[X, Y] = meshgrid(linspace(v.box(1), v.box(2), v.n), linspace(v.box(3), v.box(4), v.n));
X0 = [X(:)'; Y(:)'];
r = phasecont.ensemble(m, X0, [v.a; v.b], [0 v.steps], Precision = "double", Backend = "cpu");
out = struct("Values", v, "Model", m, "Start", X0, "Result", r);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
plot(ax, X0(1, :), X0(2, :), ".", MarkerSize = 2);
plot(ax, r.Final(1, :), r.Final(2, :), ".", MarkerSize = 2);
xlabel(ax, "x");
ylabel(ax, "y");
end
```

The parameter vector `[v.a; v.b]` follows the `par` order in `henon_map.ode` (a, then b).

- [ ] **Step 5: Run row f12_2_2 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_12_2_2.m books/strogatz/models/henon_map.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 12.2.2 (Henon map image)"
```

---

### Task 13: Row 5.2.8 (trace-determinant plot)

**Files:**
- Create: `books/strogatz/figures/fig_5_2_8.m`
- Modify: `books/strogatz/manifest.json`, `tests/StrogatzFigureTest.m`
- Add: `books/strogatz/models/damped_oscillator.ode`

**Interfaces:**
- Produces: `fig_5_2_8` with `out.Branch` (EP at the origin in c).

- [ ] **Step 1: Add the check method**

```matlab
function check_fig_5_2_8(testCase, out)
% x' = y, y' = -k x - c y has J = [0 1; -k -c], so tau = -c and
% Delta = k = 1. The type follows from tau^2 - 4 Delta = c^2 - 4 and the
% sign of tau: c < -2 unstable node, -2 < c < 0 unstable spiral,
% 0 < c < 2 stable spiral, c > 2 stable node. Points within 1e-6 of
% c = -2, 0, 2 sit on a boundary and are not checked for type.
% tdplot must draw the path at (tau, Delta), the parabola Delta = tau^2/4
% and the two axis lines.
br = out.Branch;
tbl = br.classify();
c = tbl.Alpha;
testCase.verifyEqual(tbl.T, -c, AbsTol = 1e-8);
testCase.verifyEqual(tbl.D, ones(size(c)), AbsTol = 1e-8);
type = string(tbl.Type);
region = {c < -2 - 1e-6, abs(c + 1) < 1 - 1e-6, abs(c - 1) < 1 - 1e-6, c > 2 + 1e-6};
names = ["unstable node" "unstable spiral" "stable spiral" "stable node"];
for k = 1:4
    testCase.verifyNotEmpty(type(region{k}), names(k));
    testCase.verifyTrue(all(type(region{k}) == names(k)), names(k));
end
f = figure(Visible = "off");
testCase.addTeardown(@() delete(f));
ax = axes(f);
br.tdplot(Parent = ax);
path = findall(ax, "Tag", "pc-td-path");
testCase.verifyNumElements(path, 1);
testCase.verifyEqual(path.XData(:), tbl.T, AbsTol = 1e-12);
testCase.verifyEqual(path.YData(:), tbl.D, AbsTol = 1e-12);
parabola = findall(ax, "Tag", "pc-td-parabola");
testCase.verifyNumElements(parabola, 1);
testCase.verifyEqual(parabola.YData, parabola.XData.^2/4, AbsTol = 1e-12);
testCase.verifyNumElements(findall(ax, "Tag", "pc-td-axis"), 2);
end
```

- [ ] **Step 2: Append the manifest entry**

```json
  {
    "figure": "5.2.8",
    "edition": 3,
    "model": "damped_oscillator.ode",
    "script": "fig_5_2_8",
    "api": ["continuation", "Branch.classify", "Branch.tdplot"],
    "values": {"k": 1, "c0": -3, "x0": [0, 0], "bounds": [-3, 3], "direction": 1},
    "source": {"k": "chosen", "c0": "chosen", "x0": "chosen", "bounds": "chosen",
               "direction": "chosen"},
    "check": "tau = -c and Delta = 1; types by region of c; tdplot path and parabola tau^2 = 4 Delta",
    "state": "done"
  }
```

- [ ] **Step 3: Run row f5_2_8 to see it fail**

- [ ] **Step 4: Write the script**

```matlab
function out = fig_5_2_8(options)
%FIG_5_2_8 Strogatz, 3rd edition, figure 5.2.8, from damped_oscillator.ode.
%   OUT = FIG_5_2_8() draws the path of the origin in the (tau, Delta)
%   plane as c runs over Values.bounds; FIG_5_2_8(Plot = false) draws
%   nothing. OUT.Branch is the branch of the origin in c.
arguments
    options.Plot (1,1) logical = true
end
v = struct("k", 1, "c0", -3, "x0", [0 0], "bounds", [-3 3], "direction", 1);
m = strogatzModel("damped_oscillator.ode");
br = phasecont.continuation(m, v.x0, Params = struct("k", v.k, "c", v.c0), ...
    Free = "c", Bounds = v.bounds, Direction = v.direction);
out = struct("Values", v, "Model", m, "Branch", br);
if ~options.Plot
    return
end
ax = axes(figure);
br.tdplot(Parent = ax);
end
```

- [ ] **Step 5: Run row f5_2_8 to see it pass**

- [ ] **Step 6: Commit**

Run the four checks, then:
```bash
git add books/strogatz/figures/fig_5_2_8.m books/strogatz/models/damped_oscillator.ode \
  books/strogatz/manifest.json tests/StrogatzFigureTest.m
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Add Strogatz figure 5.2.8 (trace-determinant path)"
```

---

### Task 14: Slice 0 report and STATUS

**Files:**
- Modify: `docs/STATUS.md` (new section "Strogatz slice 0 report")

- [ ] **Step 1: Run the whole class from a clean session and keep the log**

```bash
cd <worktree> && \
scripts/matlab-win "results = runtests('tests/StrogatzFigureTest.m'); disp(table(results)); fprintf('total %.1f s\n', sum([results.Duration]))" 2>&1 | tee <wip>/strogatz-slice0-run-2026-11-26.txt
```
Expected: every test passes or is skipped for a stated reason. Collect the `strogatz-time`, `strogatz-plot-time`, `strogatz-err` and `strogatz-info` lines.

- [ ] **Step 2: Run the full MATLAB suite once**

```bash
scripts/matlab-win "setenv('PATH',[getenv('PATH') ';<texbin>']); buildtool test" 2>&1 | tee <wip>/strogatz-slice0-suite-2026-11-26.txt
```
Expected: the previous totals plus the new class, 0 failed.

- [ ] **Step 3: Write the STATUS section**

Add to `docs/STATUS.md`, under the book figures item:

```markdown
### Strogatz slice 0 report (2026-11-26)

Rows done: N of 12. Moved to gaps.json: <figure: reason>, or none. Blocked: <figure: item>, or none.

| Row | API family | Plot = false (s) | Plot = true (s) | Oracle error |
|---|---|---|---|---|
| 2.2.6 | equilibrium | ... | ... | ... |
| ... (one line per row) |

Whole class on this machine: X s (test runner total). CI time: not measured, waits for the push go.

Chosen values (D2): <figure: name = value, ...> for every row, from manifest.json.

Proposed time budget for the class: <X s on this machine times a factor, with the reason>. Owner signs.
```

The proposed budget is a number for the owner to accept or change; no test checks it until the owner signs.

- [ ] **Step 4: Commit**

Run the four checks, then:
```bash
git add docs/STATUS.md
git -c user.name=Nubayat -c user.email=nubayata@gmail.com commit -m "Report Strogatz slice 0 rows, times and chosen values"
```

- [ ] **Step 5: Stop at the gate**

Present the report to the owner: rows done, rows moved and why, blocked rows, times, chosen values and the proposed budget. Pushing, CI timing and the chapter-by-chapter work wait for the owner.
