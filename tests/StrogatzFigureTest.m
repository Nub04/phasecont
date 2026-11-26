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
                orderfields(Row.values), ...
                "out.Values must equal the values in the manifest", RelTol = 1e-14);
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

        function check_fig_9_3_2(testCase, out)
            % Oracle: ode45 on the same equations with RelTol = AbsTol =
            % 1e-12, up to the first output time at or after t = 1. The
            % horizon is short because the flow is chaotic: nearby orbits
            % separate, so a long comparison would measure the separation,
            % not the solver.
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
                % All four eigenvalue pairs are real. pt.Eigenvalues is complex-typed,
                % and sort orders complex values by magnitude, so sort the real parts.
                ev = eig(cdJacobian(m, pt.U));
                testCase.verifyEqual(imag(pt.Eigenvalues), zeros(size(pt.Eigenvalues)), AbsTol = 1e-9);
                testCase.verifyEqual(sort(real(pt.Eigenvalues)), sort(real(ev)), AbsTol = 1e-6);
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

        function check_fig_3_4_2(testCase, out)
            % f(x, r) = r x - x^3, f_x = r - 3x^2, f_r = x. On x = 0, f_x = r and
            % f_r = 0, so both vanish at (0, 0): a branch point at r = 0. The other
            % branch through it is r = x^2, where f_x = r - 3r = -2r.
            brs = out.Branches;
            bp = brs(1).points("BP");
            testCase.verifyNumElements(bp, 1);
            % Point.Index is the grid row next to the point, so read the refined parameters.
            testCase.verifyEqual(bp.P(brs(1).FreeIndex), 0, AbsTol = 1e-6);
            testCase.verifyLessThan(max(abs(brs(1).U)), 1e-10);
            testCase.verifyEqual(brs(1).Eigenvalues, brs(1).Alpha, AbsTol = 1e-8);
            x = vertcat(brs(2:end).U);
            r = vertcat(brs(2:end).Alpha);
            lambda = vertcat(brs(2:end).Eigenvalues);
            testCase.verifyEqual(x.^2, r, AbsTol = 1e-8);
            testCase.verifyEqual(lambda, -2*r, AbsTol = 1e-8);
            testCase.verifyTrue(any(x < -0.9) && any(x > 0.9), "Both halves of r = x^2");
        end

        function check_fig_3_6_2(testCase, out)
            % f(x) = h + r x - x^3. A fold has f = 0 and f_x = r - 3x^2 = 0, so
            % r = 3x^2 and h = -r x + x^3 = -2x^3. Then 27 h^2 = 108 x^6 = 4 r^3.
            % At r = 1 the folds of the h branch sit at x = -+1/sqrt(3), so
            % h = +-2/(3 sqrt(3)). The cusp has f_xx = -6x = 0 as well: x = 0, so
            % (h, r) = (0, 0).
            % The checks on the special points read the refined points (Point.P).
            b = out.Branch;
            lp = b.points("LP");
            testCase.verifyNumElements(lp, 2);
            testCase.verifyEqual(sort(arrayfun(@(p) p.P(b.FreeIndex), lp(:))), [-1; 1] * 2/(3*sqrt(3)), AbsTol = 1e-6);
            c = out.Curve;
            x = c.U;
            h = c.Alpha(:, 1);
            r = c.Alpha(:, 2);
            testCase.verifyEqual(r, 3*x.^2, AbsTol = 1e-8);
            testCase.verifyEqual(h, -2*x.^3, AbsTol = 1e-8);
            testCase.verifyEqual(27*h.^2, 4*r.^3, AbsTol = 1e-8);
            cp = c.points("CP");
            testCase.verifyNumElements(cp, 1);
            testCase.verifyEqual(cp.P(c.FreeIndex).', [0 0], AbsTol = 1e-6);
            testCase.verifyTrue(any(x < 0) && any(x > 0), "The curve passes the cusp");
        end

        function check_fig_8_3_4(testCase, out)
            % f = a - x - 4xy/u, g = b x (1 - y/u), u = 1 + x^2. g = 0 with x > 0
            % gives y = u; then f = a - 5x = 0, so x* = a/5 and y* = 1 + a^2/25.
            % At the equilibrium:
            %   f_x = -1 - 4y(1 - x^2)/u^2 = (3x^2 - 5)/u,   f_y = -4x/u,
            %   g_x = b - b y (1 - x^2)/u^2 = 2b x^2/u,      g_y = -b x/u,
            % so tau = (3x^2 - 5 - b x)/u and Delta = 5 b x/u > 0. Hopf has tau = 0:
            % b = 3x - 5/x = 3a/5 - 25/a. At a = 10: x = 2, u = 5, b = 3.5 and
            % omega^2 = Delta = 7.
            % The checks on the special points read the refined points (Point.P).
            b = out.Branch;
            hp = b.points("H");
            testCase.verifyNumElements(hp, 1);
            testCase.verifyEqual(hp.P(b.FreeIndex), 3.5, AbsTol = 1e-6);
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
            % The check reads the refined point (Point.P), since Point.Index is the
            % neighbouring grid row.
            testCase.verifyEqual(hp.P(b.FreeIndex), rH, AbsTol = 1e-6);
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
            % The check reads the refined point (Point.P), since Point.Index is the
            % neighbouring grid row.
            testCase.verifyEqual(hp.P(b.FreeIndex), -1, AbsTol = 1e-6);
            testCase.verifyEqual(hp.Data.omega, 1, AbsTol = 1e-6);
            hom = out.Homoclinic;
            testCase.verifyEqual(hom.Alpha(1, 2), v.c, AbsTol = 1e-8);
            fprintf("strogatz-info 8.4.3 HOM points %d, last c %.6g mu %.6g, LC index %d\n", ...
                size(hom.Alpha, 1), hom.Alpha(end, 2), hom.Alpha(end, 1), out.Index);
            % The bracket in Values holds a sign change at c = 1. At the last HOM
            % point (c = 1.1) it does not and fzero errors, so only the first point
            % is checked.
            for k = 1
                c = hom.Alpha(k, 2);
                muc = fzero(@(mu) splitDistance(mu, c), v.bracket);
                fprintf("strogatz-err 8.4.3 c %.6g mu %.10g oracle %.10g\n", c, hom.Alpha(k, 1), muc);
                testCase.verifyEqual(hom.Alpha(k, 1), muc, AbsTol = 1e-4);
            end
        end

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
            % Point.Index is the grid row next to the point, so read the refined parameters.
            testCase.verifyEqual(bp.P(t(1).FreeIndex), 1, AbsTol = 1e-6);
            r = t(2).Alpha;
            testCase.verifyEqual(t(2).U, 1 - 1./r, AbsTol = 1e-9);
            testCase.verifyEqual(t(2).Eigenvalues, 2 - r, AbsTol = 1e-8);
            testCase.verifyGreaterThan(max(r), 1.9);
            d = out.Doubling;
            pd = d(1).points("PD");
            testCase.verifyNumElements(pd, 1);
            testCase.verifyEqual(pd.P(d(1).FreeIndex), 3, AbsTol = 1e-6);
            testCase.verifyEqual(pd.U, 2/3, AbsTol = 1e-6);
            testCase.verifyEqual(d(1).U, 1 - 1./d(1).Alpha, AbsTol = 1e-9);
            testCase.verifyEqual(d(2).Iterate, 2);
            r2 = d(2).Alpha;
            x0 = d(2).U;
            x1 = r2.*x0.*(1 - x0);
            testCase.verifyEqual(r2.*x1.*(1 - x1), x0, AbsTol = 1e-9);
            pd2 = d(2).points("PD");
            testCase.verifyNumElements(pd2, 1);
            testCase.verifyEqual(pd2.P(d(2).FreeIndex), 1 + sqrt(6), AbsTol = 1e-6);
        end

        function check_fig_12_2_2(testCase, out)
            % One step of the Henon map x' = 1 - a x^2 + y, y' = b x, applied to every
            % start point as array arithmetic, is the oracle for the ensemble image.
            v = out.Values;
            X0 = out.Start;
            F = [1 - v.a*X0(1, :).^2 + X0(2, :); v.b*X0(1, :)];
            testCase.verifySize(out.Result.Final, size(X0));
            testCase.verifyEqual(out.Result.Final, F, AbsTol = 1e-14);
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
