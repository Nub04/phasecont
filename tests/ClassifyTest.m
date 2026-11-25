classdef ClassifyTest < matlab.unittest.TestCase
    % Tests Point.classify, Point.Jacobian and Point.Type (SPEC section 8ac,
    % test 3). Every model is written here. The linear models have their
    % equilibrium at the origin, so the Points are built there directly;
    % the map points come from phasecont.equilibrium.

    properties (Constant)
        Hutchinson = "par r=1.3, tau=1" + newline + "init x=0.5" + newline + ...
            "x'=r*x*(1-delay(x,tau))" + newline + "done" + newline
        Timed = "par r=2" + newline + "init x=0.5" + newline + ...
            "x'=r*x*(1-x)+t" + newline + "done" + newline
        Logistic = "par r=2.5" + newline + "init x=0.2" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
        end
    end

    methods (Access = private)
        function m = model(testCase, text)
            m = phasecont.Model.fromSource(text, Evaluator = "jit");
            testCase.addTeardown(@() delete(m));
        end

        function m = linear(testCase, rhs, params)
            % A model with x', y', z', w' given by RHS, one line per state.
            if nargin < 3
                params = "";
            end
            states = ["x" "y" "z" "w"];
            text = "";
            if params ~= ""
                text = "par " + params + newline;
            end
            for k = 1:numel(rhs)
                text = text + states(k) + "'=" + rhs(k) + newline;
            end
            m = testCase.model(text + "done" + newline);
        end

        function pt = origin(~, m)
            n = numel(m.Variables);
            pt = phasecont.Point(U = zeros(n, 1), P = m.ParameterValues, ...
                Label = "EP", Model = m);
        end

        function verifyTdExtents(testCase, ax, tbl, isMap, at)
            % The T and D ranges of the reference curves as SPEC section 8ad
            % states them, from the path, the special points, 0 and the
            % triangle.
            sp = findobj(ax, Tag = "pc-special");
            px = zeros(0, 1);
            py = zeros(0, 1);
            for k = 1:numel(sp)
                px = [px; sp(k).XData(:)]; %#ok<AGROW>
                py = [py; sp(k).YData(:)]; %#ok<AGROW>
            end
            tr = [tbl.T; px; 0];
            dr = [tbl.D; py; 0];
            if isMap
                tr = [tr; -2; 2];
                dr = [dr; -1; 1];
            end
            w = max(tr) - min(tr);
            pad = 0.1 * max(w, 1);
            tlo = min(tr) - pad;
            thi = max(tr) + pad;
            tt = linspace(tlo, thi, 201);
            dr = [dr; tt(:).^2/4];
            dpad = 0.1 * max(max(dr) - min(dr), 1);
            dlo = min(dr) - dpad;
            dhi = max(dr) + dpad;
            axs = findobj(ax, Tag = "pc-td-axis");
            testCase.assertSize(axs, [2 1]);
            horiz = axs(arrayfun(@(a) all(a.YData == 0), axs));
            vert = axs(arrayfun(@(a) all(a.XData == 0), axs));
            testCase.assertSize(horiz, [1 1], at);
            testCase.assertSize(vert, [1 1], at);
            testCase.verifyEqual(horiz.XData, [tlo thi], "AbsTol", 1e-12, at);
            testCase.verifyEqual(vert.YData, [dlo dhi], "AbsTol", 1e-12, at);
            par = findobj(ax, Tag = "pc-td-parabola");
            testCase.verifyEqual(par.XData([1 end]), [tlo thi], "AbsTol", 1e-12, at);
            testCase.verifyEqual(par.YData, par.XData.^2/4, "AbsTol", 1e-12, at);
            testCase.verifyTrue(tlo < 0 && thi > 0, at);
        end

        function verifyClass2(testCase, c, type, stable, hyperbolic, counts, borderline)
            testCase.verifyEqual(c.Type, string(type));
            testCase.verifyEqual(c.Stable, stable);
            testCase.verifyEqual(c.Hyperbolic, hyperbolic);
            testCase.verifyEqual([c.NStable c.NUnstable c.NCenter], counts);
            testCase.verifyEqual(c.Borderline, string(borderline));
        end

        function br = henonBranch(testCase)
            % The Henon branch of fixed points in a from a = 0.2 at b = 0.3.
            b = 0.3;
            m = testCase.model(testCase.Henon);
            x = (-(1 - b) + sqrt((1 - b)^2 + 0.8)) / 0.4;
            eq = phasecont.equilibrium(m, [x b*x] + 0.01, ...
                Params = struct(a = 0.2, b = b));
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [-1 0.5]);
        end
    end

    methods (Test)
        function saddle(testCase)
            m = testCase.linear(["x" "-y"]);
            pt = testCase.origin(m);
            c = classify(pt);
            % T = 0 here, so zeroT fires even though D < 0 decides first.
            testCase.verifyClass2(c, "saddle", false, true, [1 1 0], "zeroT");
            testCase.verifyEqual([c.T c.D c.Disc], [0 -1 4]);
            testCase.verifyEmpty(c.Rotation);
            testCase.verifyEqual(c.Jacobian, [1 0; 0 -1]);
            testCase.verifyEqual(pt.Jacobian, [1 0; 0 -1]);
            testCase.verifyEqual(pt.Type, "saddle");
            testCase.verifyEqual(c.Tol, 1e-9);
        end

        function spiralsAndRotation(testCase)
            cases = {
                "-y+a*x", "x+a*y", "-0.1", "stable spiral", true, [2 0 0], "counterclockwise"
                "-y+a*x", "x+a*y", "0.1", "unstable spiral", false, [0 2 0], "counterclockwise"
                "y+a*x", "-x+a*y", "-0.1", "stable spiral", true, [2 0 0], "clockwise"
                "y+a*x", "-x+a*y", "0.1", "unstable spiral", false, [0 2 0], "clockwise"};
            for k = 1:size(cases, 1)
                m = testCase.linear([cases{k, 1} cases{k, 2}], "a=" + cases{k, 3});
                c = classify(testCase.origin(m));
                testCase.verifyClass2(c, cases{k, 4}, cases{k, 5}, true, ...
                    cases{k, 6}, strings(1, 0));
                testCase.verifyEqual(c.Rotation, cases{k, 7});
            end
        end

        function center(testCase)
            m = testCase.linear(["-y" "x"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "center", false(0, 1), false, [0 0 2], "zeroT");
            testCase.verifyEqual(c.Rotation, "counterclockwise");
        end

        function perturbedCenterIsInTheBand(testCase)
            % T = 1e-12 is inside Tol s = 1e-9 sqrt(2), and outside Tol 0.
            m = testCase.linear(["-y+q*x" "x"], "q=1e-12");
            pt = testCase.origin(m);
            c = classify(pt);
            testCase.verifyClass2(c, "center", false(0, 1), false, [0 0 2], "zeroT");
            c = classify(pt, Tol = 0);
            testCase.verifyClass2(c, "unstable spiral", false, true, [0 2 0], strings(1, 0));
            testCase.verifyEqual(c.Tol, 0);
        end

        function starAndDegenerateNode(testCase)
            m = testCase.linear(["-x" "-y"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "stable star", true, true, [2 0 0], "zeroDisc");
            m = testCase.linear(["-x+y" "-y"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "stable degenerate node", true, true, [2 0 0], "zeroDisc");
        end

        function zeroEigenvalue(testCase)
            m = testCase.linear(["x" "0"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "zero eigenvalue", false(0, 1), false, [0 1 1], "zeroD");
            m = testCase.linear(["y" "0"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "zero eigenvalue", false(0, 1), false, [0 0 2], ...
                ["zeroD" "zeroT" "zeroDisc"]);
        end

        function nodes(testCase)
            m = testCase.linear(["-x" "-2*y"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "stable node", true, true, [2 0 0], strings(1, 0));
            m = testCase.linear(["x" "2*y"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "unstable node", false, true, [0 2 0], strings(1, 0));
        end

        function oneDimension(testCase)
            m = testCase.linear("-x");
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "stable", true, true, [1 0 0], strings(1, 0));
            testCase.verifyTrue(isnan(c.T) && isnan(c.D) && isnan(c.Disc));
            m = testCase.linear("x");
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "unstable", false, true, [0 1 0], strings(1, 0));
            m = testCase.linear("0*x");
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "zero eigenvalue", false(0, 1), false, [0 0 1], strings(1, 0));
        end

        function threeDimensions(testCase)
            % Eigenvalues -0.1 +- i and 1.
            m = testCase.linear(["-0.1*x-y" "x-0.1*y" "z"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "saddle-focus (2 stable, 1 unstable)", ...
                false, true, [2 1 0], strings(1, 0));
            m = testCase.linear(["-0.1*x-y" "x-0.1*y" "-z"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "stable focus-node (3 stable, 0 unstable)", ...
                true, true, [3 0 0], strings(1, 0));
            m = testCase.linear(["-x" "-2*y" "3*z"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "saddle (2 stable, 1 unstable)", ...
                false, true, [2 1 0], strings(1, 0));
            m = testCase.linear(["x" "2*y" "3*z"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "unstable node (0 stable, 3 unstable)", ...
                false, true, [0 3 0], strings(1, 0));
            m = testCase.linear(["-y" "x" "-z"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, ...
                "non-hyperbolic (1 stable, 0 unstable, 2 on the imaginary axis)", ...
                false(0, 1), false, [1 0 2], "zeroRe");
            testCase.verifyEmpty(c.Rotation);
        end

        function fourDimensions(testCase)
            m = testCase.linear(["-x" "-2*y" "3*z" "-4*w"]);
            c = classify(testCase.origin(m));
            testCase.verifyClass2(c, "hyperbolic (3 stable, 1 unstable)", ...
                false, true, [3 1 0], strings(1, 0));
        end

        function logisticMap(testCase)
            m = testCase.model(testCase.Logistic);
            pt = phasecont.equilibrium(m, 0.5);
            c = classify(pt);
            % x = 1 - 1/r with multiplier 2 - r = -0.5.
            testCase.verifyEqual(c.Jacobian, -0.5, AbsTol = 1e-12);
            testCase.verifyClass2(c, "stable (1 inside the unit circle)", ...
                true, true, [1 0 0], strings(1, 0));
            testCase.verifyEmpty(c.Rotation);
            pt = phasecont.equilibrium(m, 0.6, Params = struct(r = 3));
            c = classify(pt);
            testCase.verifyClass2(c, ...
                "non-hyperbolic (0 inside, 0 outside, 1 on the unit circle)", ...
                false(0, 1), false, [0 0 1], "unitCircle");
        end

        function henonMap(testCase)
            m = testCase.model(testCase.Henon);
            a = 1.4;
            b = 0.3;
            x = (-(1 - b) + sqrt((1 - b)^2 + 4*a))/(2*a);
            pt = phasecont.equilibrium(m, [0.6 0.2]);
            testCase.verifyEqual(pt.U, [x; b*x], AbsTol = 1e-10);
            c = classify(pt);
            % Multipliers -a x +- sqrt(a^2 x^2 + b): one inside, one outside.
            testCase.verifyClass2(c, "unstable (1 inside, 1 outside the unit circle)", ...
                false, true, [1 1 0], strings(1, 0));
            testCase.verifyEqual(c.T, -2*a*x, AbsTol = 1e-12);
            testCase.verifyEqual(c.D, -b, AbsTol = 1e-12);
            testCase.verifyEqual(c.Disc, c.T^2 - 4*c.D, AbsTol = 1e-12);
            testCase.verifyEmpty(c.Rotation);
        end

        function logisticPeriodTwo(testCase)
            % At r = 3.2 the 2-cycle has multiplier -r^2 + 2 r + 4 = 0.16.
            m = testCase.model(testCase.Logistic);
            r = 3.2;
            pt = phasecont.equilibrium(m, 0.8, Params = struct(r = r), Iterate = 2);
            testCase.verifyEqual(pt.Iterate, 2);
            x1 = ((r + 1) + sqrt((r + 1)*(r - 3)))/(2*r);
            testCase.verifyEqual(pt.U, x1, AbsTol = 1e-10);
            c = classify(pt);
            testCase.verifyEqual(c.Jacobian, -r^2 + 2*r + 4, AbsTol = 1e-10);
            testCase.verifyClass2(c, "stable (1 inside the unit circle)", ...
                true, true, [1 0 0], strings(1, 0));
        end

        function fallbackEvaluatorAgrees(testCase)
            % model_jacobian runs on every evaluator, so a fallback model
            % gives the same J and Type as the JIT one.
            text = "par k=0.1" + newline + "x'=y" + newline + ...
                "y'=-sin(x)-k*y" + newline + "done" + newline;
            j = testCase.model(text);
            f = phasecont.Model.fromSource(text, Evaluator = "fallback");
            testCase.addTeardown(@() delete(f));
            u = [pi; 0];
            cj = classify(phasecont.Point(U = u, P = j.ParameterValues, Label = "EP", Model = j));
            cf = classify(phasecont.Point(U = u, P = f.ParameterValues, Label = "EP", Model = f));
            testCase.verifyEqual(cf.Jacobian, cj.Jacobian, RelTol = 1e-15, AbsTol = 1e-15);
            testCase.verifyEqual(cf.Type, cj.Type);
            testCase.verifyEqual(cf.Type, "saddle");
        end

        function refusedPointsAreEmpty(testCase)
            m = testCase.linear(["x" "-y"]);
            dm = testCase.model(testCase.Hutchinson);
            tm = testCase.model(testCase.Timed);
            gone = testCase.linear(["x" "-y"]);
            goneP = phasecont.Point(U = [0; 0], Label = "EP", Model = gone);
            delete(gone);
            pts = {
                phasecont.Point(U = [0; 0], Label = "LC", Model = m)
                phasecont.Point(U = [0; 0], Label = "EP")
                phasecont.Point(U = 1, P = dm.ParameterValues, Label = "EP", Model = dm)
                phasecont.Point(U = 0, P = tm.ParameterValues, Label = "EP", Model = tm)
                phasecont.Point(U = [0; 0; 0], Label = "EP", Model = m)
                goneP};
            for k = 1:numel(pts)
                pt = pts{k};
                testCase.verifyEmpty(pt.Jacobian, sprintf("case %d", k));
                testCase.verifyEmpty(pt.Type, sprintf("case %d", k));
                testCase.verifyError(@() classify(pt), "phasecont:point:noJacobian", ...
                    sprintf("case %d", k));
            end
        end

        function badTolIsRefused(testCase)
            pt = testCase.origin(testCase.linear(["x" "-y"]));
            for tol = [-1e-9 1 NaN Inf]
                testCase.verifyError(@() classify(pt, Tol = tol), ...
                    "phasecont:point:badOption", sprintf("Tol %g", tol));
            end
        end

        % Branch.classify and Branch.tdplot (SPEC section 8ad, tests 1 to 4).

        function spiralBranch(testCase)
            m = testCase.model("par r=-1" + newline + ...
                "x'=r*x-y" + newline + "y'=x+r*y" + newline + "done" + newline);
            br = phasecont.continuation(m, [0 0], Free = "r", Direction = 1, ...
                Bounds = [-1 1], Hmax = 0.05, MaxSteps = 200);
            [tbl, J] = classify(br);
            r = br.Alpha;
            npts = numel(r);
            testCase.assertGreaterThan(npts, 10);
            % The branch covers r from -1 to 1.
            testCase.verifyEqual([min(r) max(r)], [-1 1], AbsTol = 1e-9);
            testCase.verifySize(J, [2 2 npts]);
            testCase.verifyEqual(tbl.Alpha, r);
            testCase.verifyEqual(tbl.T, 2*r, AbsTol = 1e-12);
            testCase.verifyEqual(tbl.D, r.^2 + 1, AbsTol = 1e-12);
            off = ~contains(tbl.Borderline, "zeroT");
            testCase.verifyTrue(all(tbl.Type(off & r < 0) == "stable spiral"));
            testCase.verifyTrue(all(tbl.Type(off & r > 0) == "unstable spiral"));
            testCase.verifyTrue(any(r < 0) && any(r > 0));
            testCase.verifyTrue(all(tbl.Rotation == "counterclockwise"));
            for i = 1:npts
                p = br.P;
                p(br.FreeIndex) = r(i);
                pt = phasecont.Point(U = br.U(i, :).', P = p, Label = "EP", ...
                    Model = m);
                c = classify(pt);
                testCase.verifyEqual(tbl.Type(i), c.Type, sprintf("row %d", i));
                testCase.verifyEqual(J(:, :, i), c.Jacobian, sprintf("row %d", i));
                testCase.verifyEqual(tbl.Borderline(i), strjoin(c.Borderline, " "), ...
                    sprintf("row %d", i));
                testCase.verifyEqual([tbl.NStable(i) tbl.NUnstable(i) tbl.NCenter(i)], ...
                    [c.NStable c.NUnstable c.NCenter], sprintf("row %d", i));
            end
        end

        function saddleNodeBranch(testCase)
            m = testCase.model("par r=1" + newline + ...
                "x'=r-x^2" + newline + "y'=-y" + newline + "done" + newline);
            br = phasecont.continuation(m, [1 0], Free = "r", Direction = -1, ...
                Bounds = [-1 1.5], Hmax = 0.05, MaxSteps = 300);
            tbl = classify(br);
            clear0 = ~contains(tbl.Borderline, "zeroD") & ...
                ~contains(tbl.Borderline, "zeroDisc");
            nodes = clear0 & tbl.D > 0;
            saddles = clear0 & tbl.D < 0;
            testCase.verifyTrue(any(nodes) && any(saddles));
            testCase.verifyTrue(all(tbl.Type(nodes) == "stable node"));
            testCase.verifyTrue(all(tbl.Type(saddles) == "saddle"));
            testCase.verifyTrue(all(tbl.Rotation == ""));
        end

        function logisticBranch(testCase)
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0.6);
            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Bounds = [1.5 3.5], Hmax = 0.05, MaxSteps = 200);
            tbl = classify(br);
            below = tbl.Alpha < 2.99;
            above = tbl.Alpha > 3.01;
            testCase.verifyTrue(any(below) && any(above));
            testCase.verifyTrue(all(tbl.Type(below) == "stable (1 inside the unit circle)"));
            testCase.verifyTrue(all(tbl.Type(above) == ...
                "unstable (0 inside, 1 outside the unit circle)"));
            testCase.verifyTrue(all(isnan(tbl.T)));
        end

        function henonBranchTraceAndDeterminant(testCase)
            br = testCase.henonBranch();
            tbl = classify(br);
            ev = br.Eigenvalues;
            testCase.verifyEqual(tbl.T, real(sum(ev, 2)), RelTol = 1e-12, AbsTol = 1e-12);
            testCase.verifyEqual(tbl.D, real(prod(ev, 2)), RelTol = 1e-12, AbsTol = 1e-12);
        end

        function branchesWithNoJacobian(testCase)
            % A cycle branch.
            m = testCase.model("par a=1,b=1.2" + newline + ...
                "x'=a-(b+1)*x+x^2*y" + newline + "y'=b*x-x^2*y" + newline + ...
                "done" + newline);
            eq = phasecont.equilibrium(m, [1 1.2]);
            ep = phasecont.continuation(m, eq, Free = "b", Bounds = [0.5 4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 10);
            testCase.verifyError(@() classify(lc), "phasecont:branch:noJacobian");
            % The same EP branch after its model is deleted.
            testCase.verifyClass(classify(ep), "table");
            delete(m);
            testCase.verifyError(@() classify(ep), "phasecont:branch:noJacobian");
            % A delay model.
            md = testCase.model(testCase.Hutchinson);
            bd = phasecont.continuation(md, 1, Free = "r", ...
                Roots = struct(N = 30, ReMin = -0.5), Bounds = [1.2 2.0], ...
                Direction = 1, Hmax = 0.05);
            testCase.verifyEqual(bd.Type, "EP");
            testCase.verifyError(@() classify(bd), "phasecont:branch:noJacobian");
        end

        function branchBadTolIsRefused(testCase)
            br = testCase.henonBranch();
            for tol = [-1e-9 1 NaN Inf]
                testCase.verifyError(@() classify(br, Tol = tol), ...
                    "phasecont:branch:badOption", sprintf("Tol %g", tol));
            end
        end

        function tdplotOfTheSpiralBranch(testCase)
            m = testCase.model("par r=-1" + newline + ...
                "x'=r*x-y" + newline + "y'=x+r*y" + newline + "done" + newline);
            br = phasecont.continuation(m, [0 0], Free = "r", Direction = 1, ...
                Bounds = [-1 1], Hmax = 0.05, MaxSteps = 200);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            h = tdplot(br, Parent = ax);
            tbl = classify(br);
            path = findobj(ax, Tag = "pc-td-path");
            testCase.verifySize(path, [1 1]);
            testCase.verifyEqual([path.XData(:) path.YData(:)], [tbl.T tbl.D]);
            testCase.verifySize(findobj(ax, Tag = "pc-td-parabola"), [1 1]);
            testCase.verifySize(findobj(ax, Tag = "pc-td-axis"), [2 1]);
            testCase.verifyEmpty(findobj(ax, Tag = "pc-td-triangle"));
            testCase.verifyEqual(string(ax.XLabel.String), "T (trace)");
            testCase.verifyEqual(string(ax.YLabel.String), "D (determinant)");
            testCase.verifyTrue(all(isgraphics(h)));
            % The Hopf point at r = 0 sits at (0, 1).
            sp = findobj(ax, Tag = "pc-special");
            testCase.verifyNotEmpty(sp);
            xy = [vertcat(sp.XData) vertcat(sp.YData)];
            testCase.verifyLessThan(min(vecnorm(xy - [0 1], 2, 2)), 1e-6);
        end

        function tdplotOfTheHenonBranch(testCase)
            br = testCase.henonBranch();
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            tdplot(br, Parent = ax);
            tri = findobj(ax, Tag = "pc-td-triangle");
            testCase.verifySize(tri, [1 1]);
            corners = unique([tri.XData(:) tri.YData(:)], "rows");
            testCase.verifyEqual(corners, [-2 1; 0 -1; 2 1]);
        end

        function tdplotRefusesThreeStates(testCase)
            m = testCase.model("par r=-1" + newline + "x'=r*x" + newline + ...
                "y'=-y" + newline + "z'=-z" + newline + "done" + newline);
            br = phasecont.continuation(m, [0 0 0], Free = "r", Direction = 1, ...
                Bounds = [-1 -0.5], Hmax = 0.05, MaxSteps = 20);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            testCase.verifyError(@() tdplot(br, Parent = axes(f)), ...
                "phasecont:branch:badOption");
        end

        % Coverage added 2026-11-24 (SPEC section 8ad).

        function saddleNodeRowsEqualPointClassify(testCase)
            m = testCase.model("par r=1" + newline + ...
                "x'=r-x^2" + newline + "y'=-y" + newline + "done" + newline);
            br = phasecont.continuation(m, [1 0], Free = "r", Direction = -1, ...
                Bounds = [-1 1.5], Hmax = 0.05, MaxSteps = 300);
            tbl = classify(br);
            for i = 1:height(tbl)
                p = br.P;
                p(br.FreeIndex) = br.Alpha(i);
                pt = phasecont.Point(U = br.U(i, :).', P = p, Label = "EP", ...
                    Model = m);
                c = classify(pt);
                at = sprintf("row %d", i);
                testCase.verifyEqual(tbl.Type(i), c.Type, at);
                testCase.verifyEqual(tbl.Borderline(i), strjoin(c.Borderline, " "), at);
                testCase.verifyEqual([tbl.T(i) tbl.D(i) tbl.Disc(i)], ...
                    [c.T c.D c.Disc], at);
                testCase.verifyEqual(tbl.Hyperbolic(i), c.Hyperbolic, at);
            end
        end

        function nonDoubleTolFailsTheArgumentsBlock(testCase)
            m = testCase.model("par r=-1" + newline + ...
                "x'=r*x-y" + newline + "y'=x+r*y" + newline + "done" + newline);
            br = phasecont.continuation(m, [0 0], Free = "r", Direction = 1, ...
                Bounds = [-1 -0.5], Hmax = 0.05, MaxSteps = 20);
            pt = testCase.origin(m);
            bad = {[1e-9 1e-9], 1e-9 + 1i};
            for k = 1:numel(bad)
                for target = {pt, br}
                    id = "";
                    try
                        classify(target{1}, Tol = bad{k});
                    catch e
                        id = string(e.identifier);
                    end
                    testCase.verifyTrue(startsWith(id, "MATLAB:"), ...
                        sprintf("case %d, %s: %s", k, class(target{1}), id));
                end
            end
        end

        function tdplotReferenceCurvesSpanThePath(testCase)
            m = testCase.model("par r=-1" + newline + ...
                "x'=r*x-y" + newline + "y'=x+r*y" + newline + "done" + newline);
            % A T width below 1 (pad 0.1) and one above 1 (pad 10 percent).
            for b = [0.2 1]
                br = phasecont.continuation(m, [0 0], Free = "r", Direction = 1, ...
                    Bounds = [-b b], Hmax = 0.05, MaxSteps = 200);
                f = figure(Visible = "off");
                testCase.addTeardown(@() close(f));
                ax = axes(f);
                tdplot(br, Parent = ax);
                testCase.verifyTdExtents(ax, classify(br), false, sprintf("b %g", b));
            end
        end

        function tdplotReferenceCurvesSpanTheTriangle(testCase)
            br = testCase.henonBranch();
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            tdplot(br, Parent = ax);
            testCase.verifyTdExtents(ax, classify(br), true, "henon");
        end
    end
end
