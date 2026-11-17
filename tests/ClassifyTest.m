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

        function verifyClass2(testCase, c, type, stable, hyperbolic, counts, borderline)
            testCase.verifyEqual(c.Type, string(type));
            testCase.verifyEqual(c.Stable, stable);
            testCase.verifyEqual(c.Hyperbolic, hyperbolic);
            testCase.verifyEqual([c.NStable c.NUnstable c.NCenter], counts);
            testCase.verifyEqual(c.Borderline, string(borderline));
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
    end
end
