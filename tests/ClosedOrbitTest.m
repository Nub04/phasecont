classdef ClosedOrbitTest < matlab.unittest.TestCase
    % Tests phasecont.closedOrbitChecks (SPEC section 8ag). Every model is
    % written here. The cases come from Strogatz, 3rd ed., Sections 7.2 and
    % 7.3, and from Practice 6 of the Exam 2 guide; each check also has a
    % case that must not pass.

    properties (Constant)
        Box = [-2 2 -2 2]
        Pos = [0.1 5 0.1 5]
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
        function m = model(testCase, rhs, params)
            % A model with x' and y' (and z' for a third entry) given by RHS.
            if nargin < 3
                params = "";
            end
            states = ["x" "y" "z"];
            text = "";
            if params ~= ""
                text = "par " + params + newline;
            end
            for k = 1:numel(rhs)
                text = text + states(k) + "'=" + rhs(k) + newline;
            end
            m = testCase.fromText(text + "done" + newline);
        end

        function m = fromText(testCase, text)
            m = phasecont.Model.fromSource(text, Evaluator = "jit");
            testCase.addTeardown(@() delete(m));
        end

        function verifyNotRun(testCase, c)
            testCase.verifyEqual(c.Verdict, "not run");
            testCase.verifyEqual(c.Samples, 0);
            testCase.verifyTrue(isnan(c.Min) && isnan(c.Max));
        end
    end

    methods (Test)
        % Test 1: gradient systems.
        function practiceSixTwoIsAGradientSystem(testCase)
            m = testCase.model(["2*x*y+cos(y)" "x^2-x*sin(y)+3*y^2"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Gradient = true);
            testCase.verifyEqual(r.Gradient.Verdict, "holds on samples");
            testCase.verifyEqual(r.Gradient.Samples, 441);
            testCase.verifyEqual(r.Samples, 441);
            testCase.verifyLessThanOrEqual( ...
                max(abs([r.Gradient.Min r.Gradient.Max])), 1e-9 * r.Gradient.Scale);
            testCase.verifyNotRun(r.Lyapunov);
            testCase.verifyNotRun(r.Dulac);
            testCase.verifyNotRun(r.Trapping);
        end

        function practiceSixThreeIsAGradientSystem(testCase)
            m = testCase.model(["y*exp(x*y)-2*x" "x*exp(x*y)-4*y^3"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Gradient = true);
            testCase.verifyEqual(r.Gradient.Verdict, "holds on samples");
        end

        function strogatzExample721IsAGradientSystem(testCase)
            m = testCase.model(["sin(y)" "x*cos(y)"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Gradient = true);
            testCase.verifyEqual(r.Gradient.Verdict, "holds on samples");
        end

        function aRotationIsNotAGradientSystem(testCase)
            m = testCase.model(["-y" "x"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [11 11], Gradient = true);
            testCase.verifyEqual(r.Gradient.Verdict, "fails");
            testCase.verifyEqual(r.Gradient.Min, -2);
            testCase.verifyEqual(r.Gradient.Max, -2);
        end

        % Test 2: Lyapunov functions.
        function strogatzExample723OddGrid(testCase)
            m = testCase.model(["-x+4*y" "-x-y^3"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Lyapunov = "x^2 + 4*y^2");
            c = r.Lyapunov;
            testCase.verifyEqual(c.Verdict, "holds on samples");
            testCase.verifyEqual(c.Sign, "negative");
            testCase.verifyEqual(c.FixedPointSamples, 1);
            testCase.verifyEqual(c.Expression, "x^2 + 4*y^2");
            testCase.verifyEqual(c.VMin, 0, AbsTol = 1e-12);
            testCase.verifyEqual(c.VMinAt, [0 0], AbsTol = 1e-12);
            testCase.verifyEqual(c.VMax, 20, RelTol = 1e-12);
            % Vdot = -2x^2 - 8y^4, smallest at the corners.
            testCase.verifyEqual(c.Min, -136, RelTol = 1e-12);
        end

        function strogatzExample723EvenGrid(testCase)
            m = testCase.model(["-x+4*y" "-x-y^3"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [20 20], Lyapunov = "x^2 + 4*y^2");
            testCase.verifyEqual(r.Lyapunov.Verdict, "holds on samples");
            testCase.verifyEqual(r.Lyapunov.FixedPointSamples, 0);
        end

        function practiceSixFourAndFive(testCase)
            m = testCase.model(["2*y-x^3*(1+y^2)" "-5*x-2*y^5"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Lyapunov = "5*x^2 + 2*y^2");
            testCase.verifyEqual(r.Lyapunov.Verdict, "holds on samples");
            testCase.verifyEqual(r.Lyapunov.Sign, "negative");
            m = testCase.model(["y-3*x-x*y^2" "-4*x-2*y-y^3"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Lyapunov = "4*x^2 + y^2");
            testCase.verifyEqual(r.Lyapunov.Verdict, "holds on samples");
            testCase.verifyEqual(r.Lyapunov.Sign, "negative");
        end

        function aWrongLyapunovFunctionFails(testCase)
            m = testCase.model(["-x+4*y" "-x-y^3"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Lyapunov = "x^2 + y^2");
            testCase.verifyEqual(r.Lyapunov.Verdict, "fails");
            testCase.verifyLessThan(r.Lyapunov.Min, 0);
            testCase.verifyGreaterThan(r.Lyapunov.Max, 0);
        end

        function aConservedQuantityIsInconclusive(testCase)
            m = testCase.model(["y" "-x"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Lyapunov = "x^2 + y^2");
            testCase.verifyEqual(r.Lyapunov.Verdict, "inconclusive");
        end

        % Test 3: Dulac's criterion.
        function strogatzExample724(testCase)
            m = testCase.model(["x*(2-x-y)" "y*(4*x-x^2-3)"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Pos, ...
                Grid = [21 21], Dulac = "1/(x*y)");
            c = r.Dulac;
            testCase.verifyEqual(c.Verdict, "holds on samples");
            testCase.verifyEqual(c.Sign, "negative");
            testCase.verifyEqual(c.ZeroSamples, 0);
            testCase.verifyEqual(c.Expression, "1/(x*y)");
            testCase.verifyEqual(c.Min, -10, RelTol = 1e-12);
            testCase.verifyEqual(c.MinAt(2), 0.1, RelTol = 1e-12);
            testCase.verifyEqual(c.Max, -0.2, RelTol = 1e-12);
            testCase.verifyEqual(c.MaxAt(2), 5, RelTol = 1e-12);
        end

        function strogatzExample725(testCase)
            m = testCase.model(["y" "-x-y+x^2+y^2"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Dulac = "exp(-2*x)");
            c = r.Dulac;
            testCase.verifyEqual(c.Verdict, "holds on samples");
            testCase.verifyEqual(c.Min, -exp(4), RelTol = 1e-12);
            testCase.verifyEqual(c.MinAt(1), -2);
            testCase.verifyEqual(c.Max, -exp(-4), RelTol = 1e-12);
            testCase.verifyEqual(c.MaxAt(1), 2);
        end

        function practiceSixSixAndSeven(testCase)
            m = testCase.model(["x*(3-x-2*y)" "y*(2-x-y)"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Pos, ...
                Grid = [21 21], Dulac = "1/(x*y)");
            testCase.verifyEqual(r.Dulac.Verdict, "holds on samples");
            testCase.verifyEqual(r.Dulac.Sign, "negative");
            m = testCase.model(["x*(1-x/2)-2*x*y/(1+x)" "y*(-1+2*x/(1+x))"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Pos, ...
                Grid = [21 21], Dulac = "(1+x)/x");
            testCase.verifyEqual(r.Dulac.Verdict, "holds on samples");
            testCase.verifyEqual(r.Dulac.Sign, "negative");
        end

        function vanDerPolFailsBendixson(testCase)
            m = testCase.model(["y" "mu*(1-x^2)*y-x"], "mu=1");
            r = phasecont.closedOrbitChecks(m, Region = [-3 3 -3 3], ...
                Grid = [21 21], Dulac = "1");
            testCase.verifyEqual(r.Dulac.Verdict, "fails");
        end

        function zeroDivergenceIsInconclusive(testCase)
            m = testCase.model(["y" "-x"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Dulac = "1");
            testCase.verifyEqual(r.Dulac.Verdict, "inconclusive");
            testCase.verifyEqual(r.Dulac.ZeroSamples, 441);
        end

        % Test 4: trapping regions.
        function aBoxAroundTheCycleTraps(testCase)
            m = testCase.model(["x-y-x*(x^2+y^2)" "x+y-y*(x^2+y^2)"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Trapping = true);
            c = r.Trapping;
            testCase.verifyEqual(c.Verdict, "holds on samples");
            testCase.verifyEqual(c.Samples, 80);
            testCase.verifyLessThan(c.Max, 0);
            testCase.verifyEqual(c.TangentSamples, 0);
        end

        function aBoxInsideTheCycleDoesNotTrap(testCase)
            m = testCase.model(["x-y-x*(x^2+y^2)" "x+y-y*(x^2+y^2)"]);
            r = phasecont.closedOrbitChecks(m, Region = [-0.5 0.5 -0.5 0.5], ...
                Grid = [11 11], Trapping = true);
            testCase.verifyEqual(r.Trapping.Verdict, "fails");
        end

        function aSaddleDoesNotTrap(testCase)
            m = testCase.model(["x" "-y"]);
            r = phasecont.closedOrbitChecks(m, Region = [-1 1 -1 1], ...
                Grid = [11 11], Trapping = true);
            c = r.Trapping;
            testCase.verifyEqual(c.Verdict, "fails");
            testCase.verifyTrue(ismember(c.Worst, ["left" "right"]));
            testCase.verifyEqual(c.Max, 1);
        end

        function aCornerIsTestedAgainstBothSides(testCase)
            m = testCase.model(["0.5-x" "0.5-y+2*(x*y)^8"]);
            r = phasecont.closedOrbitChecks(m, Region = [0 1 0 1], ...
                Grid = [3 3], Trapping = true);
            c = r.Trapping;
            testCase.verifyEqual(c.Verdict, "fails");
            testCase.verifyEqual(c.Worst, "top");
            testCase.verifyEqual(c.MaxAt, [1 1]);
            testCase.verifyEqual(c.Max, 1.5, RelTol = 1e-12);
            testCase.verifyEqual(c.Samples, 8);
        end

        % Test 5: refusals and options.
        function refusesModelsItCannotCheck(testCase)
            texts = [
                "par a=1.4, b=0.3" + newline + "x(t+1)=1-a*x^2+y" + newline + ...
                    "y(t+1)=b*x" + newline + "done" + newline
                "x'=-delay(x,1)+y" + newline + "y'=-x" + newline + "done" + newline
                "x'=y+t" + newline + "y'=-x" + newline + "done" + newline
                "x'=y" + newline + "y'=z" + newline + "z'=-x" + newline + "done" + newline
                ];
            for k = 1:numel(texts)
                m = testCase.fromText(texts(k));
                testCase.verifyError(@() phasecont.closedOrbitChecks(m, ...
                    Region = testCase.Box, Grid = [5 5], Gradient = true), ...
                    "phasecont:orbit:badModel", texts(k));
            end
        end

        function refusesBadOptions(testCase)
            m = testCase.model(["y" "-x"]);
            bad = {
                {"Grid", [5 5], "Gradient", true}
                {"Region", testCase.Box, "Gradient", true}
                {"Region", [1 0 0 1], "Grid", [5 5], "Gradient", true}
                {"Region", [0 1 0 Inf], "Grid", [5 5], "Gradient", true}
                {"Region", [0 1 0], "Grid", [5 5], "Gradient", true}
                {"Region", testCase.Box, "Grid", [1 5], "Gradient", true}
                {"Region", testCase.Box, "Grid", [5 2.5], "Gradient", true}
                {"Region", testCase.Box, "Grid", [5 5]}
                {"Region", testCase.Box, "Grid", [5 5], "Gradient", false, "Trapping", false}
                {"Region", testCase.Box, "Grid", [5 5], "Gradient", true, "Tol", 1}
                {"Region", testCase.Box, "Grid", [5 5], "Gradient", true, "Tol", -1}
                {"Region", testCase.Box, "Grid", [5 5], "Dulac", ""}
                {"Region", testCase.Box, "Grid", [5 5], "Lyapunov", 3}
                {"Region", testCase.Box, "Grid", [5 5], "Dulac", "1" + newline + "y'=1"}
                };
            for k = 1:numel(bad)
                args = bad{k};
                testCase.verifyError(@() phasecont.closedOrbitChecks(m, args{:}), ...
                    "phasecont:model:badOption", sprintf("case %d", k));
            end
        end

        function anUnknownSymbolIsABadExpression(testCase)
            m = testCase.model(["y" "-x"]);
            testCase.verifyError(@() phasecont.closedOrbitChecks(m, ...
                Region = testCase.Box, Grid = [5 5], Dulac = "1/(x*q)"), ...
                "phasecont:orbit:badExpression");
            testCase.verifyError(@() phasecont.closedOrbitChecks(m, ...
                Region = testCase.Box, Grid = [5 5], Lyapunov = "x^2+t"), ...
                "phasecont:orbit:badExpression");
        end

        function aSingularDulacFunctionNamesTheSample(testCase)
            m = testCase.model(["y" "-x"]);
            try
                phasecont.closedOrbitChecks(m, Region = [-1 1 -1 1], ...
                    Grid = [3 3], Dulac = "1/x");
                testCase.verifyFail("no error");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:orbit:nonfinite");
                testCase.verifySubstring(string(err.message), "(0, -1)");
            end
        end

        % Test 6: parameters.
        function paramsChangeTheVerdict(testCase)
            m = testCase.model(["y" "mu*(1-x^2)*y-x"], "mu=1");
            r = phasecont.closedOrbitChecks(m, Region = [-3 3 -3 3], ...
                Grid = [21 21], Dulac = "1", Params = struct("mu", -1));
            testCase.verifyEqual(r.Dulac.Verdict, "fails");
            testCase.verifyEqual(r.P, -1);
            r = phasecont.closedOrbitChecks(m, Region = [-3 3 -3 3], ...
                Grid = [21 21], Dulac = "1", Params = struct("mu", 0));
            testCase.verifyEqual(r.Dulac.Verdict, "inconclusive");
            r = phasecont.closedOrbitChecks(m, Region = [-3 3 -3 3], ...
                Grid = [21 21], Dulac = "1", Params = 0);
            testCase.verifyEqual(r.Dulac.Verdict, "inconclusive");
        end

        function aDulacFunctionReadsTheParameters(testCase)
            % g = exp(c*x) with c = -2 is Example 7.2.5's function.
            m = testCase.model(["y" "-x-y+x^2+y^2"], "c=0");
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Dulac = "exp(c*x)", Params = struct("c", -2));
            testCase.verifyEqual(r.Dulac.Verdict, "holds on samples");
            testCase.verifyEqual(r.Dulac.Min, -exp(4), RelTol = 1e-12);
        end

        function allFourChecksInOneCall(testCase)
            m = testCase.model(["x-y-x*(x^2+y^2)" "x+y-y*(x^2+y^2)"]);
            r = phasecont.closedOrbitChecks(m, Region = testCase.Box, ...
                Grid = [21 21], Gradient = true, Lyapunov = "x^2+y^2", ...
                Dulac = "1", Trapping = true);
            testCase.verifyEqual(r.Gradient.Verdict, "fails");
            testCase.verifyEqual(r.Lyapunov.Verdict, "fails");
            testCase.verifyEqual(r.Dulac.Verdict, "fails");
            testCase.verifyEqual(r.Trapping.Verdict, "holds on samples");
            testCase.verifyEqual(r.Region, testCase.Box);
            testCase.verifyEqual(r.Grid, [21 21]);
            testCase.verifyEqual(r.Tol, 1e-9);
        end
    end
end
