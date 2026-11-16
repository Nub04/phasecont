classdef CobwebTest < matlab.unittest.TestCase
    % Tests phasecont.cobweb (SPEC section 8z, test 8). Each test writes its
    % map and computes the iterates with a loop of its own. The staircase of
    % iterates x_0, ..., x_m has the 2m + 1 vertices (x_floor(i/2),
    % x_ceil(i/2)), i = 0, ..., 2m.

    properties (Constant)
        % x -> r x (1 - x), with the fixed points 0 and 1 - 1/r.
        Logistic = "par r=2.5" + newline + "init x=0.1" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        % x -> x^2. From 2 the iterates are 2^(2^k), and 2^1024 overflows.
        Square = "init x=2" + newline + "x(t+1)=x^2" + newline + ...
            "done" + newline
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        Shift = "init x=1" + newline + "x(t+1)=x/2+t" + newline + ...
            "done" + newline
        Decay = "par a=2" + newline + "init x=1" + newline + ...
            "x'=-a*x" + newline + "done" + newline
    end

    properties (TestParameter)
        Evaluator = {"jit", "fallback"}
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
        function m = model(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end

        function ax = newAxes(testCase)
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
        end
    end

    methods (Static, Access = private)
        function x = logistic(r, x0, N)
            x = zeros(N + 1, 1);
            x(1) = x0;
            for k = 1:N
                x(k + 1) = r * x(k) * (1 - x(k));
            end
        end

        function [X, Y] = staircase(x)
            m = numel(x) - 1;
            X = zeros(2 * m + 1, 1);
            Y = zeros(2 * m + 1, 1);
            for i = 0:2 * m
                X(i + 1) = x(floor(i / 2) + 1);
                Y(i + 1) = x(ceil(i / 2) + 1);
            end
        end

        function g = rangeOf(x)
            lo = min(x);
            hi = max(x);
            w = hi - lo;
            if w == 0
                w = max(1, abs(lo));
            end
            g = [lo - w / 10, hi + w / 10];
        end
    end

    methods (Test)
        function stepsGraphAndStartOfTheLogisticMap(testCase, Evaluator)
            m = testCase.model(testCase.Logistic, Evaluator);
            ax = testCase.newAxes();
            [h, tr] = phasecont.cobweb(m, 0.1, 10, Parent = ax);

            x = CobwebTest.logistic(2.5, 0.1, 10);
            [X, Y] = CobwebTest.staircase(x);
            testCase.verifyEqual(tr.X, x);
            ref = phasecont.simulate(m, 0.1, [0 10]);
            testCase.verifyEqual(tr.T, ref.T);
            testCase.verifyEqual(tr.X, ref.X);

            testCase.verifyEqual(string(h.Steps.Tag), "pc-cobweb");
            testCase.verifyEqual(h.Steps.XData(:), X);
            testCase.verifyEqual(h.Steps.YData(:), Y);
            testCase.verifyNumElements(h.Steps.XData, 21);
            testCase.verifyEqual([h.Steps.XData(end) h.Steps.YData(end)], [x(end) x(end)]);

            testCase.verifyEqual(string(h.Start.Tag), "pc-start");
            testCase.verifyEqual([h.Start.XData h.Start.YData], [0.1 0.1]);
            testCase.verifyEqual(h.Start.Color, h.Steps.Color);

            g = CobwebTest.rangeOf(x);
            testCase.verifyEqual(string(h.Graph.Tag), "pc-cobweb-graph");
            testCase.verifyEqual(h.Graph.XData, linspace(g(1), g(2), 401));
            xs = h.Graph.XData;
            testCase.verifyEqual(h.Graph.YData, 2.5 * xs .* (1 - xs), AbsTol = 1e-14);

            testCase.verifyEqual(string(h.Diagonal.Tag), "pc-cobweb-diagonal");
            testCase.verifyEqual(h.Diagonal.XData, g);
            testCase.verifyEqual(h.Diagonal.YData, g);

            testCase.verifyEqual(string(ax.XLabel.String), "x");
            testCase.verifyEqual(string(ax.YLabel.String), "x(n+1)");
        end

        function aFixedPointStartGivesTheUnitWidth(testCase)
            % F(0) = 0 exactly, so every iterate is 0 and the range is
            % 0 -+ max(1, 0)/10.
            m = testCase.model(testCase.Logistic);
            h = phasecont.cobweb(m, 0, 5, Parent = testCase.newAxes());
            testCase.verifyEqual(h.Graph.XData([1 end]), [-0.1 0.1]);
            testCase.verifyEqual(h.Diagonal.XData, [-0.1 0.1]);
            testCase.verifyEqual(h.Steps.XData(:), zeros(11, 1));
            testCase.verifyEqual(h.Steps.YData(:), zeros(11, 1));
        end

        function paramsSetTheIteratesAndTheGraph(testCase)
            m = testCase.model(testCase.Logistic);
            [h, tr] = phasecont.cobweb(m, 0.1, 10, Params = struct("r", 3.2), ...
                Parent = testCase.newAxes());
            x = CobwebTest.logistic(3.2, 0.1, 10);
            testCase.verifyEqual(tr.X, x);
            testCase.verifyEqual(tr.P, 3.2);
            xs = h.Graph.XData;
            testCase.verifyEqual(h.Graph.YData, 3.2 * xs .* (1 - xs), AbsTol = 1e-14);
            [~, again] = phasecont.cobweb(m, 0.1, 10, Params = 3.2, ...
                Parent = testCase.newAxes());
            testCase.verifyEqual(again.X, x);
        end

        function parentKeepsTheHoldState(testCase)
            m = testCase.model(testCase.Logistic);
            ax = testCase.newAxes();
            hold(ax, "off");
            phasecont.cobweb(m, 0.1, 4, Parent = ax);
            testCase.verifyFalse(ishold(ax));
            hold(ax, "on");
            phasecont.cobweb(m, 0.3, 4, Parent = ax);
            testCase.verifyTrue(ishold(ax));
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-cobweb"), 2);
        end

        function withoutParentHoldOnKeepsTheFirstPlot(testCase)
            m = testCase.model(testCase.Logistic);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            h1 = phasecont.cobweb(m, 0.1, 4);
            ax = h1.Steps.Parent;
            testCase.verifyEqual(ax.Parent, f);
            testCase.verifyEqual(h1.Steps.Color, ax.ColorOrder(1, :));
            hold(ax, "on");
            h2 = phasecont.cobweb(m, 0.3, 4);
            testCase.verifyEqual(h2.Steps.Parent, ax);
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-cobweb"), 2);
            testCase.verifyEqual(h2.Steps.Color, ax.ColorOrder(2, :));
            testCase.verifyTrue(ishold(ax));
            hold(ax, "off");
            phasecont.cobweb(m, 0.2, 4);
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-cobweb"), 1);
        end

        function noOutputReturnsNothing(testCase)
            m = testCase.model(testCase.Logistic);
            ax = testCase.newAxes();
            out = evalc("phasecont.cobweb(m, 0.1, 4, Parent = ax)");
            testCase.verifyEqual(strtrim(out), '');
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-cobweb"), 1);
        end

        function anEarlyStopDrawsTheKeptIterates(testCase)
            m = testCase.model(testCase.Square);
            ax = testCase.newAxes();
            [h, tr] = testCase.verifyWarning( ...
                @() phasecont.cobweb(m, 2, 12, Parent = ax), ...
                "phasecont:simulate:stoppedEarly");
            x = 2 .^ (2 .^ (0:9)');
            testCase.verifyEqual(tr.X, x);
            testCase.verifyEqual(tr.Stop, "Error");
            testCase.verifyNumElements(h.Steps.XData, 19);
            [X, Y] = CobwebTest.staircase(x);
            testCase.verifyEqual(h.Steps.XData(:), X);
            testCase.verifyEqual(h.Steps.YData(:), Y);
            g = CobwebTest.rangeOf(x);
            testCase.verifyEqual(h.Graph.XData([1 end]), g);
        end

        function refusesWhatIsNotAMapOfOneVariable(testCase)
            ax = testCase.newAxes();
            ode = testCase.model(testCase.Decay);
            testCase.verifyError(@() phasecont.cobweb(ode, 1, 5, Parent = ax), ...
                "phasecont:cobweb:badModel");
            testCase.verifyErrorText(@() phasecont.cobweb(ode, 1, 5, Parent = ax), ...
                "A cobweb plot needs a map, and this model is not a map.");
            henon = testCase.model(testCase.Henon);
            testCase.verifyErrorText(@() phasecont.cobweb(henon, [0 0], 5, Parent = ax), ...
                "A cobweb plot needs a map with one variable, and this map has 2.");
            shift = testCase.model(testCase.Shift);
            testCase.verifyErrorText(@() phasecont.cobweb(shift, 1, 5, Parent = ax), ...
                "A cobweb plot needs a map that does not use t.");
            testCase.verifyEmpty(ax.Children);
        end

        function refusesABadN(testCase)
            m = testCase.model(testCase.Logistic);
            ax = testCase.newAxes();
            bad = {0, 2.5, 2e6, -1, Inf, NaN, [1 2], "3", 1i};
            for k = 1:numel(bad)
                testCase.verifyError(@() phasecont.cobweb(m, 0.1, bad{k}, Parent = ax), ...
                    "phasecont:model:badOption", sprintf("bad N number %d", k));
            end
            try
                phasecont.cobweb(m, 0.1, 0, Parent = ax);
            catch err
                testCase.verifyEqual(string(err.message), ...
                    "N must be a whole number from 1 to 1000000.");
            end
            testCase.verifyEmpty(ax.Children);
        end
    end

    methods (Access = private)
        function verifyErrorText(testCase, fn, text)
            try
                fn();
                testCase.verifyFail("expected the error: " + text);
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:cobweb:badModel");
                testCase.verifyEqual(string(err.message), text);
            end
        end
    end
end
