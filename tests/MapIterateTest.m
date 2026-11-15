classdef MapIterateTest < matlab.unittest.TestCase
    % Tests slice 1 of map support (SPEC section 8u, test 5): simulate on a
    % map iterates x(k+1) = F(x(k), p, t = k) from n0 to n1, Model.IsMap
    % reports the kind, and plot and sessions keep the map style. The map
    % models of MapRefusalTest are reused, and every other model is written
    % here. The loops in the tests keep the association of the model's
    % expression tree: x^2 is x*x and r*x*(1-x) is (r*x)*(1-x).

    properties (Constant)
        % x(k+1) = x(k) + k, so t inside F is the iterate index.
        Index = "init x=0" + newline + "x(t+1)=x+t" + newline + "done" + newline
        % x(k+1) = x(k)^2 from 10 overflows at the ninth square.
        Square = "init x=10" + newline + "x(t+1)=x*x" + newline + "done" + newline
        Total = "par r=2" + newline + "init x=0.2" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "@ total=10" + newline + ...
            "done" + newline
        DiscreteTotal = "par r=2" + newline + "init x=0.2" + newline + ...
            "x'=r*x*(1-x)" + newline + "@ meth=discrete, total=5" + newline + ...
            "done" + newline
        Hutch = "par r=1.3,tau=1" + newline + ...
            "x'=r*x*(1-delay(x,tau))" + newline + "done" + newline
    end

    properties
        Dir
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
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Dir = t.Folder;
        end
    end

    methods (Access = private)
        function m = model(testCase, text)
            m = phasecont.Model.fromSource(text, Evaluator = "jit");
            testCase.addTeardown(@() delete(m));
        end
    end

    methods (Test)
        function isMapReportsTheKind(testCase)
            for text = [MapRefusalTest.Logistic MapRefusalTest.Henon ...
                    MapRefusalTest.LogisticDiscrete MapRefusalTest.HenonDiscrete]
                testCase.verifyTrue(testCase.model(text).IsMap, text);
            end
            testCase.verifyFalse(testCase.model(MapRefusalTest.Decay).IsMap);
            testCase.verifyFalse(testCase.model(testCase.Hutch).IsMap);
        end

        function theLogisticMapEqualsALoop(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            r = 2.5;
            tr = phasecont.simulate(m, 0.1, [0 50], Params = struct("r", r));
            x = zeros(51, 1);
            x(1) = 0.1;
            for k = 1:50
                x(k + 1) = (r * x(k)) * (1 - x(k));
            end
            testCase.verifyEqual(tr.T, (0:50)');
            testCase.verifyEqual(tr.X, x);
            testCase.verifyEqual(tr.TSpan, [0 50]);
            testCase.verifyEqual(tr.Stop, "End");
            testCase.verifyEqual(tr.P, r);
        end

        function tInsideFIsTheIterateIndex(testCase)
            m = testCase.model(testCase.Index);
            tr = phasecont.simulate(m, 0, [0 4]);
            testCase.verifyEqual(tr.T, (0:4)');
            testCase.verifyEqual(tr.X, [0; 0; 1; 3; 6]);
            tr = phasecont.simulate(m, 0, [5 9]);
            testCase.verifyEqual(tr.T, (5:9)');
            testCase.verifyEqual(tr.X, [0; 5; 11; 18; 26]);
            testCase.verifyEqual(tr.TSpan, [5 9]);
        end

        function theHenonMapEqualsALoop(testCase)
            m = testCase.model(MapRefusalTest.Henon);
            a = 1.4;
            b = 0.3;
            tr = phasecont.simulate(m, [0 0], [0 100]);
            x = zeros(101, 2);
            for k = 1:100
                x(k + 1, 1) = (1 - a * (x(k, 1) * x(k, 1))) + x(k, 2);
                x(k + 1, 2) = b * x(k, 1);
            end
            testCase.verifyEqual(tr.T, (0:100)');
            testCase.verifyEqual(tr.X, x);
        end

        function aBadSpanIsRefused(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            for span = {[0 1.5], [3 3], [4 2], [0 Inf], [0 2 * flintmax], 5}
                testCase.verifyError(@() phasecont.simulate(m, 0.1, span{1}), ...
                    "phasecont:model:badOption", mat2str(span{1}));
            end
        end

        function theSolverOptionsAreRefused(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            values = struct("RelTol", 1e-6, "AbsTol", 1e-6, "MaxSteps", 10, ...
                "Hmax", 0.1);
            for name = string(fieldnames(values))'
                args = {name, values.(name)};
                testCase.verifyError( ...
                    @() phasecont.simulate(m, 0.1, [0 5], args{:}), ...
                    "phasecont:simulate:mapOption", name);
            end
            try
                phasecont.simulate(m, 0.1, [0 5], RelTol = 1e-6);
            catch err
                testCase.verifyEqual(string(err.message), ...
                    "RelTol does not apply to a map: simulate iterates a map from n0 to n1.");
            end
        end

        function tooManyIteratesAreRefusedByTheEngine(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            testCase.verifyError(@() phasecont.simulate(m, 0.1, [0 1000001]), ...
                "phasecont:engine:badProblem");
        end

        function anOverflowStopsTheRunEarly(testCase)
            m = testCase.model(testCase.Square);
            lastwarn("");
            tr = [];
            [~] = evalc("tr = phasecont.simulate(m, 10, [0 20]);");
            [msg, id] = lastwarn;
            testCase.verifyEqual(string(id), "phasecont:simulate:stoppedEarly");
            testCase.verifySubstring(string(msg), "n = 8");
            testCase.verifySubstring(string(msg), "before n = 20");
            testCase.verifyEqual(tr.Stop, "Error");
            testCase.verifySize(tr.X, [9 1]);
            x = 10;
            for k = 1:8
                x = x * x;
            end
            testCase.verifyEqual(tr.X(end), x);
        end

        function atOptionsOfAMapAreNamed(testCase)
            m = testCase.model(testCase.Total);
            lastwarn("");
            [~] = evalc("phasecont.simulate(m, 0.2, [0 3]);");
            [msg, id] = lastwarn;
            testCase.verifyEqual(string(id), "phasecont:simulate:ignoredOption");
            testCase.verifySubstring(string(msg), "sets @ total.");
            testCase.verifySubstring(string(msg), "of a map");

            m = testCase.model(MapRefusalTest.LogisticDiscrete);
            testCase.verifyWarningFree(@() phasecont.simulate(m, 0.2, [0 3]));

            m = testCase.model(testCase.DiscreteTotal);
            lastwarn("");
            [~] = evalc("phasecont.simulate(m, 0.2, [0 3]);");
            [msg, id] = lastwarn;
            testCase.verifyEqual(string(id), "phasecont:simulate:ignoredOption");
            testCase.verifySubstring(string(msg), "sets @ total.");
        end

        function plotDrawsTheIteratesAsDots(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            tr = phasecont.simulate(m, 0.2, [0 10]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot(tr, Parent = ax);
            line = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyNumElements(line, 1);
            testCase.verifyEqual(string(line.LineStyle), "none");
            testCase.verifyEqual(string(line.Marker), ".");
            testCase.verifyEqual(line.XData(:), (0:10)');
            testCase.verifyEqual(line.YData(:), tr.X);
            testCase.verifyEqual(string(ax.XLabel.String), "n");
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-start"), 1);
        end

        function aSessionKeepsTheIterates(testCase)
            m = testCase.model(MapRefusalTest.Henon);
            tr = phasecont.simulate(m, [0 0], [0 30]);
            testCase.verifyEqual(tr.Record.Type, "ITER");
            file = fullfile(testCase.Dir, "henon_iter.pcsession.json");
            save(phasecont.Session(m, {tr}), file);

            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            again = back.Items{1};
            testCase.verifyClass(again, "phasecont.Trajectory");
            testCase.verifyEqual(again.T, tr.T);
            testCase.verifyEqual(again.X, tr.X);
            testCase.verifyEqual(again.TSpan, [0 30]);
            testCase.verifyTrue(again.Map);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot(again, Parent = ax);
            line = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyEqual(string(line.LineStyle), "none");
            testCase.verifyEqual(string(line.Marker), ".");

            r = phasecont.session.replay(file);
            testCase.verifySize(r, [1 1]);
            testCase.verifyEqual(r.Kind, "simulate");
            testCase.verifyEqual(r.Type, "ITER");
            testCase.verifyTrue(r.Pass);
            testCase.verifyTrue(r.Identical);
        end
    end
end
