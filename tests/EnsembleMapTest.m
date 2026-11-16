classdef EnsembleMapTest < matlab.unittest.TestCase
    % Tests slice 7 of map support (SPEC section 8aa, test 7): ensemble on a
    % map iterates x(k+1) = F(x(k), p, t = k) from n0 to n1 with method
    % "iter", on the CPU here. The map models of MapRefusalTest and
    % MapIterateTest are reused. The loops keep the association of the
    % model's expression tree: x^2 is x*x and r*x*(1-x) is (r*x)*(1-x).

    properties (Constant)
        % x(k+1) = 2 x(k) from 1 passes 1e3 at the tenth iterate.
        Double = "init x=1" + newline + "x(t+1)=2*x" + newline + "done" + newline
        Id = "phasecont:ensemble:memberStopped"
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

        function quiet(testCase)
            % Keeps memberStopped off the screen; lastwarn still records it.
            state = warning("off", testCase.Id);
            testCase.addTeardown(@() warning(state));
        end
    end

    methods (Static, Access = private)
        function x = logistic(r, x0, n)
            % x(k) for k = 0..n in the class of r and x0.
            x = zeros(n + 1, 1, "like", x0);
            x(1) = x0;
            for k = 1:n
                x(k + 1) = (r * x(k)) * (1 - x(k));
            end
        end

        function x = henon(a, b, x0, n)
            % [x(k) y(k)] for k = 0..n in the class of a, b and x0.
            x = zeros(n + 1, 2, "like", x0);
            x(1, :) = x0;
            for k = 1:n
                x(k + 1, 1) = (1 - a * (x(k, 1) * x(k, 1))) + x(k, 2);
                x(k + 1, 2) = b * x(k, 1);
            end
        end
    end

    methods (Test)
        % ---- double and single against loops ----

        function theLogisticMapEqualsALoopInDouble(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            x0 = [0.1 0.3 0.7];
            r = phasecont.ensemble(m, x0, 2.5, [0 10 20 50], Backend = "cpu", ...
                Precision = "double");
            testCase.verifyEqual(r.Times, [10; 20; 50]);
            testCase.verifySize(r.States, [1 3 3]);
            testCase.verifyEqual(r.Status, ["ok"; "ok"; "ok"]);
            testCase.verifyEqual(r.TReached, [50; 50; 50]);
            testCase.verifyEqual(r.Accepted, [50; 50; 50]);
            testCase.verifyEqual(r.Rejected, [0; 0; 0]);
            for k = 1:3
                x = EnsembleMapTest.logistic(2.5, x0(k), 50);
                testCase.verifyEqual(r.States(1, :, k), x([11 21 51]).', ...
                    "AbsTol", 0, sprintf("member %d", k));
            end
            tr = phasecont.simulate(m, x0(2), [0 50], Params = struct("r", 2.5));
            testCase.verifyEqual(r.States(1, :, 2), tr.X([11 21 51]).', "AbsTol", 0);
            testCase.verifyEqual(r.Label.Backend, "cpu");
            testCase.verifyEqual(r.Label.Precision, "double");
            testCase.verifyFalse(r.Label.SinglePrecision);
        end

        function theHenonMapEqualsALoopInDouble(testCase)
            m = testCase.model(MapRefusalTest.Henon);
            x0 = [0 0.1; 0 -0.1];
            r = phasecont.ensemble(m, x0, [], [0 10 20 100], Backend = "cpu", ...
                Precision = "double");
            testCase.verifySize(r.States, [2 3 2]);
            testCase.verifyEqual(r.Status, ["ok"; "ok"]);
            for k = 1:2
                x = EnsembleMapTest.henon(1.4, 0.3, x0(:, k).', 100);
                testCase.verifyEqual(r.States(:, :, k), x([11 21 101], :).', ...
                    "AbsTol", 0, sprintf("member %d", k));
                tr = phasecont.simulate(m, x0(:, k).', [0 100]);
                testCase.verifyEqual(r.States(:, :, k), tr.X([11 21 101], :).', ...
                    "AbsTol", 0, sprintf("member %d", k));
            end
        end

        function theLogisticMapEqualsASingleLoopInSingle(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            x0 = [0.1 0.3];
            r = phasecont.ensemble(m, x0, 2.5, [0 10 20 50], Backend = "cpu", ...
                Precision = "single");
            testCase.verifyTrue(r.Label.SinglePrecision);
            testCase.verifyEqual(r.Label.Precision, "single");
            for k = 1:2
                x = EnsembleMapTest.logistic(single(2.5), single(x0(k)), 50);
                testCase.verifyEqual(r.States(1, :, k), double(x([11 21 51])).', ...
                    "AbsTol", 0, sprintf("member %d", k));
            end
        end

        function theHenonMapEqualsASingleLoopInSingle(testCase)
            m = testCase.model(MapRefusalTest.Henon);
            x0 = [0; 0];
            r = phasecont.ensemble(m, x0, [], [0 10 20 100], Backend = "cpu", ...
                Precision = "single");
            x = EnsembleMapTest.henon(single(1.4), single(0.3), single([0 0]), 100);
            testCase.verifyEqual(r.States, double(x([11 21 101], :)).', "AbsTol", 0);
        end

        function tInsideTheModelIsTheIterateIndex(testCase)
            m = testCase.model(MapIterateTest.Index);
            r = phasecont.ensemble(m, 0, [], [0 1 2 3 4], Backend = "cpu", ...
                Precision = "double");
            testCase.verifyEqual(r.Times, (1:4)');
            testCase.verifyEqual(r.States, [0 1 3 6]);
            % From n0 = 5 the first iterate adds 5.
            r = phasecont.ensemble(m, 0, [], [5 6 9], Backend = "cpu", ...
                Precision = "double");
            testCase.verifyEqual(r.States, [5 26]);
        end

        % ---- options and spans ----

        function stepOptionsAreRefusedForAMap(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            values = {"Dt", 0.1; "Rtol", 1e-6; "Atol", 1e-8; "Hmax", 1; "MaxSteps", 10};
            for k = 1:size(values, 1)
                testCase.verifyError(@() phasecont.ensemble(m, 0.1, [], [0 5], ...
                    "Backend", "cpu", "Precision", "double", values{k, :}), ...
                    "phasecont:ensemble:mapOption", values{k, 1});
            end
        end

        function theMapOptionsAreCheckedInOrderAfterPrecision(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            testCase.verifyError(@() phasecont.ensemble(m, 0.1, [], [0 5], ...
                Backend = "cpu", Dt = 0.1), "phasecont:ensemble:missingPrecision");
            try
                phasecont.ensemble(m, 0.1, [], [0 5], Backend = "cpu", ...
                    Precision = "double", MaxSteps = 10, Rtol = 1e-6, Dt = 0.1);
                testCase.verifyFail("no error");
            catch e
                testCase.verifyEqual(string(e.identifier), "phasecont:ensemble:mapOption");
                testCase.verifyTrue(startsWith(e.message, "Dt does not apply"), e.message);
            end
        end

        function badMapSpansAreRefused(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            spans = {[0 1.5], [0 2 1], [0 Inf], [0 2 * flintmax], [0 2^32], ...
                [-1.5 3], "0 5"};
            for k = 1:numel(spans)
                testCase.verifyError(@() phasecont.ensemble(m, 0.1, [], spans{k}, ...
                    Backend = "cpu", Precision = "double"), ...
                    "phasecont:ensemble:badOption", sprintf("span %d", k));
            end
        end

        function aLongSingleSpanIsRefusedByTheEngine(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            testCase.verifyError(@() phasecont.ensemble(m, 0.1, [], [0 2^24 + 1], ...
                Backend = "cpu", Precision = "single"), ...
                "phasecont:engine:badProblem");
        end

        % ---- stops ----

        function anOverflowWarnsWithTheEndIterate(testCase)
            m = testCase.model(MapIterateTest.Square);
            launch = @() phasecont.ensemble(m, 10, [], [0 20], Backend = "cpu", ...
                Precision = "double");
            testCase.verifyWarning(launch, testCase.Id);
            testCase.quiet();
            lastwarn("");
            r = launch();
            [msg, id] = lastwarn;
            testCase.verifyEqual(string(id), testCase.Id);
            testCase.verifyEqual(string(msg), ...
                "1 of 1 members stopped before n = 20 (notFinite). See Status.");
            testCase.verifyEqual(r.Status, "notFinite");
            testCase.verifyEqual(r.TReached, 8);
            testCase.verifyEqual(r.Accepted, 8);
            testCase.verifyTrue(isnan(r.Final));
            s = phasecont.ensemble(m, 10, [], [0 8 20], Backend = "cpu", ...
                Precision = "double");
            x = 10;
            for k = 1:8
                x = x * x;
            end
            testCase.verifyEqual(s.States(1, 1, 1), x, "AbsTol", 0);
            testCase.verifyTrue(isnan(s.States(1, 2, 1)));
        end

        function boundsStopsAMapMemberWithoutAWarning(testCase)
            m = testCase.model(testCase.Double);
            launch = @() phasecont.ensemble(m, [1 1e-4], [], [0 20], ...
                Backend = "cpu", Precision = "double", Bounds = 1e3);
            testCase.verifyWarningFree(launch);
            r = launch();
            testCase.verifyEqual(r.Status, ["bounds"; "ok"]);
            testCase.verifyEqual(r.TReached, [9; 20]);
            testCase.verifyEqual(r.Accepted, [9; 20]);
            testCase.verifyTrue(isnan(r.Final(1)));
            testCase.verifyEqual(r.Final(2), 1e-4 * 2^20);
            s = phasecont.ensemble(m, [1 1e-4], [], [0 9 20], ...
                Backend = "cpu", Precision = "double", Bounds = 1e3);
            testCase.verifyEqual(squeeze(s.States(1, 1, :)), [512; 1e-4 * 512]);
            testCase.verifyTrue(isnan(s.States(1, 2, 1)));
            testCase.verifyEqual(s.States(1, 2, 2), 1e-4 * 2^20);
        end

        % ---- refine and Job ----

        function refineOfASingleMapRunEqualsTheDoubleRun(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            x0 = [0.1 0.3];
            s = phasecont.ensemble(m, x0, 2.5, [0 10 20 50], Backend = "cpu", ...
                Precision = "single");
            d = phasecont.ensemble(m, x0, 2.5, [0 10 20 50], Backend = "cpu", ...
                Precision = "double");
            f = s.refine(2);
            testCase.verifyFalse(f.Label.SinglePrecision);
            testCase.verifyEqual(f.Label.Backend, "cpu");
            testCase.verifyEqual(f.Times, d.Times);
            testCase.verifyEqual(f.States, d.States(:, :, 2), "AbsTol", 0);
            testCase.verifyEqual(f.Job.Method, "iter");
        end

        function theJobRecordsTheIterMethod(testCase)
            m = testCase.model(MapRefusalTest.Logistic);
            r = phasecont.ensemble(m, 0.1, [], [3 10 40], Backend = "cpu", ...
                Precision = "double", Bounds = 5);
            j = r.Job;
            testCase.verifyEqual(j.Method, "iter");
            testCase.verifyEqual(j.MaxSteps, 37);
            testCase.verifyEqual(j.TSpan, [3 10 40]);
            testCase.verifyEqual(j.Bounds, 5);
            for name = ["Dt" "Rtol" "Atol" "Hmax"]
                testCase.verifyEmpty(j.(name), name);
            end
        end

        function anOdeStillRunsWithItsStepOptions(testCase)
            m = testCase.model(MapRefusalTest.Decay);
            testCase.verifyFalse(m.IsMap);
            a = phasecont.ensemble(m, 1, [], [0 1], Backend = "cpu", ...
                Precision = "double", Dt = 0.01);
            testCase.verifyEqual(a.Job.Method, "rk4");
            testCase.verifyEqual(a.Job.MaxSteps, 100000);
            testCase.verifyEqual(a.Status, "ok");
            b = phasecont.ensemble(m, 1, [], [0 1], Backend = "cpu", ...
                Precision = "double", Rtol = 1e-8, Atol = 1e-10, MaxSteps = 500);
            testCase.verifyEqual(b.Job.Method, "dp5");
            testCase.verifyEqual(b.Job.MaxSteps, 500);
            testCase.verifyEqual(b.Status, "ok");
        end
    end
end
