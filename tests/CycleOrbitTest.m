classdef CycleOrbitTest < matlab.unittest.TestCase
    % Tests the LC start from a simulated orbit (SPEC section 11, "Limit
    % cycles from a simulation"): the detected period, the Period option,
    % the struct start, the refusals and a session replay. The model is the
    % Hopf normal form x' = mu*x - y - x*r^2, y' = x + mu*y - y*r^2, whose
    % cycle at mu = 1 is the unit circle with period 2*pi.

    properties (Constant)
        Normal = "par mu=1" + newline + ...
            "x'=mu*x-y-x*(x^2+y^2)" + newline + ...
            "y'=x+mu*y-y*(x^2+y^2)" + newline + ...
            "init x=0.5,y=0" + newline + "done" + newline
        Forced = "par mu=1" + newline + ...
            "x'=mu*x-y-x*(x^2+y^2)+0.01*sin(t)" + newline + ...
            "y'=x+mu*y-y*(x^2+y^2)" + newline + ...
            "init x=0.5,y=0" + newline + "done" + newline
        % A Rossler system. With a = b = 0.2 its cycle doubles its period
        % near c = 2.83: one turn takes about 5.76 below that and two turns
        % take about 11.5 above it.
        Rossler = "par a=0.2,b=0.2,c=2.5" + newline + ...
            "x'=-y-z" + newline + "y'=x+a*y" + newline + ...
            "z'=b+z*(x-c)" + newline + "init x=1,y=1,z=0" + newline + ...
            "done" + newline
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

    methods
        function m = model(testCase, text)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = "jit");
        end

        function [m, tr] = settled(testCase)
            % About 9.5 turns from radius 0.5: the orbit has settled onto the
            % unit circle long before the end.
            m = testCase.model(testCase.Normal);
            tr = phasecont.simulate(m, [0.5 0], [0 60], RelTol = 1e-10, ...
                AbsTol = 1e-12);
        end

        function [lc, msg] = detect(testCase, c, tEnd)
            % The LC run from a Rossler orbit at c with the detected period,
            % and the text of its warning.
            m = testCase.model(testCase.Rossler);
            tr = phasecont.simulate(m, [1 1 0], [0 tEnd], ...
                Params = struct("c", c), RelTol = 1e-10, AbsTol = 1e-12);
            w = warning("off", "phasecont:continuation:detectedPeriod");
            restore = onCleanup(@() warning(w));
            lastwarn("");
            lc = phasecont.continuation(m, tr, "Type", "LC", "Free", "c", ...
                "NTST", 40, "NCOL", 4, "MaxSteps", 2);
            [msg, id] = lastwarn();
            testCase.verifyEqual(id, 'phasecont:continuation:detectedPeriod');
        end

        function [m, tr] = rossler(testCase, c, tEnd)
            m = testCase.model(testCase.Rossler);
            tr = phasecont.simulate(m, [1 1 0], [0 tEnd], ...
                Params = struct("c", c), RelTol = 1e-10, AbsTol = 1e-12);
        end

        function msg = refusal(testCase, m, tr)
            % The message of the badStart error an LC run without Period
            % gives when the period is not clear.
            msg = "";
            try
                phasecont.continuation(m, tr, "Type", "LC", "Free", "c", ...
                    "NTST", 40, "NCOL", 4, "MaxSteps", 2);
                testCase.verifyFail("the run was not refused");
            catch e
                testCase.verifyEqual(string(e.identifier), "phasecont:model:badStart");
                msg = string(e.message);
            end
        end

        function lc = runLC(~, m, start, varargin)
            lc = phasecont.continuation(m, start, "Type", "LC", "Free", "mu", ...
                "NTST", 20, "NCOL", 4, "MaxSteps", 8, "Bounds", [0.5 1.5], varargin{:});
        end
    end

    methods (Test)
        function aSettledOrbitGivesOnePeriod(testCase)
            [m, tr] = testCase.settled();
            lc = testCase.verifyWarning(@() testCase.runLC(m, tr), ...
                "phasecont:continuation:detectedPeriod");
            % One period, not a multiple of it.
            testCase.verifyEqual(lc.Period(1), 2*pi, AbsTol = 1e-6);
            [~, x] = lc.cycle(1);
            testCase.verifyEqual(vecnorm(x, 2, 2), ones(size(x, 1), 1), ...
                AbsTol = 1e-6);
            testCase.verifyGreaterThan(numel(lc.Period), 1, ...
                "the run should continue in mu");
        end

        function theWarningNamesThePeriod(testCase)
            [m, tr] = testCase.settled();
            w = warning("off", "phasecont:continuation:detectedPeriod");
            c = onCleanup(@() warning(w));
            lastwarn("");
            testCase.runLC(m, tr);
            [msg, id] = lastwarn();
            testCase.verifyEqual(id, 'phasecont:continuation:detectedPeriod');
            testCase.verifySubstring(msg, "Period");
            testCase.verifySubstring(msg, "6.28");
        end

        function aGivenPeriodSkipsTheDetection(testCase)
            [m, tr] = testCase.settled();
            w = warning("off", "phasecont:continuation:detectedPeriod");
            c = onCleanup(@() warning(w));
            detected = testCase.runLC(m, tr);
            delete(c);
            given = testCase.verifyWarningFree(@() testCase.runLC(m, tr, Period = 2*pi));
            testCase.verifyEqual(given.Period(1), detected.Period(1), AbsTol = 1e-8);
        end

        function aStructStartMatchesTheTrajectory(testCase)
            [m, tr] = testCase.settled();
            s = struct("t", tr.T, "x", tr.X, "p", struct("mu", 1));
            a = testCase.runLC(m, tr, Period = 2*pi);
            b = testCase.runLC(m, s, Period = 2*pi);
            testCase.verifyEqual(b.U(1, :), a.U(1, :), AbsTol = 1e-12);
            testCase.verifyEqual(b.Period(1), a.Period(1), AbsTol = 1e-12);
        end

        function paramsApplyAfterTheDetection(testCase)
            [m, tr] = testCase.settled();
            w = warning("off", "phasecont:continuation:detectedPeriod");
            c = onCleanup(@() warning(w));
            lc = testCase.runLC(m, tr, Params = struct("mu", 1.01));
            % The start orbit comes from mu = 1, and the run is at 1.01.
            testCase.verifyEqual(lc.Alpha(1), 1.01, AbsTol = 1e-6);
            testCase.verifyEqual(lc.Period(1), 2*pi, AbsTol = 1e-3);
        end

        function badStartsAreRefused(testCase)
            m = testCase.model(testCase.Normal);
            p = struct("mu", 1);
            bad = { ...
                struct("t", 0, "x", [1 0], "p", p), ...
                struct("t", [0; 1; 1], "x", [1 0; 0 1; -1 0], "p", p), ...
                struct("t", [0; NaN; 1], "x", [1 0; 0 1; -1 0], "p", p), ...
                struct("t", [0; 1; 2], "x", [1 0 0; 0 1 0; -1 0 0], "p", p), ...
                struct("t", [0; 1; 2], "x", [1 0; Inf 1; -1 0], "p", p)};
            for k = 1:numel(bad)
                testCase.verifyError(@() testCase.runLC(m, bad{k}), ...
                    "phasecont:model:badStart", sprintf("case %d", k));
            end
            testCase.verifyError(@() testCase.runLC(m, ...
                struct("t", [0; 1], "x", [1 0; 0 1], "p", struct("nu", 1))), ...
                "phasecont:model:unknownParameter");
        end

        function aTrajectoryOfAnotherModelIsRefused(testCase)
            [~, tr] = testCase.settled();
            other = testCase.model(testCase.Normal);
            testCase.verifyError(@() testCase.runLC(other, tr), ...
                "phasecont:model:badStart");
        end

        function aModelThatUsesTimeIsRefused(testCase)
            m = testCase.model(testCase.Forced);
            tr = phasecont.simulate(m, [0.5 0], [0 30]);
            testCase.verifyError(@() testCase.runLC(m, tr, Period = 2*pi), ...
                "phasecont:model:badStart");
        end

        function anEquilibriumOrbitIsRefused(testCase)
            m = testCase.model(testCase.Normal);
            tr = phasecont.simulate(m, [0 0], [0 10]);
            testCase.verifyError(@() testCase.runLC(m, tr), ...
                "phasecont:model:badStart");
        end

        function anOrbitWithNoReturnIsRefused(testCase)
            % Three time units turn the orbit by less than one full turn.
            m = testCase.model(testCase.Normal);
            tr = phasecont.simulate(m, [1 0], [0 3]);
            testCase.verifyError(@() testCase.runLC(m, tr), ...
                "phasecont:model:badStart");
        end

        function badPeriodsAreRefused(testCase)
            [m, tr] = testCase.settled();
            for v = {0, -1, NaN, Inf, [1 2], "6", 61}
                testCase.verifyError(@() testCase.runLC(m, tr, Period = v{1}), ...
                    "phasecont:model:badOption");
            end
            % Period applies to an LC run from an orbit only.
            eq = phasecont.equilibrium(m, [0 0]);
            testCase.verifyError(@() testCase.runLC(m, eq, Period = 1), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, eq, ...
                Free = "mu", Period = 1), "phasecont:model:badOption");
        end

        function aSettledCycleNamesNoOtherReturn(testCase)
            [lc, msg] = testCase.detect(2.5, 600);
            testCase.verifyEqual(lc.Period(1), 5.749, AbsTol = 1e-3);
            testCase.verifyFalse(contains(msg, "multiple"), msg);
        end

        function aDoubledCycleIsRefusedWithBothPeriods(testCase)
            % Past the doubling the one-turn return is far from the end, so
            % the two-turn return is the close one. The run stops and names
            % both periods, and Period picks the two-turn cycle.
            [m, tr] = testCase.rossler(3.5, 600);
            msg = testCase.refusal(m, tr);
            testCase.verifySubstring(msg, "T = 11.54");
            testCase.verifySubstring(msg, "T = 5.");
            testCase.verifySubstring(msg, "Pass Period = 5.");
            lc = phasecont.continuation(m, tr, "Type", "LC", "Free", "c", ...
                "NTST", 40, "NCOL", 4, "MaxSteps", 2, "Period", 11.545);
            testCase.verifyEqual(lc.Period(1), 11.545, AbsTol = 1e-3);
        end

        function anUnsettledOrbitNearADoublingIsRefused(testCase)
            % Below the doubling the orbit has not settled by t = 120 and the
            % one-turn return is not close enough. Without the refusal the
            % run would follow the cycle traversed twice (T about 11.52).
            % Period gives the one-turn cycle.
            [m, tr] = testCase.rossler(2.7, 120);
            msg = testCase.refusal(m, tr);
            testCase.verifySubstring(msg, "T = 11.5");
            testCase.verifySubstring(msg, "T = 5.78");
            testCase.verifySubstring(msg, "Pass Period = 5.78");
            one = phasecont.continuation(m, tr, "Type", "LC", "Free", "c", ...
                "NTST", 40, "NCOL", 4, "MaxSteps", 2, "Period", 5.79);
            testCase.verifyEqual(one.Period(1), 5.7619, AbsTol = 1e-3);
        end

        function aSessionReplaysTheRun(testCase)
            [m, tr] = testCase.settled();
            lc = testCase.runLC(m, tr, Period = 2*pi);
            file = fullfile(testCase.Dir, "orbit.pcsession.json");
            save(phasecont.Session(m, {tr, lc}), file);
            r = phasecont.session.replay(file);
            testCase.verifyEqual([r.Type], ["IVP" "LC"]);
            testCase.verifyTrue(all([r.Pass]), "a run did not reproduce");
            back = phasecont.Session.load(file);
            testCase.verifyEqual(back.Items{2}.U(1, :), lc.U(1, :));
        end
    end
end
