classdef DelayBranchTest < matlab.unittest.TestCase
    % Tests the MATLAB front end of the delay-equation engine (SPEC section
    % 8s, slice 1): a model with delay(...) calls, an EP run with Roots, the
    % delay Branch layout and a Session round trip. Every model is written
    % here. The numbers are those of SPEC section 8e, tests 1, 4, 5 and 6.

    properties (Constant)
        Hutch = "par r=1.3,tau=1" + newline + ...
            "x'=r*x*(1-delay(x,tau))" + newline + "done" + newline
        Fold = "par mu=-0.04" + newline + ...
            "x'=mu-x*x-0.5*delay(x,1)" + newline + "done" + newline
        Pitchfork = "par mu=0.35" + newline + ...
            "x'=mu*x-x*x*x-0.5*delay(x,1)" + newline + "done" + newline
        Decay = "par a=2" + newline + "init x=3" + newline + ...
            "x'=-a*x" + newline + "done" + newline
        Roots = struct(N = 30, ReMin = -0.5)
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
        function m = model(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end

        function br = hutchinson(testCase, m)
            br = phasecont.continuation(m, 1, Free = "r", ...
                Roots = testCase.Roots, Bounds = [1.2 2.0], Direction = 1, ...
                Hmax = 0.05);
        end

        function verifyErrorText(testCase, fn, id, text)
            % FN raises the error ID, and its message contains TEXT.
            try
                fn();
            catch err
                testCase.verifyEqual(string(err.identifier), string(id));
                testCase.verifySubstring(string(err.message), text);
                return
            end
            testCase.verifyFail("no error was raised, expected " + id);
        end
    end

    methods (Test)
        function theModelKnowsItIsADelayModel(testCase)
            m = testCase.model(testCase.Hutch);
            testCase.verifyTrue(m.IsDelay);
            testCase.verifyEqual(m.Variables, "x");
            testCase.verifyEqual(m.Parameters, ["r" "tau"]);
            testCase.verifyEqual(m.ParameterValues, [1.3; 1]);
            ode = testCase.model(testCase.Decay);
            testCase.verifyFalse(ode.IsDelay);
        end

        function aDelayModelRaisesNoParseWarning(testCase)
            % ODE0054 says the generators refuse delays; a delay model is
            % the case the toolbox now runs, so MATLAB stays quiet.
            testCase.verifyWarningFree( ...
                @() testCase.model(testCase.Hutch));
        end

        function aotOnADelayModelFallsBackToTheJitQuietly(testCase)
            f = @() testCase.model(testCase.Hutch, "aot");
            testCase.verifyWarningFree(f);
            m = testCase.model(testCase.Hutch, "aot");
            testCase.verifyEqual(m.Evaluator, "jit");
            testCase.verifyTrue(m.IsDelay);
        end

        function ordinaryEntriesRefuseADelayModel(testCase)
            m = testCase.model(testCase.Hutch);
            try
                phasecont.equilibrium(m, 1);
                testCase.verifyFail("equilibrium accepted a delay model");
            catch err
                testCase.verifySubstring(string(err.message), "delay terms");
            end
            try
                phasecont.simulate(m, 1, [0 1]);
                testCase.verifyFail("simulate accepted a delay model");
            catch err
                testCase.verifySubstring(string(err.message), "delay terms");
            end
        end

        function hutchinsonHasOneHopfPoint(testCase)
            m = testCase.model(testCase.Hutch);
            br = testCase.hutchinson(m);
            testCase.verifyEqual(br.Type, "EP");
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.verifyEmpty(br.Warnings);
            testCase.verifyEqual(br.Alpha(end), 2.0, AbsTol = 1e-9);
            testCase.verifyEqual(br.U, ones(size(br.U)), AbsTol = 1e-10);
            h = br.points("H");
            testCase.assertNumElements(h, 1);
            testCase.verifyEqual(h.P(1), pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(h.Data.omega, pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(h.U, 1, AbsTol = 1e-10);
            testCase.verifyEqual(h.Data.values.unstable_before, 0);
            testCase.verifyEqual(h.Data.values.unstable_after, 2);
            testCase.verifyEmpty(h.Eigenvalues);
            testCase.verifyNumElements(br.Specials, 1);
            below = br.Alpha < pi/2;
            testCase.verifyEqual(br.RootCount(below), zeros(nnz(below), 1));
            testCase.verifyEqual(br.RootCount(~below), 2*ones(nnz(~below), 1));
        end

        function theBranchLayoutHasNoEigenvaluesOrTests(testCase)
            m = testCase.model(testCase.Hutch);
            br = testCase.hutchinson(m);
            n = numel(br.Alpha);
            testCase.verifySize(br.Eigenvalues, [n 0]);
            testCase.verifySize(br.Tests, [n 0]);
            testCase.verifySize(br.RootCount, [n 1]);
            testCase.verifyEqual(br.Stable, br.RootCount == 0);
            testCase.verifyTrue(any(br.Stable) && any(~br.Stable));
            testCase.verifySize(br.StepSize, [n 1]);
            testCase.verifyTrue(all(isnan(br.StepSize)));
            testCase.verifyTrue(all(isnan(br.NewtonIters)));
        end

        function anOdeBranchHasNoRootCount(testCase)
            m = testCase.model(testCase.Decay);
            br = phasecont.continuation(m, 3, Free = "a", Bounds = [1 3], ...
                MaxSteps = 5);
            testCase.verifyEmpty(br.RootCount);
            testCase.verifyEqual(size(br.Eigenvalues, 2), 1);
        end

        function aFoldIsReportedWithTheCountChange(testCase)
            m = testCase.model(testCase.Fold);
            br = phasecont.continuation(m, -0.1, Free = "mu", ...
                Roots = testCase.Roots, Bounds = [-0.07 -0.03], Direction = -1);
            testCase.verifyEmpty(br.Warnings);
            lp = br.points("LP");
            testCase.assertNumElements(lp, 1);
            testCase.verifyEqual(lp.P(1), -0.0625, AbsTol = 1e-8);
            testCase.verifyEqual(lp.U, -0.25, AbsTol = 1e-8);
            testCase.verifyEqual(lp.Data.values.unstable_before, 0);
            testCase.verifyEqual(lp.Data.values.unstable_after, 1);
            testCase.verifyEmpty(br.points("H"));
            above = br.U > -0.25 + 1e-6;
            below = br.U < -0.25 - 1e-6;
            testCase.verifyEqual(br.RootCount(above), zeros(nnz(above), 1));
            testCase.verifyEqual(br.RootCount(below), ones(nnz(below), 1));
        end

        function aPitchforkIsReportedOnTheTrivialBranch(testCase)
            m = testCase.model(testCase.Pitchfork);
            br = phasecont.continuation(m, 0, Free = "mu", ...
                Roots = testCase.Roots, Bounds = [0.3 0.8], Direction = 1);
            testCase.verifyEmpty(br.Warnings);
            bp = br.points("BP");
            testCase.assertNumElements(bp, 1);
            testCase.verifyEqual(bp.P(1), 0.5, AbsTol = 1e-8);
            testCase.verifyEqual(bp.Data.values.unstable_before, 0);
            testCase.verifyEqual(bp.Data.values.unstable_after, 1);
            testCase.verifyEmpty(br.points("H"));
            left = br.Alpha < 0.5 - 1e-6;
            right = br.Alpha > 0.5 + 1e-6;
            testCase.verifyEqual(br.RootCount(left), zeros(nnz(left), 1));
            testCase.verifyEqual(br.RootCount(right), ones(nnz(right), 1));
        end

        function hutchinsonBelowTheHopfValueHasNoSpecial(testCase)
            m = testCase.model(testCase.Hutch);
            br = phasecont.continuation(m, 1, Free = "r", ...
                Roots = testCase.Roots, Bounds = [0.5 1.4], Direction = 1, ...
                Params = struct(r = 1), Hmax = 0.05);
            testCase.verifyEmpty(br.Specials);
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.verifyEqual(br.RootCount, zeros(numel(br.Alpha), 1));
            testCase.verifyEmpty(br.Warnings);
        end

        function detectIsLimitedToLimitAndBranchPoints(testCase)
            m = testCase.model(testCase.Fold);
            br = phasecont.continuation(m, -0.1, Free = "mu", ...
                Roots = testCase.Roots, Bounds = [-0.07 -0.03], ...
                Direction = -1, Detect = "BP");
            testCase.verifyEmpty(br.points("LP"));
            testCase.verifyNotEmpty(br.Warnings);
        end

        function rootsAreRequiredForADelayModel(testCase)
            m = testCase.model(testCase.Hutch);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, Free = "r", Bounds = [1.2 2]), ...
                "phasecont:model:badOption", "Roots");
        end

        function rootsAreRefusedForAnOdeModel(testCase)
            m = testCase.model(testCase.Decay);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 3, Free = "a", ...
                    Roots = testCase.Roots, Bounds = [1 3]), ...
                "phasecont:model:badOption", "Roots");
        end

        function badRootsAreRefused(testCase)
            m = testCase.model(testCase.Hutch);
            bad = {struct(N = 0, ReMin = -0.5), ...
                struct(N = 2.5, ReMin = -0.5), ...
                struct(N = 30, ReMin = 0), ...
                struct(N = 30, ReMin = 0.5), ...
                struct(N = 30), ...
                struct(ReMin = -0.5), ...
                struct(N = 30, ReMin = -0.5, M = 3), ...
                struct(N = [30 40], ReMin = -0.5), ...
                30};
            for k = 1:numel(bad)
                roots = bad{k};
                testCase.verifyErrorText( ...
                    @() phasecont.continuation(m, 1, Free = "r", ...
                        Roots = roots, Bounds = [1.2 2]), ...
                    "phasecont:model:badOption", "Roots");
            end
        end

        function optionsThatTheDelayRunDoesNotHaveAreRefused(testCase)
            m = testCase.model(testCase.Hutch);
            id = "phasecont:model:badOption";
            base = {"Free", "r", "Roots", testCase.Roots, "Bounds", [1.2 2]};
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, base{:}, Type = "LP"), id, "Type");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, base{:}, MeshAdapt = 2), ...
                id, "MeshAdapt");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, base{:}, Switch = "BP1"), ...
                id, "Switch");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, base{:}, Detect = ["LP" "H"]), ...
                id, "Detect");
        end

        function aSpecialStartPointIsRefused(testCase)
            m = testCase.model(testCase.Hutch);
            br = testCase.hutchinson(m);
            h = br.points("H");
            testCase.assertNumElements(h, 1);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, h, Free = "r", ...
                    Roots = testCase.Roots, Bounds = [1.2 2]), ...
                "phasecont:model:badStart", "EP");
        end

        function anEquilibriumPointCanStartARun(testCase)
            m = testCase.model(testCase.Hutch);
            pt = phasecont.Point(U = 1, P = [1.3; 1], Label = "EP", Model = m);
            back = phasecont.continuation(m, pt, Free = "r", ...
                Roots = testCase.Roots, Bounds = [1.0 1.4], Direction = -1);
            testCase.verifyEqual(back.Alpha(end), 1.0, AbsTol = 1e-9);
            testCase.verifyEqual(back.RootCount, zeros(numel(back.Alpha), 1));
        end

        function aDelayBranchSurvivesASessionRoundTrip(testCase)
            m = testCase.model(testCase.Hutch);
            br = testCase.hutchinson(m);
            file = fullfile(testCase.Dir, "delay.pcs");
            save(phasecont.Session(m, {br}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.verifyTrue(back.Model.IsDelay);
            testCase.assertNumElements(back.Items, 1);
            b = back.Items{1};
            testCase.verifyEqual(b.Type, "EP");
            testCase.verifyEqual(b.Alpha, br.Alpha);
            testCase.verifyEqual(b.U, br.U);
            testCase.verifyEqual(b.RootCount, br.RootCount);
            testCase.verifyEqual(b.Stable, br.Stable);
            testCase.verifySize(b.Eigenvalues, [numel(br.Alpha) 0]);
            testCase.verifyEqual(b.Stop, br.Stop);
            h = b.points("H");
            testCase.assertNumElements(h, 1);
            testCase.verifyEqual(h.P(1), pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(h.Data.omega, pi/2, AbsTol = 1e-9);
        end
    end
end
