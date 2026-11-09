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
        Wright = "par a=1.5, b=0" + newline + ...
            "u'=-a*delay(u,1)*(1+u)+b*u^3" + newline + "done" + newline
        Coupled = "par a=1.4, b=1.2, c=1, d=0.1" + newline + ...
            "x'=c*y-a*delay(x,1)" + newline + ...
            "y'=-d*x-b*delay(y,1.5)" + newline + "done" + newline
        % The numbers of the engine tests (SPEC 8g-2, 8k): l1 at b = 0 on
        % the Wright cubic, the Bautin value of b, and the Hopf-Hopf point.
        L1AtZero = -0.214130922441773920
        BStar = 1.1844194631859732
        AStar = 1.454526952022688
        Omega1Star = 1.5160279413134377
        Omega2Star = 1.1308210684126256
        % SPEC 8s slice 3: the zero-Hopf model of SPEC 8f test 5 and the
        % coupled model with quadratic and cubic terms.
        Zh = "par a=0, tau=1, b=1" + newline + ...
            "x'=-(a+y)*x-b*delay(x,tau)" + newline + ...
            "y'=a-y*y" + newline + "done" + newline
        CoupledNl = "par a=1.4, b=1.2, c=1, d=0.1" + newline + ...
            "x'=c*y-a*delay(x,1)+x*x+x*y-x*x*x" + newline + ...
            "y'=-d*x-b*delay(y,1.5)+y*y*y+x*y*y" + newline + "done" + newline
        % The values of the engine run (pc-ffi test, 2026-11-08), pinned
        % to 1e-12 relative. Zeros are exact zeros of the model.
        ZhValues = struct( ...
            g200_re = -1, g200_im = 0, ...
            g110_re = -0.2884004391420009, g110_im = 0.45301835045029015, ...
            g011_re = 0, g011_im = 0, g300_re = 0, g300_im = 0, ...
            g111_re = 0, g111_im = 0, ...
            g210_re = 0.024758119360069802, g210_im = -0.1894641175334449, ...
            g021_re = 0, g021_im = 0, s_product = 0, s = 0, ...
            theta = 0.2884004391420009)
        HhValues = struct( ...
            omega_large = 1.5160279413134377, ...
            omega_small = 1.1308210684126256, ...
            g2100_re = -0.48383228630347963, g2100_im = 0.011378451292987219, ...
            g1011_re = -0.8907179371819094, g1011_im = -1.0542732722112986, ...
            g1110_re = 0.25251313826270416, g1110_im = 0.16044532760123023, ...
            g0021_re = 0.829549652892998, g0021_im = -0.5605505214493246, ...
            theta = -1.0737367366445048, delta = -0.5219022074610322, ...
            product = -0.40136290516147716, sign = -1)
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

        function br = wrightCurve(testCase, lyapunov)
            m = testCase.model(testCase.Wright);
            br = phasecont.continuation(m, 0.001, Type = "H", ...
                Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                Omega = 1.57, Lyapunov = lyapunov, Bounds = [-1 1], ...
                Hmax = 0.05, Direction = 1, TolX = 1e-12, TolF = 1e-12, ...
                MaxNewton = 20);
        end

        function br = coupledCurve(testCase, extra)
            m = testCase.model(testCase.Coupled);
            br = phasecont.continuation(m, [0.001 0.001], extra{:}, Type = "H", ...
                Free = ["a" "b"], Params = struct(a = 1.46, b = 0.8, ...
                    c = 1, d = 0.1), Omega = 1.54, Bounds = [0.5 1.4], ...
                Hmax = 0.05, Direction = 1, TolX = 1e-12, TolF = 1e-12, ...
                MaxNewton = 20);
        end

        function br = zhCurve(testCase, extra)
            m = testCase.model(testCase.Zh);
            br = phasecont.continuation(m, [0.001 -0.249], extra{:}, ...
                Type = "H", Free = ["a" "tau"], Params = struct(a = 0.07, ...
                    tau = 1.4071360574511174, b = 1), Omega = 0.98, ...
                Bounds = [1.3 2.0], Hmax = 0.05, Direction = 1, ...
                TolX = 1e-12, TolF = 1e-12, MaxNewton = 20);
        end

        function br = coupledNlCurve(testCase, extra)
            m = testCase.model(testCase.CoupledNl);
            br = phasecont.continuation(m, [0.001 0.001], extra{:}, ...
                Type = "H", Free = ["a" "b"], Params = struct(a = 1.46, ...
                    b = 0.8, c = 1, d = 0.1), Omega = 1.54, ...
                Bounds = [0.5 1.4], Hmax = 0.05, Direction = 1, ...
                TolX = 1e-12, TolF = 1e-12, MaxNewton = 20);
        end

        function verifyValues(testCase, have, want)
            % Every field of WANT is in HAVE, within 1e-12 relative.
            for name = string(fieldnames(want)).'
                testCase.assertTrue(isfield(have, name), ...
                    "no field " + name + " in Data.values");
                testCase.verifyEqual(have.(name), want.(name), ...
                    "field " + name, RelTol = 1e-12, AbsTol = 1e-14);
            end
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

        % ---- Slice 2: the Hopf curve of a delay equilibrium ----

        function hutchinsonHopfCurveStartsAtTheHopfPoint(testCase)
            m = testCase.model(testCase.Hutch);
            eq = testCase.hutchinson(m);
            h = eq.points("H");
            testCase.assertNumElements(h, 1);
            c = phasecont.continuation(m, h, Type = "H", ...
                Free = ["r" "tau"], Bounds = [0.5 3], Hmax = 0.05, ...
                Direction = 1);
            testCase.verifyEqual(c.Type, "H");
            testCase.verifyEqual(c.Free, ["r" "tau"]);
            testCase.verifyEqual(c.FreeIndex, [1 2]);
            testCase.verifyEqual(c.Stop, "Bounds");
            testCase.verifyEmpty(c.Warnings);
            n = size(c.Alpha, 1);
            testCase.assertGreaterThan(n, 5);
            testCase.verifySize(c.Alpha, [n 2]);
            testCase.verifyEqual(c.Alpha(:, 1) .* c.Alpha(:, 2), ...
                (pi/2) * ones(n, 1), AbsTol = 1e-8);
            testCase.verifyEqual(c.U, ones(n, 1), AbsTol = 1e-8);
            testCase.verifyEqual(c.Kappa, c.Alpha(:, 1).^2, AbsTol = 1e-8);
            testCase.verifyEqual(c.Alpha(end, 2), 3, AbsTol = 1e-8);
        end

        function aDelayHopfBranchHasTheCurveLayout(testCase)
            m = testCase.model(testCase.Hutch);
            eq = testCase.hutchinson(m);
            c = phasecont.continuation(m, eq.points("H"), Type = "H", ...
                Free = ["r" "tau"], Bounds = [0.5 3], Hmax = 0.05);
            n = size(c.Alpha, 1);
            testCase.verifySize(c.Eigenvalues, [n 0]);
            testCase.verifyEqual(c.Stable, false(n, 1));
            testCase.verifyEmpty(c.RootCount);
            testCase.verifySize(c.Tests, [n 0]);
            testCase.verifySize(c.Codim2, [n 2]);
            testCase.verifyEqual(c.Codim2Labels, ["GH" "HH"]);
            testCase.verifyTrue(all(isnan(c.Codim2), "all"));
            testCase.verifySize(c.Kappa, [n 1]);
            testCase.verifyTrue(all(isnan(c.StepSize)));
            testCase.verifyEmpty(c.Specials);
        end

        function wrightCurveLabelsTheBautinPointWithLyapunov(testCase)
            c = testCase.wrightCurve(true);
            testCase.verifyEqual(c.Type, "H");
            testCase.verifyEqual(c.Codim2Labels, ["GH" "HH"]);
            gh = c.points("GH");
            testCase.assertNumElements(gh, 1);
            slope = 6 / (pi * (1 + pi^2/4));
            testCase.verifyEqual(gh.P(2), -testCase.L1AtZero / slope, AbsTol = 1e-6);
            testCase.verifyEqual(gh.P(1), pi/2, AbsTol = 1e-8);
            testCase.verifyEqual(gh.Data.normalization, ...
                "|q|=1, p Delta'(i omega) q=1");
            testCase.verifyTrue(isfinite(gh.Data.coefficient));
            testCase.verifyEqual(gh.Data.omega, pi/2, AbsTol = 1e-8);
            testCase.verifySize(gh.Eigenvalues, [0 1]);
            want = testCase.L1AtZero + slope * c.Alpha(:, 2);
            have = c.Codim2(:, 1);
            ok = isfinite(have);
            testCase.verifyTrue(nnz(ok) > 5);
            testCase.verifyEqual(have(ok), want(ok), AbsTol = 1e-7);
            testCase.verifyTrue(all(isnan(c.Codim2(:, 2))));
        end

        function wrightCurveHasNoLyapunovWithoutTheOption(testCase)
            c = testCase.wrightCurve(false);
            testCase.verifyTrue(all(isnan(c.Codim2(:, 1))));
            testCase.verifyEmpty(c.points("GH"));
            d = phasecont.continuation(testCase.model(testCase.Wright), ...
                0.001, Type = "H", Free = ["a" "b"], ...
                Params = struct(a = 1.57, b = 0), Omega = 1.57, ...
                Bounds = [-1 1], Hmax = 0.05, Direction = 1, TolX = 1e-12, TolF = 1e-12, ...
                MaxNewton = 20);
            testCase.verifyEqual(d.Alpha, c.Alpha);
            testCase.verifyEqual(d.Codim2, c.Codim2);
        end

        function coupledCurveLabelsTheHopfHopfPointWithRoots(testCase)
            c = testCase.coupledCurve({"Roots", struct(N = 30, ReMin = -3)});
            hh = c.points("HH");
            testCase.assertNumElements(hh, 1);
            testCase.verifyEqual(hh.P(1:2), ...
                [testCase.AStar; testCase.BStar], AbsTol = 1e-6);
            testCase.verifyEqual(hh.Data.omega, testCase.Omega1Star, ...
                AbsTol = 1e-6);
            testCase.verifyEqual(hh.Data.values.omega2, ...
                testCase.Omega2Star, AbsTol = 1e-6);
            testCase.verifyTrue(any(isfinite(c.Codim2(:, 2))));
            testCase.verifyTrue(all(isnan(c.Codim2(:, 1))));
        end

        function coupledCurveHasNoHopfHopfWithoutRoots(testCase)
            c = testCase.coupledCurve({});
            testCase.verifyEmpty(c.points("HH"));
            testCase.verifyTrue(all(isnan(c.Codim2(:, 2))));
        end

        function aDelayHopfBranchSurvivesASessionRoundTrip(testCase)
            m = testCase.model(testCase.Wright);
            c = phasecont.continuation(m, 0.001, Type = "H", ...
                Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                Omega = 1.57, Lyapunov = true, Bounds = [-1 1], ...
                Hmax = 0.05, Direction = 1, TolX = 1e-12, TolF = 1e-12, ...
                MaxNewton = 20);
            file = fullfile(testCase.Dir, "hopf.pcs");
            save(phasecont.Session(m, {c}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.assertNumElements(back.Items, 1);
            b = back.Items{1};
            testCase.verifyEqual(b.Type, "H");
            testCase.verifyEqual(b.Free, c.Free);
            testCase.verifyEqual(b.Alpha, c.Alpha);
            testCase.verifyEqual(b.U, c.U);
            testCase.verifyEqual(b.Kappa, c.Kappa);
            testCase.verifyEqual(b.Codim2, c.Codim2);
            testCase.verifyEqual(b.Codim2Labels, ["GH" "HH"]);
            testCase.verifySize(b.Eigenvalues, [numel(c.Kappa) 0]);
            testCase.verifyEqual(b.Stable, c.Stable);
            testCase.verifyEqual(b.Stop, c.Stop);
            gh = b.points("GH");
            testCase.assertNumElements(gh, 1);
            testCase.verifyEqual(gh.P, c.points("GH").P);
            testCase.verifyEqual(gh.Data.coefficient, ...
                c.points("GH").Data.coefficient);
            testCase.verifyEqual(gh.Data.normalization, ...
                "|q|=1, p Delta'(i omega) q=1");
        end

        function aNumericHopfStartNeedsOmega(testCase)
            m = testCase.model(testCase.Wright);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0.001, Type = "H", ...
                    Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                    Bounds = [-1 1]), ...
                "phasecont:model:badOption", "Omega");
        end

        function omegaWithAPointStartIsRefused(testCase)
            m = testCase.model(testCase.Hutch);
            eq = testCase.hutchinson(m);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, eq.points("H"), Type = "H", ...
                    Free = ["r" "tau"], Omega = pi/2, Bounds = [0.5 3]), ...
                "phasecont:model:badOption", "Omega");
        end

        function aHopfRunFromANonHopfPointIsRefused(testCase)
            m = testCase.model(testCase.Hutch);
            pt = phasecont.Point(U = 1, P = [1.3; 1], Label = "EP", Model = m);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, pt, Type = "H", ...
                    Free = ["r" "tau"], Bounds = [0.5 3]), ...
                "phasecont:model:badStart", "H");
        end

        function badOmegaAndLyapunovValuesAreRefused(testCase)
            m = testCase.model(testCase.Wright);
            base = {"Type", "H", "Free", ["a" "b"], "Params", ...
                struct(a = 1.57, b = 0), "Bounds", [-1 1]};
            for omega = {[1 2], "1.57", NaN, Inf, []}
                try
                    phasecont.continuation(m, 0.001, base{:}, Omega = omega{1});
                    testCase.verifyFail("accepted Omega " + string(omega{1}));
                catch err
                    testCase.verifySubstring(string(err.identifier), "phasecont:");
                end
            end
            % A frequency below the engine floor is refused by the engine.
            try
                phasecont.continuation(m, 0.001, base{:}, Omega = 0);
                testCase.verifyFail("accepted Omega 0");
            catch err
                testCase.verifySubstring(string(err.identifier), "phasecont:");
                testCase.verifySubstring(string(err.message), "omega");
            end
            for flag = {1, "yes", [true false], []}
                try
                    phasecont.continuation(m, 0.001, base{:}, Omega = 1.57, ...
                        Lyapunov = flag{1});
                    testCase.verifyFail("accepted Lyapunov");
                catch err
                    testCase.verifySubstring(string(err.identifier), "phasecont:");
                end
            end
        end

        function theOptionsOfADelayHopfRunThatDoNotExistAreRefused(testCase)
            m = testCase.model(testCase.Wright);
            id = "phasecont:model:badOption";
            base = {"Type", "H", "Free", ["a" "b"], "Params", ...
                struct(a = 1.57, b = 0), "Omega", 1.57, "Bounds", [-1 1]};
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0.001, base{:}, MeshAdapt = 2), ...
                id, "MeshAdapt");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0.001, base{:}, Switch = "BP1"), ...
                id, "Switch");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0.001, base{:}, Detect = "H"), ...
                id, "Detect");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0.001, base{:}, ...
                    Type = "LP"), id, "Type");
        end

        function omegaLyapunovAndRootsKeepTheirPlace(testCase)
            % Omega and Lyapunov belong to a delay model with Type "H";
            % Roots stay refused for an ODE model.
            ode = testCase.model(testCase.Decay);
            id = "phasecont:model:badOption";
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", ...
                    Omega = 1, Bounds = [1 3]), id, "Omega");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", ...
                    Lyapunov = true, Bounds = [1 3]), id, "Lyapunov");
            m = testCase.model(testCase.Hutch);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, Free = "r", ...
                    Roots = testCase.Roots, Omega = 1, Bounds = [1.2 2]), ...
                id, "Omega");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, Free = "r", ...
                    Roots = testCase.Roots, Lyapunov = true, ...
                    Bounds = [1.2 2]), id, "Lyapunov");
        end

        function zeroHopfCoefficientsArriveAtTheZhSpecial(testCase)
            c = testCase.zhCurve({"Coefficients", true});
            zh = c.points("ZH");
            testCase.assertNumElements(zh, 1);
            testCase.verifyValues(zh.Data.values, testCase.ZhValues);
            v = zh.Data.values;
            % The closed form of SPEC 8i test 7.
            testCase.verifyEqual(v.g200_re, -1, AbsTol = 1e-6);
            testCase.verifyEqual(complex(v.g110_re, v.g110_im), ...
                -0.288400439142001 + 0.4530183504502902i, AbsTol = 1e-6);
            testCase.verifyLessThanOrEqual(abs(v.g011_re), 1e-6);
            testCase.verifyLessThanOrEqual(abs(v.s_product), 1e-6);
            % The residual fields say how well the located point fits.
            testCase.verifyLessThanOrEqual(v.null_residual_zero, 1e-12);
            testCase.verifyLessThanOrEqual(v.null_residual_hopf, 1e-12);
            testCase.verifyLessThanOrEqual(v.equilibrium_residual, 1e-12);
            testCase.verifyEmpty(c.Warnings);
            testCase.verifyEqual(zh.Label, "ZH");
        end

        function theZhSpecialHasNoValuesWithoutTheOption(testCase)
            off = testCase.zhCurve({});
            zh = off.points("ZH");
            testCase.assertNumElements(zh, 1);
            testCase.verifyTrue(isstruct(zh.Data.values));
            testCase.verifyEmpty(fieldnames(zh.Data.values));
            % The option does not move a point or a special.
            on = testCase.zhCurve({"Coefficients", true});
            testCase.verifyEqual(on.Alpha, off.Alpha);
            testCase.verifyEqual(on.U, off.U);
            testCase.verifyEqual(on.Kappa, off.Kappa);
            testCase.verifyEqual(on.Codim2, off.Codim2);
            testCase.verifyEqual(on.points("ZH").P, zh.P);
            testCase.verifyEqual(on.points("ZH").Index, zh.Index);
            % An explicit false is the default.
            f = testCase.zhCurve({"Coefficients", false});
            testCase.verifyEmpty(fieldnames(f.points("ZH").Data.values));
            testCase.verifyEqual(f.Alpha, off.Alpha);
        end

        function hopfHopfCoefficientsArriveWithOmega2(testCase)
            c = testCase.coupledNlCurve({"Roots", struct(N = 30, ReMin = -3), ...
                "Coefficients", true});
            hh = c.points("HH");
            testCase.assertNumElements(hh, 1);
            testCase.verifyValues(hh.Data.values, testCase.HhValues);
            testCase.verifyEqual(hh.Data.values.omega2, ...
                testCase.Omega2Star, AbsTol = 1e-6);
            testCase.verifyEqual(hh.Data.values.omega_large, ...
                testCase.Omega1Star, AbsTol = 1e-6);
            testCase.verifyEqual(hh.Data.values.omega_small, ...
                testCase.Omega2Star, AbsTol = 1e-6);
            testCase.verifyLessThanOrEqual(hh.Data.values.null_residual_1, 1e-12);
            testCase.verifyLessThanOrEqual(hh.Data.values.null_residual_2, 1e-12);
            testCase.verifyEmpty(c.Warnings);
        end

        function coefficientsWithoutRootsGiveTheZhValuesOnly(testCase)
            % HH is labeled only with Roots (slice 2), so the HH values
            % need Roots too; the option alone is no error.
            c = testCase.coupledNlCurve({"Coefficients", true});
            testCase.verifyEmpty(c.points("HH"));
            testCase.verifyEmpty(c.Warnings);
        end

        function coefficientsDoNotMoveTheHopfHopfRows(testCase)
            roots = {"Roots", struct(N = 30, ReMin = -3)};
            off = testCase.coupledNlCurve(roots);
            on = testCase.coupledNlCurve([roots, {"Coefficients", true}]);
            testCase.verifyEqual(on.Alpha, off.Alpha);
            testCase.verifyEqual(on.U, off.U);
            testCase.verifyEqual(on.Codim2, off.Codim2);
            testCase.verifyEqual(on.points("HH").P, off.points("HH").P);
            testCase.verifyEqual(on.points("HH").Data.values.omega2, ...
                off.points("HH").Data.values.omega2);
        end

        function aBranchWithCoefficientsSurvivesASessionRoundTrip(testCase)
            m = testCase.model(testCase.Zh);
            c = phasecont.continuation(m, [0.001 -0.249], Type = "H", ...
                Free = ["a" "tau"], Params = struct(a = 0.07, ...
                    tau = 1.4071360574511174, b = 1), Omega = 0.98, ...
                Coefficients = true, Bounds = [1.3 2.0], Hmax = 0.05, ...
                Direction = 1, TolX = 1e-12, TolF = 1e-12, MaxNewton = 20);
            file = fullfile(testCase.Dir, "zh.pcs");
            save(phasecont.Session(m, {c}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            b = back.Items{1};
            zh = b.points("ZH");
            testCase.assertNumElements(zh, 1);
            testCase.verifyEqual(zh.Data.values, c.points("ZH").Data.values);
            testCase.verifyValues(zh.Data.values, testCase.ZhValues);
            testCase.verifyEqual(b.Alpha, c.Alpha);
        end

        function badCoefficientsValuesAreRefused(testCase)
            m = testCase.model(testCase.Zh);
            base = {"Type", "H", "Free", ["a" "tau"], "Params", ...
                struct(a = 0.07, tau = 1.4071360574511174, b = 1), ...
                "Omega", 0.98, "Bounds", [1.3 2.0]};
            for flag = {1, "yes", [true false], [], NaN}
                try
                    phasecont.continuation(m, [0.001 -0.249], base{:}, ...
                        Coefficients = flag{1});
                    testCase.verifyFail("accepted Coefficients");
                catch err
                    testCase.verifySubstring(string(err.identifier), "phasecont:");
                    testCase.verifySubstring(string(err.message), "Coefficients");
                end
            end
        end

        function coefficientsKeepTheirPlace(testCase)
            % The option belongs to a delay model with Type "H".
            id = "phasecont:model:badOption";
            ode = testCase.model(testCase.Decay);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", ...
                    Coefficients = true, Bounds = [1 3]), id, "Coefficients");
            m = testCase.model(testCase.Hutch);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 1, Free = "r", ...
                    Roots = testCase.Roots, Coefficients = true, ...
                    Bounds = [1.2 2]), id, "Coefficients");
            % Another Type is refused too (the engine has no delay LP run).
            w = testCase.model(testCase.Wright);
            try
                phasecont.continuation(w, 0.001, Type = "LP", ...
                    Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                    Coefficients = true, Bounds = [-1 1]);
                testCase.verifyFail("accepted Coefficients with Type LP");
            catch err
                testCase.verifySubstring(string(err.identifier), "phasecont:");
            end
        end
    end
end
