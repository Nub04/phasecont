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
        LyapunovNotice = "Lyapunov is not set, so this Hopf curve has no GH points " + ...
            "and the l1 column of Codim2 is NaN. Pass Lyapunov=true to compute them."
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
        % SPEC 8s slice 4: the models and numbers of SPEC 8n and 8o. OSC has
        % the Hopf point (tau, omega) below at mu = 0.3 and cycles of period
        % 2 pi at tau = pi/2. The LPC model has cycles a = (r^2 - 1)^2.
        Osc = "par mu=0.3, tau=1.3272372818926783" + newline + ...
            "x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)" + newline + ...
            "done" + newline
        OscTauH = 1.3272372818926783
        OscOmegaH = 0.9539392014169457
        LpcModel = "par a=0.25, tau=1.5707963267948966" + newline + ...
            "x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)" + newline + ...
            "done" + newline
        % SPEC 8s slice 5: OSC with a decoupled z' = (tau - m0) z - z^3. The
        % cycles of OSC are cycles of the pair with z = 0, their z multiplier
        % is exp((tau - m0) T), and a pitchfork at tau = m0 gives the branch
        % z^2 = tau - m0.
        Oscz = "par mu=0.3, tau=1.3272372818926783, m0=1.45" + newline + ...
            "x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)" + newline + ...
            "z'=(tau-m0)*z-z^3" + newline + "done" + newline
        OsczM0 = 1.45
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

        function br = oscCycle(testCase, extra)
            % The OSC cycle run of SPEC 8s slice 4, test 1, from a numeric
            % start at the Hopf point.
            m = testCase.model(testCase.Osc);
            br = phasecont.continuation(m, 0, extra{:}, Type = "LC", ...
                Free = "tau", Params = struct(mu = 0.3, tau = testCase.OscTauH), ...
                Omega = testCase.OscOmegaH, NTST = 20, NCOL = 4, Eps = 0.05, ...
                MaxNewton = 30, Hmax = 0.5, MaxSteps = 500, Direction = 1, ...
                Bounds = [1 pi/2]);
        end

        function br = lpcCycle(testCase)
            % The cycle run of SPEC 8s slice 4, test 2, with 4 multipliers.
            m = testCase.model(testCase.LpcModel);
            br = phasecont.continuation(m, 0, Type = "LC", Free = "a", ...
                Params = struct(a = 1, tau = pi/2), Omega = 1, NTST = 20, ...
                NCOL = 4, Eps = 0.05, Floquet = struct(NHist = 12, Count = 4), ...
                MaxNewton = 30, Hmax = 0.1, MaxSteps = 500, Direction = -1, ...
                Bounds = [-0.5 1.1]);
        end

        function br = osczRun(testCase, m, extra)
            % The OSCZ cycle run of SPEC 8s slice 5: a Hopf start at the
            % origin, tau from the Hopf value up to pi/2. EXTRA is a cell of
            % name-value pairs (Detect, Switch, MeshAdapt, Floquet, ...).
            br = phasecont.continuation(m, [0 0], extra{:}, Type = "LC", ...
                Free = "tau", Params = struct(mu = 0.3, ...
                    tau = testCase.OscTauH, m0 = testCase.OsczM0), ...
                Omega = testCase.OscOmegaH, NTST = 20, NCOL = 4, Eps = 0.05, ...
                MaxNewton = 30, Hmax = 0.05, MaxSteps = 400, Direction = 1, ...
                Bounds = [1 pi/2]);
        end

        function args = oscArgs(testCase)
            % Valid arguments of an OSC cycle run, for the error tests.
            args = {"Type", "LC", "Free", "tau", "Params", ...
                struct(mu = 0.3, tau = testCase.OscTauH), "Omega", ...
                testCase.OscOmegaH, "NTST", 20, "NCOL", 4, "Eps", 0.05};
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
            testCase.verifyEqual(c.Warnings, testCase.LyapunovNotice);
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

        function aDelayHopfCurveWithoutLyapunovSaysSo(testCase)
            m = testCase.model(testCase.Wright);
            d = phasecont.continuation(m, 0.001, Type = "H", ...
                Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                Omega = 1.57, Bounds = [-1 1], Hmax = 0.05, Direction = 1, ...
                TolX = 1e-12, TolF = 1e-12, MaxNewton = 20);
            testCase.verifyEqual(d.Warnings, testCase.LyapunovNotice);
            c = testCase.wrightCurve(false);
            testCase.verifyEqual(d.Alpha, c.Alpha);
            testCase.verifyEqual(d.Codim2, c.Codim2);
            % A caller that passes the option has chosen, so there is no notice.
            testCase.verifyEmpty(c.Warnings);
            testCase.verifyEmpty(testCase.wrightCurve(true).Warnings);
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
            testCase.verifyEqual(c.Warnings, testCase.LyapunovNotice);
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
            testCase.verifyEqual(c.Warnings, testCase.LyapunovNotice);
        end

        function coefficientsWithoutRootsGiveTheZhValuesOnly(testCase)
            % HH is labeled only with Roots (slice 2), so the HH values
            % need Roots too; the option alone is no error.
            c = testCase.coupledNlCurve({"Coefficients", true});
            testCase.verifyEmpty(c.points("HH"));
            testCase.verifyEqual(c.Warnings, testCase.LyapunovNotice);
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

        function oscCycleFromANumericStartHasTheEngineNumbers(testCase)
            br = testCase.oscCycle({});
            testCase.verifyEqual(br.Type, "LC");
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.verifyEmpty(br.Warnings);
            testCase.assertSize(br.Alpha, [25 1]);
            testCase.verifyEqual(br.Alpha(1), 1.3276067784641747, AbsTol = 1e-9);
            testCase.verifyEqual(br.Alpha(end), pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(br.Period(end), 2*pi, AbsTol = 5e-11);
            testCase.verifySize(br.U, [25 81]);
            testCase.verifyEqual(max(br.U(end, :)), 1, AbsTol = 1e-12);
            testCase.verifyEqual(min(br.U(end, :)), -1, AbsTol = 1e-12);
            amp = max(br.U, [], 2);
            testCase.verifyGreaterThan(min(diff(amp)), 0);
            testCase.verifyEmpty(br.Specials);
        end

        function aDelayCycleBranchHasTheCycleLayout(testCase)
            br = testCase.oscCycle({});
            npts = numel(br.Alpha);
            testCase.verifyEqual(br.Free, "tau");
            testCase.verifyEqual(br.NTST, 20);
            testCase.verifyEqual(br.NCOL, 4);
            testCase.verifySize(br.Period, [npts 1]);
            testCase.verifyEqual(br.Mesh, repmat(linspace(0, 1, 21), npts, 1), ...
                AbsTol = 1e-15);
            testCase.verifySize(br.Eigenvalues, [npts 0]);
            testCase.verifyEqual(br.Stable, false(npts, 1));
            testCase.verifySize(br.Tests, [npts 0]);
            testCase.verifySize(br.Codim2, [npts 0]);
            testCase.verifyEmpty(br.RootCount);
            testCase.verifyEqual(br.P, [0.3; testCase.OscTauH]);
            testCase.verifyEqual(br.StepSize(1), 0);
            testCase.verifyEqual(br.NewtonIters(1), 0);
        end

        function aDelayCycleBranchMatchesTheEngineRows(testCase)
            % The branch holds the rows of the engine call as the layout of
            % SPEC 8s slice 4 puts them.
            m = testCase.model(testCase.Osc);
            problem = "{""type"":""LC"",""free"":""tau"",""ntst"":20," + ...
                """ncol"":4,""omega"":0.9539392014169457,""eps"":0.05," + ...
                """options"":{""max_newton"":30,""hmax"":0.5,""max_steps"":500," + ...
                """direction"":1,""bounds"":[1,1.5707963267948966]}}";
            out = phasecont.internal.runLoop(m, problem, 0, [0.3; testCase.OscTauH]);
            rows = out.Points{1};
            testCase.assertSize(rows, [86 25]);
            br = testCase.oscCycle({});
            testCase.verifyEqual(br.Alpha, rows(1, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.U, rows(2:82, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.Period, rows(83, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.StepSize, rows(84, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.NewtonIters, rows(85, :).', RelTol = 1e-12);
            testCase.verifyTrue(all(isnan(rows(86, :))));
        end

        function aDelayCycleBranchMatchesTheEngineRowsWithMultipliers(testCase)
            % The same comparison for the run with four multipliers: the
            % columns after h and iters are Re mu, Im mu and trivial_error.
            m = testCase.model(testCase.LpcModel);
            problem = "{""type"":""LC"",""free"":""a"",""ntst"":20," + ...
                """ncol"":4,""omega"":1,""eps"":0.05," + ...
                """floquet"":{""nhist"":12,""count"":4}," + ...
                """options"":{""max_newton"":30,""hmax"":0.1,""max_steps"":500," + ...
                """direction"":-1,""bounds"":[-0.5,1.1]}}";
            out = phasecont.internal.runLoop(m, problem, 0, [1; pi/2]);
            rows = out.Points{1};
            testCase.assertSize(rows, [94 99]);
            br = testCase.lpcCycle();
            testCase.verifyEqual(br.Alpha, rows(1, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.U, rows(2:82, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.Period, rows(83, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.StepSize, rows(84, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.NewtonIters, rows(85, :).', RelTol = 1e-12);
            testCase.verifyEqual(real(br.Eigenvalues), rows(86:89, :).', ...
                RelTol = 1e-12, AbsTol = 1e-14);
            testCase.verifyEqual(imag(br.Eigenvalues), rows(90:93, :).', ...
                RelTol = 1e-12, AbsTol = 1e-14);
            % The trivial multiplier error is kept in the rows and is small.
            testCase.verifyLessThan(max(abs(rows(94, :))), 1e-6);
        end

        function aDelayCycleStartsFromAHopfPointOfAnEquilibriumBranch(testCase)
            % The H point of a slice 1 branch supplies the equilibrium, the
            % parameters and the frequency. Measured on 2026-11-09: the H
            % point is within 2.2e-14 of the exact tau and omega, and Alpha(1)
            % differs by 5.6e-10 from the value of the exact start (the first
            % orbit is a first-order correction at Eps), so the bound is 2e-9.
            m = testCase.model(testCase.Osc);
            eq = phasecont.continuation(m, 0, Free = "tau", ...
                Params = struct(mu = 0.3, tau = 1.2), ...
                Roots = struct(N = 30, ReMin = -0.5), Bounds = [1.2 1.5], ...
                Direction = 1, Hmax = 0.05);
            h = eq.points("H");
            testCase.assertNumElements(h, 1);
            br = phasecont.continuation(m, h, Type = "LC", Free = "tau", ...
                NTST = 20, NCOL = 4, Eps = 0.05, MaxNewton = 30, Hmax = 0.5, ...
                MaxSteps = 500, Direction = 1, Bounds = [1 pi/2]);
            testCase.verifyEqual(br.Type, "LC");
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.assertSize(br.Alpha, [25 1]);
            testCase.verifyEqual(br.Alpha(1), 1.3276067784641747, AbsTol = 2e-9);
            testCase.verifyEqual(br.Alpha(end), pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(br.Period(end), 2*pi, AbsTol = 5e-11);
            testCase.verifyEqual(max(br.U(end, :)), 1, AbsTol = 1e-12);
        end

        function aDelayCycleHasItsOrbitThroughCycle(testCase)
            br = testCase.oscCycle({});
            npts = numel(br.Alpha);
            [t, x] = br.cycle(npts);
            testCase.verifySize(t, [81 1]);
            testCase.verifySize(x, [81 1]);
            testCase.verifyEqual(t(1), 0);
            testCase.verifyEqual(t(end), br.Period(npts), RelTol = 1e-14);
            testCase.verifyEqual(max(x), 1, AbsTol = 1e-12);
            testCase.verifyEqual(x(end), x(1), AbsTol = 1e-9);
            % The orbit is x = sin(t + phase): its mean square is 1/2.
            testCase.verifyEqual(trapz(t, x.^2) / t(end), 0.5, AbsTol = 6e-9);
        end

        function lpcModelCycleHasMultipliersAndTheLpcSpecial(testCase)
            br = testCase.lpcCycle();
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.assertSize(br.Alpha, [99 1]);
            testCase.verifySize(br.Eigenvalues, [99 4]);
            testCase.verifyTrue(all(isfinite(br.Eigenvalues), "all"));
            testCase.verifyEqual(br.Eigenvalues(1, 1), ...
                complex(real(br.Eigenvalues(1, 1)), 0), AbsTol = 1e-9);
            testCase.verifyGreaterThan(real(br.Eigenvalues(1, 1)), 1);
            testCase.verifyEqual(br.Period, 2*pi*ones(99, 1), AbsTol = 2e-10);
            r = max(br.U, [], 2);
            testCase.verifyEqual((r.^2 - 1).^2, br.Alpha, AbsTol = 5e-8);
            % Before the fold the amplitude is sqrt(1 - sqrt(alpha)), after
            % it sqrt(1 + sqrt(alpha)). The first and last points are far
            % from the fold.
            testCase.verifyEqual(r(1), sqrt(1 - sqrt(br.Alpha(1))), AbsTol = 1e-8);
            testCase.verifyEqual(r(end), sqrt(1 + sqrt(br.Alpha(end))), AbsTol = 1e-8);
            % The cycles are unstable on the way to the fold: the leading
            % multiplier is real and a little above 1.
            mu1 = br.Eigenvalues(1, 1);
            testCase.verifyEqual(real(mu1), 1.018, AbsTol = 0.005);
            lpc = br.points("LPC");
            testCase.assertNumElements(lpc, 1);
            testCase.verifyEqual(br.Specials, lpc);
            testCase.verifyEqual(lpc.Index, 68);
            testCase.verifyEqual(lpc.P(1), 0, AbsTol = 2e-11);
            testCase.verifyEqual(lpc.P(2), pi/2);
            testCase.verifyEqual(lpc.Data.period, 2*pi, AbsTol = 2e-10);
            testCase.verifyEqual(lpc.Data.ntst, 20);
            testCase.verifyEqual(lpc.Data.ncol, 4);
            testCase.verifySize(lpc.U, [81 1]);
            testCase.verifySize(lpc.Eigenvalues, [4 1]);
            testCase.verifySize(lpc.Data.mesh, [1 21]);
            % Alpha falls to the fold and then rises again.
            [~, k] = min(br.Alpha);
            testCase.verifyLessThanOrEqual(abs(k - lpc.Index), 2);
            testCase.verifyLessThan(max(diff(br.Alpha(1:k))), 0);
            testCase.verifyGreaterThan(min(diff(br.Alpha(k:end))), 0);
        end

        function aDelayCycleBranchSurvivesASessionRoundTrip(testCase)
            m = testCase.model(testCase.LpcModel);
            c = phasecont.continuation(m, 0, Type = "LC", Free = "a", ...
                Params = struct(a = 1, tau = pi/2), Omega = 1, NTST = 20, ...
                NCOL = 4, Eps = 0.05, Floquet = struct(NHist = 12, Count = 4), ...
                MaxNewton = 30, Hmax = 0.1, MaxSteps = 500, Direction = -1, ...
                Bounds = [-0.5 1.1]);
            file = fullfile(testCase.Dir, "cycle.pcs");
            save(phasecont.Session(m, {c}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.assertNumElements(back.Items, 1);
            b = back.Items{1};
            testCase.verifyEqual(b.Type, "LC");
            testCase.verifyEqual(b.Free, c.Free);
            testCase.verifyEqual(b.Alpha, c.Alpha);
            testCase.verifyEqual(b.U, c.U);
            testCase.verifyEqual(b.Period, c.Period);
            testCase.verifyEqual(b.NTST, c.NTST);
            testCase.verifyEqual(b.NCOL, c.NCOL);
            testCase.verifyEqual(b.Mesh, c.Mesh);
            testCase.verifyEqual(b.Eigenvalues, c.Eigenvalues);
            testCase.verifyEqual(b.Stable, c.Stable);
            testCase.verifyEqual(b.Stop, c.Stop);
            testCase.verifyEqual(b.points("LPC").Index, c.points("LPC").Index);
            testCase.verifyEqual(b.points("LPC").P, c.points("LPC").P);
            testCase.verifyEqual(b.points("LPC").Data.period, ...
                c.points("LPC").Data.period);
        end

        function aDelayCycleRunNeedsItsOwnOptions(testCase)
            m = testCase.model(testCase.Osc);
            id = "phasecont:model:badOption";
            args = testCase.oscArgs();
            keys = ["NTST" "NCOL" "Eps"];
            for k = keys
                at = find(string(args(1:2:end)) == k) * 2 - 1;
                rest = args;
                rest(at:at+1) = [];
                testCase.verifyErrorText( ...
                    @() phasecont.continuation(m, 0, rest{:}), id, k);
            end
        end

        function badEpsValuesAreRefused(testCase)
            m = testCase.model(testCase.Osc);
            args = testCase.oscArgs();
            args(end-1:end) = [];
            for eps = {0, -0.05, NaN, Inf, "0.05", [0.05 0.1], [], 1 + 2i, true}
                try
                    phasecont.continuation(m, 0, args{:}, Eps = eps{1});
                    testCase.verifyFail("accepted Eps " + string(class(eps{1})));
                catch err
                    testCase.verifyEqual(string(err.identifier), ...
                        "phasecont:model:badOption");
                    testCase.verifySubstring(string(err.message), "Eps");
                end
            end
        end

        function badFloquetValuesAreRefused(testCase)
            m = testCase.model(testCase.Osc);
            args = testCase.oscArgs();
            bad = {1, "x", [], struct(NHist = 12), struct(Count = 4), ...
                struct(NHist = 0, Count = 1), struct(NHist = 12, Count = 0), ...
                struct(NHist = 1.5, Count = 1), struct(NHist = 12, Count = 1.5), ...
                struct(NHist = [1 2], Count = 1), ...
                struct(NHist = 12, Count = 4, Extra = 1), ...
                struct(NHist = "12", Count = 4), ...
                repmat(struct(NHist = 12, Count = 4), 1, 2)};
            for f = bad
                try
                    phasecont.continuation(m, 0, args{:}, Floquet = f{1});
                    testCase.verifyFail("accepted a bad Floquet");
                catch err
                    testCase.verifyEqual(string(err.identifier), ...
                        "phasecont:model:badOption");
                    testCase.verifySubstring(string(err.message), "Floquet");
                end
            end
        end

        function omegaRulesOfADelayCycleRun(testCase)
            id = "phasecont:model:badOption";
            % A numeric start needs Omega.
            m = testCase.model(testCase.Osc);
            args = testCase.oscArgs();
            at = find(string(args(1:2:end)) == "Omega") * 2 - 1;
            rest = args;
            rest(at:at+1) = [];
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, rest{:}), id, "Omega");
            % A Point start supplies it, so giving it is an error.
            h = testCase.model(testCase.Hutch);
            eq = testCase.hutchinson(h);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, eq.points("H"), Type = "LC", ...
                    Free = "r", NTST = 20, NCOL = 4, Eps = 0.05, ...
                    Omega = pi/2), id, "Omega");
            % A bad value.
            for omega = {[1 2], "1", NaN, Inf, [], 1 + 2i}
                try
                    phasecont.continuation(m, 0, args{:}, Omega = omega{1});
                    testCase.verifyFail("accepted Omega " + class(omega{1}));
                catch err
                    testCase.verifyEqual(string(err.identifier), id);
                    testCase.verifySubstring(string(err.message), "Omega");
                end
            end
            % A Point that is not an H point, even when it carries a
            % frequency: the label decides.
            pt = phasecont.Point(U = 1, P = [1.3; 1], Label = "EP", Model = h, ...
                Data = struct(omega = pi/2));
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, pt, Type = "LC", Free = "r", ...
                    NTST = 20, NCOL = 4, Eps = 0.05), ...
                "phasecont:model:badStart", "not from ""EP""");
            % An H point without a frequency in its data.
            bare = phasecont.Point(U = 1, P = [1.3; 1], Label = "H", Model = h);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, bare, Type = "LC", Free = "r", ...
                    NTST = 20, NCOL = 4, Eps = 0.05), ...
                "phasecont:model:badStart", "Data.omega");
        end

        function aFrequencyBelowTheEngineFloorIsRefusedForACycle(testCase)
            % The floor of 1e-8 is the engine's (SPEC 8s): zero and 1e-9
            % reach it through the JSON and come back as an engine error.
            m = testCase.model(testCase.Osc);
            args = testCase.oscArgs();
            at = find(string(args(1:2:end)) == "Omega") * 2 - 1;
            rest = args;
            rest(at:at+1) = [];
            for omega = [0 1e-9]
                try
                    phasecont.continuation(m, 0, rest{:}, Omega = omega);
                    testCase.verifyFail("accepted Omega " + string(omega));
                catch err
                    testCase.verifySubstring(string(err.identifier), "phasecont:");
                    testCase.verifySubstring(string(err.message), "omega");
                end
            end
        end

        function anOdeModelRefusesTheDelayCycleOptions(testCase)
            ode = testCase.model(testCase.Decay);
            id = "phasecont:model:badOption";
            common = {"Type", "LC", "Free", "a", "NTST", 20, "NCOL", 4};
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, common{:}, Eps = 0.05), ...
                id, "Eps");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, common{:}, ...
                    Floquet = struct(NHist = 12, Count = 4)), id, "Floquet");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, common{:}, Omega = 1), ...
                id, "Omega");
        end

        function optionsThatADelayCycleRunDoesNotHaveAreRefused(testCase)
            m = testCase.model(testCase.Osc);
            id = "phasecont:model:badOption";
            args = testCase.oscArgs();
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, args{:}, ...
                    Roots = struct(N = 30, ReMin = -0.5)), id, "Roots");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, args{:}, Lyapunov = true), ...
                id, "Lyapunov");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, args{:}, Coefficients = true), ...
                id, "Coefficients");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, args{:}, Detect = "LP"), ...
                id, "Detect");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, args{:}, Var = "x"), ...
                id, "Var");
            % Two free parameters are for the curves.
            rest = args;
            rest{4} = ["mu" "tau"];
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, 0, rest{:}), ...
                "phasecont:model:unknownParameter", "Free");
        end

        function epsAndFloquetBelongToADelayCycleRun(testCase)
            id = "phasecont:model:badOption";
            fl = struct(NHist = 12, Count = 4);
            ode = testCase.model(testCase.Decay);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", Eps = 0.05, ...
                    Bounds = [1 3]), id, "Eps");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", Floquet = fl, ...
                    Bounds = [1 3]), id, "Floquet");
            h = testCase.model(testCase.Hutch);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, 1, Free = "r", ...
                    Roots = testCase.Roots, Eps = 0.05, Bounds = [1.2 2]), ...
                id, "Eps");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, 1, Free = "r", ...
                    Roots = testCase.Roots, Floquet = fl, Bounds = [1.2 2]), ...
                id, "Floquet");
            w = testCase.model(testCase.Wright);
            base = {0.001, "Type", "H", "Free", ["a" "b"], "Params", ...
                struct(a = 1.57, b = 0), "Omega", 1.57, "Bounds", [-1 1]};
            testCase.verifyErrorText( ...
                @() phasecont.continuation(w, base{:}, Eps = 0.05), id, "Eps");
            testCase.verifyErrorText( ...
                @() phasecont.continuation(w, base{:}, Floquet = fl), ...
                id, "Floquet");
        end

        % ---- SPEC 8s slice 5: BPC, mesh adaptation and switching ----

        function oszBpcBranchMatchesTheEngineRowsAndHasTheBpc(testCase)
            m = testCase.model(testCase.Oscz);
            problem = "{""type"":""LC"",""free"":""tau"",""ntst"":20," + ...
                """ncol"":4,""omega"":0.9539392014169457,""eps"":0.05," + ...
                """bpc"":true,""options"":{""max_newton"":30,""hmax"":0.05," + ...
                """max_steps"":400,""direction"":1," + ...
                """bounds"":[1,1.5707963267948966]}}";
            out = phasecont.internal.runLoop(m, problem, [0; 0], ...
                [0.3; testCase.OscTauH; testCase.OsczM0]);
            rows = out.Points{1};
            testCase.assertSize(rows, [167 126]);
            br = testCase.osczRun(m, {"Detect", "BPC"});
            testCase.verifyEqual(br.Type, "LC");
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.assertSize(br.Alpha, [126 1]);
            testCase.verifyEqual(br.Alpha, rows(1, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.U, rows(2:163, :).', RelTol = 1e-12, ...
                AbsTol = 1e-14);
            testCase.verifyEqual(br.Period, rows(164, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.StepSize, rows(165, :).', RelTol = 1e-12);
            testCase.verifyEqual(br.NewtonIters, rows(166, :).', RelTol = 1e-12);
            testCase.verifyTrue(all(isnan(rows(167, :))));
            % One special, a BPC at the pitchfork tau = m0. The index of the
            % engine is 97 and the one of the Branch counts from 1.
            testCase.assertNumElements(br.Specials, 1);
            bpc = br.points("BPC");
            testCase.assertNumElements(bpc, 1);
            testCase.verifyEqual(bpc.Index, 98);
            testCase.verifyEqual(bpc.P(2), testCase.OsczM0, AbsTol = 1e-11);
            testCase.verifyEqual(bpc.Data.mesh, linspace(0, 1, 21), AbsTol = 1e-15);
            testCase.verifyEqual(bpc.Data.ntst, 20);
            testCase.verifyEqual(bpc.Data.period, out.Specials{1}(164, 1), ...
                RelTol = 1e-12);
            testCase.verifyEqual(bpc.Data.period, br.Period(bpc.Index), ...
                AbsTol = 5e-3);
            testCase.verifyEqual(bpc.U, out.Specials{1}(2:163, 1), ...
                RelTol = 1e-12, AbsTol = 1e-14);
            % The orbit is the cycle of OSC with z = 0.
            testCase.verifyEqual(br.Alpha(1), 1.3276067784641745, AbsTol = 1e-9);
            testCase.verifyEqual(br.Alpha(end), pi/2, AbsTol = 1e-9);
            testCase.verifyEqual(br.Period(end), 2*pi, AbsTol = 1e-10);
            x = br.U(end, 1:2:end);
            z = br.U(end, 2:2:end);
            testCase.verifyEqual(max(abs(x)), 1, AbsTol = 1e-11);
            testCase.verifyLessThanOrEqual(max(abs(z)), 1e-12);
            % Without Detect there is no BPC and the points are the same.
            plain = testCase.osczRun(m, {});
            testCase.verifyEmpty(plain.Specials);
            testCase.verifyEqual(plain.Alpha, br.Alpha);
            testCase.verifyEqual(plain.U, br.U);
        end

        function aBpcBranchWithMultipliersHasTheZMultiplier(testCase)
            % The z multiplier of the cycle with z = 0 is exp((tau - m0) T).
            m = testCase.model(testCase.Oscz);
            br = testCase.osczRun(m, {"Detect", "BPC", ...
                "Floquet", struct(NHist = 20, Count = 6)});
            testCase.assertSize(br.Alpha, [126 1]);
            testCase.verifySize(br.Eigenvalues, [126 6]);
            want = exp((br.Alpha - testCase.OsczM0) .* br.Period);
            nearest = min(abs(br.Eigenvalues - want), [], 2) ./ want;
            testCase.verifyLessThan(max(nearest), 1e-9);
            testCase.verifyEqual(br.points("BPC").Index, 98);
        end

        function switchingAtTheBpcGivesTwoBranchesWithThePitchforkInvariants(testCase)
            m = testCase.model(testCase.Oscz);
            fl = struct(NHist = 20, Count = 6);
            main = testCase.osczRun(m, {"Detect", "BPC", "Floquet", fl});
            zEnd = zeros(1, 2);
            for k = 1:2
                d = [1 -1];
                brs = testCase.osczRun(m, {"Detect", "BPC", "Floquet", fl, ...
                    "Switch", "BPC1", "SwitchDirection", d(k)});
                testCase.assertSize(brs, [1 2]);
                % The first branch is the run without the switch.
                testCase.verifyEqual(brs(1).Alpha, main.Alpha);
                testCase.verifyEqual(brs(1).U, main.U);
                testCase.verifyEqual(brs(1).points("BPC").Index, 98);
                sw = brs(2);
                testCase.verifyEqual(sw.Type, "LC");
                testCase.verifyEqual(sw.Free, "tau");
                testCase.verifyEqual(sw.NTST, 20);
                testCase.verifyEqual(sw.NCOL, 4);
                testCase.verifyEqual(sw.Stop, "Bounds");
                testCase.assertSize(sw.Alpha, [74 1]);
                testCase.verifyEmpty(sw.Specials);
                testCase.verifyEqual(sw.Alpha(1), testCase.OsczM0, AbsTol = 1e-9);
                testCase.verifyEqual(sw.Alpha(end), pi/2, AbsTol = 1e-9);
                testCase.verifyEqual(sw.Period(end), 2*pi, AbsTol = 1e-10);
                % z is constant along the orbit and z^2 = tau - m0.
                z = sw.U(:, 2:2:end);
                testCase.verifyLessThan(max(abs(z.^2 - (sw.Alpha - testCase.OsczM0)), ...
                    [], "all"), 1e-12);
                zEnd(k) = z(end, 1);
                testCase.verifyEqual(abs(zEnd(k)), 0.3475576596694392, AbsTol = 1e-9);
                % The z multiplier on the new branch is exp(-2 (tau - m0) T).
                want = exp(-2*(sw.Alpha - testCase.OsczM0) .* sw.Period);
                nearest = min(abs(sw.Eigenvalues - want), [], 2) ./ want;
                testCase.verifyLessThan(max(nearest), 1e-9);
            end
            testCase.verifyLessThan(zEnd(1)*zEnd(2), 0);
        end

        function meshAdaptationGivesAMeshPerPointAndPerSpecial(testCase)
            m = testCase.model(testCase.Oscz);
            br = testCase.osczRun(m, {"Detect", "BPC", "MeshAdapt", 3});
            testCase.assertSize(br.Alpha, [123 1]);
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.verifyEqual(br.Period(end), 2*pi, AbsTol = 1e-6);
            testCase.assertSize(br.Mesh, [123 21]);
            testCase.verifyEqual(br.Mesh(:, 1), zeros(123, 1));
            testCase.verifyEqual(br.Mesh(:, end), ones(123, 1));
            testCase.verifyGreaterThan(min(diff(br.Mesh, 1, 2), [], "all"), 0);
            uniform = repmat(linspace(0, 1, 21), 123, 1);
            moved = max(abs(br.Mesh - uniform), [], 2) > 1e-9;
            testCase.verifyTrue(any(moved));
            bpc = br.points("BPC");
            testCase.assertNumElements(bpc, 1);
            testCase.verifyEqual(bpc.Index, 96);
            testCase.verifyEqual(bpc.P(2), testCase.OsczM0, AbsTol = 1e-9);
            sm = bpc.Data.mesh;
            testCase.verifySize(sm, [1 21]);
            testCase.verifyEqual(sm([1 end]), [0 1]);
            testCase.verifyGreaterThan(min(diff(sm)), 0);
            testCase.verifyLessThan(max(abs(sm - br.Mesh(bpc.Index, :))), 0.05);
            % MeshAdapt = 0 is the run of slice 4: a uniform mesh.
            flat = testCase.osczRun(m, {"Detect", "BPC", "MeshAdapt", 0});
            testCase.verifyEqual(flat.Mesh, repmat(linspace(0, 1, 21), 126, 1), ...
                AbsTol = 1e-15);
        end

        function adaptedAndSwitchedBranchesSurviveASessionRoundTrip(testCase)
            m = testCase.model(testCase.Oscz);
            adapted = testCase.osczRun(m, {"Detect", "BPC", "MeshAdapt", 3});
            brs = testCase.osczRun(m, {"Detect", "BPC", "Switch", "BPC1", ...
                "SwitchDirection", -1});
            file = fullfile(testCase.Dir, "oscz.pcs");
            save(phasecont.Session(m, {adapted, brs(1), brs(2)}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.assertNumElements(back.Items, 3);
            want = {adapted, brs(1), brs(2)};
            for k = 1:3
                b = back.Items{k};
                testCase.verifyEqual(b.Type, "LC");
                testCase.verifyEqual(b.Alpha, want{k}.Alpha);
                testCase.verifyEqual(b.U, want{k}.U);
                testCase.verifyEqual(b.Period, want{k}.Period);
                testCase.verifyEqual(b.Mesh, want{k}.Mesh);
                testCase.verifyEqual(b.Stop, want{k}.Stop);
                testCase.verifyEqual(numel(b.Specials), numel(want{k}.Specials));
            end
            testCase.verifyEqual(back.Items{1}.points("BPC").Data.mesh, ...
                adapted.points("BPC").Data.mesh);
            testCase.verifyEqual(back.Items{2}.points("BPC").P, ...
                brs(1).points("BPC").P);
        end

        function theBpcSwitchAndMeshOptionsOfADelayCycleRunAreChecked(testCase)
            m = testCase.model(testCase.Oscz);
            id = "phasecont:model:badOption";
            run = @(varargin) phasecont.continuation(m, [0 0], varargin{:}, ...
                Type = "LC", Free = "tau", Params = struct(mu = 0.3, ...
                    tau = testCase.OscTauH, m0 = testCase.OsczM0), ...
                Omega = testCase.OscOmegaH, NTST = 20, NCOL = 4, Eps = 0.05);
            % A switch needs a direction, the BPC label and no adaptation.
            testCase.verifyErrorText( ...
                @() run(Detect = "BPC", Switch = "BPC1"), id, "SwitchDirection");
            testCase.verifyErrorText( ...
                @() run(Detect = "BPC", SwitchDirection = 1), id, "Switch");
            testCase.verifyErrorText( ...
                @() run(Switch = "BPC1", SwitchDirection = 1), id, "Detect");
            testCase.verifyErrorText( ...
                @() run(Detect = "BPC", Switch = "BPC1", SwitchDirection = 1, ...
                    MeshAdapt = 3), id, "MeshAdapt");
            for bad = {0, 2, -2, 1.5, "up", [1 -1], NaN}
                testCase.verifyErrorText( ...
                    @() run(Detect = "BPC", Switch = "BPC1", ...
                        SwitchDirection = bad{1}), id, "SwitchDirection");
            end
            % Detect takes BPC only on a delay cycle.
            testCase.verifyErrorText(@() run(Detect = "LPC"), id, "BPC");
            testCase.verifyErrorText(@() run(Detect = ["BPC" "LP"]), id, "BPC");
            testCase.verifyErrorText(@() run(Detect = "LP"), id, "BPC");
            % MeshAdapt is an integer of at least 0.
            for bad = {-1, 1.5, "a", [1 2], NaN, Inf}
                testCase.verifyErrorText( ...
                    @() run(MeshAdapt = bad{1}), id, "MeshAdapt");
            end
        end

        function aSwitchThatTheEngineCannotFollowRaisesAnEngineError(testCase)
            % There is no LPC and no second BPC on the OSCZ branch.
            m = testCase.model(testCase.Oscz);
            for name = ["LPC1" "BPC2"]
                try
                    testCase.osczRun(m, {"Detect", "BPC", "Switch", name, ...
                        "SwitchDirection", 1});
                    testCase.verifyFail("no error for the switch " + name);
                catch err
                    testCase.verifySubstring(string(err.identifier), "phasecont:");
                    testCase.verifyNotEmpty(err.message);
                end
            end
        end

        function theSwitchOptionsBelongToADelayCycleRun(testCase)
            id = "phasecont:model:badOption";
            h = testCase.model(testCase.Hutch);
            base = {"Free", "r", "Roots", testCase.Roots, "Bounds", [1.2 2]};
            testCase.verifyErrorText( ...
                @() phasecont.continuation(h, 1, base{:}, SwitchDirection = 1), ...
                id, "SwitchDirection");
            ode = testCase.model(testCase.Decay);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(ode, 3, Free = "a", Bounds = [1 3], ...
                    SwitchDirection = 1), id, "SwitchDirection");
            w = testCase.model(testCase.Wright);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(w, 0.001, Type = "H", ...
                    Free = ["a" "b"], Params = struct(a = 1.57, b = 0), ...
                    Omega = 1.57, Bounds = [-1 1], SwitchDirection = 1), ...
                id, "SwitchDirection");
        end
    end
end
