classdef ContinuationTest < matlab.unittest.TestCase
    % Tests phasecont.continuation: acceptance numbers, Branch fields, the
    % polling loop, interruption and start errors.

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=r*x-y-x*z" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        Fold = "par a=-1" + newline + "x'=a+x^2" + newline + ...
            "init x=1" + newline + "done" + newline
        HopfC = "par al=0.1,c=-1" + newline + ...
            "x'=al*x-y+c*x*(x^2+y^2)" + newline + ...
            "y'=x+al*y+c*y*(x^2+y^2)" + newline + ...
            "init x=0.1,y=0" + newline + "done" + newline
    end

    properties
        Dir
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
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Dir = t.Folder;
        end
    end

    methods
        function m = model(testCase, text, ev)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = ev);
        end

        function br = foldBranch(testCase, ev)
            m = testCase.model(testCase.Fold, ev);
            eq = phasecont.equilibrium(m, -1);
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 100, Bounds = [-5 5]);
        end
    end

    methods (Test)
        function test1FoldHasOneLpAtTheOrigin(testCase, Evaluator)
            br = testCase.foldBranch(Evaluator);
            lp = br.points("LP");
            testCase.assertSize(lp, [1 1]);
            fprintf("test1 [%s] LP x %.3e, a %.3e, coefficient error %.3e\n", ...
                Evaluator, lp.U, lp.P(br.FreeIndex), lp.Data.coefficient - 1);
            testCase.verifyEqual(lp.U, 0, AbsTol = 1e-10);
            testCase.verifyEqual(lp.P(br.FreeIndex), 0, AbsTol = 1e-10);
            testCase.verifyEqual(lp.Data.coefficient, 1, AbsTol = 1e-12);
        end

        function test2HopfNormalFormL1Is2c(testCase, Evaluator)
            for c = [1 -1]
                m = testCase.model(testCase.HopfC, Evaluator);
                eq = phasecont.equilibrium(m, [0 0], ...
                    Params = struct(al = -0.5, c = c));
                br = phasecont.continuation(m, eq, Free = "al", Direction = 1, ...
                    MaxSteps = 60, Bounds = [-1 1]);
                h = br.points("H");
                testCase.assertSize(h, [1 1]);
                testCase.verifySize(br.points("NSA"), [0 1]);
                fprintf("test2 [%s] c=%d al %.3e omega-1 %.3e l1-2c %.3e\n", ...
                    Evaluator, c, h.P(br.FreeIndex), h.Data.omega - 1, ...
                    h.Data.coefficient - 2*c);
                testCase.verifyEqual(h.P(br.FreeIndex), 0, AbsTol = 1e-10);
                testCase.verifyEqual(h.Data.omega, 1, AbsTol = 1e-10);
                testCase.verifyEqual(h.Data.coefficient, 2*c, AbsTol = 1e-10);
                testCase.verifyEqual(sign(h.Data.coefficient), sign(c));
            end
        end

        function test5LorenzPitchforkAndHopf(testCase, Evaluator)
            b = 8/3;
            m = testCase.model(testCase.Lorenz, Evaluator);
            eq = phasecont.equilibrium(m, [0 0 0], ...
                Params = struct(s = 10, r = 0.5, b = b));
            brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.5, MaxSteps = 1000, Bounds = [0 30], Switch = "BP1");
            testCase.assertSize(brs, [1 2]);
            bp = brs(1).points("BP");
            testCase.assertSize(bp, [1 1]);
            fprintf("test5 [%s] BP r error %.3e\n", Evaluator, ...
                bp.P(brs(1).FreeIndex) - 1);
            testCase.verifyEqual(bp.P(brs(1).FreeIndex), 1, AbsTol = 1e-10);
            sw = brs(2);
            testCase.verifyEqual(sw.Stop, "Bounds");
            testCase.verifyEqual(sw.Alpha(end), 30, AbsTol = 1e-9);
            dxy = max(abs(sw.U(:, 1) - sw.U(:, 2)));
            dz = max(abs(sw.U(:, 3) - sw.U(:, 1).^2 / b));
            h = sw.points("H");
            testCase.assertSize(h, [1 1]);
            fprintf("test5 [%s] end r err %.3e, dxy %.3e, dz %.3e, H r err %.3e, l1 %.6e\n", ...
                Evaluator, sw.Alpha(end) - 30, dxy, dz, ...
                h.P(sw.FreeIndex) - 470/19, h.Data.coefficient);
            testCase.verifyEqual(dxy, 0, AbsTol = 1e-9);
            testCase.verifyEqual(dz, 0, AbsTol = 1e-9);
            testCase.verifyEqual(h.P(sw.FreeIndex), 470/19, AbsTol = 1e-9);
            testCase.verifyGreaterThan(h.Data.coefficient, 0);
        end

        function branchFields(testCase, Evaluator)
            m = testCase.model(testCase.Lorenz, Evaluator);
            eq = phasecont.equilibrium(m, [0 0 0], ...
                Params = struct(s = 10, r = 0.5, b = 8/3));
            br = phasecont.continuation(m, eq, Free = "r", Hmax = 0.5, ...
                MaxSteps = 200, Bounds = [0 30]);
            N = numel(br.Alpha);
            testCase.verifyGreaterThan(N, 5);
            testCase.verifySize(br.Alpha, [N 1]);
            testCase.verifySize(br.U, [N 3]);
            testCase.verifySize(br.Eigenvalues, [N 3]);
            testCase.verifySize(br.Tests, [N 3]);
            testCase.verifySize(br.StepSize, [N 1]);
            testCase.verifySize(br.NewtonIters, [N 1]);
            testCase.verifySize(br.Stable, [N 1]);
            testCase.verifyClass(br.Stable, "logical");
            testCase.verifyEqual(br.Stable, all(real(br.Eigenvalues) < 0, 2));
            testCase.verifyTrue(any(br.Stable) && ~all(br.Stable));
            testCase.verifyEqual(br.Free, "r");
            testCase.verifyEqual(br.FreeIndex, 2);
            testCase.verifyEqual(br.P, [10; 0.5; 8/3]);
            testCase.verifyTrue(br.Model == m);
            sp = br.points();
            testCase.verifyEqual(numel(sp), numel(br.Specials));
            testCase.verifyGreaterThanOrEqual(numel(sp), 2);
            for k = 1:numel(sp)
                % A special lies in the step that ends at point Index.
                i = sp(k).Index;
                lo = min(br.Alpha(max(i - 1, 1)), br.Alpha(i));
                hi = max(br.Alpha(max(i - 1, 1)), br.Alpha(i));
                testCase.verifyGreaterThanOrEqual(sp(k).P(2), lo - 1e-6);
                testCase.verifyLessThanOrEqual(sp(k).P(2), hi + 1e-6);
                testCase.verifyEqual(sp(k).P([1 3]), [10; 8/3]);
            end
            testCase.verifySize(br.points("XY"), [0 1]);
            testCase.verifyClass(br.points("XY"), "phasecont.Point");
            both = [br br];
            testCase.verifyError(@() points(both), "phasecont:model:badBranch");
        end

        function plotEndsWithStyledLines(testCase)
            m = testCase.model(testCase.Fold, "jit");
            eq = phasecont.equilibrium(m, -1);
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 60, Bounds = [-5 5], Plot = true, Var = "x");
            ax = gca;
            ln = [findobj(ax, "Tag", "pc-stable"); findobj(ax, "Tag", "pc-unstable")];
            testCase.assertNotEmpty(ln);
            x = [];
            y = [];
            for k = 1:numel(ln)
                x = [x, ln(k).XData]; %#ok<AGROW>
                y = [y, ln(k).YData]; %#ok<AGROW>
            end
            testCase.verifyEqual(sort(unique(x)), sort(unique(br.Alpha')));
            testCase.verifyEqual(sort(unique(y)), sort(unique(br.U(:, 1)')));
            testCase.verifyEqual(string(ax.XLabel.String), "a");
            testCase.verifyEqual(string(ax.YLabel.String), "x");
        end

        function holdOnKeepsTheOldLine(testCase)
            m = testCase.model(testCase.Fold, "jit");
            eq = phasecont.equilibrium(m, -1);
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            ax = axes(fig);
            old = line(ax, [0 1], [0 1]);
            hold(ax, "on");
            phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 30, Bounds = [-5 5], Plot = true);
            lines = findobj(ax, "Type", "line");
            testCase.verifyGreaterThan(numel(lines), 2);
            testCase.verifyTrue(isvalid(old));
            testCase.verifyTrue(any(lines == old));
        end

        function interruptCancelsAndFreesTheRun(testCase)
            m = testCase.model(testCase.Lorenz, "fallback");
            eq = phasecont.equilibrium(m, [0 0 0]);
            calls = 0;
            id = uint64(0);
            phasecont.internal.pollHook(@hook);
            testCase.addTeardown(@() phasecont.internal.pollHook([]));
            testCase.verifyError(@() phasecont.continuation(m, eq, Free = "r", ...
                H0 = 0.001, Hmax = 0.001, MaxSteps = 1000000, ...
                Bounds = [-1000 1000]), "test:interrupt");
            testCase.verifyEqual(calls, 2);
            testCase.verifyNotEqual(id, uint64(0));
            testCase.verifyError(@() pckern("run_poll", id), ...
                "phasecont:engine:badHandle");

            function hook(run, ~)
                calls = calls + 1;
                id = run.Id;
                if calls == 2
                    error("test:interrupt", "interrupted");
                end
            end
        end

        function aShortRunDoesNotWaitForTheFullPollInterval(testCase)
            % The poll loop used to pause 50 ms after the first poll. The
            % fastest of several warm runs has a loose bound that the old
            % loop could never meet.
            m = testCase.model(testCase.Fold, "jit");
            eq = phasecont.equilibrium(m, -1);
            t = zeros(1, 8);
            for k = 1:numel(t)
                t0 = tic;
                phasecont.continuation(m, eq, Free = "a", MaxSteps = 20, ...
                    Bounds = [-5 5]);
                t(k) = toc(t0);
            end
            testCase.verifyLessThan(min(t), 0.045);
        end

        function startErrors(testCase)
            m = testCase.model(testCase.Fold, "jit");
            other = testCase.model(testCase.Fold, "jit");
            eq = phasecont.equilibrium(m, -1);
            foreign = phasecont.equilibrium(other, -1);
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "zz"), ...
                "phasecont:model:unknownParameter");
            testCase.verifyError(@() phasecont.continuation(m, eq), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, foreign, Free = "a"), ...
                "phasecont:model:badStart");
            testCase.verifyError( ...
                @() phasecont.continuation(m, [1 2], Free = "a"), ...
                "phasecont:model:badStart");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", Bounds = [0 Inf]), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", Bounds = [1 0]), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", H0 = NaN), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", Detect = "XY"), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", Direction = 2), ...
                "phasecont:model:badOption");
            testCase.verifyError( ...
                @() phasecont.continuation(m, eq, Free = "a", Plot = true, ...
                Var = "q"), "phasecont:model:unknownVariable");
        end

        function startFromTheLpPoint(testCase, Evaluator)
            % Fold x' = a + x^2: leave the LP in both directions.
            m = testCase.model(testCase.Fold, Evaluator);
            eq = phasecont.equilibrium(m, -1);
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 100, Bounds = [-5 5]);
            lp = br.points("LP");
            testCase.assertSize(lp, [1 1]);
            ends = zeros(1, 2);
            for k = 1:2
                d = [1 -1];
                nb = phasecont.continuation(m, lp, Free = "a", Direction = d(k), ...
                    H0 = 0.05, Hmax = 0.05, MaxSteps = 20, Bounds = [-5 5]);
                testCase.verifyGreaterThan(numel(nb.Alpha), 10);
                testCase.verifyEqual(nb.Alpha(1), lp.P(br.FreeIndex), AbsTol = 1e-10);
                for j = 1:numel(nb.Alpha)
                    r = m.rhs(nb.U(j, :), nb.Alpha(j));
                    testCase.verifyEqual(r, 0, AbsTol = 1e-10);
                end
                ends(k) = sign(nb.U(end, 1));
            end
            testCase.verifyNotEqual(ends(1), ends(2));
        end

        function startFromTheHopfPoint(testCase, Evaluator)
            m = testCase.model(testCase.HopfC, Evaluator);
            eq = phasecont.equilibrium(m, [0 0], Params = struct(al = -0.5, c = 1));
            br = phasecont.continuation(m, eq, Free = "al", Direction = 1, ...
                MaxSteps = 60, Bounds = [-1 1]);
            h = br.points("H");
            testCase.assertSize(h, [1 1]);
            nb = phasecont.continuation(m, h, Free = "al", MaxSteps = 20, ...
                Bounds = [-1 1]);
            testCase.verifyGreaterThan(numel(nb.Alpha), 3);
            testCase.verifySize(nb.points("H"), [0 1]);
            testCase.verifyEqual(nb.U, zeros(numel(nb.Alpha), 2), AbsTol = 1e-10);
        end

        function startFromTheBpPointNamesSwitch(testCase, Evaluator)
            m = testCase.model(testCase.Lorenz, Evaluator);
            eq = phasecont.equilibrium(m, [0 0 0], ...
                Params = struct(s = 10, r = 0.5, b = 8/3));
            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.5, MaxSteps = 100, Bounds = [0 3]);
            bp = br.points("BP");
            testCase.assertSize(bp, [1 1]);
            try
                phasecont.continuation(m, bp, Free = "r", Bounds = [0 3]);
                testCase.verifyFail("expected an error at the BP start");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:engine:badProblem");
                testCase.verifySubstring(err.message, "Switch");
            end
        end

        function startIsSpecialSkipsTheFirstStep(testCase, Evaluator)
            % The first step from al = -0.01 crosses the H at 0. An EP start
            % reports it, a start labeled "H" does not (start_is_special).
            m = testCase.model(testCase.HopfC, Evaluator);
            u = [0 0];
            p = [-0.01; 1];
            ep = phasecont.Point(U = u, P = p, Label = "EP", Model = m);
            sp = phasecont.Point(U = u, P = p, Label = "H", Model = m);
            args = {"Free", "al", "Direction", 1, "H0", 0.05, "Hmin", 1e-6, ...
                "Hmax", 0.05, "MaxSteps", 3, "Bounds", [-1 1]};
            b1 = phasecont.continuation(m, ep, args{:});
            b2 = phasecont.continuation(m, sp, args{:});
            testCase.verifySize(b1.points("H"), [1 1]);
            testCase.verifySize(b2.points("H"), [0 1]);
        end
    end
end
