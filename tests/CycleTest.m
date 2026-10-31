classdef CycleTest < matlab.unittest.TestCase
    % Tests limit cycles through phasecont.continuation(Type = "LC"): the
    % start at a Hopf point, the Branch layout of a cycle branch, the PD
    % point of the Rossler system and the start of the doubled branch there,
    % cycle(i), the plot and the option errors. Every model is written here.
    %
    % No oracle file exists for the Rossler PD point (SPEC section 14), so
    % the checks are self-consistent: a multiplier near -1 at PD and twice
    % the period on the doubled branch.

    properties (Constant)
        Brusselator = "par a=1,b=1.5" + newline + ...
            "x'=a-(b+1)*x+x^2*y" + newline + "y'=b*x-x^2*y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Rossler = "par a=0.2,b=0.2,c=2.5" + newline + ...
            "x'=-y-z" + newline + "y'=x+a*y" + newline + ...
            "z'=b+z*(x-c)" + newline + "init x=0,y=0,z=0" + newline + ...
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

        function [m, h] = brusselatorHopf(testCase)
            % With a = 1 the Hopf point is at b = 2 with omega = 1.
            m = testCase.model(testCase.Brusselator);
            eq = phasecont.equilibrium(m, [1 1.2], Params = struct(a = 1, b = 1.2));
            ep = phasecont.continuation(m, eq, Free = "b", Bounds = [0.5 4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            h = h(1);
        end

        function [m, lc] = rosslerCycles(testCase)
            m = testCase.model(testCase.Rossler);
            eq = phasecont.equilibrium(m, [0 -0.08 0.08], ...
                Params = struct(a = 0.01, b = 0.2, c = 2.5));
            ep = phasecont.continuation(m, eq, Free = "a", Bounds = [0 0.4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "a", ...
                NTST = 40, NCOL = 4, Bounds = [0.02 0.4], MaxSteps = 400);
        end
    end

    methods (Test)
        function hopfStartGivesTheBrusselatorCycle(testCase)
            [m, h] = testCase.brusselatorHopf();
            testCase.verifyEqual(h.P(2), 2, AbsTol = 1e-8);
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 30);
            testCase.verifyClass(lc, "phasecont.Branch");
            testCase.verifyEqual(lc.Type, "LC");
            testCase.verifyEqual([lc.NTST lc.NCOL], [20 4]);
            npts = numel(lc.Alpha);
            testCase.verifyGreaterThan(npts, 5);
            testCase.verifySize(lc.U, [npts 2*(20*4 + 1)]);
            testCase.verifySize(lc.Mesh, [npts 21]);
            testCase.verifySize(lc.Period, [npts 1]);
            testCase.verifySize(lc.Eigenvalues, [npts 2]);
            testCase.verifySize(lc.Tests, [npts 3]);
            % The first cycle is close to the Hopf point: period 2 pi / omega.
            testCase.verifyEqual(lc.Period(1), 2*pi, RelTol = 1e-3);
            testCase.verifyEqual(lc.Alpha(1), 2, AbsTol = 1e-3);
            % One multiplier is the trivial 1 at every point.
            testCase.verifyLessThan(max(min(abs(lc.Eigenvalues - 1), [], 2)), 1e-6);
            % The Hopf point is supercritical (l1 < 0), so the cycles that
            % grow from it are stable.
            testCase.assertLessThan(h.Data.coefficient, 0);
            testCase.verifyTrue(all(lc.Stable));
            testCase.verifyEqual(lc.Mesh(:, [1 end]), repmat([0 1], npts, 1));

            % The orbit closes: the last base point is the first one.
            [t, x] = lc.cycle(npts);
            testCase.verifySize(t, [20*4 + 1, 1]);
            testCase.verifySize(x, [20*4 + 1, 2]);
            testCase.verifyEqual(t([1 end]), [0; lc.Period(npts)], AbsTol = 1e-12);
            testCase.verifyTrue(all(diff(t) > 0));
            testCase.verifyEqual(x(end, :), x(1, :), AbsTol = 1e-8);
            testCase.verifyGreaterThan(max(x(:, 1)) - min(x(:, 1)), 1e-3);
        end

        function rosslerPeriodDoubling(testCase)
            [m, lc] = testCase.rosslerCycles();
            pd = lc.points("PD");
            testCase.assertNotEmpty(pd);
            pd = pd(1);
            testCase.verifyGreaterThan(pd.P(1), 0.2);
            testCase.verifyLessThan(pd.P(1), 0.25);
            % At a PD point one multiplier is -1.
            testCase.verifyLessThan(min(abs(pd.Eigenvalues + 1)), 1e-4);
            testCase.verifyEqual([pd.Data.ntst pd.Data.ncol], [40 4]);
            testCase.verifySize(pd.U, [3*(40*4 + 1) 1]);
            testCase.verifySize(pd.Data.mesh, [1 41]);

            dbl = phasecont.continuation(m, pd, Type = "LC", Free = "a", ...
                NTST = 60, NCOL = 4, Bounds = [0.02 0.3], MaxSteps = 200);
            testCase.verifyEqual(dbl.Type, "LC");
            testCase.verifyEqual([dbl.NTST dbl.NCOL], [60 4]);
            testCase.verifyGreaterThan(numel(dbl.Alpha), 5);
            testCase.verifyEqual(dbl.Period(1), 2*pd.Data.period, RelTol = 1e-3);
            testCase.verifyEqual(dbl.Alpha(1), pd.P(1), AbsTol = 1e-3);
            [~, x] = dbl.cycle(numel(dbl.Alpha));
            testCase.verifyEqual(x(end, :), x(1, :), AbsTol = 1e-8);
        end

        function rosslerOracle(testCase)
            % SPEC section 9: a test without its reference file is skipped
            % as "oracle missing". The file comes from AUTO-07p or MatCont on
            % the developer's machine, for the run of rosslerCycles (b = 0.2,
            % c = 2.5, free a), and lists its special points as
            % {"tool", "version", "settings", "free": "a",
            %  "specials": [{"label": "PD", "a": value}, ...]}.
            root = fileparts(fileparts(mfilename("fullpath")));
            file = fullfile(root, "references", "rossler.json");
            testCase.assumeTrue(isfile(file), ...
                "oracle missing: references/rossler.json");
            ref = jsondecode(fileread(file));
            testCase.assertEqual(string(ref.free), "a");
            specials = ref.specials;
            if iscell(specials)
                specials = [specials{:}];
            end
            want = specials(string({specials.label}) == "PD");
            testCase.assertNotEmpty(want, "the oracle lists no PD point");
            [~, lc] = testCase.rosslerCycles();
            pd = lc.points("PD");
            testCase.assertNotEmpty(pd);
            testCase.verifyEqual(pd(1).P(1), want(1).a, RelTol = 1e-6);
        end

        function plotDrawsMaxAndMin(testCase)
            [m, h] = testCase.brusselatorHopf();
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 15);
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            ax = axes(fig);
            plot(lc, Var = "x", Parent = ax);
            st = findobj(ax, "Tag", "pc-stable");
            testCase.assertNotEmpty(st);
            % One run for the maximum and one for the minimum of x.
            testCase.verifyNumElements(st, 2);
            y = sort([max(st(1).YData) max(st(2).YData)]);
            cols = 1 + 2*(0:20*4);
            testCase.verifyEqual(y(2), max(lc.U(:, cols), [], "all"), AbsTol = 1e-12);
            testCase.verifyTrue(contains(string(ax.YLabel.String), "max, min"));

            % Plot = true ends with the same styled lines.
            fig2 = figure(Visible = "off");
            testCase.addTeardown(@() close(fig2, "force"));
            phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 15, Plot = true);
            tags = string({findobj(gca, "Type", "line").Tag});
            testCase.verifyTrue(all(ismember(tags, ...
                ["pc-stable" "pc-unstable" "pc-special"])));
            testCase.verifyEqual(nnz(tags == "pc-stable"), 2);
        end

        function optionErrors(testCase)
            [m, h] = testCase.brusselatorHopf();
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = "b", NCOL = 4), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = "b", NTST = 20), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = "b", NTST = 2.5, NCOL = 4), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Free = "b", NTST = 20), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = "b", NTST = 20, NCOL = 4, Detect = "LP"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = "b", NTST = 20, NCOL = 4, Switch = "BP1"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, h, ...
                Type = "LC", Free = ["a" "b"], NTST = 20, NCOL = 4), ...
                "phasecont:model:unknownParameter");

            % A cycle point that is not PD is not a start.
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 5);
            last = phasecont.Point(U = lc.U(end, :).', P = [1; lc.Alpha(end)], ...
                Label = "LC", Data = struct("period", lc.Period(end), ...
                "mesh", lc.Mesh(end, :)), Model = m);
            testCase.verifyError(@() phasecont.continuation(m, last, ...
                Type = "LC", Free = "b", NTST = 20, NCOL = 4), ...
                "phasecont:model:badStart");
        end

        function nonHopfStartIsAnEngineError(testCase)
            % The Jacobian of a 1-state model has no complex pair, so the
            % engine refuses the start and the run fails.
            m = testCase.model("par r=1" + newline + "x'=r-x" + newline + ...
                "init x=1" + newline + "done" + newline);
            testCase.verifyError(@() phasecont.continuation(m, 1, ...
                Type = "LC", Free = "r", NTST = 10, NCOL = 3), ...
                "phasecont:engine:badProblem");
        end

        function cycleNeedsAnLcBranch(testCase)
            [m, h] = testCase.brusselatorHopf();
            ep = phasecont.continuation(m, h, Free = "b", MaxSteps = 3);
            testCase.verifyError(@() ep.cycle(1), "phasecont:model:badBranch");
        end
    end
end
