classdef MapPeriodTest < matlab.unittest.TestCase
    % Tests slice 5 of map support (SPEC section 8y, test 8): the Iterate
    % option of equilibrium and continuation, Switch = "PD1" on a map EP
    % branch, a start at a PD special with Type "EP", and curves at the
    % iterate of their special. Each test states its map and the closed form
    % it checks, and computes the iterates itself. Every model is written here.

    properties (Constant)
        % x -> r x (1 - x). Its 2-cycle is x+- = ((r+1) +- sqrt((r+1)(r-3)))/(2r)
        % for r > 3, with the multiplier -r^2 + 2r + 4 of F^2, which is -1 at
        % r = 1 + sqrt(6).
        Logistic = "par r=2" + newline + "init x=0.2" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        % (x, y) -> (1 - a x^2 + y, b x) with F_x = [-2 a x, 1; b, 0] and
        % det F_x = -b. Its fixed points have PD at a = 3(1-b)^2/4, and its
        % 2-cycle has PD at a = (1-b)^2 + (1+b)^2/4, where the multipliers
        % of F^2 are -1 and -b^2 since det F^2_x = b^2.
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        Decay = "par a=2" + newline + "init x=1" + newline + ...
            "x'=-a*x" + newline + "done" + newline
        Tol = 1e-8
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

        function [m, eq, brs] = logisticSwitch(testCase)
            % The fixed point x = 1 - 1/r from r = 2.5 to 3.6 with its PD
            % at r = 3, switched to the 2-cycle.
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0.6, Params = struct(r = 2.5));
            brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [2.5 3.6], Switch = "PD1");
            testCase.assertSize(brs, [1 2]);
        end

        function [m, pd2] = henonCyclePd(testCase)
            % The Henon fixed point from a = 0.2 at b = 0.3, switched at its
            % PD to the 2-cycle, and the PD special of the 2-cycle.
            b = 0.3;
            m = testCase.model(testCase.Henon);
            x = (-(1 - b) + sqrt((1 - b)^2 + 0.8)) / 0.4;
            eq = phasecont.equilibrium(m, [x b*x], ...
                Params = struct(a = 0.2, b = b));
            brs = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [0.2 1.2], Switch = "PD1");
            testCase.assertSize(brs, [1 2]);
            testCase.verifyEqual([brs.Iterate], [1 2]);
            pd1 = brs(1).points("PD");
            testCase.assertSize(pd1, [1 1]);
            testCase.verifyEqual(pd1.P(1), 3*(1 - b)^2/4, AbsTol = testCase.Tol);
            pd2 = brs(2).points("PD");
            testCase.assertSize(pd2, [1 1]);
            testCase.verifyEqual(pd2.Iterate, 2);
            testCase.verifyEqual(pd2.P(1), (1 - b)^2 + (1 + b)^2/4, ...
                AbsTol = testCase.Tol);
        end
    end

    methods (Static, Access = private)
        function x = logisticCycle(r)
            d = sqrt((r + 1) .* (r - 3));
            x = [(r + 1 + d) ./ (2*r), (r + 1 - d) ./ (2*r)];
        end
    end

    methods (Test)
        function equilibriumAtIterate2IsAPointOfThe2Cycle(testCase)
            m = testCase.model(testCase.Logistic);
            r = 3.2;
            want = MapPeriodTest.logisticCycle(r);
            for x0 = [0.8 0.5]
                pt = phasecont.equilibrium(m, x0, Params = struct(r = r), Iterate = 2);
                testCase.verifyEqual(pt.Iterate, 2);
                [err, j] = min(abs(want - pt.U));
                testCase.verifyLessThan(err, 1e-12, "x0 = " + x0);
                testCase.verifyEqual(pt.U, want(j), AbsTol = 1e-12);
                x1 = r*pt.U*(1 - pt.U);
                testCase.verifyEqual(r*x1*(1 - x1), pt.U, AbsTol = 1e-12);
                mu = -r^2 + 2*r + 4;
                testCase.verifyEqual(mu, 0.16, AbsTol = 1e-12);
                testCase.verifyEqual(real(pt.Eigenvalues), mu, AbsTol = 1e-10);
                testCase.verifyEqual(imag(pt.Eigenvalues), 0);
                testCase.verifyEqual(pt.Stable, true);
            end
            % Without the option the point has iterate 1.
            pt = phasecont.equilibrium(m, 0.7, Params = struct(r = r));
            testCase.verifyEqual(pt.Iterate, 1);
            testCase.verifyEqual(pt.U, 1 - 1/r, AbsTol = 1e-12);
        end

        function pdSwitchGivesTheBranchesAtIterate1And2(testCase)
            [~, ~, brs] = testCase.logisticSwitch();
            testCase.verifyEqual([brs.Iterate], [1 2]);
            b0 = brs(1);
            testCase.verifyEqual(b0.U, 1 - 1 ./ b0.Alpha, AbsTol = 1e-9);
            pd = b0.points("PD");
            testCase.assertSize(pd, [1 1]);
            testCase.verifyEqual(pd.Iterate, 1);
            testCase.verifyEqual(pd.P(1), 3, AbsTol = testCase.Tol);

            b1 = brs(2);
            r = b1.Alpha;
            testCase.verifyGreaterThan(numel(r), 5);
            x0 = b1.U;
            x1 = r.*x0.*(1 - x0);
            testCase.verifyEqual(r.*x1.*(1 - x1), x0, AbsTol = 1e-9);
            % Past r = 3 the row is x+ or x-, away from the fixed point.
            far = r > 3.01;
            cyc = MapPeriodTest.logisticCycle(r(far));
            testCase.verifyEqual(min(abs(cyc - x0(far)), [], 2), ...
                zeros(nnz(far), 1), AbsTol = 1e-9);
            testCase.verifyEqual(real(b1.Eigenvalues), -r.^2 + 2*r + 4, ...
                AbsTol = 1e-8);
            pd2 = b1.points("PD");
            testCase.assertSize(pd2, [1 1]);
            testCase.verifyEqual(pd2.Iterate, 2);
            testCase.verifyEqual(pd2.P(1), 1 + sqrt(6), AbsTol = testCase.Tol);
            testCase.verifyEqual(real(pd2.Eigenvalues), -1, AbsTol = testCase.Tol);
        end

        function startAtAPdSpecialGivesTheSwitchedBranch(testCase)
            [m, ~, brs] = testCase.logisticSwitch();
            pd = brs(1).points("PD");
            opts = {"Free", "r", "Direction", 1, "Hmax", 0.05, ...
                "MaxSteps", 1000, "Bounds", [2.5 3.6]};
            for given = {{}, {"Iterate", 2}}
                br = phasecont.continuation(m, pd, "Type", "EP", opts{:}, given{1}{:});
                testCase.assertSize(br, [1 1]);
                testCase.verifyEqual(br.Iterate, 2);
                testCase.assertSize(br.Alpha, size(brs(2).Alpha));
                testCase.verifyEqual(br.Alpha, brs(2).Alpha, AbsTol = 1e-12);
                testCase.verifyEqual(br.U, brs(2).U, AbsTol = 1e-12);
                testCase.verifyEqual(br.Eigenvalues, brs(2).Eigenvalues, AbsTol = 1e-12);
                testCase.verifyEqual([br.Specials.Label], [brs(2).Specials.Label]);
            end
            % Iterate 1 from the PD special is an ordinary start on the
            % fixed point.
            br = phasecont.continuation(m, pd, "Type", "EP", opts{:}, "Iterate", 1);
            testCase.verifyEqual(br.Iterate, 1);
            testCase.verifyEqual(br.U, 1 - 1 ./ br.Alpha, AbsTol = 1e-9);
        end

        function pdCurveOfTheHenon2CycleRunsAtIterate2(testCase)
            [m, pd2] = testCase.henonCyclePd();
            br = phasecont.continuation(m, pd2, Type = "PD", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            testCase.verifyEqual(br.Iterate, 2);
            testCase.verifyEqual(br.Type, "PD");
            testCase.verifyEqual(br.Free, ["a" "b"]);
            a = br.Alpha(:, 1);
            b = br.Alpha(:, 2);
            testCase.verifyGreaterThan(numel(a), 5);
            testCase.verifyEqual(a, (1 - b).^2 + (1 + b).^2/4, AbsTol = testCase.Tol);
            x0 = br.U(:, 1);
            y0 = br.U(:, 2);
            x1 = 1 - a.*x0.^2 + y0;
            y1 = b.*x0;
            testCase.verifyEqual(1 - a.*x1.^2 + y1, x0, AbsTol = testCase.Tol);
            testCase.verifyEqual(b.*x1, y0, AbsTol = testCase.Tol);
            mu = sort(real(br.Eigenvalues), 2);
            testCase.verifyEqual(mu, [-ones(size(b)), -b.^2], AbsTol = 1e-7);
            testCase.verifyEqual(imag(br.Eigenvalues), zeros(size(br.Eigenvalues)), ...
                AbsTol = testCase.Tol);
            testCase.verifyEqual(br.Alpha(end, 2), 0.6, AbsTol = 1e-9);
            % Iterate 2, the iterate of the special, is the same run.
            same = phasecont.continuation(m, pd2, Type = "PD", Free = ["a" "b"], ...
                Bounds = [0.1 0.6], Iterate = 2);
            testCase.verifyEqual(same.Alpha, br.Alpha);
            testCase.verifyEqual(same.U, br.U);
        end

        function iterateIsRefusedWhereItDoesNotApply(testCase)
            ode = testCase.model(testCase.Decay);
            testCase.verifyError(@() phasecont.equilibrium(ode, 1, Iterate = 2), ...
                "phasecont:model:badOption");
            eq = phasecont.equilibrium(ode, 0.1);
            testCase.verifyError(@() phasecont.continuation(ode, eq, Free = "a", ...
                Iterate = 2), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.equilibrium(ode, 1, Iterate = zeros(1, 0)), ...
                "phasecont:model:badOption");
            m = testCase.model(testCase.Logistic);
            for k = {0, 1.5, -2, 2e6, Inf, NaN, [1 2], "2", 2i, zeros(1, 0), '', {}, string.empty}
                testCase.verifyError(@() phasecont.equilibrium(m, 0.5, Iterate = k{1}), ...
                    "phasecont:model:badOption", string(class(k{1})));
            end
            [m2, pd2] = testCase.henonCyclePd();
            for k = [1 4]
                testCase.verifyError(@() phasecont.continuation(m2, pd2, ...
                    Type = "PD", Free = ["a" "b"], Iterate = k), ...
                    "phasecont:model:badOption", "Iterate = " + k);
            end
        end

        function aSessionKeepsIterateAndReplays(testCase)
            [m, ~, brs] = testCase.logisticSwitch();
            pt = phasecont.equilibrium(m, 0.8, Params = struct(r = 3.2), Iterate = 2);
            file = fullfile(testCase.Dir, "period.pcsession.json");
            save(phasecont.Session(m, {brs(1), brs(2), pt}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.assertNumElements(back.Items, 3);
            testCase.verifyEqual([back.Items{1}.Iterate, back.Items{2}.Iterate], [1 2]);
            got = back.Items{2};
            testCase.verifyEqual(got.Alpha, brs(2).Alpha);
            testCase.verifyEqual(got.U, brs(2).U);
            testCase.verifyEqual(got.points("PD").Iterate, 2);
            p = back.Items{3};
            testCase.verifyEqual(p.Iterate, 2);
            testCase.verifyEqual(p.U, pt.U);
            testCase.verifyEqual(p.Stable, true);
            r = phasecont.session.replay(file);
            testCase.verifyNotEmpty(r);
            testCase.verifyTrue(all([r.Pass]));
        end
    end
end
