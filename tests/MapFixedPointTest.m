classdef MapFixedPointTest < matlab.unittest.TestCase
    % Tests slice 2 of map support (SPEC section 8v, test 10): equilibrium on
    % a map returns a fixed point F(x) = x with its multipliers, and
    % continuation with Type "EP" follows a branch of fixed points with the
    % tests [LP BP PD NS] and labels BP, PD and NS. Each test computes the
    % expected multipliers itself. Every model is written here.

    properties (Constant)
        % x -> r x (1 - x): x = 0 with multiplier r, and x = 1 - 1/r with
        % multiplier 2 - r, so BP at r = 1 and PD at r = 3, x = 2/3, with
        % b = (f'')^2/4 + f'''/6 = 36/4 = 9.
        Logistic = "par r=2" + newline + "init x=0.2" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        % (x, y) -> (1 - a x^2 + y, b x), with the fixed point
        % x = (-(1-b) + sqrt((1-b)^2 + 4a))/(2a), y = b x, and the
        % multipliers -a x +- sqrt(a^2 x^2 + b).
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        % (x, y) -> (r x (1 - y), x), with the fixed point x = y = 1 - 1/r
        % and the multipliers the roots of mu^2 - mu + (r - 1) = 0, so NS at
        % r = 2 with theta = pi/3, d = -1 and coefficient2 = -sqrt(3).
        Delayed = "par r=1.8" + newline + "init x=0.4, y=0.4" + newline + ...
            "x(t+1)=r*x*(1-y)" + newline + "y(t+1)=x" + newline + ...
            "done" + newline
        Decay = "par a=2" + newline + "init x=3" + newline + ...
            "x'=-a*x" + newline + "done" + newline
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

        function verifyBranchShape(testCase, br, n)
            % BR has n state columns, 4 test columns, no codim-2 columns, and
            % Stable from the moduli of the multipliers.
            npts = size(br.Alpha, 1);
            testCase.verifyGreaterThan(npts, 5);
            testCase.verifySize(br.U, [npts n]);
            testCase.verifySize(br.Eigenvalues, [npts n]);
            testCase.verifySize(br.Tests, [npts 4]);
            testCase.verifySize(br.Codim2, [npts 0]);
            testCase.verifyEqual(br.Stable, all(abs(br.Eigenvalues) < 1, 2));
        end

        function br = delayedBranch(testCase)
            % The branch of the delayed logistic map from r = 1.8 through NS.
            m = testCase.model(testCase.Delayed);
            x = 1 - 1/1.8;
            eq = phasecont.equilibrium(m, [x x] + 0.01, Params = struct(r = 1.8));
            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [1.5 2.5]);
        end
    end

    methods (Static, Access = private)
        function mu = henonMultipliers(a, b)
            x = (-(1 - b) + sqrt((1 - b)^2 + 4*a)) / (2*a);
            mu = -a*x + [-1; 1] * sqrt(a^2*x^2 + b);
        end

        function mu = delayedMultipliers(r)
            mu = (1 + [-1; 1] * sqrt(complex(1 - 4*(r - 1)))) / 2;
        end
    end

    methods (Test)
        function equilibriumOfTheLogisticMap(testCase)
            % At r = 2 the fixed point from 0.9 is x = 1/2 with multiplier
            % 2 - r = 0, so it is stable.
            m = testCase.model(testCase.Logistic);
            pt = phasecont.equilibrium(m, 0.9);
            testCase.verifyEqual(pt.Label, "EP");
            testCase.verifyEqual(pt.U(:), 0.5, AbsTol = 1e-8);
            testCase.verifyEqual(real(pt.Eigenvalues(:)), 0, AbsTol = 1e-8);
            testCase.verifyEqual(imag(pt.Eigenvalues(:)), 0);
            testCase.verifyTrue(all(abs(pt.Eigenvalues) < 1));
            % At r = 3.5 the fixed point 1 - 1/r has multiplier -1.5.
            pt = phasecont.equilibrium(m, 0.7, Params = struct(r = 3.5));
            testCase.verifyEqual(pt.U(:), 1 - 1/3.5, AbsTol = 1e-8);
            testCase.verifyEqual(real(pt.Eigenvalues(:)), 2 - 3.5, AbsTol = 1e-8);
            testCase.verifyFalse(all(abs(pt.Eigenvalues) < 1));
        end

        function equilibriumOfTheHenonMap(testCase)
            a = 1.4;
            b = 0.3;
            x = (-(1 - b) + sqrt((1 - b)^2 + 4*a)) / (2*a);
            m = testCase.model(testCase.Henon);
            pt = phasecont.equilibrium(m, [0.6 0.2]);
            testCase.verifyEqual(pt.Label, "EP");
            testCase.verifyEqual(pt.U(:), [x; b*x], AbsTol = 1e-8);
            mu = testCase.henonMultipliers(a, b);
            testCase.verifyEqual(sort(real(pt.Eigenvalues(:))), mu, AbsTol = 1e-8);
            testCase.verifyEqual(imag(pt.Eigenvalues(:)), [0; 0], AbsTol = 1e-12);
            % One multiplier is below -1, so the fixed point is unstable.
            testCase.verifyFalse(all(abs(pt.Eigenvalues) < 1));
            % At a = 0.2 both multipliers lie inside the unit circle.
            pt = phasecont.equilibrium(m, [1 0.3], Params = struct(a = 0.2, b = b));
            mu = testCase.henonMultipliers(0.2, b);
            testCase.verifyEqual(sort(real(pt.Eigenvalues(:))), mu, AbsTol = 1e-8);
            testCase.verifyTrue(all(abs(pt.Eigenvalues) < 1));
        end

        function logisticBranchHasBpThenPdWithB9(testCase)
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0, Params = struct(r = 0.5));
            brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [0 4], Switch = "BP1");
            testCase.assertSize(brs, [1 2]);
            b0 = brs(1);
            testCase.verifyBranchShape(b0, 1);
            % On x = 0 the multiplier is r.
            testCase.verifyEqual(b0.U, zeros(size(b0.U)), AbsTol = 1e-10);
            testCase.verifyEqual(real(b0.Eigenvalues), b0.Alpha, AbsTol = 1e-10);
            testCase.verifyEqual(imag(b0.Eigenvalues), zeros(size(b0.Alpha)));
            bp = b0.points("BP");
            testCase.assertSize(bp, [1 1]);
            testCase.verifyEqual(bp.P(b0.FreeIndex), 1, AbsTol = 1e-8);
            testCase.verifyEqual(bp.U(:), 0, AbsTol = 1e-8);
            % psi_NS is NaN for a map with one variable.
            testCase.verifyTrue(all(isnan(b0.Tests(:, 4))));

            b1 = brs(2);
            testCase.verifyBranchShape(b1, 1);
            testCase.verifyEqual(b1.U, 1 - 1 ./ b1.Alpha, AbsTol = 1e-9);
            testCase.verifyEqual(real(b1.Eigenvalues), 2 - b1.Alpha, AbsTol = 1e-8);
            testCase.verifyEqual(imag(b1.Eigenvalues), zeros(size(b1.Alpha)));
            pd = b1.points("PD");
            testCase.assertSize(pd, [1 1]);
            fprintf("PD r error %.3e, b %.15g\n", pd.P(b1.FreeIndex) - 3, ...
                pd.Data.coefficient);
            testCase.verifyEqual(pd.P(b1.FreeIndex), 3, AbsTol = 1e-8);
            testCase.verifyEqual(pd.U(:), 2/3, AbsTol = 1e-8);
            testCase.verifyEqual(real(pd.Eigenvalues(:)), -1, AbsTol = 1e-8);
            testCase.verifyEqual(pd.Data.coefficient, 9, RelTol = 1e-7);
            testCase.verifyEmpty(b1.Warnings);
        end

        function delayedLogisticBranchHasNs(testCase)
            br = testCase.delayedBranch();
            testCase.verifyBranchShape(br, 2);
            x = 1 - 1 ./ br.Alpha;
            testCase.verifyEqual(br.U, [x x], AbsTol = 1e-9);
            for k = 1:size(br.Alpha, 1)
                ev = br.Eigenvalues(k, :).';
                mu = testCase.delayedMultipliers(br.Alpha(k));
                [~, i] = sort(imag(ev));
                testCase.verifyEqual(ev(i), mu, AbsTol = 1e-8);
            end
            ns = br.points("NS");
            testCase.assertSize(ns, [1 1]);
            fprintf("NS r error %.3e, theta error %.3e, d %.15g, coefficient2 %.15g\n", ...
                ns.P(br.FreeIndex) - 2, ns.Data.values.theta - pi/3, ...
                ns.Data.coefficient, ns.Data.coefficient2);
            testCase.verifyEqual(ns.P(br.FreeIndex), 2, AbsTol = 1e-8);
            testCase.verifyEqual(ns.U(:), [0.5; 0.5], AbsTol = 1e-8);
            testCase.verifyEqual(ns.Data.values.theta, pi/3, RelTol = 1e-7);
            testCase.verifyEqual(ns.Data.coefficient, -1, RelTol = 1e-7);
            testCase.verifyEqual(ns.Data.coefficient2, -sqrt(3), RelTol = 1e-7);
            % The fixed point is stable below r = 2 and unstable above, since
            % the multipliers have modulus sqrt(r - 1).
            away = abs(br.Alpha - 2) > 1e-6;
            testCase.verifyEqual(br.Stable(away), br.Alpha(away) < 2);
        end

        function detectOnAnOdeRefusesPdAndNs(testCase)
            m = testCase.model(testCase.Decay);
            eq = phasecont.equilibrium(m, 0);
            for label = ["PD" "NS"]
                testCase.verifyError(@() phasecont.continuation(m, eq, ...
                    Free = "a", Detect = label), "phasecont:model:badOption");
            end
        end

        function detectOnAMapTakesASubset(testCase)
            % Without "PD" the PD column is NaN and the branch has no PD.
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0.6, Params = struct(r = 2.5));
            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [2 3.5], ...
                Detect = ["LP" "BP"]);
            testCase.verifyTrue(all(isnan(br.Tests(:, 3))));
            testCase.verifyEmpty(br.points("PD"));
            testCase.verifyEqual(br.Alpha(end), 3.5, AbsTol = 1e-9);
            testCase.verifyError(@() phasecont.continuation(m, eq, ...
                Free = "r", Detect = "LC"), "phasecont:model:badOption");
        end

        function curvesFromAMapPointAreRefused(testCase)
            % The Henon branch from a = 0.2 has an LP at a = -(1-b)^2/4 below
            % and a PD at a = 3(1-b)^2/4 above. A curve from either point is
            % refused with a phasecont: error.
            b = 0.3;
            m = testCase.model(testCase.Henon);
            x = (-(1 - b) + sqrt((1 - b)^2 + 0.8)) / 0.4;
            eq = phasecont.equilibrium(m, [x b*x] + 0.01, ...
                Params = struct(a = 0.2, b = b));
            up = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [-1 0.5]);
            % Past the LP the branch runs off with x growing as a goes to 0,
            % so MaxSteps ends this run.
            down = phasecont.continuation(m, eq, Free = "a", Direction = -1, ...
                Hmax = 0.05, MaxSteps = 100);
            pd = up.points("PD");
            lp = down.points("LP");
            testCase.assertSize(pd, [1 1]);
            testCase.assertSize(lp, [1 1]);
            testCase.verifyEqual(pd.P(up.FreeIndex), 3*(1 - b)^2/4, AbsTol = 1e-8);
            testCase.verifyEqual(lp.P(down.FreeIndex), -(1 - b)^2/4, AbsTol = 1e-8);
            for start = {lp, pd}
                for type = ["LP" "PD" "NS" "LC"]
                    try
                        phasecont.continuation(m, start{1}, Type = type, ...
                            Free = ["a" "b"], MaxSteps = 3);
                    catch err
                        fprintf("Type %s from %s: %s\n", type, start{1}.Label, ...
                            err.identifier);
                        testCase.verifyTrue(startsWith(string(err.identifier), ...
                            "phasecont:"), string(err.identifier));
                        continue
                    end
                    testCase.verifyFail("Type " + type + " from a map " + ...
                        start{1}.Label + " point ran");
                end
            end
        end

        function aSessionKeepsAMapBranchAndReplays(testCase)
            br = testCase.delayedBranch();
            file = fullfile(testCase.Dir, "delayed.pcsession.json");
            save(phasecont.Session(br.Model, {br}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            got = back.Items{1};
            testCase.verifyEqual(got.Alpha, br.Alpha);
            testCase.verifyEqual(got.U, br.U);
            testCase.verifyEqual(got.Eigenvalues, br.Eigenvalues);
            testCase.verifyEqual(got.Stable, br.Stable);
            testCase.verifyEqual(got.Tests, br.Tests);
            testCase.verifySize(got.Codim2, [size(br.Alpha, 1) 0]);
            testCase.verifyEqual([got.Specials.Label], [br.Specials.Label]);
            ns = got.points("NS");
            testCase.assertSize(ns, [1 1]);
            testCase.verifyEqual(ns.Data.values.theta, pi/3, RelTol = 1e-7);
            testCase.verifyEqual(ns.Data.coefficient, -1, RelTol = 1e-7);
            r = phasecont.session.replay(file);
            testCase.verifySize(r, [1 1]);
            testCase.verifyTrue(r.Pass);
        end
    end
end
