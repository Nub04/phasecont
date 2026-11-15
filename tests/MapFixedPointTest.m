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
            testCase.verifyEqual(pt.Stable, true);
            % At r = 3.5 the fixed point 1 - 1/r has multiplier -1.5.
            pt = phasecont.equilibrium(m, 0.7, Params = struct(r = 3.5));
            testCase.verifyEqual(pt.U(:), 1 - 1/3.5, AbsTol = 1e-8);
            testCase.verifyEqual(real(pt.Eigenvalues(:)), 2 - 3.5, AbsTol = 1e-8);
            testCase.verifyFalse(all(abs(pt.Eigenvalues) < 1));
            testCase.verifyEqual(pt.Stable, false);
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
            testCase.verifyEqual(pt.Stable, false);
            % At a = 0.2 both multipliers lie inside the unit circle.
            pt = phasecont.equilibrium(m, [1 0.3], Params = struct(a = 0.2, b = b));
            mu = testCase.henonMultipliers(0.2, b);
            testCase.verifyEqual(sort(real(pt.Eigenvalues(:))), mu, AbsTol = 1e-8);
            testCase.verifyTrue(all(abs(pt.Eigenvalues) < 1));
            testCase.verifyEqual(pt.Stable, true);
        end

        function stableIsEmptyWhereItDecidesNothing(testCase)
            % At a special point a multiplier lies on the unit circle, so
            % Stable is empty there. A Point without a Model, and an EP
            % Point without eigenvalues, have nothing to test.
            br = testCase.delayedBranch();
            testCase.assertNotEmpty(br.Specials);
            for k = 1:numel(br.Specials)
                testCase.verifyEqual(br.Specials(k).Stable, false(0, 1), ...
                    "special " + br.Specials(k).Label);
            end
            testCase.verifyEqual(phasecont.Point(Eigenvalues = 0.5).Stable, false(0, 1));
            testCase.verifyEqual(phasecont.Point(Model = br.Model).Stable, false(0, 1));
            pt = phasecont.Point(Eigenvalues = 0.5, Label = "PD", Model = br.Model);
            testCase.verifyEqual(pt.Stable, false(0, 1));
            pt = phasecont.Point(Eigenvalues = [0.5; -1.5], Model = br.Model);
            testCase.verifyEqual(pt.Stable, false);
        end

        function aSessionKeepsTheStabilityOfAFixedPoint(testCase)
            % Stable is not stored, so the loaded Point computes it again
            % from the stored multipliers.
            m = testCase.model(testCase.Logistic);
            pts = {phasecont.equilibrium(m, 0.9), ...
                phasecont.equilibrium(m, 0.7, Params = struct(r = 3.5))};
            file = fullfile(testCase.Dir, "fixed.pcsession.json");
            save(phasecont.Session(m, pts), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.assertNumElements(back.Items, 2);
            testCase.verifyEqual(back.Items{1}.Stable, true);
            testCase.verifyEqual(back.Items{2}.Stable, false);
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

        function detectHOnAMapIsRefused(testCase)
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0.5);
            for detect = {"H", ["LP" "H"]}
                testCase.verifyError(@() phasecont.continuation(m, eq, ...
                    Free = "r", Detect = detect{1}), "phasecont:model:badOption");
            end
        end

        function curvesFromAMapPointRunOrAreRefused(testCase)
            % The Henon branch from a = 0.2 has an LP at a = -(1-b)^2/4 below
            % and a PD at a = 3(1-b)^2/4 above. The LP curve runs from the LP
            % point and the PD curve from the PD point (see MapCurveTest). An
            % LC run, an LP curve from the PD point and an NS curve from
            % either point are refused by the engine.
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
            free = ["a" "b"];
            lpc = phasecont.continuation(m, lp, Type = "LP", Free = free, MaxSteps = 3);
            testCase.verifyEqual(lpc.Type, "LP");
            pdc = phasecont.continuation(m, pd, Type = "PD", Free = free, MaxSteps = 3);
            testCase.verifyEqual(pdc.Type, "PD");
            refused = {lp, "LC", "phasecont:engine:unsupportedModel"; ...
                pd, "LC", "phasecont:engine:unsupportedModel"; ...
                pd, "LP", "phasecont:engine:badProblem"; ...
                lp, "NS", "phasecont:engine:badProblem"; ...
                pd, "NS", "phasecont:engine:badProblem"};
            for k = 1:size(refused, 1)
                start = refused{k, 1};
                type = refused{k, 2};
                testCase.verifyError(@() phasecont.continuation(m, start, ...
                    Type = type, Free = free, MaxSteps = 3), refused{k, 3}, ...
                    type + " from " + start.Label);
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

        function aVersion1SessionOfAMapLoadsAsStored(testCase)
            % v1.0.0 stored a map EP branch with rows of width 3n + 6 and an
            % H test where the PD test goes, and its problem asked for h.
            % Such a file is made here from a branch of this version by
            % removing the NS row and asking for h.
            m = testCase.model(testCase.Logistic);
            eq = phasecont.equilibrium(m, 0.6, Params = struct(r = 2.5));
            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [2 3.5]);
            file = fullfile(testCase.Dir, "logistic.pcsession.json");
            save(phasecont.Session(m, {br}), file);
            S = phasecont.internal.sessionCodec.readFile(file);
            testCase.assertSize(S.Records, [1 1]);
            rec = S.Records{1};
            n = 1;
            ns = 1 + n + 2*n + 4;
            testCase.assertEqual(rec.Stride, 3*n + 7);
            points = rec.Points;
            specials = rec.Specials;
            for b = 1:numel(points)
                points{b}(ns, :) = [];
                specials{b}(ns, :) = [];
            end
            json = strrep(rec.Json, """stride"":" + rec.Stride, ...
                """stride"":" + (rec.Stride - 1));
            testCase.assertNotEqual(json, rec.Json);
            problem = regexprep(rec.Problem, """detect"":\{[^}]*\}", ...
                """detect"":{""lp"":true,""bp"":true,""h"":true}");
            testCase.assertNotEqual(problem, rec.Problem);
            f = struct("Kind", rec.Kind, "Problem", problem, "U0", rec.U0, ...
                "P", rec.P, "ModelSha", rec.ModelSha, "Type", rec.Type, ...
                "Free", rec.Free, "Grid", rec.Grid, "N", rec.N, "Json", json, ...
                "Points", {points}, "Specials", {specials}, ...
                "Stride", rec.Stride - 1);
            S.Records{1} = phasecont.internal.RunRecord(f);
            header = struct("Created", "2026-11-14T00:00:00Z", ...
                "EngineName", "pckern", "EngineVersion", "1.0.0", ...
                "Platform", string(computer("arch")), "Matlab", "2026b");
            model = struct("Name", "logistic.ode", "Source", m.Source, ...
                "Sha256", m.SourceSha256, "Evaluator", "jit");
            old = fullfile(testCase.Dir, "logistic_v1.pcsession.json");
            phasecont.internal.sessionCodec.writeFile(old, header, model, ...
                S.Records, S.Items);

            back = phasecont.Session.load(old);
            testCase.addTeardown(@() delete(back.Model));
            got = back.Items{1};
            testCase.verifySize(got.Tests, [size(br.Alpha, 1) 4]);
            testCase.verifyEqual(got.Tests(:, 1:2), br.Tests(:, 1:2));
            testCase.verifyTrue(all(isnan(got.Tests(:, 3:4)), "all"));
            testCase.verifyEqual(got.Alpha, br.Alpha);
            testCase.verifyEqual(got.U, br.U);
            testCase.verifyEqual(got.Eigenvalues, br.Eigenvalues);

            testCase.applyFixture(matlab.unittest.fixtures.SuppressedWarningsFixture( ...
                "phasecont:session:runFailed"));
            lastwarn("");
            r = phasecont.session.replay(old);
            [~, id] = lastwarn;
            testCase.verifyEqual(string(id), "phasecont:session:runFailed");
            testCase.verifySize(r, [1 1]);
            testCase.verifyFalse(r.Pass);
        end
    end
end
