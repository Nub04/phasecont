classdef MapCurveTest < matlab.unittest.TestCase
    % Tests slice 3 of map support (SPEC section 8w, test 9): continuation
    % with Type "LP", "PD" or "NS" on a map follows a curve of fixed points
    % in two parameters. Each test states its map and the closed form of the
    % curve, and computes the multipliers itself. Every model is written here.

    properties (Constant)
        % (x, y) -> (1 - a x^2 + y, b x) with F_x = [-2 a x, 1; b, 0]. The LP
        % curve is a = -(1-b)^2/4 with x = 2/(1-b), y = b x and the
        % multipliers 1 and -b. The PD curve is a = 3(1-b)^2/4 with
        % x = 2/(3(1-b)), y = b x and the multipliers -1 and b.
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        % AUTO's form (x, y) -> (y, a - b x - y^2). Its fixed point x = y
        % solves y^2 + (1+b) y - a = 0 and has F_x = [0 1; -b -2y] with
        % determinant b, so the NS curve is b = 1 with x = y = -1 + sqrt(1+a),
        % kappa = -y and the multipliers -y +- i sqrt(1 - y^2).
        HenonAuto = "par a=0.69, b=0.5" + newline + "init x=0.3, y=0.3" + newline + ...
            "x(t+1)=y" + newline + "y(t+1)=a-b*x-y^2" + newline + ...
            "done" + newline
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

        function [m, lp, pd] = henonSpecials(testCase)
            % The Henon branch in a from a = 0.2 at b = 0.3 has an LP below
            % and a PD above.
            b = 0.3;
            m = testCase.model(testCase.Henon);
            x = (-(1 - b) + sqrt((1 - b)^2 + 0.8)) / 0.4;
            eq = phasecont.equilibrium(m, [x b*x] + 0.01, ...
                Params = struct(a = 0.2, b = b));
            up = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [-1 0.5]);
            down = phasecont.continuation(m, eq, Free = "a", Direction = -1, ...
                Hmax = 0.05, MaxSteps = 100);
            pd = up.points("PD");
            lp = down.points("LP");
            testCase.assertSize(pd, [1 1]);
            testCase.assertSize(lp, [1 1]);
        end

        function [m, ns] = nsSpecial(testCase)
            % The AUTO Henon branch in b from b = 0.5 at a = 0.69 has its NS
            % point at b = 1, y = 0.3.
            m = testCase.model(testCase.HenonAuto);
            y = (-1.5 + sqrt(1.5^2 + 4*0.69)) / 2;
            eq = phasecont.equilibrium(m, [y y] + 0.01);
            br = phasecont.continuation(m, eq, Free = "b", Direction = 1, ...
                Hmax = 0.05, MaxSteps = 1000, Bounds = [0 1.5]);
            ns = br.points("NS");
            testCase.assertSize(ns, [1 1]);
            testCase.verifyEqual(ns.P(2), 1, AbsTol = testCase.Tol);
            testCase.verifyEqual(ns.U(:), [0.3; 0.3], AbsTol = testCase.Tol);
        end

        function verifyCurveShape(testCase, br, type)
            % A map curve has the multipliers as Eigenvalues and no tests,
            % codim-2 values, special points or mesh.
            npts = size(br.Alpha, 1);
            testCase.verifyGreaterThan(npts, 5);
            testCase.verifyEqual(br.Type, type);
            testCase.verifySize(br.Alpha, [npts 2]);
            testCase.verifySize(br.U, [npts 2]);
            testCase.verifySize(br.Eigenvalues, [npts 2]);
            testCase.verifySize(br.Tests, [npts 0]);
            testCase.verifySize(br.Codim2, [npts 0]);
            testCase.verifyEmpty(br.Codim2Labels);
            testCase.verifyEmpty(br.Specials);
            testCase.verifyEqual(br.Stable, false(npts, 1));
            testCase.verifyTrue(isnan(br.NTST));
            if type == "NS"
                testCase.verifySize(br.Kappa, [npts 1]);
            else
                testCase.verifyEmpty(br.Kappa);
            end
            testCase.verifyError(@() br.cycle(1), "phasecont:model:badBranch");
        end

        function verifyHenonCurve(testCase, br, type)
            % BR follows the LP (sigma = -1) or PD (sigma = 1) curve of the
            % Henon map in closed form.
            testCase.verifyCurveShape(br, type);
            testCase.verifyEqual(br.Free, ["a" "b"]);
            testCase.verifyEqual(br.FreeIndex, [1 2]);
            sigma = 1;
            if type == "LP"
                sigma = -1;
            end
            a = br.Alpha(:, 1);
            b = br.Alpha(:, 2);
            if sigma < 0
                testCase.verifyEqual(a, -(1 - b).^2/4, AbsTol = testCase.Tol);
                x = 2 ./ (1 - b);
            else
                testCase.verifyEqual(a, 3*(1 - b).^2/4, AbsTol = testCase.Tol);
                x = 2 ./ (3*(1 - b));
            end
            testCase.verifyEqual(br.U, [x b.*x], AbsTol = testCase.Tol);
            for k = 1:numel(b)
                what = type + " at b = " + b(k);
                mu = br.Eigenvalues(k, :);
                J = [-2*a(k)*br.U(k, 1), 1; b(k), 0];
                testCase.verifyEqual(sort(real(mu(:))), sort(eig(J)), ...
                    "AbsTol", testCase.Tol, what);
                testCase.verifyEqual(sort(real(mu(:))), sort([-sigma; sigma*b(k)]), ...
                    "AbsTol", 1e-7, what);
                testCase.verifyEqual(imag(mu(:)), [0; 0], "AbsTol", testCase.Tol, what);
            end
            % The default direction is increasing b, to the bound 0.6.
            testCase.verifyEqual(br.Alpha(end, 2), 0.6, AbsTol = 1e-9);
        end
    end

    methods (Test)
        function lpCurveOfTheHenonMapFollowsTheClosedForm(testCase)
            [m, lp, ~] = testCase.henonSpecials();
            br = phasecont.continuation(m, lp, Type = "LP", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            testCase.verifyHenonCurve(br, "LP");
        end

        function pdCurveOfTheHenonMapFollowsTheClosedForm(testCase)
            [m, ~, pd] = testCase.henonSpecials();
            br = phasecont.continuation(m, pd, Type = "PD", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            testCase.verifyHenonCurve(br, "PD");
        end

        function nsCurveOfTheAutoHenonMapIsBEqual1(testCase)
            [m, ns] = testCase.nsSpecial();
            br = phasecont.continuation(m, ns, Type = "NS", Free = ["b" "a"], ...
                Bounds = [-0.5 2.5]);
            testCase.verifyCurveShape(br, "NS");
            testCase.verifyEqual(br.Free, ["b" "a"]);
            testCase.verifyEqual(br.FreeIndex, [2 1]);
            b = br.Alpha(:, 1);
            a = br.Alpha(:, 2);
            y = -1 + sqrt(1 + a);
            testCase.verifyEqual(b, ones(size(b)), AbsTol = testCase.Tol);
            testCase.verifyEqual(br.U, [y y], AbsTol = testCase.Tol);
            testCase.verifyEqual(br.Kappa, -y, AbsTol = testCase.Tol);
            for k = 1:numel(a)
                what = "NS at a = " + a(k);
                mu = br.Eigenvalues(k, :);
                J = [0 1; -b(k) -2*br.U(k, 2)];
                testCase.verifyEqual(sort(imag(mu(:))), sort(imag(eig(J))), ...
                    "AbsTol", testCase.Tol, what);
                testCase.verifyEqual(real(mu(:)), [-y(k); -y(k)], ...
                    "AbsTol", testCase.Tol, what);
                s = sqrt(1 - y(k)^2);
                testCase.verifyEqual(sort(imag(mu(:))), [-s; s], ...
                    "AbsTol", testCase.Tol, what);
            end
            testCase.verifyGreaterThan(br.Alpha(end, 2), 2.45);
        end

        function aSessionKeepsMapCurvesAndReplays(testCase)
            [m, lp, pd] = testCase.henonSpecials();
            lpc = phasecont.continuation(m, lp, Type = "LP", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            pdc = phasecont.continuation(m, pd, Type = "PD", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            [mn, ns] = testCase.nsSpecial();
            nsc = phasecont.continuation(mn, ns, Type = "NS", Free = ["b" "a"], ...
                Bounds = [-0.5 2.5]);
            files = [fullfile(testCase.Dir, "henon.pcsession.json"), ...
                fullfile(testCase.Dir, "henon_auto.pcsession.json")];
            save(phasecont.Session(m, {lpc, pdc}), files(1));
            save(phasecont.Session(mn, {nsc}), files(2));
            want = {{lpc, pdc}, {nsc}};
            for f = 1:2
                back = phasecont.Session.load(files(f));
                testCase.addTeardown(@() delete(back.Model));
                testCase.assertSize(back.Items, size(want{f}));
                for k = 1:numel(want{f})
                    got = back.Items{k};
                    br = want{f}{k};
                    testCase.verifyEqual(got.Type, br.Type);
                    testCase.verifyEqual(got.Free, br.Free);
                    testCase.verifyEqual(got.Alpha, br.Alpha);
                    testCase.verifyEqual(got.U, br.U);
                    testCase.verifyEqual(got.Eigenvalues, br.Eigenvalues);
                    testCase.verifyEqual(got.Kappa, br.Kappa);
                    testCase.verifyEqual(got.Stable, br.Stable);
                    testCase.verifyCurveShape(got, br.Type);
                end
                r = phasecont.session.replay(files(f));
                testCase.verifySize(r, [1 numel(want{f})]);
                testCase.verifyTrue(all([r.Pass]));
            end
        end

        function optionsOfCyclesAndEpRunsAreRefusedOnAMapCurve(testCase)
            [m, lp, ~] = testCase.henonSpecials();
            free = ["a" "b"];
            bad = {{"Detect", "LP"}, {"Switch", "BP1"}, {"Var", "x"}, ...
                {"NTST", 10}, {"NCOL", 4}, {"MeshAdapt", 1}};
            for type = ["LP" "PD" "NS"]
                for k = 1:numel(bad)
                    testCase.verifyError(@() phasecont.continuation(m, lp, ...
                        bad{k}{:}, Type = type, Free = free), ...
                        "phasecont:model:badOption", ...
                        type + " with " + bad{k}{1});
                end
                testCase.verifyError(@() phasecont.continuation(m, lp, ...
                    Type = type, Free = "a"), "phasecont:model:badOption", type);
            end
        end

        function aStartOffTheCurveIsRefusedByTheEngine(testCase)
            [m, lp, pd] = testCase.henonSpecials();
            free = ["a" "b"];
            cases = {lp, "PD"; pd, "LP"; pd, "NS"; lp, "NS"};
            for k = 1:size(cases, 1)
                start = cases{k, 1};
                type = cases{k, 2};
                try
                    phasecont.continuation(m, start, Type = type, Free = free);
                catch err
                    testCase.verifyEqual(string(err.identifier), ...
                        "phasecont:engine:badProblem", type + " from " + start.Label);
                    if type ~= "NS"
                        testCase.verifySubstring(string(err.message), ...
                            "not near the " + type + " curve");
                    end
                    continue
                end
                testCase.verifyFail(type + " from the " + start.Label + " point ran");
            end
        end

        function cycleTypesAreRefusedOnAMap(testCase)
            % H, LC, LPC and HOM go to the engine, which refuses them with
            % its map text before it reads any other key.
            [m, lp, pd] = testCase.henonSpecials();
            for start = {lp, pd}
                for type = ["H" "LC" "LPC" "HOM"]
                    try
                        phasecont.continuation(m, start{1}, Type = type, ...
                            Free = ["a" "b"]);
                    catch err
                        what = type + " from " + start{1}.Label;
                        testCase.verifyEqual(string(err.identifier), ...
                            "phasecont:engine:unsupportedModel", what);
                        testCase.verifySubstring(string(err.message), ...
                            "does not follow cycles of a map yet", what);
                        continue
                    end
                    testCase.verifyFail("Type " + type + " from a map " + ...
                        start{1}.Label + " point ran");
                end
            end
        end

        function aVersion1MapLpCurveLoadsInItsOldLayout(testCase)
            % v1.0.0 ran an LP curve of a map as the ODE LP curve of F and
            % stored rows of width 3n + 7, with the codim-2 tests CP, BT and
            % ZH before h. Such a file is made here from a curve of this
            % version by adding three NaN rows. It loads in the ODE layout,
            % and its replay gives the new layout, so Pass is false.
            [m, lp, ~] = testCase.henonSpecials();
            br = phasecont.continuation(m, lp, Type = "LP", Free = ["a" "b"], ...
                Bounds = [0.1 0.6]);
            file = fullfile(testCase.Dir, "henon_lp.pcsession.json");
            save(phasecont.Session(m, {br}), file);
            S = phasecont.internal.sessionCodec.readFile(file);
            testCase.assertSize(S.Records, [1 1]);
            rec = S.Records{1};
            n = 2;
            testCase.assertEqual(rec.Stride, 3*n + 4);
            h = rec.Stride - 1;
            points = rec.Points;
            specials = rec.Specials;
            for b = 1:numel(points)
                k = size(points{b}, 2);
                points{b} = [points{b}(1:h-1, :); NaN(3, k); points{b}(h:end, :)];
                k = size(specials{b}, 2);
                specials{b} = [specials{b}(1:h-1, :); NaN(3, k); specials{b}(h:end, :)];
            end
            json = strrep(rec.Json, """stride"":" + rec.Stride, ...
                """stride"":" + (rec.Stride + 3));
            testCase.assertNotEqual(json, rec.Json);
            f = struct("Kind", rec.Kind, "Problem", rec.Problem, "U0", rec.U0, ...
                "P", rec.P, "ModelSha", rec.ModelSha, "Type", rec.Type, ...
                "Free", rec.Free, "Grid", rec.Grid, "N", rec.N, "Json", json, ...
                "Points", {points}, "Specials", {specials}, ...
                "Stride", rec.Stride + 3);
            S.Records{1} = phasecont.internal.RunRecord(f);
            header = struct("Created", "2026-11-14T00:00:00Z", ...
                "EngineName", "pckern", "EngineVersion", "1.0.0", ...
                "Platform", string(computer("arch")), "Matlab", "2026b");
            model = struct("Name", "henon.ode", "Source", m.Source, ...
                "Sha256", m.SourceSha256, "Evaluator", "jit");
            old = fullfile(testCase.Dir, "henon_lp_v1.pcsession.json");
            phasecont.internal.sessionCodec.writeFile(old, header, model, ...
                S.Records, S.Items);

            back = phasecont.Session.load(old);
            testCase.addTeardown(@() delete(back.Model));
            got = back.Items{1};
            npts = size(br.Alpha, 1);
            testCase.verifyEqual(got.Type, "LP");
            testCase.verifyEqual(got.Alpha, br.Alpha);
            testCase.verifyEqual(got.U, br.U);
            testCase.verifyEqual(got.Eigenvalues, br.Eigenvalues);
            testCase.verifyEqual(got.Codim2Labels, ["CP" "BT" "ZH"]);
            testCase.verifySize(got.Codim2, [npts 3]);
            testCase.verifyTrue(all(isnan(got.Codim2), "all"));

            testCase.applyFixture(matlab.unittest.fixtures.SuppressedWarningsFixture( ...
                "phasecont:session:runFailed"));
            r = phasecont.session.replay(old);
            testCase.verifySize(r, [1 1]);
            testCase.verifyFalse(r.Pass);
        end

        function mapCurveRowsByHand(testCase)
            % rows2branch on a map model with an empty grid reads the map
            % layouts: [a2, x(1..n), a1, mu_re, mu_im, h, it] for LP and PD
            % and [a2, x, a1, kappa, mu_re, mu_im, h, it] for NS.
            m = testCase.model(testCase.Henon);
            st = struct("points", 2, "stop", "Bounds", "warnings", [], ...
                "specials", []);
            pts = [ ...
                0.3, 1 2, -0.1, 1 -0.3, 0 0, 0.01, 2; ...
                0.4, 3 4, -0.2, 1 -0.4, 0 0, 0.02, 3].';
            for kind = ["LP" "PD"]
                br = phasecont.internal.rows2branch(m, [1 2], [0; 0], 2, ...
                    pts, zeros(10, 0), st, kind, []);
                testCase.verifyEqual(br.Type, kind);
                testCase.verifyEqual(br.Alpha, [-0.1 0.3; -0.2 0.4]);
                testCase.verifyEqual(br.U, [1 2; 3 4]);
                testCase.verifyEqual(br.Eigenvalues, ...
                    complex([1 -0.3; 1 -0.4], 0));
                testCase.verifySize(br.Tests, [2 0]);
                testCase.verifySize(br.Codim2, [2 0]);
                testCase.verifyEmpty(br.Codim2Labels);
                testCase.verifyEqual(br.Stable, false(2, 1));
                testCase.verifyEqual(br.StepSize, [0.01; 0.02]);
                testCase.verifyEqual(br.NewtonIters, [2; 3]);
                testCase.verifyTrue(isnan(br.NTST));
            end
            ns = [ ...
                0.3, 1 2, -0.1, 0.5, 0.5 0.5, 0.8 -0.8, 0.01, 2; ...
                0.4, 3 4, -0.2, 0.6, 0.6 0.6, 0.7 -0.7, 0.02, 3].';
            sst = st;
            sst.specials = struct("label", "NS", "index", 1);
            br = phasecont.internal.rows2branch(m, [1 2], [0; 0], 2, ...
                ns, ns(:, 2), sst, "NS", []);
            testCase.verifyEqual(br.Type, "NS");
            testCase.verifyEqual(br.Alpha, [-0.1 0.3; -0.2 0.4]);
            testCase.verifyEqual(br.Kappa, [0.5; 0.6]);
            testCase.verifyEqual(br.Eigenvalues, ...
                [complex(0.5, 0.8) complex(0.5, -0.8); ...
                 complex(0.6, 0.7) complex(0.6, -0.7)]);
            testCase.verifySize(br.Codim2, [2 0]);
            testCase.verifyEqual(br.Specials(1).Data.kappa, 0.6);
            testCase.verifyEqual(br.Specials(1).U, [3; 4]);
        end

        function aVersion1MapCycleCurveKeepsTheCycleLayout(testCase)
            % v1.0.0 stored a PD or NS curve of a map as a curve of cycles
            % with a grid. On a map model with a grid, rows2branch keeps that
            % layout, so the Branch has a mesh and its cycles.
            m = testCase.model(testCase.Henon);
            % [a2, orbit(10), T, a1, mesh(3), mult_re(2), mult_im(2),
            % LPPD PDNS GPD R2, h, it] with n = 2, NTST = 2, NCOL = 2.
            pts = [ ...
                0.5, 1 2 2 3 3 4 2 1 1 2, 6, -0.1, 0 0.4 1, -1 1, 0 0, 0.1 0.2 0.3 0.4, 0.01, 2; ...
                0.6, 1 2 5 6 7 8 6 5 1 2, 7, -0.2, 0 0.5 1, -1 1, 0 0, NaN NaN NaN NaN, 0.02, 3].';
            st = struct("points", 2, "stop", "Bounds", "warnings", [], ...
                "specials", []);
            br = phasecont.internal.rows2branch(m, [1 2], [0; 0], 2, ...
                pts, zeros(26, 0), st, "PD", [2 2]);
            testCase.verifyEqual(br.Type, "PD");
            testCase.verifyEqual([br.NTST br.NCOL], [2 2]);
            testCase.verifyEqual(br.Period, [6; 7]);
            testCase.verifyEqual(br.Codim2Labels, ["LPPD" "PDNS" "GPD" "R2"]);
            [t, x] = br.cycle(1);
            testCase.verifyEqual(t, [0; 0.2; 0.4; 0.7; 1] * 6, AbsTol = 1e-15);
            testCase.verifyEqual(x, [1 2; 2 3; 3 4; 2 1; 1 2]);
        end
    end
end
