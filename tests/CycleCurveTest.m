classdef CycleCurveTest < matlab.unittest.TestCase
    % Tests curves of LPC and PD points through phasecont.continuation(Type
    % = "LPC" | "PD"): the start at a special point of an LC branch, the
    % Branch layout of a curve of cycles, the exact LPC curve of a Bautin
    % model, the PD curve of the Rossler system, the fold curve of the
    % Morris-Lecar model and the option errors. Every model is written here.
    %
    % The Bautin model r' = r (b1 + b2 r^2 - r^4), theta' = 1 + g r^2 has
    % the LPC curve b1 = -b2^2/4 with r^2 = b2/2 and T = 2 pi/(1 + g b2/2).
    % The Rossler and Morris-Lecar values come from pc-cli runs of the same
    % problems, and crates/pc-engine/tests/cycle_curve_models.rs checks these
    % curves against LC runs and an RK4 shooting oracle. The oracle files of
    % SPEC section 14 do not exist yet, so the tests that read them are
    % skipped as "oracle missing".

    properties (Constant)
        Bautin = "par b1=-0.25,b2=1,g=0.5" + newline + ...
            "x'=(b1+b2*(x^2+y^2)-(x^2+y^2)^2)*x-(1+g*(x^2+y^2))*y" + newline + ...
            "y'=(1+g*(x^2+y^2))*x+(b1+b2*(x^2+y^2)-(x^2+y^2)^2)*y" + newline + ...
            "init x=0.7,y=0" + newline + "done" + newline
        Rossler = "par a=0.2,b=0.2,c=2.5" + newline + ...
            "x'=-y-z" + newline + "y'=x+a*y" + newline + ...
            "z'=b+z*(x-c)" + newline + "init x=0,y=0,z=0" + newline + ...
            "done" + newline
        MorrisLecar = "par I=100,phi=0.04" + newline + ...
            "par C=20,gL=2,gK=8,gCa=4.4" + newline + ...
            "par vL=-60,vK=-84,vCa=120" + newline + ...
            "par v1=-1.2,v2=18,v3=2,v4=30" + newline + ...
            "minf=0.5*(1+tanh((v-v1)/v2))" + newline + ...
            "winf=0.5*(1+tanh((v-v3)/v4))" + newline + ...
            "tauw=1/cosh((v-v3)/(2*v4))" + newline + ...
            "v'=(I-gL*(v-vL)-gK*w*(v-vK)-gCa*minf*(v-vCa))/C" + newline + ...
            "w'=phi*(winf-w)/tauw" + newline + ...
            "init v=-20,w=0.1" + newline + "done" + newline
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

        function [m, lpc] = bautinFold(testCase)
            % The LC branch from the Hopf point at b1 = 0 (b2 = 1) turns at
            % the LPC point b1 = -1/4.
            m = testCase.model(testCase.Bautin);
            eq = phasecont.equilibrium(m, [0 0]);
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-1 1]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "b1", ...
                NTST = 40, NCOL = 4, Bounds = [-0.3 0.1], MaxSteps = 400);
            lpc = lc.points("LPC");
            testCase.assertNumElements(lpc, 1);
        end

        function [m, pd] = rosslerFlip(testCase)
            % The LC branch in a at c = 2.5 has its PD point near a = 0.226.
            m = testCase.model(testCase.Rossler);
            eq = phasecont.equilibrium(m, [0 -0.08 0.08], ...
                Params = struct(a = 0.01, b = 0.2, c = 2.5));
            ep = phasecont.continuation(m, eq, Free = "a", Bounds = [0 0.4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "a", ...
                NTST = 40, NCOL = 4, Bounds = [0.02 0.4], MaxSteps = 400);
            pd = lc.points("PD");
            testCase.assertNumElements(pd, 1);
        end

        function [m, lpc] = morrisLecarFold(testCase)
            % At phi = 0.04 the rest state loses stability in a subcritical
            % Hopf point near I = 93.86, and its cycles fold at I = 88.29.
            m = testCase.model(testCase.MorrisLecar);
            eq = phasecont.equilibrium(m, [-36.75 0.07], ...
                Params = struct(I = 60, phi = 0.04));
            ep = phasecont.continuation(m, eq, Free = "I", Bounds = [60 120]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "I", ...
                NTST = 160, NCOL = 4, Bounds = [80 120], Hmax = 0.5, ...
                MaxNewton = 15, MaxSteps = 300);
            lpc = lc.points("LPC");
            testCase.assertNotEmpty(lpc);
            lpc = lpc(1);
        end

        function pd = rosslerCurve(~, m, flip, a)
            % The PD curve in (c, a) from FLIP up to A, or down to it.
            a0 = flip.P(1);
            pd = phasecont.continuation(m, flip, Type = "PD", Free = ["c" "a"], ...
                Bounds = sort([a, 2*a0 - a]), Direction = sign(a - a0), ...
                MaxNewton = 15, Hmax = 0.05);
        end

        function c = morrisLecarCurve(~, m, fold, phi)
            % The LPC curve in (I, phi) from FOLD up to PHI.
            c = phasecont.continuation(m, fold, Type = "LPC", Free = ["I" "phi"], ...
                Bounds = [0.02 phi], Direction = 1, Hmax = 0.5, ...
                MaxNewton = 15, MaxSteps = 200);
        end

        function ref = oracle(testCase, name, free)
            % The points of references/NAME.json, a curve oracle of SPEC
            % section 14 made with AUTO-07p or MatCont on the developer's
            % machine: {"tool", "version", "settings", "free": [a1, a2],
            %  "points": [{a1: value, a2: value, "period": value}, ...]}.
            root = fileparts(fileparts(mfilename("fullpath")));
            file = fullfile(root, "references", name + ".json");
            testCase.assumeTrue(isfile(file), ...
                "oracle missing: references/" + name + ".json");
            ref = jsondecode(fileread(file));
            testCase.assertEqual(reshape(string(ref.free), 1, []), free);
            ref = ref.points;
            if iscell(ref)
                ref = [ref{:}];
            end
            testCase.assertNotEmpty(ref, "the oracle lists no point");
        end
    end

    methods (Test)
        function bautinLpcCurveIsExact(testCase)
            [m, lpc] = testCase.bautinFold();
            testCase.verifyEqual(lpc.P(1), -0.25, AbsTol = 1e-6);
            testCase.verifyEqual(lpc.Label, "LPC");

            c = phasecont.continuation(m, lpc, Type = "LPC", Free = ["b1" "b2"], ...
                Bounds = [0.5 1.5], Hmax = 0.05, MaxNewton = 15);
            testCase.verifyClass(c, "phasecont.Branch");
            testCase.verifyEqual(c.Type, "LPC");
            testCase.verifyEqual(c.Free, ["b1" "b2"]);
            testCase.verifyEqual(c.Stop, "Bounds");
            % NTST and NCOL default to those of the start point.
            testCase.verifyEqual([c.NTST c.NCOL], [40 4]);
            npts = size(c.Alpha, 1);
            testCase.verifyGreaterThan(npts, 5);
            testCase.verifySize(c.Alpha, [npts 2]);
            testCase.verifySize(c.U, [npts 2*(40*4 + 1)]);
            testCase.verifySize(c.Mesh, [npts 41]);
            testCase.verifySize(c.Period, [npts 1]);
            testCase.verifySize(c.Eigenvalues, [npts 2]);
            testCase.verifySize(c.Tests, [npts 0]);
            testCase.verifyEqual(c.Codim2Labels, ["LPPD" "LPNS" "CPC" "R1"]);
            testCase.verifySize(c.Codim2, [npts 4]);
            testCase.verifySize(c.Kappa, [0 1]);
            testCase.verifyEqual(c.Alpha(1, 2), 1, AbsTol = 1e-12);
            testCase.verifyTrue(any(abs(c.Alpha(end, 2) - [0.5 1.5]) < 1e-10));

            % b1 = -b2^2/4, T = 2 pi/(1 + b2/4) and r^2 = b2/2 along the curve.
            b1 = c.Alpha(:, 1);
            b2 = c.Alpha(:, 2);
            testCase.verifyLessThan(max(abs(b1 + b2.^2/4)), 1e-8);
            testCase.verifyEqual(c.Period, 2*pi./(1 + b2/4), RelTol = 1e-8);
            r2 = c.U(:, 1:2:end).^2 + c.U(:, 2:2:end).^2;
            testCase.verifyLessThan(max(abs(r2 - b2/2), [], "all"), 1e-8);
            % A fold of cycles has a double multiplier 1.
            testCase.verifyLessThan(max(abs(c.Eigenvalues - 1), [], "all"), 1e-5);

            [t, x] = c.cycle(npts);
            testCase.verifyEqual(t([1 end]), [0; c.Period(npts)], AbsTol = 1e-12);
            testCase.verifyEqual(x(end, :), x(1, :), AbsTol = 1e-8);
        end

        function bautinLpcCurveOnAnotherMesh(testCase)
            % Other NTST and NCOL than those of the start point regrid its orbit.
            [m, lpc] = testCase.bautinFold();
            c = phasecont.continuation(m, lpc, Type = "LPC", Free = ["b1" "b2"], ...
                NTST = 20, NCOL = 3, Bounds = [0.9 1.1], Hmax = 0.05, MaxNewton = 15);
            testCase.verifyEqual(c.Stop, "Bounds");
            testCase.verifyEqual([c.NTST c.NCOL], [20 3]);
            testCase.verifySize(c.Mesh, [size(c.Alpha, 1) 21]);
            testCase.verifySize(c.U, [size(c.Alpha, 1) 2*(20*3 + 1)]);
            testCase.verifyEqual(c.Alpha(end, 2), 1.1, AbsTol = 1e-10);
            b2 = c.Alpha(:, 2);
            testCase.verifyLessThan(max(abs(c.Alpha(:, 1) + b2.^2/4)), 1e-8);
            testCase.verifyEqual(c.Period, 2*pi./(1 + b2/4), RelTol = 1e-6);
            r2 = c.U(:, 1:2:end).^2 + c.U(:, 2:2:end).^2;
            testCase.verifyLessThan(max(abs(r2 - b2/2), [], "all"), 1e-4);
        end

        function rosslerPdCurveMeetsTheEngineValues(testCase)
            [m, flip] = testCase.rosslerFlip();
            testCase.verifyGreaterThan(flip.P(1), 0.2);
            testCase.verifyLessThan(flip.P(1), 0.25);
            % The engine finds the PD curve at c = 2.2814357594 for a = 0.25
            % and at c = 2.8324450280 for a = 0.2 (b = 0.2, N = 40, m = 4).
            for want = [0.25 2.2814357594; 0.2 2.8324450280].'
                pd = testCase.rosslerCurve(m, flip, want(1));
                testCase.verifyEqual(pd.Type, "PD");
                testCase.verifyEqual(pd.Stop, "Bounds");
                testCase.verifyEqual([pd.NTST pd.NCOL], [40 4]);
                testCase.verifyEqual(pd.Codim2Labels, ["LPPD" "PDNS" "GPD" "R2"]);
                testCase.verifyEqual(pd.Alpha(1, :), [2.5 flip.P(1)], AbsTol = 1e-10);
                testCase.verifyEqual(pd.Alpha(end, 2), want(1), AbsTol = 1e-10);
                testCase.verifyEqual(pd.Alpha(end, 1), want(2), AbsTol = 1e-6);
                % One multiplier is -1 at every point.
                testCase.verifyLessThan( ...
                    max(min(abs(pd.Eigenvalues + 1), [], 2)), 1e-8);
                [~, x] = pd.cycle(size(pd.Alpha, 1));
                testCase.verifyEqual(x(end, :), x(1, :), AbsTol = 1e-8);
            end
        end

        function morrisLecarLpcCurveMeetsTheEngineValues(testCase)
            [m, fold] = testCase.morrisLecarFold();
            testCase.verifyEqual(fold.P(1), 88.2932505431, AbsTol = 1e-6);
            c = testCase.morrisLecarCurve(m, fold, 0.06);
            testCase.verifyEqual(c.Stop, "Bounds");
            testCase.verifyEqual([c.NTST c.NCOL], [160 4]);
            testCase.verifyEqual(c.Alpha(end, 2), 0.06, AbsTol = 1e-10);
            testCase.verifyEqual(c.Alpha(end, 1), 91.2171895691, AbsTol = 1e-6);
            testCase.verifyEqual(c.Period(end), 101.4575045510, RelTol = 1e-6);
        end

        function rosslerPdOracle(testCase)
            ref = testCase.oracle("rossler_pd", ["c" "a"]);
            [m, flip] = testCase.rosslerFlip();
            for k = 1:numel(ref)
                pd = testCase.rosslerCurve(m, flip, ref(k).a);
                testCase.verifyEqual(pd.Alpha(end, 2), ref(k).a, AbsTol = 1e-10);
                testCase.verifyEqual(pd.Alpha(end, 1), ref(k).c, RelTol = 1e-6);
                testCase.verifyEqual(pd.Period(end), ref(k).period, RelTol = 1e-6);
            end
        end

        function morrisLecarLpcOracle(testCase)
            ref = testCase.oracle("morris_lecar_lpc", ["I" "phi"]);
            [m, fold] = testCase.morrisLecarFold();
            for k = 1:numel(ref)
                c = testCase.morrisLecarCurve(m, fold, ref(k).phi);
                testCase.verifyEqual(c.Alpha(end, 2), ref(k).phi, AbsTol = 1e-10);
                testCase.verifyEqual(c.Alpha(end, 1), ref(k).I, RelTol = 1e-6);
                testCase.verifyEqual(c.Period(end), ref(k).period, RelTol = 1e-6);
            end
        end

        function optionErrors(testCase)
            m = testCase.model(testCase.Bautin);
            % A start point with the layout of an LPC point on a mesh of
            % 2 intervals of 2 points. No error below needs the engine.
            u = zeros(2*(2*2 + 1), 1);
            data = struct("period", 2*pi, "mesh", [0 0.5 1], "ntst", 2, "ncol", 2);
            lpc = phasecont.Point(U = u, P = [-0.25; 1; 0.5], Label = "LPC", ...
                Data = data, Model = m);
            free = ["b1" "b2"];
            bad = "phasecont:model:badOption";
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = "b1"), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = ["b1" "b1"]), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = free, Detect = "LP"), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = free, Switch = "BP1"), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = free, Var = "x"), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = free, NTST = 0), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = free, NCOL = 2.5), bad);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "LPC", Free = ["b1" "q"]), ...
                "phasecont:model:unknownParameter");

            start = "phasecont:model:badStart";
            % The label must match the type of the curve.
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "PD", Free = free), start);
            testCase.verifyError(@() phasecont.continuation(m, lpc, ...
                Type = "NS", Free = free), start);
            % A state vector or a point without the cycle data is no start.
            testCase.verifyError(@() phasecont.continuation(m, [0 0], ...
                Type = "LPC", Free = free), start);
            plain = phasecont.Point(U = u, P = [-0.25; 1; 0.5], Label = "LPC", ...
                Data = struct("period", 2*pi), Model = m);
            testCase.verifyError(@() phasecont.continuation(m, plain, ...
                Type = "LPC", Free = free), start);
            short = phasecont.Point(U = u, P = [-0.25; 1], Label = "LPC", ...
                Data = data, Model = m);
            testCase.verifyError(@() phasecont.continuation(m, short, ...
                Type = "LPC", Free = free), start);
            other = testCase.model(testCase.Bautin);
            testCase.verifyError(@() phasecont.continuation(other, lpc, ...
                Type = "LPC", Free = free), start);
        end
    end
end
