classdef CurveTest < matlab.unittest.TestCase
    % Tests the LP and Hopf curves of phasecont.continuation: the codim-2
    % labels with their coefficients, the Branch layout of a curve, the plot in
    % the (a1, a2) plane and the option errors. Every model is written here.

    properties (Constant)
        Cusp = "par b1=0,b2=1" + newline + "x'=b1+b2*x-x^3" + newline + ...
            "init x=-1" + newline + "done" + newline
        Bt = "par b1=0,b2=0,s=1" + newline + "x'=y" + newline + ...
            "y'=b1+b2*y+x^2+s*x*y" + newline + "init x=0,y=0" + newline + ...
            "done" + newline
        Brusselator = "par a=1,b=1.5" + newline + ...
            "x'=a-(b+1)*x+x^2*y" + newline + "y'=b*x-x^2*y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        % Normal form z' = (b1 + i) z + b2 z|z|^2 + s z|z|^4 in z = x + i y.
        Bautin = "par b1=0,b2=0.5,s=1" + newline + ...
            "x'=b1*x-y+b2*x*(x^2+y^2)+s*x*(x^2+y^2)^2" + newline + ...
            "y'=x+b1*y+b2*y*(x^2+y^2)+s*y*(x^2+y^2)^2" + newline + ...
            "init x=0,y=0" + newline + "done" + newline
        % Zero-Hopf normal form with cubic terms; the origin at b1 = b2 = 0 has
        % b = B, c = 2 C, d = dr + i (di - om K / B) and e from the formula
        % in the engine tests.
        ZeroHopf = "par b1=0,b2=0,B=1.5,C=0.8,K=0.3,L=0.6,om=2,dr=1,di=0.4," + ...
            "er=0.5,ei=-0.3,mr=-0.4,mi=0.7" + newline + ...
            "rr=y^2+z^2" + newline + ...
            "al=b2+dr*x+er*x^2+mr*rr" + newline + ...
            "be=om+di*x+ei*x^2+mi*rr" + newline + ...
            "x'=b1+B*x^2+C*rr+K*x^3+L*x*rr" + newline + ...
            "y'=al*y-be*z" + newline + "z'=al*z+be*y" + newline + ...
            "init x=0,y=0,z=0" + newline + "done" + newline
        % Two Hopf normal forms with cubic coupling, w1 = x1 + i y1 and
        % w2 = x2 + i y2.
        DoubleHopf = "par b1=0,b2=0,om1=2,om2=0.9,a11r=-0.8,a11i=0.3," + ...
            "a12r=1.7,a12i=-0.4,a21r=0.6,a21i=0.5,a22r=-1.1,a22i=0.2" + newline + ...
            "r1=x1^2+y1^2" + newline + "r2=x2^2+y2^2" + newline + ...
            "al1=b1+a11r*r1+a12r*r2" + newline + ...
            "be1=om1+a11i*r1+a12i*r2" + newline + ...
            "al2=b2+a21r*r1+a22r*r2" + newline + ...
            "be2=om2+a21i*r1+a22i*r2" + newline + ...
            "x1'=al1*x1-be1*y1" + newline + "y1'=al1*y1+be1*x1" + newline + ...
            "x2'=al2*x2-be2*y2" + newline + "y2'=al2*y2+be2*x2" + newline + ...
            "init x1=0,y1=0,x2=0,y2=0" + newline + "done" + newline
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

        function [m, lp] = cuspFold(testCase)
            % The cusp model with the fold of its equilibrium branch in b1.
            m = testCase.model(testCase.Cusp);
            eq = phasecont.equilibrium(m, -1);
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-1 1]);
            lp = ep.points("LP");
            testCase.assertNotEmpty(lp);
            lp = lp(1);
        end

        function [m, h] = brusselatorHopf(testCase)
            m = testCase.model(testCase.Brusselator);
            eq = phasecont.equilibrium(m, [1 1.2], Params = struct(a = 1, b = 1.2));
            ep = phasecont.continuation(m, eq, Free = "b", Bounds = [0.5 4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            h = h(1);
        end

        function sp = zeroHopf(testCase, type)
            % The ZH point of the LP or Hopf curve of the zero-Hopf model.
            m = testCase.model(testCase.ZeroHopf);
            if type == "LP"
                eq = phasecont.equilibrium(m, [-1 0 0], ...
                    Params = struct(b1 = -1.2, b2 = 0.5));
                ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-2 1]);
                start = ep.points("LP");
                opts = {"Direction", -1, "Bounds", [-0.5 1]};
            else
                eq = phasecont.equilibrium(m, [-1 0 0], ...
                    Params = struct(b1 = -1.2, b2 = 0.3));
                ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-2 1]);
                start = ep.points("H");
                opts = {"Bounds", [-0.5 1]};
            end
            testCase.assertNotEmpty(start);
            br = phasecont.continuation(m, start(1), "Type", type, ...
                "Free", ["b1" "b2"], opts{:});
            sp = br.points("ZH");
            testCase.assertSize(sp, [1 1]);
        end
    end

    methods (Test)
        function lpCurveOfTheCuspFindsTheCusp(testCase)
            [m, lp] = testCase.cuspFold();
            br = phasecont.continuation(m, lp, Type = "LP", ...
                Free = ["b1" "b2"], Bounds = [-0.5 1.5]);
            testCase.verifyEqual(br.Type, "LP");
            testCase.verifyEqual(br.Free, ["b1" "b2"]);
            testCase.verifyEqual(br.FreeIndex, [1 2]);
            N = size(br.Alpha, 1);
            testCase.verifyGreaterThan(N, 5);
            testCase.verifySize(br.Alpha, [N 2]);
            testCase.verifySize(br.U, [N 1]);
            testCase.verifySize(br.Eigenvalues, [N 1]);
            testCase.verifySize(br.Codim2, [N 3]);
            testCase.verifyEqual(br.Codim2Labels, ["CP" "BT" "ZH"]);
            testCase.verifySize(br.Tests, [N 0]);
            % The folds of x' = b1 + b2 x - x^3 are 27 b1^2 = 4 b2^3.
            b1 = br.Alpha(:, 1);
            b2 = br.Alpha(:, 2);
            testCase.verifyEqual(27 * b1.^2, 4 * b2.^3, AbsTol = 1e-8);
            cp = br.points("CP");
            testCase.assertSize(cp, [1 1]);
            fprintf("cusp CP: b1 %.2e, b2 %.2e, x %.2e, coefficient %.12f\n", ...
                cp.P(1), cp.P(2), cp.U, cp.Data.coefficient);
            testCase.verifyEqual(cp.P(1:2), [0; 0], AbsTol = 1e-8);
            testCase.verifyEqual(cp.U, 0, AbsTol = 1e-8);
            testCase.verifyEqual(cp.Data.coefficient, -1, AbsTol = 1e-8);
            testCase.verifyTrue(cp.Data.refined);
            testCase.verifyTrue(isnan(cp.Data.coefficient2));
        end

        function lpCurveOfTheBtModelFindsTheBt(testCase)
            m = testCase.model(testCase.Bt);
            eq = phasecont.equilibrium(m, [-1 0], ...
                Params = struct(b1 = -1, b2 = 0.5, s = 1));
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-1 1]);
            lp = ep.points("LP");
            testCase.assertNotEmpty(lp);
            br = phasecont.continuation(m, lp(1), Type = "LP", ...
                Free = ["b1" "b2"], Direction = -1, Bounds = [-0.5 1]);
            bt = br.points("BT");
            testCase.assertSize(bt, [1 1]);
            fprintf("BT: a %.12f, b %.12f\n", bt.Data.coefficient, ...
                bt.Data.coefficient2);
            testCase.verifyEqual(bt.U, [0; 0], AbsTol = 1e-8);
            testCase.verifyEqual(bt.P(1:2), [0; 0], AbsTol = 1e-8);
            testCase.verifyEqual(bt.Data.coefficient, 1, AbsTol = 1e-8);
            testCase.verifyEqual(bt.Data.coefficient2, 1, AbsTol = 1e-8);
            testCase.verifyTrue(isstruct(bt.Data.values));
        end

        function hopfCurveOfTheBrusselatorFollowsBEqualsOnePlusASquared(testCase)
            [m, h] = testCase.brusselatorHopf();
            br = phasecont.continuation(m, h, Type = "H", ...
                Free = ["a" "b"], Bounds = [1.5 5]);
            testCase.verifyEqual(br.Type, "H");
            N = size(br.Alpha, 1);
            testCase.verifyGreaterThan(N, 5);
            testCase.verifySize(br.Alpha, [N 2]);
            testCase.verifySize(br.U, [N 2]);
            testCase.verifySize(br.Eigenvalues, [N 2]);
            testCase.verifySize(br.Codim2, [N 4]);
            testCase.verifyEqual(br.Codim2Labels, ["GH" "BT" "ZH" "HH"]);
            a = br.Alpha(:, 1);
            b = br.Alpha(:, 2);
            testCase.verifyEqual(b, 1 + a.^2, AbsTol = 1e-8);
            % A Hopf point has the pair +-i sqrt(kappa) with kappa = a^2.
            re = real(br.Eigenvalues);
            testCase.verifyLessThan(max(abs(re), [], 2), 1e-6 * ones(N, 1));
            testCase.verifyTrue(all(br.Stable == false));
            testCase.verifyEmpty(br.points());
        end

        function hopfCurveOfTheBautinModelFindsTheGh(testCase)
            m = testCase.model(testCase.Bautin);
            eq = phasecont.equilibrium(m, [0 0], ...
                Params = struct(b1 = -0.5, b2 = 0.5, s = 1));
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-1 1]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            br = phasecont.continuation(m, h(1), Type = "H", ...
                Free = ["b1" "b2"], Direction = -1, Bounds = [-0.5 0.8]);
            gh = br.points("GH");
            testCase.assertSize(gh, [1 1]);
            fprintf("GH: b1 %.2e, b2 %.2e, l2 %.6f, omega %.12f\n", ...
                gh.P(1), gh.P(2), gh.Data.coefficient, gh.Data.omega);
            testCase.verifyEqual(gh.P(1:2), [0; 0], AbsTol = 1e-8);
            testCase.verifyEqual(gh.U, [0; 0], AbsTol = 1e-8);
            testCase.verifyEqual(gh.Data.omega, 1, AbsTol = 1e-8);
            % l2 = 4 s for s = 1.
            testCase.verifyGreaterThan(gh.Data.coefficient, 0);
            testCase.verifyTrue(gh.Data.refined);
        end

        function zeroHopfOnTheLpAndTheHopfCurve(testCase)
            % b = B, c = 2 C, Re d = dr, Im d = di - om K / B and e = er +
            % dr s / 2 - mr B / C with s = 2 mr / C - 3 K / B + L / C.
            B = 1.5; C = 0.8; K = 0.3; L = 0.6; om = 2; dr = 1; di = 0.4; er = 0.5;
            mr = -0.4;
            s = 2 * mr / C - 3 * K / B + L / C;
            want = struct(b = B, c = 2 * C, re_d = dr, im_d = di - om * K / B, ...
                e = er + dr * s / 2 - mr * B / C);
            for type = ["LP" "H"]
                sp = testCase.zeroHopf(type);
                fprintf("ZH on %s: b %.10f, c %.10f, Re d %.10f, Im d %.10f, e %.10f\n", ...
                    type, sp.Data.coefficient, sp.Data.coefficient2, ...
                    sp.Data.values.re_d, sp.Data.values.im_d, sp.Data.values.e);
                testCase.verifyEqual(sp.U, [0; 0; 0], AbsTol = 1e-8);
                testCase.verifyEqual(sp.P(1:2), [0; 0], AbsTol = 1e-8);
                testCase.verifyTrue(sp.Data.refined);
                testCase.verifyEqual(sp.Data.coefficient, want.b, AbsTol = 1e-8);
                testCase.verifyEqual(sp.Data.coefficient2, want.c, AbsTol = 1e-8);
                testCase.verifyEqual(sp.Data.values.re_d, want.re_d, AbsTol = 1e-8);
                testCase.verifyEqual(sp.Data.values.im_d, want.im_d, AbsTol = 1e-8);
                testCase.verifyEqual(sp.Data.values.e, want.e, AbsTol = 1e-8);
                testCase.verifyEqual(sp.Data.omega, om, AbsTol = 1e-8);
                testCase.verifySubstring(sp.Data.normalization, "b>0");
            end
        end

        function doubleHopfOnTheHopfCurve(testCase)
            % Re P11 = 4 a11r, Re P12 = 2 a12r, Re P21 = 2 a21r and
            % Re P22 = 4 a22r, because om1 > om2.
            m = testCase.model(testCase.DoubleHopf);
            eq = phasecont.equilibrium(m, zeros(1, 4), ...
                Params = struct(b1 = -0.3, b2 = 0.4));
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-0.5 0.5]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            br = phasecont.continuation(m, h(1), Type = "H", ...
                Free = ["b1" "b2"], Direction = -1, Bounds = [-0.5 1]);
            hh = br.points("HH");
            testCase.assertSize(hh, [1 1]);
            d = hh.Data;
            fprintf("HH: P11 %.8f, P12 %.8f, P21 %.8f, P22 %.8f, omega %.8f, %.8f\n", ...
                d.coefficient, d.values.re_p12, d.values.re_p21, ...
                d.values.re_p22, d.omega, d.values.omega2);
            testCase.verifyEqual(hh.P(1:2), [0; 0], AbsTol = 1e-8);
            testCase.verifyTrue(d.refined);
            testCase.verifyEqual(d.coefficient, 4 * -0.8, AbsTol = 1e-8);
            testCase.verifyEqual(d.values.re_p12, 2 * 1.7, AbsTol = 1e-8);
            testCase.verifyEqual(d.values.re_p21, 2 * 0.6, AbsTol = 1e-8);
            testCase.verifyEqual(d.values.re_p22, 4 * -1.1, AbsTol = 1e-8);
            testCase.verifyEqual(d.omega, 2, AbsTol = 1e-8);
            testCase.verifyEqual(d.values.omega2, 0.9, AbsTol = 1e-8);
            testCase.verifySubstring(d.normalization, "w1>w2");
        end

        function aStateVectorStartsACurveToo(testCase)
            [m, lp] = testCase.cuspFold();
            br = phasecont.continuation(m, lp.U, Type = "LP", ...
                Free = ["b1" "b2"], Params = lp.P, Bounds = [-0.5 1.5]);
            testCase.verifyEqual(br.Type, "LP");
            testCase.verifyNotEmpty(br.points("CP"));
        end

        function curveBranchPlotsInTheParameterPlane(testCase)
            [m, lp] = testCase.cuspFold();
            br = phasecont.continuation(m, lp, Type = "LP", ...
                Free = ["b1" "b2"], Bounds = [-0.5 1.5]);
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            ax = axes(fig);
            plot(br, Parent = ax);
            line = findobj(ax, "Tag", "pc-curve");
            testCase.assertSize(line, [1 1]);
            testCase.verifyEqual(line.XData(:), br.Alpha(:, 1));
            testCase.verifyEqual(line.YData(:), br.Alpha(:, 2));
            testCase.verifyEmpty(findobj(ax, "Tag", "pc-stable"));
            testCase.verifyEmpty(findobj(ax, "Tag", "pc-unstable"));
            marks = findobj(ax, "Tag", "pc-special");
            labels = findobj(ax, "Tag", "pc-label");
            testCase.verifySize(marks, [numel(br.Specials) 1]);
            testCase.verifySize(labels, [numel(br.Specials) 1]);
            testCase.verifyTrue(ismember("CP", string({labels.String})));
            testCase.verifyEqual(string(ax.XLabel.String), "b1");
            testCase.verifyEqual(string(ax.YLabel.String), "b2");
            testCase.verifyError(@() plot(br, Var = "x", Parent = ax), ...
                "phasecont:model:badOption");
        end

        function livePlotOfACurveEndsInTheStyledPlot(testCase)
            [m, lp] = testCase.cuspFold();
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            br = phasecont.continuation(m, lp, Type = "LP", ...
                Free = ["b1" "b2"], Bounds = [-0.5 1.5], Plot = true);
            ax = gca;
            line = findobj(ax, "Tag", "pc-curve");
            testCase.assertSize(line, [1 1]);
            testCase.verifyEqual(line.XData(:), br.Alpha(:, 1));
            testCase.verifyEqual(line.YData(:), br.Alpha(:, 2));
        end

        function optionErrors(testCase)
            [m, lp] = testCase.cuspFold();
            id = "phasecont:model:badOption";
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "XX", Free = "b1"), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = "b1"), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "b2" "b1"]), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "b1"]), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "zz"]), ...
                "phasecont:model:unknownParameter");
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "b2"], Switch = "BP1"), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "b2"], Detect = "LP"), id);
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Type = "LP", Free = ["b1" "b2"], Var = "x"), id);
            % An EP run still takes exactly one name.
            testCase.verifyError(@() phasecont.continuation(m, lp, ...
                Free = ["b1" "b2"]), "phasecont:model:unknownParameter");
        end

        function anEquilibriumBranchHasTheOldLayout(testCase)
            [m, ~] = testCase.cuspFold();
            eq = phasecont.equilibrium(m, -1);
            br = phasecont.continuation(m, eq, Free = "b1", Bounds = [-1 1]);
            N = numel(br.Alpha);
            testCase.verifyEqual(br.Type, "EP");
            testCase.verifySize(br.Alpha, [N 1]);
            testCase.verifySize(br.Tests, [N 3]);
            testCase.verifyEqual(br.Free, "b1");
            testCase.verifyEqual(br.FreeIndex, 1);
            testCase.verifySize(br.Codim2, [N 0]);
            testCase.verifyEqual(br.Codim2Labels, strings(1, 0));
        end
    end
end
