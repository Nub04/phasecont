classdef AppTest < matlab.unittest.TestCase
    % Tests phasecont.app.PhasePlane through its methods, with the figure
    % hidden: the options read from the model, trajectories and
    % equilibria, the graphics tags, the field and the nullclines, a
    % parameter change, and a save and open round trip. Every test loads
    % pckern, so the suite needs the MEX binary.

    properties (Constant)
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
        Rotation = "x'=y" + newline + "y'=-x" + newline + ...
            "init x=1,y=0" + newline + "done" + newline
        Spiral = "x'=-x+y" + newline + "y'=-x-y" + newline + ...
            "init x=1,y=0" + newline + "done" + newline
        Saddle = "x'=x" + newline + "y'=-y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Decay = "x'=-x" + newline + "y'=-2*y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Param = "par a=-1" + newline + "x'=a*x" + newline + "y'=-y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Three = "x'=-x" + newline + "y'=-2*y" + newline + "z'=-3*z" + newline + ...
            "init x=1,y=1,z=1" + newline + "done" + newline
        Forced = "x'=-x+sin(t)" + newline + "y'=-y" + newline + ...
            "init x=0,y=1" + newline + "done" + newline
        OneState = "x'=-x" + newline + "init x=1" + newline + "done" + newline
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
        function [app, m] = makeApp(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
            app = phasecont.app.PhasePlane(m, Visible = "off");
            testCase.addTeardown(@() delete(app));
        end

        function n = count(~, app, tag)
            n = numel(findobj(app.Axes, "Tag", tag));
        end
    end

    methods (Test)
        function optionsSetAxesRangeAndT(testCase)
            text = "@ total=200, xp=w, yp=v, xlo=0, xhi=0.8, ylo=-80, yhi=60" ...
                + newline + testCase.MorrisLecar;
            [app, m] = testCase.makeApp(text);
            [~, iv] = ismember("v", m.Variables);
            [~, iw] = ismember("w", m.Variables);
            testCase.verifyEqual(app.XIndex, iw);
            testCase.verifyEqual(app.YIndex, iv);
            testCase.verifyEqual(app.Window, [0 0.8 -80 60]);
            testCase.verifyEqual(app.T, 200);
            testCase.verifyEmpty(app.Notes);
            testCase.verifyEqual(app.Axes.XLim, [0 0.8]);
            testCase.verifyEqual(app.Axes.YLim, [-80 60]);
            testCase.verifyEqual(app.Axes.XLabel.String, 'w');
            testCase.verifyEqual(app.Axes.YLabel.String, 'v');
        end

        function windowDefaultsToInitWithoutOptions(testCase)
            app = testCase.makeApp(testCase.MorrisLecar);
            testCase.verifyEqual(app.XIndex, 1);
            testCase.verifyEqual(app.YIndex, 2);
            testCase.verifyEqual(app.Window, [-40 0 -0.9 1.1], AbsTol = 1e-12);
            testCase.verifyEmpty(app.T);
        end

        function badRangeFallsBackWithANote(testCase)
            text = "@ xlo=2, xhi=1" + newline + testCase.Rotation;
            app = testCase.makeApp(text);
            testCase.verifyEqual(app.Window, [0 2 -1 1]);
            testCase.verifyNotEmpty(app.Notes);
            testCase.verifyTrue(any(contains(app.Notes, "xlo")));
        end

        function unusedMethodOptionsAreNamed(testCase)
            text = "@ meth=rungekutta, dt=0.1, total=5" + newline + testCase.Rotation;
            app = testCase.makeApp(text);
            testCase.verifyTrue(any(contains(app.Notes, "Dormand-Prince")));
            testCase.verifyTrue(any(contains(app.Notes, "meth")));
            testCase.verifyEqual(app.T, 5);
        end

        function nonAutonomousModelGetsANote(testCase)
            app = testCase.makeApp(testCase.Forced);
            testCase.verifyTrue(any(contains(app.Notes, "uses t")));
        end

        function clickIntegratesFromThePointInTheChosenAxes(testCase)
            text = "@ total=200, xp=w, yp=v" + newline + testCase.MorrisLecar;
            app = testCase.makeApp(text);
            [~, iv] = ismember("v", app.Model.Variables);
            [~, iw] = ismember("w", app.Model.Variables);
            tr = app.clickAt(0.3, -10);
            testCase.verifyClass(tr, "phasecont.Trajectory");
            x0 = zeros(2, 1);
            x0(iw) = 0.3;
            x0(iv) = -10;
            testCase.verifyEqual(tr.X(1, :).', x0);
            testCase.verifyEqual(tr.T(end), 200, AbsTol = 1e-9);
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 1);
            testCase.verifyEqual(testCase.count(app, "pc-start"), 1);
            testCase.verifyEqual(numel(app.Items), 1);
        end

        function rotationClosesAfterOnePeriod(testCase)
            T = 2 * pi;
            text = sprintf("@ total=%.17g", T) + newline + testCase.Rotation;
            app = testCase.makeApp(text);
            tr = app.addTrajectory(1, 0);
            testCase.verifyEqual(tr.X(end, :), [1 0], AbsTol = 1e-6);
            h = findobj(app.Axes, "Tag", "pc-trajectory");
            testCase.verifyEqual(numel(h.XData), numel(tr.T));
        end

        function bothDirectionsGiveTwoTrajectories(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Decay);
            app.setDirection("Both");
            tr = app.addTrajectory(1, 1);
            testCase.verifyEqual(numel(tr), 2);
            testCase.verifyEqual(tr(1).X(end, 1), exp(-1), AbsTol = 1e-6);
            testCase.verifyEqual(tr(2).X(end, 1), exp(1), AbsTol = 1e-6);
            testCase.verifyEqual(tr(2).T(end), -1, AbsTol = 1e-9);
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 2);
            testCase.verifyEqual(testCase.count(app, "pc-start"), 2);
        end

        function backwardDirectionRunsToMinusT(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Decay);
            app.setDirection("Backward");
            tr = app.addTrajectory(1, 1);
            testCase.verifyEqual(tr.T(end), -1, AbsTol = 1e-9);
            testCase.verifyEqual(tr.X(end, 2), exp(2), AbsTol = 1e-5);
        end

        function missingTimeAsksForT(testCase)
            app = testCase.makeApp(testCase.Rotation);
            testCase.verifyError(@() app.addTrajectory(1, 0), ...
                "phasecont:app:missingTime");
            testCase.verifyEmpty(app.Items);
            app.setT(3);
            tr = app.addTrajectory(1, 0);
            testCase.verifyEqual(tr.T(end), 3, AbsTol = 1e-9);
        end

        function badTIsRejected(testCase)
            app = testCase.makeApp(testCase.Rotation);
            testCase.verifyError(@() app.setT(-1), "phasecont:app:badValue");
            testCase.verifyError(@() app.setT(NaN), "phasecont:app:badValue");
            testCase.verifyError(@() app.setT([1 2]), "phasecont:app:badValue");
            app.setT(4);
            app.setT([]);
            testCase.verifyEmpty(app.T);
        end

        function stableEquilibriumGetsAFilledMarker(testCase)
            app = testCase.makeApp(testCase.Spiral);
            pt = app.findEquilibrium(0.5, 0.5);
            testCase.verifyClass(pt, "phasecont.Point");
            testCase.verifyEqual(pt.U, [0; 0], AbsTol = 1e-9);
            testCase.verifyLessThan(max(real(pt.Eigenvalues)), 0);
            h = findobj(app.Axes, "Tag", "pc-equilibrium-stable");
            testCase.verifyEqual(numel(h), 1);
            testCase.verifyEqual(h.MarkerFaceColor, [0 0 0]);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-unstable"), 0);
        end

        function unstableEquilibriumGetsAnOpenMarker(testCase)
            app = testCase.makeApp(testCase.Saddle);
            pt = app.findEquilibrium(0.5, 0.5);
            testCase.verifyGreaterThan(max(real(pt.Eigenvalues)), 0);
            h = findobj(app.Axes, "Tag", "pc-equilibrium-unstable");
            testCase.verifyEqual(numel(h), 1);
            testCase.verifyEqual(h.MarkerFaceColor, 'none');
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-stable"), 0);
        end

        function clickModeChoosesWhatAClickDoes(testCase)
            app = testCase.makeApp("@ total=2" + newline + testCase.Spiral);
            testCase.verifyClass(app.clickAt(0.5, 0.5), "phasecont.Trajectory");
            app.setClickMode("Equilibrium");
            testCase.verifyClass(app.clickAt(0.5, 0.5), "phasecont.Point");
            testCase.verifyEqual(numel(app.Items), 2);
            testCase.verifyError(@() app.setClickMode("Both"), ...
                "MATLAB:validators:mustBeMember");
        end

        function morrisLecarEquilibriumMatchesItsEigenvalues(testCase)
            app = testCase.makeApp(testCase.MorrisLecar);
            pt = app.findEquilibrium(-20, 0.1);
            if all(real(pt.Eigenvalues) < 0)
                tag = "pc-equilibrium-stable";
            else
                tag = "pc-equilibrium-unstable";
            end
            testCase.verifyEqual(testCase.count(app, tag), 1);
            f = app.Model.rhs(pt.U, app.Params);
            testCase.verifyLessThan(norm(f), 1e-8);
        end

        function fieldPointsAlongTheVectorField(testCase, Evaluator)
            app = testCase.makeApp(testCase.Rotation, Evaluator);
            app.showField(true);
            h = findobj(app.Axes, "Tag", "pc-field");
            testCase.assertEqual(numel(h), 1);
            X = h.XData;
            Y = h.YData;
            U = h.UData;
            V = h.VData;
            testCase.verifyEqual(size(X), [20 20]);
            % f = (y, -x): the arrow is parallel to it and not opposed.
            along = U .* Y + V .* (-X);
            across = U .* (-X) - V .* Y;
            testCase.verifyLessThan(max(abs(across(:))), 1e-9);
            testCase.verifyGreaterThanOrEqual(min(along(:)), 0);
            % Every arrow has the same length: the field is normalized.
            len = hypot(U, V);
            testCase.verifyEqual(len, len(1) * ones(size(len)), AbsTol = 1e-12);
            app.showField(false);
            testCase.verifyEqual(testCase.count(app, "pc-field"), 0);
        end

        function nullclinesLieOnTheZeroSetsOfTheField(testCase, Evaluator)
            app = testCase.makeApp(testCase.Rotation, Evaluator);
            app.showNullclines(true);
            h = findobj(app.Axes, "Tag", "pc-nullcline");
            testCase.assertEqual(numel(h), 2);
            names = string(arrayfun(@(g) string(g.UserData), h));
            hx = h(names == "x");
            hy = h(names == "y");
            % x' = y vanishes on y = 0 and y' = -x on x = 0.
            testCase.verifyEqual(max(abs(hx.YData), [], "omitnan"), 0, AbsTol = 1e-9);
            testCase.verifyEqual(max(abs(hy.XData), [], "omitnan"), 0, AbsTol = 1e-9);
            testCase.verifyTrue(any(~isnan(hx.XData)));
            testCase.verifyTrue(any(~isnan(hy.YData)));
            app.showNullclines(false);
            testCase.verifyEqual(testCase.count(app, "pc-nullcline"), 0);
        end

        function morrisLecarNullclinesCrossAtTheEquilibrium(testCase)
            text = "@ xlo=-80, xhi=60, ylo=-0.1, yhi=0.9" + newline + testCase.MorrisLecar;
            app = testCase.makeApp(text);
            app.showNullclines(true);
            h = findobj(app.Axes, "Tag", "pc-nullcline");
            testCase.verifyEqual(numel(h), 2);
            for k = 1:numel(h)
                testCase.verifyTrue(any(~isnan(h(k).XData)), ...
                    "nullcline " + string(h(k).UserData) + " is empty");
            end
        end

        function parameterChangeRemovesRunsAndRedrawsTheField(testCase)
            app = testCase.makeApp("@ total=2" + newline + testCase.Param);
            app.showField(true);
            app.addTrajectory(1, 1);
            app.findEquilibrium(0.5, 0.5);
            testCase.verifyEqual(numel(app.Items), 2);
            before = findobj(app.Axes, "Tag", "pc-field");
            u0 = before.UData;
            app.setParameter("a", 1);
            testCase.verifyEqual(app.Params, 1);
            testCase.verifyEmpty(app.Items);
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-start"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-stable") ...
                + testCase.count(app, "pc-equilibrium-unstable"), 0);
            after = findobj(app.Axes, "Tag", "pc-field");
            testCase.verifyEqual(numel(after), 1);
            testCase.verifyNotEqual(after.UData, u0);
            % The next run uses the new value: x' = x grows.
            tr = app.addTrajectory(1, 1);
            testCase.verifyEqual(tr.X(end, 1), exp(2), AbsTol = 1e-5);
            testCase.verifyEqual(tr.P, 1);
        end

        function unknownParameterIsAnError(testCase)
            app = testCase.makeApp(testCase.Param);
            testCase.verifyError(@() app.setParameter("b", 1), ...
                "phasecont:app:unknownParameter");
            testCase.verifyError(@() app.setParameter("a", Inf), ...
                "MATLAB:validators:mustBeFinite");
        end

        function otherAxesUseTheSliceValues(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Three);
            app.setAxes("x", "z");
            testCase.verifyEqual([app.XIndex app.YIndex], [1 3]);
            testCase.verifyEqual(app.Axes.XLabel.String, 'x');
            testCase.verifyEqual(app.Axes.YLabel.String, 'z');
            tr = app.addTrajectory(0.5, 0.25);
            testCase.verifyEqual(tr.X(1, :), [0.5 1 0.25]);
            app.setSlice("y", 2);
            tr = app.addTrajectory(0.5, 0.25);
            testCase.verifyEqual(tr.X(1, :), [0.5 2 0.25]);
            testCase.verifyEqual(tr.X(end, 2), 2 * exp(-2), AbsTol = 1e-6);
            h = findobj(app.Axes, "Tag", "pc-trajectory");
            testCase.verifyEqual(numel(h), 2);
        end

        function axisAndSliceErrors(testCase)
            app = testCase.makeApp(testCase.Three);
            testCase.verifyError(@() app.setAxes("x", "x"), ...
                "phasecont:app:sameAxes");
            testCase.verifyError(@() app.setAxes("x", "q"), ...
                "phasecont:app:unknownVariable");
            testCase.verifyError(@() app.setSlice("q", 1), ...
                "phasecont:app:unknownVariable");
        end

        function windowChangesTheViewOnly(testCase)
            app = testCase.makeApp(testCase.Rotation);
            app.setWindow([-3 3 -2 2]);
            testCase.verifyEqual(app.Axes.XLim, [-3 3]);
            testCase.verifyEqual(app.Axes.YLim, [-2 2]);
            testCase.verifyError(@() app.setWindow([1 1 0 1]), ...
                "phasecont:app:badWindow");
            testCase.verifyEqual(app.Window, [-3 3 -2 2]);
        end

        function nonAutonomousModelRunsButHasNoFieldOrEquilibria(testCase)
            app = testCase.makeApp("@ total=5" + newline + testCase.Forced);
            tr = app.addTrajectory(0, 1);
            testCase.verifyEqual(tr.T(end), 5, AbsTol = 1e-9);
            % x' = -x + sin t with x(0) = 0 has the exact solution below.
            t = 5;
            exact = (sin(t) - cos(t) + exp(-t)) / 2;
            testCase.verifyEqual(tr.X(end, 1), exact, AbsTol = 1e-6);
            testCase.verifyError(@() app.showField(true), ...
                "phasecont:app:nonAutonomous");
            testCase.verifyError(@() app.showNullclines(true), ...
                "phasecont:app:nonAutonomous");
            testCase.verifyError(@() app.findEquilibrium(0, 0), ...
                "phasecont:app:nonAutonomous");
            testCase.verifyEqual(testCase.count(app, "pc-field"), 0);
        end

        function clearRemovesEveryItem(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Spiral);
            app.addTrajectory(1, 0);
            app.findEquilibrium(0.5, 0.5);
            app.clear();
            testCase.verifyEmpty(app.Items);
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-start"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-stable"), 0);
        end

        function aModelNeedsTwoStates(testCase)
            m = phasecont.Model.fromSource(testCase.OneState);
            testCase.addTeardown(@() delete(m));
            testCase.verifyError( ...
                @() phasecont.app.PhasePlane(m, Visible = "off"), ...
                "phasecont:app:needTwoStates");
        end

        function aBadSourceIsRejected(testCase)
            testCase.verifyError(@() phasecont.app.PhasePlane(42), ...
                "phasecont:app:badSource");
        end

        function sessionRoundTripKeepsTheItems(testCase)
            text = "@ total=100, xp=w, yp=v" + newline + testCase.MorrisLecar;
            app = testCase.makeApp(text);
            tr = app.addTrajectory(0.3, -10);
            pt = app.findEquilibrium(-20, 0.1);
            file = fullfile(testCase.Dir, "app.pcsession.json");
            app.saveSession(file);
            testCase.assertTrue(isfile(file));

            app2 = phasecont.app.PhasePlane(file, Visible = "off");
            testCase.addTeardown(@() delete(app2));
            testCase.assertEqual(numel(app2.Items), 2);
            testCase.verifyEqual(app2.Items{1}.X, tr.X);
            testCase.verifyEqual(app2.Items{1}.T, tr.T);
            testCase.verifyEqual(app2.Items{2}.U, pt.U);
            testCase.verifyEqual(app2.XIndex, app.XIndex);
            testCase.verifyEqual(app2.YIndex, app.YIndex);
            testCase.verifyEqual(app2.T, 100);
            testCase.verifyEqual(testCase.count(app2, "pc-trajectory"), 1);
            testCase.verifyEqual(testCase.count(app2, "pc-start"), 1);
            testCase.verifyEqual(testCase.count(app2, "pc-equilibrium-stable") ...
                + testCase.count(app2, "pc-equilibrium-unstable"), 1);
        end

        function openSessionReplacesTheModelAndTheItems(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Spiral);
            app.addTrajectory(1, 0);
            app.findEquilibrium(0.5, 0.5);
            file = fullfile(testCase.Dir, "spiral.pcsession.json");
            app.saveSession(file);

            other = testCase.makeApp("@ total=1" + newline + testCase.Three);
            other.addTrajectory(1, 1);
            testCase.verifyEqual(numel(other.Model.Variables), 3);
            other.openSession(file);
            testCase.verifyEqual(other.Model.Variables, ["x" "y"]);
            testCase.verifyEqual(numel(other.Items), 2);
            testCase.verifyEqual(other.T, 1);
            testCase.verifyEqual(testCase.count(other, "pc-trajectory"), 1);
            testCase.verifyEqual(testCase.count(other, "pc-equilibrium-stable"), 1);
            % A run in the opened model works.
            tr = other.addTrajectory(1, 0);
            testCase.verifyEqual(size(tr.X, 2), 2);
            testCase.verifyEqual(numel(other.Items), 3);
        end

        function openedSessionKeepsItsParameterValues(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Param);
            app.setParameter("a", -3);
            app.addTrajectory(1, 1);
            file = fullfile(testCase.Dir, "param.pcsession.json");
            app.saveSession(file);
            app2 = phasecont.app.PhasePlane(file, Visible = "off");
            testCase.addTeardown(@() delete(app2));
            testCase.verifyEqual(app2.Params, -3);
        end

        function closingTheFigureFreesAnOwnedModel(testCase)
            app = testCase.makeApp("@ total=1" + newline + testCase.Spiral);
            app.addTrajectory(1, 0);
            file = fullfile(testCase.Dir, "close.pcsession.json");
            app.saveSession(file);
            app2 = phasecont.app.PhasePlane(file, Visible = "off");
            m2 = app2.Model;
            fig = app2.Figure;
            delete(app2);
            testCase.verifyFalse(isvalid(fig));
            testCase.verifyFalse(isvalid(m2));
        end
    end
end
