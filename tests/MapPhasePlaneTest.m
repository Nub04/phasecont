classdef MapPhasePlaneTest < matlab.unittest.TestCase
    % Tests the map view and the cobweb view of phasecont.app.PhasePlane
    % (SPEC section 8z, tests 1 to 7), with the figure hidden. Every model
    % is written here, and every expected value is computed here with a
    % loop or a formula of its own. The loops keep the association of the
    % model's expression tree, so x^2 is x*x and r*x*(1-x) is (r*x)*(1-x).

    properties (Constant)
        % (x, y) -> (1 - a x^2 + y, b x).
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        % F(x) - x = (y, 0.5 x), so the nullclines are y = 0 and x = 0.
        Shear = "init x=0.25, y=0.25" + newline + "x(t+1)=x+y" + newline + ...
            "y(t+1)=0.5*x+y" + newline + "done" + newline
        % A map with two variables that reads t.
        Forced = "init x=0, y=0" + newline + "x(t+1)=0.5*x+y+t" + newline + ...
            "y(t+1)=0.5*y" + newline + "done" + newline
        % x -> r x (1 - x), with the fixed points 0 and 1 - 1/r and the
        % multiplier 2 - r at 1 - 1/r.
        Logistic = "par r=2.5" + newline + "init x=0.1" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        % x -> x^2. From 2 the iterates are 2^(2^k), and 2^1024 overflows.
        Square = "init x=2" + newline + "x(t+1)=x^2" + newline + ...
            "done" + newline
        % A map with one variable that reads t.
        Shift = "init x=1" + newline + "x(t+1)=x/2+t" + newline + ...
            "done" + newline
        Rotation = "u'=v" + newline + "v'=-u" + newline + ...
            "init u=1,v=0" + newline + "done" + newline
        OneState = "x'=-x" + newline + "init x=1" + newline + "done" + newline
        CobwebNote = "The cobweb view draws no direction field or nullclines. " ...
            + "The fixed points lie where the graph crosses the diagonal."
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

    methods (Access = private)
        function m = model(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end

        function [app, m] = makeApp(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = testCase.model(text, ev);
            app = phasecont.app.PhasePlane(m, Visible = "off");
            testCase.addTeardown(@() delete(app));
        end

        function app = openFile(testCase, file)
            app = phasecont.app.PhasePlane(file, Visible = "off");
            testCase.addTeardown(@() delete(app));
        end

        function n = count(~, app, tag)
            n = numel(findobj(app.Axes, "Tag", tag));
        end

        function h = line1(testCase, app, tag)
            h = findobj(app.Axes, "Tag", tag);
            testCase.assertNumElements(h, 1, "one line with Tag " + tag);
        end

        function verifyErrorText(testCase, fn, id, text)
            try
                fn();
                testCase.verifyFail("expected the error " + id);
            catch err
                testCase.verifyEqual(string(err.identifier), id);
                testCase.verifyEqual(string(err.message), text);
            end
        end

        function verifyIdsFreed(testCase, first, last)
            % Handle ids count up in steps of 2, so every id strictly
            % between FIRST and LAST must be unknown to the engine.
            ids = first + uint64(2):uint64(2):last - uint64(2);
            testCase.verifyNotEmpty(ids, "the session load made no handle");
            for id = ids
                testCase.verifyError(@() pckern("model_info", id), ...
                    "phasecont:engine:badHandle", ...
                    sprintf("handle %d is still alive", id));
            end
        end

        function c = control(testCase, app, type, row, col)
            % The control of TYPE at row ROW and column COL of the left grid.
            hs = findall(app.Figure, "Type", type);
            keep = arrayfun(@(h) isequal(h.Layout.Row, row) ...
                && isequal(h.Layout.Column, col), hs);
            c = hs(keep);
            testCase.assertNumElements(c, 1, sprintf("one %s at (%d, %s)", ...
                type, row, mat2str(col)));
        end

        function c = checkbox(testCase, app, text)
            hs = findall(app.Figure, "Type", "uicheckbox");
            c = hs(arrayfun(@(h) string(h.Text) == text, hs));
            testCase.assertNumElements(c, 1, "one check box " + text);
        end

        function verifyControls(testCase, app, label, enabled, xlab, ylab)
            % LABEL is the T label text. ENABLED holds the Enable state of
            % [XDrop YDrop DirDrop FieldBox NullBox].
            testCase.verifyEqual(string(testCase.control(app, "uilabel", 3, 1).Text), label);
            hs = {testCase.control(app, "uidropdown", 1, 2), ...
                testCase.control(app, "uidropdown", 2, 2), ...
                testCase.control(app, "uidropdown", 4, 2), ...
                testCase.checkbox(app, "Direction field"), ...
                testCase.checkbox(app, "Nullclines")};
            names = ["x axis" "y axis" "Direction" "Direction field" "Nullclines"];
            for k = 1:numel(hs)
                testCase.verifyEqual(string(hs{k}.Enable) == "on", enabled(k), ...
                    names(k) + " enabled state");
            end
            testCase.verifyEqual(string(app.Axes.XLabel.String), xlab);
            testCase.verifyEqual(string(app.Axes.YLabel.String), ylab);
        end
    end

    methods (Static, Access = private)
        function X = henon(a, b, x0, N)
            X = zeros(N + 1, 2);
            X(1, :) = x0;
            for k = 1:N
                X(k + 1, 1) = (1 - a * (X(k, 1) * X(k, 1))) + X(k, 2);
                X(k + 1, 2) = b * X(k, 1);
            end
        end

        function x = logistic(r, x0, N)
            x = zeros(N + 1, 1);
            x(1) = x0;
            for k = 1:N
                x(k + 1) = r * x(k) * (1 - x(k));
            end
        end

        function [X, Y] = staircase(x)
            m = numel(x) - 1;
            X = zeros(2 * m + 1, 1);
            Y = zeros(2 * m + 1, 1);
            for i = 0:2 * m
                X(i + 1) = x(floor(i / 2) + 1);
                Y(i + 1) = x(ceil(i / 2) + 1);
            end
        end
    end

    methods (Test)
        % Test 1.
        function henonIteratesForwardFromAClick(testCase, Evaluator)
            app = testCase.makeApp(testCase.Henon, Evaluator);
            testCase.verifyControls(app, "N", [true true false false true], "x", "y");
            testCase.verifyEqual(app.Direction, "Forward");
            testCase.verifyEmpty(app.T);

            app.setT(20);
            tr = app.clickAt(0.1, 0.2);
            testCase.verifyClass(tr, "phasecont.Trajectory");
            testCase.verifyEqual(tr.T, (0:20).');
            testCase.verifyEqual(tr.X, testCase.henon(1.4, 0.3, [0.1 0.2], 20));
            testCase.verifyEqual(tr.Stop, "End");
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 1);
            testCase.verifyEqual(app.Status, "Iterated from (0.1, 0.2) for 20 steps.");

            for bad = {2.5, 0, 2e6, -1, NaN, [1 2]}
                testCase.verifyErrorText(@() app.setT(bad{1}), ...
                    "phasecont:app:badValue", ...
                    "N must be a whole number from 1 to 1000000, or empty.");
            end
            testCase.verifyEqual(app.T, 20);
            app.setT(1e6);
            testCase.verifyEqual(app.T, 1e6);
            app.setT([]);
            testCase.verifyErrorText(@() app.clickAt(0.1, 0.2), ...
                "phasecont:app:missingTime", ...
                "The model has no @ total. Enter the number of iterates N first.");
            for dir = ["Backward" "Both"]
                testCase.verifyErrorText(@() app.setDirection(dir), ...
                    "phasecont:app:mapModel", "A map is iterated forward only.");
            end
            app.setDirection("Forward");
            testCase.verifyErrorText(@() app.showField(true), ...
                "phasecont:app:mapModel", "A map has no direction field.");
            testCase.verifyEqual(testCase.count(app, "pc-field"), 0);
        end

        % Test 2.
        function henonFixedPointsAreMarkedByTheirMultipliers(testCase, Evaluator)
            app = testCase.makeApp(testCase.Henon, Evaluator);
            app.setClickMode("Equilibrium");
            b = 0.3;
            for a = [1.4 0.2]
                app.setParameter("a", a);
                xs = (-(1 - b) + sqrt((1 - b)^2 + 4 * a)) / (2 * a);
                ys = b * xs;
                mu = eig([-2 * a * xs, 1; b, 0]);
                top = max(abs(mu));
                pt = app.clickAt(xs + 0.02, ys - 0.01);
                testCase.verifyClass(pt, "phasecont.Point");
                testCase.verifyEqual(pt.U, [xs; ys], AbsTol = 1e-10);
                testCase.verifyEqual(sort(abs(pt.Eigenvalues)), sort(abs(mu)), ...
                    AbsTol = 1e-10);
                if a == 1.4
                    testCase.verifyGreaterThan(top, 1);
                    want = "pc-equilibrium-unstable";
                    other = "pc-equilibrium-stable";
                    kind = "unstable";
                else
                    testCase.verifyLessThan(top, 1);
                    want = "pc-equilibrium-stable";
                    other = "pc-equilibrium-unstable";
                    kind = "stable";
                end
                h = testCase.line1(app, want);
                testCase.verifyEqual([h.XData h.YData], [xs ys], AbsTol = 1e-10);
                testCase.verifyEqual(testCase.count(app, other), 0);
                testCase.verifyEqual(app.Status, sprintf( ...
                    "Fixed point at (%.6g, %.6g): %s, largest |multiplier| %.3g.", ...
                    pt.U(1), pt.U(2), kind, top));
            end
        end

        % Test 3.
        function mapNullclinesAreWhereTheMapKeepsAVariable(testCase, Evaluator)
            app = testCase.makeApp(testCase.Shear, Evaluator);
            note = "On a map, the nullcline of a variable is the curve " ...
                + "where the map leaves that variable unchanged.";
            testCase.verifyTrue(any(app.Notes == note));
            % A window whose grid has no node on x = 0 or y = 0.
            app.setWindow([-0.75 1.25 -0.75 1.25]);
            cell = 2 / 100;
            app.showNullclines(true);
            h = findobj(app.Axes, "Tag", "pc-nullcline");
            testCase.assertNumElements(h, 2);
            names = string(arrayfun(@(g) string(g.UserData), h));
            hx = h(names == "x");
            hy = h(names == "y");
            testCase.assertNumElements(hx, 1);
            testCase.assertNumElements(hy, 1);
            % x(t+1) - x = y vanishes on y = 0, y(t+1) - y = 0.5 x on x = 0.
            testCase.verifyLessThanOrEqual(max(abs(hx.YData), [], "omitnan"), cell);
            testCase.verifyLessThanOrEqual(max(abs(hy.XData), [], "omitnan"), cell);
            testCase.verifyTrue(any(isfinite(hx.XData)));
            testCase.verifyTrue(any(isfinite(hy.YData)));
            app.showNullclines(false);
            testCase.verifyEqual(testCase.count(app, "pc-nullcline"), 0);
        end

        % Test 4.
        function aMapThatReadsTIteratesWithTEqualToN(testCase, Evaluator)
            app = testCase.makeApp(testCase.Forced, Evaluator);
            note = "The model uses t. Iterates start at n = 0 with t = n. " ...
                + "The nullclines and the fixed points need an autonomous model.";
            testCase.verifyTrue(any(app.Notes == note));
            testCase.verifyFalse(any(startsWith(app.Notes, "On a map")));
            app.setT(10);
            tr = app.clickAt(0.1, 0.2);
            X = zeros(11, 2);
            X(1, :) = [0.1 0.2];
            for k = 1:10
                n = k - 1;
                X(k + 1, 1) = (0.5 * X(k, 1) + X(k, 2)) + n;
                X(k + 1, 2) = 0.5 * X(k, 2);
            end
            testCase.verifyEqual(tr.X, X);
            testCase.verifyEqual(tr.T, (0:10).');
            testCase.verifyErrorText(@() app.showNullclines(true), ...
                "phasecont:app:nonAutonomous", ...
                "The nullclines needs an autonomous model, and this model uses t.");
            app.setClickMode("Equilibrium");
            testCase.verifyError(@() app.clickAt(0.1, 0.2), ...
                "phasecont:app:nonAutonomous");
            testCase.verifyErrorText(@() app.showField(true), ...
                "phasecont:app:mapModel", "A map has no direction field.");
        end

        function ignoredOptionsAndTotalOfAMap(testCase, Evaluator)
            text = "@ meth=euler, dt=0.1, total=5" + newline + testCase.Henon;
            app = testCase.makeApp(text, Evaluator);
            testCase.verifyEqual(app.T, 5);
            testCase.verifyTrue(any(app.Notes == "The model sets @ dt, meth. " ...
                + "The app iterates the map one step at a time and ignores them."));

            % A meth that starts with d makes x'=... a map and gets no note.
            text = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
                "x'=1-a*x^2+y" + newline + "y'=b*x" + newline + ...
                "@ meth=discrete" + newline + "done" + newline;
            app = testCase.makeApp(text, Evaluator);
            testCase.verifyEqual(app.Direction, "Forward");
            testCase.verifyEqual(string(testCase.control(app, "uilabel", 3, 1).Text), "N");
            testCase.verifyFalse(any(startsWith(app.Notes, "The model sets @")));

            text = "@ total=2.5" + newline + testCase.Henon;
            app = testCase.makeApp(text, Evaluator);
            testCase.verifyEmpty(app.T);
            testCase.verifyTrue(any(app.Notes == ...
                "@ total = 2.5 is not a whole number from 1 to 1000000; enter N."));
        end

        % Test 5.
        function logisticOpensTheCobwebView(testCase, Evaluator)
            app = testCase.makeApp(testCase.Logistic, Evaluator);
            testCase.verifyControls(app, "N", [false false false false false], ...
                "x", "x(n+1)");
            testCase.verifyEqual([app.XIndex app.YIndex], [1 1]);
            testCase.verifyFalse(logical(testCase.control(app, "uitable", 13, [1 2]).Visible));
            testCase.verifyEqual(app.Window, [-0.9 1.1 -0.9 1.1], AbsTol = 4 * eps);
            note = "In the cobweb view a click starts at its x value. " ...
                + "The fixed points lie where the graph crosses the diagonal.";
            testCase.verifyTrue(any(app.Notes == note));
            testCase.verifyFalse(any(startsWith(app.Notes, "On a map")));

            r = 2.5;
            g = testCase.line1(app, "pc-cobweb-graph");
            testCase.verifyNumElements(g.XData, 401);
            testCase.verifyEqual(g.XData([1 end]), [-0.9 1.1], AbsTol = 4 * eps);
            testCase.verifyEqual(g.YData, r * g.XData .* (1 - g.XData), AbsTol = 1e-14);
            d = testCase.line1(app, "pc-cobweb-diagonal");
            testCase.verifyEqual(d.XData, [-0.9 1.1], AbsTol = 4 * eps);
            testCase.verifyEqual(d.YData, d.XData);

            app.setT(10);
            tr = app.clickAt(0.1, 0.9);
            testCase.verifyEqual(tr.X(1), 0.1);
            x = testCase.logistic(r, 0.1, 10);
            testCase.verifyEqual(tr.X, x);
            [X, Y] = testCase.staircase(x);
            s = testCase.line1(app, "pc-cobweb");
            testCase.verifyNumElements(s.XData, 21);
            testCase.verifyEqual(s.XData(:), X);
            testCase.verifyEqual(s.YData(:), Y);
            p = testCase.line1(app, "pc-start");
            testCase.verifyEqual([p.XData p.YData], [0.1 0.1]);
            testCase.verifyEqual(testCase.count(app, "pc-trajectory"), 0);
            testCase.verifyEqual(app.Status, "Iterated from x = 0.1 for 10 steps.");

            app.setClickMode("Equilibrium");
            pt = app.clickAt(0.62, 0.3);
            testCase.verifyEqual(pt.U, 0.6, AbsTol = 1e-10);
            testCase.verifyEqual(pt.Eigenvalues, 2 - r, AbsTol = 1e-10);
            h = testCase.line1(app, "pc-equilibrium-stable");
            testCase.verifyEqual([h.XData h.YData], [0.6 0.6], AbsTol = 1e-10);
            testCase.verifyEqual(app.Status, sprintf( ...
                "Fixed point at x = %.6g: stable, |multiplier| %.3g.", pt.U, 0.5));

            r = 3.2;
            app.setParameter("r", r);
            testCase.verifyEqual(testCase.count(app, "pc-cobweb"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-start"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-stable"), 0);
            testCase.verifyFalse(any(cellfun(@(c) isa(c, "phasecont.Trajectory") ...
                || isa(c, "phasecont.Point"), app.Items)));
            g = testCase.line1(app, "pc-cobweb-graph");
            testCase.verifyEqual(g.YData, r * g.XData .* (1 - g.XData), AbsTol = 1e-14);
            pt = app.clickAt(0.7, 0.3);
            testCase.verifyEqual(pt.U, 1 - 1 / r, AbsTol = 1e-10);
            testCase.verifyEqual(pt.Eigenvalues, 2 - r, AbsTol = 1e-10);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-unstable"), 1);
            testCase.verifyEqual(testCase.count(app, "pc-equilibrium-stable"), 0);
            testCase.verifySubstring(app.Status, "unstable");

            testCase.verifyErrorText(@() app.showField(true), ...
                "phasecont:app:mapModel", testCase.CobwebNote);
            testCase.verifyErrorText(@() app.showNullclines(true), ...
                "phasecont:app:mapModel", testCase.CobwebNote);
            testCase.verifyError(@() app.setAxes("x", "x"), "phasecont:app:sameAxes");
            testCase.verifyError(@() app.setAxes("x", "z"), ...
                "phasecont:app:unknownVariable");
            app.showField(false);
            app.showNullclines(false);

            app.setWindow([0 1 -0.5 2]);
            g = testCase.line1(app, "pc-cobweb-graph");
            testCase.verifyEqual(g.XData, linspace(0, 1, 401));
            testCase.verifyEqual(g.YData, r * g.XData .* (1 - g.XData), AbsTol = 1e-14);
            d = testCase.line1(app, "pc-cobweb-diagonal");
            testCase.verifyEqual([d.XData d.YData], [-0.5 2 -0.5 2]);
        end

        function cobwebSessionRoundTripDrawsTheSameStaircase(testCase, Evaluator)
            app = testCase.makeApp(testCase.Logistic, Evaluator);
            app.setT(10);
            app.clickAt(0.2, 0);
            s = testCase.line1(app, "pc-cobweb");
            file = fullfile(testCase.Dir, "cobweb.pcsession.json");
            app.saveSession(file);

            other = testCase.openFile(file);
            s2 = testCase.line1(other, "pc-cobweb");
            testCase.verifyEqual(s2.XData, s.XData);
            testCase.verifyEqual(s2.YData, s.YData);
            p = testCase.line1(other, "pc-start");
            testCase.verifyEqual([p.XData p.YData], [0.2 0.2]);
            testCase.verifyEqual(string(other.Axes.YLabel.String), "x(n+1)");
            testCase.verifyEqual(testCase.count(other, "pc-cobweb-graph"), 1);
        end

        function cobwebWindowFromTheOptions(testCase, Evaluator)
            text = "@ xlo=0, xhi=1" + newline + testCase.Logistic;
            app = testCase.makeApp(text, Evaluator);
            testCase.verifyEqual(app.Window, [0 1 0 1]);
            text = "@ xlo=0, xhi=1, ylo=-1, yhi=2" + newline + testCase.Logistic;
            app = testCase.makeApp(text, Evaluator);
            testCase.verifyEqual(app.Window, [0 1 -1 2]);
        end

        function cobwebStopsEarlyAtAnOverflow(testCase, Evaluator)
            app = testCase.makeApp(testCase.Square, Evaluator);
            app.setT(12);
            tr = app.clickAt(2, 0);
            testCase.verifyEqual(tr.T, (0:9).');
            testCase.verifyEqual(tr.Stop, "Error");
            x = zeros(10, 1);
            x(1) = 2;
            for k = 1:9
                x(k + 1) = x(k) * x(k);
            end
            testCase.verifyEqual(tr.X, x);
            [X, Y] = testCase.staircase(x);
            s = testCase.line1(app, "pc-cobweb");
            testCase.verifyNumElements(s.XData, 19);
            testCase.verifyEqual(s.XData(:), X);
            testCase.verifyEqual(s.YData(:), Y);
            testCase.verifyEqual([s.XData(end) s.YData(end)], [x(10) x(10)]);
            testCase.verifyTrue(endsWith(app.Status, "Stopped early at n = 9 (Error)."));
        end

        % Test 6.
        function aMapWithOneVariableThatReadsTIsRefused(testCase, Evaluator)
            text = "The cobweb view needs a map that does not use t.";
            m = testCase.model(testCase.Shift, Evaluator);
            testCase.verifyErrorText( ...
                @() phasecont.app.PhasePlane(m, Visible = "off"), ...
                "phasecont:app:nonAutonomous", text);
            % The caller's model is not deleted.
            testCase.verifyTrue(isvalid(m));

            app = testCase.makeApp(testCase.Henon, Evaluator);
            first = testCase.model(testCase.Shift);
            file = fullfile(testCase.Dir, "shift.pcsession.json");
            save(phasecont.Session(first, {}), file);
            testCase.verifyErrorText( ...
                @() phasecont.app.PhasePlane(file, Visible = "off"), ...
                "phasecont:app:nonAutonomous", text);
            testCase.verifyErrorText(@() app.openSession(file), ...
                "phasecont:app:nonAutonomous", text);
            after = testCase.model(testCase.Shift);
            testCase.verifyIdsFreed(first.Id, after.Id);
            % The app keeps its model.
            testCase.verifyTrue(isvalid(app.Model));
            testCase.verifyEqual(app.Model.Variables, ["x" "y"]);

            testCase.verifyErrorText( ...
                @() phasecont.app.PhasePlane(testCase.model(testCase.OneState, Evaluator), ...
                Visible = "off"), "phasecont:app:needTwoStates", ...
                "A phase plane needs a model with at least two variables.");
        end

        % Test 7.
        function openSessionSwitchesTheView(testCase, Evaluator)
            files = strings(1, 3);
            texts = [testCase.Rotation testCase.Henon testCase.Logistic];
            for k = 1:3
                files(k) = fullfile(testCase.Dir, sprintf("view%d.pcsession.json", k));
                save(phasecont.Session(testCase.model(texts(k)), {}), files(k));
            end
            app = testCase.makeApp(testCase.Rotation, Evaluator);
            testCase.verifyControls(app, "T", true(1, 5), "u", "v");

            app.openSession(files(2));
            testCase.verifyControls(app, "N", [true true false false true], "x", "y");
            testCase.verifyEqual(app.Direction, "Forward");

            app.openSession(files(3));
            testCase.verifyControls(app, "N", false(1, 5), "x", "x(n+1)");
            testCase.verifyEqual([app.XIndex app.YIndex], [1 1]);
            testCase.verifyEqual(testCase.count(app, "pc-cobweb-graph"), 1);

            app.openSession(files(1));
            testCase.verifyControls(app, "T", true(1, 5), "u", "v");
            testCase.verifyEqual([app.XIndex app.YIndex], [1 2]);
            testCase.verifyEqual(testCase.count(app, "pc-cobweb-graph"), 0);
            testCase.verifyEqual(testCase.count(app, "pc-cobweb-diagonal"), 0);
            app.setDirection("Both");
            app.showField(true);
            testCase.verifyGreaterThan(testCase.count(app, "pc-field"), 0);
        end
    end
end
