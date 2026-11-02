classdef TrajectoryTest < matlab.unittest.TestCase
    % Tests phasecont.simulate and phasecont.Trajectory against exact
    % solutions, the stop reasons, the argument errors and the plot tags.
    % Every test loads pckern, so the suite needs the MEX binary.

    properties (Constant)
        Rotation = "x'=y" + newline + "y'=-x" + newline + ...
            "init x=1,y=0" + newline + "done" + newline
        Decay = "par a=1" + newline + "x'=-a*x" + newline + ...
            "init x=1" + newline + "done" + newline
        Forced = "x'=-x+sin(t)" + newline + "init x=0" + newline + ...
            "done" + newline
        Blow = "x'=x^2" + newline + "init x=1" + newline + "done" + newline
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
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end
    end

    methods (Test)
        function rotationMatchesTheExactSolution(testCase, Evaluator)
            m = testCase.model(testCase.Rotation, Evaluator);
            tr = phasecont.simulate(m, [1 0], [0 10]);
            testCase.verifyEqual(tr.Stop, "End");
            testCase.verifyEqual(tr.T(1), 0);
            testCase.verifyEqual(tr.T(end), 10);
            testCase.verifyTrue(issorted(tr.T, "strictascend"));
            testCase.verifyEqual(size(tr.X, 2), 2);
            testCase.verifyEqual(size(tr.X, 1), numel(tr.T));
            exact = [cos(tr.T), -sin(tr.T)];
            testCase.verifyLessThan(max(abs(tr.X - exact), [], "all"), 1e-6);
            testCase.verifyEqual(tr.X0, [1; 0]);
            testCase.verifyEqual(tr.TSpan, [0 10]);
            testCase.verifyEqual(tr.Model, m);
        end

        function backwardIntegrationReturnsToTheStart(testCase)
            m = testCase.model(testCase.Rotation);
            fwd = phasecont.simulate(m, [1 0], [0 5]);
            back = phasecont.simulate(m, fwd.X(end, :), [5 0]);
            testCase.verifyEqual(back.Stop, "End");
            testCase.verifyEqual(back.T(1), 5);
            testCase.verifyEqual(back.T(end), 0);
            testCase.verifyTrue(issorted(back.T, "strictdescend"));
            testCase.verifyLessThan(norm(back.X(end, :) - [1 0]), 1e-6);
        end

        function aForcedModelSeesTheTime(testCase)
            m = testCase.model(testCase.Forced);
            testCase.verifyFalse(m.Autonomous);
            tr = phasecont.simulate(m, 0, [0 8]);
            exact = (sin(tr.T) - cos(tr.T) + exp(-tr.T)) / 2;
            testCase.verifyLessThan(max(abs(tr.X - exact)), 1e-6);
        end

        function aTimeShiftedStartSeesTheShiftedTime(testCase)
            m = testCase.model(testCase.Forced);
            x0 = (sin(3) - cos(3) + exp(-3)) / 2;
            tr = phasecont.simulate(m, x0, [3 9]);
            exact = (sin(tr.T) - cos(tr.T) + exp(-tr.T)) / 2;
            testCase.verifyLessThan(max(abs(tr.X - exact)), 1e-6);
        end

        function paramsOverrideTheModelValues(testCase)
            m = testCase.model(testCase.Decay);
            tr = phasecont.simulate(m, 1, [0 1], Params = struct("a", 2));
            testCase.verifyEqual(tr.P, 2);
            testCase.verifyEqual(tr.X(end), exp(-2), AbsTol = 1e-7);
            tr = phasecont.simulate(m, 1, [0 1], Params = 3);
            testCase.verifyEqual(tr.X(end), exp(-3), AbsTol = 1e-7);
            tr = phasecont.simulate(m, 1, [0 1]);
            testCase.verifyEqual(tr.P, 1);
        end

        function aPointStartsTheRunWithItsParameters(testCase)
            m = testCase.model(testCase.Decay);
            eq = phasecont.equilibrium(m, 0.5, Params = struct("a", 2));
            tr = phasecont.simulate(m, eq, [0 1]);
            testCase.verifyEqual(tr.P, 2);
            testCase.verifyEqual(tr.X0, eq.U);
            tr = phasecont.simulate(m, eq, [0 1], Params = struct("a", 4));
            testCase.verifyEqual(tr.P, 4);
        end

        function tolerancesControlTheError(testCase)
            m = testCase.model(testCase.Rotation);
            loose = phasecont.simulate(m, [1 0], [0 10], RelTol = 1e-4, AbsTol = 1e-6);
            tight = phasecont.simulate(m, [1 0], [0 10], RelTol = 1e-10, AbsTol = 1e-12);
            errLoose = abs(loose.X(end, 1) - cos(10));
            errTight = abs(tight.X(end, 1) - cos(10));
            testCase.verifyLessThan(errTight, errLoose);
            testCase.verifyLessThan(numel(loose.T), numel(tight.T));
        end

        function hmaxCapsTheStepLength(testCase)
            m = testCase.model(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 1], Hmax = 0.01);
            testCase.verifyLessThanOrEqual(max(diff(tr.T)), 0.01 * (1 + 1e-12));
            free = phasecont.simulate(m, [1 0], [0 1], Hmax = Inf);
            testCase.verifyLessThan(numel(free.T), numel(tr.T));
        end

        function maxStepsStopsAndWarns(testCase)
            m = testCase.model(testCase.Rotation);
            tr = testCase.verifyWarning( ...
                @() phasecont.simulate(m, [1 0], [0 100], MaxSteps = 5), ...
                "phasecont:simulate:stoppedEarly");
            testCase.verifyEqual(tr.Stop, "MaxSteps");
            testCase.verifyLessThan(tr.T(end), 100);
            testCase.verifyLessThanOrEqual(numel(tr.T), 6);
        end

        function aBlowUpStopsBeforeTheSingularity(testCase)
            m = testCase.model(testCase.Blow);
            tr = testCase.verifyWarning( ...
                @() phasecont.simulate(m, 1, [0 2]), ...
                "phasecont:simulate:stoppedEarly");
            testCase.verifyTrue(any(tr.Stop == ["Error" "StepTooSmall"]), tr.Stop);
            testCase.verifyLessThan(tr.T(end), 1);
            testCase.verifyGreaterThan(tr.T(end), 0.99);
        end

        function theRecordHoldsTheRun(testCase)
            m = testCase.model(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 2], MaxSteps = 1000, Hmax = 0.5);
            rec = tr.Record;
            testCase.verifyEqual(rec.Kind, "simulate");
            testCase.verifyEqual(rec.Type, "IVP");
            testCase.verifyEqual(rec.U0, [1; 0]);
            testCase.verifyEqual(rec.ModelSha, m.SourceSha256);
            testCase.verifyEqual(rec.Stride, 3);
            testCase.verifyEqual(rec.Points{1}, [tr.T, tr.X].');
            problem = jsondecode(rec.Problem);
            testCase.verifyEqual(problem.type, 'IVP');
            testCase.verifyEqual(problem.tspan, [0; 2]);
            testCase.verifyEqual(problem.options.max_steps, 1000);
            testCase.verifyEqual(problem.options.hmax, 0.5);
            testCase.verifyFalse(isfield(problem.options, "rtol"));
        end

        function plotDrawsThePhasePlaneByDefault(testCase)
            m = testCase.model(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 6]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            h = plot(tr, Parent = ax);
            line = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyNumElements(line, 1);
            testCase.verifyEqual(line.XData(:), tr.X(:, 1));
            testCase.verifyEqual(line.YData(:), tr.X(:, 2));
            start = findobj(ax, "Tag", "pc-start");
            testCase.verifyNumElements(start, 1);
            testCase.verifyEqual([start.XData start.YData], [1 0]);
            testCase.verifyNumElements(h, 2);
            testCase.verifyEqual(string(ax.XLabel.String), "x");
            testCase.verifyEqual(string(ax.YLabel.String), "y");
        end

        function plotCanDrawOneVariableAgainstTime(testCase)
            m = testCase.model(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 6]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot(tr, Var = "y", Parent = ax);
            line = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyEqual(line.XData(:), tr.T);
            testCase.verifyEqual(line.YData(:), tr.X(:, 2));
            testCase.verifyEqual(string(ax.XLabel.String), "t");
            testCase.verifyEqual(string(ax.YLabel.String), "y");
            plot(tr, Var = [2 1], Parent = ax);
            lines = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyNumElements(lines, 2);
        end

        function aOneVariableModelPlotsAgainstTime(testCase)
            m = testCase.model(testCase.Decay);
            tr = phasecont.simulate(m, 1, [0 2]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot(tr, Parent = ax);
            line = findobj(ax, "Tag", "pc-trajectory");
            testCase.verifyEqual(line.XData(:), tr.T);
            testCase.verifyEqual(string(ax.XLabel.String), "t");
        end

        function plotRespectsHoldAndAnArrayOfTrajectories(testCase)
            m = testCase.model(testCase.Rotation);
            a = phasecont.simulate(m, [1 0], [0 3]);
            b = phasecont.simulate(m, [0.5 0], [0 3]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot([a b], Parent = ax);
            testCase.verifyNumElements(findobj(ax, "Tag", "pc-trajectory"), 2);
            testCase.verifyNotEqual( ...
                findobj(ax, "Tag", "pc-start", "-and", "Type", "line").Color, ...
                [0 0 0]);
            testCase.verifyFalse(ishold(ax));
        end

        function plotRejectsAnUnknownVariable(testCase)
            m = testCase.model(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 1]);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            testCase.verifyError(@() plot(tr, Var = "z", Parent = ax), ...
                "phasecont:model:unknownVariable");
            testCase.verifyError(@() plot(tr, Var = 3, Parent = ax), ...
                "phasecont:model:unknownVariable");
            testCase.verifyError(@() plot(tr, Var = [1 2 1], Parent = ax), ...
                "phasecont:model:unknownVariable");
        end

        function aTrajectoryOutlivesItsModel(testCase)
            m = phasecont.Model.fromSource(testCase.Rotation);
            tr = phasecont.simulate(m, [1 0], [0 1]);
            delete(m);
            testCase.verifyNotEmpty(tr.X);
            f = figure(Visible = "off");
            testCase.addTeardown(@() close(f));
            ax = axes(f);
            plot(tr, Parent = ax);
            testCase.verifyEqual(string(ax.XLabel.String), "x1");
        end

        function badArgumentsRaiseNamedErrors(testCase)
            m = testCase.model(testCase.Rotation);
            bad = @(varargin) phasecont.simulate(m, varargin{:});
            testCase.verifyError(@() bad([1 0], [0 0]), "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 Inf]), "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], 5), "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 1 2]), "phasecont:model:badOption");
            testCase.verifyError(@() bad(1, [0 1]), "phasecont:model:badStart");
            testCase.verifyError(@() bad([1 NaN], [0 1]), "phasecont:model:badStart");
            testCase.verifyError(@() bad("x", [0 1]), "phasecont:model:badStart");
            testCase.verifyError(@() bad([1 0], [0 1], RelTol = 0), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 1], AbsTol = -1), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 1], MaxSteps = 2.5), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 1], Hmax = NaN), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() bad([1 0], [0 1], Hmax = 0), ...
                "phasecont:model:badOption");
        end

        function aPointOfAnotherModelIsRefused(testCase)
            a = testCase.model(testCase.Decay);
            b = testCase.model(testCase.Decay);
            eq = phasecont.equilibrium(a, 0.5);
            testCase.verifyError(@() phasecont.simulate(b, eq, [0 1]), ...
                "phasecont:model:badStart");
        end
    end
end
