classdef FitTest < matlab.unittest.TestCase
    % Tests phasecont.fit and FitResult (SPEC, "Data fitting (P37)").
    % Every test runs on the CPU and needs the pckern binary.

    properties (Constant)
        Decay = "par a=1" + newline + "x'=-a*x" + newline + ...
            "init x=1" + newline + "done" + newline
        TwoPar = "par a=1, b=2" + newline + "x'=-a*x+b" + newline + ...
            "init x=0" + newline + "done" + newline
        Line = "par b=1" + newline + "x'=b" + newline + ...
            "init x=0" + newline + "done" + newline
        Rotation = "par w=1" + newline + "x'=w*y" + newline + ...
            "y'=-w*x" + newline + "init x=1,y=0" + newline + "done" + newline
        Blow = "par c=1" + newline + "x'=c*x^2" + newline + ...
            "init x=1" + newline + "done" + newline
        Hidden = "par a=1, c=1" + newline + "x'=-a*x" + newline + ...
            "y'=-c*y" + newline + "init x=1,y=1" + newline + "done" + newline
        MemberStopped = "phasecont:ensemble:memberStopped"
    end

    properties (TestParameter)
        Required = struct( ...
            "Fit", struct("Name", "Fit", "Id", "missingFit"), ...
            "Observe", struct("Name", "Observe", "Id", "missingObserve"), ...
            "T0", struct("Name", "T0", "Id", "missingTime"), ...
            "FdStep", struct("Name", "FdStep", "Id", "missingFdStep"), ...
            "Tau", struct("Name", "Tau", "Id", "missingTau"), ...
            "GradTol", struct("Name", "GradTol", "Id", "missingGradTol"), ...
            "StepTol", struct("Name", "StepTol", "Id", "missingStepTol"), ...
            "MaxIter", struct("Name", "MaxIter", "Id", "missingMaxIter"))
        BadValue = struct( ...
            "FdStepZero", {{"FdStep", 0}}, ...
            "FdStepNegative", {{"FdStep", -1e-7}}, ...
            "TauNaN", {{"Tau", NaN}}, ...
            "GradTolInf", {{"GradTol", Inf}}, ...
            "StepTolText", {{"StepTol", "small"}}, ...
            "MaxIterZero", {{"MaxIter", 0}}, ...
            "MaxIterFraction", {{"MaxIter", 2.5}}, ...
            "MaxIterVector", {{"MaxIter", [10 20]}})
        Method = struct( ...
            "Rk4", {{"Dt", 0.01}}, ...
            "Dp5", {{"Rtol", 1e-8, "Atol", 1e-10}})
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
        end
    end

    methods
        function m = model(testCase, text)
            m = phasecont.Model.fromSource(text);
            testCase.addTeardown(@() delete(m));
        end

        function c = options(~, drop, over)
            % The name-value options every test starts from, as a cell
            % array. DROP names options to leave out, OVER is a struct of
            % values that replace the defaults.
            if nargin < 2 || isempty(drop)
                drop = string.empty;
            end
            if nargin < 3
                over = struct();
            end
            drop = string(drop);
            s = struct("Fit", "a", "Observe", "x", "T0", 0, "Dt", 0.01, ...
                "FdStep", 1e-7, "Tau", 1e-3, "GradTol", 1e-10, ...
                "StepTol", 1e-10, "MaxIter", 100);
            names = string(fieldnames(over))';
            for k = 1:numel(names)
                s.(names(k)) = over.(names(k));
            end
            for k = 1:numel(drop)
                s = rmfield(s, drop(k));
            end
            c = namedargs2cell(s);
        end

        function Y = ensembleData(~, m, x0, p, ts, method)
            % Observed first variable of the model, made by ensemble at
            % the step method of the fit.
            res = phasecont.ensemble(m, x0, p, [0 ts(:)'], ...
                "Backend", "cpu", "Precision", "double", method{:});
            Y = reshape(res.States(1, :, 1), numel(ts), 1);
        end
    end

    methods (Test)
        % ---- argument checks: nothing here has a default ----

        function everyRequiredOptionIsNamed(testCase, Required)
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options(Required.Name);
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:}), ...
                "phasecont:fit:" + Required.Id);
        end

        function theMethodIsRequired(testCase)
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options("Dt");
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:}), ...
                "phasecont:ensemble:missingMethod");
        end

        function aBadOptionValueIsRefused(testCase, BadValue)
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct(BadValue{1}, BadValue{2}));
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:}), ...
                "phasecont:fit:badOption");
        end

        function aDataTimeAtOrBeforeTheStartIsRefused(testCase)
            m = testCase.model(testCase.Decay);
            c = testCase.options();
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], [0; 1], [1; 0.3], c{:}), ...
                "phasecont:fit:badTime");
            c = testCase.options([], struct("T0", 0.6));
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], [0.5; 1], [0.7; 0.3], c{:}), ...
                "phasecont:fit:badTime");
        end

        function dataTimesMustIncrease(testCase)
            m = testCase.model(testCase.Decay);
            c = testCase.options();
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], [1; 0.5], [0.3; 0.6], c{:}), ...
                "phasecont:fit:badTime");
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], [1; 1], [0.3; 0.3], c{:}), ...
                "phasecont:fit:badTime");
        end

        function theDataMustFitTheTimesAndTheObservedVariables(testCase)
            m = testCase.model(testCase.Decay);
            c = testCase.options();
            t = (1:3)';
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, [0.3; 0.2], c{:}), ...
                "phasecont:fit:badData");
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, [0.3 0.3; 0.2 0.2; 0.1 0.1], c{:}), ...
                "phasecont:fit:badData");
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], t, [0.3; NaN; 0.1], c{:}), ...
                "phasecont:fit:badData");
        end

        function namesMustExist(testCase)
            m = testCase.model(testCase.Decay);
            t = (1:3)';
            Y = exp(-1.3*t);
            c = testCase.options([], struct("Observe", "z"));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:fit:unknownVariable");
            c = testCase.options([], struct("Fit", "zz"));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:model:unknownParameter");
            c = testCase.options([], struct("Fit", ["a" "a"]));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:fit:badFit");
            c = testCase.options([], struct("Observe", ["x" "x"]));
            testCase.verifyError(@() phasecont.fit(m, 1, [], [t t], [Y Y], c{:}), ...
                "phasecont:fit:badObserve");
        end

        % ---- the algorithm ----

        function aLinearModelHasGainRatioOne(testCase)
            % x' = b has x(t) = b t, so the residual is affine in b and the
            % quadratic model of Algorithm 3.16 is exact: rho is 1.
            m = testCase.model(testCase.Line);
            t = (0.5:0.5:3)';
            noise = [0.1; -0.05; 0.02; 0.07; -0.03; 0.04];
            Y = 2*t + noise;
            c = testCase.options([], struct("Fit", "b", "Dt", 0.05));
            r = phasecont.fit(m, 0, [], t, Y, c{:});
            testCase.verifyTrue(r.History.Accepted(1));
            testCase.verifyEqual(r.History.Rho(1), 1, "AbsTol", 1e-6);
            closedForm = sum(t .* Y) / sum(t.^2);
            testCase.verifyEqual(r.Estimate, closedForm, "AbsTol", 1e-6);
            testCase.verifyTrue(r.Converged);
            testCase.verifyTrue(any(r.Stop == ["gradient" "step"]));
        end

        function theJacobianMatchesTheClosedFormSensitivity(testCase, Method)
            % d x / d a = -t exp(-a t) for x' = -a x, x(0) = 1.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            over = struct("GradTol", 1e-6, "StepTol", 1e-9);
            c = testCase.options("Dt", over);
            r = phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:}, Method{:});
            exact = -t .* exp(-r.Estimate * t);
            testCase.verifySize(r.Jacobian, [numel(t) 1]);
            testCase.verifyLessThanOrEqual( ...
                max(abs(r.Jacobian - exact) ./ abs(exact)), 1e-5);
        end

        % ---- recovery ----

        function ensembleDataGivesBackTheParameters(testCase)
            m = testCase.model(testCase.TwoPar);
            ts = (0.5:0.5:4)';
            Y = testCase.ensembleData(m, 0, [1.5; 0.7], ts, {"Dt", 0.01});
            c = testCase.options([], struct("Fit", ["a" "b"]));
            r = phasecont.fit(m, 0, [], ts, Y, c{:});
            testCase.verifyEqual(r.Estimate, [1.5; 0.7], "AbsTol", 1e-6);
            testCase.verifyLessThanOrEqual(r.Cost, 1e-12);
            testCase.verifyTrue(r.Converged);
            testCase.verifyEqual(r.Parameters.a, r.Estimate(1));
            testCase.verifyEqual(r.Parameters.b, r.Estimate(2));
        end

        function theClosedFormDecayIsFoundAtAFineStep(testCase)
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct("Dt", 0.001));
            r = phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:});
            testCase.verifyEqual(r.Estimate, 1.3, "AbsTol", 1e-8);
        end

        function theStartTimeShiftsTheData(testCase)
            m = testCase.model(testCase.Decay);
            t = (1.5:0.5:3)';
            c = testCase.options([], struct("T0", 1));
            r = phasecont.fit(m, exp(-1.3), [], t, exp(-1.3*t), c{:});
            testCase.verifyEqual(r.Estimate, 1.3, "AbsTol", 1e-6);
        end

        function twoObservedVariablesKeepTheirOrder(testCase)
            m = testCase.model(testCase.Rotation);
            t = (0.5:0.5:3)';
            Y = [cos(1.2*t) -sin(1.2*t)];
            c = testCase.options([], struct("Fit", "w", "Observe", ["x" "y"]));
            r = phasecont.fit(m, [1; 0], [], t, Y, c{:});
            testCase.verifyEqual(r.Estimate, 1.2, "AbsTol", 1e-6);
            testCase.verifySize(r.Fitted, [6 2]);
            testCase.verifySize(r.Residual, [6 2]);
            testCase.verifyEqual(r.Residual, r.Fitted - Y, "AbsTol", 1e-12);
        end

        function aSubsetFitLeavesTheOthersUntouched(testCase)
            m = testCase.model(testCase.TwoPar);
            ts = (0.5:0.5:4)';
            Y = testCase.ensembleData(m, 0, [1.5; 0.9], ts, {"Dt", 0.01});
            c = testCase.options();
            r = phasecont.fit(m, 0, [1.2; 0.9], ts, Y, c{:});
            testCase.verifyEqual(r.Estimate, 1.5, "AbsTol", 1e-6);
            testCase.verifyEqual(r.P(2), 0.9);
            testCase.verifyEqual(r.P(1), r.Estimate);
            % A struct changes named values and keeps the model's others.
            r = phasecont.fit(m, 0, struct("a", 1.2), ts, Y, c{:});
            testCase.verifyEqual(r.P(2), m.ParameterValues(2));
        end

        function reachingMaxIterIsNotAnError(testCase)
            m = testCase.model(testCase.TwoPar);
            ts = (0.5:0.5:4)';
            Y = testCase.ensembleData(m, 0, [1.5; 0.7], ts, {"Dt", 0.01});
            c = testCase.options([], struct("Fit", ["a" "b"], "MaxIter", 1));
            r = phasecont.fit(m, 0, [], ts, Y, c{:});
            testCase.verifyFalse(r.Converged);
            testCase.verifyEqual(r.Stop, "maxIter");
            testCase.verifyEqual(r.Iterations, 1);
        end

        % ---- members that do not finish ----

        function aStartPointThatBlowsUpIsRefused(testCase)
            % x' = c x^2 with x(0) = 1 and c = 1 blows up at t = 1.
            m = testCase.model(testCase.Blow);
            c = testCase.options([], struct("Fit", "c"));
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, [], [2; 3], [2; 3], c{:}), ...
                "phasecont:fit:startFailed");
        end

        function aTrialPointThatBlowsUpIsARejectedStep(testCase)
            % x(t) = 1/(1 - c t). From c = 0.1 the first Gauss-Newton step
            % lands beyond the pole at c = 0.5 for t = 2.
            m = testCase.model(testCase.Blow);
            t = (0.5:0.5:2)';
            Y = 1 ./ (1 - 0.45*t);
            c = testCase.options([], struct("Fit", "c", "Tau", 1e-6));
            r = phasecont.fit(m, 1, 0.1, t, Y, c{:});
            h = r.History;
            testCase.verifyTrue(any(~h.Accepted & isnan(h.Rho)));
            testCase.verifyEqual(r.Estimate, 0.45, "AbsTol", 1e-6);
            testCase.verifyTrue(r.Converged);
        end

        function aPerturbedPointThatBlowsUpAtTheStartIsRefused(testCase)
            % The start point is 1e-8 below the pole at t = 2; the point
            % one forward-difference step above it is past the pole.
            m = testCase.model(testCase.Blow);
            c = testCase.options("Dt", struct("Fit", "c", ...
                "Rtol", 1e-8, "Atol", 1e-10));
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, 0.5*(1 - 1e-8), [1; 2], [2; 10], c{:}), ...
                "phasecont:fit:jacobianFailed");
        end

        function aParameterThatDoesNotReachTheDataIsRefused(testCase)
            m = testCase.model(testCase.Hidden);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct("Fit", ["a" "c"]));
            try
                phasecont.fit(m, [1; 1], [], t, exp(-1.3*t), c{:});
                testCase.verifyFail("fit did not raise unidentifiable");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:fit:unidentifiable");
                testCase.verifyTrue(contains(err.message, """c"""));
            end
        end

        % ---- warnings ----

        function theMemberStoppedWarningIsRestored(testCase)
            id = testCase.MemberStopped;
            saved = warning("query", id);
            testCase.addTeardown(@() warning(saved));
            m = testCase.model(testCase.Blow);
            t = (0.5:0.5:2)';
            Y = 1 ./ (1 - 0.45*t);
            c = testCase.options([], struct("Fit", "c", "Tau", 1e-6));
            for state = ["on" "off"]
                warning(state, id);
                phasecont.fit(m, 1, 0.1, t, Y, c{:});
                testCase.verifyEqual(string(warning("query", id).state), state);
                testCase.verifyError( ...
                    @() phasecont.fit(m, 1, [], [2; 3], [2; 3], c{:}), ...
                    "phasecont:fit:startFailed");
                testCase.verifyEqual(string(warning("query", id).state), state);
            end
        end

        function aFitIssuesNoWarningOfItsOwn(testCase)
            warning("on", testCase.MemberStopped);
            m = testCase.model(testCase.Blow);
            t = (0.5:0.5:2)';
            Y = 1 ./ (1 - 0.45*t);
            c = testCase.options([], struct("Fit", "c", "Tau", 1e-6));
            testCase.verifyWarningFree(@() phasecont.fit(m, 1, 0.1, t, Y, c{:}));
        end
    end
end
