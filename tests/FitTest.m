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
        Ramp = "par b=1" + newline + "x'=b*t" + newline + ...
            "init x=0" + newline + "done" + newline
        MemberStopped = "phasecont:ensemble:memberStopped"
    end

    properties (TestParameter)
        Required = struct( ...
            "Fit", struct("Name", "Fit", "Id", "missingFit"), ...
            "Observe", struct("Name", "Observe", "Id", "missingObserve"), ...
            "T0", struct("Name", "T0", "Id", "missingTime"), ...
            "FdStep", struct("Name", "FdStep", "Id", "missingFdStep"), ...
            "FdScale", struct("Name", "FdScale", "Id", "missingFdScale"), ...
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
            "MaxIterVector", {{"MaxIter", [10 20]}}, ...
            "MaxIterNaN", {{"MaxIter", NaN}}, ...
            "MaxIterNegative", {{"MaxIter", -3}}, ...
            "TauZero", {{"Tau", 0}}, ...
            "StepTolNegative", {{"StepTol", -1e-8}}, ...
            "T0NaN", {{"T0", NaN}}, ...
            "T0Text", {{"T0", "0"}}, ...
            "T0Vector", {{"T0", [0 1]}}, ...
            "FdStepTooSmallToMoveTheParameter", {{"FdStep", 1e-17}}, ...
            "FdScaleZero", {{"FdScale", 0}}, ...
            "FdScaleNegative", {{"FdScale", -1}}, ...
            "FdScaleNaN", {{"FdScale", NaN}}, ...
            "FdScaleInf", {{"FdScale", Inf}}, ...
            "FdScaleText", {{"FdScale", "one"}}, ...
            "FdScaleTooLong", {{"FdScale", [1 1]}}, ...
            "FdScaleComplex", {{"FdScale", 1 + 1i}})
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
                "FdStep", 1e-7, "FdScale", 1, "Tau", 1e-3, "GradTol", 1e-10, ...
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

        function verifyDampingLaw(testCase, h)
            % Replays the update of mu and nu (Algorithm 3.16) over the
            % rows of History, so a change of the law is seen.
            nu = 2;
            for k = 1:height(h) - 1
                if h.Accepted(k)
                    expected = h.Mu(k) * max(1/3, 1 - (2*h.Rho(k) - 1)^3);
                    nu = 2;
                else
                    expected = h.Mu(k) * nu;
                    nu = 2 * nu;
                end
                testCase.verifyEqual(h.Mu(k+1), expected, "RelTol", 1e-12);
            end
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
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, [Y Y], c{:}), ...
                "phasecont:fit:badObserve");
        end

        function namesMustBeNonemptyText(testCase)
            m = testCase.model(testCase.Decay);
            t = (1:3)';
            Y = exp(-1.3*t);
            c = testCase.options([], struct("Fit", ""));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:fit:badFit");
            c = testCase.options([], struct("Fit", 5));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:fit:badFit");
            c = testCase.options([], struct("Observe", 5));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "phasecont:fit:badObserve");
        end

        function theStartStateMustBeOneFiniteVector(testCase)
            m = testCase.model(testCase.Rotation);
            t = (1:3)';
            Y = cos(t);
            c = testCase.options([], struct("Fit", "w"));
            for bad = {1, [1; NaN], [1 0; 0 1], "ab", true(2, 1)}
                testCase.verifyError( ...
                    @() phasecont.fit(m, bad{1}, [], t, Y, c{:}), ...
                    "phasecont:model:badStart");
            end
        end

        function theParametersMustFitTheModel(testCase)
            m = testCase.model(testCase.TwoPar);
            t = (1:3)';
            Y = t;
            c = testCase.options();
            testCase.verifyError( ...
                @() phasecont.fit(m, 0, [1; 2; 3], t, Y, c{:}), ...
                "phasecont:model:badParameter");
            testCase.verifyError( ...
                @() phasecont.fit(m, 0, struct("zz", 1), t, Y, c{:}), ...
                "phasecont:model:unknownParameter");
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
            % History: the last row holds the cost and gradient that the
            % result reports, and Iterations is the row count.
            h = r.History;
            testCase.verifyEqual(r.Iterations, height(h));
            testCase.verifyEqual(h.Cost(end), r.Cost);
            testCase.verifyEqual(h.GradNorm(end), r.GradNorm);
            testCase.verifyEqual(h.Iter, (1:height(h))');
        end

        function theResultMatchesAnIndependentComputation(testCase)
            % x' = b, x(0) = 0 has x(t) = b t, so the model, the residual,
            % the cost and the Jacobian at the estimate have closed forms.
            m = testCase.model(testCase.Line);
            t = (0.5:0.5:3)';
            noise = [0.1; -0.05; 0.02; 0.07; -0.03; 0.04];
            Y = 2*t + noise;
            c = testCase.options([], struct("Fit", "b"));
            r = phasecont.fit(m, 0, [], t, Y, c{:});
            b = sum(t .* Y) / sum(t.^2);
            testCase.verifyEqual(r.Estimate, b, "AbsTol", 1e-8);
            testCase.verifyEqual(r.Fitted, b*t, "AbsTol", 1e-8);
            testCase.verifyEqual(r.Residual, b*t - Y, "AbsTol", 1e-8);
            testCase.verifyEqual(r.Cost, 0.5*sum((b*t - Y).^2), "RelTol", 1e-8);
            testCase.verifyEqual(r.Jacobian, t, "AbsTol", 1e-6);
            testCase.verifyEqual(r.GradNorm, ...
                norm(r.Jacobian' * r.Residual(:), Inf), "AbsTol", 1e-12);
            testCase.verifyEqual(r.Fitted - Y, r.Residual, "AbsTol", 1e-12);
        end

        function aZeroStartStepsByFdStepTimesFdScale(testCase)
            % At q = 0 the step is FdStep*FdScale. With FdStep = 0.1 the
            % forward difference of x = exp(-a t) at a = 0 is
            % (exp(-eta t) - 1)/eta, so the first damping
            % Tau*sum(J.^2) tells which step was taken.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            Y = exp(-0.5*t);
            for scale = [1 2]
                eta = 0.1 * scale;
                c = testCase.options([], struct("FdStep", 0.1, "FdScale", scale));
                r = phasecont.fit(m, 1, 0, t, Y, c{:});
                J = (exp(-eta*t) - 1) / eta;
                testCase.verifyEqual(r.History.Mu(1), 1e-3 * sum(J.^2), "RelTol", 1e-6);
            end
        end

        function aTinyParameterGetsTheFdScaleFloor(testCase)
            % x = exp(-a t) with a = 1e-12 stays near 1. A step of
            % FdStep*abs(a) = 1e-19 cannot change a state of size 1 (the
            % rounding unit there is 1e-16), so the column of J was exactly
            % zero. With FdScale = 1 the step is 1e-7.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            a = 1e-12;
            Y = testCase.ensembleData(m, 1, a, t, {"Dt", 0.01});
            c = testCase.options([], struct("FdScale", 1));
            r = phasecont.fit(m, 1, a, t, Y, c{:});
            testCase.verifyEqual(r.Iterations, 0);
            testCase.verifyEqual(r.Stop, "gradient");
            exact = -t .* exp(-a*t);
            testCase.verifyLessThanOrEqual( ...
                max(abs(r.Jacobian - exact) ./ abs(exact)), 1e-5);
        end

        function aFdScaleSmallerThanTheParameterChangesNothing(testCase)
            % max(abs(q), s) = abs(q) when s <= abs(q), so the Jacobian is
            % the one of FdStep*abs(q), bit for bit.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            Y = exp(-1.3*t);
            c1 = testCase.options([], struct("FdScale", 1e-3));
            c2 = testCase.options([], struct("FdScale", 1e-9));
            r1 = phasecont.fit(m, 1, [], t, Y, c1{:});
            r2 = phasecont.fit(m, 1, [], t, Y, c2{:});
            testCase.verifyEqual(r1.Jacobian, r2.Jacobian);
            testCase.verifyEqual(r1.Estimate, r2.Estimate);
        end

        function aFdScaleVectorFollowsTheOrderOfFit(testCase)
            % Fit = ["c" "a"] while the model lists a before c. a = 1e-12
            % needs the scale 1 and c = 1 does not. If the vector were read
            % in the order of the model, a would get 1e-12 and its column
            % would be zero.
            m = testCase.model(testCase.Hidden);
            t = (0.5:0.5:3)';
            a = 1e-12;
            res = phasecont.ensemble(m, [1; 1], [a; 1], [0; t]', ...
                "Backend", "cpu", "Precision", "double", "Dt", 0.01);
            Y = reshape(res.States(:, :, 1), 2, numel(t)).';
            c = testCase.options([], struct("Fit", ["c" "a"], ...
                "Observe", ["x" "y"], "FdScale", [1e-12 1]));
            r = phasecont.fit(m, [1 1], [a; 1], t, Y, c{:});
            testCase.verifyEqual(r.Fit, ["c" "a"]);
            testCase.verifyEqual(r.Iterations, 0);
            exactA = [-t .* exp(-a*t); zeros(numel(t), 1)];
            exactC = [zeros(numel(t), 1); -t .* exp(-t)];
            testCase.verifyEqual(r.Jacobian(:, 2), exactA, "RelTol", 1e-5, "AbsTol", 1e-12);
            testCase.verifyEqual(r.Jacobian(:, 1), exactC, "RelTol", 1e-5, "AbsTol", 1e-12);
            % The same values swapped give a the scale 1e-12, a step of
            % 1e-19 and a zero column. A scalar read of the vector would
            % treat both orders alike.
            c = testCase.options([], struct("Fit", ["c" "a"], ...
                "Observe", ["x" "y"], "FdScale", [1 1e-12]));
            try
                phasecont.fit(m, [1 1], [a; 1], t, Y, c{:});
                testCase.verifyFail("fit did not raise unidentifiable");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:fit:unidentifiable");
                testCase.verifyTrue(contains(err.message, """a"""));
            end
        end

        function aFdScaleVectorIsCheckedEntryByEntry(testCase)
            % Two fitted parameters; one bad entry, a wrong length or a
            % matrix is refused before the model runs.
            m = testCase.model(testCase.TwoPar);
            t = (0.5:0.5:3)';
            Y = 2*(1 - exp(-t));
            for bad = {[1 0], [1 NaN], [1 -1], [1 Inf], [1 1 1], ones(2)}
                c = testCase.options([], struct("Fit", ["a" "b"], ...
                    "FdScale", bad{1}));
                testCase.verifyError(@() phasecont.fit(m, 0, [], t, Y, c{:}), ...
                    "phasecont:fit:badOption");
            end
        end

        function aLargeStepToleranceStopsAtTheStart(testCase)
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct("StepTol", 10));
            r = phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:});
            testCase.verifyEqual(r.Stop, "step");
            testCase.verifyTrue(r.Converged);
            testCase.verifyEqual(r.Iterations, 0);
            testCase.verifyEmpty(r.History);
            testCase.verifyEqual(r.Estimate, 1);
        end

        function aHugeMaxIterNeedsNoMemoryOfItsOwn(testCase)
            m = testCase.model(testCase.Line);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct("Fit", "b", "MaxIter", 1e9));
            r = phasecont.fit(m, 0, [], t, 2*t, c{:});
            testCase.verifyTrue(r.Converged);
            testCase.verifyLessThan(r.Iterations, 10);
        end

        function aFdStepThatDoesNotMoveTheParameterIsRefused(testCase)
            % q = 1 and 1 + 1e-17 is 1 in double precision.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            c = testCase.options([], struct("FdStep", 1e-17));
            try
                phasecont.fit(m, 1, [], t, exp(-1.3*t), c{:});
                testCase.verifyFail("fit did not raise badOption");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:fit:badOption");
                testCase.verifyTrue(contains(err.message, "FdStep"));
                testCase.verifyTrue(contains(err.message, """a"""));
            end
        end

        function theJacobianDividesByTheRealizedStep(testCase)
            % At a = 1 the step 1e-15 is taken as (1 + 1e-15) - 1, about
            % 1.11e-15. Dividing by 1e-15 instead would be 11% off. The
            % data is the model at a = 1, so the fit stops at the start and
            % its Jacobian is the one of the start point.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            h = 1e-15;
            taken = (1 + h) - 1;
            testCase.assertGreaterThan(abs(taken - h) / h, 0.05);
            res = phasecont.ensemble(m, [1 1], [1 1 + h], [0; t]', ...
                "Backend", "cpu", "Precision", "double", "Dt", 0.01);
            X = reshape(res.States, numel(t), 2);
            c = testCase.options([], struct("FdStep", h, "FdScale", 1));
            r = phasecont.fit(m, 1, 1, t, X(:, 1), c{:});
            testCase.assertEqual(r.Iterations, 0);
            testCase.verifyEqual(r.Jacobian, (X(:, 2) - X(:, 1)) / taken, ...
                "RelTol", 1e-12);
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
            testCase.verifyDampingLaw(r.History);
        end

        function aFitOrderThatIsNotTheModelOrderKeepsEveryColumnInPlace(testCase)
            % x' = -a x + b, x(0) = 0 has x = (b/a)(1 - exp(-a t)).
            m = testCase.model(testCase.TwoPar);
            ts = (0.5:0.5:4)';
            Y = testCase.ensembleData(m, 0, [1.5; 0.7], ts, {"Dt", 0.01});
            % Fit b alone, with a fixed at its true value.
            c = testCase.options([], struct("Fit", "b"));
            r = phasecont.fit(m, 0, [1.5; 2], ts, Y, c{:});
            testCase.verifyEqual(r.Estimate, 0.7, "AbsTol", 1e-6);
            testCase.verifyEqual(r.P, [1.5; 0.7], "AbsTol", 1e-6);
            % Fit b first, then a.
            c = testCase.options([], struct("Fit", ["b" "a"]));
            r = phasecont.fit(m, 0, [1.2; 0.9], ts, Y, c{:});
            testCase.verifyEqual(r.Estimate, [0.7; 1.5], "AbsTol", 1e-6);
            testCase.verifyEqual(r.P, [1.5; 0.7], "AbsTol", 1e-6);
            testCase.verifyEqual(r.Parameters.a, r.Estimate(2));
            testCase.verifyEqual(r.Parameters.b, r.Estimate(1));
            b = r.Estimate(1);
            a = r.Estimate(2);
            e = exp(-a*ts);
            dB = (1 - e) / a;
            dA = b * (-(1 - e) / a^2 + ts .* e / a);
            testCase.verifyEqual(r.Jacobian, [dB dA], "RelTol", 1e-4, "AbsTol", 1e-6);
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

        function theStartTimeIsTheTimeOfTheModel(testCase)
            % x' = b t from x(1) = 0.5 gives x(t) = 0.5 + b (t^2 - 1)/2.
            m = testCase.model(testCase.Ramp);
            t = (1.5:0.5:3)';
            Y = 0.5 + 3 * (t.^2 - 1) / 2;
            c = testCase.options([], struct("Fit", "b", "T0", 1));
            r = phasecont.fit(m, 0.5, [], t, Y, c{:});
            testCase.verifyEqual(r.Estimate, 3, "AbsTol", 1e-6);
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

        function anObservedOrderThatIsNotTheModelOrderKeepsItsColumns(testCase)
            m = testCase.model(testCase.Rotation);
            t = (0.5:0.5:3)';
            x = cos(1.2*t);
            y = -sin(1.2*t);
            c = testCase.options([], struct("Fit", "w", "Observe", "y"));
            r = phasecont.fit(m, [1; 0], [], t, y, c{:});
            testCase.verifyEqual(r.Estimate, 1.2, "AbsTol", 1e-6);
            c = testCase.options([], struct("Fit", "w", "Observe", ["y" "x"]));
            r = phasecont.fit(m, [1; 0], [], t, [y x], c{:});
            testCase.verifyEqual(r.Estimate, 1.2, "AbsTol", 1e-6);
            testCase.verifyEqual(r.Fitted, [y x], "AbsTol", 1e-5);
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

        function aStartAtTheOptimumStopsWithoutAStep(testCase)
            m = testCase.model(testCase.TwoPar);
            ts = (0.5:0.5:4)';
            Y = testCase.ensembleData(m, 0, [1.5; 0.7], ts, {"Dt", 0.01});
            c = testCase.options([], struct("Fit", ["a" "b"]));
            r = phasecont.fit(m, 0, struct("a", 1.5, "b", 0.7), ts, Y, c{:});
            testCase.verifyEqual(r.Iterations, 0);
            testCase.verifyEqual(r.Stop, "gradient");
            testCase.verifyTrue(r.Converged);
            testCase.verifyEmpty(r.History);
            testCase.verifyEqual(r.Estimate, [1.5; 0.7]);
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
            testCase.verifyDampingLaw(h);
            % A rejected step leaves the iterate, so its cost and gradient
            % are those of the row before.
            for k = find(~h.Accepted(:))'
                if k > 1
                    testCase.verifyEqual(h.Cost(k), h.Cost(k-1));
                    testCase.verifyEqual(h.GradNorm(k), h.GradNorm(k-1));
                end
            end
        end

        function aStartThatNeedsMoreStepsThanEnsembleAllowsIsRefused(testCase)
            % ensemble stops a trajectory after 100000 steps. Here Dt = 0.01
            % and the last time is 2000.
            m = testCase.model(testCase.Decay);
            c = testCase.options();
            try
                phasecont.fit(m, 1, [], 2000, 0, c{:});
                testCase.verifyFail("fit did not raise startFailed");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:fit:startFailed");
                testCase.verifyTrue(contains(err.message, "maxSteps"));
                testCase.verifyTrue(contains(err.message, "MaxSteps"));
            end
        end

        function maxStepsLetsALongSpanFinish(testCase)
            % The same span as above needs 200000 steps. MaxSteps = 300000
            % lets it finish, and the fit finds the rate that made the data.
            m = testCase.model(testCase.Decay);
            t = [1000; 2000];
            Y = reshape(phasecont.ensemble(m, 1, 2e-4, [0; t]', ...
                "Backend", "cpu", "Precision", "double", "Dt", 0.01, ...
                "MaxSteps", 300000).States, numel(t), 1);
            over = struct("FdScale", 1e-4, "GradTol", 1e-6, "StepTol", 1e-8, ...
                "MaxSteps", 300000);
            c = testCase.options([], over);
            r = phasecont.fit(m, 1, 1e-4, t, Y, c{:});
            testCase.verifyEqual(r.Estimate, 2e-4, "AbsTol", 1e-8);
            testCase.verifyTrue(r.Converged);
            % Without MaxSteps, or with an empty one, the limit is the
            % 100000 of ensemble and the start does not finish.
            for drop = [true false]
                if drop
                    c = testCase.options([], rmfield(over, "MaxSteps"));
                else
                    over.MaxSteps = [];
                    c = testCase.options([], over);
                end
                try
                    phasecont.fit(m, 1, 1e-4, t, Y, c{:});
                    testCase.verifyFail("fit did not raise startFailed");
                catch err
                    testCase.verifyEqual(string(err.identifier), ...
                        "phasecont:fit:startFailed");
                    testCase.verifyTrue(contains(err.message, "maxSteps"));
                    testCase.verifyTrue(contains(err.message, "MaxSteps"));
                end
            end
        end

        function aBadMaxStepsIsRefusedByEnsemble(testCase)
            % fit passes a given MaxSteps on unchanged, so the errors are
            % those of ensemble: badOption for a value, and the
            % mustBeScalarOrEmpty validator of ensemble for a size.
            m = testCase.model(testCase.Decay);
            t = (0.5:0.5:3)';
            Y = exp(-1.3*t);
            for bad = {0, 2.5, -1, NaN, Inf, 2^32}
                c = testCase.options([], struct("MaxSteps", bad{1}));
                testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                    "phasecont:ensemble:badOption");
            end
            c = testCase.options([], struct("MaxSteps", [1 2]));
            testCase.verifyError(@() phasecont.fit(m, 1, [], t, Y, c{:}), ...
                "MATLAB:validators:mustBeScalarOrEmpty");
        end

        function aMapIsRefused(testCase)
            % fit fits ODE models. A map is refused before any option is read.
            maps = ["par r=2" + newline + "init x=0.2" + newline + ...
                    "x(t+1)=r*x*(1-x)" + newline + "done" + newline, ...
                    "par r=2" + newline + "init x=0.2" + newline + ...
                    "x'=r*x*(1-x)" + newline + "@ meth=discrete" + newline + ...
                    "done" + newline];
            for text = maps
                m = testCase.model(text);
                testCase.verifyTrue(m.IsMap);
                c = testCase.options([], struct("Fit", "r"));
                testCase.verifyError(@() phasecont.fit(m, 0.2, [], (1:3)', ...
                    [0.3; 0.4; 0.5], c{:}), "phasecont:fit:mapModel");
                testCase.verifyError(@() phasecont.fit(m, 0.2, [], (1:3)', ...
                    [0.3; 0.4; 0.5]), "phasecont:fit:mapModel");
            end
        end

        function aPerturbedPointPastThePoleAtAnAcceptedStepIsRefused(testCase)
            % x(2) = 1/(1 - 2 c) has its pole at c = 0.5. From c = 0.38 the
            % first step is accepted at about 0.44, and the point one
            % forward-difference step above it is past the pole.
            m = testCase.model(testCase.Blow);
            over = struct("Fit", "c", "FdStep", 0.2, "FdScale", 0.01, ...
                "Tau", 1e-6);
            % The Jacobian at the start is defined: a huge GradTol stops
            % there without an error.
            stop = over;
            stop.GradTol = 1e10;
            c = testCase.options([], stop);
            r = phasecont.fit(m, 1, 0.38, 2, 10, c{:});
            testCase.assertEqual(r.Iterations, 0);
            testCase.assertEqual(r.Stop, "gradient");
            c = testCase.options([], over);
            testCase.verifyError( ...
                @() phasecont.fit(m, 1, 0.38, 2, 10, c{:}), ...
                "phasecont:fit:jacobianFailed");
        end

        function aPerturbedPointAtTheStepLimitNamesMaxSteps(testCase)
            % The rotation x' = w y, y' = -w x needs about twice the
            % Dormand-Prince steps at w = 2 as at w = 1. With FdStep = 1
            % the perturbed point of w = 1 is w = 2, so a MaxSteps between
            % the two counts stops only the perturbed point.
            saved = warning("off", testCase.MemberStopped);
            testCase.addTeardown(@() warning(saved));
            m = testCase.model(testCase.Rotation);
            tol = {"Rtol", 1e-8, "Atol", 1e-10};
            probe = phasecont.ensemble(m, [1 1; 0 0], [1 2], [0 10], ...
                "Backend", "cpu", "Precision", "double", tol{:});
            testCase.assertEqual(probe.Status, ["ok"; "ok"]);
            limit = floor(mean(probe.Accepted));
            check = phasecont.ensemble(m, [1 1; 0 0], [1 2], [0 10], ...
                "Backend", "cpu", "Precision", "double", tol{:}, ...
                "MaxSteps", limit);
            testCase.assertEqual(check.Status, ["ok"; "maxSteps"]);
            c = testCase.options("Dt", struct("Fit", "w", "FdStep", 1, ...
                "Rtol", 1e-8, "Atol", 1e-10, "MaxSteps", limit));
            try
                phasecont.fit(m, [1 0], 1, 10, cos(10), c{:});
                testCase.verifyFail("fit did not raise jacobianFailed");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:fit:jacobianFailed");
                testCase.verifyTrue(contains(err.message, "maxSteps"));
                testCase.verifyTrue(contains(err.message, "MaxSteps"));
            end
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
            saved = warning("query", testCase.MemberStopped);
            testCase.addTeardown(@() warning(saved));
            warning("on", testCase.MemberStopped);
            m = testCase.model(testCase.Blow);
            t = (0.5:0.5:2)';
            Y = 1 ./ (1 - 0.45*t);
            c = testCase.options([], struct("Fit", "c", "Tau", 1e-6));
            testCase.verifyWarningFree(@() phasecont.fit(m, 1, 0.1, t, Y, c{:}));
        end
    end
end
