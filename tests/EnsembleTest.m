classdef EnsembleTest < matlab.unittest.TestCase
    % Tests phasecont.ensemble, EnsembleResult, gpuDevices and gpuSelfTest.
    % The CPU tests run on any machine that has the pckern binary. A GPU
    % test with no device of its backend prints SKIPPED and is filtered.
    % Set PC_REQUIRE_GPU=cuda,opencl to turn such a skip into a failure, so
    % a green run cannot hide that no GPU ran.

    properties (Constant)
        Rotation = "x'=y" + newline + "y'=-x" + newline + ...
            "init x=1,y=0" + newline + "done" + newline
        Decay = "par a=1" + newline + "x'=-a*x" + newline + ...
            "init x=1" + newline + "done" + newline
        Blow = "x'=x^2" + newline + "init x=1" + newline + "done" + newline
        TwoPar = "par a=1, b=2" + newline + "x'=-a*x+b" + newline + ...
            "init x=0" + newline + "done" + newline
    end

    properties (TestParameter)
        GpuBackend = {"cuda", "opencl"}
        GpuPrecision = {"single", "double"}
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

        function skipUnlessDevice(testCase, backend, precision)
            % Filters the test when the machine has no such device, or
            % fails it when PC_REQUIRE_GPU names the backend.
            devs = phasecont.gpuDevices();
            rows = devs(devs.Backend == backend, :);
            if precision == "double"
                rows = rows(rows.Fp64, :);
            end
            if ~isempty(rows)
                return
            end
            required = strtrim(split(string(getenv("PC_REQUIRE_GPU")), ","));
            if any(required == backend)
                testCase.verifyFail(sprintf( ...
                    "PC_REQUIRE_GPU names %s but no %s device runs %s precision.", ...
                    backend, backend, precision));
                testCase.assumeFail("required device missing");
            end
            fprintf("SKIPPED: no %s device for %s precision\n", backend, precision);
            testCase.assumeFail("no " + backend + " device");
        end
    end

    methods (Test)
        % ---- argument checks: nothing here has a default ----

        function precisionIsRequired(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Dt = 0.01), "phasecont:ensemble:missingPrecision");
        end

        function precisionMustBeSingleOrDouble(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "half", Dt = 0.01), "phasecont:ensemble:badOption");
        end

        function methodIsRequired(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double"), "phasecont:ensemble:missingMethod");
        end

        function bothToleranceValuesAreRequired(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Rtol = 1e-6), ...
                "phasecont:ensemble:missingTolerance");
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Atol = 1e-9), ...
                "phasecont:ensemble:missingTolerance");
        end

        function stepLengthAndToleranceTogetherAreRejected(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, Rtol = 1e-6, Atol = 1e-9), ...
                "phasecont:ensemble:ambiguousMethod");
        end

        function hmaxNeedsTolerances(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, Hmax = 0.1), ...
                "phasecont:ensemble:badOption");
        end

        function timesMustIncrease(testCase)
            m = testCase.model(testCase.Rotation);
            bad = {[0 0], [1 0], [0 2 1 3], [0 Inf], 5, [0 NaN 1]};
            for k = 1:numel(bad)
                testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], bad{k}, ...
                    Precision = "double", Dt = 0.01), ...
                    "phasecont:ensemble:badOption");
            end
        end

        function startStatesMustMatchTheModel(testCase)
            m = testCase.model(testCase.Rotation);
            for x0 = {zeros(3, 2), [1 NaN; 0 0], [], "ab", 1+2i}
                testCase.verifyError(@() phasecont.ensemble(m, x0{1}, [], [0 1], ...
                    Precision = "double", Dt = 0.01), "phasecont:model:badStart");
            end
        end

        function parametersMustMatchTheModel(testCase)
            m = testCase.model(testCase.Decay);
            x0 = ones(1, 4);
            for p = {ones(2, 4), ones(1, 3), NaN(1, 4), "a"}
                testCase.verifyError(@() phasecont.ensemble(m, x0, p{1}, [0 1], ...
                    Precision = "double", Dt = 0.01), ...
                    "phasecont:model:badParameter");
            end
            testCase.verifyError(@() phasecont.ensemble(m, x0, struct("nope", 1), ...
                [0 1], Precision = "double", Dt = 0.01), ...
                "phasecont:model:unknownParameter");
            r = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(r, [1; 0], 1, [0 1], ...
                Precision = "double", Dt = 0.01), "phasecont:model:badParameter");
        end

        function maxStepsAndBoundsAreChecked(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, MaxSteps = 0), ...
                "phasecont:ensemble:badOption");
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, MaxSteps = 1.5), ...
                "phasecont:ensemble:badOption");
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, Bounds = -1), ...
                "phasecont:ensemble:badOption");
        end

        function deviceNeedsAnExplicitBackend(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Precision = "double", Dt = 0.01, Device = 0), ...
                "phasecont:ensemble:badOption");
        end

        function toleranceBelowThePrecisionIsAnError(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Backend = "cpu", Precision = "single", Rtol = 1e-8, Atol = 1e-10), ...
                "phasecont:ensemble:toleranceTooTight");
            % The same tolerance is fine in double precision.
            r = phasecont.ensemble(m, [1; 0], [], [0 1], Backend = "cpu", ...
                Precision = "double", Rtol = 1e-8, Atol = 1e-10);
            testCase.verifyEqual(r.Status, "ok");
        end

        function missingDeviceIsAnEngineError(testCase)
            m = testCase.model(testCase.Rotation);
            testCase.verifyError(@() phasecont.ensemble(m, [1; 0], [], [0 1], ...
                Backend = "opencl", Device = 99, Precision = "double", Dt = 0.01), ...
                "phasecont:engine:gpuUnavailable");
        end

        % ---- CPU results against closed forms ----

        function rotationMatchesTheExactSolutionInDouble(testCase)
            m = testCase.model(testCase.Rotation);
            x0 = [1 0 0.5; 0 1 -2];
            r = phasecont.ensemble(m, x0, [], [0 1 2.5 4], Backend = "cpu", ...
                Precision = "double", Dt = 1e-3);
            testCase.verifyClass(r, "phasecont.EnsembleResult");
            testCase.verifyEqual(r.Times, [1; 2.5; 4]);
            testCase.verifySize(r.States, [2 3 3]);
            testCase.verifyEqual(r.Members, 3);
            testCase.verifyEqual(r.Status, ["ok"; "ok"; "ok"]);
            t = reshape(r.Times, 1, [], 1);
            a = reshape(x0(1, :), 1, 1, []);
            b = reshape(x0(2, :), 1, 1, []);
            exact = [a .* cos(t) + b .* sin(t); -a .* sin(t) + b .* cos(t)];
            testCase.verifyLessThan(max(abs(r.States - exact), [], "all"), 1e-9);
            testCase.verifyEqual(r.Final, reshape(r.States(:, end, :), 2, 3), ...
                "AbsTol", 0);
            testCase.verifyEqual(r.TReached, [4; 4; 4], "AbsTol", 1e-12);
            % Each of the three segments rounds its step count up, so allow a few.
            testCase.verifyGreaterThanOrEqual(r.Accepted, [4000; 4000; 4000]);
            testCase.verifyLessThanOrEqual(r.Accepted, [4003; 4003; 4003]);
            testCase.verifyEqual(r.Rejected, [0; 0; 0]);
            testCase.verifyEqual(r.Label.Backend, "cpu");
            testCase.verifyEqual(r.Label.Precision, "double");
            testCase.verifyFalse(r.Label.SinglePrecision);
        end

        function adaptiveDecayWithOneRatePerMember(testCase)
            m = testCase.model(testCase.Decay);
            rates = [0.5 1 2 4];
            r = phasecont.ensemble(m, ones(1, 4), rates, [0 3], Backend = "cpu", ...
                Precision = "double", Rtol = 1e-10, Atol = 1e-12);
            testCase.verifySize(r.States, [1 1 4]);
            testCase.verifyEqual(r.P, rates);
            testCase.verifyEqual(r.Final, exp(-3 * rates), ...
                "RelTol", 1e-8, "AbsTol", 1e-10);
            testCase.verifyGreaterThan(min(r.Accepted), 0);
        end

        function aSharedParameterSetIsRepeated(testCase)
            m = testCase.model(testCase.TwoPar);
            r = phasecont.ensemble(m, zeros(1, 3), [2; 4], [0 50], ...
                Backend = "cpu", Precision = "double", Rtol = 1e-10, Atol = 1e-12);
            testCase.verifyEqual(r.P, repmat([2; 4], 1, 3));
            testCase.verifyEqual(r.Final, [2 2 2], "AbsTol", 1e-8);
        end

        function aStructChangesNamedParametersOnly(testCase)
            m = testCase.model(testCase.TwoPar);
            r = phasecont.ensemble(m, zeros(1, 2), struct("b", 6), [0 50], ...
                Backend = "cpu", Precision = "double", Rtol = 1e-10, Atol = 1e-12);
            testCase.verifyEqual(r.P, repmat([1; 6], 1, 2));
            testCase.verifyEqual(r.Final, [6 6], "AbsTol", 1e-8);
        end

        function emptyParametersUseTheModelValues(testCase)
            m = testCase.model(testCase.TwoPar);
            r = phasecont.ensemble(m, 0, [], [0 50], Backend = "cpu", ...
                Precision = "double", Rtol = 1e-10, Atol = 1e-12);
            testCase.verifyEqual(r.P, [1; 2]);
            testCase.verifyEqual(r.Final, 2, "AbsTol", 1e-8);
        end

        function aRowVectorIsOneMember(testCase)
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1 0], [], [0 1], Backend = "cpu", ...
                Precision = "double", Dt = 0.01);
            testCase.verifyEqual(r.Members, 1);
            testCase.verifyEqual(r.X0, [1; 0]);
        end

        function singlePrecisionIsLabelled(testCase)
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1; 0], [], [0 5], Backend = "cpu", ...
                Precision = "single", Dt = 0.01);
            testCase.verifyTrue(r.Label.SinglePrecision);
            testCase.verifyEqual(r.Label.Precision, "single");
            testCase.verifyLessThan(abs(r.Final(1) - cos(5)), 1e-3);
        end

        function repeatedRunsAreIdentical(testCase)
            m = testCase.model(testCase.Rotation);
            x0 = [linspace(-1, 1, 40); linspace(2, -2, 40)];
            args = {"Backend", "cpu", "Precision", "double", "Rtol", 1e-8, "Atol", 1e-10};
            a = phasecont.ensemble(m, x0, [], [0 5], args{:});
            b = phasecont.ensemble(m, x0, [], [0 5], args{:});
            c = phasecont.ensemble(m, x0(:, 7:9), [], [0 5], args{:});
            testCase.verifyEqual(a.States, b.States, "AbsTol", 0);
            testCase.verifyEqual(c.States, a.States(:, :, 7:9), "AbsTol", 0);
        end

        % ---- stop reasons ----

        function boundsStopsAMemberWithoutAWarning(testCase)
            m = testCase.model(testCase.Blow);
            testCase.verifyWarningFree(@() phasecont.ensemble(m, [0 1], [], [0 2], ...
                Backend = "cpu", Precision = "double", Rtol = 1e-8, Atol = 1e-10, ...
                Bounds = 1e3));
            r = phasecont.ensemble(m, [0 1], [], [0 2], Backend = "cpu", ...
                Precision = "double", Rtol = 1e-8, Atol = 1e-10, Bounds = 1e3);
            testCase.verifyEqual(r.Status, ["ok"; "bounds"]);
            testCase.verifyEqual(r.StatusCode, uint8([0; 1]));
            testCase.verifyLessThan(r.TReached(2), 1);
            testCase.verifyEqual(r.TReached(1), 2);
        end

        function maxStepsWarnsAndKeepsTheLastState(testCase)
            m = testCase.model(testCase.Rotation);
            launch = @() phasecont.ensemble(m, [1; 0], [], [0 10], Backend = "cpu", ...
                Precision = "double", Dt = 0.01, MaxSteps = 10);
            testCase.verifyWarning(launch, "phasecont:ensemble:memberStopped");
            state = warning("off", "phasecont:ensemble:memberStopped");
            testCase.addTeardown(@() warning(state));
            r = launch();
            testCase.verifyEqual(r.Status, "maxSteps");
            testCase.verifyEqual(r.TReached, 0.1, "AbsTol", 1e-12);
        end

        % ---- refine ----

        function refineRerunsOneMemberInDouble(testCase)
            m = testCase.model(testCase.Rotation);
            x0 = [1 0 0.5; 0 1 -2];
            r = phasecont.ensemble(m, x0, [], [0 1 4], Backend = "cpu", ...
                Precision = "single", Rtol = 1e-5, Atol = 1e-8);
            f = r.refine(3);
            testCase.verifyClass(f, "phasecont.EnsembleResult");
            testCase.verifyEqual(f.Members, 1);
            testCase.verifyFalse(f.Label.SinglePrecision);
            testCase.verifyEqual(f.Label.Backend, "cpu");
            testCase.verifyEqual(f.X0, x0(:, 3));
            testCase.verifyEqual(f.Times, r.Times);
            t = f.Times.';
            exact = [0.5 * cos(t) - 2 * sin(t); -0.5 * sin(t) - 2 * cos(t)];
            testCase.verifyLessThan(max(abs(squeeze(f.States) - exact), [], "all"), 1e-4);
        end

        function refineKeepsTheMethodOfTheRun(testCase)
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1 0.5; 0 0], [], [0 2], Backend = "cpu", ...
                Precision = "single", Dt = 0.01, MaxSteps = 500);
            f = r.refine(2);
            testCase.verifyEqual(f.Job.Method, "rk4");
            testCase.verifyEqual(f.Job.Dt, 0.01);
            testCase.verifyEqual(f.Job.MaxSteps, 500);
            testCase.verifyEqual(f.Accepted, 200, "AbsTol", 1);
        end

        function refineChecksTheIndex(testCase)
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1 0.5; 0 0], [], [0 1], Backend = "cpu", ...
                Precision = "double", Dt = 0.01);
            for k = {0, 3, 1.5, NaN, [1 3], -1}
                testCase.verifyError(@() r.refine(k{1}), "phasecont:ensemble:badIndex");
            end
        end

        % ---- devices and the self-test ----

        function theCpuIsAlwaysListed(testCase)
            [d, b] = phasecont.gpuDevices();
            testCase.verifyEqual(string(d.Properties.VariableNames), ...
                ["Backend" "Index" "Name" "Driver" "Fp32" "Fp64" "MemoryMiB"]);
            testCase.verifyEqual(string(b.Properties.VariableNames), ...
                ["Backend" "State" "Why"]);
            cpu = d(d.Backend == "cpu", :);
            testCase.verifyEqual(height(cpu), 1);
            testCase.verifyEqual(cpu.Index, 0);
            testCase.verifyTrue(cpu.Fp32 && cpu.Fp64);
            testCase.verifyTrue(any(b.Backend == "cpu" & b.State == "available"));
            testCase.verifyTrue(all(ismember(b.State, ...
                ["available" "not_built" "unavailable"])));
        end

        function autoPicksADeviceThatRunsThePrecision(testCase)
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1; 0], [], [0 1], Backend = "auto", ...
                Precision = "double", Dt = 0.01);
            d = phasecont.gpuDevices();
            testCase.verifyTrue(any(d.Backend == r.Label.Backend & d.Fp64));
            testCase.verifyEqual(r.Label.Precision, "double");
        end

        function selfTestPassesOnTheCpu(testCase, GpuPrecision)
            s = phasecont.gpuSelfTest(Backend = "cpu", Precision = GpuPrecision);
            testCase.verifyTrue(s.Pass, evalc("disp(s.Checks)"));
            testCase.verifyEqual(s.Backend, "cpu");
            testCase.verifyEqual(s.Precision, GpuPrecision);
            testCase.verifyEqual(s.SinglePrecision, GpuPrecision == "single");
            testCase.verifyEqual(height(s.Checks), 4);
            testCase.verifyEqual(s.Checks.MaxError(3:4), [0; 0]);
        end

        function selfTestNeedsAPrecision(testCase)
            testCase.verifyError(@() phasecont.gpuSelfTest(Backend = "cpu"), ...
                "phasecont:ensemble:missingPrecision");
        end

        function aShortCpuRunDoesNotWaitForTheFullPollInterval(testCase)
            % The poll loop used to pause 50 ms after the first poll, so no
            % call took less than that. The fastest of several warm runs
            % has a loose bound that the old loop could never meet.
            m = testCase.model(testCase.Rotation);
            t = zeros(1, 8);
            for k = 1:numel(t)
                t0 = tic;
                phasecont.ensemble(m, [1 0.5; 0 0], [], [0 3], Backend = "cpu", ...
                    Precision = "double", Rtol = 1e-9, Atol = 1e-12);
                t(k) = toc(t0);
            end
            testCase.verifyLessThan(min(t), 0.045);
        end

        function selfTestOnAGpu(testCase, GpuBackend, GpuPrecision)
            testCase.skipUnlessDevice(GpuBackend, GpuPrecision);
            s = phasecont.gpuSelfTest(Backend = GpuBackend, Precision = GpuPrecision);
            testCase.verifyTrue(s.Pass, string(evalc("disp(s)")) + string(evalc("disp(s.Checks)")));
            testCase.verifyEqual(s.Backend, GpuBackend);
            testCase.verifyEqual(s.Precision, GpuPrecision);
            testCase.verifyEqual(s.SinglePrecision, GpuPrecision == "single");
        end

        function aGpuResultCanBeRefinedOnTheCpu(testCase, GpuBackend)
            testCase.skipUnlessDevice(GpuBackend, "single");
            m = testCase.model(testCase.Rotation);
            r = phasecont.ensemble(m, [1 0.5; 0 0], [], [0 3], Backend = GpuBackend, ...
                Precision = "single", Rtol = 1e-5, Atol = 1e-8);
            testCase.verifyTrue(r.Label.SinglePrecision);
            f = r.refine(1);
            testCase.verifyFalse(f.Label.SinglePrecision);
            testCase.verifyEqual(f.Label.Backend, "cpu");
            testCase.verifyLessThan(abs(f.Final(1) - cos(3)), 1e-4);
        end
    end
end
