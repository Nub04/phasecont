classdef RunBoundedTest < matlab.unittest.TestCase
    % Tests phasecont.internal.runBounded with small system programs, so
    % no TeX install is needed. Every test skips when MATLAB has no JVM.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
            testCase.assumeTrue(usejava("jvm"), "MATLAB runs without a JVM");
        end
    end

    methods (Test)
        function finiteLimitStopsAProgramThatDoesNotEnd(testCase)
            clock = tic;
            r = phasecont.internal.runBounded(RunBoundedTest.sleeper(60), ...
                string(tempdir), 1);
            testCase.verifyLessThan(toc(clock), 30);
            testCase.verifyTrue(r.TimedOut);
            testCase.verifyEqual(r.Status, -1);
        end

        function finiteLimitLeavesAFastProgramAlone(testCase)
            r = phasecont.internal.runBounded(["whoami"], string(tempdir), 60);
            testCase.verifyFalse(r.TimedOut);
            testCase.verifyEqual(r.Status, 0);
            testCase.verifyNotEmpty(strtrim(r.Output));
        end

        function infiniteLimitRunsThroughTheJvmWithoutAWarning(testCase)
            testCase.verifyWarningFree(@() phasecont.internal.runBounded( ...
                ["whoami"], string(tempdir), Inf));
            r = phasecont.internal.runBounded(["whoami"], string(tempdir), Inf);
            testCase.verifyFalse(r.TimedOut);
            testCase.verifyEqual(r.Status, 0);
            testCase.verifyNotEmpty(strtrim(r.Output));
        end

        function infiniteLimitWaitsForAProgramThatTakesAWhile(testCase)
            clock = tic;
            r = phasecont.internal.runBounded(RunBoundedTest.sleeper(2), ...
                string(tempdir), Inf);
            testCase.verifyGreaterThanOrEqual(toc(clock), 1.5);
            testCase.verifyFalse(r.TimedOut);
            testCase.verifyEqual(r.Status, 0);
        end

        function infiniteLimitTakesTheJvmRouteNotSystem(testCase)
            % The system route refuses an argument with a space and raises
            % badArgument. The JVM route passes it on, whoami fails on it, and
            % the call returns that exit code.
            r = phasecont.internal.runBounded(["whoami", "a b"], ...
                string(tempdir), Inf);
            testCase.verifyNotEqual(r.Status, 0);
            testCase.verifyNotEqual(r.Status, -1);
            testCase.verifyFalse(r.TimedOut);
        end

        function aProgramThatCannotStartFallsBackQuietlyForInf(testCase)
            name = "phasecont-no-such-program";
            testCase.verifyWarningFree(@() RunBoundedTest.tryRun(name, Inf));
            testCase.verifyWarning(@() RunBoundedTest.tryRun(name, 30), ...
                "phasecont:latex:noTimeout");
        end

        function aCmdWrapperOnThePathStartsOnWindows(testCase)
            testCase.assumeTrue(ispc, "Only Windows needs the lookup");
            folder = string(tempname(tempdir));
            mkdir(folder);
            testCase.addTeardown(@() rmdir(folder, "s"));
            fid = fopen(fullfile(folder, "pcwrapper.cmd"), "w");
            fprintf(fid, "@echo wrapper says hi\r\n");
            fclose(fid);
            oldPath = getenv("PATH");
            setenv("PATH", folder + ";" + oldPath);
            testCase.addTeardown(@() setenv("PATH", oldPath));
            testCase.verifyWarningFree(@() phasecont.internal.runBounded( ...
                ["pcwrapper"], string(tempdir), 60));
            r = phasecont.internal.runBounded(["pcwrapper"], string(tempdir), 60);
            testCase.verifyEqual(r.Status, 0);
            testCase.verifySubstring(r.Output, "wrapper says hi");
            testCase.verifyFalse(r.TimedOut);
        end
    end

    methods (Static, Access = private)
        function argv = sleeper(seconds)
            % A program that ends only after SECONDS, on every platform.
            if ispc
                argv = ["ping", "-n", string(seconds + 1), "127.0.0.1"];
            else
                argv = ["sleep", string(seconds)];
            end
        end

        function tryRun(name, timeout)
            % The system fallback runs a program that does not exist and
            % returns a nonzero status. Only warnings matter here.
            try
                phasecont.internal.runBounded(name, string(tempdir), timeout);
            catch
                % The fallback may refuse or fail. The warning is the point.
            end
        end
    end
end
