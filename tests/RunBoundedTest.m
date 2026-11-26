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

        function aProgramThatIsNotFoundRaisesNoProgram(testCase)
            % No fallback to SYSTEM, for Inf and for a finite limit alike.
            name = "phasecont-no-such-program";
            testCase.verifyError(@() phasecont.internal.runBounded( ...
                name, string(tempdir), Inf), "phasecont:internal:noProgram");
            testCase.verifyError(@() phasecont.internal.runBounded( ...
                name, string(tempdir), 30), "phasecont:internal:noProgram");
        end

        function aProgramPlantedInTheFolderNeverRuns(testCase)
            % A file named like the program, in the folder the program runs
            % in but not on PATH, must not run on either route.
            folder = string(tempname(tempdir));
            mkdir(folder);
            testCase.addTeardown(@() rmdir(folder, "s"));
            marker = fullfile(folder, "planted-ran.txt");
            name = "pcplanted";
            if ispc
                file = fullfile(folder, name + ".bat");
                body = "@echo ran> """ + marker + """" + newline;
            else
                file = fullfile(folder, name);
                body = "#!/bin/sh" + newline + "echo ran > '" + marker + "'" + newline;
            end
            fid = fopen(file, "w");
            fprintf(fid, "%s", body);
            fclose(fid);
            if ~ispc
                fileattrib(char(file), "+x");
            end
            for limit = [Inf, 30]
                testCase.verifyError(@() phasecont.internal.runBounded( ...
                    name, folder, limit), "phasecont:internal:noProgram");
            end
            testCase.verifyFalse(isfile(marker));
        end

        function everyExitLeavesNoOutputFileInTheTempFolder(testCase)
            % The output file is deleted after the program stops, on a normal
            % end, on a timeout and when the program cannot start.
            before = string({dir(fullfile(tempdir, "tp*.txt")).name});
            phasecont.internal.runBounded(["whoami"], string(tempdir), Inf);
            phasecont.internal.runBounded(RunBoundedTest.sleeper(60), ...
                string(tempdir), 1);
            try
                phasecont.internal.runBounded("pc-no-such-program-xyz", ...
                    string(tempdir), Inf);
            catch
                % noProgram is expected. Only the temp folder matters here.
            end
            after = string({dir(fullfile(tempdir, "tp*.txt")).name});
            testCase.verifyEmpty(setdiff(after, before));
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

        function aTimeoutStopsTheChildOfACmdWrapper(testCase)
            % cmd.exe runs the wrapper and ping runs as its child. If the
            % child outlived the timeout it would hold the output file open,
            % so the delete would warn and leave the file behind.
            testCase.assumeTrue(ispc, "Only Windows runs .cmd wrappers");
            folder = string(tempname(tempdir));
            mkdir(folder);
            testCase.addTeardown(@() rmdir(folder, "s"));
            fid = fopen(fullfile(folder, "pcsleeper.cmd"), "w");
            fprintf(fid, "@ping -n 61 127.0.0.1\r\n");
            fclose(fid);
            oldPath = getenv("PATH");
            setenv("PATH", folder + ";" + oldPath);
            testCase.addTeardown(@() setenv("PATH", oldPath));
            before = string({dir(fullfile(tempdir, "tp*.txt")).name});
            clock = tic;
            r = testCase.verifyWarningFree(@() phasecont.internal.runBounded( ...
                ["pcsleeper"], string(tempdir), 1));
            testCase.verifyLessThan(toc(clock), 30);
            testCase.verifyTrue(r.TimedOut);
            testCase.verifyEqual(r.Status, -1);
            after = string({dir(fullfile(tempdir, "tp*.txt")).name});
            testCase.verifyEmpty(setdiff(after, before));
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
    end
end
