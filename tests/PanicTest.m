classdef PanicTest < matlab.unittest.TestCase
    % A panic in the engine reaches MATLAB as an error and does not
    % damage later calls.

    properties
        File
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
            testCase.File = fullfile(t.Folder, "p.ode");
            fid = fopen(testCase.File, "w");
            fprintf(fid, "par a=2\ninit x=1\nx'=-a*x\ndone\n");
            fclose(fid);
        end
    end

    methods (Test)
        function panicThenModelWorks(testCase)
            try
                pckern("panic");
                testCase.verifyFail("pckern(""panic"") did not raise");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:engine:panic");
                testCase.verifySubstring(err.message, ".rs:");
            end
            m = phasecont.Model(testCase.File);
            testCase.verifyEqual(m.rhs(3), -6);
        end

        function panicInWorkerReachesPoll(testCase)
            r = pckern("panic_in_worker");
            cleanup = onCleanup(@() pckern("free", r));
            t0 = tic;
            id = "";
            while toc(t0) < 10
                try
                    pckern("run_poll", r);
                catch err
                    id = string(err.identifier);
                    break
                end
                pause(0.01);
            end
            testCase.verifyEqual(id, "phasecont:engine:panic");
            clear cleanup
        end
    end
end
