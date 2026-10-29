classdef BuildTest < matlab.unittest.TestCase
    % Checks the binary folder rule and the mex task's missing-source error.

    methods (Test)
        function binFolderHonorsEnvironmentOverride(testCase)
            testCase.applyFixture( ...
                matlab.unittest.fixtures.EnvironmentVariableFixture( ...
                "PHASECONT_BIN", "override-folder"));
            testCase.verifyEqual(fixtures.binFolder("anything"), ...
                "override-folder");
        end

        function wslRootOnWindowsUsesStageFolder(testCase)
            testCase.assumeTrue(ispc);
            testCase.applyFixture( ...
                matlab.unittest.fixtures.EnvironmentVariableFixture( ...
                "PHASECONT_BIN", ""));
            root = "\\wsl.localhost\X\home\u\repo";
            expected = string(fullfile(getenv("USERPROFILE"), ...
                "phasecont-stage", "bin", "win64"));
            testCase.verifyEqual(fixtures.binFolder(root), expected);
        end

        function wslRootElsewhereUsesToolboxBin(testCase)
            testCase.assumeFalse(ispc);
            testCase.applyFixture( ...
                matlab.unittest.fixtures.EnvironmentVariableFixture( ...
                "PHASECONT_BIN", ""));
            root = "\\wsl.localhost\X\home\u\repo";
            testCase.verifyEqual(fixtures.binFolder(root), ...
                string(fullfile(root, "toolbox", "bin", computer("arch"))));
        end

        function plainRootUsesToolboxBin(testCase)
            testCase.applyFixture( ...
                matlab.unittest.fixtures.EnvironmentVariableFixture( ...
                "PHASECONT_BIN", ""));
            root = string(tempdir) + "plainroot";
            testCase.verifyEqual(fixtures.binFolder(root), ...
                string(fullfile(root, "toolbox", "bin", computer("arch"))));
        end

        function mexTaskWithoutGatewayRaisesNoGateway(testCase)
            % Called directly: running "mex" through the plan would run the
            % cargo task first, which starts a real build.
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.assumeFalse(isfile(fullfile(root, "mex", ...
                "pckern_gateway.c")), "gateway source exists");
            plan = matlab.buildtool.Plan.load(root);
            testCase.verifyEqual(string(plan("mex").Dependencies), "cargo");
            % run() reports a failed task in its result and prints the
            % error, so check both.
            [output, result] = evalc("runMexTaskOnly(root)");
            testCase.verifyTrue(any([result.TaskResults.Failed]));
            testCase.verifySubstring(output, "phasecont:build:noGateway");
        end
    end
end

function result = runMexTaskOnly(root)
% Run the mex task alone, without its cargo dependency.
plan = matlab.buildtool.Plan.load(root);
plan("mex").Dependencies = string.empty;
result = run(plan, "mex");
end
