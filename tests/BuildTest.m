classdef BuildTest < matlab.unittest.TestCase
    % Checks the binary folder rule and that the gateway source and build exist.

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

        function gatewayIsBuiltIntoTheBinFolder(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.verifyTrue(isfile(fullfile(root, "mex", ...
                "pckern_gateway.c")));
            built = fullfile(fixtures.binFolder(root), "pckern." + mexext);
            testCase.assumeTrue(isfile(built), ...
                "pckern has not been built (run buildtool mex)");
            testCase.verifyTrue(isfile(built));
        end
    end
end
