classdef BuildfileValueTest < matlab.unittest.TestCase
    % scripts/buildfileValue reads the version and the identifier from
    % buildfile.m, which scripts/cleanInstallCheck.m relies on.

    methods (TestClassSetup)
        function addScriptsToPath(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(root, "scripts")));
        end
    end

    methods (Test)
        function readsVersionAndUuidFromRepoBuildfile(testCase)
            testCase.verifyTrue(matches(buildfileValue("toolboxVersion"), ...
                digitsPattern + "." + digitsPattern + "." + digitsPattern));
            testCase.verifyNotEmpty(regexp(buildfileValue("toolboxUuid"), ...
                "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", "once"));
        end

        function skipsCommentLinesAfterHeader(testCase)
            tmp = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            f = fullfile(tmp.Folder, "buildfile.m");
            writelines(["function v = toolboxVersion"; "% note"; "% more"; ...
                "v = ""1.2.3"";"; "end"], f);
            testCase.verifyEqual(buildfileValue("toolboxVersion", f), "1.2.3");
        end

        function failsLoudlyWhenMissing(testCase)
            tmp = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            f = fullfile(tmp.Folder, "buildfile.m");
            writelines(["function v = other"; "v = ""1.2.3"";"; "end"], f);
            testCase.verifyError(@() buildfileValue("toolboxVersion", f), ...
                "phasecont:build:buildfileValue");
        end
    end
end
