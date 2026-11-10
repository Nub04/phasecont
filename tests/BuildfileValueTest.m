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

        function readsFileWithWindowsLineEndings(testCase)
            f = testCase.writeBuildfile( ...
                join(["function v = toolboxVersion"; "% note"; "v = ""4.5.6"";"; "end"], ...
                char(13) + newline) + char(13) + newline);
            testCase.verifyEqual(buildfileValue("toolboxVersion", f), "4.5.6");
        end

        function refusesLiteralsItCannotReadExactly(testCase)
            % The reader takes one double-quoted literal and nothing else;
            % any other form must fail instead of returning a part of it.
            cases = dictionary( ...
                "single quoted", "v = '1.2.3';", ...
                "concatenated", "v = ""1.2"" + "".3"";", ...
                "computed", "v = string(1) + "".2.3"";", ...
                "no semicolon", "v = ""1.2.3""", ...
                "empty", "v = """";");
            for k = keys(cases)'
                f = testCase.writeBuildfile( ...
                    ["function v = toolboxVersion"; cases(k); "end"]);
                testCase.verifyError(@() buildfileValue("toolboxVersion", f), ...
                    "phasecont:build:buildfileValue", k);
            end
        end

        function doesNotMatchAFunctionWithALongerName(testCase)
            f = testCase.writeBuildfile( ...
                ["function v = toolboxVersion2"; "v = ""1.2.3"";"; "end"]);
            testCase.verifyError(@() buildfileValue("toolboxVersion", f), ...
                "phasecont:build:buildfileValue");
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

    methods
        function f = writeBuildfile(testCase, text)
            % Write text (lines or one string) to a buildfile.m in a new
            % temporary folder and return its path.
            tmp = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            f = fullfile(tmp.Folder, "buildfile.m");
            fid = fopen(f, "w");
            fwrite(fid, char(join(text(:), newline)), "char");
            fclose(fid);
        end
    end
end
