classdef WslArgTest < matlab.unittest.TestCase
    % scripts/wslArg checks a path before buildfile.m puts it inside a
    % double-quoted wsl.exe argument that bash then reads.

    methods (TestClassSetup)
        function addScriptsToPath(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(root, "scripts")));
        end
    end

    methods (Test)
        function returnsAPlainPathUnchanged(testCase)
            testCase.verifyEqual(wslArg("/home/u/my repo/x-1_2.y"), ...
                "/home/u/my repo/x-1_2.y");
            testCase.verifyEqual(wslArg("C:/Users/u/phasecont-stage/bin/win64"), ...
                "C:/Users/u/phasecont-stage/bin/win64");
        end

        function acceptsACharVector(testCase)
            testCase.verifyEqual(wslArg('/home/u/repo'), "/home/u/repo");
        end

        function refusesCharactersBashOrCmdWouldExpand(testCase)
            bad = ["/home/u/$HOME/repo", "/home/u/`id`/repo", ...
                "/home/u/a""b/repo", "/home/u/a\b/repo", "/home/u/100%/repo", ...
                "/home/u/a" + newline + "b"];
            for k = 1:numel(bad)
                testCase.verifyError(@() wslArg(bad(k)), ...
                    "phasecont:build:unsafePath", bad(k));
            end
        end

        function namesTheOffendingPathInTheMessage(testCase)
            try
                wslArg("/home/u/$HOME/repo");
                testCase.verifyFail("wslArg accepted a path with a dollar sign");
            catch err
                testCase.verifySubstring(err.message, "/home/u/$HOME/repo");
            end
        end
    end
end
