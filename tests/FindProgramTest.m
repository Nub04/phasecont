classdef FindProgramTest < matlab.unittest.TestCase
    % Tests phasecont.internal.findProgram with programs it writes into a
    % temporary folder.

    properties
        Folder string
        Name = "pcfindprogramprobe"
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (TestMethodSetup)
        function writeProgram(testCase)
            tmp = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Folder = string(tmp.Folder);
            if ispc
                file = fullfile(testCase.Folder, testCase.Name + ".cmd");
                body = "@echo hi" + newline;
            else
                file = fullfile(testCase.Folder, testCase.Name);
                body = "#!/bin/sh" + newline + "echo hi" + newline;
            end
            fid = fopen(file, "w");
            fprintf(fid, "%s", body);
            fclose(fid);
            if ~ispc
                fileattrib(char(file), "+x");
            end
            old = getenv("PATH");
            testCase.addTeardown(@() setenv("PATH", old));
        end
    end

    methods (Test)
        function aProgramOnThePathComesBackAsAFullPath(testCase)
            setenv("PATH", testCase.Folder + pathsep + getenv("PATH"));
            p = phasecont.internal.findProgram(testCase.Name);
            testCase.verifyTrue(startsWith(p, testCase.Folder));
            testCase.verifyTrue(isfile(p));
        end

        function theCurrentFolderIsNotSearched(testCase)
            testCase.applyFixture( ...
                matlab.unittest.fixtures.CurrentFolderFixture(testCase.Folder));
            testCase.verifyEqual(phasecont.internal.findProgram(testCase.Name), "");
        end

        function relativePathEntriesAreSkipped(testCase)
            testCase.applyFixture( ...
                matlab.unittest.fixtures.CurrentFolderFixture(testCase.Folder));
            setenv("PATH", "." + pathsep + pathsep + getenv("PATH"));
            testCase.verifyEqual(phasecont.internal.findProgram(testCase.Name), "");
        end

        function anExplicitPathIsKept(testCase)
            stem = fullfile(testCase.Folder, testCase.Name);
            p = phasecont.internal.findProgram(stem);
            testCase.verifyTrue(startsWith(p, stem));
            testCase.verifyEqual(phasecont.internal.findProgram(stem + "-missing"), "");
        end

        function aMissingNameGivesEmpty(testCase)
            testCase.verifyEqual( ...
                phasecont.internal.findProgram("pc-no-such-program-7d2e"), "");
            testCase.verifyEqual(phasecont.internal.findProgram(""), "");
        end
    end
end
