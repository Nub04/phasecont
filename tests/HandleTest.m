classdef HandleTest < matlab.unittest.TestCase
    % Handle lifetime tests. Model-level cases; run cases are added later.

    properties (Constant)
        Text = "par a=2" + newline + "init x=1" + newline + ...
            "x'=-a*x" + newline + "done" + newline
    end

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
            testCase.File = fullfile(t.Folder, "h.ode");
            fid = fopen(testCase.File, "w");
            fwrite(fid, char(testCase.Text), "char");
            fclose(fid);
        end
    end

    methods (Test)
        function freedHandleIsBadHandle(testCase)
            m = phasecont.Model(testCase.File);
            pckern("free", m.Id);
            testCase.verifyError(@() m.rhs(1), "phasecont:engine:badHandle");
            testCase.verifyWarningFree(@() delete(m));
        end

        function deleteTwiceIsFine(testCase)
            m = phasecont.Model(testCase.File);
            delete(m);
            testCase.verifyWarningFree(@() delete(m));
        end

        function badHandleKinds(testCase)
            testCase.verifyError( ...
                @() pckern("model_rhs", 12345678901234, 1, 2), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_rhs", uint64(0), 1, 2), ...
                "phasecont:engine:badHandle");
            testCase.verifyError( ...
                @() pckern("model_rhs", uint64(12345678901234), 1, 2), ...
                "phasecont:engine:badHandle");
        end

        function modelsFromOneFileAreIndependent(testCase)
            a = phasecont.Model(testCase.File);
            b = phasecont.Model(testCase.File);
            testCase.verifyNotEqual(a.Id, b.Id);
            delete(a);
            testCase.verifyEqual(b.rhs(3), -6);
        end

        function clearFreesTheHandle(testCase)
            m = phasecont.Model(testCase.File);
            id = m.Id;
            clear m
            testCase.verifyError(@() pckern("model_info", id), ...
                "phasecont:engine:badHandle");
        end
    end
end
