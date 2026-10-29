classdef PointTest < matlab.unittest.TestCase
    % Tests the phasecont.Point value class.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            testCase.applyFixture(fixtures.BinFixture);
        end
    end

    methods (Test)
        function defaults(testCase)
            p = phasecont.Point();
            testCase.verifyEqual(p.U, zeros(0,1));
            testCase.verifyEqual(p.P, zeros(0,1));
            testCase.verifyEqual(p.Eigenvalues, zeros(0,1));
            testCase.verifyEqual(p.Label, "EP");
            testCase.verifyTrue(isnan(p.Index));
            testCase.verifyEqual(p.Data, struct());
            testCase.verifyClass(p.Model, "phasecont.Model");
            testCase.verifyEmpty(p.Model);
        end

        function nameValueConstruction(testCase)
            p = phasecont.Point(U = [1 2 3], P = [4; 5], ...
                Eigenvalues = [1+2i; 1-2i], Label = "H", Index = 7, ...
                Data = struct(omega = 2));
            testCase.verifyEqual(p.U, [1; 2; 3]);
            testCase.verifyEqual(p.P, [4; 5]);
            testCase.verifyEqual(p.Eigenvalues, [1+2i; 1-2i]);
            testCase.verifyEqual(p.Label, "H");
            testCase.verifyEqual(p.Index, 7);
            testCase.verifyEqual(p.Data.omega, 2);
        end

        function propertiesAreImmutable(testCase)
            p = phasecont.Point(U = 1);
            testCase.verifyError(@() setU(p), "MATLAB:class:SetProhibited");
            function setU(q)
                q.U = 2;
            end
        end

        function modelPropertyAcceptsModelOnly(testCase)
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
            file = fullfile(tempname, "m.ode");
            mkdir(fileparts(file));
            cleanup = onCleanup(@() rmdir(fileparts(file), "s"));
            fid = fopen(file, "w");
            fprintf(fid, "init x=1\nx'=-x\ndone\n");
            fclose(fid);
            m = phasecont.Model(file);
            p = phasecont.Point(Model = m);
            testCase.verifyTrue(p.Model == m);
            testCase.verifyError(@() phasecont.Point(Model = 3), ...
                "MATLAB:validation:UnableToConvert");
            clear cleanup
        end
    end
end
