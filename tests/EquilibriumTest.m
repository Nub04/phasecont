classdef EquilibriumTest < matlab.unittest.TestCase
    % Tests phasecont.equilibrium.

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        NoEq = "par a=1" + newline + "init x=0" + newline + ...
            "x'=x^2+a" + newline + "done" + newline
    end

    properties
        Dir
    end

    properties (TestParameter)
        Evaluator = {"jit", "fallback"}
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
            testCase.Dir = t.Folder;
        end
    end

    methods
        function m = model(testCase, text, ev)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = ev);
        end
    end

    methods (Test)
        function lorenzOrigin(testCase, Evaluator)
            m = testCase.model(testCase.Lorenz, Evaluator);
            pt = phasecont.equilibrium(m, [0.1 0.1 0.1]);
            testCase.verifyClass(pt, "phasecont.Point");
            testCase.verifyEqual(pt.U, zeros(3, 1), AbsTol = 1e-12);
            s = 10; r = 28; b = 8/3;
            J = [-s s 0; r -1 0; 0 0 -b];
            testCase.verifyEqual(sort(real(pt.Eigenvalues)), sort(eig(J)), ...
                RelTol = 1e-10);
            testCase.verifyEqual(pt.Label, "EP");
            testCase.verifyEqual(pt.P, [10; 28; 8/3]);
            testCase.verifyTrue(pt.Model == m);
        end

        function lorenzNontrivial(testCase, Evaluator)
            m = testCase.model(testCase.Lorenz, Evaluator);
            pt = phasecont.equilibrium(m, [8; 8; 27]);
            b = 8/3; r = 28;
            c = sqrt(b * (r - 1));
            testCase.verifyEqual(pt.U, [c; c; r - 1], AbsTol = 1e-10);
            testCase.verifySize(pt.Eigenvalues, [3 1]);
        end

        function paramsChangeOnlyTheNamedOne(testCase)
            m = testCase.model(testCase.Lorenz, "jit");
            pt = phasecont.equilibrium(m, [0.1 0.1 0.1], Params = struct(b = 2));
            testCase.verifyEqual(pt.P, [10; 28; 2]);
            pt = phasecont.equilibrium(m, [0.1 0.1 0.1], Params = [1 2 3]);
            testCase.verifyEqual(pt.P, [1; 2; 3]);
        end

        function noEquilibriumIsNumericError(testCase, Evaluator)
            m = testCase.model(testCase.NoEq, Evaluator);
            testCase.verifyError(@() phasecont.equilibrium(m, 0.3), ...
                "phasecont:engine:numeric");
        end

        function badLengthIsEngineBadArg(testCase)
            m = testCase.model(testCase.Lorenz, "jit");
            testCase.verifyError(@() phasecont.equilibrium(m, [1 2]), ...
                "phasecont:engine:badArg");
        end

        function badOptions(testCase)
            m = testCase.model(testCase.Lorenz, "jit");
            testCase.verifyError( ...
                @() phasecont.equilibrium(m, [1 1 1], MaxIter = 0), ...
                "MATLAB:validators:mustBePositive");
            testCase.verifyError( ...
                @() phasecont.equilibrium(m, [1 1 1], Tol = Inf), ...
                "phasecont:model:badOption");
        end
    end
end
