classdef JsonRoundTripTest < matlab.unittest.TestCase
    % Tests that numbers written by num2json reach the engine unchanged.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            testCase.applyFixture(fixtures.BinFixture);
        end
    end

    methods (Test)
        function stringsRoundTripBitwise(testCase)
            rng(1);
            r = randn(1, 1000) .* 10 .^ randi([-300 300], 1, 1000);
            xs = [1e-10, 8/3, 470/19, 0.1, 1/3, pi, 1e300, 5e-324, r];
            for x = xs
                s = phasecont.internal.num2json(x);
                testCase.assertEqual(typecast(str2double(s), "uint64"), ...
                    typecast(x, "uint64"), "str2double " + s);
                y = jsondecode("[" + s + "]");
                testCase.assertEqual(typecast(y, "uint64"), ...
                    typecast(x, "uint64"), "jsondecode " + s);
            end
        end

        function nonFiniteIsBadOption(testCase)
            testCase.verifyError(@() phasecont.internal.num2json(NaN), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.num2json(Inf), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.num2json(-Inf), ...
                "phasecont:model:badOption");
        end

        function engineAcceptsTheseNumbers(testCase)
            f = fullfile(tempname, "fold.ode");
            mkdir(fileparts(f));
            cleanup = onCleanup(@() rmdir(fileparts(f), "s"));
            fid = fopen(f, "w");
            fprintf(fid, "par a=-1\ninit x=1\nx'=a+x^2\ndone\n");
            fclose(fid);
            m = phasecont.Model(f);
            eq = phasecont.equilibrium(m, -1, Tol = 1e-10);
            br = phasecont.continuation(m, eq, Free = "a", MaxSteps = 20, ...
                Bounds = [-5 470/19], TolX = 1e-10, H0 = 8/3*1e-3);
            testCase.verifyClass(br, "phasecont.Branch");
            testCase.verifyGreaterThan(numel(br.Alpha), 2);
            testCase.verifyNotEqual(br.Stop, "Error");
            testCase.verifyNotEqual(br.Stop, "Cancelled");
            clear cleanup
        end
    end
end
