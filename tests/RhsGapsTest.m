classdef RhsGapsTest < matlab.unittest.TestCase
    % Tests phasecont.internal.rhsGaps (SPEC section 8z), which the cobweb
    % graph and the nullcline and field grids of PhasePlane use. The engine
    % refuses a value that is not finite, so a column whose value overflows
    % becomes NaN and every other column is the value of the batch rhs.

    properties (Constant)
        % x -> x^2 overflows for |x| above 2^512, y -> x*y does not at the
        % points below.
        Pair = "init x=1, y=1" + newline + "x(t+1)=x^2" + newline + ...
            "y(t+1)=x*y" + newline + "done" + newline
        Spring = "par k=2" + newline + "init u=1, v=0" + newline + ...
            "u'=v" + newline + "v'=-k*u" + newline + "done" + newline
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
        end
    end

    methods (Access = private)
        function m = model(testCase, text, ev)
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end
    end

    methods (Test)
        function anOverflowingColumnBecomesNaN(testCase, Evaluator)
            m = testCase.model(testCase.Pair, Evaluator);
            S = [1, 2^600, 3, -2^513, 0.5; 2, 2^-600, 4, 0, -1];
            F = phasecont.internal.rhsGaps(m, S, zeros(0, 1));
            testCase.verifySize(F, [2 5]);
            testCase.verifyTrue(all(isnan(F(:, [2 4])), "all"));
            ok = [1 3 5];
            testCase.verifyFalse(any(isnan(F(:, ok)), "all"));
            testCase.verifyEqual(F(:, ok), m.rhs(S(:, ok)));
            testCase.verifyEqual(F(:, ok), [S(1, ok) .^ 2; S(1, ok) .* S(2, ok)]);
        end

        function aFiniteBatchIsTheBatchRhs(testCase, Evaluator)
            m = testCase.model(testCase.Spring, Evaluator);
            S = [linspace(-1, 1, 7); linspace(2, -2, 7)];
            F = phasecont.internal.rhsGaps(m, S, 3);
            testCase.verifyEqual(F, m.rhs(S, 3));
            testCase.verifyEqual(F, [S(2, :); -3 * S(1, :)]);
        end

        function anErrorThatIsNotNumericIsRethrown(testCase, Evaluator)
            m = testCase.model(testCase.Pair, Evaluator);
            testCase.verifyError( ...
                @() phasecont.internal.rhsGaps(m, ones(3, 4), zeros(0, 1)), ...
                "phasecont:model:badState");
        end
    end
end
