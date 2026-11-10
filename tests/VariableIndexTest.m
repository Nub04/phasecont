classdef VariableIndexTest < matlab.unittest.TestCase
    % phasecont.internal.variableIndex turns a variable name or index into a
    % column of U. Branch plots and continuation live plots use it.

    properties (Constant)
        Names = ["v" "w" "z"]
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function emptyMeansTheFirstVariable(testCase)
            testCase.verifyEqual(phasecont.internal.variableIndex([], 3, testCase.Names), 1);
        end

        function takesAnIndexInRange(testCase)
            testCase.verifyEqual( ...
                phasecont.internal.variableIndex(int8(2), 3, testCase.Names), 2);
            testCase.verifyClass( ...
                phasecont.internal.variableIndex(int8(2), 3, testCase.Names), "double");
            testCase.verifyEqual(phasecont.internal.variableIndex(3, 3, testCase.Names), 3);
        end

        function takesANameAsStringOrChar(testCase)
            testCase.verifyEqual(phasecont.internal.variableIndex("w", 3, testCase.Names), 2);
            testCase.verifyEqual(phasecont.internal.variableIndex('z', 3, testCase.Names), 3);
        end

        function worksWithoutNames(testCase)
            % A branch with no model knows only how many states it has.
            testCase.verifyEqual(phasecont.internal.variableIndex(2, 3, strings(0, 1)), 2);
            testCase.verifyError( ...
                @() phasecont.internal.variableIndex("w", 3, strings(0, 1)), ...
                "phasecont:model:unknownVariable");
        end

        function refusesWhatItCannotPlace(testCase)
            bad = {0, 4, 1.5, NaN, [1 2], "", "q", ["v" "w"], {1}, true};
            for k = 1:numel(bad)
                testCase.verifyError( ...
                    @() phasecont.internal.variableIndex(bad{k}, 3, testCase.Names), ...
                    "phasecont:model:unknownVariable", "case " + k);
            end
        end

        function errorListsTheNames(testCase)
            try
                phasecont.internal.variableIndex("q", 3, testCase.Names);
                testCase.verifyFail("no error");
            catch err
                testCase.verifySubstring(err.message, "v, w, z");
            end
        end
    end
end
