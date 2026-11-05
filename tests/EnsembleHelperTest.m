classdef EnsembleHelperTest < matlab.unittest.TestCase
    % Tests the ensemble helper that needs no engine: internal.jsonRows.
    % These run without pckern.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function emptyArrayGivesZeroRows(testCase)
            s = phasecont.internal.jsonRows([], ["a" "b"]);
            testCase.verifyEqual(size(s), [0 1]);
            testCase.verifyEqual(string(fieldnames(s)).', ["a" "b"]);
        end

        function structArrayBecomesAColumn(testCase)
            v = jsondecode('[{"a":1,"b":"x"},{"a":2,"b":"y"}]');
            s = phasecont.internal.jsonRows(v, ["a" "b"]);
            testCase.verifyEqual(size(s), [2 1]);
            testCase.verifyEqual([s.a], [1 2]);
            testCase.verifyEqual(string({s.b}), ["x" "y"]);
        end

        function missingKeysAreFilledWithEmpty(testCase)
            v = jsondecode('[{"a":1,"b":2},{"a":3}]');
            s = phasecont.internal.jsonRows(v, ["a" "b"]);
            testCase.verifyEqual(size(s), [2 1]);
            testCase.verifyEqual(s(1).b, 2);
            testCase.verifyEmpty(s(2).b);
            testCase.verifyEqual(s(2).a, 3);
        end

        function extraKeysAreDropped(testCase)
            v = jsondecode('[{"a":1,"c":9}]');
            s = phasecont.internal.jsonRows(v, "a");
            testCase.verifyEqual(string(fieldnames(s)), "a");
        end

        function singleObjectIsOneRow(testCase)
            v = jsondecode('[{"a":1}]');
            s = phasecont.internal.jsonRows(v, "a");
            testCase.verifyEqual(size(s), [1 1]);
            testCase.verifyEqual(s.a, 1);
        end
    end
end
