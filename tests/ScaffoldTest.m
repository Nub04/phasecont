classdef ScaffoldTest < matlab.unittest.TestCase
    % Checks the build skeleton until P7a adds the real test suites.

    methods (Test)
        function planHasTheFourTasks(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            plan = matlab.buildtool.Plan.load(root);
            testCase.verifyEqual(sort(string({plan.Tasks.Name})), ...
                ["cargo" "mex" "package" "test"]);
        end
    end
end
