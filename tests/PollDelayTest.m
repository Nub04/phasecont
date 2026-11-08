classdef PollDelayTest < matlab.unittest.TestCase
    % Tests phasecont.internal.pollDelay, the wait between engine polls.
    % These run without pckern.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function theFirstWaitIsOneMillisecond(testCase)
            testCase.verifyEqual(phasecont.internal.pollDelay(1), 0.001);
        end

        function theWaitDoublesUntilItReachesTheCap(testCase)
            d = arrayfun(@phasecont.internal.pollDelay, 1:8);
            testCase.verifyEqual(d, [0.001 0.002 0.004 0.008 0.016 0.019 0.05 0.05]);
        end

        function theWaitNeverShrinksOrPassesTheCap(testCase)
            d = arrayfun(@phasecont.internal.pollDelay, 1:200);
            testCase.verifyGreaterThan(min(d), 0);
            testCase.verifyLessThanOrEqual(max(d), 0.05);
            testCase.verifyGreaterThanOrEqual(diff(d), zeros(1, 199));
        end

        function aHugePollCountStaysAtTheCap(testCase)
            testCase.verifyEqual(phasecont.internal.pollDelay(1e6), 0.05);
        end

        function theRampEndsOnTheOldFiftyMillisecondGrid(testCase)
            d = arrayfun(@phasecont.internal.pollDelay, 1:6);
            testCase.verifyEqual(sum(d), 0.05, AbsTol = 1e-12);
        end

        function everyOldPollTimeIsStillAPollTime(testCase)
            % The old loop polled at 0, 50, 100, ... ms. A run that ended
            % at one of those polls must still be seen no later than that.
            d = arrayfun(@phasecont.internal.pollDelay, 1:40);
            times = cumsum(d);
            for j = 1:30
                testCase.verifyTrue(any(abs(times - 0.05 * j) < 1e-12), ...
                    sprintf("no poll at %d ms", 50 * j));
            end
        end

        function aBadCountIsRefused(testCase)
            testCase.verifyError(@() phasecont.internal.pollDelay(0), ...
                "MATLAB:validators:mustBePositive");
            testCase.verifyError(@() phasecont.internal.pollDelay(1.5), ...
                "MATLAB:validators:mustBeInteger");
        end
    end
end
