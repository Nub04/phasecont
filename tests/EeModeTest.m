classdef EeModeTest < matlab.unittest.TestCase
    % Tests the EE mode, phasecont.ee.solve, phasecont.ee.inverse and
    % phasecont.ee.jstr (SPEC section 8ah). Every matrix is written here.

    properties (Constant)
        SolveMethods = ["cramer" "adjugate" "inverse" "gaussjordan"]
        InverseMethods = ["adjugate" "gaussjordan"]
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function integerTwoByTwo(testCase)
            A = [2 1; 1 3];
            b = [4; 7];
            for m = testCase.SolveMethods
                x = phasecont.ee.solve(A, b, Method = m);
                testCase.verifyEqual(x, [1; 2], "AbsTol", 1e-12, m);
            end
            for m = testCase.InverseMethods
                X = phasecont.ee.inverse(A, Method = m);
                testCase.verifyEqual(X, [3 -1; -1 2] / 5, "AbsTol", 1e-12, m);
            end
            [~, info] = phasecont.ee.solve(A, b, Method = "cramer");
            testCase.verifyEqual(info.Det, 5, "AbsTol", 1e-12);
            testCase.verifyEqual(info.DetB, [5; 10], "AbsTol", 1e-12);
        end

        function integerThreeByThree(testCase)
            A = [1 0 2; 2 1 0; 0 3 1];
            b = [5; 1; -1];
            adj = [1 6 -2; -2 1 4; 6 -3 1];
            for m = testCase.SolveMethods
                x = phasecont.ee.solve(A, b, Method = m);
                testCase.verifyEqual(x, [1; -1; 2], "AbsTol", 1e-12, m);
            end
            for m = testCase.InverseMethods
                X = phasecont.ee.inverse(A, Method = m);
                testCase.verifyEqual(X, adj / 13, "AbsTol", 1e-12, m);
            end
            [~, info] = phasecont.ee.solve(A, b, Method = "cramer");
            testCase.verifyEqual(info.Det, 13, "AbsTol", 1e-12);
            testCase.verifyEqual(info.DetB, [13; -13; 26], "AbsTol", 1e-12);
            [~, info] = phasecont.ee.inverse(A, Method = "adjugate");
            testCase.verifyEqual(info.Adjugate, adj, "AbsTol", 1e-12);
            testCase.verifyEqual(info.Det, 13, "AbsTol", 1e-12);
        end

        function infoFields(testCase)
            A = [2 1; 1 3];
            b = [4; 7];
            [~, c] = phasecont.ee.solve(A, b, Method = "cramer");
            [~, g] = phasecont.ee.solve(A, b, Method = "gaussjordan");
            [~, v] = phasecont.ee.solve(A, b, Method = "inverse");
            [~, a] = phasecont.ee.inverse(A, Method = "adjugate");
            testCase.verifyEqual(c.Method, "cramer");
            testCase.verifyEqual(c.RCond, rcond(A));
            testCase.verifyEqual(c.Adjugate, []);
            testCase.verifyEqual(size(c.Steps), [0 1]);
            testCase.verifyTrue(isnan(g.Det));
            testCase.verifyEqual(g.DetB, []);
            testCase.verifyNotEmpty(g.Steps);
            testCase.verifyEqual(size(g.Steps(end).Matrix), [2 3]);
            testCase.verifyTrue(isnan(v.Det));
            testCase.verifyEqual(size(v.Steps(end).Matrix), [2 4]);
            testCase.verifyEqual(a.DetB, []);
            testCase.verifyEqual(size(a.Steps), [0 1]);
        end

        function agreeWithBackslash(testCase)
            s = RandStream("mt19937ar", "Seed", 7);
            cases = {randn(s, 6), randn(s, 8) + 1j * randn(s, 8)};
            for k = 1:numel(cases)
                A = cases{k};
                n = size(A, 1);
                b = randn(s, n, 1) + 1j * randn(s, n, 1) * (k == 2);
                ref = A \ b;
                for m = testCase.SolveMethods
                    x = phasecont.ee.solve(A, b, Method = m);
                    testCase.verifyLessThanOrEqual(norm(x - ref), ...
                        1e-10 * norm(ref), m);
                    testCase.verifyLessThanOrEqual(norm(A * x - b), ...
                        1e-10 * norm(b), m);
                end
                for m = testCase.InverseMethods
                    X = phasecont.ee.inverse(A, Method = m);
                    testCase.verifyLessThanOrEqual(norm(A * X - eye(n)), ...
                        1e-10, m);
                end
            end
        end

        function complexNonSymmetric(testCase)
            % Nonzero imaginary parts above and below the diagonal, so a
            % conjugating transpose in the adjugate changes the answer.
            A = [2+1j, 1-2j, 0; 3j, 4, 1+1j; 1, -1j, 5-2j];
            b = [1; 2j; 3-1j];
            ref = A \ b;
            for m = testCase.SolveMethods
                x = phasecont.ee.solve(A, b, Method = m);
                testCase.verifyEqual(x, ref, "AbsTol", 1e-12, m);
            end
            for m = testCase.InverseMethods
                X = phasecont.ee.inverse(A, Method = m);
                testCase.verifyEqual(X, inv(A), "AbsTol", 1e-12, m);
            end
        end

        function stepsReplay(testCase)
            A = [0 1 2; 1 0 1; 2 1 0];
            b = [3; 2; 3];
            [x, info] = phasecont.ee.solve(A, b, Method = "gaussjordan");
            testCase.verifyEqual(x, [1; 1; 1], "AbsTol", 1e-12);
            testCase.verifyEqual(info.Steps(1).Op, "swap");
            testCase.verifyEqual(info.Steps(1).Rows, [1 3]);
            M = [A b];
            for k = 1:numel(info.Steps)
                M = EeModeTest.replay(M, info.Steps(k));
                testCase.verifyTrue(isequal(M, info.Steps(k).Matrix), ...
                    "step " + k);
            end
            testCase.verifyTrue(isequal(M(:, 1:3), eye(3)));
            [X, info] = phasecont.ee.inverse(A, Method = "gaussjordan");
            M = [A eye(3)];
            for k = 1:numel(info.Steps)
                M = EeModeTest.replay(M, info.Steps(k));
            end
            testCase.verifyTrue(isequal(M(:, 4:6), X));
            testCase.verifyTrue(isequal(M(:, 1:3), eye(3)));
        end

        function pivotTieTakesSmallestRow(testCase)
            A = [0.5 1 0; 2 0 1; -2 1 1];
            [~, info] = phasecont.ee.inverse(A, Method = "gaussjordan");
            testCase.verifyEqual(info.Steps(1).Op, "swap");
            testCase.verifyEqual(info.Steps(1).Rows, [1 2]);
        end

        function noStepWhenNothingChanges(testCase)
            % a11 is 1 and a21 is 0, so column 1 needs no step at all.
            [~, info] = phasecont.ee.inverse([1 2; 0 4], Method = "gaussjordan");
            ops = [info.Steps.Op];
            rows = vertcat(info.Steps.Rows);
            testCase.verifyEqual(ops, ["scale" "add"]);
            testCase.verifyEqual(rows, [2 2; 1 2]);
            testCase.verifyEqual(info.Steps(2).Factor, -2);
        end

        function jDisplay(testCase)
            z = [3-4j, -2-3j, 6+8j, 5, -4j, 0, complex(0, -0), ...
                complex(2, -0), complex(-0, 3), complex(-0, 0)];
            want = ["3 - j4" "-2 - j3" "6 + j8" "5" "-j4" "0" "0" ...
                "2" "j3" "0"];
            for k = 1:numel(z)
                testCase.verifyEqual(phasecont.ee.jstr(z(k)), want(k), ...
                    "entry " + k);
            end
            testCase.verifyEqual(phasecont.ee.jstr([1+2j, 3; -1j, 0]), ...
                ["1 + j2" "3"; "-j1" "0"]);
            testCase.verifyEqual(phasecont.ee.jstr(pi + 1j * exp(1), ...
                Digits = 3), "3.14 + j2.72");
            testCase.verifyEqual(phasecont.ee.jstr(-2.5), "-2.5");
            testCase.verifyEqual(phasecont.ee.jstr(complex(1, NaN)), ...
                "1 + jNaN");
            testCase.verifyEqual(phasecont.ee.jstr(complex(1, -Inf)), ...
                "1 - jInf");
        end

        function jDisplayRefusals(testCase)
            testCase.verifyError(@() phasecont.ee.jstr(1, Digits = 0), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.jstr(1, Digits = 18), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.jstr(1, Digits = 2.5), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.jstr("1"), ...
                "phasecont:ee:badSize");
        end

        function singularRefused(testCase)
            A = [1 2 3; 4 5 6; 5 7 9];
            b = [1; 2; 3];
            for m = testCase.SolveMethods
                testCase.verifyError(@() phasecont.ee.solve(A, b, Method = m), ...
                    "phasecont:ee:singular", m);
            end
            for m = testCase.InverseMethods
                testCase.verifyError(@() phasecont.ee.inverse(A, Method = m), ...
                    "phasecont:ee:singular", m);
            end
            testCase.verifyError(@() phasecont.ee.inverse(0, Method = "adjugate"), ...
                "phasecont:ee:singular");
        end

        function tolCut(testCase)
            % rcond of [1 1; 1 1+d] is about d/4.
            below = [1 1; 1 1 + 1e-11];
            above = [1 1; 1 1 + 1e-7];
            b = [2; 2];
            for m = testCase.SolveMethods
                testCase.verifyError(@() phasecont.ee.solve(below, b, ...
                    Method = m), "phasecont:ee:singular", m);
                x = phasecont.ee.solve(below, b, Method = m, Tol = 1e-13);
                testCase.verifyEqual(x, [2; 0], "AbsTol", 1e-3, m);
                x = phasecont.ee.solve(above, b, Method = m);
                testCase.verifyEqual(x, [2; 0], "AbsTol", 1e-6, m);
            end
        end

        function sizeAndValueRefusals(testCase)
            ok = [2 1; 1 3];
            testCase.verifyError(@() phasecont.ee.inverse(eye(9), ...
                Method = "adjugate"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.inverse(zeros(0), ...
                Method = "adjugate"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.inverse(ones(2, 3), ...
                Method = "gaussjordan"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.inverse(single(ok), ...
                Method = "gaussjordan"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.solve(ok, [1; 2; 3], ...
                Method = "cramer"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.solve(ok, [1 2], ...
                Method = "cramer"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.solve(ok, int32([1; 2]), ...
                Method = "cramer"), "phasecont:ee:badSize");
            testCase.verifyError(@() phasecont.ee.solve([1 NaN; 0 1], ...
                [1; 2], Method = "gaussjordan"), "phasecont:ee:nonfinite");
            testCase.verifyError(@() phasecont.ee.solve(ok, [1; Inf], ...
                Method = "cramer"), "phasecont:ee:nonfinite");
            testCase.verifyError(@() phasecont.ee.inverse([Inf 0; 0 1], ...
                Method = "adjugate"), "phasecont:ee:nonfinite");
        end

        function optionRefusals(testCase)
            ok = [2 1; 1 3];
            b = [4; 7];
            testCase.verifyError(@() phasecont.ee.solve(ok, b), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.inverse(ok), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.solve(ok, b, Method = "lu"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.inverse(ok, Method = "cramer"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.inverse(ok, Method = "inverse"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.solve(ok, b, ...
                Method = "cramer", Tol = 1), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.solve(ok, b, ...
                Method = "cramer", Tol = -1e-3), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.ee.solve(ok, b, ...
                Method = "cramer", Tol = NaN), "phasecont:model:badOption");
        end
    end

    methods (Static, Access = private)
        function M = replay(M, step)
            % The row operations of SPEC 8ah, done the same way.
            r = step.Rows;
            switch step.Op
                case "swap"
                    M([r(1) r(2)], :) = M([r(2) r(1)], :);
                case "scale"
                    M(r(1), :) = step.Factor * M(r(1), :);
                    M(r(1), r(1)) = 1;
                case "add"
                    M(r(1), :) = M(r(1), :) + step.Factor * M(r(2), :);
                    M(r(1), r(2)) = 0;
            end
        end
    end
end
