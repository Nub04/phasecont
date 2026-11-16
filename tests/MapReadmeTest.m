classdef MapReadmeTest < matlab.unittest.TestCase
    % Tests slice 8 of map support (SPEC section 8ab, test 1): the examples
    % of the "Maps" section of README.md. Each test writes the README's
    % .ode text into a temporary folder, runs the README calls as written
    % there and checks the values the README states in its comments. The
    % plot call of the first example is left out, since MapIterateTest
    % covers it.

    properties (Constant)
        % x -> r x (1 - x), with the fixed point x = 1 - 1/r, its multiplier
        % 2 - r and PD at r = 3, x = 2/3. Its 2-cycle has the multiplier
        % -r^2 + 2r + 4 of F^2, which is 0.16 at r = 3.2 and -1 at
        % r = 1 + sqrt(6).
        Logistic = "par r=2.5" + newline + "x(t+1)=r*x*(1-x)" + newline + ...
            "init x=0.1" + newline + "done" + newline
        % (x, y) -> (1 - a x^2 + y, b x), whose fixed points have PD at
        % a = 3(1 - b)^2/4.
        Henon = "par a=0.2, b=0.3" + newline + "x(t+1)=1-a*x^2+y" + newline + ...
            "y(t+1)=b*x" + newline + "init x=1.09, y=0.33" + newline + ...
            "done" + newline
        % The README lines that the tests below run or check, as the README
        % writes them, with the values its comments state.
        Lines = string({ ...
            'tr = phasecont.simulate(m, 0.1, [0 50]);   % rows for n = 0, 1, ..., 50', ...
            'eq = phasecont.equilibrium(m, 0.6);                % x = 0.6, multiplier -0.5', ...
            'br = phasecont.continuation(m, eq, Free = "r", Direction = 1, Bounds = [2 3.5]);', ...
            'br.points("PD")                                    % r = 3, x = 2/3', ...
            'm   = phasecont.Model("henon.ode");', ...
            'eq  = phasecont.equilibrium(m, [1.09 0.33]);', ...
            'br  = phasecont.continuation(m, eq, Free = "a", Direction = 1, Bounds = [-1 0.5]);', ...
            'pd  = br.points("PD");                              % a = 3(1 - b)^2/4 = 0.3675', ...
            'pdc = phasecont.continuation(m, pd, Type = "PD", Free = ["a" "b"], Bounds = [0.1 0.6]);', ...
            'brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, Bounds = [2 3.6], Switch = "PD1");', ...
            'brs(2).Iterate                                      % 2', ...
            'brs(2).points("PD")                                 % r = 1 + sqrt(6)', ...
            'p   = phasecont.equilibrium(m, 0.8, Params = struct(r = 3.2), Iterate = 2);   % x = 0.7995, multiplier 0.16'})
        Tol = 1e-8
    end

    properties
        Root
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.Root = root;
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
        end
    end

    methods (TestMethodSetup)
        function enterTemporaryFolder(testCase)
            % The README loads its models by file name from the current
            % folder.
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.applyFixture( ...
                matlab.unittest.fixtures.CurrentFolderFixture(t.Folder));
            writelines(testCase.Logistic, "logistic.ode", LineEnding = "\n");
            writelines(testCase.Henon, "henon.ode", LineEnding = "\n");
        end
    end

    methods (Test)
        function readmeHoldsTheseModelsAndLines(testCase)
            % The tests below copy the README. This test fails when the
            % README changes, so the copy and the README stay the same.
            readme = string(fileread(fullfile(testCase.Root, "README.md")));
            readme = replace(readme, sprintf("\r\n"), newline);
            testCase.verifySubstring(readme, testCase.Logistic + "```");
            testCase.verifySubstring(readme, testCase.Henon + "```");
            for line = testCase.Lines
                testCase.verifySubstring(readme, line);
            end
        end

        function iterationAndFixedPoints(testCase)
            m  = phasecont.Model("logistic.ode");
            testCase.addTeardown(@() delete(m));
            tr = phasecont.simulate(m, 0.1, [0 50]);
            testCase.verifyEqual(tr.T(:), (0:50)');

            eq = phasecont.equilibrium(m, 0.6);
            testCase.verifyEqual(eq.U, 0.6, AbsTol = testCase.Tol);
            testCase.verifyEqual(eq.Eigenvalues, -0.5, AbsTol = testCase.Tol);

            br = phasecont.continuation(m, eq, Free = "r", Direction = 1, Bounds = [2 3.5]);
            pd = br.points("PD");
            testCase.verifyNumElements(pd, 1);
            testCase.verifyEqual(pd.P(1), 3, AbsTol = testCase.Tol);
            testCase.verifyEqual(pd.U, 2/3, AbsTol = testCase.Tol);
        end

        function henonPdCurve(testCase)
            m   = phasecont.Model("henon.ode");
            testCase.addTeardown(@() delete(m));
            eq  = phasecont.equilibrium(m, [1.09 0.33]);
            br  = phasecont.continuation(m, eq, Free = "a", Direction = 1, Bounds = [-1 0.5]);
            pd  = br.points("PD");
            testCase.verifyNumElements(pd, 1);
            testCase.verifyEqual(pd.P(1), 3*(1 - 0.3)^2/4, AbsTol = testCase.Tol);
            testCase.verifyEqual(pd.P(1), 0.3675, AbsTol = testCase.Tol);

            pdc = phasecont.continuation(m, pd, Type = "PD", Free = ["a" "b"], Bounds = [0.1 0.6]);
            testCase.verifyEqual(pdc.Type, "PD");
            a = pdc.Alpha(:, 1);
            b = pdc.Alpha(:, 2);
            testCase.verifyEqual(a, 3*(1 - b).^2/4, AbsTol = testCase.Tol);
            testCase.verifyGreaterThanOrEqual(b, 0.1 - testCase.Tol);
            testCase.verifyLessThanOrEqual(b, 0.6 + testCase.Tol);
            testCase.verifyEqual(b(1), 0.3, AbsTol = testCase.Tol);
            testCase.verifyEqual(max(b), 0.6, AbsTol = testCase.Tol);
        end

        function periodTwoPoints(testCase)
            m   = phasecont.Model("logistic.ode");
            testCase.addTeardown(@() delete(m));
            eq  = phasecont.equilibrium(m, 0.6);
            brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, Bounds = [2 3.6], Switch = "PD1");
            testCase.verifySize(brs, [1 2]);
            testCase.verifyEqual(brs(2).Iterate, 2);
            testCase.verifyEqual([brs.Iterate], [1 2]);
            pd2 = brs(2).points("PD");
            testCase.assertNotEmpty(pd2);
            testCase.verifyEqual(pd2(1).P(1), 1 + sqrt(6), AbsTol = testCase.Tol);

            p   = phasecont.equilibrium(m, 0.8, Params = struct(r = 3.2), Iterate = 2);
            testCase.verifyEqual(p.U, 0.7995, AbsTol = 1e-4);
            testCase.verifyEqual(p.Eigenvalues, 0.16, AbsTol = testCase.Tol);
            testCase.verifyTrue(p.Stable);
        end
    end
end
