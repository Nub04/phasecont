classdef EquilibriaTest < matlab.unittest.TestCase
    % Tests phasecont.equilibria (SPEC 8ai-2).

    properties (Constant)
        Bistable = "init x=0.5, y=0.5" + newline + "x'=x-x^3" + newline + ...
            "y'=-y" + newline + "done" + newline
        Logistic = "par r=3.2" + newline + "init x=0.3" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
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

        function verifyGroups(testCase, pts, raw, X0, varargin)
            % Each group's point is phasecont.equilibrium from its first
            % converging start, and every start in a group is close to it.
            testCase.verifyEqual(raw.Converged, raw.Group > 0);
            for g = 1:numel(pts)
                rep = find(raw.Group == g, 1);
                testCase.verifyEqual(find(raw.Converged & raw.Group >= g, 1), rep);
                one = phasecont.equilibrium(pts(g).Model, X0(:, rep), varargin{:});
                testCase.verifyTrue(isequaln(pts(g), one));
                testCase.verifyEqual(pts(g).Label, "EP");
                testCase.verifyEqual(pts(g).Record.Kind, "equilibrium");
                testCase.verifyEqual(pts(g).Record.U0, X0(:, rep));
                testCase.verifyEqual(raw.U(:, rep), one.U);
                testCase.verifyTrue(isequal(raw.Eigenvalues(:, rep), one.Eigenvalues));
            end
            testCase.verifyTrue(all(isnan(raw.U(:, ~raw.Converged)), "all"));
        end
    end

    methods (Test)
        function gridOfStartsOnABistableModel(testCase, Evaluator)
            m = testCase.model(testCase.Bistable, Evaluator);
            [X, Y] = meshgrid(linspace(-1.5, 1.5, 5));
            X0 = [X(:)'; Y(:)'];
            [pts, raw] = phasecont.equilibria(m, X0);
            testCase.verifyClass(pts, "phasecont.Point");
            testCase.verifySize(pts, [1 3]);
            testCase.verifyEqual(arrayfun(@(q) q.U(1), pts), [-1 0 1], AbsTol = 1e-12);
            testCase.verifyEqual(arrayfun(@(q) q.U(2), pts), [0 0 0], AbsTol = 1e-12);
            testCase.verifySize(raw.U, [2 25]);
            testCase.verifySize(raw.Eigenvalues, [2 25]);
            testCase.verifySize(raw.Residual, [1 25]);
            testCase.verifyClass(raw.Converged, "logical");
            testCase.verifyTrue(all(raw.Converged));
            testCase.verifyLessThanOrEqual(raw.Residual, 1e-10);
            testCase.verifyGroups(pts, raw, X0);
            testCase.verifyEqual([pts.Stable], [true false true]);
        end

        function cyclePointsOfTheLogisticMap(testCase, Evaluator)
            m = testCase.model(testCase.Logistic, Evaluator);
            X0 = linspace(-0.1, 1.1, 25);
            [pts, raw] = phasecont.equilibria(m, X0, Iterate = 2);
            r = 3.2;
            s = sqrt((r - 3) * (r + 1));
            want = sort([0, 1 - 1/r, (r + 1 - s)/(2*r), (r + 1 + s)/(2*r)]);
            testCase.verifySize(pts, [1 4]);
            testCase.verifyEqual(sort(arrayfun(@(q) q.U, pts)), want, AbsTol = 1e-12);
            testCase.verifyEqual(unique([pts.Iterate]), 2);
            testCase.verifyGroups(pts, raw, X0, Iterate = 2);
        end

        function aBoxWithNoEquilibriumGivesNoPoints(testCase)
            m = testCase.model(testCase.NoEq, "jit");
            [pts, raw] = phasecont.equilibria(m, [-1 0 1]);
            testCase.verifyClass(pts, "phasecont.Point");
            testCase.verifySize(pts, [0 0]);
            testCase.verifyEqual(raw.Converged, false(1, 3));
            testCase.verifyEqual(raw.Group, zeros(1, 3));
            testCase.verifyTrue(all(isnan(raw.U)));
        end

        function zeroStartsGiveNoPoints(testCase)
            m = testCase.model(testCase.Bistable, "jit");
            [pts, raw] = phasecont.equilibria(m, zeros(2, 0));
            testCase.verifySize(pts, [0 0]);
            testCase.verifySize(raw.U, [2 0]);
            testCase.verifySize(raw.Group, [1 0]);
        end

        function theGatewayReturnsEmptyOutputsForZeroStarts(testCase)
            m = testCase.model(testCase.Bistable, "jit");
            [U, E, res, status] = pckern("model_equilibrium_many", m.Id, ...
                zeros(2, 0), zeros(0, 1), '');
            testCase.verifySize(U, [2 0]);
            testCase.verifySize(E, [2 0]);
            testCase.verifySize(res, [1 0]);
            testCase.verifySize(status, [1 0]);
            testCase.verifyClass(status, "int32");
        end

        function badOptionsAreRefused(testCase)
            m = testCase.model(testCase.Bistable, "jit");
            X0 = [0.5; 0.5];
            for mt = {0, -1, Inf, NaN, "a", [1 2], 1i}
                testCase.verifyError(@() phasecont.equilibria(m, X0, MergeTol = mt{1}), ...
                    "phasecont:model:badOption");
            end
            testCase.verifyError(@() phasecont.equilibria(m, [0.5; 0.5; 0.5]), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.equilibria(m, [0.5 + 1i; 0.5]), ...
                "MATLAB:validators:mustBeReal");
            testCase.verifyError(@() phasecont.equilibria(m, X0, Iterate = 2), ...
                "phasecont:model:badOption");
        end
    end
end
