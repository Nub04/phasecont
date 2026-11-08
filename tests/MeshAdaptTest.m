classdef MeshAdaptTest < matlab.unittest.TestCase
    % Tests the MeshAdapt option of phasecont.continuation (SPEC section 11,
    % P33): 0 keeps the start mesh, 1 moves it, and the option is refused
    % for the types that have no mesh and for values that are not counts.
    % Every model is written here.

    properties (Constant)
        Brusselator = "par a=1,b=1.5" + newline + ...
            "x'=a-(b+1)*x+x^2*y" + newline + "y'=b*x-x^2*y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Sandstede = "par a=-0.25,b=1,c=-2,mu=0,mut=0,alpha=0,beta=0,gamma=0" + newline + ...
            "x'=a*x+b*y-a*x^2+(mut-alpha*z)*x*(2-3*x)" + newline + ...
            "y'=b*x+a*y-1.5*b*x^2-1.5*a*x*y-2*y*(mut-alpha*z)" + newline + ...
            "z'=c*z+mu*x+gamma*x*z+alpha*beta*(x^2*(1-x)-y^2)" + newline + ...
            "init x=0,y=0,z=0" + newline + "done" + newline
    end

    properties
        Dir
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
        function m = model(testCase, text)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = "jit");
        end

        function [m, eq, h] = brusselatorHopf(testCase)
            m = testCase.model(testCase.Brusselator);
            eq = phasecont.equilibrium(m, [1 1.2], Params = struct(a = 1, b = 1.2));
            ep = phasecont.continuation(m, eq, Free = "b", Bounds = [0.5 4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            h = h(1);
        end

        function start = sandstedeOrbit(~, a, period)
            f = @(~, w) (1 - w.^2) .* (1 - a*w) / 2;
            o = odeset(RelTol = 1e-12, AbsTol = 1e-14);
            ts = linspace(0, period/2, 2001).';
            [~, wp] = ode45(f, ts, 0, o);
            [~, wm] = ode45(f, -ts, 0, o);
            t = [-flipud(ts(2:end)); ts];
            w = [flipud(wm(2:end)); wp];
            x = [1 - w.^2, -w.*(1 - w.^2), zeros(size(w))];
            start = struct(t = t, x = x, p = struct(a = a, b = 1, c = -2));
        end
    end

    methods (Test)
        function zeroKeepsTheStartMeshOfACycle(testCase)
            [m, ~, h] = testCase.brusselatorHopf();
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 30, ...
                MeshAdapt = 0);
            k = size(lc.Mesh, 1);
            testCase.assertGreaterThan(k, 5);
            testCase.verifyEqual(lc.Mesh, repmat(linspace(0, 1, 21), k, 1), ...
                AbsTol = 1e-14);
        end

        function oneMovesTheMeshOfACycle(testCase)
            [m, ~, h] = testCase.brusselatorHopf();
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, Bounds = [1.5 3], MaxSteps = 30, ...
                MeshAdapt = 1);
            k = size(lc.Mesh, 1);
            testCase.assertGreaterThan(k, 5);
            testCase.verifyGreaterThan( ...
                max(abs(lc.Mesh - repmat(linspace(0, 1, 21), k, 1)), [], "all"), 1e-6);
            testCase.verifyEqual(lc.Mesh(:, [1 end]), repmat([0 1], k, 1));
            testCase.verifyTrue(all(diff(lc.Mesh, 1, 2) > 0, "all"));
        end

        function zeroKeepsTheStartMeshOfAHomCurve(testCase)
            m = testCase.model(testCase.Sandstede);
            hom = phasecont.continuation(m, testCase.sandstedeOrbit(-0.25, 20), ...
                Type = "HOM", Free = ["mut" "a"], NTST = 160, NCOL = 4, ...
                Bounds = [-0.26 0.26], Hmax = 0.05, MaxNewton = 15, ...
                MaxSteps = 12, MeshAdapt = 0);
            k = size(hom.Mesh, 1);
            testCase.assertGreaterThan(k, 5);
            testCase.verifyEqual(hom.Mesh, repmat(hom.Mesh(1, :), k, 1), ...
                AbsTol = 1e-14);
        end

        function badValuesAndTypesAreRefused(testCase)
            [m, eq, h] = testCase.brusselatorHopf();
            id = "phasecont:model:badOption";
            lc = @(v) phasecont.continuation(m, h, Type = "LC", Free = "b", ...
                NTST = 20, NCOL = 4, MaxSteps = 2, MeshAdapt = v);
            for v = {-1, 1.5, NaN, Inf, [1 2], "3", true}
                testCase.verifyError(@() lc(v{1}), id);
            end
            for v = [0 3]
                testCase.verifyError(@() phasecont.continuation(m, eq, ...
                    Free = "b", MaxSteps = 2, MeshAdapt = v), id);
                testCase.verifyError(@() phasecont.continuation(m, h, Type = "H", ...
                    Free = ["a" "b"], MaxSteps = 2, MeshAdapt = v), id);
            end
        end
    end
end
