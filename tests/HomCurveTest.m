classdef HomCurveTest < matlab.unittest.TestCase
    % Tests curves of orbits homoclinic to a saddle through
    % phasecont.continuation(Type = "HOM"): the start from an explicit
    % orbit, the start from a long-period point of an LC branch, the Branch
    % layout of a HOM curve and the option errors. Every model is written
    % here.
    %
    % Sandstede's model (J. Dyn. Diff. Eq. 9 (1997) 269-288, SPEC section
    % 8b) has at mu = mut = alpha = 0 an orbit homoclinic to the origin in
    % the plane z = 0 for |a| < b: x = 1 - w^2, y = -w (1 - w^2) with w' =
    % (1 - w^2)(b - a w)/2 and w(0) = 0. Along the curve in (mut, a) the
    % saddle stays at the origin, mut stays near 0 and NSA = 2a is the only
    % label. The Bogdanov-Takens values come from SPEC section 8b, test 4,
    % where the shooting in crates/pc-engine/tests/hom_curves.rs gives b1 =
    % -0.0023714546120554 at b2 = -0.1.

    properties (Constant)
        Sandstede = "par a=-0.25,b=1,c=-2,mu=0,mut=0,alpha=0,beta=0,gamma=0" + newline + ...
            "x'=a*x+b*y-a*x^2+(mut-alpha*z)*x*(2-3*x)" + newline + ...
            "y'=b*x+a*y-1.5*b*x^2-1.5*a*x*y-2*y*(mut-alpha*z)" + newline + ...
            "z'=c*z+mu*x+gamma*x*z+alpha*beta*(x^2*(1-x)-y^2)" + newline + ...
            "init x=0,y=0,z=0" + newline + "done" + newline
        BT = "par b1=0.001,b2=-0.1" + newline + ...
            "x'=y" + newline + "y'=b1+b2*x+x^2+x*y" + newline + ...
            "init x=0,y=0" + newline + "done" + newline
        Labels = ["NSA" "BLS" "BLU" "NSF" "NDS" "NDU" "TLS" "TLU" ...
            "ZES" "ZEU" "OFS" "OFU" "IFS" "IFU"]
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

        function start = sandstedeOrbit(~, a, period)
            % The orbit of Sandstede's model at b = 1 over [-T/2, T/2],
            % from w' = (1 - w^2)(1 - a w)/2 with w(0) = 0 by ode45.
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

        function [m, lc, i] = btLongCycle(testCase, maxSteps)
            % The LC branch of MAXSTEPS steps from the Hopf point of the BT
            % normal form at b1 = 0, b2 = -0.1, and the first of its points
            % with period above 100.
            m = testCase.model(testCase.BT);
            eq = phasecont.equilibrium(m, [0.0113 0]);
            ep = phasecont.continuation(m, eq, Free = "b1", Bounds = [-0.01 0.01], ...
                Direction = -1);
            h = ep.points("H");
            testCase.assertNumElements(h, 1);
            testCase.assertEqual(h.P(1), 0, AbsTol = 1e-10);
            lc = phasecont.continuation(m, h, Type = "LC", Free = "b1", ...
                NTST = 80, NCOL = 4, Bounds = [-0.01 0.01], Hmax = 0.05, ...
                MaxNewton = 15, MaxSteps = maxSteps);
            i = find(lc.Period > 100, 1);
            if maxSteps >= 1700
                testCase.assertNotEmpty(i, "the LC branch never passed period 100");
            end
        end
    end

    methods (Test)
        function sandstedeCurveFromAnOrbit(testCase)
            m = testCase.model(testCase.Sandstede);
            [ntst, ncol, n] = deal(160, 4, 3);
            hom = phasecont.continuation(m, testCase.sandstedeOrbit(-0.25, 20), ...
                Type = "HOM", Free = ["mut" "a"], NTST = ntst, NCOL = ncol, ...
                Bounds = [-0.26 0.26], Hmax = 0.05, MaxNewton = 15);
            testCase.verifyEqual(hom.Type, "HOM");
            testCase.verifyEqual(hom.Stop, "Bounds");
            k = numel(hom.Period);
            testCase.verifyGreaterThan(k, 5);
            % The layout of SPEC section 8b, test 6.
            testCase.verifySize(hom.U, [k n*(ntst*ncol + 1)]);
            testCase.verifySize(hom.X0, [k n]);
            testCase.verifySize(hom.Alpha, [k 2]);
            testCase.verifySize(hom.Mesh, [k ntst + 1]);
            testCase.verifySize(hom.Eigenvalues, [k n]);
            testCase.verifySize(hom.Codim2, [k 14]);
            % The IF columns are NaN without InclinationFlips.
            testCase.verifyTrue(all(isnan(hom.Codim2(:, 13:14)), "all"));
            testCase.verifyEqual(hom.Codim2Labels, testCase.Labels);
            testCase.verifyEqual([hom.NTST hom.NCOL], [ntst ncol]);
            testCase.verifyEqual(hom.Period, repmat(20, k, 1), AbsTol = 1e-12);
            testCase.verifyLessThan(max(abs(hom.X0), [], "all"), 1e-10);
            testCase.verifyLessThan(max(abs(hom.Alpha(:, 1))), 1e-6);
            testCase.verifyLessThan(max(abs(hom.U(:, 3:3:end)), [], "all"), 1e-10);
            % At the origin the eigenvalues are a - 1, a + 1 and c = -2.
            ev = sort(real(hom.Eigenvalues(1, :)));
            testCase.verifyEqual(ev, sort([hom.Alpha(1, 2) - 1, hom.Alpha(1, 2) + 1, -2]), ...
                AbsTol = 1e-9);
            testCase.verifyEqual(hom.Codim2(:, 1), 2*hom.Alpha(:, 2), AbsTol = 1e-9);

            testCase.assertNumElements(hom.Specials, 1);
            sp = hom.Specials(1);
            testCase.verifyEqual(sp.Label, "NSA");
            testCase.verifyEqual(sp.P(m.Parameters == "a"), 0, AbsTol = 1e-8);
            testCase.verifySize(sp.Data.x0, [n 1]);
            testCase.verifyLessThan(max(abs(sp.Data.x0)), 1e-10);
            testCase.verifyEqual(sp.Data.period, 20, AbsTol = 1e-12);

            % The orbit over T is not closed: both ends lie near the saddle.
            [t, x] = hom.cycle(1);
            testCase.verifyEqual(t([1 end]), [0; 20], AbsTol = 1e-12);
            testCase.verifySize(x, [ntst*ncol + 1, n]);
            testCase.verifyLessThan(norm(x(1, :) - hom.X0(1, :)), 1e-2);
            testCase.verifyLessThan(norm(x(end, :) - hom.X0(1, :)), 1e-2);
            testCase.verifyGreaterThan(max(x(:, 1)), 0.9);
        end

        function btCurveFromALongCycle(testCase)
            [m, lc, i] = testCase.btLongCycle(1700);
            hom = phasecont.continuation(m, lc, Type = "HOM", Index = i, ...
                Free = ["b1" "b2"], Bounds = [-0.1 -0.05], Hmax = 0.01, ...
                MaxNewton = 15);
            testCase.verifyEqual(hom.Stop, "Bounds");
            testCase.verifyEqual([hom.NTST hom.NCOL], [80 4]);
            testCase.verifyEqual(hom.Period(1), lc.Period(i), AbsTol = 1e-12);
            testCase.verifyEqual(hom.Alpha(1, 2), -0.1, AbsTol = 1e-12);
            testCase.verifyEqual(hom.Alpha(1, 1), -0.0023714546120554, AbsTol = 1e-8);
            testCase.verifyEmpty(hom.Specials);
            % The saddle is the equilibrium x = (-b2 + sqrt(b2^2 - 4 b1))/2.
            b = hom.Alpha;
            testCase.verifyEqual(hom.X0(:, 1), (-b(:, 2) + sqrt(b(:, 2).^2 - 4*b(:, 1)))/2, ...
                AbsTol = 1e-10);
            testCase.verifyEqual(hom.X0(:, 2), zeros(size(b, 1), 1), AbsTol = 1e-10);

            % Another NTST regrids the orbit.
            hom2 = phasecont.continuation(m, lc, Type = "HOM", Index = i, ...
                Free = ["b1" "b2"], NTST = 100, Bounds = [-0.1 -0.09], ...
                Hmax = 0.01, MaxNewton = 15);
            testCase.verifyEqual([hom2.NTST hom2.NCOL], [100 4]);
            testCase.verifySize(hom2.U, [numel(hom2.Period) 2*(100*4 + 1)]);
            testCase.verifyEqual(hom2.Alpha(1, 1), -0.0023714546120554, AbsTol = 1e-8);
        end

        function inclinationFlipsAddTheIFSLabel(testCase)
            % SPEC section 8b-3, test 3: at (a, b, c, beta) = (-0.25, 1, -3, 1)
            % from alpha = 0 to 2.1 at T = 20 there is one IFS label, within
            % 5e-3 of the oracle zero alpha = 1.8961, and no other label.
            m = testCase.model(testCase.Sandstede);
            orbit = testCase.sandstedeOrbit(-0.25, 20);
            orbit.p.c = -3;
            orbit.p.beta = 1;
            hom = @(varargin) phasecont.continuation(m, orbit, "Type", "HOM", ...
                "Free", ["mut" "alpha"], "NTST", 160, "NCOL", 4, ...
                "Bounds", [-0.1 2.1], "Hmax", 0.05, "MaxNewton", 15, varargin{:});

            off = hom();
            testCase.verifyEqual(off.Stop, "Bounds");
            testCase.verifyEmpty(off.Specials);
            testCase.verifyTrue(all(isnan(off.Codim2(:, 13:14)), "all"));

            on = hom("InclinationFlips", true);
            testCase.verifyEqual(on.Stop, "Bounds");
            testCase.verifySize(on.Codim2, [numel(on.Period) 14]);
            testCase.verifyFalse(any(isnan(on.Codim2(:, 13:14)), "all"));
            testCase.assertNumElements(on.Specials, 1);
            testCase.verifyEqual(on.Specials.Label, "IFS");
            testCase.verifyEqual(on.Specials.P(m.Parameters == "alpha"), 1.8961, ...
                AbsTol = 5e-3);

            % An explicit false is the same as leaving the option out.
            same = hom("InclinationFlips", false);
            testCase.verifyEmpty(same.Specials);
            testCase.verifyTrue(all(isnan(same.Codim2(:, 13:14)), "all"));
        end

        function optionErrors(testCase)
            m = testCase.model(testCase.Sandstede);
            orbit = testCase.sandstedeOrbit(-0.25, 20);
            hom = @(start, varargin) phasecont.continuation(m, start, ...
                "Type", "HOM", varargin{:});
            % Two free parameters.
            testCase.verifyError(@() hom(orbit, Free = "a", NTST = 20, NCOL = 4), ...
                "phasecont:model:badOption");
            % NTST and NCOL are required from a struct, and Index is refused.
            testCase.verifyError(@() hom(orbit, Free = ["mut" "a"], NCOL = 4), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() hom(orbit, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4, Index = 1), "phasecont:model:badOption");
            % Detect does not apply.
            testCase.verifyError(@() hom(orbit, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4, Detect = "LP"), "phasecont:model:badOption");
            % The struct needs t, x and p, increasing times and n columns.
            testCase.verifyError(@() hom(rmfield(orbit, "p"), Free = ["mut" "a"], ...
                NTST = 20, NCOL = 4), "phasecont:model:badStart");
            bad = orbit;
            bad.t = flipud(bad.t);
            testCase.verifyError(@() hom(bad, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4), "phasecont:model:badStart");
            bad = orbit;
            bad.x = bad.x(:, 1:2);
            testCase.verifyError(@() hom(bad, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4), "phasecont:model:badStart");
            % A Branch start must be an LC branch and needs Index in range.
            eq = phasecont.equilibrium(m, [0 0 0]);
            ep = phasecont.continuation(m, eq, Free = "mu", Bounds = [-0.1 0.1], ...
                MaxSteps = 3);
            testCase.verifyError(@() hom(ep, Free = ["mut" "a"], Index = 1), ...
                "phasecont:model:badStart");
            % Index applies to HOM only.
            testCase.verifyError(@() phasecont.continuation(m, eq, Free = "mu", ...
                Index = 1), "phasecont:model:badOption");
            % InclinationFlips applies to HOM only, even when false, and
            % takes a logical scalar.
            testCase.verifyError(@() phasecont.continuation(m, eq, Free = "mu", ...
                InclinationFlips = true), "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuation(m, eq, Free = "mu", ...
                InclinationFlips = false), "phasecont:model:badOption");
            testCase.verifyError(@() hom(orbit, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4, InclinationFlips = 1), "phasecont:model:badOption");
            testCase.verifyError(@() hom(orbit, Free = ["mut" "a"], NTST = 20, ...
                NCOL = 4, InclinationFlips = [true false]), "phasecont:model:badOption");
        end

        function indexChecksOnAnLCBranch(testCase)
            [m, lc] = testCase.btLongCycle(5);
            hom = @(varargin) phasecont.continuation(m, lc, "Type", "HOM", ...
                "Free", ["b1" "b2"], varargin{:});
            testCase.verifyError(@() hom(), "phasecont:model:badOption");
            testCase.verifyError(@() hom(Index = 0), "phasecont:model:badOption");
            testCase.verifyError(@() hom(Index = numel(lc.Period) + 1), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() hom(Index = 1.5), "phasecont:model:badOption");
        end
    end
end
