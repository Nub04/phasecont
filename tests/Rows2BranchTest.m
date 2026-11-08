classdef Rows2BranchTest < matlab.unittest.TestCase
    % Tests phasecont.internal.rows2branch on hand-made row matrices.
    % Most cases need no binary: the Model is empty and nothing calls pckern.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function handMadeMatrix(testCase)
            n = 2;
            nan = NaN;
            % Rows: [alpha, u1 u2, re1 re2, im1 im2, psiLP psiBP psiH, h, it]
            pts = [ ...
                0.0, 1 2, -1 -2,  0  0, nan nan nan, 0.1, 3; ...
                0.5, 3 4, -1  1,  0  0, 0.5 nan 0.2, 0.2, 4; ...
                1.0, 5 6,  0.5 0.5, 2 -2, 0.1 0.2 0.3, 0.3, 5].';
            specials = [1.0, 5 6, 0.5 0.5, 2 -2, 0.1 0.2 0.3, 0.3, 5].';
            st = struct("points", 3, "stop", "Bounds", ...
                "warnings", {{"w1", "w2"}}, ...
                "specials", struct("label", "H", "index", 2, ...
                    "coefficient", 0.25, "omega", 2, "normalization", "n1", ...
                    "refined", true));
            br = phasecont.internal.rows2branch( ...
                phasecont.Model.empty, 2, [10; 20; 30], n, pts, specials, st);
            testCase.verifyClass(br, "phasecont.Branch");
            testCase.verifyEqual(br.Alpha, [0; 0.5; 1]);
            testCase.verifyEqual(br.U, [1 2; 3 4; 5 6]);
            testCase.verifyEqual(br.Eigenvalues, ...
                [complex(-1, 0) complex(-2, 0); ...
                 complex(-1, 0) complex(1, 0); ...
                 complex(0.5, 2) complex(0.5, -2)]);
            testCase.verifyEqual(br.Stable, [true; false; false]);
            testCase.verifyEqual(br.Tests, [nan nan nan; 0.5 nan 0.2; 0.1 0.2 0.3]);
            testCase.verifyEqual(br.StepSize, [0.1; 0.2; 0.3]);
            testCase.verifyEqual(br.NewtonIters, [3; 4; 5]);
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.verifyEqual(br.Warnings, ["w1"; "w2"]);
            testCase.verifyEqual(br.P, [10; 20; 30]);
            testCase.verifyEqual(br.FreeIndex, 2);
            testCase.verifySize(br.Specials, [1 1]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Label, "H");
            testCase.verifyEqual(sp.Index, 3);
            testCase.verifyEqual(sp.P, [10; 1.0; 30]);
            testCase.verifyEqual(sp.U, [5; 6]);
            testCase.verifyEqual(sp.Eigenvalues, [complex(0.5, 2); complex(0.5, -2)]);
            testCase.verifyEqual(sp.Data.coefficient, 0.25);
            testCase.verifyEqual(sp.Data.omega, 2);
            testCase.verifyEqual(sp.Data.normalization, "n1");
            testCase.verifyEqual(sp.Data.refined, true);
            testCase.verifyEqual(size(br.points("H")), [1 1]);
            testCase.verifySize(br.points("LP"), [0 1]);
            testCase.verifySize(br.points(), [1 1]);
        end

        function nullsAndNoSpecials(testCase)
            n = 1;
            pts = [0.0, 1, -1, 0, nan nan nan, 0.1, 2].';
            st = struct("points", 1, "stop", [], "warnings", [], ...
                "specials", []);
            br = phasecont.internal.rows2branch( ...
                phasecont.Model.empty, 1, 7, n, pts, zeros(9, 0), st);
            testCase.verifyEqual(br.Stop, "");
            testCase.verifySize(br.Warnings, [0 1]);
            testCase.verifySize(br.Specials, [0 1]);
            testCase.verifyEqual(br.Stable, true);

            sp = struct("label", "LP", "index", 0, "coefficient", [], ...
                "omega", [], "normalization", [], "refined", false);
            st.specials = sp;
            row = [0.0, 1, 0, 0, nan nan nan, 0.1, 2].';
            br = phasecont.internal.rows2branch( ...
                phasecont.Model.empty, 1, 7, n, pts, row, st);
            d = br.Specials(1).Data;
            testCase.verifyTrue(isnan(d.coefficient));
            testCase.verifyTrue(isnan(d.omega));
            testCase.verifyEqual(d.normalization, "");
            testCase.verifyEqual(br.Specials(1).Index, 1);
        end

        function pointsOnAnArrayIsAnError(testCase)
            n = 1;
            pts = [0.0, 1, -1, 0, nan nan nan, 0.1, 2].';
            st = struct("points", 1, "stop", "MaxSteps", "warnings", [], ...
                "specials", []);
            br = phasecont.internal.rows2branch( ...
                phasecont.Model.empty, 1, 7, n, pts, zeros(9, 0), st);
            both = [br br];
            testCase.verifyError(@() points(both), "phasecont:model:badBranch");
        end

        function unknownFreeNameWithARealModel(testCase)
            % This case needs a real Model, so it needs the binary.
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
            d = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            file = fullfile(d.Folder, "fold.ode");
            fid = fopen(file, "w");
            fprintf(fid, "par a=-1\ninit x=1\nx'=a+x^2\ndone\n");
            fclose(fid);
            m = phasecont.Model(file);
            n = 1;
            pts = [0.0, 1, -1, 0, nan nan nan, 0.1, 2].';
            st = struct("points", 1, "stop", "MaxSteps", "warnings", [], ...
                "specials", []);
            testCase.verifyError(@() phasecont.internal.rows2branch( ...
                m, "zz", -1, n, pts, zeros(9, 0), st), ...
                "phasecont:model:unknownParameter");
            br = phasecont.internal.rows2branch( ...
                m, "a", -1, n, pts, zeros(9, 0), st);
            testCase.verifyEqual(br.FreeIndex, 1);
        end

        function equilibriumBranchHasNoCurveFields(testCase)
            n = 1;
            pts = [0.0, 1, -1, 0, nan nan nan, 0.1, 2].';
            st = struct("points", 1, "stop", [], "warnings", [], ...
                "specials", []);
            br = phasecont.internal.rows2branch( ...
                phasecont.Model.empty, 1, 7, n, pts, zeros(9, 0), st);
            testCase.verifyEqual(br.Type, "EP");
            testCase.verifySize(br.Alpha, [1 1]);
            testCase.verifySize(br.Codim2, [1 0]);
            testCase.verifyEqual(br.Codim2Labels, strings(1, 0));
        end

        function lpCurveRows(testCase)
            n = 2;
            nan = NaN;
            % Rows: [a2, x1 x2, a1, re1 re2, im1 im2, CP BT ZH, h, it]
            pts = [ ...
                0.10, 1 2, 0.40, -1 -2, 0  0, nan nan nan, 0.1, 3; ...
                0.15, 3 4, 0.45,  0  0, 1 -1, 0.5 nan nan, 0.2, 4; ...
                0.20, 5 6, 0.50, 0.1 0.2, 0 0, nan nan nan, 0.3, 5].';
            specials = pts(:, 2);
            q = struct("q", 2);
            st = struct("points", 3, "stop", "Bounds", "warnings", [], ...
                "specials", struct("label", "CP", "index", 1, ...
                    "coefficient", -1, "coefficient2", [], "coefficients", q, ...
                    "omega", [], "normalization", [], "refined", true));
            % free = [a1 a2]: a1 is parameter 3 and a2 is parameter 1.
            br = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [3 1], [10; 20; 30], n, pts, specials, st, "LP");
            testCase.verifyEqual(br.Type, "LP");
            testCase.verifyEqual(br.Free, ["p3" "p1"]);
            testCase.verifyEqual(br.FreeIndex, [3 1]);
            testCase.verifyEqual(br.Alpha, [0.40 0.10; 0.45 0.15; 0.50 0.20]);
            testCase.verifyEqual(br.U, [1 2; 3 4; 5 6]);
            testCase.verifyEqual(br.Eigenvalues, ...
                [complex(-1, 0) complex(-2, 0); ...
                 complex(0, 1) complex(0, -1); ...
                 complex(0.1, 0) complex(0.2, 0)]);
            testCase.verifyEqual(br.Stable, false(3, 1));
            testCase.verifySize(br.Tests, [3 0]);
            testCase.verifyEqual(br.Codim2, [nan nan nan; 0.5 nan nan; nan nan nan]);
            testCase.verifyEqual(br.Codim2Labels, ["CP" "BT" "ZH"]);
            testCase.verifyEqual(br.StepSize, [0.1; 0.2; 0.3]);
            testCase.verifyEqual(br.NewtonIters, [3; 4; 5]);
            testCase.verifyEqual(br.P, [10; 20; 30]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Label, "CP");
            testCase.verifyEqual(sp.Index, 2);
            testCase.verifyEqual(sp.P, [0.15; 20; 0.45]);
            testCase.verifyEqual(sp.U, [3; 4]);
            testCase.verifyEqual(sp.Data.coefficient, -1);
            testCase.verifyTrue(isnan(sp.Data.coefficient2));
            testCase.verifyEqual(sp.Data.values, q);
            testCase.verifyEqual(sp.Data.refined, true);
            testCase.verifySize(br.points("CP"), [1 1]);
        end

        function hopfCurveRowsDropKappa(testCase)
            n = 2;
            % Rows: [a2, x1 x2, a1, kappa, re1 re2, im1 im2, GH BT ZH HH, h, it]
            pts = [0.3, 1 2, 0.7, 9, 0 0, 1 -1, nan nan nan nan, 0.1, 3].';
            st = struct("points", 1, "stop", [], "warnings", [], ...
                "specials", []);
            br = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [1 2], [0; 0], n, pts, zeros(15, 0), st, "H");
            testCase.verifyEqual(br.Type, "H");
            testCase.verifyEqual(br.Alpha, [0.7 0.3]);
            testCase.verifyEqual(br.U, [1 2]);
            testCase.verifySize(br.Codim2, [1 4]);
            testCase.verifyEqual(br.Codim2Labels, ["GH" "BT" "ZH" "HH"]);
            testCase.verifyEqual(br.Eigenvalues, [complex(0, 1) complex(0, -1)]);
        end

        function curveRowErrors(testCase)
            n = 2;
            st = struct("points", 0, "stop", [], "warnings", [], ...
                "specials", []);
            e = phasecont.Model.empty;
            % An EP-sized matrix is the wrong size for a curve.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], n, zeros(12, 0), zeros(12, 0), st, "LP"), ...
                "phasecont:model:badRows");
            % A curve needs two free parameters and an EP run needs one.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], n, zeros(13, 0), zeros(13, 0), st, "LP"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], n, zeros(12, 0), zeros(12, 0), st), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], n, zeros(13, 0), zeros(13, 0), st, "XX"), ...
                "phasecont:model:badOption");
        end

        function cycleRows(testCase)
            % n = 2, NTST = 2, NCOL = 2: rows of [alpha, orbit(10), T,
            % mesh(3), mult_re(2), mult_im(2), psiLPC psiPD psiNS, h, it].
            nan = NaN;
            pts = [ ...
                0.1, 1 2 2 3 3 4 2 1 1 2, 6, 0 0.4 1, 1 0.5, 0 0, 0.1 0.2 0.3, 0.01, 2; ...
                0.2, 1 2 5 6 7 8 6 5 1 2, 7, 0 0.5 1, -1.5 1, 0 0, nan nan nan, 0.02, 3; ...
                0.3, 0 0 1 1 2 2 1 1 0 0, 8, 0 0.6 1, nan 1, 0 0, 0.4 0.5 0.6, 0.03, 4].';
            specials = pts(:, 2);
            st = struct("points", 3, "stop", "max_steps", "warnings", [], ...
                "specials", struct("label", "PD", "index", 1));
            br = phasecont.internal.rows2branch(phasecont.Model.empty, 1, ...
                [0; 9], 2, pts, specials, st, "LC", [2 2]);
            testCase.verifyEqual(br.Type, "LC");
            testCase.verifyEqual(br.Alpha, [0.1; 0.2; 0.3]);
            testCase.verifyEqual(br.U, pts(2:11, :).');
            testCase.verifyEqual(br.Period, [6; 7; 8]);
            testCase.verifyEqual(br.Mesh, [0 0.4 1; 0 0.5 1; 0 0.6 1]);
            testCase.verifyEqual([br.NTST br.NCOL], [2 2]);
            testCase.verifyEqual(real(br.Eigenvalues(1, :)), [1 0.5]);
            testCase.verifyEqual(imag(br.Eigenvalues), zeros(3, 2));
            % Stable means every multiplier but the one closest to 1 is inside
            % the unit circle; a NaN multiplier is not stable.
            testCase.verifyEqual(br.Stable, [true; false; false]);
            testCase.verifyEqual(br.Tests, [0.1 0.2 0.3; nan nan nan; 0.4 0.5 0.6]);
            testCase.verifyEqual(br.StepSize, [0.01; 0.02; 0.03]);
            testCase.verifyEqual(br.NewtonIters, [2; 3; 4]);
            testCase.verifySize(br.Codim2, [3 0]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Label, "PD");
            testCase.verifyEqual(sp.Index, 2);
            testCase.verifyEqual(sp.P, [0.2; 9]);
            testCase.verifyEqual(sp.U, [1 2 5 6 7 8 6 5 1 2].');
            testCase.verifyEqual(sp.Data.period, 7);
            testCase.verifyEqual(sp.Data.mesh, [0 0.5 1]);
            testCase.verifyEqual([sp.Data.ntst sp.Data.ncol], [2 2]);

            % The base points of interval j are tau_j + (k/NCOL) h_j, then 1.
            [t, x] = br.cycle(1);
            testCase.verifyEqual(t, [0; 0.2; 0.4; 0.7; 1] * 6, AbsTol = 1e-15);
            testCase.verifyEqual(x, [1 2; 2 3; 3 4; 2 1; 1 2]);
            testCase.verifyError(@() br.cycle(4), "phasecont:model:badBranch");
            testCase.verifyError(@() br.cycle(1.5), "phasecont:model:badBranch");
            ep = phasecont.internal.rows2branch(phasecont.Model.empty, 1, ...
                [0; 0], 2, zeros(12, 0), zeros(12, 0), ...
                struct("points", 0, "stop", [], "warnings", [], "specials", []));
            testCase.verifyError(@() ep.cycle(1), "phasecont:model:badBranch");
        end

        function cycleRowErrors(testCase)
            st = struct("points", 0, "stop", [], "warnings", [], ...
                "specials", []);
            e = phasecont.Model.empty;
            % Stride for n = 2, NTST = 2, NCOL = 2 is 24.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], 2, zeros(24, 0), zeros(24, 0), st, "LC"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], 2, zeros(24, 0), zeros(24, 0), st, "LC", [2 0]), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], 2, zeros(20, 0), zeros(20, 0), st, "LC", [2 2]), ...
                "phasecont:model:badRows");
            br = phasecont.internal.rows2branch(e, 1, [0; 0], 2, ...
                zeros(24, 0), zeros(24, 0), st, "LC", [2 2]);
            testCase.verifySize(br.U, [0 10]);
            testCase.verifySize(br.Mesh, [0 3]);
            testCase.verifySize(br.Period, [0 1]);
        end

        function cycleCurveRows(testCase)
            % LPC curve, n = 2, NTST = 2, NCOL = 2: rows of [a2, orbit(10),
            % T, a1, mesh(3), mult_re(2), mult_im(2), LPPD LPNS CPC R1, h,
            % it].
            nan = NaN;
            pts = [ ...
                0.5, 1 2 2 3 3 4 2 1 1 2, 6, -0.1, 0 0.4 1, 1 1, 0 0, 0.1 0.2 0.3 0.4, 0.01, 2; ...
                0.6, 1 2 5 6 7 8 6 5 1 2, 7, -0.2, 0 0.5 1, 1 1, 0 0, nan nan nan nan, 0.02, 3].';
            specials = pts(:, 2);
            st = struct("points", 2, "stop", "Bounds", "warnings", [], ...
                "specials", struct("label", "LPPD", "index", 1));
            % free = [a1 a2]: a1 is parameter 3 and a2 is parameter 1.
            br = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [3 1], [10; 20; 30], 2, pts, specials, st, "LPC", [2 2]);
            testCase.verifyEqual(br.Type, "LPC");
            testCase.verifyEqual(br.Free, ["p3" "p1"]);
            testCase.verifyEqual(br.Alpha, [-0.1 0.5; -0.2 0.6]);
            testCase.verifyEqual(br.U, pts(2:11, :).');
            testCase.verifyEqual(br.Period, [6; 7]);
            testCase.verifyEqual(br.Mesh, [0 0.4 1; 0 0.5 1]);
            testCase.verifyEqual([br.NTST br.NCOL], [2 2]);
            testCase.verifyEqual(br.Eigenvalues, complex(ones(2, 2), 0));
            testCase.verifyEqual(br.Stable, false(2, 1));
            testCase.verifySize(br.Tests, [2 0]);
            testCase.verifyEqual(br.Codim2, [0.1 0.2 0.3 0.4; nan nan nan nan]);
            testCase.verifyEqual(br.Codim2Labels, ["LPPD" "LPNS" "CPC" "R1"]);
            testCase.verifySize(br.Kappa, [0 1]);
            testCase.verifyEqual(br.StepSize, [0.01; 0.02]);
            testCase.verifyEqual(br.NewtonIters, [2; 3]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Label, "LPPD");
            testCase.verifyEqual(sp.Index, 2);
            testCase.verifyEqual(sp.P, [0.6; 20; -0.2]);
            testCase.verifyEqual(sp.U, [1 2 5 6 7 8 6 5 1 2].');
            testCase.verifyEqual(sp.Data.period, 7);
            testCase.verifyEqual(sp.Data.mesh, [0 0.5 1]);
            testCase.verifyEqual([sp.Data.ntst sp.Data.ncol], [2 2]);
            testCase.verifyFalse(isfield(sp.Data, "kappa"));
            [t, x] = br.cycle(1);
            testCase.verifyEqual(t, [0; 0.2; 0.4; 0.7; 1] * 6, AbsTol = 1e-15);
            testCase.verifyEqual(x, [1 2; 2 3; 3 4; 2 1; 1 2]);

            % A PD curve has the same layout and its own labels.
            pd = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [3 1], [10; 20; 30], 2, pts, specials, st, "PD", [2 2]);
            testCase.verifyEqual(pd.Type, "PD");
            testCase.verifyEqual(pd.Codim2Labels, ["LPPD" "PDNS" "GPD" "R2"]);
            testCase.verifyEqual(pd.Alpha, [-0.1 0.5; -0.2 0.6]);
        end

        function nsCurveRowsHaveKappa(testCase)
            % NS curve, n = 2, NTST = 2, NCOL = 2: rows of [a2, orbit(10),
            % T, a1, kappa, mesh(3), mult_re(2), mult_im(2), R1 R2 R3 R4
            % LPNS PDNS CH, h, it].
            pts = [ ...
                0.5, 1 2 2 3 3 4 2 1 1 2, 6, -0.1, 0.3, 0 0.4 1, 0.3 0.3, 0.9 -0.9, 1 2 3 4 5 6 7, 0.01, 2; ...
                0.6, 1 2 5 6 7 8 6 5 1 2, 7, -0.2, 0.2, 0 0.5 1, 0.2 0.2, 0.9 -0.9, 7 6 5 4 3 2 1, 0.02, 3].';
            st = struct("points", 2, "stop", [], "warnings", [], ...
                "specials", struct("label", "R4", "index", 0));
            br = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [1 2], [0; 0], 2, pts, pts(:, 1), st, "NS", [2 2]);
            testCase.verifyEqual(br.Type, "NS");
            testCase.verifyEqual(br.Alpha, [-0.1 0.5; -0.2 0.6]);
            testCase.verifyEqual(br.Period, [6; 7]);
            testCase.verifyEqual(br.Kappa, [0.3; 0.2]);
            testCase.verifyEqual(br.Mesh, [0 0.4 1; 0 0.5 1]);
            testCase.verifyEqual(br.Eigenvalues(1, :), [complex(0.3, 0.9) complex(0.3, -0.9)]);
            testCase.verifyEqual(br.Codim2, [1 2 3 4 5 6 7; 7 6 5 4 3 2 1]);
            testCase.verifyEqual(br.Codim2Labels, ["R1" "R2" "R3" "R4" "LPNS" "PDNS" "CH"]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Index, 1);
            testCase.verifyEqual(sp.P, [-0.1; 0.5]);
            testCase.verifyEqual(sp.Data.kappa, 0.3);
            testCase.verifyEqual(sp.Data.period, 6);
            [t, ~] = br.cycle(2);
            testCase.verifyEqual(t(end), 7);
        end

        function cycleCurveRowErrors(testCase)
            st = struct("points", 0, "stop", [], "warnings", [], ...
                "specials", []);
            e = phasecont.Model.empty;
            % Strides for n = 2, NTST = 2, NCOL = 2: 26 for LPC and PD, 30
            % for NS.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(26, 0), zeros(26, 0), st, "LPC"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(26, 0), zeros(26, 0), st, "PD", [0 2]), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], 2, zeros(26, 0), zeros(26, 0), st, "PD", [2 2]), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(26, 0), zeros(26, 0), st, "NS", [2 2]), ...
                "phasecont:model:badRows");
            br = phasecont.internal.rows2branch(e, [1 2], [0; 0], 2, ...
                zeros(30, 0), zeros(30, 0), st, "NS", [2 2]);
            testCase.verifySize(br.Alpha, [0 2]);
            testCase.verifySize(br.U, [0 10]);
            testCase.verifySize(br.Kappa, [0 1]);
            testCase.verifySize(br.Codim2, [0 7]);
            br = phasecont.internal.rows2branch(e, [1 2], [0; 0], 2, ...
                zeros(26, 0), zeros(26, 0), st, "LPC", [2 2]);
            testCase.verifySize(br.Alpha, [0 2]);
            testCase.verifySize(br.Period, [0 1]);
        end

        function oldCurveRowsArePaddedWithNaN(testCase)
            % A v1.0.0 session stored LPC and PD rows with two codim-2 tests
            % (24 rows for n = 2, NTST = 2, NCOL = 2) and NS rows with six
            % (29 rows). The new tests read NaN.
            e = phasecont.Model.empty;
            st = struct("points", 2, "stop", [], "warnings", [], ...
                "specials", struct("label", "LPPD", "index", 1));
            lpc = [ ...
                0.5, 1 2 2 3 3 4 2 1 1 2, 6, -0.1, 0 0.4 1, 1 1, 0 0, 0.1 0.2, 0.01, 2; ...
                0.6, 1 2 5 6 7 8 6 5 1 2, 7, -0.2, 0 0.5 1, 1 1, 0 0, 0.3 0.4, 0.02, 3].';
            testCase.verifySize(lpc, [24 2]);
            for kind = ["LPC" "PD"]
                br = phasecont.internal.rows2branch(e, [3 1], [10; 20; 30], 2, ...
                    lpc, lpc(:, 2), st, kind, [2 2]);
                testCase.verifyEqual(br.Codim2, [0.1 0.2 NaN NaN; 0.3 0.4 NaN NaN]);
                testCase.verifyEqual(br.StepSize, [0.01; 0.02]);
                testCase.verifyEqual(br.NewtonIters, [2; 3]);
                testCase.verifyEqual(br.Eigenvalues, complex(ones(2, 2), 0));
                testCase.verifyEqual(br.Period, [6; 7]);
                testCase.verifyEqual(br.Specials(1).Index, 2);
                testCase.verifyEqual(br.Specials(1).Label, "LPPD");
            end

            ns = [ ...
                0.5, 1 2 2 3 3 4 2 1 1 2, 6, -0.1, 0.3, 0 0.4 1, 0.3 0.3, 0.9 -0.9, 1 2 3 4 5 6, 0.01, 2; ...
                0.6, 1 2 5 6 7 8 6 5 1 2, 7, -0.2, 0.2, 0 0.5 1, 0.2 0.2, 0.9 -0.9, 6 5 4 3 2 1, 0.02, 3].';
            testCase.verifySize(ns, [29 2]);
            br = phasecont.internal.rows2branch(e, [1 2], [0; 0], 2, ...
                ns, ns(:, 1), st, "NS", [2 2]);
            testCase.verifyEqual(br.Codim2, [1 2 3 4 5 6 NaN; 6 5 4 3 2 1 NaN]);
            testCase.verifyEqual(br.Kappa, [0.3; 0.2]);
            testCase.verifyEqual(br.Eigenvalues(1, :), [complex(0.3, 0.9) complex(0.3, -0.9)]);
            testCase.verifyEqual(br.StepSize, [0.01; 0.02]);
            testCase.verifyEqual(br.NewtonIters, [2; 3]);
            testCase.verifyEqual(br.Specials(1).Data.kappa, 0.3);

            % An old width with no points pads to the new width.
            br = phasecont.internal.rows2branch(e, [1 2], [0; 0], 2, ...
                zeros(29, 0), zeros(29, 0), ...
                struct("points", 0, "stop", [], "warnings", [], "specials", []), ...
                "NS", [2 2]);
            testCase.verifySize(br.Codim2, [0 7]);
            % The points and the specials must both be old, and no other
            % width is accepted.
            none = struct("points", 0, "stop", [], "warnings", [], "specials", []);
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(29, 0), zeros(30, 0), none, "NS", [2 2]), ...
                "phasecont:model:badRows");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(28, 0), zeros(28, 0), none, "NS", [2 2]), ...
                "phasecont:model:badRows");
            % An LC or HOM row never changed width, so a narrow one is an error.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(24, 0), zeros(24, 0), none, "HOM", [2 2]), ...
                "phasecont:model:badRows");
        end

        function homCurveRows(testCase)
            % HOM curve, n = 2, NTST = 2, NCOL = 2: rows of [a2, orbit(10),
            % T, x0(2), a1, mesh(3), eig_re(2), eig_im(2), codim2(14), h,
            % it], 38 rows in all.
            c2 = [1:14; NaN(1, 12) 0.5 0.6];
            pts = [ ...
                0.5, 0 0 1 2 3 4 1 1 0.1 0.1, 20, 0.01 0.02, -0.1, 0 0.4 1, -1 2, 0 0, c2(1, :), 0.01, 2; ...
                0.6, 0 0 2 3 4 5 2 2 0.1 0.1, 20, 0.03 0.04, -0.2, 0 0.5 1, -1.5 2.5, 0 0, c2(2, :), 0.02, 3].';
            st = struct("points", 2, "stop", "Bounds", "warnings", [], ...
                "specials", struct("label", "OFS", "index", 1));
            br = phasecont.internal.rows2branch(phasecont.Model.empty, ...
                [3 1], [10; 20; 30], 2, pts, pts(:, 2), st, "HOM", [2 2]);
            testCase.verifyEqual(br.Type, "HOM");
            testCase.verifyEqual(br.Free, ["p3" "p1"]);
            testCase.verifyEqual(br.Alpha, [-0.1 0.5; -0.2 0.6]);
            testCase.verifyEqual(br.U, pts(2:11, :).');
            testCase.verifyEqual(br.Period, [20; 20]);
            testCase.verifyEqual(br.X0, [0.01 0.02; 0.03 0.04]);
            testCase.verifyEqual(br.Mesh, [0 0.4 1; 0 0.5 1]);
            testCase.verifyEqual([br.NTST br.NCOL], [2 2]);
            testCase.verifyEqual(br.Eigenvalues, complex([-1 2; -1.5 2.5], 0));
            testCase.verifyEqual(br.Stable, false(2, 1));
            testCase.verifySize(br.Tests, [2 0]);
            testCase.verifyEqual(br.Codim2, c2);
            testCase.verifyEqual(br.Codim2Labels, ["NSA" "BLS" "BLU" "NSF" ...
                "NDS" "NDU" "TLS" "TLU" "ZES" "ZEU" "OFS" "OFU" "IFS" "IFU"]);
            testCase.verifySize(br.Kappa, [0 1]);
            testCase.verifyEqual(br.StepSize, [0.01; 0.02]);
            testCase.verifyEqual(br.NewtonIters, [2; 3]);
            sp = br.Specials(1);
            testCase.verifyEqual(sp.Label, "OFS");
            testCase.verifyEqual(sp.Index, 2);
            testCase.verifyEqual(sp.P, [0.6; 20; -0.2]);
            testCase.verifyEqual(sp.U, [0 0 2 3 4 5 2 2 0.1 0.1].');
            testCase.verifyEqual(sp.Eigenvalues, [-1.5; 2.5]);
            testCase.verifyEqual(sp.Data.x0, [0.03; 0.04]);
            testCase.verifyEqual(sp.Data.period, 20);
            testCase.verifyEqual(sp.Data.mesh, [0 0.5 1]);
            testCase.verifyEqual([sp.Data.ntst sp.Data.ncol], [2 2]);
            % The orbit over T, not closed: it starts and ends near X0.
            [t, x] = br.cycle(1);
            testCase.verifyEqual(t, [0; 0.2; 0.4; 0.7; 1] * 20, AbsTol = 1e-13);
            testCase.verifyEqual(x, [0 0; 1 2; 3 4; 1 1; 0.1 0.1]);
        end

        function homCurveRowErrors(testCase)
            st = struct("points", 0, "stop", [], "warnings", [], ...
                "specials", []);
            e = phasecont.Model.empty;
            % The stride for n = 2, NTST = 2, NCOL = 2 is 38.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(38, 0), zeros(38, 0), st, "HOM"), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.internal.rows2branch(e, [1 2], ...
                [0; 0], 2, zeros(26, 0), zeros(26, 0), st, "HOM", [2 2]), ...
                "phasecont:model:badRows");
            br = phasecont.internal.rows2branch(e, [1 2], [0; 0], 2, ...
                zeros(38, 0), zeros(38, 0), st, "HOM", [2 2]);
            testCase.verifySize(br.Alpha, [0 2]);
            testCase.verifySize(br.U, [0 10]);
            testCase.verifySize(br.X0, [0 2]);
            testCase.verifySize(br.Codim2, [0 14]);
            testCase.verifySize(br.Period, [0 1]);
        end

        function delayRows(testCase)
            % A delay row is [alpha, u(1..n), count], n + 2 values, with
            % the count of roots in the right half plane stored as a double.
            n = 2;
            pts = [ ...
                1.0, 0.5 0.25, 0; ...
                1.5, 0.6 0.30, 0; ...
                2.0, 0.7 0.35, 2].';
            specials = [1.5708, 0.6 0.30, 2].';
            st = struct("points", 3, "stop", "Bounds", "warnings", {{}}, ...
                "specials", struct("label", "H", "index", 1, ...
                    "coefficient", [], "omega", 1.5708, ...
                    "normalization", [], "refined", false, ...
                    "coefficients", struct("unstable_before", 0, ...
                        "unstable_after", 2)));
            br = phasecont.internal.rows2branch(phasecont.Model.empty, 1, ...
                [1.3; 1], n, pts, specials, st, "DDE");
            testCase.verifyEqual(br.Type, "EP");
            testCase.verifyEqual(br.Alpha, [1.0; 1.5; 2.0]);
            testCase.verifyEqual(br.U, [0.5 0.25; 0.6 0.30; 0.7 0.35]);
            testCase.verifyEqual(br.RootCount, [0; 0; 2]);
            testCase.verifyEqual(br.Stable, [true; true; false]);
            testCase.verifySize(br.Eigenvalues, [3 0]);
            testCase.verifySize(br.Tests, [3 0]);
            testCase.verifySize(br.Codim2, [3 0]);
            testCase.verifyEqual(br.StepSize, NaN(3, 1));
            testCase.verifyEqual(br.NewtonIters, NaN(3, 1));
            testCase.verifyEqual(br.Stop, "Bounds");
            testCase.assertNumElements(br.Specials, 1);
            sp = br.Specials;
            testCase.verifyEqual(sp.Label, "H");
            testCase.verifyEqual(sp.Index, 2);
            testCase.verifyEqual(sp.P, [1.5708; 1]);
            testCase.verifyEqual(sp.U, [0.6; 0.30]);
            testCase.verifyEmpty(sp.Eigenvalues);
            testCase.verifyEqual(sp.Data.omega, 1.5708);
            testCase.verifyEqual(sp.Data.values.unstable_before, 0);
            testCase.verifyEqual(sp.Data.values.unstable_after, 2);
        end

        function delayRowErrors(testCase)
            st = struct("points", 0, "stop", [], "warnings", [], ...
                "specials", []);
            e = phasecont.Model.empty;
            % The stride for n = 2 is 4. An equilibrium row of an ODE has 11.
            testCase.verifyError(@() phasecont.internal.rows2branch(e, 1, ...
                [0; 0], 2, zeros(11, 0), zeros(11, 0), st, "DDE"), ...
                "phasecont:model:badRows");
            br = phasecont.internal.rows2branch(e, 1, [0; 0], 2, ...
                zeros(4, 0), zeros(4, 0), st, "DDE");
            testCase.verifySize(br.Alpha, [0 1]);
            testCase.verifySize(br.RootCount, [0 1]);
            testCase.verifySize(br.Eigenvalues, [0 0]);
        end
    end
end
