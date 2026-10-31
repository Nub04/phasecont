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
    end
end
