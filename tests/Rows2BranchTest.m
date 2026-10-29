classdef Rows2BranchTest < matlab.unittest.TestCase
    % Tests phasecont.internal.rows2branch on hand-made row matrices.
    % No binary is needed: the Model is empty and nothing calls pckern.

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
    end
end
