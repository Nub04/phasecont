classdef RunRecordTest < matlab.unittest.TestCase
    % Tests phasecont.internal.RunRecord and Run.decode. No test here calls
    % pckern, so the suite runs without the MEX binary.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function holdsTheFieldsItIsGiven(testCase)
            rec = phasecont.internal.RunRecord(sample());
            testCase.verifyEqual(rec.Kind, "continuation");
            testCase.verifyEqual(rec.Problem, "{""type"":""EP""}");
            testCase.verifyEqual(rec.U0, [1; 2]);
            testCase.verifyEqual(rec.P, [0.5; 3]);
            testCase.verifyEqual(rec.Free, "r");
            testCase.verifyEqual(rec.Stride, 3);
            testCase.verifyEqual(rec.branchCount(), 2);
            testCase.verifyEqual(size(rec.Points{2}), [3 1]);
            testCase.verifyEqual(rec.Result, struct());
        end

        function aHandleIsSharedBetweenBranches(testCase)
            rec = phasecont.internal.RunRecord(sample());
            a = phasecont.Branch(struct("Record", rec, "RunBranch", 1));
            b = phasecont.Branch(struct("Record", rec, "RunBranch", 2));
            testCase.verifyTrue(a.Record == b.Record);
            testCase.verifyEqual([a.RunBranch b.RunBranch], [1 2]);
            fresh = phasecont.Branch();
            testCase.verifyEmpty(fresh.Record);
            testCase.verifyTrue(isnan(fresh.RunBranch));
        end

        function aPointKeepsItsRecordOnlyWhenGivenOne(testCase)
            rec = phasecont.internal.RunRecord( ...
                struct("Kind", "equilibrium", "U0", [1; 2], "P", 3, ...
                    "Result", struct("U", [1; 2], "Eigenvalues", [-1; -2])));
            pt = phasecont.Point(U = [1; 2], Record = rec);
            testCase.verifyTrue(pt.Record == rec);
            testCase.verifyEmpty(phasecont.Point(U = [1; 2]).Record);
        end

        function rejectsAnUnknownField(testCase)
            testCase.verifyError( ...
                @() phasecont.internal.RunRecord(struct("Colour", "red")), ...
                "phasecont:session:badRecord");
        end

        function rejectsAnUnknownKind(testCase)
            testCase.verifyError( ...
                @() phasecont.internal.RunRecord(struct("Kind", "shooting")), ...
                "phasecont:session:badRecord");
        end

        function rejectsPointsWithoutMatchingSpecials(testCase)
            f = sample();
            f.Specials = {zeros(3, 0)};
            testCase.verifyError(@() phasecont.internal.RunRecord(f), ...
                "phasecont:session:badRecord");
        end

        function decodeGivesStructArraysForBranchesAndSpecials(testCase)
            % Two branches: the first has two specials with the same fields,
            % the second has none. jsondecode gives a struct array and [].
            raw = "{""state"":""done"",""stride"":3,""branches"":[" + ...
                "{""points"":4,""stop"":""Bounds"",""warnings"":[]," + ...
                """specials"":[{""label"":""H"",""index"":2}," + ...
                "{""label"":""LP"",""index"":3}]}," + ...
                "{""points"":1,""stop"":""End"",""warnings"":[],""specials"":[]}]}";
            s = phasecont.internal.Run.decode(raw);
            testCase.verifyEqual(s.state, "done");
            testCase.verifyEqual(numel(s.branches), 2);
            testCase.verifyEqual(numel(s.branches(1).specials), 2);
            testCase.verifyEqual(string(s.branches(1).specials(2).label), "LP");
            testCase.verifyEqual(numel(s.branches(2).specials), 0);
        end

        function decodeAcceptsSpecialsWithDifferentFields(testCase)
            raw = "{""state"":""done"",""stride"":3,""branches"":[" + ...
                "{""points"":4,""stop"":""Bounds"",""warnings"":[]," + ...
                """specials"":[{""label"":""H"",""index"":2,""omega"":1.5}," + ...
                "{""label"":""LP"",""index"":3}]}]}";
            s = phasecont.internal.Run.decode(raw);
            testCase.verifyEqual(numel(s.branches(1).specials), 2);
            testCase.verifyEqual(s.branches(1).specials(1).omega, 1.5);
        end
    end
end

function f = sample()
% A two-branch continuation record with stride-3 rows.
f = struct("Kind", "continuation", "Problem", "{""type"":""EP""}", ...
    "U0", [1; 2], "P", [0.5; 3], "ModelSha", repmat('0', 1, 64), ...
    "Type", "EP", "Free", "r", "Grid", zeros(1, 0), "N", 2, ...
    "Json", "{}", "Points", {{zeros(3, 4), zeros(3, 1)}}, ...
    "Specials", {{zeros(3, 2), zeros(3, 0)}}, "Stride", 3);
end
