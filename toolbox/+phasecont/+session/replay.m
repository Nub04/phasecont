function r = replay(file, options)
%REPLAY Run every run of a session file again and compare with the file.
%   R = phasecont.session.replay(FILE) builds the model from the stored
%   source, checks its hash, runs each stored run again and compares the
%   result with the stored one. R is a 1-by-K struct array, one entry per
%   run, with the fields
%     Kind         "continuation", "simulate" or "equilibrium"
%     Type         "EP", "LC", "IVP" and so on; "" for an equilibrium
%     Rows         struct with Stored and New, the number of points of each
%                  branch in the file and in the rerun
%     Identical    true when every number matches bit for bit and the
%                  status agrees
%     MaxRelDiff   largest |a-b|/max(1,|b|) over the finite numbers, with a
%                  the new value and b the stored one
%     LabelsMatch  the special points have the same labels and indices
%     StopMatch    every branch stopped for the same reason
%     Pass         the shapes, the NaN and Inf positions, the point counts,
%                  the labels and the stop reasons agree and MaxRelDiff is
%                  at most Tol
%
%   R = phasecont.session.replay(FILE, Tol = 1e-8, Evaluator = "jit")
%   sets the tolerance (default 1e-8) and builds the model with another
%   evaluator than the recorded one. A different platform or evaluator can
%   change the last digits, and a different step sequence in an adaptive
%   run; Identical reports exact equality separately.
%
% The error phasecont:session:hashMismatch means the stored source does not
% match its stored SHA-256, and phasecont:session:badFile means the file is
% not a version 1 session file.
arguments
    file (1,1) string
    options.Tol (1,1) double {mustBePositive, mustBeFinite} = 1e-8
    options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
end
S = phasecont.internal.sessionCodec.readFile(file);
m = phasecont.internal.sessionCodec.buildModel(S, options.Evaluator);
empty = struct("Kind", "", "Type", "", "Rows", struct("Stored", [], "New", []), ...
    "Identical", false, "MaxRelDiff", Inf, "LabelsMatch", false, ...
    "StopMatch", false, "Pass", false);
r = repmat(empty, 1, numel(S.Records));
for k = 1:numel(S.Records)
    rec = S.Records{k};
    if rec.Kind == "equilibrium"
        one = replayEquilibrium(m, rec);
    else
        out = phasecont.internal.runLoop(m, rec.Problem, rec.U0, rec.P);
        c = phasecont.internal.sessionCodec.compareRun(rec, out.Points, ...
            out.Specials, out.Json);
        one = struct("Kind", rec.Kind, "Type", rec.Type, ...
            "Rows", struct("Stored", c.Stored, "New", c.New), ...
            "Identical", c.Same && c.Identical, "MaxRelDiff", c.MaxRelDiff, ...
            "LabelsMatch", c.Labels, "StopMatch", c.Stops, ...
            "Pass", c.Same && c.Structure && c.Labels && c.Stops ...
                && c.MaxRelDiff <= options.Tol);
    end
    if rec.Kind == "equilibrium"
        one.Pass = one.Pass && one.MaxRelDiff <= options.Tol;
    end
    r(k) = one;
end
end

function one = replayEquilibrium(m, rec)
% Run model_equilibrium again with the stored guess and options.
[u, ev] = pckern("model_equilibrium", m.Id, rec.U0, rec.P, char(rec.Problem));
stored = [rec.Result.U(:); real(rec.Result.Eigenvalues(:)); imag(rec.Result.Eigenvalues(:))];
fresh = [u(:); real(ev(:)); imag(ev(:))];
cmp = phasecont.internal.sessionCodec.compareArrays(fresh, stored);
one = struct("Kind", "equilibrium", "Type", "", ...
    "Rows", struct("Stored", 1, "New", 1), ...
    "Identical", cmp.Same && cmp.Identical, "MaxRelDiff", cmp.MaxRelDiff, ...
    "LabelsMatch", true, "StopMatch", true, "Pass", cmp.Same);
end
