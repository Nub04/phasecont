function F = rhsGaps(m, S, p)
%RHSGAPS The right-hand side at many points, with NaN where it is not finite.
%   F = phasecont.internal.rhsGaps(M, S, P) evaluates the model M with
%   parameter values P at the columns of S through the batch rhs, in one
%   call. The engine refuses a batch in which some value is not finite, so
%   on that refusal each column is evaluated on its own, and a column whose
%   value is not finite becomes NaN. Every other value that is not finite
%   becomes NaN as well, so a plot has gaps there. F has one row per
%   variable and one column per column of S.
arguments
    m (1,1) phasecont.Model
    S (:,:) double
    p (:,1) double
end
try
    F = m.rhs(S, p);
catch err
    if err.identifier ~= "phasecont:engine:numeric"
        rethrow(err);
    end
    F = NaN(numel(m.Variables), size(S, 2));
    for k = 1:size(S, 2)
        try
            F(:, k) = m.rhs(S(:, k), p);
        catch err
            if err.identifier ~= "phasecont:engine:numeric"
                rethrow(err);
            end
        end
    end
end
F(~isfinite(F)) = NaN;
end
