function [xs, ys] = cobwebGraph(m, p, lo, hi, count)
%COBWEBGRAPH Samples of the graph y = F(x) of a map with one variable.
%   [XS, YS] = phasecont.internal.cobwebGraph(M, P, LO, HI, COUNT) evaluates
%   the map M with parameter values P at COUNT points from LO to HI through
%   the batch rhs, in one call. The engine refuses a batch in which some
%   value is not finite, so on that refusal each point is evaluated on its
%   own, and a point whose value is not finite becomes NaN. The line then
%   has gaps there. XS and YS are row vectors.
arguments
    m (1,1) phasecont.Model
    p (:,1) double
    lo (1,1) double
    hi (1,1) double
    count (1,1) double
end
xs = linspace(lo, hi, count);
try
    ys = m.rhs(xs, p);
catch err
    if err.identifier ~= "phasecont:engine:numeric"
        rethrow(err);
    end
    ys = zeros(1, count);
    for k = 1:count
        try
            ys(k) = m.rhs(xs(k), p);
        catch err
            if err.identifier ~= "phasecont:engine:numeric"
                rethrow(err);
            end
            ys(k) = NaN;
        end
    end
end
ys(~isfinite(ys)) = NaN;
end
