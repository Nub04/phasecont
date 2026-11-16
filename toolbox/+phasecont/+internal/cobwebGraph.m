function [xs, ys] = cobwebGraph(m, p, lo, hi, count)
%COBWEBGRAPH Samples of the graph y = F(x) of a map with one variable.
%   [XS, YS] = phasecont.internal.cobwebGraph(M, P, LO, HI, COUNT) evaluates
%   the map M with parameter values P at COUNT points from LO to HI through
%   phasecont.internal.rhsGaps, so a point whose value is not finite
%   becomes NaN and the line has a gap there. XS and YS are row vectors.
arguments
    m (1,1) phasecont.Model
    p (:,1) double
    lo (1,1) double
    hi (1,1) double
    count (1,1) double
end
xs = linspace(lo, hi, count);
ys = phasecont.internal.rhsGaps(m, xs, p);
end
