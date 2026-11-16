function [h, tr] = cobweb(m, x0, N, options)
%COBWEB Draw the cobweb plot of a map with one variable.
%   H = phasecont.cobweb(M, X0, N) iterates the map M N times from X0 and
%   draws its cobweb in the current axes: the graph y = F(x), the diagonal
%   y = x and the staircase (x_0, x_0), (x_0, x_1), (x_1, x_1), ...,
%   (x_N, x_N), with a marker at (x_0, x_0). The fixed points of F lie where
%   the graph crosses the diagonal. X0 is a number or a phasecont.Point.
%
%   [H, TR] = phasecont.cobweb(...) also returns the phasecont.Trajectory of
%   the run, which is phasecont.simulate(M, X0, [0 N], Params = ...).
%
%   Options:
%     Params  parameter values, a struct or a vector, as simulate takes them.
%     Parent  axes to draw into. Without it the plot uses newplot, so hold on
%             is respected. Either way the hold state of the axes is the
%             same after the call as before it.
%
%   H is a struct with the fields Graph, Diagonal, Steps and Start, the
%   lines with the Tags "pc-cobweb-graph", "pc-cobweb-diagonal",
%   "pc-cobweb" and "pc-start". The graph and the diagonal span the
%   smallest to the largest finite iterate, widened by a tenth of that width
%   on each side (by a tenth of max(1, |x|) when every iterate is the same),
%   with 401 samples of F at the parameter values of the run. The staircase
%   takes the next color of the color order of the axes, and the next plot
%   takes the color after it.
%
%   M must be a map with one variable that does not use t, else the call
%   raises phasecont:cobweb:badModel. N must be a whole number from 1 to
%   1000000, else the call raises phasecont:model:badOption. A run that
%   stops early warns phasecont:simulate:stoppedEarly and draws the
%   iterates it kept.
%
%   See also phasecont.simulate, phasecont.app.PhasePlane.
arguments
    m (1,1) phasecont.Model
    x0
    N
    options.Params = []
    options.Parent = []
end
if ~m.IsMap
    error("phasecont:cobweb:badModel", ...
        "A cobweb plot needs a map, and this model is not a map.");
end
n = numel(m.Variables);
if n ~= 1
    error("phasecont:cobweb:badModel", ...
        "A cobweb plot needs a map with one variable, and this map has %d.", n);
end
if ~m.Autonomous
    error("phasecont:cobweb:badModel", ...
        "A cobweb plot needs a map that does not use t.");
end
if ~(isnumeric(N) && isreal(N) && isscalar(N) && isfinite(N) ...
        && N == round(N) && N >= 1 && N <= 1e6)
    error("phasecont:model:badOption", ...
        "N must be a whole number from 1 to 1000000.");
end
tr = phasecont.simulate(m, x0, [0 double(N)], Params = options.Params);

x = tr.X(:, 1);
kept = x(isfinite(x));
lo = min(kept);
hi = max(kept);
w = hi - lo;
if w == 0
    w = max(1, abs(lo));
end
[gx, gy] = phasecont.internal.cobwebGraph(m, tr.P, lo - w / 10, hi + w / 10, 401);
[sx, sy] = phasecont.internal.cobwebSteps(x);

if isempty(options.Parent)
    ax = newplot;
else
    ax = options.Parent;
end
wasHeld = ishold(ax);
hold(ax, "on");
restore = onCleanup(@() restoreHold(ax, wasHeld));
[color, after] = phasecont.internal.nextColor(ax);
h = struct();
h.Graph = plot(ax, gx, gy, "-", Color = [0 0 0], LineWidth = 1.2, ...
    Tag = "pc-cobweb-graph");
h.Diagonal = plot(ax, gx([1 end]), gx([1 end]), "--", ...
    Color = [0.55 0.55 0.55], Tag = "pc-cobweb-diagonal");
h.Steps = plot(ax, sx, sy, "-", Color = color, Tag = "pc-cobweb");
h.Start = plot(ax, x(1), x(1), "o", Color = color, MarkerFaceColor = color, ...
    MarkerSize = 5, Tag = "pc-start");
ax.ColorOrderIndex = after;
name = m.Variables(1);
xlabel(ax, name);
ylabel(ax, name + "(n+1)");
clear restore
if nargout == 0
    clear h
end
end

function restoreHold(ax, wasHeld)
% Put the hold state of AX back as it was before the call.
if isvalid(ax) && ~wasHeld
    hold(ax, "off");
end
end
