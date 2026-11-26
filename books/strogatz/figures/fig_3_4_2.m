function out = fig_3_4_2(options)
%FIG_3_4_2 Strogatz, 3rd edition, figure 3.4.2, from pitchfork.ode.
%   OUT = FIG_3_4_2() draws the figure; FIG_3_4_2(Plot = false) draws
%   nothing. OUT.Branches holds the branch x = 0, the branch that leaves
%   its branch point, and a run along r = x^2 from x = -1.
arguments
    options.Plot (1,1) logical = true
end
v = struct("r0", -1, "x0", 0, "bounds", [-1 1], "hmax", 0.05, ...
    "x1", -1, "r1", 1, "bounds1", [-1 1.5], "direction1", -1);
m = strogatzModel("pitchfork.ode");
brs = phasecont.continuation(m, v.x0, Params = struct("r", v.r0), Free = "r", ...
    Bounds = v.bounds, Hmax = v.hmax, Switch = "BP1");
more = phasecont.continuation(m, v.x1, Params = struct("r", v.r1), Free = "r", ...
    Bounds = v.bounds1, Hmax = v.hmax, Direction = v.direction1);
brs = [brs more];
out = struct("Values", v, "Model", m, "Branches", brs);
if ~options.Plot
    return
end
ax = axes(figure);
brs.plot(Parent = ax);
xlabel(ax, "r");
ylabel(ax, "x");
end
