function out = fig_10_3_4(options)
%FIG_10_3_4 Strogatz, 3rd edition, figure 10.3.4, from logistic_map.ode.
%   OUT = FIG_10_3_4() draws the fixed points and the 2-cycle against r;
%   FIG_10_3_4(Plot = false) draws nothing. OUT.Transcritical holds x = 0
%   and the branch through its branch point, OUT.Doubling the fixed points
%   from r1 and the 2-cycle that leaves their period doubling.
arguments
    options.Plot (1,1) logical = true
end
v = struct("r0", 0.5, "x0", 0, "bounds", [0.5 2], "r1", 2.5, "x1", 0.6, ...
    "bounds1", [2.5 3.6], "hmax", 0.05);
m = strogatzModel("logistic_map.ode");
t = phasecont.continuation(m, v.x0, Params = struct("r", v.r0), Free = "r", ...
    Bounds = v.bounds, Switch = "BP1");
d = phasecont.continuation(m, v.x1, Params = struct("r", v.r1), Free = "r", ...
    Bounds = v.bounds1, Hmax = v.hmax, Switch = "PD1");
out = struct("Values", v, "Model", m, "Transcritical", t, "Doubling", d);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
t.plot(Parent = ax);
d.plot(Parent = ax);
xlabel(ax, "r");
ylabel(ax, "x");
end
