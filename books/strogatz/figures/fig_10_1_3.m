function out = fig_10_1_3(options)
%FIG_10_1_3 Strogatz, 3rd edition, figure 10.1.3, from cosine_map.ode.
%   OUT = FIG_10_1_3() draws the cobweb; FIG_10_1_3(Plot = false) draws
%   nothing and gets the same iterates from simulate. OUT.Trajectory holds
%   the iterates and OUT.Point the fixed point.
arguments
    options.Plot (1,1) logical = true
end
v = struct("x0", 1, "N", 20, "guess", 0.7);
m = strogatzModel("cosine_map.ode");
if options.Plot
    ax = axes(figure);
    [~, tr] = phasecont.cobweb(m, v.x0, v.N, Parent = ax);
    xlabel(ax, "x_n");
    ylabel(ax, "x_{n+1}");
else
    tr = phasecont.simulate(m, v.x0, [0 v.N]);
end
pt = phasecont.equilibrium(m, v.guess);
out = struct("Values", v, "Model", m, "Trajectory", tr, "Point", pt);
end
