function out = fig_2_2_6(options)
%FIG_2_2_6 Strogatz, 3rd edition, figure 2.2.6, from cosine_flow.ode.
%   OUT = FIG_2_2_6() draws the figure; FIG_2_2_6(Plot = false) draws
%   nothing. OUT.Values holds every value the PhaseCont calls use, and
%   OUT.Point the equilibrium.
arguments
    options.Plot (1,1) logical = true
end
v = struct("x0", 0.7, "window", [-2 2]);
m = strogatzModel("cosine_flow.ode");
pt = phasecont.equilibrium(m, v.x0);
out = struct("Values", v, "Model", m, "Point", pt);
if ~options.Plot
    return
end
xs = linspace(v.window(1), v.window(2), 400);
ax = axes(figure);
plot(ax, xs, xs, xs, xs - m.rhs(xs));   % x - f(x) = cos x
hold(ax, "on");
plot(ax, pt.U, pt.U, "ko", MarkerFaceColor = "w");
xlabel(ax, "x");
legend(ax, "y = x", "y = cos x", "x*", Location = "best");
end
