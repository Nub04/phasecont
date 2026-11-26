function out = fig_9_3_2(options)
%FIG_9_3_2 Strogatz, 3rd edition, figure 9.3.2, from lorenz.ode.
%   OUT = FIG_9_3_2() draws the figure; FIG_9_3_2(Plot = false) draws
%   nothing. OUT.Values holds every value the PhaseCont calls use, and
%   OUT.Trajectory the orbit.
arguments
    options.Plot (1,1) logical = true
end
v = struct("sigma", 10, "r", 28, "b", 8/3, "x0", [0 1 0], "tspan", [0 50], ...
    "relTol", 1e-10, "absTol", 1e-12);
m = strogatzModel("lorenz.ode");
tr = phasecont.simulate(m, v.x0, v.tspan, ...
    Params = struct("sigma", v.sigma, "r", v.r, "b", v.b), ...
    RelTol = v.relTol, AbsTol = v.absTol);
out = struct("Values", v, "Model", m, "Trajectory", tr);
if ~options.Plot
    return
end
ax = axes(figure);
tr.plot(Var = ["x" "z"], Parent = ax);
xlabel(ax, "x");
ylabel(ax, "z");
end
