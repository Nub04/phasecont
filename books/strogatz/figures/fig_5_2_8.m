function out = fig_5_2_8(options)
%FIG_5_2_8 Strogatz, 3rd edition, figure 5.2.8, from damped_oscillator.ode.
%   OUT = FIG_5_2_8() draws the path of the origin in the (tau, Delta)
%   plane as c runs over Values.bounds; FIG_5_2_8(Plot = false) draws
%   nothing. OUT.Branch is the branch of the origin in c.
arguments
    options.Plot (1,1) logical = true
end
v = struct("k", 1, "c0", -3, "x0", [0 0], "bounds", [-3 3], "direction", 1);
m = strogatzModel("damped_oscillator.ode");
br = phasecont.continuation(m, v.x0, Params = struct("k", v.k, "c", v.c0), ...
    Free = "c", Bounds = v.bounds, Direction = v.direction);
out = struct("Values", v, "Model", m, "Branch", br);
if ~options.Plot
    return
end
ax = axes(figure);
br.tdplot(Parent = ax);
end
