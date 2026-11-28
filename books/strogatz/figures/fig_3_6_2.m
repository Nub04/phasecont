function out = fig_3_6_2(options)
%FIG_3_6_2 Strogatz, 3rd edition, figure 3.6.2, from imperfect.ode.
%   OUT = FIG_3_6_2() draws the fold curve in the (h, r) plane;
%   FIG_3_6_2(Plot = false) draws nothing. OUT.Branch is the equilibrium
%   branch in h at r = r0 and OUT.Curve the fold curve from its first fold.
arguments
    options.Plot (1,1) logical = true
end
v = struct("h0", 0, "r0", 1, "x0", -1, "hBounds", [-1 1], "rBounds", [-0.5 1.5], ...
    "direction", 1);
m = strogatzModel("imperfect.ode");
b = phasecont.continuation(m, v.x0, Params = struct("h", v.h0, "r", v.r0), ...
    Free = "h", Bounds = v.hBounds);
lp = b.points("LP");
c = phasecont.continuation(m, lp(1), Type = "LP", Free = ["h" "r"], ...
    Bounds = v.rBounds, Direction = v.direction);
out = struct("Values", v, "Model", m, "Branch", b, "Curve", c);
if ~options.Plot
    return
end
ax = axes(figure);
c.plot(Parent = ax);
xlabel(ax, "h");
ylabel(ax, "r");
end
