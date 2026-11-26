function out = fig_9_5_1(options)
%FIG_9_5_1 Strogatz, 3rd edition, figure 9.5.1, from lorenz.ode.
%   OUT = FIG_9_5_1() draws the equilibrium branch and the cycles that leave
%   its Hopf point; FIG_9_5_1(Plot = false) draws nothing. OUT.Branch is
%   the branch of C+ in r and OUT.Cycles the LC branch.
arguments
    options.Plot (1,1) logical = true
end
v = struct("sigma", 10, "b", 8/3, "r0", 20, "guess", [7 7 19], "epBounds", [20 30], ...
    "ntst", 40, "ncol", 4, "lcBounds", [13 25], "direction", 1, "maxSteps", 70);
m = strogatzModel("lorenz.ode");
pt = phasecont.equilibrium(m, v.guess, Params = struct("sigma", v.sigma, "r", v.r0, "b", v.b));
b = phasecont.continuation(m, pt, Free = "r", Bounds = v.epBounds);
hp = b.points("H");
lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "r", NTST = v.ntst, ...
    NCOL = v.ncol, Bounds = v.lcBounds, Direction = v.direction, MaxSteps = v.maxSteps);
out = struct("Values", v, "Model", m, "Branch", b, "Cycles", lc);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
b.plot(Var = "z", Parent = ax);
lc.plot(Var = "z", Parent = ax);
xlabel(ax, "r");
ylabel(ax, "z");
end
