function out = fig_8_4_3(options)
%FIG_8_4_3 Strogatz, 3rd edition, figure 8.4.3, from homoclinic.ode.
%   OUT = FIG_8_4_3() draws phase portraits below, at and above the
%   homoclinic value of mu; FIG_8_4_3(Plot = false) draws nothing.
%   OUT.Homoclinic is the HOM curve in (mu, c) and OUT.Homoclinic.Alpha(1, 1)
%   the homoclinic mu at c = Values.c.
arguments
    options.Plot (1,1) logical = true
end
v = struct("mu0", -1.5, "c", 1, "x0", [1 0], "epBounds", [-1.5 -0.5], "ntst", 80, ...
    "ncol", 4, "hmax", 0.05, "maxNewton", 15, "maxSteps", 1700, "periodT", 50, ...
    "homBounds", [0.9 1.1], "homHmax", 0.01, "bracket", [-0.95 -0.8], ...
    "muBelow", -0.92, "muAbove", -0.8, "starts", [1.1 0; 0.01 0.01], "T", 15);
m = strogatzModel("homoclinic.ode");
b = phasecont.continuation(m, v.x0, Params = struct("mu", v.mu0, "c", v.c), ...
    Free = "mu", Bounds = v.epBounds);
hp = b.points("H");
lc = phasecont.continuation(m, hp(1), Type = "LC", Free = "mu", NTST = v.ntst, ...
    NCOL = v.ncol, Hmax = v.hmax, MaxNewton = v.maxNewton, MaxSteps = v.maxSteps);
i = find(lc.Period > v.periodT, 1);
hom = phasecont.continuation(m, lc, Type = "HOM", Index = i, Free = ["mu" "c"], ...
    Bounds = v.homBounds, Hmax = v.homHmax, MaxNewton = v.maxNewton);
out = struct("Values", v, "Model", m, "Branch", b, "Cycles", lc, "Index", i, ...
    "Homoclinic", hom);
if ~options.Plot
    return
end
mus = [v.muBelow hom.Alpha(1, 1) v.muAbove];
tl = tiledlayout(figure, 1, 3);
for k = 1:3
    ax = nexttile(tl);
    hold(ax, "on");
    for s = 1:size(v.starts, 1)
        tr = phasecont.simulate(m, v.starts(s, :), [0 v.T], ...
            Params = struct("mu", mus(k), "c", v.c));
        tr.plot(Parent = ax);
    end
    title(ax, sprintf("mu = %.4f", mus(k)));
end
end
