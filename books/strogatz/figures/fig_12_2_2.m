function out = fig_12_2_2(options)
%FIG_12_2_2 Strogatz, 3rd edition, figure 12.2.2, from henon_map.ode.
%   OUT = FIG_12_2_2() draws a grid of start points and its image after
%   Values.steps iterates; FIG_12_2_2(Plot = false) draws nothing.
%   OUT.Start holds the start points and OUT.Result the ensemble result.
arguments
    options.Plot (1,1) logical = true
end
v = struct("a", 1.4, "b", 0.3, "box", [-1.5 1.5 -0.4 0.4], "n", 100, "steps", 1);
m = strogatzModel("henon_map.ode");
[X, Y] = meshgrid(linspace(v.box(1), v.box(2), v.n), linspace(v.box(3), v.box(4), v.n));
X0 = [X(:)'; Y(:)'];
r = phasecont.ensemble(m, X0, [v.a; v.b], [0 v.steps], Precision = "double", Backend = "cpu");
out = struct("Values", v, "Model", m, "Start", X0, "Result", r);
if ~options.Plot
    return
end
ax = axes(figure);
hold(ax, "on");
plot(ax, X0(1, :), X0(2, :), ".", MarkerSize = 2);
plot(ax, r.Final(1, :), r.Final(2, :), ".", MarkerSize = 2);
xlabel(ax, "x");
ylabel(ax, "y");
end
