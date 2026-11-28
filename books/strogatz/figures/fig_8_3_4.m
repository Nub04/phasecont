function out = fig_8_3_4(options)
%FIG_8_3_4 Strogatz, 3rd edition, figure 8.3.4, from chemical_oscillator.ode.
%   OUT = FIG_8_3_4() draws the Hopf curve in the (a, b) plane;
%   FIG_8_3_4(Plot = false) draws nothing. OUT.Branch is the equilibrium
%   branch in b and OUT.Curves the Hopf curve runs from its Hopf point.
arguments
    options.Plot (1,1) logical = true
end
v = struct("a0", 10, "b0", 2, "x0", [2 5], "bBounds", [2 5], "curveBounds", [0.5 10], ...
    "directions", [1 -1]);
m = strogatzModel("chemical_oscillator.ode");
b = phasecont.continuation(m, v.x0, Params = struct("a", v.a0, "b", v.b0), ...
    Free = "b", Bounds = v.bBounds);
hp = b.points("H");
curves = phasecont.Branch.empty(1, 0);
for d = v.directions
    curves(end + 1) = phasecont.continuation(m, hp(1), Type = "H", Free = ["a" "b"], ...
        Bounds = v.curveBounds, Direction = d); %#ok<AGROW>
end
out = struct("Values", v, "Model", m, "Branch", b, "Curves", curves);
if ~options.Plot
    return
end
ax = axes(figure);
curves.plot(Parent = ax);
xlabel(ax, "a");
ylabel(ax, "b");
end
