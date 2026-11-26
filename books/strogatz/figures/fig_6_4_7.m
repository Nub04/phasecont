function out = fig_6_4_7(options)
%FIG_6_4_7 Strogatz, 3rd edition, figure 6.4.7, from rabbits_sheep.ode.
%   OUT = FIG_6_4_7() opens the PhasePlane app; FIG_6_4_7(Plot = false)
%   builds it with Visible = "off". OUT.App is the app, OUT.Points the four
%   equilibria and OUT.Trajectories the four orbits, in the order of
%   OUT.Values.points.
arguments
    options.Plot (1,1) logical = true
end
v = struct("points", [0.1 0.1; 0.1 1.9; 2.9 0.1; 0.9 1.1], ...
    "window", [0 3.5 0 2.5], "T", 20);
m = strogatzModel("rabbits_sheep.ode");
visible = "off";
if options.Plot
    visible = "on";
end
app = phasecont.app.PhasePlane(m, Visible = visible);
app.setWindow(v.window);
app.setT(v.T);
app.showNullclines(true);
n = size(v.points, 1);
points = cell(1, n);
trajectories = cell(1, n);
for k = 1:n
    points{k} = app.findEquilibrium(v.points(k, 1), v.points(k, 2));
    trajectories{k} = app.addTrajectory(v.points(k, 1), v.points(k, 2));
end
out = struct("Values", v, "Model", m, "App", app);
out.Points = points;
out.Trajectories = trajectories;
end
