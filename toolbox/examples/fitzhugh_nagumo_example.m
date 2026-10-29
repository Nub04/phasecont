function br = fitzhugh_nagumo_example(options)
%FITZHUGH_NAGUMO_EXAMPLE Hopf points of the FitzHugh-Nagumo model.
%   BR = fitzhugh_nagumo_example() loads fitzhugh_nagumo.ode from the folder
%   of this file, finds the equilibrium at the model's drive I, continues I
%   in both directions and prints the special points (label, I, v, omega).
%   BR is the 1-by-2 array of phasecont.Branch objects, one per direction.
%
%   Name-value options
%     Plot       true draws both branches (default true)
%     Evaluator  "jit", "fallback" or "aot" (default "jit")
%
%   The toolbox and the folder with the pckern binary must be on the path.
%   Reference: FitzHugh, Biophysical Journal 1 (1961) 445-466.
arguments
    options.Plot (1,1) logical = true
    options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["jit" "fallback" "aot"])} = "jit"
end

here = fileparts(mfilename("fullpath"));
m = phasecont.Model(fullfile(here, "fitzhugh_nagumo.ode"), Evaluator = options.Evaluator);
eq = phasecont.equilibrium(m, m.Init);

% Continue the drive I upward and downward from the equilibrium. Every
% setting is given here: the drive runs in [-1 3], at most 2000 steps per
% direction, and the run reports limit points, branch points and Hopf points.
branches = phasecont.Branch.empty(1, 0);
for dir = [1 -1]
    branches(end+1) = phasecont.continuation(m, eq, Free = "I", ...
        Direction = dir, Bounds = [-1 3], MaxSteps = 2000, ...
        Detect = ["LP" "BP" "H"], Plot = options.Plot, Var = "v"); %#ok<AGROW>
    if options.Plot && dir == 1
        hold(gca, "on")
    end
end
if options.Plot
    hold(gca, "off")
end

% Table of the special points.
n = 0;
label = strings(0, 1);
drive = zeros(0, 1);
v = zeros(0, 1);
omega = zeros(0, 1);
for k = 1:numel(branches)
    sp = branches(k).points();
    for j = 1:numel(sp)
        n = n + 1;
        label(n, 1) = sp(j).Label;
        drive(n, 1) = sp(j).P(branches(k).FreeIndex);
        v(n, 1) = sp(j).U(1);
        omega(n, 1) = sp(j).Data.omega;
    end
end
disp(table(label, drive, v, omega, VariableNames = ["Label" "I" "v" "omega"]))
br = branches;
end
