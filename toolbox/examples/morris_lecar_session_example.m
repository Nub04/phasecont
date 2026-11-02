function r = morris_lecar_session_example(options)
%MORRIS_LECAR_SESSION_EXAMPLE Save a Morris-Lecar diagram and replay it.
%   R = morris_lecar_session_example() loads morris_lecar.ode from the folder
%   of this file and builds a small bifurcation diagram: the branch of rest
%   states in I with its Hopf point, the branch of limit cycles from that
%   point with its fold of cycles, the curve of folds in (I, phi), the curve of
%   Hopf points in (I, phi) and one trajectory. It saves the diagram to a
%   session file, reruns every run from that file and prints the result. R is
%   the struct array that phasecont.session.replay returns, one element per
%   run.
%
%   Name-value options
%     File       session file to write (default: morris_lecar.pcsession.json
%                in the temporary folder)
%     Open       true opens the saved file in the phase-plane app after the
%                replay (default false)
%     Evaluator  "jit", "fallback" or "aot" (default "jit")
%
%   The toolbox and the folder with the pckern binary must be on the path.
%   Reference: Morris and Lecar, Biophysical Journal 35 (1981) 193-213.
arguments
    options.File (1,1) string = fullfile(tempdir, "morris_lecar.pcsession.json")
    options.Open (1,1) logical = false
    options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["jit" "fallback" "aot"])} = "jit"
end

here = fileparts(mfilename("fullpath"));
m = phasecont.Model(fullfile(here, "morris_lecar.ode"), Evaluator = options.Evaluator);

% Branch of rest states in I, from I = 0 up to 150. The run reports the Hopf
% point.
rest = phasecont.equilibrium(m, [-60 0], Params = struct(I = 0, phi = 0.04));
ep = phasecont.continuation(m, rest, Free = "I", Bounds = [0 150], ...
    MaxSteps = 400, Hmax = 2, H0 = 0.5, Plot = false);
hopf = ep.points("H");
if isempty(hopf)
    error("morris_lecar_session_example:noHopf", ...
        "The branch of rest states has no Hopf point.");
end

% Limit cycles from the first Hopf point, on a coarse mesh (NTST 40, NCOL 4).
% The run reports the fold of cycles.
lc = phasecont.continuation(m, hopf(1), Type = "LC", Free = "I", ...
    NTST = 40, NCOL = 4, Bounds = [80 120], Hmax = 0.5, ...
    MaxNewton = 15, MaxSteps = 300, Plot = false);
fold = lc.points("LPC");
if isempty(fold)
    error("morris_lecar_session_example:noFold", ...
        "The branch of cycles has no fold of cycles.");
end

% The fold of cycles and the Hopf point, followed in I and phi.
lpc = phasecont.continuation(m, fold(1), Type = "LPC", Free = ["I" "phi"], ...
    Bounds = [0.02 0.06], Hmax = 0.5, MaxNewton = 15, MaxSteps = 100, Plot = false);
hcurve = phasecont.continuation(m, hopf(1), Type = "H", Free = ["I" "phi"], ...
    Bounds = [0.02 0.06], Hmax = 0.5, MaxNewton = 15, MaxSteps = 200, Plot = false);

% One trajectory at I = 100, which ends on the stable cycle.
tr = phasecont.simulate(m, [-20 0.1], [0 300], Params = struct(I = 100, phi = 0.04));

save(phasecont.Session(m, {ep, lc, lpc, hcurve, tr}), options.File);
fprintf("Saved %s\n", options.File);

% Rerun every run from the file. A run passes when its rows agree to 1e-8
% (relative, with a floor of 1) and its labels and stop reasons are the same.
r = phasecont.session.replay(options.File);
disp(table([r.Kind]', [r.Type]', [r.Pass]', [r.Identical]', [r.MaxRelDiff]', ...
    VariableNames = ["Kind" "Type" "Pass" "Identical" "MaxRelDiff"]))

if options.Open
    phasecont.app.PhasePlane(options.File);
end
end
