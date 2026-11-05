function s = gpuSelfTest(options)
%GPUSELFTEST Check one ensemble backend against closed-form solutions.
%   S = phasecont.gpuSelfTest(Precision = "single") runs small ensembles on
%   the chosen backend and device and compares them with exact solutions.
%   It answers one question: does this backend, on this machine, give the
%   right numbers? It does not measure speed.
%
%   Options
%     Precision   "single" or "double". Required; it has no default.
%     Backend     "auto" (default), "cpu", "cuda" or "opencl"
%     Device      index within the backend, counting from 0 (default 0)
%
%   Checks
%     rotation    fixed-step Runge-Kutta 4 on x' = y, y' = -x from 64 start
%                 states, with the states recorded at four times
%     decay       Dormand-Prince 5(4) on x' = -a x with one a per member
%     repeat      the rotation run a second time gives identical numbers
%     subset      the first 8 members run alone give the same numbers as in
%                 the full ensemble
%   The first two compare with the exact solution. The tolerances are the
%   test's own thresholds, not a general accuracy claim: in single precision
%   5e-4 for the rotation (1000 steps of about 6e-8 rounding each, with
%   margin) and 1e-3 for the decay, in double precision 1e-8 and 1e-7.
%
%   S has the fields Backend, Device, Driver, Precision, SinglePrecision
%   (what ran, as in EnsembleResult.Label), MaxError (the larger of the two
%   exact-solution errors), Pass and Checks (a table with one row per
%   check). The command line tool does the full tiers: pc-cli gpu selftest.
%
%   See also phasecont.gpuDevices, phasecont.ensemble.
arguments
    options.Backend (1,1) string = "auto"
    options.Device = []
    options.Precision (1,1) string = ""
end

precision = lower(options.Precision);
if strlength(precision) == 0
    error("phasecont:ensemble:missingPrecision", ...
        "Precision is required: ""single"" or ""double"". It has no default.");
end
if ~any(precision == ["single" "double"])
    error("phasecont:ensemble:badOption", ...
        "Precision must be ""single"" or ""double"", got ""%s"".", options.Precision);
end
if precision == "single"
    tolRotation = 5e-4;
    tolDecay = 1e-3;
    rtol = 1e-5;
    atol = 1e-8;
else
    tolRotation = 1e-8;
    tolDecay = 1e-7;
    rtol = 1e-9;
    atol = 1e-12;
end

target = {"Backend", options.Backend, "Precision", precision};
if ~isempty(options.Device)
    target = [target, {"Device", options.Device}];
end

rotation = phasecont.Model.fromSource(join([ ...
    "x'=y", "y'=-x", "init x=1,y=0", "done", ""], newline));
cleanRotation = onCleanup(@() delete(rotation)); %#ok<NASGU>
decay = phasecont.Model.fromSource(join([ ...
    "par a=1", "x'=-a*x", "init x=1", "done", ""], newline));
cleanDecay = onCleanup(@() delete(decay)); %#ok<NASGU>

% Rotation: exact solution x(t) = x0 cos t + y0 sin t, y(t) = -x0 sin t + y0 cos t.
members = 64;
radius = linspace(0.5, 2, members);
angle = 2 * pi * (0:members - 1) / members;
x0 = [radius .* cos(angle); radius .* sin(angle)];
tspan = [0 2.5 5 7.5 10];
r = phasecont.ensemble(rotation, x0, [], tspan, target{:}, Dt = 0.01);
t = reshape(r.Times, 1, [], 1);
a = reshape(x0(1, :), 1, 1, []);
b = reshape(x0(2, :), 1, 1, []);
exact = [a .* cos(t) + b .* sin(t); -a .* sin(t) + b .* cos(t)];
errRotation = max(abs(r.States - exact), [], "all");

% Decay with one rate per member: x(t) = exp(-a t).
rates = linspace(0.5, 2, members);
d = phasecont.ensemble(decay, ones(1, members), rates, [0 1 2.5 5], ...
    target{:}, Rtol = rtol, Atol = atol);
td = reshape(d.Times, 1, [], 1);
exactDecay = exp(-reshape(rates, 1, 1, []) .* td);
errDecay = max(abs(d.States - exactDecay), [], "all");

% Same input, same output; and the answer for a member does not depend on the others.
again = phasecont.ensemble(rotation, x0, [], tspan, target{:}, Dt = 0.01);
subset = phasecont.ensemble(rotation, x0(:, 1:8), [], tspan, target{:}, Dt = 0.01);
diffRepeat = max(abs(again.States - r.States), [], "all");
diffSubset = max(abs(subset.States - r.States(:, :, 1:8)), [], "all");

Name = ["rotation"; "decay"; "repeat"; "subset"];
MaxError = [errRotation; errDecay; diffRepeat; diffSubset];
Tolerance = [tolRotation; tolDecay; 0; 0];
Pass = MaxError <= Tolerance & all(isfinite(MaxError));
Checks = table(Name, MaxError, Tolerance, Pass);

s = struct( ...
    "Backend", r.Label.Backend, ...
    "Device", r.Label.Device, ...
    "Driver", r.Label.Driver, ...
    "Precision", r.Label.Precision, ...
    "SinglePrecision", r.Label.SinglePrecision, ...
    "MaxError", max(errRotation, errDecay), ...
    "Pass", all(Pass), ...
    "Checks", Checks);
end
