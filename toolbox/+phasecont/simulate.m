function tr = simulate(m, x0, tspan, options)
%SIMULATE Integrate a model from an initial state.
%   TR = phasecont.simulate(M, X0, [T0 T1]) integrates the model M from the
%   state X0 at time T0 to time T1 and returns a phasecont.Trajectory. X0 is
%   a state vector or a phasecont.Point. T1 < T0 integrates backward. The
%   call blocks while the engine runs and polls it about every 50 ms. Ctrl+C
%   cancels the run and returns nothing.
%
%   The solver is the Dormand-Prince 5(4) pair with step size control. It
%   returns one row per accepted step, so a plot of TR.X is as smooth as the
%   steps are short; Hmax caps the step length. A model that uses the time t
%   is integrated with t running from T0 to T1.
%
%   TR = phasecont.simulate(M, X0, [0 50], Params = struct("I", 80)) changes
%   the named parameters, or sets them all when Params is a vector.
%
%   Name-value options
%     Params    struct or vector that overrides the parameters of X0 (or of
%               the model when X0 is a vector)
%     RelTol    relative error tolerance, positive (default 1e-8)
%     AbsTol    absolute error tolerance, positive (default 1e-10)
%     MaxSteps  most accepted steps, a positive integer (default 100000)
%     Hmax      largest step length, positive; Inf means no cap (default)
%
%   TR.Stop is "End" when the run reached T1. When it stops for another
%   reason ("MaxSteps", "StepTooSmall" or "Error", for example because the
%   solution blows up), simulate warns with the ID
%   phasecont:simulate:stoppedEarly and TR holds the steps up to then.
arguments
    m (1,1) phasecont.Model
    x0
    tspan
    options.Params = []
    options.RelTol = []
    options.AbsTol = []
    options.MaxSteps = []
    options.Hmax = []
end
import phasecont.internal.num2json

if ~(isnumeric(tspan) && isreal(tspan) && numel(tspan) == 2 ...
        && all(isfinite(tspan)) && tspan(1) ~= tspan(2))
    error("phasecont:model:badOption", ...
        "TSPAN must be [t0 t1], two different finite real numbers.");
end
tspan = double(tspan(:)).';
[u0, p] = startValues(m, x0, options.Params);

parts = strings(1, 0);
parts(end+1) = numberOption("rtol", options.RelTol, "RelTol", false);
parts(end+1) = numberOption("atol", options.AbsTol, "AbsTol", false);
parts(end+1) = numberOption("max_steps", options.MaxSteps, "MaxSteps", true);
if ~isempty(options.Hmax)
    h = options.Hmax;
    if ~(isnumeric(h) && isreal(h) && isscalar(h) && h > 0 && ~isnan(h))
        error("phasecont:model:badOption", "Hmax must be positive or Inf.");
    end
    % The engine default is no cap, so Inf adds no key.
    if isfinite(h)
        parts(end+1) = """hmax"":" + num2json(double(h));
    end
end
parts = parts(strlength(parts) > 0);
problem = "{""type"":""IVP"",""tspan"":[" + num2json(tspan(1)) + "," ...
    + num2json(tspan(2)) + "],""options"":{" + strjoin(parts, ",") + "}}";

n = numel(m.Variables);
out = phasecont.internal.runLoop(m, problem, u0, p);
st = out.Status;
rows = out.Points{1};
record = phasecont.internal.RunRecord(struct( ...
    "Kind", "simulate", "Problem", problem, "U0", u0(:), "P", p(:), ...
    "ModelSha", m.SourceSha256, "Type", "IVP", "N", n, "Json", out.Json, ...
    "Points", {out.Points}, "Specials", {out.Specials}, "Stride", out.Stride));
stop = "";
if isfield(st.branches, "stop") && ~isempty(st.branches(1).stop)
    stop = string(st.branches(1).stop);
end
warn = strings(0, 1);
if isfield(st.branches, "warnings") && ~isempty(st.branches(1).warnings)
    warn = string(st.branches(1).warnings);
end
tr = phasecont.Trajectory(struct( ...
    "T", rows(1, :).', "X", rows(2:end, :).', "X0", u0(:), "TSpan", tspan, ...
    "P", p(:), "Stop", stop, "Warnings", warn(:), "Model", m));
tr.attachRecord(record);
if stop ~= "End"
    warning("phasecont:simulate:stoppedEarly", ...
        "The run stopped at t = %.6g with reason ""%s"" before t = %.6g.", ...
        tr.T(end), stop, tspan(2));
end
end

function [u0, p] = startValues(m, start, params)
% Start state and parameters from a Point or a state vector.
n = numel(m.Variables);
np = numel(m.Parameters);
if isa(start, "phasecont.Point")
    if ~isscalar(start)
        error("phasecont:model:badStart", "X0 must be one Point.");
    end
    if ~isempty(start.Model) && start.Model ~= m
        error("phasecont:model:badStart", ...
            "The start point belongs to a different model.");
    end
    u0 = start.U(:);
    p = start.P(:);
    if numel(u0) ~= n || numel(p) ~= np
        error("phasecont:model:badStart", ...
            "The start point needs %d states and %d parameters.", n, np);
    end
    if isstruct(params)
        v = m.paramVector(params);
        idx = find(ismember(m.Parameters, string(fieldnames(params))));
        p(idx) = v(idx);
    elseif ~isempty(params)
        p = m.paramVector(params);
    end
elseif isnumeric(start) && isreal(start) && isvector(start)
    if numel(start) ~= n
        error("phasecont:model:badStart", ...
            "The start vector needs %d states, got %d.", n, numel(start));
    end
    if ~all(isfinite(start))
        error("phasecont:model:badStart", "The start vector must be finite.");
    end
    u0 = double(start(:));
    if isempty(params) && ~isstruct(params)
        p = m.ParameterValues;
    else
        p = m.paramVector(params);
    end
else
    error("phasecont:model:badStart", ...
        "X0 must be a phasecont.Point or a real state vector.");
end
end

function s = numberOption(key, value, name, isCount)
% One "key":value entry, or "" when the option is not set.
s = "";
if isempty(value)
    return
end
if ~(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) ...
        && value > 0)
    error("phasecont:model:badOption", ...
        "%s must be a positive finite number.", name);
end
if isCount && value ~= round(value)
    error("phasecont:model:badOption", "%s must be an integer.", name);
end
s = """" + key + """:" + phasecont.internal.num2json(double(value));
end
