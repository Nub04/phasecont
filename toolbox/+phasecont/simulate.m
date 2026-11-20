function tr = simulate(m, x0, tspan, options)
%SIMULATE Integrate a model, or iterate a map, from an initial state.
%   TR = phasecont.simulate(M, X0, [T0 T1]) integrates the model M from the
%   state X0 at time T0 to time T1 and returns a phasecont.Trajectory. X0 is
%   a state vector or a phasecont.Point. T1 < T0 integrates backward. The
%   call blocks while the engine runs and polls it (first after 1 ms, then at
%   growing intervals up to 50 ms). Ctrl+C cancels the run and returns
%   nothing.
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
%   A map, written x(t+1)=... or with @ meth=discrete (M.IsMap is true), is
%   iterated: TR = phasecont.simulate(M, X0, [N0 N1]) sets x(N0) = X0 and
%   x(k+1) = F(x(k)) for k = N0, ..., N1-1, with t = k inside F. N0 and N1
%   are integers with N0 < N1 and magnitude at most flintmax, and one run
%   makes at most 1000000 iterates. TR.T is (N0:N1)', with one row of TR.X
%   per iterate. RelTol, AbsTol, MaxSteps and Hmax do not apply to a map,
%   and each one passed raises phasecont:simulate:mapOption.
%
%   simulate reads none of the model's @ options (meth, dt, total, bound
%   and the others), so they do not change the result. The one exception
%   is a meth value that starts with d, such as meth=discrete: it makes the
%   model a map, as in XPPAUT. The first simulate call that runs on a model
%   that sets any other @ option warns with the ID
%   phasecont:simulate:ignoredOption and names them all. While that warning
%   is turned off, the one notice is kept for a later call.
%
%   TR.Stop is "End" when the run reached T1. When it stops for another
%   reason ("MaxSteps", "StepTooSmall" or "Error", for example because the
%   solution blows up), simulate warns with the ID
%   phasecont:simulate:stoppedEarly and TR holds the steps up to then. A
%   map run stops with "Error" at the first iterate that is NaN or
%   infinite and keeps the iterates before it.
%
%   A model file with an only line, such as only x,t,w, gives TR.Output:
%   a table of those columns in that order, where t is TR.T, a state is
%   its column of TR.X and an aux quantity holds its value at each row. A
%   non-finite aux value is kept as computed and does not stop the run.
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

if m.IsMap
    % SPEC 8u: TSPAN first, then the options that do not apply to a map.
    tspan = mapSpan(tspan);
    refuseMapOptions(options);
    [u0, p] = startValues(m, x0, options.Params);
    type = "ITER";
    problem = "{""type"":""ITER"",""tspan"":[" + num2json(tspan(1)) + "," ...
        + num2json(tspan(2)) + "]" + auxKey(m) + "}";
else
    if ~(isnumeric(tspan) && isreal(tspan) && numel(tspan) == 2 ...
            && all(isfinite(tspan)) && tspan(1) ~= tspan(2))
        error("phasecont:model:badOption", ...
            "TSPAN must be [t0 t1], two different finite real numbers.");
    end
    tspan = double(tspan(:)).';
    [u0, p] = startValues(m, x0, options.Params);
    type = "IVP";
    problem = ivpProblem(tspan, options, auxKey(m));
end

n = numel(m.Variables);
out = phasecont.internal.runLoop(m, problem, u0, p);
% After the run, so that a run the engine refuses gets no warning that the
% options were ignored.
warnIgnoredOptions(m);
st = out.Status;
rows = out.Points{1};
record = phasecont.internal.RunRecord(struct( ...
    "Kind", "simulate", "Problem", problem, "U0", u0(:), "P", p(:), ...
    "ModelSha", m.SourceSha256, "Type", type, "N", n, "Json", out.Json, ...
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
    "T", rows(1, :).', "X", rows(2:n+1, :).', "X0", u0(:), "TSpan", tspan, ...
    "P", p(:), "Stop", stop, "Warnings", warn(:), "Model", m, ...
    "Map", m.IsMap));
tr.attachAux(rows(n+2:end, :));
tr.attachRecord(record);
if stop ~= "End"
    if m.IsMap
        warning("phasecont:simulate:stoppedEarly", ...
            "The run stopped at n = %d with reason ""%s"" before n = %d.", ...
            tr.T(end), stop, tspan(2));
    else
        warning("phasecont:simulate:stoppedEarly", ...
            "The run stopped at t = %.6g with reason ""%s"" before t = %.6g.", ...
            tr.T(end), stop, tspan(2));
    end
end
end

function key = auxKey(m)
% The aux key of the problem: rows carry the aux columns only when the only
% list names an aux quantity (SPEC section 10).
key = "";
if any(ismember(lower(m.Only), lower(m.AuxNames)))
    key = ",""aux"":true";
end
end

function problem = ivpProblem(tspan, options, aux)
% The IVP problem JSON, with an options entry for each option given.
import phasecont.internal.num2json
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
    + num2json(tspan(2)) + "],""options"":{" + strjoin(parts, ",") + "}" ...
    + aux + "}";
end

function span = mapSpan(tspan)
% [n0 n1] of a map run as doubles, or the badOption error. An integer
% class value that does not survive the trip through double is refused.
ok = isnumeric(tspan) && isreal(tspan) && numel(tspan) == 2;
if ok
    span = double(tspan(:)).';
    ok = all(isfinite(span)) && all(span == round(span)) ...
        && all(abs(span) <= flintmax) && span(1) < span(2) ...
        && isequal(cast(span, class(tspan)), reshape(tspan, 1, 2));
end
if ~ok
    error("phasecont:model:badOption", ...
        "For a map, TSPAN must be [n0 n1], two integers with n0 < n1.");
end
end

function refuseMapOptions(options)
% The solver options of an ODE run do not apply to a map.
for name = ["RelTol" "AbsTol" "MaxSteps" "Hmax"]
    if ~isempty(options.(name))
        error("phasecont:simulate:mapOption", ...
            "%s does not apply to a map: simulate iterates a map from n0 to n1.", ...
            name);
    end
end
end

function warnIgnoredOptions(m)
% Name the @ options of the model, once per model. A warning that is turned
% off does not use up the notice. On a map the meth entry whose value starts
% with d is left out, since it is what makes the file a map.
id = "phasecont:simulate:ignoredOption";
names = string(fieldnames(m.Options)).';
if m.IsMap
    keep = true(size(names));
    for k = 1:numel(names)
        keep(k) = ~(startsWith(names(k), "meth", IgnoreCase = true) ...
            && startsWith(m.Options.(names(k)), ["d" "D"]));
    end
    names = names(keep);
end
if isempty(names)
    return
end
s = warning("query", id);
if string(s.state) == "off" || ~m.firstNotice(id)
    return
end
if m.IsMap
    text = "The model sets @ %s. simulate reads no @ options of a map: " + ...
        "it iterates the map from n0 to n1 of TSPAN. This warning comes " + ...
        "once per model.";
else
    text = "The model sets @ %s. simulate reads no @ options: it " + ...
        "integrates over TSPAN with Dormand-Prince 5(4), set only by " + ...
        "RelTol, AbsTol, MaxSteps and Hmax. This warning comes once per model.";
end
warning(id, text, strjoin(names, ", "));
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
