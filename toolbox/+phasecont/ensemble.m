function r = ensemble(m, x0, p, tspan, options)
%ENSEMBLE Integrate many trajectories of one model at once.
%   R = phasecont.ensemble(M, X0, P, TSPAN, Precision = "double", ...
%       Rtol = 1e-8, Atol = 1e-10) integrates the model M from every column
%   of X0 and returns a phasecont.EnsembleResult. It runs on the CPU or on a
%   GPU, one thread per trajectory. It does not speed up one trajectory or a
%   continuation; those stay in double precision on the CPU.
%
%   X0 is n-by-M, one start state per column (a single state vector is one
%   member). P holds the parameters: np-by-1 for one set shared by all
%   members, np-by-M for one set per member, a scalar struct that changes
%   the named parameters of the model, or [] for the values in the model.
%
%   TSPAN is [t0 t1], or [t0 s1 s2 ... t1] to record the states at the
%   times s1, s2, ... as well. The times must increase; the engine only
%   integrates forward. With two entries, R holds the state at t1 only.
%
%   On a map (M.IsMap), ensemble iterates the map instead, one iterate per
%   unit of time, and t inside the model is the iterate index. TSPAN is then
%   [n0 n1] or [n0 s1 ... n1]: increasing whole numbers of magnitude at most
%   flintmax, with n1 - n0 at most 4294967295. In single precision the engine
%   also needs every entry to be at most 2^24 in magnitude and two
%   neighboring entries to be at most 2^24 apart. Precision is
%   still required, and Backend, Device and Bounds work as for an ODE. Dt,
%   Rtol, Atol, Hmax and MaxSteps do not apply to a map, and each one passed
%   raises phasecont:ensemble:mapOption. A member of a map stops early only
%   with Status "bounds", "notFinite" or "recursionDepth".
%
%   Nothing here has a default for a value that changes a result. These
%   options are required (for a map, only Precision):
%     Precision   "single" or "double". Single precision is for screening
%                 (see EnsembleResult); double is exact to the same level as
%                 phasecont.simulate on the CPU.
%     Rtol, Atol  tolerances of the Dormand-Prince 5(4) method with one step
%                 size per trajectory, or
%     Dt          step length of the fixed-step Runge-Kutta 4 method.
%                 Give Rtol and Atol, or Dt, not both.
%   A relative tolerance below what the precision resolves (13 times the
%   unit roundoff: 7.8e-7 in single, 1.4e-15 in double) raises
%   phasecont:ensemble:toleranceTooTight. It is never changed for you.
%
%   Other name-value options
%     Backend     "auto" (default), "cpu", "cuda", "opencl" or "metal".
%                 "auto" takes the first of CUDA, OpenCL, Metal and then the
%                 CPU that has a device for the Precision. On macOS the
%                 order is Metal, OpenCL and then the CPU. R.Label says
%                 which one ran.
%     Device      index of the device within its backend, counting from 0, as
%                 in phasecont.gpuDevices (default 0). Not with "auto".
%     Hmax        largest step length for Rtol/Atol, positive; Inf means no
%                 cap (default)
%     MaxSteps    most steps of one trajectory, a positive integer; [] means
%                 100000 (default). A trajectory that needs more stops with
%                 Status "maxSteps"
%     Bounds      positive; a trajectory stops with Status "bounds" when a
%                 state passes this size in absolute value. Inf turns the test
%                 off (default)
%     FastMath    false (default) or true. true lets the CUDA compiler use
%                 fused multiply-add, which runs double precision faster on
%                 some GPUs. It works only with Backend "cuda" and Precision
%                 "double"; anything else raises
%                 phasecont:ensemble:fastMathBackend, including a Backend
%                 "auto" that picks another device. Every such run warns with
%                 phasecont:ensemble:fastMath, and R.Label.Math is "fma"
%                 instead of "exact". The results are not bit-reproducible:
%                 call refine before they feed continuation.
%
%   A trajectory that stops with "stepTooSmall", "maxSteps", "notFinite" or
%   "recursionDepth" is reported with the warning ID
%   phasecont:ensemble:memberStopped. A member ends with "recursionDepth"
%   when a recursive function call in its right-hand side needs more than
%   1,000,000 active frames; the other members go on. The
%   call blocks while the engine runs and polls it, first after 1 ms and then
%   at growing intervals up to 50 ms, so a short run returns within a few
%   milliseconds. Ctrl+C cancels the run.
%
%   See also phasecont.gpuDevices, phasecont.EnsembleResult, phasecont.simulate.
arguments
    m (1,1) phasecont.Model
    x0
    p
    tspan
    options.Backend (1,1) string = "auto"
    options.Device = []
    options.Precision (1,1) string = ""
    options.Dt = []
    options.Rtol = []
    options.Atol = []
    options.Hmax = []
    options.MaxSteps double {mustBeScalarOrEmpty} = []
    options.Bounds (1,1) double = Inf
    options.FastMath (1,1) logical = false
end
import phasecont.internal.num2json

n = numel(m.Variables);
np = numel(m.Parameters);

% Required choices first, so a missing one is reported before anything else.
precision = lower(options.Precision);
if strlength(precision) == 0
    error("phasecont:ensemble:missingPrecision", ...
        "Precision is required: ""single"" or ""double"". It has no default.");
end
if ~any(precision == ["single" "double"])
    error("phasecont:ensemble:badOption", ...
        "Precision must be ""single"" or ""double"", got ""%s"".", ...
        options.Precision);
end
isMap = m.IsMap;
if isMap
    method = "iter";
    mapOptions(options);
else
    method = chooseMethod(options);
end

[x0, members] = startStates(x0, n);
[pm, shared] = parameterSets(m, p, members, np);

spanOk = isnumeric(tspan) && isreal(tspan) && isvector(tspan) ...
    && numel(tspan) >= 2 && all(isfinite(tspan)) ...
    && issorted(tspan, "strictascend");
if isMap
    mapOk = spanOk;
    if mapOk
        t = double(tspan);
        mapOk = all(t == fix(t)) && all(abs(t) <= flintmax) ...
            && t(end) - t(1) <= double(intmax("uint32"));
    end
    if ~mapOk
        error("phasecont:ensemble:badOption", ...
            "For a map, TSPAN must be [n0 n1] or [n0 s1 ... n1]: increasing " ...
            + "whole numbers of magnitude at most flintmax, with n1 - n0 at " ...
            + "most %d.", intmax("uint32"));
    end
elseif ~spanOk
    error("phasecont:ensemble:badOption", ...
        "TSPAN must be [t0 t1] or [t0 s1 ... t1]: finite, real and increasing.");
end
tspan = double(tspan(:)).';

if isMap
    % One iterate per unit of time, so the cap never stops a map run.
    maxSteps = tspan(end) - tspan(1);
elseif isempty(options.MaxSteps)
    maxSteps = 100000;
else
    maxSteps = options.MaxSteps;
    if ~(isreal(maxSteps) && maxSteps >= 1 && maxSteps == fix(maxSteps) ...
            && maxSteps <= double(intmax("uint32")))
        error("phasecont:ensemble:badOption", ...
            "MaxSteps must be an integer from 1 to %d.", intmax("uint32"));
    end
end
if ~(options.Bounds > 0)
    error("phasecont:ensemble:badOption", ...
        "Bounds must be positive; Inf turns the test off.");
end

[backend, device] = chooseDevice(options, precision);
math = "exact";
if options.FastMath
    if backend ~= "cuda" || precision ~= "double"
        error("phasecont:ensemble:fastMathBackend", ...
            "FastMath runs only with Backend ""cuda"" and Precision ""double"", not " ...
            + "on %s in %s precision. Set FastMath = false.", backend, precision);
    end
    math = "fma";
end

% The job text. Numbers go through num2json so they read back exactly.
if method == "iter"
    methodJson = "{""type"":""iter""}";
elseif method == "rk4"
    methodJson = "{""type"":""rk4"",""dt"":" + num2json(double(options.Dt)) + "}";
else
    hmax = "null";
    if isfinite(options.Hmax)
        hmax = num2json(double(options.Hmax));
    end
    methodJson = "{""type"":""dp5"",""rtol"":" + num2json(double(options.Rtol)) ...
        + ",""atol"":" + num2json(double(options.Atol)) + ",""hmax"":" + hmax + "}";
end
if numel(tspan) == 2
    outputJson = """final""";
else
    samples = strings(1, numel(tspan) - 1);
    for k = 2:numel(tspan)
        samples(k - 1) = num2json(tspan(k));
    end
    outputJson = "{""samples"":[" + strjoin(samples, ",") + "]}";
end
boundsJson = "null";
if isfinite(options.Bounds)
    boundsJson = num2json(options.Bounds);
end
params = "per_member";
pvec = pm(:);
if shared
    params = "shared";
    pvec = pm(:, 1);
end
job = "{""backend"":""" + backend + """,""device"":" + sprintf("%d", device) ...
    + ",""precision"":""" + precision + """" ...
    + ",""math"":""" + math + """" ...
    + ",""t0"":" + num2json(tspan(1)) + ",""t1"":" + num2json(tspan(end)) ...
    + ",""method"":" + methodJson + ",""output"":" + outputJson ...
    + ",""bounds"":" + boundsJson ...
    + ",""max_steps"":" + sprintf("%d", maxSteps) ...
    + ",""params"":""" + params + """}";

try
    ens = phasecont.internal.Ens(m, job, x0(:), pvec);
catch err
    throw(mapError(err));
end
cleanup = onCleanup(@() delete(ens)); %#ok<NASGU>
if math == "fma"
    warning("phasecont:ensemble:fastMath", "%s", ...
        "FastMath is on: this run uses fused multiply-add on CUDA. Results are " ...
        + "not bit-reproducible against the default, or against another backend. " ...
        + "For chaotic models they can differ from the default run by more than " ...
        + "rounding, and step counts and Status can differ too. Do not use these " ...
        + "states as input to continuation, a normal form or Floquet multipliers; " ...
        + "call refine first.");
end
polls = 0;
while true
    st = ens.poll();
    if st.state ~= "running"
        break
    end
    polls = polls + 1;
    pause(phasecont.internal.pollDelay(polls));
end
if st.state ~= "done"
    error("phasecont:ensemble:cancelled", "The ensemble run was cancelled.");
end

S = st.samples;
if st.members ~= members || st.n ~= n
    error("phasecont:engine:internal", ...
        "The engine returned %d members of %d states, expected %d of %d.", ...
        st.members, st.n, members, n);
end
states = reshape(ens.copy(phasecont.internal.Ens.States, members * S * n), n, S, members);
times = ens.copy(phasecont.internal.Ens.Times, S);
code = ens.copy(phasecont.internal.Ens.Status, members);
reached = ens.copy(phasecont.internal.Ens.TReached, members);
accepted = ens.copy(phasecont.internal.Ens.Accepted, members);
rejected = ens.copy(phasecont.internal.Ens.Rejected, members);

% Code 5 means a member still running, which a finished run never returns.
names = ["ok" "bounds" "stepTooSmall" "maxSteps" "notFinite" "running" "recursionDepth"];
label = struct("Backend", string(st.label.backend), ...
    "Device", string(st.label.device), ...
    "Driver", string(st.label.driver), ...
    "Precision", string(st.label.precision), ...
    "SinglePrecision", logical(st.label.single_precision), ...
    "Math", string(st.label.math));
jobInfo = struct("TSpan", tspan, "Method", method, "Dt", double(options.Dt), ...
    "Rtol", double(options.Rtol), "Atol", double(options.Atol), ...
    "Hmax", Inf, "MaxSteps", maxSteps, "Bounds", options.Bounds, ...
    "FastMath", math == "fma");
if method == "iter"
    jobInfo.Hmax = [];
elseif method == "dp5" && ~isempty(options.Hmax)
    jobInfo.Hmax = double(options.Hmax);
end
r = phasecont.EnsembleResult(struct( ...
    "Model", m, "X0", x0, "P", pm, "Times", times, "States", states, ...
    "Status", reshape(names(code + 1), [], 1), ...
    "StatusCode", uint8(code), "TReached", reached, ...
    "Accepted", accepted, "Rejected", rejected, ...
    "Label", label, "Job", jobInfo));

failed = ismember(r.Status, ["stepTooSmall" "maxSteps" "notFinite" "recursionDepth"]);
if any(failed)
    why = strjoin(unique(r.Status(failed), "stable"), ", ");
    if isMap
        warning("phasecont:ensemble:memberStopped", ...
            "%d of %d members stopped before n = %d (%s). See Status.", ...
            nnz(failed), members, tspan(end), why);
    else
        warning("phasecont:ensemble:memberStopped", ...
            "%d of %d members stopped before t = %.6g (%s). See Status.", ...
            nnz(failed), members, tspan(end), why);
    end
end
end

function method = chooseMethod(options)
% "rk4" when Dt is given, "dp5" when Rtol and Atol are. Never a default.
haveDt = ~isempty(options.Dt);
haveTol = ~isempty(options.Rtol) || ~isempty(options.Atol);
if haveDt && haveTol
    error("phasecont:ensemble:ambiguousMethod", ...
        "Give Dt (Runge-Kutta 4) or Rtol and Atol (Dormand-Prince 5(4)), not both.");
end
if haveDt
    method = "rk4";
    positive(options.Dt, "Dt");
    if ~isempty(options.Hmax)
        error("phasecont:ensemble:badOption", "Hmax applies only with Rtol and Atol.");
    end
elseif haveTol
    method = "dp5";
    if isempty(options.Rtol) || isempty(options.Atol)
        error("phasecont:ensemble:missingTolerance", ...
            "Rtol and Atol are both required. They have no default.");
    end
    positive(options.Rtol, "Rtol");
    positive(options.Atol, "Atol");
    if ~isempty(options.Hmax) && ~(isnumeric(options.Hmax) ...
            && isreal(options.Hmax) && isscalar(options.Hmax) && options.Hmax > 0)
        error("phasecont:ensemble:badOption", "Hmax must be positive or Inf.");
    end
else
    error("phasecont:ensemble:missingMethod", ...
        "Give Rtol and Atol, or Dt. The step control has no default.");
end
end

function mapOptions(options)
% A map is iterated, so no option of a step method applies. Each one passed is
% an error, checked in a fixed order.
for name = ["Dt" "Rtol" "Atol" "Hmax" "MaxSteps"]
    if ~isempty(options.(name))
        error("phasecont:ensemble:mapOption", ...
            "%s does not apply to a map: ensemble iterates a map from n0 to n1.", name);
    end
end
end

function positive(v, name)
if ~(isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && v > 0)
    error("phasecont:ensemble:badOption", ...
        "%s must be a positive finite number.", name);
end
end

function [x0, members] = startStates(x0, n)
% The start states as an n-by-M matrix.
if ~(isnumeric(x0) && isreal(x0) && ismatrix(x0) && ~isempty(x0))
    error("phasecont:model:badStart", ...
        "X0 must be a nonempty real n-by-M matrix, one start state per column.");
end
if size(x0, 1) ~= n
    if isvector(x0) && numel(x0) == n
        x0 = x0(:);
    else
        error("phasecont:model:badStart", ...
            "X0 needs %d rows, one per state, but has %d.", n, size(x0, 1));
    end
end
if ~all(isfinite(x0), "all")
    error("phasecont:model:badStart", "X0 must be finite.");
end
x0 = double(x0);
members = size(x0, 2);
end

function [pm, shared] = parameterSets(m, p, members, np)
% The parameters as np-by-M, and whether every member has the same set.
shared = true;
if isstruct(p) || (isnumeric(p) && isempty(p) && np > 0)
    % A struct changes named values; [] keeps the values of the model.
    if isempty(p) && ~isstruct(p)
        v = m.ParameterValues(:);
    else
        v = m.paramVector(p);
    end
    pm = repmat(v(:), 1, members);
    return
end
if ~(isnumeric(p) && isreal(p) && ismatrix(p) && all(isfinite(p), "all"))
    error("phasecont:model:badParameter", ...
        "P must be a struct, [] or a finite real np-by-1 or np-by-M matrix.");
end
if np == 0
    if ~isempty(p)
        error("phasecont:model:badParameter", ...
            "The model has no parameters, so P must be empty.");
    end
    pm = zeros(0, members);
    return
end
if size(p, 1) ~= np || ~any(size(p, 2) == [1 members])
    error("phasecont:model:badParameter", ...
        "P must be %d-by-1 or %d-by-%d (np-by-1 or np-by-M), got %d-by-%d.", ...
        np, np, members, size(p, 1), size(p, 2));
end
shared = size(p, 2) == 1;
pm = repmat(double(p), 1, members / size(p, 2));
end

function [backend, device] = chooseDevice(options, precision)
% The backend name and device index for the job.
backend = lower(options.Backend);
if ~any(backend == ["auto" "cpu" "cuda" "opencl" "metal"])
    error("phasecont:ensemble:badOption", ...
        "Backend must be ""auto"", ""cpu"", ""cuda"", ""opencl"" or ""metal"", got ""%s"".", ...
        options.Backend);
end
dev = options.Device;
if ~isempty(dev) && ~(isnumeric(dev) && isreal(dev) && isscalar(dev) ...
        && dev >= 0 && dev == fix(dev))
    error("phasecont:ensemble:badOption", ...
        "Device must be a non-negative integer, counting from 0.");
end
if backend ~= "auto"
    device = 0;
    if ~isempty(dev)
        device = double(dev);
    end
    return
end
if ~isempty(dev)
    error("phasecont:ensemble:badOption", ...
        "Device cannot be used with Backend = ""auto"". Name the backend.");
end
list = phasecont.gpuDevices();
if precision == "single"
    list = list(list.Fp32, :);
else
    list = list(list.Fp64, :);
end
for b = phasecont.internal.autoBackendOrder()
    hit = list(list.Backend == b, :);
    if ~isempty(hit)
        backend = b;
        device = hit.Index(1);
        return
    end
end
error("phasecont:engine:gpuUnavailable", ...
    "No device runs %s precision.", precision);
end

function err = mapError(err)
% The engine reports a tolerance below the precision as a bad problem. Give
% it its own ID so a caller can catch it.
if err.identifier == "phasecont:engine:badProblem" ...
        && contains(err.message, "tolerance too tight")
    err = MException("phasecont:ensemble:toleranceTooTight", "%s", err.message);
end
end
