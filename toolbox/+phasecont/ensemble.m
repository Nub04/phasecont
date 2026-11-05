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
%   Nothing here has a default for a value that changes a result. These
%   options are required:
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
%                 CPU that has a device for the Precision. R.Label says
%                 which one ran.
%     Device      index of the device within its backend, counting from 0, as
%                 in phasecont.gpuDevices (default 0). Not with "auto".
%     Hmax        largest step length for Rtol/Atol, positive; Inf means no
%                 cap (default)
%     MaxSteps    most steps of one trajectory, a positive integer (default
%                 100000). A trajectory that needs more stops with Status
%                 "maxSteps"
%     Bounds      positive; a trajectory stops with Status "bounds" when a
%                 state passes this size in absolute value. Inf turns the test
%                 off (default)
%
%   A trajectory that stops with "stepTooSmall", "maxSteps" or "notFinite"
%   is reported with the warning ID phasecont:ensemble:memberStopped. The
%   call blocks while the engine runs and polls it about every 50 ms. Ctrl+C
%   cancels the run.
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
    options.MaxSteps (1,1) double = 100000
    options.Bounds (1,1) double = Inf
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
method = chooseMethod(options);

[x0, members] = startStates(x0, n);
[pm, shared] = parameterSets(m, p, members, np);

if ~(isnumeric(tspan) && isreal(tspan) && isvector(tspan) ...
        && numel(tspan) >= 2 && all(isfinite(tspan)) ...
        && issorted(tspan, "strictascend"))
    error("phasecont:ensemble:badOption", ...
        "TSPAN must be [t0 t1] or [t0 s1 ... t1]: finite, real and increasing.");
end
tspan = double(tspan(:)).';

if ~(options.MaxSteps >= 1 && options.MaxSteps == fix(options.MaxSteps) ...
        && options.MaxSteps <= double(intmax("uint32")))
    error("phasecont:ensemble:badOption", ...
        "MaxSteps must be an integer from 1 to %d.", intmax("uint32"));
end
if ~(options.Bounds > 0)
    error("phasecont:ensemble:badOption", ...
        "Bounds must be positive; Inf turns the test off.");
end

[backend, device] = chooseDevice(options, precision);

% The job text. Numbers go through num2json so they read back exactly.
if method == "rk4"
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
    + ",""t0"":" + num2json(tspan(1)) + ",""t1"":" + num2json(tspan(end)) ...
    + ",""method"":" + methodJson + ",""output"":" + outputJson ...
    + ",""bounds"":" + boundsJson ...
    + ",""max_steps"":" + sprintf("%d", options.MaxSteps) ...
    + ",""params"":""" + params + """}";

try
    ens = phasecont.internal.Ens(m, job, x0(:), pvec);
catch err
    throw(mapError(err));
end
cleanup = onCleanup(@() delete(ens)); %#ok<NASGU>
while true
    st = ens.poll();
    if st.state ~= "running"
        break
    end
    pause(0.05);
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

names = ["ok" "bounds" "stepTooSmall" "maxSteps" "notFinite"];
label = struct("Backend", string(st.label.backend), ...
    "Device", string(st.label.device), ...
    "Driver", string(st.label.driver), ...
    "Precision", string(st.label.precision), ...
    "SinglePrecision", logical(st.label.single_precision));
jobInfo = struct("TSpan", tspan, "Method", method, "Dt", double(options.Dt), ...
    "Rtol", double(options.Rtol), "Atol", double(options.Atol), ...
    "Hmax", Inf, "MaxSteps", options.MaxSteps, "Bounds", options.Bounds);
if method == "dp5" && ~isempty(options.Hmax)
    jobInfo.Hmax = double(options.Hmax);
end
r = phasecont.EnsembleResult(struct( ...
    "Model", m, "X0", x0, "P", pm, "Times", times, "States", states, ...
    "Status", reshape(names(code + 1), [], 1), ...
    "StatusCode", uint8(code), "TReached", reached, ...
    "Accepted", accepted, "Rejected", rejected, ...
    "Label", label, "Job", jobInfo));

failed = ismember(r.Status, ["stepTooSmall" "maxSteps" "notFinite"]);
if any(failed)
    warning("phasecont:ensemble:memberStopped", ...
        "%d of %d members stopped before t = %.6g (stepTooSmall, maxSteps or notFinite). See Status.", ...
        nnz(failed), members, tspan(end));
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
for b = ["cuda" "opencl" "metal" "cpu"]
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
