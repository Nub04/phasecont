function br = continuation(m, start, options)
%CONTINUATION Follow a branch of equilibria in one parameter.
%   BR = phasecont.continuation(M, START, Free = "r") continues START along
%   the free parameter. START is a phasecont.Point or a state vector. The
%   call blocks while the engine runs, polls it about every 50 ms and
%   returns a phasecont.Branch. Ctrl+C cancels the run and returns nothing.
%
%   BRS = phasecont.continuation(M, START, Free = "r", Switch = "BP1")
%   follows the branch and then the branch that starts at its first branch
%   point. BRS is a 1-by-2 array: the followed branch, then the switched one.
%
%   Name-value options
%     Free       free parameter, one of M.Parameters (required)
%     MaxSteps   most steps per branch
%     Direction  1 or -1, the first direction along the parameter
%     Bounds     [lo hi], finite, the run stops outside them
%     H0 Hmin Hmax   first, smallest and largest step length
%     TolX TolF  Newton tolerances on the step and the residual
%     MaxNewton  most Newton iterations per step
%     Detect     labels to detect, a subset of ["LP" "BP" "H"] (default all)
%     Switch     special point to switch at, like "BP1"
%     Plot       true draws the branch while the run goes and ends with the
%                styled plot of Branch.plot (default false)
%     Var        variable to plot, a name or an index into M.Variables
%                (default the first variable of M)
%     Params     struct or vector that overrides the parameters of START
%
%   Options that are not given keep the engine defaults. If START is a
%   special point (any label except "EP"), the run does not report that
%   point again at its first step.
arguments
    m (1,1) phasecont.Model
    start
    options.Free = []
    options.MaxSteps = []
    options.Direction = []
    options.Bounds = []
    options.H0 = []
    options.Hmin = []
    options.Hmax = []
    options.TolX = []
    options.TolF = []
    options.MaxNewton = []
    options.Detect = ["LP" "BP" "H"]
    options.Switch = []
    options.Plot (1,1) logical = false
    options.Var = []
    options.Params = []
end
import phasecont.internal.num2json

if isempty(options.Free)
    error("phasecont:model:badOption", "The Free option is required.");
end
free = string(options.Free);
if ~isscalar(free) || ~any(m.Parameters == free)
    error("phasecont:model:unknownParameter", ...
        "Free must be one of: %s.", strjoin(m.Parameters, ", "));
end
[u0, p, special] = startValues(m, start, options.Params);

parts = strings(1, 0);
parts(end+1) = numberOption("h0", options.H0, "H0");
parts(end+1) = numberOption("hmin", options.Hmin, "Hmin");
parts(end+1) = numberOption("hmax", options.Hmax, "Hmax");
parts(end+1) = numberOption("max_steps", options.MaxSteps, "MaxSteps", true);
parts(end+1) = numberOption("tol_x", options.TolX, "TolX");
parts(end+1) = numberOption("tol_f", options.TolF, "TolF");
parts(end+1) = numberOption("max_newton", options.MaxNewton, "MaxNewton", true);
if ~isempty(options.Direction)
    d = options.Direction;
    if ~(isnumeric(d) && isscalar(d) && (d == 1 || d == -1))
        error("phasecont:model:badOption", "Direction must be 1 or -1.");
    end
    parts(end+1) = """direction"":" + num2json(d);
end
if ~isempty(options.Bounds)
    b = options.Bounds;
    if ~(isnumeric(b) && isreal(b) && numel(b) == 2 && all(isfinite(b)) ...
            && b(1) < b(2))
        error("phasecont:model:badOption", ...
            "Bounds must be [lo hi] with finite values and lo < hi.");
    end
    parts(end+1) = """bounds"":[" + num2json(b(1)) + "," + num2json(b(2)) + "]";
end
detect = string(options.Detect);
if ~all(ismember(detect, ["LP" "BP" "H"]))
    error("phasecont:model:badOption", ...
        "Detect entries must be among ""LP"", ""BP"" and ""H"".");
end
parts(end+1) = """detect"":{""lp"":" + lower(string(any(detect == "LP"))) + ...
    ",""bp"":" + lower(string(any(detect == "BP"))) + ...
    ",""h"":" + lower(string(any(detect == "H"))) + "}";
parts = parts(strlength(parts) > 0);

problem = "{""type"":""EP"",""free"":" + jsonencode(free) + ...
    ",""start_is_special"":" + lower(string(special)) + ...
    ",""options"":{" + strjoin(parts, ",") + "}";
hasSwitch = ~isempty(options.Switch);
if hasSwitch
    sw = string(options.Switch);
    if ~isscalar(sw)
        error("phasecont:model:badOption", "Switch must be one string.");
    end
    problem = problem + ",""switch"":" + jsonencode(sw);
end
problem = problem + "}";

varIdx = 0;
if options.Plot || ~isempty(options.Var)
    if isempty(options.Var)
        varIdx = 1;
    else
        v = options.Var;
        if isnumeric(v) && isscalar(v) && v >= 1 && v <= numel(m.Variables) ...
                && v == round(v)
            varIdx = double(v);
        elseif (isstring(v) || ischar(v)) && isscalar(string(v))
            varIdx = find(m.Variables == string(v), 1);
        else
            varIdx = [];
        end
        if isempty(varIdx)
            error("phasecont:model:unknownVariable", ...
                "Var must be one of: %s.", strjoin(m.Variables, ", "));
        end
    end
end
ax = [];
if options.Plot
    ax = newplot;
    xlabel(ax, free);
    ylabel(ax, m.Variables(varIdx));
end

n = numel(m.Variables);
run = phasecont.internal.Run(m, problem, u0, p);
cleanup = onCleanup(@() delete(run)); %#ok<NASGU>
pts = cell(1, 0);
lines = gobjects(1, 0);
pause_s = 0.05;
while true
    st = run.poll();
    nb = numel(st.branches);
    for b = 1:nb
        if b > numel(pts)
            pts{b} = zeros(st.stride, 0);
        end
        have = size(pts{b}, 2);
        avail = st.branches(b).points;
        if avail > have
            pts{b} = [pts{b}, run.copy(b, 0, have + 1, avail - have, st.stride)];
        end
        if options.Plot
            if b > numel(lines)
                lines(b) = line(ax, NaN, NaN);
            end
            set(lines(b), XData = pts{b}(1, :), YData = pts{b}(1 + varIdx, :));
        end
    end
    hook = phasecont.internal.pollHook();
    if ~isempty(hook)
        hook(run, st);
    end
    if options.Plot
        drawnow limitrate
    end
    if st.state ~= "running"
        break
    end
    pause(pause_s);
end
nb = numel(st.branches);
br = phasecont.Branch.empty(1, 0);
for b = 1:nb
    nsp = numel(st.branches(b).specials);
    sp = run.copy(b, 1, 1, nsp, st.stride);
    one = phasecont.internal.rows2branch(m, free, p, n, pts{b}, sp, st.branches(b));
    br(1, b) = one;
end
if options.Plot
    % Replace the live lines with the styled plot.
    delete(lines(isgraphics(lines)));
    for b = 1:nb
        plot(br(b), Var = varIdx, Parent = ax);
    end
    drawnow
end
delete(run);
end

function [u0, p, special] = startValues(m, start, params)
% Start state, parameters and whether the start is a special point.
n = numel(m.Variables);
np = numel(m.Parameters);
if isa(start, "phasecont.Point")
    if ~isscalar(start)
        error("phasecont:model:badStart", "START must be one Point.");
    end
    if ~isempty(start.Model) && start.Model ~= m
        error("phasecont:model:badStart", ...
            "The start point belongs to a different model.");
    end
    u0 = start.U;
    p = start.P;
    special = start.Label ~= "EP";
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
    u0 = double(start(:));
    special = false;
    if isempty(params) && ~isstruct(params)
        p = m.ParameterValues;
    else
        p = m.paramVector(params);
    end
else
    error("phasecont:model:badStart", ...
        "START must be a phasecont.Point or a real state vector.");
end
end

function s = numberOption(key, value, name, isCount)
% One "key":value entry, or "" when the option is not set.
if nargin < 4
    isCount = false;
end
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
