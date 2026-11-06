function fig = figureFromBranch(br, options)
%FIGUREFROMBRANCH Figure data for phasecont.latex.pgfplots from a Branch.
%   FIG = phasecont.latex.figureFromBranch(BR) returns the struct that
%   phasecont.latex.pgfplots renders, with the same content as plot(BR).
%
%   A branch of equilibria is one series of Var against the free
%   parameter. A branch of limit cycles (Type "LC") is two series, the
%   maximum and the minimum of Var over each orbit. A curve (Type "LP",
%   "H", "LPC", "PD", "NS" or "HOM") is one series in the plane of its two
%   free parameters and takes no Var. Stable runs are solid and unstable runs
%   dashed, and each special point is a mark with its label.
%
%   Options
%     Var    name or index of the state variable, default the first
%     Color  "#RRGGBB" for every series, default "#000000". One colour keeps
%            the stable and unstable runs of a branch the same colour.
%     Title  title of the axes, default none
%
%   FIG has the fields XLabel, YLabel, Title, Series (X, Y, Class, Color)
%   and Marks (Label, X, Y). Edit it before rendering to rename the axes.
arguments
    br (1,1) phasecont.Branch
    options.Var = []
    options.Color (1,1) string = "#000000"
    options.Title (1,1) string = ""
end

isCurve = ~any(br.Type == ["EP" "LC"]);
if isCurve
    if ~isempty(options.Var)
        error("phasecont:model:badOption", ...
            "A %s curve is drawn in its parameter plane and takes no Var.", br.Type);
    end
    [series, marks, xlabel, ylabel] = curveData(br, options.Color);
else
    [series, marks, xlabel, ylabel] = branchData(br, options.Var, options.Color);
end
fig = struct("XLabel", xlabel, "YLabel", ylabel, "Title", options.Title, ...
    "Series", series, "Marks", marks);
end

function [series, marks, xlabel, ylabel] = curveData(br, color)
series = emptySeries();
marks = emptyMarks();
if ~isempty(br.Alpha)
    series = oneSeries(br.Alpha(:, 1), br.Alpha(:, 2), "stable", color);
end
labels = strings(numel(br.Specials), 1);
px = zeros(numel(br.Specials), 1);
py = px;
for k = 1:numel(br.Specials)
    labels(k) = br.Specials(k).Label;
    px(k) = br.Specials(k).P(br.FreeIndex(1));
    py(k) = br.Specials(k).P(br.FreeIndex(2));
end
marks = markList(labels, px, py);
xlabel = br.Free(1);
ylabel = br.Free(2);
end

function [series, marks, xlabel, ylabel] = branchData(br, var, color)
isCycle = br.Type == "LC";
if isCycle
    n = size(br.U, 2) / (br.NTST * br.NCOL + 1);
else
    n = size(br.U, 2);
end
if isempty(br.Model)
    names = "u" + (1:n);
else
    names = br.Model.Variables(:).';
end
vi = variableIndex(var, names, n);
ylabel = names(vi);
xlabel = br.Free;

cols = [];
if isCycle
    cols = vi + n * (0:br.NTST * br.NCOL);
end
series = emptySeries();
if ~isempty(br.Alpha)
    if isCycle
        ymax = max(br.U(:, cols), [], 2);
        ymin = min(br.U(:, cols), [], 2);
        series = [stabilityRuns(br.Alpha, ymax, br.Stable, color); ...
            stabilityRuns(br.Alpha, ymin, br.Stable, color)];
        ylabel = "max, min " + ylabel;
    else
        series = stabilityRuns(br.Alpha, br.U(:, vi), br.Stable, color);
    end
end

labels = strings(numel(br.Specials), 1);
px = zeros(numel(br.Specials), 1);
py = px;
for k = 1:numel(br.Specials)
    sp = br.Specials(k);
    labels(k) = sp.Label;
    px(k) = sp.P(br.FreeIndex);
    if isCycle
        py(k) = max(sp.U(cols));
    else
        py(k) = sp.U(vi);
    end
end
marks = markList(labels, px, py);
end

function vi = variableIndex(var, names, n)
if isempty(var)
    vi = 1;
elseif isnumeric(var) && isscalar(var) && var == round(var) && var >= 1 && var <= n
    vi = double(var);
elseif (isstring(var) || ischar(var)) && isscalar(string(var)) ...
        && any(names == string(var))
    vi = find(names == string(var), 1);
else
    error("phasecont:model:unknownVariable", ...
        "Var must be one of: %s.", strjoin(names, ", "));
end
end

function series = stabilityRuns(x, y, stable, color)
% One series per run of equal stability. Neighbouring runs share their
% boundary point, so the drawn line has no gap, as in Branch.plot.
npts = numel(x);
if npts == 1
    series = oneSeries(x, y, classOf(stable(1)), color);
    return
end
change = find(diff(stable) ~= 0);
first = [1; change + 1];
last = [change; npts];
series = emptySeries();
for r = 1:numel(first)
    rows = first(r):min(last(r) + 1, npts);
    series = [series; oneSeries(x(rows), y(rows), classOf(stable(first(r))), color)]; %#ok<AGROW>
end
end

function c = classOf(isStable)
if isStable
    c = "stable";
else
    c = "unstable";
end
end

function s = oneSeries(x, y, class, color)
s = struct("X", x(:), "Y", y(:), "Class", class, "Color", color);
end

function s = emptySeries()
s = struct("X", {}, "Y", {}, "Class", {}, "Color", {});
s = s(:);
end

function m = emptyMarks()
m = struct("Label", {}, "X", {}, "Y", {});
m = m(:);
end

function m = markList(labels, px, py)
m = emptyMarks();
for k = 1:numel(labels)
    m = [m; struct("Label", labels(k), "X", px(k), "Y", py(k))]; %#ok<AGROW>
end
end
