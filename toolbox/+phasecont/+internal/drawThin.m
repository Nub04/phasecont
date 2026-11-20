function drawThin(h, x, y, rows)
% DRAWTHIN Thin a drawn line to what its axes can show (SPEC section 8af-3).
%
%   phasecont.internal.drawThin(H, X, Y, ROWS) takes the line H, drawn with
%   the columns X and Y, whose rows are the rows ROWS of a Trajectory or a
%   Branch. It sets H.UserData.PhaseContRows to the indices of the drawn
%   rows and, on 2-D linear or log axes, keeps one MarkedClean listener on
%   the axes that draws only the rows the rule of SPEC 8af-3 keeps. The
%   listener lives in the line's application data and so is deleted with
%   the line. Every drawn vertex is an exact data row.
%
%   Internal use.

x = x(:);
y = y(:);
rows = double(rows(:));
setRows(h, rows);
if numel(x) < 3 || ~isfloat(x) || ~isfloat(y)
    return
end
ax = ancestor(h, "axes");
if isempty(ax)
    return
end
setappdata(h, "PhaseContFull", {x, y, rows});
setappdata(h, "PhaseContState", []);
try
    l = listener(ax, "MarkedClean", @(~, ~) refresh(h, ax));
catch
    % No listener: the line stays exact.
    rmappdata(h, "PhaseContFull");
    return
end
setappdata(h, "PhaseContListener", l);
end

function refresh(h, ax)
% Draw the rows the rule keeps for the current state of the axes, or every
% row when the state does not allow thinning. An unchanged state does
% nothing.
if ~isvalid(h) || ~isvalid(ax) || ~isappdata(h, "PhaseContFull")
    return
end
p = getpixelposition(ax);
w = p(3);
ht = p(4);
logx = ax.XScale == "log";
logy = ax.YScale == "log";
state = [ax.XLim ax.YLim logx logy w ht ax.View];
if isequal(state, getappdata(h, "PhaseContState"))
    return
end
setappdata(h, "PhaseContState", state);
full = getappdata(h, "PhaseContFull");
x = full{1};
y = full{2};
rows = full{3};
n = numel(x);
if ~isequal(ax.View, [0 90]) || n <= 4 * w || w <= 0 || ht <= 0
    keep = true(n, 1);
else
    keep = cellRule(x, y, ax.XLim, ax.YLim, logx, logy, w, ht);
end
if all(keep)
    if numel(h.XData) ~= n
        set(h, "XData", x, "YData", y);
    end
    setRows(h, rows);
else
    set(h, "XData", x(keep), "YData", y(keep));
    setRows(h, rows(keep));
end
end

function keep = cellRule(x, y, xl, yl, logx, logy, w, ht)
% The rows SPEC 8af-3 draws: a row whose 0.35-pixel cell differs from the
% cell of the row before it, the ends, every row with no pixel coordinate
% and the rows holding the extremes of x and y.
side = 0.35;
[cx, fx] = cells(x, xl, logx, w / side);
[cy, fy] = cells(y, yl, logy, ht / side);
n = numel(x);
keep = [true; cx(2:end) ~= cx(1:n-1) | cy(2:end) ~= cy(1:n-1)];
keep(n) = true;
keep(~isfinite(cx) | ~isfinite(cy)) = true;
keep(extremes(fx)) = true;
keep(extremes(fy)) = true;
end

function [c, v] = cells(v, lim, islog, scale)
% The cell index of each value along one axis, and the values on the scale
% the extremes are taken on (NaN for a value the axis cannot place).
v(~isfinite(v)) = NaN;
if islog
    v(v <= 0) = NaN;
    t = log10(v);
    lo = log10(lim(1));
    span = log10(lim(2)) - lo;
else
    t = v;
    lo = lim(1);
    span = lim(2) - lo;
end
c = floor((t - lo) / span * scale);
end

function k = extremes(v)
% The rows of the smallest and the largest value, ignoring NaN; none when
% every value is NaN.
k = zeros(0, 1);
if all(isnan(v))
    return
end
[~, i] = min(v);
[~, j] = max(v);
k = [i; j];
end

function setRows(h, rows)
% Record the drawn rows in the line's UserData, keeping any other field.
ud = h.UserData;
if isempty(ud)
    ud = struct();
end
if isstruct(ud) && isscalar(ud)
    ud.PhaseContRows = rows;
    h.UserData = ud;
end
end
