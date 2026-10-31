classdef Branch < handle
    % A computed branch with its special points: a branch of equilibria
    % (Type "EP"), a curve of fold points ("LP") or Hopf points ("H"), or a
    % branch of periodic orbits ("LC").
    %
    %   br = phasecont.continuation(m, eq, Free = "r")
    %   h  = br.points("H")     % special points with label H
    %   all = br.points()       % every special point
    %   plot(br, Var = "x")     % stable solid, unstable dashed, labeled specials
    %
    %   c = phasecont.continuation(m, lp, Type = "LP", Free = ["a" "b"])
    %   plot(c)                 % the curve in the (a, b) plane
    %
    %   lc = phasecont.continuation(m, h(1), Type = "LC", Free = "r", ...
    %       NTST = 40, NCOL = 4)
    %   [t, x] = lc.cycle(10)   % the orbit at point 10
    %   plot(lc, Var = "x")     % max and min of x over the orbit
    %
    % A Branch holds copied data and no engine handle, so it stays valid
    % after the Model is deleted. Rows are points: N points give N-by-k or
    % N-by-n arrays. Call points on one Branch at a time, not on an array.
    %
    % Properties
    %   Type         "EP", "LP", "H" or "LC"
    %   Free         name of the free parameter; [a1 a2] for a curve
    %   FreeIndex    its index in Model.Parameters; [i1 i2] for a curve
    %   P            parameter vector the run started from
    %   Alpha        free parameter value at each point, N-by-1; [a1 a2] in
    %                the order of Free, N-by-2, for a curve
    %   U            states, N-by-n (on a curve, the state at each point of
    %                it); for "LC", the orbit at the NTST*NCOL+1 base points,
    %                point-major, N-by-n(NTST*NCOL+1) (see cycle)
    %   Eigenvalues  N-by-n complex; the Floquet multipliers for "LC"
    %   Stable       true where all eigenvalues have negative real part; for
    %                "LC", where all multipliers except the one closest to 1
    %                have modulus below 1; false everywhere on a curve, whose
    %                points are never hyperbolic
    %   Tests        [psi_LP psi_BP psi_H] at each point, N-by-3; [psi_LPC
    %                psi_PD psi_NS] for "LC"; N-by-0 on a curve
    %   Codim2       codim-2 test values at each point of a curve, N-by-k
    %                (N-by-0 for Type "EP" and "LC")
    %   Codim2Labels the k labels of the columns of Codim2: ["CP" "BT" "ZH"]
    %                for "LP", ["GH" "BT" "ZH" "HH"] for "H"
    %   Period       period of the orbit at each point, N-by-1 ("LC" only)
    %   Mesh         mesh points from 0 to 1 at each point, N-by-(NTST+1)
    %                ("LC" only); the mesh moves when the run adapts it
    %   NTST NCOL    mesh intervals and collocation points of an "LC" run
    %   StepSize     step length used to reach each point
    %   NewtonIters  Newton iterations used at each point
    %   Specials     column of phasecont.Point
    %   Stop         why the run stopped ("" when unknown)
    %   Warnings     warning messages from the run
    %   Model        the phasecont.Model the run used
    %
    % Methods
    %   points, plot, cycle

    properties (SetAccess = private)
        Type (1,1) string = "EP"
        Free (1,:) string = ""
        FreeIndex (1,:) double = NaN
        P (:,1) double = zeros(0,1)
        Alpha (:,:) double = zeros(0,1)
        U (:,:) double = zeros(0,0)
        Eigenvalues (:,:) double = zeros(0,0)
        Stable (:,1) logical = false(0,1)
        Tests (:,:) double = zeros(0,3)
        Codim2 (:,:) double = zeros(0,0)
        Codim2Labels (1,:) string = strings(1,0)
        Period (:,1) double = zeros(0,1)
        Mesh (:,:) double = zeros(0,0)
        NTST (1,1) double = NaN
        NCOL (1,1) double = NaN
        StepSize (:,1) double = zeros(0,1)
        NewtonIters (:,1) double = zeros(0,1)
        Specials (:,1) phasecont.Point = phasecont.Point.empty(0,1)
        Stop (1,1) string = ""
        Warnings (:,1) string = strings(0,1)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
    end

    methods
        function obj = Branch(values)
            % Internal use: values is a struct with the property names as fields.
            arguments
                values (1,1) struct = struct()
            end
            for name = string(fieldnames(values))'
                obj.(name) = values.(name);
            end
        end

        function pts = points(br, label)
            % PTS = BR.POINTS() returns every special point, BR.POINTS(LABEL)
            % those with that label. An unknown label gives an empty array.
            if ~isscalar(br)
                error("phasecont:model:badBranch", ...
                    "points works on one Branch at a time; call it per element.");
            end
            pts = br.Specials;
            if nargin < 2 || isempty(pts)
                return
            end
            label = string(label);
            if ~isscalar(label)
                error("phasecont:model:badBranch", "The label must be one string.");
            end
            labels = strings(numel(pts), 1);
            for k = 1:numel(pts)
                labels(k) = pts(k).Label;
            end
            pts = pts(labels == label);
            if isempty(pts)
                pts = phasecont.Point.empty(0, 1);
            end
        end

        function [t, x] = cycle(br, i)
            % [T, X] = BR.CYCLE(I) returns the periodic orbit at point I of an
            % LC branch. T holds the NTST*NCOL+1 base-point times from 0 to
            % the period, as a column. X holds the state at those times, one
            % row per time and one column per variable. The last row closes
            % the orbit and equals the first.
            if ~isscalar(br)
                error("phasecont:model:badBranch", ...
                    "cycle works on one Branch at a time; call it per element.");
            end
            if br.Type ~= "LC"
                error("phasecont:model:badBranch", ...
                    "cycle needs a Branch of Type ""LC"", not ""%s"".", br.Type);
            end
            npts = numel(br.Period);
            if ~(isnumeric(i) && isscalar(i) && isreal(i) && i == round(i) ...
                    && i >= 1 && i <= npts)
                error("phasecont:model:badBranch", ...
                    "The point index must be an integer from 1 to %d.", npts);
            end
            N = br.NTST;
            m = br.NCOL;
            tau = br.Mesh(i, :);
            % Base points: m equally spaced points in each mesh interval, then 1.
            base = tau(1:N) + ((0:m-1).' / m) .* diff(tau);
            t = [base(:); tau(end)] * br.Period(i);
            x = reshape(br.U(i, :), [], N*m + 1).';
        end

        function h = plot(br, options)
            % H = PLOT(BR) draws the branch against its free parameter.
            % PLOT(BR, Var = "x") picks the variable by name or index,
            % PLOT(BR, Parent = AX) draws into AX without resetting it. BR may
            % be a 1-by-K array. Stable runs are solid (Tag "pc-stable"),
            % unstable runs dashed (Tag "pc-unstable"). Special points get a
            % marker (Tag "pc-special") and a text label (Tag "pc-label").
            % A curve is one solid line (Tag "pc-curve") in the plane of its
            % two free parameters and takes no Var. An LC branch draws the
            % max and the min of Var over each orbit, both styled by
            % stability, with its special points at the max.
            % Without Parent the plot uses newplot, so hold on is respected.
            % H holds the lines, markers and texts as a column vector.
            arguments
                br phasecont.Branch
                options.Var = []
                options.Parent = []
            end
            h = gobjects(0, 1);
            if isempty(br)
                return
            end
            if isempty(options.Parent)
                ax = newplot;
            else
                ax = options.Parent;
            end
            for k = 1:numel(br)
                h = [h; br(k).drawOne(ax, options.Var)]; %#ok<AGROW>
            end
            if nargout == 0
                clear h
            end
        end
    end

    methods (Access = private)
        function h = drawOne(br, ax, var)
            % Draw this branch on ax: stability runs, then special points.
            if br.Type == "LP" || br.Type == "H"
                h = br.drawCurve(ax, var);
                return
            end
            vi = br.variableIndex(var);
            if isempty(br.Alpha)
                h = gobjects(0, 1);
                return
            end
            alpha = br.Alpha;
            color = nextColor(ax);
            ylab = br.variableName(vi);
            if br.Type == "LC"
                % The variable at every base point of each orbit.
                cols = vi + br.stateCount() * (0:br.NTST*br.NCOL);
                y = max(br.U(:, cols), [], 2);
                ymin = min(br.U(:, cols), [], 2);
                h = [stabilityRuns(ax, alpha, y, br.Stable, color);
                    stabilityRuns(ax, alpha, ymin, br.Stable, color)];
                yall = [y; ymin];
                ylab = "max, min " + ylab;
            else
                y = br.U(:, vi);
                h = stabilityRuns(ax, alpha, y, br.Stable, color);
                yall = y;
            end

            px = zeros(numel(br.Specials), 1);
            py = px;
            for k = 1:numel(br.Specials)
                sp = br.Specials(k);
                px(k) = sp.P(br.FreeIndex);
                if br.Type == "LC"
                    py(k) = max(sp.U(cols));
                else
                    py(k) = sp.U(vi);
                end
            end
            h = [h; specialMarks(ax, br.Specials, px, py, ...
                0.015 * (max(alpha) - min(alpha)), 0.015 * (max(yall) - min(yall)))];
            xlabel(ax, br.Free);
            ylabel(ax, ylab);
        end

        function h = drawCurve(br, ax, var)
            % Draw a curve in the (a1, a2) plane with its special points.
            if ~isempty(var)
                error("phasecont:model:badOption", ...
                    "A %s curve is plotted in its parameter plane and takes no Var.", ...
                    br.Type);
            end
            h = gobjects(0, 1);
            if isempty(br.Alpha)
                return
            end
            x = br.Alpha(:, 1);
            y = br.Alpha(:, 2);
            color = nextColor(ax);
            if numel(x) == 1
                h(end+1, 1) = line(ax, x, y, Color = color, LineStyle = "none", ...
                    Marker = ".", MarkerSize = 12, Tag = "pc-curve");
            else
                h(end+1, 1) = line(ax, x, y, Color = color, LineStyle = "-", ...
                    Tag = "pc-curve");
            end
            px = zeros(numel(br.Specials), 1);
            py = px;
            for k = 1:numel(br.Specials)
                px(k) = br.Specials(k).P(br.FreeIndex(1));
                py(k) = br.Specials(k).P(br.FreeIndex(2));
            end
            h = [h; specialMarks(ax, br.Specials, px, py, ...
                0.015 * (max(x) - min(x)), 0.015 * (max(y) - min(y)))];
            xlabel(ax, br.Free(1));
            ylabel(ax, br.Free(2));
        end

        function vi = variableIndex(br, var)
            % Index into the columns of U from a name or an index.
            if isempty(br.Model)
                n = br.stateCount();
                names = strings(0, 1);
            else
                names = br.Model.Variables;
                n = numel(names);
            end
            if isempty(var)
                vi = 1;
            elseif isnumeric(var) && isscalar(var) && var == round(var) ...
                    && var >= 1 && var <= n
                vi = double(var);
            elseif (isstring(var) || ischar(var)) && isscalar(string(var)) ...
                    && any(names == string(var))
                vi = find(names == string(var), 1);
            else
                error("phasecont:model:unknownVariable", ...
                    "Var must be one of: %s.", strjoin(names, ", "));
            end
        end

        function name = variableName(br, vi)
            if isempty(br.Model)
                name = "u" + vi;
            else
                name = br.Model.Variables(vi);
            end
        end

        function n = stateCount(br)
            % Number of states, from the columns of U.
            n = size(br.U, 2);
            if br.Type == "LC"
                n = n / (br.NTST*br.NCOL + 1);
            end
        end
    end
end

function color = nextColor(ax)
% The next color of the axes color order, which then moves on by one.
order = ax.ColorOrder;
idx = ax.ColorOrderIndex;
color = order(mod(idx - 1, size(order, 1)) + 1, :);
ax.ColorOrderIndex = mod(idx, size(order, 1)) + 1;
end

function h = stabilityRuns(ax, x, y, stable, color)
% Stable runs solid (Tag "pc-stable"), unstable runs dashed ("pc-unstable").
% Neighbouring runs share their boundary point. One point is a marker.
h = gobjects(0, 1);
npts = numel(x);
if npts == 1
    tag = "pc-unstable";
    if stable(1)
        tag = "pc-stable";
    end
    h(end+1, 1) = line(ax, x, y, Color = color, ...
        LineStyle = "none", Marker = ".", MarkerSize = 12, Tag = tag);
    return
end
change = find(diff(stable) ~= 0);
first = [1; change + 1];
last = [change; npts];
for r = 1:numel(first)
    rows = first(r):min(last(r) + 1, npts);
    if stable(first(r))
        style = "-";
        tag = "pc-stable";
    else
        style = "--";
        tag = "pc-unstable";
    end
    h(end+1, 1) = line(ax, x(rows), y(rows), Color = color, ...
        LineStyle = style, Tag = tag); %#ok<AGROW>
end
end

function h = specialMarks(ax, specials, px, py, dx, dy)
% A marker (Tag "pc-special") and a text label ("pc-label") per special point.
h = gobjects(0, 1);
for k = 1:numel(specials)
    h(end+1, 1) = line(ax, px(k), py(k), LineStyle = "none", ...
        Marker = "o", MarkerSize = 7, MarkerEdgeColor = "k", ...
        MarkerFaceColor = "w", Tag = "pc-special"); %#ok<AGROW>
    h(end+1, 1) = text(ax, px(k) + dx, py(k) + dy, specials(k).Label, ...
        HorizontalAlignment = "left", VerticalAlignment = "bottom", ...
        Tag = "pc-label"); %#ok<AGROW>
end
end
