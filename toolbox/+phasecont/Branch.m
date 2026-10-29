classdef Branch < handle
    % A computed branch of equilibria with its special points.
    %
    %   br = phasecont.continuation(m, eq, Free = "r")
    %   h  = br.points("H")     % special points with label H
    %   all = br.points()       % every special point
    %   plot(br, Var = "x")     % stable solid, unstable dashed, labeled specials
    %
    % A Branch holds copied data and no engine handle, so it stays valid
    % after the Model is deleted. Rows are points: N points give N-by-1 or
    % N-by-n arrays. Call points on one Branch at a time, not on an array.
    %
    % Properties
    %   Free         name of the free parameter
    %   FreeIndex    its index in Model.Parameters
    %   P            parameter vector the run started from
    %   Alpha        free parameter value at each point
    %   U            states, N-by-n
    %   Eigenvalues  N-by-n complex
    %   Stable       true where all eigenvalues have negative real part
    %   Tests        [psi_LP psi_BP psi_H] at each point, N-by-3
    %   StepSize     step length used to reach each point
    %   NewtonIters  Newton iterations used at each point
    %   Specials     column of phasecont.Point
    %   Stop         why the run stopped ("" when unknown)
    %   Warnings     warning messages from the run
    %   Model        the phasecont.Model the run used
    %
    % Methods
    %   points, plot

    properties (SetAccess = private)
        Free (1,1) string = ""
        FreeIndex (1,1) double = NaN
        P (:,1) double = zeros(0,1)
        Alpha (:,1) double = zeros(0,1)
        U (:,:) double = zeros(0,0)
        Eigenvalues (:,:) double = zeros(0,0)
        Stable (:,1) logical = false(0,1)
        Tests (:,:) double = zeros(0,3)
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

        function h = plot(br, options)
            % H = PLOT(BR) draws the branch against its free parameter.
            % PLOT(BR, Var = "x") picks the variable by name or index,
            % PLOT(BR, Parent = AX) draws into AX without resetting it. BR may
            % be a 1-by-K array. Stable runs are solid (Tag "pc-stable"),
            % unstable runs dashed (Tag "pc-unstable"). Special points get a
            % marker (Tag "pc-special") and a text label (Tag "pc-label").
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
            vi = br.variableIndex(var);
            if isempty(br.Alpha)
                h = gobjects(0, 1);
                return
            end
            alpha = br.Alpha;
            y = br.U(:, vi);
            npts = numel(alpha);
            order = ax.ColorOrder;
            idx = ax.ColorOrderIndex;
            color = order(mod(idx - 1, size(order, 1)) + 1, :);
            ax.ColorOrderIndex = mod(idx, size(order, 1)) + 1;

            h = gobjects(0, 1);
            if npts == 1
                tag = "pc-unstable";
                if br.Stable(1)
                    tag = "pc-stable";
                end
                h(end+1, 1) = line(ax, alpha, y, Color = color, ...
                    LineStyle = "none", Marker = ".", MarkerSize = 12, Tag = tag);
            else
                % Maximal runs of equal Stable; neighbours share a boundary point.
                change = find(diff(br.Stable) ~= 0);
                first = [1; change + 1];
                last = [change; npts];
                for r = 1:numel(first)
                    rows = first(r):min(last(r) + 1, npts);
                    if br.Stable(first(r))
                        style = "-";
                        tag = "pc-stable";
                    else
                        style = "--";
                        tag = "pc-unstable";
                    end
                    h(end+1, 1) = line(ax, alpha(rows), y(rows), Color = color, ...
                        LineStyle = style, Tag = tag); %#ok<AGROW>
                end
            end

            dx = 0.015 * (max(alpha) - min(alpha));
            dy = 0.015 * (max(y) - min(y));
            for k = 1:numel(br.Specials)
                sp = br.Specials(k);
                px = sp.P(br.FreeIndex);
                py = sp.U(vi);
                h(end+1, 1) = line(ax, px, py, LineStyle = "none", ...
                    Marker = "o", MarkerSize = 7, MarkerEdgeColor = "k", ...
                    MarkerFaceColor = "w", Tag = "pc-special"); %#ok<AGROW>
                h(end+1, 1) = text(ax, px + dx, py + dy, sp.Label, ...
                    HorizontalAlignment = "left", VerticalAlignment = "bottom", ...
                    Tag = "pc-label"); %#ok<AGROW>
            end
            xlabel(ax, br.Free);
            ylabel(ax, br.variableName(vi));
        end

        function vi = variableIndex(br, var)
            % Index into the columns of U from a name or an index.
            if isempty(br.Model)
                n = size(br.U, 2);
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
    end
end
