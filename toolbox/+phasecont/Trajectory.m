classdef Trajectory < handle
    % A computed solution of an initial value problem.
    %
    %   tr = phasecont.simulate(m, [0.1 0.2], [0 50])
    %   plot(tr)                  % phase plane of the first two variables
    %   plot(tr, Var = "x")       % x against t
    %   plot(tr, Var = ["x" "y"]) % y against x
    %
    % A Trajectory holds copied data and no engine handle, so it stays
    % valid after the Model is deleted. Rows are time points: N points give
    % N-by-n arrays.
    %
    % Properties
    %   T         time at each accepted step, N-by-1, with the start time first
    %   X         states, N-by-n
    %   X0        start state, n-by-1
    %   TSpan     [t0 t1] as given; t1 < t0 integrates backward
    %   P         parameter vector of the run
    %   Stop      why the run stopped: "End" when it reached t1, else
    %             "MaxSteps", "StepTooSmall", "Cancelled" or "Error"
    %             ("" when unknown)
    %   Warnings  warning messages from the run
    %   Model     the phasecont.Model the run used
    %
    % Methods
    %   plot

    properties (SetAccess = private)
        T (:,1) double = zeros(0,1)
        X (:,:) double = zeros(0,0)
        X0 (:,1) double = zeros(0,1)
        TSpan (1,2) double = [0 0]
        P (:,1) double = zeros(0,1)
        Stop (1,1) string = ""
        Warnings (:,1) string = strings(0,1)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
    end

    properties (SetAccess = private, Hidden)
        % The run that produced the trajectory. A session stores the run.
        Record phasecont.internal.RunRecord {mustBeScalarOrEmpty} = phasecont.internal.RunRecord.empty
    end

    methods (Hidden)
        function attachRecord(obj, record)
            % Internal use: remember the run that produced the trajectory.
            obj.Record = record;
        end
    end

    methods
        function obj = Trajectory(values)
            % Internal use: values is a struct with the property names as fields.
            arguments
                values (1,1) struct = struct()
            end
            for name = string(fieldnames(values))'
                obj.(name) = values.(name);
            end
        end

        function h = plot(tr, options)
            % H = PLOT(TR) draws the path in the plane of the first two
            % variables, or x against t for a model with one variable.
            % PLOT(TR, Var = "x") draws that variable against t, and Var =
            % ["x" "y"] draws y against x. A variable is a name or an index
            % into Model.Variables. PLOT(TR, Parent = AX) draws into AX
            % without resetting it. TR may be a 1-by-K array. Each path is
            % a line with Tag "pc-trajectory" and a marker at its start
            % (Tag "pc-start"). Without Parent the plot uses newplot, so
            % hold on is respected. H holds the lines and markers as a
            % column vector.
            arguments
                tr phasecont.Trajectory
                options.Var = []
                options.Parent = []
            end
            h = gobjects(0, 1);
            if isempty(tr)
                return
            end
            if isempty(options.Parent)
                ax = newplot;
            else
                ax = options.Parent;
            end
            for k = 1:numel(tr)
                h = [h; tr(k).drawOne(ax, options.Var)]; %#ok<AGROW>
            end
            if nargout == 0
                clear h
            end
        end
    end

    methods (Access = private)
        function h = drawOne(tr, ax, var)
            % Draw this trajectory on ax: one line and a marker at the start.
            n = size(tr.X, 2);
            if isempty(var)
                if n >= 2
                    idx = [1 2];
                else
                    idx = 1;
                end
            else
                idx = tr.variableIndex(var);
            end
            if isscalar(idx)
                x = tr.T;
                y = tr.X(:, idx);
                xlab = "t";
                ylab = tr.variableName(idx);
            else
                x = tr.X(:, idx(1));
                y = tr.X(:, idx(2));
                xlab = tr.variableName(idx(1));
                ylab = tr.variableName(idx(2));
            end
            wasHeld = ishold(ax);
            hold(ax, "on");
            color = nextColor(ax);
            line = plot(ax, x, y, "-", Color = color, Tag = "pc-trajectory");
            mark = plot(ax, x(1), y(1), "o", Color = color, ...
                MarkerFaceColor = color, MarkerSize = 5, Tag = "pc-start");
            if ~wasHeld
                hold(ax, "off");
            end
            xlabel(ax, xlab);
            ylabel(ax, ylab);
            h = [line; mark];
        end

        function idx = variableIndex(tr, var)
            % One or two indices into the variables, from names or numbers.
            names = tr.variableNames();
            n = numel(names);
            if isnumeric(var)
                idx = double(var(:)).';
                ok = isreal(idx) && all(idx >= 1 & idx <= n & idx == round(idx));
            else
                v = string(var);
                idx = zeros(1, numel(v));
                ok = true;
                for k = 1:numel(v)
                    hit = find(names == v(k), 1);
                    if isempty(hit)
                        ok = false;
                    else
                        idx(k) = hit;
                    end
                end
            end
            if ~ok || ~any(numel(idx) == [1 2])
                error("phasecont:model:unknownVariable", ...
                    "Var must be one or two of: %s.", strjoin(names, ", "));
            end
        end

        function names = variableNames(tr)
            % The variable names, or x1, x2, ... when the Model is gone.
            if isempty(tr.Model) || ~isvalid(tr.Model)
                names = "x" + (1:size(tr.X, 2));
            else
                names = tr.Model.Variables(:).';
            end
        end

        function name = variableName(tr, vi)
            names = tr.variableNames();
            name = names(vi);
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
