classdef PhasePlane < handle
    % An interactive phase plane of a model, in a uifigure.
    %
    %   app = phasecont.app.PhasePlane(m)
    %   app = phasecont.app.PhasePlane("diagram.pcsession.json")
    %
    % M is a phasecont.Model with at least two variables. A session file is
    % loaded with phasecont.Session.load and its trajectories and
    % equilibria are drawn. A click in the plot starts a run from that
    % point (the other variables take the slice values): Click mode
    % "Integrate" integrates the model, "Equilibrium" runs Newton's method
    % from the point and marks the result, filled when every eigenvalue has
    % a negative real part and open otherwise. The check boxes draw a
    % normalized direction field on a 20-by-20 grid and the nullclines on a
    % 101-by-101 grid. The model must be autonomous for the field, the
    % nullclines and the equilibria.
    %
    % Settings that come from the model
    %   @ xp, yp      the variables on the two axes (default: the first two)
    %   @ xlo, xhi    the x range, when both are set and xlo < xhi
    %   @ ylo, yhi    the y range, same rule. The ranges belong to the pair
    %                 the model names (the first two variables when it
    %                 names none); any other pair starts at
    %                 Init +/- max(1, |Init|)
    %   @ total       the integration length T. Without it the T field is
    %                 empty and a click asks for T instead of integrating.
    %   @ meth, dt, bounds
    %                 not used. The app integrates with the Dormand-Prince
    %                 5(4) pair and a note says so.
    % The view range only changes the display. Each integration uses
    % Hmax = T/500 so the drawn lines stay smooth.
    %
    % A model that uses t integrates with t running from 0 to T (Forward),
    % 0 to -T (Backward) or both. Changing a parameter removes the
    % trajectories and equilibria, because they belong to the old values,
    % and redraws the field and the nullclines. Branches loaded from a
    % session are kept and saved again but not drawn.
    %
    % Graphics tags: pc-trajectory, pc-start, pc-equilibrium-stable,
    % pc-equilibrium-unstable, pc-field, pc-nullcline (one line per
    % variable, with the variable name in UserData).
    %
    % Methods
    %   clickAt, addTrajectory, findEquilibrium, setParameter, setSlice,
    %   setAxes, setWindow, setT, setDirection, setClickMode, showField,
    %   showNullclines, clear, saveSession, openSession, delete
    %
    % Properties (read only)
    %   Model, Items, Params, Slice, XIndex, YIndex, Window, T, Direction,
    %   ClickMode, Notes, Status, Figure, Axes

    properties (SetAccess = private)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
        % Trajectory, Point and Branch objects the app holds, in the order
        % they were added or loaded.
        Items (1,:) cell = cell(1, 0)
        Params (:,1) double = zeros(0, 1)
        % One value per variable. The entries of the two plotted variables
        % are not used; the others hold the slice.
        Slice (:,1) double = zeros(0, 1)
        XIndex (1,1) double = 1
        YIndex (1,1) double = 2
        % [xlo xhi ylo yhi] of the view
        Window (1,4) double = [-1 1 -1 1]
        % Integration length, or empty when it is not known
        T double {mustBeScalarOrEmpty} = []
        Direction (1,1) string = "Forward"
        ClickMode (1,1) string = "Integrate"
        Notes (:,1) string = strings(0, 1)
        Status (1,1) string = ""
        Figure = []
        Axes = []
    end

    properties (Access = private)
        Ui (1,1) struct = struct()
        OwnsModel (1,1) logical = false
        % The pair of variables that @ xp, @ yp and the @ xlo.. range name.
        OptionPair (1,2) double = [1 2]
        FieldOn (1,1) logical = false
        NullclinesOn (1,1) logical = false
    end

    properties (Constant, Access = private)
        FieldGrid = 20
        NullclineGrid = 101
        HmaxDivisor = 500
    end

    methods
        function app = PhasePlane(source, options)
            % APP = PhasePlane(M) or PhasePlane(FILE). Visible = "off"
            % builds the window without showing it.
            arguments
                source
                options.Visible (1,1) string {mustBeMember(options.Visible, ["on" "off"])} = "on"
            end
            items = cell(1, 0);
            if isa(source, "phasecont.Model") && isscalar(source)
                m = source;
                owns = false;
            elseif (isstring(source) || ischar(source)) && isscalar(string(source))
                S = phasecont.Session.load(string(source));
                m = S.Model;
                items = S.Items;
                owns = true;
            else
                error("phasecont:app:badSource", ...
                    "PhasePlane takes a phasecont.Model or the name of a session file.");
            end
            if numel(m.Variables) < 2
                if owns
                    delete(m);
                end
                error("phasecont:app:needTwoStates", ...
                    "A phase plane needs a model with at least two variables.");
            end
            app.Model = m;
            app.OwnsModel = owns;
            app.Items = items;
            app.resetToModel();
            app.buildUi(options.Visible);
            app.refreshControls();
            app.redraw();
            if ~isempty(items)
                app.setStatus(app.describeLoaded());
            else
                app.setStatus("Click in the plot to start a run.");
            end
        end

        function tr = clickAt(app, x, y)
            % TR = CLICKAT(APP, X, Y) does what a click at (X, Y) does:
            % integrates in Click mode "Integrate" and returns the
            % Trajectory objects, or finds an equilibrium in mode
            % "Equilibrium" and returns the Point.
            if app.ClickMode == "Equilibrium"
                tr = app.findEquilibrium(x, y);
            else
                tr = app.addTrajectory(x, y);
            end
        end

        function tr = addTrajectory(app, x, y)
            % TR = ADDTRAJECTORY(APP, X, Y) integrates from the point
            % (X, Y) in the plane, with the other variables at the slice
            % values, in the chosen direction, and draws the path. TR is a
            % 1-by-k array of Trajectory objects, k = 2 for "Both". The
            % error phasecont:app:missingTime means T is not set.
            arguments
                app (1,1) phasecont.app.PhasePlane
                x (1,1) double {mustBeReal, mustBeFinite}
                y (1,1) double {mustBeReal, mustBeFinite}
            end
            if isempty(app.T)
                error("phasecont:app:missingTime", ...
                    "The model has no @ total. Enter the integration length T first.");
            end
            T = app.T;
            switch app.Direction
                case "Forward"
                    spans = {[0 T]};
                case "Backward"
                    spans = {[0 -T]};
                otherwise
                    spans = {[0 T], [0 -T]};
            end
            x0 = app.startState(x, y);
            ws = warning("off", "phasecont:simulate:stoppedEarly");
            restore = onCleanup(@() warning(ws));
            runs = cell(1, numel(spans));
            for k = 1:numel(spans)
                runs{k} = phasecont.simulate(app.Model, x0, spans{k}, ...
                    Params = app.Params, Hmax = T / app.HmaxDivisor);
            end
            tr = [runs{:}];
            msg = sprintf("Integrated from (%.4g, %.4g) over |t| up to %.4g.", x, y, T);
            for k = 1:numel(tr)
                app.Items{end+1} = tr(k);
                app.drawItem(tr(k));
                if tr(k).Stop ~= "End"
                    msg = msg + " Stopped early at t = " + sprintf("%.4g", tr(k).T(end)) ...
                        + " (" + tr(k).Stop + ").";
                end
            end
            app.setStatus(msg);
        end

        function pt = findEquilibrium(app, x, y)
            % PT = FINDEQUILIBRIUM(APP, X, Y) runs Newton's method from the
            % point (X, Y), with the other variables at the slice values,
            % and marks the equilibrium. The marker is filled when every
            % eigenvalue has a negative real part and open otherwise. A
            % failure to converge raises the engine error.
            arguments
                app (1,1) phasecont.app.PhasePlane
                x (1,1) double {mustBeReal, mustBeFinite}
                y (1,1) double {mustBeReal, mustBeFinite}
            end
            app.requireAutonomous(true, "equilibrium search");
            pt = phasecont.equilibrium(app.Model, app.startState(x, y), ...
                Params = app.Params);
            app.Items{end+1} = pt;
            app.drawItem(pt);
            top = max(real(pt.Eigenvalues));
            if top < 0
                kind = "stable";
            else
                kind = "unstable";
            end
            app.setStatus(sprintf( ...
                "Equilibrium at (%.6g, %.6g): %s, largest real part %.3g.", ...
                pt.U(app.XIndex), pt.U(app.YIndex), kind, top));
        end

        function setParameter(app, name, value)
            % SETPARAMETER(APP, NAME, VALUE) sets a parameter, removes the
            % trajectories and equilibria and redraws the field and the
            % nullclines.
            arguments
                app (1,1) phasecont.app.PhasePlane
                name (1,1) string
                value (1,1) double {mustBeReal, mustBeFinite}
            end
            idx = find(app.Model.Parameters == name, 1);
            if isempty(idx)
                error("phasecont:app:unknownParameter", ...
                    "The model has no parameter named ""%s"".", name);
            end
            app.Params(idx) = value;
            removed = app.dropRuns();
            app.refreshControls();
            app.redraw();
            msg = sprintf("%s = %.6g.", name, value);
            if removed > 0
                msg = msg + sprintf(" Removed %d trajectories and equilibria computed with the old values.", removed);
            end
            app.setStatus(msg);
        end

        function setSlice(app, name, value)
            % SETSLICE(APP, NAME, VALUE) sets the value that variable NAME
            % takes in new runs and in the field and nullclines.
            arguments
                app (1,1) phasecont.app.PhasePlane
                name (1,1) string
                value (1,1) double {mustBeReal, mustBeFinite}
            end
            idx = app.variableIndex(name);
            app.Slice(idx) = value;
            app.refreshControls();
            app.redrawBackground();
            app.setStatus(sprintf("Slice value %s = %.6g.", name, value));
        end

        function setAxes(app, xname, yname)
            % SETAXES(APP, XNAME, YNAME) plots variable XNAME against YNAME
            % and resets the view to the default range of that pair.
            arguments
                app (1,1) phasecont.app.PhasePlane
                xname (1,1) string
                yname (1,1) string
            end
            ix = app.variableIndex(xname);
            iy = app.variableIndex(yname);
            if ix == iy
                error("phasecont:app:sameAxes", ...
                    "The two axes need different variables.");
            end
            app.XIndex = ix;
            app.YIndex = iy;
            app.Window = app.defaultWindow(ix, iy);
            app.refreshControls();
            app.redraw();
            app.setStatus(sprintf("Plotting %s against %s.", yname, xname));
        end

        function setWindow(app, window)
            % SETWINDOW(APP, [XLO XHI YLO YHI]) sets the view. It changes
            % the display only.
            arguments
                app (1,1) phasecont.app.PhasePlane
                window (1,4) double {mustBeReal, mustBeFinite}
            end
            if window(1) >= window(2) || window(3) >= window(4)
                error("phasecont:app:badWindow", ...
                    "The view needs xlo < xhi and ylo < yhi.");
            end
            app.Window = window;
            app.refreshControls();
            app.redrawBackground();
        end

        function setT(app, value)
            % SETT(APP, T) sets the integration length, a positive number.
            % SETT(APP, []) clears it.
            if ~isempty(value) && ~(isnumeric(value) && isreal(value) ...
                    && isscalar(value) && isfinite(value) && value > 0)
                error("phasecont:app:badValue", ...
                    "T must be a positive finite number, or empty.");
            end
            app.T = double(value);
            app.refreshControls();
        end

        function setDirection(app, direction)
            % SETDIRECTION(APP, "Forward" | "Backward" | "Both")
            arguments
                app (1,1) phasecont.app.PhasePlane
                direction (1,1) string {mustBeMember(direction, ["Forward" "Backward" "Both"])}
            end
            app.Direction = direction;
            app.refreshControls();
        end

        function setClickMode(app, mode)
            % SETCLICKMODE(APP, "Integrate" | "Equilibrium")
            arguments
                app (1,1) phasecont.app.PhasePlane
                mode (1,1) string {mustBeMember(mode, ["Integrate" "Equilibrium"])}
            end
            app.ClickMode = mode;
            app.refreshControls();
        end

        function showField(app, on)
            % SHOWFIELD(APP, TF) draws or removes the direction field.
            arguments
                app (1,1) phasecont.app.PhasePlane
                on (1,1) logical
            end
            app.requireAutonomous(on, "direction field");
            app.FieldOn = on;
            app.refreshControls();
            app.redrawBackground();
        end

        function showNullclines(app, on)
            % SHOWNULLCLINES(APP, TF) draws or removes the nullclines.
            arguments
                app (1,1) phasecont.app.PhasePlane
                on (1,1) logical
            end
            app.requireAutonomous(on, "nullclines");
            app.NullclinesOn = on;
            app.refreshControls();
            app.redrawBackground();
        end

        function clear(app)
            % CLEAR(APP) removes every item the app holds.
            count = numel(app.Items);
            app.Items = cell(1, 0);
            app.redrawItems();
            app.setStatus(sprintf("Cleared %d items.", count));
        end

        function saveSession(app, file)
            % SAVESESSION(APP, FILE) saves the model and every item the
            % app holds as a session file.
            arguments
                app (1,1) phasecont.app.PhasePlane
                file (1,1) string
            end
            s = phasecont.Session(app.Model, app.Items);
            save(s, file);
            app.setStatus(sprintf("Saved %d items to %s.", numel(app.Items), file));
        end

        function openSession(app, file)
            % OPENSESSION(APP, FILE) replaces the model and the items with
            % those of a session file. Nothing runs; the stored rows are
            % drawn.
            arguments
                app (1,1) phasecont.app.PhasePlane
                file (1,1) string
            end
            S = phasecont.Session.load(file);
            if numel(S.Model.Variables) < 2
                delete(S.Model);
                error("phasecont:app:needTwoStates", ...
                    "A phase plane needs a model with at least two variables.");
            end
            old = app.Model;
            ownedOld = app.OwnsModel;
            app.Model = S.Model;
            app.OwnsModel = true;
            app.Items = S.Items;
            if ownedOld
                delete(old);
            end
            app.resetToModel();
            app.refreshControls();
            app.redraw();
            app.setStatus(app.describeLoaded());
        end

        function delete(app)
            % Close the window and free a model that the app loaded itself.
            if ~isempty(app.Figure) && isvalid(app.Figure)
                delete(app.Figure);
            end
            if app.OwnsModel && ~isempty(app.Model) && isvalid(app.Model)
                delete(app.Model);
            end
        end
    end

    methods (Access = private)
        function resetToModel(app)
            % Take the state of the app from the model: parameters, slice,
            % axes, view, T and the notes.
            m = app.Model;
            n = numel(m.Variables);
            notes = strings(0, 1);
            o = m.Options;
            app.Params = m.ParameterValues;
            init = m.Init;
            if numel(init) ~= n
                init = zeros(n, 1);
            end
            app.Slice = init;

            ix = 1;
            iy = 2;
            if isfield(o, "xp") || isfield(o, "yp")
                [ix, iy, notes] = app.axesFromOptions(o, notes);
            end
            app.XIndex = ix;
            app.YIndex = iy;
            app.OptionPair = [ix iy];
            [app.Window, notes] = app.windowFor(ix, iy, notes);

            app.T = [];
            if isfield(o, "total")
                v = str2double(o.total);
                if isfinite(v) && v > 0
                    app.T = v;
                else
                    notes(end+1, 1) = "@ total = " + o.total + " is not a positive number; enter T.";
                end
            end
            ignored = intersect(["meth" "dt" "bounds"], string(fieldnames(o)).');
            if ~isempty(ignored)
                notes(end+1, 1) = "The model sets @ " + strjoin(ignored, ", ") ...
                    + ". The app integrates with Dormand-Prince 5(4) and ignores them.";
            end
            if ~m.Autonomous
                notes(end+1, 1) = "The model uses t. Runs start at t = 0. " ...
                    + "The field, the nullclines and the equilibria need an autonomous model.";
            end
            app.Notes = notes;
            app.FieldOn = false;
            app.NullclinesOn = false;

            % Show the parameters the stored runs used.
            drawn = app.Items(cellfun(@(c) isa(c, "phasecont.Trajectory") ...
                || isa(c, "phasecont.Point"), app.Items));
            if ~isempty(drawn)
                first = drawn{1}.P(:);
                if numel(first) == numel(app.Params)
                    app.Params = first;
                end
                same = cellfun(@(c) isequal(c.P(:), first), drawn);
                if ~all(same)
                    app.Notes(end+1, 1) = "The stored runs use different parameter values. " ...
                        + "The table shows those of the first run.";
                end
            end
        end

        function [ix, iy, notes] = axesFromOptions(app, o, notes)
            % The axes named by @ xp and @ yp, with a note for each name
            % that is not a variable.
            vars = app.Model.Variables;
            ix = 1;
            iy = 2;
            hx = [];
            hy = [];
            if isfield(o, "xp")
                hx = find(strcmpi(vars, o.xp), 1);
                if isempty(hx)
                    notes(end+1, 1) = "@ xp = " + o.xp + " is not a variable.";
                end
            end
            if isfield(o, "yp")
                hy = find(strcmpi(vars, o.yp), 1);
                if isempty(hy)
                    notes(end+1, 1) = "@ yp = " + o.yp + " is not a variable.";
                end
            end
            if ~isempty(hx)
                ix = hx;
            end
            if ~isempty(hy)
                iy = hy;
            end
            if ix == iy
                % Either the names agree or one name is missing. Move y to
                % the first variable that is not x.
                iy = find((1:numel(vars)) ~= ix, 1);
                notes(end+1, 1) = "The axes from @ xp and @ yp coincide; plotting " ...
                    + vars(iy) + " on the y axis.";
            end
        end

        function [w, notes] = windowFor(app, ix, iy, notes)
            % The view for the pair (ix, iy): the @ xlo.. range when the
            % pair is the one the model names, else Init +/- max(1, |Init|).
            c = app.Slice([ix iy]);
            r = max(1, abs(c));
            w = [c(1) - r(1), c(1) + r(1), c(2) - r(2), c(2) + r(2)];
            if ~isequal([ix iy], app.OptionPair)
                return
            end
            o = app.Model.Options;
            [xr, notes] = rangeFrom(o, "xlo", "xhi", notes);
            [yr, notes] = rangeFrom(o, "ylo", "yhi", notes);
            if ~isempty(xr)
                w(1:2) = xr;
            end
            if ~isempty(yr)
                w(3:4) = yr;
            end
        end

        function w = defaultWindow(app, ix, iy)
            w = app.windowFor(ix, iy, strings(0, 1));
        end

        function idx = variableIndex(app, name)
            idx = find(app.Model.Variables == name, 1);
            if isempty(idx)
                error("phasecont:app:unknownVariable", ...
                    "The model has no variable named ""%s"". The variables are: %s.", ...
                    name, strjoin(app.Model.Variables, ", "));
            end
        end

        function x0 = startState(app, x, y)
            x0 = app.Slice;
            x0(app.XIndex) = x;
            x0(app.YIndex) = y;
        end

        function requireAutonomous(app, on, what)
            if on && ~app.Model.Autonomous
                error("phasecont:app:nonAutonomous", ...
                    "The %s needs an autonomous model, and this model uses t.", what);
            end
        end

        function removed = dropRuns(app)
            % Remove the trajectories and equilibria; keep the branches.
            isRun = cellfun(@(c) isa(c, "phasecont.Trajectory") ...
                || isa(c, "phasecont.Point"), app.Items);
            removed = nnz(isRun);
            app.Items = app.Items(~isRun);
        end

        function text = describeLoaded(app)
            nTr = sum(cellfun(@(c) isa(c, "phasecont.Trajectory"), app.Items));
            nEq = sum(cellfun(@(c) isa(c, "phasecont.Point"), app.Items));
            nBr = sum(cellfun(@(c) isa(c, "phasecont.Branch"), app.Items));
            text = sprintf("Loaded %d trajectories and %d equilibria.", nTr, nEq);
            if nBr > 0
                text = text + sprintf(" %d branches are kept and saved again but not drawn.", nBr);
            end
        end

        function setStatus(app, text)
            app.Status = text;
            if isfield(app.Ui, "Status") && isvalid(app.Ui.Status)
                app.Ui.Status.Text = char(text);
            end
        end

        function report(app, err)
            % Show an error from a callback.
            app.setStatus(string(err.message));
            if ~isempty(app.Figure) && isvalid(app.Figure) && logical(app.Figure.Visible)
                uialert(app.Figure, err.message, "PhaseCont", Icon = "warning");
            end
        end

        % ---- drawing ----

        function redraw(app)
            ax = app.Axes;
            ax.XLim = app.Window(1:2);
            ax.YLim = app.Window(3:4);
            xlabel(ax, app.Model.Variables(app.XIndex));
            ylabel(ax, app.Model.Variables(app.YIndex));
            app.redrawBackground();
            app.redrawItems();
        end

        function redrawItems(app)
            ax = app.Axes;
            delete(findobj(ax, "-regexp", "Tag", "^pc-(trajectory|start|equilibrium-.*)$"));
            for k = 1:numel(app.Items)
                app.drawItem(app.Items{k});
            end
        end

        function drawItem(app, it)
            ax = app.Axes;
            if isa(it, "phasecont.Trajectory")
                h = plot(it, Parent = ax, Var = [app.XIndex app.YIndex]);
                set(h, PickableParts = "none");
            elseif isa(it, "phasecont.Point")
                if all(real(it.Eigenvalues) < 0)
                    tag = "pc-equilibrium-stable";
                    face = [0 0 0];
                else
                    tag = "pc-equilibrium-unstable";
                    face = "none";
                end
                line(ax, it.U(app.XIndex), it.U(app.YIndex), Marker = "o", ...
                    MarkerSize = 8, LineStyle = "none", Color = [0 0 0], ...
                    MarkerFaceColor = face, Tag = tag, PickableParts = "none");
            end
        end

        function redrawBackground(app)
            ax = app.Axes;
            ax.XLim = app.Window(1:2);
            ax.YLim = app.Window(3:4);
            delete(findobj(ax, "Tag", "pc-field"));
            delete(findobj(ax, "Tag", "pc-nullcline"));
            if app.NullclinesOn
                app.drawNullclines();
            end
            if app.FieldOn
                app.drawField();
            end
        end

        function drawField(app)
            ax = app.Axes;
            [F, X, Y] = app.gridRhs(app.FieldGrid, app.FieldGrid);
            w = app.Window;
            sx = F(app.XIndex, :) / (w(2) - w(1));
            sy = F(app.YIndex, :) / (w(4) - w(3));
            len = hypot(sx, sy);
            good = isfinite(len) & len > 0;
            u = zeros(size(sx));
            v = zeros(size(sy));
            u(good) = sx(good) ./ len(good);
            v(good) = sy(good) ./ len(good);
            dx = 0.8 * (w(2) - w(1)) / (app.FieldGrid - 1);
            dy = 0.8 * (w(4) - w(3)) / (app.FieldGrid - 1);
            h = quiver(ax, X, Y, reshape(u * dx, size(X)), reshape(v * dy, size(Y)), ...
                AutoScale = "off", Color = [0.55 0.55 0.55], Tag = "pc-field", ...
                PickableParts = "none");
            app.sendToBottom(h);
        end

        function drawNullclines(app)
            ax = app.Axes;
            [F, X, ~] = app.gridRhs(app.NullclineGrid, app.NullclineGrid);
            xs = linspace(app.Window(1), app.Window(2), app.NullclineGrid);
            ys = linspace(app.Window(3), app.Window(4), app.NullclineGrid);
            idx = [app.XIndex app.YIndex];
            colors = [0.85 0.33 0.10; 0.00 0.45 0.74];
            for k = 1:2
                Z = reshape(F(idx(k), :), size(X));
                Z(~isfinite(Z)) = NaN;
                xy = nullclineLines(xs, ys, Z);
                h = line(ax, xy(:, 1), xy(:, 2), Color = colors(k, :), ...
                    LineWidth = 1.2, Tag = "pc-nullcline", ...
                    UserData = app.Model.Variables(idx(k)), PickableParts = "none");
                app.sendToBottom(h);
            end
        end

        function [F, X, Y] = gridRhs(app, nx, ny)
            % The right-hand side on an ny-by-nx grid of the plane, other
            % variables at the slice values. F is n-by-(nx*ny).
            xs = linspace(app.Window(1), app.Window(2), nx);
            ys = linspace(app.Window(3), app.Window(4), ny);
            [X, Y] = meshgrid(xs, ys);
            S = repmat(app.Slice, 1, numel(X));
            S(app.XIndex, :) = X(:).';
            S(app.YIndex, :) = Y(:).';
            F = app.Model.rhs(S, app.Params);
        end

        function sendToBottom(app, h)
            ax = app.Axes;
            kids = ax.Children;
            ax.Children = [kids(kids ~= h); h];
        end

        % ---- controls ----

        function buildUi(app, visible)
            fig = uifigure(Visible = visible, Name = "PhaseCont phase plane", ...
                Position = [100 100 1100 700]);
            fig.CloseRequestFcn = @(~, ~) delete(app);
            root = uigridlayout(fig, [1 2]);
            root.ColumnWidth = {330, "1x"};
            root.RowHeight = {"1x"};
            left = uigridlayout(root, [16 2]);
            left.Layout.Row = 1;
            left.Layout.Column = 1;
            left.ColumnWidth = {80, "1x"};
            left.RowHeight = [repmat({22}, 1, 11), {"1x"}, {90}, {28}, {64}, {48}];
            left.Padding = [6 6 6 6];
            left.RowSpacing = 4;
            ui = struct();

            ui.XDrop = uidropdown(left, ValueChangedFcn = @(~, ~) app.onAxesChanged());
            place(uilabel(left, Text = "x axis"), 1, 1);
            place(ui.XDrop, 1, 2);
            ui.YDrop = uidropdown(left, ValueChangedFcn = @(~, ~) app.onAxesChanged());
            place(uilabel(left, Text = "y axis"), 2, 1);
            place(ui.YDrop, 2, 2);
            ui.TField = uieditfield(left, "text", ...
                ValueChangedFcn = @(~, ~) app.onTChanged());
            place(uilabel(left, Text = "T"), 3, 1);
            place(ui.TField, 3, 2);
            ui.DirDrop = uidropdown(left, Items = {'Forward', 'Backward', 'Both'}, ...
                ValueChangedFcn = @(~, ~) app.onDirection());
            place(uilabel(left, Text = "Direction"), 4, 1);
            place(ui.DirDrop, 4, 2);
            ui.ModeDrop = uidropdown(left, Items = {'Integrate', 'Equilibrium'}, ...
                ValueChangedFcn = @(~, ~) app.onClickMode());
            place(uilabel(left, Text = "Click"), 5, 1);
            place(ui.ModeDrop, 5, 2);
            names = ["XLo" "XHi" "YLo" "YHi"];
            texts = ["x from" "x to" "y from" "y to"];
            for k = 1:4
                ui.(names(k)) = uieditfield(left, "numeric", ...
                    ValueChangedFcn = @(~, ~) app.onWindowChanged());
                place(uilabel(left, Text = texts(k)), 5 + k, 1);
                place(ui.(names(k)), 5 + k, 2);
            end
            ui.FieldBox = uicheckbox(left, Text = "Direction field", ...
                ValueChangedFcn = @(~, ~) app.onField());
            place(ui.FieldBox, 10, [1 2]);
            ui.NullBox = uicheckbox(left, Text = "Nullclines", ...
                ValueChangedFcn = @(~, ~) app.onNullclines());
            place(ui.NullBox, 11, [1 2]);
            ui.ParamTable = uitable(left, ColumnEditable = [false true], ...
                ColumnName = {'Parameter', 'Value'}, RowName = {}, ...
                CellEditCallback = @(src, ev) app.onParamEdited(src, ev));
            place(ui.ParamTable, 12, [1 2]);
            ui.SliceTable = uitable(left, ColumnEditable = [false true], ...
                ColumnName = {'Slice', 'Value'}, RowName = {}, ...
                CellEditCallback = @(src, ev) app.onSliceEdited(src, ev));
            place(ui.SliceTable, 13, [1 2]);
            buttons = uigridlayout(left, [1 3]);
            buttons.Padding = [0 0 0 0];
            place(buttons, 14, [1 2]);
            ui.Clear = uibutton(buttons, Text = "Clear", ...
                ButtonPushedFcn = @(~, ~) app.onClear());
            ui.Save = uibutton(buttons, Text = "Save session", ...
                ButtonPushedFcn = @(~, ~) app.onSave());
            ui.Open = uibutton(buttons, Text = "Open session", ...
                ButtonPushedFcn = @(~, ~) app.onOpen());
            ui.Note = uilabel(left, Text = "", WordWrap = "on", VerticalAlignment = "top");
            place(ui.Note, 15, [1 2]);
            ui.Status = uilabel(left, Text = "", WordWrap = "on", VerticalAlignment = "top", ...
                FontWeight = "bold");
            place(ui.Status, 16, [1 2]);
            ui.Left = left;

            ax = uiaxes(root);
            ax.Layout.Row = 1;
            ax.Layout.Column = 2;
            ax.NextPlot = "add";
            ax.Box = "on";
            grid(ax, "on");
            ax.Interactions = [];
            ax.Toolbar.Visible = "off";
            ax.ButtonDownFcn = @(~, ~) app.onClick();

            app.Ui = ui;
            app.Figure = fig;
            app.Axes = ax;
        end

        function refreshControls(app)
            % Put the state of the app into the controls. Setting a Value
            % in code does not call the callbacks.
            ui = app.Ui;
            m = app.Model;
            vars = cellstr(m.Variables);
            ui.XDrop.Items = vars;
            ui.XDrop.Value = vars{app.XIndex};
            ui.YDrop.Items = vars;
            ui.YDrop.Value = vars{app.YIndex};
            if isempty(app.T)
                ui.TField.Value = '';
            else
                ui.TField.Value = sprintf('%.12g', app.T);
            end
            ui.DirDrop.Value = char(app.Direction);
            ui.ModeDrop.Value = char(app.ClickMode);
            ui.XLo.Value = app.Window(1);
            ui.XHi.Value = app.Window(2);
            ui.YLo.Value = app.Window(3);
            ui.YHi.Value = app.Window(4);
            ui.FieldBox.Value = app.FieldOn;
            ui.NullBox.Value = app.NullclinesOn;
            auto = m.Autonomous;
            ui.FieldBox.Enable = auto;
            ui.NullBox.Enable = auto;
            ui.ParamTable.Data = table(m.Parameters(:), app.Params(:), ...
                VariableNames = {'Parameter', 'Value'});
            others = setdiff(1:numel(vars), [app.XIndex app.YIndex], "stable");
            sliceNames = m.Variables(others);
            sliceValues = app.Slice(others);
            ui.SliceTable.Data = table(sliceNames(:), sliceValues(:), ...
                VariableNames = {'Slice', 'Value'});
            hasSlice = ~isempty(others);
            ui.SliceTable.Visible = hasSlice;
            ui.Left.RowHeight{13} = 90 * hasSlice;
            ui.Note.Text = char(strjoin(app.Notes, newline));
            ui.Status.Text = char(app.Status);
        end

        % ---- callbacks ----

        function guard(app, action)
            % Run a callback body. A failure goes to the status line and
            % the controls go back to the state of the app.
            try
                action();
            catch err
                app.refreshControls();
                app.report(err);
            end
        end

        function onClick(app)
            cp = app.Axes.CurrentPoint;
            app.guard(@() app.clickAt(cp(1, 1), cp(1, 2)));
        end

        function onAxesChanged(app)
            app.guard(@() app.setAxes(string(app.Ui.XDrop.Value), ...
                string(app.Ui.YDrop.Value)));
        end

        function onTChanged(app)
            text = strtrim(string(app.Ui.TField.Value));
            if text == ""
                app.guard(@() app.setT([]));
            else
                app.guard(@() app.setT(str2double(text)));
            end
        end

        function onDirection(app)
            app.guard(@() app.setDirection(string(app.Ui.DirDrop.Value)));
        end

        function onClickMode(app)
            app.guard(@() app.setClickMode(string(app.Ui.ModeDrop.Value)));
        end

        function onField(app)
            app.guard(@() app.showField(logical(app.Ui.FieldBox.Value)));
        end

        function onNullclines(app)
            app.guard(@() app.showNullclines(logical(app.Ui.NullBox.Value)));
        end

        function onWindowChanged(app)
            ui = app.Ui;
            app.guard(@() app.setWindow([ui.XLo.Value ui.XHi.Value ...
                ui.YLo.Value ui.YHi.Value]));
        end

        function onParamEdited(app, src, ev)
            row = ev.Indices(1);
            name = string(src.Data.Parameter(row));
            app.guard(@() app.setParameter(name, ev.NewData));
        end

        function onSliceEdited(app, src, ev)
            row = ev.Indices(1);
            name = string(src.Data.Slice(row));
            app.guard(@() app.setSlice(name, ev.NewData));
        end

        function onClear(app)
            app.clear();
        end

        function onSave(app)
            [f, d] = uiputfile("*.json", "Save session", "session.pcsession.json");
            figure(app.Figure);
            if isequal(f, 0)
                return
            end
            app.guard(@() app.saveSession(fullfile(d, f)));
        end

        function onOpen(app)
            [f, d] = uigetfile("*.json", "Open session");
            figure(app.Figure);
            if isequal(f, 0)
                return
            end
            app.guard(@() app.openSession(fullfile(d, f)));
        end
    end

end

function place(component, row, col)
% Put a component in a grid cell.
component.Layout.Row = row;
component.Layout.Column = col;
end

function [r, notes] = rangeFrom(o, lo, hi, notes)
% The range [lo hi] from two @ options, or [] with a note when it is not
% usable. Nothing is returned when neither option is set.
r = [];
hasLo = isfield(o, lo);
hasHi = isfield(o, hi);
if ~hasLo && ~hasHi
    return
end
if hasLo && hasHi
    a = str2double(o.(lo));
    b = str2double(o.(hi));
    if isfinite(a) && isfinite(b) && a < b
        r = [a b];
        return
    end
end
notes(end+1, 1) = "@ " + lo + " and @ " + hi + " need two numbers with " + lo ...
    + " < " + hi + "; using the default range.";
end

function xy = nullclineLines(xs, ys, Z)
% Points of the zero level of Z on the grid xs by ys, one polyline after
% the other with NaN rows between them.
xy = [NaN NaN];
try
    C = contourc(xs, ys, Z, [0 0]);
catch
    return
end
parts = {};
k = 1;
while k < size(C, 2)
    len = C(2, k);
    parts{end+1} = [C(:, k+1:k+len).'; NaN NaN]; %#ok<AGROW>
    k = k + len + 1;
end
if ~isempty(parts)
    xy = vertcat(parts{:});
end
end
