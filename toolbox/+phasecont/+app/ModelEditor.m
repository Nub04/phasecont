classdef ModelEditor < handle
    % A window for writing a model without XPP syntax.
    %
    %   app = phasecont.app.ModelEditor
    %   app = phasecont.app.ModelEditor(Visible = "off")
    %
    % Three tabs build a model, and Check compiles it:
    %
    %   Equations  equation lines (x' = ... or dx/dt = ...), a Parameters box
    %              and an Initial values box with one `name = value` per
    %              line (commas also separate entries; # and % start a
    %              comment). Values are numbers.
    %   Table      States (Name, Rhs, Init), Parameters (Name, Value) and
    %              Aux (Name, Expr), one row per entry.
    %   Source     a complete XPP/XPPAUT .ode text. Open file fills it.
    %
    % Nothing is defaulted. A state with no initial value, a parameter with
    % no value and a symbol that is not declared each give a diagnostic,
    % and the model is not built. A failed Check keeps the last good Model.
    %
    % Open file reads .ode, .xml and .sbml (SBML) and .m (a MATLAB function
    % file, which needs Init, and often Params and StateNames). The result
    % is shown on the Source tab as .ode text, so you can read and edit what
    % the importer made. Open in phase plane starts phasecont.app.PhasePlane
    % on the last good model. Save as .ode writes that model's source text.
    %
    % Diagnostics is a struct array with the fields Line, Column, Message
    % and Id. Line and Column are NaN when the message has no position. On
    % the Equations tab a line number is a line of the equation box. On the
    % Table tab it counts the States rows, then the Aux rows.
    %
    % Methods
    %   setEquations, setParameters, setInitial, setTable, setSource,
    %   selectTab, check, openFile, saveOde, openPhasePlane, delete
    %
    % Properties (read only)
    %   Model, Source, Diagnostics, Tab, Status, Figure

    properties (SetAccess = private)
        % The last model that built without a diagnostic, or empty.
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
        % The .ode text of Model.
        Source (1,1) string = ""
        Diagnostics (:,1) struct = repmat(struct("Line", NaN, "Column", NaN, ...
            "Message", "", "Id", ""), 0, 1)  % same fields as emptyDiagnostics
        % "Equations", "Table" or "Source"
        Tab (1,1) string = "Equations"
        Status (1,1) string = ""
        Figure = []
    end

    properties (Access = private)
        Ui (1,1) struct = struct()
        % Name used in diagnostics for the Source tab.
        SourceName (1,1) string = "source.ode"
        % Row counts of the States and Aux tables at the last build.
        TableRows (1,2) double = [0 0]
        Selected (1,1) struct = struct("States", [], "Params", [], "Aux", [])
    end

    methods
        function app = ModelEditor(options)
            % APP = ModelEditor builds the window. Visible = "off" builds
            % it without showing it.
            arguments
                options.Visible (1,1) string {mustBeMember(options.Visible, ["on" "off"])} = "on"
            end
            app.buildUi(options.Visible);
            app.setStatus("Enter equations, then click Check.");
        end

        function setEquations(app, text)
            % SETEQUATIONS(APP, TEXT) fills the equation box on the
            % Equations tab. TEXT is a string, or a string array with one
            % line per entry.
            arguments
                app (1,1) phasecont.app.ModelEditor
                text string {mustBeNonmissing}
            end
            app.Ui.Eq.Value = textLines(text);
        end

        function setParameters(app, text)
            % SETPARAMETERS(APP, TEXT) fills the Parameters box, for example
            % "w = 2" + newline + "c = 0.1".
            arguments
                app (1,1) phasecont.app.ModelEditor
                text string {mustBeNonmissing}
            end
            app.Ui.Par.Value = textLines(text);
        end

        function setInitial(app, text)
            % SETINITIAL(APP, TEXT) fills the Initial values box.
            arguments
                app (1,1) phasecont.app.ModelEditor
                text string {mustBeNonmissing}
            end
            app.Ui.Ini.Value = textLines(text);
        end

        function setTable(app, options)
            % SETTABLE(APP, States = T, Params = P, Aux = A) fills the
            % Table tab. Each argument is a table with the columns the tab
            % shows (Name and Rhs, and Init, for States; Name and Value for
            % Params; Name and Expr for Aux). An argument that is left out
            % keeps its rows.
            arguments
                app (1,1) phasecont.app.ModelEditor
                options.States table = table()
                options.Params table = table()
                options.Aux table = table()
            end
            given = ["States" "Params" "Aux"];
            cols = {["Name" "Rhs" "Init"], ["Name" "Value"], ["Name" "Expr"]};
            for k = 1:3
                t = options.(given(k));
                if width(t) == 0 && height(t) == 0
                    continue
                end
                app.Ui.(given(k)).Data = tableData(t, cols{k}, given(k));
            end
        end

        function setSource(app, text)
            % SETSOURCE(APP, TEXT) fills the Source tab with .ode text.
            arguments
                app (1,1) phasecont.app.ModelEditor
                text string {mustBeNonmissing}
            end
            app.Ui.Src.Value = textLines(text);
        end

        function selectTab(app, name)
            % SELECTTAB(APP, NAME) shows "Equations", "Table" or "Source".
            arguments
                app (1,1) phasecont.app.ModelEditor
                name (1,1) string {mustBeMember(name, ["Equations" "Table" "Source"])}
            end
            app.Ui.Group.SelectedTab = app.Ui.Tabs.(name);
            app.Tab = name;
        end

        function ok = check(app)
            % OK = CHECK(APP) builds the model from the tab that is
            % showing. On success Model and Source take the new model and
            % Diagnostics is empty. On failure Diagnostics lists the
            % problems and Model and Source keep their last good values. OK
            % tells which happened.
            try
                m = app.build();
            catch err
                if ~startsWith(string(err.identifier), "phasecont:")
                    rethrow(err);
                end
                app.Diagnostics = app.diagnose(err);
                app.showDiagnostics();
                n = numel(app.Diagnostics);
                if isempty(app.Model)
                    keep = "No model yet.";
                else
                    keep = "The last good model is kept.";
                end
                app.setStatus(sprintf("%d problem(s). %s", n, keep));
                ok = false;
                return
            end
            app.Model = m;
            app.Source = m.Source;
            app.Diagnostics = emptyDiagnostics();
            app.showDiagnostics();
            app.setStatus(sprintf("OK: %d variable(s) (%s), %d parameter(s).", ...
                numel(m.Variables), join(m.Variables, ", "), ...
                numel(m.Parameters)));
            ok = true;
        end

        function openFile(app, file, options)
            % OPENFILE(APP, FILE) reads an .ode, .xml, .sbml or .m file,
            % shows its .ode text on the Source tab and checks it. Init,
            % Params and StateNames apply to .m files only, as in
            % phasecont.Model.fromFile, and an .m file needs Init.
            arguments
                app (1,1) phasecont.app.ModelEditor
                file (1,1) string
                options.Init double = zeros(0, 1)
                options.Params (1,1) struct = struct()
                options.StateNames string = strings(0, 1)
            end
            [~, stem, ext] = fileparts(file);
            ext = lower(string(ext));
            given = ~isempty(options.Init) || ~isempty(fieldnames(options.Params)) ...
                || ~isempty(options.StateNames);
            if ext == ".m"
                [text, opts, skipped] = phasecont.internal.importRequest(file, ...
                    options.Init(:), options.Params, options.StateNames(:), "");
                ode = phasecont.internal.importText("matlab", text, opts, skipped);
            elseif any(ext == [".ode" ".xml" ".sbml"])
                if given
                    error("phasecont:model:badOption", ...
                        "Init, Params and StateNames apply to .m files only. " + ...
                        "A %s file holds its own values.", ext);
                end
                if ext == ".ode"
                    if ~isfile(file)
                        error("phasecont:import:file", ...
                            "Cannot open the file %s.", file);
                    end
                    ode = string(fileread(file, Encoding = "UTF-8"));
                else
                    ode = phasecont.internal.importText("sbml", ...
                        phasecont.internal.readSbml(file));
                end
            else
                error("phasecont:model:badOption", ...
                    "Cannot read a %s file. The extensions are .ode, .xml, " + ...
                    ".sbml and .m.", ext);
            end
            app.SourceName = string(stem) + ".ode";
            app.setSource(ode);
            app.selectTab("Source");
            app.check();
        end

        function saveOde(app, file)
            % SAVEODE(APP, FILE) writes the .ode text of the last good
            % model, as UTF-8. The error phasecont:app:noModel means Check
            % has not built a model yet.
            arguments
                app (1,1) phasecont.app.ModelEditor
                file (1,1) string
            end
            if isempty(app.Model)
                error("phasecont:app:noModel", ...
                    "There is no model to save. Click Check first.");
            end
            fid = fopen(file, "w", "n", "UTF-8");
            if fid < 0
                error("phasecont:app:saveFailed", "Cannot write %s.", file);
            end
            closer = onCleanup(@() fclose(fid));
            fprintf(fid, "%s", char(app.Source));
            clear closer
            app.setStatus("Saved " + file);
        end

        function pp = openPhasePlane(app, options)
            % PP = OPENPHASEPLANE(APP) starts a phasecont.app.PhasePlane on
            % the last good model. The phase plane keeps its own reference,
            % so a later Check does not change it.
            arguments
                app (1,1) phasecont.app.ModelEditor
                options.Visible (1,1) string {mustBeMember(options.Visible, ["on" "off"])} = "on"
            end
            if isempty(app.Model)
                error("phasecont:app:noModel", ...
                    "There is no model yet. Click Check first.");
            end
            pp = phasecont.app.PhasePlane(app.Model, Visible = options.Visible);
        end

        function delete(app)
            % Close the window.
            if ~isempty(app.Figure) && isvalid(app.Figure)
                delete(app.Figure);
            end
        end
    end

    methods (Access = private)
        function m = build(app)
            % The model of the tab that is showing. Each tab goes through
            % the public builders, so it gets their checks.
            switch app.Tab
                case "Equations"
                    eq = app.lines(app.Ui.Eq);
                    if all(strtrim(eq) == "")
                        error("phasecont:app:empty", ...
                            "Nothing to check. Enter equations first.");
                    end
                    par = parseAssignments(app.lines(app.Ui.Par), "Parameters");
                    ini = parseAssignments(app.lines(app.Ui.Ini), "Initial values");
                    m = phasecont.Model.fromEquations(eq, Params = par, ...
                        Init = ini, Name = "equations.ode");
                case "Table"
                    states = tableFrom(app.Ui.States.Data, ...
                        ["Name" "Rhs" "Init"], [false false true]);
                    params = tableFrom(app.Ui.Params.Data, ...
                        ["Name" "Value"], [false true]);
                    aux = tableFrom(app.Ui.Aux.Data, ["Name" "Expr"], [false false]);
                    app.TableRows = [height(states) height(aux)];
                    if height(states) == 0
                        error("phasecont:app:empty", ...
                            "Nothing to check. Add a row to States first.");
                    end
                    m = phasecont.Model.fromTable(States = states, ...
                        Params = params, Aux = aux, Name = "table.ode");
                otherwise
                    lines = app.lines(app.Ui.Src);
                    if all(strtrim(lines) == "")
                        error("phasecont:app:empty", ...
                            "Nothing to check. Enter .ode text or open a file.");
                    end
                    m = phasecont.Model.fromSource( ...
                        join(lines, newline) + newline, Name = app.SourceName);
            end
        end

        function lines = lines(~, area)
            lines = string(area.Value);
            lines = lines(:);
        end

        function d = diagnose(app, err)
            % Turn an error into Diagnostics. The engine reports a parse
            % problem as <file>:<line>:<col>: <message>, one per line.
            d = emptyDiagnostics();
            id = string(err.identifier);
            parts = splitlines(string(err.message));
            parts(strtrim(parts) == "") = [];
            if isempty(parts)
                parts = id;
            end
            for k = 1:numel(parts)
                tok = [];
                if id == "phasecont:engine:model"
                    tok = regexp(parts(k), '^.*?:(\d+):(\d+):\s*(.*)$', ...
                        "tokens", "once");
                end
                if isempty(tok)
                    one = struct("Line", NaN, "Column", NaN, ...
                        "Message", parts(k), "Id", id);
                else
                    one = struct("Line", str2double(tok{1}), ...
                        "Column", str2double(tok{2}), ...
                        "Message", string(tok{3}), "Id", id);
                end
                d(end+1, 1) = one; %#ok<AGROW>
            end
        end

        function showDiagnostics(app)
            d = app.Diagnostics;
            items = cell(1, numel(d));
            for k = 1:numel(d)
                items{k} = char(app.locate(d(k)) + d(k).Message);
            end
            app.Ui.List.Items = items;
        end

        function text = locate(app, d)
            % "line 3, column 5: " in front of a message, or "".
            text = "";
            if isnan(d.Line)
                return
            end
            if app.Tab == "Table"
                nStates = app.TableRows(1);
                if d.Line <= nStates
                    text = sprintf("States row %d: ", d.Line);
                else
                    text = sprintf("Aux row %d: ", d.Line - nStates);
                end
            elseif isnan(d.Column)
                text = sprintf("line %d: ", d.Line);
            else
                text = sprintf("line %d, column %d: ", d.Line, d.Column);
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

        % ---- controls ----

        function buildUi(app, visible)
            fig = uifigure(Visible = visible, Name = "PhaseCont model editor", ...
                Position = [100 100 980 680]);
            fig.CloseRequestFcn = @(~, ~) delete(app);
            root = uigridlayout(fig, [4 1]);
            root.RowHeight = {"1x", 30, 96, 40};
            ui = struct();

            ui.Group = uitabgroup(root, ...
                SelectionChangedFcn = @(~, ev) app.onTab(ev));
            ui.Group.Layout.Row = 1;
            ui.Tabs = struct();
            ui.Tabs.Equations = uitab(ui.Group, Title = "Equations", Tag = "Equations");
            ui.Tabs.Table = uitab(ui.Group, Title = "Table", Tag = "Table");
            ui.Tabs.Source = uitab(ui.Group, Title = "XPP .ode", Tag = "Source");

            g = uigridlayout(ui.Tabs.Equations, [4 2]);
            g.ColumnWidth = {"2x", "1x"};
            g.RowHeight = {22, "1x", 22, "1x"};
            place(uilabel(g, Text = "Equations (x' = ... or dx/dt = ...)"), 1, 1);
            ui.Eq = uitextarea(g, ValueChangedFcn = @(~, ~) app.onTextEdited());
            place(ui.Eq, [2 4], 1);
            place(uilabel(g, Text = "Parameters (name = value)"), 1, 2);
            ui.Par = uitextarea(g, ValueChangedFcn = @(~, ~) app.onTextEdited());
            place(ui.Par, 2, 2);
            place(uilabel(g, Text = "Initial values (name = value)"), 3, 2);
            ui.Ini = uitextarea(g, ValueChangedFcn = @(~, ~) app.onTextEdited());
            place(ui.Ini, 4, 2);

            t = uigridlayout(ui.Tabs.Table, [3 3]);
            t.ColumnWidth = {"1.6x", "1x", "1x"};
            t.RowHeight = {22, "1x", 28};
            defs = {"States", "States (Name, Rhs, Init)", ...
                    {'Name', 'Rhs', 'Init'}, {'char', 'char', 'numeric'}; ...
                "Params", "Parameters (Name, Value)", ...
                    {'Name', 'Value'}, {'char', 'numeric'}; ...
                "Aux", "Aux (Name, Expr)", ...
                    {'Name', 'Expr'}, {'char', 'char'}};
            for k = 1:3
                key = defs{k, 1};
                place(uilabel(t, Text = defs{k, 2}), 1, k);
                ui.(key) = uitable(t, ColumnName = defs{k, 3}, ...
                    ColumnFormat = defs{k, 4}, ...
                    ColumnEditable = true(1, numel(defs{k, 3})), RowName = {}, ...
                    Data = cell(0, numel(defs{k, 3})), ...
                    CellSelectionCallback = @(~, ev) app.onSelect(key, ev));
                place(ui.(key), 2, k);
                b = uigridlayout(t, [1 2]);
                b.Padding = [0 0 0 0];
                place(b, 3, k);
                uibutton(b, Text = "Add row", ...
                    ButtonPushedFcn = @(~, ~) app.onAddRow(key));
                uibutton(b, Text = "Delete row", ...
                    ButtonPushedFcn = @(~, ~) app.onDeleteRow(key));
            end

            s = uigridlayout(ui.Tabs.Source, [1 1]);
            ui.Src = uitextarea(s, ValueChangedFcn = @(~, ~) app.onTextEdited());

            buttons = uigridlayout(root, [1 4]);
            buttons.Padding = [0 0 0 0];
            buttons.Layout.Row = 2;
            ui.OpenFile = uibutton(buttons, Text = "Open file", ...
                ButtonPushedFcn = @(~, ~) app.onOpen());
            ui.Check = uibutton(buttons, Text = "Check", ...
                ButtonPushedFcn = @(~, ~) app.onCheck());
            ui.Phase = uibutton(buttons, Text = "Open in phase plane", ...
                ButtonPushedFcn = @(~, ~) app.onPhasePlane());
            ui.Save = uibutton(buttons, Text = "Save as .ode", ...
                ButtonPushedFcn = @(~, ~) app.onSave());

            ui.List = uilistbox(root, Items = cell(1, 0));
            ui.List.Layout.Row = 3;
            ui.Status = uilabel(root, Text = "", WordWrap = "on", ...
                VerticalAlignment = "top", FontWeight = "bold");
            ui.Status.Layout.Row = 4;

            app.Ui = ui;
            app.Figure = fig;
        end

        % ---- callbacks ----

        function guard(app, action)
            % Run a callback body. A failure goes to the status line.
            try
                action();
            catch err
                app.report(err);
            end
        end

        function onTab(app, ev)
            app.Tab = string(ev.NewValue.Tag);
        end

        function onTextEdited(app)
            app.guard(@() app.check());
        end

        function onCheck(app)
            app.guard(@() app.check());
        end

        function onSelect(app, key, ev)
            if isempty(ev.Indices)
                app.Selected.(key) = [];
            else
                app.Selected.(key) = unique(ev.Indices(:, 1)).';
            end
        end

        function onAddRow(app, key)
            data = app.Ui.(key).Data;
            blank = cell(1, size(data, 2));
            for k = 1:numel(blank)
                if any(strcmp(app.Ui.(key).ColumnFormat{k}, "numeric"))
                    blank{k} = [];
                else
                    blank{k} = '';
                end
            end
            app.Ui.(key).Data = [data; blank];
        end

        function onDeleteRow(app, key)
            data = app.Ui.(key).Data;
            rows = app.Selected.(key);
            rows = rows(rows >= 1 & rows <= size(data, 1));
            if isempty(rows)
                app.setStatus("Select a cell in the row to delete.");
                return
            end
            data(rows, :) = [];
            app.Ui.(key).Data = data;
            app.Selected.(key) = [];
        end

        function onOpen(app)
            [f, d] = uigetfile({'*.ode;*.xml;*.sbml;*.m', ...
                'Models (*.ode, *.xml, *.sbml, *.m)'}, 'Open model');
            figure(app.Figure);
            if isequal(f, 0)
                return
            end
            file = string(fullfile(d, f));
            [~, ~, ext] = fileparts(file);
            if lower(string(ext)) ~= ".m"
                app.guard(@() app.openFile(file));
                return
            end
            answer = inputdlg({ ...
                "Initial values, one per state (numbers separated by spaces or commas)", ...
                "State names (optional, separated by spaces or commas)", ...
                "Parameters (optional, name = value, name = value)"}, ...
                "Open MATLAB function", [1 70]);
            figure(app.Figure);
            if isempty(answer)
                return
            end
            app.guard(@() app.openMatlabFile(file, answer));
        end

        function openMatlabFile(app, file, answer)
            init = str2double(splitWords(answer{1}));
            if any(~isfinite(init))
                error("phasecont:app:badAssignment", ...
                    "Initial values must be finite numbers.");
            end
            names = splitWords(answer{2});
            par = parseAssignments(string(answer{3}), "Parameters");
            app.openFile(file, Init = init, Params = par, StateNames = names);
        end

        function onPhasePlane(app)
            app.guard(@() app.openPhasePlane());
        end

        function onSave(app)
            [f, d] = uiputfile("*.ode", "Save model", "model.ode");
            figure(app.Figure);
            if isequal(f, 0)
                return
            end
            app.guard(@() app.saveOde(fullfile(d, f)));
        end
    end
end

function d = emptyDiagnostics()
d = repmat(struct("Line", NaN, "Column", NaN, "Message", "", "Id", ""), 0, 1);
end

function place(component, row, col)
% Put a component in a grid cell or a span of cells.
component.Layout.Row = row;
component.Layout.Column = col;
end

function c = textLines(text)
% Lines of text as the cell array of char that a text area takes.
text = text(:);
if isempty(text)
    c = {''};
    return
end
lines = splitlines(join(text, newline));
c = cellstr(lines);
end

function w = splitWords(text)
% Words separated by blanks or commas, as a string column. Empty text gives
% no words.
text = strtrim(string(text));
if text == ""
    w = strings(0, 1);
    return
end
w = split(text, [" ", ","]);
w = w(w ~= "");
w = w(:);
end

function s = parseAssignments(lines, label)
% A struct from lines of `name = value`. Commas and semicolons separate
% entries on one line, # and % start a comment, and a value is a number.
s = struct();
for k = 1:numel(lines)
    line = strtrim(regexprep(lines(k), '[%#].*$', ''));
    if line == ""
        continue
    end
    parts = strtrim(split(line, [",", ";"]));
    parts = parts(parts ~= "");
    for j = 1:numel(parts)
        tok = regexp(parts(j), '^([A-Za-z_]\w*)\s*=\s*(\S+)$', "tokens", "once");
        if isempty(tok)
            error("phasecont:app:badAssignment", ...
                "%s, line %d: expected name = number, got ""%s"".", ...
                label, k, parts(j));
        end
        v = str2double(tok{2});
        if ~isfinite(v)
            error("phasecont:app:badAssignment", ...
                "%s, line %d: ""%s"" is not a finite number.", ...
                label, k, tok{2});
        end
        if isfield(s, tok{1})
            error("phasecont:model:duplicateName", ...
                "%s: the name %s is listed more than once.", label, tok{1});
        end
        s.(tok{1}) = v;
    end
end
end

function t = tableFrom(data, vars, numeric)
% A table of the rows of a uitable. A row with every cell empty is dropped.
% An empty text cell is "" and an empty number is NaN.
if isempty(data)
    data = cell(0, numel(vars));
end
keep = true(size(data, 1), 1);
for r = 1:size(data, 1)
    keep(r) = ~all(cellfun(@isBlank, data(r, :)));
end
data = data(keep, :);
cols = cell(1, numel(vars));
for k = 1:numel(vars)
    if numeric(k)
        col = nan(size(data, 1), 1);
        for r = 1:size(data, 1)
            if ~isempty(data{r, k})
                col(r) = double(data{r, k});
            end
        end
    else
        col = strings(size(data, 1), 1);
        for r = 1:size(data, 1)
            if ~isempty(data{r, k})
                col(r) = string(data{r, k});
            end
        end
    end
    cols{k} = col;
end
t = table(cols{:}, VariableNames = cellstr(vars));
end

function tf = isBlank(v)
tf = isempty(v) || ((ischar(v) || isstring(v)) && strtrim(string(v)) == "") ...
    || (isnumeric(v) && isscalar(v) && isnan(v));
end

function data = tableData(t, vars, label)
% The cell array a uitable takes, from a table with the columns VARS. The
% columns after Name and the first value column are optional.
have = string(t.Properties.VariableNames);
need = vars(1:min(2, numel(vars)));
absent = need(~ismember(need, have));
if ~isempty(absent)
    error("phasecont:model:badTable", "%s table needs the column %s.", ...
        label, join(absent, ", "));
end
use = vars(ismember(vars, have));
data = cell(height(t), numel(vars));
for k = 1:numel(vars)
    if ~ismember(vars(k), use)
        for r = 1:height(t)
            data{r, k} = defaultCell(vars(k));
        end
        continue
    end
    col = t.(vars(k));
    for r = 1:height(t)
        if iscell(col)
            v = col{r};
        else
            v = col(r);
        end
        if isstring(v) || iscategorical(v)
            v = char(string(v));
        elseif ischar(v)
            v = char(v);
        elseif isnumeric(v) || islogical(v)
            v = double(v);
            if isscalar(v) && isnan(v)
                v = [];    % a NaN cell means "no value"; show it blank
            end
        end
        data{r, k} = v;
    end
end
end

function v = defaultCell(var)
% An empty cell of the column's kind: [] for Init and Value, '' otherwise.
if any(var == ["Init" "Value"])
    v = [];
else
    v = '';
end
end
