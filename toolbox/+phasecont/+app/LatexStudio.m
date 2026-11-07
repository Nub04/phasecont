classdef LatexStudio < handle
    % A window for writing a LaTeX document with PhaseCont figures and tables.
    %
    %   app = phasecont.app.LatexStudio
    %   app = phasecont.app.LatexStudio(Visible = "off")
    %   app = phasecont.app.LatexStudio(Executable = "C:\texlive\bin\latexmk.exe")
    %   app = phasecont.app.LatexStudio(Timeout = 60)
    %
    % The editor on the left holds a whole LaTeX document. Insert figure and
    % Insert table add the LaTeX of a Branch from the base workspace, as a
    % pgfplots figure or a booktabs table of its special points, just before
    % the last \end{document} (at the end when there is none). The figure
    % always writes its points into the document, so one file compiles
    % alone. A new window starts with an article that loads pgfplots and
    % booktabs.
    %
    % Digits, the number of significant digits of every number that Insert
    % writes, has no default. Leave it empty and an Insert raises
    % phasecont:latex:missingDigits.
    %
    % Compile runs phasecont.latex.build on the editor text, so it needs
    % latexmk from TeX Live or MiKTeX on the system path, or Executable. The
    % window blocks while latexmk runs. By default there is no time limit,
    % so a document that never ends blocks it until you press Ctrl-C in the
    % MATLAB window; Timeout sets a limit in seconds. The right side shows the result of the last compile. The Errors tab lists each TeX error with its line,
    % and selecting one shows the lines of the document around it. The Log
    % tab holds the TeX log. Open PDF hands the file to the system viewer;
    % there is no preview inside the window.
    %
    % Methods
    %   newDocument, setSource, setDigits, openFile, saveFile, insertFigure,
    %   insertTable, compile, selectError, openPdf, delete
    %
    % Properties (read only)
    %   Source, Digits, Result, File, Status, Figure
    %
    % Result is the struct of phasecont.latex.build (Ok, Pdf, Folder, Log,
    % Errors, Command, TimedOut) for the last Compile. Digits is NaN while
    % the field is empty or is not a number.

    properties (Dependent, SetAccess = private)
        % The text of the editor.
        Source (1,1) string
        % The number in the Digits field, NaN when it has none.
        Digits (1,1) double
    end

    properties (SetAccess = private)
        % The result of the last Compile.
        Result (1,1) struct = struct("Ok", false, "Pdf", "", "Folder", "", ...
            "Log", "", "Errors", repmat(struct("File", "", "Line", 0, ...
            "Message", ""), 0, 1), "Command", "", "TimedOut", false)
        % The file that Open or Save last used, or "".
        File (1,1) string = ""
        Status (1,1) string = ""
        Figure = []
    end

    properties (Access = private)
        Ui (1,1) struct = struct()
        Executable (1,1) string = ""
        Timeout (1,1) double = Inf
        Opener = []
        % The document text that the last Compile read.
        CompiledSource (1,1) string = ""
    end

    methods
        function app = LatexStudio(options)
            % APP = LatexStudio builds the window. Visible = "off" builds it
            % without showing it. Executable is the path of latexmk, and
            % Opener is a function handle that takes the PDF path, both for
            % tests and for a TeX install that is not on the path. Timeout is
            % the number of seconds that one Compile may take, Inf for no
            % limit.
            arguments
                options.Visible (1,1) string {mustBeMember(options.Visible, ["on" "off"])} = "on"
                options.Executable (1,1) string = ""
                options.Timeout (1,1) double {mustBePositive} = Inf
                options.Opener = []
            end
            if ~isempty(options.Opener) && ~isa(options.Opener, "function_handle")
                error("phasecont:app:badOpener", ...
                    "Opener must be a function handle that takes the PDF path.");
            end
            app.Executable = options.Executable;
            app.Timeout = options.Timeout;
            if isempty(options.Opener)
                app.Opener = @defaultOpener;
            else
                app.Opener = options.Opener;
            end
            app.buildUi(options.Visible);
            app.newDocument();
            app.refreshBranches();
            app.setStatus("Set Digits, then insert a figure or a table, or click Compile.");
        end

        function s = get.Source(app)
            s = join(string(app.Ui.Src.Value(:)), newline);
        end

        function d = get.Digits(app)
            d = str2double(strtrim(string(app.Ui.Digits.Value)));
        end

        function newDocument(app)
            % NEWDOCUMENT(APP) replaces the text with an empty article that
            % loads pgfplots and booktabs.
            arguments
                app (1,1) phasecont.app.LatexStudio
            end
            app.Ui.Src.Value = textLines(templateText());
            app.File = "";
            app.setStatus("New document.");
        end

        function setSource(app, text)
            % SETSOURCE(APP, TEXT) replaces the text of the editor. TEXT is
            % a string, or a string array with one line per entry.
            arguments
                app (1,1) phasecont.app.LatexStudio
                text string {mustBeNonmissing}
            end
            app.Ui.Src.Value = textLines(text);
        end

        function setDigits(app, n)
            % SETDIGITS(APP, N) sets the Digits field. The range 1 to 17 is
            % checked when a figure or a table is inserted.
            arguments
                app (1,1) phasecont.app.LatexStudio
                n (1,1) double
            end
            app.Ui.Digits.Value = char(string(n));
        end

        function openFile(app, file)
            % OPENFILE(APP, FILE) reads a UTF-8 .tex file into the editor.
            arguments
                app (1,1) phasecont.app.LatexStudio
                file (1,1) string
            end
            if ~isfile(file)
                error("phasecont:app:openFailed", "Cannot read %s.", file);
            end
            text = string(fileread(file, Encoding = "UTF-8"));
            app.Ui.Src.Value = textLines(text);
            app.File = file;
            app.setStatus("Opened " + file);
        end

        function saveFile(app, file)
            % SAVEFILE(APP, FILE) writes the editor text as UTF-8.
            arguments
                app (1,1) phasecont.app.LatexStudio
                file (1,1) string
            end
            fid = fopen(file, "w", "n", "UTF-8");
            if fid < 0
                error("phasecont:app:saveFailed", "Cannot write %s.", file);
            end
            closer = onCleanup(@() fclose(fid));
            fprintf(fid, "%s", char(app.Source));
            clear closer
            app.File = file;
            app.setStatus("Saved " + file);
        end

        function insertFigure(app, br, options)
            % INSERTFIGURE(APP, BR) adds the pgfplots figure of the Branch
            % BR before the last \end{document}. Var, Color and Title are
            % the options of phasecont.latex.figureFromBranch. The Digits
            % field must hold a number.
            arguments
                app (1,1) phasecont.app.LatexStudio
                br (1,1) phasecont.Branch
                options.Var = []
                options.Color (1,1) string = "#000000"
                options.Title (1,1) string = ""
            end
            fig = phasecont.latex.figureFromBranch(br, Var = options.Var, ...
                Color = options.Color, Title = options.Title);
            r = phasecont.latex.pgfplots(fig, Digits = app.digitsValue());
            app.insertText(r.Tex);
            app.setStatus("Inserted a figure.");
        end

        function insertTable(app, source)
            % INSERTTABLE(APP, SOURCE) adds the table of special points of
            % SOURCE before the last \end{document}. SOURCE is a Branch or
            % the struct that phasecont.latex.specialTable takes. The
            % Digits field must hold a number.
            arguments
                app (1,1) phasecont.app.LatexStudio
                source (1,1)
            end
            r = phasecont.latex.specialTable(source, Digits = app.digitsValue());
            app.insertText(r.Tex);
            app.setStatus("Inserted a table.");
        end

        function result = compile(app)
            % RESULT = COMPILE(APP) builds the document with latexmk and
            % shows the outcome. A TeX error is part of RESULT and does not
            % raise, and neither does a run that the Timeout stopped (TimedOut
            % is true). A missing latexmk raises phasecont:latex:engineNotFound.
            arguments
                app (1,1) phasecont.app.LatexStudio
            end
            tex = app.Source;
            if strtrim(tex) == ""
                error("phasecont:app:empty", ...
                    "Nothing to compile. Write or open a document first.");
            end
            app.setStatus("Compiling ...");
            app.setBusy(true);
            idle = onCleanup(@() app.setBusy(false));
            args = {"Timeout", app.Timeout};
            if app.Executable ~= ""
                args = [args, {"Executable", app.Executable}];
            end
            r = phasecont.latex.build(tex, args{:});
            clear idle
            app.Result = r;
            app.CompiledSource = tex;
            app.showResult();
            result = r;
        end

        function lines = selectError(app, k)
            % LINES = SELECTERROR(APP, K) selects the K-th error of the last
            % Compile and shows the lines of the document around it. LINES
            % is the text that the window shows, one string per line, with
            % ">" in front of the line that TeX named.
            arguments
                app (1,1) phasecont.app.LatexStudio
                k (1,1) double {mustBeInteger}
            end
            errs = app.Result.Errors;
            if k < 1 || k > numel(errs)
                error("phasecont:app:noSuchError", ...
                    "There are %d errors, so %d is not one of them.", numel(errs), k);
            end
            lines = app.errorContext(errs(k));
            app.Ui.List.Value = app.Ui.List.Items{k};
            app.Ui.Context.Value = textLines(lines);
        end

        function openPdf(app)
            % OPENPDF(APP) opens the PDF of the last Compile in the system
            % viewer. The error phasecont:app:noPdf means there is none.
            arguments
                app (1,1) phasecont.app.LatexStudio
            end
            pdf = app.Result.Pdf;
            if pdf == ""
                error("phasecont:app:noPdf", ...
                    "There is no PDF. Click Compile and fix the errors first.");
            end
            app.Opener(char(pdf));
            app.setStatus("Opened " + pdf);
        end

        function refreshBranches(app)
            % REFRESHBRANCHES(APP) lists the Branch variables of the base
            % workspace in the Branch menu.
            arguments
                app (1,1) phasecont.app.LatexStudio
            end
            vars = evalin("base", "whos");
            keep = false(1, numel(vars));
            for k = 1:numel(vars)
                keep(k) = string(vars(k).class) == "phasecont.Branch" ...
                    && prod(vars(k).size) == 1;
            end
            names = string({vars(keep).name});
            app.Ui.Branch.ItemsData = {};
            if isempty(names)
                app.Ui.Branch.Items = {'(no Branch in the workspace)'};
                app.Ui.Branch.ItemsData = {''};
            else
                app.Ui.Branch.Items = cellstr(names);
                app.Ui.Branch.ItemsData = cellstr(names);
            end
        end

        function delete(app)
            % Close the window.
            if ~isempty(app.Figure) && isvalid(app.Figure)
                delete(app.Figure);
            end
        end
    end

    methods (Access = private)
        function d = digitsValue(app)
            % The Digits field as a number. An empty field gives NaN, which
            % the render call turns into phasecont:latex:missingDigits.
            text = strtrim(string(app.Ui.Digits.Value));
            if text == ""
                d = NaN;
                return
            end
            d = str2double(text);
            if isnan(d)
                error("phasecont:latex:badDigits", ...
                    "Digits must be a whole number from 1 to 17, not ""%s"".", text);
            end
        end

        function insertText(app, tex)
            % Put TEX on its own lines before the last \end{document}.
            lines = string(app.Ui.Src.Value(:));
            block = splitlines(string(regexprep(char(tex), '\s+$', '')));
            at = find(startsWith(strtrim(lines), "\end{document}"), 1, "last");
            if isempty(at)
                at = numel(lines) + 1;
            end
            merged = [lines(1:at-1); ""; block(:); ""; lines(at:end)];
            app.Ui.Src.Value = textLines(merged);
        end

        function showResult(app)
            r = app.Result;
            yesNo = ["no" "yes"];
            app.Ui.Summary.Value = textLines([ ...
                "Compiled: " + yesNo(1 + r.Ok) + timedOutNote(r.TimedOut, app.Timeout); ...
                "PDF: " + r.Pdf; ...
                "Folder: " + r.Folder; ...
                "Command: " + r.Command]);
            n = numel(r.Errors);
            items = cell(1, n);
            for k = 1:n
                e = r.Errors(k);
                place = string(sprintf("line %d", e.Line));
                if ~endsWith(e.File, "main.tex")
                    place = e.File + ", " + place;
                end
                items{k} = char(string(sprintf("%d. ", k)) + place + ": " + e.Message);
            end
            app.Ui.List.Items = items;
            app.Ui.Log.Value = textLines(r.Log);
            app.Ui.Context.Value = {''};
            if n > 0
                app.selectError(1);
            end
            if r.Ok
                app.setStatus("Compiled. Click Open PDF to read it.");
            elseif r.TimedOut
                app.setStatus(sprintf( ...
                    "Compile stopped after %g s. Raise Timeout, or look for a loop in the document.", ...
                    app.Timeout));
            elseif n > 0
                app.setStatus(sprintf("Compile failed with %d error(s).", n));
            else
                app.setStatus("Compile failed. The Log tab has the output.");
            end
        end

        function lines = errorContext(app, e)
            % Three lines of the document on each side of the error line.
            if ~endsWith(e.File, "main.tex")
                lines = e.File + ", line " + e.Line + ": " + e.Message;
                return
            end
            src = splitlines(app.CompiledSource);
            first = max(1, e.Line - 3);
            last = min(numel(src), e.Line + 3);
            lines = strings(0, 1);
            for k = first:last
                mark = " ";
                if k == e.Line
                    mark = ">";
                end
                lines(end+1, 1) = string(sprintf("%s %4d  %s", mark, k, src(k))); %#ok<AGROW>
            end
            lines = [lines; ""; string(e.Message)];
        end

        function setBusy(app, busy)
            if isfield(app.Ui, "Compile") && isvalid(app.Ui.Compile)
                if busy
                    app.Ui.Compile.Enable = "off";
                else
                    app.Ui.Compile.Enable = "on";
                end
                drawnow
            end
        end

        function setStatus(app, text)
            app.Status = text;
            if isfield(app.Ui, "StatusLabel") && isvalid(app.Ui.StatusLabel)
                app.Ui.StatusLabel.Text = char(text);
            end
        end

        function report(app, err)
            % Show an error from a callback.
            app.setStatus(string(err.message));
            if ~isempty(app.Figure) && isvalid(app.Figure) && logical(app.Figure.Visible)
                uialert(app.Figure, err.message, "PhaseCont", Icon = "warning");
            end
        end

        function guard(app, action)
            % Run a callback body. A failure goes to the status line.
            try
                action();
            catch err
                app.report(err);
            end
        end

        % ---- window ----

        function buildUi(app, visible)
            fig = uifigure(Visible = visible, Name = "PhaseCont LaTeX studio", ...
                Position = [100 100 1100 720]);
            fig.CloseRequestFcn = @(~, ~) delete(app);
            root = uigridlayout(fig, [4 1]);
            root.RowHeight = {32, "1x", 170, 36};
            ui = struct();

            bar = uigridlayout(root, [1 11]);
            bar.Padding = [0 0 0 0];
            bar.ColumnWidth = {56, 56, 56, 46, 48, "1x", 64, 100, 92, 72, 80};
            bar.Layout.Row = 1;
            uibutton(bar, Text = "New", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.newDocument()));
            uibutton(bar, Text = "Open", ButtonPushedFcn = @(~, ~) app.onOpen());
            uibutton(bar, Text = "Save", ButtonPushedFcn = @(~, ~) app.onSave());
            uilabel(bar, Text = "Digits", HorizontalAlignment = "right");
            ui.Digits = uieditfield(bar, "text", Value = '');
            ui.Branch = uidropdown(bar, Items = {'(no Branch in the workspace)'}, ...
                ItemsData = {''});
            uibutton(bar, Text = "Refresh", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.refreshBranches()));
            uibutton(bar, Text = "Insert figure", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.onInsert("figure")));
            uibutton(bar, Text = "Insert table", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.onInsert("table")));
            ui.Compile = uibutton(bar, Text = "Compile", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.compile()));
            uibutton(bar, Text = "Open PDF", ...
                ButtonPushedFcn = @(~, ~) app.guard(@() app.openPdf()));

            body = uigridlayout(root, [1 2]);
            body.Padding = [0 0 0 0];
            body.ColumnWidth = {"3x", "2x"};
            body.Layout.Row = 2;
            ui.Src = uitextarea(body, FontName = "monospaced");
            ui.Src.Layout.Column = 1;

            side = uigridlayout(body, [4 1]);
            side.Padding = [0 0 0 0];
            side.RowHeight = {22, 96, 22, "1x"};
            side.Layout.Column = 2;
            uilabel(side, Text = "Result of the last compile");
            ui.Summary = uitextarea(side, Editable = "off");
            uilabel(side, Text = "Document around the selected error");
            ui.Context = uitextarea(side, Editable = "off", FontName = "monospaced");

            tabs = uitabgroup(root);
            tabs.Layout.Row = 3;
            errTab = uitab(tabs, Title = "Errors");
            logTab = uitab(tabs, Title = "Log");
            eg = uigridlayout(errTab, [1 1]);
            ui.List = uilistbox(eg, Items = cell(1, 0), ...
                ValueChangedFcn = @(~, ev) app.guard(@() app.onError(ev.Value)));
            lg = uigridlayout(logTab, [1 1]);
            ui.Log = uitextarea(lg, Editable = "off", FontName = "monospaced");

            ui.StatusLabel = uilabel(root, Text = "", WordWrap = "on", ...
                VerticalAlignment = "top", FontWeight = "bold");
            ui.StatusLabel.Layout.Row = 4;

            app.Ui = ui;
            app.Figure = fig;
        end

        % ---- callbacks ----

        function onOpen(app)
            [f, d] = uigetfile({'*.tex', 'LaTeX (*.tex)'}, 'Open document');
            figure(app.Figure);
            if isequal(f, 0)
                return
            end
            app.guard(@() app.openFile(string(fullfile(d, f))));
        end

        function onSave(app)
            file = app.File;
            if file == ""
                [f, d] = uiputfile({'*.tex', 'LaTeX (*.tex)'}, 'Save document', ...
                    'document.tex');
                figure(app.Figure);
                if isequal(f, 0)
                    return
                end
                file = string(fullfile(d, f));
            end
            app.guard(@() app.saveFile(file));
        end

        function onInsert(app, what)
            name = string(app.Ui.Branch.Value);
            if name == ""
                error("phasecont:app:noBranch", ...
                    "There is no Branch to insert. Compute one, then click Refresh.");
            end
            br = evalin("base", name);
            if ~isa(br, "phasecont.Branch") || ~isscalar(br)
                error("phasecont:app:noBranch", ...
                    "%s is not a single Branch. Click Refresh.", name);
            end
            if what == "figure"
                app.insertFigure(br);
            else
                app.insertTable(br);
            end
        end

        function onError(app, value)
            k = find(strcmp(app.Ui.List.Items, value), 1);
            if ~isempty(k)
                app.selectError(k);
            end
        end
    end
end

function text = templateText()
text = [ ...
    "\documentclass{article}"
    "\usepackage{pgfplots}"
    "\pgfplotsset{compat=1.18}"
    "\usepackage{booktabs}"
    "\begin{document}"
    ""
    "\end{document}"];
end

function defaultOpener(pdf)
% Open a file with the program that the system chose for it.
if ispc
    winopen(pdf);
    return
end
if ismac
    program = "open";
else
    program = "xdg-open";
end
[status, out] = system(program + " """ + string(pdf) + """ &");
if status ~= 0
    error("phasecont:app:openFailed", "Cannot open %s: %s", pdf, strtrim(out));
end
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

function s = timedOutNote(timedOut, seconds)
% " (stopped after 60 s)" for a run that the time limit ended, else "".
s = "";
if timedOut
    s = string(sprintf(" (stopped after %g s)", seconds));
end
end
