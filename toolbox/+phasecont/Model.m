classdef Model < handle
    % An ODE model read from an XPP/XPPAUT .ode file.
    %
    %   m = phasecont.Model(file)
    %   m = phasecont.Model(file, Evaluator = "fallback")
    %   f = m.rhs(x)
    %   f = m.rhs(x, p)       % p: numeric vector or struct of parameters
    %   f = m.rhs(x, p, t)    % t: time, needed when the equations use t
    %   F = m.rhs(X, p)       % X is n-by-k, one point per column; F is n-by-k
    %   m = phasecont.Model.fromSource(text, Name = "model.ode")
    %   m = phasecont.Model.fromEquations(text, Params = P, Init = X0)
    %   m = phasecont.Model.fromTable(States = T, Params = P, Init = X0)
    %   m = phasecont.Model.fromFunction(@(t,x,p) ..., Init = X0)
    %   m = phasecont.Model.fromFile("model.xml")   % .ode, .xml, .sbml, .m
    %
    % Every route ends as .ode text handed to fromSource, so sessions,
    % continuation and the ensemble engine treat all models alike.
    %
    % The model lives in the engine and is freed when the object is deleted
    % or cleared. Saving a Model stores its source text, and loading it
    % compiles the text again.

    properties (SetAccess = private)
        File (1,1) string = ""
        Source (1,1) string = ""
        Variables (1,:) string = strings(1,0)
        Parameters (1,:) string = strings(1,0)
        ParameterValues (:,1) double = zeros(0,1)
        Init (:,1) double = zeros(0,1)
        Evaluator (1,1) string = "jit"
        % AotInfo holds Compiler, Version, Flags and Library of the AOT
        % library when Evaluator is "aot"; the fields are "" otherwise.
        AotInfo (1,1) struct = struct("Compiler", "", "Version", "", ...
            "Flags", "", "Library", "")
        % Autonomous is false when the equations use t.
        Autonomous (1,1) logical = true
        % IsDelay is true when the equations use delay(...). The engine runs
        % such a model with the JIT only, and only through the delay entries.
        IsDelay (1,1) logical = false
        % SourceSha256 is the SHA-256 of the exact source text, as 64 lower-case
        % hex digits. A saved session stores it to check that a replay runs
        % the same model.
        SourceSha256 (1,1) string = ""
        % Options holds the `@` options of the file as a struct of strings,
        % with lower-case field names and the last value of a repeated name.
        % An option with blanks around `=` is left out: XPPAUT 8.0 ignores it.
        Options (1,1) struct = struct()
        Diagnostics (:,1) struct = repmat(struct("code", [], "severity", [], ...
            "file", [], "line", [], "col", [], "end_line", [], "end_col", [], ...
            "message", [], "fix_it", [], "notes", []), 0, 1)
    end

    properties (Hidden, Transient, SetAccess = private)
        Id (1,1) uint64 = uint64(0)
    end

    methods
        function obj = Model(file, options)
            arguments
                file (1,1) string
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
                % Source: model text to use instead of reading FILE, which is then
                % only the name. Use Model.fromSource rather than this option.
                options.Source (1,1) string = string(missing)
            end
            if exist("pckern", "file") ~= 3
                error("phasecont:engine:load", ...
                    "pckern.%s is not on the MATLAB path. Add the folder " + ...
                    "that holds the PhaseCont binaries to the path.", mexext);
            end
            if ismissing(options.Source)
                full = phasecont.Model.fullPath(file);
                if ~isfile(full)
                    error("phasecont:model:fileNotFound", "%s", ...
                        "The model file """ + full + """ does not exist.");
                end
                text = string(fileread(full, Encoding = "UTF-8"));
            else
                full = file;
                text = options.Source;
            end
            obj.File = full;
            obj.Source = text;
            if options.Evaluator == ""
                % No Evaluator given: the session option decides.
                obj.Evaluator = phasecont.config("Codegen");
            else
                obj.Evaluator = options.Evaluator;
            end
            obj.build();
        end

        function f = rhs(obj, x, p, t)
            % F = M.RHS(X), M.RHS(X, P) or M.RHS(X, P, T): the right-hand side
            % at X. A model that uses t (Autonomous is false) needs T; a model
            % that does not use t ignores it. P defaults to ParameterValues
            % when it is omitted.
            %
            % A vector X with one entry per variable is one point and F is a
            % column. A matrix X with one row per variable is k points, one per
            % column, and F is n-by-k. A model with one variable takes a row
            % vector of k points the same way.
            arguments
                obj (1,1) phasecont.Model
                x {mustBeNumeric, mustBeReal}
                p = []
                t = []
            end
            haveT = nargin > 3;
            if haveT
                if ~(isnumeric(t) && isreal(t) && isscalar(t) && isfinite(t))
                    error("phasecont:model:badTime", ...
                        "t must be a finite real numeric scalar.");
                end
            elseif ~obj.Autonomous
                name = obj.File;
                if name == ""
                    name = "model";
                end
                error("phasecont:model:missingTime", ...
                    "%s uses t; call rhs(x, p, t)", name);
            end
            if nargin < 3
                p = obj.ParameterValues;
            else
                p = obj.paramVector(p);
            end
            n = numel(obj.Variables);
            if ~ismatrix(x)
                error("phasecont:model:badState", ...
                    "x must be a vector or a matrix, not an array with %d dimensions.", ...
                    ndims(x));
            end
            if n >= 1 && isequal(size(x), [n 0])
                f = zeros(n, 0);   % no points, no call
                return
            end
            single = isempty(x) || (isvector(x) && (numel(x) == n || n ~= 1));
            if single
                if haveT
                    f = pckern("model_rhs", obj.Id, double(x), p, double(t));
                else
                    f = pckern("model_rhs", obj.Id, double(x), p);
                end
                f = f(:);
                return
            end
            if size(x, 1) ~= n
                error("phasecont:model:badState", ...
                    "x must have %d rows, one per variable, but has %d.", ...
                    n, size(x, 1));
            end
            if haveT
                f = pckern("model_rhs_batch", obj.Id, double(x), p, double(t));
            else
                f = pckern("model_rhs_batch", obj.Id, double(x), p);
            end
        end

        function delete(obj)
            if isempty(obj.Id) || obj.Id == 0
                return
            end
            try
                pckern("free", obj.Id);
            catch err
                if err.identifier ~= "phasecont:engine:badHandle"
                    warning("phasecont:model:freeFailed", "%s", err.message);
                end
            end
            obj.Id = uint64(0);
        end
    end

    methods (Hidden)
        function p = paramVector(obj, p)
            % Parameter vector from a numeric vector or a scalar struct.
            if isstruct(p)
                if ~isscalar(p)
                    error("phasecont:model:badParameter", ...
                        "Parameters given as a struct must be a scalar struct.");
                end
                v = obj.ParameterValues;
                names = string(fieldnames(p));
                for k = 1:numel(names)
                    idx = find(obj.Parameters == names(k), 1);
                    if isempty(idx)
                        error("phasecont:model:unknownParameter", ...
                            "The model has no parameter named ""%s"".", names(k));
                    end
                    val = p.(names(k));
                    if ~(isnumeric(val) && isreal(val) && isscalar(val) ...
                            && isfinite(val))
                        error("phasecont:model:badParameter", ...
                            "Parameter ""%s"" must be a finite real numeric scalar.", ...
                            names(k));
                    end
                    v(idx) = double(val);
                end
                p = v;
            elseif isnumeric(p) && isreal(p) && (isvector(p) || isempty(p)) ...
                    && all(isfinite(p), "all") && numel(p) == numel(obj.Parameters)
                p = double(p(:));
            else
                error("phasecont:model:badParameter", ...
                    "Parameters must be a struct or a finite real numeric " + ...
                    "vector with %d values.", numel(obj.Parameters));
            end
        end

        function info = engineInfo(obj)
            info = jsondecode(pckern("model_info", obj.Id));
        end
    end

    methods (Static)
        function obj = fromSource(text, options)
            % M = phasecont.Model.fromSource(TEXT, Name = "model.ode",
            % Evaluator = "jit") builds a model from source text. Name goes
            % into File and the diagnostics; it is not read from disk. The
            % text is hashed as it is, so the same text gives the same
            % SourceSha256 on every platform.
            arguments
                text (1,1) string {mustBeNonmissing}
                options.Name (1,1) string = "model.ode"
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            obj = phasecont.Model(options.Name, ...
                Evaluator = options.Evaluator, Source = text);
        end

        function obj = fromEquations(text, options)
            % M = phasecont.Model.fromEquations(TEXT, Params = struct(...),
            % Init = struct(...), Name = "equations.ode") builds a model from
            % equation lines such as
            %
            %   x' = y
            %   dy/dt = -x - c*y
            %
            % Params and Init are structs of numbers. The engine reads them as
            % `par` and `init` lines added after the equations, so a parser
            % diagnostic keeps the line number of TEXT. TEXT is one string or
            % a string array with one line per entry.
            %
            % Nothing is defaulted. A state without an Init entry raises
            % phasecont:model:missingInit, an Init name that is not a state
            % raises phasecont:model:unknownInit, and a symbol that is neither
            % a state nor a parameter is a parse error with its line and
            % column. TEXT must not hold `init` lines; use fromSource for a
            % complete .ode text.
            arguments
                text string {mustBeNonmissing}
                options.Params (1,1) struct = struct()
                options.Init (1,1) struct = struct()
                options.Name (1,1) string = "equations.ode"
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            text = join(text(:), newline);
            lines = splitlines(text);
            if any(~cellfun(@isempty, regexp(cellstr(lines), ...
                    '^\s*(i|ic|init)\s+[A-Za-z_]', 'once', 'ignorecase')))
                error("phasecont:model:initInText", ...
                    "The equation text has an init line. Remove it and " + ...
                    "pass Init, or use Model.fromSource for a complete " + ...
                    ".ode text.");
            end
            source = phasecont.internal.odeText(text, options.Params, ...
                options.Init);
            given = string(fieldnames(options.Init));
            try
                obj = phasecont.Model.fromSource(source, Name = options.Name, ...
                    Evaluator = options.Evaluator);
            catch err
                % The parser refuses an init line for a name that is not a
                % state, and its message does not say that Init is the
                % cause. Parse the equations without Init to read the
                % states, and name the extra entries when there are any.
                extra = string.empty;
                if err.identifier == "phasecont:engine:model" && ~isempty(given)
                    try
                        probe = phasecont.Model.fromSource( ...
                            phasecont.internal.odeText(text, options.Params, struct()), ...
                            Name = options.Name, Evaluator = options.Evaluator);
                        extra = setdiff(given, probe.Variables, "stable");
                    catch
                    end
                end
                if ~isempty(extra)
                    error("phasecont:model:unknownInit", ...
                        "Init has names that are not states of the model: %s.", ...
                        join(extra, ", "));
                end
                rethrow(err);
            end
            missing = setdiff(obj.Variables, given, "stable");
            if ~isempty(missing)
                error("phasecont:model:missingInit", ...
                    "No initial value for %s. Give one in Init.", ...
                    join(missing, ", "));
            end
            extra = setdiff(given, obj.Variables, "stable");
            if ~isempty(extra)
                error("phasecont:model:unknownInit", ...
                    "Init has names that are not states of the model: %s.", ...
                    join(extra, ", "));
            end
        end

        function obj = fromTable(options)
            % M = phasecont.Model.fromTable(States = T, Params = P, Init = X,
            % Aux = A) builds a model from tables or structs.
            %
            %   States  table with Name and Rhs columns (and an optional Init
            %           column), or a struct of right-hand sides
            %   Params  table with Name and Value columns, or a struct
            %   Init    table with Name and Value columns, or a struct
            %   Aux     table with Name and Expr columns, or a struct
            %
            % A right-hand side is text, or a number for a constant. A NaN in
            % the Init column of States means no value was given, and the
            % model then raises phasecont:model:missingInit. The rules of
            % fromEquations apply from there on.
            arguments
                options.States = []
                options.Params = struct()
                options.Init = struct()
                options.Aux = struct()
                options.Name (1,1) string = "table.ode"
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            if isnumeric(options.States) && isempty(options.States)
                error("phasecont:model:badTable", ...
                    "States is required: a table with Name and Rhs columns " + ...
                    "or a struct of right-hand sides.");
            end
            [eq, p, x0] = phasecont.internal.tableToEquations( ...
                options.States, options.Params, options.Init, options.Aux);
            obj = phasecont.Model.fromEquations(eq, Params = p, Init = x0, ...
                Name = options.Name, Evaluator = options.Evaluator);
        end

        function obj = fromFunction(h, options)
            % M = phasecont.Model.fromFunction(@(t,x,p) [...], Init = x0,
            % Params = struct(...), StateNames = ["x" "y"]) translates a
            % MATLAB function handle into a model. The engine reads the text
            % of the handle and never calls it, so only a subset of MATLAB
            % works: arithmetic, indexing, the usual math functions, and
            % assignments in a function file. Loops, if and unknown calls
            % raise phasecont:import:unsupported with their position.
            %
            % Numeric vectors the handle captured become constants. Params is
            % a struct whose field order gives p(1), p(2), and so on. Init
            % holds one value per state and has no default; leaving it out
            % raises phasecont:import:missingValue.
            arguments
                h (1,1) function_handle
                options.Init double = zeros(0, 1)
                options.Params (1,1) struct = struct()
                options.StateNames string = strings(0, 1)
                options.Name (1,1) string = "function.ode"
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            [text, opts, skipped] = phasecont.internal.importRequest(h, ...
                options.Init(:), options.Params, options.StateNames(:), "");
            ode = phasecont.internal.importText("matlab", text, opts, skipped);
            obj = phasecont.Model.fromSource(ode, Name = options.Name, ...
                Evaluator = options.Evaluator);
        end

        function obj = fromFile(file, options)
            % M = phasecont.Model.fromFile(FILE) reads a model by file
            % extension:
            %
            %   .ode            an XPP/XPPAUT file, as phasecont.Model(FILE)
            %   .xml, .sbml     SBML levels 2 and 3 (species, parameters,
            %                   compartments, reactions, rate and assignment
            %                   rules; events, delays and algebraic rules are
            %                   refused by name)
            %   .m              a MATLAB function file, translated as in
            %                   fromFunction
            %
            % Init, Params and StateNames apply to .m files only. Passing them
            % with another extension raises phasecont:model:badOption.
            arguments
                file (1,1) string
                options.Init double = zeros(0, 1)
                options.Params (1,1) struct = struct()
                options.StateNames string = strings(0, 1)
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            [~, stem, ext] = fileparts(file);
            ext = lower(string(ext));
            given = ~isempty(options.Init) || ~isempty(fieldnames(options.Params)) ...
                || ~isempty(options.StateNames);
            if ext == ".m"
                [text, opts, skipped] = phasecont.internal.importRequest( ...
                    file, options.Init(:), options.Params, ...
                    options.StateNames(:), "");
                ode = phasecont.internal.importText("matlab", text, opts, skipped);
            elseif any(ext == [".ode" ".xml" ".sbml"])
                if given
                    error("phasecont:model:badOption", ...
                        "Init, Params and StateNames apply to .m files only. " + ...
                        "A %s file holds its own values.", ext);
                end
                if ext == ".ode"
                    obj = phasecont.Model(file, Evaluator = options.Evaluator);
                    return
                end
                text = phasecont.internal.readSbml(file);
                ode = phasecont.internal.importText("sbml", text);
            else
                error("phasecont:model:badOption", ...
                    "Cannot read a %s file. The extensions are .ode, .xml, " + ...
                    ".sbml and .m.", ext);
            end
            obj = phasecont.Model.fromSource(ode, Name = string(stem) + ".ode", ...
                Evaluator = options.Evaluator);
        end

        function obj = loadobj(s)
            % Compile the saved source text again. Id is transient.
            obj = s;
            obj.Id = uint64(0);
            obj.build();
        end
    end

    methods (Access = private)
        function build(obj)
            % Parse, compile and read back the model for obj.Source.
            text = obj.Source;
            file = obj.File;
            [id, diag, msg] = pckern("model_from_ode", char(text));
            d = jsondecode(diag);
            entries = phasecont.Model.toStructs(d.diagnostics);
            if id == 0
                text = "";
                for k = 1:numel(entries)
                    if string(entries(k).severity) == "error"
                        e = entries(k);
                        text = sprintf("%s:%d:%d: %s", file, e.line, e.col, e.message);
                        break
                    end
                end
                if text == ""
                    text = file + ": " + string(msg);
                end
                error("phasecont:engine:model", "%s", text);
            end
            aot = phasecont.Model.emptyAotInfo();
            try
                % A delay model has no AOT route; use the JIT without a warning.
                isDelay = false;
                first = jsondecode(pckern("model_info", id));
                if isfield(first, "delay")
                    isDelay = logical(first.delay);
                end
                if isDelay && obj.Evaluator == "aot"
                    obj.Evaluator = "jit";
                end
                if obj.Evaluator == "aot"
                    [ok, aot] = phasecont.Model.compileAot(id);
                    if ~ok
                        obj.Evaluator = "jit";
                    end
                end
                if obj.Evaluator ~= "aot"
                    pckern("model_compile", id, ...
                        sprintf('{"evaluator":"%s"}', obj.Evaluator));
                end
                info = jsondecode(pckern("model_info", id));
            catch err
                try
                    pckern("free", id);
                catch
                end
                rethrow(err);
            end
            obj.Id = id;
            % Warn after obj.Id is set, so delete frees the handle if a warning throws.
            for k = 1:numel(entries)
                if string(entries(k).severity) == "warning"
                    e = entries(k);
                    % ODE0054 says a delay model needs the dde-* commands of the
                    % CLI. This front end runs it, so only Diagnostics keeps it.
                    if isDelay && string(e.code) == "ODE0054"
                        continue
                    end
                    warning("phasecont:model:parseWarning", "%s", ...
                        sprintf("%s:%d:%d: %s", file, e.line, e.col, e.message));
                end
            end
            obj.Variables = phasecont.Model.toNames(info.variables);
            obj.Parameters = phasecont.Model.toNames(info.parameters);
            obj.ParameterValues = phasecont.Model.toColumn(info.values);
            obj.Init = phasecont.Model.toColumn(info.init);
            obj.Evaluator = string(info.evaluator);
            obj.AotInfo = aot;
            obj.Autonomous = logical(info.autonomous);
            obj.IsDelay = isDelay;
            obj.SourceSha256 = string(info.source_sha256);
            obj.Options = phasecont.Model.toOptions(info.options);
            obj.Diagnostics = entries;
        end
    end

    methods (Static, Access = private)
        function s = emptyAotInfo()
            s = struct("Compiler", "", "Version", "", "Flags", "", "Library", "");
        end

        function [ok, aot] = compileAot(id)
            % Build and load the AOT library for model id. On any failure
            % warn and return ok = false; the caller then builds the JIT.
            ok = false;
            aot = phasecont.Model.emptyAotInfo();
            try
                [lib, built] = phasecont.internal.aotBuild(id);
            catch err
                warning("phasecont:aot:compile", ...
                    "AOT compile failed, using the JIT: %s", err.message);
                return
            end
            try
                pckern("model_compile", id, jsonencode( ...
                    struct("evaluator", "aot", "aot_lib", char(lib))));
            catch err
                warning("phasecont:aot:load", ...
                    "AOT library could not be loaded, using the JIT: %s", ...
                    err.message);
                return
            end
            ok = true;
            aot = built;
        end

        function full = fullPath(file)
            % Absolute path: keep an absolute name, resolve others against pwd.
            f = char(file);
            if ispc
                isAbs = ~isempty(regexp(f, '^([A-Za-z]:[\\/]|\\\\)', 'once'));
            else
                isAbs = startsWith(f, "/");
            end
            if isAbs
                full = string(file);
            else
                full = string(fullfile(pwd, f));
            end
        end

        function names = toNames(v)
            % jsondecode gives a cell for a string list and [] for an empty one.
            if isempty(v)
                names = strings(1, 0);
            else
                names = reshape(string(v), 1, []);
            end
        end

        function s = toOptions(v)
            % The name/value list of model_info as a struct of strings.
            % jsondecode gives [] for no entries and a struct array otherwise.
            s = struct();
            for k = 1:numel(v)
                s.(matlab.lang.makeValidName(string(v(k).name))) = ...
                    string(v(k).value);
            end
        end

        function col = toColumn(v)
            if isempty(v)
                col = zeros(0, 1);
            else
                col = double(v(:));
            end
        end

        function s = toStructs(v)
            % Diagnostics as a column struct array, 0-by-1 when empty.
            if isempty(v)
                s = repmat(struct("code", [], "severity", [], "file", [], ...
                    "line", [], "col", [], "end_line", [], "end_col", [], ...
                    "message", [], "fix_it", [], "notes", []), 0, 1);
            elseif iscell(v)
                % Entries with different fields: give each the union.
                names = strings(1, 0);
                for k = 1:numel(v)
                    names = union(names, string(fieldnames(v{k}))', "stable");
                end
                s = repmat(cell2struct(cell(numel(names), 1), names, 1), numel(v), 1);
                for k = 1:numel(v)
                    for n = string(fieldnames(v{k}))'
                        s(k).(n) = v{k}.(n);
                    end
                end
                s = s(:);
            else
                s = v(:);
            end
        end
    end
end
