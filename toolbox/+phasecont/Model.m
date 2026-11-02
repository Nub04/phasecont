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
