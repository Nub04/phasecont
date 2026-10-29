classdef Model < handle
    % An ODE model read from an XPP/XPPAUT .ode file.
    %
    %   m = phasecont.Model(file)
    %   m = phasecont.Model(file, Evaluator = "fallback")
    %   f = m.rhs(x)
    %   f = m.rhs(x, p)       % p: numeric vector or struct of parameters
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
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["jit" "fallback"])} = "jit"
            end
            if exist("pckern", "file") ~= 3
                error("phasecont:engine:load", ...
                    "pckern.%s is not on the MATLAB path. Add the folder " + ...
                    "that holds the PhaseCont binaries to the path.", mexext);
            end
            full = phasecont.Model.fullPath(file);
            if ~isfile(full)
                error("phasecont:model:fileNotFound", "%s", ...
                    "The model file """ + full + """ does not exist.");
            end
            text = string(fileread(full, Encoding = "UTF-8"));
            obj.File = full;
            obj.Source = text;
            obj.Evaluator = options.Evaluator;
            obj.build();
        end

        function f = rhs(obj, x, p)
            % F = M.RHS(X) or M.RHS(X, P): the right-hand side at X.
            arguments
                obj (1,1) phasecont.Model
                x {mustBeNumeric, mustBeReal}
                p = []
            end
            if nargin < 3
                p = obj.ParameterValues;
            else
                p = obj.paramVector(p);
            end
            f = pckern("model_rhs", obj.Id, double(x), p);
            f = f(:);
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
            for k = 1:numel(entries)
                if string(entries(k).severity) == "warning"
                    e = entries(k);
                    warning("phasecont:model:parseWarning", "%s", ...
                        sprintf("%s:%d:%d: %s", file, e.line, e.col, e.message));
                end
            end
            try
                pckern("model_compile", id, ...
                    sprintf('{"evaluator":"%s"}', obj.Evaluator));
                info = jsondecode(pckern("model_info", id));
            catch err
                try
                    pckern("free", id);
                catch
                end
                rethrow(err);
            end
            obj.Id = id;
            obj.Variables = phasecont.Model.toNames(info.variables);
            obj.Parameters = phasecont.Model.toNames(info.parameters);
            obj.ParameterValues = phasecont.Model.toColumn(info.values);
            obj.Init = phasecont.Model.toColumn(info.init);
            obj.Evaluator = string(info.evaluator);
            obj.Diagnostics = entries;
        end
    end

    methods (Static, Access = private)
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
