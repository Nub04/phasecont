function [text, optsJson, skipped] = importRequest(source, init, params, stateNames, title)
%IMPORTREQUEST Text and options JSON for the MATLAB importer.
%   [TEXT, OPTS, SKIPPED] = phasecont.internal.importRequest(SOURCE, INIT,
%   PARAMS, STATENAMES, TITLE) prepares one call of
%   pckern("import", "matlab", TEXT, OPTS).
%
%   SOURCE is a function handle or the path of a .m file. For an anonymous
%   handle TEXT is func2str(H) and the numeric variables it captured become
%   constants of the model. For a handle to a function in its own .m file,
%   TEXT is that file. Built-in, compiled, local and nested functions have
%   no text to translate and raise phasecont:import:unsupported.
%
%   INIT is a numeric vector, one value per state, or empty. PARAMS is a
%   scalar struct of finite real numbers; its field order gives p(1), p(2)
%   and so on. STATENAMES is a string vector or empty. TITLE is a string or
%   "". Nothing is defaulted: what is missing stays missing and the
%   importer says so.
%
%   SKIPPED lists captured variables that could not be passed on (matrices,
%   text, handles, non-finite values), so an `undeclared` error from the
%   importer can name them.
arguments
    source
    init (:,1) double = zeros(0, 1)
    params (1,1) struct = struct()
    stateNames (:,1) string = strings(0, 1)
    title (1,1) string = ""
end

captures = struct("name", {}, "value", {});
skipped = strings(0, 1);
if isa(source, "function_handle")
    [text, ws] = handleText(source);
    names = fieldnames(ws);
    for k = 1:numel(names)
        v = ws.(names{k});
        if isnumeric(v) || islogical(v)
            v = double(v);
        end
        if isa(v, "double") && isreal(v) && ~isempty(v) && isvector(v) ...
                && all(isfinite(v))
            captures(end+1) = struct("name", string(names{k}), ...
                "value", {numberList(v)}); %#ok<AGROW>
        else
            skipped(end+1, 1) = string(names{k}); %#ok<AGROW>
        end
    end
else
    file = string(source);
    if ~isfile(file)
        error("phasecont:import:file", "Cannot find the file %s.", file);
    end
    text = string(fileread(file, "Encoding", "UTF-8"));
end

parts = strings(0, 1);
if ~isempty(captures)
    items = strings(numel(captures), 1);
    for k = 1:numel(captures)
        items(k) = "{""name"":" + jsonencode(captures(k).name) + ...
            ",""value"":" + captures(k).value + "}";
    end
    parts(end+1) = """captures"":[" + join(items.', ",") + "]";
end
pNames = string(fieldnames(params));
if ~isempty(pNames)
    items = strings(numel(pNames), 1);
    for k = 1:numel(pNames)
        v = params.(pNames(k));
        if ~(isnumeric(v) || islogical(v)) || ~isreal(v) || ~isscalar(v) ...
                || ~isfinite(v)
            error("phasecont:model:badValue", ...
                "Params.%s must be a finite real numeric scalar.", pNames(k));
        end
        items(k) = "{""name"":" + jsonencode(pNames(k)) + ...
            ",""value"":" + phasecont.internal.num2json(double(v)) + "}";
    end
    parts(end+1) = """params"":[" + join(items.', ",") + "]";
end
if ~isempty(stateNames)
    quoted = strings(numel(stateNames), 1);
    for k = 1:numel(stateNames)
        quoted(k) = jsonencode(stateNames(k));
    end
    parts(end+1) = """state_names"":[" + join(quoted.', ",") + "]";
end
if ~isempty(init)
    if ~all(isfinite(init))
        error("phasecont:model:badValue", ...
            "Init must hold finite real numbers.");
    end
    parts(end+1) = """init"":" + numberList(init, true);
end
if title ~= ""
    parts(end+1) = """title"":" + jsonencode(title);
end
if isempty(parts)
    optsJson = "{}";
else
    optsJson = "{" + join(parts.', ",") + "}";
end
end

function [text, ws] = handleText(h)
% The source text of a handle and the variables an anonymous one captured.
info = functions(h);
ws = struct();
switch string(info.type)
    case "anonymous"
        text = string(func2str(h));
        for k = 1:numel(info.workspace)
            w = info.workspace{k};
            for f = fieldnames(w).'
                ws.(f{1}) = w.(f{1});
            end
        end
    case "simple"
        file = string(info.file);
        if file == "" || ~endsWith(file, ".m", IgnoreCase = true) ...
                || ~isfile(file)
            error("phasecont:import:unsupported", ...
                "The function %s has no .m file to translate. Built-in, " + ...
                "compiled and encrypted functions cannot be imported.", ...
                func2str(h));
        end
        text = string(fileread(file, "Encoding", "UTF-8"));
    otherwise
        error("phasecont:import:unsupported", ...
            "Local and nested functions cannot be translated. Put the " + ...
            "function in its own .m file or write an anonymous function.");
end
end

function s = numberList(v, forceList)
% JSON for a numeric vector: a bare number for one value, else a list.
if nargin < 2
    forceList = false;
end
items = strings(1, numel(v));
for k = 1:numel(v)
    items(k) = phasecont.internal.num2json(v(k));
end
if isscalar(v) && ~forceList
    s = items;
else
    s = "[" + join(items, ",") + "]";
end
end
