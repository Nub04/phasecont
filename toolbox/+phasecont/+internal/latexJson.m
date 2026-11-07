function json = latexJson(kind, spec, opts)
%LATEXJSON Build the JSON document that the LaTeX renderer reads.
%   J = phasecont.internal.latexJson(KIND, SPEC, OPTS) returns the text for
%   pckern("latex", KIND, J). KIND is "pgfplots" or "table".
%
%   For "pgfplots", SPEC has fields XLabel, YLabel, Series (struct array
%   with X, Y, Class, and optional Name and Color) and optionally Title and
%   Marks (struct array with Label, X, Y). For "table", SPEC has Columns
%   (string array) and Rows (struct array with Label, Type, Values).
%
%   OPTS has Digits and Standalone, and for "pgfplots" also Data
%   ("inline" or "external") and Basename.
%
%   The JSON is written by hand. jsonencode turns a one-element vector into
%   a scalar and can drop digits, and the renderer needs arrays and every
%   digit. Numbers print with %.17g. NaN and Inf print as null, which the
%   renderer reports as a non-finite value with its index. A missing Digits
%   raises phasecont:latex:missingDigits, and a field of the wrong shape
%   raises phasecont:latex:badInput.
arguments
    kind (1,1) string {mustBeMember(kind, ["pgfplots" "table"])}
    spec (1,1) struct
    opts (1,1) struct
end

if ~isfield(opts, "Digits") || ~isscalar(opts.Digits) || isnan(opts.Digits)
    error("phasecont:latex:missingDigits", ...
        "Digits is required: the number of significant digits to print, 1 to 17.");
end

doc = struct();
switch kind
    case "pgfplots"
        need(spec, ["XLabel" "YLabel" "Series"], "figure");
        doc.xlabel = text(spec.XLabel, "XLabel");
        doc.ylabel = text(spec.YLabel, "YLabel");
        if isfield(spec, "Title") && strlength(text(spec.Title, "Title")) > 0
            doc.title = text(spec.Title, "Title");
        end
        if opts.Data == "external"
            doc.data = "external";
            doc.basename = text(opts.Basename, "Basename");
        end
        doc.series = seriesDocs(spec.Series);
        if isfield(spec, "Marks") && ~isempty(spec.Marks)
            doc.marks = markDocs(spec.Marks);
        end
    case "table"
        need(spec, ["Columns" "Rows"], "table");
        doc.columns = textArray(spec.Columns, "Columns");
        doc.rows = rowDocs(spec.Rows);
end
doc.digits = opts.Digits;
if opts.Standalone
    doc.standalone = true;
end
json = enc(doc, "");
end

function docs = seriesDocs(series)
if ~isstruct(series)
    badInput("Series must be a struct array.");
end
docs = cell(numel(series), 1);
for k = 1:numel(series)
    s = series(k);
    need(s, ["X" "Y" "Class"], "series " + k);
    d = struct();
    if isfield(s, "Name") && strlength(text(s.Name, "Name")) > 0
        d.name = text(s.Name, "Name");
    end
    d.class = text(s.Class, "Class");
    if isfield(s, "Color") && strlength(text(s.Color, "Color")) > 0
        d.color = text(s.Color, "Color");
    end
    d.x = numbers(s.X, "series " + k + " X");
    d.y = numbers(s.Y, "series " + k + " Y");
    docs{k} = d;
end
end

function docs = markDocs(marks)
if ~isstruct(marks)
    badInput("Marks must be a struct array.");
end
docs = cell(numel(marks), 1);
for k = 1:numel(marks)
    m = marks(k);
    need(m, ["Label" "X" "Y"], "mark " + k);
    d = struct();
    d.label = text(m.Label, "Label");
    d.x = numbers(m.X, "mark " + k + " X");
    d.y = numbers(m.Y, "mark " + k + " Y");
    if ~isscalar(d.x) || ~isscalar(d.y)
        badInput("mark " + k + " needs one X and one Y.");
    end
    docs{k} = d;
end
end

function docs = rowDocs(rows)
if ~isstruct(rows)
    badInput("Rows must be a struct array.");
end
docs = cell(numel(rows), 1);
for k = 1:numel(rows)
    r = rows(k);
    need(r, ["Label" "Type" "Values"], "row " + k);
    d = struct();
    d.label = text(r.Label, "Label");
    d.type = text(r.Type, "Type");
    d.values = numbers(r.Values, "row " + k + " Values");
    docs{k} = d;
end
end

function need(s, names, what)
missing = names(~isfield(s, names));
if ~isempty(missing)
    badInput(sprintf("%s needs the field %s.", what, join(missing, ", ")));
end
end

function s = text(v, what)
if ~(isstring(v) || ischar(v)) || ~isscalar(string(v)) || ismissing(string(v))
    badInput(sprintf("%s must be one string.", what));
end
s = string(v);
end

function s = textArray(v, what)
if ~(isstring(v) || ischar(v) || iscellstr(v))
    badInput(sprintf("%s must be a string array.", what));
end
s = string(v);
if any(ismissing(s(:)))
    badInput(sprintf("%s must not hold a missing string.", what));
end
s = s(:);
end

function x = numbers(v, what)
if ~(isnumeric(v) || islogical(v)) || ~isreal(v)
    badInput(sprintf("%s must be real numbers.", what));
end
x = double(v(:));
end

function badInput(message)
error("phasecont:latex:badInput", "%s", message);
end

function s = enc(v, key, parent)
% JSON text for v. A field named in the list is always an array, even with
% one element, except x and y inside a mark, which are single numbers. PARENT
% is the name of the array that holds the struct being encoded.
if nargin < 3
    parent = "";
end
alwaysArray = any(key == ["x" "y" "values" "columns" "series" "marks" "rows"]);
if parent == "marks" && any(key == ["x" "y"])
    alwaysArray = false;
end
if isstruct(v)
    if isscalar(v) && ~alwaysArray
        names = string(fieldnames(v));
        parts = strings(0, 1);
        for k = 1:numel(names)
            parts(end+1, 1) = jsonencode(char(names(k))) + ":" + ...
                enc(v.(names(k)), names(k), parent); %#ok<AGROW>
        end
        s = "{" + join(parts, ",") + "}";
    else
        parts = strings(0, 1);
        for k = 1:numel(v)
            parts(end+1, 1) = enc(v(k), "", key); %#ok<AGROW>
        end
        s = list(parts);
    end
elseif iscell(v)
    parts = strings(0, 1);
    for k = 1:numel(v)
        parts(end+1, 1) = enc(v{k}, "", key); %#ok<AGROW>
    end
    s = list(parts);
elseif isstring(v) || ischar(v)
    v = string(v);
    if isscalar(v) && ~alwaysArray
        s = string(jsonencode(char(v)));
    else
        parts = strings(0, 1);
        for k = 1:numel(v)
            parts(end+1, 1) = string(jsonencode(char(v(k)))); %#ok<AGROW>
        end
        s = list(parts);
    end
elseif islogical(v) && isscalar(v)
    if v
        s = "true";
    else
        s = "false";
    end
elseif isnumeric(v)
    if isscalar(v) && ~alwaysArray
        s = number(v);
    else
        parts = strings(numel(v), 1);
        for k = 1:numel(v)
            parts(k) = number(v(k));
        end
        s = list(parts);
    end
else
    badInput("A value of class " + class(v) + " cannot go into the document.");
end
end

function s = number(x)
if isfinite(x)
    s = string(sprintf("%.17g", x));
else
    s = "null";
end
end

function s = list(parts)
if isempty(parts)
    s = "[]";
else
    s = "[" + join(parts, ",") + "]";
end
end
