function [equations, params, init] = tableToEquations(states, params, init, aux)
%TABLETOEQUATIONS Turn struct or table input into equation text and structs.
%   [EQ, P, X0] = phasecont.internal.tableToEquations(STATES, PARAMS, INIT,
%   AUX) reads each argument as a struct (name -> value) or a table:
%
%     STATES  table with Name and Rhs columns, and an optional Init column;
%             or a struct whose fields are the states and whose values are
%             the right-hand sides (text, or a number for a constant).
%     PARAMS  table with Name and Value columns, or a struct of numbers.
%     INIT    table with Name and Value columns, or a struct of numbers.
%     AUX     table with Name and Expr columns, or a struct of text.
%
%   EQ is one line per state (`x' = rhs`) followed by one line per aux
%   (`aux name=expr`). P and X0 are structs ready for phasecont.internal.
%   odeText. A NaN in the Init column of STATES counts as "no initial value
%   given"; the model build then names that state in missingInit. Nothing
%   is defaulted.
%
%   Raises phasecont:model:badTable, badName, badValue, duplicateName and
%   ambiguousInit.
arguments
    states
    params = struct()
    init = struct()
    aux = struct()
end

[sNames, sRhs, sExtra] = rows(states, "States", "Rhs", "Init");
if isempty(sNames)
    error("phasecont:model:badTable", "States must list at least one state.");
end
checkNames(sNames, "States");
lines = strings(numel(sNames), 1);
for k = 1:numel(sNames)
    lines(k) = sNames(k) + "' = " + exprText(sRhs{k}, ...
        sprintf("States: the right-hand side of %s", sNames(k)));
end

[aNames, aExpr] = rows(aux, "Aux", "Expr");
checkNames(aNames, "Aux");
auxLines = strings(numel(aNames), 1);
for k = 1:numel(aNames)
    auxLines(k) = "aux " + aNames(k) + "=" + exprText(aExpr{k}, ...
        sprintf("Aux: the expression of %s", aNames(k)));
end
equations = join([lines; auxLines], newline);

[pNames, pVals] = rows(params, "Params", "Value");
checkNames(pNames, "Params");
for k = 1:numel(pVals)
    if isnumeric(pVals{k}) && isscalar(pVals{k}) && isnan(pVals{k})
        error("phasecont:model:missingValue", ...
            "Params: %s has no value.", pNames(k));
    end
end
params = toStruct(pNames, pVals);

[iNames, iVals] = rows(init, "Init", "Value");
checkNames(iNames, "Init");
init = toStruct(iNames, iVals);

if isfield(sExtra, "Init")
    if ~isempty(iNames)
        error("phasecont:model:ambiguousInit", ...
            "Initial values come from the Init column of States and from " + ...
            "the Init argument. Give them in one place.");
    end
    have = false(numel(sNames), 1);
    for k = 1:numel(sNames)
        v = sExtra.Init{k};
        have(k) = ~(isnumeric(v) && isscalar(v) && isnan(v));
    end
    init = toStruct(sNames(have), sExtra.Init(have));
end
end

function [names, values, extra] = rows(x, label, valueVar, optionalVar)
% Names and values of a struct or table, plus the optional columns of a table.
extra = struct();
if isstruct(x)
    if ~isscalar(x)
        error("phasecont:model:badTable", ...
            "%s must be a scalar struct or a table.", label);
    end
    names = string(fieldnames(x));
    values = struct2cell(x);
    values = values(:);
elseif istable(x)
    vars = string(x.Properties.VariableNames);
    need = ["Name" string(valueVar)];
    absent = need(~ismember(need, vars));
    if ~isempty(absent)
        error("phasecont:model:badTable", ...
            "%s table needs the column %s.", label, join(absent, ", "));
    end
    names = string(x.Name);
    values = cells(x.(valueVar), label);
    if nargin > 3 && ismember(string(optionalVar), vars)
        extra.(optionalVar) = cells(x.(optionalVar), label);
    end
else
    error("phasecont:model:badTable", ...
        "%s must be a struct or a table, got %s.", label, class(x));
end
names = names(:);
end

function c = cells(col, label)
% A column as a cell array with one value per row.
if iscell(col)
    c = col(:);
elseif iscategorical(col)
    c = num2cell(string(col(:)));
elseif isstring(col) || isnumeric(col) || islogical(col)
    c = num2cell(col(:));
else
    error("phasecont:model:badTable", ...
        "%s: a column of class %s is not supported.", label, class(col));
end
end

function checkNames(names, label)
if isempty(names)
    return
end
for k = 1:numel(names)
    if isempty(regexp(names(k), '^[A-Za-z_][A-Za-z0-9_]*$', 'once'))
        error("phasecont:model:badName", ...
            "%s: ""%s"" is not a valid name; use letters, digits and " + ...
            "underscores, starting with a letter or underscore.", ...
            label, names(k));
    end
end
[u, ~, idx] = unique(names);
count = accumarray(idx(:), 1);
dup = u(count > 1);
if ~isempty(dup)
    error("phasecont:model:duplicateName", ...
        "%s: the name %s is listed more than once.", label, dup(1));
end
end

function s = toStruct(names, values)
s = struct();
for k = 1:numel(names)
    s.(names(k)) = values{k};
end
end

function t = exprText(v, where)
% Right-hand sides are text on one line, or a finite real number.
if ischar(v) && (isrow(v) || isempty(v))
    v = string(v);
end
if isstring(v) && isscalar(v) && ~ismissing(v)
    t = strtrim(v);
    if t == ""
        error("phasecont:model:badValue", "%s is empty.", where);
    end
    if contains(t, newline) || contains(t, sprintf("\r"))
        error("phasecont:model:badValue", ...
            "%s must fit on one line.", where);
    end
elseif (isnumeric(v) || islogical(v)) && isreal(v) && isscalar(v) && isfinite(v)
    t = phasecont.internal.num2ode(double(v));
else
    error("phasecont:model:badValue", ...
        "%s must be text or a finite real number.", where);
end
end
