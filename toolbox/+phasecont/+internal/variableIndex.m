function vi = variableIndex(var, n, names)
%VARIABLEINDEX Column of U for a variable name or index.
%   VI = phasecont.internal.variableIndex(VAR, N, NAMES) returns the index of
%   the variable VAR in a model with N states named NAMES. VAR is an integer
%   from 1 to N or a name in NAMES. An empty VAR gives 1. NAMES may be empty
%   when only N is known; then only an index is accepted. Anything else
%   raises phasecont:model:unknownVariable.
arguments
    var
    n (1,1) double
    names string
end
if isempty(var)
    vi = 1;
elseif isnumeric(var) && isscalar(var) && var == round(var) ...
        && var >= 1 && var <= n
    vi = double(var);
elseif (isstring(var) || ischar(var)) && isscalar(string(var)) ...
        && any(names == string(var))
    vi = find(names == string(var), 1);
else
    error("phasecont:model:unknownVariable", ...
        "Var must be one of: %s.", strjoin(names, ", "));
end
end
