function [method, tol] = eeOptions(options, allowed)
%EEOPTIONS Check the Method and Tol options of the EE mode (SPEC 8ah).
%   [METHOD, TOL] = phasecont.internal.eeOptions(OPTIONS, ALLOWED) returns
%   the method as a string and Tol as a double. Method has no default.
method = options.Method;
if isempty(method)
    error("phasecont:model:badOption", ...
        "Method is required: %s.", strjoin('"' + allowed + '"', ", "));
end
if ~((isstring(method) && isscalar(method) && ~ismissing(method)) || ...
        (ischar(method) && isrow(method))) || ~any(string(method) == allowed)
    error("phasecont:model:badOption", ...
        "Method must be one of %s.", strjoin('"' + allowed + '"', ", "));
end
method = string(method);
tol = options.Tol;
if ~(isnumeric(tol) && isreal(tol) && isscalar(tol) && isfinite(tol) ...
        && tol >= 0 && tol < 1)
    error("phasecont:model:badOption", ...
        "Tol must be a finite real number with 0 <= Tol < 1.");
end
tol = double(tol);
end
