function value = config(name, value)
%CONFIG Read or set a PhaseCont session option.
%   V = phasecont.config("Codegen") returns the code path used for new
%   models that do not name an Evaluator, "jit" (default) or "aot".
%   phasecont.config("Codegen", "aot") sets it for this MATLAB session.
%   Nothing is saved between sessions.
%
%   An unknown name raises phasecont:config:badName and a value that the
%   option does not accept raises phasecont:config:badValue.
arguments
    name (1,1) string
    value = []
end
persistent codegen
if isempty(codegen)
    codegen = "jit";
end
if name ~= "Codegen"
    error("phasecont:config:badName", ...
        "Unknown option ""%s""; the only option is ""Codegen"".", name);
end
if nargin > 1
    if ~((isstring(value) || ischar(value)) && isscalar(string(value)) ...
            && any(string(value) == ["jit" "aot"]))
        error("phasecont:config:badValue", ...
            "Codegen must be ""jit"" or ""aot"".");
    end
    codegen = string(value);
end
if nargin < 2 || nargout > 0
    value = codegen;
end
end
