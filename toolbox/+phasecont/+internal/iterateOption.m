function k = iterateOption(m, value)
% The Iterate option of equilibrium and continuation (SPEC 8y): [] when it
% is not given, else a whole number from 1 to 1e6. The option applies to a
% map only. Only the default, a 0-by-0 double, means not given, so any
% other empty value is refused like any other bad value.
k = [];
if isa(value, "double") && isequal(size(value), [0 0])
    return
end
if ~m.IsMap
    error("phasecont:model:badOption", "Iterate applies to a map only.");
end
if ~(isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value) ...
        && value == round(value) && value >= 1 && value <= 1e6)
    error("phasecont:model:badOption", ...
        "Iterate must be a whole number from 1 to 1000000.");
end
k = double(value);
end
