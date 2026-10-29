function s = num2json(x)
%NUM2JSON Write a finite double as a JSON number with 17 significant digits.
%   S = phasecont.internal.num2json(X) returns sprintf('%.17g', X), which
%   reads back as the same double. jsonencode can drop digits, so numbers
%   in the options JSON of the engine never go through it. NaN and Inf
%   raise phasecont:model:badOption.
arguments
    x (1,1) double
end
if ~isfinite(x)
    error("phasecont:model:badOption", ...
        "Options must be finite numbers, got %s.", num2str(x));
end
s = string(sprintf("%.17g", x));
end
