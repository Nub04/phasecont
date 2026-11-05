function s = num2ode(x)
%NUM2ODE Write a finite double as .ode number text, with the fewest digits.
%   S = phasecont.internal.num2ode(X) returns the shortest of %.15g, %.16g
%   and %.17g that reads back as the same double. The .ode lexer reads
%   every form this gives, including 1e-05 and 1e+300. NaN and Inf raise
%   phasecont:model:badValue.
arguments
    x (1,1) double
end
if ~isfinite(x)
    error("phasecont:model:badValue", ...
        "A number in the model must be finite, got %s.", num2str(x));
end
for digits = 15:17
    s = string(sprintf("%.*g", digits, x));
    if str2double(s) == x
        return
    end
end
end
