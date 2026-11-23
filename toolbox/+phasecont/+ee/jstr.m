function s = jstr(z, options)
%JSTR Print complex numbers in the engineering j form (SPEC section 8ah).
%   S = phasecont.ee.jstr(Z) returns a string array the size of the
%   numeric array Z, with each entry a + jb written as "a + j|b|" or
%   "a - j|b|". A zero imaginary part prints "a", a zero real part prints
%   "j|b|" or "-j|b|", and -0 prints as 0. Parts print with
%   sprintf("%.*g", Digits, .); Digits (default 6) is a whole number from
%   1 to 17. NaN and Inf print as MATLAB prints them.
%
%   Errors: phasecont:ee:badSize (Z not numeric) and
%   phasecont:model:badOption (bad Digits).
%
%   See also phasecont.ee.solve, phasecont.ee.inverse.
arguments
    z
    options.Digits = 6
end
dg = options.Digits;
if ~(isnumeric(dg) && isreal(dg) && isscalar(dg) && isfinite(dg) ...
        && dg == round(dg) && dg >= 1 && dg <= 17)
    error("phasecont:model:badOption", ...
        "Digits must be a whole number from 1 to 17.");
end
if ~isnumeric(z)
    error("phasecont:ee:badSize", "z must be numeric, not %s.", class(z));
end
dg = double(dg);
s = strings(size(z));
for k = 1:numel(z)
    a = double(real(z(k)));
    b = double(imag(z(k)));
    if b == 0
        s(k) = part(a, dg);
    elseif a == 0
        if b < 0
            s(k) = "-j" + part(-b, dg);
        else
            s(k) = "j" + part(b, dg);
        end
    elseif b < 0
        s(k) = part(a, dg) + " - j" + part(-b, dg);
    else
        % b > 0 or NaN
        s(k) = part(a, dg) + " + j" + part(b, dg);
    end
end
end

function t = part(v, dg)
if v == 0
    v = 0; % -0 prints as 0
end
t = string(sprintf("%.*g", dg, v));
end
