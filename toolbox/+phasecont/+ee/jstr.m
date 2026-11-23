function s = jstr(z, options)
%JSTR Print complex numbers in the engineering j form (SPEC section 8ah).
%   S = phasecont.ee.jstr(Z) returns a string array the size of the
%   numeric array Z, with each entry a + jb written as "a + j|b|" or
%   "a - j|b|". A zero imaginary part prints "a", a zero real part prints
%   "j|b|" or "-j|b|", and -0 prints as 0. Each part is rounded to Digits
%   significant digits (default 6, a whole number from 1 to 17) and
%   printed in engineering E form: the exponent is a multiple of 3 and is
%   left out when it is 0, so 978 prints "978", 9780 prints "9.78E3" and
%   0.5 prints "500E-3". NaN and Inf print as MATLAB prints them.
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
s = strings(size(z));
if isempty(z)
    return
end
dg = double(dg);
a = double(real(z(:)));
b = double(imag(z(:)));
n = numel(a);
p = phasecont.internal.eeNum([a; abs(b)], dg);
t = p(1:n);                            % b == 0, including -0
sb = p(n+1:end);
neg = b < 0;
im = b ~= 0;                           % true for NaN
re = a ~= 0;                           % true for NaN
only = im & ~re;
t(only & neg) = "-j" + sb(only & neg);
t(only & ~neg) = "j" + sb(only & ~neg);
both = im & re;
t(both & neg) = t(both & neg) + " - j" + sb(both & neg);
t(both & ~neg) = t(both & ~neg) + " + j" + sb(both & ~neg);
s(:) = t;
end
