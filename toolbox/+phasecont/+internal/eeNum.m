function s = eeNum(v, dg)
%EENUM Print real values in the engineering E form (SPEC section 8ah).
%   S = phasecont.internal.eeNum(V, DG) returns a string array the size of
%   the real array V. A finite nonzero entry is rounded to DG significant
%   digits as sprintf("%.*e", DG - 1, .) rounds it and written with an
%   exponent E that is a multiple of 3 and a mantissa from 1 to below 1000:
%   "9.78E3", "500E-3", "-20E-6". When E is 0 only the mantissa prints
%   ("978"). Zeros at the end of the fraction are dropped. A zero (or -0)
%   prints "0"; NaN, Inf and -Inf print as "NaN", "Inf" and "-Inf".
s = strings(size(v));
if isempty(v)
    return
end
v = double(v(:));
out = strings(numel(v), 1);
out(v == 0) = "0";
bad = ~isfinite(v);
if any(bad)
    out(bad) = compose("%g", v(bad));
end
ok = isfinite(v) & v ~= 0;
if any(ok)
    w = v(ok);
    t = compose("%." + (dg - 1) + "e", abs(w));
    e = double(extractAfter(t, "e"));
    E = 3 * floor(e / 3);
    k = e - E + 1;                       % digits before the point, 1 to 3
    d = erase(extractBefore(t, "e"), ".") + "00";
    m = extractBefore(d, k + 1);
    fr = strip(extractAfter(d, k), "right", "0");
    has = strlength(fr) > 0;
    m(has) = m(has) + "." + fr(has);
    has = E ~= 0;
    m(has) = m(has) + "E" + string(E(has));
    neg = w < 0;
    m(neg) = "-" + m(neg);
    out(ok) = m;
end
s(:) = out;
end
