function eeBackstop(d, varargin)
%EEBACKSTOP Refuse a result of the EE mode that left the double range.
%   phasecont.internal.eeBackstop(D, V1, V2, ...) raises
%   phasecont:ee:outOfRange when D (det(A), or [] when the method does not
%   use it) is 0, subnormal, Inf or NaN, or when any of V1, V2, ... holds
%   Inf or NaN (SPEC section 8ah, check 5). It catches what the entry
%   window of eeCheck does not bound, such as a Tol near 0.
if ~isempty(d) && ~(isfinite(d) && abs(d) >= realmin)
    error("phasecont:ee:outOfRange", ...
        "det(A) = %.3g is outside the normal double range; use Method = ""gaussjordan"" or rescale A.", ...
        abs(d));
end
for k = 1:numel(varargin)
    v = varargin{k};
    if isstruct(v)
        bad = any(arrayfun(@(s) ~all(isfinite(s.Factor)) ...
            || ~all(isfinite(s.Matrix), "all"), v));
    else
        bad = ~all(isfinite(v), "all");
    end
    if bad
        error("phasecont:ee:outOfRange", ...
            "A result left the double range (Inf or NaN); rescale A and b.");
    end
end
end
