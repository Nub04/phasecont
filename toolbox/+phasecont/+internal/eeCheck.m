function rc = eeCheck(A, tol, b)
%EECHECK Shared checks of the EE mode (SPEC section 8ah).
%   RC = phasecont.internal.eeCheck(A, TOL) checks that A is a square
%   double matrix of order 1 to 8 with no NaN or Inf entry and that
%   rcond(A) > TOL, and returns rcond(A). RC = eeCheck(A, TOL, B) also
%   checks that B is a double column of length size(A, 1) with no NaN or
%   Inf entry. Every nonzero entry of A and B must have a magnitude from
%   2^-100 to 2^100, so that det(A), det(B_i) and the adjugate stay inside
%   the double range at order 8. Every method of phasecont.ee.solve and phasecont.ee.inverse
%   calls it first, so they refuse the same matrices.
if ~(isa(A, "double") && ~issparse(A) && ismatrix(A) ...
        && size(A, 1) == size(A, 2) && size(A, 1) >= 1 && size(A, 1) <= 8)
    error("phasecont:ee:badSize", ...
        "A must be a square double matrix of order 1 to 8, not a %s %s.", ...
        strjoin(string(size(A)), " by "), class(A));
end
n = size(A, 1);
hasB = nargin >= 3;
if ~hasB
    b = zeros(0, 1);
end
if hasB && ~(isa(b, "double") && ~issparse(b) && iscolumn(b) && numel(b) == n)
    error("phasecont:ee:badSize", ...
        "b must be a double column of length %d, not a %s %s.", n, ...
        strjoin(string(size(b)), " by "), class(b));
end
if ~all(isfinite(A), "all") || (hasB && ~all(isfinite(b)))
    error("phasecont:ee:nonfinite", "A and b must not hold NaN or Inf.");
end
lo = 2^-100;
hi = 2^100;
v = abs([A(:); b(:)]);
v = v(v ~= 0);
if any(v < lo | v > hi)
    error("phasecont:ee:outOfRange", ...
        "Every nonzero entry of A and b must have a magnitude from 2^-100 to 2^100 (about %.3g to %.3g); rescale A and b.", ...
        lo, hi);
end
rc = rcond(A);
if ~(rc > tol)
    error("phasecont:ee:singular", ...
        "A is singular to the tolerance: rcond(A) = %.3g, not above Tol = %.3g.", ...
        rc, tol);
end
end
