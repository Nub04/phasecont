function [X, info] = inverse(A, options)
%INVERSE Invert a small matrix by a hand method (SPEC section 8ah).
%   [X, INFO] = phasecont.ee.inverse(A, Method = M) gives inv(A) for a real
%   or complex double A of order 1 to 8. Method is required:
%
%     "adjugate"     X = adj(A)/det(A), adj(A) the transpose of the
%                    cofactor matrix
%     "gaussjordan"  Gauss-Jordan on [A | I]; X is the right block
%
%   Tol (default 1e-9) is the singular test: A is refused when
%   rcond(A) <= Tol. INFO has the same fields as in phasecont.ee.solve.
%
%   Errors: phasecont:ee:badSize, phasecont:ee:nonfinite,
%   phasecont:ee:outOfRange, phasecont:ee:singular and phasecont:model:badOption, as in
%   phasecont.ee.solve.
%
%   See also phasecont.ee.solve, phasecont.ee.jstr.
arguments
    A
    options.Method = []
    options.Tol = 1e-9
end
[method, tol] = phasecont.internal.eeOptions(options, ["adjugate" "gaussjordan"]);
rc = phasecont.internal.eeCheck(A, tol);
n = size(A, 1);
info = struct("Method", method, "RCond", rc, "Det", NaN, "DetB", [], ...
    "Adjugate", [], "Steps", ...
    reshape(struct("Op", {}, "Rows", {}, "Factor", {}, "Matrix", {}), 0, 1));
switch method
    case "adjugate"
        d = det(A);
        adj = phasecont.internal.eeAdjugate(A);
        phasecont.internal.eeBackstop(d);
        X = adj / d;
        phasecont.internal.eeBackstop([], adj, X);
        info.Det = d;
        info.Adjugate = adj;
    case "gaussjordan"
        [M, steps] = phasecont.internal.eeGaussJordan([A eye(n)], n);
        X = M(:, n+1:end);
        phasecont.internal.eeBackstop([], steps, X);
        info.Steps = steps;
end
end
