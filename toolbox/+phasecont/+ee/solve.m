function [x, info] = solve(A, b, options)
%SOLVE Solve a small linear system by a hand method (SPEC section 8ah).
%   [X, INFO] = phasecont.ee.solve(A, B, Method = M) solves A*X = B for a
%   real or complex double A of order 1 to 8 and a double column B, by the
%   method a circuits course uses by hand. Method is required:
%
%     "cramer"       X(i) = det(B_i)/det(A), B_i is A with column i
%                    replaced by B
%     "adjugate"     X = adj(A)*B/det(A)
%     "inverse"      inv(A) by Gauss-Jordan on [A | I], then inv(A)*B
%     "gaussjordan"  Gauss-Jordan on [A | B]; X is the last column
%
%   Tol (default 1e-9) is the singular test: A is refused when
%   rcond(A) <= Tol. INFO has the fields Method, RCond (rcond(A)), Det
%   (det(A), NaN for "inverse" and "gaussjordan"), DetB (det(B_i) for
%   "cramer", [] otherwise), Adjugate ("adjugate", [] otherwise) and Steps
%   (the row operations of "inverse" and "gaussjordan", each with Op,
%   Rows, Factor and Matrix; a 0 by 1 struct otherwise).
%
%   For large systems use A\B. These methods show the hand steps.
%
%   Errors: phasecont:ee:badSize (A not square double of order 1 to 8, or
%   B not a double column of length n), phasecont:ee:nonfinite (NaN or
%   Inf), phasecont:ee:outOfRange (a nonzero entry of A or B outside
%   2^-100 to 2^100 in magnitude, or det(A) or a result outside the double
%   range), phasecont:ee:singular (rcond(A) <= Tol) and
%   phasecont:model:badOption (missing or unknown Method, Tol outside
%   0 <= Tol < 1).
%
%   See also phasecont.ee.inverse, phasecont.ee.jstr.
arguments
    A
    b
    options.Method = []
    options.Tol = 1e-9
end
[method, tol] = phasecont.internal.eeOptions(options, ...
    ["cramer" "adjugate" "inverse" "gaussjordan"]);
rc = phasecont.internal.eeCheck(A, tol, b);
n = size(A, 1);
info = struct("Method", method, "RCond", rc, "Det", NaN, "DetB", [], ...
    "Adjugate", [], "Steps", ...
    reshape(struct("Op", {}, "Rows", {}, "Factor", {}, "Matrix", {}), 0, 1));
switch method
    case "cramer"
        d = det(A);
        detB = zeros(n, 1);
        for i = 1:n
            Bi = A;
            Bi(:, i) = b;
            detB(i) = det(Bi);
        end
        phasecont.internal.eeBackstop(d);
        x = detB / d;
        phasecont.internal.eeBackstop([], detB, x);
        info.Det = d;
        info.DetB = detB;
    case "adjugate"
        d = det(A);
        adj = phasecont.internal.eeAdjugate(A);
        phasecont.internal.eeBackstop(d);
        x = adj * b / d;
        phasecont.internal.eeBackstop([], adj, x);
        info.Det = d;
        info.Adjugate = adj;
    case "inverse"
        [M, steps] = phasecont.internal.eeGaussJordan([A eye(n)], n);
        x = M(:, n+1:end) * b;
        phasecont.internal.eeBackstop([], steps, x);
        info.Steps = steps;
    case "gaussjordan"
        [M, steps] = phasecont.internal.eeGaussJordan([A b], n);
        x = M(:, end);
        phasecont.internal.eeBackstop([], steps, x);
        info.Steps = steps;
end
end
