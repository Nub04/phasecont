function r = closedOrbitChecks(m, options)
%CLOSEDORBITCHECKS Test closed-orbit hypotheses on a grid over a rectangle.
%   R = phasecont.closedOrbitChecks(M, Region = [X0 X1 Y0 Y1], Grid = [NX NY],
%   Gradient = true, Lyapunov = V, Dulac = G, Trapping = true) evaluates
%   the hypotheses of the closed-orbit tests of Strogatz, 3rd ed., Sections
%   7.2 and 7.3, at NX*NY samples of the rectangle of a planar ODE M:
%
%     Gradient  J12 = J21 at every sample (a gradient system, Theorem 7.2.3)
%     Lyapunov  Vdot = grad(V).f has one strict sign away from fixed points
%     Dulac     div(G*f) = G*tr(J) + grad(G).f has one strict sign
%               (G = "1" is Bendixson's criterion)
%     Trapping  f.n <= 0 on the boundary, n the outward normal of each side
%
%   What the result does not claim. A check that holds says that its
%   hypothesis holds at every sample, not on the whole rectangle. No result
%   says that closed orbits are absent or present. The trapping check tests
%   the boundary only and says nothing about fixed points inside it, which
%   the Poincare-Bendixson theorem also needs.
%
%   Every derivative is exact: the Jacobian of M and the gradient of V and G
%   come from the engine's symbolic derivatives, not finite differences. V
%   and G are .ode expressions in the state names, the parameter names and
%   the built-in functions; a fixed quantity, a user function or a table of
%   M is not visible in them.
%
%   Name-value options
%     Region    [X0 X1 Y0 Y1], finite, X0 < X1 and Y0 < Y1, in the order of
%               M.Variables (required)
%     Grid      [NX NY], whole numbers of at least 2; the samples are
%               linspace(X0, X1, NX) by linspace(Y0, Y1, NY), corners
%               included (required)
%     Params    struct or numeric vector of parameter values; a struct
%               changes only the fields it names (default M.ParameterValues)
%     Gradient  true to run the gradient check (default false)
%     Lyapunov  expression V to run the Lyapunov check (default none)
%     Dulac     expression G to run the Dulac check (default none)
%     Trapping  true to run the trapping check (default false)
%     Tol       relative tolerance, 0 <= Tol < 1 (default 1e-9): a value q
%               is zero when |q| <= Tol*S, with S the scale of the check
%
%   At least one check must be asked for.
%
%   R is a struct with the fields Region, Grid, P (the parameters used), Tol,
%   Samples (NX*NY) and one struct per check: Gradient, Lyapunov, Dulac and
%   Trapping. Each has
%     Verdict   "holds on samples", "fails", "inconclusive" or "not run"
%     Reason    one sentence
%     Min, Max  the extremes of the tested quantity
%     MinAt, MaxAt  the (x, y) of each, the first in sample order on a tie
%     Samples   how many samples the check tested
%     Scale     the magnitude S the tolerance is relative to
%   A check that did not run has NaN extremes, [NaN NaN] locations and
%   Samples 0. The tested quantities and the extra fields are:
%     Gradient  q = J12 - J21, S = max norm(J, "fro")
%     Lyapunov  q = Vdot, S = max(|V_x f1| + |V_y f2|); a sample with
%               norm(f) <= Tol*max norm(f) is a fixed-point sample and
%               does not count against the check; extra fields Sign,
%               FixedPointSamples, VMin, VMinAt, VMax, VMaxAt (the range of
%               V) and Expression. V is not tested for positivity.
%     Dulac     q = div(G*f), S = max(|G tr J| + |G_x f1| + |G_y f2|);
%               extra fields Sign, ZeroSamples and Expression
%     Trapping  q = f.n on each side, a corner tested against both of its
%               sides; S = max norm(f) on the boundary; Samples is the
%               number of boundary samples, 2*(NX+NY)-4; extra fields Worst
%               (the side of the largest value) and TangentSamples (values
%               that are zero)
%
%   Errors
%     phasecont:orbit:badModel       a map, a delay model, a model that
%                                    uses t, or a model without exactly two
%                                    variables
%     phasecont:model:badOption      a bad or missing option
%     phasecont:orbit:badExpression  V or G does not parse or uses t
%     phasecont:orbit:nonfinite      NaN or Inf in f, J, V, grad V, G or
%                                    grad G at a sample; the message names
%                                    the sample and the quantity
%
%   Example (Strogatz, Example 7.2.4)
%     m = phasecont.Model.fromEquations(["x' = x*(2-x-y)" ...
%         "y' = y*(4*x-x^2-3)"], Init = struct("x", 1, "y", 1));
%     r = phasecont.closedOrbitChecks(m, Region = [0.1 5 0.1 5], ...
%         Grid = [21 21], Dulac = "1/(x*y)");
%     r.Dulac.Verdict   % "holds on samples", Sign "negative"
%
%   See also phasecont.equilibrium, phasecont.Point.
arguments
    m (1,1) phasecont.Model
    options.Region = []
    options.Grid = []
    options.Params = []
    options.Gradient = false
    options.Lyapunov = []
    options.Dulac = []
    options.Trapping = false
    options.Tol = 1e-9
end

checkModel(m);
[x0, x1, y0, y1] = regionOption(options.Region);
[nx, ny] = gridOption(options.Grid);
doGradient = flagOption(options.Gradient, "Gradient");
doTrapping = flagOption(options.Trapping, "Trapping");
lyapunov = expressionOption(options.Lyapunov, "Lyapunov");
dulac = expressionOption(options.Dulac, "Dulac");
tol = options.Tol;
if ~(isnumeric(tol) && isreal(tol) && isscalar(tol) && isfinite(tol) ...
        && tol >= 0 && tol < 1)
    error("phasecont:model:badOption", ...
        "Tol must be a finite real number with 0 <= Tol < 1.");
end
tol = double(tol);
if ~(doGradient || doTrapping || lyapunov ~= "" || dulac ~= "")
    error("phasecont:model:badOption", ...
        "Ask for at least one check: Gradient = true, Lyapunov = V, " + ...
        "Dulac = G or Trapping = true.");
end
if isempty(options.Params) && ~isstruct(options.Params)
    p = m.ParameterValues;
else
    p = m.paramVector(options.Params);
end
p = p(:);

[XX, YY] = ndgrid(linspace(x0, x1, nx), linspace(y0, y1, ny));
X = [XX(:)'; YY(:)'];
K = size(X, 2);
F = phasecont.internal.rhsGaps(m, X, p);   % NaN where the engine refuses
mustBeFiniteAt(F, X, "the vector field f");

r = struct("Region", [x0 x1 y0 y1], "Grid", [nx ny], "P", p, "Tol", tol, ...
    "Samples", K, "Gradient", notRun(), "Lyapunov", lyapunovNotRun(), ...
    "Dulac", dulacNotRun(), "Trapping", trappingNotRun());

J = [];
if doGradient || dulac ~= ""
    J = jacobians(m.Id, X, p, "the Jacobian of f");
end
if doGradient
    r.Gradient = gradientCheck(J, X, tol);
end
if lyapunov ~= ""
    [V, dV] = expressionValues(m, lyapunov, "Lyapunov", "V", p, X);
    r.Lyapunov = lyapunovCheck(V, dV, F, X, tol, lyapunov);
end
if dulac ~= ""
    [G, dG] = expressionValues(m, dulac, "Dulac", "g", p, X);
    r.Dulac = dulacCheck(G, dG, J, F, X, tol, dulac);
end
if doTrapping
    r.Trapping = trappingCheck(F, X, nx, ny, tol);
end
end

% ---- options ---------------------------------------------------------------

function checkModel(m)
if m.IsMap
    why = "it is a map";
elseif m.IsDelay
    why = "it has delay terms";
elseif ~m.Autonomous
    why = "it uses t";
elseif numel(m.Variables) ~= 2
    why = sprintf("it has %d variables", numel(m.Variables));
else
    return
end
error("phasecont:orbit:badModel", ...
    "closedOrbitChecks needs an ODE with two variables that does not " + ...
    "use t and has no delay terms, and this model cannot be checked " + ...
    "because %s.", why);
end

function [x0, x1, y0, y1] = regionOption(v)
if ~(isnumeric(v) && isreal(v) && numel(v) == 4 && all(isfinite(v)))
    error("phasecont:model:badOption", ...
        "Region must be four finite real numbers [x0 x1 y0 y1].");
end
v = double(v(:)');
[x0, x1, y0, y1] = deal(v(1), v(2), v(3), v(4));
if ~(x0 < x1 && y0 < y1)
    error("phasecont:model:badOption", ...
        "Region [x0 x1 y0 y1] needs x0 < x1 and y0 < y1.");
end
end

function [nx, ny] = gridOption(v)
if ~(isnumeric(v) && isreal(v) && numel(v) == 2 && all(isfinite(v)) ...
        && all(v == round(v)) && all(v >= 2))
    error("phasecont:model:badOption", ...
        "Grid must be two whole numbers [nx ny], each at least 2.");
end
nx = double(v(1));
ny = double(v(2));
end

function tf = flagOption(v, name)
if ~((islogical(v) || isnumeric(v)) && isscalar(v) && isreal(v) ...
        && (v == 0 || v == 1))
    error("phasecont:model:badOption", "%s must be true or false.", name);
end
tf = logical(v);
end

function e = expressionOption(v, name)
if isempty(v) && isnumeric(v)
    e = "";
    return
end
if ~((isstring(v) && isscalar(v) && ~ismissing(v)) || ...
        (ischar(v) && (isrow(v) || isempty(v))))
    error("phasecont:model:badOption", ...
        "%s must be a string expression.", name);
end
e = strtrim(string(v));
if e == ""
    error("phasecont:model:badOption", "%s must not be empty.", name);
end
if contains(e, [newline sprintf("\r")])
    error("phasecont:model:badOption", ...
        "%s must be one expression on one line.", name);
end
end

% ---- evaluation ------------------------------------------------------------

function J = jacobians(id, X, p, what)
% J is 4-by-K: rows J11, J21, J12, J22 (the columns of each 2-by-2 J).
K = size(X, 2);
J = zeros(4, K);
for k = 1:K
    try
        Jk = pckern("model_jacobian", id, X(:, k), p, '');
    catch err
        if err.identifier ~= "phasecont:engine:numeric"
            rethrow(err);
        end
        nonfinite(X(:, k), what);
    end
    if ~all(isfinite(Jk), "all")
        nonfinite(X(:, k), what);
    end
    J(:, k) = Jk(:);
end
end

function [E, dE] = expressionValues(m, e, option, symbol, p, X)
% E (1-by-K) and its gradient dE (2-by-K) from an auxiliary model.
names = m.Variables;
params = struct();
for k = 1:numel(m.Parameters)
    params.(m.Parameters(k)) = p(k);
end
init = struct(names(1), 0, names(2), 0);
lines = [names(1) + "' = " + e; names(2) + "' = 0"];
try
    aux = phasecont.Model.fromEquations(lines, Params = params, Init = init, ...
        Name = option + ".ode");
catch err
    if startsWith(err.identifier, "phasecont:engine:") || ...
            startsWith(err.identifier, "phasecont:model:")
        cause = MException("phasecont:orbit:badExpression", ...
            "%s = ""%s"" is not an expression this model can evaluate: %s", ...
            option, e, err.message);
        cause = addCause(cause, err);
        throw(cause);
    end
    rethrow(err);
end
cleanup = onCleanup(@() delete(aux));
if ~aux.Autonomous || aux.IsDelay || aux.IsMap
    error("phasecont:orbit:badExpression", ...
        "%s = ""%s"" must not use t or delay terms.", option, e);
end
if ~isequal(aux.Variables(:), names(:))
    error("phasecont:orbit:badExpression", ...
        "%s = ""%s"" changed the state names of the model.", option, e);
end
pa = aux.ParameterValues;
full = phasecont.internal.rhsGaps(aux, X, pa);
E = full(1, :);
mustBeFiniteAt(E, X, sprintf("the %s function %s", option, symbol));
dJ = jacobians(aux.Id, X, pa, sprintf("the gradient of %s", symbol));
dE = dJ([1 3], :);   % row 1 of each Jacobian: (d e/dx, d e/dy)
clear cleanup
end

function mustBeFiniteAt(A, X, what)
bad = find(~all(isfinite(A), 1), 1);
if ~isempty(bad)
    nonfinite(X(:, bad), what);
end
end

function nonfinite(x, what)
error("phasecont:orbit:nonfinite", "%s is NaN or Inf at the sample (%s, %s).", ...
    what, sprintf("%.15g", x(1)), sprintf("%.15g", x(2)));
end

% ---- checks ----------------------------------------------------------------

function c = gradientCheck(J, X, tol)
q = J(3, :) - J(2, :);
s = max(sqrt(sum(J.^2, 1)));
c = extremes(notRun(), q, X, s);
if all(abs(q) <= tol * s)
    c.Verdict = "holds on samples";
    c.Reason = "J12 = J21 within the tolerance at every sample, the " + ...
        "condition for a gradient system (Strogatz, Theorem 7.2.3), " + ...
        "tested at the samples only.";
else
    c.Verdict = "fails";
    c.Reason = sprintf("J12 - J21 is %.6g at (%.6g, %.6g), so J is " + ...
        "not symmetric there and the field is not a gradient field.", ...
        worstValue(c), worstAt(c));
end
end

function c = lyapunovCheck(V, dV, F, X, tol, e)
terms = dV .* F;
q = sum(terms, 1);
s = max(sum(abs(terms), 1));
normF = sqrt(sum(F.^2, 1));
fixed = normF <= tol * max(normF);
c = extremes(lyapunovNotRun(), q, X, s);
c.Expression = e;
c.FixedPointSamples = nnz(fixed);
[c.VMin, i] = min(V);
c.VMinAt = X(:, i)';
[c.VMax, i] = max(V);
c.VMaxAt = X(:, i)';
pos = q > tol * s;
neg = q < -tol * s;
away = ~fixed;
if any(pos) && any(neg)
    c.Verdict = "fails";
    c.Sign = "";
    c.Reason = sprintf("Vdot takes both signs on the samples, from %.6g " + ...
        "at (%.6g, %.6g) to %.6g at (%.6g, %.6g).", c.Min, c.MinAt, ...
        c.Max, c.MaxAt);
elseif any(away) && all(neg(away))
    c.Verdict = "holds on samples";
    c.Sign = "negative";
    c.Reason = "Vdot < 0 at every sample that is not a fixed-point " + ...
        "sample, tested at the samples only; V was not tested for " + ...
        "positivity.";
elseif any(away) && all(pos(away))
    c.Verdict = "holds on samples";
    c.Sign = "positive";
    c.Reason = "Vdot > 0 at every sample that is not a fixed-point " + ...
        "sample, tested at the samples only; V was not tested for " + ...
        "positivity.";
elseif ~any(away)
    c.Verdict = "inconclusive";
    c.Sign = "";
    c.Reason = "f is zero within the tolerance at every sample, so no " + ...
        "sample tests the sign of Vdot.";
else
    c.Verdict = "inconclusive";
    c.Sign = "";
    c.Reason = sprintf("Vdot is zero within the tolerance at %d " + ...
        "samples where f is not.", nnz(away & ~pos & ~neg));
end
end

function c = dulacCheck(G, dG, J, F, X, tol, e)
trJ = J(1, :) + J(4, :);
terms = [G .* trJ; dG .* F];
q = sum(terms, 1);
s = max(sum(abs(terms), 1));
c = extremes(dulacNotRun(), q, X, s);
c.Expression = e;
pos = q > tol * s;
neg = q < -tol * s;
c.ZeroSamples = nnz(~pos & ~neg);
if any(pos) && any(neg)
    c.Verdict = "fails";
    c.Sign = "";
    c.Reason = sprintf("div(g f) takes both signs on the samples, from " + ...
        "%.6g at (%.6g, %.6g) to %.6g at (%.6g, %.6g).", c.Min, c.MinAt, ...
        c.Max, c.MaxAt);
elseif all(neg)
    c.Verdict = "holds on samples";
    c.Sign = "negative";
    c.Reason = "div(g f) < 0 at every sample, tested at the samples only.";
elseif all(pos)
    c.Verdict = "holds on samples";
    c.Sign = "positive";
    c.Reason = "div(g f) > 0 at every sample, tested at the samples only.";
else
    c.Verdict = "inconclusive";
    c.Sign = "";
    c.Reason = sprintf("div(g f) is zero within the tolerance at %d of " + ...
        "%d samples.", c.ZeroSamples, numel(q));
end
end

function c = trappingCheck(F, X, nx, ny, tol)
idx = reshape(1:nx*ny, nx, ny);
sides = ["left" "right" "bottom" "top"];
at = {idx(1, :), idx(nx, :), idx(:, 1)', idx(:, ny)'};
q = [-F(1, at{1}), F(1, at{2}), -F(2, at{3}), F(2, at{4})];
cols = [at{:}];
side = repelem(1:4, [ny ny nx nx]);
boundary = unique(cols);
s = max(sqrt(sum(F(:, boundary).^2, 1)));
c = trappingNotRun();
c.Samples = numel(boundary);
c.Scale = s;
[c.Min, i] = min(q);
c.MinAt = X(:, cols(i))';
[c.Max, i] = max(q);
c.MaxAt = X(:, cols(i))';
c.Worst = sides(side(i));
pos = q > tol * s;
c.TangentSamples = nnz(abs(q) <= tol * s);
if s == 0
    c.Verdict = "holds on samples";
    c.Reason = "f is zero at every boundary sample, so no value " + ...
        "points outward.";
elseif any(pos)
    c.Verdict = "fails";
    c.Reason = sprintf("f points outward on the %s side at (%.6g, %.6g), " + ...
        "where f.n = %.6g.", c.Worst, c.MaxAt, c.Max);
else
    c.Verdict = "holds on samples";
    c.Reason = "f.n <= 0 at every boundary sample, tested at the " + ...
        "samples only; the check says nothing about fixed points inside.";
end
end

function c = extremes(c, q, X, s)
c.Samples = numel(q);
c.Scale = s;
[c.Min, i] = min(q);
c.MinAt = X(:, i)';
[c.Max, i] = max(q);
c.MaxAt = X(:, i)';
end

function v = worstValue(c)
if abs(c.Min) >= abs(c.Max)
    v = c.Min;
else
    v = c.Max;
end
end

function xy = worstAt(c)
if abs(c.Min) >= abs(c.Max)
    xy = c.MinAt;
else
    xy = c.MaxAt;
end
end

% ---- results of checks that did not run -------------------------------------

function c = notRun()
c = struct("Verdict", "not run", "Reason", "The check was not asked for.", ...
    "Min", NaN, "MinAt", [NaN NaN], "Max", NaN, "MaxAt", [NaN NaN], ...
    "Samples", 0, "Scale", NaN);
end

function c = lyapunovNotRun()
c = notRun();
c.Sign = "";
c.FixedPointSamples = 0;
c.VMin = NaN;
c.VMinAt = [NaN NaN];
c.VMax = NaN;
c.VMaxAt = [NaN NaN];
c.Expression = "";
end

function c = dulacNotRun()
c = notRun();
c.Sign = "";
c.ZeroSamples = 0;
c.Expression = "";
end

function c = trappingNotRun()
c = notRun();
c.Worst = "";
c.TangentSamples = 0;
end
