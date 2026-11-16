function r = fit(m, x0, P, t, Y, options)
%FIT Fit model parameters to trajectory data (Levenberg-Marquardt).
%   R = phasecont.fit(M, X0, P, T, Y, Fit = "a", Observe = "x", T0 = 0, ...
%       Dt = 0.01, FdStep = 1e-7, FdScale = 1, Tau = 1e-3, GradTol = 1e-8, ...
%       StepTol = 1e-8, MaxIter = 200) changes the parameters named in Fit
%   until the model, started from X0 at time T0, matches the data Y at the
%   times T. It returns a phasecont.FitResult.
%
%   X0 is one start state, n-by-1; it is not fitted. P is an np-by-1
%   vector, a scalar struct that changes named parameters, or [] for the
%   values in the model. It holds the start values of the fit and the fixed
%   values of the parameters that are not in Fit. T is a vector of S times
%   that increase and are all greater than T0. Y is S-by-K, and column k
%   holds the data of the variable Observe(k).
%
%   The fit minimizes F = 1/2 * sum of (model - data).^2 with Algorithm 3.16
%   of Madsen, Nielsen and Tingleff, Methods for Non-Linear Least Squares
%   Problems, 2nd ed., 2004. The Jacobian is a forward difference. Each
%   iteration is one phasecont.ensemble call on the CPU in double precision.
%   The residuals are not weighted and there are no bounds.
%
%   Nothing here has a default. These options are required:
%     Fit        string array of parameter names to fit
%     Observe    string array of variable names the data belongs to
%     T0         time of the start state
%     Dt         step of Runge-Kutta 4, or
%     Rtol, Atol tolerances of Dormand-Prince 5(4), as in phasecont.ensemble
%     FdStep     relative step of the forward difference: column j of the
%                Jacobian uses FdStep*max(abs(q_j), s_j). Nielsen et al. use
%                1e-7 in their examples.
%     FdScale    s_j, the size of parameter j below which the step stops
%                shrinking, as TypicalX does in the MathWorks optimizers. A
%                positive finite scalar, or one value per fitted parameter in
%                the order of Fit. Choose a value near the smallest size of
%                the parameter that the model can tell apart.
%     Tau       start damping, mu = Tau*max(diag(J'*J)). They give 1e-6 for
%                a good start and 1e-3 or 1 otherwise.
%     GradTol    stop when max(abs(J'*f)) <= GradTol
%     StepTol    stop when norm(h) <= StepTol*(norm(q) + StepTol)
%     MaxIter    most trial steps
%   Reaching MaxIter is not an error; R.Converged is false then. FdStep and
%   FdScale must be large enough to move every fitted parameter: a step that
%   leaves q_j unchanged in double precision raises phasecont:fit:badOption.
%
%   One option is optional:
%     MaxSteps   most steps of one member, passed to phasecont.ensemble,
%                whose limit is 100000 when MaxSteps is not given or is
%                empty, as for Dt, Rtol and Atol
%
%   A trial point that does not finish is a rejected step. A start point
%   that does not finish raises phasecont:fit:startFailed, and so does a
%   start that needs more steps than MaxSteps allows. A
%   perturbed point that does not finish, at the start or at an accepted
%   trial point, raises phasecont:fit:jacobianFailed. Both messages name
%   MaxSteps when the member stopped at the step limit. A parameter whose
%   Jacobian column is exactly zero raises phasecont:fit:unidentifiable; this
%   is reliable with Dt and is not guaranteed with Rtol and Atol. The warning
%   phasecont:ensemble:memberStopped is off while the fit runs. Ctrl+C
%   cancels the run. A map (M.IsMap) is refused with phasecont:fit:mapModel.
%
%   See also phasecont.ensemble, phasecont.FitResult, phasecont.simulate.
arguments
    m (1,1) phasecont.Model
    x0
    P
    t
    Y
    options.Fit = []
    options.Observe = []
    options.T0 = []
    options.Dt = []
    options.Rtol = []
    options.Atol = []
    options.FdStep = []
    options.FdScale = []
    options.Tau = []
    options.GradTol = []
    options.StepTol = []
    options.MaxIter = []
    options.MaxSteps = []
end

if m.IsMap
    error("phasecont:fit:mapModel", ...
        "fit fits ODE models, and this model is a map (x(t+1)=... or @ meth=discrete). Fitting a map is not supported.");
end

% A missing required option is reported first, then a bad value. Dt, Rtol and
% Atol are checked by ensemble on the first call.
required = ["Fit" "Observe" "T0" "FdStep" "FdScale" "Tau" "GradTol" "StepTol" ...
    "MaxIter"];
ids = ["missingFit" "missingObserve" "missingTime" "missingFdStep" ...
    "missingFdScale" "missingTau" "missingGradTol" "missingStepTol" ...
    "missingMaxIter"];
for k = 1:numel(required)
    if isempty(options.(required(k)))
        error("phasecont:fit:" + ids(k), ...
            "%s is required. It has no default.", required(k));
    end
end
t0 = options.T0;
if ~(isnumeric(t0) && isreal(t0) && isscalar(t0) && isfinite(t0))
    error("phasecont:fit:badOption", "T0 must be a finite real number.");
end
for name = ["FdStep" "Tau" "GradTol" "StepTol"]
    v = options.(name);
    if ~(isnumeric(v) && isreal(v) && isscalar(v) && isfinite(v) && v > 0)
        error("phasecont:fit:badOption", ...
            "%s must be a positive finite number.", name);
    end
end
maxIter = options.MaxIter;
if ~(isnumeric(maxIter) && isreal(maxIter) && isscalar(maxIter) ...
        && isfinite(maxIter) && maxIter >= 1 && maxIter == fix(maxIter))
    error("phasecont:fit:badOption", "MaxIter must be a positive integer.");
end
t0 = double(t0);
fdStep = double(options.FdStep);
tau = double(options.Tau);
gradTol = double(options.GradTol);
stepTol = double(options.StepTol);
maxIter = double(maxIter);

n = numel(m.Variables);
fitNames = nameList(options.Fit, "badFit", "Fit");
obsNames = nameList(options.Observe, "badObserve", "Observe");
[found, fitIdx] = ismember(fitNames, m.Parameters);
if ~all(found)
    error("phasecont:model:unknownParameter", ...
        "The model has no parameter named ""%s"".", fitNames(find(~found, 1)));
end
[found, obsIdx] = ismember(obsNames, m.Variables);
if ~all(found)
    error("phasecont:fit:unknownVariable", ...
        "The model has no variable named ""%s"".", obsNames(find(~found, 1)));
end
fitIdx = fitIdx(:);
obsIdx = obsIdx(:);
nf = numel(fitIdx);
nObs = numel(obsIdx);
fdScale = options.FdScale;
if ~(isnumeric(fdScale) && isreal(fdScale) && isvector(fdScale) ...
        && (isscalar(fdScale) || numel(fdScale) == nf) ...
        && all(isfinite(fdScale)) && all(fdScale > 0))
    error("phasecont:fit:badOption", ...
        "FdScale must be a positive finite number or a vector of %d of them, one per name in Fit.", ...
        nf);
end
fdScale = double(fdScale(:)) .* ones(nf, 1);

if ~(isnumeric(x0) && isreal(x0) && isvector(x0) && numel(x0) == n && all(isfinite(x0)))
    error("phasecont:model:badStart", ...
        "X0 must be one finite real start state with %d values.", n);
end
x0 = double(x0(:));
if isnumeric(P) && isempty(P)
    p0 = m.ParameterValues(:);
else
    p0 = m.paramVector(P);
end

if ~(isnumeric(t) && isreal(t) && isvector(t) && all(isfinite(t)))
    error("phasecont:fit:badTime", "T must be a vector of finite real times.");
end
t = double(t(:));
if ~issorted(t, "strictascend") || t(1) <= t0
    error("phasecont:fit:badTime", ...
        "The data times must increase and all be greater than T0 = %.6g.", t0);
end
nTimes = numel(t);
if ~(isnumeric(Y) && isreal(Y) && ismatrix(Y) ...
        && isequal(size(Y), [nTimes nObs]) && all(isfinite(Y), "all"))
    error("phasecont:fit:badData", ...
        "Y must be a finite real %d-by-%d matrix: one row per time, one column per observed variable.", ...
        nTimes, nObs);
end
Y = double(Y);

ctx = struct("m", m, "x0", x0, "p0", p0, "fitIdx", fitIdx, "obsIdx", obsIdx, ...
    "fdStep", fdStep, "fdScale", fdScale, "tspan", [t0; t].', "nTimes", nTimes, "Y", Y, ...
    "method", {methodArgs(options)}, "fitNames", fitNames);

% The warning for members that stop is expected here: a trial point may
% leave the region where the model has a solution. Restore it on any exit.
warnId = "phasecont:ensemble:memberStopped";
saved = warning("query", warnId);
warning("off", warnId);
restore = onCleanup(@() warning(saved)); %#ok<NASGU>

q = p0(fitIdx);
ev = evaluate(ctx, q);
if ~ev.ok0
    error("phasecont:fit:startFailed", ...
        "The model did not reach the last data time from the start values (status ""%s"").%s", ...
        ev.status(1), stepsHint(ev.status(1)));
end
requireJacobian(ctx, ev);
f = ev.resid;
model = ev.model;
J = ev.J;
cost = 0.5 * (f' * f);
g = J' * f;
mu = tau * max(sum(J .^ 2, 1));
nu = 2;
gradNorm = norm(g, Inf);
done = gradNorm <= gradTol;
stop = "";
if done
    stop = "gradient";
end

cap = min(maxIter, 64);   % History rows grow by doubling, up to MaxIter
hist = nan(cap, 7);       % Iter Cost GradNorm StepNorm Mu Rho Accepted
iter = 0;
while ~done && iter < maxIter
    h = [J; sqrt(mu) * eye(nf)] \ [-f; zeros(nf, 1)];
    stepNorm = norm(h);
    if stepNorm <= stepTol * (norm(q) + stepTol)
        done = true;
        stop = "step";
        break
    end
    iter = iter + 1;
    qNew = q + h;
    trial = evaluate(ctx, qNew);
    rho = NaN;
    accept = false;
    if trial.ok0
        fNew = trial.resid;
        costNew = 0.5 * (fNew' * fNew);
        rho = (cost - costNew) / (0.5 * (h' * (mu * h - g)));
        accept = rho > 0;
    end
    muUsed = mu;
    if accept
        requireJacobian(ctx, trial);
        q = qNew;
        f = fNew;
        model = trial.model;
        J = trial.J;
        cost = costNew;
        g = J' * f;
        gradNorm = norm(g, Inf);
        mu = mu * max(1/3, 1 - (2 * rho - 1)^3);
        nu = 2;
        if gradNorm <= gradTol
            done = true;
            stop = "gradient";
        end
    else
        mu = mu * nu;
        nu = 2 * nu;
    end
    if iter > cap
        cap = min(maxIter, 2 * cap);
        hist(cap, 1) = NaN;
    end
    hist(iter, :) = [iter cost gradNorm stepNorm muUsed rho accept];
end
if ~done
    stop = "maxIter";
end

hist = hist(1:iter, :);
history = table(hist(:, 1), hist(:, 2), hist(:, 3), hist(:, 4), hist(:, 5), ...
    hist(:, 6), logical(hist(:, 7)), VariableNames = ...
    ["Iter" "Cost" "GradNorm" "StepNorm" "Mu" "Rho" "Accepted"]);

pAll = p0;
pAll(fitIdx) = q;
params = struct();
for k = 1:nf
    params.(fitNames(k)) = q(k);
end
r = phasecont.FitResult(struct("Fit", fitNames, "Observe", obsNames, ...
    "Times", t, "Estimate", q, "P", pAll, "Parameters", params, ...
    "Fitted", model, "Residual", reshape(f, nTimes, nObs), ...
    "Cost", cost, "GradNorm", gradNorm, "Jacobian", J, ...
    "Iterations", iter, "Converged", stop ~= "maxIter", "Stop", stop, ...
    "History", history));
end

function names = nameList(v, id, label)
% A nonempty row of distinct, nonempty names.
if ischar(v) || iscellstr(v) %#ok<ISCLSTR>
    v = string(v);
end
if ~(isstring(v) && ~isempty(v) && isvector(v) ...
        && ~any(ismissing(v)) && all(strlength(v) > 0))
    error("phasecont:fit:" + id, ...
        "%s must be a string array of names, for example [""a"" ""b""].", label);
end
names = v(:).';
if numel(unique(names)) ~= numel(names)
    error("phasecont:fit:" + id, "%s names the same entry twice.", label);
end
end

function c = methodArgs(options)
% The step method options of ensemble, as given. Their rules are checked
% by ensemble.
c = {};
for name = ["Dt" "Rtol" "Atol" "MaxSteps"]
    if ~isempty(options.(name))
        c = [c, {char(name), options.(name)}]; %#ok<AGROW>
    end
end
end

function ev = evaluate(ctx, q)
% One ensemble call: member 1 is the point q, member j+1 is q with fitted
% parameter j moved by a forward-difference step. Returns the residual at q
% and, when every member finished, the Jacobian.
nf = numel(q);
eta = ctx.fdStep * max(abs(q), ctx.fdScale);
pq = ctx.p0;
pq(ctx.fitIdx) = q;
pm = repmat(pq, 1, nf + 1);
for j = 1:nf
    moved = q(j) + eta(j);
    eta(j) = moved - q(j);            % the step that was actually taken
    if ~(isfinite(eta(j)) && eta(j) ~= 0)
        error("phasecont:fit:badOption", ...
            "FdStep = %g does not move parameter ""%s"" (value %g). Use a larger FdStep or FdScale.", ...
            ctx.fdStep, ctx.fitNames(j), q(j));
    end
    pm(ctx.fitIdx(j), j + 1) = moved;
end
res = phasecont.ensemble(ctx.m, repmat(ctx.x0, 1, nf + 1), pm, ctx.tspan, ...
    "Backend", "cpu", "Precision", "double", ctx.method{:});
ev = struct("status", res.Status, "ok0", res.Status(1) == "ok", ...
    "okJ", res.Status(2:end) == "ok", "resid", [], "model", [], "J", []);
if ~ev.ok0
    return
end
nTimes = ctx.nTimes;
nObs = numel(ctx.obsIdx);
model0 = observed(res, 1, ctx.obsIdx, nTimes, nObs);
ev.model = model0;
ev.resid = reshape(model0 - ctx.Y, [], 1);
J = zeros(numel(model0), nf);
for j = 1:nf
    if ev.okJ(j)
        mj = observed(res, j + 1, ctx.obsIdx, nTimes, nObs);
        J(:, j) = (mj(:) - model0(:)) / eta(j);
    end
end
ev.J = J;
end

function v = observed(res, member, obsIdx, nTimes, nObs)
% States of the observed variables of one member, S-by-K.
v = reshape(res.States(obsIdx, :, member), nObs, nTimes).';
end

function requireJacobian(ctx, ev)
% A Jacobian needs every perturbed point to have finished, and every
% column to depend on its parameter.
bad = find(~ev.okJ, 1);
if ~isempty(bad)
    error("phasecont:fit:jacobianFailed", ...
        "The perturbed point of parameter ""%s"" did not finish (status ""%s""), so the Jacobian is not defined here.%s", ...
        ctx.fitNames(bad), ev.status(bad + 1), stepsHint(ev.status(bad + 1)));
end
flat = find(~any(ev.J, 1), 1);
if ~isempty(flat)
    error("phasecont:fit:unidentifiable", ...
        "The observed data does not depend on parameter ""%s"": its Jacobian column is exactly zero.", ...
        ctx.fitNames(flat));
end
end

function hint = stepsHint(status)
% Added to the message of a member that stopped at the step limit.
hint = "";
if status == "maxSteps"
    hint = " Raise MaxSteps if the model needs more steps.";
end
end
