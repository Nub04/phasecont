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
%   Two options are optional:
%     MaxSteps   most steps of one member, passed to phasecont.ensemble,
%                whose limit is 100000 when MaxSteps is not given or is
%                empty, as for Dt, Rtol and Atol
%     Starts     |Fit|-by-K matrix of start values, one column per start,
%                rows in the order of Fit. They replace the fitted values
%                in P, which still gives the other parameters. The K fits
%                run in lockstep, one ensemble call per iteration for all
%                starts still running, and each start gives the same bits
%                as a fit from that start alone. R is the start with the
%                lowest Cost (a tie keeps the lower index); R.Starts holds
%                every start and R.BestStart the index of R. A start that
%                would raise startFailed, jacobianFailed, unidentifiable or
%                the badOption of FdStep has Stop "failed" and its error in
%                Error, and the fit warns phasecont:fit:someStartsFailed.
%                When every start fails, the error of the first is raised
%                with "In start k:" before its message. An empty Starts of
%                any size is the same as none. A Starts that is not a
%                finite real |Fit|-by-K matrix raises phasecont:fit:badStarts.
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
    options.Starts = []
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
starts = options.Starts;
given = ~(isnumeric(starts) && isempty(starts));
if given
    if ~(isnumeric(starts) && isreal(starts) && ismatrix(starts) ...
            && size(starts, 1) == nf && size(starts, 2) >= 1 ...
            && all(isfinite(starts), "all"))
        error("phasecont:fit:badStarts", ...
            "Starts must be a finite real %d-by-K matrix: one row per name in Fit, one column per start.", ...
            nf);
    end
    Q0 = double(starts);
else
    Q0 = p0(fitIdx);
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

cfg = struct("tau", tau, "gradTol", gradTol, "stepTol", stepTol, ...
    "maxIter", maxIter, "nf", nf);
K = size(Q0, 2);
S = cell(1, K);
for k = 1:K
    S{k} = newStart(Q0(:, k), maxIter);
end
% Each pass makes one ensemble call for the points of every start that is
% still running: its start point or its trial point. A fit without Starts
% is the case K = 1.
while true
    live = find(cellfun(@(s) s.phase ~= "end", S));
    if isempty(live)
        break
    end
    blocks = cell(1, numel(live));
    etas = cell(1, numel(live));
    keep = false(1, numel(live));
    for i = 1:numel(live)
        k = live(i);
        try
            [blocks{i}, etas{i}] = perturb(ctx, S{k}.point);
            keep(i) = true;
        catch err
            S{k} = failStart(S{k}, err);
        end
    end
    live = live(keep);
    if isempty(live)
        continue
    end
    blocks = blocks(keep);
    etas = etas(keep);
    pm = [blocks{:}];
    res = phasecont.ensemble(ctx.m, repmat(ctx.x0, 1, size(pm, 2)), pm, ...
        ctx.tspan, "Backend", "cpu", "Precision", "double", ctx.method{:});
    for i = 1:numel(live)
        k = live(i);
        ev = unpack(ctx, res, (i - 1) * (nf + 1), etas{i});
        try
            S{k} = advance(S{k}, ev, ctx, cfg);
        catch err
            S{k} = failStart(S{k}, err);
        end
    end
end

results = cell(1, K);
for k = 1:K
    results{k} = resultFields(S{k}, ctx, obsNames, t, nObs);
end
failed = cellfun(@(s) s.stop == "failed", S);
if ~given
    if failed
        rethrow(S{1}.err);
    end
    v = results{1};
    v.Starts = phasecont.FitResult.empty(1, 0);
    r = phasecont.FitResult(v);
    return
end
if all(failed)
    err = S{1}.err;
    throw(MException(err.identifier, "%s", "In start 1: " + string(err.message)));
end
each = phasecont.FitResult.empty(1, 0);
for k = 1:K
    each(k) = phasecont.FitResult(results{k});
end
costs = cellfun(@(v) v.Cost, results);
costs(failed) = NaN;
[~, best] = min(costs);        % min skips NaN and keeps the first of a tie
v = results{best};
v.Starts = each;
v.BestStart = best;
r = phasecont.FitResult(v);
if any(failed)
    warning("phasecont:fit:someStartsFailed", ...
        "%d of %d starts failed. Their entries in R.Starts have Stop ""failed"" and hold the error in Error.", ...
        nnz(failed), K);
end
end

function s = newStart(q, maxIter)
% The state of one start before its first ensemble call.
cap = min(maxIter, 64);   % History rows grow by doubling, up to MaxIter
s = struct("phase", "start", "point", q, "q", q, "f", [], "model", [], ...
    "J", [], "cost", NaN, "g", [], "mu", NaN, "nu", 2, "gradNorm", NaN, ...
    "done", false, "stop", "", "iter", 0, "rows", 0, "cap", cap, ...
    "hist", nan(cap, 7), "h", [], "stepNorm", NaN, "err", []);
end

function s = advance(s, ev, ctx, cfg)
% One step of Algorithm 3.16 for one start, given the ensemble result of
% its point. Errors are those of a fit from this start alone.
if s.phase == "start"
    if ~ev.ok0
        error("phasecont:fit:startFailed", ...
            "The model did not reach the last data time from the start values (status ""%s"").%s", ...
            ev.status(1), stepsHint(ev.status(1)));
    end
    requireJacobian(ctx, ev);
    s.f = ev.resid;
    s.model = ev.model;
    s.J = ev.J;
    s.cost = 0.5 * (s.f' * s.f);
    s.g = s.J' * s.f;
    s.mu = cfg.tau * max(sum(s.J .^ 2, 1));
    s.nu = 2;
    s.gradNorm = norm(s.g, Inf);
    s.done = s.gradNorm <= cfg.gradTol;
    if s.done
        s.stop = "gradient";
    end
else
    h = s.h;
    rho = NaN;
    accept = false;
    if ev.ok0
        fNew = ev.resid;
        costNew = 0.5 * (fNew' * fNew);
        rho = (s.cost - costNew) / (0.5 * (h' * (s.mu * h - s.g)));
        accept = rho > 0;
    end
    muUsed = s.mu;
    if accept
        requireJacobian(ctx, ev);
        s.q = s.point;
        s.f = fNew;
        s.model = ev.model;
        s.J = ev.J;
        s.cost = costNew;
        s.g = s.J' * s.f;
        s.gradNorm = norm(s.g, Inf);
        s.mu = s.mu * max(1/3, 1 - (2 * rho - 1)^3);
        s.nu = 2;
        if s.gradNorm <= cfg.gradTol
            s.done = true;
            s.stop = "gradient";
        end
    else
        s.mu = s.mu * s.nu;
        s.nu = 2 * s.nu;
    end
    if s.iter > s.cap
        s.cap = min(cfg.maxIter, 2 * s.cap);
        s.hist(s.cap, 1) = NaN;
    end
    s.hist(s.iter, :) = [s.iter s.cost s.gradNorm s.stepNorm muUsed rho accept];
    s.rows = s.iter;
end
s = nextStep(s, cfg);
end

function s = nextStep(s, cfg)
% The next trial point, or the end of this start.
if ~s.done && s.iter < cfg.maxIter
    h = [s.J; sqrt(s.mu) * eye(cfg.nf)] \ [-s.f; zeros(cfg.nf, 1)];
    stepNorm = norm(h);
    if stepNorm <= cfg.stepTol * (norm(s.q) + cfg.stepTol)
        s.done = true;
        s.stop = "step";
        s.phase = "end";
        return
    end
    s.iter = s.iter + 1;
    s.h = h;
    s.stepNorm = stepNorm;
    s.point = s.q + h;
    s.phase = "trial";
    return
end
if ~s.done
    s.stop = "maxIter";
end
s.phase = "end";
end

function s = failStart(s, err)
% Record an error of this start. Only the errors that a fit raises from a
% point are recorded; any other error ends the call.
if ~ismember(string(err.identifier), ["phasecont:fit:startFailed" ...
        "phasecont:fit:jacobianFailed" "phasecont:fit:unidentifiable" ...
        "phasecont:fit:badOption"])
    rethrow(err);
end
s.err = err;
s.stop = "failed";
s.phase = "end";
end

function v = resultFields(s, ctx, obsNames, t, nObs)
% The FitResult fields of one start.
nf = numel(s.q);
hist = s.hist(1:s.rows, :);
history = table(hist(:, 1), hist(:, 2), hist(:, 3), hist(:, 4), hist(:, 5), ...
    hist(:, 6), logical(hist(:, 7)), VariableNames = ...
    ["Iter" "Cost" "GradNorm" "StepNorm" "Mu" "Rho" "Accepted"]);
pAll = ctx.p0;
pAll(ctx.fitIdx) = s.q;
params = struct();
for k = 1:nf
    params.(ctx.fitNames(k)) = s.q(k);
end
residual = zeros(0, 0);
model = zeros(0, 0);
J = zeros(0, 0);
if ~isempty(s.f)
    residual = reshape(s.f, ctx.nTimes, nObs);
    model = s.model;
    J = s.J;
end
err = MException.empty;
if s.stop == "failed"
    err = s.err;
end
v = struct("Fit", ctx.fitNames, "Observe", obsNames, ...
    "Times", t, "Estimate", s.q, "P", pAll, "Parameters", params, ...
    "Fitted", model, "Residual", residual, ...
    "Cost", s.cost, "GradNorm", s.gradNorm, "Jacobian", J, ...
    "Iterations", s.rows, "Converged", s.stop ~= "maxIter" && s.stop ~= "failed", ...
    "Stop", s.stop, "History", history, "BestStart", [], "Error", err);
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

function [pm, eta] = perturb(ctx, q)
% The parameter columns of one point: column 1 is the point q, column j+1
% is q with fitted parameter j moved by a forward-difference step. eta holds
% the steps that were actually taken.
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
end

function ev = unpack(ctx, res, offset, eta)
% The residual at one point and, when every member finished, the Jacobian.
% The members of the point are offset + 1 to offset + nf + 1 of res.
nf = numel(eta);
status = res.Status(offset + (1:nf + 1));
ev = struct("status", status, "ok0", status(1) == "ok", ...
    "okJ", status(2:end) == "ok", "resid", [], "model", [], "J", []);
if ~ev.ok0
    return
end
nTimes = ctx.nTimes;
nObs = numel(ctx.obsIdx);
model0 = observed(res, offset + 1, ctx.obsIdx, nTimes, nObs);
ev.model = model0;
ev.resid = reshape(model0 - ctx.Y, [], 1);
J = zeros(numel(model0), nf);
for j = 1:nf
    if ev.okJ(j)
        mj = observed(res, offset + j + 1, ctx.obsIdx, nTimes, nObs);
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
