function [pts, raw] = equilibria(m, X0, options)
%EQUILIBRIA Find the equilibria of a model from many starting points.
%   PTS = phasecont.equilibria(M, X0) runs the Newton solve of
%   phasecont.equilibrium from every column of X0, an n-by-K matrix of
%   starts, in one engine call that spreads the starts over the CPU threads.
%   The converged starts are merged into groups, and PTS is a 1-by-G array
%   of phasecont.Point, one per group, in the order the groups were made.
%   Each point is the one phasecont.equilibrium returns from the first
%   start of its group. When no start converges, PTS is empty and no error
%   is raised.
%
%   Merging takes the converged starts in start order. Start j joins the
%   first group whose point v has max(abs(u - v)) <= MergeTol*max(1,
%   max(abs(v))), where u is the point start j reached; otherwise it makes
%   a new group.
%
%   [PTS, RAW] = phasecont.equilibria(...) also returns a struct with one
%   column or entry per start:
%     U            n-by-K points (NaN for a start that failed)
%     Eigenvalues  n-by-K eigenvalues, or multipliers on a map
%     Residual     1-by-K max(abs(G(u))), where G is the function Newton
%                  solved; for a failed start the last Newton residual
%                  when Newton ran out of iterations, else NaN
%     Converged    1-by-K logical
%     Group        1-by-K index into PTS, 0 for a start that failed
%
%   Name-value options
%     MergeTol  relative distance under which two points are the same
%               equilibrium, a positive finite number (default 1e-6)
%     Params, MaxIter, Tol, Iterate  as in phasecont.equilibrium
%
%   See also phasecont.equilibrium.
arguments
    m (1,1) phasecont.Model
    X0 (:,:) {mustBeNumeric, mustBeReal}
    options.MergeTol = 1e-6
    options.Params = []
    options.MaxIter (1,1) {mustBeNumeric, mustBeInteger, mustBePositive} = 50
    options.Tol (1,1) {mustBeNumeric, mustBeReal} = 1e-10
    options.Iterate = []
end
mt = options.MergeTol;
if ~(isnumeric(mt) && isreal(mt) && isscalar(mt) && isfinite(mt) && mt > 0)
    error("phasecont:model:badOption", "MergeTol must be a positive finite number.");
end
mt = double(mt);
if ~(isfinite(options.Tol) && options.Tol > 0)
    error("phasecont:model:badOption", "Tol must be a positive finite number.");
end
n = numel(m.Variables);
if size(X0, 1) ~= n
    error("phasecont:model:badOption", ...
        "X0 must have %d rows, one per variable, but it has %d.", n, size(X0, 1));
end
if isempty(options.Params) && ~isstruct(options.Params)
    p = m.ParameterValues;
else
    p = m.paramVector(options.Params);
end
k = phasecont.internal.iterateOption(m, options.Iterate);
opts = "{""max_newton"":" + phasecont.internal.num2json(options.MaxIter) + ...
    ",""tol_x"":" + phasecont.internal.num2json(options.Tol) + ...
    ",""tol_f"":" + phasecont.internal.num2json(options.Tol);
if isempty(k)
    k = 1;
else
    opts = opts + ",""iterate"":" + phasecont.internal.num2json(k);
end
opts = opts + "}";
X0 = double(X0);
[U, E, res, status] = pckern("model_equilibrium_many", m.Id, X0, p, char(opts));
K = size(X0, 2);
conv = status == 0;
group = zeros(1, K);
reps = zeros(1, 0);
for j = find(conv)
    u = U(:, j);
    for g = 1:numel(reps)
        v = U(:, reps(g));
        if max(abs(u - v)) <= mt * max(1, max(abs(v)))
            group(j) = g;
            break
        end
    end
    if group(j) == 0
        reps(end + 1) = j; %#ok<AGROW>
        group(j) = numel(reps);
    end
end
if isempty(reps)
    pts = phasecont.Point.empty;
else
    c = cell(1, numel(reps));
    for g = 1:numel(reps)
        j = reps(g);
        u = U(:, j);
        ev = E(:, j);
        record = phasecont.internal.RunRecord(struct( ...
            "Kind", "equilibrium", "Problem", opts, "U0", X0(:, j), "P", p(:), ...
            "ModelSha", m.SourceSha256, "N", n, ...
            "Result", struct("U", u, "Eigenvalues", ev)));
        c{g} = phasecont.Point(U = u, P = p, Eigenvalues = ev, Label = "EP", ...
            Model = m, Record = record, Iterate = k);
    end
    pts = [c{:}];
end
if nargout > 1
    raw = struct("U", U, "Eigenvalues", E, "Residual", res, ...
        "Converged", conv, "Group", group);
end
end
