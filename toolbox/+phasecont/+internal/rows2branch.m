function br = rows2branch(m, free, p, n, points, specials, branchStatus, kind, grid)
%ROWS2BRANCH Build a phasecont.Branch from engine rows. Internal.
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST)
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST, KIND)
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST, "LC", [NTST NCOL])
%
%   M         the phasecont.Model (empty is allowed in tests)
%   FREE      name of the free parameter, or its 1-based index; a curve
%             takes two, [a1 a2]
%   P         parameter vector the run started from
%   N         number of states
%   POINTS    stride-by-N matrix from run_copy, one column per point
%   SPECIALS  stride-by-K matrix of the special points
%   ST        the status struct of this branch (stop, warnings, specials)
%   KIND      "EP" (default), "LP", "H" or "LC"
%   GRID      [NTST NCOL] of an LC run: mesh intervals and collocation points
%
% An equilibrium column has the layout [alpha, u(1..n), eig_re(1..n),
% eig_im(1..n), psi_LP, psi_BP, psi_H, h, newton_iters].
% A curve column is [a2, x(1..n), a1, (kappa for H,) eig_re(1..n),
% eig_im(1..n), codim2(1..k), h, newton_iters], where k is the number of
% codim-2 labels of the kind.
% A cycle column is [alpha, orbit(1..n(NTST*NCOL+1)), T, mesh(1..NTST+1),
% mult_re(1..n), mult_im(1..n), psi_LPC, psi_PD, psi_NS, h, newton_iters],
% with the orbit at the base points, point-major. All have NaN for missing
% values. The function makes no engine calls.
if nargin < 8
    kind = "EP";
end
kind = string(kind);
nmesh = 0;
switch kind
    case "EP"
        block = n;
        labels = strings(1, 0);
        extra = 3;
    case "LP"
        block = n + 1;
        labels = ["CP" "BT" "ZH"];
        extra = numel(labels);
    case "H"
        block = n + 2;
        labels = ["GH" "BT" "ZH" "HH"];
        extra = numel(labels);
    case "LC"
        if nargin < 9 || ~isnumeric(grid) || numel(grid) ~= 2 ...
                || any(grid < 1 | grid ~= round(grid))
            error("phasecont:model:badOption", ...
                "An LC branch needs [NTST NCOL], two positive integers.");
        end
        ntst = double(grid(1));
        ncol = double(grid(2));
        block = n*(ntst*ncol + 1) + 1;
        nmesh = ntst + 1;
        labels = strings(1, 0);
        extra = 3;
    otherwise
        error("phasecont:model:badOption", ...
            "The kind must be ""EP"", ""LP"", ""H"" or ""LC"".");
end
isCurve = kind == "LP" || kind == "H";

[name, idx] = freeParameters(m, free, 1 + isCurve);
p = double(p(:));
stride = 1 + block + nmesh + 2*n + extra + 2;
if size(points, 1) ~= stride || size(specials, 1) ~= stride
    error("phasecont:model:badRows", ...
        "Expected rows of %d values, got %d and %d.", stride, ...
        size(points, 1), size(specials, 1));
end

[alpha, s, mesh, ev, tail, h, iters] = unpack(points, n, block, nmesh, extra);
npts = size(alpha, 1);
tests = tail;
codim2 = zeros(npts, 0);
switch kind
    case "EP"
        u = s;
        stable = all(real(ev) < 0, 2);
    case "LC"
        u = s(:, 1:end-1);
        period = s(:, end);
        stable = cycleStable(ev);
    otherwise
        u = s(:, 1:n);
        alpha = [s(:, n+1) alpha];
        tests = zeros(npts, 0);
        codim2 = tail;
        stable = false(npts, 1);
end
stop = "";
if isfield(branchStatus, "stop") && ~isempty(branchStatus.stop)
    stop = string(branchStatus.stop);
end
warn = strings(0, 1);
if isfield(branchStatus, "warnings") && ~isempty(branchStatus.warnings)
    warn = string(branchStatus.warnings);
    warn = warn(:);
end

info = specialList(branchStatus);
pts = phasecont.Point.empty(0, 1);
if numel(info) ~= size(specials, 2)
    error("phasecont:model:badRows", ...
        "The status lists %d special points but %d rows were copied.", ...
        numel(info), size(specials, 2));
end
for k = 1:numel(info)
    [a2, sk, mk, ek] = unpack(specials(:, k), n, block, nmesh, extra);
    pk = p;
    if isCurve
        pk(idx) = [sk(n+1), a2];
    else
        pk(idx) = a2;
    end
    if kind == "LC"
        uk = sk(1:end-1);
    else
        uk = sk(1:n);
    end
    data = struct( ...
        "coefficient", numberField(info{k}, "coefficient"), ...
        "coefficient2", numberField(info{k}, "coefficient2"), ...
        "omega", numberField(info{k}, "omega"), ...
        "normalization", textField(info{k}, "normalization"), ...
        "refined", flagField(info{k}, "refined"), ...
        "values", structField(info{k}, "coefficients"));
    if kind == "LC"
        data.period = sk(end);
        data.mesh = mk;
        data.ntst = ntst;
        data.ncol = ncol;
    end
    pts(end+1, 1) = phasecont.Point(U = uk(:), P = pk, ...
        Eigenvalues = ek(:), Label = string(info{k}.label), ...
        Index = double(info{k}.index) + 1, Data = data, Model = m); %#ok<AGROW>
end

values = struct( ...
    "Type", kind, "Free", name, "FreeIndex", idx, "P", p, "Alpha", alpha, ...
    "U", u, "Eigenvalues", ev, "Stable", stable, "Tests", tests, ...
    "Codim2", codim2, "Codim2Labels", labels, ...
    "StepSize", h, "NewtonIters", iters, "Specials", pts, ...
    "Stop", stop, "Warnings", warn, "Model", m);
if kind == "LC"
    values.Period = period;
    values.Mesh = mesh;
    values.NTST = ntst;
    values.NCOL = ncol;
end
br = phasecont.Branch(values);
end

function [name, idx] = freeParameters(m, free, count)
% Names and 1-based indices of the free parameters, in the order given.
if numel(free) ~= count
    error("phasecont:model:badOption", ...
        "Expected %d free parameter(s), got %d.", count, numel(free));
end
if isnumeric(free)
    idx = double(free(:)).';
    if isempty(m)
        name = "p" + idx;
    else
        name = reshape(m.Parameters(idx), 1, []);
    end
else
    name = reshape(string(free), 1, []);
    idx = zeros(1, count);
    for i = 1:count
        k = find(m.Parameters == name(i), 1);
        if isempty(k)
            if isempty(m)
                continue
            end
            error("phasecont:model:unknownParameter", ...
                "The model has no parameter named ""%s"".", name(i));
        end
        idx(i) = k;
    end
end
end

function [alpha, s, mesh, ev, tail, h, iters] = unpack(rows, n, block, nmesh, extra)
% Split stride-by-N rows into N-by-... arrays: the parameter in row 1, the
% state block S (u; x and a1 on a curve; the orbit and T of a cycle), the
% mesh of a cycle, the eigenvalues or multipliers, the test values, h and
% the Newton iterations.
alpha = rows(1, :).';
s = rows(2:1+block, :).';
mesh = rows(2+block:1+block+nmesh, :).';
e = 1 + block + nmesh;
ev = complex(rows(e+1:e+n, :).', rows(e+n+1:e+2*n, :).');
t = e + 2*n;
tail = rows(t+1:t+extra, :).';
h = rows(t+extra+1, :).';
iters = rows(t+extra+2, :).';
if isempty(alpha)
    s = zeros(0, block);
    mesh = zeros(0, nmesh);
    ev = complex(zeros(0, n));
    tail = zeros(0, extra);
end
end

function stable = cycleStable(mu)
% A cycle is stable when every multiplier except the one closest to 1 lies
% inside the unit circle. A row with a NaN multiplier is not stable.
stable = false(size(mu, 1), 1);
for i = 1:size(mu, 1)
    [~, k] = min(abs(mu(i, :) - 1));
    rest = mu(i, [1:k-1, k+1:end]);
    stable(i) = all(isfinite(mu(i, :))) && all(abs(rest) < 1);
end
end

function list = specialList(st)
% The specials of a status struct as a cell row of scalar structs.
list = {};
if ~isfield(st, "specials") || isempty(st.specials)
    return
end
s = st.specials;
if iscell(s)
    list = reshape(s, 1, []);
else
    list = num2cell(reshape(s, 1, []));
end
end

function v = numberField(s, name)
v = NaN;
if isfield(s, name) && ~isempty(s.(name))
    v = double(s.(name));
end
end

function v = textField(s, name)
v = "";
if isfield(s, name) && ~isempty(s.(name))
    v = string(s.(name));
end
end

function v = flagField(s, name)
v = false;
if isfield(s, name) && ~isempty(s.(name))
    v = logical(s.(name));
end
end

function v = structField(s, name)
v = struct();
if isfield(s, name) && isstruct(s.(name)) && isscalar(s.(name))
    v = s.(name);
end
end
