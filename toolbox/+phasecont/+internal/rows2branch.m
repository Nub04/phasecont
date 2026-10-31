function br = rows2branch(m, free, p, n, points, specials, branchStatus, kind)
%ROWS2BRANCH Build a phasecont.Branch from engine rows. Internal.
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST)
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST, KIND)
%
%   M         the phasecont.Model (empty is allowed in tests)
%   FREE      name of the free parameter, or its 1-based index; a curve
%             takes two, [a1 a2]
%   P         parameter vector the run started from
%   N         number of states
%   POINTS    stride-by-N matrix from run_copy, one column per point
%   SPECIALS  stride-by-K matrix of the special points
%   ST        the status struct of this branch (stop, warnings, specials)
%   KIND      "EP" (default), "LP" or "H"
%
% An equilibrium column has the layout [alpha, u(1..n), eig_re(1..n),
% eig_im(1..n), psi_LP, psi_BP, psi_H, h, newton_iters].
% A curve column is [a2, x(1..n), a1, (kappa for H,) eig_re(1..n),
% eig_im(1..n), codim2(1..k), h, newton_iters], where k is the number of
% codim-2 labels of the kind. Both have NaN for missing values.
% The function makes no engine calls.
if nargin < 8
    kind = "EP";
end
kind = string(kind);
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
    otherwise
        error("phasecont:model:badOption", ...
            "The kind must be ""EP"", ""LP"" or ""H"".");
end
isCurve = kind ~= "EP";

[name, idx] = freeParameters(m, free, 1 + isCurve);
p = double(p(:));
stride = 1 + block + 2*n + extra + 2;
if size(points, 1) ~= stride || size(specials, 1) ~= stride
    error("phasecont:model:badRows", ...
        "Expected rows of %d values, got %d and %d.", stride, ...
        size(points, 1), size(specials, 1));
end

[alpha, a1, u, ev, tail, h, iters] = unpack(points, n, block, extra);
if isCurve
    alpha = [a1 alpha];
    tests = zeros(size(alpha, 1), 0);
    codim2 = tail;
    stable = false(size(alpha, 1), 1);
else
    tests = tail;
    codim2 = zeros(size(alpha, 1), 0);
    stable = all(real(ev) < 0, 2);
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
    [a2, a1k, uk, ek] = unpack(specials(:, k), n, block, extra);
    pk = p;
    if isCurve
        pk(idx) = [a1k, a2];
    else
        pk(idx) = a2;
    end
    data = struct( ...
        "coefficient", numberField(info{k}, "coefficient"), ...
        "coefficient2", numberField(info{k}, "coefficient2"), ...
        "omega", numberField(info{k}, "omega"), ...
        "normalization", textField(info{k}, "normalization"), ...
        "refined", flagField(info{k}, "refined"), ...
        "values", structField(info{k}, "coefficients"));
    pts(end+1, 1) = phasecont.Point(U = uk(:), P = pk, ...
        Eigenvalues = ek(:), Label = string(info{k}.label), ...
        Index = double(info{k}.index) + 1, Data = data, Model = m); %#ok<AGROW>
end

br = phasecont.Branch(struct( ...
    "Type", kind, "Free", name, "FreeIndex", idx, "P", p, "Alpha", alpha, ...
    "U", u, "Eigenvalues", ev, "Stable", stable, "Tests", tests, ...
    "Codim2", codim2, "Codim2Labels", labels, ...
    "StepSize", h, "NewtonIters", iters, "Specials", pts, ...
    "Stop", stop, "Warnings", warn, "Model", m));
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

function [alpha, a1, u, ev, tail, h, iters] = unpack(rows, n, block, extra)
% Split stride-by-N rows into N-by-... arrays. a1 is empty for an
% equilibrium run, where the state block is just u.
alpha = rows(1, :).';
u = rows(2:1+n, :).';
a1 = zeros(size(alpha));
if block > n
    a1 = rows(2+n, :).';
end
e = 1 + block;
ev = complex(rows(e+1:e+n, :).', rows(e+n+1:e+2*n, :).');
t = e + 2*n;
tail = rows(t+1:t+extra, :).';
h = rows(t+extra+1, :).';
iters = rows(t+extra+2, :).';
if isempty(alpha)
    u = zeros(0, n);
    ev = complex(zeros(0, n));
    tail = zeros(0, extra);
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
