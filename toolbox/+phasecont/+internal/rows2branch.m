function br = rows2branch(m, free, p, n, points, specials, branchStatus)
%ROWS2BRANCH Build a phasecont.Branch from engine rows. Internal.
%   BR = rows2branch(M, FREE, P, N, POINTS, SPECIALS, ST)
%
%   M         the phasecont.Model (empty is allowed in tests)
%   FREE      name of the free parameter, or its 1-based index
%   P         parameter vector the run started from
%   N         number of states
%   POINTS    stride-by-N matrix from run_copy, one column per point
%   SPECIALS  stride-by-K matrix of the special points
%   ST        the status struct of this branch (stop, warnings, specials)
%
% A column has the layout [alpha, u(1..n), eig_re(1..n), eig_im(1..n),
% psi_LP, psi_BP, psi_H, h, newton_iters], with NaN for missing values.
% The function makes no engine calls.
if isnumeric(free)
    idx = double(free);
    if isempty(m)
        name = "p" + idx;
    else
        name = m.Parameters(idx);
    end
else
    name = string(free);
    idx = find(m.Parameters == name, 1);
end
p = double(p(:));
stride = 3*n + 6;
if size(points, 1) ~= stride || size(specials, 1) ~= stride
    error("phasecont:model:badRows", ...
        "Expected rows of %d values, got %d and %d.", stride, ...
        size(points, 1), size(specials, 1));
end

[alpha, u, ev, tests, h, iters] = unpack(points, n);
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
    [a, uk, ek] = unpack(specials(:, k), n);
    pk = p;
    pk(idx) = a;
    data = struct( ...
        "coefficient", numberField(info{k}, "coefficient"), ...
        "omega", numberField(info{k}, "omega"), ...
        "normalization", textField(info{k}, "normalization"), ...
        "refined", flagField(info{k}, "refined"));
    pts(end+1, 1) = phasecont.Point(U = uk(:), P = pk, ...
        Eigenvalues = ek(:), Label = string(info{k}.label), ...
        Index = double(info{k}.index) + 1, Data = data, Model = m); %#ok<AGROW>
end

br = phasecont.Branch(struct( ...
    "Free", name, "FreeIndex", idx, "P", p, "Alpha", alpha, "U", u, ...
    "Eigenvalues", ev, "Stable", all(real(ev) < 0, 2), "Tests", tests, ...
    "StepSize", h, "NewtonIters", iters, "Specials", pts, ...
    "Stop", stop, "Warnings", warn, "Model", m));
end

function [alpha, u, ev, tests, h, iters] = unpack(rows, n)
% Split stride-by-N rows into N-by-... arrays.
alpha = rows(1, :).';
u = rows(2:1+n, :).';
ev = complex(rows(2+n:1+2*n, :).', rows(2+2*n:1+3*n, :).');
tests = rows(2+3*n:4+3*n, :).';
h = rows(5+3*n, :).';
iters = rows(6+3*n, :).';
if isempty(alpha)
    u = zeros(0, n);
    ev = complex(zeros(0, n));
    tests = zeros(0, 3);
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
