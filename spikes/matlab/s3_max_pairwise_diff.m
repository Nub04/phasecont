function d = s3_max_pairwise_diff(xs, xref)
%S3_MAX_PAIRWISE_DIFF Largest norm(xi - xj) / norm(xref) over all pairs of solutions.
%   xs    cell array of solution vectors, all with as many entries as xref
%   xref  the reference solution (the sparse LU result); only its 2-norm is used
%   The result is 0 for fewer than two solutions. A NaN in any pair makes the result
%   NaN, so a broken solution cannot pass a "d <= tol" test.
n = numel(xs);
for k = 1:n
    if numel(xs{k}) ~= numel(xref)
        error('spikes:badsize', 'solution %d has %d entries, expected %d', ...
            k, numel(xs{k}), numel(xref));
    end
end
d = 0;
scale = norm(xref(:));
for i = 1:n - 1
    for j = i + 1:n
        v = norm(xs{i}(:) - xs{j}(:)) / scale;
        if isnan(v)
            d = NaN;
            return
        end
        d = max(d, v);
    end
end
end
