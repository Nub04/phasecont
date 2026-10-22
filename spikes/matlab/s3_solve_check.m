function [ok, x] = s3_solve_check(h, A, b, nz, method, threads, reps)
%S3_SOLVE_CHECK Solves A*x = b in Rust with one method and checks the result.
%   h        problem handle from pckern("s3_build", ...)
%   A, b     the sparse matrix and right-hand side that pckern("s3_copy", ...) returned
%   nz       the stored nonzero count that s3_build returned
%   method   'sparse_lu', 'condensed_sparse' or 'condensed_dense'
%   threads  thread count for the solve, reps  timed runs
%   Checks the JSON report with s3_check_solve and the backward error of x in MATLAB
%   (at most 1e-13). An error from pckern is a reported failure. x is [] whenever the
%   solve failed or returned a vector of the wrong size.
tag = sprintf('%s threads=%d', method, threads);
nr = size(A, 1);
x = [];
args = sprintf('method=%s threads=%d reps=%d', method, threads, reps);
try
    [xr, json] = pckern("s3_solve", h, nr, args);
    s = jsondecode(json);
catch e
    ok = report(false, [tag ' s3_solve call'], sprintf('%s | %s', e.identifier, e.message));
    return
end
want = struct('method', method, 'threads', threads, 'reps', reps, 'nr', nr, 'nnz', nz);
ok = s3_check_solve(s, want);
right_shape = isa(xr, 'double') && isequal(size(xr), [nr 1]);
ok = report(right_shape, [tag ' x is an nr-by-1 double'], ...
    sprintf('size %s, class %s', mat2str(size(xr)), class(xr))) && ok;
if ~right_shape
    return
end
x = xr;
be = s3_backward_error(A, x, b);
ok = report(be <= 1e-13, [tag ' MATLAB backward error <= 1e-13'], sprintf('%.3e', be)) && ok;
end
