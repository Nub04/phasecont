function res = s3_time_matlab(A, b)
%S3_TIME_MATLAB Times MATLAB's own sparse solves of A*x = b with timeit.
%   res.t_decomp      seconds for decomposition(A) plus one solve (the factorization is included)
%   res.t_backslash   seconds for A\b
%   res.x_decomp, res.x_backslash   the two solutions
%   The timing lives in a function file because scripts run slower.
res.x_decomp = decomp_solve(A, b);
res.x_backslash = A \ b;
res.t_decomp = timeit(@() decomp_solve(A, b));
res.t_backslash = timeit(@() A \ b);
end

function x = decomp_solve(A, b)
d = decomposition(A);
x = d \ b;
end
