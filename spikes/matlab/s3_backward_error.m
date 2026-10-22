function be = s3_backward_error(A, x, b)
%S3_BACKWARD_ERROR Normwise backward error of x for A*x = b in the infinity norm.
%   be = norm(A*x-b,inf) / (norm(A,inf)*norm(x,inf) + norm(b,inf))
be = norm(A * x - b, inf) / (norm(A, inf) * norm(x, inf) + norm(b, inf));
end
