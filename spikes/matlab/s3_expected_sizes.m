function z = s3_expected_sizes(n, m, N)
%S3_EXPECTED_SIZES Sizes of the S3 collocation system, from the spike contract.
%   z.nr   = n(Nm+1)+2                                  matrix order
%   z.nnz  = N*n*m*(n(m+1)+2) + 2n + n(Nm+1) + nr       stored nonzeros
%   z.nred = n(N+1)+2                                   order of the condensed system
z.nr = n * (N * m + 1) + 2;
z.nnz = N * n * m * (n * (m + 1) + 2) + 2 * n + n * (N * m + 1) + z.nr;
z.nred = n * (N + 1) + 2;
end
