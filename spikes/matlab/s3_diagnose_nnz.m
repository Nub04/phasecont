function txt = s3_diagnose_nnz(I, J, V, nnz_expected)
%S3_DIAGNOSE_NNZ Explains why nnz(sparse(I,J,V)) can differ from the nnz the engine reported.
%   sparse() drops exact zeros and adds up repeated (row, column) pairs, so a
%   count mismatch means V holds zeros, [I J] holds repeated positions, or both.
nzeros = nnz(V == 0);
ndup = numel(I) - size(unique([I(:) J(:)], 'rows'), 1);
A = sparse(I, J, V);
if nzeros > 0 && ndup > 0
    cause = 'zeros in V and duplicate rows in [I J]';
elseif nzeros > 0
    cause = 'zeros in V';
elseif ndup > 0
    cause = 'duplicate rows in [I J]';
else
    cause = sprintf('neither (numel(V) is %d, expected nnz %d)', numel(V), nnz_expected);
end
txt = sprintf(['zeros in V: %d; duplicate positions in [I J]: %d; numel(V): %d; ' ...
    'expected nnz: %d; nnz(A): %d; cause: %s'], nzeros, ndup, numel(V), nnz_expected, nnz(A), cause);
end
