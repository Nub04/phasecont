function ok = s3_check_solve(s, want)
%S3_CHECK_SOLVE Checks and prints one decoded s3_solve report.
%   s     jsondecode of the JSON that pckern("s3_solve", ...) returned
%   want  struct with method, threads, reps (what the call asked for) and
%         nr, nnz (the size of the matrix MATLAB got from s3_copy)
%   The report's backward error must be at most 1e-13. A null decodes to [] and fails.
tag = sprintf('%s threads=%d', want.method, want.threads);
z = s3_expected_sizes(s.n, s.m, s.N);
fprintf('  %s: reps %d, n %d, m %d, N %d, nr %d, nnz %d, nred %d, backward_error %s\n', tag, ...
    s.reps, s.n, s.m, s.N, s.nr, s.nnz, s.nred, s4_numstr(s.backward_error, '%.3e'));
ok = report(strcmp(s.spike, 's3') && strcmp(s.method, want.method), [tag ' spike and method'], ...
    sprintf('%s, %s', s.spike, s.method));
ok = report(s.threads == want.threads && s.reps == want.reps, [tag ' threads and reps'], ...
    sprintf('threads %d, reps %d', s.threads, s.reps)) && ok;
ok = report(s.nr == want.nr && s.nnz == want.nnz, [tag ' sizes match the matrix MATLAB got'], ...
    sprintf('report nr %d nnz %d, matrix nr %d nnz %d', s.nr, s.nnz, want.nr, want.nnz)) && ok;
ok = report(s.nr == z.nr && s.nnz == z.nnz && s.nred == z.nred, [tag ' sizes match the contract formulas'], ...
    sprintf('nr %d (%d), nnz %d (%d), nred %d (%d)', s.nr, z.nr, s.nnz, z.nnz, s.nred, z.nred)) && ok;
ok = report(~isempty(s.backward_error) && s.backward_error <= 1e-13, [tag ' report backward error <= 1e-13'], ...
    s4_numstr(s.backward_error, '%.3e')) && ok;
ok = s3_check_timings(tag, s.timings, want.method) && ok;
end
