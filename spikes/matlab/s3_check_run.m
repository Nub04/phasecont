function ok = s3_check_run(r, want)
%S3_CHECK_RUN Checks and prints one decoded run_json("s3") report.
%   r     jsondecode of the JSON that pckern("run_json", "s3", args) returned
%   want  struct with threads (the parallel thread count that was asked for), reps,
%         and nr, nnz (the size of the matrix MATLAB got from s3_copy)
%   Checks the method list and order, every backward error (at most 1e-13),
%   max_pairwise_rel_diff (at most 1e-10), the bitwise flag and the report's own pass flag.
z = s3_expected_sizes(r.n, r.m, r.N);
fprintf('  target "%s", profile "%s", n %d, m %d, N %d, nr %d, nnz %d, nred %d, threads %d, reps %d\n', ...
    r.target, r.profile, r.n, r.m, r.N, r.nr, r.nnz, r.nred, r.threads, r.reps);
ok = report(strcmp(r.spike, 's3'), 'run spike', r.spike);
ok = report(r.threads == want.threads && r.reps == want.reps, 'run threads and reps', ...
    sprintf('threads %d, reps %d', r.threads, r.reps)) && ok;
ok = report(r.nr == want.nr && r.nnz == want.nnz, 'run sizes match the matrix MATLAB got', ...
    sprintf('report nr %d nnz %d, matrix nr %d nnz %d', r.nr, r.nnz, want.nr, want.nnz)) && ok;
ok = report(r.nr == z.nr && r.nnz == z.nnz && r.nred == z.nred, 'run sizes match the contract formulas', ...
    sprintf('nr %d (%d), nnz %d (%d), nred %d (%d)', r.nr, z.nr, r.nnz, z.nnz, r.nred, z.nred)) && ok;

% Method order from the contract: sparse_lu, condensed_sparse, condensed_dense;
% each at 1 thread, then at `threads` when that is above 1. No dense entries when nred > 5000.
expected = {'sparse_lu@1', 'condensed_sparse@1'};
if want.threads > 1
    expected{end + 1} = sprintf('condensed_sparse@%d', want.threads);
end
if z.nred <= 5000
    expected{end + 1} = 'condensed_dense@1';
    if want.threads > 1
        expected{end + 1} = sprintf('condensed_dense@%d', want.threads);
    end
end
methods = s3_rows(r.methods);
got = cell(1, numel(methods));
for k = 1:numel(methods)
    got{k} = sprintf('%s@%d', methods(k).method, methods(k).threads);
end
ok = report(isequal(got, expected), 'run method list', strjoin(got, ', ')) && ok;

for k = 1:numel(methods)
    m = methods(k);
    tag = sprintf('%s threads=%d', m.method, m.threads);
    ok = report(~isempty(m.backward_error) && m.backward_error <= 1e-13, [tag ' report backward error <= 1e-13'], ...
        s4_numstr(m.backward_error, '%.3e')) && ok;
    ok = s3_check_timings(tag, m.timings, m.method) && ok;
end

d = r.max_pairwise_rel_diff;
ok = report(~isempty(d) && d <= 1e-10, 'run max_pairwise_rel_diff <= 1e-10', s4_numstr(d, '%.3e')) && ok;
ok = report(isequal(r.sparse_threads_bitwise_equal, true), 'run sparse_threads_bitwise_equal', ...
    mat2str(r.sparse_threads_bitwise_equal)) && ok;
fprintf('  cond1_est %s (informational)\n', s4_numstr(r.cond1_est, '%.3e'));
ok = report(isequal(r.pass, true), 'run pass flag', mat2str(r.pass)) && ok;
end
