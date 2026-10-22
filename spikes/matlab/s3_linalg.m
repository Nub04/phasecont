% Spike S3 inside MATLAB: linear algebra for collocation systems.
%
%   spikes/stage.sh
%   scripts/matlab-win --run spikes/matlab/s3_linalg.m
%
% For N = 40, 100 and 400 (n = m = 4) the script
%   - builds the bordered collocation matrix in Rust ("s3_build"), copies it to
%     MATLAB ("s3_copy") and makes a sparse matrix from the triplets
%   - solves A*x = b in Rust: sparse LU at 1 thread, condensed sparse and condensed
%     dense at 1 thread and at maxNumCompThreads (no dense solve when the reduced
%     order is above 5000)
%   - solves it in MATLAB with decomposition(A) and with A\b, timed with timeit
%   - checks every backward error (at most 1e-13) and the differences between all
%     solutions (at most 1e-10), and prints condest(A) next to that difference
%   - runs the same suite inside Rust with run_json("s3") and checks its report
% Needs the staged pckern.dll and stops with spikes:nodll without it.

here = fileparts(mfilename('fullpath'));
addpath(here);
stage = 'C:\Users\<user>\phasecont-stage\spikes';
if ~isfile(fullfile(stage, 'pckern.dll'))
    error('spikes:nodll', 'no pckern.dll in %s (run spikes/stage.sh first)', stage);
end
fprintf('MATLAB %s, %s, maxNumCompThreads %d\n', version, computer('arch'), maxNumCompThreads);
cd(stage);
[info, dll] = pckern("info");
fprintf('%s\nloaded %s\n', info, dll);

n = 4;
m = 4;
reps = 20;
T = maxNumCompThreads;
ok = true;
for N = [40 100 400]
    fprintf('\n== N = %d (n = %d, m = %d, %d thread(s) for the parallel runs) ==\n', N, n, m, T);
    z = s3_expected_sizes(n, m, N);
    [h, nr, nz] = pckern("s3_build", sprintf("n=%d m=%d N=%d", n, m, N));
    ok = report(isa(h, 'uint64') && isscalar(h), 'handle is a uint64 scalar', class(h)) && ok;
    ok = report(nr == z.nr && nz == z.nnz, 's3_build sizes match the contract formulas', ...
        sprintf('nr %d (%d), nnz %d (%d)', nr, z.nr, nz, z.nnz)) && ok;

    % Copy the triplets and build the sparse matrix. sparse() drops zero values and
    % adds duplicate positions, so a count mismatch is traced to one of the two.
    [I, J, V, b] = pckern("s3_copy", h, nr, nz);
    try
        A = sparse(I, J, V, nr, nr);
    catch e
        ok = report(false, 'sparse(I, J, V, nr, nr)', sprintf('%s | %s', e.identifier, e.message)) && ok;
        pckern("free_handle", h);
        continue
    end
    ok = report(isequal(size(A), [nr nr]) && numel(b) == nr, 'size(A) and numel(b)', ...
        sprintf('A %dx%d, numel(b) %d, nr %d', size(A, 1), size(A, 2), numel(b), nr)) && ok;
    ok = report(nnz(A) == nz, 'nnz(A) equals the nnz from s3_build', ...
        sprintf('nnz(A) %d, s3_build %d', nnz(A), nz)) && ok;
    if nnz(A) ~= nz
        fprintf('  %s\n', s3_diagnose_nnz(I, J, V, nz));
    end

    % Rust solves.
    runs = {'sparse_lu', 1};
    for t = unique([1 T])
        runs(end + 1, :) = {'condensed_sparse', t}; %#ok<SAGROW>
    end
    if z.nred <= 5000
        for t = unique([1 T])
            runs(end + 1, :) = {'condensed_dense', t}; %#ok<SAGROW>
        end
    else
        fprintf('condensed_dense skipped: reduced order %d is above 5000\n', z.nred);
    end
    xs = cell(1, 0);
    for k = 1:size(runs, 1)
        [ok_k, x] = s3_solve_check(h, A, b, nz, runs{k, 1}, runs{k, 2}, reps);
        ok = ok_k && ok;
        if ~isempty(x)
            xs{end + 1} = x; %#ok<SAGROW>
        end
    end

    % MATLAB solves, timed in s3_time_matlab.
    res = s3_time_matlab(A, b);
    be_decomp = s3_backward_error(A, res.x_decomp, b);
    be_back = s3_backward_error(A, res.x_backslash, b);
    ok = report(be_decomp <= 1e-13, 'MATLAB decomposition(A) + solve backward error <= 1e-13', ...
        sprintf('%.3e, %.1f us', be_decomp, 1e6 * res.t_decomp)) && ok;
    ok = report(be_back <= 1e-13, 'MATLAB A\b backward error <= 1e-13', ...
        sprintf('%.3e, %.1f us', be_back, 1e6 * res.t_backslash)) && ok;
    xs = [xs, {res.x_decomp, res.x_backslash}]; %#ok<AGROW>

    % All solutions against each other. The first one is the sparse LU solution
    % unless that solve failed.
    d = s3_max_pairwise_diff(xs, xs{1});
    ok = report(d <= 1e-10, 'largest pairwise solution difference <= 1e-10', ...
        sprintf('%.3e over %d solutions, condest(A) %.3e', d, numel(xs), condest(A))) && ok;

    % The same suite inside Rust.
    try
        json = pckern("run_json", "s3", sprintf("n=%d m=%d N=%d threads=%d reps=%d", n, m, N, T, reps));
        fprintf('run_json("s3"):\n%s\n', json);
        r = jsondecode(json);
        want = struct('threads', T, 'reps', reps, 'nr', nr, 'nnz', nz);
        ok = s3_check_run(r, want) && ok;
    catch e
        ok = report(false, 'run_json s3', sprintf('%s | %s', e.identifier, e.message)) && ok;
    end
    ok = s1_expect_ok(sprintf('free_handle, N = %d', N), @() pckern("free_handle", h)) && ok;
end

assert(ok, 'spikes:s3:fail', 'S3 checks failed');
disp('S3 checks: all passed');
