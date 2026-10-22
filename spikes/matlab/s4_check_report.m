function ok = s4_check_report(r, want_minor)
%S4_CHECK_REPORT Prints a decoded run_json("s4") report and checks it against the contract.
%   r           jsondecode of the JSON that pckern("run_json", "s4", "") returned
%   want_minor  optional NVRTC minor version this process must have loaded (4 for CUDA 13.4)
%   JSON nulls decode to [] and print as null. The check fails unless the report's own
%   pass flag is true and agrees with the chosen path.
if nargin < 2
    want_minor = [];
end
fprintf('  target %s, profile %s, n %s, a %s, reps %s\n', q(r.target), q(r.profile), ...
    s4_numstr(r.n, '%d'), s4_numstr(r.a, '%.9g'), s4_numstr(r.reps, '%d'));
fprintf('  driver_version %s, NVRTC %s.%s\n', s4_numstr(r.driver_version, '%d'), ...
    s4_numstr(r.nvrtc_major, '%d'), s4_numstr(r.nvrtc_minor, '%d'));
fprintf('  device %s, compute capability %s.%s\n', q(r.device), s4_numstr(r.cc_major, '%d'), ...
    s4_numstr(r.cc_minor, '%d'));
fprintf('  nvrtc_path %s, chosen %s\n', q(r.nvrtc_path), q(r.chosen));
paths = s3_rows(r.paths);
for k = 1:numel(paths)
    print_path(paths(k));
end

ok = report(strcmp(r.spike, 's4'), 'S4 spike', r.spike);
ok = report(numel(paths) == 2 && strcmp(paths(1).name, 'ptx') && strcmp(paths(2).name, 'cubin'), ...
    'S4 paths are ptx then cubin', sprintf('%d path(s)', numel(paths))) && ok;

% Selection rule from the contract.
if r.driver_version >= r.nvrtc_major * 1000 + r.nvrtc_minor * 10
    want_chosen = 'ptx';
elseif r.driver_version >= 13000
    want_chosen = 'cubin';
else
    want_chosen = '';
end
ok = report(strcmp(r.chosen, want_chosen), 'S4 chosen follows the selection rule', ...
    sprintf('chosen %s, rule gives %s', q(r.chosen), q(want_chosen))) && ok;

for k = 1:numel(paths)
    p = paths(k);
    rows = s3_rows(p.timings);
    if ~isempty(p.error)
        shape = isempty(rows) && isequal(p.pass, false);
        what = 'failed path: error text, no timing rows, pass false';
    elseif isequal(p.pass, true)
        shape = isscalar(rows) && strcmp(rows(1).name, 'kernel') && ~isempty(rows(1).median_ns) ...
            && rows(1).median_ns > 0 && ~isempty(p.gbps) && p.gbps > 0 ...
            && ~isempty(p.compile_ms) && ~isempty(p.load_ms);
        what = 'working path: one kernel row, gbps > 0, compile_ms and load_ms set';
    else
        shape = true;
        what = 'path ran but failed the ulp check';
    end
    ok = report(shape, ['S4 path ' p.name ' shape'], what) && ok;
    want_pass = isempty(p.error) && ~isempty(p.max_ulp_fma) && p.max_ulp_fma <= 1;
    ok = report(isequal(p.pass, want_pass), ['S4 path ' p.name ' pass flag'], ...
        sprintf('pass %s, error empty and max_ulp_fma <= 1 gives %s', mat2str(p.pass), mat2str(want_pass))) && ok;
end

idx = find(strcmp({paths.name}, r.chosen), 1);
chosen_ok = ~isempty(idx) && isequal(paths(idx).pass, true);
ok = report(chosen_ok, 'S4 the chosen path exists and passes', sprintf('chosen %s', q(r.chosen))) && ok;
ok = report(isequal(r.pass, chosen_ok), 'S4 pass flag agrees with the chosen path', ...
    sprintf('pass %s, chosen path passes %s', mat2str(r.pass), mat2str(chosen_ok))) && ok;
ok = report(isequal(r.pass, true), 'S4 report pass flag is true', mat2str(r.pass)) && ok;
if ~isempty(want_minor)
    ok = report(r.nvrtc_minor == want_minor, 'S4 NVRTC minor version', ...
        sprintf('loaded 13.%d, wanted 13.%d', r.nvrtc_minor, want_minor)) && ok;
end
end

function print_path(p)
fprintf('  path %s: arch %s, compile_ms %s, load_ms %s, gbps %s\n', q(p.name), q(p.arch), ...
    s4_numstr(p.compile_ms, '%.3f'), s4_numstr(p.load_ms, '%.3f'), s4_numstr(p.gbps, '%.1f'));
fprintf('    max_ulp_fma %s, max_ulp_unfused %s, pass %s\n', s4_numstr(p.max_ulp_fma, '%g'), ...
    s4_numstr(p.max_ulp_unfused, '%g'), mat2str(p.pass));
fprintf('    error %s\n', q(p.error));
rows = s3_rows(p.timings);
for j = 1:numel(rows)
    fprintf('    timing %s: median %s us, min %s us, max %s us\n', q(rows(j).name), ...
        s4_numstr(rows(j).median_ns / 1e3, '%.1f'), s4_numstr(rows(j).min_ns / 1e3, '%.1f'), ...
        s4_numstr(rows(j).max_ns / 1e3, '%.1f'));
end
end

function s = q(t)
% Quotes a string, so an empty string still fills its slot in fprintf.
s = ['"' char(t) '"'];
end
