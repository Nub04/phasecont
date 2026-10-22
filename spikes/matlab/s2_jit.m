% Spike S2 inside MATLAB: the Cranelift JIT against AOT Rust, AOT C and a
% MATLAB anonymous-function baseline.
%
%   spikes/stage.sh
%   scripts/matlab-win --run spikes/matlab/s2_jit.m
%
% The baseline needs no pckern.dll and always runs. The pckern runs need
% the staged DLL: "run_json" times the JIT and Rust code, and "bench_c"
% runs the same suite with spikes/mex/fhn_aot.c as compiled by mex (MSVC).

here = fileparts(mfilename('fullpath'));
addpath(here);
stage = 'C:\Users\<user>\phasecont-stage\spikes';
fprintf('MATLAB %s, %s, maxNumCompThreads %d\n', version, computer('arch'), maxNumCompThreads);

b = s2_baseline(1e5, 5);
fprintf('matlab anonymous rhs: median %.1f ns per call (%.1f to %.1f), 1e7 calls take about %.2f s\n', ...
    median(b.rhs_ns), min(b.rhs_ns), max(b.rhs_ns), b.rhs_1e7_s);
fprintf('matlab anonymous jac: median %.1f ns per call (%.1f to %.1f), 1e7 calls take about %.2f s\n', ...
    median(b.jac_ns), min(b.jac_ns), max(b.jac_ns), b.jac_1e7_s);

if ~isfile(fullfile(stage, 'pckern.dll'))
    fprintf('SKIP pckern runs: no pckern.dll in %s\n', stage);
    return
end
cd(stage);
[info, dll] = pckern("info");
fprintf('%s\nloaded %s\n', info, dll);
% Both runs print before any check, so a failed check still shows both.
% scripts/matlab-win starts a new MATLAB process, so run_json makes the first
% JIT compile in it and bench_c makes a second, warm one.
decoded = struct();
for entry = ["run_json", "bench_c"]
    if entry == "run_json"
        json = pckern("run_json", "s2", "");
        note = 'first JIT compile in this process';
    else
        json = pckern("bench_c", "");
        note = 'second compile, warm';
    end
    fprintf('%s:\n%s\n', entry, json);
    s = jsondecode(json);
    decoded.(entry) = s;
    fprintf('  compile_ms %.3f (%s)\n', s.compile_ms, note);
    for t = s.timings'
        fprintf('  %-22s median %.2f ns (%.2f to %.2f)\n', t.name, t.median_ns, t.min_ns, t.max_ns);
    end
    % jit time over the other variant's time, taken rep by rep.
    if isfield(s, 'ratios')
        for q = s.ratios'
            fprintf('  %-16s median %.3f (%.3f to %.3f), per rep %s\n', q.name, q.median, ...
                q.min, q.max, strtrim(sprintf('%.3f ', q.per_rep)));
        end
    end
    jit = s.timings(strcmp({s.timings.name}, 'jit rhs')).median_ns;
    fprintf('  matlab anonymous rhs / jit rhs = %.0f\n', median(b.rhs_ns)/jit);
end
% In each run the rhs variants must share one checksum, the jac variants
% another, and every max_ulp_vs_rust value must be 0.
for entry = ["run_json", "bench_c"]
    s = decoded.(entry);
    assert(s.rhs_checksums_match && s.jac_checksums_match, 'S2 %s: checksums differ', entry);
    assert(all(structfun(@(u) u == 0, s.max_ulp_vs_rust)), ...
        'S2 %s: a max_ulp_vs_rust value is not 0', entry);
end
