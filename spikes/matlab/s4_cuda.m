% Spike S4 inside MATLAB, variant a: NVRTC and the CUDA driver found by the default
% Windows DLL search.
%
%   spikes/stage.sh
%   scripts/matlab-win --run spikes/matlab/s4_cuda.m
%
% Each scripts/matlab-win call starts a fresh MATLAB process, and this script makes
% no CUDA call before run_json("s4"). MATLAB ships its own nvrtc64_130_0.dll in
% bin\win64, so the process may load that copy (NVRTC 13.1) and not the 13.4 toolkit
% copy; the report's NVRTC version says which one it was. Variant b, s4_cuda_preload.m,
% loads the toolkit copy first. Needs the staged pckern.dll and stops with
% spikes:nodll without it.

here = fileparts(mfilename('fullpath'));
addpath(here);
stage = fullfile(getenv('USERPROFILE'), 'phasecont-stage', 'spikes');
if ~isfile(fullfile(stage, 'pckern.dll'))
    error('spikes:nodll', 'no pckern.dll in %s (run spikes/stage.sh first)', stage);
end
fprintf('MATLAB %s, %s, maxNumCompThreads %d\n', version, computer('arch'), maxNumCompThreads);
cd(stage);
[info, dll] = pckern("info");
fprintf('%s\nloaded %s\n', info, dll);

ok = s4_run_report();
assert(ok, 'spikes:s4:fail', 'S4 checks failed (variant a)');
disp('S4 checks, variant a (default DLL search): all passed');
