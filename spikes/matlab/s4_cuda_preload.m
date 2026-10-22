% Spike S4 inside MATLAB, variant b: load the CUDA 13.4 toolkit's NVRTC first.
%
%   spikes/stage.sh
%   scripts/matlab-win --run spikes/matlab/s4_cuda_preload.m
%
% Needs the CUDA 13.4 toolkit (environment variable CUDA_PATH_V13_4) and the staged
% pckern.dll, and stops with spikes:nodll or spikes:nocuda without them. Run it as its
% own scripts/matlab-win call, because it must be the first CUDA user in the process.
% It loads nvrtc-builtins64_134.dll and then nvrtc64_130_0.dll from the toolkit's
% bin\x64 folder with pckern("preload", ...), before any CUDA call. After that
% run_json("s4") must report NVRTC 13.4, and the loaded nvrtc64_130_0.dll must lie
% in the toolkit folder. Variant a, s4_cuda.m, uses the default DLL search.

here = fileparts(mfilename('fullpath'));
addpath(here);
stage = 'C:\Users\<user>\phasecont-stage\spikes';
if ~isfile(fullfile(stage, 'pckern.dll'))
    error('spikes:nodll', 'no pckern.dll in %s (run spikes/stage.sh first)', stage);
end
fprintf('MATLAB %s, %s, maxNumCompThreads %d\n', version, computer('arch'), maxNumCompThreads);
cuda = getenv('CUDA_PATH_V13_4');
if isempty(cuda)
    error('spikes:nocuda', 'CUDA_PATH_V13_4 is not set; install the CUDA 13.4 toolkit first');
end
bin = fullfile(cuda, 'bin', 'x64');
fprintf('CUDA_PATH_V13_4 %s\ntoolkit bin folder %s\n', cuda, bin);
cd(stage);
[info, dll] = pckern("info");
fprintf('%s\nloaded %s\n', info, dll);
ok = true;

% The builtins DLL goes first, then NVRTC itself. Nothing may touch CUDA before this.
for name = {s4_builtins_name(13, 4), 'nvrtc64_130_0.dll'}
    try
        loaded = pckern("preload", fullfile(bin, name{1}));
        ok = report(true, ['preload ' name{1}], loaded) && ok;
    catch e
        ok = report(false, ['preload ' name{1}], sprintf('%s | %s', e.identifier, e.message)) && ok;
    end
end

ok = s4_run_report(4) && ok;
nvrtc_path = pckern("module_path", "nvrtc64_130_0.dll");
ok = report(s4_path_in_folder(nvrtc_path, bin), 'nvrtc64_130_0.dll was loaded from the toolkit folder', ...
    sprintf('loaded %s, toolkit folder %s', nvrtc_path, bin)) && ok;

assert(ok, 'spikes:s4:fail', 'S4 checks failed (variant b)');
disp('S4 checks, variant b (toolkit NVRTC preloaded): all passed');
