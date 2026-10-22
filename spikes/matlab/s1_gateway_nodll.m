% Spike S1, part 1: gateway checks that need no pckern.dll.
%
%   spikes/stage.sh --mex-only
%   scripts/matlab-win --run spikes/matlab/s1_gateway_nodll.m
%
% Copies pckern.mexw64 from the stage folder into an empty temporary folder,
% so the missing-DLL error can be checked even after pckern.dll is staged.
% It also builds the gateway once with the repo's \\wsl.localhost\ folder as
% the current folder, to check whether mex works there. Nothing is written
% to the repo.

here = fileparts(mfilename('fullpath'));
addpath(here);
repo = fileparts(fileparts(here));
stage = 'C:\Users\<user>\phasecont-stage\spikes';
fprintf('MATLAB %s\nrepo:  %s\nstage: %s\n', version, repo, stage);

probe = fullfile(tempdir, 'pckern_s1_nodll');
if isfolder(probe)
    rmdir(probe, 's');
end
mkdir(probe);
ok = true;

% mex with a UNC current folder and relative source paths. This answers an
% environment question, so it is reported but does not fail the script.
cd(repo);
try
    mex('-R2018a', '-outdir', probe, '-output', 'unc_probe', ...
        fullfile('spikes', 'mex', 'pckern_gateway.c'), fullfile('spikes', 'mex', 'fhn_aot.c'));
    built = isfile(fullfile(probe, 'unc_probe.mexw64'));
    report(built, 'mex with a UNC current folder', pwd);
catch e
    report(false, 'mex with a UNC current folder', e.message);
end

% The gateway with no pckern.dll next to it.
copyfile(fullfile(stage, 'pckern.mexw64'), probe);
cd(probe);
assert(~isfile('pckern.dll'));
fprintf('gateway: %s\n', which('pckern'));

ok = expect_error('no command', @() pckern(), 'phasecont:engine:badArg', 'usage') && ok;
ok = expect_error('numeric command', @() pckern(42), 'phasecont:engine:badArg', ...
    'character vector or string scalar') && ok;
ok = expect_error('string array command', @() pckern(["info", "scale"]), ...
    'phasecont:engine:badArg', 'must be a string scalar') && ok;
ok = expect_error('empty string array command', @() pckern(string.empty), ...
    'phasecont:engine:badArg', 'must be a string scalar') && ok;
ok = expect_error('char matrix command', @() pckern(['ab'; 'cd']), ...
    'phasecont:engine:badArg', 'character vector or string scalar') && ok;
ok = expect_error('missing string command', @() pckern(string(missing)), ...
    'phasecont:engine:badArg', '') && ok;
ok = expect_error('unknown command', @() pckern('nope'), 'phasecont:engine:badArg', ...
    'unknown command "nope"') && ok;
for k = 1:2
    ok = expect_error(sprintf('info without pckern.dll, call %d', k), @() pckern("info"), ...
        'phasecont:engine:load', 'pckern.dll: The specified module could not be found. (Windows error 126)') && ok;
end
ok = expect_error('scale without pckern.dll', @() pckern('scale', 1, 2), ...
    'phasecont:engine:load', 'Windows error 126') && ok;

% Commands that work without pckern.dll.
k32 = pckern("module_path", "kernel32.dll");
ok = report(endsWith(k32, '\kernel32.dll', 'IgnoreCase', true), 'module_path kernel32.dll', k32) && ok;
none = pckern("module_path", "pckern.dll");
ok = report(isempty(none), 'module_path pckern.dll is empty', ['[' none ']']) && ok;
ok = expect_error('preload a missing file', @() pckern("preload", "C:\nope\missing.dll"), ...
    'phasecont:engine:load', 'cannot load C:\nope\missing.dll') && ok;
loaded = pckern("preload", fullfile(getenv('SystemRoot'), 'System32', 'kernel32.dll'));
ok = report(strcmpi(loaded, k32), 'preload kernel32.dll by full path', loaded) && ok;

if ~ok
    error('spikes:s1:fail', 'S1 gateway checks failed');
end
disp('S1 gateway checks without pckern.dll: all passed');
