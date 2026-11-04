% Spike S1, part 2: the FFI round trip through pckern.dll.
%
%   spikes/stage.sh
%   scripts/matlab-win --run spikes/matlab/s1_ffi.m
%
% Needs the staged pckern.dll and stops with spikes:nodll without it. Checks:
%   - "scale" for empty, scalar, row, column, matrix, N-d and NaN/Inf inputs,
%     called with a string command and with a char command
%   - a panic on the MATLAB thread and one on a worker thread, each raised twice
%     as a MATLAB error, with "scale" still working after every one
%   - a test handle: the first free works, the second free is an error
%   - the argument checks of "scale"

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
ok = true;

% "scale" round trip. The result must equal 2*x, with the size of x and class double.
m34 = reshape(1:12, 3, 4) / 7;
cases = {
    'empty 0x0', zeros(0, 0);
    'empty 1x0', zeros(1, 0);
    'empty 0x3', zeros(0, 3);
    'scalar', 3.25;
    'row', [1 -2.5 0 1e300 -1e-300];
    'column', (1:4)' / 3;
    'matrix 3x4', m34;
    'array 2x3x2', reshape(1:12, 2, 3, 2) / 5;
    'NaN and Inf', [NaN Inf -Inf 0 -0 1.5];
    };
for k = 1:size(cases, 1)
    x = cases{k, 2};
    ok = s1_check_scale(['string command, ' cases{k, 1}], @(v) pckern("scale", v, 2), x) && ok;
    ok = s1_check_scale(['char command,   ' cases{k, 1}], @(v) pckern('scale', v, 2), x) && ok;
end
% 0.5 times 4 is exactly 2, so this fails if Rust ignores the factor.
ok = s1_check_scale('factor 0.5, times 4', @(v) 4 * pckern("scale", v, 0.5), m34) && ok;

% A panic must come back as a MATLAB error and leave the library usable.
probe = [1 2 3];
for k = 1:2
    ok = s1_expect_error(sprintf('panic, call %d', k), @() pckern("panic"), ...
        'phasecont:engine:panic', 'panic: spike panic: 100% on purpose') && ok;
    ok = s1_check_scale(sprintf('scale after panic %d', k), @(v) pckern("scale", v, 2), probe) && ok;
end
for k = 1:2
    ok = s1_expect_error(sprintf('panic_in_worker, call %d', k), @() pckern("panic_in_worker"), ...
        'phasecont:engine:panic', {'spike worker panic at item', '(raised on another thread)'}) && ok;
    ok = s1_check_scale(sprintf('scale after panic_in_worker %d', k), @(v) pckern("scale", v, 2), ...
        probe) && ok;
end

% Handles are opaque uint64 ids. A freed or never issued id is an error.
h = pckern("test_handle");
ok = report(isa(h, 'uint64') && isscalar(h), 'test_handle returns a uint64 scalar', ...
    sprintf('class %s, size %s, value %u', class(h), mat2str(size(h)), h)) && ok;
ok = s1_expect_ok('free_handle on a live handle', @() pckern("free_handle", h)) && ok;
ok = s1_expect_error('second free_handle', @() pckern("free_handle", h), ...
    'phasecont:engine:badHandle', 'unknown handle 0x') && ok;
ok = s1_expect_error('free_handle on id 0', @() pckern("free_handle", uint64(0)), ...
    'phasecont:engine:badHandle', 'unknown handle 0x') && ok;
ok = s1_expect_error('free_handle on a double', @() pckern("free_handle", 12345), ...
    'phasecont:engine:badArg', 'the handle must be a uint64 scalar') && ok;

% Argument checks of "scale", each raised by the gateway.
bad_x = 'x must be a real, full double array';
ok = s1_expect_error('scale rejects single x', @() pckern("scale", single([1 2 3]), 2), ...
    'phasecont:engine:badArg', bad_x) && ok;
ok = s1_expect_error('scale rejects int32 x', @() pckern("scale", int32([1 2 3]), 2), ...
    'phasecont:engine:badArg', bad_x) && ok;
ok = s1_expect_error('scale rejects complex x', @() pckern("scale", [1 + 2i, 3], 2), ...
    'phasecont:engine:badArg', bad_x) && ok;
ok = s1_expect_error('scale rejects sparse x', @() pckern("scale", sparse([1 0 2]), 2), ...
    'phasecont:engine:badArg', bad_x) && ok;
bad_a = 'a must be a real double scalar';
ok = s1_expect_error('scale rejects single a', @() pckern("scale", probe, single(2)), ...
    'phasecont:engine:badArg', bad_a) && ok;
ok = s1_expect_error('scale rejects vector a', @() pckern("scale", probe, [1 2]), ...
    'phasecont:engine:badArg', bad_a) && ok;
ok = s1_check_scale('scale after the argument errors', @(v) pckern("scale", v, 2), probe) && ok;

assert(ok, 'spikes:s1:fail', 'S1 FFI checks failed');
disp('S1 FFI checks: all passed');
