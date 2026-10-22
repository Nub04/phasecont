function [ok, r, mods] = s4_run_report(want_minor)
%S4_RUN_REPORT Runs pckern("run_json", "s4", ""), prints and checks the report and the loaded DLLs.
%   want_minor  optional NVRTC minor version this process must have loaded (4 for CUDA 13.4)
%   ok    true when the call worked and s4_check_report passed
%   r     the decoded report, or [] when the call failed
%   mods  struct from s4_module_info: where Windows loaded NVRTC, nvcuda and the builtins DLL from
%   An error from run_json is a reported failure. The loaded DLL paths are still printed,
%   because they show which NVRTC copy the process found.
if nargin < 1
    want_minor = [];
end
lookup = @(name) pckern("module_path", name);
r = [];
try
    json = pckern("run_json", "s4", "");
    fprintf('run_json("s4"):\n%s\n', json);
    r = jsondecode(json);
catch e
    ok = report(false, 'run_json s4', sprintf('%s | %s', e.identifier, e.message));
    fprintf('NVRTC version unknown, looking for the 13.0 builtins DLL name\n');
    mods = s4_module_info(struct('nvrtc_major', 13, 'nvrtc_minor', 0), lookup);
    return
end
ok = s4_check_report(r, want_minor);
mods = s4_module_info(r, lookup);
end
