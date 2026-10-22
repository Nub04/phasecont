function info = s4_module_info(r, lookup)
%S4_MODULE_INFO Prints where Windows loaded the NVRTC, driver and builtins DLLs from.
%   r       decoded S4 report (uses nvrtc_major and nvrtc_minor)
%   lookup  lookup(name) returns the full path of a loaded module, or '' when it is
%           not loaded. The spike scripts pass @(name) pckern("module_path", name).
%   The NVRTC version in the report is authoritative. These paths are supporting
%   evidence for which copy of the library the process loaded.
info = struct('nvrtc_name', '', 'nvrtc_path', '', 'nvcuda_path', '', ...
    'builtins_name', '', 'builtins_path', '');
names = s4_nvrtc_names();
for k = 1:numel(names)
    p = char(lookup(names{k}));
    if ~isempty(p)
        info.nvrtc_name = names{k};
        info.nvrtc_path = p;
        break
    end
end
if isempty(info.nvrtc_name)
    fprintf('module_path: none of these is loaded: %s\n', strjoin(names, ', '));
else
    fprintf('module_path: %s -> %s\n', info.nvrtc_name, info.nvrtc_path);
end
info.nvcuda_path = char(lookup('nvcuda.dll'));
fprintf('module_path: nvcuda.dll -> %s\n', path_or_none(info.nvcuda_path));
info.builtins_name = s4_builtins_name(r.nvrtc_major, r.nvrtc_minor);
info.builtins_path = char(lookup(info.builtins_name));
fprintf('module_path: %s -> %s\n', info.builtins_name, path_or_none(info.builtins_path));
end

function s = path_or_none(p)
if isempty(p)
    s = '(not loaded)';
else
    s = p;
end
end
