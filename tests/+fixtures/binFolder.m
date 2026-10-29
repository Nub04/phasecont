function folder = binFolder(root)
% Folder that holds pckern.<mexext> and the pckern library for this MATLAB.
%
% 1. PHASECONT_BIN, when set and non-empty.
% 2. On Windows with the repo under \\wsl.localhost\ or \\wsl$\, the stage
%    folder on the user's profile (Windows MATLAB must load binaries from C:).
% 3. Otherwise toolbox/bin/<arch> under the repo root.
%
% The folder is not created.
override = string(getenv("PHASECONT_BIN"));
if strlength(override) > 0
    folder = override;
elseif ispc && startsWith(string(root), ["\\wsl.localhost\" "\\wsl$\"], ...
        IgnoreCase = true)
    folder = string(fullfile(getenv("USERPROFILE"), "phasecont-stage", ...
        "bin", "win64"));
else
    folder = string(fullfile(root, "toolbox", "bin", computer("arch")));
end
end
