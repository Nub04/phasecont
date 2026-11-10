function value = wslArg(path)
% Check a path that buildfile.m puts inside a double-quoted wsl.exe argument.
%   value = wslArg(path) returns the path as a string. It raises
%   phasecont:build:unsafePath for a double quote, a dollar sign, a backtick,
%   a backslash, a percent sign or a line break. In a double-quoted argument
%   bash expands the dollar sign and the backtick, cmd.exe expands a percent
%   sign, and wsl.exe drops a backslash, so these cannot be escaped reliably
%   from here. The caller passes Windows paths with forward slashes.
arguments
    path (1,1) string
end
if contains(path, ["""", "$", "`", "\", "%", newline, char(13)])
    error("phasecont:build:unsafePath", ...
        "The path %s has a character that the wsl.exe command cannot pass " + ...
        "safely (one of "" $ ` \ %% or a line break). Move the checkout or " + ...
        "the stage folder to a path without it.", path);
end
value = path;
end
