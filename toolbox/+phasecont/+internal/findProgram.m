function path = findProgram(name)
%FINDPROGRAM Absolute path of a program on PATH, never from the current folder.
%   P = phasecont.internal.findProgram(NAME) looks for the program NAME in
%   the folders on the PATH environment variable and returns the first
%   match as an absolute path, or "" when there is none. On Windows it tries
%   NAME with each extension in PATHEXT (".COM .EXE .BAT .CMD" when PATHEXT
%   is empty), and NAME itself when it already has an extension. Elsewhere a
%   match must be a file the user may execute.
%
%   NAME that holds a / or \ is a path the caller chose. It comes back
%   unchanged when that file exists (on Windows also with a PATHEXT
%   extension), else "".
%
%   The current folder is not searched, and neither is a PATH entry that is
%   empty, "." or any other relative folder, since those name the current
%   folder too. A bare name run through cmd.exe or CreateProcess would look
%   in the current folder first, which inside MATLAB is any folder the user
%   opened, so a file there named like the program would run instead.
arguments
    name (1,1) string {mustBeNonmissing}
end

path = "";
if name == ""
    return
end
if ispc
    exts = split(string(getenv("PATHEXT")), ";");
    exts = exts(exts ~= "");
    if isempty(exts)
        exts = [".COM"; ".EXE"; ".BAT"; ".CMD"];
    end
    [~, ~, own] = fileparts(name);
    if own == ""
        names = name + exts;
    else
        names = [name; name + exts];
    end
    sep = ";";
else
    names = name;
    sep = ":";
end

if contains(name, ["/", "\"])
    for n = reshape(names, 1, [])
        if runnable(n)
            path = n;
            return
        end
    end
    return
end

folders = split(string(getenv("PATH")), sep);
folders = strtrim(erase(folders, """"));
for f = reshape(folders, 1, [])
    if ~isAbsolute(f)
        continue
    end
    for n = reshape(names, 1, [])
        candidate = string(fullfile(f, n));
        if runnable(candidate)
            path = candidate;
            return
        end
    end
end
end

function tf = runnable(file)
tf = isfile(file);
if tf && ~ispc
    [ok, a] = fileattrib(char(file));
    tf = ok && isequal(a.UserExecute, 1);
end
end

function tf = isAbsolute(folder)
if ispc
    % C:\x or C:/x, or a UNC path \\server\share. "C:x" is relative.
    tf = ~isempty(regexp(folder, "^([A-Za-z]:[\\/]|\\\\[^\\])", "once"));
else
    tf = startsWith(folder, "/");
end
end
