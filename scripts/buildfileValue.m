function value = buildfileValue(name, buildfile)
% Read the string literal returned by a local function of buildfile.m.
%   value = buildfileValue("toolboxVersion")
%   value = buildfileValue("toolboxUuid", file)
% It matches the function header, any comment lines after it, and the first
% assignment of a double-quoted string, so the literal lives in buildfile.m
% only. It raises phasecont:build:buildfileValue if the pattern is not found.
arguments
    name (1,1) string
    buildfile (1,1) string = fullfile(fileparts(fileparts(mfilename("fullpath"))), "buildfile.m")
end
text = fileread(buildfile);
pattern = "function\s+\w+\s*=\s*" + name + "\s*(?:\r?\n\s*%[^\n]*)*\r?\n\s*\w+\s*=\s*""([^""]+)""\s*;";
tok = regexp(text, pattern, "tokens", "once");
if isempty(tok)
    error("phasecont:build:buildfileValue", ...
        "Could not find the string literal of %s in %s", name, buildfile);
end
value = string(tok{1});
end
