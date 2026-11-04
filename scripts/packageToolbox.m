function outputFile = packageToolbox(opts)
% Stage the toolbox files and build a .mltbx with matlab.addons.toolbox.
%
%   outputFile = packageToolbox(Root = r, StageFolder = s, OutputFile = o, ...
%       Version = "1.0.0", Identifier = uuid, BinFolder = b, ...
%       Platforms = ["win64" "glnxa64" "maca64"], LicensesFile = l)
%
% Every input is required. buildfile.m calls this from the package task, and
% tests/PackageTest.m calls it with temporary folders.
%
%   Root          repo root (holds toolbox/, LICENSE, THIRD_PARTY_NOTICES.md)
%   StageFolder   folder to build the package tree in; deleted first. It must
%                 be an absolute path directly inside a folder named "package".
%   OutputFile    the .mltbx to write
%   Version       dotted number such as "1.0.0"
%   Identifier    fixed toolbox UUID
%   BinFolder     folder with one subfolder per platform, each holding the
%                 MEX file and the pckern library for that platform:
%                   win64/pckern.mexw64      win64/pckern.dll
%                   glnxa64/pckern.mexa64    glnxa64/libpckern.so
%                   maca64/pckern.mexmaca64  maca64/libpckern.dylib
%   Platforms     the platforms to ship, a non-empty string array drawn from
%                 "win64", "glnxa64" and "maca64". A release ships all three;
%                 a local build ships the one it can make.
%   LicensesFile  text file written into the package as THIRD_PARTY_LICENSES.txt
%
% Each platform goes to bin/<platform> in the package and on the toolbox
% path. MATLAB loads only the MEX file with its own extension, and the
% gateway loads the library from the folder of its MEX file.
arguments
    opts.Root
    opts.StageFolder
    opts.OutputFile
    opts.Version
    opts.Identifier
    opts.BinFolder
    opts.Platforms
    opts.LicensesFile
end

% Every input is required text. Check them all before anything is
% validated, created or deleted.
required = ["Root", "StageFolder", "OutputFile", "Version", "Identifier", ...
    "BinFolder", "LicensesFile"];
absent = strings(1, 0);
for name = required
    if ~isfield(opts, name) || ~isTextValue(opts.(name))
        absent(end+1) = name; %#ok<AGROW>
    else
        opts.(name) = string(opts.(name));
    end
end
if ~isfield(opts, "Platforms") || ~isPlatformList(opts.Platforms)
    absent(end+1) = "Platforms";
end
if ~isempty(absent)
    error("phasecont:package:missingInput", ...
        "Missing required input(s), each must be non-empty text: %s", ...
        strjoin(absent, ", "));
end
platforms = orderedPlatforms(string(opts.Platforms));

if isempty(regexp(opts.Version, ...
        '^(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})$', "once"))
    error("phasecont:package:badVersion", ...
        "Version must be a dotted number such as 1.0.0, got '%s'", opts.Version);
end
if isempty(regexp(opts.Identifier, ...
        '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', "once"))
    error("phasecont:package:badIdentifier", ...
        "Identifier must be a lower-case UUID, got '%s'", opts.Identifier);
end

stage = char(opts.StageFolder);
checkStage(stage);
stagePath = normalizedPath(stage, "phasecont:package:badStage");
if isUnder(normalizedPath(pwd, "phasecont:package:badStage"), stagePath)
    error("phasecont:package:badStage", ...
        "The current folder is inside the stage folder, which is deleted first: %s", pwd);
end
outPath = checkOutput(opts.OutputFile, stagePath);
rootPath = normalizedPath(opts.Root, "phasecont:package:missingFile");
binPath = normalizedPath(opts.BinFolder, "phasecont:package:missingFile");
licPath = normalizedPath(opts.LicensesFile, "phasecont:package:missingFile");
for pair = {"Root", rootPath; "BinFolder", binPath; "LicensesFile", licPath}'
    if isUnder(pair{2}, stagePath)
        error("phasecont:package:inputInsideStage", ...
            "%s lies inside the stage folder, which is deleted first: %s", ...
            pair{1}, pair{2});
    end
end
if ~isfolder(fullfile(opts.Root, "toolbox"))
    error("phasecont:package:missingFile", ...
        "Folder not found: %s", fullfile(opts.Root, "toolbox"));
end
if ~isfolder(fullfile(opts.Root, "toolbox", "examples"))
    error("phasecont:package:missingFile", ...
        "Folder not found: %s", fullfile(opts.Root, "toolbox", "examples"));
end

% Check every input before deleting or writing anything.
binaries = platformBinaries(opts.BinFolder, platforms);
for k = 1:numel(binaries)
    if ~isfile(binaries(k))
        error("phasecont:package:missingBinary", ...
            "Binary not found: %s", binaries(k));
    end
end
if ~isfile(opts.LicensesFile) || strlength(strtrim(string(fileread(opts.LicensesFile)))) == 0
    error("phasecont:package:licenses", ...
        "THIRD_PARTY_LICENSES.txt source is missing or empty: %s", opts.LicensesFile);
end
for name = ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    if ~isfile(fullfile(opts.Root, name))
        error("phasecont:package:missingFile", ...
            "File not found: %s", fullfile(opts.Root, name));
    end
end

% Make sure the output folder exists now, so that a failure here cannot
% happen after the stage has been deleted.
outFolder = fileparts(outPath);
if ~isfolder(outFolder)
    [made, msg] = mkdir(outFolder);
    if ~made
        error("phasecont:package:badOutput", ...
            "Cannot create the output folder %s: %s", outFolder, msg);
    end
end

if isfolder(stage)
    rmdir(stage, "s");
end
mkdir(stage);

% toolbox/ goes to the stage root. toolbox/bin is skipped: binaries come
% from BinFolder only.
entries = dir(fullfile(opts.Root, "toolbox"));
for e = entries(:)'
    if startsWith(e.name, ".") || e.name == "bin"
        continue
    end
    copyfile(fullfile(e.folder, e.name), fullfile(stage, e.name));
end

% Dotfiles such as .gitkeep stay out of the package, at any depth.
hidden = dir(fullfile(stage, "**", ".*"));
for h = hidden(~[hidden.isdir])'
    delete(fullfile(h.folder, h.name));
end

binStages = strings(1, numel(platforms));
for p = 1:numel(platforms)
    binStages(p) = fullfile(stage, "bin", platforms(p));
    mkdir(binStages(p));
    for f = platformBinaries(opts.BinFolder, platforms(p))
        copyfile(f, binStages(p));
    end
end
copyfile(fullfile(opts.Root, "LICENSE"), stage);
copyfile(fullfile(opts.Root, "THIRD_PARTY_NOTICES.md"), stage);
copyfile(opts.LicensesFile, fullfile(stage, "THIRD_PARTY_LICENSES.txt"));

o = matlab.addons.toolbox.ToolboxOptions(stage, opts.Identifier);
o.ToolboxName = "PhaseCont";
o.ToolboxVersion = opts.Version;
o.AuthorName = "Nubayat";
o.Summary = "Equilibrium continuation and bifurcation detection for XPP .ode models.";
o.Description = "PhaseCont reads XPP and XPPAUT .ode files and follows equilibria " + ...
    "from MATLAB, detecting saddle-node and Hopf points. The continuation engine " + ...
    "is written in Rust and compiles your model to native code inside MATLAB, " + ...
    "so no compiler is needed. This package runs on " + platformText(platforms) + ".";
files = dir(fullfile(stage, "**", "*"));
files = files(~[files.isdir]);
o.ToolboxFiles = string(fullfile({files.folder}, {files.name}))';
o.ToolboxMatlabPath = [string(stage), ...
    string(fullfile(stage, "examples")), binStages];
o.SupportedPlatforms.Win64 = any(platforms == "win64");
o.SupportedPlatforms.Glnxa64 = any(platforms == "glnxa64");
o.SupportedPlatforms.Mac = any(platforms == "maca64");
o.SupportedPlatforms.MatlabOnline = false;
o.MinimumMatlabRelease = "R2026b";
o.OutputFile = outPath;

matlab.addons.toolbox.packageToolbox(o);
outputFile = string(outPath);
end

function files = platformBinaries(binFolder, platforms)
% MEX file and library for each platform, under BinFolder/<platform>.
files = strings(1, 0);
for p = platforms
    switch p
        case "win64"
            names = ["pckern.mexw64", "pckern.dll"];
        case "glnxa64"
            names = ["pckern.mexa64", "libpckern.so"];
        case "maca64"
            names = ["pckern.mexmaca64", "libpckern.dylib"];
    end
    files = [files, string(fullfile(binFolder, p, names))]; %#ok<AGROW>
end
end

function tf = isPlatformList(v)
% True for a non-empty list of distinct known platform names.
known = ["win64", "glnxa64", "maca64"];
if iscellstr(v) || ischar(v)
    v = string(v);
end
tf = isstring(v) && ~isempty(v) && isvector(v) && ~any(ismissing(v)) ...
    && all(ismember(v, known)) && numel(unique(v)) == numel(v);
end

function p = orderedPlatforms(v)
% Platforms in a fixed order, so the package does not depend on input order.
known = ["win64", "glnxa64", "maca64"];
p = known(ismember(known, v));
end

function text = platformText(platforms)
% "Windows, Linux and macOS (Apple silicon)" for the platforms given.
names = dictionary(["win64", "glnxa64", "maca64"], ...
    ["Windows", "Linux", "macOS on Apple silicon"]);
words = names(platforms);
if isscalar(words)
    text = words;
else
    text = strjoin(words(1:end-1), ", ") + " and " + words(end);
end
end

function tf = isTextValue(v)
% True for a non-empty, non-missing text scalar (string or char row).
if isstring(v) && isscalar(v)
    tf = ~ismissing(v) && strlength(strtrim(v)) > 0;
elseif ischar(v) && isrow(v)
    tf = ~isempty(strtrim(v));
else
    tf = false;
end
end

function checkSegments(parts, what, shown)
% Reject segments that Windows would read differently from what they say.
reserved = ["CON", "PRN", "AUX", "NUL", "COM" + (1:9), "LPT" + (1:9)];
for k = 1:numel(parts)
    seg = parts(k);
    bad = strlength(strtrim(seg)) == 0 || seg == "." || seg == ".." ...
        || endsWith(seg, [" ", "."]) || contains(seg, ["<", ">", """", "|", "*"]) ...
        || any(double(char(seg)) < 32) || (k > 1 && contains(seg, [":", "?"])) ...
        || any(upper(extractBefore(seg + ".", ".")) == reserved) ...
        || (what == "Stage" && (contains(seg, ["&", ";"]) ...
        || (k == numel(parts) && startsWith(seg, "."))));
    if bad
        error("phasecont:package:bad" + what, ...
            "Refusing '%s': the segment '%s' is empty, relative, ends in a dot or space, is a reserved device name, or has a character that is not allowed here", ...
            shown, seg);
    end
end
end

function checkStage(stage)
% Refuse any stage folder that is not an absolute path of the form
% <something>/package/<name>, with no suspicious segments, so that a typo
% cannot make the rmdir below delete real files.
parts = split(string(stage), ["\", "/"]);
isAbsolute = ~isempty(regexp(stage, '^([A-Za-z]:[\\/]|[\\/]{2})', "once"));
if isAbsolute && numel(parts) >= 2 && parts(end) ~= "" && parts(end-1) == "package"
    % Empty segments come from doubled separators or the UNC prefix.
    checkSegments(parts(parts ~= ""), "Stage", stage);
else
    error("phasecont:package:badStage", ...
        "Refusing to use '%s' as the stage folder: it must be an absolute path directly inside a folder named package", ...
        stage);
end
end

function outPath = checkOutput(output, stagePath)
% Validate OutputFile and return its absolute, normalized path.
outPath = normalizedPath(output, "phasecont:package:badOutput");
parts = split(string(outPath), "\");
leaf = parts(end);
if ~endsWith(leaf, ".mltbx", IgnoreCase = true) || strlength(leaf) <= strlength(".mltbx")
    error("phasecont:package:badOutput", ...
        "OutputFile must be a name ending in .mltbx, got '%s'", output);
end
checkSegments(parts(parts ~= ""), "Output", output);
if isfolder(outPath) || isUnder(outPath, stagePath)
    error("phasecont:package:badOutput", ...
        "OutputFile is a folder or lies at or inside the stage folder: %s", output);
end
if isfile(outPath)
    [~, attrs] = fileattrib(outPath);
    if ~attrs.UserWrite
        error("phasecont:package:badOutput", ...
            "OutputFile exists and is read-only: %s", output);
    end
end
end

function p = normalizedPath(p, id)
% Absolute path with "." and ".." resolved, "\" separators and no trailing
% separator. Pure string work: the folder need not exist. A path must start
% with a drive letter and a separator, with two separators (UNC), or be
% relative to the current folder. A single leading separator or "C:x" is
% refused with the error id given in id.
p = char(p);
if isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/]{2})', "once"))
    if ~isempty(regexp(p, '^([\\/]|[A-Za-z]:)', "once"))
        error(id, ...
            "Path '%s' must start with a drive letter and a separator, with a UNC prefix, or be relative to the current folder", p);
    end
    p = fullfile(pwd, p);
end
unc = ~isempty(regexp(p, '^[\\/]{2}', "once"));
parts = split(string(p), ["\", "/"]);
kept = strings(1, 0);
for part = parts(:)'
    if part == "" || part == "."
        continue
    elseif part == ".."
        if ~isempty(kept)
            kept(end) = [];
        end
    else
        kept(end+1) = part; %#ok<AGROW>
    end
end
p = char(strjoin(kept, "\"));
if unc
    p = ['\\' p];
end
end

function tf = isUnder(p, folder)
% True when p equals folder or lies below it (case-insensitive).
p = lower(string(p));
folder = lower(string(folder));
tf = p == folder || startsWith(p, folder + "\");
end
