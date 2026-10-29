function outputFile = packageToolbox(opts)
% Stage the toolbox files and build a .mltbx with matlab.addons.toolbox.
%
%   outputFile = packageToolbox(Root = r, StageFolder = s, OutputFile = o, ...
%       Version = "0.3.0", Identifier = uuid, BinFolder = b, LicensesFile = l)
%
% Every input is required. buildfile.m calls this from the package task, and
% tests/PackageTest.m calls it with temporary folders.
%
%   Root          repo root (holds toolbox/, LICENSE, THIRD_PARTY_NOTICES.md)
%   StageFolder   folder to build the package tree in; deleted first. It must
%                 be a direct child of a folder named "package".
%   OutputFile    the .mltbx to write
%   Version       dotted number such as "0.3.0"
%   Identifier    fixed toolbox UUID
%   BinFolder     folder with pckern.mexw64 and pckern.dll
%   LicensesFile  text file written into the package as THIRD_PARTY_LICENSES.txt
%
% v0.3 supports win64 only. To add a platform (P8), extend platformBinaries
% and the SupportedPlatforms and path lists below.
arguments
    opts.Root (1,1) string
    opts.StageFolder (1,1) string
    opts.OutputFile (1,1) string
    opts.Version (1,1) string
    opts.Identifier (1,1) string
    opts.BinFolder (1,1) string
    opts.LicensesFile (1,1) string
end

% Every input is required. Check them all before anything is validated or deleted.
required = ["Root", "StageFolder", "OutputFile", "Version", "Identifier", ...
    "BinFolder", "LicensesFile"];
missing = strings(1, 0);
for name = required
    if ~isfield(opts, name) || strlength(strtrim(opts.(name))) == 0
        missing(end+1) = name; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error("phasecont:package:missingInput", ...
        "Missing required input(s): %s", strjoin(missing, ", "));
end

if isempty(regexp(opts.Version, '^\d+\.\d+\.\d+$', "once"))
    error("phasecont:package:badVersion", ...
        "Version must be a dotted number such as 0.3.0, got '%s'", opts.Version);
end
if isempty(regexp(opts.Identifier, ...
        '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', "once"))
    error("phasecont:package:badIdentifier", ...
        "Identifier must be a lower-case UUID, got '%s'", opts.Identifier);
end
if ~endsWith(opts.OutputFile, ".mltbx", IgnoreCase = true)
    error("phasecont:package:badOutput", ...
        "OutputFile must end in .mltbx, got '%s'", opts.OutputFile);
end

stage = char(opts.StageFolder);
checkStage(stage);
if ~isfolder(fullfile(opts.Root, "toolbox"))
    error("phasecont:package:missingFile", ...
        "Folder not found: %s", fullfile(opts.Root, "toolbox"));
end

% Check every input before deleting or writing anything.
binaries = platformBinaries(opts.BinFolder);
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

binStage = fullfile(stage, "bin", "win64");
mkdir(binStage);
for k = 1:numel(binaries)
    copyfile(binaries(k), binStage);
end
copyfile(fullfile(opts.Root, "LICENSE"), stage);
copyfile(fullfile(opts.Root, "THIRD_PARTY_NOTICES.md"), stage);
copyfile(opts.LicensesFile, fullfile(stage, "THIRD_PARTY_LICENSES.txt"));

outFolder = fileparts(opts.OutputFile);
if strlength(outFolder) > 0 && ~isfolder(outFolder)
    mkdir(outFolder);
end

o = matlab.addons.toolbox.ToolboxOptions(stage, opts.Identifier);
o.ToolboxName = "PhaseCont";
o.ToolboxVersion = opts.Version;
o.AuthorName = "Nubayat";
o.Summary = "Equilibrium continuation and bifurcation detection for XPP .ode models.";
o.Description = "PhaseCont reads XPP and XPPAUT .ode files and follows equilibria " + ...
    "from MATLAB, detecting saddle-node and Hopf points. The continuation engine " + ...
    "is written in Rust and compiles your model to native code inside MATLAB, " + ...
    "so no compiler is needed. This release runs on Windows only.";
files = dir(fullfile(stage, "**", "*"));
files = files(~[files.isdir]);
o.ToolboxFiles = string(fullfile({files.folder}, {files.name}))';
o.ToolboxMatlabPath = [string(stage), ...
    string(fullfile(stage, "examples")), string(binStage)];
o.SupportedPlatforms.Win64 = true;
o.SupportedPlatforms.Glnxa64 = false;
o.SupportedPlatforms.Mac = false;
o.SupportedPlatforms.MatlabOnline = false;
o.MinimumMatlabRelease = "R2026b";
o.OutputFile = opts.OutputFile;

matlab.addons.toolbox.packageToolbox(o);
outputFile = opts.OutputFile;
end

function files = platformBinaries(binFolder)
% Binaries that go in bin/win64. Add other platforms here in P8.
files = [fullfile(binFolder, "pckern.mexw64"), fullfile(binFolder, "pckern.dll")];
end

function checkStage(stage)
% Refuse any stage folder that is not an absolute path of the form
% <something>/package/<name>, with no "." or ".." segments, so that a typo
% cannot make the rmdir below delete real files.
parts = split(string(stage), ["\", "/"]);
isAbsolute = ~isempty(regexp(stage, '^([A-Za-z]:[\\/]|\\\\|/)', "once"));
ok = isAbsolute && numel(parts) >= 2 && strlength(parts(end)) > 0 ...
    && ~any(parts == "." | parts == "..") && parts(end-1) == "package";
if ~ok
    error("phasecont:package:badStage", ...
        "Refusing to use '%s' as the stage folder: it must be an absolute path directly inside a folder named package, with no . or .. segments", ...
        stage);
end
end
