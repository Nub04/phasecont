function plan = buildfile
% Build plan for PhaseCont.
%
% In MATLAB, from the repo root:  buildtool <task>
% From WSL:                       scripts/matlab-win "buildtool <task>"
%
% "buildtool -tasks" lists the tasks. The cargo and mex tasks write to the
% folder that tests/+fixtures/binFolder.m names. The package task builds
% build/PhaseCont-<version>.mltbx from those binaries for this platform, or
% for win64, glnxa64 and maca64 when PHASECONT_BINROOT is set (see packageTask).

plan = buildplan(localfunctions);

plan("test") = matlab.buildtool.tasks.TestTask("tests", ...
    Description = "Run the matlab.unittest suites in tests/", ...
    TestResults = fullfile("build", "test-results", "junit.xml"));

plan("mex").Dependencies = "cargo";
plan("package").Dependencies = ["mex", "test"];
plan.DefaultTasks = "test";
end

function cargoTask(context)
% Build pckern for this MATLAB and put the library in the binary folder.
%   PHASECONT_PREBUILT set:  copy the library from that folder.
%   Windows MATLAB, repo under \\wsl.localhost\:  run scripts/build-pckern in WSL.
%   Linux and macOS:  cargo build in the repo, then copy the library.
root = context.Plan.RootFolder;
cleanup = addTestsToPath(root); %#ok<NASGU>
bin = fixtures.binFolder(root);
if ~isfolder(bin)
    mkdir(bin);
end
lib = libraryFileName();

prebuilt = string(getenv("PHASECONT_PREBUILT"));
if strlength(prebuilt) > 0
    src = fullfile(prebuilt, lib);
    if ~isfile(src)
        error("phasecont:build:prebuilt", ...
            "Prebuilt library not found: %s", src);
    end
    copyfile(src, bin);
    fprintf("Copied %s to %s\n", src, bin);
elseif ispc && isWslPath(root)
    [distro, linuxPath] = splitWslPath(root);
    % wsl.exe drops backslashes inside the quoted argument, so pass the
    % Windows path with forward slashes (wslpath -u accepts C:/Users/...).
    % It also takes a quoted -d value literally, so the distro name (which
    % has no spaces) is left unquoted.
    cmd = sprintf('wsl.exe -d %s --cd "%s" -- bash scripts/build-pckern win64 "%s"', ...
        distro, linuxPath, strrep(bin, "\", "/"));
    status = runFromLocalFolder(cmd);
    if status ~= 0
        error("phasecont:build:cargo", ...
            "scripts/build-pckern failed (exit %d); see the output above", status);
    end
elseif ispc
    error("phasecont:build:cargo", ...
        "Set PHASECONT_PREBUILT to a folder with pckern.dll, or build from a WSL checkout.");
else
    cmd = sprintf('cd "%s" && cargo build --release --locked -p pc-ffi', root);
    status = system(cmd, "-echo");
    if status ~= 0
        error("phasecont:build:cargo", ...
            "cargo build failed (exit %d); see the output above", status);
    end
    built = fullfile(root, "target", "release", lib);
    copyfile(built, bin);
    fprintf("Copied %s to %s\n", built, bin);
end
end

function mexTask(context)
% Build the MEX gateway from mex/pckern_gateway.c into the binary folder.
% mex works when the current folder is \\wsl.localhost\... (spike S1 checked
% this), but Windows MATLAB should load binaries from C:, so the output goes
% to the binary folder next to pckern.dll.
root = context.Plan.RootFolder;
src = fullfile(root, "mex", "pckern_gateway.c");
if ~isfile(src)
    error("phasecont:build:noGateway", ...
        "mex/pckern_gateway.c does not exist yet");
end
cleanup = addTestsToPath(root); %#ok<NASGU>
bin = fixtures.binFolder(root);
if ~isfolder(bin)
    mkdir(bin);
end
inc = fullfile(root, "crates", "pc-ffi", "include");
args = {"-R2018a", "-I" + inc, "-outdir", char(bin), "-output", "pckern", src};
if isunix && ~ismac
    args{end+1} = "-ldl";
end
mex(args{:});
end

function packageTask(context)
% Package the toolbox as build/PhaseCont-<version>.mltbx.
% scripts/packageToolbox.m does the staging and packaging; this task gathers
% the inputs: the binaries and a THIRD_PARTY_LICENSES.txt (made by pc-cli
% licenses, or with PHASECONT_PREBUILT set, the file PHASECONT_LICENSES names).
%   PHASECONT_BINROOT set:  a folder holding win64/, glnxa64/ and maca64/,
%       each with the MEX file and the library (the release job sets it);
%       the package covers all three platforms.
%   Otherwise:  the package covers this platform only and takes the outputs
%       of the cargo and mex tasks.
root = context.Plan.RootFolder;
oldPath = path;
restorePath = onCleanup(@() path(oldPath));
addpath(fullfile(root, "tests"), fullfile(root, "scripts"));

binRoot = string(getenv("PHASECONT_BINROOT"));
if strlength(binRoot) > 0
    platforms = ["win64", "glnxa64", "maca64"];
else
    platforms = string(computer("arch"));
    if ~any(platforms == ["win64", "glnxa64", "maca64"])
        error("phasecont:package:platform", ...
            "PhaseCont packages win64, glnxa64 and maca64; this MATLAB is %s.", platforms);
    end
    binRoot = localBinRoot(root, platforms);
end

pkgVersion = toolboxVersion();
packageToolbox( ...
    Root = root, ...
    StageFolder = fullfile(root, "build", "package", "PhaseCont"), ...
    OutputFile = fullfile(root, "build", "PhaseCont-" + pkgVersion + ".mltbx"), ...
    Version = pkgVersion, ...
    Identifier = toolboxUuid(), ...
    BinFolder = binRoot, ...
    Platforms = platforms, ...
    LicensesFile = licensesFile(root, platforms));
end

function binRoot = localBinRoot(root, arch)
% Copy this platform's MEX file and library from the binary folder into
% build/binroot/<arch>/, the layout packageToolbox expects, and return
% build/binroot. A missing file is left for packageToolbox to report.
binRoot = fullfile(root, "build", "binroot");
dest = fullfile(binRoot, arch);
if isfolder(binRoot)
    rmdir(binRoot, "s");
end
mkdir(dest);
bin = fixtures.binFolder(root);
for f = ["pckern." + mexext, libraryFileName()]
    src = fullfile(bin, f);
    if isfile(src)
        copyfile(src, dest);
    end
end
end

function file = licensesFile(root, platforms)
% Path of the third-party license text to ship: the file that
% PHASECONT_LICENSES names (the release job and PHASECONT_PREBUILT builds set
% it), or else the output of pc-cli licenses for the Rust targets of the
% packaged platforms.
file = string(getenv("PHASECONT_LICENSES"));
if strlength(file) > 0 || strlength(string(getenv("PHASECONT_PREBUILT"))) > 0
    if strlength(file) == 0 || ~isfile(file) || dir(file).bytes == 0
        error("phasecont:package:licenses", ...
            "Set PHASECONT_LICENSES to a non-empty THIRD_PARTY_LICENSES.txt (the output of pc-cli licenses). It is required when PHASECONT_PREBUILT is set.");
    end
    return
end
outDir = fullfile(root, "build");
if ~isfolder(outDir)
    mkdir(outDir);
end
file = fullfile(outDir, "THIRD_PARTY_LICENSES.txt");
% Remove any old output so a failed run cannot leave a stale file in use.
if isfile(file)
    delete(file);
end
triples = dictionary(["win64", "glnxa64", "maca64"], ...
    ["x86_64-pc-windows-gnu", "x86_64-unknown-linux-gnu", "aarch64-apple-darwin"]);
args = "cargo run -q --locked -p pc-cli -- licenses" + ...
    join(" --target " + triples(platforms), "") + ...
    " --out build/THIRD_PARTY_LICENSES.txt --overrides licenses/overrides";
if ispc && isWslPath(root)
    [distro, linuxPath] = splitWslPath(root);
    cmd = sprintf('wsl.exe -d %s --cd "%s" -- %s', distro, linuxPath, args);
    status = runFromLocalFolder(cmd);
else
    if ispc
        cdCmd = "cd /d";
    else
        cdCmd = "cd";
    end
    status = system(sprintf('%s "%s" && %s', cdCmd, root, args), "-echo");
end
if status ~= 0
    error("phasecont:package:licenses", ...
        "pc-cli licenses failed (exit %d); see the output above", status);
end
if ~isfile(file) || dir(file).bytes == 0
    error("phasecont:package:licenses", ...
        "pc-cli licenses exited 0 but %s is missing or empty", file);
end
end

function v = toolboxVersion
% Toolbox version, used for the package metadata and the .mltbx file name.
v = "1.0.0";
end

function id = toolboxUuid
% Fixed toolbox identifier, generated once with uuidgen. Every release must
% use the same value so that installing a new .mltbx replaces the old one.
id = "1abed604-32e4-4ec7-9cea-49389af763ff";
end

function cleanup = addTestsToPath(root)
% Put tests/ on the path (for +fixtures) until the returned object is cleared.
oldPath = path;
addpath(fullfile(root, "tests"));
cleanup = onCleanup(@() path(oldPath));
end

function name = libraryFileName
% File name of the pckern library on this platform
switch computer("arch")
    case "win64"
        name = "pckern.dll";
    case "glnxa64"
        name = "libpckern.so";
    otherwise
        name = "libpckern.dylib";
end
end

function tf = isWslPath(folder)
tf = startsWith(string(folder), ["\\wsl.localhost\" "\\wsl$\"], IgnoreCase = true);
end

function [distro, linuxPath] = splitWslPath(folder)
% \\wsl.localhost\<distro>\home\u\repo gives <distro> and /home/u/repo
parts = split(extractAfter(string(folder), 2), "\");
distro = parts(2);
linuxPath = "/" + strjoin(parts(3:end), "/");
end

function status = runFromLocalFolder(cmd)
% cmd.exe cannot start in a \\wsl.localhost\ folder, so run from tempdir.
oldFolder = pwd;
cd(tempdir);
restore = onCleanup(@() cd(oldFolder));
status = system(cmd, "-echo");
clear restore
end
