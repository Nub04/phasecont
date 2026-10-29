function plan = buildfile
% Build plan for PhaseCont.
%
% In MATLAB, from the repo root:  buildtool <task>
% From WSL:                       scripts/matlab-win "buildtool <task>"
%
% "buildtool -tasks" lists the tasks. The cargo and mex tasks write to the
% folder that tests/+fixtures/binFolder.m names. The package task builds
% build/PhaseCont-0.3.0.mltbx from those binaries (win64 only, Windows MATLAB);
% it needs the cargo and mex outputs and a THIRD_PARTY_LICENSES.txt (see packageTask).

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
% Package the toolbox as build/PhaseCont-<version>.mltbx (win64 only in v0.3).
% scripts/packageToolbox.m does the staging and packaging; this task gathers
% the inputs. It needs the binaries from the cargo and mex tasks, and a
% THIRD_PARTY_LICENSES.txt: made by pc-cli licenses, or with PHASECONT_PREBUILT
% set, the file that PHASECONT_LICENSES names.
root = context.Plan.RootFolder;
if computer("arch") ~= "win64"
    error("phasecont:package:platform", ...
        "v0.3 packages win64 only; run the package task in Windows MATLAB (this is %s).", ...
        computer("arch"));
end
oldPath = path;
restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(fullfile(root, "tests"), fullfile(root, "scripts"));

pkgVersion = toolboxVersion();
packageToolbox( ...
    Root = root, ...
    StageFolder = fullfile(root, "build", "package", "PhaseCont"), ...
    OutputFile = fullfile(root, "build", "PhaseCont-" + pkgVersion + ".mltbx"), ...
    Version = pkgVersion, ...
    Identifier = toolboxUuid(), ...
    BinFolder = fixtures.binFolder(root), ...
    LicensesFile = licensesFile(root));
end

function file = licensesFile(root)
% Path of the third-party license text to ship. With PHASECONT_PREBUILT set it
% is the file named by PHASECONT_LICENSES; otherwise pc-cli licenses writes it.
if strlength(string(getenv("PHASECONT_PREBUILT"))) > 0
    file = string(getenv("PHASECONT_LICENSES"));
    if strlength(file) == 0 || ~isfile(file) || dir(file).bytes == 0
        error("phasecont:package:licenses", ...
            "With PHASECONT_PREBUILT set, set PHASECONT_LICENSES to a non-empty THIRD_PARTY_LICENSES.txt (the output of pc-cli licenses).");
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
args = "cargo run -q --locked -p pc-cli -- licenses --target x86_64-pc-windows-gnu" + ...
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
v = "0.3.0";
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
