function plan = buildfile
% Build plan for PhaseCont.
%
% In MATLAB, from the repo root:  buildtool <task>
% From WSL:                       scripts/matlab-win "buildtool <task>"
%
% "buildtool -tasks" lists the tasks. The cargo and mex tasks write to the
% folder that tests/+fixtures/binFolder.m names. The package task stops with
% an error until P8a.

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
args = {"-R2018a", "-outdir", char(bin), "-output", "pckern", src};
if isunix && ~ismac
    args{end+1} = "-ldl";
end
mex(args{:});
end

function packageTask(~)
% Package the toolbox as a .mltbx with matlab.addons.toolbox.ToolboxOptions
error("phasecont:build:notImplemented", ...
    "The package task arrives in P8a. Toolbox UUID: %s", toolboxUuid());
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
