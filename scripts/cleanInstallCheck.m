% Install the packaged toolbox into MATLAB, run an example from the installed
% files and uninstall it.
%
%   PHASECONT_MLTBX          path of the .mltbx to check (needed by the
%                            install and remove steps)
%   PHASECONT_INSTALL_STEP   (optional) "install", "example" or "remove" runs
%                            that step alone; empty runs all three. CI runs
%                            each step in its own MATLAB, because a MATLAB
%                            licensed on a GitHub-hosted runner cannot start
%                            a second licensed MATLAB.
%   PHASECONT_CHILD_MATLAB   (optional, all steps only) program that starts
%                            the MATLAB for the example step, called as
%                            PROGRAM "statement". Without it that MATLAB is
%                            started as  matlabroot/bin/matlab -batch "statement".
%
% install  checks the version in the file name, stops if PhaseCont is
%          already installed, installs the .mltbx and checks the version and
%          Guid. It checks that phasecont.Model resolves into the add-on
%          folder and not the repo, that THIRD_PARTY_LICENSES.txt is there,
%          and that the root, bin/<arch> and examples folders are on the
%          path. It writes those folders to a state file in tempdir for the
%          later steps. It does not load pckern.
% example  reports whether this MATLAB lists the install and found
%          phasecont.Model on its path when it started, adds the installed
%          folders with addpath either way, and runs the FitzHugh-Nagumo
%          example, which must find 2 Hopf points.
% remove   uninstalls the toolbox by Guid and checks that it is gone from the
%          list of installed toolboxes and from disk. An install made by a
%          MATLAB started with -batch is not registered for later MATLAB
%          sessions, though its files stay on disk (seen on R2026b), so when
%          the toolbox is not listed but its folder exists, this step installs
%          the same .mltbx again first.
%
% The example runs in a MATLAB process that did not install the toolbox. The
% gateway never unloads pckern (worker threads may still run its code), so on
% Windows a MATLAB that has loaded it keeps the file locked and the uninstall
% could not delete it.
%
% Run it from a folder outside the repo (for example tempdir) under
% matlab -batch, so the repo toolbox is not on the path. Any error makes the
% exit code non-zero. A failure in the install step, or in the example while
% all steps run, uninstalls the toolbox again, but only a copy this script
% installed: if PhaseCont is already installed the install step stops and
% prints the command to remove it.

cleanInstallMain(string(getenv("PHASECONT_INSTALL_STEP")), ...
    string(fileparts(mfilename("fullpath"))));

function cleanInstallMain(step, scriptsFolder)
if ~contains(string(path), scriptsFolder)
    addpath(scriptsFolder);   % buildfileValue must not depend on the current folder
    restore = onCleanup(@() rmpath(scriptsFolder));
end
cfg.RepoRoot = string(fileparts(scriptsFolder));
cfg.ScriptFile = fullfile(scriptsFolder, "cleanInstallCheck.m");
buildfile = fullfile(cfg.RepoRoot, "buildfile.m");
cfg.Version = buildfileValue("toolboxVersion", buildfile);
cfg.Guid = buildfileValue("toolboxUuid", buildfile);
cfg.StateFile = string(fullfile(tempdir, "phasecont-install-check.txt"));
clear restore

switch step
    case "install"
        installStep(cfg);
    case "example"
        exampleStep(cfg);
    case "remove"
        removeStep(cfg);
    case ""
        installStep(cfg);
        try
            exampleInNewMatlab(cfg);
        catch err
            removeAfterFailure(cfg.Guid);
            rethrow(err);
        end
        removeStep(cfg);
    otherwise
        error("phasecont:install:step", ...
            "PHASECONT_INSTALL_STEP must be install, example, remove or empty, got '%s'", step);
end
end

function installStep(cfg)
file = mltbxFile();
[~, base, ext] = fileparts(file);
name = string(base) + string(ext);
tok = regexp(name, "^PhaseCont-(.+)\.mltbx$", "tokens", "once");
if ~isempty(tok) && tok{1} ~= cfg.Version
    error("phasecont:install:version", ...
        "File name %s has version %s but buildfile.m says %s", name, tok{1}, cfg.Version);
end

before = matlab.addons.toolbox.installedToolboxes;
if ~isempty(before)
    hit = before(strcmpi({before.Guid}, cfg.Guid) | strcmp({before.Name}, "PhaseCont"));
    if ~isempty(hit)
        error("phasecont:install:alreadyInstalled", ...
            "PhaseCont %s is already installed. It may be deliberate, so this script leaves it alone. To remove it, run:  t = matlab.addons.toolbox.installedToolboxes; matlab.addons.toolbox.uninstallToolbox(t(strcmp({t.Guid}, '%s')))", ...
            hit(1).Version, hit(1).Guid);
    end
end
if ~isempty(which("phasecont.Model"))
    error("phasecont:install:onPath", ...
        "phasecont.Model is already on the path (%s). Run this from a folder outside the repo, with the repo toolbox off the path.", ...
        which("phasecont.Model"));
end

fprintf("Installing %s\n", file);
installed = matlab.addons.toolbox.installToolbox(file);
guid = string(installed.Guid);
try
    if string(installed.Version) ~= cfg.Version
        error("phasecont:install:version", "Installed version %s, expected %s", ...
            installed.Version, cfg.Version);
    end
    if lower(guid) ~= lower(cfg.Guid)
        error("phasecont:install:guid", "Installed Guid %s, expected %s", guid, cfg.Guid);
    end
    fprintf("Installed %s %s (%s)\n", installed.Name, installed.Version, guid);

    model = string(which("phasecont.Model"));
    if strlength(model) == 0
        error("phasecont:install:notOnPath", "phasecont.Model is not on the path after the install");
    end
    root = string(fileparts(fileparts(model)));   % parent of +phasecont
    checkAddonRoot(root, cfg.RepoRoot);
    lic = fullfile(root, "THIRD_PARTY_LICENSES.txt");
    if ~isfile(lic) || dir(lic).bytes == 0
        error("phasecont:install:licenses", "Missing or empty: %s", lic);
    end
    fprintf("phasecont.Model: %s\nLicenses: %s\n", model, lic);

    dirs = string(strsplit(path, pathsep));
    dirs = dirs(startsWith(dirs, root));
    archBin = fullfile(root, "bin", computer("arch"));
    if numel(dirs) < 3 || ~any(strcmpi(dirs, archBin))
        error("phasecont:install:path", ...
            "Expected the root, %s and examples on the path, found %d folders under %s", ...
            archBin, numel(dirs), root);
    end
    % Line 1 is the install root, the other lines are its folders on the path.
    writelines([root; dirs(:)], cfg.StateFile);
catch err
    removeAfterFailure(guid);
    rethrow(err);
end
fprintf("Install step passed.\n");
end

function exampleStep(cfg)
% Whether this MATLAB lists the install is only reported, since an install
% made under -batch is not registered for later sessions.
[root, dirs] = readState(cfg);
fprintf("This MATLAB lists PhaseCont (%s) as installed: %s\n", cfg.Guid, ...
    string(~isempty(listed(cfg.Guid))));
cd(tempdir);   % keep the repo scripts folder out of the current folder
model = string(which("phasecont.Model"));
if strlength(model) > 0
    fprintf("This MATLAB had phasecont.Model on its path at startup: %s\n", model);
    checkAddonRoot(string(fileparts(fileparts(model))), cfg.RepoRoot);
else
    fprintf("This MATLAB did not have phasecont.Model on its path at startup.\n");
end
addpath(char(strjoin(dirs, pathsep)));
model = string(which("phasecont.Model"));
if ~startsWith(lower(model), lower(root) + filesep)
    error("phasecont:install:notAddon", ...
        "After addpath, phasecont.Model resolves to %s, outside %s", model, root);
end

branches = fitzhugh_nagumo_example(Plot = false);
hopf = [];
for k = 1:numel(branches)
    hopf = [hopf; branches(k).points("H")]; %#ok<AGROW>
end
if numel(hopf) ~= 2
    error("phasecont:install:hopf", "Expected 2 Hopf points, got %d", numel(hopf));
end
idx = branches(1).FreeIndex;
for k = 1:numel(hopf)
    fprintf("Hopf %d: I = %.8g, omega = %.8g\n", k, hopf(k).P(idx), hopf(k).Data.omega);
end
fprintf("Example step passed.\n");
end

function exampleInNewMatlab(cfg)
% Run the example step in a new MATLAB process, which has not loaded pckern.
cd(tempdir);   % cmd.exe cannot start in a \\wsl.localhost\ folder
statement = sprintf("setenv('PHASECONT_INSTALL_STEP', 'example'); run('%s')", ...
    strrep(cfg.ScriptFile, "'", "''"));
exe = string(getenv("PHASECONT_CHILD_MATLAB"));
if strlength(exe) > 0
    command = sprintf('"%s" "%s"', exe, statement);
else
    command = sprintf('"%s" -batch "%s"', fullfile(matlabroot, "bin", "matlab"), statement);
end
status = system(command, "-echo");
if status ~= 0
    error("phasecont:install:example", "The example failed in the new MATLAB (exit %d)", status);
end
end

function removeStep(cfg)
[root, ~] = readState(cfg);
t = listed(cfg.Guid);
if isempty(t)
    % An install made under -batch is not registered for later sessions,
    % though its files stay on disk (seen on R2026b). Installing the same
    % file again registers it in this MATLAB.
    if ~isfolder(root)
        error("phasecont:install:notInstalled", ...
            "PhaseCont (%s) is not installed and %s does not exist, so there is nothing to remove.", ...
            cfg.Guid, root);
    end
    file = mltbxFile();
    fprintf("This MATLAB does not list PhaseCont, so it installs %s again before the uninstall.\n", file);
    matlab.addons.toolbox.installToolbox(file);
    t = listed(cfg.Guid);
    model = string(which("phasecont.Model"));
    again = string(fileparts(fileparts(model)));
    if isempty(t) || lower(again) ~= lower(root)
        removeAfterFailure(cfg.Guid);
        error("phasecont:install:reinstall", ...
            "Installing again put phasecont.Model at %s, expected it under %s", model, root);
    end
end
matlab.addons.toolbox.uninstallToolbox(t(1));
if ~isempty(listed(cfg.Guid))
    error("phasecont:install:notRemoved", "PhaseCont is still installed after the uninstall");
end
if isfolder(root)
    error("phasecont:install:notRemoved", "The uninstall left the folder %s", root);
end
delete(cfg.StateFile);
fprintf("Clean install check passed; toolbox uninstalled.\n");
end

function checkAddonRoot(root, repoRoot)
% Fail unless root lies in the add-on folder and not in the repo.
if startsWith(lower(root), lower(repoRoot) + filesep)
    error("phasecont:install:notAddon", ...
        "phasecont.Model resolves into the repo (%s), not the add-on folder", root);
end
% The add-on folder setting exists on R2026b; if it does not on another
% release only the repo check above applies.
try
    addonFolder = string(settings().matlab.addons.InstallationFolder.ActiveValue);
catch
    addonFolder = "";
end
if strlength(addonFolder) > 0 && ~startsWith(lower(root), lower(addonFolder))
    error("phasecont:install:notAddon", ...
        "phasecont.Model resolves to %s, outside the add-on folder %s", root, addonFolder);
end
end

function file = mltbxFile()
% The .mltbx named by PHASECONT_MLTBX.
file = string(getenv("PHASECONT_MLTBX"));
if strlength(file) == 0
    error("phasecont:install:noFile", "Set PHASECONT_MLTBX to the path of the .mltbx.");
end
if ~isfile(file)
    error("phasecont:install:noFile", "PHASECONT_MLTBX does not name a file: %s", file);
end
end

function [root, dirs] = readState(cfg)
% Install root and its folders on the path, as the install step wrote them.
if ~isfile(cfg.StateFile)
    error("phasecont:install:state", ...
        "Missing %s, which the install step writes; run it first.", cfg.StateFile);
end
lines = readlines(cfg.StateFile, EmptyLineRule = "skip");
if numel(lines) < 4
    error("phasecont:install:state", "%s holds %d lines, expected at least 4.", ...
        cfg.StateFile, numel(lines));
end
root = lines(1);
dirs = lines(2:end)';
end

function t = listed(guid)
% Installed toolboxes with this Guid.
t = matlab.addons.toolbox.installedToolboxes;
if ~isempty(t)
    t = t(strcmpi({t.Guid}, guid));
end
end

function removeAfterFailure(guid)
% Uninstall the toolbox this script installed, after an error.
try
    t = listed(guid);
    if ~isempty(t)
        matlab.addons.toolbox.uninstallToolbox(t(1));
    end
catch err
    fprintf(2, "Uninstall failed: %s\n", err.message);
end
end
