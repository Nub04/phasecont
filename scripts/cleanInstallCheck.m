% Install the packaged toolbox into MATLAB and run an example from it.
%
%   PHASECONT_MLTBX   path of the .mltbx to check (required)
%
% Run it from a folder outside the repo (for example tempdir), so the repo
% toolbox is not on the path, under matlab -batch. Any error makes the exit
% code non-zero. The script uninstalls the toolbox at the end, also after a
% failure, but only a copy that it installed itself: if PhaseCont is already
% installed it stops and prints the command to remove it.

ciScriptsFolder = fileparts(mfilename("fullpath"));
ciExpectedVersion = buildfileValue("toolboxVersion", fullfile(ciScriptsFolder, "..", "buildfile.m"));
ciExpectedGuid = buildfileValue("toolboxUuid", fullfile(ciScriptsFolder, "..", "buildfile.m"));

ciFile = string(getenv("PHASECONT_MLTBX"));
if strlength(ciFile) == 0
    error("phasecont:install:noFile", "Set PHASECONT_MLTBX to the path of the .mltbx.");
end
if ~isfile(ciFile)
    error("phasecont:install:noFile", "PHASECONT_MLTBX does not name a file: %s", ciFile);
end
ciName = extractAfter(string(ciFile), strlength(fileparts(ciFile)) + 1);
ciTok = regexp(ciName, "^PhaseCont-(.+)\.mltbx$", "tokens", "once");
if ~isempty(ciTok) && ciTok{1} ~= ciExpectedVersion
    error("phasecont:install:version", ...
        "File name %s has version %s but buildfile.m says %s", ciName, ciTok{1}, ciExpectedVersion);
end

ciBefore = matlab.addons.toolbox.installedToolboxes;
if ~isempty(ciBefore)
    ciHit = ciBefore(strcmpi({ciBefore.Guid}, ciExpectedGuid) | strcmp({ciBefore.Name}, "PhaseCont"));
    if ~isempty(ciHit)
        error("phasecont:install:alreadyInstalled", ...
            "PhaseCont %s is already installed. It may be deliberate, so this script leaves it alone. To remove it, run:  t = matlab.addons.toolbox.installedToolboxes; matlab.addons.toolbox.uninstallToolbox(t(strcmp({t.Guid}, '%s')))", ...
            ciHit(1).Version, ciHit(1).Guid);
    end
end

if ~isempty(which("phasecont.Model"))
    error("phasecont:install:onPath", ...
        "phasecont.Model is already on the path (%s). Run this from a folder outside the repo, with the repo toolbox off the path.", ...
        which("phasecont.Model"));
end

fprintf("Installing %s\n", ciFile);
ciInstalled = matlab.addons.toolbox.installToolbox(ciFile);
ciGuid = string(ciInstalled.Guid);
ciRemove = onCleanup(@() cleanInstallRemove(ciGuid));

if string(ciInstalled.Version) ~= ciExpectedVersion
    error("phasecont:install:version", "Installed version %s, expected %s", ...
        ciInstalled.Version, ciExpectedVersion);
end
if lower(ciGuid) ~= lower(ciExpectedGuid)
    error("phasecont:install:guid", "Installed Guid %s, expected %s", ciGuid, ciExpectedGuid);
end
fprintf("Installed %s %s (%s)\n", ciInstalled.Name, ciInstalled.Version, ciGuid);

ciModel = string(which("phasecont.Model"));
if strlength(ciModel) == 0
    error("phasecont:install:notOnPath", "phasecont.Model is not on the path after the install");
end
ciRoot = fileparts(fileparts(ciModel));   % parent of +phasecont
ciLic = fullfile(ciRoot, "THIRD_PARTY_LICENSES.txt");
if ~isfile(ciLic) || dir(ciLic).bytes == 0
    error("phasecont:install:licenses", "Missing or empty: %s", ciLic);
end
fprintf("phasecont.Model: %s\nLicenses: %s\n", ciModel, ciLic);

% Run the example in a fresh MATLAB process, as a user would after the install.
% The gateway never unloads pckern.dll (rayon threads may still run its code),
% so a MATLAB that has called into it keeps the file locked and the uninstall
% below could not delete it. This process never loads the DLL.
cd(tempdir);   % cmd.exe cannot start in a \\wsl.localhost\ folder
ciChildFile = fullfile(tempname, "cleanInstallChild.m");
mkdir(fileparts(ciChildFile));
% A new MATLAB process does not list this install yet when started from a
% -batch session, so pass it the folders that the install put on this path.
ciDirs = string(strsplit(path, pathsep));
ciDirs = ciDirs(startsWith(ciDirs, ciRoot));
if numel(ciDirs) < 3
    error("phasecont:install:path", "Expected the root, bin/win64 and examples on the path, found %d folders under %s", ...
        numel(ciDirs), ciRoot);
end
writelines([ ...
    "addpath(" + strjoin("'" + ciDirs + "'", ", ") + ");"
    "branches = fitzhugh_nagumo_example(Plot = false);"
    "hopf = [];"
    "for k = 1:numel(branches)"
    "    hopf = [hopf; branches(k).points(""H"")];"
    "end"
    "if numel(hopf) ~= 2"
    "    error(""phasecont:install:hopf"", ""Expected 2 Hopf points, got %d"", numel(hopf));"
    "end"
    "idx = branches(1).FreeIndex;"
    "for k = 1:numel(hopf)"
    "    fprintf(""Hopf %d: I = %.8g, omega = %.8g\n"", k, hopf(k).P(idx), hopf(k).Data.omega);"
    "end"
    ], ciChildFile);
ciExe = fullfile(matlabroot, "bin", "matlab");
ciStatus = system(sprintf('"%s" -batch "cd(''%s''); run(''%s'')"', ...
    ciExe, fileparts(ciChildFile), ciChildFile), "-echo");
rmdir(fileparts(ciChildFile), "s");
if ciStatus ~= 0
    error("phasecont:install:example", "The example failed in the fresh MATLAB (exit %d)", ciStatus);
end

% Uninstall now (not only at exit), so the checks below see the result.
clear ciRemove
ciAfter = matlab.addons.toolbox.installedToolboxes;
if ~isempty(ciAfter) && any(strcmpi({ciAfter.Guid}, ciExpectedGuid))
    error("phasecont:install:notRemoved", "PhaseCont is still installed after the uninstall");
end
if isfolder(ciRoot)
    error("phasecont:install:notRemoved", "The uninstall left the folder %s", ciRoot);
end
fprintf("Clean install check passed; toolbox uninstalled.\n");

function cleanInstallRemove(guid)
% Uninstall the toolbox this script installed (runs also after an error).
try
    t = matlab.addons.toolbox.installedToolboxes;
    if ~isempty(t)
        t = t(strcmpi({t.Guid}, guid));
    end
    if ~isempty(t)
        matlab.addons.toolbox.uninstallToolbox(t(1));
    end
catch err
    fprintf(2, "Uninstall failed: %s\n", err.message);
end
end
