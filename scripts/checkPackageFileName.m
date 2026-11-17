function platform = checkPackageFileName(name, version, arch)
% Check that a package file name, as packageFileName writes it, has this
% version and can install on the platform arch. Returns the platform in the
% name, or "" for the package that holds all three platforms.
%
% Raises phasecont:install:fileName for a name packageFileName does not
% write, phasecont:install:version for another version and
% phasecont:install:platform for a package of another platform.
arguments
    name (1,1) string
    version (1,1) string
    arch (1,1) string
end
tok = regexp(name, '^PhaseCont-([0-9]+\.[0-9]+\.[0-9]+)(-win64|-glnxa64|-maca64)?\.mltbx$', ...
    "tokens", "once");
if isempty(tok)
    error("phasecont:install:fileName", ...
        "%s is not a package file name (PhaseCont-<version>[-<platform>].mltbx).", name);
end
if tok(1) ~= version
    error("phasecont:install:version", ...
        "File name %s has version %s but buildfile.m says %s", name, tok(1), version);
end
platform = "";
if numel(tok) > 1 && strlength(tok(2)) > 0
    platform = extractAfter(tok(2), 1);
end
if platform ~= "" && platform ~= arch
    error("phasecont:install:platform", ...
        "%s is the %s package and cannot install on %s.", name, platform, arch);
end
end
