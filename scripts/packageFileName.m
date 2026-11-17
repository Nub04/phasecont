function name = packageFileName(version, platforms)
% File name of a PhaseCont package: PhaseCont-<version>.mltbx for the
% package that holds win64, glnxa64 and maca64, and
% PhaseCont-<version>-<platform>.mltbx for a package that holds one of them.
%
%   packageFileName("1.0.0", "win64")                         % PhaseCont-1.0.0-win64.mltbx
%   packageFileName("1.0.0", ["win64", "glnxa64", "maca64"])  % PhaseCont-1.0.0.mltbx
%
% Raises phasecont:package:badVersion for a version that is not N.N.N and
% phasecont:package:platforms for any other platform list.
arguments
    version (1,1) string
    platforms (1,:) string
end
if isempty(regexp(version, '^[0-9]+\.[0-9]+\.[0-9]+$', "once"))
    error("phasecont:package:badVersion", ...
        "Version must be N.N.N, got ""%s"".", version);
end
known = ["win64", "glnxa64", "maca64"];
if isscalar(platforms) && any(platforms == known)
    name = "PhaseCont-" + version + "-" + platforms + ".mltbx";
elseif numel(platforms) == numel(known) && all(ismember(known, platforms))
    name = "PhaseCont-" + version + ".mltbx";
else
    error("phasecont:package:platforms", ...
        "A package holds one of win64, glnxa64 and maca64, or all three; got [%s].", ...
        strjoin(platforms, ", "));
end
end
