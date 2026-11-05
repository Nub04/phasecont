function paths = latexWrite(folder, basename, tex, files)
%LATEXWRITE Write a rendered document and its data files into a folder.
%   P = phasecont.internal.latexWrite(FOLDER, BASENAME, TEX, FILES) writes
%   FOLDER/BASENAME.tex and each element of FILES (fields Name and
%   Content) into FOLDER, which is created when it is missing. The files
%   are UTF-8. P lists the written paths, the .tex file first.
%
%   BASENAME must match [A-Za-z0-9_-]{1,100}, else phasecont:latex:badName.
%   A file that cannot be opened raises phasecont:latex:writeFailed.
arguments
    folder (1,1) string {mustBeNonmissing}
    basename (1,1) string {mustBeNonmissing}
    tex (1,1) string {mustBeNonmissing}
    files (:,1) struct
end

if isempty(regexp(basename, "^[A-Za-z0-9_-]{1,100}$", "once"))
    error("phasecont:latex:badName", ...
        "Basename must be 1 to 100 letters, digits, ""_"" or ""-"", got ""%s"".", basename);
end
if ~isfolder(folder)
    [ok, msg] = mkdir(folder);
    if ~ok
        error("phasecont:latex:writeFailed", ...
            "Cannot create the folder ""%s"": %s", folder, msg);
    end
end
paths = fullfile(folder, basename + ".tex");
writeText(paths, tex);
for k = 1:numel(files)
    p = fullfile(folder, files(k).Name);
    writeText(p, files(k).Content);
    paths(end+1, 1) = p; %#ok<AGROW>
end
end

function writeText(path, text)
[fid, msg] = fopen(path, "w");
if fid < 0
    error("phasecont:latex:writeFailed", "Cannot write ""%s"": %s", path, msg);
end
closer = onCleanup(@() fclose(fid));
fwrite(fid, unicode2native(char(text), "UTF-8"), "uint8");
clear closer
end
