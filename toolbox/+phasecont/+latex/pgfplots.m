function result = pgfplots(fig, options)
%PGFPLOTS Render a figure as pgfplots LaTeX.
%   R = phasecont.latex.pgfplots(FIG, Digits = 6) returns the LaTeX for FIG,
%   the struct that phasecont.latex.figureFromBranch returns. The text is a
%   fragment for \input, or a whole document with Standalone = true.
%
%   Options
%     Digits      significant digits of every number, 1 to 17. Required:
%                 without it the call raises phasecont:latex:missingDigits.
%     Data        "inline" (default) writes the points into the document.
%                 "external" writes one table file per series next to it.
%     Basename    name of the .tex file, and the prefix of the data files,
%                 default "figure". Letters, digits, "_" and "-" only.
%     Folder      writes the files here, creating the folder. Without it
%                 nothing is written, and external data comes back in
%                 R.Files only.
%     Standalone  true gives a document with its own preamble, default false.
%
%   R has the fields Tex (string), Files (struct array with Name and
%   Content, the external data tables), Paths (the files written, empty
%   without Folder) and Data ("inline" or "external").
%
%   A NaN or Inf in the data, a series without points, or a mismatch
%   between X and Y raises phasecont:latex:nonFinite, emptySeries or
%   lengthMismatch, and the message names the series and index.
arguments
    fig (1,1) struct
    options.Digits (1,1) double = NaN
    options.Data (1,1) string {mustBeMember(options.Data, ["inline" "external"])} = "inline"
    options.Basename (1,1) string = "figure"
    options.Folder (1,1) string = ""
    options.Standalone (1,1) logical = false
end

opts = struct("Digits", options.Digits, "Standalone", options.Standalone, ...
    "Data", options.Data, "Basename", options.Basename);
json = phasecont.internal.latexJson("pgfplots", fig, opts);
[tex, files] = phasecont.internal.latexRender("pgfplots", json);
paths = strings(0, 1);
if options.Folder ~= ""
    paths = phasecont.internal.latexWrite(options.Folder, options.Basename, tex, files);
end
result = struct("Tex", tex, "Files", files, "Paths", paths, "Data", options.Data);
end
