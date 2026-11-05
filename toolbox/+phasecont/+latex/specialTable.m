function result = specialTable(source, options)
%SPECIALTABLE Render special points as a booktabs table.
%   R = phasecont.latex.specialTable(BR, Digits = 6) returns the LaTeX of a
%   table with one row per special point of the Branch BR: its label, its
%   type and the values of the free parameters. A branch of equilibria and
%   an LP or H curve add one column per state variable.
%
%   SOURCE can also be a struct with the fields Columns (string array, one
%   per value) and Rows (struct array with Label, Type and Values), which is
%   what phasecont.internal.latexTable builds. Edit that struct to rename
%   columns or rows before rendering.
%
%   Options
%     Digits      significant digits of every number, 1 to 17. Required:
%                 without it the call raises phasecont:latex:missingDigits.
%     Basename    name of the .tex file, default "table".
%     Folder      writes the .tex file here, creating the folder. Without it
%                 nothing is written.
%     Standalone  true gives a document with its own preamble, default false.
%
%   R has the fields Tex (string), Files (always empty, kept so that R reads
%   like the result of pgfplots), Paths (the files written) and Data
%   ("inline").
%
%   A branch without special points raises phasecont:latex:emptyTable.
arguments
    source (1,1)
    options.Digits (1,1) double = NaN
    options.Basename (1,1) string = "table"
    options.Folder (1,1) string = ""
    options.Standalone (1,1) logical = false
end

if isa(source, "phasecont.Branch")
    spec = phasecont.internal.latexTable(source);
elseif isstruct(source)
    spec = source;
else
    error("phasecont:latex:badInput", ...
        "The source must be a phasecont.Branch or a struct with Columns and Rows.");
end
opts = struct("Digits", options.Digits, "Standalone", options.Standalone);
json = phasecont.internal.latexJson("table", spec, opts);
[tex, files] = phasecont.internal.latexRender("table", json);
paths = strings(0, 1);
if options.Folder ~= ""
    paths = phasecont.internal.latexWrite(options.Folder, options.Basename, tex, files);
end
result = struct("Tex", tex, "Files", files, "Paths", paths, "Data", "inline");
end
