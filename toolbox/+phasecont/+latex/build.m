function result = build(tex, options)
%BUILD Compile a LaTeX document to PDF with latexmk.
%   R = phasecont.latex.build(TEX) writes TEX to main.tex in a new folder
%   under tempdir and runs latexmk there. TEX is a whole document, so render
%   it with Standalone = true, or wrap a fragment yourself.
%
%   Options
%     Files       struct array with Name and Content, the data files that
%                 the document reads (R.Files of phasecont.latex.pgfplots)
%     Engine      "latexmk", the only engine for now
%     Executable  path of the latexmk program, default the name "latexmk"
%                 on the system path. Letters, digits, spaces and
%                 _ - . / \ : ( ) + only, else phasecont:latex:badEngine.
%     Timeout     seconds that latexmk may run, default 300. Inf sets no
%                 limit. When the time is up, latexmk and the TeX programs
%                 it started are stopped.
%
%   R has the fields Ok (true when a PDF came out), Pdf (its path, "" when
%   there is none), Folder (the build folder, which is not deleted), Log
%   (the TeX log, or the program output when there is no log), Errors (a
%   struct array with File, Line and Message from the -file-line-error
%   lines of the log), Command (the command that ran) and TimedOut (true
%   when the time limit stopped latexmk; Ok is false and Pdf is "" then).
%
%   The document name is fixed, so no text of TEX reaches the command line.
%   Shell escape stays off, and latexmk stops at the first error. A TeX
%   error or a timeout is reported in R and does not raise. A missing
%   latexmk raises phasecont:latex:engineNotFound.
%
%   A finite Timeout starts latexmk through the JVM of MATLAB, since SYSTEM
%   cannot be interrupted. When that is not possible the call warns with
%   phasecont:latex:noTimeout and runs latexmk with no limit, as it does
%   for Timeout = Inf. Details are in phasecont.internal.runBounded.
arguments
    tex (1,1) string {mustBeNonmissing}
    options.Files (:,1) struct = repmat(struct("Name", "", "Content", ""), 0, 1)
    options.Engine (1,1) string {mustBeMember(options.Engine, "latexmk")} = "latexmk"
    options.Executable (1,1) string = ""
    options.Timeout (1,1) double {mustBePositive} = 300
end

exe = options.Executable;
if exe == ""
    exe = options.Engine;
end
if isempty(regexp(exe, "^[A-Za-z0-9 _\-./\\:()+]+$", "once"))
    error("phasecont:latex:badEngine", ...
        "The latexmk path may hold letters, digits, spaces and _ - . / \\ : ( ) + only.");
end

[status, ~] = system(quote(exe) + " -v");
if status ~= 0
    error("phasecont:latex:engineNotFound", ...
        "Cannot run ""%s"" (status %d). Install TeX Live or MiKTeX, or pass Executable.", ...
        exe, status);
end

folder = string(tempname(tempdir));
phasecont.internal.latexWrite(folder, "main", tex, options.Files);

argv = [exe, "-pdf", "-norc", "-no-shell-escape", "-interaction=nonstopmode", ...
    "-halt-on-error", "-file-line-error", "-output-directory=out", "main.tex"];
job = phasecont.internal.runBounded(argv, folder, options.Timeout);

pdf = fullfile(folder, "out", "main.pdf");
if job.TimedOut || ~isfile(pdf)
    pdf = "";    % a PDF that a stopped run left behind is not a result
end
logFile = fullfile(folder, "out", "main.log");
if isfile(logFile)
    log = string(fileread(logFile, Encoding = "UTF-8"));
else
    log = job.Output;
end
result = struct("Ok", job.Status == 0 && pdf ~= "", "Pdf", pdf, "Folder", folder, ...
    "Log", log, "Errors", parseErrors(log), "Command", job.Command, ...
    "TimedOut", job.TimedOut);
end

function s = quote(path)
s = """" + path + """";
end

function errors = parseErrors(log)
% File, Line and Message of every "./main.tex:12: message" line.
tokens = regexp(log, "(?m)^(?:\./)?([^\s:]+\.tex):(\d+): (.*?)\r?$", "tokens");
errors = repmat(struct("File", "", "Line", 0, "Message", ""), 0, 1);
for k = 1:numel(tokens)
    t = tokens{k};
    message = strtrim(t(3));
    if startsWith(message, "==>")
        continue    % TeX's closing "Fatal error occurred" line repeats the error
    end
    errors(end+1, 1) = struct("File", t(1), "Line", str2double(t(2)), ... %#ok<AGROW>
        "Message", message);
end
end
