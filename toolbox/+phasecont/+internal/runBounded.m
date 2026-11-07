function r = runBounded(argv, folder, timeout)
%RUNBOUNDED Run a program in a folder and stop it after a time limit.
%   R = phasecont.internal.runBounded(ARGV, FOLDER, TIMEOUT) runs the
%   program ARGV(1) with the arguments ARGV(2:end) in FOLDER. It waits at
%   most TIMEOUT seconds. When the limit passes it stops the program and
%   every process the program started, then returns.
%
%   R has the fields Status (the exit code, -1 when the program had to be
%   stopped), Output (what the program printed, standard output and error
%   together), TimedOut (true when the limit stopped it) and Command (the
%   command line, for display).
%
%   Base MATLAB has no way to put a limit on SYSTEM, and SYSTEM cannot be
%   interrupted. So the program starts through java.lang.ProcessBuilder in
%   the JVM of MATLAB, for a finite TIMEOUT and for Inf alike. The wait loop
%   pauses between checks, so Ctrl-C still works, and it stops the program
%   and what it started when it unwinds. TIMEOUT of Inf only turns the
%   clock off. No text goes through a shell on that route.
%
%   On Windows, Java adds ".exe" to a bare program name and nothing else, so
%   a program that is a .cmd or .bat wrapper is not found. The name is
%   looked up on the path with PATHEXT first, as cmd does, and Java gets the
%   full path.
%
%   SYSTEM is the fallback when MATLAB has no JVM or Java cannot start the
%   program. It cannot be stopped from here, and it needs every argument
%   after the first to hold letters, digits and _ - . / \ : = + only, else
%   phasecont:internal:badArgument. A finite TIMEOUT cannot be enforced
%   there, so the call warns with phasecont:latex:noTimeout. TIMEOUT of Inf
%   asks for no limit, so it falls back without a warning.
%
%   When the JVM cannot list child processes (it needs Java 9), only the
%   main program is stopped and the call warns with
%   phasecont:latex:killIncomplete.
arguments
    argv (1,:) string {mustBeNonempty, mustBeNonmissing}
    folder (1,1) string {mustBeNonmissing}
    timeout (1,1) double {mustBePositive}
end

command = quote(argv(1)) + strjoin(" " + argv(2:end), "");
r = struct("Status", -1, "Output", "", "TimedOut", false, "Command", command);

if ~usejava("jvm")
    if ~isinf(timeout)
        warning("phasecont:latex:noTimeout", ...
            "MATLAB runs without a JVM, so the time limit cannot be enforced.");
    end
    r = runUnbounded(r, argv, folder);
    return
end

outFile = string(tempname(tempdir)) + ".txt";
removeOut = onCleanup(@() deleteFile(outFile));
try
    list = java.util.ArrayList();
    list.add(char(resolveProgram(argv(1))));
    for k = 2:numel(argv)
        list.add(char(argv(k)));
    end
    builder = java.lang.ProcessBuilder(list);
    builder.directory(java.io.File(char(folder)));
    builder.redirectErrorStream(true);
    builder.redirectOutput(java.io.File(char(outFile)));
    proc = builder.start();
catch err
    if ~isinf(timeout)
        warning("phasecont:latex:noTimeout", ...
            "Java could not start ""%s"" (%s), so the time limit cannot be enforced.", ...
            argv(1), firstLine(err.message));
    end
    r = runUnbounded(r, argv, folder);
    return
end
stopProc = onCleanup(@() stopTree(proc));
try
    proc.getOutputStream().close();     % no input is coming
catch
    % The program may already have ended.
end

clock = tic;
while proc.isAlive()
    if toc(clock) >= timeout
        r.TimedOut = true;
        stopTree(proc);
        break
    end
    pause(0.05);
end
try
    proc.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
    if ~proc.isAlive() && ~r.TimedOut
        r.Status = double(proc.exitValue());
    end
catch
    % Status stays -1.
end
r.Output = readOutput(outFile);
end

function path = resolveProgram(name)
% Windows only: the full path of NAME as cmd would find it, or NAME when
% nothing matches (Java then reports the failure). A name without a folder
% is searched in every folder of the path, trying each PATHEXT extension.
path = name;
if ~ispc
    return
end
exts = split(string(getenv("PATHEXT")), ";");
exts = exts(exts ~= "");
if isempty(exts)
    exts = [".COM"; ".EXE"; ".BAT"; ".CMD"];
end
[~, ~, own] = fileparts(name);
if own == ""
    names = name + exts;
else
    names = [name; name + exts];
end
if contains(name, ["/", "\"])
    folders = "";
else
    folders = split(string(getenv("PATH")), ";");
    folders = erase(folders, """");
    folders = folders(folders ~= "");
end
for f = reshape(folders, 1, [])
    for n = reshape(names, 1, [])
        candidate = n;
        if f ~= ""
            candidate = fullfile(f, n);
        end
        if isfile(candidate)
            path = string(candidate);
            return
        end
    end
end
end

function r = runUnbounded(r, argv, folder)
for k = 2:numel(argv)
    if isempty(regexp(argv(k), "^[A-Za-z0-9_\-./\\:=+]+$", "once"))
        error("phasecont:internal:badArgument", ...
            "Without a time limit every argument after the first may hold letters, digits and _ - . / \\ : = + only.");
    end
end
here = pwd;
back = onCleanup(@() cd(here));
cd(folder);
[status, out] = system(r.Command);
clear back
r.Status = status;
r.Output = string(out);
end

function stopTree(proc)
% Stop the program and what it started. Descendants are listed first,
% because a program that has died no longer has any.
try
    if ~proc.isAlive()
        return
    end
catch
    return
end
descendants = {};
try
    it = proc.toHandle().descendants().iterator();
    while it.hasNext()
        descendants{end+1} = it.next(); %#ok<AGROW>
    end
catch
    warning("phasecont:latex:killIncomplete", ...
        "This JVM cannot list child processes, so only the main program was stopped.");
end
try
    proc.destroyForcibly();
catch
    % Already gone.
end
for k = 1:numel(descendants)
    try
        descendants{k}.destroyForcibly();
    catch
        % Already gone.
    end
end
end

function text = readOutput(file)
text = "";
if isfile(file)
    try
        text = string(fileread(file, Encoding = "UTF-8"));
    catch
        text = "";
    end
end
end

function deleteFile(file)
if isfile(file)
    delete(file);
end
end

function s = quote(path)
s = """" + path + """";
end

function s = firstLine(message)
parts = splitlines(string(message));
s = parts(1);
end
