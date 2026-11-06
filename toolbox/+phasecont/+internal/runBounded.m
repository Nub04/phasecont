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
%   Base MATLAB has no way to put a limit on SYSTEM, so a finite TIMEOUT
%   starts the program through java.lang.ProcessBuilder in the JVM of
%   MATLAB. The wait loop pauses between checks, so Ctrl-C still works, and
%   it stops the program when it unwinds. No text goes through a shell on
%   that route.
%
%   Two cases use SYSTEM instead, which cannot be stopped from here. TIMEOUT
%   of Inf asks for no limit and runs silently. When MATLAB has no JVM, or
%   Java cannot start the program, the call warns with
%   phasecont:latex:noTimeout and runs it without a limit. That route
%   needs every argument after the first to hold letters, digits and
%   _ - . / \ : = + only, else phasecont:internal:badArgument.
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

if isinf(timeout)
    r = runUnbounded(r, argv, folder);
    return
end
if ~usejava("jvm")
    warning("phasecont:latex:noTimeout", ...
        "MATLAB runs without a JVM, so the time limit cannot be enforced.");
    r = runUnbounded(r, argv, folder);
    return
end

outFile = string(tempname(tempdir)) + ".txt";
removeOut = onCleanup(@() deleteFile(outFile));
try
    list = java.util.ArrayList();
    for k = 1:numel(argv)
        list.add(char(argv(k)));
    end
    builder = java.lang.ProcessBuilder(list);
    builder.directory(java.io.File(char(folder)));
    builder.redirectErrorStream(true);
    builder.redirectOutput(java.io.File(char(outFile)));
    proc = builder.start();
catch err
    warning("phasecont:latex:noTimeout", ...
        "Java could not start ""%s"" (%s), so the time limit cannot be enforced.", ...
        argv(1), firstLine(err.message));
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
