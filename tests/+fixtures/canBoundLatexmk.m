function ok = canBoundLatexmk()
%CANBOUNDLATEXMK True when phasecont.internal.runBounded can start latexmk.
%   A finite build time limit only works on the JVM route. When Java cannot
%   start the program, runBounded warns and runs it without a limit, so a
%   test that builds a document which never ends would hang the suite.
%   Such tests call this function and skip when it is false. The function
%   runs "latexmk -v" through runBounded itself, so it sees the same path
%   lookup (a .cmd wrapper on Windows) that the build does.
ok = false;
if ~usejava("jvm")
    return
end
lastwarn("");
try
    r = phasecont.internal.runBounded(["latexmk", "-v"], string(tempdir), 60);
catch
    return
end
[~, id] = lastwarn;
ok = ~strcmp(id, "phasecont:latex:noTimeout") && ~r.TimedOut && r.Status == 0;
end
