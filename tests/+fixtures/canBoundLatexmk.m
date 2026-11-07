function ok = canBoundLatexmk()
%CANBOUNDLATEXMK True when the JVM of MATLAB can start latexmk itself.
%   A finite build time limit only works on that route. When Java cannot
%   start the program (for example latexmk is a .cmd wrapper on Windows),
%   phasecont.internal.runBounded warns and runs it without a limit, so a
%   test that builds a document which never ends would hang the suite.
%   Such tests call this function and skip when it is false.
ok = false;
if ~usejava("jvm")
    return
end
proc = [];
try
    list = java.util.ArrayList();
    list.add('latexmk');
    list.add('-v');
    builder = java.lang.ProcessBuilder(list);
    builder.redirectErrorStream(true);
    proc = builder.start();
    proc.getOutputStream().close();
    ok = true;
catch
    ok = false;
end
if ~isempty(proc)
    try
        proc.destroyForcibly();
    catch
        % Already gone.
    end
end
end
