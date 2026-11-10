function root = aotCacheRoot()
%AOTCACHEROOT The folder that holds the compiled AOT libraries.
%   ROOT = phasecont.internal.aotCacheRoot() returns the cache folder and
%   creates it when it is missing. MATLAB loads the libraries in this
%   folder, so nobody else may be able to put a file there.
%
%   Windows and macOS give every user a private tempdir, so the folder is
%   tempdir/phasecont-aot. On Linux tempdir is a shared /tmp. The folder
%   is tempdir/phasecont-aot-<uid>, created with mode 700, and it must be
%   owned by the current user and not writable by the group or others.
%   Otherwise this function raises phasecont:aot:cacheFolder, and the
%   caller falls back to the JIT.
if ~isunix || ismac
    root = string(fullfile(tempdir, "phasecont-aot"));
    if ~isfolder(root)
        mkdir(root);
    end
    return
end
[status, uidText] = system("id -u");
uid = strtrim(string(uidText));
if status ~= 0 || isempty(regexp(uid, "^[0-9]+$", "once"))
    error("phasecont:aot:cacheFolder", "Cannot read the user id: %s", uidText);
end
root = string(fullfile(tempdir, "phasecont-aot-" + uid));
if contains(root, "'")
    error("phasecont:aot:cacheFolder", ...
        "The cache folder path contains a single quote: %s", root);
end
if ~isfolder(root)
    % One call sets the mode, so the folder is never visible with a looser
    % mode, and it fails if another user created the name first.
    system("mkdir -m 700 '" + root + "'");
end
[status, out] = system("stat -c '%u %a' '" + root + "'");
if status ~= 0 || ~phasecont.internal.aotCacheOwnerOk(out, uid)
    error("phasecont:aot:cacheFolder", ...
        "%s is not owned by user %s or can be written by others. Remove it.", ...
        root, uid);
end
end
