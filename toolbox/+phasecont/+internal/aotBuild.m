function [lib, info] = aotBuild(id)
%AOTBUILD Compile the C text of a model into a library with mex.
%   [LIB, INFO] = phasecont.internal.aotBuild(ID) asks the engine for the C
%   text of model ID, compiles it together with a stub mexFunction and
%   returns the absolute path of the library. INFO has the fields Compiler,
%   Version, Flags and Library.
%
%   The library lives under tempdir/phasecont-aot. Its folder name joins a
%   hash of the C text and the build flags with the compiler name and
%   version, so the same text built by the same compiler is compiled once.
%   Floating-point contraction
%   is turned off (/fp:strict on MSVC, -ffp-contract=off elsewhere) so the
%   results match the JIT. A failed compile raises phasecont:aot:compile.
arguments
    id (1,1) uint64
end
cfgs = mex.getCompilerConfigurations("C", "Selected");
if isempty(cfgs)
    error("phasecont:aot:compile", ...
        "No mex C compiler is selected; run mex -setup C.");
end
cfg = cfgs(1);
name = string(cfg.Name);
version = string(cfg.Version);
if contains(name, "Visual C++", IgnoreCase = true) ...
        || contains(string(cfg.Manufacturer), "Microsoft", IgnoreCase = true)
    flags = "/fp:strict";
    flagArgs = {"COMPFLAGS=$COMPFLAGS /fp:strict"};
else
    % mex links with an export list (a version script on Linux, an
    % exported-symbols list on macOS) that keeps only mexFunction, so the
    % engine could not find pc_aot_init. Empty export settings keep every
    % symbol that the C text marks with default visibility.
    flags = "-ffp-contract=off LINKEXPORT= LINKEXPORTVER=";
    flagArgs = {"CFLAGS=$CFLAGS -ffp-contract=off", "LINKEXPORT=", "LINKEXPORTVER="};
end

root = fullfile(tempdir, "phasecont-aot");
emitDir = fullfile(root, "emit");
if ~isfolder(emitDir)
    mkdir(emitDir);
end
pckern("model_compile", id, jsonencode(struct("emit_c", emitDir)));
cfile = string(jsondecode(pckern("model_info", id)).c_file);
text = string(fileread(cfile, Encoding = "UTF-8"));
hook = phasecont.internal.aotHook();
if ~isempty(hook)
    text = string(hook(text));
end

compilerKey = matlab.lang.makeValidName(name + "_" + version);
% The flags are part of the hash, so a library built with other flags or
% link settings is never reused.
key = textHash(text + newline + flags) + "_" + compilerKey;
dest = fullfile(root, key);
output = "pcaot_" + extractBefore(key + "_", "_");
lib = string(fullfile(dest, output + "." + mexext));
info = struct("Compiler", name, "Version", version, "Flags", flags, ...
    "Library", lib);
if isfile(lib)
    return
end

if ~isfolder(dest)
    mkdir(dest);
end
src = fullfile(dest, output + ".c");
stub = fullfile(dest, output + "_stub.c");
writeText(src, text);
writeText(stub, "void mexFunction(int nlhs, void **plhs, int nrhs, " + ...
    "const void **prhs) { (void)nlhs; (void)plhs; (void)nrhs; (void)prhs; }" + newline);
args = [{"-R2018a", "-silent"}, flagArgs, {"-outdir", char(dest), ...
    "-output", char(output), char(src), char(stub)}];
try
    log = evalc("mex(args{:})"); %#ok<NASGU>
catch err
    error("phasecont:aot:compile", "mex failed: %s", err.message);
end
if ~isfile(lib)
    error("phasecont:aot:compile", "mex did not produce %s.", lib);
end
end

function writeText(file, text)
fid = fopen(file, "w", "n", "UTF-8");
if fid < 0
    error("phasecont:aot:compile", "Cannot write %s.", file);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, char(text), "char");
end

function h = textHash(text)
% Two 32-bit FNV-1a hashes with different seeds of the UTF-8 bytes, as 16
% hex digits. It names the cache folder, so it must be stable, not secret.
bytes = uint64(unicode2native(char(text), "UTF-8"));
a = uint64(2166136261);
b = uint64(3735928559);
mask = uint64(4294967295);
prime = uint64(16777619);
for k = 1:numel(bytes)
    a = bitand(bitxor(a, bytes(k)) * prime, mask);
    b = bitand(bitxor(b, bytes(k) + 31) * prime, mask);
end
h = string(sprintf("%08x%08x", a, b));
end
