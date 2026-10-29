function f = aotHook(varargin)
%AOTHOOK Test hook that rewrites the emitted C text before it is compiled.
%   F = phasecont.internal.aotHook() returns the stored function handle
%   (empty by default). phasecont.internal.aotHook(F) stores F and
%   phasecont.internal.aotHook([]) clears it. While a handle is stored,
%   aotBuild compiles F(text) instead of the emitted text, so a test can
%   break the C code. It is not a public option.
persistent hook
if nargin > 0
    hook = varargin{1};
end
if nargout > 0 || nargin == 0
    f = hook;
end
end
