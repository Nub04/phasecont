function f = pollHook(varargin)
%POLLHOOK Test hook called on every pass of the continuation polling loop.
%   F = phasecont.internal.pollHook() returns the stored function handle
%   (empty by default). phasecont.internal.pollHook(F) stores F, and
%   phasecont.internal.pollHook([]) clears it. While a handle is stored,
%   continuation calls F(run, status) on each pass. It exists so a test can
%   raise an error inside the loop the way Ctrl+C does. It is not a public
%   option.
persistent hook
if nargin > 0
    hook = varargin{1};
end
if nargout > 0 || nargin == 0
    f = hook;
end
end
