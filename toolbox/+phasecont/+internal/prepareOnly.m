function f = prepareOnly(varargin)
%PREPAREONLY Switch continuation into the mode continuationBatch uses. Internal.
%   F = phasecont.internal.prepareOnly() returns true while the mode is on
%   (false by default). phasecont.internal.prepareOnly(TF) sets it. While it
%   is on, phasecont.continuation checks its options and builds the problem
%   as usual, then returns a job struct (Model, Problem, U0, P, Finish,
%   RunError) without starting a run. continuationBatch turns it on while it
%   collects jobs and turns it off again through onCleanup, so an error does
%   not leave it on. It is not a public option.
persistent on
if isempty(on)
    on = false;
end
if nargin > 0
    on = logical(varargin{1});
end
if nargout > 0 || nargin == 0
    f = on;
end
end
