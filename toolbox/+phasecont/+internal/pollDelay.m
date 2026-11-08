function d = pollDelay(k)
%POLLDELAY Seconds to wait before the next engine poll. Internal.
%   D = phasecont.internal.pollDelay(K) returns the wait after the K-th poll
%   of a run, for K = 1, 2, .... The first six waits are 1, 2, 4, 8, 16 and
%   19 ms, which add up to 50 ms, and every later wait is 50 ms. The polls
%   therefore fall at 0, 1, 3, 7, 15, 31 and 50 ms and then every 50 ms, so
%   they include every time the old fixed 50 ms loop polled. A run that
%   stops within a few milliseconds is seen within a few milliseconds, and no
%   run is seen later than it was before. continuation, simulate, the session
%   replay and ensemble share this schedule.
arguments
    k (1,1) double {mustBeInteger, mustBePositive}
end
ramp = [0.001 0.002 0.004 0.008 0.016 0.019];
if k <= numel(ramp)
    d = ramp(k);
else
    d = 0.05;
end
end
