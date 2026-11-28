function out = runLoop(m, problem, u0, p, onPoll)
%RUNLOOP Start an engine run, poll it to the end and copy its rows. Internal.
%   OUT = phasecont.internal.runLoop(M, PROBLEM, U0, P) starts the run that
%   the problem JSON PROBLEM describes, polls it until the worker stops
%   (see phasecont.internal.pollDelay for the wait between polls) and returns
%   the rows of every branch.
%   OUT = runLoop(M, PROBLEM, U0, P, ONPOLL) also calls ONPOLL(POINTS, ST)
%   after each poll, where POINTS is the cell of rows copied so far and ST
%   the decoded status. Use it to draw the run while it goes.
%
%   OUT is a struct with the fields
%     Status    decoded final status; branches is a column struct array
%     Json      the final status as the engine wrote it, a string
%     Points    1-by-nb cell, one stride-by-k matrix per branch
%     Specials  1-by-nb cell, the special-point rows of each branch
%     Stride    values per row
%
% An error in the loop (Ctrl+C included) deletes the run, which cancels it and
% waits for the worker. continuation, simulate and the session replay share
% this loop.
if nargin < 5
    onPoll = [];
end
run = phasecont.internal.Run(m, problem, u0, p);
cleanup = onCleanup(@() delete(run));
rows = phasecont.internal.RunRows(~isempty(onPoll));
polls = 0;
while true
    st = run.poll();
    pts = rows.pull(run, st);
    if ~isempty(onPoll)
        onPoll(pts, st);
    end
    hook = phasecont.internal.pollHook();
    if ~isempty(hook)
        hook(run, st);
    end
    if st.state ~= "running"
        break
    end
    polls = polls + 1;
    pause(phasecont.internal.pollDelay(polls));
end
out = rows.collect(run, st);
end
