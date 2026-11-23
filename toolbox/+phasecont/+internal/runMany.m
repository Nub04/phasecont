function outs = runMany(jobs, cap, onError)
%RUNMANY Run several engine runs at the same time. Internal.
%   OUTS = phasecont.internal.runMany(JOBS, CAP, ONERROR) starts the runs
%   that the cell JOBS describes, at most CAP at a time, and polls every live
%   run on each pass. Each job is a struct with the fields Model, Problem, U0
%   and P that runLoop takes. When a run stops, its rows are collected and
%   the next job starts. OUTS is a 1-by-numel(JOBS) cell in input order; each
%   entry is the struct runLoop returns.
%
%   An engine error in job K (at the start, a poll or a copy) calls
%   ONERROR(K, ERR), which must throw. Any error, an error of the poll hook
%   and Ctrl+C included, deletes every live run, which cancels it and waits
%   for its worker.
%
% Each run waits on its own pollDelay schedule, and a pass waits for the
% shortest of them, so a run that just started is polled as soon as runLoop
% would poll it. The engine runs each job on its own worker thread with its
% own copy of the model, so the rows do not depend on CAP.
nj = numel(jobs);
outs = cell(1, nj);
runs = cell(1, 0);    % live Run objects
idx = zeros(1, 0);    % job index of each live run
rows = cell(1, 0);    % RunRows of each live run
polls = zeros(1, 0);  % passes each live run has had
% The cleanup holds the live runs in a handle, because an onCleanup that
% reads this workspace runs after the workspace is gone.
registry = containers.Map("KeyType", "double", "ValueType", "any");
cleanup = onCleanup(@() freeAll(registry));
next = 1;
while next <= nj && numel(runs) < cap
    startJob();
end
while ~isempty(runs)
    hook = phasecont.internal.pollHook();
    s = 1;
    while s <= numel(runs)
        k = idx(s);
        run = runs{s};
        try
            st = run.poll();
            rows{s}.pull(run, st);
        catch err
            onError(k, err);
        end
        if ~isempty(hook)
            hook(run, st);
        end
        if st.state ~= "running"
            try
                outs{k} = rows{s}.collect(run, st);
            catch err
                onError(k, err);
            end
            delete(run);
            remove(registry, k);
            runs(s) = [];
            idx(s) = [];
            rows(s) = [];
            polls(s) = [];
            if next <= nj
                startJob();
            end
        else
            polls(s) = polls(s) + 1;
            s = s + 1;
        end
    end
    if ~isempty(runs)
        % A run started in this pass has had no wait yet, so it does not
        % hold up the others.
        waits = arrayfun(@phasecont.internal.pollDelay, max(polls, 1));
        pause(min(waits));
    end
end

    function startJob()
        % Starts job NEXT and adds it to the live set.
        job = jobs{next};
        try
            r = phasecont.internal.Run(job.Model, job.Problem, job.U0, job.P);
        catch err
            onError(next, err);
        end
        registry(next) = r;
        runs{end+1} = r;
        idx(end+1) = next;
        rows{end+1} = phasecont.internal.RunRows(false);
        polls(end+1) = 0;
        next = next + 1;
    end
end

function freeAll(registry)
% Deletes every run that is still live, which cancels it and waits for its
% worker.
live = values(registry);
for j = 1:numel(live)
    delete(live{j});
end
remove(registry, keys(registry));
end
