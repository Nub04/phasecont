function brs = continuationBatch(jobs, options)
%CONTINUATIONBATCH Run several continuations at the same time.
%   BRS = phasecont.continuationBatch(JOBS) runs every job in the cell JOBS
%   and returns a 1-by-numel(JOBS) cell. Each job is a cell of the arguments
%   phasecont.continuation takes, {M, START, Name, Value, ...}, and BRS{K}
%   is what phasecont.continuation(JOBS{K}{:}) returns, bit for bit. The
%   engine runs each job on its own worker thread, so independent curves
%   (both directions of a branch, several start points, a family of models)
%   use several cores at once.
%
%   BRS = phasecont.continuationBatch(JOBS, MaxConcurrent = N) keeps at most
%   N runs live at a time. The default is maxNumCompThreads. N changes speed
%   and memory, never the results: each live run holds its own compiled
%   model and Jacobians.
%
%   Every job is checked before any run starts. The first error, in the
%   options of a job or in the engine, cancels the runs that are live and is
%   rethrown with the same identifier and "In job K:" in front of the
%   message. Ctrl+C cancels every live run. Plot cannot be used in a batch;
%   plot each branch after the batch returns.
%
%   Example:
%     m = phasecont.Model("brusselator.ode");
%     eq = phasecont.equilibrium(m, [1 1.2]);
%     brs = phasecont.continuationBatch({
%         {m, eq, "Free", "b", "Direction", 1}
%         {m, eq, "Free", "b", "Direction", -1}});
%
%   See also phasecont.continuation.
arguments
    jobs cell
    options.MaxConcurrent (1,1) double {mustBeInteger, mustBePositive} = maxNumCompThreads
end
nj = numel(jobs);
brs = cell(1, nj);
if nj == 0
    return
end
for k = 1:nj
    job = jobs{k};
    if ~iscell(job) || numel(job) < 2
        error("phasecont:model:badOption", ...
            "Job %d of continuationBatch must be a cell {m, start, Name, Value, ...}.", k);
    end
end
prepared = cell(1, nj);
phasecont.internal.prepareOnly(true);
restore = onCleanup(@() phasecont.internal.prepareOnly(false));
for k = 1:nj
    try
        prepared{k} = phasecont.continuation(jobs{k}{:});
    catch err
        throwInJob(k, err);
    end
end
clear restore
outs = phasecont.internal.runMany(prepared, options.MaxConcurrent, @runFailed);
for k = 1:nj
    try
        brs{k} = prepared{k}.Finish(outs{k});
    catch err
        throwInJob(k, err);
    end
end

    function runFailed(k, err)
        try
            prepared{k}.RunError(err);
        catch err2
            throwInJob(k, err2);
        end
        throwInJob(k, err);
    end
end

function throwInJob(k, err)
% Rethrows ERR with the same identifier and the job number in the message.
id = err.identifier;
if id == ""
    id = "phasecont:continuationBatch:jobFailed";
end
e = MException(id, "%s", sprintf("In job %d: %s", k, err.message));
e = e.addCause(err);
throw(e);
end
