classdef Ens < handle
    % An ensemble run in the engine. Internal: use phasecont.ensemble.
    %
    %   run = phasecont.internal.Ens(model, jobJson, x0, p)
    %   s = run.poll()
    %   v = run.copy(what, count)
    %   run.cancel()
    %
    % The run owns an engine handle that delete frees. Freeing a running
    % ensemble cancels it and waits for the worker, so Ctrl+C in a polling
    % loop stops the run. The run keeps its own copy of the model, so it does
    % not depend on the Model object after it starts.

    properties (Constant)
        % The what selectors of the engine's ens_copy.
        States = 0
        Times = 1
        Status = 2
        TReached = 3
        Accepted = 4
        Rejected = 5
    end

    properties (Hidden, SetAccess = private)
        Id (1,1) uint64 = uint64(0)
    end

    methods
        function obj = Ens(m, job, x0, p)
            arguments
                m (1,1) phasecont.Model
                job (1,1) string
                x0 (:,1) double
                p (:,1) double
            end
            obj.Id = pckern("ens_start", m.Id, char(job), x0, p);
        end

        function s = poll(obj)
            % The decoded status. state is "running", "cancelled" or "done".
            % A run that failed raises its engine error here.
            s = jsondecode(string(pckern("ens_poll", obj.Id)));
            s.state = string(s.state);
        end

        function v = copy(obj, what, count)
            % One array of a finished run as a column of COUNT values.
            v = pckern("ens_copy", obj.Id, what, count);
            v = v(:);
        end

        function cancel(obj)
            pckern("ens_cancel", obj.Id);
        end

        function delete(obj)
            if isempty(obj.Id) || obj.Id == 0
                return
            end
            id = obj.Id;
            obj.Id = uint64(0);
            try
                pckern("free", id);
            catch err
                if err.identifier ~= "phasecont:engine:badHandle"
                    warning("phasecont:model:freeFailed", "%s", err.message);
                end
            end
        end
    end
end
