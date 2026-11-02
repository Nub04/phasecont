classdef Run < handle
    % A continuation run in the engine. Internal: use phasecont.continuation.
    %
    %   run = phasecont.internal.Run(model, problemJson, u0, p)
    %   s = run.poll()
    %   rows = run.copy(branch, what, start, count, stride)
    %   run.cancel()
    %
    % The run owns an engine handle that delete frees. Freeing a running run
    % cancels it and waits for the worker. The run keeps its own copy of the
    % model, so it does not depend on the Model object after it starts.

    properties (Hidden, SetAccess = private)
        Id (1,1) uint64 = uint64(0)
        % The last status as the engine wrote it, a JSON string.
        LastStatus (1,1) string = ""
    end

    methods
        function obj = Run(m, problem, u0, p)
            arguments
                m (1,1) phasecont.Model
                problem (1,1) string
                u0 (:,1) double
                p (:,1) double
            end
            obj.Id = pckern("run_start", m.Id, char(problem), u0, p);
        end

        function s = poll(obj)
            % The decoded status. branches is a struct array.
            raw = string(pckern("run_poll", obj.Id));
            obj.LastStatus = raw;
            s = phasecont.internal.Run.decode(raw);
        end

        function rows = copy(obj, branch, what, start, count, stride)
            % Rows of one branch as a stride-by-count matrix.
            % branch and start are 1-based. what: 0 points, 1 specials.
            rows = pckern("run_copy", obj.Id, branch - 1, what, ...
                start - 1, count, stride);
        end

        function cancel(obj)
            pckern("run_cancel", obj.Id);
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

    methods (Static)
        function s = decode(raw)
            % Decode a status JSON string the way poll does. A session load
            % uses it on the stored final status.
            s = jsondecode(raw);
            s.state = string(s.state);
            s.branches = phasecont.internal.Run.structArray(s.branches);
            for k = 1:numel(s.branches)
                if isfield(s.branches, "specials")
                    s.branches(k).specials = ...
                        phasecont.internal.Run.structArray(s.branches(k).specials);
                end
            end
        end
    end

    methods (Static, Access = private)
        function s = structArray(v)
            % jsondecode gives a struct array, a cell (entries with different
            % fields) or [] for a JSON array of objects. Return a column struct
            % array, 0-by-1 when empty.
            if isempty(v)
                s = repmat(struct(), 0, 1);
            elseif iscell(v)
                names = strings(1, 0);
                for k = 1:numel(v)
                    names = union(names, string(fieldnames(v{k}))', "stable");
                end
                s = repmat(cell2struct(cell(numel(names), 1), names, 1), numel(v), 1);
                for k = 1:numel(v)
                    for n = string(fieldnames(v{k}))'
                        s(k).(n) = v{k}.(n);
                    end
                end
            else
                s = v(:);
            end
        end
    end
end
